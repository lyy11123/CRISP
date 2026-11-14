"""Read-only C MF6/Split report, including original rigid EIR search costs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics

import fg_batch
import hi_closeout_c as run
import hi_h4_report as fault_report
from post_fg_diag import table

ROOT = run.ROOT
COST_COMPONENTS = ("backend_seconds", "backend_copy_prepare_seconds", "R0_prepare_seconds", "L1_search_seconds",
    "calibration_seconds", "search_seconds", "restore_seconds", "final_cpcm_seconds",
    "evaluation_CPCM_seconds", "split_axis_construction_seconds", "other_task_seconds")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backend(record):
    assert set(record["backend"]) == set(range(1, 11))
    assert not record["fault_violations"]
    return [record["backend"][s][0] for s in range(1, 11)]


def graph_metrics(export, spec, dead):
    data = run.split.static.read_pickle(ROOT / export)
    events = pickle.loads(data["events_blob"])
    hp = run.split.platform(spec, dead)
    metrics = run.split.event_metrics(events, hp)
    bounds = run.split.tensor_bounds(events, hp)
    return metrics, bounds, block_times(events, hp)


def block_times(events, hp):
    values = [run.split.service(e, hp) for e in events.values()
              if e.event_type == "computation" and e.comp_device == "tensorcore"]
    return dict(min_ns=min(values), max_ns=max(values), mean_ns=statistics.mean(values),
                total_ns=sum(values), counts=dict(Counter(map(str, values))))


def structural_rows(model, method, metrics, bounds, blocks, rigid_bound, info=None):
    load = dict(model=model, method=method, tensor_blocks=bounds["blocks"], live_tensor_units=bounds["live_units"],
        block_time_min_ns=blocks["min_ns"], block_time_max_ns=blocks["max_ns"],
        block_time_mean_ns=blocks["mean_ns"], block_time_total_ns=blocks["total_ns"],
        block_time_distribution_ns=blocks["counts"], basic_lower_bound_ns=bounds["basic_ns"],
        count_lower_bound_ns=bounds["count_ns"], current_tensor_lower_bound_ns=bounds["lower_bound_ns"],
        original_post_fault_lower_bound_ns=None if method == "healthy" else rigid_bound,
        busiest_tensor_ns=metrics["compute_peak_ns"]["tensorcore"],
        effective_flops=metrics["tensor_effective_flops"], rounded_flops=metrics["tensor_rounded_flops"])
    communication = dict(model=model, method=method, **metrics["communication"])
    if info is not None:
        communication.update(affected_weight_input_bytes_before=info["affected_weight_network_bytes_before"],
            affected_weight_input_bytes_after=info["affected_weight_network_bytes_after"],
            affected_activation_input_bytes_after=info["affected_activation_network_bytes_after"],
            healthy_computation_moved=info["healthy_computation_moved"])
    return load, communication


def quality_row(model, phase_op, method, values, estimate, healthy, r0mean, bound):
    return dict(model=model, phase_op=phase_op, method=method, backend_samples=len(values),
        backend_mean_ns=statistics.mean(values), backend_min_ns=min(values), backend_max_ns=max(values), T_hat_ns=estimate,
        relative_to_healthy=statistics.mean(values)/healthy-1,
        relative_to_R0=statistics.mean(values)/r0mean-1 if r0mean is not None else None,
        rigid_tensor_lower_bound_ns=bound,
        backend_below_rigid_bound=sum(v<bound for v in values) if method != "healthy" else None,
        all_backend_below_rigid_bound=max(values)<bound if method != "healthy" else None)


def complete_cost(cost):
    measured = sum(cost[key] for key in COST_COMPONENTS if key != "other_task_seconds")
    cost["other_task_seconds"] = cost["task_body_seconds"] - measured
    cost["startup_wrapper_poll_seconds"] = cost["wall_seconds"] - cost["task_body_seconds"]
    assert all(cost[key] >= -1e-6 for key in COST_COMPONENTS), cost
    assert cost["startup_wrapper_poll_seconds"] >= -1e-6, cost
    cost["exclusive_task_sum_seconds"] = sum(cost[key] for key in COST_COMPONENTS)
    cost["task_partition_error_seconds"] = cost["exclusive_task_sum_seconds"] - cost["task_body_seconds"]
    cost["receipt_partition_error_seconds"] = cost["exclusive_task_sum_seconds"] + cost["startup_wrapper_poll_seconds"] - cost["wall_seconds"]
    assert math.isclose(cost["exclusive_task_sum_seconds"], cost["task_body_seconds"], abs_tol=1e-6)
    assert abs(cost["receipt_partition_error_seconds"]) < 1e-6
    return cost


def offline_migration(state, info):
    """Match the EIR migration definition: new DDR copies, not PE input reads."""
    _, original_flows = run.split.input_flows(state)
    compared = 0
    for transfer in info["transfers"]:
        if not transfer["offline"]:
            continue
        candidates = [f for f in original_flows if f["offline"] and tuple(f["data"]) == tuple(transfer["data"])
                      and run.split.overlap(f["region"], transfer["region"]) == tuple(map(tuple, transfer["region"]))]
        assert candidates
        assert tuple(transfer["source"]) in {tuple(f["source"]) for f in candidates}
        compared += 1
    return dict(unique_ddr_migration_bytes=0, unique_ddr_migration_blocks=0,
                matched_offline_transfers=compared,
                definition="new DDR source copies of unique weight slices, as in EIR; local input traffic reported separately")


def collect(out, progress=False):
    run.split.setup()
    out = Path(out).resolve()
    local = run.private_out(out)
    manifest = run.frozen.read_json(run.MANIFEST)
    jobs = {j["id"]: j for j in manifest["jobs"]}
    present, missing, checks = {}, [], []
    for name, job in jobs.items():
        path = out / job["result"]
        receipt_path = out / "receipts" / (name + ".json")
        if not path.exists() or not receipt_path.exists():
            missing.append(name)
            continue
        result = run.frozen.read_json(path)
        receipt = run.frozen.read_json(receipt_path)
        assert receipt["identity"] == fg_batch.identity(job) and receipt["rc"] == 0
        assert result["job_id"] == name and result["kind"] == job["kind"]
        assert result["status"] in ("ok", "reused")
        if result["status"] == "ok":
            assert result["spec"] == job["spec"]
        else:
            assert job["kind"] == "fault_rep"
        present[name] = dict(spec=job["spec"], result=result, receipt=receipt)
        checks.append(dict(job_id=name, identity=receipt["identity"], result_sha256=sha(path),
                           receipt_sha256=sha(receipt_path), result_status=result["status"]))
    if not progress:
        assert not missing, missing
        for model in run.hi_config.MODELS:
            r = run.frozen.read_json(local / "representatives" / (model + ".json"))
            assert not r["applicable"] or "C_" + model + "__stop300" in jobs

    samples, all_states, costs, inventories = [], [], [], []
    for name, item in present.items():
        s, r = item["spec"], item["result"]
        cost = dict(job_id=name, model=s["model"], kind=s["kind"], status=r["status"],
                    wall_seconds=item["receipt"]["wall_seconds"], task_body_seconds=r["wall_seconds"],
                    peak_rss_gb=item["receipt"]["peak_rss_gb"],
                    actual_search_trajectories=0, actual_backend_calls=0,
                    **dict.fromkeys(COST_COMPONENTS, 0.),
                    inherited_R0_seconds=0., inherited_L1_seconds=0.,
                    receipt_reconstructed_from_task=item["receipt"].get("recovered_after_controller_restart", False),
                    backend_timer_scope="not separately timed; any backend work remains in other_task_seconds")
        if r["status"] == "reused":
            costs.append(complete_cost(cost))
            continue
        if s["kind"] == "fault_select":
            assert r["anchor_backend_calls"] == 1
            assert r["faults"]["MF6"]["failed_nodes"] == [] and r["faults"]["MF6"]["failed_links"] == []
            assert len(r["faults"]["MF6"]["failed_pes"]) == 1
            cost["actual_backend_calls"] = 1
        elif s["kind"] in ("fault_pre", "fault_l1", "fault_rep"):
            raw = run.frozen.read_pickle(r["sink"])
            assert raw["backend_seeds"] == list(range(1, 11))
            if s["kind"] == "fault_pre":
                entries = [("healthy", raw["record"])]
                assert raw["T"] == r["T"]
                reused = "reused_healthy_record" in r
                cost["healthy_reused"] = reused
            elif s["kind"] == "fault_l1":
                entries = [("R0", raw["R0"]), ("L1", raw["L1"])]
                reused = False
                cost.update(R0_prepare_seconds=raw["t_R0"], L1_search_seconds=raw["t_L1"])
            else:
                entries = [("L2", raw["record"])]
                reused = False
                assert s["steps"] in run.frozen.STEPS and s["seed"] in run.frozen.SEEDS
                assert raw["proposals"]["steps"] == s["steps"]
                assert [row["step"] for row in raw["search"]["rows"]] == list(range(1, s["steps"] + 1))
                assert raw["spec"]["method"] == "L2Gsync" and raw["spec"]["checks"] is False
                assert raw["T_L1_check"] == raw["search"]["T_start"]
                cost.update(actual_search_trajectories=1, calibration_seconds=raw["search"]["t_calibration"],
                    search_seconds=raw["search"]["t_L2_search"], restore_seconds=raw["search"]["t_restore_best"],
                    inherited_R0_seconds=raw["timing"]["t_R0"], inherited_L1_seconds=raw["timing"]["t_L1"],
                    calibration_proposals=raw["proposals"]["calibration"], search_proposals=raw["proposals"]["steps"],
                    best_step=raw["search"]["best_step"], finalize_inclusive_seconds=raw["timing"]["t_finalize"])
                assert math.isclose(raw["timing"]["t_finalize"], raw["search"]["t_restore_best"] + raw["record"]["t_finalize"], abs_tol=1e-9)
            for label, record in entries:
                backend(record)
                state_row = fault_report.state_rows(s, label, record, r, samples)
                state_row["reused_healthy_record"] = r.get("reused_healthy_record")
                all_states.append(state_row)
            if not reused:
                cost["actual_backend_calls"] = 10 * len(entries)
                cost["backend_seconds"] = sum(sum(rec["backend_wall"]) for _, rec in entries)
                cost["backend_copy_prepare_seconds"] = sum(rec["t_backend"] for _, rec in entries) - cost["backend_seconds"]
                cost["final_cpcm_seconds"] = sum(rec["t_finalize"] for _, rec in entries)
                cost["backend_timer_scope"] = "EIR direct backend calls; deep-copy and setup measured separately"
            inventories.append(dict(job_id=name, raw_sha256=sha(ROOT / r["sink"]),
                exports_sha256={k: sha(ROOT / p) for k, p in r["exports"].items()}))
        elif s["kind"] == "c_split_eval":
            assert r["seed"] == s["seed"] and r["actual_backend_calls"] == 1
            cost.update(actual_backend_calls=1, backend_seconds=r["backend_seconds"],
                        evaluation_CPCM_seconds=r["cpcm_seconds"],
                        backend_timer_scope="Split saved backend interval includes the platform copy passed to event_driver; event copy is in other_task_seconds")
            samples.append(dict(job_id=name, model=s["model"], phase_op=s["phase_op"],
                state=s["method"], backend_seed=s["seed"], backend_ns=r["backend_triple_ns"][0],
                unhidden_compute_ns=r["backend_triple_ns"][1], unhidden_communication_ns=r["backend_triple_ns"][2],
                backend_call_seconds=r["backend_seconds"]))
        elif s["kind"] == "c_split_prepare":
            cost["split_axis_construction_seconds"] = sum(row["build_seconds"] for row in r["choices"])
        costs.append(complete_cost(cost))

    stops, _, trajectories = fault_report.stop_tables(jobs, present) if any(j["kind"] == "fault_stop" for j in jobs.values()) else ([], [], [])
    quality, static_rows, choices, coverage, migrations = [], [], [], [], []
    loads, communications, comparisons, arm_coverage, current_l1 = [], [], [], [], []
    for model in run.hi_config.MODELS:
        prefix = "C_" + model + "__"
        rep_path = local / "representatives" / (model + ".json")
        if not rep_path.exists():
            coverage.append(dict(model=model, state="pending anchor/structural selection"))
            continue
        representative = run.frozen.read_json(rep_path)
        coverage.append(dict(model=model, phase_op=representative["phase_op"], applicable=representative["applicable"],
                             failed_pe=representative["failed_pe"], reason=representative["reason"],
                             rule=representative["rule"], inspections=representative["inspections"],
                             planned_jobs=sum(n.startswith(prefix) for n in jobs),
                             complete_jobs=sum(n.startswith(prefix) for n in present)))
        if not representative["applicable"]:
            continue
        if prefix + "pre" not in present:
            continue
        pre = present[prefix + "pre"]["result"]
        pre_raw = run.frozen.read_pickle(pre["sink"])
        healthy = statistics.mean(backend(pre_raw["record"]))
        s = present[prefix + "pre"]["spec"]
        prepared_item = present.get(prefix + "split_prepare")
        if prepared_item is None:
            continue
        prepared = prepared_item["result"]
        chosen_state = run.load_split_state(present[prefix + "split_prepare"]["spec"], local)
        choices += [dict(model=model, **r) for r in prepared["choices"]]
        bound = run.frozen.read_json(local / "metadata" / f"C_{model}_m1.json")["rigid_tensor_load_bound"]["lower_bound_ns"]
        final_arms = [("healthy", backend(pre_raw["record"]), pre_raw["record"]["T"], pre["exports"]["healthy"], pre_raw["record"])]
        r0mean, stop = None, None
        if prefix + "l1" in present:
            l1 = present[prefix + "l1"]["result"]
            l1_raw = run.frozen.read_pickle(l1["sink"])
            r0mean = statistics.mean(backend(l1_raw["R0"]))
            final_arms.append(("R0", backend(l1_raw["R0"]), l1_raw["R0"]["T"], l1["exports"]["R0"], l1_raw["R0"]))
        if prefix + "stop300" in present:
            stop = present[prefix + "stop300"]["result"]
            selected_raw = run.frozen.read_pickle(present[stop["source_job"]]["result"]["sink"])
            final_record = selected_raw[stop["level"]] if stop["level"] in ("R0", "L1") else selected_raw["record"]
            final_arms.append(("rigid EIR", backend(final_record), final_record["T"], stop["export"], final_record))
        for method, values, estimate, exported, raw_record in final_arms:
            dead = None if method == "healthy" else tuple(representative["failed_pe"])
            metrics, state_bound, blocks = graph_metrics(exported, s, dead)
            row = dict(quality_row(model, s["phase_op"], method, values, estimate, healthy, r0mean, bound),
                tensor_peak_ns=metrics["compute_peak_ns"]["tensorcore"], tensor_load_bound_ns=state_bound["lower_bound_ns"],
                EIR_stop_level=stop["level"] if method == "rigid EIR" else None,
                EIR_stop_reason=stop["reason"] if method == "rigid EIR" else None,
                EIR_repair_seed=stop.get("seed") if method == "rigid EIR" else None)
            quality.append(row)
            static_rows.append(dict(model=model, method=method, **metrics))
            migrations.append(dict(model=model, method=method, **raw_record.get("migration_vs_start", {})))
            load, traffic = structural_rows(model, method, metrics, state_bound, blocks, bound)
            loads.append(load); communications.append(traffic)
        for factor, method in ((1, "Local-m1"), (2, "Split-m2")):
            if any(prefix + f"m{factor}_s{seed}" not in present for seed in range(1, 11)):
                continue
            members = [present[prefix + f"m{factor}_s{seed}"]["result"] for seed in range(1, 11)]
            info = run.frozen.read_json(local / "metadata" / f"C_{model}_m{factor}.json")
            graph_info = prepared["graphs"][str(factor)]
            payload = run.frozen.read_pickle(graph_info["graph"])
            events = pickle.loads(payload["events_blob"])
            hp = run.split.platform(s, tuple(representative["failed_pe"]))
            metrics = run.split.event_metrics(events, hp)
            state_bound = run.split.tensor_bounds(events, hp)
            assert metrics == info["metrics"] and state_bound == info["tensor_load_bound"]
            assert run.split.graph_digest(events) == graph_info["graph_sha256"]
            assert all(r["graph_sha256"] == graph_info["graph_sha256"] for r in members)
            values = [r["backend_triple_ns"][0] for r in members]
            assert len({r["T_hat_ns"] for r in members}) == 1
            assert info["coverage_checked"] and info["healthy_computation_moved"] == 0
            assert info["metrics"]["tensor_effective_flops"] == info["baseline_metrics"]["tensor_effective_flops"]
            quality.append(dict(quality_row(model, s["phase_op"], method, values, members[0]["T_hat_ns"], healthy, r0mean, bound),
                tensor_peak_ns=info["metrics"]["compute_peak_ns"]["tensorcore"],
                tensor_load_bound_ns=info["tensor_load_bound"]["lower_bound_ns"], axis=info["axis"],
                decision_seconds=info["axis_selection_seconds"]))
            static_rows.append(dict(model=model, method=method, **info["metrics"],
                block_time_distribution_ns=info["rigid_block_time_distribution_ns"],
                parent_arithmetic=info["parent_arithmetic"],
                affected_weight_network_bytes_before=info["affected_weight_network_bytes_before"],
                affected_weight_network_bytes_after=info["affected_weight_network_bytes_after"],
                affected_activation_network_bytes_after=info["affected_activation_network_bytes_after"]))
            migrations.append(dict(model=model, method=method, **offline_migration(chosen_state, info)))
            load, traffic = structural_rows(model, method, metrics, state_bound, block_times(events, hp), bound, info)
            loads.append(load); communications.append(traffic)
        model_quality = {r["method"]: r for r in quality if r["model"] == model}
        l1_state = next((r for r in all_states if r["model"] == model and r["state"] == "L1"), None)
        if l1_state is not None and stop is None:
            current_l1.append(dict(model=model, phase_op=s["phase_op"], source_job=l1_state["job_id"],
                backend_mean_ns=l1_state["backend_mean_ns"], backend_min_ns=l1_state["backend_min_ns"],
                backend_max_ns=l1_state["backend_max_ns"], relative_to_healthy=l1_state["backend_mean_ns"]/healthy-1,
                relative_to_R0=l1_state["backend_mean_ns"]/r0mean-1, T_hat_ns=l1_state["T_hat_ns"],
                T_hat_relative_to_healthy=l1_state["T_hat_ns"]/pre_raw["record"]["T"]-1,
                final_rigid_EIR_status="pending"))
        for method in ("healthy", "R0", "rigid EIR", "Local-m1", "Split-m2"):
            arm_coverage.append(dict(model=model, method=method, status="complete" if method in model_quality else "pending"))
        if all(method in model_quality for method in ("Local-m1", "Split-m2")):
            left, right = (model_quality[method] for method in ("Local-m1", "Split-m2"))
            traffic = {r["method"]: r for r in communications if r["model"] == model}
            comparison = dict(model=model, phase_op=s["phase_op"], old_rigid_bound_ns=bound,
                backend_mean_ratio_m2_vs_m1=right["backend_mean_ns"]/left["backend_mean_ns"]-1,
                tensor_peak_ratio_m2_vs_m1=right["tensor_peak_ns"]/left["tensor_peak_ns"]-1,
                tensor_bound_ratio_m2_vs_m1=right["tensor_load_bound_ns"]/left["tensor_load_bound_ns"]-1,
                m1_backend_below_old_bound=left["backend_below_rigid_bound"], m2_backend_below_old_bound=right["backend_below_rigid_bound"],
                backend_samples_per_arm=10)
            for key in ("affected_activation_input_bytes_after", "route_link_bytes", "ddr_read_bytes"):
                comparison[key+"_m1"] = traffic["Local-m1"][key]
                comparison[key+"_m2"] = traffic["Split-m2"][key]
                comparison[key+"_ratio_m2_vs_m1"] = traffic["Split-m2"][key]/traffic["Local-m1"][key]-1
            if l1_state is not None:
                comparison.update(L1_backend_mean_ns=l1_state["backend_mean_ns"],
                    m1_backend_relative_to_L1=left["backend_mean_ns"]/l1_state["backend_mean_ns"]-1,
                    m2_backend_relative_to_L1=right["backend_mean_ns"]/l1_state["backend_mean_ns"]-1,
                    m1_backend_relative_to_R0=left["relative_to_R0"], m2_backend_relative_to_R0=right["relative_to_R0"])
            comparisons.append(comparison)

    summary = dict(status="running" if missing else "complete", planned_jobs=len(jobs), complete_jobs=len(present),
        missing=missing, actual_search_trajectories=sum(r["actual_search_trajectories"] for r in costs),
        actual_backend_calls=sum(r["actual_backend_calls"] for r in costs),
        reused_stop_jobs=sum(r["status"] == "reused" for r in costs),
        reused_healthy_jobs=sum(r.get("healthy_reused", False) for r in costs),
        process_hours=sum(r["wall_seconds"] for r in costs)/3600,
        task_body_seconds=sum(r["task_body_seconds"] for r in costs),
        startup_wrapper_poll_seconds=sum(r["startup_wrapper_poll_seconds"] for r in costs),
        task_cost_component_totals={key: sum(r[key] for r in costs) for key in COST_COMPONENTS},
        task_partition_max_abs_error_seconds=max((abs(r["task_partition_error_seconds"]) for r in costs), default=0),
        receipt_partition_max_abs_error_seconds=max((abs(r["receipt_partition_error_seconds"]) for r in costs), default=0),
        maximum_job_seconds=max((r["wall_seconds"] for r in costs), default=0),
        peak_rss_gb=max((r["peak_rss_gb"] for r in costs), default=0),
        coverage=coverage, quality=quality, manifest_sha256=sha(run.MANIFEST),
        newly_run_by_report=dict(backend_calls=0, search_trajectories=0))
    cost_components = [dict(job_id=row["job_id"], model=row["model"], kind=row["kind"], component=key, seconds=row[key])
                       for row in costs for key in COST_COMPONENTS]
    return summary, dict(c_quality=quality, c_backend_samples=samples, c_all_EIR_states=all_states,
        c_costs=costs, c_stops=stops, c_trajectories=trajectories, c_static=static_rows,
        c_axis_choices=choices, c_migration=migrations, c_block_load=loads, c_communication=communications,
        c_split_comparison=comparisons, c_arm_coverage=arm_coverage, c_cost_components=cost_components,
        c_current_L1=current_l1), dict(job_checks=checks, raw_inventory=inventories,
        cost_accounting=dict(status="passed", jobs=len(costs), exclusive_task_components=list(COST_COMPONENTS),
            equations=["sum(exclusive_task_components) = task_body_seconds", "task_body_seconds + startup_wrapper_poll_seconds = receipt wall_seconds"],
            inherited_R0_L1_and_finalize_inclusive_are_informational_only=True,
            reconstructed_receipt_jobs=[r["job_id"] for r in costs if r["receipt_reconstructed_from_task"]]))


def plot_figures(summary, tables):
    comparisons = {r["model"]: r for r in tables["c_split_comparison"]}
    if set(comparisons) != set(run.hi_config.MODELS):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = run.DEST / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    models = ("llama", "qwen3moe")
    names = dict(llama="Llama", qwen3moe="Qwen3-MoE")
    methods = ("healthy", "R0", "rigid EIR", "Local-m1", "Split-m2")
    colors = ("#8c939b", "#d98939", "#43856a", "#427ca5", "#7b63a3")
    coverage = {r["model"]: r for r in summary["coverage"]}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "figure.facecolor": "white"})
    files = []
    def save(fig, name):
        for suffix in ("png", "svg"):
            path = folder / (name + "." + suffix)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            files.append(dict(path=path.relative_to(run.DEST).as_posix(), sha256=sha(path)))
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, model in zip(axes, models):
        quality = {r["method"]: r for r in summary["quality"] if r["model"] == model}
        top = max(r["backend_max_ns"] for r in quality.values()) / 1e6
        for index, method in enumerate(methods):
            if method not in quality:
                ax.text(index, top * .035, "Pending", ha="center", va="bottom", fontsize=9, rotation=90)
                continue
            row = quality[method]
            mean, low, high = (row[k] / 1e6 for k in ("backend_mean_ns", "backend_min_ns", "backend_max_ns"))
            ax.bar(index, mean, width=.64, color=colors[index],
                   yerr=[[mean-low], [high-mean]], error_kw=dict(capsize=3, lw=1, ecolor="#343434"))
            ax.text(index, high + .025*top, f"{mean:.1f}", ha="center", va="bottom", fontsize=9)
        bound = comparisons[model]["old_rigid_bound_ns"] / 1e6
        ax.axhline(bound, color="#555555", linestyle="--", lw=1, label="Post-fault original-block bound")
        ax.set_xticks(range(len(methods)), ("Healthy", "R0", "Rigid\nEIR", "Local-m1", "Split-m2"))
        ax.set_ylim(0, top*1.18)
        ax.set_ylabel("Backend latency (million ns)")
        cov = coverage[model]
        ax.set_title(f"{names[model]} ({cov['complete_jobs']}/{cov['planned_jobs']} jobs)")
        ax.grid(axis="y", alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Ten backend seeds per completed state; whiskers show min–max", fontsize=11)
    fig.tight_layout()
    save(fig, "c_backend_latency")
    labels = ("Busiest Tensor", "Tensor lower bound", "Affected activation\ninput bytes", "All route-link bytes", "Backend mean")
    keys = ("tensor_peak_ratio_m2_vs_m1", "tensor_bound_ratio_m2_vs_m1",
            "affected_activation_input_bytes_after_ratio_m2_vs_m1", "route_link_bytes_ratio_m2_vs_m1", "backend_mean_ratio_m2_vs_m1")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, model in zip(axes, models):
        values = [comparisons[model][k] * 100 for k in keys]
        ax.barh(range(len(keys)), values, color=["#427ca5" if v<=0 else "#d98939" for v in values], height=.57)
        span = max(20, max(abs(v) for v in values))
        for i, value in enumerate(values):
            ax.text(value + (span*.025 if value>=0 else -span*.025), i, f"{value:+.2f}%",
                    ha="left" if value>=0 else "right", va="center", fontsize=9)
        ax.set_yticks(range(len(labels)), labels);ax.invert_yaxis()
        ax.set_xlim(min(values)-span*.5, max(values)+span*.4)
        ax.axvline(0, color="#555555", lw=1)
        ax.grid(axis="x", alpha=.2);ax.set_axisbelow(True)
        ax.set_title(names[model]);ax.set_xlabel("Change from Local-m1 (%)")
    fig.suptitle("Split-m2: saved compute-load and communication changes", fontsize=11)
    fig.tight_layout()
    save(fig, "c_split_tradeoff")
    return files


def report(out, progress=False):
    summary, tables, evidence = collect(out, progress)
    for name, rows in tables.items():
        table(run.DEST / "tables" / (name + ".csv"), rows)
    figures = plot_figures(summary, tables)
    run.frozen.write_json(run.DEST / "evidence/c_completion.json", dict(summary=summary, **evidence, figures=figures))
    text = ["# 两个新模型的 MF6 与局部切分", "", "本报告只读取保存结果；不调用后端或搜索。", "",
        f"已完成 {summary['complete_jobs']}/{summary['planned_jobs']} 个作业；实际新增后端 {summary['actual_backend_calls']} 次、L2 搜索 {summary['actual_search_trajectories']} 条。停档复用 {summary['reused_stop_jobs']} 项、健康状态复用 {summary['reused_healthy_jobs']} 项。", "",
        "故障由健康 A3h decode MHA 的一次 seed1 后端选定：非角落 PE 的 Tensor/Vector 忙碌时间最大者，平局按坐标。PE 的计算单元失效，路由器与链路保留。随后按预先固定的 FFN、projection 顺序选择有孤儿 Tensor 且可合法二分的算子，没有用修复结果筛选位置或算子。", ""]
    for row in summary["coverage"]:
        text.append(f"- {row['model']}：{row.get('phase_op') or '尚无合法代表点'}，已完成 {row.get('complete_jobs', 0)}/{row.get('planned_jobs', 45)} 项。")
    pending = [r for r in tables["c_arm_coverage"] if r["status"] == "pending"]
    if pending:
        text += ["", "待完成状态：" + "、".join(f"{r['model']} {r['method']}" for r in pending) + "。下表仅纳入已有十个后端种子的状态；未完成 EIR 不提前选取修复结果。"]
    text += ["", "| 模型/算子 | 状态 | 后端均值ns [最小,最大] | 相对健康 | 相对R0 | 低于故障后原固定块下界的样本 |", "|---|---|---:|---:|---:|---|"]
    for r in summary["quality"]:
        relative = "—" if r["relative_to_R0"] is None else f"{r['relative_to_R0']:+.2%}"
        below = "—（健康平台）" if r["method"] == "healthy" else f"{r['backend_below_rigid_bound']}/10"
        text.append(f"| {r['model']}/{r['phase_op']} | {r['method']} | {r['backend_mean_ns']:,.1f} [{r['backend_min_ns']:,}, {r['backend_max_ns']:,}] | {r['relative_to_healthy']:+.2%} | {relative} | {below} |")
    if tables["c_current_L1"]:
        text += ["", "正在继续的刚性 EIR：已完成 L1 阶段", "",
            "| 模型 | L1后端均值ns [最小,最大] | 相对健康 | 相对R0 | L1 CPCM ns | CPCM相对健康 | 最终EIR状态 |",
            "|---|---:|---:|---:|---:|---:|---|"]
        for r in tables["c_current_L1"]:
            text.append(f"| {r['model']} | {r['backend_mean_ns']:,.1f} [{r['backend_min_ns']:,}, {r['backend_max_ns']:,}] | {r['relative_to_healthy']:+.2%} | {r['relative_to_R0']:+.2%} | {r['T_hat_ns']:,} | {r['T_hat_relative_to_healthy']:+.2%} | 后续L2进行中 |")
        text += ["", "这一行保留已经完成的 L1 十种子评估；最终刚性 EIR 行等待既定 30/100/300 档停止规则给出结果。"]
    text += ["", "rigid EIR 按原 2% 及 30/100/300 独立预算停止，内部五条修复轨迹和每条十次后端全部计入成本；上表取原停止档的后端中位代表，完整五种子结果另表保存。Local-m1 与 Split-m2 从同一健康图的故障孤儿块出发，用同一放置和路径规则，健康核原计算落点不变。", "",
        "Split 仅比较合法的 N、M 轴，依次最小化实际取整 FLOP 增量、完整图链路字节、固定轴顺序。表中保存所有候选轴、实际输入复制、输出区域覆盖和取整变化；不假设通信按二分比例减少。", "",
        "固定块下界按真实块时长和故障后可用 Tensor 单元计算：max(max c、sum c/P、ceil(K/P) min c)。同时报告新图下界与最忙核负载。只有后端总延迟实际低于旧块下界，才构成超出旧分块放置可达到范围的证据。", "",
        "唯一权重迁移字节沿用 EIR 的 DDR 来源新增副本定义；局部切分保留原 DDR 源，额外 PE 输入复制和链路传输另列。健康输入和全体原始候选未修改。", "",
        "### 实际块大小、下界和计算负载", "",
        "服务时间由保存图里的真实 Tensor 形状及当前硬件计算。原固定块下界使用故障后 63 个 Tensor 单元；健康行使用 64 个。计数项为 ceil(K/P) × 最短块，基本项包含最长块及总工作量/P，两者取较大值。", "",
        "| 模型 | 状态 | Tensor块数 | 存活单元 | 块时长最小/最大（Mns） | 基本下界（Mns） | 计数下界（Mns） | 实际分块下界（Mns） | 最忙Tensor负载（Mns） |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in tables["c_block_load"]:
        text.append(f"| {r['model']} | {r['method']} | {r['tensor_blocks']} | {r['live_tensor_units']} | {r['block_time_min_ns']/1e6:.3f}/{r['block_time_max_ns']/1e6:.3f} | {r['basic_lower_bound_ns']/1e6:.3f} | {r['count_lower_bound_ns']/1e6:.3f} | {r['current_tensor_lower_bound_ns']/1e6:.3f} | {r['busiest_tensor_ns']/1e6:.3f} |")
    text += ["", "### 通信与复制", "",
        "下表用十进制 MB/GB（1 MB = 10⁶字节，1 GB = 10⁹字节）。route-link 字节对每条实际链路计数；受影响输入字节只统计局部改图部分，不与全图 DDR 读取相加。复制量是实际传输，不表示新增独立权重副本，也不表示峰值 SRAM 占用。", "",
        "| 模型 | 状态 | 全图逻辑传输（GB） | 全图route-link（GB） | DDR读取（GB） | 受影响权重输入（MB） | 受影响激活输入（MB） |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in tables["c_communication"]:
        weight = f"{r['affected_weight_input_bytes_after']/1e6:.3f}" if "affected_weight_input_bytes_after" in r else "—"
        activation = f"{r['affected_activation_input_bytes_after']/1e6:.3f}" if "affected_activation_input_bytes_after" in r else "—"
        text.append(f"| {r['model']} | {r['method']} | {r['logical_network_bytes']/1e9:.3f} | {r['route_link_bytes']/1e9:.3f} | {r['ddr_read_bytes']/1e9:.3f} | {weight} | {activation} |")
    text += ["", "### 从这两个保存点能看到什么", ""]
    for r in tables["c_split_comparison"]:
        text.append(f"- {r['model']}：m2 相对 m1 的最忙 Tensor 负载变化 {r['tensor_peak_ratio_m2_vs_m1']:+.2%}，实际分块下界变化 {r['tensor_bound_ratio_m2_vs_m1']:+.2%}；受影响激活输入变化 {r['affected_activation_input_bytes_after_ratio_m2_vs_m1']:+.2%}，全图 route-link 字节变化 {r['route_link_bytes_ratio_m2_vs_m1']:+.2%}。后端均值变化 {r['backend_mean_ratio_m2_vs_m1']:+.2%}，m1/m2 分别有 {r['m1_backend_below_old_bound']}/10、{r['m2_backend_below_old_bound']}/10 个样本低于旧分块下界 {r['old_rigid_bound_ns']/1e6:.3f} Mns。")
        if "L1_backend_mean_ns" in r:
            text.append(f"  与同点 R0 相比，m1/m2 的后端均值分别变化 {r['m1_backend_relative_to_R0']:+.2%}/{r['m2_backend_relative_to_R0']:+.2%}；与已完成 L1 相比分别为 {r['m1_backend_relative_to_L1']:+.2%}/{r['m2_backend_relative_to_L1']:+.2%}。m1 与 m2 的差值单独反映本次二分带来的变化。")
    for model in run.hi_config.MODELS:
        group = {r["method"]: r for r in tables["c_block_load"] if r["model"] == model}
        if "Local-m1" not in group or "Split-m2" not in group:
            continue
        left, right = group["Local-m1"], group["Split-m2"]
        before = "块数及最短块形成的计数项" if left["count_lower_bound_ns"] > left["basic_lower_bound_ns"] else "总工作量/P与最长块形成的基本项"
        after = "计数项" if right["count_lower_bound_ns"] > right["basic_lower_bound_ns"] else "基本项"
        text.append(f"{model} 的原图有 {left['tensor_blocks']} 个 Tensor 块，块时长分为 {len(left['block_time_distribution_ns'])} 档。m1 下界由{before}决定；m2 把块数增加到 {right['tensor_blocks']} 后，下界由{after}决定，最忙 Tensor 实际负载为 {right['busiest_tensor_ns']/1e6:.3f} Mns。计算负载、输入复制和真实后端时间一起说明该点的切分效果。")
        final = next((r for r in tables["c_stops"] if r["model"] == model and r["budget"] == 300), None)
        if final is not None:
            q = {r["method"]: r for r in summary["quality"] if r["model"] == model}
            reason = {"within_epsilon": "回到健康值的 2% 以内", "tier_gain": "相邻两档改进小于 2%", "budget": "达到 300 步预算"}[final["reason"]]
            text.append(f"{model} 的刚性 EIR 最终采用 {final['level']}，停止原因是{reason}；该档五条轨迹的 CPCM 均值为 {final['stop_mean_T_hat_ns']/1e6:.3f} Mns，较健康 {final['stop_mean_T_hat_ns']/final['T_pre_ns']-1:+.2%}。十次后端均值的中位代表为修复 seed{final['representative_repair_seed']}，其后端均值 {q['rigid EIR']['backend_mean_ns']/1e6:.3f} Mns。m1、m2 相对这一最终 EIR 的后端均值分别变化 {q['Local-m1']['backend_mean_ns']/q['rigid EIR']['backend_mean_ns']-1:+.2%}、{q['Split-m2']['backend_mean_ns']/q['rigid EIR']['backend_mean_ns']-1:+.2%}。")
            if model == "llama":
                text.append("Llama 的二分降低了最忙核负载及分块下界，并在十个后端样本中的一个样本达到旧块下界以下；其后端均值仍高于该下界。这里体现的是切分扩大可实现的范围，而非十个种子一致突破。")
            elif model == "qwen3moe":
                text.append("Qwen3-MoE 的 m1 已把最忙核负载恢复到健康图的水平；二分增加了块数和激活传输，却没有继续降低最忙核负载或总工作量下界。m2 与 m1、最终 EIR 的后端均值相差都不到 0.3%，各自十个样本的取值范围重叠；本点主要体现局部搬移的恢复作用，未观察到二分带来的额外收益。")
    text += [""]
    if figures:
        text += ["![十种子后端延迟](figures/c_backend_latency.png)", "",
            "柱为十种子均值，线段为最小至最大值；虚线为故障后旧分块下界。" + ("Pending 表示该状态尚未收齐。" if pending else ""), "",
            "![切分的计算与通信变化](figures/c_split_tradeoff.png)", "",
            "所有变化以同模型 Local-m1 为参照。对应 SVG 与 PNG 一并保存在 figures；绝对值与逐种子样本在表中。", ""]
    text += ["### 运行成本逐项相加", "",
        "任务体采用互不重叠的计时项。EIR 的 final_cpcm_seconds 读取 raw.record.t_finalize；raw.timing.t_finalize 已包含恢复最佳状态，只作为说明列，不能再相加。继承的 R0/L1 时间以及复用的健康评估时间均不计作本次新增成本。", "",
        "| 类型 | 作业数 | 后端及复制准备秒 | R0/L1秒 | 校准与搜索秒 | 恢复及最终CPCM秒 | 全轴切分构图秒 | 其他任务体秒 | 任务体总秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in dict.fromkeys(r["kind"] for r in tables["c_costs"]):
        rows = [r for r in tables["c_costs"] if r["kind"] == kind]
        total = lambda *keys: sum(r[k] for r in rows for k in keys)
        text.append(f"| {kind} | {len(rows)} | {total('backend_seconds','backend_copy_prepare_seconds'):.3f} | {total('R0_prepare_seconds','L1_search_seconds'):.3f} | {total('calibration_seconds','search_seconds'):.3f} | {total('restore_seconds','final_cpcm_seconds','evaluation_CPCM_seconds'):.3f} | {total('split_axis_construction_seconds'):.3f} | {total('other_task_seconds'):.3f} | {total('task_body_seconds'):.3f} |")
    text += ["", "后端明细中，EIR 的直接调用与复制准备分开计时；Split 保存的后端时间段包含调用参数里的平台复制，事件复制归入其他任务体。未单独计时的锚点后端、输入加载、快照导出、检查及记录都保留在其他任务体。切分构图列相加全部实际候选轴的构造时间，每次只计入准备作业，不随十个后端样本重复收费。", "",
        "| 模型 | 作业数 | 任务体秒 | 启动、包装及轮询差秒 | 回执总秒 |",
        "|---|---:|---:|---:|---:|"]
    for model in run.hi_config.MODELS:
        rows = [r for r in tables["c_costs"] if r["model"] == model]
        text.append(f"| {model} | {len(rows)} | {sum(r['task_body_seconds'] for r in rows):.3f} | {sum(r['startup_wrapper_poll_seconds'] for r in rows):.3f} | {sum(r['wall_seconds'] for r in rows):.3f} |")
    text += ["", f"逐行核对：全部 {len(tables['c_costs'])} 项的任务体分项之和等于任务体总时，再加启动、包装及轮询差等于回执总时。最大绝对差分别为 {summary['task_partition_max_abs_error_seconds']:.3g} 秒、{summary['receipt_partition_max_abs_error_seconds']:.3g} 秒；未四舍五入的值在 c_costs.csv 和 c_cost_components.csv。", "",
        f"累计作业成本 {summary['process_hours']:.4f} 进程小时，最长作业 {summary['maximum_job_seconds']/60:.2f} 分钟，最高 RSS {summary['peak_rss_gb']:.3f} GiB。这是并行质量运行成本；正式单进程计时由 J 报告。", "",
        "复算：`python e2e/hi_closeout_c_report.py --out e2e/hi/out/closeout_20261115`。逐种子样本、全部 EIR 档位、停止决策、通信/下界、轴选择、迁移及逐项成本在 `tables/c_*.csv`；输入/回执/原始状态摘要在 `evidence/c_completion.json`。", ""]
    (run.DEST / "C_NEW_MODEL_SPLIT.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("coverage", "quality", "missing")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=run.OUT)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    report(args.out, args.progress)
