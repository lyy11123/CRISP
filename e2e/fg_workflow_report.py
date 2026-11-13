"""Tabulate saved F/G fault repair and P/D experiments without running simulations.

python -B e2e/fg_workflow_report.py --run-out e2e/fg/out/main --out e2e/fg/tables
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import pickle
import statistics

from e_run import ROOT, write_json


CONFIG = ("config_id", "stage", "hardware_profile", "shape", "split_mode", "d2d_scale")
SEEDS = (123, 1, 2, 3, 4)
TIERS = (30, 100, 300)
MIGRATION = ("moved_behas", "n_behas", "moved_behas_share", "ddr_read_moved_bytes",
             "ddr_read_total_bytes", "ddr_read_moved_share", "unique_block_bytes", "unique_blocks")
FAULT_CHANGES = ("backend_change_vs_healthy", "backend_change_vs_A3h_R0", "backend_gap_vs_FULL")
JOINT_METRICS = ("P_completion_ns", "D_latency_ns", "paired_delta_P_ns", "P_solo_ns", "D_solo_ns",
                 "P_self_amplification", "D_self_amplification", "P_common_ref_ratio", "D_common_ref_ratio",
                 "Lhat_P", "Lhat_D", "Lhat_max", "KV_D_interval_overlap_ns", "D_start_delay_ns",
                 "D_latency_change_vs_B_R0", "P_completion_change_vs_B_R0", "D_latency_change_vs_CRISP",
                 "P_completion_change_vs_CRISP", "wall_seconds", "simulation_seconds", "peak_rss_gb")
JOINT_METRICS += ("KV_start_ns", "KV_end_ns", "actual_D_start_ns", "actual_D_end_ns",
                  "KV_interval_overlap_fraction", "simulation_total_ns", "unhidden_compute_ns",
                  "unhidden_communication_ns")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def config(spec):
    return {key: spec[key] for key in CONFIG}


def stats(values, prefix):
    values = list(values)
    if not values:
        return {prefix + "_" + key: None for key in ("mean", "median", "min", "max")}
    return {prefix + "_mean": statistics.mean(values), prefix + "_median": statistics.median(values),
            prefix + "_min": min(values), prefix + "_max": max(values)}


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6)


def write_table(out, name, rows):
    write_json(out / (name + ".json"), rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def load_results(jobs, run_out, allow_incomplete):
    records = {p.stem: read(p) for p in (run_out / "results").glob("*.json")}
    missing = [name for name in jobs if name not in records]
    if not allow_incomplete:
        assert not missing, missing
    for name, job in jobs.items():
        if name not in records:
            continue
        row = records[name]
        assert row["status"] in ("ok", "reused"), (name, row["status"])
        if row["status"] != "reused":
            spec = row if job["kind"] == "search" else row["spec"]
            assert all(spec[key] == value for key, value in job["spec"].items()), name
    return records, missing


def raw_details(records, jobs, run_out, skip_raw):
    """Keep numeric backend samples, not the large restored objects."""
    if skip_raw:
        return {}, []
    import e2e_pilot
    e2e_pilot.C._artifact_path()
    details, samples = {}, []
    for name, job in jobs.items():
        if name not in records or job["kind"] not in ("fault_pre", "fault_l1", "fault_rep"):
            continue
        row = records[name]
        if row["status"] == "reused":
            continue
        with gzip.open(run_out / "raw" / (name + ".pkl.gz"), "rb") as file:
            raw = pickle.load(file)
        s = job["spec"]
        states = {"healthy": "record"} if job["kind"] == "fault_pre" else (
            {"R0": "R0", **({"L1": "L1"} if s["start_arm"] == "A3h" else {})}
            if job["kind"] == "fault_l1" else {"s" + str(s["steps"]): "record"})
        detail = {"states": {}, "state_timing": {},
                  "timing": {key: raw[key] for key in ("t_R0", "t_L1") if key in raw}}
        for state, key in states.items():
            backend = raw[key]["backend"]
            assert sorted(map(int, backend)) == list(range(1, 11)), name
            detail["states"][state] = {int(seed): tuple(triple) for seed, triple in backend.items()}
            detail["state_timing"][state] = dict(t_finalize=raw[key]["t_finalize"],
                                                t_backend_offline=raw[key]["t_backend"])
            for seed, triple in sorted(backend.items()):
                total, compute, communication = triple
                samples.append(dict(config(s), job_id=name, phase_op=s["phase_op"],
                    fault_id=s.get("fault_id", "healthy"), start_arm=s["start_arm"], state=state,
                    sa_seed=s.get("seed"), backend_seed=int(seed), backend_ns=total,
                    unhidden_compute_ns=compute, unhidden_communication_ns=communication,
                    total_minus_unhidden_components_ns=total - compute - communication, source_format="raw"))
        details[name] = detail
        del raw
    return details, samples


def backend_values(details, name, state):
    return [triple[0] for triple in details[name]["states"][state].values()] if name in details else []


def fault_tables(jobs, records, details, allow_incomplete):
    configurations, summary, seeds, tiers, stops = [], [], [], [], []
    full = defaultdict(list)
    for row in records.values():
        if row["kind"] == "search" and row["family"] == "full":
            full[row["config_id"], row["phase"], row["op"]].append(row)
    full_best = {}
    for key, rows in full.items():
        if len(rows) == 14:
            assert {r["candidate_order"] for r in rows} == set(range(14))
            full_best[key] = min(rows, key=lambda r: (statistics.mean(x["cycles"] for x in r["backend"]), r["candidate_order"]))
    for name, job in jobs.items():
        if name not in records:
            continue
        s, r = job["spec"], records[name]
        if job["kind"] == "fault_select":
            used = {j["spec"]["fault_id"] for j in jobs.values()
                    if j["kind"] == "fault_l1" and j["spec"]["config_id"] == s["config_id"]}
            for fid in sorted(used):
                configurations.append(dict(config(s), fault_id=fid, **r["faults"][fid]))
        elif job["kind"] == "fault_rep":
            row = dict(config(s), job_id=name, phase_op=s["phase_op"], fault_id=s["fault_id"],
                       steps=s["steps"], sa_seed=s["seed"], status=r["status"],
                       wall_seconds=r["wall_seconds"], previous_stop=s["previous_stop"])
            if r["status"] == "ok":
                row.update(T_hat_ns=r["T"], backend_mean_ns=r["backend_mean"],
                           migration=r["migration"], timing=r["timing"], peak_rss_gb=r["peak_rss_gb"])
            tiers.append(row)
        elif job["kind"] == "fault_stop":
            stops.append(dict(config(s), job_id=name, phase_op=s["phase_op"], fault_id=s["fault_id"],
                requested_steps=s["steps"], stop=r["stop"], level=r["level"], reason=r["reason"],
                T_hat_ns=r["T"], T_pre_ns=r["T_pre"], T_hat_change_vs_pre=r["T"] / r["T_pre"] - 1,
                gain_from_previous=r.get("gain_from_previous"), source_job=r["source_job"],
                representative_seed=r.get("seed"), representative_backend_mean_ns=r.get("backend_mean")))
    targets = [j for j in jobs.values() if j["kind"] == "fault_stop" and j["spec"]["steps"] == 300]
    for target in targets:
        s = target["spec"]
        cid, op, fid = s["config_id"], s["phase_op"], s["fault_id"]
        phase, operator = ("prefill" if op.startswith("pf_") else "decode"), op.split("_", 1)[1]
        names = {arm: (f"{cid}__pre_{op}_{arm}", f"{cid}__l1_{op}_{arm}_{fid}") for arm in ("A1", "A3h")}
        required = [target["id"], *(n for pair in names.values() for n in pair)]
        if not all(n in records for n in required):
            assert allow_incomplete
            continue
        final = records[target["id"]]
        context = dict(config(s), phase=phase, op=operator, phase_op=op, fault_id=fid)
        healthy = {arm: records[pair[0]] for arm, pair in names.items()}
        l1 = {arm: records[pair[1]] for arm, pair in names.items()}
        r0_a3h = l1["A3h"]["R0_backend_mean"]
        full_row = full_best.get((cid, phase, operator)) if fid == "PF4" else None
        full_mean = statistics.mean(v["cycles"] for v in full_row["backend"]) if full_row else None
        cells = []
        for arm in ("A1", "A3h"):
            record = l1[arm]
            cells.append((arm + "-R0", arm, [dict(job_id=record["job_id"], sa_seed=None,
                level="R0", T=record["T_R0"], backend_mean=record["R0_backend_mean"],
                migration=record["migration_R0"], backend_values=backend_values(details, record["job_id"], "R0"))]))
        level = final["level"]
        if level in ("R0", "L1"):
            record = l1["A3h"]
            finals = [dict(job_id=record["job_id"], sa_seed=None, level=level, T=record["T_" + level],
                backend_mean=record[level + "_backend_mean"],
                migration=record["migration_R0" if level == "R0" else "migration"],
                backend_values=backend_values(details, record["job_id"], level))]
        else:
            finals = []
            for seed in SEEDS:
                record = records[f"{cid}__rep_{op}_{fid}_{level}_r{seed}"]
                assert record["status"] == "ok"
                finals.append(dict(job_id=record["job_id"], sa_seed=seed, level=level, T=record["T"],
                    backend_mean=record["backend_mean"], migration=record["migration"], timing=record["timing"],
                    backend_values=backend_values(details, record["job_id"], level)))
            assert close(statistics.mean(r["T"] for r in finals), final["T"])
            representative = min((r for r in finals if r["backend_mean"] == statistics.median(x["backend_mean"] for x in finals)),
                                 key=lambda r: r["sa_seed"])
            assert representative["job_id"] == final["source_job"]
        cells.append(("EIR", "A3h", finals))
        if full_row:
            cells.append(("A3h-FULL", "A3h", [dict(job_id=full_row["candidate_id"], sa_seed=full_row["x"],
                level="FULL", T=full_row["T_hat"], backend_mean=full_mean,
                backend_values=[v["cycles"] for v in full_row["backend"]])]))
        actual_tiers = [records[f"{cid}__rep_{op}_{fid}_s{steps}_r{seed}"] for steps in TIERS for seed in SEEDS
                        if f"{cid}__rep_{op}_{fid}_s{steps}_r{seed}" in records
                        and records[f"{cid}__rep_{op}_{fid}_s{steps}_r{seed}"]["status"] == "ok"]
        for scheme, arm, states in cells:
            values = [r["backend_mean"] for r in states]
            samples = [v for r in states for v in r["backend_values"]]
            row = dict(context, scheme=scheme,
                source_candidate=full_row["candidate_id"] if scheme == "A3h-FULL" else healthy[arm]["source_candidate"],
                healthy_source_candidate=healthy[arm]["source_candidate"],
                healthy_job=healthy[arm]["job_id"], healthy_backend_mean_ns=healthy[arm]["backend_mean"],
                healthy_T_hat_ns=healthy[arm]["T"], backend_mean_ns=statistics.mean(values),
                backend_median_ns=statistics.median(values), state_backend_min_ns=min(values), state_backend_max_ns=max(values),
                n_sa_states=len(states), sa_seeds=[r["sa_seed"] for r in states if r["sa_seed"] is not None],
                source_jobs=[r["job_id"] for r in states],
                T_hat_mean_ns=statistics.mean(r["T"] for r in states),
                backend_change_vs_healthy=statistics.mean(values) / healthy[arm]["backend_mean"] - 1,
                backend_change_vs_matched_R0=statistics.mean(values) / l1[arm]["R0_backend_mean"] - 1,
                backend_change_vs_A3h_R0=statistics.mean(values) / r0_a3h - 1,
                backend_gap_vs_FULL=statistics.mean(values) / full_mean - 1 if full_mean else None,
                FULL_backend_mean_ns=full_mean, backend_sample_count=len(samples),
                backend_sample_min_ns=min(samples) if samples else None, backend_sample_max_ns=max(samples) if samples else None,
                state_statistics="five SA seed means" if len(states) == 5 else "one saved state")
            for metric in MIGRATION:
                m = [r["migration"][metric] for r in states if "migration" in r]
                row.update(stats(m, metric))
            if scheme == "EIR":
                selected = next(r for r in states if r["job_id"] == final["source_job"])
                if level in ("R0", "L1") and final["source_job"] in details:
                    times = details[final["source_job"]]["timing"]
                    selected["timing"] = dict(t_R0=times["t_R0"], t_L1=times["t_L1"] if level == "L1" else 0,
                        t_decision=times["t_R0"] + (times["t_L1"] if level == "L1" else 0),
                        **details[final["source_job"]]["state_timing"][level])
                row.update(stop_level=level, stop_reason=final["reason"], stop_T_hat_ns=final["T"],
                    representative_job=final["source_job"], representative_sa_seed=selected["sa_seed"],
                    representative_backend_mean_ns=selected["backend_mean"],
                    selected_search_steps=int(level[1:]) if level.startswith("s") else 0,
                    executed_L2_jobs=len(actual_tiers), executed_L2_steps=sum(r["steps"] for r in actual_tiers),
                    executed_L2_search_seconds=sum(r["timing"]["t_L2_search"] for r in actual_tiers))
                for metric in ("t_R0", "t_L1", "t_calibration", "t_L2_search", "t_finalize", "t_backend_offline", "t_decision"):
                    row.update(stats([r["timing"][metric] for r in states if metric in r.get("timing", {})], "selected_" + metric + "_seconds"))
            summary.append(row)
            if scheme == "EIR":
                for state in states:
                    seeds.append(dict(context, scheme=scheme, source_job=state["job_id"], sa_seed=state["sa_seed"],
                        stop_level=level, stop_reason=final["reason"], representative=state["job_id"] == final["source_job"],
                        T_hat_ns=state["T"], backend_mean_ns=state["backend_mean"],
                        backend_change_vs_healthy=state["backend_mean"] / healthy[arm]["backend_mean"] - 1,
                        backend_sample_count=len(state["backend_values"]),
                        **stats(state["backend_values"], "backend_sample_ns"), **state["migration"],
                        timing=state.get("timing", {})))
    return dict(fault_configs=configurations, fault_summary=summary, fault_seed_results=seeds,
                fault_tiers=tiers, fault_stops=stops,
                fault_aggregate=fault_aggregate(summary), fault_layers=fault_layers(summary))


def fault_aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["config_id"], row["fault_id"], row["scheme"]].append(row)
    output = []
    for (_, fault, scheme), cells in sorted(groups.items()):
        row = dict(config(cells[0]), fault_id=fault, scheme=scheme, operator_count=len(cells))
        for metric in FAULT_CHANGES:
            row.update(stats([c[metric] for c in cells if c[metric] is not None], metric))
        row["stop_levels"] = dict(Counter(c["stop_level"] for c in cells if "stop_level" in c))
        row["stop_reasons"] = dict(Counter(c["stop_reason"] for c in cells if "stop_reason" in c))
        output.append(row)
    return output


def fault_layers(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["config_id"], row["fault_id"], row["scheme"]][row["phase"], row["op"]] = row
    output = []
    for (_, fault, scheme), cells in sorted(groups.items()):
        if len(cells) != 8:
            continue
        row = dict(config(next(iter(cells.values()))), fault_id=fault, scheme=scheme,
                   assembly="sum of per-operator state means; fixed LN, eight MHA micrographs", ln_count="fixed")
        for phase in ("prefill", "decode"):
            weights = {"ln": 2 if phase == "prefill" else 1, "mha": 8, "proj": 1, "ffn": 1}
            row[phase + "_ns"] = sum(w * cells[phase, op]["backend_mean_ns"] for op, w in weights.items())
            row["healthy_" + phase + "_ns"] = sum(w * cells[phase, op]["healthy_backend_mean_ns"] for op, w in weights.items())
        for n in (512, 2048):
            row[f"request_{n}_ns"] = row["prefill_ns"] + n * row["decode_ns"]
            row[f"healthy_request_{n}_ns"] = row["healthy_prefill_ns"] + n * row["healthy_decode_ns"]
        for metric in ("prefill", "decode", "request_512", "request_2048"):
            row[metric + "_change_vs_healthy"] = row[metric + "_ns"] / row["healthy_" + metric + "_ns"] - 1
        output.append(row)
    lookup = {(r["config_id"], r["fault_id"], r["scheme"]): r for r in output}
    for row in output:
        for reference, suffix in (("A3h-R0", "A3h_R0"), ("A3h-FULL", "FULL")):
            other = lookup.get((row["config_id"], row["fault_id"], reference))
            for metric in ("prefill", "decode", "request_512", "request_2048"):
                row[metric + "_change_vs_" + suffix] = row[metric + "_ns"] / other[metric + "_ns"] - 1 if other else None
    return output


def joint_run(row, records):
    s, sim = row["spec"], row["simulation"]
    ref = records[s["reference_job"]]
    result = dict(config(s), job_id=row["job_id"], condition=s["condition"], arm=s["arm"], phase=s["phase"],
        seed=s["seed"], backend=s["backend"], policy=s["policy"], p_op=s.get("p_op", "mha"),
        source_config_id=s["source_config_id"], mapping_reuse=s["mapping_reuse"], graph_job=s["graph_job"],
        reference_job=s["reference_job"], P_solo_job=s["P_solo"], D_solo_job=s["D_solo"],
        tau_ns=row["tau"], reference_P_ns=ref["T_P"], reference_D_ns=ref["T_D"],
        reference_KV_start_ns=ref["KV_start"], reference_KV_end_ns=ref["KV_end"],
        P_deadline_ns=1.5 * ref["T_P"], D_deadline_ns=row["tau"] + 1.5 * ref["T_D"],
        actual_D_start_ns=sim["streams"]["1"]["first_start"], actual_D_end_ns=sim["streams"]["1"]["C"],
        wall_seconds=row["wall_seconds"], simulation_seconds=sim["wall"], peak_rss_gb=row["peak_rss_gb"],
        shared_P_D_links=row["shared"]["P_D_links"], shared_KV_D_links=row["shared"]["KV_D_links"],
        shared_compute=row["shared"]["compute"], audit_n=sim.get("audit", {}).get("n"), audit_bad=sim.get("audit", {}).get("bad"),
        simulation_total_ns=sim["out"][0], unhidden_compute_ns=sim["out"][1], unhidden_communication_ns=sim["out"][2],
        P_events=sim["streams"]["0"]["n"], D_events=sim["streams"]["1"]["n"])
    for key, value in row["metrics"].items():
        result[key + "_ns" if key in ("P_completion", "D_latency", "paired_delta_P", "P_solo", "D_solo") else key] = value
    result.update(P_common_ref_ratio=result["P_completion_ns"] / ref["T_P"], D_common_ref_ratio=result["D_latency_ns"] / ref["T_D"],
                  D_start_delay_ns=result["actual_D_start_ns"] - row["tau"])
    assert close(result["P_solo_ns"], records[s["P_solo"]]["simulation"]["streams"]["0"]["C"]), row["job_id"]
    assert close(result["D_solo_ns"], records[s["D_solo"]]["simulation"]["streams"]["1"]["C"]), row["job_id"]
    assert close(result["P_self_amplification"], result["P_completion_ns"] / result["P_solo_ns"]), row["job_id"]
    assert close(result["paired_delta_P_ns"], result["P_completion_ns"] - result["P_solo_ns"]), row["job_id"]
    assert close(result["D_latency_ns"], result["actual_D_end_ns"] - row["tau"]), row["job_id"]
    assert close(result["D_self_amplification"], result["D_latency_ns"] / result["D_solo_ns"]), row["job_id"]
    assert row["tau"] == round(ref["KV_start"] + s["phase"] * (ref["KV_end"] - ref["KV_start"])), row["job_id"]
    if "Lhat_D" in result:
        assert close(result["Lhat_D"], result["D_common_ref_ratio"] - 1.5)
        assert close(result["Lhat_P"], result["P_common_ref_ratio"] - 1.5)
    if s["policy"] == "C4":
        graph = records[s["graph_job"]]
        p_deadline, d_deadline = 1.5 * graph["that"]["0"], row["tau"] + 1.5 * graph["that"]["1"]
        result.update(C4_internal_P_deadline_ns=p_deadline, C4_internal_D_deadline_ns=d_deadline,
                      C4_first_stream=0 if p_deadline <= d_deadline else 1)
    if "kv_window" in sim:
        begin, end = sim["kv_window"]
        overlap = max(0, min(end, result["actual_D_end_ns"]) - max(begin, result["actual_D_start_ns"]))
        result.update(KV_start_ns=begin, KV_end_ns=end, KV_D_interval_overlap_ns=overlap,
                      KV_D_intervals_overlap=overlap > 0, KV_interval_overlap_fraction=overlap / (end - begin))
    if s["backend"] == "hop":
        assert result["audit_bad"] == 0, row["job_id"]
    return result


def joint_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["config_id"], row["condition"], row["arm"], row["phase"]].append(row)
    output = []
    for (_, condition, arm, phase), cells in sorted(groups.items()):
        first = cells[0]
        row = dict(config(first), condition=condition, arm=arm, phase=phase, seed_count=len(cells),
            seeds=sorted(c["seed"] for c in cells), backend=first["backend"], policy=first["policy"],
            source_config_id=first["source_config_id"], mapping_reuse=first["mapping_reuse"],
            tau_ns=first["tau_ns"], reference_P_ns=first["reference_P_ns"], reference_D_ns=first["reference_D_ns"],
            KV_D_intervals_overlap_count=sum(c.get("KV_D_intervals_overlap", False) for c in cells),
            audit_n=sum(c["audit_n"] for c in cells if c["audit_n"] is not None),
            audit_bad=sum(c["audit_bad"] for c in cells if c["audit_bad"] is not None))
        assert len({c["tau_ns"] for c in cells}) == 1
        for metric in JOINT_METRICS:
            row.update(stats([c[metric] for c in cells if metric in c], metric))
        output.append(row)
    return output


def joint_comparisons(rows):
    """Compare the three-seed cell means; keep this separate from mean paired ratios."""
    lookup = {(r["config_id"], r["condition"], r["phase"], r["arm"]): r for r in rows}
    comparisons = []
    for row in rows:
        if row["arm"] != "CRISP" or row["stage"] == "F_bw":
            continue
        for arm in ("B-R0", "-DSTU", "-priority", "-EIR"):
            ref = lookup.get((row["config_id"], row["condition"], row["phase"], arm))
            if ref is None:
                continue
            value = dict(config(row), condition=row["condition"], phase=row["phase"], method="CRISP", reference_arm=arm,
                seed_count=row["seed_count"], comparison="ratio of three-backend-seed cell means",
                Lhat_max_difference=row["Lhat_max_mean"] - ref["Lhat_max_mean"],
                paired_delta_P_difference_ns=row["paired_delta_P_ns_mean"] - ref["paired_delta_P_ns_mean"])
            for metric in ("D_latency", "P_completion", "P_solo", "D_solo"):
                value[metric + "_ratio_change"] = row[metric + "_ns_mean"] / ref[metric + "_ns_mean"] - 1
                value[metric + "_ns"] = row[metric + "_ns_mean"]
                value["reference_" + metric + "_ns"] = ref[metric + "_ns_mean"]
            comparisons.append(value)
    groups = defaultdict(list)
    for row in comparisons:
        for scope in ("ALL", row["stage"], row["config_id"]):
            groups[scope, row["reference_arm"]].append(row)
    summaries = []
    for (scope, reference), cells in sorted(groups.items()):
        row = dict(scope=scope, reference_arm=reference, cells=len(cells), method="CRISP",
                   comparison="distribution across condition/phase cells; each cell first averages three backend seeds")
        for metric in ("D_latency_ratio_change", "P_completion_ratio_change", "P_solo_ratio_change",
                       "Lhat_max_difference", "paired_delta_P_difference_ns"):
            values = [c[metric] for c in cells]
            row.update(stats(values, metric))
            row[metric + "_lower_count"] = sum(v < -1e-12 for v in values)
            row[metric + "_equal_count"] = sum(abs(v) <= 1e-12 for v in values)
            row[metric + "_higher_count"] = sum(v > 1e-12 for v in values)
        summaries.append(row)
    return comparisons, summaries


def joint_tables(records):
    runs, ffn, refs, graphs, solos = [], [], [], [], []
    for record in records.values():
        kind = record["kind"]
        if kind in ("joint", "joint_ffn_check"):
            row = joint_run(record, records)
            if kind == "joint_ffn_check":
                assert row["shared_P_D_links"] == row["shared_KV_D_links"] == row["shared_compute"] == 0
                ffn.append(row)
            else:
                runs.append(row)
        elif kind == "joint_reference":
            refs.append(dict(config(record["spec"]), job_id=record["job_id"], reference=record["reference"],
                T_P_ns=record["T_P"], T_D_ns=record["T_D"], KV_start_ns=record["KV_start"], KV_end_ns=record["KV_end"],
                backend_seeds=record["backend_seeds"]))
        elif kind == "joint_solo":
            s, sim = record["spec"], record["simulation"]
            stream = sim["streams"][str(s["stream"])]
            row = dict(config(s), job_id=record["job_id"], graph_job=s["graph_job"], condition=s["condition"],
                p_op=s.get("p_op", "mha"), backend=s["backend"], stream=s["stream"], seed=s["seed"],
                completion_ns=stream["C"], first_start_ns=stream["first_start"], events=stream["n"],
                unhidden_compute_ns=sim["out"][1], unhidden_communication_ns=sim["out"][2],
                wall_seconds=record["wall_seconds"], simulation_seconds=sim["wall"], peak_rss_gb=record["peak_rss_gb"],
                audit_n=sim.get("audit", {}).get("n"), audit_bad=sim.get("audit", {}).get("bad"))
            if "kv_window" in sim:
                row.update(KV_start_ns=sim["kv_window"][0], KV_end_ns=sim["kv_window"][1])
            solos.append(row)
        elif kind == "joint_graph":
            row = dict(config(record["spec"]), job_id=record["job_id"], condition=record["spec"]["condition"],
                p_op=record["spec"].get("p_op", "mha"), cache_key=record["cache_key"], status=record["status"],
                chain=record["chain"], chain_edges=record["chain_edges"], shared=record["shared"], kv=record["kv"],
                source_exports=record["source_exports"], files=record["files"], fault=record["fault"],
                platform_sha256=record["platform_sha256"],
                wall_seconds=record["wall_seconds"], peak_rss_gb=record["peak_rss_gb"])
            for stream, prefix in (("0", "P"), ("1", "D")):
                row.update({prefix + "_" + key: value for key, value in record["sizes"][stream].items()})
            graphs.append(row)
    lookup = {(r["config_id"], r["condition"], r["arm"], r["phase"], r["seed"]): r for r in runs}
    for row in runs:
        for reference, suffix in (("B-R0", "B_R0"), ("CRISP", "CRISP")):
            other = lookup.get((row["config_id"], row["condition"], reference, row["phase"], row["seed"]))
            if other:
                for metric in ("D_latency", "P_completion"):
                    row[metric + "_change_vs_" + suffix] = row[metric + "_ns"] / other[metric + "_ns"] - 1
    summaries = joint_summary(runs)
    comparisons, comparison_summary = joint_comparisons(summaries)
    solo_groups = defaultdict(list)
    for row in solos:
        solo_groups[row["graph_job"], row["backend"], row["stream"]].append(row)
    solo_summaries = []
    for (graph, backend, stream), rows in sorted(solo_groups.items()):
        row = dict(config(rows[0]), graph_job=graph, backend=backend, stream=stream,
            condition=rows[0]["condition"], p_op=rows[0]["p_op"], seed_count=len(rows), seeds=sorted(r["seed"] for r in rows))
        for metric in ("completion_ns", "wall_seconds", "simulation_seconds", "peak_rss_gb"):
            row.update(stats((r[metric] for r in rows), metric))
        row["completion_ns_range"] = row["completion_ns_max"] - row["completion_ns_min"]
        solo_summaries.append(row)
    bandwidth = [dict(r) for r in summaries if r["hardware_profile"] == "watos" and r["condition"] == "healthy" and r["phase"] == 0.5]
    baseline = {r["arm"]: r for r in bandwidth if r["d2d_scale"] == 1}
    for row in bandwidth:
        row["experiment"] = "fixed WATOS 1x mappings; only D2D bandwidth and matching common reference change"
        for metric in ("D_latency", "P_completion", "P_solo", "D_solo"):
            row[metric + "_ns_ratio_to_1x"] = row[metric + "_ns_mean"] / baseline[row["arm"]][metric + "_ns_mean"]
    return dict(joint_runs=runs, joint_summary=summaries, joint_references=refs, joint_graphs=graphs,
                joint_ffn_checks=ffn, joint_solos=solos, joint_solo_summary=solo_summaries,
                joint_comparisons=comparisons, joint_comparison_summary=comparison_summary, bandwidth_joint=bandwidth)


def summarize(manifest, run_out, out, allow_incomplete=False, skip_raw=False):
    items = read(manifest)
    jobs = {j["id"]: j for j in items}
    records, missing = load_results(jobs, run_out, allow_incomplete)
    details, backend = raw_details(records, jobs, run_out, skip_raw)
    tables = fault_tables(jobs, records, details, allow_incomplete) | joint_tables(records)
    for row in tables["fault_summary"]:
        if row["scheme"] != "A3h-FULL":
            continue
        candidate = records[row["source_candidate"]]
        for value in candidate["backend"]:
            total, compute, communication = value["triple"]
            backend.append(dict(config(row), job_id=candidate["candidate_id"], phase_op=row["phase_op"],
                fault_id=row["fault_id"], start_arm="A3h", state="FULL", sa_seed=candidate["x"],
                backend_seed=value["seed"], backend_ns=total, unhidden_compute_ns=compute,
                unhidden_communication_ns=communication, total_minus_unhidden_components_ns=total - compute - communication,
                source_format="result JSON"))
    tables["fault_backend_samples"] = backend
    for row in tables["fault_summary"]:
        if not skip_raw and row["scheme"] != "A3h-FULL":
            assert row["backend_sample_count"] == 10 * row["n_sa_states"], row
            assert close(row["backend_mean_ns"], statistics.mean(v["backend_ns"] for v in backend
                if v["job_id"] in row["source_jobs"] and v["state"] == (row["stop_level"] if row["scheme"] == "EIR" else "R0")))
    counts = {kind: dict(planned=sum(j["kind"] == kind for j in jobs.values()),
                        completed=sum(j["kind"] == kind and j["id"] in records for j in jobs.values()),
                        reused=sum(j["kind"] == kind and j["id"] in records and records[j["id"]]["status"] == "reused" for j in jobs.values()))
              for kind in sorted({j["kind"] for j in jobs.values()})}
    checks = dict(planned_jobs=len(jobs), completed_jobs=len(jobs) - len(missing), missing_jobs=missing,
        by_kind=counts, raw_backend_samples_read=not skip_raw, table_rows={k: len(v) for k, v in tables.items()},
        paired_joint_metrics_recomputed=True, stopping_representatives_recomputed=True,
        arbitration_bad=sum(r["audit_bad"] or 0 for r in tables["joint_runs"] + tables["joint_ffn_checks"]))
    summary = dict(sources=dict(manifest=str(manifest), run_output=str(run_out)), checks=checks,
        definitions=dict(time_unit="ns; 1 simulation tick = 1 ns", healthy="paired healthy injection from the same selected source",
            fault_backend_mean="mean of ten backend seeds per state; EIR L2 rows then average five SA state means",
            state_backend_min_max="range across SA state means, separate from backend_sample_min/max",
            EIR_representative="median backend mean among the five stopped-tier states, tie smallest SA seed; used only by joint simulation",
            fault_layer="sum of per-operator state means; prefill 2LN+8MHA+proj+FFN, decode LN+8MHA+proj+FFN",
            FULL="A3h PF4 only, best backend mean of fourteen 1000-step candidates",
            decision_timing="saved parallel-run elapsed measurements; serial paper timing remains stage J",
            early_stop_timing="R0/L1 t_decision follows the original EIR sum of t_R0 and applicable t_L1; final state evaluation and ten-seed backend time are listed separately",
            KV_overlap="intersection of the KV bounding interval and actual D first-start/end interval; not a per-link waiting trace",
            unhidden_components="original backend triple; not DDR/NoC/D2D queue attribution",
            bandwidth_joint="WATOS 1x mapping frozen across 0.5x/1x/2x bandwidth"),
        tables=tables)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_table(out, name, rows)
    write_json(out / "workflow_checks.json", checks)
    write_json(out / "workflow_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "e2e/fg/manifest/all_jobs.json")
    parser.add_argument("--run-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "e2e/fg/tables")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-raw", action="store_true", help="JSON-only progress snapshot; backend raw distributions remain empty")
    args = parser.parse_args()
    summary = summarize(args.manifest, args.run_out, args.out, args.allow_incomplete, args.skip_raw)
    checks = summary["checks"]
    print(json.dumps({key: value for key, value in checks.items() if key != "missing_jobs"}
                     | {"missing_job_count": len(checks["missing_jobs"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
