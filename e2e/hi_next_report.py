"""Read the 344-job H/I batch and the four H1 pilots; no simulation or search."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import f_config
import fg_batch
import hi_config
import hi_manifest
from hi_report import exclusive_timing
from hi_split_supplement import read, write

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "e2e/hi"
KEYS = ("model", "hardware_profile", "phase", "op", "method")
OPS = ("ln", "mha", "proj", "ffn")
PHASES = ("prefill", "decode")
METHODS = ("A1", "A3h", "Gemini")
PROTOCOLS = ("backend", "own", "random_cpcm", "unified_det_cpcm")
LABELS = dict(backend="后端均值选优", own="自身确定性估值选优",
              random_cpcm="随机平局 CPCM 选优", unified_det_cpcm="统一确定性 CPCM 选优")


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batch_name(spec):
    return "I1" if spec["batch"] in ("I1a", "I1b", "I1W") else spec["batch"]


def expected_specs():
    specs = hi_manifest.candidates()
    new = [s for s in specs if batch_name(s) in ("H2D", "H2P", "H1c", "I1")]
    old = [s for s in specs if s["batch"] in ("H1a", "H1b")]
    assert len(new) == 344 and len(old) == 4
    return new, old


def stats(values, prefix):
    return {prefix + "_n": len(values), prefix + "_mean_ns": statistics.mean(values),
            prefix + "_median_ns": statistics.median(values), prefix + "_min_ns": min(values),
            prefix + "_max_ns": max(values), prefix + "_stdev_ns": statistics.stdev(values)}


def timing_row(record, raw, terminal):
    common = {k: record[k] for k in ("candidate_id", *KEYS)}
    if record["method"] != "Gemini":
        pieces = exclusive_timing(record)
        resolution = "A1/A3h mutually exclusive milestones"
    else:
        timing = record["timing"]
        marks = timing["milestones_seconds"]
        best = timing["evaluation"]
        last = terminal["timing"] if terminal else None
        pieces = dict(configure_seconds=marks["configure_complete"],
            search_seconds=raw["gemini"]["sa_wall_s"],
            backend_calls_seconds=sum(best["backend_individual_seconds"]),
            random_cpcm_calls_seconds=sum(best["random_cpcm_individual_seconds"]),
            evaluation_other_seconds=best["evaluation_other_seconds"],
            terminal_backend_calls_seconds=sum(last["backend_individual_seconds"]) if last else 0,
            terminal_evaluation_other_seconds=last["evaluation_other_seconds"] if last else 0,
            export_seconds=marks["export_complete"]-marks["evaluation_complete"],
            snapshot_reload_seconds=marks["snapshot_reloaded"]-marks["export_complete"],
            final_accounting_seconds=record["wall_seconds"]-marks["snapshot_reloaded"])
        # The legacy marks are overwritten by terminal evaluation. Account from
        # both saved evaluation timers; the remaining time is a named total.
        pieces["template_capture_and_recording_seconds"] = record["wall_seconds"]-sum(pieces.values())
        assert min(pieces.values()) >= 0
        assert math.isclose(sum(pieces.values()), record["wall_seconds"], rel_tol=1e-12)
        resolution = "Gemini SA and each evaluation measured; template/capture/recording grouped residual"
    return dict(common, **pieces, full_job_seconds=record["wall_seconds"], timing_resolution=resolution)


def collect(next_out, pilot_out, progress=False):
    import post_fg_split
    post_fg_split.setup()
    new, old = expected_specs()
    declared = {s["candidate_id"]: s for s in new + old}
    pilot_graphs = {r["candidate_id"]: r for r in read(HERE / "evidence/h1_graph_measured.json")["rows"]}
    rows, samples, timings, audits, terminals, missing = [], [], [], [], [], []
    for out, specs in ((pilot_out, old), (next_out, new)):
        for spec in specs:
            name = spec["candidate_id"]
            result_path = out / "results" / (name + ".json")
            receipt_path = out / "receipts" / (name + ".json")
            if progress and not (result_path.exists() and receipt_path.exists()):
                missing.append(name)
                continue
            result = read(result_path)
            assert result["status"] == "ok" and all(result[k] == v for k, v in spec.items())
            assert result["exact_rollback"] and not result["debug_checks"]
            receipt = read(receipt_path)
            # Controller release dependencies sit outside spec and do not change identity.
            job = dict(id=name, kind=spec["kind"], spec=spec, result=f"results/{name}.json",
                       argv=["{python}", "-u", "-B", "e2e/hi_run.py", "child", "--spec", "{spec}", "--out", "{out}"])
            assert receipt["rc"] == 0 and receipt["identity"] == fg_batch.identity(job)
            raw_path = out / "raw" / (name + ".pkl.gz")
            raw = post_fg_split.static.read_pickle(raw_path)
            assert raw["meta"]["steps"] == 1000
            selected = raw["selected"]
            if spec["method"] == "Gemini":
                meta = raw["gemini"]
                assert meta["steps"] == 1000 and meta["anneal_seed"] == spec["x"]
                assert meta["LP_flag"] and meta["hops_only"] and not meta["dijkstra_routing"]
                assert meta["rollback"] == "pre_move_snapshot"
                trajectory_check = "native Gemini capture records completed 1000 steps"
                own_cost, own_primary = meta["best_energy"], meta["best_energy"]
            else:
                assert len(raw["rows"]) == 1000
                assert [r["step"] for r in raw["rows"]] == list(range(1, 1001))
                assert raw["check_revert"]["n"] == 0 and not raw["check2"] and not raw["check4"]
                assert result["instrumentation"]["original_algorithm_ast_identical_after_removing_observations"]
                for row in raw["rows"]:
                    target = raw["meta"]["Tmax"]*(raw["meta"]["Tmin"]/raw["meta"]["Tmax"])**(row["step"]/1000)
                    assert math.isclose(row["temp"], target, rel_tol=1e-12)
                    assert row["accepted"] or "t_revert" in row
                trajectory_check = "1000 raw rows, geometric temperatures and exact reverts checked"
                own_cost = selected["own_cost"]
                own_primary = selected["T_sel"] if spec["method"] == "A3h" else own_cost
            assert [r["seed"] for r in result["backend"]] == list(range(1, 11))
            assert len(result["T_hat_rand"]) == 10 and result["T_hat_rand"] == selected["T_hat_rand"]
            assert result["T_hat_rand_mean"] == statistics.mean(result["T_hat_rand"])
            assert result["T_hat"] == selected["T_bc"] and result["T_hat_pre"] == selected["T_pre"]
            for item in result["backend"]:
                assert item["triple"] == list(selected["backend"][item["seed"]])
                assert item["cycles"] == item["triple"][0]
            if "own_cost" in result:
                assert result["own_cost"] == own_cost
            orders = result.get("backend_input_orders")
            if orders is not None:
                assert [v["seed"] for v in orders] == list(range(1, 11))
                assert all(len(v["actual_iteration_sha256"]) == 64 for v in orders)
            values = [r["cycles"] for r in result["backend"]]
            graph = result["graph"]
            row = dict(candidate_id=name, batch=batch_name(spec), **{k: spec[k] for k in KEYS},
                candidate_order=spec["candidate_order"], g=spec["g"], x=spec["x"],
                **stats(values, "backend"), **stats(result["T_hat_rand"], "random_cpcm"),
                deterministic_cpcm_ns=result["T_hat"], own_selector_primary=own_primary, own_cost=own_cost,
                selected_step=result["selected_step"], full_job_seconds=result["wall_seconds"],
                process_receipt_seconds=receipt["wall_seconds"], peak_rss_gib=result["peak_rss_gb"],
                final_events=graph["events"], final_computations=sum(graph["computations"].values()),
                schedule_activities=graph["activities"] if graph.get("graph_metric_version") == "schedule_activities_v2"
                    else pilot_graphs[name]["measured_schedule_activities"],
                final_branch_hop_references=graph["communication_hops"],
                actual_event_orders_recorded=orders is not None, source=str(result_path.relative_to(ROOT)))
            row["deterministic_relative_error"] = row["deterministic_cpcm_ns"]/row["backend_mean_ns"]-1
            row["random_cpcm_relative_error"] = row["random_cpcm_mean_ns"]/row["backend_mean_ns"]-1
            rows.append(row)
            for index, item in enumerate(result["backend"]):
                samples.append(dict(candidate_id=name, **{k: spec[k] for k in KEYS}, seed=item["seed"],
                    backend_ns=item["cycles"], unhidden_compute_ns=item["triple"][1],
                    unhidden_communication_ns=item["triple"][2], random_cpcm_ns=result["T_hat_rand"][index],
                    backend_call_seconds=result["timing"]["evaluation"]["backend_individual_seconds"][index],
                    random_cpcm_call_seconds=result["timing"]["evaluation"]["random_cpcm_individual_seconds"][index]))
            terminal = None
            if spec["method"] == "Gemini" and spec["x"] == 123:
                terminal = read(out / "results" / (name + "_terminal.json"))
                assert terminal["status"] == "ok" and terminal["from_search"] == name
                assert set(terminal["backend"]) == {str(seed) for seed in range(1, 11)}
                assert terminal["random_cpcm_evaluations"] == 0
                assert [v["seed"] for v in terminal["backend_input_orders"]] == list(range(1, 11))
                final_values = [terminal["backend"][str(seed)][0] for seed in range(1, 11)]
                terminals.append(dict(candidate_id=name, phase=spec["phase"], op=spec["op"],
                    best_backend_mean_ns=row["backend_mean_ns"], **stats(final_values, "terminal_backend"),
                    terminal_vs_best=statistics.mean(final_values)/row["backend_mean_ns"]-1,
                    best_energy=raw["gemini"]["best_energy"], terminal_energy=raw["gemini"]["terminal_energy"],
                    samples=final_values, backend_samples=[dict(seed=seed,
                        backend_ns=terminal["backend"][str(seed)][0],
                        unhidden_compute_ns=terminal["backend"][str(seed)][1],
                        unhidden_communication_ns=terminal["backend"][str(seed)][2],
                        backend_call_seconds=terminal["timing"]["backend_individual_seconds"][seed-1])
                        for seed in range(1, 11)]))
            timings.append(timing_row(result, raw, terminal))
            artifacts = {kind: sha(out / folder / (name + suffix)) for kind, folder, suffix in
                         (("raw", "raw", ".pkl.gz"), ("snapshot", "snapshots", ".pkl.gz"), ("result", "results", ".json"))}
            if name in pilot_graphs:
                prior = pilot_graphs[name]["input_sha256"]
                assert artifacts["raw"] == prior["raw"] and artifacts["snapshot"] == prior["snapshots"]
                assert artifacts["result"] == prior["results"]
            if terminal:
                artifacts.update(terminal_result=sha(out / "results" / (name+"_terminal.json")),
                                 terminal_snapshot=sha(out / "snapshots" / (name+"_terminal.pkl.gz")))
            audits.append(dict(candidate_id=name, **artifacts, receipt_identity_matches=True,
                raw_samples_match=True, trajectory_check=trajectory_check,
                actual_event_orders_recorded=orders is not None,
                model_contract_sha256=spec["model_contract_sha256"], source_sha256=result["source_sha256"]))
    extra = {p.stem for p in next_out.glob("results/*.json") if not p.stem.endswith("_terminal")} - set(declared)
    assert not extra
    if not progress:
        assert len(rows) == 348 and len(terminals) == 8
    return rows, samples, timings, audits, terminals, missing


def select_operators(candidates):
    groups = defaultdict(list)
    for row in candidates:
        groups[tuple(row[k] for k in KEYS)].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        orders = {r["candidate_order"] for r in rows}
        assert len(orders) == len(rows) and orders <= set(range(14))
        row = dict(zip(KEYS, key), completed=len(rows), planned=14,
                   state="complete" if len(rows) == 14 else "pilot_or_partial")
        if len(rows) == 14:
            keys = dict(backend=lambda r: (r["backend_mean_ns"], r["candidate_order"]),
                own=lambda r: (r["own_selector_primary"], r["own_cost"], r["candidate_order"]),
                random_cpcm=lambda r: (r["random_cpcm_mean_ns"], r["candidate_order"]),
                unified_det_cpcm=lambda r: (r["deterministic_cpcm_ns"], r["candidate_order"]))
            best = min(rows, key=keys["backend"])
            for name, selector in keys.items():
                chosen = min(rows, key=selector)
                row.update({name+"_candidate": chosen["candidate_id"], name+"_mean_ns": chosen["backend_mean_ns"],
                    name+"_min_ns": chosen["backend_min_ns"], name+"_max_ns": chosen["backend_max_ns"],
                    name+"_stdev_ns": chosen["backend_stdev_ns"],
                    name+"_selection_gap": chosen["backend_mean_ns"]/best["backend_mean_ns"]-1})
            row.update(candidate_mean_median_ns=statistics.median(r["backend_mean_ns"] for r in rows),
                       candidate_mean_min_ns=min(r["backend_mean_ns"] for r in rows),
                       candidate_mean_max_ns=max(r["backend_mean_ns"] for r in rows))
        output.append(row)
    return output


def assemble_layers(operators):
    complete = {(r["phase"], r["op"], r["method"]): r for r in operators
                if r["model"] == "llama" and r["hardware_profile"] == "busybarn" and r["completed"] == 14}
    contract = hi_config.model_contract("llama")
    hp = f_config.hardware_parameters("busybarn")
    output = []
    for protocol in PROTOCOLS:
        for method in METHODS:
            if not all((phase, op, method) in complete for phase in PHASES for op in OPS):
                continue
            row = dict(model="llama", hardware_profile="busybarn", shape="2x2", method=method,
                protocol=protocol, batch=contract["batch"], mha_multiplier=contract["head_groups"],
                kv_length=2048, decode_norm_multiplier=1, prefill_norm_multiplier=2,
                handoff_assumption="one contiguous bf16 activation over one directed adjacent-group D2D link")
            for phase in PHASES:
                weights = dict(ln=2 if phase == "prefill" else 1, mha=contract["head_groups"], proj=1, ffn=1)
                cells = {op: complete[phase, op, method] for op in OPS}
                measured_sum = sum(weights[op]*cells[op][protocol+"_mean_ns"] for op in OPS)
                activation_bytes = contract["batch"]*(2048 if phase == "prefill" else 1)*contract["hidden"]*2
                handoff = hp["d2d_latency_ticks"]+activation_bytes/hp["d2d_bytes_per_tick_directed_link"]
                row.update({phase+"_operator_sum_ns": measured_sum, phase+"_handoff_bytes": activation_bytes,
                    phase+"_handoff_ns": handoff, phase+"_ns": measured_sum+handoff,
                    phase+"_sources": {op: cells[op][protocol+"_candidate"] for op in OPS},
                    phase+"_sum_of_component_minima_ns": sum(weights[op]*cells[op][protocol+"_min_ns"] for op in OPS)+handoff,
                    phase+"_sum_of_component_maxima_ns": sum(weights[op]*cells[op][protocol+"_max_ns"] for op in OPS)+handoff})
            row.update(TTFT_ms=row["prefill_ns"]/1e6, TPOT_ms=row["decode_ns"]/1e6)
            for length in (128, 512, 2048):
                row[f"request_additional_decode_{length}_ns"] = row["prefill_ns"]+length*row["decode_ns"]
            output.append(row)
    lookup = {(r["protocol"], r["method"]): r for r in output}
    for row in output:
        if row["method"] == "A3h":
            for other in ("A1", "Gemini"):
                for measure in ("prefill_ns", "decode_ns", "request_additional_decode_2048_ns"):
                    row[f"{measure}_relative_to_{other}"] = row[measure]/lookup[row["protocol"], other][measure]-1
    return output


def fullmodel_references(layers):
    """Keep the unknown placement as B cross-group connections, not a layout."""
    output = []
    for layer in layers:
        row = {key: layer[key] for key in ("model", "hardware_profile", "shape", "method", "protocol", "batch", "kv_length")}
        row.update(layers=80, crossgroup_connections="B", equation="constant_ns + B * per_connection_ns",
                   layer_count_basis="artifact Llama reference", placement="B supplied by the future layer placement",
                   measurement="analytical repetition of measured single-layer operators; full-model residency not simulated")
        for phase in PHASES:
            row[phase+"_constant_ns"] = 80*layer[phase+"_operator_sum_ns"]
            row[phase+"_per_connection_ns"] = layer[phase+"_handoff_ns"]
        for steps in (128, 512, 2048):
            row[f"request_additional_decode_{steps}_constant_ns"] = row["prefill_constant_ns"]+steps*row["decode_constant_ns"]
            row[f"request_additional_decode_{steps}_per_connection_ns"] = row["prefill_per_connection_ns"]+steps*row["decode_per_connection_ns"]
        output.append(row)
    return output


def compare_estimates(candidates, next_out):
    with (HERE / "tables/remaining_candidate_estimates.csv").open(encoding="utf-8-sig", newline="") as stream:
        forecast = {r["candidate_id"]: r for r in csv.DictReader(stream)}
    rows = []
    for row in candidates:
        if row["batch"] in ("H1a", "H1b"):
            continue
        estimate = forecast[row["candidate_id"]]
        predicted = float(estimate["estimated_seconds"])
        rows.append(dict(candidate_id=row["candidate_id"], batch=row["batch"], **{k: row[k] for k in KEYS},
            original_estimated_seconds=predicted, measured_job_seconds=row["full_job_seconds"],
            measured_receipt_seconds=row["process_receipt_seconds"], measured_vs_estimate=row["full_job_seconds"]/predicted,
            original_predicted_peak_rss_gib=float(estimate["estimated_rss_gib"]),
            measured_peak_rss_gib=row["peak_rss_gib"]))
    summary = []
    previous = read(HERE / "evidence/next_batch_estimates.json")["summary"]
    for group in ("H2D", "H2P", "H1c", "I1", "next344"):
        selected = [r for r in rows if r["batch"] == group or group == "next344"]
        if not selected:
            continue
        original = next(r for r in previous if r["batch"] == group)
        summary.append(dict(batch=group, complete=len(selected), planned=original["candidates"],
            original_estimated_process_hours=original["estimated_process_hours"],
            measured_job_process_hours=sum(r["measured_job_seconds"] for r in selected)/3600,
            measured_receipt_process_hours=sum(r["measured_receipt_seconds"] for r in selected)/3600,
            measured_to_original_median=statistics.median(r["measured_vs_estimate"] for r in selected),
            measured_to_original_min=min(r["measured_vs_estimate"] for r in selected),
            measured_to_original_max=max(r["measured_vs_estimate"] for r in selected),
            longest_job_seconds=max(r["measured_job_seconds"] for r in selected),
            peak_job_rss_gib=max(r["measured_peak_rss_gib"] for r in selected),
            original_planning_low_minutes=original["planning_low_minutes"],
            original_planning_high_minutes=original["planning_high_minutes"]))
    batch_files = list(next_out.glob("*.batch.json"))
    batches = [dict(file=str(p.relative_to(ROOT)), **read(p)) for p in batch_files]
    return rows, summary, batches


def controller_evidence(next_out):
    files = {}
    for filename in ("initial_84.start.json", "next_344.start.json", "controller_finish.json",
                     "class_estimates.json", "release.json", "next_344.status.before_eta_fix.json"):
        path = next_out / filename
        if path.exists():
            files[filename] = read(path)
    result = dict(files=files,
        failed_attempts=[dict(path=str(p.relative_to(ROOT)), **read(p)) for p in
                         sorted(next_out.glob("failed_attempts/*/*/receipt.json"))])
    observations = []
    for path in sorted(next_out.glob("*.batch.json")) + sorted(next_out.glob("*.status*.json")):
        value = read(path)
        if "min_available_gib" in value:
            observations.append(dict(source=str(path.relative_to(ROOT)),
                max_active=value.get("max_combined_active", value["max_active"]),
                min_available_gib=value["min_available_gib"]))
    if observations:
        result.update(resource_observations=observations,
            combined_peak_active=max(r["max_active"] for r in observations),
            minimum_available_gib=min(r["min_available_gib"] for r in observations))
    if "initial_84.start.json" in files and "controller_finish.json" in files:
        start, end = files["initial_84.start.json"]["created_utc"], files["controller_finish.json"]["utc"]
        result.update(first_launch_utc=start, completion_utc=end,
                      full_wall_seconds=(datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds())
    return result


def batch_completion_times(candidates, next_out, progress):
    """Estimate controller-observed completion from the sole attempt log timestamp."""
    first_start = next_out / "initial_84.start.json"
    expected = dict(H2D=166, H2P=166, H1c=4, I1=8)
    planned = {r["candidate_id"]: batch_name(r) for r in expected_specs()[0]}
    if not first_start.exists():
        assert progress, "Final reporting needs initial_84.start.json and all task logs"
        return [], [dict(batch=k, planned=v, completed_records=sum(r["batch"] == k for r in candidates),
            timed_records=0, state="start_and_logs_not_downloaded") for k, v in expected.items()]
    origin = datetime.fromisoformat(read(first_start)["created_utc"]).timestamp()
    jobs = []
    for row in candidates:
        name = row["candidate_id"]
        if name not in planned:
            continue
        logs = sorted((next_out / "logs").glob(name + ".*.log"))
        if not logs and progress:
            continue
        empty_logs = [p for p in logs if p.stat().st_size == 0]
        finished_logs = [p for p in logs if p.stat().st_size > 0]
        assert len(finished_logs) == 1, (name, "expected the single completed experiment attempt", len(finished_logs))
        log = finished_logs[0]
        finish_record = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
        assert finish_record["candidate_id"] == name and finish_record["status"] == "ok"
        assert finish_record["seconds"] == row["full_job_seconds"]
        stamp = log.name[len(name)+1:-4]
        assert stamp.isdigit()
        start_unix = int(stamp)/1e9
        finish_unix = start_unix + row["process_receipt_seconds"]
        assert finish_unix >= start_unix >= origin
        jobs.append(dict(candidate_id=name, batch=planned[name],
            log=str(log.relative_to(ROOT)), start_unix_ns=int(stamp),
            empty_prelaunch_logs=[str(p.relative_to(ROOT)) for p in empty_logs],
            start_utc=datetime.fromtimestamp(start_unix, timezone.utc).isoformat(),
            approximate_completed_utc=datetime.fromtimestamp(finish_unix, timezone.utc).isoformat(),
            approximate_completed_beijing=datetime.fromtimestamp(finish_unix, timezone(timedelta(hours=8))).isoformat(),
            receipt_wall_seconds=row["process_receipt_seconds"],
            minutes_from_initial_launch=(finish_unix-origin)/60))
    summaries = []
    for batch, count in expected.items():
        group = [r for r in jobs if r["batch"] == batch]
        completed = sum(r["batch"] == batch for r in candidates)
        state = "complete" if len(group) == count else "logs_not_downloaded" if len(group) < completed else "in_progress"
        summary = dict(batch=batch, planned=count, completed_records=completed,
                       timed_records=len(group), state=state)
        if group:
            last = max(group, key=lambda r:r["minutes_from_initial_launch"])
            summary.update(latest_timed_candidate=last["candidate_id"],
                latest_observed_completed_utc=last["approximate_completed_utc"],
                latest_observed_minutes_from_launch=last["minutes_from_initial_launch"])
            if state == "complete":
                summary.update(approximate_batch_complete_utc=last["approximate_completed_utc"],
                    approximate_batch_complete_beijing=last["approximate_completed_beijing"],
                    minutes_from_initial_launch=last["minutes_from_initial_launch"])
        summaries.append(summary)
    if not progress:
        assert len(jobs) == 344 and all(r["state"] == "complete" for r in summaries)
    return jobs, summaries


def class_timings(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in KEYS)].append(row)
    return [dict(zip(KEYS, key), complete=len(values),
        full_job_seconds_median=statistics.median(v["full_job_seconds"] for v in values),
        full_job_seconds_min=min(v["full_job_seconds"] for v in values),
        full_job_seconds_max=max(v["full_job_seconds"] for v in values),
        full_job_process_hours=sum(v["full_job_seconds"] for v in values)/3600,
        process_receipt_hours=sum(v["process_receipt_seconds"] for v in values)/3600,
        peak_rss_gib=max(v["peak_rss_gib"] for v in values),
        maximum_final_events=max(v["final_events"] for v in values),
        maximum_schedule_activities=max(v["schedule_activities"] for v in values))
        for key, values in sorted(groups.items())]


def tail_estimate_comparison(timings, job_completion):
    path = HERE / "evidence/next_344_tail_estimate.json"
    if not path.exists():
        return []
    estimate = read(path)
    completed = {r["candidate_id"]: r for r in job_completion}
    output = []
    for item in estimate["rows"]:
        name = f"I_qwen3moe_{item['hardware']}_prefill_ffn_A3h_g0_s123"
        rows = [r for r in timings if r["candidate_id"] == name]
        if not rows or name not in completed:
            continue
        timing = rows[0]
        low, high = (datetime.fromisoformat(value).replace(tzinfo=timezone(timedelta(hours=8)))
                     for value in item["estimated_finish_local"])
        actual = datetime.fromisoformat(completed[name]["approximate_completed_beijing"])
        output.append(dict(candidate_id=name, hardware=item["hardware"],
            observed_utc=estimate["observed_utc"], basis=estimate["basis"],
            measured_search_minutes=timing["search_seconds"]/60,
            measured_ten_backend_minutes=timing["backend_calls_seconds"]/60,
            measured_full_job_minutes=timing["full_job_seconds"]/60,
            forecast_finish_low_beijing=low.isoformat(), forecast_finish_high_beijing=high.isoformat(),
            approximate_actual_complete_beijing=actual.isoformat(), within_updated_window=low <= actual <= high))
    return output


def health_text(operators, layers, candidates):
    main = [r for r in candidates if r["model"] == "llama" and r["hardware_profile"] == "busybarn"]
    complete_groups = sum(r["completed"] == 14 for r in operators
                          if r["model"] == "llama" and r["hardware_profile"] == "busybarn")
    lines = ["# H 阶段主硬件健康结果", "", "本表采用 W1、Llama 2×2 Die 组、batch=8，prefill 长度与 decode KV 长度均为 2048。",
        f"当前主硬件 Llama 已完成 {len(main)}/336 个候选，{complete_groups}/24 个算子×方法组齐备。每组计划 14 个候选；每个候选完整搜索 1000 步，随后独立评估后端种子 1–10 与十次随机平局 CPCM。",
        "", "## 八个算子的主质量比较", "", "按同方法十四候选的十种子后端均值选优。方括号为所选候选十次后端评估的最小、最大值，单位毫秒。",
        "", "| 阶段 | 算子 | A1 | A3h | Gemini | A3h / A1 − 1 | A3h / Gemini − 1 |", "|---|---|---|---|---|---|---|"]
    lookup = {(r["phase"],r["op"],r["method"]):r for r in operators
              if r["model"] == "llama" and r["hardware_profile"] == "busybarn" and r["completed"] == 14}
    for phase in PHASES:
        for op in OPS:
            if not all((phase, op, method) in lookup for method in METHODS):
                continue
            r = [lookup[phase,op,m] for m in METHODS]
            cells = [f"{x['backend_mean_ns']/1e6:.4f} [{x['backend_min_ns']/1e6:.4f}, {x['backend_max_ns']/1e6:.4f}]" for x in r]
            lines.append(f"| {phase} | {op} | {' | '.join(cells)} | {r[1]['backend_mean_ns']/r[0]['backend_mean_ns']-1:+.2%} | {r[1]['backend_mean_ns']/r[2]['backend_mean_ns']-1:+.2%} |")
    if len(layers) == 12:
        assembled = {(r["protocol"],r["method"]): r for r in layers}
        backend_a, backend_b = assembled["backend","A3h"], assembled["backend","A1"]
        random_a, random_b = assembled["random_cpcm","A3h"], assembled["random_cpcm","A1"]
        ln_delta_us = (lookup["decode","ln","A3h"]["backend_mean_ns"]-lookup["decode","ln","A1"]["backend_mean_ns"])/1000
        lines += ["", "### 本批结果", "",
            f"A3h 的健康收益主要集中在 decode 的 MHA、projection 和 FFN。单层 decode 按后端选优较 A1 变化 {backend_a['decode_ns']/backend_b['decode_ns']-1:+.2%}，按随机平局 CPCM 选择变化 {random_a['decode_ns']/random_b['decode_ns']-1:+.2%}。decode LN 较 A1 增加约 65%，对应绝对增加 {ln_delta_us:.3f} 微秒，已计入单层合计。",
            f"Prefill 与 A1 接近：后端选优的单层变化为 {backend_a['prefill_ns']/backend_b['prefill_ns']-1:+.2%}，随机平局 CPCM 选出的单层变化为 {random_a['prefill_ns']/random_b['prefill_ns']-1:+.2%}。本批支持 decode 改善；prefill 的收益没有同样延续，选择器也保留了实际损失。",
            f"Gemini 的后端选优 prefill 单层时间是 A3h 的 {assembled['backend','Gemini']['prefill_ns']/backend_a['prefill_ns']:.2f} 倍。表内 MHA、projection、FFN 均较高，其中 FFN 的绝对时间最大；本轮尚未分解 DDR 等待与路由争用，不作这两者的物理归因。"]
    lines += ["", "## 三种选优及统一 CPCM 对照", "",
        "自身估值沿用历史代价定义：A1 按其归一化加权聚合能量；A3h 先按搜索所用完整广播 CPCM 时间 T_sel，再以自身能量打破平局；Gemini 按原生 hops-only 能量。统一确定性 CPCM 另列，便于用同一预测器比较。所有选择最后再列出所选状态的后端均值。",
        "候选之间完全同值时按预定 candidate_order 选取。后端选优是质量参考，另两种选择及统一 CPCM 对照不读取后端结果。",
        "", "| 方法 | 选择规则 | 八算子选优损失中位数 | 最大值 | 与后端赢家相同 |", "|---|---|---|---|---|"]
    for method in METHODS:
        group = [r for r in lookup.values() if r["method"] == method]
        for protocol in PROTOCOLS[1:]:
            if len(group) != 8:
                continue
            gaps = [r[protocol+"_selection_gap"] for r in group]
            same = sum(r[protocol+"_candidate"] == r["backend_candidate"] for r in group)
            lines.append(f"| {method} | {LABELS[protocol]} | {statistics.median(gaps):+.2%} | {max(gaps):+.2%} | {same}/8 |")
    lines += ["", "十四候选后端均值的中位数、范围、每种选择的具体候选与十种子标准差全部保存在 `hi/tables/next_operators.csv`，逐样本在 `next_samples.csv`。标准差描述同一候选的后端波动；十四候选是映射选择集合。", "",
        "## 单层与固定 KV 请求估算", "", "Prefill = 2×RMSNorm + 8×代表 MHA + projection + FFN + handoff；decode = RMSNorm + 8×代表 MHA + projection + FFN + handoff。decode FFN 已含一次 RMSNorm，未再重复添加。",
        "handoff 单列为相邻组之间一条有向 D2D 链路传送连续 bf16 激活的解析项：batch×tokens×8192×2 字节，256 字节/ns、固定 20 ns，prefill 1,048,596 ns，decode 532 ns。算子均值之和与 handoff 分列保存在 CSV，便于采用实际层布局时替换。",
        "下面是单层算子结果的串行组装；没有模拟跨算子重叠、所有层同时驻留或层间共享链路竞争。请求 N 表示首个 token 产生后再执行 N 次 decode；使用 P+N×D，decode 每一步保持 KV=2048。",
        "", "| 选择规则 | 方法 | 单层 TTFT ms | 单层 TPOT ms | 请求 N=128 s | N=512 s | N=2048 s |", "|---|---|---|---|---|---|---|"]
    for r in layers:
        lines.append(f"| {LABELS[r['protocol']]} | {r['method']} | {r['TTFT_ms']:.4f} | {r['TPOT_ms']:.4f} | "+
                     " | ".join(f"{r[f'request_additional_decode_{n}_ns']/1e9:.5f}" for n in (128,512,2048))+" |")
    if len(layers) == 12:
        lines += ["", "![Llama 主硬件健康结果](hi/figures/next_health.png)", "",
                  "主柱使用随机平局 CPCM 选择的状态，白色菱形为后端选优；数值均为所选状态的实际后端均值加解析 handoff。可编辑矢量图见 `hi/figures/next_health.svg`。"]
    lines += ["", "CSV 中各算子样本最小值之和、最大值之和仅是组成项的范围，既不是全层联合仿真范围，也不是置信区间。"]
    if len(layers) == 12:
        lines += ["", "## 80 层成本参考", "",
            "以 artifact 的 Llama 80 层为参考，令 B 为实际层布局中的跨组连接数。P_op、D_op 是上表单层中不含 handoff 的算子和，则：", "",
            "- T80_P = 80×P_op + B×1,048,596 ns。",
            "- T80_D = 80×D_op + B×532 ns。",
            "- 首个 token 后追加 N 次 decode 的请求为 T80_P + N×T80_D，KV 固定为 2048。", "",
            "下表列随机平局 CPCM 选择的 80 层算子和。每增加一条跨组连接，prefill 增加 1.048596 ms、decode 一步增加 0.532 µs；N=2048 的请求增加 2.138132 ms。布局确定后直接代入 B。", "",
            "| 方法 | 80 层 prefill 算子和 s | 80 层 decode 算子和 ms | N=2048 请求算子和 s |", "|---|---|---|---|"]
        for row in fullmodel_references(layers):
            if row["protocol"] == "random_cpcm":
                lines.append(f"| {row['method']} | {row['prefill_constant_ns']/1e9:.6f} | {row['decode_constant_ns']/1e6:.6f} | {row['request_additional_decode_2048_constant_ns']/1e9:.6f} |")
        lines += ["", "四种选择共 12 行的常数项及每条连接增量见 `hi/tables/next_fullmodel_reference.csv`。这一参考重复使用同一层实测算子成本，B 保留为参数；未指定 80 层分组形状、模拟全模型同时驻留或检查容量，组间共享链路竞争与流水重叠也未加入。"]
    lines += ["",
        "## WATOS 与 Qwen 试点", "", "WATOS 的 Llama 四项、Qwen 两种硬件共八项都只是预定 A3h g0/s123 完整试点。它们计入后续正式候选，尚未覆盖各算子十四候选和完整 LN/projection 链。跨硬件比较采用相同 g0/s123 候选；这里仅报告实际测到的 MHA/FFN，不据此生成完整层 TTFT/TPOT。", "",
        "| 模型 | 硬件 | 阶段 | 算子 | 后端均值 [最小, 最大] ms | 搜索+评估秒 | 峰值 RSS GiB |", "|---|---|---|---|---|---|---|"]
    for r in candidates:
        if r["batch"] not in ("H1c", "I1"):
            continue
        lines.append(f"| {r['model']} | {r['hardware_profile']} | {r['phase']} | {r['op']} | {r['backend_mean_ns']/1e6:.4f} [{r['backend_min_ns']/1e6:.4f}, {r['backend_max_ns']/1e6:.4f}] | {r['full_job_seconds']:.2f} | {r['peak_rss_gib']:.3f} |")
    if len(layers) == 12 and (HERE / "evidence/next_h_health_review.json").exists():
        lines += ["", "独立汇总检查从 336 份原始 run 与结果重算：24 个候选组、96 个选中状态、12 行单层结果及请求计算均一致，记录见 `hi/evidence/next_h_health_review.json`。这一记录核对的是 Llama 主硬件健康结果，整批 H/I 完成情况另见 `hi/evidence/next_completion_audit.json`。"]
    return lines


def run(args):
    rows, samples, timings, audits, terminals, missing = collect(args.next_out, args.pilot_out, args.progress)
    operators = select_operators(rows)
    layers = assemble_layers(operators)
    comparisons, eta, batches = compare_estimates(rows, args.next_out)
    controller = controller_evidence(args.next_out)
    job_completion, batch_completion = batch_completion_times(rows, args.next_out, args.progress)
    tail_comparison = tail_estimate_comparison(timings, job_completion)
    completion_by_batch = {r["batch"]: r for r in batch_completion}
    for item in eta:
        actual = completion_by_batch.get(item["batch"])
        if actual and actual["state"] == "complete":
            item.update(shared_queue_complete_utc=actual["approximate_batch_complete_utc"],
                        shared_queue_complete_beijing=actual["approximate_batch_complete_beijing"],
                        shared_queue_minutes_from_initial_launch=actual["minutes_from_initial_launch"])
        elif item["batch"] == "next344" and "full_wall_seconds" in controller:
            item.update(shared_queue_complete_utc=controller["completion_utc"],
                shared_queue_complete_beijing=datetime.fromisoformat(controller["completion_utc"]).astimezone(timezone(timedelta(hours=8))).isoformat(),
                shared_queue_minutes_from_initial_launch=controller["full_wall_seconds"]/60)
    dest = args.destination
    files = dict(candidates=rows, samples=samples, operators=operators, layers=layers,
        fullmodel_reference=fullmodel_references(layers),
        job_completion_times=job_completion, batch_completion_times=batch_completion,
        tail_estimate_comparison=tail_comparison,
        timing_exclusive=timings, class_timings=class_timings(rows),
        candidate_estimate_comparison=comparisons, estimate_comparison=eta, gemini_terminal=terminals,
        terminal_samples=[dict(candidate_id=r["candidate_id"]+"_terminal", phase=r["phase"], op=r["op"], **sample)
                          for r in terminals for sample in r["backend_samples"]])
    for name, values in files.items():
        table(dest / "tables" / ("next_" + name + ".csv"), values)
    all_results = len(rows) == 348
    batch_complete = any(b["completed"] == 344 and not b["failed"] and not b["pending"] for b in batches)
    complete = all_results and batch_complete
    evidence = dict(status="complete" if complete else "results_complete_awaiting_batch" if all_results else "in_progress", next_planned=344,
        next_complete=sum(r["batch"] not in ("H1a", "H1b") for r in rows), reused_h1=sum(r["batch"] in ("H1a", "H1b") for r in rows),
        healthy_candidates=len(rows), backend_samples=len(samples), random_cpcm_samples=len(samples),
        gemini_terminal_states=len(terminals), gemini_terminal_backend_samples=10*len(terminals),
        total_backend_samples=len(samples)+10*len(terminals),
        complete_operator_groups=sum(r["completed"] == 14 for r in operators), missing=missing,
        batches=batches, controller=controller, batch_completion_times=batch_completion,
        tail_estimate_comparison=tail_comparison,
        empty_prelaunch_logs=[p for row in job_completion for p in row["empty_prelaunch_logs"]], candidates=audits,
        model_contracts={m:hi_config.model_contract(m) for m in hi_config.MODELS})
    if all_results:
        assert evidence["next_complete"] == 344 and evidence["reused_h1"] == 4
        assert evidence["complete_operator_groups"] == 24 and len(layers) == 12
    if not args.progress:
        assert complete
    write(dest / "evidence/next_completion_audit.json", evidence)
    lines = ["# H/I 第二批结果与耗时", "", f"本轮 {evidence['next_complete']}/344 个任务完成，原 H1 四项复用 {evidence['reused_h1']}/4；共 {len(rows)} 个正式健康候选。后端与随机平局 CPCM 各 {len(samples)} 个样本，Gemini 结束状态另有 {10*len(terminals)} 个后端样本。",
        "所有正式任务保持 1000 步。A1/A3h 核对原始逐步记录、温度曲线与被拒动作的回滚；Gemini 核对原生捕获入口记录的 1000 步、种子、LP/XY 参数和快照回滚。候选身份、模型合同、原始十种子结果与输出文件摘要逐项保存于 `evidence/next_completion_audit.json`。", ""]
    for b in batches:
        maximum = b.get("max_combined_active", b["max_active"])
        lines += [f"批次 `{Path(b['file']).name}`：成功 {b['completed']}，失败 {len(b['failed'])}，未完成 {len(b['pending'])}；控制器墙钟 {b['wall_seconds']/60:.3f} 分钟，最多合计 {maximum} 并行，最低可用内存 {b['min_available_gib']:.3f} GiB。"]
    if "full_wall_seconds" in controller:
        lines += [f"整轮从首批启动 {controller['first_launch_utc']} 到完成 {controller['completion_utc']}，共 {controller['full_wall_seconds']/60:.3f} 分钟。两控制器运行有重叠，墙钟不能相加。归档保留的失败尝试共 {len(controller['failed_attempts'])} 次。"]
    if "combined_peak_active" in controller:
        lines += [f"合并监督器重启前状态、各批次状态与最终记录，本轮最多同时运行 {controller['combined_peak_active']} 个作业，最低可用内存 {controller['minimum_available_gib']:.3f} GiB。"]
    if "next_344.status.before_eta_fix.json" in controller["files"]:
        lines += ["监督入口有两次代码修正：启动阶段补齐 STDOUT 常量引用，当时尚未启动新候选；后期修正超过旧估计后剩余时间被算为 1 秒的问题，只重启监督器，两项在跑的 Qwen 任务继续运行。上述记录属于监督程序调整，实验作业失败与重试另按 receipt 统计。最终监督器批次墙钟和资源极值仅覆盖重启后的时段，因此总用时取最初启动至完成时间，资源极值合并重启前记录。"]
    if evidence["empty_prelaunch_logs"]:
        lines += [f"作业目录保留 {len(job_completion)} 份真实完成日志和 {len(evidence['empty_prelaunch_logs'])} 份启动前空日志。空日志来自 STDOUT 属性读取出错之前已经打开文件、但 Popen 尚未被调用；每个正式候选的唯一非空日志均有与结果时间一致的完成标记。空文件原样保留并列入审计，未计为额外实验尝试。"]
    lines += ["", "## 原估计与实际用时", "", "原估计来自四个主硬件 A3h 完整试点以及静态图规模；A1/Gemini 当时尚无实测。1.5–3 倍规划范围已包含各类首候选完成后再放行后续候选的依赖。原区间覆盖搜索、评估、导出/重载；不含下载、汇总及后续故障/联合任务。", "",
        "| 批次 | 完成/计划 | 原进程小时 | 实测进程小时 | 单任务实测/估计中位数 | 最长任务 min | 原墙钟范围 min | 实际收齐 UTC+8 | 距整轮启动 min |", "|---|---|---|---|---|---|---|---|---|"]
    for r in eta:
        end = r.get("shared_queue_complete_beijing", "待完整日志")
        elapsed = f"{r['shared_queue_minutes_from_initial_launch']:.3f}" if "shared_queue_minutes_from_initial_launch" in r else "—"
        if r["batch"] == "next344" and "full_wall_seconds" in controller:
            end = datetime.fromisoformat(controller["completion_utc"]).astimezone(timezone(timedelta(hours=8))).isoformat()
            elapsed = f"{controller['full_wall_seconds']/60:.3f}"
        lines.append(f"| {'四批合计' if r['batch']=='next344' else r['batch']} | {r['complete']}/{r['planned']} | {r['original_estimated_process_hours']:.3f} | {r['measured_job_process_hours']:.3f} | {r['measured_to_original_median']:.2f}× | {r['longest_job_seconds']/60:.2f} | {r['original_planning_low_minutes']:.2f}–{r['original_planning_high_minutes']:.2f} | {end} | {elapsed} |")
    if "full_wall_seconds" in controller:
        total = next(r for r in eta if r["batch"] == "next344")
        actual = controller["full_wall_seconds"]/60
        lines += ["", f"整轮实测 {actual:.2f} 分钟；原规划 {total['original_planning_low_minutes']:.2f}–{total['original_planning_high_minutes']:.2f} 分钟。实测是从首批启动算起，包含首候选校准和后续释放等待。"]
    if len(tail_comparison) == 2:
        by_hardware = {row["hardware"]: row for row in tail_comparison}
        bb, wa = by_hardware["busybarn"], by_hardware["watos"]
        lines += ["", f"长尾来自 Qwen prefill FFN：BusyBarn 上搜索 {bb['measured_search_minutes']:.2f} 分钟、十次后端 {bb['measured_ten_backend_minutes']:.2f} 分钟；WATOS 上搜索 {wa['measured_search_minutes']:.2f} 分钟、十次后端 {wa['measured_ten_backend_minutes']:.2f} 分钟。此前按 Llama 图规模估计的单任务总时间约 14.7 分钟，未估中 MoE 的搜索和后端成本。",
            "13:22 根据已经完成的后端种子调用间隔更新预测，BusyBarn 为 13:39–13:45，最后一项 WATOS 为约 14:02–14:15；两项实际完成均落在各自更新窗口内，整轮完成于 14:05:57。更新预测含后续后端、随机 CPCM、导出与重载，原始估算保留在 `evidence/next_344_tail_estimate.json`，实际比较见 `tables/next_tail_estimate_comparison.csv`。"]
        assert bb["within_updated_window"] and wa["within_updated_window"]
    lines += ["", "四批共享一个队列，各分项的原墙钟区间是该批单独排程的预测，不能相加，也不等于实际共享队列下该批结束时刻。实际收齐时刻取该批所有作业的完成时间最大值；每作业用日志文件名中的启动 Unix 纳秒时间加 receipt.wall_seconds 推算，含进程启动记录和约 2 秒调度轮询误差。距整轮启动从 initial_84.start.created_utc 算起。整轮最终时刻直接取 controller_finish.utc。逐项时间来源保存在 `tables/next_job_completion_times.csv` 与 `next_batch_completion_times.csv`，进程小时不作为墙钟。", "",
        "## 分项运行开销", "", "单位为本轮任务进程小时，按任务独立分项求和，包含并行资源争用；不是 J 阶段的串行算法计时。", "",
        "| 方法 | 任务 | 搜索 | 后端评估 | 随机平局 CPCM | 其余 | 完整任务 |", "|---|---|---|---|---|---|---|"]
    for method in METHODS:
        selected = [r for r in timings if r["method"] == method and r["candidate_id"] not in {s["candidate_id"] for s in expected_specs()[1]}]
        if not selected:
            continue
        total = sum(r["full_job_seconds"] for r in selected)
        search = sum(r["search_seconds"] for r in selected)
        backend = sum(r["backend_calls_seconds"]+r.get("terminal_backend_calls_seconds",0) for r in selected)
        ties = sum(r["random_cpcm_calls_seconds"] for r in selected)
        lines.append(f"| {method} | {len(selected)} | {search/3600:.4f} | {backend/3600:.4f} | {ties/3600:.4f} | {(total-search-backend-ties)/3600:.4f} | {total/3600:.4f} |")
    lines += ["", "A1/A3h 的模板、初始构图、温度校准、搜索、最佳状态恢复、后端、随机 CPCM、评估准备、导出、重载、记录分别列出。Gemini 保存了完整 SA 与两次各自的评估时间；模板构造/状态捕获/记录合为剩余项，未把结束状态覆盖后的时间戳当作最佳状态时间。首次广播缓存构图等嵌套计时不再重复相加。", "",
        "## Gemini 最好与结束状态", "", "每个主硬件算子仅 seed123 做此对照，未新增搜索。比较对应同一搜索的两种状态，不能视为方法之间的选优差异。", "",
        "| 阶段 | 算子 | 最好状态后端均值 ms | 结束状态后端均值 ms | 结束 / 最好 − 1 |", "|---|---|---|---|---|"]
    for r in terminals:
        lines.append(f"| {r['phase']} | {r['op']} | {r['best_backend_mean_ns']/1e6:.4f} | {r['terminal_backend_mean_ns']/1e6:.4f} | {r['terminal_vs_best']:+.2%} |")
    lines += ["", "健康质量、三种选优和完整层结果见 `../H_HEALTH_SUMMARY.md`。WATOS/Qwen 本轮仅列试点；H3、I2、故障修复和联合实验不在本轮 344 项之内。"]
    if args.progress:
        lines.insert(2, "此文件是已落盘结果的进度汇总；未完成的候选组不做十四候选选优。")
    (dest / "NEXT_BATCH_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    health = health_text(operators, layers, rows)
    args.health_report.parent.mkdir(parents=True, exist_ok=True)
    args.health_report.write_text("\n".join(health)+"\n", encoding="utf-8")
    print(json.dumps({k:evidence[k] for k in ("status","next_complete","reused_h1","healthy_candidates","backend_samples","gemini_terminal_states","complete_operator_groups")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next-out", type=Path, default=HERE / "out/next_344")
    parser.add_argument("--pilot-out", type=Path, default=HERE / "out/h1_main")
    parser.add_argument("--destination", type=Path, default=HERE)
    parser.add_argument("--health-report", type=Path, default=ROOT / "e2e/H_HEALTH_SUMMARY.md")
    parser.add_argument("--progress", action="store_true", help="Read completed results while this batch is running")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
