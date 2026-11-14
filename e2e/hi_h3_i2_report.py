"""Read H3/I2, I3 and the previous 348; retain each run's separate costs."""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import statistics

import f_config
import fg_batch
import hi_config
import hi_manifest
import hi_next_report as previous
from hi_split_supplement import read, write

ROOT, HERE = previous.ROOT, previous.HERE
KEYS, OPS, PHASES = previous.KEYS, previous.OPS, previous.PHASES
METHODS, PROTOCOLS, LABELS = previous.METHODS, previous.PROTOCOLS, previous.LABELS
COUNTS = dict(H3=108, I2D=166, I2P=166)
I3_COUNTS = dict(I3=108)
PREFIX = "h3_i2_"


def expected(counts=COUNTS):
    rows = [r for r in hi_manifest.candidates() if r["batch"] in counts]
    assert Counter(r["batch"] for r in rows) == counts
    return rows


def collect_new(out, progress, counts=COUNTS):
    import post_fg_split
    post_fg_split.setup()
    candidates, samples, times, audits, terminals, missing = [], [], [], [], [], []
    specs = expected(counts)
    for spec in specs:
        name = spec["candidate_id"]
        paths = {"result": out/"results"/(name+".json"), "raw": out/"raw"/(name+".pkl.gz"),
                 "snapshot": out/"snapshots"/(name+".pkl.gz"), "receipt": out/"receipts"/(name+".json")}
        if progress and not all(paths[k].exists() for k in ("result", "receipt")):
            missing.append(name)
            continue
        result, receipt = read(paths["result"]), read(paths["receipt"])
        assert result["status"] == "ok" and receipt["rc"] == 0
        assert all(result[k] == v for k, v in spec.items())
        job = dict(id=name, kind=spec["kind"], spec=spec, result=f"results/{name}.json",
                   argv=["{python}","-u","-B","e2e/hi_run.py","child","--spec","{spec}","--out","{out}"])
        assert receipt["identity"] == fg_batch.identity(job)
        raw = post_fg_split.static.read_pickle(paths["raw"])
        selected = raw["selected"]
        assert raw["meta"]["steps"] == 1000 and result["exact_rollback"] and not result["debug_checks"]
        if spec["method"] == "Gemini":
            meta = raw["gemini"]
            assert meta["steps"] == 1000 and meta["anneal_seed"] == spec["x"]
            assert meta["LP_flag"] and meta["hops_only"] and not meta["dijkstra_routing"]
            assert meta["rollback"] == "pre_move_snapshot"
            own_primary = own_cost = meta["best_energy"]
            trajectory = "native Gemini completed 1000 steps and pre-move snapshot rollback"
        else:
            assert len(raw["rows"]) == 1000 and [r["step"] for r in raw["rows"]] == list(range(1,1001))
            assert raw["check_revert"]["n"] == 0 and not raw["check2"] and not raw["check4"]
            assert result["instrumentation"]["original_algorithm_ast_identical_after_removing_observations"]
            for r in raw["rows"]:
                temperature = raw["meta"]["Tmax"]*(raw["meta"]["Tmin"]/raw["meta"]["Tmax"])**(r["step"]/1000)
                assert math.isclose(r["temp"], temperature, rel_tol=1e-12)
                assert r["accepted"] or "t_revert" in r
            own_cost = selected["own_cost"]
            own_primary = selected["T_sel"] if spec["method"] == "A3h" else own_cost
            trajectory = "1000 raw rows, geometric temperatures, recorded exact reverts"
        assert result["own_cost"] == own_cost
        assert [v["seed"] for v in result["backend"]] == list(range(1,11))
        assert len(result["T_hat_rand"]) == 10 and result["T_hat_rand"] == selected["T_hat_rand"]
        assert result["T_hat_rand_mean"] == statistics.mean(result["T_hat_rand"])
        assert result["T_hat"] == selected["T_bc"] and result["T_hat_pre"] == selected["T_pre"]
        assert [r["seed"] for r in result["backend_input_orders"]] == list(range(1,11))
        assert all(len(r["actual_iteration_sha256"]) == 64 for r in result["backend_input_orders"])
        for item in result["backend"]:
            assert item["triple"] == list(selected["backend"][item["seed"]]) and item["cycles"] == item["triple"][0]
        graph = result["graph"]
        assert graph["graph_metric_version"] == "schedule_activities_v2"
        row = dict(candidate_id=name, batch=spec["batch"], **{k:spec[k] for k in KEYS},
            candidate_order=spec["candidate_order"], g=spec["g"], x=spec["x"],
            **previous.stats([r["cycles"] for r in result["backend"]], "backend"),
            **previous.stats(result["T_hat_rand"], "random_cpcm"),
            deterministic_cpcm_ns=result["T_hat"], own_selector_primary=own_primary, own_cost=own_cost,
            selected_step=result["selected_step"], full_job_seconds=result["wall_seconds"],
            process_receipt_seconds=receipt["wall_seconds"], peak_rss_gib=result["peak_rss_gb"],
            final_events=graph["events"], final_computations=sum(graph["computations"].values()),
            schedule_activities=graph["activities"], final_branch_hop_references=graph["communication_hops"],
            actual_event_orders_recorded=True, source=str(paths["result"].relative_to(ROOT)))
        row.update(deterministic_relative_error=row["deterministic_cpcm_ns"]/row["backend_mean_ns"]-1,
                   random_cpcm_relative_error=row["random_cpcm_mean_ns"]/row["backend_mean_ns"]-1)
        candidates.append(row)
        for index, backend in enumerate(result["backend"]):
            samples.append(dict(candidate_id=name, **{k:spec[k] for k in KEYS}, seed=backend["seed"],
                backend_ns=backend["cycles"], unhidden_compute_ns=backend["triple"][1],
                unhidden_communication_ns=backend["triple"][2], random_cpcm_ns=result["T_hat_rand"][index],
                backend_call_seconds=result["timing"]["evaluation"]["backend_individual_seconds"][index],
                random_cpcm_call_seconds=result["timing"]["evaluation"]["random_cpcm_individual_seconds"][index]))
        terminal = None
        artifact_hashes = {k:previous.sha(p) for k,p in paths.items()}
        if spec["method"] == "Gemini" and spec["x"] == 123:
            path = out/"results"/(name+"_terminal.json")
            terminal = read(path)
            assert terminal["status"] == "ok" and terminal["from_search"] == name
            assert set(terminal["backend"]) == {str(seed) for seed in range(1,11)}
            assert terminal["random_cpcm_evaluations"] == 0
            assert [r["seed"] for r in terminal["backend_input_orders"]] == list(range(1,11))
            values = [terminal["backend"][str(seed)][0] for seed in range(1,11)]
            terminals.append(dict(candidate_id=name, phase=spec["phase"], op=spec["op"],
                best_backend_mean_ns=row["backend_mean_ns"], **previous.stats(values, "terminal_backend"),
                terminal_vs_best=statistics.mean(values)/row["backend_mean_ns"]-1,
                best_energy=meta["best_energy"], terminal_energy=meta["terminal_energy"], samples=values,
                backend_samples=[dict(seed=seed, backend_ns=terminal["backend"][str(seed)][0],
                    unhidden_compute_ns=terminal["backend"][str(seed)][1],
                    unhidden_communication_ns=terminal["backend"][str(seed)][2],
                    backend_call_seconds=terminal["timing"]["backend_individual_seconds"][seed-1]) for seed in range(1,11)]))
            artifact_hashes.update(terminal_result=previous.sha(path),
                terminal_snapshot=previous.sha(out/"snapshots"/(name+"_terminal.pkl.gz")))
        times.append(previous.timing_row(result, raw, terminal))
        audits.append(dict(candidate_id=name, **artifact_hashes, receipt_identity_matches=True,
            raw_samples_match=True, trajectory_check=trajectory,
            model_contract_sha256=spec["model_contract_sha256"], source_sha256=result["source_sha256"]))
    primary_ids = {s["candidate_id"] for s in specs}
    terminal_ids = {s["candidate_id"]+"_terminal" for s in specs if s["method"] == "Gemini" and s["x"] == 123}
    result_ids = {p.stem for p in (out/"results").glob("*.json")}
    assert result_ids <= primary_ids | terminal_ids
    if not progress:
        assert len(candidates) == len(specs) and len(terminals) == len(terminal_ids)
        assert result_ids == primary_ids | terminal_ids
        assert {p.stem for p in (out/"receipts").glob("*.json")} == primary_ids
    return candidates, samples, times, audits, terminals, missing


def qwen_layers(operators):
    contract = hi_config.model_contract("qwen3moe")
    assert contract["head_groups"] == 4 and contract["decode_chain"]["norm"] == 1
    complete = {(r["phase"],r["op"],r["method"]):r for r in operators
        if r["model"] == "qwen3moe" and r["hardware_profile"] == "busybarn" and r["completed"] == 14}
    hp = f_config.hardware_parameters("busybarn")
    output = []
    for protocol in PROTOCOLS:
        for method in METHODS:
            if not all((phase,op,method) in complete for phase in PHASES for op in OPS):
                continue
            row = dict(model="qwen3moe", hardware_profile="busybarn", shape="2x2", method=method,
                protocol=protocol, batch=contract["batch"], mha_multiplier=4, kv_length=2048,
                decode_norm_multiplier=1, prefill_norm_multiplier=2,
                handoff_assumption="one adjacent-group directed D2D link carrying contiguous bf16 activation")
            for phase in PHASES:
                weights = dict(ln=2 if phase=="prefill" else 1, mha=4, proj=1, ffn=1)
                cells = {op:complete[phase,op,method] for op in OPS}
                byte_count = contract["batch"]*(2048 if phase=="prefill" else 1)*contract["hidden"]*2
                handoff = hp["d2d_latency_ticks"]+byte_count/hp["d2d_bytes_per_tick_directed_link"]
                value = sum(weights[op]*cells[op][protocol+"_mean_ns"] for op in OPS)
                row.update({phase+"_operator_sum_ns":value, phase+"_handoff_bytes":byte_count,
                    phase+"_handoff_ns":handoff, phase+"_ns":value+handoff,
                    phase+"_sources":{op:cells[op][protocol+"_candidate"] for op in OPS},
                    phase+"_sum_of_component_minima_ns":sum(weights[op]*cells[op][protocol+"_min_ns"] for op in OPS)+handoff,
                    phase+"_sum_of_component_maxima_ns":sum(weights[op]*cells[op][protocol+"_max_ns"] for op in OPS)+handoff})
            row.update(TTFT_ms=row["prefill_ns"]/1e6, TPOT_ms=row["decode_ns"]/1e6)
            for count in (128,512,2048):
                row[f"request_additional_decode_{count}_ns"] = row["prefill_ns"]+count*row["decode_ns"]
            output.append(row)
    return output


def qwen_48_reference(layers):
    output = []
    for layer in layers:
        row = {k:layer[k] for k in ("model","hardware_profile","method","protocol","batch","kv_length")}
        row.update(layers=48, crossgroup_connections="B", equation="constant_ns + B * per_connection_ns",
                   measurement="analytical same-layer repetition; placement and full-model residency not simulated")
        for phase in PHASES:
            row[phase+"_constant_ns"] = 48*layer[phase+"_operator_sum_ns"]
            row[phase+"_per_connection_ns"] = layer[phase+"_handoff_ns"]
        for count in (128,512,2048):
            row[f"request_additional_decode_{count}_constant_ns"] = row["prefill_constant_ns"]+count*row["decode_constant_ns"]
            row[f"request_additional_decode_{count}_per_connection_ns"] = row["prefill_per_connection_ns"]+count*row["decode_per_connection_ns"]
        output.append(row)
    return output


def hardware_comparison(operators, model="llama"):
    lookup = {tuple(r[k] for k in KEYS):r for r in operators if r["completed"] == 14}
    output = []
    for phase in PHASES:
        for op in ("mha","ffn"):
            for method in ("A1","A3h"):
                keys = [(model,hw,phase,op,method) for hw in ("busybarn","watos")]
                if not all(key in lookup for key in keys):
                    continue
                baseline, watos = (lookup[key] for key in keys)
                for protocol in PROTOCOLS:
                    output.append(dict(model=model,phase=phase,op=op,method=method,protocol=protocol,
                        candidates_per_hardware=14, busybarn_candidate=baseline[protocol+"_candidate"],
                        watos_candidate=watos[protocol+"_candidate"], busybarn_ns=baseline[protocol+"_mean_ns"],
                        watos_ns=watos[protocol+"_mean_ns"],
                        watos_relative_to_busybarn=watos[protocol+"_mean_ns"]/baseline[protocol+"_mean_ns"]-1))
    return output


def watos_method_comparison(operators, model="llama"):
    cells = {(r["phase"],r["op"],r["method"]):r for r in operators
             if r["model"] == model and r["hardware_profile"] == "watos" and r["completed"] == 14}
    output = []
    for phase in PHASES:
        for op in ("mha","ffn"):
            if not all((phase,op,m) in cells for m in ("A1","A3h")):
                continue
            a,b = cells[phase,op,"A3h"],cells[phase,op,"A1"]
            for protocol in PROTOCOLS:
                output.append(dict(model=model,hardware_profile="watos",phase=phase,op=op,protocol=protocol,
                    A1_candidate=b[protocol+"_candidate"],A3h_candidate=a[protocol+"_candidate"],
                    A1_backend_mean_ns=b[protocol+"_mean_ns"],A3h_backend_mean_ns=a[protocol+"_mean_ns"],
                    A3h_relative_to_A1=a[protocol+"_mean_ns"]/b[protocol+"_mean_ns"]-1))
    return output


def workload_rows():
    rows = []
    for phase in PHASES:
        path = HERE/"out/model_checks/results"/f"check_qwen3moe_busybarn_{phase}_ffn_A3h.json"
        data = read(path)
        semantic, graph = data["semantic"], data["graph"]
        assert data["status"] == "ok"
        rows.append(dict(model="qwen3moe", phase=phase, input="H0/I0 A3h g0/s123 initial graph",
            experts=semantic["experts"], top_k=semantic["top_k"], expert_dim=semantic["expert_dim"],
            expert_batch_assignment=semantic["expert_batch_assignment"],
            tensor_blocks=semantic["tensor_blocks"], tensor_effective_flops=semantic["tensor_effective_flops"],
            vector_elements_by_type=semantic["vector_elements_by_type"],
            dispatch_behaviours=semantic["dispatches"], combine_behaviours=semantic["combines"],
            embedded_rmsnorm_behaviours=semantic["embedded_rmsnorm_behaviours"],
            initial_events=graph["events"], initial_communication_bytes=graph["communication_bytes"],
            tensor_service_ns=graph["tensor_service_ns"], vector_service_ns=graph["vector_service_ns"],
            source=str(path.relative_to(ROOT)), source_sha256=previous.sha(path)))
    return rows


def execution_summary(out, candidates, progress, expected_count=440):
    batches = [dict(file=str(p.relative_to(ROOT)), **read(p)) for p in sorted(out.glob("*.batch.json"))]
    starts = [dict(file=str(p.relative_to(ROOT)), **read(p)) for p in sorted(out.glob("*start*.json"))]
    starts = [r for r in starts if "created_utc" in r]
    finish_path = out/"controller_finish.json"
    finish = read(finish_path) if finish_path.exists() else None
    first = min(starts,key=lambda r:r["created_utc"]) if starts else None
    failure_history = [dict(file=str(p.relative_to(ROOT)), **read(p)) for p in sorted(out.glob("failed_attempts/*/*/receipt.json"))]
    failed_log_names = {Path(r["log"]).name for r in failure_history}
    clocks = []
    if first:
        origin = datetime.fromisoformat(first["created_utc"]).timestamp()
        for row in candidates:
            name = row["candidate_id"]
            logs = list((out/"logs").glob(name+".*.log"))
            if not logs and progress:
                continue
            complete, failed_logs, empty_logs = [], [], []
            for path in logs:
                if not path.stat().st_size:
                    empty_logs.append(path)
                    continue
                last_line = path.read_text(encoding="utf-8").splitlines()[-1]
                last = json.loads(last_line) if last_line.startswith("{") else {}
                if last.get("status") == "ok" and last.get("candidate_id") == name:
                    assert last["seconds"] == row["full_job_seconds"]
                    complete.append(path)
                else:
                    assert path.name in failed_log_names, (name, "non-complete log needs archived failed receipt", path.name)
                    failed_logs.append(path)
            assert len(complete) == 1, (name, len(complete))
            path = complete[0]
            start = int(path.name[len(name)+1:-4])/1e9
            end = start+row["process_receipt_seconds"]
            clocks.append(dict(candidate_id=name, batch=row["batch"], log=str(path.relative_to(ROOT)),
                approximate_started_utc=datetime.fromtimestamp(start,timezone.utc).isoformat(),
                minutes_before_launch=(start-origin)/60, receipt_wall_seconds=row["process_receipt_seconds"],
                approximate_completed_utc=datetime.fromtimestamp(end,timezone.utc).isoformat(),
                approximate_completed_beijing=datetime.fromtimestamp(end,timezone(timedelta(hours=8))).isoformat(),
                minutes_from_stage_start=(end-origin)/60,
                empty_prelaunch_logs=[str(p.relative_to(ROOT)) for p in empty_logs],
                failed_attempt_logs=[str(p.relative_to(ROOT)) for p in failed_logs]))
    observations = []
    for path in sorted(out.glob("*.batch.json"))+sorted(out.glob("*.status*.json")):
        value = read(path)
        if "min_available_gib" in value:
            observations.append(dict(file=str(path.relative_to(ROOT)),
                maximum_active=value.get("max_combined_active",value.get("max_active",0)), minimum_free_gib=value["min_available_gib"]))
    result = dict(batches=batches, starts=starts, controller_finish=finish, resource_observations=observations,
                  failed_attempts=failure_history)
    if observations:
        result.update(maximum_active=max(r["maximum_active"] for r in observations),
                      minimum_available_gib=min(r["minimum_free_gib"] for r in observations))
    if first and finish:
        result.update(first_start_utc=first["created_utc"], completion_utc=finish["utc"],
                      full_wall_seconds=datetime.fromisoformat(finish["utc"]).timestamp()-origin)
    if not progress:
        assert len(clocks) == expected_count and first and finish and finish["rc"] == 0 and finish["completed"] == expected_count
        assert any(r["completed"] == expected_count and not r["failed"] and not r["pending"] for r in batches)
    return result, clocks


def timing_summary(candidates, times, clocks, counts=COUNTS, total="all440"):
    clock_map = {r["candidate_id"]:r for r in clocks}
    timing_map = {r["candidate_id"]:r for r in times}
    rows = []
    for batch in (*counts,*([total] if len(counts)>1 else [])):
        group = [r for r in candidates if r["batch"] == batch or batch == total]
        if not group:
            continue
        sub = [timing_map[r["candidate_id"]] for r in group]
        row = dict(batch=batch, completed=len(group), planned=sum(counts.values()) if batch==total else counts[batch],
            job_process_hours=sum(r["full_job_seconds"] for r in group)/3600,
            receipt_process_hours=sum(r["process_receipt_seconds"] for r in group)/3600,
            search_process_hours=sum(r["search_seconds"] for r in sub)/3600,
            backend_process_hours=sum(r["backend_calls_seconds"]+r.get("terminal_backend_calls_seconds",0) for r in sub)/3600,
            random_cpcm_process_hours=sum(r["random_cpcm_calls_seconds"] for r in sub)/3600,
            longest_job_seconds=max(r["full_job_seconds"] for r in group),
            maximum_peak_rss_gib=max(r["peak_rss_gib"] for r in group))
        row["other_process_hours"] = row["job_process_hours"]-sum(row[k] for k in ("search_process_hours","backend_process_hours","random_cpcm_process_hours"))
        if len(group) == row["planned"] and all(r["candidate_id"] in clock_map for r in group):
            last = max((clock_map[r["candidate_id"]] for r in group),key=lambda r:r["minutes_from_stage_start"])
            row.update(approximate_complete_beijing=last["approximate_completed_beijing"],
                       minutes_from_stage_start=last["minutes_from_stage_start"])
        rows.append(row)
    return rows


def estimate_comparison(candidates, execution, clocks, stage="h3_i2"):
    path = HERE/"tables"/(stage+"_estimates.csv")
    with path.open(encoding="utf-8-sig",newline="") as stream:
        predictions = {r["candidate_id"]:r for r in csv.DictReader(stream)}
    plan = read(HERE/"evidence"/(stage+"_estimates.json"))
    if stage == "h3_i2":
        assert len(predictions) == 440 and plan["counts"] == COUNTS
        summaries, total = plan["summary"], "all440"
    else:
        assert stage == "i3" and len(predictions) == plan["jobs"] == 108
        summaries = [dict(batch="I3", jobs=108, process_hours=plan["process_hours"],
            queue_minutes=plan["queue_minutes_at_84_slots"], planning_minutes=plan["planning_minutes"])]
        total = "I3"
    jobs = []
    for row in candidates:
        pred = predictions[row["candidate_id"]]
        jobs.append(dict(candidate_id=row["candidate_id"],batch=row["batch"],**{k:row[k] for k in KEYS},
            original_estimated_seconds=float(pred["estimated_seconds"]), measured_job_seconds=row["full_job_seconds"],
            measured_receipt_seconds=row["process_receipt_seconds"],
            measured_vs_original=row["full_job_seconds"]/float(pred["estimated_seconds"]),
            original_estimated_rss_gib=float(pred["estimated_rss_gib"]), measured_peak_rss_gib=row["peak_rss_gib"],
            estimate_source=pred["cost_source"], estimate_anchors=pred["anchors"],
            original_included_terminal_seconds=float(pred["terminal_extra_seconds"]) if stage == "h3_i2" else 0.))
    output = []
    for predicted in summaries:
        batch = predicted["batch"]
        group = [r for r in jobs if r["batch"] == batch or batch == total]
        row = dict(batch=batch,planned=predicted["jobs"],completed=len(group),
            original_estimated_process_hours=predicted["process_hours"],
            original_queue_minutes=predicted["queue_minutes"],
            original_planning_low_minutes=predicted["planning_minutes"][0],
            original_planning_high_minutes=predicted["planning_minutes"][1],
            measured_job_process_hours=sum(r["measured_job_seconds"] for r in group)/3600,
            measured_receipt_process_hours=sum(r["measured_receipt_seconds"] for r in group)/3600)
        if group:
            row.update(measured_vs_original_median=statistics.median(r["measured_vs_original"] for r in group),
                longest_job_seconds=max(r["measured_job_seconds"] for r in group))
        completed_clocks = [r for r in clocks if r["batch"] == batch or batch == total]
        if len(completed_clocks) == row["planned"]:
            last = max(completed_clocks,key=lambda r:r["minutes_from_stage_start"])
            row.update(approximate_complete_beijing=last["approximate_completed_beijing"],
                       actual_minutes_from_stage_start=last["minutes_from_stage_start"])
        if batch == total and "full_wall_seconds" in execution:
            row.update(actual_minutes_from_stage_start=execution["full_wall_seconds"]/60,
                approximate_complete_beijing=datetime.fromisoformat(execution["completion_utc"]).astimezone(timezone(timedelta(hours=8))).isoformat())
        output.append(row)
    return jobs, output


def eta_history(out, execution):
    path = out/"h3_i2.log"
    if not path.exists():
        return []
    finish = datetime.fromisoformat(execution["completion_utc"]) if "completion_utc" in execution else None
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if record.get("event") != "status" or "estimate" not in record:
            continue
        pred = record["estimate"]
        row = dict(updated_utc=pred["updated_utc"], completed=record["completed"],
            remaining_jobs=pred["remaining_jobs"], remaining_queue_seconds=pred["queue_seconds"],
            planning_remaining_seconds=pred["planning_seconds"], estimated_finish_utc=pred["estimated_finish_utc"],
            measured_class_jobs=pred["measured_class_jobs"], unmeasured_class_jobs=pred["unmeasured_class_jobs"],
            display_factor=1.5 if pred["unmeasured_class_jobs"] == 0 else 3.0,
            estimate_queue_order="estimated duration only; actual execution also uses priority",
            total_forecast_available=pred["queue_seconds"] is not None,
            reason=pred.get("reason",pred.get("assumption","")))
        if finish:
            row["actual_remaining_seconds"] = (finish-datetime.fromisoformat(pred["updated_utc"])).total_seconds()
        output.append(row)
    return output


def i3_eta_history(out, execution):
    path = out/"i3.log"
    if not path.exists():
        return []
    finish = datetime.fromisoformat(execution["completion_utc"]) if "completion_utc" in execution else None
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if record.get("event") != "status" or "estimate" not in record:
            continue
        prediction = record["estimate"]
        row = dict(completed=record["completed"], prediction=prediction,
            queue_seconds=prediction.get("queue_seconds"), planning_seconds=prediction.get("planning_seconds"),
            estimated_finish_utc=prediction.get("estimated_finish_utc"),
            updated_utc=prediction.get("updated_utc"), capacity_assumption=prediction["capacity_assumption"])
        if finish and row["updated_utc"]:
            row["actual_remaining_seconds"] = (finish-datetime.fromisoformat(row["updated_utc"])).total_seconds()
        output.append(row)
    return output


def qwen_watos_document(operators, candidates):
    cells = [r for r in operators if r["model"] == "qwen3moe" and r["hardware_profile"] == "watos" and r["completed"] == 14]
    count = sum(r["model"] == "qwen3moe" and r["hardware_profile"] == "watos" for r in candidates)
    lines = ["## I3：Qwen WATOS 算子", "",
        f"已完成 {count}/112 个候选、{len(cells)}/8 个完整算子×方法组，包括原 I1 四个 A3h 试点及新 I3 的 108 项。只覆盖 prefill/decode 的 MHA、FFN；未测 WATOS LN/projection，不组装 WATOS 完整层。", "",
        "| 阶段 | 算子 | A1 后端选优 ms | A3h 后端选优 ms | A3h/A1−1 | 随机 CPCM 选优 A3h/A1−1 |", "|---|---|---|---|---|---|"]
    comparisons = watos_method_comparison(operators,"qwen3moe")
    lookup = {(r["phase"],r["op"],r["protocol"]):r for r in comparisons}
    for phase in PHASES:
        for op in ("mha","ffn"):
            if (phase,op,"backend") not in lookup:
                continue
            row, random = lookup[phase,op,"backend"],lookup[phase,op,"random_cpcm"]
            lines.append(f"| {phase} | {op} | {row['A1_backend_mean_ns']/1e6:.4f} | {row['A3h_backend_mean_ns']/1e6:.4f} | {row['A3h_relative_to_A1']:+.2%} | {random['A3h_relative_to_A1']:+.2%} |")
    lines += ["", "只对十四候选齐备的方法组作比较。全部选优规则、候选身份、与主硬件的同方法比较另见 `hi/tables/i3_watos_method_comparison.csv`、`i3_hardware_comparison.csv`。"]
    return lines


def quality_document(operators, layers, references, candidates, workload):
    qwen = [r for r in candidates if r["model"] == "qwen3moe" and r["hardware_profile"] == "busybarn"]
    lookup = {(r["phase"],r["op"],r["method"]):r for r in operators
        if r["model"] == "qwen3moe" and r["hardware_profile"] == "busybarn" and r["completed"] == 14}
    lines = ["# I 阶段 Qwen 健康结果", "",
        f"主硬件已完成 {len(qwen)}/336 候选，完整算子×方法组 {len(lookup)}/24。固定 batch=16、hidden=2048、prefill/KV=2048；每个代表 MHA 含 8 个 Q 头与一组 K/V，层内 MHA 乘数为 4。",
        "每组十四候选，每候选完整 1000 步、后端种子 1–10、十次随机平局 CPCM。I0 的 28 个历史候选未满足执行次序复用要求，本阶段按新模型合同搜索；八个 I1 试点各自计入原定候选集合。", "",
        "## 八算子结果", "", "后端十种子均值在同方法十四候选内选优。数值为毫秒 [最小, 最大]。", "",
        "| 阶段 | 算子 | A1 | A3h | Gemini | A3h/A1−1 | A3h/Gemini−1 |", "|---|---|---|---|---|---|---|"]
    for phase in PHASES:
        for op in OPS:
            if not all((phase,op,m) in lookup for m in METHODS):
                continue
            group = [lookup[phase,op,m] for m in METHODS]
            values = [f"{r['backend_mean_ns']/1e6:.4f} [{r['backend_min_ns']/1e6:.4f}, {r['backend_max_ns']/1e6:.4f}]" for r in group]
            lines.append(f"| {phase} | {op} | {' | '.join(values)} | {group[1]['backend_mean_ns']/group[0]['backend_mean_ns']-1:+.2%} | {group[1]['backend_mean_ns']/group[2]['backend_mean_ns']-1:+.2%} |")
    if all((phase,op,m) in lookup for phase in PHASES for op in ("ln","mha","proj") for m in METHODS):
        delta = lambda phase,op: lookup[phase,op,"A3h"]["backend_mean_ns"]/lookup[phase,op,"A1"]["backend_mean_ns"]-1
        ln_diff_us = (lookup["decode","ln","A3h"]["backend_mean_ns"]-lookup["decode","ln","A1"]["backend_mean_ns"])/1000
        lines += ["", f"已齐的 LN/MHA/projection 中，A3h 相对 A1 的 prefill/decode MHA 分别变化 {delta('prefill','mha'):+.2%}/{delta('decode','mha'):+.2%}，decode projection 变化 {delta('decode','proj'):+.2%}。Prefill LN 与 projection 分别变化 {delta('prefill','ln'):+.2%}/{delta('prefill','proj'):+.2%}。decode LN 变化 {delta('decode','ln'):+.2%}，绝对差为 {ln_diff_us:+.3f} µs；所有差值均保留在层组装中。"]
        if len(lookup) < 24:
            lines += ["FFN 的方法组仍未全部齐备，完整层结果等待这些候选完成。"]
    lines += ["", "## 三种选优与统一 CPCM 对照", "",
        "自身代价沿用原定义：A1 为归一化加权聚合能量，A3h 先按搜索 T_sel 再按自身能量，Gemini 为 hops-only 能量；另列统一确定性 CPCM。随机选择使用十次平局估值均值。完全同值按预定 candidate_order 处理。后端选优提供质量参考，其余选择不读取后端答案。", "",
        "| 方法 | 规则 | 八算子选择损失中位数 | 最大值 | 相同后端赢家 |", "|---|---|---|---|---|"]
    for method in METHODS:
        group = [r for r in lookup.values() if r["method"] == method]
        if len(group) != 8:
            continue
        for protocol in PROTOCOLS[1:]:
            gaps = [r[protocol+"_selection_gap"] for r in group]
            lines.append(f"| {method} | {LABELS[protocol]} | {statistics.median(gaps):+.2%} | {max(gaps):+.2%} | {sum(r[protocol+'_candidate']==r['backend_candidate'] for r in group)}/8 |")
    lines += ["", "## 单层与请求", "",
        "Prefill = 2×RMSNorm + 4×代表 MHA + projection + FFN + handoff；decode = RMSNorm + 4×代表 MHA + projection + FFN + handoff。decode FFN 已含一次 RMSNorm。handoff 为 batch×tokens×2048×2 字节经过一条相邻组 D2D 的解析项：256 B/ns + 20 ns，prefill 524,308 ns、decode 276 ns。",
        "请求 N 表示首 token 之后追加的 decode 步数，使用 P+N×D，KV 保持 2048。算子和、handoff 和来源候选分列保留；这是算子结果串行组装。", "",
        "| 规则 | 方法 | TTFT ms | TPOT ms | N=128 s | N=512 s | N=2048 s |", "|---|---|---|---|---|---|---|"]
    for row in layers:
        values = " | ".join(f"{row[f'request_additional_decode_{n}_ns']/1e9:.5f}" for n in (128,512,2048))
        lines.append(f"| {LABELS[row['protocol']]} | {row['method']} | {row['TTFT_ms']:.4f} | {row['TPOT_ms']:.4f} | {values} |")
    if len(layers) == 12:
        cell = {(r["protocol"],r["method"]):r for r in layers}
        for protocol in ("backend","random_cpcm"):
            a,b = cell[protocol,"A3h"],cell[protocol,"A1"]
            lines += ["", f"{LABELS[protocol]}：A3h 相对 A1，prefill {a['prefill_ns']/b['prefill_ns']-1:+.2%}，decode {a['decode_ns']/b['decode_ns']-1:+.2%}，N=2048 请求 {a['request_additional_decode_2048_ns']/b['request_additional_decode_2048_ns']-1:+.2%}。"]
    lines += ["", "## 48 层解析参考", "",
        "以 artifact 的 48 层为参考，B 为实际布局中的跨组层连接数：T48_P=48×P_operator_sum+B×524,308 ns；T48_D=48×D_operator_sum+B×276 ns。请求仍为 P+N×D。布局确定后代入 B；完整模型同时驻留、容量、共享链路竞争与流水重叠未在这里模拟。", "",
        "| 随机 CPCM 选优 | 48 层 P 算子和 s | 48 层 D 算子和 ms | N=2048 算子和 s |", "|---|---|---|---|"]
    for row in references:
        if row["protocol"] == "random_cpcm":
            lines.append(f"| {row['method']} | {row['prefill_constant_ns']/1e9:.6f} | {row['decode_constant_ns']/1e6:.6f} | {row['request_additional_decode_2048_constant_ns']/1e9:.6f} |")
    lines += ["", "## 专家工作量与图规模", "",
        "固定 128 个专家、top-k=8、每专家维度 768，按 expert_idx % 16 分配 batch。以下为 H0/I0 已核对的主硬件 A3h g0/s123 初始图，工作量按实际张量块计入 grain 舍入；它说明输入规模，不代表各个搜索终态的路由。", "",
        "| 阶段 | Tensor 块 | 有效计算 FLOP | 初始事件 | 初始通信 GiB | 内嵌 RMSNorm 行为 |", "|---|---|---|---|---|---|"]
    for row in workload:
        lines.append(f"| {row['phase']} | {row['tensor_blocks']} | {row['tensor_effective_flops']:,} | {row['initial_events']} | {row['initial_communication_bytes']/2**30:.3f} | {row['embedded_rmsnorm_behaviours']} |")
    lines += ["", "逐候选终态事件数、实际调度活动数、峰值 RSS、完整耗时和十种子原始值见 `hi/tables/h3_i2_candidates.csv`、`h3_i2_samples.csv`；专家向量工作量分项见 `h3_i2_workload.csv`。", ""]
    lines += qwen_watos_document(operators,candidates)
    return lines


def render_stage(candidates, new, operators, hardware, timing_rows, execution, missing, estimates, watos_methods, clocks, i3_count):
    counts = Counter(r["model"] for r in candidates)
    h_watos = [r for r in candidates if r["model"] == "llama" and r["hardware_profile"] == "watos"]
    h_watos_groups = sum(r["completed"] == 14 for r in operators if r["model"] == "llama" and r["hardware_profile"] == "watos")
    i_main_groups = sum(r["completed"] == 14 for r in operators if r["model"] == "qwen3moe" and r["hardware_profile"] == "busybarn")
    lines = ["# H3 / I2 / I3 健康实验", "",
        f"H3/I2 已完成 {len(new)}/440 个新候选，随后授权的 I3 已完成 {i3_count}/108，复用前两批 348 个，共 {len(candidates)}/896 个；Llama {counts['llama']}/448，Qwen {counts['qwen3moe']}/448。H3=108、I2D=166、I2P=166 保留原批次；I3 独立运行、独立记时。",
        "旧候选保持原样；新旧候选均核对模型合同、完整 1000 步、后端十种子、十次随机平局 CPCM、receipt 身份和文件摘要。", ""]
    if "full_wall_seconds" in execution:
        lines += [f"H3/I2 这 440 项从 {execution['first_start_utc']} 到 {execution['completion_utc']}，墙钟 {execution['full_wall_seconds']/60:.3f} 分钟。"]
    if "maximum_active" in execution:
        lines += [f"峰值合计 {execution['maximum_active']} 并行，最低可用内存 {execution['minimum_available_gib']:.3f} GiB；归档失败尝试 {len(execution['failed_attempts'])} 次。"]
    lines += ["", "## 预计用时与实际进展", "",
        "这轮预估使用已测 Qwen FFN 的完整作业成本，包含十种子后端；未测 A1/Gemini 暂估搜索比例，LN/projection 使用已有图规模。24 个未测类的首候选优先开始，空闲槽位继续运行其余候选；首候选结束后更新该类完整耗时和 RSS。初始范围为排程值至其两倍，不是置信区间；包含搜索、校准、评估、导出/重载，不含后续故障、联合、J、下载与汇总。", "",
        "| 批次 | 完成/计划 | 原进程小时 | 实际进程小时 | 原墙钟范围 min | 共享队列收齐 min |", "|---|---|---|---|---|---|"]
    for row in estimates:
        actual = f"{row['actual_minutes_from_stage_start']:.3f}" if "actual_minutes_from_stage_start" in row else "未收齐/日志待下载"
        lines.append(f"| {row['batch']} | {row['completed']}/{row['planned']} | {row['original_estimated_process_hours']:.3f} | {row['measured_job_process_hours']:.3f} | {row['original_planning_low_minutes']:.2f}–{row['original_planning_high_minutes']:.2f} | {actual} |")
    lines += ["", "三批共享 128 个槽位；各批原墙钟是独立排程预测，不能相加。实际收齐从本轮最初启动时刻统一起算，整轮以 controller_finish 为准。逐作业估计差值及监督器历次剩余时间预测另存 `h3_i2_candidate_estimate_comparison.csv`、`h3_i2_eta_history.csv`。",
        "初始计划范围采用排程值的 1–2 倍；运行中显示沿用已有监督器，仍有未测类别时显示 1–3 倍，全部剩余类别已测时显示 1–1.5 倍。因此首条约 116–349 分钟来自旧的显示系数，尚无新的变慢测量。在线 ETA 按预计时长排序，而实际队列还采用首候选和 Qwen prefill FFN 优先级；完整记录保留这一区别。作业超过现有估计时，监督器展示阶段/步数并暂不提供总剩余时间。"]
    lines += ["", "## 分项用时", "", "进程小时为作业用时之和；各批实际收齐时间按日志启动时间 + receipt 用时取最大值，含约 2 秒轮询误差。", "",
        "| 批次 | 完成/计划 | 搜索 PH | 后端 PH | 随机 CPCM PH | 其余 PH | 完整 PH | 最长任务 min | 从启动到收齐 min |", "|---|---|---|---|---|---|---|---|---|"]
    for row in timing_rows:
        done = f"{row['minutes_from_stage_start']:.3f}" if "minutes_from_stage_start" in row else "未收齐/日志待下载"
        lines.append(f"| {row['batch']} | {row['completed']}/{row['planned']} | {row['search_process_hours']:.3f} | {row['backend_process_hours']:.3f} | {row['random_cpcm_process_hours']:.3f} | {row['other_process_hours']:.3f} | {row['job_process_hours']:.3f} | {row['longest_job_seconds']/60:.2f} | {done} |")
    lines += ["", "完整分项保留模板、校准、搜索、状态恢复、后端、随机 CPCM、评估准备、导出和重载；Gemini 的 seed123 最好/终态两次评估分别记账。并行用时含资源争用，正式串行计时另属 J 阶段。", "",
        "## H3：Llama WATOS 算子", "", "主硬件与 WATOS 均在相同方法的十四候选中选取，以下为后端均值选优。两种硬件的候选数相等。", "",
        "| 阶段 | 算子 | 方法 | BusyBarn ms | WATOS ms | WATOS / BusyBarn − 1 |", "|---|---|---|---|---|---|"]
    for row in hardware:
        if row["protocol"] == "backend":
            lines.append(f"| {row['phase']} | {row['op']} | {row['method']} | {row['busybarn_ns']/1e6:.4f} | {row['watos_ns']/1e6:.4f} | {row['watos_relative_to_busybarn']:+.2%} |")
    if len(watos_methods) == 16:
        by_cell = {(r["phase"],r["op"],r["protocol"]):r for r in watos_methods}
        lines += ["", "### 相同 WATOS 硬件上的 A1 与 A3h", "",
            "| 阶段 | 算子 | 后端选优 A3h/A1−1 | 随机 CPCM 选优 A3h/A1−1 |", "|---|---|---|---|"]
        for phase in PHASES:
            for op in ("mha","ffn"):
                back, rand = by_cell[phase,op,"backend"],by_cell[phase,op,"random_cpcm"]
                lines.append(f"| {phase} | {op} | {back['A3h_relative_to_A1']:+.2%} | {rand['A3h_relative_to_A1']:+.2%} |")
        lines += ["", f"在 WATOS 上，A3h 的 decode MHA/FFN 分别较 A1 变化 {by_cell['decode','mha','backend']['A3h_relative_to_A1']:+.2%}/{by_cell['decode','ffn','backend']['A3h_relative_to_A1']:+.2%}；prefill MHA 变化 {by_cell['prefill','mha','backend']['A3h_relative_to_A1']:+.2%}。Prefill FFN 的后端选优差异仅 {by_cell['prefill','ffn','backend']['A3h_relative_to_A1']:+.2%}，随机 CPCM 选择后为 {by_cell['prefill','ffn','random_cpcm']['A3h_relative_to_A1']:+.2%}，因此该点没有同样的改善。",
            "跨硬件变化主要体现在 prefill FFN：两种方法在 WATOS 上的时间均下降约 75%；decode FFN 的硬件变化较小。以上比较采用各硬件自己的十四候选，不把单个预定试点与十四候选赢家相比。",
            "", "| WATOS 方法 | 随机 CPCM 选优损失中位数 | 最大值 | 与后端选优同一状态 |", "|---|---|---|---|"]
        for method in ("A1","A3h"):
            group = [r for r in operators if r["model"] == "llama" and r["hardware_profile"] == "watos" and r["method"] == method]
            gaps = [r["random_cpcm_selection_gap"] for r in group]
            lines.append(f"| {method} | {statistics.median(gaps):+.2%} | {max(gaps):+.2%} | {sum(r['random_cpcm_candidate']==r['backend_candidate'] for r in group)}/4 |")
        h_timing = next(r for r in timing_rows if r["batch"] == "H3")
        if "minutes_from_stage_start" in h_timing:
            last = max((r for r in clocks if r["batch"] == "H3"),key=lambda r:r["minutes_from_stage_start"])
            lines += ["", f"H3 的 108 项新增候选累计 {h_timing['job_process_hours']:.3f} 进程小时，最长单任务 {h_timing['longest_job_seconds']/60:.2f} 分钟；在共享队列中于 {h_timing['approximate_complete_beijing']} 收齐，距整轮启动 {h_timing['minutes_from_stage_start']:.3f} 分钟。最后完成的 `{last['candidate_id']}` 在启动后 {last['minutes_before_launch']:.3f} 分钟进入运行，说明该批收齐时间包含与 Qwen 长任务共享槽位的等待。"]
        if (HERE/"figures/h3_i2_llama_hardware.png").exists():
            lines += ["", "![Llama 两种硬件的算子结果](figures/h3_i2_llama_hardware.png)", "",
                "图中每个算子以对应 BusyBarn A1 的随机 CPCM 选择结果为固定分母；柱为随机 CPCM 选择，白色菱形为后端选优，全部来自同样十四候选的实际后端均值。"]
    lines += ["", f"当前 Llama WATOS 已完成 {len(h_watos)}/112 个候选、{h_watos_groups}/8 个完整算子×方法组，覆盖 P/D 的 MHA/FFN。LN/projection 不在当前 WATOS 计划内，因此不据此组装完整层。全部三种选择及统一 CPCM 对照在 `tables/h3_i2_operators.csv` 与 `h3_i2_hardware_comparison.csv`。", "",
        "## I2：Qwen 主硬件", "", f"Qwen 主硬件目前 {i_main_groups}/24 个完整算子×方法组。逐算子结果及齐备后生成的完整单层、48 层参数化参考见 `../I_HEALTH_SUMMARY.md`。头组乘数为 4，decode FFN 已包含一次 RMSNorm；组装使用同一模型合同。", "",
        "## 新旧样本计数", "", "H3/I2 的 440 项计划包含 4400 个主后端样本 + 8 个新 Qwen Gemini 终态×10=4480 个后端，随机 CPCM 4400 次。I3 另有 1080 个后端和 1080 个随机 CPCM，无 Gemini。结合旧 348 候选与旧 8 个终态，健康全量为 8960 个主后端 +160 个终态后端=9120 个，随机 CPCM 8960 次。终态对照复用同一搜索，不计新候选。"]
    if missing:
        lines += ["", f"当前尚缺 {len(missing)} 个新候选；完整组才生成十四候选选择，尚未完成的组不作完整比较。"]
    return lines


def render_i3(candidates, operators, times, execution, missing, estimates):
    lines = ["", "## I3 独立运行与用时", "",
        f"I3 新增 {len(candidates)}/108 项，目录 `out/i3_108`。原 I1 四个 WATOS A3h 试点计入 112 个正式候选，未重跑。首四个 A1 类代表优先开始，其余 prefill FFN 优先；全部候选直接释放。",
        "新池先为原 440 项中尚未成功的任务保留名额，两池合计上限 128，系统预留 3 GiB。后续其他已批准批次另记；同机资源是共享的，I3 的墙钟不会与 H3/I2 墙钟相加。", "",
        "| 完成/计划 | 原进程小时 | 实际进程小时 | 初始墙钟范围 min | I3 启动到完成 min |", "|---|---|---|---|---|"]
    for row in estimates:
        actual = f"{row['actual_minutes_from_stage_start']:.3f}" if "actual_minutes_from_stage_start" in row else "未收齐/日志待下载"
        lines.append(f"| {row['completed']}/108 | {row['original_estimated_process_hours']:.3f} | {row['measured_job_process_hours']:.3f} | {row['original_planning_low_minutes']:.2f}–{row['original_planning_high_minutes']:.2f} | {actual} |")
    if "full_wall_seconds" in execution:
        lines += ["",f"I3 从 {execution['first_start_utc']} 到 {execution['completion_utc']}，共 {execution['full_wall_seconds']/60:.3f} 分钟。"]
    lines += ["", "| 搜索 PH | 后端 PH | 随机 CPCM PH | 其余 PH | 完整 PH | 最长任务 min | 峰值单作业 RSS GiB |", "|---|---|---|---|---|---|---|"]
    for row in times:
        lines.append(f"| {row['search_process_hours']:.3f} | {row['backend_process_hours']:.3f} | {row['random_cpcm_process_hours']:.3f} | {row['other_process_hours']:.3f} | {row['job_process_hours']:.3f} | {row['longest_job_seconds']/60:.2f} | {row['maximum_peak_rss_gib']:.3f} |")
    lines += ["", "I3 初始使用 84 个可用槽、200 GiB 作业预算估计，实际槽位随旧任务完成而增加。初始 99–198 分钟包含完整搜索、十后端、十随机 CPCM 与导出；数据下载与汇总另计。逐作业实测/原估计、I3 自身完成时间及在线预测存于 `tables/i3_candidate_estimate_comparison.csv`、`i3_job_completion_times.csv`、`i3_eta_history.csv`。"]
    if missing:
        lines += ["",f"I3 当前尚缺 {len(missing)} 个结果；候选组齐备后才生成 WATOS 方法比较。"]
    return lines


def run(args):
    old = previous.collect(args.previous_out, args.pilot_out, False)
    new = collect_new(args.out, args.progress)
    i3 = collect_new(args.i3_out, args.progress, I3_COUNTS)
    candidates, samples, times, audits, terminals = [list(old[i])+list(new[i])+list(i3[i]) for i in range(5)]
    assert len(old[0]) == 348 and len({r["candidate_id"] for r in candidates}) == len(candidates)
    operators = previous.select_operators(candidates)
    layers = qwen_layers(operators)
    references = qwen_48_reference(layers)
    hardware = hardware_comparison(operators)
    watos_methods = watos_method_comparison(operators)
    workload = workload_rows()
    execution, clocks = execution_summary(args.out, new[0], args.progress)
    timing_rows = timing_summary(new[0], new[2], clocks)
    estimate_jobs, estimates = estimate_comparison(new[0],execution,clocks)
    history = eta_history(args.out,execution)
    i3_execution, i3_clocks = execution_summary(args.i3_out,i3[0],args.progress,108)
    i3_times = timing_summary(i3[0],i3[2],i3_clocks,I3_COUNTS,"I3")
    i3_jobs_estimates, i3_estimates = estimate_comparison(i3[0],i3_execution,i3_clocks,"i3")
    i3_history = i3_eta_history(args.i3_out,i3_execution)
    finish = execution["controller_finish"]
    complete440 = len(new[0]) == 440 and finish is not None and finish["rc"] == 0 and finish["completed"] == 440
    i3_finish = i3_execution["controller_finish"]
    complete_i3 = len(i3[0]) == 108 and i3_finish is not None and i3_finish["rc"] == 0 and i3_finish["completed"] == 108
    complete = complete440 and complete_i3
    evidence = dict(status="complete" if complete else "in_progress", new_expected=440, new_complete=len(new[0]),
        h3_i2_complete=complete440,i3_expected=108,i3_complete=len(i3[0]),i3_finished=complete_i3,
        previous_candidates=348, candidates=len(candidates), per_model=dict(Counter(r["model"] for r in candidates)),
        new_main_backend_samples=len(new[1]),i3_main_backend_samples=len(i3[1]),previous_main_backend_samples=len(old[1]), main_backend_samples=len(samples),
        new_terminal_states=len(new[4]), previous_terminal_states=len(old[4]), terminal_backend_samples=10*len(terminals),
        total_backend_samples=len(samples)+10*len(terminals), random_cpcm_samples=len(samples),
        complete_operator_groups=sum(r["completed"] == 14 for r in operators), qwen_layer_rows=len(layers),
        execution=execution,i3_execution=i3_execution, original_estimate_sha256=previous.sha(HERE/"evidence/h3_i2_estimates.json"),
        i3_original_estimate_sha256=previous.sha(HERE/"evidence/i3_estimates.json"),
        report_sources_sha256={path.name:previous.sha(path) for path in (Path(__file__),Path(previous.__file__))},
        missing=new[5],i3_missing=i3[5], input_audits=audits,
        model_contracts={m:hi_config.model_contract(m) for m in hi_config.MODELS})
    if not args.progress:
        assert complete and len(candidates) == 896 and len(samples) == 8960 and len(terminals) == 16
        assert evidence["per_model"] == dict(llama=448,qwen3moe=448)
        assert evidence["complete_operator_groups"] == 64 and len(layers) == len(references) == 12
        formal = hi_manifest.candidates()
        assert {r["candidate_id"] for r in candidates} == {r["candidate_id"] for r in formal}
        expected_groups = Counter(tuple(s[k] for k in KEYS) for s in formal)
        assert {tuple(r[k] for k in KEYS):r["completed"] for r in operators} == dict(expected_groups)
    files = dict(candidates=candidates, samples=samples, timing_exclusive=times, operators=operators,
        qwen_layers=layers, qwen_fullmodel_reference=references, hardware_comparison=hardware,
        watos_method_comparison=watos_methods,
        class_timings=previous.class_timings(candidates), stage_timings=timing_rows, job_completion_times=clocks,
        candidate_estimate_comparison=estimate_jobs, estimate_comparison=estimates, eta_history=history,
        workload=workload, gemini_terminal=[dict(model="qwen3moe" if r["candidate_id"].startswith("I_") else "llama",**r) for r in terminals],
        terminal_samples=[dict(candidate_id=r["candidate_id"]+"_terminal",phase=r["phase"],op=r["op"],**sample)
                          for r in terminals for sample in r["backend_samples"]])
    for name, rows in files.items():
        previous.table(args.destination/"tables"/(PREFIX+name+".csv"), rows)
    i3_files = dict(candidates=i3[0],samples=i3[1],timing_exclusive=i3[2],stage_timings=i3_times,
        job_completion_times=i3_clocks,candidate_estimate_comparison=i3_jobs_estimates,estimate_comparison=i3_estimates,
        eta_history=i3_history,hardware_comparison=hardware_comparison(operators,"qwen3moe"),
        watos_method_comparison=watos_method_comparison(operators,"qwen3moe"))
    for name,rows in i3_files.items():
        previous.table(args.destination/"tables"/("i3_"+name+".csv"),rows)
    write(args.destination/"evidence/h3_i2_completion_audit.json", evidence)
    write(args.destination/"evidence/i3_completion_audit.json",dict(status="complete" if complete_i3 else "in_progress",
        expected=108,completed=len(i3[0]),main_backend_samples=len(i3[1]),random_cpcm_samples=len(i3[1]),
        missing=i3[5],execution=i3_execution,input_audits=i3[3]))
    document = render_stage(candidates,new[0],operators,hardware,timing_rows,execution,new[5],estimates,watos_methods,clocks,len(i3[0]))
    document += render_i3(i3[0],operators,i3_times,i3_execution,i3[5],i3_estimates)
    (args.destination/"H3_I2_REPORT.md").write_text("\n".join(document)+"\n",encoding="utf-8")
    args.health_report.parent.mkdir(parents=True,exist_ok=True)
    args.health_report.write_text("\n".join(quality_document(operators,layers,references,candidates,workload))+"\n",encoding="utf-8")
    print(json.dumps({k:evidence[k] for k in ("status","new_complete","i3_complete","previous_candidates","candidates","main_backend_samples","terminal_backend_samples","complete_operator_groups","qwen_layer_rows")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE/"out/h3_i2_440")
    parser.add_argument("--i3-out",type=Path,default=HERE/"out/i3_108")
    parser.add_argument("--previous-out",type=Path,default=HERE/"out/next_344")
    parser.add_argument("--pilot-out",type=Path,default=HERE/"out/h1_main")
    parser.add_argument("--destination",type=Path,default=HERE)
    parser.add_argument("--health-report",type=Path,default=ROOT/"e2e/I_HEALTH_SUMMARY.md")
    parser.add_argument("--progress",action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
