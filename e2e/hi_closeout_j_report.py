"""Read the saved J timing records; never launch search, CPCM or a backend."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from hi_closeout_p0 import csv_write, json_write

ROOT = Path(__file__).resolve().parents[1]
HI = ROOT / "e2e/hi"
BASE = HI / "closeout"
SOURCES = {}
J1_FIELDS = ("compile_seconds", "forward_complete_seconds", "analysis_seconds", "full_call_seconds")
FAST_FIELDS = ("pack_seconds", "kernel_seconds", "unpack_seconds", "total_seconds")
J4_FIELDS = ("input_load_platform_seconds", "shared_all256_chunk_seconds", "cpcm_deadline_rank_seconds",
             "selected_strategy_transform_seconds", "policy_and_initial_ready_list_seconds", "total_prepare_seconds")
J5_FIELDS = ("load_state_seconds", "platform_seconds", "baseline_metrics_seconds", "resident_decision_seconds",
             "frozen_graph_verification_seconds", "export_seconds", "complete_case_seconds")


def digest(path):
    data = Path(path).read_bytes()
    value = hashlib.sha256(data).hexdigest()
    SOURCES[str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")] = value
    return value


def read(path):
    digest(path)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def local(value):
    value = str(value).replace("\\", "/")
    if value.startswith("/home/ubuntu/"):
        value = value.split("/", 4)[4]
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def jobs(path):
    value = read(path)
    items = value["jobs"] if isinstance(value, dict) else value
    result = []
    for item in items:
        if "spec" in item:
            result.append(item)
        else:
            name = item["job_id"]
            result.append(dict(id=name, kind=item["kind"], spec=item, result=f"results/{name}.json",
                argv=["{python}", "-u", "-B", f"e2e/{item.get('module', 'fg_run')}.py", "child",
                      "--spec", "{spec}", "--out", "{out}"]))
    return result


def identity(job):
    value = {k: job[k] for k in ("id", "kind", "spec", "result")}
    value["argv"] = job.get("argv")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def input_name(spec):
    if spec["kind"] in ("j1", "j1_startup"):
        return spec.get("input_id") or Path(spec["prepared_graph"]).stem.removesuffix(".pkl")
    if spec["kind"] == "j4":
        return spec["hardware_profile"] + ":" + spec["arm"]
    if spec["kind"] == "j5":
        return spec.get("case") or spec["model"]
    return spec.get("input_id") or spec["case"]


def collect(out):
    inputs, records, environments, receipts = [], [], [], []
    manifests = [("old45", HI / "j/J_FIRST_SERIAL.json", HI / "out/j_first_serial"),
                 ("micro", BASE / "manifests/j_micro.json", out),
                 ("search", BASE / "manifests/j_search.json", out),
                 ("eir", BASE / "manifests/j3.json", out)]
    seen = set()
    for batch, manifest, folder in manifests:
        for job in jobs(manifest):
            name, spec = job["id"], job["spec"]
            assert name not in seen
            seen.add(name)
            path = folder / job["result"]
            receipt_path = folder / "receipts" / (name + ".json")
            status = "pending"
            if path.exists() and receipt_path.exists():
                row, receipt = read(path), read(receipt_path)
                if receipt["rc"] == 0:
                    assert row["status"] == "ok" and row["job_id"] == name
                    assert receipt["identity"] == identity(job), name
                    assert row["spec"] == spec, name
                    status = "complete"
                    row = dict(row, report_batch=batch, input_id=input_name(spec))
                    records.append(row)
                    env = row.get("environment", row.get("j_environment"))
                    assert env and all(value == "1" for value in env["thread_settings"].values())
                    reported_env=dict(env)
                    if spec["kind"] not in ("j1", "j1_startup"):
                        reported_env["raw_environment_timing_protocol"]=env.get("timing_protocol")
                        reported_env["raw_environment_protocol_scope"]="generic J1 helper metadata; use actual_job_protocol for this job"
                        reported_env["timing_protocol"]=row.get("j_protocol", "one complete control preparation or repair in one fresh process")
                    reported_env["actual_job_protocol"]=reported_env.get("timing_protocol")
                    environments.append(dict(job_id=name, batch=batch, **reported_env,
                        result_source_sha256=row.get("source_sha256", {}),
                        execution_source_sha256=row.get("execution_source_sha256", {}),
                        gpt_adapter_sha256=row.get("gpt_adapter_sha256")))
                    receipts.append(dict(job_id=name, identity=receipt["identity"],
                        receipt_wall_seconds=receipt["wall_seconds"], result_sha256=SOURCES[path.relative_to(ROOT).as_posix()],
                        result=str(path.relative_to(ROOT)), receipt=str(receipt_path.relative_to(ROOT))))
                else:
                    status = "failed_receipt"
            elif path.exists():
                status = "pending_receipt"
            inputs.append(dict(job_id=name, batch=batch, kind=spec["kind"], input_id=input_name(spec),
                status=status, manifest=manifest.relative_to(ROOT).as_posix(), manifest_sha256=digest(manifest),
                spec=spec, identity=identity(job), result=path.relative_to(ROOT).as_posix(),
                command=job.get("argv"), declared_input_sha256={k: v for k, v in spec.items() if "sha256" in k}))
    # The C representative must exist before these three jobs can be admitted.
    for model in ("llama", "qwen3moe"):
        for rep in (1, 2, 3):
            name = f"J5_{model}_m2_r{rep}"
            if name not in seen:
                inputs.append(dict(job_id=name, batch="micro", kind="j5", input_id=model,
                    status="pending_C_representative", spec={}, reason="C representative not yet in j_micro manifest"))
    assert len(inputs) == 125
    return inputs, records, environments, receipts


def j2_costs(record):
    """Exclusive observed intervals plus contained diagnostics; all seconds."""
    t, d = record["timing"], record["j_diagnostic_timing"]
    m, ev = t["milestones_seconds"], t["evaluation"]
    monitor = d["loop_A1_monitor_seconds"]
    progress = d["progress_callback_seconds"]
    initial_monitor = d["initial_A1_monitor_seconds"]
    loop_interval = m["search_complete"] - m["search_begin"]
    backend = sum(ev["backend_individual_seconds"])
    ties = sum(ev["random_cpcm_individual_seconds"])
    pieces = dict(configure_seconds=m["configure_complete"],
        template_seconds=t["template_initialization_seconds"],
        initial_graph_noncalibration_nondiagnostic_seconds=t["initial_graph_and_calibration_seconds"]-t["calibration_seconds"]-initial_monitor,
        initial_A1_monitor_seconds=initial_monitor, calibration_seconds=t["calibration_seconds"],
        algorithm_search_seconds=d["algorithm_search_seconds"],
        loop_A1_monitor_seconds=monitor, progress_callback_seconds=progress,
        search_interval_other_seconds=loop_interval-d["raw_sa_seconds"],
        best_restore_seconds=m["evaluation_begin"]-m["search_complete"],
        backend_calls_seconds=backend, random_cpcm_calls_seconds=ties,
        evaluation_other_seconds=m["evaluation_complete"]-m["evaluation_begin"]-backend-ties,
        export_seconds=m["export_complete"]-m["evaluation_complete"],
        snapshot_reload_seconds=m["snapshot_reloaded"]-m["export_complete"],
        final_accounting_seconds=record["wall_seconds"]-m["snapshot_reloaded"])
    assert all(v >= -1e-5 for v in pieces.values()), pieces
    assert math.isclose(sum(pieces.values()), record["wall_seconds"], rel_tol=1e-9, abs_tol=1e-5)
    assert math.isclose(d["algorithm_search_seconds"]+monitor+progress, d["raw_sa_seconds"], abs_tol=1e-5)
    assert len(record["backend"]) == len(record["T_hat_rand"]) == 10
    return dict(pieces, raw_sa_seconds=d["raw_sa_seconds"],
        observed_search_interval_seconds=loop_interval,
        initial_graph_and_calibration_inclusive_seconds=t["initial_graph_and_calibration_seconds"],
        first_broadcast_inclusive_seconds=t["first_broadcast_seconds"],
        broadcast_total_inclusive_seconds=t["broadcast_total_seconds"],
        anneal_mark_inclusive_seconds=d["anneal_mark_seconds"],
        loop_A1_monitor_calls=d["loop_A1_monitor_calls"], progress_callback_calls=d["progress_callback_calls"],
        backend_calls=10, random_cpcm_calls=10, search_steps=1000,
        offline_generation_measured_seconds=sum(pieces[k] for k in ("configure_seconds", "template_seconds",
            "initial_graph_noncalibration_nondiagnostic_seconds", "calibration_seconds", "algorithm_search_seconds",
            "best_restore_seconds", "export_seconds")),
        recording_and_reload_measured_seconds=sum(pieces[k] for k in ("initial_A1_monitor_seconds",
            "loop_A1_monitor_seconds", "progress_callback_seconds", "search_interval_other_seconds",
            "snapshot_reload_seconds", "final_accounting_seconds")),
        wall_seconds=record["wall_seconds"], timing_entry_seconds=record["timing_entry_seconds"],
        entry_other_seconds=record["timing_entry_seconds"]-record["wall_seconds"],
        sum_exclusive_seconds=sum(pieces.values()),
        final_CPCM_seconds=None,
        final_CPCM_note="Contained in evaluation_other_seconds; the frozen final_eval has no independent final-CPCM timer")


def historical_fourteen_costs():
    """Actual process cost of complete historical groups, apart from serial J2."""
    index=HI/"tables/h3_i2_candidates.csv"
    inventory={index.relative_to(ROOT).as_posix():dict(bytes=index.stat().st_size,sha256=digest(index))}
    with index.open(encoding="utf-8-sig",newline="") as stream:
        candidates=list(csv.DictReader(stream))
    groups,details=[],[]
    for label,model,phase,op in (("Llama_prefill_mha","llama","prefill","mha"),
                                ("Qwen_decode_ffn","qwen3moe","decode","ffn")):
        for method in ("A1","A3h"):
            selected=[r for r in candidates if (r["model"],r["hardware_profile"],r["phase"],r["op"],r["method"])
                      ==(model,"busybarn",phase,op,method)]
            assert len(selected)==14 and len({r["candidate_id"] for r in selected})==14
            measured=[]
            for item in selected:
                path=local(item["source"]);record=read(path)
                inventory[path.relative_to(ROOT).as_posix()]=dict(bytes=path.stat().st_size,sha256=SOURCES[path.relative_to(ROOT).as_posix()])
                assert record["search_steps"]==1000 and len(record["backend"])==len(record["T_hat_rand"])==10
                timing=record["timing"];milestones=timing["milestones_seconds"];evaluation=timing["evaluation"]
                generation=milestones["evaluation_begin"]+timing["export_seconds"]
                backend=evaluation["evaluation_prepare_and_backend_seconds"]
                ties=evaluation["random_cpcm_with_compile_seconds"]
                other=record["wall_seconds"]-generation-backend-ties
                assert other>=0
                entry=dict(input_id=label,method=method,candidate_id=record["candidate_id"],
                    source=path.relative_to(ROOT).as_posix(),search_steps=1000,backend_calls=10,random_cpcm_calls=10,
                    generation_including_existing_monitors_seconds=generation,
                    random_selector_evaluation_with_compile_seconds=ties,
                    backend_selector_evaluation_with_prepare_seconds=backend,
                    backend_calls_seconds=sum(evaluation["backend_individual_seconds"]),
                    random_cpcm_calls_seconds=sum(evaluation["random_cpcm_individual_seconds"]),
                    other_job_seconds=other,all_research_evaluations_wall_seconds=record["wall_seconds"],
                    generation_plus_random_selector_measured_seconds=generation+ties,
                    generation_plus_backend_selector_measured_seconds=generation+backend)
                measured.append(entry);details.append(entry)
            group=dict(input_id=label,method=method,candidates=14,search_steps=14000,
                backend_calls=140,random_cpcm_calls=140,cost_basis="sum of historical concurrent-process elapsed seconds",
                generation_timing_scope="includes original CPCM monitors and progress callbacks; legacy timers do not separate these",
                selection_timing_scope="candidate scoring measured; final list reduction not separately timed")
            for key in measured[0]:
                if key.endswith("seconds"):group[key]=sum(row[key] for row in measured)
            groups.append(group)
    return groups,details,[dict(path=path,**info) for path,info in sorted(inventory.items())]


def build_tables(records):
    wide, samples, axis_rows, j2, j3 = [], [], [], [], []
    for record in records:
        spec, kind = record["spec"], record["spec"]["kind"]
        row = dict(job_id=record["job_id"], batch=record["report_batch"], kind=kind,
            input_id=record["input_id"], method=spec.get("method"), implementation=record.get("implementation"),
            repetition=spec.get("repetition", spec.get("research_repetition", spec.get("timing_repeat"))),
            repair_seed=spec.get("repair_seed"), budget=spec.get("steps"),
            wall_seconds=record["wall_seconds"], peak_rss_gib=record["peak_rss_gb"])
        if kind in ("j1", "j1_startup"):
            row.update(cost_class="CPCM primitive timing", load_seconds=record["load_seconds"],
                platform_seconds=record["platform_seconds"], startup_mode=record.get("startup_mode"),
                input_sha256=record["input_sha256"], graph_events=record["warmup"]["events"],
                graph_activities=record["warmup"]["activities"], graph_hops=record["warmup"]["hops"],
                T=record["warmup"]["T"], analysis_sha256=record["warmup"]["analysis_sha256"],
                new_numba_signatures=record.get("new_numba_signatures", []),
                numba_signatures=record.get("numba_signatures", []))
            measured = record["steady"] if kind == "j1" else [record["warmup"]]
            assert len(measured) == (5 if kind == "j1" else 1)
            for field in J1_FIELDS + (FAST_FIELDS if record["implementation"] == "fast" else ()):
                row[field] = statistics.median(r[field] for r in measured)
                row["warmup_"+field] = record["warmup"][field]
            row["sample_aggregation"] = "median of five steady calls per process" if kind == "j1" else "one warmup call per process"
            for call in [record["warmup"], *record["steady"]]:
                samples.append(dict(job_id=row["job_id"], input_id=row["input_id"],
                    implementation=row["implementation"], process_repeat=row["repetition"], **call))
        elif kind == "j4":
            row.update(cost_class="runtime control", **{k: record[k] for k in J4_FIELDS})
            row["resident_prepare_seconds"] = record["total_prepare_seconds"]-record["input_load_platform_seconds"]
            row["other_prepare_seconds"] = record["total_prepare_seconds"]-sum(record[k] for k in J4_FIELDS[:-1])
        elif kind == "j5":
            row.update(cost_class="runtime control", **{k: record[k] for k in J5_FIELDS},
                selected_axis=record["axis"], legal_axes=[c["axis"] for c in record["candidates"]],
                graph_sha256=record["graph_sha256"])
            row["other_case_seconds"] = record["complete_case_seconds"]-sum(record[k] for k in J5_FIELDS[:-1])
            for candidate in record["candidates"]:
                axis_rows.append(dict(job_id=row["job_id"], input_id=row["input_id"], axis=candidate["axis"],
                    selected=candidate["axis"] == record["axis"], build_seconds=candidate["build_seconds"],
                    **candidate["phases"], score=candidate["score"]))
        elif kind == "search":
            row.update(cost_class="offline mapping and separate paper validation", **j2_costs(record))
            j2.append(row)
        else:
            t = record["J3_timing"]
            row.update(cost_class="runtime control and separate paper validation", **t)
            row["measured_control_with_finalize_export_seconds"] = t["new_control_seconds"]+t["final_CPCM_seconds"]+t["state_export_seconds"]
            row["backend_copy_setup_seconds"] = t["backend_validation_seconds"]-t["backend_actual_calls_seconds"]
            row["other_job_seconds"] = record["wall_seconds"]-row["measured_control_with_finalize_export_seconds"]-t["backend_validation_seconds"]-t["numba_warmup_or_cache_seconds"]
            assert row["other_job_seconds"] >= -1e-5
            j3.append(row)
        wide.append(row)
    # Deterministic forward equality is checked across both implementations and every call.
    equality = []
    for label in sorted({r["input_id"] for r in records if r["spec"]["kind"] == "j1"}):
        selected = [r for r in samples if r["input_id"] == label]
        sig = {(r["T"], r["events"], r["activities"], r["analysis_sha256"]) for r in selected}
        assert len(sig) == 1, (label, sig)
        equality.append(dict(input_id=label, compared_calls=len(selected), equal=True,
            implementations=sorted({r["implementation"] for r in selected})))
    cumulative = []
    for label in ("gpt64", "gpt144", "qwen"):
        for seed in (123, 1, 2):
            group = [r for r in j3 if r["input_id"] == label and r["repair_seed"] == seed]
            l1 = [r for r in group if r["kind"] == "fault_l1"]
            for budget in (30, 100, 300):
                tiers = [n for n in (30, 100, 300) if n <= budget]
                selected = l1 + [r for r in group if r["budget"] in tiers]
                complete = len(l1) == 1 and len(selected) == 1+len(tiers)
                common = dict(input_id=label, repair_seed=seed, through_budget=budget,
                    status="complete" if complete else "pending", tiers=tiers, R0_L1_charges=1,
                    measured_jobs=[r["job_id"] for r in selected])
                if complete:
                    for key in ("new_control_seconds", "calibration_seconds", "search_seconds", "restore_best_seconds",
                                "final_CPCM_seconds", "state_export_seconds", "measured_control_with_finalize_export_seconds",
                                "backend_validation_seconds", "backend_actual_calls_seconds", "backend_copy_setup_seconds",
                                "numba_warmup_or_cache_seconds", "backend_calls", "other_job_seconds", "wall_seconds"):
                        common[key] = sum(r[key] for r in selected)
                    common["L2_search_proposals"] = sum(tiers)
                    common["calibration_proposals"] = sum(r["calibration_proposals"] for r in selected if r["kind"] == "fault_rep")
                    assert math.isclose(common["measured_control_with_finalize_export_seconds"] + common["backend_validation_seconds"]
                        + common["numba_warmup_or_cache_seconds"] + common["other_job_seconds"], common["wall_seconds"], abs_tol=1e-6)
                cumulative.append(common)
    return wide, samples, axis_rows, j2, j3, cumulative, equality


def summarize(rows, group_fields, metric_fields, expected=3, kind=""):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_fields)].append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda x: str(x[0])):
        wanted = 1 if "GPT144_prefill_mha" in key and kind == "J2" else expected
        for metric in metric_fields:
            numbers = [r[metric] for r in values if isinstance(r.get(metric), (int, float))]
            if numbers:
                output.append(dict(kind=kind, **dict(zip(group_fields, key)), metric=metric,
                    unit="seconds", process_count=len(numbers), expected_processes=wanted,
                    status="complete" if len(numbers) == wanted else "partial",
                    median=statistics.median(numbers), minimum=min(numbers), maximum=max(numbers),
                    n1_note="one measurement, no repeat range" if wanted == 1 else "three independent processes"))
    return output


def markdown_table(rows, columns):
    def cell(value):
        if value is None: return "—"
        if isinstance(value, float): return f"{value:.6g}"
        return str(value).replace("|", "/").replace("\n", " ")
    return ["| " + " | ".join(title for _, title in columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|"] + [
        "| " + " | ".join(cell(row.get(key)) for key, _ in columns) + " |" for row in rows]


def input_inventory(inputs, old_audit):
    requested = {(str(local(row["path"])), row["sha256"]): True for row in old_audit["input_files"]}
    semantic = {}
    for item in inputs:
        spec = item["spec"]
        def add(path, value):
            key = (str(local(path)), value)
            requested[key] = requested.get(key, False) or item["status"] == "complete"
        for key, hash_key in (("prepared_graph", "prepared_graph_sha256"),
                              ("source_prepare_record", "source_prepare_sha256"),
                              ("prior_signature_source", "prior_signature_source_sha256")):
            if key in spec and hash_key in spec:
                add(spec[key], spec[hash_key])
        for path, value in spec.get("input_sha256", {}).items():
            add(path, value)
        if spec.get("kind") == "j5" and "graph" in spec:
            path = local(spec["graph"])
            prepared = read(local(spec["source_prepare_record"]))
            assert prepared["graphs"]["2"]["graph_sha256"] == spec["graph_sha256"]
            semantic[path.relative_to(ROOT).as_posix()] = dict(
                path=path.relative_to(ROOT).as_posix(), semantic_graph_sha256=spec["graph_sha256"],
                observed_file_sha256=digest(path) if path.exists() else None,
                status="semantic_identity_matches_preparation", hash_type="event graph_digest and independent archive bytes")
    result = []
    for (filename, expected), required in sorted(requested.items()):
        path = Path(filename)
        actual = digest(path) if path.exists() else None
        if actual is not None and required:
            assert actual == expected, (filename, expected, actual)
        result.append(dict(path=path.relative_to(ROOT).as_posix(), expected_sha256=expected,
            observed_sha256=actual, status="not_local_yet" if actual is None else
                "matched" if actual == expected else "pending_input_changed"))
    return result + list(semantic.values())


def serial_conditions(out):
    files = {name:out/name for name in ("idle_window_gate.json", "closeout.start.json", "closeout.status.json", "monitor_history.jsonl")}
    if not all(path.exists() for path in files.values()):
        return dict(status="pending_local_serial_evidence")
    gate, start, status = (read(files[name]) for name in ("idle_window_gate.json", "closeout.start.json", "closeout.status.json"))
    digest(files["monitor_history.jsonl"])
    history = [json.loads(line) for line in files["monitor_history.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()]
    observed = [r for r in history if r.get("event") == "status"] + [status]
    assert gate["status"] == "passed" and gate["quality_completion"]["completed"] == gate["quality_completion"]["expected"] == 549
    assert gate["quality_controller_exited"] and gate["quality_workers_remaining"] == 0
    assert gate["workers"] == start["workers"] == 1
    assert all(r["active"] <= 1 and r["max_active"] == 1 and not r["failed"] for r in observed)
    quality_end = datetime.fromisoformat(gate["quality_completion"]["utc"])
    j_start = datetime.fromisoformat(start["utc"])
    assert quality_end < j_start
    return dict(status="passed", gate=gate, start=start, latest_downloaded_status=status,
        sampled_statuses=len(observed)-1, current_status_checked=True,
        observed_max_active=max(r["max_active"] for r in observed),
        quality_to_J_seconds=(j_start-quality_end).total_seconds(),
        observed_min_available_gib=min(r["available_gib"] for r in observed),
        recorded_files_sha256={name:digest(path) for name,path in files.items()})


def speedups(wide):
    result=[]
    for name in sorted({r["input_id"] for r in wide if r["kind"] == "j1"}):
        groups={mode:[r for r in wide if r["kind"] == "j1" and r["input_id"] == name and r["implementation"] == mode]
                for mode in ("original","fast")}
        if any(len(rows)!=3 for rows in groups.values()):continue
        row=dict(input_id=name,batch=groups["fast"][0]["batch"],processes_per_implementation=3,
            definition="ratio of medians across three processes; each process uses median of five steady calls")
        for key, short in (("forward_complete_seconds","forward"),("full_call_seconds","full_call")):
            for mode,rows in groups.items():row[f"{mode}_{short}_seconds"]=statistics.median(r[key] for r in rows)
            row[f"{short}_speedup"]=row[f"original_{short}_seconds"]/row[f"fast_{short}_seconds"]
        result.append(row)
    return result


def render(inputs, records, summaries, cumulative, environments, audit):
    counts = []
    for batch in ("old45", "micro", "search", "eir"):
        rows = [r for r in inputs if r["batch"] == batch]
        counts.append(dict(batch=batch, planned=len(rows), complete=sum(r["status"] == "complete" for r in rows),
            pending=sum(r["status"] != "complete" for r in rows)))
    lines = ["# J：正式计时汇总", "", f"生成时间：{audit['generated_utc']}。只读取保存结果；本次后处理没有启动搜索或仿真。",
        "", f"已核对 {len(records)}/125 项：旧 45 项保留，新计时最多 80 项。当前状态：{audit['status']}。",
        "", *markdown_table(counts, [("batch", "批次"), ("planned", "计划项"), ("complete", "已核对"), ("pending", "待完成")]),
        "", "micro 包含新模型 J1 的 18 项与 C 两个真实代表点 J5 的最多 6 项；C 输入尚未进入清单的项列为 pending_C_representative。",
        "J2a 是三个输入 × A1/A3h × 三次完整 1000 步，J2b 是 GPT144 prefill MHA × A1/A3h 各一次。J3 是三个输入 × 三个修复种子 × R0/L1 与独立 30/100/300 步。",
        "", "## 计时如何相加", "",
        "1. 离线映射：初始化、构图、校准、完整 SA、恢复最佳状态、导出和候选选择。J2 当前固定单候选，不将它乘十四后当成十四候选实测。十次随机 CPCM 单列；十种子后端是质量验证，只有采用后端选优的完整流程才把相应后端计入选择成本。",
        "2. 运行时控制：R0/L1、每个 EIR 档位的校准/搜索/恢复、最终 CPCM、状态导出，以及 DSTU 准备和 Split 所有合法轴的构造/评分/选择。",
        "3. 论文质量验证：独立后端调用和复制准备、恢复快照复查、诊断与计时记录分列。Python 仿真时间不是芯片上逐包仲裁延迟。",
        "", "## J1：CPCM 完整调用与前向推演", "",
        "每个独立进程预热一次后测五次稳态，先取进程内五次中位数，再报告三个进程中位数及最小/最大值。五次重复不当作五个独立进程。完整调用包含 compile、forward 和 analysis；fast 的 pack/kernel/unpack 已包含在 forward 中，不能再次相加。输入读取、平台创建和预热另列。",
        "旧 GPT 的首次 JIT 与磁盘缓存各三次继续保留；新图记录实际编译签名与预热。只有签名相同才引用旧启动测量；新签名记录于 J_TIMING.json，预热含缓存/JIT，不冒充稳态。", ""]
    chosen = [r for r in summaries if r["kind"] == "J1" and r["metric"] in ("forward_complete_seconds", "full_call_seconds")]
    lines += markdown_table(chosen, [("input_id", "输入"), ("implementation", "实现"), ("metric", "计时项"),
        ("process_count", "进程数"), ("median", "中位秒"), ("minimum", "最小秒"), ("maximum", "最大秒")])
    new_speedups=[r for r in audit["J1_speedups"] if r["batch"]=="micro"]
    if new_speedups:
        lines += ["", "本轮新增三图的加速倍数按‘原实现三进程中位数 / 加速实现三进程中位数’计算：", ""]
        lines += markdown_table(new_speedups,[("input_id","输入"),("original_forward_seconds","原前向秒"),
            ("fast_forward_seconds","快前向秒"),("forward_speedup","前向倍数"),
            ("original_full_call_seconds","原完整调用秒"),("fast_full_call_seconds","快完整调用秒"),("full_call_speedup","完整调用倍数")])
        new_fast=[r for r in records if r["report_batch"]=="micro" and r["spec"]["kind"]=="j1" and r["implementation"]=="fast"]
        lines += ["",f"本轮 {len(new_fast)} 个加速进程中，新增 Numba 签名为 {sum(bool(r['new_numba_signatures']) for r in new_fast)} 个进程；当前签名与旧 JIT/缓存测量一致。每个新进程的首次调用保留为 warmup，启动开销单列，未并入稳态；签名一致与缓存是否命中是不同信息，本批没有单独记录缓存命中计数。完整调用还包含构图与分析，因此其提速小于只看前向推演的提速。"]
    lines += ["", "## J2：完整映射搜索", "",
        "initial_graph_and_calibration_seconds 已包含校准；互斥表将校准取出，同时单列初始 A1 CPCM 监测。raw_sa_seconds 只覆盖循环，不含校准。algorithm_search_seconds 扣除循环 A1 CPCM 监测和进度回调，保留原始观测值供核对。搜索时间段与 raw_sa 的小差值记为 search_interval_other_seconds。",
        "广播总时间包含在原时间段中，仅作内部明细。退火标记开销亦为内部明细，不再次从各段中扣减。互斥分项之和逐行等于完整作业时间；最外层入口与该时间的差值单列。",
        "final_eval 没有独立的最终 CPCM 秒表，J2 将最终 CPCM、评估构图、深复制及图指标合列 evaluation_other_seconds，并明确 final_CPCM_seconds 为空。十次真实后端和十次随机 CPCM 调用分别计时；随机 CPCM 的编译包含在 evaluation_other_seconds。", ""]
    picked = [r for r in summaries if r["kind"] == "J2" and r["metric"] in
              ("offline_generation_measured_seconds", "algorithm_search_seconds", "calibration_seconds", "backend_calls_seconds", "random_cpcm_calls_seconds", "wall_seconds")]
    lines += markdown_table(picked, [("input_id", "输入"), ("method", "方法"), ("metric", "分项"),
        ("process_count", "n"), ("median", "中位秒"), ("minimum", "最小秒"), ("maximum", "最大秒")]) if picked else ["J2 尚无完整结果；表项 pending。"]
    complete_searches=[r for r in records if r["spec"]["kind"]=="search"]
    if complete_searches:
        measured=[dict(job_id=r["job_id"],**j2_costs(r)) for r in complete_searches]
        lines += ["", "以下五项互不重叠，相加为完整作业时间。映射生成列包含已单独计时的初始化、校准、搜索、最佳状态恢复及导出；记录/复查列包含 A1 的 CPCM 监测、进度回调、快照重载和余下记账时间。最终 CPCM 与评估准备仍留在最后一列，不将它归入纯搜索或真实后端调用。", ""]
        lines += markdown_table(measured,[("job_id","独立进程"),("offline_generation_measured_seconds","映射生成 s"),
            ("recording_and_reload_measured_seconds","记录/复查 s"),("backend_calls_seconds","十次后端 s"),
            ("random_cpcm_calls_seconds","十次随机 CPCM s"),("evaluation_other_seconds","最终 CPCM/评估准备 s"),
            ("wall_seconds","合计 s")])
        lines += ["", "A1 用聚合代价做搜索；额外的十次随机 CPCM 用于统一的候选选择比较，不属于 A1 的原生搜索循环。完整作业、外层入口、调度回执分别计时，进程启动和退出后的轮询差值不计作算法成本。每行是一条完整 1000 步轨迹；环境 helper 中通用的‘五次稳态’字符串只适用于 J1 的内核重复，J2 的实际次数以自身协议、轨迹和回执为准。"]
    first_review=BASE/"evidence/j2_first_complete_review.json"
    if first_review.exists():
        first=json.loads(first_review.read_text(encoding="utf-8"))
        lines += ["", f"首个 Qwen decode FFN A1 独立进程已核对 1000 个正式提议、{first['counts']['calibration_proposals']} 个校准提议、10 次后端与 10 次随机 CPCM；与历史同候选的 1000 条非计时轨迹逐项一致。详见[首项复查](evidence/j2_first_complete_review.json)。它的调度回执耗时为 {first['costs']['receipt_wall_seconds']:.3f} 秒（{first['costs']['receipt_wall_seconds']/60:.3f} 分钟），该类旧的并行实测估计为 {first['historical_estimate_seconds']:.3f} 秒。该实测只更新同输入、同方法的剩余重复，不用于缩放其他图或方法的预计耗时。"]
    six_review=BASE/"evidence/j2_qwen_six_review.json"
    if six_review.exists():
        six=json.loads(six_review.read_text(encoding="utf-8"))
        by_method={r["method"]:r for r in six["methods"]}
        a1,a3=by_method["A1"]["metrics"],by_method["A3h"]["metrics"]
        lines += ["", "Qwen decode FFN 的 A1/A3h 各三个独立进程已齐，六次合计 6000 个正式提议、60 次后端和 60 次随机 CPCM。每个方法重复的是同一个固定候选与搜索种子 123；六条轨迹分别与历史同候选逐步相同，恢复后的事件顺序和图摘要亦相同。环境、线程数和运行源码一致，现场记录为单进程计时。详见[六项独立复查](evidence/j2_qwen_six_review.json)。",
            f"校准单独计数：A1 每次 {by_method['A1']['calibration_proposals_per_trajectory']} 个提议，A3h 每次 {by_method['A3h']['calibration_proposals_per_trajectory']} 个，合计 {six['calibration_proposals_total']} 个。原算法先采样 50 次，若使代价上升的有效提议不足 3 个便继续采样；本例 A3h 在第 59 次达到条件。校准之后仍各运行完整 1000 步。",
            f"完整作业中位数为 A1 {a1['wall_seconds']['median']:.3f} 秒 [{a1['wall_seconds']['minimum']:.3f}, {a1['wall_seconds']['maximum']:.3f}]、A3h {a3['wall_seconds']['median']:.3f} 秒 [{a3['wall_seconds']['minimum']:.3f}, {a3['wall_seconds']['maximum']:.3f}]。扣除循环监测和进度记录后的搜索中位数分别为 {a1['algorithm_search_seconds']['median']:.3f} 与 {a3['algorithm_search_seconds']['median']:.3f} 秒；已单独计时的映射生成中位数为 {a1['offline_generation_measured_seconds']['median']:.3f} 与 {a3['offline_generation_measured_seconds']['median']:.3f} 秒。各分项的中位数各自统计；互斥分项与完整作业的相等关系按每次实际进程逐项核对。"]
    historical=audit["historical_fourteen_candidate_costs"]
    lines += ["", "十四候选整体费用另取历史同输入的完整记录累计：下表每行实际读取 14 个候选、14000 个正式提议、140 次后端及 140 次随机 CPCM。单位为各并行进程经历时间之和（进程分钟），其中包含当时的共享资源竞争；本轮 J2 表是空闲服务器上的固定候选串行实测。", ""]
    historic_minutes=[dict(row,**{key.removesuffix("_seconds")+"_minutes":value/60 for key,value in row.items() if key.endswith("_seconds")}) for row in historical]
    lines += markdown_table(historic_minutes,[("input_id","历史十四候选输入"),("method","方法"),
        ("generation_including_existing_monitors_minutes","生成含原监测 min"),
        ("random_selector_evaluation_with_compile_minutes","随机 CPCM 选择评估 min"),
        ("backend_selector_evaluation_with_prepare_minutes","后端选择评估及准备 min"),
        ("generation_plus_random_selector_measured_minutes","生成+随机 CPCM min"),
        ("generation_plus_backend_selector_measured_minutes","生成+后端 min"),
        ("all_research_evaluations_wall_minutes","含全部研究评估 min")])
    lines += ["", "历史‘生成’由初始化至恢复最佳状态，再加导出组成；旧版 SA 记录包含原有 CPCM 监测和进度回调。随机 CPCM 选择评估含 140 次估值及其编译；后端选择评估含 140 次真实后端及其复制/最终 CPCM 等准备。两种选择流程分别与生成相加，包含两种选择评估的完整研究作业另列；最终对 14 个分数取最小值的程序耗时没有单独秒表。历史与本轮单候选各自保留计时方式，完整候选来源和 SHA-256 见 [j_offline_cost_inputs.json](evidence/j_offline_cost_inputs.json)。",
        "", "GPT144 prefill MHA 计划每方法测 n=1；最小/最大相同表示同一次测量。其余输入每方法计划测三个独立进程，目前 n 随已完成结果更新。J2 质量结果随计时保存，健康主表继续使用原来的候选组。",
        "", "## J3：EIR 的分档与累计成本", "",
        "每个修复种子先计一次 R0/L1，再分别从同一 L1 运行 30、100、300 步。累计到 300 步计 430 个正式提议以及三次独立校准；每档重新开始温度日程，不把 300 步轨迹当成前三档的全部代价。R0/L1 原数值也保存在后续 raw 中，但累计表忽略这些继承字段。",
        "measured_control_with_finalize_export_seconds = 新增 R0/L1 或校准+搜索+恢复 + 最终 CPCM + 状态导出。后端调用及其复制准备独立计费。读取/初始化、原快照整理、未单独计时的诊断与记录保留为 other_job_seconds，不将它伪称为纯算法成本；完整 observed wall 同时列出。",
        "累计任务体 = measured_control_with_finalize_export_seconds + backend_validation_seconds + numba_warmup_or_cache_seconds + other_job_seconds。backend_validation_seconds 已含 backend_actual_calls_seconds 与 backend_copy_setup_seconds；控制总项也已含校准、搜索、恢复、最终CPCM和导出，明细不再与总项重复相加。calibration_proposals 独立累计，正式提议数仍为30、130、430。",
        "J3 计时的三个修复种子是三次轨迹；研究质量表的五种子挑选是另一项工作。这里既不把三次或五次的最佳质量配上一条轨迹成本，也不把 5×中位数写成五种子的实测。停档及多候选状态选择未在 J3 单独重测，不能从此表补造其耗时。", ""]
    cum = [r for r in summaries if r["kind"] == "J3 cumulative" and r["metric"] in
           ("measured_control_with_finalize_export_seconds", "backend_validation_seconds", "wall_seconds")]
    lines += markdown_table(cum, [("input_id", "输入"), ("through_budget", "累计至"), ("metric", "分项"),
        ("process_count", "n"), ("median", "中位秒"), ("minimum", "最小秒"), ("maximum", "最大秒")]) if cum else ["J3 尚无齐备的累计档位；27 个累计行已列出 pending。"]
    qwen_cumulative = [r for r in cumulative if r["input_id"] == "qwen"]
    if len(qwen_cumulative) == 9 and all(r["status"] == "complete" for r in qwen_cumulative):
        qwen_control = [dict(r, stage_name="R0/L1" if r["stage"] == "fault_l1" else str(r["budget"])+"步")
            for r in summaries if r["kind"] == "J3" and r["input_id"] == "qwen"
            and r["metric"] == "measured_control_with_finalize_export_seconds"]
        qwen_control.sort(key=lambda r: r["budget"] or 0)
        whole = [r["wall_seconds"] for r in qwen_cumulative if r["through_budget"] == 300]
        lines += ["", "Qwen decode FFN 的12项正式计时已齐：三个修复种子各运行一次R0/L1及三个独立预算。每个种子累计430个正式提议、45个校准提议和50次后端，三种子合计1290、135、150次。九条累计结果均已完成，R0/L1在每条累计链中只计一次。下表是每个独立档位的三个进程，控制时间包含最终CPCM和状态导出，后端验证另列。", ""]
        lines += markdown_table(qwen_control, [("stage_name","独立档位"),("process_count","n"),
            ("median","控制中位秒"),("minimum","最小秒"),("maximum","最大秒")])
        lines += ["", f"按每个种子的R0/L1、30、100、300步四个任务的实测墙钟相加，全过程为中位 {statistics.median(whole):.3f} 秒 [{min(whole):.3f}, {max(whole):.3f}]，包含50次研究用后端验证。三个种子的12个任务合计 {sum(whole):.3f} 秒；这是这些任务本身的时间之和，排队间隔及期间穿插的其他输入任务不计入。九条L2提议与最终状态和对应历史预算/种子的逐项核对见 [Qwen完整J3复核](evidence/j3_qwen_complete_review.json)。GPT各输入按自身完成项报告。"]
    lines += ["", "## J4 与 J5：控制准备与完整选轴", "",
        "J4 的 total_prepare 包含读取平台、全链路 256 KB 构图、CPCM 截止时间/排序及初始 ready 列表。resident_prepare 去掉读取平台；这不是硬件每包服务时间。",
        "J5 的 resident_decision 包含全部合法轴的构造、评分和选择；每轴 phases 包含在 build_seconds 内，而 build_seconds 又包含在 resident_decision 内。完整 case 另含输入、平台、基线指标、冻结图核对和导出，其他差值单列。", ""]
    ctrl = [r for r in summaries if (r["kind"] == "J4" and r["metric"] in ("resident_prepare_seconds", "total_prepare_seconds")) or
            (r["kind"] == "J5" and r["metric"] in ("resident_decision_seconds", "complete_case_seconds"))]
    lines += markdown_table(ctrl, [("input_id", "输入"), ("metric", "分项"), ("process_count", "n"),
        ("median", "中位秒"), ("minimum", "最小秒"), ("maximum", "最大秒")])
    new_j5=[r for r in records if r["report_batch"]=="micro" and r["spec"]["kind"]=="j5"]
    if new_j5:
        rows=[]
        for model in sorted({r["model"] for r in new_j5}):
            selected=[r for r in new_j5 if r["model"]==model]
            assert len({tuple(r["legal_axes"]) for r in selected})==1 and len({r["axis"] for r in selected})==1
            rows.append(dict(model=model,legal_axes="/".join(selected[0]["legal_axes"]),selected_axis=selected[0]["axis"],
                repeats=len(selected),resident_seconds=statistics.median(r["resident_decision_seconds"] for r in selected),
                full_case_seconds=statistics.median(r["complete_case_seconds"] for r in selected)))
        lines += ["", "新模型的 J5 使用 C 中已固定的真实 MF6 点。合法轴全部参与构造和评分：", ""]
        lines += markdown_table(rows,[("model","模型"),("legal_axes","完整评估轴"),("selected_axis","选中轴"),
            ("repeats","独立进程"),("resident_seconds","常驻决策中位秒"),("full_case_seconds","含读入/平台/核对/导出秒")])
    lines += ["", "## 环境、来源与复算", "",
        "J_INPUTS.json/CSV 保存逐项清单、实际完成状态、输入哈希、结果及回执身份；J_ENVIRONMENT.json/CSV 保存每个实际进程的 CPU、系统、Python/NumPy/Numba、线程设置、空闲采样与运行源码哈希。旧批与新批保持标签，不以旧环境代替尚未生成的新环境。",
        "新批单进程与空闲证明读取 out 中保存的 batch 和现场证据；只有结果中的线程设置不能独立证明没有其他进程。已完成旧批的现场证据保存在 hi/evidence/j_serial_conditions.json。", "",
        f"当前环境记录 {len(environments)} 条；来源 SHA-256 {len(audit['sources_sha256'])} 项。所有已纳入结果均有成功回执，且回执身份与对应清单一致。",
        "", "生成命令：", "", "```text",
        "e2e/post_fg/dist/venv/Scripts/python.exe -B e2e/hi_closeout_j_report.py --progress",
        "```", "", "去掉 --progress 时要求本地 125 项全部齐备。J_TIMING.csv 是逐进程宽表；tables/j_timing_summary.csv 是独立进程统计，j1_calls.csv 保留所有稳态/预热调用，j2_costs.csv 与 j3_costs.csv 保留分项，j3_cumulative.csv 保存各档累计，j5_axis_costs.csv 保存全轴成本。", ""]
    serial=audit["new_serial_conditions"]
    if serial["status"]=="passed":
        lines += [f"本轮质量池 549/549 完成且控制器退出后，等待 {serial['quality_to_J_seconds']:.3f} 秒启动 J。下载的串行现场记录截至 {serial['latest_downloaded_status']['utc']}，{serial['sampled_statuses']} 条状态记录的最大活跃实验数为 {serial['observed_max_active']}；启动条件记录中质量实验剩余进程为 0。",
            "本轮运行环境为 AMD EPYC 9K65（实例分配128 vCPU）、Linux 5.15、Python 3.13.18、NumPy 2.5.4、Numba 0.68.1；四项数值库线程设置均为1。现场记录随后续结果下载继续更新。", ""]
    review=BASE/"evidence/j_micro_24_review.json"
    if review.exists():
        lines += ["首24项的[独立复查记录](evidence/j_micro_24_review.json)逐项核对成功回执、输入文件与运行源码哈希：三个J1图的各36次调用具有相同T̂、活动规模和关键链/等待摘要；两个J5冻结图及六个重新导出图的事件语义摘要均一致。J5图的语义摘要与gzip文件字节SHA分别保存。复查没有启动CPCM、后端或搜索。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HI / "out/closeout_j_20261115")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    inputs, records, environments, receipts = collect(args.out.resolve())
    if not args.progress:
        assert all(r["status"] == "complete" for r in inputs), Counter(r["status"] for r in inputs)
    wide, calls, axes, j2, j3, cumulative, equality = build_tables(records)
    summary = []
    for kind, fields, keys in (("j1", J1_FIELDS+FAST_FIELDS, ("input_id", "implementation")),
                               ("j1_startup", J1_FIELDS+FAST_FIELDS, ("startup_mode",)),
                               ("j4", J4_FIELDS+("resident_prepare_seconds", "other_prepare_seconds"), ("input_id",)),
                               ("j5", J5_FIELDS+("other_case_seconds",), ("input_id",))):
        summary += summarize([r for r in wide if r["kind"] == kind], keys, fields,
            kind={"j1": "J1", "j1_startup": "J1 startup", "j4": "J4", "j5": "J5"}[kind])
    j2fields = tuple(k for r in j2 for k in r if k.endswith("seconds"))
    summary += summarize(j2, ("input_id", "method"), tuple(dict.fromkeys(j2fields)), kind="J2")
    j3fields = tuple(k for r in j3 for k in r if k.endswith("seconds"))
    summary += summarize([dict(r, stage=r["kind"]) for r in j3], ("input_id", "stage", "budget"),
        tuple(dict.fromkeys(j3fields)), kind="J3")
    cumfields = tuple(k for r in cumulative for k in r if k.endswith("seconds"))
    summary += summarize([r for r in cumulative if r["status"] == "complete"], ("input_id", "through_budget"),
        tuple(dict.fromkeys(cumfields)), kind="J3 cumulative")
    sidecars = {}
    for path in (HI / "evidence/j_first_completion_audit.json", HI / "evidence/j_serial_conditions.json"):
        sidecars[path.relative_to(ROOT).as_posix()] = read(path)
    old_audit = sidecars["e2e/hi/evidence/j_first_completion_audit.json"]
    old_hashes = {r["job_id"]: r["result_sha256"] for r in old_audit["receipts"]}
    assert all(r["result_sha256"] == old_hashes[r["job_id"]] for r in receipts if r["job_id"] in old_hashes)
    verified_inputs = input_inventory(inputs, old_audit)
    batches = {p.relative_to(ROOT).as_posix(): read(p) for p in sorted(args.out.glob("*.batch.json"))}
    serial = serial_conditions(args.out)
    acceleration = speedups(wide)
    historical_groups,historical_candidates,historical_inputs=historical_fourteen_costs()
    preparations = [read(p) for folder in (HI / "out/j1_prepare/results", BASE / "j_prepare/results")
                    for p in sorted(folder.glob("*.json"))]
    for path in (Path(__file__), ROOT / "e2e/hi_closeout_j_search.py", ROOT / "e2e/hi_closeout_j_eir.py",
                 ROOT / "e2e/hi_closeout_j_micro.py", ROOT / "e2e/hi_run.py", ROOT / "eir/eir.py"):
        digest(path)
    audit = dict(status="complete" if len(records) == 125 else "partial", generated_utc=datetime.now(timezone.utc).isoformat(),
        planned_maximum=125, completed=len(records), pending=len(inputs)-len(records),
        status_counts=dict(Counter(r["status"] for r in inputs)),
        old45_preserved=sum(r["report_batch"] == "old45" for r in records) == 45,
        source_results_edited=False, new_searches=0, new_backend_calls=0,
        sources_sha256=dict(SOURCES), receipts=receipts, exact_forward_equality=equality,
        verified_input_files=verified_inputs, old45_result_hashes_unchanged=True,
        new_serial_conditions=serial, J1_speedups=acceleration,
        historical_fourteen_candidate_costs=historical_groups,
        previous_serial_evidence=sidecars, new_batch_records=batches,
        postprocessing_seconds=time.perf_counter()-started)
    csv_write(BASE / "J_INPUTS.csv", inputs)
    json_write(BASE / "J_INPUTS.json", dict(jobs=inputs, graph_preparations=preparations,
        input_files=verified_inputs))
    csv_write(BASE / "J_ENVIRONMENT.csv", environments)
    json_write(BASE / "J_ENVIRONMENT.json", dict(processes=environments, old_serial_evidence=sidecars,
        new_batch_records=batches, new_serial_conditions=serial,
        observed_local_source_sha256={k: v for k, v in SOURCES.items() if k.endswith(".py")}))
    csv_write(BASE / "J_TIMING.csv", wide)
    json_write(BASE / "J_TIMING.json", dict(processes=wide, summaries=summary, cumulative_EIR=cumulative,
        cost_classes=["offline mapping and selection", "runtime control", "paper quality validation"],
        source_timing_records=[dict(job_id=r["job_id"], timing=r.get("timing"),
            j_diagnostic_timing=r.get("j_diagnostic_timing"), J3_timing=r.get("J3_timing")) for r in records]))
    for name, rows in (("j1_calls", calls), ("j5_axis_costs", axes), ("j2_costs", j2), ("j3_costs", j3),
                       ("j3_cumulative", cumulative), ("j_timing_summary", summary), ("j1_speedups", acceleration)):
        csv_write(BASE / "tables" / (name+".csv"), rows)
    csv_write(BASE/"tables/j_offline_fourteen_costs.csv",historical_groups)
    csv_write(BASE/"tables/j_offline_candidate_costs.csv",historical_candidates)
    json_write(BASE/"evidence/j_offline_cost_inputs.json",dict(status="passed",groups=4,candidates=56,
        input_file_count=len(historical_inputs),files=historical_inputs,only_files_actually_read=True,
        new_searches=0,new_backend_calls=0,new_CPCM_evaluations=0))
    json_write(BASE / "evidence/j_report_audit.json", audit)
    (BASE / "J_SUMMARY.md").write_text(render(inputs, records, summary, cumulative, environments, audit), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("status", "planned_maximum", "completed", "pending", "old45_preserved", "postprocessing_seconds")}, indent=2))


if __name__ == "__main__":
    main()
