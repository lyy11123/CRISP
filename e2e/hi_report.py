"""Read H1 pilot evidence and estimate the next approved-plan batches."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import hi_config
import hi_manifest
from hi_split_supplement import read, write

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "e2e/hi"


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=list(rows[0]))
        output.writeheader()
        output.writerows(rows)


def exclusive_timing(record):
    timing = record["timing"]
    marks = timing["milestones_seconds"]
    backend = sum(timing["evaluation"]["backend_individual_seconds"])
    ties = sum(timing["evaluation"]["random_cpcm_individual_seconds"])
    measured_eval = marks["evaluation_complete"] - marks["evaluation_begin"]
    result = dict(configure_seconds=marks["configure_complete"],
        template_seconds=timing["template_initialization_seconds"],
        initial_graph_noncalibration_seconds=timing["initial_graph_and_calibration_seconds"]-timing["calibration_seconds"],
        calibration_seconds=timing["calibration_seconds"],
        search_seconds=marks["search_complete"]-marks["search_begin"],
        best_restore_seconds=marks["evaluation_begin"]-marks["search_complete"],
        backend_calls_seconds=backend, random_cpcm_calls_seconds=ties,
        evaluation_other_seconds=measured_eval-backend-ties,
        export_seconds=marks["export_complete"]-marks["evaluation_complete"],
        snapshot_reload_seconds=marks["snapshot_reloaded"]-marks["export_complete"],
        final_accounting_seconds=record["wall_seconds"]-marks["snapshot_reloaded"])
    assert all(value >= 0 for value in result.values())
    assert math.isclose(sum(result.values()), record["wall_seconds"], rel_tol=1e-12)
    return result


def audit_pilots(out):
    import post_fg_split
    import fg_batch
    post_fg_split.setup()
    specs = {s["candidate_id"]: s for s in hi_manifest.candidates() if s["batch"] in ("H1a", "H1b")}
    batch = read(out / "h1_main.batch.json")
    assert batch["completed"] == 4 and not batch["failed"] and not batch["pending"]
    manifest_jobs = {job["id"]: job for job in fg_batch.job_items(HERE / "manifest/h1_main.json")}
    records, summaries, samples, timing_rows, audit_rows = [], [], [], [], []
    measured = {row["candidate_id"]: row for row in read(HERE / "evidence/h1_graph_measured.json")["rows"]}
    for name, spec in specs.items():
        record = read(out / "results" / (name + ".json"))
        receipt = read(out / "receipts" / (name + ".json"))
        assert receipt["rc"] == 0 and receipt["identity"] == fg_batch.identity(manifest_jobs[name])
        assert record["status"] == "ok" and all(record[key] == value for key, value in spec.items())
        raw_path = out / "raw" / (name + ".pkl.gz")
        raw = post_fg_split.static.read_pickle(raw_path)
        replay = measured[name]
        assert replay["status"] == "passed" and replay["exact_mapping_routes_and_layout"]
        assert replay["input_sha256"]["raw"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
        assert replay["input_sha256"]["results"] == hashlib.sha256((out / "results" / (name+".json")).read_bytes()).hexdigest()
        assert raw["meta"]["steps"] == len(raw["rows"]) == 1000
        assert [r["step"] for r in raw["rows"]] == list(range(1, 1001))
        assert record["exact_rollback"] and not record["debug_checks"]
        assert raw["check_revert"]["n"] == 0 and not raw["check2"] and not raw["check4"]
        rejected = [row for row in raw["rows"] if not row["accepted"]]
        assert all("t_revert" in row for row in rejected)
        assert record["instrumentation"]["original_algorithm_ast_identical_after_removing_observations"]
        for row in raw["rows"]:
            target = raw["meta"]["Tmax"] * (raw["meta"]["Tmin"]/raw["meta"]["Tmax"])**(row["step"]/1000)
            assert math.isclose(row["temp"], target, rel_tol=1e-12)
        assert [r["seed"] for r in record["backend"]] == list(range(1, 11))
        assert len(record["T_hat_rand"]) == 10
        assert record["T_hat_rand"] == raw["selected"]["T_hat_rand"]
        for sample in record["backend"]:
            assert sample["triple"] == list(raw["selected"]["backend"][sample["seed"]])
        mean = statistics.mean(sample["cycles"] for sample in record["backend"])
        ties_mean = statistics.mean(record["T_hat_rand"])
        assert ties_mean == record["T_hat_rand_mean"]
        timing = exclusive_timing(record)
        computations = sum(record["graph"]["computations"].values())
        summaries.append(dict(candidate_id=name, phase=record["phase"], op=record["op"], steps=1000,
            selected_step=record["selected_step"], backend_n=10, backend_mean_ns=mean,
            backend_min_ns=min(row["cycles"] for row in record["backend"]),
            backend_max_ns=max(row["cycles"] for row in record["backend"]),
            deterministic_cpcm_ns=record["T_hat"], random_cpcm_n=10, random_cpcm_mean_ns=ties_mean,
            deterministic_relative_error=record["T_hat"]/mean-1, random_cpcm_relative_error=ties_mean/mean-1,
            wall_seconds=record["wall_seconds"], process_receipt_seconds=receipt["wall_seconds"],
            peak_rss_gib=record["peak_rss_gb"], final_events=record["graph"]["events"],
            final_computations=computations, final_dependency_edges=record["graph"]["dependency_edges"],
            measured_schedule_activities=replay["measured_schedule_activities"],
            final_branch_hop_references=record["graph"]["communication_hops"],
            computation_plus_branch_hop_references=computations+record["graph"]["communication_hops"],
            rejected_moves=len(rejected), exact_reverts_recorded=sum("t_revert" in row for row in rejected)))
        timing_rows.append(dict(candidate_id=name, phase=record["phase"], op=record["op"], **timing,
            full_job_seconds=record["wall_seconds"], first_bc_cache_graph_seconds_included=record["timing"]["first_broadcast_seconds"],
            all_bc_cache_graph_seconds_included=record["timing"]["broadcast_total_seconds"]))
        for index, (backend, tie) in enumerate(zip(record["backend"], record["T_hat_rand"])):
            samples.append(dict(candidate_id=name, phase=record["phase"], op=record["op"], seed=backend["seed"],
                backend_ns=backend["cycles"], unhidden_compute_ns=backend["triple"][1],
                unhidden_communication_ns=backend["triple"][2], random_cpcm_ns=tie,
                backend_call_seconds=record["timing"]["evaluation"]["backend_individual_seconds"][index],
                random_cpcm_call_seconds=record["timing"]["evaluation"]["random_cpcm_individual_seconds"][index]))
        audit_rows.append(dict(candidate_id=name, raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            result_sha256=hashlib.sha256((out / "results" / (name+".json")).read_bytes()).hexdigest(),
            snapshot_sha256=hashlib.sha256((out / "snapshots" / (name+".pkl.gz")).read_bytes()).hexdigest(),
            full_steps=True, exact_rollback_kept=True, debug_checks_off=True, original_temperature_schedule=True,
            raw_backend_and_random_cpcm_match=True, receipt_identity_matches=True,
            independent_snapshot_mapping_routes_layout_and_T_match=True))
        records.append(record)
    return records, summaries, samples, timing_rows, audit_rows, batch


def simulate_queue(rows, memory_gib, workers=128):
    pending = sorted(rows, key=lambda row: -row["estimated_seconds"])
    ids = {row["candidate_id"] for row in rows}
    completed = set()
    running, t, max_running, max_memory = [], 0.0, 0, 0.0
    while pending or running:
        occupied = sum(memory for _, memory, _ in running)
        index = 0
        while len(running) < workers and index < len(pending):
            job = pending[index]
            predecessor = job["calibration_prerequisite"]
            released = not predecessor or predecessor not in ids or predecessor in completed
            if released and occupied + job["memory_hint_gib"] <= memory_gib:
                running.append((t+job["estimated_seconds"], job["memory_hint_gib"], job["candidate_id"]))
                occupied += job["memory_hint_gib"]
                pending.pop(index)
            else:
                index += 1
        assert running
        max_running, max_memory = max(max_running, len(running)), max(max_memory, occupied)
        t = min(end for end, _, _ in running)
        completed.update(name for end, _, name in running if end <= t)
        running = [(end, memory, name) for end, memory, name in running if end > t]
    return dict(queue_seconds=t, max_active=max_running, peak_reserved_gib=max_memory)


def estimate(records, checks, memory_gib):
    anchors = {(r["phase"], r["op"]): r for r in records}
    check_map = {(r["spec"]["model"],r["spec"]["hardware_profile"],r["spec"]["phase"],r["spec"]["op"],r["spec"]["method"]): r for r in checks}
    complete = {r["candidate_id"] for r in records}
    fixed_eval = min(exclusive_timing(row)["evaluation_other_seconds"] for row in records)
    rows = []
    for spec in hi_manifest.candidates():
        if spec["candidate_id"] in complete:
            continue
        proxy_op = "mha" if spec["op"] in ("ln", "mha") else "ffn"
        anchor = anchors[(spec["phase"], proxy_op)]
        target = check_map[(spec["model"], spec["hardware_profile"], spec["phase"], spec["op"],
                            "Gemini" if spec["method"] == "Gemini" else "A3h")]
        base = check_map[("llama", "busybarn", spec["phase"], proxy_op, "A3h")]
        event_ratio = target["graph"]["events"] / base["graph"]["events"]
        activity_ratio = (target["graph"]["computation_events"]+target["graph"]["hops"]) / (base["graph"]["computation_events"]+base["graph"]["hops"])
        scale = (event_ratio+activity_ratio)/2
        timing = exclusive_timing(anchor)
        fixed = timing["configure_seconds"] + fixed_eval
        template = timing["template_seconds"] * target["build_seconds"]/base["build_seconds"]
        variable = anchor["wall_seconds"]-fixed-timing["template_seconds"]
        seconds = fixed + template + variable*scale
        terminal_extra = 0.0
        if spec["method"] == "Gemini" and spec["x"] == 123:
            terminal_extra = fixed_eval + (timing["backend_calls_seconds"]+timing["evaluation_other_seconds"]-fixed_eval+timing["export_seconds"])*scale
            seconds += terminal_extra
        rss = anchor["peak_rss_gb"] * max(1, scale)
        batch = "I1" if spec["batch"] in ("I1a", "I1b", "I1W") else spec["batch"]
        rows.append(dict(candidate_id=spec["candidate_id"], batch=batch, model=spec["model"],
            hardware=spec["hardware_profile"], phase=spec["phase"], op=spec["op"], method=spec["method"],
            g=spec["g"], x=spec["x"], source_anchor=anchor["candidate_id"],
            observed_target_events=target["graph"]["events"], observed_target_branch_hops=target["graph"]["hops"],
            event_ratio=event_ratio, computation_and_path_reference_ratio=activity_ratio, size_scale=scale,
            estimated_seconds=seconds, estimated_rss_gib=rss, memory_hint_gib=1.25*rss,
            included_gemini_terminal_seconds=terminal_extra,
            included_gemini_terminal_backend_samples=10 if terminal_extra else 0,
            cost_source="full A3h pilot; g0 graph size proxy; A1/Gemini algorithm time unmeasured",
            search_steps=1000, independent_backend_samples=10, random_cpcm_evaluations=10))
    categories = {}
    for row in rows:
        key = (row["model"], row["hardware"], row["phase"], row["op"], row["method"])
        categories.setdefault(key, []).append(row)
    candidate_order = {row["candidate_id"]:row["candidate_order"] for row in hi_manifest.candidates()}
    for key, group in categories.items():
        measured = key[0] == "llama" and key[1] == "busybarn" and key[3] in ("mha", "ffn") and key[4] == "A3h"
        first = min(group, key=lambda row: candidate_order[row["candidate_id"]])
        for row in group:
            row["calibration_prerequisite"] = "" if measured or row is first else first["candidate_id"]
    summaries = []
    for label in ("H2D", "H2P", "H1c", "I1", "H2D+H2P", "next344"):
        batches = {"H2D", "H2P"} if label=="H2D+H2P" else {"H2D","H2P","H1c","I1"} if label=="next344" else {label}
        selected = [row for row in rows if row["batch"] in batches]
        parallel = min(128, len(selected))
        total = sum(row["estimated_seconds"] for row in selected)
        longest = max(row["estimated_seconds"] for row in selected)
        schedule = simulate_queue(selected, memory_gib, parallel)
        summaries.append(dict(batch=label, candidates=len(selected), estimated_process_hours=total/3600,
            ideal_work_div_workers_minutes=total/parallel/60, longest_job_minutes=longest/60,
            predicted_queue_minutes=schedule["queue_seconds"]/60,
            planning_low_minutes=1.5*schedule["queue_seconds"]/60,
            planning_high_minutes=3*schedule["queue_seconds"]/60,
            maximum_estimated_rss_gib=max(row["estimated_rss_gib"] for row in selected),
            planned_peak_memory_gib=schedule["peak_reserved_gib"], planned_max_active=schedule["max_active"],
            memory_budget_gib=memory_gib, cpu_limit=128, reserve_gib=3))
    expected = dict(H2D=166,H2P=166,H1c=4,I1=8)
    assert all(row["candidates"] == expected[row["batch"]] for row in summaries if row["batch"] in expected)
    return rows, summaries, fixed_eval


def planned_batches(destination):
    all_candidates = hi_manifest.candidates()
    plans = []
    for batch in ("H2D", "H2P", "H1c", "I1"):
        rows = [row for row in all_candidates if row["batch"] == batch or
                batch == "I1" and row["batch"] in ("I1a", "I1b", "I1W")]
        groups = {}
        for row in rows:
            groups.setdefault((row["model"],row["hardware_profile"],row["phase"],row["op"],row["method"]), []).append(row)
        initially_released, release_after = [], []
        for key, group in sorted(groups.items()):
            first = min(group, key=lambda row: row["candidate_order"])
            already_measured = key[0] == "llama" and key[1] == "busybarn" and key[3] in ("mha","ffn") and key[4] == "A3h"
            if already_measured:
                initially_released.extend(row["candidate_id"] for row in group)
            else:
                initially_released.append(first["candidate_id"])
                release_after.append(dict(category=list(key), prerequisite_candidate=first["candidate_id"],
                    release_after_success=[row["candidate_id"] for row in group if row is not first],
                    update_using=["complete job seconds including all evaluations", "peak RSS", "successful matching receipt"]))
        folder = destination / "manifest"
        write(folder / f"{batch.lower()}_planned.json", dict(status="planned_only_not_started", batch=batch,
            candidate_count=len(rows), jobs=rows,
            release_control=f"e2e/hi/manifest/{batch.lower()}_initial_release.json"))
        write(folder / f"{batch.lower()}_initial_release.json", dict(all=False, ids=initially_released))
        plan = dict(batch=batch, total=len(rows), initial_release=len(initially_released),
                    class_calibrations=release_after, already_measured_A3h_groups_released=True,
                    launch_example=f"python e2e/fg_batch.py --manifest e2e/hi/manifest/{batch.lower()}_planned.json --out <shared-output> --workers 128 --reserve-gib 3 --name {batch.lower()} --release-file e2e/hi/manifest/{batch.lower()}_initial_release.json",
                    note="Planned only. In a combined queue, merge these unchanged job specs and release IDs so total concurrency remains <=128.")
        plans.append(plan)
    write(destination / "manifest/next_batch_release_plan.json", dict(status="planned_only_not_started", batches=plans))
    return plans


def run(out, check_out, destination):
    records, summary, samples, timings, audit, batch = audit_pilots(out)
    checks = [read(filename) for filename in sorted((check_out / "results").glob("check_*.json"))]
    assert len(checks) == 40 and all(row["status"] == "ok" for row in checks)
    estimates, batch_estimates, fixed_eval = estimate(records, checks, batch["min_available_gib"]-3)
    release_plans = planned_batches(destination)
    tables = destination / "tables"
    for filename, rows in (("h1_pilots.csv", summary), ("h1_samples.csv", samples),
                           ("h1_timing_exclusive.csv", timings), ("remaining_candidate_estimates.csv", estimates),
                           ("next_batch_estimates.csv", batch_estimates)):
        table(tables / filename, rows)
    evidence = dict(status="passed", candidates=4, full_searches=4, full_search_steps=4000,
        backend_samples=40, random_cpcm_evaluations=40, audits=audit,
        algorithm_wall_process_hours=sum(r["wall_seconds"] for r in records)/3600,
        receipt_process_hours=sum(row["process_receipt_seconds"] for row in summary)/3600,
        batch={key:batch[key] for key in ("completed","failed","pending","wall_seconds","max_active","min_available_gib")},
        forecast=dict(created_date="2026-11-14", source="four complete 1000-step A3h pilots plus 40 static graph checks",
            g0_graph_proxies=True, A1_Gemini_search_times_measured=False, fixed_evaluation_seconds=fixed_eval,
            size_scale="mean(target_events/anchor_events, target_(compute+branch_hops)/anchor_(compute+branch_hops))",
            formula="configure + min_observed_evaluation_other + scaled_template_build + size_scale*(anchor_full - configure - min_observed_evaluation_other - anchor_template)",
            queue="same-class first-candidate completion gates remaining candidates; longest ready job first, 128 processes maximum, individual memory hints at 1.25*estimated RSS, 3 GiB reserve",
            planning_factor=[1.5,3.0], forecast_includes="search, calibration, state export/reload, 10 backend and 10 CPCM calls; Gemini seed123 adds 10 terminal-state backend calls; excludes later faults/joint/J and transfer/reporting"),
        planned_release_batches=release_plans)
    write(destination / "evidence/h1_pilot_audit.json", evidence)
    write(destination / "evidence/next_batch_estimates.json", dict(summary=batch_estimates, assumptions=evidence["forecast"]))
    render(destination, summary, timings, batch_estimates, evidence)


def render(destination, rows, timings, estimates, evidence):
    lines = ["# H1 主硬件四个完整试点", "", "2026-11-14，Codex。Llama 主硬件 BusyBarn 2×2，A3h g0/seed123。四个试点已计入 H 的 448 个正式候选，不重复运行。", "",
        f"四项全部成功，完整搜索合计 4,000 步，40 个独立后端样本和 40 次随机平局 CPCM。整批墙钟 {evidence['batch']['wall_seconds']:.3f} 秒，最多同时四项；完整作业累计 {evidence['algorithm_wall_process_hours']:.6f} 进程小时，含解释器启动的 receipt 累计 {evidence['receipt_process_hours']:.6f} 进程小时。最低可用内存 {evidence['batch']['min_available_gib']:.3f} GiB。", "",
        "## 单候选结果", "", "| 阶段/算子 | 后端均值 [最小,最大]，ns | 确定性 CPCM，ns | 随机 CPCM 均值，ns | 完整作业秒 | 峰值 RSS GiB |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['phase']}/{row['op']} | {row['backend_mean_ns']:,.1f} [{row['backend_min_ns']:,}, {row['backend_max_ns']:,}] | {row['deterministic_cpcm_ns']:,} | {row['random_cpcm_mean_ns']:,.1f} | {row['wall_seconds']:.3f} | {row['peak_rss_gib']:.3f} |")
    lines += ["", "每行仅一个预定 A3h 候选，不能作为十四候选选优后的方法排名，也尚未得到完整单层 TTFT/TPOT。逐种子数值与独立调用耗时在 `tables/h1_samples.csv`。", "",
        "四点随机平局 CPCM 均值相对后端均值的偏差依次为 " + "、".join(f"{r['random_cpcm_relative_error']:+.2%}" for r in rows) +
        "；确定性估值偏差依次为 " + "、".join(f"{r['deterministic_relative_error']:+.2%}" for r in rows) +
        "。这四个固定候选继续显示随机平局估值对最终后端均值的贴合，但十四候选选择效果要等下一批齐备后再比较。", "", "## 分项开销", "",
        "以下各列互不重叠，总和等于完整作业时间。BcCache.graph 的首次及累计调用是其中的子项，另存 CSV，不再加到总计。模板进入搜索之前的 initial_event_dict/add_broadcast 已计入模板初始化；first_broadcast_seconds 字段只指首次广播缓存构图，不能解释成所有首次广播开销。", "",
        "| 阶段/算子 | 配置+模板 | 初始图+校准 | 1000步SA | 最好状态恢复 | 10后端纯调用 | 10随机CPCM纯调用 | 评估其余 | 导出+重载+记账 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in timings:
        lines.append(f"| {r['phase']}/{r['op']} | {r['configure_seconds']+r['template_seconds']:.3f} | {r['initial_graph_noncalibration_seconds']+r['calibration_seconds']:.3f} | {r['search_seconds']:.3f} | {r['best_restore_seconds']:.3f} | {r['backend_calls_seconds']:.3f} | {r['random_cpcm_calls_seconds']:.3f} | {r['evaluation_other_seconds']:.3f} | {r['export_seconds']+r['snapshot_reload_seconds']+r['final_accounting_seconds']:.3f} |")
    lines += ["", "单位为秒。评估其余包括状态复制、完整广播/图编译、统计和准备；不能将它归成后端核本身。完整十二项及首次广播缓存构图耗时见 `tables/h1_timing_exclusive.csv`。", "",
        "## 最终图与原协议核对", "", "| 阶段/算子 | 事件 | 计算事件 | 依赖边 | 实际执行活动 | 分支路径跳引用 | 拒绝动作/精确回滚 | 选中步 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['phase']}/{r['op']} | {r['final_events']} | {r['final_computations']} | {r['final_dependency_edges']} | {r['measured_schedule_activities']} | {r['final_branch_hop_references']} | {r['rejected_moves']}/{r['exact_reverts_recorded']} | {r['selected_step']} |")
    lines += ["", "保存的四份 raw 均为完整 1–1000 步；逐步温度与原几何降温一致。去除新增时间戳/进度记录后，搜索函数 AST 与原实现一致。全部拒绝动作都有原精确回滚记录；逐步额外调试核对关闭。十种子后端和随机 CPCM 与 raw 逐项一致，manifest/spec/receipt 身份一致，保存图和 raw 的 SHA 已记录。", "",
        "确定性 bb 加速用于搜索和最后一次快照重载的 T_pre 自检；质量 final_eval 的确定性估值、十个独立后端及十次随机平局 CPCM 仍用原实现。", "",
        "原结果 `graph.activities` 实际保存了计算数加分支路径跳引用，多播共享前缀可能重复。本表实际执行活动来自 `evidence/h1_graph_measured.json` 的独立原 CPCM 重放，该重放同时核对四份映射、路由、布局和 T_pre/T_hat 全部精确相等。原结果保留；路径引用只作为大小特征。", "",
        "## 下一批分片与时间估计", "", "以下是 2026-11-14 基于这四个完整试点和 40 次构图核对的规划值，尚未启动后续 H/I 全量。", "",
        "| 批次 | 候选 | 预计进程小时 | 工作量/可用并行，分钟 | 最长作业，分钟 | 规划墙钟，分钟 | 排程最高并行 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in estimates:
        label = "四批合计" if r['batch'] == "next344" else r['batch']
        lines.append(f"| {label} | {r['candidates']} | {r['estimated_process_hours']:.2f} | {r['ideal_work_div_workers_minutes']:.2f} | {r['longest_job_minutes']:.2f} | {r['planning_low_minutes']:.1f}–{r['planning_high_minutes']:.1f} | {r['planned_max_active']} |")
    lines += ["", "H2D/H2P 各 166 项；H1c 为 WATOS 四项完整 A3h 试点；I1 为两硬件 P/D MHA/FFN 八项。旧 Qwen 28 项未满足历史实际集合遍历身份要求，因此不复用，I1 包含主硬件 decode FFN 新试点，其余 27 项进入 I2D。", "",
        "估计按同阶段 MHA/FFN 的完整 A3h 作业拆出固定准备，再按新图事件数与计算数+路径跳引用的平均比值缩放。LN 暂以 MHA、projection 暂以 FFN 作为成本参照；Gemini 使用其自身已生成图的大小，A1 使用同语义 A3h 初图。各 g1 及 A1/Gemini 的完整搜索耗时尚未测量，规划暂采用同大小 A3h 成本，不称作实测。没有使用旧 30 步轨迹乘以 1000/30。", "",
        "表中工作量/并行与最长作业先各自列出；简化调度同时计入同类首候选完成后才释放剩余候选的先后依赖，再按长作业优先和单作业 RSS 预算排程，最后乘 1.5–3，容纳 128 并行资源争用及尚未测量的方法差异。因此较长的未测类别需要两段连续等待，不能只用一个最长候选代替整批时间。实际调度器每两秒分批启动，未在模拟中逐次复刻，因此表格不是精确完成时刻。这是资源安排区间，不是统计置信区间。完整成本包含校准、1000 步、状态导出、十次后端及十次随机 CPCM；Gemini seed123 另含十个结束状态后端样本（H2D/H2P 各四例）。故障、联合、正式 J 和下载汇总另计。", "",
        "建议按以下顺序：", "",
        "1. H2D/H2P 分别保存清单，共用不超过 128 进程的队列。prefill 较大图优先，decode 和较小图补空位。各算子 A1/Gemini、LN/projection 先启动第一个正式候选，完成后用其完整成本和 RSS 更新队列；这些均属于既定 166 项。",
        "2. H1c 四项作为第二硬件成本校准，可与上述队列穿插。模型与路径已构图核对，完整候选仍待实测。",
        "3. I1 的八项优先安排较大的 MoE prefill FFN，其余七项并行。两个 prefill FFN 的 RSS 和后端开销将决定后续 I2P 资源预算。",
        "4. 并行上限 128，系统保留 3 GiB；本表使用本批最低可用内存减 3 GiB 排程，并按预测 RSS 的 1.25 倍占位，正式执行按实时可用内存及实测峰值更新。",
        "", "准确清单已保存 `manifest/h2d_planned.json`、`h2p_planned.json`、`h1c_planned.json`、`i1_planned.json`，均标记 planned_only_not_started；四个 initial_release 文件和 `manifest/next_batch_release_plan.json` 给出先发候选与成功后释放的同类候选。H2D/H2P 已测过的 A3h MHA/FFN 可直接排队，其余每类先一个正式候选；更新该类完整耗时、峰值 RSS 并核对成功 receipt 后，再加入对应 release IDs。合并清单共用一个 128 进程上限，不能为各批各开 128。", "",
        "逐候选规划见 `tables/remaining_candidate_estimates.csv`，批次摘要及假设见 `evidence/next_batch_estimates.json`；本报告没有启动 H2D/H2P/H1c/I1。", ""]
    (destination / "H1_PILOT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "out/h1_main")
    parser.add_argument("--checks", type=Path, default=HERE / "out/model_checks")
    parser.add_argument("--destination", type=Path, default=HERE)
    args = parser.parse_args()
    run(args.out, args.checks, args.destination)
