"""H3 + I2D + I2P only: 440 jobs in the existing memory-aware pool."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import types

import hi_next as previous
from e_run import ROOT, write_json

HERE = ROOT / "e2e/hi"
NAME = "h3_i2"
COUNTS = {"H3": 108, "I2D": 166, "I2P": 166}
KEYS = ("model", "hardware_profile", "phase", "op", "method")
MANIFEST = HERE / "manifest/h3_i2_440.json"
RELEASE_PLAN = HERE / "manifest/h3_i2_release_plan.json"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return tuple(row[k] for k in KEYS)


def measurements():
    candidates = read_csv(HERE / "tables/next_candidates.csv")
    timing = {r["candidate_id"]: r for r in read_csv(HERE / "tables/next_timing_exclusive.csv")}
    groups = defaultdict(list)
    for row in candidates:
        groups[key(row)].append(row)
    assert len(candidates) == 348 and len(groups) == 36
    return candidates, timing, groups


def estimates(specs):
    candidates, timing, groups = measurements()
    graphs = {key(r["spec"]): r["graph"] for r in
              (previous.read(path) for path in (HERE / "out/model_checks/results").glob("*.json"))}

    def median(rows, field):
        return statistics.median(float(r[field]) for r in rows)

    def search(rows):
        return statistics.median(float(timing[r["candidate_id"]]["search_seconds"]) for r in rows)

    def terminal_cost(rows):
        return statistics.median(sum(float(timing[r["candidate_id"]].get(k) or 0) for k in
            ("backend_calls_seconds", "evaluation_other_seconds", "export_seconds")) for r in rows)

    output = []
    for spec in specs:
        target = key(spec)
        model, hardware, phase, op, method = target
        terminal = 0.0
        if target in groups:
            anchors = groups[target]
            seconds = median(anchors, "process_receipt_seconds")
            rss = max(float(r["peak_rss_gib"]) for r in anchors)
            source, ratio = "same_model_hardware_phase_operator_method_measured", 1.0
        elif (model, hardware, phase, op, "A3h") in groups:
            anchors = groups[model, hardware, phase, op, "A3h"]
            counterpart = groups["llama", "busybarn", phase, op, method]
            baseline = groups["llama", "busybarn", phase, op, "A3h"]
            ratio = search(counterpart) / search(baseline)
            seconds = median(anchors, "process_receipt_seconds") + search(anchors)*(ratio-1)
            rss = max(float(r["peak_rss_gib"]) for r in anchors)
            source = "same_model_A3h_measured_nonsearch_plus_Llama_method_search_ratio"
            if method == "Gemini" and spec["x"] == 123:
                terminal = terminal_cost(anchors)
        else:
            anchors = groups["llama", "busybarn", phase, op, method]
            graph_method = "Gemini" if method == "Gemini" else "A3h"
            target_graph = graphs[model, hardware, phase, op, graph_method]
            base_graph = graphs["llama", "busybarn", phase, op, graph_method]
            ratio = (target_graph["events"]/base_graph["events"] +
                     (target_graph["computation_events"]+target_graph["hops"])/
                     (base_graph["computation_events"]+base_graph["hops"]))/2
            seconds = median(anchors, "process_receipt_seconds") * ratio
            rss = max(float(r["peak_rss_gib"]) for r in anchors) * max(1, ratio)
            source = "Llama_same_operator_method_measured_graph_size_proxy_first_candidate_required"
            if method == "Gemini" and spec["x"] == 123:
                terminal = terminal_cost(anchors) * ratio
        output.append(dict(candidate_id=spec["candidate_id"], batch=spec["batch"], **dict(zip(KEYS, target)),
            candidate_order=spec["candidate_order"], estimated_seconds=seconds+terminal,
            estimated_without_terminal_seconds=seconds, terminal_extra_seconds=terminal,
            terminal_backend_samples=10 if terminal else 0, measured_same_class=target in groups,
            estimated_rss_gib=rss, memory_hint_gib=1.25*rss, ratio=ratio, cost_source=source,
            anchors=";".join(r["candidate_id"] for r in anchors), prerequisite=""))
    return output, groups


def data():
    all_specs = previous.read(HERE / "manifest/all_candidates_planned.json")["jobs"]
    specs = [spec for spec in all_specs if spec["batch"] in COUNTS]
    assert Counter(s["batch"] for s in specs) == COUNTS
    completed = {r["candidate_id"] for r in read_csv(HERE / "tables/next_candidates.csv")}
    assert len(specs) == len({s["candidate_id"] for s in specs}) == 440
    assert not completed.intersection(s["candidate_id"] for s in specs)
    forecast, measured = estimates(specs)
    rows = {r["candidate_id"]: r for r in forecast}
    groups = defaultdict(list)
    for spec in specs:
        assert spec["search_steps"] == 1000 and spec["backend_seeds"] == spec["random_cpcm_seeds"] == list(range(1, 11))
        assert spec["module"] == "hi_run"
        groups[key(spec)].append(spec)
    initial, calibrations = {s["candidate_id"] for s in specs}, []
    for category, group in sorted(groups.items()):
        if category not in measured:
            first = min(group, key=lambda s: s["candidate_order"])["candidate_id"]
            dependents = [s["candidate_id"] for s in group if s["candidate_id"] != first]
            calibrations.append(dict(category=list(category), prerequisite_candidate=first,
                                     release_after_success=[], observed_class_candidates=dependents,
                                     role="priority measurement; other candidates do not wait"))
    assert len(initial) == 440 and len(calibrations) == 24
    first_ids = {r["prerequisite_candidate"] for r in calibrations}
    assert not any(row["prerequisite"] for row in forecast)
    jobs = []
    for spec in specs:
        row, name = rows[spec["candidate_id"]], spec["candidate_id"]
        jobs.append(dict(id=name, kind=spec["kind"], spec=spec, result=f"results/{name}.json",
            argv=["{python}", "-u", "-B", "e2e/hi_run.py", "child", "--spec", "{spec}", "--out", "{out}"],
            deps=[row["prerequisite"]] if row["prerequisite"] else [],
            estimated_seconds=row["estimated_seconds"], peak_rss_gb=row["memory_hint_gib"],
            priority=2 if name in first_ids else 1 if spec["model"] == "qwen3moe" and spec["phase"] == "prefill" and spec["op"] == "ffn" else 0))
    return jobs, initial, calibrations, forecast, measured


def assemble():
    jobs, initial, calibrations, _, _ = data()
    return jobs, initial, calibrations


def queue_estimate(jobs, memory_gib, workers=128):
    remaining, running, done = list(jobs), [], set()
    now, maximum = 0.0, 0
    while remaining or running:
        occupied = sum(row[1] for row in running)
        for job in sorted(remaining[:], key=lambda j: (j["priority"], j["estimated_seconds"]), reverse=True):
            if len(running) >= workers:
                break
            if set(job["deps"]) <= done and occupied+job["peak_rss_gb"] <= memory_gib:
                running.append((now+job["estimated_seconds"], job["peak_rss_gb"], job["id"]))
                occupied += job["peak_rss_gb"]
                remaining.remove(job)
        assert running
        maximum = max(maximum, len(running))
        now = min(row[0] for row in running)
        done.update(row[2] for row in running if row[0] <= now)
        running = [row for row in running if row[0] > now]
    return dict(queue_seconds=now, peak_parallel=maximum)


def plan(memory_gib=240):
    jobs, initial, calibrations, forecast, _ = data()
    write_json(MANIFEST, dict(status="authorized", counts=COUNTS, jobs=jobs,
                              excluded="I3 and the 348 completed candidates"))
    write_json(RELEASE_PLAN, dict(all=True, initial_release_ids=sorted(initial), calibrations=calibrations,
                                  measured_classes=8, new_classes=24, dependent_jobs=0,
                                  priority_rules="24 unmeasured-class representatives first; remaining Qwen prefill FFN next; then longest jobs"))
    table(HERE / "tables/h3_i2_estimates.csv", forecast)
    summary = []
    for batch in (*COUNTS, "all440"):
        selected = [j for j in jobs if j["spec"]["batch"] == batch or batch == "all440"]
        queue = queue_estimate(selected, memory_gib)
        summary.append(dict(batch=batch, jobs=len(selected), process_hours=sum(j["estimated_seconds"] for j in selected)/3600,
            longest_job_minutes=max(j["estimated_seconds"] for j in selected)/60,
            queue_minutes=queue["queue_seconds"]/60, planning_minutes=[queue["queue_seconds"]/60, 2*queue["queue_seconds"]/60],
            maximum_parallel=queue["peak_parallel"], memory_budget_gib=memory_gib))
    evidence = dict(status="passed", counts=COUNTS, summary=summary, initial=440, measured_classes=8, calibrations=24,
        delayed_jobs=0, terminal_backend_samples=sum(r["terminal_backend_samples"] for r in forecast),
        new_backend_samples=4400, random_cpcm_samples=4400,
        note="full jobs include all search/evaluation/export; initial range is queue estimate to twice estimate, not a confidence interval; measured first candidates update it",
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (HERE/"tables/next_candidates.csv", HERE/"tables/next_timing_exclusive.csv", HERE/"manifest/all_candidates_planned.json")})
    assert evidence["terminal_backend_samples"] == 80
    write_json(HERE / "evidence/h3_i2_estimates.json", evidence)
    lines = ["# H3 与 I2 运行计划", "", "2026-11-14。只启动 H3 108、I2D 166、I2P 166，共 440 个候选。I3 未在本轮授权内，之前完成的 348 个候选不重跑。", "",
        "## 任务与启动顺序", "", "H3 补齐 Llama WATOS 的 P/D MHA、FFN，方法 A1/A3h；I2D/I2P 补齐 Qwen 主硬件四算子的 A1/A3h/Gemini。", "",
        "440 项全部进入可运行队列，没有首候选完成后才释放同类任务的等待条件。24 个未测类的代表候选优先级最高，先获得实测进度；"
        "同类其他候选可随空闲槽位直接跟进。任一候选结果与 receipt 身份核对成功后，立即更新该类完整耗时和 RSS。"
        "Qwen LN/projection 的 A3h 也在这 24 个优先观测类别内；原八个已测 A3h 类的 104 项不加新限制。", "",
        "24 个代表项之后，Qwen prefill FFN 剩余项优先，其余按预计完整耗时从长到短填充。"
        "所有任务共用最多 128 个进程，预留 3 GiB，继续按实时可用内存与实测峰值安排。内存终止保留失败记录，预算恢复后自动续跑。", "",
        "## 根据完整任务估时", "", "Qwen 主硬件 prefill FFN A3h 已测完整作业为 4,480.50 秒（74.68 分钟），包含约 30.49 分钟搜索与 40.86 分钟十种子后端。"
        "这次直接采用同模型、硬件、阶段、算子和方法的完整实测值；已有类别使用 receipt 耗时中位数、RSS 最大值。", "",
        "未测 A1/Gemini 优先保留该模型 A3h 实测的评估及其他成本，只按 Llama 同算子的实测方法比例暂估搜索部分。"
        "无同模型完整输入的 Qwen LN/projection 使用 Llama 同算子成本和已构图规模暂估。首候选完成后即改用真实完整耗时。"
        "Gemini 八个 seed123 还各含十次结束状态后端，已额外计入预计时间及 80 个样本。", "",
        "| 批次 | 作业 | 预计进程小时 | 最长作业分钟 | 队列/内存排程分钟 | 初始安排范围分钟 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['batch']} | {r['jobs']} | {r['process_hours']:.2f} | {r['longest_job_minutes']:.1f} | {r['queue_minutes']:.1f} | {r['planning_minutes'][0]:.1f}–{r['planning_minutes'][1]:.1f} |")
    lines += ["", "上表按全体任务可运行、24 个代表项优先排程，计入完整 1000 步、温度校准、十次后端、十次随机 CPCM、导出和重载。"
        "内存排程暂用 240 GiB 作业预算，实际运行按实时内存。范围上限为排程值的两倍，用来安排未测方法与并行争用；不是统计置信区间。"
        "下载、汇总、推送以及后续故障/联合/J 另计；各批共享队列，分项墙钟不相加。", "",
        "## 固定的方法和保存内容", "", "全部 spec 从 all_candidates_planned.json 原样取出，仍为 1000 步、原温度/精确回滚、确定性 bb 加速搜索、关闭逐步额外 checks；"
        "每候选十种子独立后端和十次随机平局 CPCM。Gemini seed123 的最好/结束状态对照沿用 hi_run。健康故障状态不变，8Q+1KV 及模型合同不变。", "",
        "复用 hi_next.Controller 和 fg_batch 的进程池，新目录为 `out/h3_i2_440`。保存 `h3_i2.status.json`、`h3_i2.batch.json`、`h3_i2.start.json`、`h3_i2.log`、"
        "每项 spec/log/raw/snapshot/result/receipt、release.json、class_estimates.json 和 estimated_finish.json。三分钟 COS 备份由主控沿用现有设置。", "",
        "本机检查与云端启动（从仓库根目录）：", "", "```text", "python e2e/hi_h3_i2.py plan", "python e2e/hi_h3_i2.py launch", "```", "",
        "本机 plan 只读取已完成结果并生成清单、估时和本文件，不启动实验。", ""]
    (HERE / "H3_I2_RUN_PLAN.md").write_text("\n".join(lines), encoding="utf-8")
    compile(previous.wrapped_pool_source(), "<H3 I2 existing pool>", "exec")
    return evidence


class Controller(previous.Controller):
    def __init__(self, out, workers):
        previous.assemble = assemble
        super().__init__(out, workers)
        self.manifest = MANIFEST
        _, _, groups = measurements()
        for job in self.jobs:
            k = key(job["spec"])
            if k in groups and previous.category(job) not in self.observed:
                self.observed[previous.category(job)] = [dict(wall_seconds=float(r["process_receipt_seconds"]),
                    peak_rss_gb=float(r["peak_rss_gib"])) for r in groups[k]]

    def eta(self, status):
        result = super().eta(status)
        active = {row["id"]: row for row in status["running"]}
        overdue = [name for name, row in active.items() if row["seconds"] >= self.by_id[name]["estimated_seconds"]]
        if overdue:
            result.update(queue_seconds=None, planning_seconds=None, estimated_finish_utc=None, overdue_ids=overdue,
                          reason="Active jobs exceeded their class estimate; use recorded phase/step progress until updated complete samples arrive.")
        return result

    def write(self, path, value):
        path = Path(path)
        if path.name == NAME + ".status.json":
            value["estimate"] = self.eta(value)
            value["failed_attempt_count"] = len(list((self.out/"failed_attempts").glob("*/*/receipt.json")))
            value["class_calibrations_complete"] = sum(r["prerequisite_candidate"] in self.success for r in self.calibrations)
            write_json(self.out/"estimated_finish.json", value["estimate"])
            write_json(path, value)
        else:
            super().write(path, value)

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out/"receipts").mkdir(exist_ok=True)
        self.recover()
        for path in self.receipt_paths():
            receipt = previous.read(path)
            if receipt["rc"] == 0:
                self.success_receipt(receipt)
        self.save_release()
        pool = previous.pool
        pool.category, pool.memory_key = previous.category, previous.memory_key
        pool.memory_requirement = lambda job, *args: max(previous.RAW_MEMORY_REQUIREMENT(job, *args), job["peak_rss_gb"])
        pool.job_items = lambda path: self.jobs
        pool.write_json = self.write
        pool.subprocess = types.SimpleNamespace(Popen=self.spawn, STDOUT=subprocess.STDOUT)
        namespace = dict(vars(pool), _hi=self)
        exec(compile(previous.wrapped_pool_source(), "<H3 I2 existing pool>", "exec"), namespace)
        rc = namespace["execute"](self.manifest, self.out, self.workers, 3, NAME, 4, self.release)
        write_json(self.out/"controller_finish.json", dict(rc=rc, completed=len(self.success), expected=440,
                   utc=datetime.now(timezone.utc).isoformat()))
        return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "launch", "run"))
    parser.add_argument("--out", type=Path, default=HERE/"out/h3_i2_440")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--memory-budget-gib", type=float, default=240)
    args = parser.parse_args()
    assert 1 <= args.workers <= 128
    if args.command in ("plan", "check"):
        result = plan(args.memory_budget_gib)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "launch":
        jobs, _, _ = assemble()
        declared = previous.RAW_JOB_ITEMS(MANIFEST)
        assert [previous.pool.identity(j) for j in jobs] == [previous.pool.identity(j) for j in declared]
        start = args.out/(NAME+".start.json")
        if start.exists() and previous.pool.process_alive(previous.read(start)["pid"]):
            raise RuntimeError("H3/I2 controller already running")
        write_json(args.out/"scheduler_policy.json", dict(running_growth_fraction=.25))
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                   CRISP_E_ARTIFACT=str(ROOT/"e2e/work/artifact"), PYTHONUTF8="1")
        command = [sys.executable,"-u","-B","e2e/hi_h3_i2.py","run","--out",str(args.out),"--workers",str(args.workers)]
        options = dict(start_new_session=True) if sys.platform != "win32" else dict(
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        with (args.out/(NAME+".log")).open("ab") as stream:
            process = previous.RAW_POPEN(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,
                                         stderr=subprocess.STDOUT,**options)
        write_json(start,dict(pid=process.pid,command=command,created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256={name:hashlib.sha256((ROOT/"e2e"/name).read_bytes()).hexdigest()
                           for name in ("hi_h3_i2.py","hi_next.py","fg_batch.py","hi_run.py")}))
        print(json.dumps(dict(pid=process.pid,jobs=440,out=str(args.out))),flush=True)
        return 0
    return Controller(args.out,args.workers).run()


if __name__ == "__main__":
    raise SystemExit(main())
