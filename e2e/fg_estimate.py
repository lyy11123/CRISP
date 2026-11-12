"""Estimate F/G remaining work from completed jobs, without running simulations.

Example (write a new scheduling copy; the input manifest stays unchanged):
  python e2e/fg_estimate.py --manifest e2e/fg/manifest/all_jobs.json \
    --out e2e/fg/out/main --status e2e/fg/out/main/main.status.json \
    --report e2e/fg/tables/workflow_estimate.json \
    --updated-jobs e2e/fg/tables/jobs_with_measured_times.json
"""

import argparse
from collections import Counter, defaultdict, deque
import copy
from datetime import datetime, timezone
import hashlib
import heapq
import gzip
import json
from pathlib import Path
import pickle
import statistics


HERE = Path(__file__).resolve().parent
SUCCESS = {None, "ok", "complete", "completed", "reused"}
PLATFORM = ("hardware_profile", "shape", "split_mode", "d2d_scale")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def identity(job):
    payload = {key: job[key] for key in ("id", "kind", "spec", "result")}
    payload["argv"] = job.get("argv")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def category(job, healthy=False):
    """Pool restart seeds, retaining hardware, actual workload and search budget."""
    spec = job["spec"]
    platform = tuple(spec.get(k) for k in PLATFORM)
    if healthy:
        platform = platform[:3] + (1.0,)
    common = (job["kind"],) + platform
    if job["kind"] == "search":
        return common + (("healthy" if healthy else spec["family"]),
                         ("healthy" if healthy else spec["fault_id"]),
                         spec["phase"], spec["op"], spec["method"],
                         spec["search_steps"], tuple(spec["backend_seeds"]))
    return common + tuple(spec.get(k) for k in
                          ("phase", "op", "phase_op", "start_arm", "fault_id", "steps",
                           "condition", "backend", "stream", "arm", "policy", "p_op")) + (
                               tuple(spec.get("backend_seeds", [])), tuple(spec.get("backends", [])))


def priority(job):
    spec = job["spec"]
    if spec["hardware_profile"] == "watos":
        base = 500
    elif spec["shape"] == "3x3" and spec["split_mode"] == "growth":
        base = 400
    elif spec["stage"] == "F" or spec["split_mode"] == "growth":
        base = 300
    else:
        base = 200
    return base + (1000 if job["kind"] != "search" else 0)


def has_terminal(job):
    return job["kind"] == "search" and job["spec"]["method"] == "Gemini" and job["spec"]["x"] == 123


def observations(jobs, out):
    rows, rejected = {}, []
    for job in jobs:
        name = job["id"]
        result_path = out / job["result"]
        receipt_path = out / "receipts" / (name + ".json")
        if not result_path.exists():
            continue
        result = read(result_path)
        if result.get("status") not in SUCCESS:
            continue
        source_spec = result if job["kind"] == "search" else result.get("spec", {})
        receipt = read(receipt_path) if receipt_path.exists() else None
        if receipt and (receipt["rc"] != 0 or receipt["identity"] != identity(job)):
            rejected.append({"id": name, "reason": "receipt identity or return code differs"})
            continue
        if not receipt and not all(source_spec.get(k) == v for k, v in job["spec"].items()):
            rejected.append({"id": name, "reason": "result without receipt does not identify this spec"})
            continue
        terminal_seconds = 0.0
        if has_terminal(job):
            terminal = out / "results" / (name + "_terminal.json")
            if terminal.exists():
                terminal_seconds = float(read(terminal)["wall_seconds"])
            else:
                rejected.append({"id": name, "reason": "Gemini seed 123 terminal result absent"})
                continue
        seconds = float(receipt["wall_seconds"] if receipt else result["wall_seconds"] + terminal_seconds)
        rows[name] = dict(id=name, seconds=seconds,
                          primary_seconds=max(0.0, seconds - terminal_seconds),
                          terminal_seconds=terminal_seconds,
                          peak_rss_gb=(receipt or result).get("peak_rss_gb"),
                          timing_source="receipt" if receipt else "result (child launch overhead excluded)",
                          reused=result.get("status") == "reused", result=result)
    return rows, rejected


def sample_summary(samples, terminal=False):
    values = [r["primary_seconds"] + (r["terminal_seconds"] if terminal else 0) for r in samples]
    return dict(samples=len(values), low=min(values), mean=statistics.mean(values), high=max(values),
                source_jobs=[r["id"] for r in samples])


def d_references(root):
    """Real D cloud backend measurements; keep each solo/overlap timing separate."""
    cloud_names, descriptions = set(), {}
    for path in root.glob("remote/pkg*/top/runs_*.json"):
        cloud_names.update(row["name"] for row in read(path) if row.get("rc") == 0 and "wall" in row)
    for path in root.glob("remote/pkg*/top/d_jobs.json"):
        descriptions.update({row["name"]: row for row in read(path)})
    pools = defaultdict(list)
    for name in sorted(cloud_names):
        path = root / "prod/sims" / (name + ".pkl.gz")
        description = descriptions.get(name)
        if not path.exists() or not description or description["kind"] != "mha":
            continue
        with gzip.open(path, "rb") as handle:
            result = pickle.load(handle)
        for run in result["runs"]:
            if run.get("stop") or run["kind"] not in ("solo", "ov"):
                continue
            stream = next(iter(run["streams"])) if run["kind"] == "solo" else None
            key = ("joint_solo" if run["kind"] == "solo" else "joint", run["backend"], run["policy"], stream)
            pools[key].append(dict(seconds=float(run["wall"]), name=name, label=run["label"], path=str(path)))
    return pools


def joint_proxy(job, observed, references):
    spec = job["spec"]
    if job["kind"] not in ("joint", "joint_solo") or spec.get("p_op", "mha") != "mha":
        return None
    key = (job["kind"], spec["backend"], spec.get("policy", "P1"), spec.get("stream"))
    samples = references.get(key, [])
    if not samples:
        return None
    # Scale only by graph sizes observed in this F/G pilot. Hardware timing and
    # chunked-hop growth remain assumptions of this historical approximation.
    sizes = defaultdict(list)
    for item in observed.values():
        r = item["result"]
        if r.get("kind") == "search" and r.get("family") == "healthy" and r.get("method") in ("A1", "A3h"):
            sizes[(r["config_id"], r["phase"], r["op"])].append(r["graph_events"])
    base = "F_watos_2x2_fixed_d2d1"
    terms = [("prefill", "mha", 1)]
    if spec.get("stream") == 1:
        terms = []
    if job["kind"] == "joint" or spec.get("stream") == 1:
        terms += [("decode", "ln", 1), ("decode", "mha", 8), ("decode", "proj", 1), ("decode", "ffn", 1)]
    config = spec.get("source_config_id", spec["config_id"])
    if any(not sizes[(c, phase, op)] for c in (base, config) for phase, op, _ in terms):
        return None
    def events(c):
        return sum(statistics.mean(sizes[(c, phase, op)]) * multiplier for phase, op, multiplier in terms)
    ratio = events(config) / events(base)
    values = [row["seconds"] * ratio for row in samples]
    return dict(classification="proxy", complete=False, samples=len(samples),
                basis="D cloud per-evaluation time, scaled by observed F/G unchunked event-count ratio; launch overhead and hardware/chunk growth not calibrated",
                low=min(values), mean=statistics.mean(values), high=max(values),
                source_jobs=sorted({row["name"] for row in samples})[:5],
                historical_event_ratio=ratio, historical_key=list(key))


def estimates(jobs, observed, references=None):
    pools = defaultdict(list)
    for job in jobs:
        row = observed.get(job["id"])
        if row and not row["reused"]:
            pools[category(job)].append(row)
    rows = {}
    for job in jobs:
        name, spec = job["id"], job["spec"]
        actual = observed.get(name)
        if actual:
            rows[name] = dict(id=name, classification="measured", complete=True,
                              basis=actual["timing_source"], samples=1,
                              low=actual["seconds"], mean=actual["seconds"], high=actual["seconds"],
                              source_jobs=[name], reused=actual["reused"])
            continue
        samples = pools.get(category(job), [])
        label, basis = "measured", "same hardware/workload/method/budget, other restart"
        if not samples and job["kind"] == "search" and spec["family"] in ("full", "bandwidth"):
            samples = pools.get(category(job, healthy=True), [])
            label = "proxy"
            basis = ("matching healthy search; fault can change duration" if spec["family"] == "full"
                     else "matching WATOS 1x healthy search; bandwidth can change duration")
        if samples:
            rows[name] = dict(id=name, classification=label, complete=False, basis=basis,
                              **sample_summary(samples, terminal=has_terminal(job)))
        elif references and (proxy := joint_proxy(job, observed, references)):
            rows[name] = dict(id=name, **proxy)
        else:
            rows[name] = dict(id=name, classification="unmeasured", complete=False,
                              basis="no completed matching formal job", samples=0,
                              low=None, mean=None, high=None, source_jobs=[])
    return rows


def topology(jobs):
    by_id = {job["id"]: job for job in jobs}
    children = defaultdict(list)
    indegree = {name: len(job["deps"]) for name, job in by_id.items()}
    for job in jobs:
        for dependency in job["deps"]:
            if dependency not in by_id:
                raise ValueError(f"Dependency outside manifest: {job['id']} -> {dependency}")
            children[dependency].append(job["id"])
    ready = deque(name for name, n in indegree.items() if n == 0)
    order = []
    while ready:
        name = ready.popleft()
        order.append(name)
        for child in children[name]:
            indegree[child] -= 1
            if not indegree[child]:
                ready.append(child)
    if len(order) != len(jobs):
        raise ValueError("Manifest contains a dependency cycle")
    return by_id, children, order


def schedule(jobs, durations, workers, running=None):
    """Priority then longest job first; dependencies respected, one process per job."""
    by_id, children, order = topology(jobs)
    tail = {}
    for name in reversed(order):
        tail[name] = durations[name] + max((tail[c] for c in children[name]), default=0.0)
    need = {name: len(job["deps"]) for name, job in by_id.items()}
    def rank(name):
        return (-by_id[name].get("priority", priority(by_id[name])), -durations[name], name)
    ready = [rank(name) for name, n in need.items() if not n]
    heapq.heapify(ready)
    active, done = [], set()
    clock, max_active = 0.0, 0
    # Reserve slots for jobs actually running in the supplied status snapshot.
    for name in (running or {}):
        if name in by_id and durations[name] > 0:
            heapq.heappush(active, (durations[name], name))
    running_ids = {name for _, name in active}
    while ready or active:
        while ready and len(active) < workers:
            _, _, name = heapq.heappop(ready)
            if name in done or name in running_ids:
                continue
            if durations[name] == 0:
                done.add(name)
                for child in children[name]:
                    need[child] -= 1
                    if need[child] == 0:
                        heapq.heappush(ready, rank(child))
            else:
                heapq.heappush(active, (clock + durations[name], name))
        max_active = max(max_active, len(active))
        if active:
            clock = active[0][0]
            while active and active[0][0] <= clock:
                _, name = heapq.heappop(active)
                done.add(name)
                for child in children[name]:
                    need[child] -= 1
                    if need[child] == 0:
                        heapq.heappush(ready, rank(child))
    return dict(seconds=clock, hours=clock / 3600, max_active=max_active,
                critical_path_seconds=max(tail.values(), default=0.0))


def analyse(jobs, out, workers=128, status=None, smoke_path=None, d_root=None):
    observed, rejected = observations(jobs, out)
    references = d_references(d_root) if d_root else {}
    rows = estimates(jobs, observed, references)
    status = status or {}
    running = {r["id"]: r for r in status.get("running", []) if r["id"] not in observed}
    by_id, _, order = topology(jobs)
    unknown = {name for name, row in rows.items() if row["classification"] == "unmeasured"}
    blocked = set(unknown)
    for name in order:
        if name not in observed and any(dep in blocked for dep in by_id[name]["deps"]):
            blocked.add(name)
    process_hours, schedules = {}, {}
    overruns = []
    for name, current in running.items():
        row = rows.get(name)
        if row and row["high"] is not None and current["seconds"] > row["high"]:
            overruns.append(dict(id=name, elapsed_seconds=current["seconds"], measured_max=row["high"]))
    for column in ("low", "mean", "high"):
        durations = {}
        for name, row in rows.items():
            duration = 0.0 if row["complete"] or row[column] is None else row[column]
            if name in running:
                duration = max(0.0, duration - running[name]["seconds"])
            durations[name] = duration
        process_hours[column] = sum(durations.values()) / 3600
        schedules[column] = schedule(jobs, durations, workers, running)
    prefix_jobs = [job for job in jobs if job["id"] not in blocked]
    prefix_durations = {job["id"]: (0.0 if rows[job["id"]]["complete"] else
                        max(0.0, rows[job["id"]]["mean"] - running.get(job["id"], {}).get("seconds", 0)))
                        for job in prefix_jobs}
    prefix = schedule(prefix_jobs, prefix_durations, workers, running)
    groups = defaultdict(list)
    for job in jobs:
        label = "search/" + job["spec"]["family"] if job["kind"] == "search" else job["kind"]
        groups[label].append(rows[job["id"]])
    classes = {}
    for label, members in groups.items():
        counts = Counter(row["classification"] for row in members)
        complete = [row for row in members if row["complete"]]
        remaining = [row for row in members if not row["complete"]]
        classes[label] = dict(total=len(members), complete=len(complete), remaining=len(remaining),
                              counts={k: counts[k] for k in ("measured", "proxy", "unmeasured")},
                              measured_completed_process_hours=sum(r["mean"] for r in complete) / 3600,
                              covered_remaining_process_hours={k: sum(max(0.0, r[k] - running.get(r["id"], {}).get("seconds", 0))
                                  for r in remaining if r[k] is not None) / 3600 for k in ("low", "mean", "high")})
    missing = defaultdict(list)
    for name in unknown:
        missing[category(by_id[name])].append(name)
    missing_categories = [dict(category=list(key), jobs=len(names), examples=sorted(names)[:2])
                          for key, names in sorted(missing.items(), key=lambda item: str(item[0]))]
    smoke = read(smoke_path) if smoke_path and Path(smoke_path).exists() else None
    smoke_reference = None
    if smoke:
        smoke_reference = dict(path=str(smoke_path), scope=smoke["scope"],
            joint_fixture=smoke["joint_fixture"], used_for_duration_projection=False,
            reason="Local two-step repair and 28-event P / 308-event D fixtures do not measure formal repair budgets or GPT prefill.")
    full_available = not unknown and not overruns
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(), jobs=len(jobs), completed=len(observed),
        remaining=len(jobs) - len(observed), workers=workers, classes=classes,
        completed_process_hours=sum(r["seconds"] for r in observed.values()) / 3600,
        full_remaining_estimate_available=full_available,
        full_remaining_process_hours=process_hours if full_available else None,
        full_remaining_schedule=schedules if full_available else None,
        covered_remaining_process_hours=process_hours,
        covered_work_schedule_with_unmeasured_duration_zero=schedules,
        measured_prefix=dict(jobs=len(prefix_jobs), pending=sum(not rows[j["id"]]["complete"] for j in prefix_jobs), **prefix),
        unmeasured_jobs=len(unknown), known_jobs_behind_unmeasured_dependencies=len(blocked - unknown),
        missing_categories=missing_categories, rejected_observations=rejected,
        running_snapshot=dict(count=len(running), source_wall_seconds=status.get("wall_seconds"),
                              elapsed_process_hours=sum(r["seconds"] for r in running.values()) / 3600,
                              unmeasured_running=[r for name, r in running.items() if name in unknown],
                              beyond_observed_range=overruns),
        smoke_reference=smoke_reference,
        historical_D_reference={str(key): dict(evaluations=len(values),
            mean_seconds=statistics.mean(r["seconds"] for r in values),
            min_seconds=min(r["seconds"] for r in values), max_seconds=max(r["seconds"] for r in values),
            source_files=sorted({r["path"] for r in values})) for key, values in references.items()},
        assumptions=[
            "Durations are process wall time, including contention; process-hours are not CPU utilization measurements.",
            "Low/high are observed sample minima/maxima, not confidence bounds. A single sample gives no variability estimate.",
            "Restarts share a timing category only when platform, workload, method and budget match.",
            "Gemini seed 123 includes terminal evaluation once; other restarts do not inherit that extra cost.",
            "FULL and bandwidth proxies use matching healthy searches; they remain proxies until measured.",
            "All repair tiers are counted until real stop results are available; no early-stop rate is assumed.",
            "Historical D proxies use individual real cloud run timings, with orig/hop and solo P/D/overlap separated; no division of a mixed job total is used.",
            "D proxies scale by observed F/G unchunked event counts relative to WATOS 2x2; this does not establish chunked-hop complexity or changed hardware/launch costs.",
            "The dependency simulation has at most the requested process count. Memory restrictions, future interference and launch spacing are not modeled.",
            "Priority order: ready repair/joint tasks, WATOS, G3x3 growth, other F/G growth, G fixed; longer jobs first within a priority.",
            "The covered-work schedule sets unknown durations to zero and is only a partial-work projection, not a whole-run finishing time.",
            "Status is a supplied snapshot. Elapsed running time is subtracted only from measured/proxy duration; unfinished samples never become completed timings.",
        ], job_estimates=list(rows.values()))


def updated_jobs(jobs, report):
    """Return a new manifest with timing hints only; preserve scheduler identity."""
    estimates_by_id = {row["id"]: row for row in report["job_estimates"]}
    output = copy.deepcopy(jobs)
    for job in output:
        row = estimates_by_id[job["id"]]
        job["priority"] = priority(job)
        if row["mean"] is not None:
            job["estimated_seconds"] = row["mean"]
            job["timing_basis"] = f"FG {row['classification']}: {row['basis']} ({row['samples']} samples)"
    assert all(identity(a) == identity(b) for a, b in zip(jobs, output))
    return output


def markdown(report):
    lines = ["# F/G 运行时间更新", "", f"已完成 {report['completed']} / {report['jobs']} 个作业。", "",
             "| 作业 | 完成/全部 | 同类实测 | 代理估计 | 未测 | 已覆盖的剩余进程小时 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in report["classes"].items():
        counts = row["counts"]
        lines.append(f"| {name} | {row['complete']}/{row['total']} | {counts['measured']} | {counts['proxy']} | {counts['unmeasured']} | {row['covered_remaining_process_hours']['mean']:.2f} |")
    covered = report["covered_remaining_process_hours"]
    schedules = report["covered_work_schedule_with_unmeasured_duration_zero"]
    lines += ["", f"已有时间依据的剩余工作：{covered['mean']:.2f} 进程小时（样本最小/最大值换算：{covered['low']:.2f}–{covered['high']:.2f}）。",
              f"按 {report['workers']} 进程并遵守作业依赖，这部分工作约 {schedules['mean']['hours']:.2f} 小时（{schedules['low']['hours']:.2f}–{schedules['high']['hours']:.2f}）。计算暂把未测作业耗时设为 0；此数不能作为全部完成时间。",
              f"目前不经过未测作业便可执行的已测部分：{report['measured_prefix']['pending']} 个作业，约 {report['measured_prefix']['hours']:.2f} 小时。"]
    if report["full_remaining_estimate_available"]:
        lines.append(f"全流程剩余估计：{report['full_remaining_schedule']['mean']['hours']:.2f} 小时。")
    else:
        lines.append(f"全流程完成时间尚不能计算：{report['unmeasured_jobs']} 个作业没有同类正式测量，另有 {len(report['running_snapshot']['beyond_observed_range'])} 个在跑作业已超出对应样本最大耗时。")
    lines += ["", "进程小时按每个子进程的实际经过时间累加，含并发争用。样本最小/最大值不是置信区间；只有一个样本时尚无波动范围。排程未计内存导致的降并发。",
              "", "本机两步修复与 28 事件 P、308 事件 D 的小图只用于验证连接和调用，未用于推算真实 prefill 修复或联合实验。30/100/300 步修复按清单分别保留，未假定提前停止比例。"]
    if report["historical_D_reference"]:
        lines += ["", "联合实验的代理来自 D 阶段真实云端单次评估；orig/hop、P/D 单跑与联合分别统计。新规模暂按已测图的事件数量相对 2×2 变化外推，尚未校准硬件变化、切块增长和逐作业启动成本。其范围为历史样本最小/最大值。"]
    if report["running_snapshot"]["unmeasured_running"]:
        longest = max(report["running_snapshot"]["unmeasured_running"], key=lambda r: r["seconds"])
        lines += ["", f"未完成样本中最长已运行 {longest['seconds'] / 60:.1f} 分钟：`{longest['id']}`；这只是已经过时间。"]
    lines += ["", "JSON 中 missing_categories 列出待测配置/算子/预算，job_estimates 给出每个作业的来源。",
              "--updated-jobs 写入新的任务清单副本，只更新外层 estimated_seconds、timing_basis、priority；spec、依赖和任务身份保持一致。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", "--run-out", type=Path, required=True)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--smoke", type=Path, default=HERE / "fg/evidence/repair_joint_smoke.json")
    parser.add_argument("--d-root", type=Path, help="Optional D output root for historical cloud joint proxies")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--updated-jobs", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 128:
        parser.error("workers must be between 1 and 128")
    if args.updated_jobs and args.updated_jobs.resolve() == args.manifest.resolve():
        parser.error("--updated-jobs must be a new path, not the input manifest")
    jobs = read(args.manifest)
    report = analyse(jobs, args.out, args.workers, read(args.status) if args.status else None, args.smoke, args.d_root)
    report.update(manifest=str(args.manifest), output=str(args.out), status_source=str(args.status) if args.status else None)
    write(args.report, report)
    summary = markdown(report)
    summary_path = args.summary or args.report.with_suffix(".md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    if args.updated_jobs:
        write(args.updated_jobs, updated_jobs(jobs, report))
    print(summary)


if __name__ == "__main__":
    main()
