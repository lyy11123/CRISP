"""Read closeout receipts and the dynamic registry; write dated ETA snapshots.

Execution files and previous estimates are never changed. Unfinished jobs use
the median of completed, non-reused jobs in the same experimental class. Before
B/C fault measurements arrive, current healthy backend time replaces only the
historical backend part of their estimates. An overdue live job has unknown
remaining time and is reported separately.
"""
import argparse
import collections
from datetime import datetime, timezone, timedelta
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import statistics

import fg_batch
from e_run import ROOT, write_json

BASE = ROOT / "e2e/hi/closeout"
OUT = ROOT / "e2e/hi/out/closeout_20261115"
REGISTRY = BASE / "manifests/registry.json"
OK = {"ok", "reused", "complete", "completed"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def root_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def package(job):
    name = job["id"]
    for prefix in ("A", "B", "C", "S"):
        if name.startswith(prefix + "_"):
            return prefix
    raise ValueError(f"Unrecognized closeout package: {name}")


def category(job):
    s = job["spec"]
    return (package(job), s.get("model", "gpt"), s.get("hardware_profile"),
            s.get("phase"), s.get("op", s.get("phase_op")), job["kind"],
            s.get("start_arm"), s.get("steps"), s.get("fault_id"),
            s.get("method"), s.get("factor"), s.get("mapping_state"),
            s.get("transport"), s.get("stream"))


def is_reused(result):
    return result["status"] == "reused" or "reused_healthy_record" in result


def registry_jobs(filename):
    registry = read(filename)
    sources, jobs = [], []
    for value in registry["manifests"]:
        p = root_path(value)
        sources.append(dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        jobs.extend(fg_batch.job_items(p))
    by_id = {j["id"]: j for j in jobs}
    assert len(by_id) == len(jobs)
    assert {d for j in jobs for d in j.get("deps", [])} <= set(by_id)
    return registry, jobs, sources


def completed_records(jobs, out):
    records, evidence, timings, unconfirmed = {}, {}, {}, []
    for job in jobs:
        name = job["id"]
        receipt_path = out / "receipts" / (name + ".json")
        receipt = read(receipt_path) if receipt_path.exists() else None
        if receipt is not None and (receipt["rc"] != 0 or receipt["identity"] != fg_batch.identity(job)):
            continue
        paths = [out / job["result"]]
        if package(job) == "C":
            paths.append(out / "c_mf6" / job["result"])
        result_path = next((p for p in paths if p.exists()), None)
        if result_path is None:
            continue
        result = read(result_path)
        if result.get("status") not in OK:
            continue
        if "spec" in result:
            if result["spec"] != job["spec"]:
                unconfirmed.append(dict(job_id=name, reason="result spec differs from registered job"))
                continue
        elif receipt is None:
            unconfirmed.append(dict(job_id=name, reason="result has no full spec and no matching successful receipt"))
            continue
        assert result.get("job_id", name) == name
        records[name] = result
        evidence[name] = "matching_success_receipt" if receipt else "matching_result_only"
        timings[name] = receipt["wall_seconds"] if receipt else result["wall_seconds"]
    return records, evidence, timings, unconfirmed


def operator_key(spec):
    return (spec["model"], spec["hardware_profile"], spec["phase_op"], spec["shape"],
            spec["split_mode"], spec.get("d2d_scale", 1), spec["workload_version"], spec["start_arm"])


def healthy_backend_observations(jobs, completed, evidence):
    """Use the ten backend calls themselves, excluding reuse and preparation."""
    initial, observed = {}, collections.defaultdict(list)
    prepared_pickle_types = False
    for job in jobs:
        if package(job) not in ("B", "C") or job["kind"] != "fault_pre":
            continue
        key = operator_key(job["spec"])
        # Both approved planners define healthy pre as old ten-backend + 20s.
        initial[package(job), key] = float(job["estimated_seconds"]) - 20.
        assert initial[package(job), key] > 0
        result = completed.get(job["id"])
        if result is None or is_reused(result):
            continue
        if not prepared_pickle_types:
            import post_fg_split
            post_fg_split.setup()
            prepared_pickle_types = True
        path = root_path(result["sink"])
        with gzip.open(path, "rb") as stream:
            raw = pickle.load(stream)
        assert raw["backend_seeds"] == list(range(1, 11))
        record = raw["record"]
        assert set(record["backend"]) == set(range(1, 11)) and len(record["backend_wall"]) == 10
        observed[key].append(dict(job_id=job["id"], package=package(job),
            backend_ten_seconds=sum(record["backend_wall"]),
            backend_individual_seconds=record["backend_wall"], evidence=evidence[job["id"]],
            raw=result["sink"], raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            meaning="current healthy ten-backend calls; not a fault measurement"))
    return initial, observed


def full_estimate(job, same_class, initial_backends, healthy_observed):
    initial = float(job["estimated_seconds"])
    if same_class:
        return statistics.median(x["seconds"] for x in same_class), "same_class_measured_median", {}
    if package(job) in ("B", "C") and job["kind"] in ("fault_l1", "fault_rep"):
        key = operator_key(job["spec"])
        healthy = healthy_observed.get(key, [])
        if healthy:
            old = initial_backends[package(job), key]
            new = statistics.median(r["backend_ten_seconds"] for r in healthy)
            multiplier = 2 if job["kind"] == "fault_l1" else 1
            non_backend = initial - multiplier * old
            assert non_backend >= 0
            return non_backend + multiplier * new, "current_healthy_backend_projection", dict(
                backend_replacement_multiplier=multiplier, original_backend_ten_seconds=old,
                current_healthy_backend_ten_seconds=new, preserved_non_backend_seconds=non_backend,
                healthy_backend_source_jobs=[r["job_id"] for r in healthy],
                projection_note="Healthy backend runtime projected onto fault evaluations; fault runtime is not measured yet.")
    return initial, "manifest_initial_estimate", {}


def running_records(out, by_id, completed, now, live):
    filename = out / "closeout.status.json"
    status = read(filename) if filename.exists() else None
    status_time = datetime.fromisoformat(status["utc"]).timestamp() if status and status.get("utc") else None
    age = now.timestamp() - status_time if status_time is not None else None
    active = {}
    if status:
        for row in status.get("running", []):
            if row["id"] in by_id and row["id"] not in completed:
                active[row["id"]] = dict(row, elapsed_seconds=row["seconds"], source="status_snapshot")
    live_path = out / "live_workers.json"
    if live_path.exists():
        for name, row in read(live_path).items():
            if name not in by_id or name in completed:
                continue
            assert row["identity"] == fg_batch.identity(by_id[name])
            if live:
                if not fg_batch.process_alive(row["pid"]):
                    active.pop(name, None)
                    continue
                elapsed = max(0, now.timestamp() - row["started_unix"])
            else:
                if status_time is None:
                    continue
                elapsed = max(0, status_time - row["started_unix"])
            active[name] = dict(active.get(name, {}), pid=row["pid"], elapsed_seconds=elapsed, source="live_worker_record")
    return active, status, age


def stopped_ancestor(name, by_id, records, cache):
    if name in cache:
        return cache[name]
    result = records.get(name)
    if result and result.get("stop") is True:
        cache[name] = name
    else:
        prior = by_id[name]["spec"].get("previous_stop")
        cache[name] = stopped_ancestor(prior, by_id, records, cache) if prior else None
    return cache[name]


def dependency_costs(by_id, rows):
    point, known, paths = {}, {}, {}
    pending = set(by_id)
    while pending:
        ready = sorted(n for n in pending if set(by_id[n].get("deps", [])) <= set(point))
        assert ready, "Dependency cycle"
        for name in ready:
            deps = by_id[name].get("deps", [])
            if rows[name]["state"] == "completed":
                point[name], known[name], paths[name] = 0., 0., []
                pending.remove(name)
                continue
            predecessor = max(deps, key=lambda n: known[n], default=None)
            row = rows[name]
            duration = row["remaining_seconds"]
            known[name] = (known[predecessor] if predecessor else 0) + (duration or 0)
            paths[name] = (paths[predecessor] if predecessor else []) + ([name] if row["state"] != "completed" else [])
            point[name] = (None if duration is None or any(point[d] is None for d in deps)
                           else max((point[d] for d in deps), default=0) + duration)
            pending.remove(name)
    return point, known, paths


def table(filename, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with filename.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                         for k, v in row.items()} for row in rows)


def estimate(out, registry_path, destination, live=False):
    now = datetime.now(timezone.utc)
    registry, jobs, sources = registry_jobs(registry_path)
    by_id = {j["id"]: j for j in jobs}
    completed, completion_evidence, measured_wall, unconfirmed = completed_records(jobs, out)
    active, status, status_age = running_records(out, by_id, completed, now, live)
    observations = collections.defaultdict(list)
    for name, record in completed.items():
        if not is_reused(record):
            observations[category(by_id[name])].append(dict(job_id=name, seconds=measured_wall[name], evidence=completion_evidence[name]))
    initial_backends, healthy_observed = healthy_backend_observations(jobs, completed, completion_evidence)
    rows, stopped = {}, {}
    for job in jobs:
        name, spec = job["id"], job["spec"]
        samples = observations[category(job)]
        initial = float(job["estimated_seconds"])
        full, source, projection = full_estimate(job, samples, initial_backends, healthy_observed)
        row = dict(job_id=name, package=package(job), kind=job["kind"], model=spec.get("model", "gpt"),
                   hardware=spec.get("hardware_profile"), phase=spec.get("phase"), op=spec.get("op", spec.get("phase_op")),
                   steps=spec.get("steps"), initial_seconds=initial, current_full_seconds=full,
                   estimate_source=source, observed_count=len(samples), observed_jobs=[x["job_id"] for x in samples],
                   state="completed" if name in completed else "running" if name in active else "pending",
                   elapsed_seconds=0., overdue_seconds=0., stopped_by=None, **projection)
        if name in completed:
            row.update(remaining_seconds=0., actual_seconds=measured_wall[name], reused=is_reused(completed[name]),
                       completion_evidence=completion_evidence[name])
        elif name in active:
            elapsed = active[name]["elapsed_seconds"]
            row.update(elapsed_seconds=elapsed, pid=active[name]["pid"], active_evidence=active[name]["source"],
                       remaining_seconds=full - elapsed if elapsed < full else None,
                       overdue_seconds=max(0., elapsed - full))
        else:
            previous_stop = spec.get("previous_stop")
            ancestor = stopped_ancestor(previous_stop, by_id, completed, stopped) if previous_stop else None
            if ancestor and job["kind"] in ("fault_rep", "s_rep", "fault_stop", "s_stop"):
                row.update(remaining_seconds=1., current_full_seconds=1., estimate_source="known_stopped_chain_control_only",
                           stopped_by=ancestor, observed_count=0, observed_jobs=[])
            else:
                row["remaining_seconds"] = full
        rows[name] = row
    point, known, paths = dependency_costs(by_id, rows)
    groups = []
    for label in ("A", "B", "C", "S"):
        group = [r for r in rows.values() if r["package"] == label]
        if not group:
            groups.append(dict(package=label, registered_jobs=0, state="not_yet_registered"))
            continue
        remaining = [r for r in group if r["state"] != "completed"]
        unknown = [r["job_id"] for r in remaining if r["remaining_seconds"] is None]
        total_known = sum(r["remaining_seconds"] or 0 for r in remaining)
        longest = max(group, key=lambda r: known[r["job_id"]])["job_id"]
        exact_chain = all(point[r["job_id"]] is not None for r in group)
        groups.append(dict(package=label, registered_jobs=len(group), completed=len(group) - len(remaining),
                           completed_reused=sum(r.get("reused", False) for r in group),
                           running=sum(r["state"] == "running" for r in group), pending=sum(r["state"] == "pending" for r in group),
                           remaining_process_hours=total_known / 3600 if not unknown else None,
                           remaining_known_estimate_process_hours=total_known / 3600,
                           unknown_running=unknown, longest_dependency_minutes=max((point[r["job_id"]] for r in group), default=0) / 60 if exact_chain else None,
                           longest_known_component_minutes=known[longest] / 60, longest_known_component_jobs=paths[longest],
                           initial_total_process_hours=sum(r["initial_seconds"] for r in group) / 3600,
                           successful_process_hours=sum(r.get("actual_seconds", 0) for r in group) / 3600,
                           initial_remaining_full_job_hours=sum(r["initial_seconds"] for r in remaining) / 3600,
                           reduced_after_stop=sum(r["stopped_by"] is not None for r in group)))
    overdue = [r for r in rows.values() if r["state"] == "running" and r["remaining_seconds"] is None]
    longest_point = max((value for value in point.values() if value is not None), default=0) if not overdue else None
    known_process_hours = sum(r["remaining_seconds"] or 0 for r in rows.values()) / 3600
    result = dict(created_utc=now.isoformat(), mode="live_host_read" if live else "offline_file_snapshot",
                  registry_accepting=registry["accepting"], registered_jobs=len(jobs), verified_completed=len(completed),
                  status_reported_completed=status.get("completed") if status else None,
                  status_age_seconds=status_age, status_available_gib=status.get("available_gib") if status else None,
                  reserve_gib=3, process_limit=128, packages=groups, overdue_running=overdue,
                  remaining_process_hours=known_process_hours if not overdue else None,
                  remaining_known_estimate_process_hours=known_process_hours,
                  longest_dependency_minutes=longest_point / 60 if longest_point is not None else None,
                  earliest_dependency_only_utc=(now + timedelta(seconds=longest_point)).isoformat()
                    if live and status_age is not None and status_age <= 120 and longest_point is not None else None,
                  unconfirmed_results=unconfirmed, source_manifests=sources,
                  healthy_backend_observations=[dict(operator=list(key), samples=values,
                    median_backend_ten_seconds=statistics.median(v["backend_ten_seconds"] for v in values))
                    for key,values in healthy_observed.items()],
                  healthy_backend_projection_jobs=sum(r["estimate_source"] == "current_healthy_backend_projection" for r in rows.values()),
                  class_observations=[dict(category=list(key), count=len(values), median_seconds=statistics.median(v["seconds"] for v in values),
                                           minimum_seconds=min(v["seconds"] for v in values), maximum_seconds=max(v["seconds"] for v in values),
                                           samples=values) for key, values in observations.items() if values],
                  includes="currently registered A/B/C/S complete remaining jobs, including their final backends and exports",
                  excludes="not-yet-registered jobs, J serial timing, COS transfer, reports; successful_process_hours does not include failed attempts",
                  interpretation="Dependency estimates assume ready jobs have slots and memory. Known components are estimated work, not mathematical lower bounds. Overdue running time is unknown, never clamped to a fictitious positive remainder.")
    destination.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    stem = destination / ("eta_" + stamp)
    write_json(stem.with_suffix(".json"), result)
    table(stem.with_name(stem.name + "_packages.csv"), groups)
    table(stem.with_name(stem.name + "_jobs.csv"), list(rows.values()))
    text = ["# 收尾实验用时更新", "", f"记录时间：{now.astimezone(timezone(timedelta(hours=8))).isoformat()}。已登记 {len(jobs)} 项，本地可核对完成 {len(completed)} 项。", "",
            "| 工作包 | 已完成/登记 | 运行中 | 待运行 | 剩余进程小时 | 最长剩余依赖链 |", "|---|---:|---:|---:|---:|---:|"]
    for row in groups:
        if not row["registered_jobs"]:
            text.append(f"| {row['package']} | 尚未登记 | — | — | — | — |")
            continue
        ph = f"{row['remaining_process_hours']:.2f}" if row["remaining_process_hours"] is not None else f"未定；已知部分 {row['remaining_known_estimate_process_hours']:.2f}"
        chain = f"{row['longest_dependency_minutes']:.1f} 分钟" if row["longest_dependency_minutes"] is not None else f"未定；已知部分 {row['longest_known_component_minutes']:.1f} 分钟"
        text.append(f"| {row['package']} | {row['completed']}/{row['registered_jobs']} | {row['running']} | {row['pending']} | {ph} | {chain} |")
    text += ["", "同类故障作业有完整实测时，优先使用实测中位数。B/C尚无同类L1/L2实测时，用本轮同模型、硬件和算子健康pre的十次后端耗时替换原估时的后端部分；L1替换两份，L2替换一份，搜索和固定成本保留。这是健康后端对故障时长的推算，尚未测得的故障开销会在首项完成后更新。其余类别保留原清单初值。", "",
             "复用记录不用于估算搜索/后端耗时，明确停档后的作业只计短控制时间。各项原始初值、保留成本与替换来源保留在CSV中。", "",
             "依赖链假设任务就绪后有可用进程和内存；总进程小时除以128不能替代它。上述范围只包含当前登记的质量任务，不含后续新增任务、J串行计时、下载与汇总。"]
    for key,values in healthy_observed.items():
        value = statistics.median(r["backend_ten_seconds"] for r in values)
        text += [f"", f"健康后端参考：{key[0]} / {key[1]} / {key[2]}，十次后端中位累计{value/60:.2f}分钟，来源 " + ", ".join(r["job_id"] for r in values) + "。"]
    if not live:
        text += ["", "这是本机文件快照，不能代表服务器此刻的完成数或据此给出结束时刻。"]
    if status_age is not None and status_age > 120:
        text += ["", f"状态文件距本次读取已 {status_age / 60:.1f} 分钟，运行时长按所保存时刻解释。"]
    if overdue:
        text += ["", "以下作业已超过当前同类估计，剩余时间尚不能由已有完整样本确定："]
        text += [f"- {r['job_id']}：已运行 {r['elapsed_seconds']/60:.1f} 分钟，原/更新估计 {r['current_full_seconds']/60:.1f} 分钟。" for r in overdue]
    stem.with_suffix(".md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return dict(json=str(stem.with_suffix(".json")), markdown=str(stem.with_suffix(".md")),
                registered_jobs=len(jobs), verified_completed=len(completed), overdue=len(overdue), packages=groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--destination", type=Path, default=BASE / "evidence")
    parser.add_argument("--live", action="store_true", help="Read process liveness on this same server; otherwise preserve snapshot semantics")
    args = parser.parse_args()
    print(json.dumps(estimate(root_path(args.out), root_path(args.registry), root_path(args.destination), args.live), ensure_ascii=False))
