"""Run F/G jobs in fresh processes, ordered by duration and available memory.

python e2e/fg_batch.py --manifest e2e/fg/pilot.json --out e2e/fg/out/run --workers 128
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from e_run import ROOT, write_json


def available_gib():
    if sys.platform == "win32":
        import ctypes

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (key, ctypes.c_ulonglong) for key in
                ("total", "available", "page_total", "page_available", "virtual_total",
                 "virtual_available", "extended")]

        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        return memory.available / 2**30
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines())
    return int(info["MemAvailable"].split()[0]) / 2**20


def process_rss(pid):
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return 0.0
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def job_items(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["jobs"] if isinstance(data, dict) else data
    jobs = []
    for item in items:
        if "spec" in item:
            jobs.append(item)
            continue
        name = item.get("job_id") or item["candidate_id"]
        job = dict(id=name, kind=item["kind"], deps=item.get("depends_on", []), spec=item,
                   result=f"results/{name}.json")
        module = item.get("module", "fg_run")
        job["argv"] = ["{python}", "-u", "-B", f"e2e/{module}.py", "child",
                       "--spec", "{spec}", "--out", "{out}"]
        jobs.append(job)
    return jobs


def category(job):
    s = job["spec"]
    return (job["kind"], s.get("hardware_profile"), s.get("shape"), s.get("split_mode"),
            s.get("phase"), s.get("op"), s.get("method"))


def identity(job):
    payload = {key: job[key] for key in ("id", "kind", "spec", "result")}
    payload["argv"] = job.get("argv")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def execute(manifest, out, workers, reserve_gib, run_name, memory_hint, release_file=None):
    out = out.resolve()
    for folder in ("logs", "specs", "receipts"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    jobs = job_items(manifest)
    assert len({j["id"] for j in jobs}) == len(jobs)
    ids = {j["id"] for j in jobs}
    for job in jobs:
        assert set(job.get("deps", ())) <= ids, (job["id"], job.get("deps"))
    completed, failed, running, pending, rows = set(), set(), {}, {}, []
    paused_categories = set()
    duration, memory = {}, {}
    for path in (out / "receipts").glob("*.json"):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt["rc"] == 0:
            key = tuple(receipt["category"])
            duration.setdefault(key, []).append(receipt["wall_seconds"])
            memory[key] = max(memory.get(key, 0), receipt.get("peak_rss_gb", 0))
    for job in jobs:
        receipt_path = out / "receipts" / (job["id"] + ".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt["identity"] == identity(job) and receipt["rc"] == 0
                    and (out / job["result"]).exists()):
                completed.add(job["id"])
                rows.append(dict(receipt, cached=True))
                continue
        pending[job["id"]] = job
    historical = json.loads((ROOT / "e2e/e/evidence/fg_estimates.json").read_text(encoding="utf-8"))
    timing = historical["per_job_duration_sources"]
    hints_path = ROOT / "e2e/fg/manifest/memory_hints.json"
    hints = json.loads(hints_path.read_text(encoding="utf-8")) if hints_path.exists() else {}
    config_memory = {r["config_id"]: r["peak_rss_gb"] for r in hints.get("by_configuration", [])}

    def seconds(job):
        key = category(job)
        if key in duration:
            return sum(duration[key]) / len(duration[key])
        s = job["spec"]
        name = "/".join(str(s.get(k, "")) for k in ("phase", "op", "method"))
        return job.get("estimated_seconds", timing.get(name, {}).get("mean_job_seconds", 900))

    def gib(job):
        measured = memory.get(category(job), 0)
        configuration_peak = config_memory.get(job["spec"].get("config_id"), 0)
        return max(job.get("peak_rss_gb", memory_hint), 1.25 * measured, 1.25 * configuration_peak)

    started, next_status = time.monotonic(), 0
    min_available, max_active = available_gib(), 0
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    print(json.dumps(dict(event="start", jobs=len(jobs), cached=len(completed), workers=workers,
                          reserve_gib=reserve_gib, pid=os.getpid())), flush=True)
    while pending or running:
        for name, active in list(running.items()):
            active["peak"] = max(active["peak"], process_rss(active["process"].pid))
            key = category(active["job"])
            memory[key] = max(memory.get(key, 0), active["peak"])
            rc = active["process"].poll()
            if rc is None:
                continue
            active["log"].close()
            job = active["job"]
            result_path = out / job["result"]
            peak = active["peak"]
            if rc == 0:
                if not result_path.exists():
                    rc = 90
                else:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if result.get("status") not in (None, "ok", "complete", "completed", "reused"):
                        rc = 91
                    peak = max(peak, result.get("peak_rss_gb") or 0)
            receipt = dict(id=name, identity=identity(job), category=category(job), rc=rc,
                           wall_seconds=time.monotonic() - active["started"], peak_rss_gb=peak,
                           result=job["result"], cached=False)
            write_json(out / "receipts" / (name + ".json"), receipt)
            rows.append(receipt)
            key = category(job)
            memory[key] = max(memory.get(key, 0), peak)
            if rc == 0:
                completed.add(name)
                duration.setdefault(key, []).append(receipt["wall_seconds"])
            else:
                failed.add(name)
                paused_categories.add(key)
            print(json.dumps(dict(event="done", **receipt)), flush=True)
            del running[name]

        free = available_gib()
        min_available = min(min_available, free)
        committed = sum(max(0, gib(a["job"]) - process_rss(a["process"].pid))
                        for a in running.values())
        budget = free - reserve_gib - committed
        release = json.loads(release_file.read_text(encoding="utf-8")) if release_file else {"all": True}
        released = ids if release.get("all") else set(release["ids"])
        ready = sorted((j for j in pending.values() if j["id"] in released
                        and set(j.get("deps", ())) <= completed
                        and category(j) not in paused_categories),
                       key=seconds, reverse=True)
        # A short ramp lets new workers report RSS before the next eight start.
        launched = 0
        for job in ready:
            if len(running) >= workers or launched >= 8:
                break
            need = gib(job)
            if need > budget:
                continue
            name = job["id"]
            spec_path = out / "specs" / (name + ".json")
            write_json(spec_path, job["spec"])
            argv = job.get("argv", [sys.executable, "-u", "-B", "e2e/fg_run.py", "child",
                                    "--spec", "{spec}", "--out", "{out}"])
            argv = [arg.replace("{python}", sys.executable).replace("{spec}", str(spec_path))
                    .replace("{out}", str(out)) for arg in argv]
            stamp = time.time_ns()
            log = (out / "logs" / f"{name}.{stamp}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[name] = dict(process=process, log=log, job=job, peak=0.0, started=time.monotonic())
            del pending[name]
            budget -= need
            launched += 1
        max_active = max(max_active, len(running))
        now = time.monotonic()
        if now >= next_status:
            status = dict(event="status", pid=os.getpid(), manifest=str(manifest), jobs=len(jobs),
                          completed=len(completed), failed=sorted(failed), active=len(running),
                          pending=len(pending), available_gib=free, min_available_gib=min_available,
                          wall_seconds=now-started, max_active=max_active,
                          waiting_for_release=sum(n not in released for n in pending),
                          running=[dict(id=n, pid=a["process"].pid,
                                        seconds=now-a["started"], rss_gb=process_rss(a["process"].pid))
                                   for n, a in running.items()])
            write_json(out / (run_name + ".status.json"), status)
            print(json.dumps({k:v for k,v in status.items() if k != "running"}), flush=True)
            next_status = now + 30
        if not running and pending and not ready and not any(n not in released for n in pending):
            print(json.dumps(dict(event="blocked_dependencies", jobs=sorted(pending))), flush=True)
            break
        time.sleep(2)
    batch = dict(manifest=str(manifest), workers=workers, wall_seconds=time.monotonic()-started,
                 runs=rows, completed=len(completed), failed=sorted(failed), pending=sorted(pending),
                 min_available_gib=min_available, max_active=max_active,
                 python=sys.version, platform=sys.platform)
    write_json(out / (run_name + ".batch.json"), batch)
    print(json.dumps(dict(event="finished", completed=len(completed), failed=len(failed),
                          pending=len(pending), wall_seconds=batch["wall_seconds"])), flush=True)
    return int(bool(failed or pending))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--reserve-gib", type=float, default=16)
    parser.add_argument("--memory-hint-gib", type=float, default=1.5)
    parser.add_argument("--name", default="batch")
    parser.add_argument("--release-file", type=Path)
    args = parser.parse_args()
    assert 1 <= args.workers <= 128
    return execute(args.manifest, args.out, args.workers, args.reserve_gib, args.name,
                   args.memory_hint_gib, args.release_file)


if __name__ == "__main__":
    raise SystemExit(main())
