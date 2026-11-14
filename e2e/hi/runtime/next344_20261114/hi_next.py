"""One existing F/G process pool for the authorized H2D/H2P/H1c/I1 jobs."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import types

import fg_batch as pool
from e_run import ROOT, write_json

HERE = ROOT / "e2e/hi"
BATCHES = {"h2d": 166, "h2p": 166, "h1c": 4, "i1": 8}
RAW_CATEGORY = pool.category
RAW_MEMORY_KEY = pool.memory_key
RAW_MEMORY_REQUIREMENT = pool.memory_requirement
RAW_POPEN = subprocess.Popen
RAW_JOB_ITEMS = pool.job_items


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def category(job):
    return (job["spec"]["model"],) + RAW_CATEGORY(job)


def memory_key(job):
    return (job["spec"]["model"],) + RAW_MEMORY_KEY(job)


def wrapped_pool_source():
    source = inspect.getsource(pool.execute)
    replacements = {
        'for path in (out / "receipts").glob("*.json"):\n': 'for path in _hi.receipt_paths():\n',
        '        if external:\n            external_state = external.poll()\n':
        '        if external:\n            external_state = external.poll()\n            external_state = _hi.observe_external(external, external_state, external_ids, pending, failed)\n',
        '                failed.add(name)\n                paused_categories.add(key)\n':
        '                if _hi.failure(receipt, job, active):\n                    pending[name] = job\n                else:\n                    failed.add(name)\n                    paused_categories.add(key)\n'}
    for old, new in replacements.items():
        assert source.count(old) == 1
        source = source.replace(old, new)
    assert source.count('key = tuple(receipt["category"])') == 2
    source = source.replace('key = tuple(receipt["category"])', 'key = category(jobs_by_id[receipt["id"]])')
    return source


def assemble():
    jobs, initial = [], set()
    estimates = {row["candidate_id"]: row for row in csv.DictReader(
        (HERE / "tables/remaining_candidate_estimates.csv").open(encoding="utf-8-sig", newline=""))}
    for name, count in BATCHES.items():
        subset = pool.job_items(HERE / "manifest" / (name + "_planned.json"))
        assert len(subset) == count
        jobs.extend(subset)
        initial.update(read(HERE / "manifest" / (name + "_initial_release.json"))["ids"])
    by_id = {job["id"]: job for job in jobs}
    assert len(jobs) == len(by_id) == 344
    predecessors = {}
    plan = read(HERE / "manifest/next_batch_release_plan.json")
    calibration = []
    for batch in plan["batches"]:
        calibration.extend(batch["class_calibrations"])
        for row in batch["class_calibrations"]:
            first = row["prerequisite_candidate"]
            assert first in initial
            for name in row["release_after_success"]:
                assert name not in predecessors and name not in initial
                assert category(by_id[name]) == category(by_id[first])
                predecessors[name] = first
    assert set(by_id) == initial | set(predecessors)
    assert len(initial) == 84 and len(calibration) == 32 and len(predecessors) == 260
    for job in jobs:
        spec, estimate = job["spec"], estimates[job["id"]]
        assert spec["search_steps"] == 1000 and spec["backend_seeds"] == list(range(1, 11))
        assert spec["random_cpcm_seeds"] == list(range(1, 11)) and spec["module"] == "hi_run"
        job["deps"] = [predecessors[job["id"]]] if job["id"] in predecessors else []
        job["estimated_seconds"] = float(estimate["estimated_seconds"])
        job["peak_rss_gb"] = float(estimate["memory_hint_gib"])
    return jobs, initial, calibration


class Controller:
    def __init__(self, out, workers, external_manifest=None, external_status=None):
        self.out, self.workers = out, workers
        self.jobs, self.initial, self.calibrations = assemble()
        self.by_id = {job["id"]: job for job in self.jobs}
        self.success, self.history, self.observed = {}, [], defaultdict(list)
        self.live = read(out / "live_workers.json") if (out / "live_workers.json").exists() else {}
        self.manifest, self.release = out / "authorized_344.json", out / "release.json"
        self.base_estimates = {j["id"]: j["estimated_seconds"] for j in self.jobs}
        self.started = time.time()
        self.external_manifest, self.external_status = external_manifest, external_status
        self.external_running = {}
        self.external_failure_seen = set()

    def valid_receipt(self, path):
        receipt = read(path)
        job = self.by_id.get(receipt["id"])
        return job is not None and receipt["identity"] == pool.identity(job)

    def receipt_paths(self):
        return [p for p in (self.out / "receipts").glob("*.json") if self.valid_receipt(p)]

    def success_receipt(self, receipt):
        name = receipt["id"]
        job = self.by_id[name]
        assert receipt["identity"] == pool.identity(job) and receipt["result"] == job["result"]
        result = read(self.out / job["result"])
        assert result["status"] == "ok" and all(result[k] == v for k, v in job["spec"].items())
        if name in self.success:
            return
        self.success[name] = receipt
        key = category(job)
        self.observed[key].append(receipt)
        seconds = statistics.median(r["wall_seconds"] for r in self.observed[key])
        rss = 1.25 * max(r["peak_rss_gb"] for r in self.observed[key])
        for other in self.jobs:
            if category(other) == key:
                other["estimated_seconds"] = seconds
                other["peak_rss_gb"] = max(other["peak_rss_gb"], rss)
        self.save_release()

    def save_release(self):
        released = set(self.initial)
        for row in self.calibrations:
            if row["prerequisite_candidate"] in self.success:
                released.update(row["release_after_success"])
        write_json(self.release, dict(all=False, ids=sorted(released), successful_prerequisites=
            [r["prerequisite_candidate"] for r in self.calibrations if r["prerequisite_candidate"] in self.success]))
        write_json(self.out / "class_estimates.json", dict(updated_utc=datetime.now(timezone.utc).isoformat(),
            groups=[dict(category=list(key), samples=len(rows), wall_seconds_median=statistics.median(r["wall_seconds"] for r in rows),
                         peak_rss_gib=max(r["peak_rss_gb"] for r in rows)) for key, rows in self.observed.items()]))

    def failure(self, receipt, job, active):
        stamp = str(time.time_ns())
        folder = self.out / "failed_attempts" / job["id"] / stamp
        folder.mkdir(parents=True, exist_ok=True)
        log_path = Path(active["log"].name)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        oom = receipt["rc"] in (-9, 137, -1073741801) or any(
            text in log for text in ("MemoryError", "Cannot allocate memory", "std::bad_alloc", "Out of memory"))
        evidence = dict(receipt, log=str(log_path), memory_related=oom, available_gib=pool.available_gib())
        write_json(folder / "receipt.json", evidence)
        result = self.out / job["result"]
        if result.exists():
            shutil.copy2(result, folder / "result.json")
        self.history.append(evidence)
        if oom:
            hint = max(1.5 * job["peak_rss_gb"], 1.5 * receipt["peak_rss_gb"])
            for other in self.jobs:
                if category(other) == category(job):
                    other["peak_rss_gb"] = max(other["peak_rss_gb"], hint)
            write_json(self.out / "memory_retry.json", dict(job_id=job["id"], required_gib=hint,
                       available_gib=pool.available_gib(), reserve_gib=3, action="retry when pool memory budget fits",
                       failed_attempt=str(folder), utc=datetime.now(timezone.utc).isoformat()))
        return oom

    def observe_external(self, external, state, external_ids, pending, failed):
        self.external_running = state["running"]
        for receipt in external.completed.values():
            self.success_receipt(receipt)
        # A stopped initial controller no longer owns failed OOM jobs.
        if not state["controller_alive"] and not state["active"]:
            for name in list(external.failed):
                if name in self.external_failure_seen:
                    continue
                receipt = read(self.out / "receipts" / (name + ".json"))
                logs = sorted((self.out / "logs").glob(name + ".*.log"))
                active = dict(log=types.SimpleNamespace(name=str(logs[-1])))
                self.external_failure_seen.add(name)
                if self.failure(receipt, self.by_id[name], active):
                    external_ids.discard(name)
                    external.jobs.pop(name)
                    external.failed.pop(name)
                    pending[name] = self.by_id[name]
                    failed.discard(name)
        return external.poll()

    def spawn(self, argv, **kwargs):
        spec = read(argv[argv.index("--spec") + 1])
        name = spec["candidate_id"]
        assert name in self.by_id and spec == self.by_id[name]["spec"]
        process = RAW_POPEN(argv, **kwargs)
        self.live[name] = dict(pid=process.pid, started_unix=time.time(), identity=pool.identity(self.by_id[name]),
                               log=str(kwargs["stdout"].name))
        write_json(self.out / "live_workers.json", self.live)
        return process

    def eta(self, status):
        active = dict(self.external_running)
        active.update({row["id"]: row for row in status["running"]})
        memory_budget = max(0, status["available_gib"] - 3 + sum(row["rss_gb"] for row in active.values()))
        overdue = [name for name, row in active.items() if category(self.by_id[name]) not in self.observed
                   and row["seconds"] >= self.by_id[name]["estimated_seconds"]]
        progress_rows = []
        for name in sorted(overdue):
            path = self.out / "progress" / (name + ".json")
            progress = read(path) if path.exists() else {}
            milestones = progress.get("milestones", {})
            step = progress.get("step", 1000 if "search_complete" in milestones else None)
            elapsed = progress.get("search_elapsed_seconds")
            if elapsed is None and "search_complete" in milestones:
                elapsed = milestones["search_complete"] - milestones["search_begin"]
            search_remaining = elapsed * (1000-step) / step if step and elapsed is not None else None
            progress_rows.append(dict(candidate_id=name, phase=progress.get("phase"), step=step, total_steps=1000,
                search_elapsed_seconds=elapsed, search_remaining_seconds=search_remaining,
                post_search_evaluation_remaining_seconds=None, elapsed_job_seconds=active[name]["seconds"],
                earlier_estimated_job_seconds=self.by_id[name]["estimated_seconds"],
                progress_age_seconds=time.time()-path.stat().st_mtime if path.exists() else None,
                basis="linear extrapolation from completed search steps only; remaining evaluation/export time unknown"))
        if overdue:
            remaining = [job for job in self.jobs if job["id"] not in self.success]
            measured = sum(category(job) in self.observed for job in remaining)
            return dict(updated_utc=datetime.now(timezone.utc).isoformat(), remaining_jobs=len(remaining),
                measured_class_jobs=measured, unmeasured_class_jobs=len(remaining)-measured,
                queue_seconds=None, planning_seconds=None, estimated_finish_utc=None,
                overdue_unmeasured_ids=sorted(overdue), overdue_unmeasured_progress=progress_rows,
                reason="Active jobs exceeded the earlier estimate and have no completed same-class sample; total completion time awaits measurement.",
                includes="remaining complete jobs: search, calibration, backend/tie evaluations, export/reload",
                excludes="later faults/joint/J, transfer and report", memory_budget_gib=memory_budget)
        done = set(self.success)
        pending = [j for j in self.jobs if j["id"] not in done and j["id"] not in active]
        running = [(max(1.0, self.by_id[name]["estimated_seconds"] - row["seconds"]),
                    self.by_id[name]["peak_rss_gb"], name) for name, row in active.items()]
        t, estimate_possible = 0.0, True
        while pending or running:
            occupied = sum(row[1] for row in running)
            for job in sorted(pending[:], key=lambda j: -j["estimated_seconds"]):
                if len(running) == self.workers:
                    break
                if set(job["deps"]) <= done and occupied + job["peak_rss_gb"] <= memory_budget:
                    running.append((t + job["estimated_seconds"], job["peak_rss_gb"], job["id"]))
                    occupied += job["peak_rss_gb"]
                    pending.remove(job)
            if not running:
                estimate_possible = False
                break
            t = min(row[0] for row in running)
            done.update(row[2] for row in running if row[0] <= t)
            running = [row for row in running if row[0] > t]
        now = datetime.now(timezone.utc)
        remaining = [j for j in self.jobs if j["id"] not in self.success]
        measured = sum(category(j) in self.observed for j in remaining)
        factor = 1.5 if measured == len(remaining) else 3.0
        return dict(updated_utc=now.isoformat(), remaining_jobs=len(remaining), measured_class_jobs=measured,
                    unmeasured_class_jobs=len(remaining)-measured, queue_seconds=t if estimate_possible else None,
                    planning_seconds=[t, t*factor] if estimate_possible else None,
                    estimated_finish_utc=[(now+timedelta(seconds=t*f)).isoformat() for f in (1, factor)] if estimate_possible else None,
                    includes="remaining complete jobs: search, calibration, backend/tie evaluations, export/reload",
                    excludes="later faults/joint/J, transfer and report", memory_budget_gib=memory_budget,
                    assumption="measured class median or H1 graph-size estimate; 128-slot memory/dependency scheduling; upper factor allows contention")

    def write(self, path, value):
        path = Path(path)
        if path.parent.name == "receipts":
            if value["rc"] == 0:
                self.success_receipt(value)
            self.live.pop(value["id"], None)
            write_json(self.out / "live_workers.json", self.live)
        elif path.name == "next_344.status.json":
            value["estimate"] = self.eta(value)
            value["failed_attempt_count"] = len(list((self.out / "failed_attempts").glob("*/*/receipt.json")))
            value["class_calibrations_complete"] = sum(row["prerequisite_candidate"] in self.success for row in self.calibrations)
            write_json(self.out / "estimated_finish.json", value["estimate"])
        write_json(path, value)

    def recover(self):
        # Let workers from a previous controller finish before taking ownership.
        while self.live:
            for name, row in list(self.live.items()):
                assert row["identity"] == pool.identity(self.by_id[name])
                if pool.process_alive(row["pid"]):
                    continue
                job = self.by_id[name]
                path = self.out / job["result"]
                if path.exists():
                    result = read(path)
                    if result["status"] == "ok" and all(result[k] == v for k, v in job["spec"].items()):
                        receipt = dict(id=name, identity=pool.identity(job), category=category(job), rc=0,
                            wall_seconds=result["wall_seconds"], peak_rss_gb=result["peak_rss_gb"],
                            result=job["result"], cached=False, recovered_after_controller_restart=True)
                        write_json(self.out / "receipts" / (name + ".json"), receipt)
                del self.live[name]
            write_json(self.out / "live_workers.json", self.live)
            if self.live:
                write_json(self.out / "recovery.json", dict(state="waiting for prior workers", workers=self.live,
                           utc=datetime.now(timezone.utc).isoformat()))
                time.sleep(2)

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "receipts").mkdir(exist_ok=True)
        write_json(self.manifest, dict(jobs=self.jobs, authorization="H2D166 H2P166 H1c4 I1eight only"))
        self.recover()
        for path in self.receipt_paths():
            receipt = read(path)
            if receipt["rc"] == 0:
                self.success_receipt(receipt)
        self.save_release()
        pool.category, pool.memory_key = category, memory_key
        pool.memory_requirement = lambda job, *args: max(RAW_MEMORY_REQUIREMENT(job, *args), job["peak_rss_gb"])
        pool.job_items = lambda path: self.jobs if Path(path).resolve() == self.manifest.resolve() else RAW_JOB_ITEMS(path)
        pool.write_json = self.write
        pool.subprocess = types.SimpleNamespace(Popen=self.spawn, STDOUT=subprocess.STDOUT)
        source = wrapped_pool_source()
        namespace = dict(vars(pool), _hi=self)
        exec(compile(source, "<HI authorized existing pool>", "exec"), namespace)
        rc = namespace["execute"](self.manifest, self.out, self.workers, 3, "next_344", 4, self.release,
                                  self.external_manifest, self.external_status)
        write_json(self.out / "controller_finish.json", dict(rc=rc, completed=len(self.success), expected=344,
                   utc=datetime.now(timezone.utc).isoformat()))
        return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "launch", "run"))
    parser.add_argument("--out", type=Path, default=HERE / "out/next_344")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--external-status", type=Path)
    args = parser.parse_args()
    assert 1 <= args.workers <= 128
    args.out = args.out.resolve()
    assert (args.external_manifest is None) == (args.external_status is None)
    if args.command == "check":
        jobs, initial, calibrations = assemble()
        compile(wrapped_pool_source(), "<HI authorized existing pool>", "exec")
        if args.external_manifest:
            external = RAW_JOB_ITEMS(args.external_manifest)
            by_id = {job["id"]: job for job in jobs}
            assert {job["id"] for job in external} == initial
            assert all(pool.identity(job) == pool.identity(by_id[job["id"]]) for job in external)
        print(json.dumps(dict(status="passed", jobs=len(jobs), initially_released=len(initial),
              calibration_classes=len(calibrations), dependent_jobs=sum(bool(j["deps"]) for j in jobs),
              categories=len({category(j) for j in jobs}), memory_keys=len({memory_key(j) for j in jobs}),
              proposed_out=str(args.out)), ensure_ascii=False))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "launch":
        previous = args.out / "next_344.start.json"
        if previous.exists() and pool.process_alive(read(previous)["pid"]):
            raise RuntimeError("next_344 controller already running")
        write_json(args.out / "scheduler_policy.json", dict(running_growth_fraction=.25))
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                   CRISP_E_ARTIFACT=str(ROOT / "e2e/work/artifact"), PYTHONUTF8="1")
        command = [sys.executable, "-u", "-B", "e2e/hi_next.py", "run", "--out", str(args.out), "--workers", str(args.workers)]
        if args.external_manifest:
            command.extend(["--external-manifest", str(args.external_manifest), "--external-status", str(args.external_status)])
        options = dict(start_new_session=True) if sys.platform != "win32" else dict(
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        with (args.out / "next_344.log").open("ab") as stream:
            process = RAW_POPEN(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, **options)
        write_json(previous, dict(pid=process.pid, command=command, created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256={name: hashlib.sha256((ROOT / "e2e" / name).read_bytes()).hexdigest()
                           for name in ("hi_next.py", "fg_batch.py", "hi_run.py")}))
        print(json.dumps(dict(pid=process.pid, out=str(args.out), jobs=344)), flush=True)
        return 0
    return Controller(args.out, args.workers, args.external_manifest, args.external_status).run()


if __name__ == "__main__":
    raise SystemExit(main())
