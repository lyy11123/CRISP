"""Estimate the remaining serial J work from completed repetitions."""
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import json
import statistics

import fg_batch
from e_run import ROOT, write_json

BASE = ROOT / "e2e/hi/closeout"
OUT = ROOT / "e2e/hi/out/closeout_j_20261115"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def group(job):
    s = job["spec"]
    return (job["kind"], s.get("input_id", s.get("case", s.get("model"))),
            s.get("method", s.get("implementation")), s.get("steps"))


def main():
    now = datetime.now(timezone.utc)
    jobs = [j for name in ("j_micro", "j_search", "j3")
            for j in fg_batch.job_items(BASE / "manifests" / (name + ".json"))]
    done, measured = {}, defaultdict(list)
    for job in jobs:
        receipt = OUT / "receipts" / (job["id"] + ".json")
        if receipt.exists():
            value = read(receipt)
            if value["rc"] == 0:
                assert value["identity"] == fg_batch.identity(job)
                done[job["id"]] = value["wall_seconds"]
                measured[group(job)].append(value["wall_seconds"])
    live_path = OUT / "live_workers.json"
    live = read(live_path) if live_path.exists() else {}
    rows = []
    for job in jobs:
        name, key = job["id"], group(job)
        seconds = statistics.median(measured[key]) if measured[key] else job["estimated_seconds"]
        elapsed = max(0, now.timestamp() - live[name]["started_unix"]) if name in live else 0
        remaining = 0 if name in done else (seconds - elapsed if elapsed < seconds else None)
        rows.append(dict(job_id=name, group=key, completed=name in done, running=name in live,
            elapsed_seconds=elapsed, estimated_full_seconds=seconds, remaining_seconds=remaining,
            source="current_serial_repetitions" if measured[key] else "original_planning_estimate",
            completed_repetitions=len(measured[key]), actual_seconds=done.get(name)))
    unknown = [r["job_id"] for r in rows if r["remaining_seconds"] is None]
    known = sum(r["remaining_seconds"] or 0 for r in rows)
    started = (OUT / "closeout.start.json").exists()
    completion = read(OUT / "controller_finish.json") if (OUT / "controller_finish.json").exists() else None
    complete = bool(completion and completion["rc"] == 0 and len(done) == len(jobs) == 80 and not live)
    estimate = dict(utc=now.isoformat(), status="complete" if complete else "running" if started else "waiting_for_quality",
        jobs_expected=80, jobs_prepared=len(jobs), completed=len(done), active=len(live), workers=1,
        measured_process_hours=sum(done.values()) / 3600, remaining_known_hours=known / 3600,
        overdue_jobs=unknown, remaining_hours=None if unknown or len(jobs) != 80 else known / 3600,
        actual_finish_utc=completion["utc"] if complete else None,
        estimated_finish_utc=completion["utc"] if complete else (now + timedelta(seconds=known)).isoformat()
            if started and not unknown and len(jobs) == 80 else None,
        note="Includes all remaining J jobs and their validation backends; download/report time excluded. Historical parallel estimates remain until the same class has a completed serial repetition.",
        jobs=rows)
    destination = BASE / "evidence" / ("j_eta_" + now.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    write_json(destination, estimate)
    print(json.dumps({k: v for k, v in estimate.items() if k != "jobs"}))


if __name__ == "__main__":
    main()
