"""Read live I3 backend progress; no search, backend call or scheduler mutation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "e2e/hi"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(job):
    payload = {k: job[k] for k in ("id", "kind", "spec", "result")}
    payload["argv"] = job.get("argv")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def group(spec):
    return tuple(spec[k] for k in ("model", "hardware_profile", "phase", "op", "method"))


def marks(out, name):
    logs = sorted((out / "logs").glob(name + ".*.log"), key=lambda p: p.stat().st_mtime)
    milestones = {}
    last_phase = None
    if logs:
        for line in logs[-1].read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith('{"candidate_id"'):
                continue
            item = json.loads(line)
            if item["candidate_id"] == name:
                last_phase = item["phase"]
                milestones[last_phase] = item["elapsed_seconds"]
    progress = out / "progress" / (name + ".json")
    if progress.exists():
        item = read(progress)
        milestones.update(item.get("milestones", {}))
        last_phase = item["phase"]
    return milestones, last_phase, str(logs[-1]) if logs else None


def estimate(out, manifest, reference_outs):
    now = datetime.now(timezone.utc)
    status_path = out / "i3.status.json"
    status = read(status_path)
    # Run this command on the server: mtime then represents the actual status write.
    status_age = max(0., time.time() - status_path.stat().st_mtime)
    jobs = read(manifest)["jobs"]
    by_id = {j["id"]: j for j in jobs}
    completed = set()
    for job in jobs:
        path = out / "receipts" / (job["id"] + ".json")
        if path.exists():
            receipt = read(path)
            if receipt["rc"] == 0 and receipt["identity"] == identity(job):
                result = read(out / job["result"])
                assert result["status"] == "ok" and all(result[k] == v for k, v in job["spec"].items())
                completed.add(job["id"])
    samples = defaultdict(list)
    for folder in dict.fromkeys([out] + reference_outs):
        for path in (folder / "results").glob("*.json"):
            item = read(path)
            if item.get("kind") != "search" or item.get("status") != "ok":
                continue
            if (item["model"], item["hardware_profile"], item["phase"], item["op"]) != ("qwen3moe", "watos", "prefill", "ffn"):
                continue
            timing = item["timing"]
            milestone = timing["milestones_seconds"]
            starts = [milestone[f"backend_seed_{s}_begin"] for s in range(1, 11)]
            intervals = [b - a for a, b in zip(starts, starts[1:])] + [milestone["random_cpcm_begin"] - starts[-1]]
            samples[group(item)].append(dict(candidate_id=item["candidate_id"], source=str(path),
                backend_intervals_seconds=intervals,
                tail_seconds=item["wall_seconds"] - milestone["random_cpcm_begin"]))
    rows = []
    for live in status["running"]:
        name = live["id"]
        if name in completed:
            continue
        spec = by_id[name]["spec"]
        milestone, phase, log = marks(out, name)
        elapsed = live["seconds"] + status_age
        same = samples[group(spec)]
        basis = "completed same model/hardware/operator/method"
        if not same:
            same = [sample for key, values in samples.items() if key[:-1] == group(spec)[:-1] for sample in values]
            basis = "completed same model/hardware/operator, other method; tail proxy"
        tail_values = [s["tail_seconds"] for s in same]
        tail = statistics.median(tail_values) if tail_values else None
        begins = sorted((int(match.group(1)), seconds) for label, seconds in milestone.items()
                        if (match := re.fullmatch(r"backend_seed_(\d+)_begin", label)))
        intervals = [b[1] - a[1] for a, b in zip(begins, begins[1:])]
        historical = [x for sample in same for x in sample["backend_intervals_seconds"]]
        duration_samples = intervals or historical
        item = dict(id=name, method=spec["method"], phase=phase, live_elapsed_seconds=elapsed,
                    search_complete_seconds=milestone.get("search_complete"),
                    status_reported_seconds=live["seconds"], log=log,
                    backend_started=len(begins), observed_backend_intervals_seconds=intervals,
                    completed_tail_samples=[dict(candidate_id=s["candidate_id"], tail_seconds=s["tail_seconds"], source=s["source"]) for s in same],
                    tail_basis=basis, tail_estimate_seconds=tail,
                    remaining_seconds=None, remaining_range_seconds=[None, None])
        if "random_cpcm_begin" in milestone and tail is not None:
            spent = elapsed - milestone["random_cpcm_begin"]
            item.update(remaining_seconds=max(0., tail - spent),
                        remaining_range_seconds=[max(0., min(tail_values) * .8 - spent), max(0., max(tail_values) * 1.2 - spent)],
                        estimation_stage="post-backend random CPCM, export and reload")
            if spent > max(tail_values) * 1.2:
                item.update(remaining_seconds=None, remaining_range_seconds=[0., None], overdue_tail=True)
        elif begins and duration_samples and tail is not None and "search_complete" in milestone:
            unit = statistics.median(intervals[-3:] or historical)
            low = min(statistics.median(duration_samples) * .9, unit * .8)
            high = max(statistics.median(duration_samples) * 1.1, unit * 1.2)
            seed, started = begins[-1]
            spent = elapsed - started
            remaining = max(0., unit - spent) + (10 - seed) * unit + tail
            lower = max(0., low - spent) + (10 - seed) * low + min(tail_values) * .8
            upper = max(0., high - spent) + (10 - seed) * high + max(tail_values) * 1.2
            item.update(remaining_seconds=remaining, remaining_range_seconds=[lower, upper],
                        active_backend_seed=seed, active_seed_elapsed_seconds=spent,
                        per_seed_seconds=unit, per_seed_range_seconds=[low, high],
                        estimation_stage="remaining independent backend seeds, ten random CPCM evaluations, export and reload")
            if spent > high:
                item.update(overdue_active_seed=True, remaining_seconds=None, remaining_range_seconds=[lower, None])
        rows.append(item)
    known = [r["remaining_seconds"] for r in rows if r["remaining_seconds"] is not None]
    unknown = [r["id"] for r in rows if r["remaining_seconds"] is None]
    pending = set(by_id) - completed - {r["id"] for r in rows}
    point = max(known, default=0.) if not unknown and not pending else None
    ranges = [max((r["remaining_range_seconds"][i] for r in rows if r["remaining_range_seconds"][i] is not None), default=0.) for i in (0, 1)]
    if unknown or pending:
        ranges[1] = None
    grouped = []
    for method in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method]
        values = [r["remaining_seconds"] for r in subset if r["remaining_seconds"] is not None]
        grouped.append(dict(method=method, running=len(subset), unknown=sum(r["remaining_seconds"] is None for r in subset),
                            remaining_minutes_range=[min(values) / 60, max(values) / 60] if values else None,
                            slowest_known_id=max((r for r in subset if r["remaining_seconds"] is not None), key=lambda r:r["remaining_seconds"])["id"] if values else None))
    return dict(updated_utc=now.isoformat(), source_status_age_seconds=status_age, completed=len(completed), expected=len(jobs),
                running=len(rows), pending_ids=sorted(pending), original_start_estimate_minutes=[99, 198],
                remaining_seconds=point, remaining_range_seconds=ranges,
                finish_utc=(now + timedelta(seconds=point)).isoformat() if point is not None else None,
                finish_range_utc=[(now + timedelta(seconds=v)).isoformat() if v is not None else None for v in ranges],
                unresolved_ids=unknown, method_groups=grouped, jobs=rows,
                assumptions=["Run on the live server; status mtime is the last status write, not a downloaded file timestamp.",
                             "All remaining searches must have completed 1000 steps; otherwise that job has no numerical estimate.",
                             "Each job uses its last three completed backend intervals including between-seed copy overhead; use same-class historical intervals only before any live interval completes.",
                             "Tail is measured complete job time after random_cpcm_begin, including ten random-CPCM calls and export/reload; other-method same-operator tail is explicitly marked when needed.",
                             "Range is a timing extrapolation (roughly 0.8-1.2 of measured seed/tail costs), not a statistical confidence interval; future contention or OOM retries are not predicted.",
                             "A seed or tail already exceeding the timing range remains unknown, rather than being reported as one second from completion.",
                             "Concurrent remaining jobs contribute the maximum remaining time; download/report time is excluded."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "out/i3_108")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest/i3_108.json")
    parser.add_argument("--output", type=Path, default=HERE / "evidence/i3_live_estimate.json")
    args = parser.parse_args()
    result = estimate(args.out.resolve(), args.manifest.resolve(), [HERE / "out/next_344"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ("jobs", "assumptions")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
