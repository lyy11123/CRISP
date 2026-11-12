"""Offline audit and timing projection for the frozen CRISP E cloud pilot."""

import argparse
from collections import Counter, defaultdict
import heapq
import itertools
import json
import math
from pathlib import Path
import statistics
import sys


SEEDS = list(range(1, 11))
TERMINAL = "_terminal"
FIELDS = ("kind", "shape", "phase", "op", "method")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def timing_class(record):
    return tuple(record[key] for key in FIELDS)


def describe(key):
    return dict(zip(FIELDS, key))


def terminal_spec(spec):
    return dict(spec, kind="terminal", candidate_id=spec["candidate_id"] + TERMINAL)


def needs_terminal(spec):
    return spec["method"] == "Gemini" and spec["x"] == 123


def audit(records, pilot, batch):
    expected = {s["candidate_id"]: s for s in pilot}
    expected.update({s["candidate_id"] + TERMINAL: terminal_spec(s) for s in pilot if needs_terminal(s)})
    issues = []
    missing = sorted(expected.keys() - records.keys())
    extra = sorted(records.keys() - expected.keys())
    if missing:
        issues.append({"missing_results": missing})
    if extra:
        issues.append({"unexpected_results": extra})
    batch_ids = [row["candidate_id"] for row in batch["runs"]]
    primary = {s["candidate_id"] for s in pilot}
    if set(batch_ids) != primary or len(batch_ids) != len(primary):
        issues.append({"batch_manifest_mismatch": {"missing": sorted(primary - set(batch_ids)), "extra": sorted(set(batch_ids) - primary)}})
    failed = [row for row in batch["runs"] if row["rc"] != 0]
    recovered = []
    unresolved = []
    for row in failed:
        name = row["candidate_id"]
        record = records.get(name, {})
        if record.get("status") == "ok" and record.get("original_wall_seconds") == row.get("wall_seconds"):
            recovered.append({"candidate_id": name, "original_rc": row["rc"]})
        else:
            unresolved.append(name)
    if unresolved:
        issues.append({"unresolved_worker_processes": unresolved})
    recovered_names = {row["candidate_id"] for row in recovered}

    signatures = defaultdict(dict)
    restoration = Counter()
    gemini_audits = Counter()
    for name in sorted(expected.keys() & records.keys()):
        r, spec = records[name], expected[name]
        problems = [key for key in (*FIELDS, "x", "candidate_order", "search_steps", "backend_seeds") if r.get(key) != spec[key]]
        if r.get("status") != "ok":
            problems.append("status")
        if sorted(item["seed"] for item in r["backend"]) != SEEDS or any(item["cycles"] <= 0 for item in r["backend"]):
            problems.append("backend seeds or latency")
        for key in ("wall_seconds", "evaluation_seconds"):
            if not math.isfinite(r[key]) or r[key] <= 0:
                problems.append(key)
        peak = r.get("peak_rss_gb")
        if (peak is None and name not in recovered_names) or (peak is not None and (not math.isfinite(peak) or peak <= 0)):
            problems.append("peak_rss_gb")
        checks = r["checks"]
        if spec["kind"] == "terminal":
            source = name.removesuffix(TERMINAL)
            if checks.get("from_search") != source or source not in records:
                problems.append("terminal source")
            elif r["workload_signature"] != records[source]["workload_signature"]:
                problems.append("terminal workload")
        else:
            key = r["shape"], r["phase"], r["op"]
            signatures[key][r["method"]] = r["workload_signature"]
            if len(r["T_hat_rand"]) != len(SEEDS):
                problems.append("CPCM random-tie samples")
            check_names = ("T_pre_same", "T_bc_same") if spec["kind"] == "reeval" else ("snapshot_workload_same", "snapshot_T_pre_same")
            if all(checks.get(key) is True for key in check_names):
                restoration[spec["kind"]] += 1
            else:
                problems.append(" / ".join(check_names))
            if r["method"] == "Gemini":
                meta = checks["gemini"]
                if meta["steps"] != spec["search_steps"] or meta["anneal_seed"] != spec["x"] or meta["rollback"] != "pre_move_snapshot":
                    problems.append("Gemini anneal protocol")
                for move, result in meta["native_revert_audit"].items():
                    gemini_audits[move] += 1
                    if result.get("adapter_restore_equal") is not True:
                        problems.append(f"Gemini {move} rollback")
            elif spec["kind"] == "search":
                revert = checks["revert"]
                if any(revert[key] for key in ("E_mismatch", "T_mismatch", "state_mismatch")):
                    problems.append("A1/A3h rollback")
        if problems:
            issues.append({"candidate": name, "issues": problems})

    comparisons = []
    for (shape, phase, op), methods in sorted(signatures.items()):
        for a, b in itertools.combinations(sorted(methods), 2):
            comparisons.append({"shape": shape, "phase": phase, "op": op, "methods": [a, b],
                                "equal": methods[a] == methods[b], "sha256": [methods[a]["sha256"], methods[b]["sha256"]]})
    differences = [row for row in comparisons if not row["equal"]]
    if differences:
        issues.append({"workload_signature_differences": differences})
    return {"expected": dict(Counter(s["kind"] for s in expected.values())),
            "present": dict(Counter(records[name]["kind"] for name in expected.keys() & records.keys())),
            "snapshot_checks_passed": dict(restoration), "gemini_adapter_audits_by_move": dict(gemini_audits),
            "workload_pairs": len(comparisons), "workload_pairs_equal": len(comparisons) - len(differences),
            "workload_comparisons": comparisons, "recovered_worker_processes": recovered, "issues": issues}


def schedule(times, workers):
    available = [0.0] * workers
    for elapsed in times:
        heapq.heapreplace(available, available[0] + elapsed)
    return max(available)


def projection(records, full, workers):
    observed = defaultdict(list)
    completed = set()
    for record in records.values():
        if record["status"] == "ok":
            observed[timing_class(record)].append(record)
            completed.add(record["candidate_id"])
    plan = Counter(timing_class(s) for s in full)
    plan.update(timing_class(terminal_spec(s)) for s in full if needs_terminal(s))
    classes = []
    estimates = {}
    for key, count in sorted(plan.items()):
        kind, shape, phase, op, method = key
        matches = observed[key]
        basis = "measured"
        if not matches and kind == "reeval" and shape == "2x2" and method == "A1":
            matches = observed[kind, shape, phase, op, "A3h"]
            basis = "same 2x2 phase/op A3h reevaluation proxy"
        row = describe(key) | {"planned": count, "observed": len(observed[key]), "remaining": count - len(observed[key])}
        if matches:
            estimates[key] = statistics.mean(r["wall_seconds"] for r in matches)
            peaks = [r["peak_rss_gb"] for r in matches if r.get("peak_rss_gb") is not None]
            row.update(timing_basis=basis, mean_seconds=estimates[key],
                       mean_search_seconds=statistics.mean(r["search_seconds"] for r in matches),
                       mean_evaluation_seconds=statistics.mean(r["evaluation_seconds"] for r in matches),
                       peak_rss_gb=max(peaks, default=None),
                       projected_total_process_hours=count * estimates[key] / 3600,
                       projected_remaining_process_hours=row["remaining"] * estimates[key] / 3600)
        else:
            row.update(timing_basis="unobserved")
        classes.append(row)

    buckets = {}
    for kind in ("search", "reeval", "terminal"):
        rows = [r for r in classes if r["kind"] == kind]
        buckets[kind] = {"planned": sum(r["planned"] for r in rows), "observed": sum(r["observed"] for r in rows),
                         "measured_classes": sum(r["timing_basis"] == "measured" for r in rows),
                         "proxy_jobs": sum(r["planned"] for r in rows if "proxy" in r["timing_basis"]),
                         "unknown_jobs": sum(r["planned"] for r in rows if r["timing_basis"] == "unobserved"),
                         "projected_total_process_hours": sum(r.get("projected_total_process_hours", 0) for r in rows),
                         "projected_remaining_process_hours": sum(r.get("projected_remaining_process_hours", 0) for r in rows)}

    projected = []
    if len(estimates) == len(plan):
        all_times, remaining_times = [], []
        for s in full:
            elapsed = estimates[timing_class(s)]
            name = s["candidate_id"]
            remaining = elapsed if name not in completed else 0
            if needs_terminal(s):
                terminal_time = estimates[timing_class(terminal_spec(s))]
                elapsed += terminal_time
                if name + TERMINAL not in completed:
                    remaining += terminal_time
            all_times.append(elapsed)
            if remaining:
                remaining_times.append(remaining)
        for n in sorted({32, 64, workers}):
            projected.append({"workers": n, "ideal_full_hours": schedule(all_times, n) / 3600,
                              "ideal_remaining_hours": schedule(remaining_times, n) / 3600})
    return {"by_class": classes, "buckets": buckets, "manifest_order_projection": projected,
            "assumption": "Per-class pilot wall time remains constant when worker count changes. List scheduling includes long-job tail but no saturation, contention, or I/O penalty."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path, help="Directory containing results/ and batch.json")
    parser.add_argument("--e2e", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--workers", type=int, default=128, help="Target parallelism for the timing model")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.e2e.resolve()))
    from e_run import candidates, pilot_candidates

    records = {}
    for path in sorted((args.pilot_dir / "results").glob("*.json")):
        row = read_json(path)
        name = row["candidate_id"]
        if name in records or name != path.stem:
            raise ValueError(f"Duplicate or incorrect result identity in {path}")
        records[name] = row
    batch = read_json(args.pilot_dir / "batch.json")
    verified = audit(records, pilot_candidates(), batch)
    measured = [r for r in records.values() if r["status"] == "ok"]
    peaks = [r for r in measured if r.get("peak_rss_gb") is not None]
    peak_record = max(peaks, key=lambda r: r["peak_rss_gb"]) if peaks else None
    worker_seconds = sum(r.get("wall_seconds", 0) for r in batch["runs"])
    result = {"audit": verified, "pilot_observed": {"workers": batch["workers"], "batch_wall_seconds": batch["wall_seconds"],
              "process_hours": sum(r["wall_seconds"] for r in measured) / 3600,
              "rss_records_measured": len(peaks), "rss_records_missing": len(measured) - len(peaks),
              "max_peak_rss_gib": peak_record["peak_rss_gb"] if peak_record else None,
              "worker_process_hours": worker_seconds / 3600,
              "mean_busy_worker_slots": worker_seconds / batch["wall_seconds"],
              "cached_worker_records": sum(r.get("cached", False) for r in batch["runs"]),
              "peak_record": peak_record["candidate_id"] if peak_record else None,
              "requested_workers_times_largest_measured_peak_rss_gib": args.workers * peak_record["peak_rss_gb"] if peak_record else None},
              "projection": projection(records, candidates(), args.workers)}
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return int(bool(verified["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
