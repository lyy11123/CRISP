"""Check F/G candidate outputs and estimate the remaining search workload."""

import argparse
from collections import Counter, defaultdict
import heapq
import json
import math
from pathlib import Path
import statistics

from e_run import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def terminal_spec(spec):
    return spec | dict(candidate_id=spec["candidate_id"] + "_terminal", kind="terminal")


def expected(specs):
    result = {s["candidate_id"]: s for s in specs}
    result.update({s["candidate_id"] + "_terminal": terminal_spec(s)
                   for s in specs if s["method"] == "Gemini" and s["x"] == 123})
    return result


def check(specs, out, allow_incomplete=False):
    planned = expected(specs)
    records, issues, signatures = {}, [], defaultdict(set)
    for name, spec in planned.items():
        path = out / "results" / (name + ".json")
        if not path.exists():
            if not allow_incomplete:
                issues.append(dict(candidate_id=name, issue="missing result"))
            continue
        row = records[name] = read(path)
        errors = [key for key in spec if row.get(key) != spec[key]]
        if row.get("status") != "ok":
            errors.append("status")
        if sorted(v["seed"] for v in row["backend"]) != spec["backend_seeds"]:
            errors.append("backend seeds")
        if any(v["cycles"] <= 0 or v["cycles"] != v["triple"][0] for v in row["backend"]):
            errors.append("backend times")
        for field in ("wall_seconds", "evaluation_seconds", "peak_rss_gb", "T_hat", "T_hat_pre"):
            if not isinstance(row[field], (int, float)) or not math.isfinite(row[field]) or row[field] <= 0:
                errors.append(field)
        snapshot = out / "snapshots" / (name + ".pkl.gz")
        if not snapshot.is_file() or snapshot.stat().st_size == 0:
            errors.append("snapshot")
        if spec["kind"] == "terminal":
            if row["checks"]["from_search"] != name.removesuffix("_terminal"):
                errors.append("terminal source")
            if row["T_hat_rand"]:
                errors.append("terminal random ties")
        else:
            for field in ("snapshot_workload_same", "snapshot_T_pre_same"):
                if row["checks"].get(field) is not True:
                    errors.append(field)
            ties = row["T_hat_rand"]
            if len(ties) != 10 or row["T_hat_rand_mean"] != statistics.mean(ties):
                errors.append("random CPCM evaluations")
            if spec["method"] == "Gemini":
                meta = row["checks"]["gemini"]
                if (meta["anneal_seed"] != spec["x"] or meta["steps"] != spec["search_steps"]
                        or meta["rollback"] != "pre_move_snapshot"
                        or not meta["LP_flag"] or not meta["hops_only"] or meta["dijkstra_routing"]):
                    errors.append("Gemini protocol")
                if any(not v["adapter_restore_equal"] for v in meta["native_revert_audit"].values()):
                    errors.append("Gemini rollback")
            elif any(row["checks"]["revert"].get(k) != 0 for k in ("E_mismatch", "T_mismatch", "state_mismatch")):
                errors.append("A1/A3h rollback")
            key = (spec["shape"], spec["split_mode"], spec["phase"], spec["op"])
            signatures[key].add(row["workload_signature"]["sha256"])
        metrics = row["fg_metrics"]
        if metrics["cpcm_measured_ticks"] != row["T_hat"]:
            errors.append("repeated CPCM result")
        if metrics["graph_events"] != row["graph_events"]:
            errors.append("event count")
        if errors:
            issues.append(dict(candidate_id=name, issues=errors))
    for key, hashes in signatures.items():
        if len(hashes) != 1:
            issues.append(dict(workload=key, issue="different workload across hardware/method candidates"))
    return dict(expected_records=len(planned), completed_records=len(records),
                missing_records=len(planned) - len(records), primary_candidates=len(specs),
                backend_samples=sum(len(r["backend"]) for r in records.values()),
                backend_calls=sum(len(r["backend"]) for r in records.values() if "evaluation_reused_from" not in r),
                reused_backend_samples=sum(len(r["backend"]) for r in records.values() if "evaluation_reused_from" in r),
                random_tie_calls=sum(len(r["T_hat_rand"]) for r in records.values()),
                issues=issues), records


def timing_key(row):
    return tuple(row[k] for k in ("hardware_profile", "shape", "split_mode", "d2d_scale", "phase", "op", "method"))


def estimate(specs, records, workers):
    observed = defaultdict(list)
    for row in records.values():
        if row["kind"] != "terminal":
            observed[(row["family"], *timing_key(row))].append(row)
    categories = defaultdict(list)
    for spec in specs:
        categories[(spec["family"], *timing_key(spec))].append(spec)
    details, remaining_durations = [], []
    for key, pending in sorted(categories.items()):
        matches = observed[key]
        basis = "same configuration/phase/op/method pilot"
        if not matches and pending[0]["family"] == "full":
            matches = observed[("healthy", *key[1:])]
            basis = "same-configuration healthy pilot proxy; fault search unmeasured"
        if not matches and pending[0]["family"] == "bandwidth":
            sample_key = list(key[1:]); sample_key[3] = 1.0
            matches = observed[("healthy", *sample_key)]
            basis = "WATOS 1x pilot proxy; changed bandwidth unmeasured"
        count_remaining = sum(s["candidate_id"] not in records for s in pending)
        item = dict(category=list(key), planned=len(pending), remaining=count_remaining,
                    measured_candidates=len(matches))
        if matches:
            duration = statistics.mean(r["wall_seconds"] for r in matches)
            # seed123 Gemini child evaluates a terminal in addition to the primary.
            terminal = [records[r["candidate_id"] + "_terminal"]["wall_seconds"] for r in matches
                        if r["candidate_id"] + "_terminal" in records]
            item.update(mean_seconds=duration, peak_rss_gb=max(r["peak_rss_gb"] for r in matches),
                        timing_basis=basis, remaining_process_hours=count_remaining * duration / 3600)
            for spec in pending:
                if spec["candidate_id"] not in records:
                    extra = statistics.mean(terminal) if terminal and spec["method"] == "Gemini" and spec["x"] == 123 else 0
                    remaining_durations.append(duration + extra)
        else:
            item["timing_basis"] = "unmeasured"
        details.append(item)
    queue = [0.0] * workers
    for duration in sorted(remaining_durations, reverse=True):
        heapq.heapreplace(queue, queue[0] + duration)
    return dict(by_category=details, remaining_process_hours=sum(remaining_durations) / 3600,
                remaining_jobs_estimated=len(remaining_durations),
                remaining_jobs_unmeasured=sum(r["remaining"] for r in details if r["timing_basis"] == "unmeasured"),
                ideal_longest_first_hours=max(queue) / 3600,
                longest_estimated_job_minutes=max(remaining_durations, default=0) / 60,
                workers=workers,
                scope="Search jobs only; excludes repair, standalone references and joint simulations. Measured pilot times held constant; final scheduling adds dependencies and memory limits.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Run output directory")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--full-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    audit, records = check(read(args.manifest), args.out, args.allow_incomplete)
    result = dict(audit=audit)
    if args.full_manifest:
        result["search_projection"] = estimate(read(args.full_manifest), records, args.workers)
    write_json(args.report, result)
    print(json.dumps(audit, ensure_ascii=False))
    return bool(audit["issues"])


if __name__ == "__main__":
    raise SystemExit(main())
