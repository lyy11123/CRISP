"""Audit the collected E pilot and full result records without running simulations."""

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path, PurePosixPath
import re

from e_pilot_check import SEEDS, needs_terminal, read_json, terminal_spec
from e_run import HERE, ROOT, candidates, pilot_candidates, write_json


PATH_FIELDS = ("run", "snapshot")
REVERT_ERRORS = ("E_mismatch", "T_mismatch", "state_mismatch")
GEMINI_MOVES = {"ddr", "relocate", "swap"}


def normalized_path(value):
    return PurePosixPath(value.replace("\\", "/")).as_posix() if isinstance(value, str) else value


def config_differences(record, spec):
    differences = [key for key in spec if key not in PATH_FIELDS and record.get(key) != spec[key]]
    differences += [key for key in PATH_FIELDS
                    if normalized_path(record.get(key)) != normalized_path(spec.get(key))]
    return differences


def positive_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def bounded_integer(value, upper):
    return type(value) is int and 0 <= value <= upper


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def verified_recovery(directory, row, recovery, record):
    name = row["candidate_id"]
    failed_log = directory / "logs" / f"{name}.log"
    failure = {
        "candidate_id": name,
        "batch": directory.name,
        "original_rc": row["rc"],
        "original_wall_seconds": row.get("wall_seconds"),
        "original_log": str(failed_log),
        "recovered": False,
    }
    if recovery is None:
        return failure
    failure.update(error=recovery.get("original_error"), method=recovery.get("method"))
    recovery_spec = directory / "recovery_specs" / f"{name}.json"
    recovery_log = directory / "recovery_logs" / f"{name}.log"
    failure.update(recovery_log=str(recovery_log), recovery_spec=str(recovery_spec))
    if (recovery.get("original_batch_row") != row
            or recovery.get("verified") is not True or recovery.get("rc") != 0
            or not recovery.get("method") or not recovery.get("original_error")
            or normalized_path(recovery.get("recovery_spec")) != f"recovery_specs/{name}.json"
            or normalized_path(recovery.get("recovery_log")) != f"recovery_logs/{name}.log"
            or not nonempty(failed_log) or not nonempty(recovery_spec) or not recovery_log.is_file()
            or recovery["original_error"] not in failed_log.read_text(encoding="utf-8", errors="replace")):
        return failure
    spec = read_json(recovery_spec)
    failure["recovered"] = (
        record.get("status") == "ok"
        and spec.get("candidate_id") == name
        and spec.get("original_wall_seconds") == row["wall_seconds"] == record.get("original_wall_seconds")
        and not config_differences(record, spec)
    )
    return failure


def check_batch(directory, primary, records, issues):
    batch_path = directory / "batch.json"
    if not batch_path.is_file():
        issues.append({"batch": str(directory), "missing_batch_record": str(batch_path)})
        return []
    rows = read_json(batch_path)["runs"]
    row_counts = Counter(row["candidate_id"] for row in rows)
    if row_counts != Counter(primary.keys()):
        issues.append({"batch": str(directory), "batch_identity_mismatch": {
            "missing": sorted(primary.keys() - row_counts.keys()),
            "unexpected": sorted(row_counts.keys() - primary.keys()),
            "duplicate": sorted(name for name, n in row_counts.items() if n > 1),
        }})
    ledger_path = directory / "recovery.json"
    ledger = read_json(ledger_path) if ledger_path.is_file() else {}
    failed_ids = {row["candidate_id"] for row in rows if row["rc"] != 0}
    if ledger.keys() - failed_ids:
        issues.append({"batch": str(directory), "recovery_without_failed_batch_row": sorted(ledger.keys() - failed_ids)})
    failures = []
    for row in rows:
        if row["rc"] != 0:
            name = row["candidate_id"]
            failures.append(verified_recovery(directory, row, ledger.get(name), records.get(name, {})))
    return failures


def rollback_problems(record, spec):
    checks = record.get("checks", {})
    steps = spec["search_steps"]
    if spec["method"] != "Gemini":
        revert = checks.get("revert", {})
        if (not bounded_integer(checks.get("selected_step"), steps)
                or not bounded_integer(revert.get("n"), steps)
                or any(revert.get(key) != 0 for key in REVERT_ERRORS)):
            return ["A1/A3h rollback audit or selected step"]
        return []

    meta = checks.get("gemini", {})
    problems = []
    if (meta.get("initializer_seed") != 123 or meta.get("anneal_seed") != spec["x"] or meta.get("steps") != steps
            or meta.get("Tmax") != 10 or meta.get("Tmin") != 1e-6
            or meta.get("LP_flag") is not True or meta.get("hops_only") is not True
            or meta.get("dijkstra_routing") is not False or meta.get("rollback") != "pre_move_snapshot"
            or not bounded_integer(meta.get("best_step"), steps)
            or not positive_number(meta.get("best_energy")) or not positive_number(meta.get("terminal_energy"))
            or meta["best_energy"] > meta["terminal_energy"]):
        problems.append("Gemini annealing protocol")
    rejected = meta.get("rejections", {})
    audit = meta.get("native_revert_audit", {})
    if (set(rejected) != GEMINI_MOVES or not all(bounded_integer(n, steps) for n in rejected.values())
            or sum(rejected.values()) > steps or set(audit) != {move for move, n in rejected.items() if n > 0}
            or any(row.get("adapter_restore_equal") is not True for row in audit.values())):
        problems.append("Gemini exact rollback audit")
    return problems


def check_result(record, spec, directory, recovered, totals):
    name, kind = spec["candidate_id"], spec["kind"]
    problems = config_differences(record, spec)
    if record.get("status") != "ok":
        problems.append("status")
    backend = record.get("backend", [])
    if (sorted(item["seed"] for item in backend) != SEEDS
            or any(not positive_number(item["cycles"]) or len(item["triple"]) != 3
                   or item["cycles"] != item["triple"][0] for item in backend)):
        problems.append("backend evaluations")
    else:
        totals["backend_evaluations"] += len(backend)
    if (not positive_number(record.get("wall_seconds")) or not positive_number(record.get("evaluation_seconds"))
            or (record.get("peak_rss_gb") is not None and not positive_number(record["peak_rss_gb"]))
            or (not recovered and (record.get("peak_rss_gb") is None or "original_wall_seconds" in record))):
        problems.append("timing or recovery provenance")

    if not positive_number(record.get("T_hat")) or not positive_number(record.get("T_hat_pre")):
        problems.append("deterministic CPCM")
    else:
        totals[f"{kind}_deterministic_checks"] += 1
    ties = record.get("T_hat_rand", [])
    mean = record.get("T_hat_rand_mean")
    if kind != "terminal":
        if len(ties) != 10 or not all(positive_number(t) for t in ties) or mean != sum(ties) / 10:
            problems.append("random-tie CPCM samples or mean")
        else:
            totals["random_tie_evaluations"] += len(ties)
    elif ties or mean is not None:
        problems.append("unexpected terminal random-tie evaluations")

    checks = record.get("checks", {})
    if kind == "reeval":
        if not all(checks.get(key) is True for key in ("T_pre_same", "T_bc_same")):
            problems.append("historical snapshot CPCM reload")
        else:
            totals["historical_snapshot_reload_checks"] += 1
        for key in PATH_FIELDS:
            if not nonempty(ROOT / normalized_path(spec[key])):
                problems.append(f"historical {key} file")
    elif kind == "search":
        if not all(checks.get(key) is True for key in ("snapshot_workload_same", "snapshot_T_pre_same")):
            problems.append("search snapshot reload")
        else:
            totals["search_snapshot_reload_checks"] += 1
        rollback = rollback_problems(record, spec)
        problems.extend(rollback)
        if not rollback:
            totals[f"{spec['method']}_rollback_checks"] += 1
        if not nonempty(directory / "raw" / f"{name}.pkl.gz"):
            problems.append("saved raw result")
    if kind in ("search", "terminal") and not nonempty(directory / "snapshots" / f"{name}.pkl.gz"):
        problems.append("saved snapshot")
    if kind != "terminal":
        submitted = directory / "specs" / f"{name}.json"
        if not nonempty(submitted) or config_differences(read_json(submitted), spec):
            problems.append("submitted spec")

    signature = record.get("workload_signature", {})
    if (not re.fullmatch(r"[0-9a-f]{64}", signature.get("sha256", ""))
            or not positive_number(signature.get("behaviours"))
            or not positive_number(signature.get("tensors"))
            or signature.get("behaviours") != sum(signature.get("types", {}).values())):
        problems.append("workload signature")
    return problems


def audit(pilot_dir, full_dir=None, *, allow_missing=False):
    primary = {s["candidate_id"]: s for s in candidates()}
    pilot = {s["candidate_id"]: s for s in pilot_candidates()}
    full = {name: spec for name, spec in primary.items() if name not in pilot}
    expected = primary | {s["candidate_id"] + "_terminal": terminal_spec(s)
                          for s in primary.values() if needs_terminal(s)}
    planned = Counter(spec["kind"] for spec in expected.values())
    if planned != {"search": 420, "reeval": 252, "terminal": 16}:
        raise ValueError(f"Unexpected E canonical counts: {dict(planned)}")
    issues, records, origins, failures = [], {}, {}, []
    for directory, submitted in ((pilot_dir, pilot), (full_dir, full)):
        if directory is None:
            continue
        directory = directory.resolve()
        batch_records = {}
        names = set(submitted) | {s["candidate_id"] + "_terminal" for s in submitted.values() if needs_terminal(s)}
        for path in sorted((directory / "results").glob("*.json")):
            record = read_json(path)
            name = record.get("candidate_id")
            if name != path.stem or name not in names or name in records:
                issues.append({"invalid_or_duplicate_result": str(path), "candidate_id": name})
                continue
            records[name] = batch_records[name] = record
            origins[name] = directory
        failures.extend(check_batch(directory, submitted, batch_records, issues))

    recovered = {row["candidate_id"] for row in failures if row["recovered"]}
    unresolved = [row["candidate_id"] for row in failures if not row["recovered"]]
    if unresolved:
        issues.append({"unresolved_original_worker_failures": unresolved})

    totals = Counter()
    workloads = defaultdict(dict)
    for name, record in sorted(records.items()):
        spec = expected[name]
        problems = check_result(record, spec, origins[name], name in recovered, totals)
        if spec["kind"] == "terminal":
            source = name.removesuffix("_terminal")
            if (record.get("checks", {}).get("from_search") != source or source not in records
                    or origins[source] != origins[name]
                    or record.get("workload_signature") != records[source].get("workload_signature")):
                problems.append("terminal seed-123 source or workload")
            else:
                totals["terminal_workload_checks"] += 1
        else:
            group = tuple(spec[key] for key in ("shape", "phase", "op"))
            workloads[group][name] = record.get("workload_signature")
        if problems:
            issues.append({"candidate": name, "problems": problems})

    workload_groups = []
    for (shape, phase, op), candidates_in_group in sorted(workloads.items()):
        first = next(iter(candidates_in_group.values()))
        different = [name for name, signature in candidates_in_group.items() if signature != first]
        group = {"shape": shape, "phase": phase, "op": op, "present": len(candidates_in_group),
                 "planned": 42, "sha256": first.get("sha256") if first else None, "equal": not different}
        workload_groups.append(group)
        if different:
            issues.append({"workload_mismatch": group, "different_candidates": different})

    missing = sorted(expected.keys() - records.keys())
    if missing and not allow_missing:
        issues.append({"missing_results": len(missing)})
    return {
        "status": "invalid" if issues else "partial" if missing else "complete",
        "planned": dict(planned),
        "present": dict(Counter(record["kind"] for record in records.values())),
        "missing": dict(Counter(expected[name]["kind"] for name in missing)),
        "missing_ids": missing,
        "checks_passed": dict(totals),
        "workload_groups": workload_groups,
        "original_worker_failures": failures,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=HERE / "e/out/pilot_remote")
    parser.add_argument("--full-dir", type=Path, default=HERE / "e/out/full_remote")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true",
                        help="Allow missing full results during a pilot-only check")
    args = parser.parse_args()
    full = args.full_dir if args.full_dir.is_dir() else None
    result = audit(args.pilot_dir, full, allow_missing=args.allow_missing)
    write_json(args.out, result)
    print(f"{result['status']}: {sum(result['present'].values())}/{sum(result['planned'].values())} "
          f"results, {len(result['issues'])} issue groups; {args.out}")
    return int(bool(result["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())

