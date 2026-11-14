"""Audit saved closeout jobs, raw samples and receipts; never run an experiment.

The quality inventory is A5 + B46 + C90 + S408. J timing has its own inventory
and totals. ``--progress`` permits missing outputs, not mismatched identities.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "e2e/hi/closeout"
OUT = ROOT / "e2e/hi/out/closeout_20261115"
JOUT = ROOT / "e2e/hi/out/closeout_j_20261115"
PACKAGES = {"A": ("manifests/a.json", 5), "B": ("manifests/b.json", 46),
            "C": ("manifests/c_mf6.json", 90), "S": ("s_manifest.json", 408)}
INPUTS = {"A": "manifests/a_inputs.json", "B": "manifests/b_inputs.json",
          "C": "manifests/c_inputs.json", "S": "s_inputs.json"}
JMANIFESTS = ("manifests/j_micro.json", "manifests/j_search.json", "manifests/j3.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def local(value):
    value = str(value).replace("\\", "/")
    if value.startswith("/home/ubuntu/") and "/e2e/" in value:
        value = "e2e/" + value.split("/e2e/", 1)[1]
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def identity(job):
    value = {k: job[k] for k in ("id", "kind", "spec", "result")}
    value["argv"] = job.get("argv")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def quality_jobs():
    jobs, sources = {}, []
    for package, (filename, expected) in PACKAGES.items():
        path = BASE / filename
        rows = read(path)["jobs"]
        assert len(rows) == expected, (package, len(rows), expected)
        sources.append(dict(package=package, path=relative(path), sha256=sha(path), jobs=len(rows)))
        for job in rows:
            assert job["id"] not in jobs, job["id"]
            jobs[job["id"]] = (package, job)
    assert len(jobs) == 549
    assert {d for _, j in jobs.values() for d in j.get("deps", [])} <= set(jobs)
    return jobs, sources


def source_inputs():
    declared, seen, inventory = [], {}, []
    for package, filename in INPUTS.items():
        path = BASE / filename
        value = read(path)
        if package == "A":
            entries = [(v, value["sha256"][k], None) for k, v in value["paths"].items()]
        elif package in ("B", "C"):
            entries = [(v[k], v["sha256"][k], None) for v in value["candidates"].values()
                       for k in ("record", "raw", "snapshot")]
        else:
            entries = [(r["path"], r["sha256"], r["bytes"]) for r in value["files"]]
        declared.append(dict(package=package, path=relative(path), sha256=sha(path), declarations=len(entries)))
        for filename, expected, size in entries:
            path = local(filename)
            key = relative(path)
            if key in seen:
                assert seen[key]["expected_sha256"] == expected, key
                seen[key]["packages"] = sorted(set(seen[key]["packages"] + [package]))
                continue
            actual = sha(path) if path.exists() else None
            row = dict(path=key, packages=[package], expected_sha256=expected, actual_sha256=actual,
                       bytes=path.stat().st_size if path.exists() else None,
                       matches=actual == expected and (size is None or path.stat().st_size == size))
            seen[key] = row
            inventory.append(row)
    return dict(status="passed" if all(r["matches"] for r in inventory) else "failed",
                declarations=declared, unique_files=len(inventory),
                total_bytes=sum(r["bytes"] or 0 for r in inventory), files=inventory)


def raw_read(path):
    # Register only the classes needed to deserialize the historical objects.
    # No platform, graph compilation, CPCM or backend is invoked here.
    import post_fg_split
    post_fg_split.setup()
    with gzip.open(local(path), "rb") as stream:
        return pickle.load(stream)


def backend_record(record):
    assert set(record["backend"]) == set(range(1, 11))
    assert len(record["backend_wall"]) == 10
    assert not record["fault_violations"]
    assert all(len(v) == 3 for v in record["backend"].values())
    return len(record["backend"]), sum(record["backend_wall"])


def costs(package, job, result, out):
    spec, kind = job["spec"], job["kind"]
    row = dict(actual_search_trajectories=0, search_proposals=0, calibration_proposals=0,
               accepted_proposals=0, actual_backend_calls=0, historical_backend_samples_reused=0,
               backend_call_seconds=0., calibration_seconds=0., search_seconds=0.,
               restore_best_seconds=0., reused=result["status"] == "reused",
               healthy_reused="reused_healthy_record" in result)
    if result["status"] == "reused":
        assert result.get("backend_calls", 0) == 0
        if kind in ("fault_rep", "s_rep"):
            assert result["previous_stop"] == spec["previous_stop"]
            stop = read(out / "results" / (spec["previous_stop"] + ".json"))
            assert stop["stop"] and result["export"] == stop["export"]
            row["reuse_source"] = spec["previous_stop"]
        elif kind in ("s_solo", "s_joint_reuse"):
            assert result["reuse_identity_checked"]
            row["reuse_source"] = result.get("reused_from_job") or result["reused_from"]
            source = (out / "results" / (result["reused_from_job"] + ".json") if "reused_from_job" in result
                      else local(result["reused_from"]))
            old = read(source)
            assert old["evaluation_identity"] == result["evaluation_identity"]
            assert old["initial_order_sha256"] == result["initial_order_sha256"]
            assert old["fresh_protocol"] == result["fresh_protocol"] == "fresh_transform_v1"
            row["reuse_source_sha256"] = sha(source)
            if kind == "s_joint_reuse":
                assert row["reuse_source_sha256"] == spec["reuse_sha256"] == result["reused_sha256"]
                assert old["spec"]["reference"] == result["resolved_simulation_spec"]["reference"]
            row["historical_backend_samples_reused"] = 1
        else:
            raise AssertionError((job["id"], "unexpected reuse kind", kind))
        return row
    if kind in ("fault_pre", "fault_l1", "fault_rep", "s_l1", "s_rep"):
        raw = raw_read(result["sink"])
        assert raw["backend_seeds"] == list(range(1, 11))
        row["raw_path"] = relative(local(result["sink"]))
        row["raw_sha256"] = sha(local(result["sink"]))
        entries = [raw["R0"], raw["L1"]] if kind in ("fault_l1", "s_l1") else [raw["record"]]
        measured = [backend_record(rec) for rec in entries]
        if row["healthy_reused"]:
            assert kind == "fault_pre" and result["actual_backend_calls"] == 0
            row["historical_backend_samples_reused"] = sum(n for n, _ in measured)
            row["reuse_source"] = result["reused_healthy_record"]
        else:
            row["actual_backend_calls"] = sum(n for n, _ in measured)
            row["backend_call_seconds"] = sum(seconds for _, seconds in measured)
        row["exports_sha256"] = {k: sha(local(v)) for k, v in result["exports"].items()}
        if kind in ("fault_rep", "s_rep"):
            search = raw["search"]
            assert [r["step"] for r in search["rows"]] == list(range(1, spec["steps"] + 1))
            assert raw["proposals"]["steps"] == spec["steps"]
            assert raw["spec"]["checks"] is False and search["check_revert"]["n"] == 0
            assert raw["T_L1_check"] == search["T_start"]
            row.update(actual_search_trajectories=1, search_proposals=len(search["rows"]),
                       calibration_proposals=raw["proposals"]["calibration"],
                       accepted_proposals=sum(r["accepted"] for r in search["rows"]),
                       calibration_seconds=search["t_calibration"], search_seconds=search["t_L2_search"],
                       restore_best_seconds=search["t_restore_best"])
            if package == "A":
                assert spec["steps"] == result["actual_search_proposals"] == 1000
                assert result["checkpoint_backend_calls"] == 0 and not result["original_stop300_inherited"]
                assert result["finish_reason"] == "independent_1000_step_budget_completed"
                assert result["source_input_sha256"] == read(BASE / INPUTS["A"])["sha256"]
                assert [r["step"] for r in search["checkpoints"]] == [30,100,300,500,700,1000]
                for checkpoint in result["checkpoints"]:
                    assert sha(local(checkpoint["snapshot"])) == checkpoint["snapshot_sha256"]
        if "backend_calls" in result:
            assert row["actual_backend_calls"] == result["backend_calls"]
    elif kind == "fault_select":
        row["actual_backend_calls"] = result["anchor_backend_calls"]
        assert row["actual_backend_calls"] == int(package == "C")
    elif kind == "c_split_eval":
        assert result["actual_backend_calls"] == 1 and result["seed"] == spec["seed"]
        assert len(result["backend_triple_ns"]) == 3
        row.update(actual_backend_calls=1, backend_call_seconds=result["backend_seconds"])
    elif kind in ("s_solo", "s_joint"):
        assert result["backend_calls"] == 1 and result["fresh_protocol"] == "fresh_transform_v1"
        assert result["evaluation_identity"] and result["initial_order_sha256"]
        row.update(actual_backend_calls=1, backend_call_seconds=result["simulation_seconds"])
    else:
        assert kind in ("fault_selection", "fault_stop", "c_representative", "c_split_prepare", "s_stop", "s_graph")
        assert result.get("backend_calls", result.get("actual_backend_calls", 0)) == 0
    return row


def collect(jobs, out):
    ledger, missing, errors = [], [], []
    for name, (package, job) in jobs.items():
        rp, cp, sp = out / job["result"], out / "receipts" / (name + ".json"), out / "specs" / (name + ".json")
        row = dict(package=package, job_id=name, kind=job["kind"], identity=identity(job),
                   model=job["spec"].get("model", "gpt"), hardware=job["spec"].get("hardware_profile"),
                   phase_op=job["spec"].get("phase_op"), steps=job["spec"].get("steps", 0),
                   seed=job["spec"].get("seed"), complete=False)
        if not (rp.exists() and cp.exists() and sp.exists()):
            row["state"] = "pending_files"
            row["missing_files"] = [relative(p) for p in (rp, cp, sp) if not p.exists()]
            missing.append(name)
        else:
            result, receipt = read(rp), read(cp)
            try:
                assert receipt["rc"] == 0 and receipt["identity"] == identity(job)
                assert receipt["id"] == name and receipt["result"] == job["result"]
                assert result["job_id"] == name and result["kind"] == job["kind"]
                assert result["status"] in ("ok", "reused")
                assert read(sp) == job["spec"]
                if "spec" in result:
                    assert result["spec"] == job["spec"]
                else:
                    assert result["status"] == "reused" and job["kind"] == "fault_rep"
                row.update(costs(package, job, result, out))
                row.update(complete=True, state=result["status"], receipt_seconds=receipt["wall_seconds"],
                           task_seconds=result["wall_seconds"], receipt_peak_rss_gib=receipt["peak_rss_gb"],
                           task_peak_rss_gib=result.get("peak_rss_gb", 0), result_sha256=sha(rp), receipt_sha256=sha(cp),
                           recovered_receipt=receipt.get("recovered_after_controller_restart", False))
            except (AssertionError, FileNotFoundError, KeyError) as exc:
                row.update(state="verification_failed", error=f"{type(exc).__name__}: {exc}")
                errors.append(dict(job_id=name, error=row["error"]))
        ledger.append(row)
    return ledger, missing, errors


def totals(rows):
    done = [r for r in rows if r["complete"]]
    fields = ("actual_search_trajectories", "search_proposals", "calibration_proposals", "accepted_proposals",
              "actual_backend_calls", "historical_backend_samples_reused", "backend_call_seconds")
    return dict(expected_jobs=len(rows), completed_jobs=len(done),
                **{k:sum(r[k] for r in done) for k in fields},
                reused_jobs=sum(r["reused"] for r in done), healthy_reused_jobs=sum(r["healthy_reused"] for r in done),
                reused_repair_jobs=sum(r["reused"] and r["kind"] in ("fault_rep", "s_rep") for r in done),
                receipt_process_hours=sum(r["receipt_seconds"] for r in done)/3600,
                task_process_hours=sum(r["task_seconds"] for r in done)/3600,
                maximum_job_seconds=max((r["receipt_seconds"] for r in done), default=0),
                peak_task_rss_gib=max((max(r["receipt_peak_rss_gib"], r["task_peak_rss_gib"]) for r in done), default=0),
                receipts_recovered_from_result=sum(r["recovered_receipt"] for r in done),
                kinds=dict(Counter(r["kind"] for r in done)))


def monitor(out, jobs, completed):
    records = []
    for filename in ("monitor_history.jsonl", "host_monitor.jsonl", "closeout.log"):
        path = out / filename
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("{"):
                    value = json.loads(line)
                    if "utc" in value:
                        records.append(value)
    for path in [out / "closeout.status.json", *out.glob("controller_restarts/*/closeout.status.json")]:
        if path.exists(): records.append(read(path))
    timestamps = sorted(r["utc"] for r in records if "utc" in r)
    starts = [read(p) for p in [out / "closeout.start.json", *out.glob("controller_restarts/*/closeout.start.json")] if p.exists()]
    first = min((r["utc"] for r in starts), default=timestamps[0] if timestamps else None)
    finished = sorted(r["utc"] for r in records if r.get("jobs") == 549 and r.get("completed") == 549)
    endpoint = finished[0] if finished else timestamps[-1] if timestamps else None
    elapsed = (datetime.fromisoformat(endpoint)-datetime.fromisoformat(first)).total_seconds() if endpoint and first else None
    failures = []
    for path in sorted(out.glob("failed_attempts/*/*/receipt.json")):
        r = read(path)
        assert r["id"] in jobs and r["identity"] == identity(jobs[r["id"]][1])
        log = local(r["log"]) if r.get("log") else None
        text = log.read_text(encoding="utf-8") if log and log.exists() else ""
        path_error = "ValueError:" in text and "is not in the subpath of" in text
        failures.append(dict(path=relative(path), sha256=sha(path), job_id=r["id"], rc=r["rc"],
            wall_seconds=r["wall_seconds"], memory_related=r.get("memory_related", False),
            input_path_prepare_error=path_error, recovered=r["id"] in completed,
            log=relative(log) if log and log.exists() else None,
            log_sha256=sha(log) if log and log.exists() else None))
    active = [max(r.get("active", 0), r.get("max_active", 0)) for r in records]
    available = [r[k] for r in records for k in ("available_gib", "min_available_gib") if k in r]
    maximum = max(active, default=None)
    minimum = min(available, default=None)
    return dict(first_start_utc=first, snapshot_utc=timestamps[-1] if timestamps else None,
        first_all_549_observed_utc=finished[0] if finished else None,
        elapsed_seconds=elapsed, elapsed_scope="to first complete observation" if finished else "to latest saved observation; still partial",
        controller_starts=starts, peak_active_workers=maximum, configured_worker_limit=128,
        worker_limit_observed=maximum is not None and maximum <= 128,
        minimum_available_gib=minimum, configured_reserve_gib=3,
        reserve_observed=minimum is not None and minimum >= 3, records=len(records),
        failed_attempts=failures, failed_attempt_count=len(failures),
        oom_attempt_count=sum(r["memory_related"] for r in failures),
        input_path_prepare_errors=sum(r["input_path_prepare_error"] for r in failures),
        recovered_failure_count=sum(r["recovered"] for r in failures),
        failed_attempt_process_hours=sum(r["wall_seconds"] for r in failures)/3600)


def package_reports(package_totals):
    definitions = {
        "A": ("evidence/a_report_audit.json", None,
              {"new_1000_step_searches":"actual_search_trajectories", "new_1000_backend_samples":"actual_backend_calls"}),
        "B": ("evidence/b_completion.json", None,
              {"completed_jobs":"completed_jobs", "actual_repair_trajectories":"actual_search_trajectories",
               "reused_repair_jobs":"reused_repair_jobs", "backend_samples":"actual_backend_calls"}),
        "C": ("evidence/c_completion.json", "summary",
              {"complete_jobs":"completed_jobs", "actual_search_trajectories":"actual_search_trajectories",
               "actual_backend_calls":"actual_backend_calls", "reused_healthy_jobs":"healthy_reused_jobs"}),
        "S": ("s_completion_check.json", None,
              {"jobs":"completed_jobs", "actual_L2":"actual_search_trajectories",
               "actual_search_proposals":"search_proposals", "actual_backend_calls":"actual_backend_calls"})}
    rows = []
    for package, (filename, child, fields) in definitions.items():
        path = BASE / filename
        if not path.exists():
            rows.append(dict(package=package, status="not_generated", path=relative(path)))
            continue
        report = read(path)
        report = report[child] if child else report
        checks = {source:dict(report=report[source], ledger=package_totals[package][target],
                              matches=report[source] == package_totals[package][target]) for source,target in fields.items()}
        done = package_totals[package]["completed_jobs"] == package_totals[package]["expected_jobs"]
        status = "passed" if all(r["matches"] for r in checks.values()) else "different_snapshot"
        rows.append(dict(package=package, path=relative(path), sha256=sha(path), status=status,
                         report_status=report["status"], quality_complete=done, checks=checks))
    return rows


def j_inventory(out):
    rows, manifests = [], []
    for filename in JMANIFESTS:
        path = BASE / filename
        jobs = read(path)["jobs"]
        manifests.append(dict(path=relative(path), sha256=sha(path), jobs=len(jobs)))
        for job in jobs:
            rp, cp = out / job["result"], out / "receipts" / (job["id"] + ".json")
            row = dict(job_id=job["id"], kind=job["kind"], complete=False)
            if rp.exists() and cp.exists():
                result, receipt = read(rp), read(cp)
                assert receipt["rc"] == 0 and receipt["identity"] == identity(job)
                assert result["status"] in ("ok", "reused", "complete", "completed")
                assert result["job_id"] == job["id"] and result["spec"] == job["spec"]
                row.update(complete=True, receipt_seconds=receipt["wall_seconds"],
                           result_sha256=sha(rp), receipt_sha256=sha(cp))
            rows.append(row)
    assert len({r["job_id"] for r in rows}) == len(rows)
    return dict(expected_final_jobs=80, manifest_jobs=len(rows), completed_jobs=sum(r["complete"] for r in rows),
        status="complete" if len(rows)==80 and all(r["complete"] for r in rows) else "pending",
        receipt_process_hours=sum(r.get("receipt_seconds", 0) for r in rows)/3600,
        excluded_from_quality_totals=True, manifests=manifests, jobs=rows)


def audit(out, jout, destination, progress):
    jobs, manifests = quality_jobs()
    inputs = source_inputs()
    ledger, missing, errors = collect(jobs, out)
    overall = totals(ledger)
    packages = {p:totals([r for r in ledger if r["package"]==p]) for p in PACKAGES}
    observations = monitor(out, jobs, {r["job_id"] for r in ledger if r["complete"]})
    checks = package_reports(packages)
    completed = overall["completed_jobs"] == 549
    report_ready = all(r["status"] == "passed" and r["report_status"] in ("passed", "complete") for r in checks)
    passed = completed and not errors and inputs["status"] == "passed" and report_ready
    value = dict(status="passed" if passed else "in_progress", generated_utc=datetime.now(timezone.utc).isoformat(),
        source=relative(Path(__file__)), source_sha256=sha(__file__), output=relative(out),
        read_only=True, newly_executed=dict(searches=0, backend_calls=0),
        totals=overall, packages=packages, manifests=manifests, missing_jobs=missing, errors=errors,
        source_inputs=inputs, monitoring=observations, package_report_crosschecks=checks, j_timing=j_inventory(jout),
        accounting_notes=["Only identity-matched successful receipts and results enter quality totals.",
            "Actual search trajectories count independent L2 trajectories; L1 candidates are not separate L2 searches.",
            "Search proposals exclude calibration proposals, reported separately.",
            "Inherited R0/L1 timings and reused historical backend samples are not new work.",
            "Receipt and task-body process hours are separate. Failed attempts are an additional cost.",
            "Recovered receipts may record task-body time because the original controller did not retain wrapper timing.",
            "Batch duration spans controller restarts and is based on saved UTC observations, not summed job duration.",
            "J timing is counted independently and never added to quality backend or search totals."])
    table(destination / "tables/quality_job_ledger.csv", ledger)
    table(destination / "tables/quality_package_ledger.csv", [dict(package=p, **v) for p,v in packages.items()])
    write_json(destination / "evidence/quality_completion_audit.json", value)
    assert not errors, errors
    assert inputs["status"] == "passed", "source input hash mismatch or missing file"
    if not progress:
        assert passed, {"completed":overall["completed_jobs"], "reports":checks}
        assert observations["worker_limit_observed"], observations["peak_active_workers"]
        assert all(r["recovered"] for r in observations["failed_attempts"])
    print(json.dumps(dict(status=value["status"], **overall,
                         package_completed={p:v["completed_jobs"] for p,v in packages.items()},
                         failed_attempts=observations["failed_attempt_count"], oom=observations["oom_attempt_count"],
                         J_manifest=value["j_timing"]["manifest_jobs"], J_complete=value["j_timing"]["completed_jobs"]), indent=2))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--j-out", type=Path, default=JOUT)
    parser.add_argument("--output", type=Path, default=BASE)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    audit(args.out.resolve(), args.j_out.resolve(), args.output.resolve(), args.progress)
