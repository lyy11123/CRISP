"""Account for successful jobs, the separate policy check, and retained attempts."""
import hashlib
import json
from pathlib import Path

from e_run import ROOT, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    folder = ROOT / "e2e/post_fg"
    audit = read(folder / "evidence/completion_audit.json")
    initial = read(folder / "evidence/dstu_initial_batch_checks.json")
    policy = read(folder / "out/policy_validation/verification.json")
    assert audit["status"] == policy["status"] == "passed"
    assert all(policy["fields"].values()) and policy["issues"]["different_rows"] == 0
    attempt_root = folder / "out/main/attempts/20261113_before_fresh_baselines"
    failures = []
    for row in initial["failed_replays"]:
        path = attempt_root / "receipts" / (row["job"] + ".json")
        receipt = read(path)
        assert receipt["rc"] == row["rc"] == 1
        assert receipt["wall_seconds"] == row["wall_seconds"]
        failures.append(dict(job_id=row["job"], rc=receipt["rc"],
                             completed_backend_calls=1, wall_seconds=receipt["wall_seconds"],
                             receipt=str(path.relative_to(folder)),
                             receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             reason=row["reason"]))
    assert len(failures) == initial["failed_completed_backend_runs"]
    report = dict(status="passed", main_logical_jobs=audit["jobs"],
                  main_successful_backend_calls=audit["backend_evaluations"],
                  independent_policy_validation_backend_calls=policy["new_backend_calls"],
                  failed_comparison_completed_backend_calls=len(failures),
                  total_backend_invocations=audit["backend_evaluations"] + policy["new_backend_calls"] + len(failures),
                  main_successful_process_hours=audit["successful_process_hours"],
                  independent_policy_process_seconds=policy["new_wall_seconds"],
                  failed_comparison_process_seconds=sum(row["wall_seconds"] for row in failures),
                  measured_job_process_hours=audit["successful_process_hours"] +
                      (policy["new_wall_seconds"] + sum(row["wall_seconds"] for row in failures)) / 3600,
                  failed_attempts=failures,
                  counting="Backend simulations only; CPCM, graph construction, local checks and file analysis are separate. No historical reused simulation is counted as a new invocation.")
    write_json(folder / "evidence/final_execution_ledger.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "failed_attempts"}))


if __name__ == "__main__":
    main()
