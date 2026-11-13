"""Count completed Post-F/G jobs against their exact submitted manifests."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from e_run import write_json
from fg_batch import identity, job_items


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(manifests, out, destination):
    jobs = {}
    sources = []
    for path in manifests:
        for job in job_items(path):
            if job["id"] in jobs:
                assert identity(jobs[job["id"]]) == identity(job), job["id"]
            jobs[job["id"]] = job
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    counts = collections.Counter()
    rows = []
    for name, job in sorted(jobs.items()):
        receipt = read(out / "receipts" / (name + ".json"))
        result_path = out / job["result"]
        result = read(result_path)
        assert receipt["identity"] == identity(job), name
        assert receipt["rc"] == 0 and receipt["result"] == job["result"], name
        assert result["job_id"] == name and result["status"] == "ok", name
        actual_spec = result["spec"] if "spec" in result else result
        assert all(actual_spec[k] == v for k, v in job["spec"].items()), name
        kind = job["kind"]
        role = ("timing_validation" if kind.startswith("speed_") else
                "historical_reuse" if "reused_from" in result else
                "same_batch_reuse" if "batch_alias_from" in result else "new_backend")
        calls = result.get("backend_calls", int(kind == "split_eval"))
        assert calls == int(role == "new_backend"), (name, role, calls)
        counts[role] += 1
        rows.append(dict(job_id=name, kind=kind, role=role, backend_calls=calls,
                         receipt_seconds=receipt["wall_seconds"],
                         peak_rss_gib=receipt["peak_rss_gb"],
                         result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                         identity=receipt["identity"]))
    evidence = dict(status="passed", jobs=len(jobs), counts=dict(counts),
                    backend_evaluations=sum(r["backend_calls"] for r in rows),
                    successful_process_hours=sum(r["receipt_seconds"] for r in rows)/3600,
                    max_job_peak_rss_gib=max(r["peak_rss_gib"] for r in rows),
                    manifests=sources, jobs_verified=rows)
    write_json(destination, evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ("manifests", "jobs_verified")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("e2e/post_fg/out/main"))
    parser.add_argument("--evidence", type=Path, default=Path("e2e/post_fg/evidence/completion_audit.json"))
    args = parser.parse_args()
    audit(args.manifest, args.out, args.evidence)
