"""Compare early fault selection with saved full-selection records."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from e_run import write_json
from fg_batch import identity, job_items
import fg_fault
from fg_pipeline import advance_fault_dependencies


def check(manifest, results_out, report):
    jobs = job_items(manifest)
    by_id = {job["id"]: job for job in jobs}
    historical = []
    for path in sorted((results_out / "results").glob("*__selection.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        spec = saved["spec"]
        groups = defaultdict(list)
        for name in spec["candidate_ids"]:
            row = fg_fault.read_json(results_out / "results" / (name + ".json"))
            op = ("pf" if row["phase"] == "prefill" else "dc") + "_" + row["op"]
            groups[(op, row["method"])].append(row)
        old_out = Path(saved["operators"]["dc_mha"]["A3h"]["record"]).parent.parent
        for (op, method), rows in groups.items():
            assert fg_fault.choose_candidate(rows, old_out) == saved["operators"][op][method], (spec["config_id"], op, method)
        anchor = fg_fault.fault_anchor_source(spec, results_out)
        assert anchor["candidate_id"] == saved["operators"]["dc_mha"]["A3h"]["candidate_id"]
        historical.append(dict(config_id=spec["config_id"], matched_groups=len(groups),
                               matched_anchor=anchor["candidate_id"]))
    assert len(historical) == 4
    cid = "G_busybarn_3x3_growth_d2d1"
    updated, audit = advance_fault_dependencies(jobs, cid)
    target = by_id[cid + "__fault_select"]
    anchor_names = fg_fault.fault_anchor_candidates(target["spec"])
    assert all((results_out / "results" / (name + ".json")).exists() for name in anchor_names)
    assert not (results_out / "selection" / (cid + ".json")).exists()
    anchor = fg_fault.fault_anchor_source(target["spec"], results_out)
    assert anchor["candidate_id"] in anchor_names
    assert [identity(job) for job in jobs] == [identity(job) for job in updated]
    assert target["spec"]["depends_on"] == [cid + "__selection"]
    changed = [job["id"] for job, new in zip(jobs, updated) if job != new]
    assert changed == [target["id"]]
    new_target = next(job for job in updated if job["id"] == target["id"])
    assert set(new_target["deps"]) == set(anchor_names)
    full = [job for job in updated if job["spec"].get("config_id") == cid
            and job["spec"].get("family") == "full"]
    assert len(full) == 112 and all(job["deps"] == [target["id"]] for job in full)
    record = dict(status="passed", historical=historical,
                  matched_historical_groups=sum(row["matched_groups"] for row in historical),
                  target_anchor=anchor, target_candidates_present=len(anchor_names),
                  changed_jobs=changed, unchanged_identities=len(jobs), full_searches_reachable=112,
                  dependency_change=audit)
    write_json(report, record)
    print(json.dumps({k: record[k] for k in ("status", "matched_historical_groups",
          "target_candidates_present", "unchanged_identities", "full_searches_reachable")}, ensure_ascii=False))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    check(args.manifest, args.results_out, args.report)
