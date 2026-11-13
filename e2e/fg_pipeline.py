"""Start physical fault selection when its fourteen decode MHA candidates exist.

Only the runtime manifest's outer deps change. Specs, job identities, and the
full operator selection stay the same, so existing results remain reusable.
"""
import argparse
import copy
from pathlib import Path

from e_run import write_json
from fg_batch import identity, job_items
from fg_fault import fault_anchor_candidates


def advance_fault_dependencies(jobs, config_id):
    updated = copy.deepcopy(jobs)
    by_id = {job["id"]: job for job in updated}
    target = by_id[config_id + "__fault_select"]
    before = list(target["deps"])
    candidates = fault_anchor_candidates(target["spec"])
    assert len(candidates) == len(set(candidates)) == 14
    for name in candidates:
        job = by_id[name]
        spec = job["spec"]
        assert (job["kind"], spec["family"], spec["phase"], spec["op"], spec["method"]) == (
            "search", "healthy", "decode", "mha", "A3h")
        assert not job["deps"]
    target["deps"] = candidates
    assert all(identity(old) == identity(new) for old, new in zip(jobs, updated))
    assert all(old == new for old, new in zip(jobs, updated) if old["id"] != target["id"])
    audit = dict(job=target["id"], before_dependencies=before, after_dependencies=candidates,
                 jobs=len(updated), all_job_identities_unchanged=True,
                 note="Outer deps describe runtime scheduling; spec.depends_on retains the original task definition.")
    return updated, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", default="G_busybarn_3x3_growth_d2d1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    jobs, audit = advance_fault_dependencies(job_items(args.manifest), args.config)
    write_json(args.out, jobs)
    write_json(args.audit, audit)
    print(f"{audit['job']}: {len(audit['after_dependencies'])} candidate dependencies; "
          f"{audit['jobs']} unchanged job identities")


if __name__ == "__main__":
    main()
