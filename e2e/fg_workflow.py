"""Combine the frozen F/G candidates with repair and joint dependencies."""
import argparse
from collections import Counter
import json
from pathlib import Path

from e_run import write_json
from fg_batch import job_items
import fg_fault
import fg_joint
import fg_manifest


def wrap(spec):
    name = spec["job_id"]
    return dict(id=name, kind=spec["kind"], spec=spec, deps=spec["depends_on"],
                result=f"results/{name}.json", argv=["{python}", "-u", "-B",
                f"e2e/{spec['module']}.py", "child", "--spec", "{spec}", "--out", "{out}"])


def closure(jobs, names):
    selected = set(names)
    todo = list(names)
    by_id = {j["id"]: j for j in jobs}
    while todo:
        for dependency in by_id[todo.pop()]["deps"]:
            if dependency not in selected:
                selected.add(dependency)
                todo.append(dependency)
    return [job for job in jobs if job["id"] in selected]


def build(directory, run_out):
    configs = fg_manifest.configurations()
    specs = fg_manifest.candidates(run_out)
    searches = fg_manifest.jobs(specs, run_out)
    repairs = [wrap(s) for s in fg_fault.jobs(configs, specs, run_out)]
    joint = [wrap(s) for s in fg_joint.jobs(configs, run_out)]
    jobs = searches + repairs + joint
    by_id = {j["id"]: j for j in jobs}
    assert len(by_id) == len(jobs)
    visited, pending = set(), dict(by_id)
    while pending:
        ready = [n for n,j in pending.items() if set(j["deps"]) <= visited]
        assert ready, list(pending)[:8]
        visited.update(ready)
        for n in ready:
            del pending[n]
    targets = []
    for cfg in configs:
        if cfg["stage"] == "G" and cfg["split_mode"] != "growth":
            continue
        cid = cfg["config_id"]
        targets.extend([f"{cid}__stop_pf_mha_PF4_300",
                        f"{cid}__joint_healthy_CRISP_g0.5_s1"])
    pilot_extended = closure(jobs, targets)
    independent = [j for j in searches if j["spec"]["family"] != "full"]
    write_json(directory / "all_jobs.json", jobs)
    write_json(directory / "independent_jobs.json", independent)
    write_json(directory / "repair_joint_pilot_jobs.json", pilot_extended)
    write_json(directory / "initial_release.json", dict(all=False, ids=sorted(
        {j["id"] for j in independent} | {j["id"] for j in pilot_extended})))
    summary = dict(jobs=len(jobs), by_kind=dict(Counter(j["kind"] for j in jobs)),
                   independent_search_jobs=len(independent),
                   repair_joint_pilot_targets=targets,
                   repair_joint_pilot_including_prerequisites=len(pilot_extended),
                   repair_stage_jobs_max=1920, search_jobs=3024, joint_jobs=609,
                   note="Repair jobs after an earlier stop publish a reused result. Pilot dependency closure reuses completed healthy candidates.")
    assert summary["by_kind"]["search"] == 3024
    assert summary["by_kind"]["joint"] == 609
    assert summary["by_kind"]["fault_rep"] == 1920
    write_json(directory / "workflow_counts.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("e2e/fg/manifest"))
    parser.add_argument("--run-out", default="e2e/fg/out/main")
    args = parser.parse_args()
    build(args.out, args.run_out)
