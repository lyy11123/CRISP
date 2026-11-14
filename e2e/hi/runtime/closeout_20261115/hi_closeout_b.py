"""The two remaining Qwen prefill FFN PF4 chains, with frozen EIR rules."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import hi_fault
import hi_i4
import hi_h4
import hi_next
import hi_h3_i2
from e_run import ROOT, write_json

BASE = ROOT / "e2e/hi/closeout"
OUT = ROOT / "e2e/hi/out/closeout_20261115"
INPUTS = BASE / "manifests/b_inputs.json"
MANIFEST = BASE / "manifests/b.json"


def bind():
    hi_i4.INPUTS = INPUTS
    hi_i4.NAME = "closeout_b"
    hi_i4.OUT = OUT
    hi_i4.MANIFEST = MANIFEST


def plan(out=OUT):
    master = {r["candidate_id"]: r for r in hi_fault.hi_manifest.candidates()}
    specs = []
    for hardware in ("busybarn", "watos"):
        subset = [s for s in hi_fault.jobs("qwen3moe", hardware, out, "pf_ffn")
                  if s.get("phase_op") in (None, "pf_ffn") and s.get("fault_id") != "MF6"]
        names = {s["job_id"]: "B__" + s["job_id"] for s in subset}
        for spec in subset:
            spec.update(module="hi_closeout_b", launch_status="authorized_HI_CLOSEOUT_B")
            spec["job_id"] = names[spec["job_id"]]
            for field in ("previous_stop", "l1_job"):
                if field in spec:
                    spec[field] = names[spec[field]]
            if "rep_jobs" in spec:
                spec["rep_jobs"] = [names[n] for n in spec["rep_jobs"]]
            for field in ("pre_sink", "l1_sink"):
                if field in spec:
                    old = Path(spec[field])
                    spec[field] = (old.parent / ("B__" + old.name)).as_posix()
            if spec["kind"] == "fault_selection":
                spec["candidate_ids"] = [n for n in spec["candidate_ids"]
                    if master[n]["method"] == "A3h" and
                    (master[n]["phase"], master[n]["op"]) in (("decode", "mha"), ("prefill", "ffn"))]
                spec["depends_on"] = []
            else:
                spec["depends_on"] = [names[n] for n in spec["depends_on"]]
            specs.append(spec)
    assert Counter(s["kind"] for s in specs) == dict(fault_selection=2, fault_select=2,
        fault_pre=2, fault_l1=2, fault_stop=8, fault_rep=30)
    needed = {n for s in specs if s["kind"] == "fault_selection" for n in s["candidate_ids"]}
    assert len(needed) == 56
    index, records = {}, {}
    for folder in hi_i4.SOURCES:
        source = ROOT / "e2e/hi/out" / folder
        for p in (source / "results").glob("*.json"):
            if p.stem not in needed:
                continue
            r = hi_next.read(p)
            assert r["status"] == "ok" and all(r[k] == v for k, v in master[p.stem].items())
            paths = dict(record=p, raw=source/"raw"/(p.stem+".pkl.gz"),
                         snapshot=source/"snapshots"/(p.stem+".pkl.gz"))
            index[p.stem] = {k: v.relative_to(ROOT).as_posix() for k,v in paths.items()}
            index[p.stem]["sha256"] = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
            records[p.stem] = r
    assert set(index) == needed
    write_json(INPUTS, dict(status="passed", candidates=index, count=56,
        selection="backend_mean_then_candidate_order", new_healthy_searches=0))
    selected = {}
    for hw in ("busybarn", "watos"):
        rows = [r for r in records.values() if r["hardware_profile"] == hw and r["op"] == "ffn"]
        assert len(rows) == 14
        selected[hw] = min(rows, key=lambda r:(statistics.mean(v["cycles"] for v in r["backend"]),r["candidate_order"]))
    jobs = []
    for s in specs:
        r = selected[s["hardware_profile"]]
        backend = sum(r["timing"]["evaluation"]["backend_individual_seconds"])
        seconds = 1 if s["kind"] in ("fault_selection", "fault_select", "fault_stop") else 20+backend*(2 if s["kind"]=="fault_l1" else 1)
        if s["kind"] == "fault_rep":
            seconds += r["search_seconds"]*s["steps"]/1000
        jobs.append(dict(id=s["job_id"],kind=s["kind"],spec=s,deps=s["depends_on"],result=f"results/{s['job_id']}.json",
            argv=["{python}","-u","-B","e2e/hi_closeout_b.py","child","--spec","{spec}","--out","{out}"],
            estimated_seconds=seconds,peak_rss_gb=max(.6,2*r["peak_rss_gb"]),priority=10))
    assert {d for j in jobs for d in j["deps"]} <= {j["id"] for j in jobs}
    write_json(MANIFEST,dict(status="authorized",jobs=jobs,maximum_backend_samples=360,maximum_L2=30))
    estimate = hi_h3_i2.queue_estimate(jobs,240,128)
    report = dict(status="passed",jobs=46,healthy_inputs=56,maximum_backend_samples=360,maximum_L2=30,
        process_hours=sum(j["estimated_seconds"] for j in jobs)/3600,
        dependency_minutes_all_tiers=estimate["queue_seconds"]/60,
        selected={k:dict(candidate_id=r["candidate_id"],backend_ten_seconds=sum(r["timing"]["evaluation"]["backend_individual_seconds"])) for k,r in selected.items()})
    write_json(BASE/"evidence/b_plan.json",report)
    return report


def child(spec,out):
    bind()
    hi_i4.child(spec,out)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("plan","child"))
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--spec",type=Path)
    a=p.parse_args()
    if a.command=="plan":print(json.dumps(plan(a.out),indent=2))
    else:child(hi_next.read(a.spec),a.out)
