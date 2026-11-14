"""Read-only I4 PF4 readiness and full-job estimates; never launches experiments."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import hi_fault
import hi_next
import hi_h3_i2
from e_run import ROOT, write_json

HERE=ROOT/"e2e/hi"
SOURCES=("h1_main","next_344","h3_i2_440","i3_108")


def prepare():
    target=HERE/"out/i4_pf4"
    planned=[]
    for hardware in ("busybarn","watos"):
        planned.extend(s for s in hi_fault.jobs("qwen3moe",hardware,target,"pf_ffn") if s.get("fault_id")!="MF6")
    needed={n for s in planned if s["kind"]=="fault_selection" for n in s["candidate_ids"]}
    assert len(planned)==264 and len(needed)==224
    master={s["candidate_id"]:s for s in hi_next.read(HERE/"manifest/all_candidates_planned.json")["jobs"]}
    records={}
    for folder in SOURCES:
        for p in (HERE/"out"/folder/"results").glob("*.json"):
            name=p.stem
            if name not in needed:continue
            r=hi_next.read(p)
            assert r["status"]=="ok" and all(r[k]==v for k,v in master[name].items())
            assert [x["seed"] for x in r["backend"]]==list(range(1,11))
            base=p.parent.parent
            paths=dict(record=p,raw=base/"raw"/(name+".pkl.gz"),snapshot=base/"snapshots"/(name+".pkl.gz"))
            if all(q.exists() for q in paths.values()):records[name]=dict(record=r,paths=paths)
    groups=defaultdict(list)
    for name in needed:groups[hi_h3_i2.key(master[name])].append(name)
    group_rows=[]
    cost={}
    for key,names in sorted(groups.items()):
        available=[records[n] for n in names if n in records]
        ready=len(available)==14
        selected=min(available,key=lambda x:(statistics.mean(s["cycles"] for s in x["record"]["backend"]),x["record"]["candidate_order"])) if available else None
        r=selected["record"] if selected else None
        if not selected:
            anchors=[records[n] for n in records if hi_h3_i2.key(master[n])==(*key[:-1],"A3h")]
            assert anchors,key
            selected=anchors[0];r=selected["record"]
        backend=sum(r["timing"]["evaluation"]["backend_individual_seconds"])
        search=float(r["search_seconds"])
        row=dict(model=key[0],hardware_profile=key[1],phase=key[2],op=key[3],method=key[4],
            complete_candidates=len(available),required=14,ready=ready,
            selected_candidate=r["candidate_id"] if ready else None,provisional_timing_anchor=None if ready else r["candidate_id"],
            backend_ten_seconds=backend,healthy_search_1000_seconds=search,
            raw_and_snapshot_present=True,missing=[n for n in sorted(names) if n not in records])
        group_rows.append(row);cost[key]=row
    jobs=[]
    for s in planned:
        kind=s["kind"]
        if kind in ("fault_select","fault_selection","fault_stop"):seconds=1.
        else:
            c=cost["qwen3moe",s["hardware_profile"],s["phase"],s["op"],s["start_arm"]]
            seconds=20+c["backend_ten_seconds"]*(2 if kind=="fault_l1" and s["start_arm"]=="A3h" else 1)
            if kind=="fault_rep":seconds+=c["healthy_search_1000_seconds"]*s["steps"]/1000
        jobs.append(dict(id=s["job_id"],spec=s,deps=[] if kind=="fault_selection" else s["depends_on"],
            estimated_seconds=seconds,peak_rss_gb=2.,priority=0))
    queues=[dict(slots=n,minutes=hi_h3_i2.queue_estimate(jobs,200,n)["queue_seconds"]/60) for n in (14,32,64,128)]
    evidence=dict(status="ready" if all(r["ready"] for r in group_rows) else "waiting_for_healthy_groups",
        inspected_utc=datetime.now(timezone.utc).isoformat(),scope="readiness and estimates only; no manifest submitted, no new experiments",
        healthy_candidates_present=len(records),healthy_candidates_required=224,
        complete_groups=sum(r["ready"] for r in group_rows),required_groups=16,
        groups=group_rows,maximum_jobs=264,control_jobs=52,healthy_replays=16,R0_L1_jobs=16,
        maximum_L2_trajectories=180,maximum_backend_samples=2240,
        estimated_upper_process_hours=sum(j["estimated_seconds"] for j in jobs)/3600,
        queue_estimates_all_L2_executed=queues,
        estimate_assumptions="Uses existing full backend and 1000-step search costs, actual backend-mean winner only when all fourteen candidates are present. Incomplete groups use existing same-operator timing anchors. R0/L1 or L2 early stopping not predicted; all 180 L2 trajectories charged. Cloud contention and fault-specific work are not yet measured.",
        provisional_groups=[{k:r[k] for k in ("hardware_profile","phase","op","method","provisional_timing_anchor")} for r in group_rows if not r["ready"]],
        source_results_sha256={v["paths"]["record"].relative_to(ROOT).as_posix():hashlib.sha256(v["paths"]["record"].read_bytes()).hexdigest() for v in records.values()})
    write_json(HERE/"evidence/i4_pf4_readiness.json",evidence)
    return evidence


if __name__=="__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    result=prepare()
    print(json.dumps({k:result[k] for k in ("status","healthy_candidates_present","healthy_candidates_required","complete_groups","required_groups","estimated_upper_process_hours","queue_estimates_all_L2_executed")},ensure_ascii=False,indent=2))
