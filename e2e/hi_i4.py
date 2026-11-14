"""I4's ready PF4 operators: no prefill FFN or WATOS decode FFN."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import hi_h4 as shared
import hi_next as previous
import hi_h3_i2 as common
import hi_fault
import hi_config
import fg_fault as frozen
from e_run import ROOT,write_json

HERE=ROOT/"e2e/hi"
NAME="i4_light"
OUT=HERE/"out/i4_light"
MANIFEST=HERE/"manifest/i4_light.json"
INPUTS=HERE/"manifest/i4_light_healthy_inputs.json"
PARENTS=shared.PARENTS+((HERE/"manifest/h4_pf4.json",HERE/"out/h4_pf4","h4_pf4.status.json"),)
SOURCES=("h1_main","next_344","h3_i2_440","i3_108")


def bind_shared():
    # These are process-local entry bindings; all existing source files stay frozen.
    shared.NAME,shared.OUT,shared.MANIFEST,shared.INPUTS,shared.PARENTS=NAME,OUT,MANIFEST,INPUTS,PARENTS


def fixed_pf4(spec,out):
    source=frozen.fault_anchor_source(spec,out)
    _,_,snapshot,_,_=frozen.load_healthy(frozen.op_spec(spec,"dc_mha"),source,out)
    fault=hi_fault.physical_faults(snapshot[3],{},source["candidate_id"])["PF4"]
    fault.update(backend_seed=None,selection_backend_calls=0,rule="Fixed die0/core11 <-> die1/core8; no busy-time selection")
    result=dict(config_id=spec["config_id"],model="qwen3moe",faults={"PF4":fault},anchor=source,
        model_contract_sha256=spec["model_contract_sha256"],fault_selection_version="hi_i4_fixed_pf4_v1",
        anchor_backend_calls=0,physical_links_verified=True)
    frozen.write_json(Path(out)/"faults"/(spec["config_id"]+".json"),result)
    return result


def child(spec,out):
    bind_shared()
    assert spec["kind"] in hi_fault.KINDS and spec.get("fault_id") in (None,"PF4")
    assert spec["model"]=="qwen3moe" and spec["model_contract_sha256"]==hi_config.digest(hi_config.model_contract("qwen3moe"))
    assert spec["config_id"]==hi_config.identity(spec) and spec["backend_seeds"]==list(range(1,11))
    if spec["kind"]=="fault_rep":
        assert spec["steps"] in frozen.STEPS and spec["seed"] in frozen.SEEDS and spec["start_arm"]=="A3h"
    with shared.healthy_routes(),hi_fault.model_adapter(spec,out):
        original=frozen.select_faults;frozen.select_faults=fixed_pf4
        try:frozen.child(spec,out)
        finally:frozen.select_faults=original


def planned(out):
    specs=[]
    for hardware in ("busybarn","watos"):
        for s in hi_fault.jobs("qwen3moe",hardware,out,"pf_ffn"):
            if s.get("fault_id")=="MF6":continue
            phase_op=s.get("phase_op")
            if phase_op and ((hardware=="busybarn" and phase_op=="pf_ffn") or
                (hardware=="watos" and phase_op not in ("pf_mha","dc_mha"))):continue
            s.update(module="hi_i4",launch_status="authorized_I4_ready_PF4_subset")
            if s["kind"]=="fault_selection":
                allowed={(x["phase"],x["op"],x["method"]) for x in hi_fault.hi_manifest.candidates()
                    if x["model"]=="qwen3moe" and x["hardware_profile"]==hardware
                    and ((hardware=="busybarn" and (x["phase"]!="prefill" or x["op"]!="ffn")
                          and (x["method"]=="A3h" or (x["phase"]=="decode" and x["method"]=="A1")))
                         or (hardware=="watos" and x["op"]=="mha" and x["method"]=="A3h"))}
                lookup={x["candidate_id"]:x for x in hi_fault.hi_manifest.candidates()}
                s["candidate_ids"]=[n for n in s["candidate_ids"] if (lookup[n]["phase"],lookup[n]["op"],lookup[n]["method"]) in allowed]
                s["depends_on"]=[]
            specs.append(s)
    assert Counter(s["kind"] for s in specs)=={"fault_selection":2,"fault_select":2,"fault_pre":13,"fault_l1":13,"fault_stop":36,"fault_rep":135}
    return specs


def plan(out=OUT):
    specs=planned(out)
    needed={n for s in specs if s["kind"]=="fault_selection" for n in s["candidate_ids"]}
    assert len(needed)==182
    master={s["candidate_id"]:s for s in previous.read(HERE/"manifest/all_candidates_planned.json")["jobs"]}
    records,index={},{}
    for folder in SOURCES:
        base=HERE/"out"/folder
        for p in (base/"results").glob("*.json"):
            name=p.stem
            if name not in needed:continue
            r=previous.read(p)
            assert r["status"]=="ok" and all(r[k]==v for k,v in master[name].items())
            paths=dict(record=p,raw=base/"raw"/(name+".pkl.gz"),snapshot=base/"snapshots"/(name+".pkl.gz"))
            if not all(q.exists() for q in paths.values()):continue
            records[name]=r
            index[name]={k:q.relative_to(ROOT).as_posix() for k,q in paths.items()}
            index[name]["sha256"]={k:hashlib.sha256(q.read_bytes()).hexdigest() for k,q in paths.items()}
    missing=sorted(needed-set(records))
    if missing:
        result=dict(status="waiting_for_download_or_results",required=182,present=len(records),missing=missing)
        write_json(HERE/"evidence/i4_light_readiness.json",result)
        return result
    groups=defaultdict(list)
    for r in records.values():groups[common.key(r)].append(r)
    assert len(groups)==13 and all(len(rows)==14 for rows in groups.values())
    write_json(INPUTS,dict(status="passed",count=182,candidates=index,selection="backend_mean_then_candidate_order",new_healthy_searches=0))
    jobs=[]
    for s in specs:
        kind=s["kind"]
        if kind in ("fault_selection","fault_select","fault_stop"):seconds,rss=1.,.6
        else:
            rows=groups["qwen3moe",s["hardware_profile"],s["phase"],s["op"],s["start_arm"]]
            chosen=min(rows,key=lambda r:(statistics.mean(x["cycles"] for x in r["backend"]),r["candidate_order"]))
            backend=sum(chosen["timing"]["evaluation"]["backend_individual_seconds"])
            seconds=20+backend*(2 if kind=="fault_l1" and s["start_arm"]=="A3h" else 1)
            if kind=="fault_rep":seconds+=chosen["search_seconds"]*s["steps"]/1000
            rss=max(.6,2*chosen["peak_rss_gb"])
        jobs.append(dict(id=s["job_id"],kind=kind,spec=s,result=f"results/{s['job_id']}.json",deps=s["depends_on"],
            argv=["{python}","-u","-B","e2e/hi_i4.py","child","--spec","{spec}","--out","{out}"],
            estimated_seconds=seconds,peak_rss_gb=rss,priority=1 if kind in ("fault_selection","fault_select","fault_pre","fault_l1") else 0))
    assert {d for j in jobs for d in j["deps"]}<={j["id"] for j in jobs}
    write_json(MANIFEST,dict(status="authorized",jobs=jobs,new_healthy_searches=0,a3h_pf4_units=9,a1_r0_units=4,
        maximum_L2_trajectories=135,excluded=["all prefill FFN", "WATOS decode FFN", "MF6"]))
    queues=[dict(slots=n,minutes=common.queue_estimate(jobs,200,n)["queue_seconds"]/60) for n in (14,32,64,128)]
    result=dict(status="passed",jobs=201,counts=dict(Counter(s["kind"] for s in specs)),healthy_inputs=182,complete_groups=13,
        control_jobs=40,a3h_pf4_units=9,a1_r0_units=4,maximum_L2_trajectories=135,maximum_backend_samples=1700,
        estimated_upper_process_hours=sum(j["estimated_seconds"] for j in jobs)/3600,
        queue_estimates_all_L2_executed=queues,
        estimate_basis="Complete selected healthy candidate wall costs, all independent L2 tiers charged; actual old stop rule may skip later tiers",
        excluded=["all prefill FFN", "WATOS decode FFN", "MF6"],new_healthy_searches=0,
        combined_process_rule="new active plus all unfinished H3I2, I3 and H4 jobs <=128",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (MANIFEST,INPUTS,ROOT/"e2e/hi_i4.py")})
    write_json(HERE/"evidence/i4_light_plan.json",result)
    compile(shared.wrapped_source(),"<I4 shared-slot pool>","exec")
    return result


class Controller(shared.Controller):
    def __init__(self,out,workers):
        bind_shared();super().__init__(out,workers)
        self.parent_reserved_slots=sum(len(p["jobs"]) for p in self.parents)

    def reserve_parents(self):
        total,growth=0,0.
        for parent in self.parents:
            for name,job in parent["jobs"].items():
                if name in parent["success"]:continue
                path=parent["out"]/"receipts"/(name+".json")
                if not path.exists():continue
                receipt=previous.read(path)
                if receipt["rc"]!=0 or receipt["identity"]!=previous.pool.identity(job) or receipt["result"]!=job["result"]:continue
                result=previous.read(parent["out"]/job["result"])
                if job["kind"]=="search":
                    valid=result["status"]=="ok" and all(result[k]==v for k,v in job["spec"].items())
                else:
                    valid=result["job_id"]==name and (result["status"]=="ok" and result["spec"]==job["spec"] or
                        result["status"]=="reused" and job["kind"]=="fault_rep" and result["previous_stop"]==job["spec"]["previous_stop"])
                if valid:parent["success"].add(name)
            remaining=len(parent["jobs"])-len(parent["success"]);total+=remaining
            if remaining:growth+=float(previous.read(parent["out"]/parent["status"])["growth_reserve_gib"])
        self.parent_reserved_slots,self.parent_growth_reserve_gib=total,growth
        return total


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","check","child","launch","run"))
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--workers",type=int,default=128)
    args=parser.parse_args();args.out=args.out.resolve()
    assert 1<=args.workers<=128
    if args.command in ("plan","check"):
        result=plan(args.out);print(json.dumps(result,ensure_ascii=False,indent=2));return int(result["status"]!="passed")
    if args.command=="child":child(previous.read(args.spec),args.out);return 0
    bind_shared();args.out.mkdir(parents=True,exist_ok=True)
    if args.command=="launch":
        start=args.out/(NAME+".start.json")
        if start.exists() and previous.pool.process_alive(previous.read(start)["pid"]):raise RuntimeError("I4 controller already running")
        write_json(args.out/"scheduler_policy.json",dict(running_growth_fraction=.25))
        env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMBA_NUM_THREADS="1",CRISP_E_ARTIFACT=str(ROOT/"e2e/work/artifact"),PYTHONUTF8="1")
        command=[sys.executable,"-u","-B","e2e/hi_i4.py","run","--out",str(args.out),"--workers",str(args.workers)]
        options=dict(start_new_session=True) if sys.platform!="win32" else dict(creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        with (args.out/(NAME+".log")).open("ab") as stream:process=previous.RAW_POPEN(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,**options)
        write_json(start,dict(pid=process.pid,command=command,created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256={n:hashlib.sha256((ROOT/"e2e"/n).read_bytes()).hexdigest() for n in ("hi_i4.py","hi_h4.py","hi_fault.py","hi_next.py","fg_batch.py","fg_fault.py")}))
        print(json.dumps(dict(pid=process.pid,jobs=201,out=str(args.out))),flush=True);return 0
    return Controller(args.out,args.workers).run()


if __name__=="__main__":raise SystemExit(main())
