"""H4 PF4 repairs and four H5 A1-R0 inputs, using existing healthy states."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import hi_next as previous
import hi_h3_i2 as common
import hi_fault
import hi_config
import fg_fault as frozen
from e_run import ROOT, write_json

HERE=ROOT/"e2e/hi"
NAME="h4_pf4"
OUT=HERE/"out/h4_pf4"
MANIFEST=HERE/"manifest/h4_pf4.json"
INPUTS=HERE/"manifest/h4_healthy_inputs.json"
PARENTS=((HERE/"manifest/h3_i2_440.json",HERE/"out/h3_i2_440","h3_i2.status.json"),
         (HERE/"manifest/i3_108.json",HERE/"out/i3_108","i3.status.json"))


def assemble():
    jobs=previous.read(MANIFEST)["jobs"]
    return jobs,{j["id"] for j in jobs},[]


def input_index():
    return previous.read(INPUTS)["candidates"]


@contextmanager
def healthy_routes():
    index=input_index()
    original_result,original_raw,original_choose=frozen.result_path,frozen.raw_path,frozen.choose_candidate
    def result_path(out,name):
        return ROOT/index[name]["record"] if name in index else original_result(out,name)
    def raw_path(out,name):
        return ROOT/index[name]["raw"] if name in index else original_raw(out,name)
    def choose(rows,out):
        result=original_choose(rows,out)
        source=index[result["candidate_id"]]
        result.update(record=source["record"],raw=source["raw"],snapshot=source["snapshot"])
        return result
    frozen.result_path,frozen.raw_path,frozen.choose_candidate=result_path,raw_path,choose
    try:yield
    finally:frozen.result_path,frozen.raw_path,frozen.choose_candidate=original_result,original_raw,original_choose


def fixed_pf4(spec,out):
    source=frozen.fault_anchor_source(spec,out)
    _,_,snapshot,_,_=frozen.load_healthy(frozen.op_spec(spec,"dc_mha"),source,out)
    hp=snapshot[3]
    fault=hi_fault.physical_faults(hp,{},source["candidate_id"])["PF4"]
    fault.update(backend_seed=None,selection_backend_calls=0,rule="Fixed die0/core11 <-> die1/core8; no busy-time selection")
    result=dict(config_id=spec["config_id"],model="llama",faults={"PF4":fault},anchor=source,
        model_contract_sha256=spec["model_contract_sha256"],fault_selection_version="hi_h4_fixed_pf4_v1",
        anchor_backend_calls=0,physical_links_verified=True)
    frozen.write_json(Path(out)/"faults"/(spec["config_id"]+".json"),result)
    return result


def child(spec,out):
    assert spec["kind"] in hi_fault.KINDS and spec.get("fault_id") in (None,"PF4")
    assert spec["model"]=="llama" and spec["model_contract_sha256"]==hi_config.digest(hi_config.model_contract("llama"))
    assert spec["config_id"]==hi_config.identity(spec) and spec["backend_seeds"]==list(range(1,11))
    if spec["kind"]=="fault_rep":
        assert spec["steps"] in frozen.STEPS and spec["seed"] in frozen.SEEDS and spec["start_arm"]=="A3h"
    with healthy_routes(),hi_fault.model_adapter(spec,out):
        original=frozen.select_faults
        frozen.select_faults=fixed_pf4
        try:frozen.child(spec,out)
        finally:frozen.select_faults=original


def plan(out=OUT):
    candidates={r["candidate_id"]:r for r in common.read_csv(HERE/"tables/h3_i2_candidates.csv")}
    jobspecs=[]
    for hw in ("busybarn","watos"):
        for spec in hi_fault.jobs("llama",hw,out,"pf_ffn"):
            if spec.get("fault_id")=="MF6":continue
            spec.update(module="hi_h4",launch_status="authorized_H4_PF4_and_H5_A1_R0")
            if spec["kind"]=="fault_selection":spec["depends_on"]=[]
            jobspecs.append(spec)
    counts=Counter(s["kind"] for s in jobspecs)
    assert counts=={"fault_selection":2,"fault_select":2,"fault_pre":16,"fault_l1":16,"fault_stop":48,"fault_rep":180}
    wanted={name for s in jobspecs if s["kind"]=="fault_selection" for name in s["candidate_ids"]}
    assert len(wanted)==224 and wanted<=set(candidates)
    master={s["candidate_id"]:s for s in previous.read(HERE/"manifest/all_candidates_planned.json")["jobs"]}
    index={}
    for name in sorted(wanted):
        record=ROOT/Path(candidates[name]["source"].replace("\\","/"))
        base=record.parent.parent
        paths={"record":record,"raw":base/"raw"/(name+".pkl.gz"),"snapshot":base/"snapshots"/(name+".pkl.gz")}
        saved=previous.read(record)
        assert saved["status"]=="ok" and all(saved[k]==v for k,v in master[name].items())
        assert all(p.exists() for p in paths.values())
        index[name]={k:p.relative_to(ROOT).as_posix() for k,p in paths.items()}
        index[name]["sha256"]={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
        index[name]["backend_mean_ns"]=statistics.mean(x["cycles"] for x in saved["backend"])
        index[name]["healthy_T_hat"]=saved["T_hat"]
    write_json(INPUTS,dict(status="passed",candidates=index,count=224,selection="backend_mean_then_candidate_order",new_healthy_searches=0))
    jobs=[]
    # Initial ordering estimates use the same selected healthy backend cost and measured graph RSS.
    bygroup=defaultdict(list)
    for name in wanted:
        bygroup[common.key(candidates[name])].append(candidates[name])
    for spec in jobspecs:
        kind=spec["kind"]
        if kind in ("fault_selection","fault_select","fault_stop"):
            seconds,rss=1.,.6
        else:
            method=spec["start_arm"]
            group=bygroup["llama",spec["hardware_profile"],spec["phase"],spec["op"],method]
            chosen=min(group,key=lambda r:(float(r["backend_mean_ns"]),int(r["candidate_order"])))
            record=previous.read(ROOT/index[chosen["candidate_id"]]["record"])
            backend=sum(record["timing"]["evaluation"]["backend_individual_seconds"])
            search=float(record["search_seconds"])
            seconds=backend*(2 if kind=="fault_l1" and method=="A3h" else 1)+20
            if kind=="fault_rep":seconds+=search*spec["steps"]/1000
            rss=max(.6,2*float(chosen["peak_rss_gib"]))
        name=spec["job_id"]
        jobs.append(dict(id=name,kind=kind,spec=spec,result=f"results/{name}.json",deps=spec["depends_on"],
            argv=["{python}","-u","-B","e2e/hi_h4.py","child","--spec","{spec}","--out","{out}"],
            estimated_seconds=seconds,peak_rss_gb=rss,priority=1 if kind in ("fault_selection","fault_select","fault_pre","fault_l1") else 0))
    assert {d for j in jobs for d in j["deps"]}<={j["id"] for j in jobs}
    write_json(MANIFEST,dict(status="authorized",jobs=jobs,counts=dict(counts),new_healthy_searches=0,
        a3h_pf4_units=12,a1_r0_units=4,maximum_L2_trajectories=180,
        selection="existing fourteen-candidate backend-mean selection, with random-CPCM choice recorded separately"))
    evidence=dict(status="passed",jobs=len(jobs),counts=dict(counts),healthy_inputs=224,
        a3h_pf4_units=12,a1_r0_units=4,maximum_L2_trajectories=180,maximum_new_backend_calls=2240,
        estimated_upper_process_hours=sum(j["estimated_seconds"] for j in jobs)/3600,
        estimate_basis="selected healthy backend wall cost; L2 search scales by 30/100/300 of measured 1000-step healthy search; fault-specific costs await real pilot",
        stop_rule_sha256=hashlib.sha256(common.previous.inspect.getsource(frozen.stop_job).encode()).hexdigest(),
        models=["llama"],faults=["PF4"],new_healthy_searches=0,new_MF6_backend_calls=0,
        combined_process_rule="new active + all unfinished H3I2 + all unfinished I3 <= 128",
        independent_budgets=[30,100,300],repair_seeds=list(frozen.SEEDS),backend_seeds=list(range(1,11)))
    write_json(HERE/"evidence/h4_pf4_plan.json",evidence)
    compile(wrapped_source(),"<H4 shared-slot existing pool>","exec")
    return evidence


def wrapped_source():
    source=previous.wrapped_pool_source()
    for old,new in {
        "        external_slots = 0\n":"        external_slots = _hi.reserve_parents()\n",
        "        budget = free - reserve_gib - reserved_growth\n":"        budget = free - reserve_gib - reserved_growth - _hi.parent_growth_reserve_gib\n"}.items():
        assert source.count(old)==1;source=source.replace(old,new)
    return source


class Controller(previous.Controller):
    def __init__(self,out,workers):
        previous.assemble=assemble
        super().__init__(out,workers)
        self.manifest=MANIFEST
        self.parents=[dict(out=o,status=s,jobs={j["id"]:j for j in previous.RAW_JOB_ITEMS(m)},success=set()) for m,o,s in PARENTS]
        self.parent_reserved_slots=548
        self.parent_growth_reserve_gib=0.

    def spawn(self,argv,**kwargs):
        spec=previous.read(argv[argv.index("--spec")+1]);name=spec["job_id"]
        assert name in self.by_id and spec==self.by_id[name]["spec"]
        process=previous.RAW_POPEN(argv,**kwargs)
        self.live[name]=dict(pid=process.pid,started_unix=time.time(),identity=previous.pool.identity(self.by_id[name]),log=str(kwargs["stdout"].name))
        write_json(self.out/"live_workers.json",self.live)
        return process

    def recover(self):
        while self.live:
            for name,row in list(self.live.items()):
                assert row["identity"]==previous.pool.identity(self.by_id[name])
                if previous.pool.process_alive(row["pid"]):continue
                job=self.by_id[name];path=self.out/job["result"]
                if path.exists():
                    result=previous.read(path)
                    if result["status"] in ("ok","reused") and result["job_id"]==name:
                        receipt=dict(id=name,identity=previous.pool.identity(job),category=previous.category(job),rc=0,
                            wall_seconds=result["wall_seconds"],peak_rss_gb=result.get("peak_rss_gb",0),result=job["result"],cached=False,
                            recovered_after_controller_restart=True)
                        self.success_receipt(receipt)
                        write_json(self.out/"receipts"/(name+".json"),receipt)
                del self.live[name]
            write_json(self.out/"live_workers.json",self.live)
            if self.live:time.sleep(2)

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
                if result["status"]=="ok" and all(result[k]==v for k,v in job["spec"].items()):parent["success"].add(name)
            remaining=len(parent["jobs"])-len(parent["success"])
            total+=remaining
            if remaining:
                status=previous.read(parent["out"]/parent["status"])
                growth+=float(status["growth_reserve_gib"])
        self.parent_reserved_slots,self.parent_growth_reserve_gib=total,growth
        return total

    def success_receipt(self,receipt):
        name=receipt["id"];job=self.by_id[name]
        assert receipt["identity"]==previous.pool.identity(job) and receipt["result"]==job["result"]
        result=previous.read(self.out/job["result"])
        assert result["job_id"]==name and result["status"] in ("ok","reused")
        if result["status"]=="ok":assert result["spec"]==job["spec"]
        else:
            stop=previous.read(frozen.result_path(self.out,job["spec"]["previous_stop"]))
            assert job["kind"]=="fault_rep" and stop["stop"] and result["export"]==stop["export"]
        if name in self.success:return
        self.success[name]=receipt
        if result["status"]=="ok":
            key=previous.category(job);self.observed[key].append(receipt)
            seconds=statistics.median(r["wall_seconds"] for r in self.observed[key])
            rss=1.25*max(r["peak_rss_gb"] for r in self.observed[key])
            for other in self.jobs:
                if previous.category(other)==key and other["spec"].get("steps")==job["spec"].get("steps"):
                    other["estimated_seconds"]=seconds;other["peak_rss_gb"]=max(other["peak_rss_gb"],rss)
        self.save_release()

    def write(self,path,value):
        path=Path(path)
        if path.name==NAME+".status.json":
            capacity=max(0,self.workers-self.parent_reserved_slots)
            value.update(parent_reserved_slots=self.parent_reserved_slots,allowed_new_active=capacity,
                parent_growth_reserve_gib=self.parent_growth_reserve_gib,
                parent_successes=[dict(out=str(p["out"]),completed=len(p["success"]),expected=len(p["jobs"])) for p in self.parents])
            assert value["active"]<=capacity
            value["failed_attempt_count"]=len(list((self.out/"failed_attempts").glob("*/*/receipt.json")))
            write_json(path,value)
        else:super().write(path,value)

    def run(self):
        self.out.mkdir(parents=True,exist_ok=True);(self.out/"receipts").mkdir(exist_ok=True)
        self.recover()
        for path in self.receipt_paths():
            receipt=previous.read(path)
            if receipt["rc"]==0:self.success_receipt(receipt)
        self.save_release()
        pool=previous.pool
        pool.category,pool.memory_key=previous.category,previous.memory_key
        pool.memory_requirement=lambda job,*args:max(previous.RAW_MEMORY_REQUIREMENT(job,*args),job["peak_rss_gb"])
        pool.job_items=lambda path:self.jobs;pool.write_json=self.write
        pool.subprocess=types.SimpleNamespace(Popen=self.spawn,STDOUT=subprocess.STDOUT)
        namespace=dict(vars(pool),_hi=self)
        exec(compile(wrapped_source(),"<H4 shared-slot existing pool>","exec"),namespace)
        rc=namespace["execute"](self.manifest,self.out,self.workers,3,NAME,4,self.release)
        write_json(self.out/"controller_finish.json",dict(rc=rc,completed=len(self.success),expected=len(self.jobs),utc=datetime.now(timezone.utc).isoformat()))
        return rc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","check","child","launch","run"))
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--workers",type=int,default=128)
    args=parser.parse_args();args.out=args.out.resolve()
    assert 1<=args.workers<=128
    if args.command in ("plan","check"):
        print(json.dumps(plan(args.out),ensure_ascii=False,indent=2));return 0
    if args.command=="child":child(previous.read(args.spec),args.out);return 0
    args.out.mkdir(parents=True,exist_ok=True)
    if args.command=="launch":
        start=args.out/(NAME+".start.json")
        if start.exists() and previous.pool.process_alive(previous.read(start)["pid"]):raise RuntimeError("H4 controller already running")
        write_json(args.out/"scheduler_policy.json",dict(running_growth_fraction=.25))
        env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMBA_NUM_THREADS="1",CRISP_E_ARTIFACT=str(ROOT/"e2e/work/artifact"),PYTHONUTF8="1")
        command=[sys.executable,"-u","-B","e2e/hi_h4.py","run","--out",str(args.out),"--workers",str(args.workers)]
        options=dict(start_new_session=True) if sys.platform!="win32" else dict(creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        with (args.out/(NAME+".log")).open("ab") as stream:process=previous.RAW_POPEN(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,**options)
        write_json(start,dict(pid=process.pid,command=command,created_utc=datetime.now(timezone.utc).isoformat(),source_sha256={n:hashlib.sha256((ROOT/"e2e"/n).read_bytes()).hexdigest() for n in ("hi_h4.py","hi_fault.py","hi_next.py","fg_batch.py","fg_fault.py")}))
        print(json.dumps(dict(pid=process.pid,jobs=264,out=str(args.out))),flush=True);return 0
    return Controller(args.out,args.workers).run()


if __name__=="__main__":raise SystemExit(main())
