"""Llama H5: existing P/D joint backends over formal healthy and PF4 states."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import time

for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[k]="1"
import hi_config as H
import hi_joint as model_joint
from hi_run import execution_order_digest
import fg_joint as frozen
import fg_fault
import hi_i4_dffn as prior
import hi_next as previous
import hi_h3_i2 as common
from e_run import ROOT,write_json

HERE=ROOT/"e2e/hi"
NAME="h5"
OUT=HERE/"out/h5"
MANIFEST=HERE/"manifest/h5.json"
PILOT=HERE/"manifest/h5_pilot.json"
INPUTS=HERE/"manifest/h5_inputs.json"
H4=HERE/"out/h4_pf4"
PARENTS=prior.PARENTS+((prior.MANIFEST,prior.OUT,"i4_watos_dffn.status.json"),)
CID=H.identity(dict(model="llama",hardware_profile="busybarn",shape="2x2",split_mode="fixed",d2d_scale=1))
PHASES=(.1,.5,.9)
SEEDS=(1,2,3)


def bind(manifest=MANIFEST):
    prior.base.NAME,prior.base.OUT,prior.base.MANIFEST,prior.base.PARENTS=NAME,OUT,manifest,PARENTS
    prior.base.bind_shared()


def input_record():return previous.read(INPUTS)


def stream_signature(events):
    # Include actual event insertion order, all fields, and stable set ordering.
    def canonical(value):
        if isinstance(value,dict):return [(canonical(k),canonical(v)) for k,v in value.items()]
        if isinstance(value,set):return sorted((canonical(v) for v in value),key=repr)
        if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
        if hasattr(value,"__dict__"):return (type(value).__module__,type(value).__qualname__,canonical(vars(value)))
        if hasattr(value,"tolist"):return canonical(value.tolist())
        if hasattr(value,"item"):return value.item()
        return value
    rows=[(tag,canonical(vars(event))) for tag,event in events.items()]
    return hashlib.sha256(repr(rows).encode()).hexdigest()


@contextmanager
def adapter(spec,out):
    import f_config
    original_prepare,original_resolve,original_kv,original_read,original_chain=f_config.prepare,frozen.resolve_export,frozen.build_kv,frozen.read_json,frozen.CHAIN
    f_config.prepare=H.prepare
    frozen.CHAIN=model_joint.decode_chain_order("llama")
    assert frozen.CHAIN==["dc_ln"]+["dc_mha"]*8+["dc_proj","dc_ffn"]
    def resolve(root,reference):return reference["file"] if "file" in reference else original_resolve(root,reference)
    def read(path):
        if fg_fault.path(path)==Path(out)/"faults"/(CID+".json"):path=input_record()["fault_file"]
        return original_read(path)
    def kv(events,hp,ctx,right,c):
        bundle=fg_fault.read_pickle(resolve(out,spec["sources"]["P"]))
        tags={(tag[0],10,0)+tuple(tag[3:]) for tag in bundle["kv_tags"]}
        assert tags<=set(events)
        info=model_joint.redirect_kv(events,hp,tags,right,ctx,c.BCAST_SEED)
        assert info["payload_bytes"]==bundle["kv"]["logical_bytes"]
        return tags,dict(messages=info["messages"],unique_payload_bytes=info["payload_bytes"],targets=info["target_ports"],
            multicast_link_bytes=info["routed_unique_link_bytes"],cross_group_edges=info["crossing_links"],
            tensor_contract=bundle["kv"],identification="exact Ktranspose/V tensor producer tags and region bytes")
    frozen.resolve_export,frozen.read_json,frozen.build_kv=resolve,read,kv
    try:yield
    finally:
        f_config.prepare,frozen.resolve_export,frozen.build_kv,frozen.read_json,frozen.CHAIN=original_prepare,original_resolve,original_kv,original_read,original_chain


def child(spec,out):
    out=Path(out);started=time.perf_counter()
    assert spec["model"]=="llama" and spec["model_contract_sha256"]==H.digest(H.model_contract("llama"))
    if spec["kind"]=="hi_joint_export":
        source=input_record()["P"][spec["mapping"]]
        target=out/"exports"/(spec["job_id"]+".pkl.gz")
        bundle=model_joint.export_candidate(ROOT/source["record"],ROOT/source["raw"],ROOT/source["snapshot"],target)
        result=dict(export=fg_fault.relative(target),candidate_id=bundle["candidate_id"],kv=bundle["kv"],
            platform_sha256=bundle["platform_sha256"],job_id=spec["job_id"],kind=spec["kind"],status="ok",spec=spec,
            wall_seconds=time.perf_counter()-started,new_backend_calls=0)
        import e2e_pilot_peak
        result["peak_rss_gb"]=e2e_pilot_peak.peak()/2**30
        write_json(frozen.result_path(out,spec["job_id"]),result)
        return
    with adapter(spec,out):
        if spec["kind"]=="joint_graph":
            result=frozen.graph_job(spec,out)
            result["model"]="llama";result["model_contract_sha256"]=spec["model_contract_sha256"]
            result["decode_chain_contract"]=dict(mha_groups=8,external_norms=1,ffn_embedded_norms=1,stages=11)
            assert result["chain"]==model_joint.decode_chain_order("llama")
            result["stream_identity"]={key:stream_signature(fg_fault.read_pickle(file)) for key,file in result["files"].items()}
            result["actual_iteration_sha256"]={key:execution_order_digest(fg_fault.read_pickle(file)) for key,file in result["files"].items()}
            result.update(job_id=spec["job_id"],kind=spec["kind"],status=result.get("status","ok"),spec=spec,
                wall_seconds=time.perf_counter()-started)
            import e2e_pilot_peak
            result["peak_rss_gb"]=e2e_pilot_peak.peak()/2**30
            write_json(frozen.result_path(out,spec["job_id"]),result)
        else:
            if spec["kind"]=="joint":
                graph=previous.read(frozen.result_path(out,spec["graph_job"]))
                for stream,label in ((0,"P_solo"),(1,"D_solo")):
                    solo=previous.read(frozen.result_path(out,spec[label]))
                    paired=previous.read(frozen.result_path(out,solo["graph_job"]))
                    assert paired["platform_sha256"]==graph["platform_sha256"]
                    assert paired["stream_identity"][f"{spec['backend']}_{stream}"]==graph["stream_identity"][f"{spec['backend']}_{stream}"]
                    assert paired["actual_iteration_sha256"][f"{spec['backend']}_{stream}"]==graph["actual_iteration_sha256"][f"{spec['backend']}_{stream}"]
            frozen.child(spec,out)


def plan():
    rows=common.read_csv(HERE/"tables/h3_i2_candidates.csv")
    P={}
    for method in ("A1","A3h"):
        group=[r for r in rows if r["model"]=="llama" and r["hardware_profile"]=="busybarn" and r["phase"]=="prefill" and r["op"]=="mha" and r["method"]==method]
        assert len(group)==14
        win=min(group,key=lambda r:(float(r["backend_mean_ns"]),int(r["candidate_order"])))
        record=ROOT/Path(win["source"].replace("\\","/"));folder=record.parent.parent;name=win["candidate_id"]
        files=dict(record=record,raw=folder/"raw"/(name+".pkl.gz"),snapshot=folder/"snapshots"/(name+".pkl.gz"))
        assert all(p.exists() for p in files.values())
        P[method]={k:p.relative_to(ROOT).as_posix() for k,p in files.items()}
        P[method].update(candidate_id=name,sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()})
    D={}
    for state,arm,kind,key in (("healthy_A1","A1","pre","exports.healthy"),("healthy_A3h","A3h","pre","exports.healthy"),
        ("D_A1","A1","l1","exports.R0"),("D_A3hR0","A3h","l1","exports.R0"),("D_A3hEIR","A3h","stop","export")):
        D[state]={}
        for op in frozen.DC:
            suffix=f"pre_{op}_{arm}" if kind=="pre" else f"l1_{op}_{arm}_PF4" if kind=="l1" else f"stop_{op}_PF4_300"
            path=H4/"results"/(CID+"__"+suffix+".json");record=previous.read(path)
            assert record["status"]=="ok"
            value=record
            for token in key.split("."):value=value[token]
            source=ROOT/value;assert source.exists()
            D[state][op]=dict(file=value,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_job=path.relative_to(ROOT).as_posix())
    fault=H4/"faults"/(CID+".json");assert previous.read(fault)["faults"]["PF4"]["failed_links"]==[[[0,11],[1,8]],[[1,8],[0,11]]]
    write_json(INPUTS,dict(status="passed",model="llama",P=P,D=D,fault_file=fault.relative_to(ROOT).as_posix(),new_healthy_searches=0))
    base=dict(model="llama",hardware_profile="busybarn",shape="2x2",split_mode="fixed",d2d_scale=1,config_id=CID,
        source_config_id=CID,module="hi_h5",protocol_version="hi_h5_formal_v1",model_contract_sha256=H.digest(H.model_contract("llama")),p_op="mha")
    specs=[]
    def add(suffix,kind,deps,**kw):
        name="H5_llama__"+suffix
        specs.append(dict(base,job_id=name,kind=kind,depends_on=list(dict.fromkeys(deps)),**kw));return name
    exports={m:add("P_export_"+m,"hi_joint_export",[],mapping=m) for m in P}
    definitions=(("healthy_A1","healthy","A1",["orig"]),("healthy_A3h","healthy","A3h",["orig","hop"]),
        ("D_A1","D","A1",["orig"]),("D_A3hR0","D","A3h",["orig"]),("D_A3hEIR","D","A3h",["orig","hop"]))
    graphs={};solos={};aliases=[];identities={}
    for state,condition,mapping,backends in definitions:
        sources={op:{"file":record["file"]} for op,record in D[state].items()};sources["P"]={"job":exports[mapping],"key":"export"}
        graph=add("graph_"+state,"joint_graph",[exports[mapping]],sources=sources,condition=condition,backends=backends,mapping_state=state)
        graphs[state]=graph
        for backend in backends:
            for stream in (0,1):
                stream_source=P[mapping]["candidate_id"] if stream==0 else tuple(D[state][op]["file"] for op in frozen.DC)
                for seed in SEEDS:
                    identity=(condition,backend,stream,seed,stream_source)
                    logical=(state,backend,stream,seed)
                    if identity in identities:
                        solos[logical]=identities[identity];aliases.append(dict(logical=list(logical),job=solos[logical]));continue
                    solo=add(f"solo_{state}_{backend}_{stream}_{seed}","joint_solo",[graph],graph_job=graph,backend=backend,stream=stream,seed=seed,condition=condition)
                    identities[identity]=solo;solos[logical]=solo
    ps=[solos["healthy_A1","orig",0,s] for s in SEEDS];ds=[solos["healthy_A1","orig",1,s] for s in SEEDS]
    reference=add("reference","joint_reference",ps+ds,P_solos=ps,D_solos=ds)
    arms=(("healthy_A1","A1_no_DSTU","orig"),("healthy_A3h","A3h_no_DSTU","orig"),("healthy_A3h","A3h_A256","hop"),
          ("D_A1","A1_R0_no_DSTU","orig"),("D_A3hR0","A3h_R0_no_DSTU","orig"),("D_A3hEIR","A3h_EIR_no_DSTU","orig"),("D_A3hEIR","A3h_EIR_A256","hop"))
    for state,arm,backend in arms:
        for phase in PHASES:
            for seed in SEEDS:
                ps,ds=solos[state,backend,0,seed],solos[state,backend,1,seed]
                add(f"joint_{state}_{arm}_g{phase}_s{seed}","joint",[graphs[state],reference,ps,ds],graph_job=graphs[state],reference_job=reference,
                    P_solo=ps,D_solo=ds,arm=arm,condition="D" if state.startswith("D_") else "healthy",backend=backend,policy="C4" if backend=="hop" else "P1",phase=phase,seed=seed)
    assert sum(s["kind"]=="joint" for s in specs)==63 and sum(s["kind"]=="joint_solo" for s in specs)<=42
    jobs=[]
    for s in specs:
        seconds={"hi_joint_export":15,"joint_graph":120,"joint_solo":120,"joint_reference":1,"joint":180}[s["kind"]]
        jobs.append(dict(id=s["job_id"],kind=s["kind"],spec=s,result=f"results/{s['job_id']}.json",deps=s["depends_on"],
            argv=["{python}","-u","-B","e2e/hi_h5.py","child","--spec","{spec}","--out","{out}"],
            estimated_seconds=seconds,peak_rss_gb=3.,priority=0))
    by_id={j["id"]:j for j in jobs}
    assert {d for j in jobs for d in j["deps"]}<=set(by_id)
    selected=next(s["job_id"] for s in specs if s["kind"]=="joint" and s["arm"]=="A3h_A256" and s["phase"]==.5 and s["seed"]==1)
    pilot=set()
    def need(name):
        if name in pilot:return
        pilot.add(name)
        for dep in by_id[name]["deps"]:need(dep)
    need(selected)
    write_json(MANIFEST,dict(status="authorized_prepared_not_launched",jobs=jobs,logical_paired_solos=42,actual_paired_solos=len(identities),solo_aliases=aliases))
    write_json(PILOT,dict(status="authorized_pilot",jobs=[j for j in jobs if j["id"] in pilot],target=selected,
        counts=dict(Counter(j["kind"] for j in jobs if j["id"] in pilot))))
    result=dict(status="passed",jobs=len(jobs),counts=dict(Counter(s["kind"] for s in specs)),joint_units=63,logical_paired_solos=42,
        actual_paired_solos=len(identities),solo_aliases=aliases,pilot_jobs=len(pilot),pilot_target=selected,
        fault_conditions=["healthy","D_PF4"],phases=list(PHASES),backend_seeds=list(SEEDS),decode_chain=model_joint.decode_chain_order("llama"),
        phase_reference="same healthy A1/original three-seed P/D mean and KV window for all methods",
        waiting_for="actual pilot time/RSS before full launch",provisional_timing_only=True)
    write_json(HERE/"evidence/h5_plan.json",result);return result


class Controller(prior.base.Controller):
    def __init__(self,out,workers,manifest):
        bind(manifest);super().__init__(out,workers)
    def success_receipt(self,receipt):
        name=receipt["id"];job=self.by_id[name]
        assert receipt["identity"]==previous.pool.identity(job) and receipt["result"]==job["result"]
        result=previous.read(self.out/job["result"])
        assert result["job_id"]==name and result["status"] in ("ok","reused") and result["spec"]==job["spec"]
        if name in self.success:return
        self.success[name]=receipt
        self.observed[previous.category(job)].append(receipt)
        self.save_release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","check","child","launch","run"))
    parser.add_argument("--manifest",type=Path,default=MANIFEST)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--workers",type=int,default=128)
    args=parser.parse_args();args.out=args.out.resolve();args.manifest=args.manifest.resolve()
    assert 1<=args.workers<=128
    if args.command in ("plan","check"):print(json.dumps(plan(),ensure_ascii=False,indent=2));return 0
    if args.command=="child":child(previous.read(args.spec),args.out);return 0
    bind(args.manifest);args.out.mkdir(parents=True,exist_ok=True)
    if args.command=="launch":
        start=args.out/(NAME+".start.json")
        if start.exists() and previous.pool.process_alive(previous.read(start)["pid"]):raise RuntimeError("H5 controller already running")
        if start.exists() and Path(previous.read(start)["manifest"]).resolve()!=args.manifest:
            archive=args.out/"pilot_run"
            archive.mkdir(parents=True,exist_ok=True)
            for filename in ("h5.start.json","h5.status.json","h5.batch.json","controller_finish.json","estimated_finish.json"):
                source=args.out/filename
                if source.exists():
                    target=archive/filename
                    assert not target.exists()
                    source.rename(target)
        write_json(args.out/"scheduler_policy.json",dict(running_growth_fraction=.25))
        env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMBA_NUM_THREADS="1",CRISP_E_ARTIFACT=str(ROOT/"e2e/work/artifact"),PYTHONUTF8="1")
        command=[sys.executable,"-u","-B","e2e/hi_h5.py","run","--manifest",str(args.manifest),"--out",str(args.out),"--workers",str(args.workers)]
        options=dict(start_new_session=True) if sys.platform!="win32" else dict(creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        with (args.out/(NAME+".log")).open("ab") as stream:process=previous.RAW_POPEN(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,**options)
        write_json(start,dict(pid=process.pid,command=command,manifest=str(args.manifest),created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256={n:hashlib.sha256((ROOT/"e2e"/n).read_bytes()).hexdigest() for n in ("hi_h5.py","hi_joint.py","hi_i4.py","hi_h4.py","fg_joint.py","d_joint.py")}))
        print(json.dumps(dict(pid=process.pid,jobs=len(previous.RAW_JOB_ITEMS(args.manifest)),out=str(args.out))),flush=True);return 0
    return Controller(args.out,args.workers,args.manifest).run()


if __name__=="__main__":raise SystemExit(main())
