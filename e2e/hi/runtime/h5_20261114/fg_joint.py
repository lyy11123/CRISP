"""F/G paired P/D experiments on two adjacent groups, using frozen D backends."""
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import statistics
import time

from fg_fault import (ROOT, EMPTY_FAULT, path, relative, read_json, write_json, read_pickle,
                      write_pickle, result_path, platform_signature, job_id)

PHASES = (.1,.5,.9)
SEEDS = (1,2,3)
DC = ("dc_ln","dc_mha","dc_proj","dc_ffn")
CHAIN = ["dc_ln"] + ["dc_mha"]*8 + ["dc_proj","dc_ffn"]
ARMS = ("B-R0","CRISP","-DSTU","-priority")


def maps(shape):
    rows,cols=map(int,shape.split("x"))
    return {"L":{r*cols+c:r*(2*cols)+c for r in range(rows) for c in range(cols)},
            "R":{r*cols+c:r*(2*cols)+c+cols for r in range(rows) for c in range(cols)}}


def shape_joint(shape):
    rows,cols=map(int,shape.split("x"))
    return f"{rows}x{2*cols}"


def crossing_fault(shape):
    _,cols=map(int,shape.split("x"))
    a,b=[cols-1,11],[cols,8]
    return dict(EMPTY_FAULT,failed_links=[[a,b],[b,a]])


def d_modules():
    os.environ["D_MAIN"]=str(ROOT)
    import d_joint as d
    import event_notation
    event_notation.event_notation.__hash__=lambda self:hash(self.event_tag)
    import cpcm,fast_cpcm
    cpcm.simulate=fast_cpcm.simulate
    return d


def resolve_export(out,reference):
    value=read_json(result_path(out,reference["job"]))
    for key in reference["key"].split("."):
        value=value[key]
    return value


def build_kv(events,hp,ctx,right,c):
    import numpy as np
    targets={loc[:-1] for loc in hp.ddr_dict if loc[0] in right}
    tags=sorted(t for t,e in events.items() if e.event_type=="communication" and
                any(name in e.event_name for name in ("kconv","vconv")) and
                isinstance(e.target_location,set) and all(x[1]>=16 for x in e.target_location))
    before=random.getstate(),np.random.get_state()
    random.seed(c.BCAST_SEED); np.random.seed(c.BCAST_SEED)
    actual=0
    crossings=set()
    for tag in tags:
        e=events[tag]
        e.target_location=set(targets)
        e.paths=hp.record_dijkstra_multicast_path(comm_pairs=[[tag,e.source_location,set(targets),e.comm_bytes]],
            alpha=ctx["alpha"],beta=ctx["beta"],gamma=ctx["gamma"])[2][tag]
        e.path_list=e.get_paths(e.paths)
        links={link for branch in e.path_list for link in branch}
        assert {b[-1][1] for b in e.path_list if b}==targets
        assert links<=set(hp.links_dict)
        actual+=int(e.comm_bytes)*len(links)
        crossings.update(link for link in links if (link[0][0] in right)!=(link[1][0] in right))
    random.setstate(before[0]); np.random.set_state(before[1])
    return set(tags),dict(messages=len(tags),unique_payload_bytes=sum(int(events[t].comm_bytes) for t in tags),
                          targets=len(targets),multicast_link_bytes=actual,cross_group_edges=len(crossings))


def graph_job(spec,out):
    import f_config
    s=dict(spec,phase="prefill",op=spec.get("p_op","mha"),method="A3h",g=0,x=123)
    _,c,_=f_config.prepare(s,out)
    d=d_modules()
    mp=maps(spec["shape"])
    fault=EMPTY_FAULT
    if spec["condition"]=="D":
        local_fault=read_json(Path(out)/"faults"/(spec["source_config_id"]+".json"))["faults"]["PF4"]
        fault=d.remap_fault(local_fault,mp["R"])
    elif spec["condition"]=="cross":
        local_fault=EMPTY_FAULT
        fault=crossing_fault(spec["shape"])
    else:
        local_fault=EMPTY_FAULT
    references={key:resolve_export(out,r) for key,r in spec["sources"].items()}
    identity={key:spec[key] for key in ("hardware_profile","shape","split_mode","d2d_scale","condition")}
    identity.update(sources=references,p_op=spec.get("p_op","mha"),fault=fault,
                    protocol_version=spec.get("protocol_version","structural_fixture"),chain=CHAIN,chunk_bytes=256*1024)
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    folder=Path(out)/"joint_cache"/digest
    meta_file=folder/"graph.json"
    if meta_file.exists():
        return dict(read_json(meta_file),status="reused")
    hp=f_config.build_platform(s,out,faults=fault,shape_override=shape_joint(spec["shape"]))
    exports={k:read_pickle(v) for k,v in references.items()}
    for group,keys,local in (("L",["P"],EMPTY_FAULT),("R",list(DC),local_fault)):
        local_hp=f_config.build_platform(s,out,faults=local)
        structural=d.check_group(local_hp,hp,mp[group])
        assert structural["links_bad"]==0 and structural["modules_equal"] and structural["links_22"]==structural["links_24_in_group"],structural
        for key,field in (("tensorcore_dict","tensorcore_cfg"),("vectorunit_dict","vectorunit_cfg")):
            for loc,module in getattr(local_hp,key).items():
                target=getattr(hp,key)[d.rn(loc,mp[group])]
                assert getattr(module,field)==getattr(target,field)
        if spec.get("mapping_reuse")!="watos_1x":
            assert all(exports[key]["platform_sha256"]==platform_signature(local_hp) for key in keys)
    p=d.retag(d.remap_events(pickle.loads(exports["P"]["events_blob"]),mp["L"]),0)
    kv,kv_info=build_kv(p,hp,exports["P"]["ctx"],set(mp["R"].values()),c) if spec.get("p_op","mha")=="mha" else (set(),{})
    chain={}; previous=None; edges=0
    for index,op in enumerate(CHAIN):
        ev=d.retag(d.remap_events(pickle.loads(exports[op]["events_blob"]),mp["R"]),1,index)
        sources={t for t,e in ev.items() if not e.dependency_set}
        sinks={t for t,e in ev.items() if not e.issue_set}
        if previous is not None:
            for tag in sources:ev[tag].dependency_set.update(previous)
            for tag in previous:chain[tag].issue_set.update(sources)
            edges+=len(sources)*len(previous)
        chain.update(ev);previous=sinks
    for ev in (p,chain):
        assert d.check_graph(ev,hp)=={"missing_compute":0,"missing_links":0}
    p_links=d.links_of({t:e for t,e in p.items() if t not in kv})
    kv_links=d.links_of({t:e for t,e in p.items() if t in kv})
    d_links=d.links_of(chain)
    p_compute={(e.comp_device,e.comp_location) for e in p.values() if e.event_type=="computation"}
    d_compute={(e.comp_device,e.comp_location) for e in chain.values() if e.event_type=="computation"}
    shared=dict(P_D_links=len(p_links&d_links),KV_D_links=len(kv_links&d_links),compute=len(p_compute&d_compute))
    if spec.get("p_op")=="ffn":assert shared==dict(P_D_links=0,KV_D_links=0,compute=0),shared
    streams={0:p,1:chain}; files={}; that={}; sizes={}
    import dstu2
    for stream,ev in streams.items():
        filename=folder/f"orig_{stream}.pkl.gz"
        write_pickle(filename,ev);files[f"orig_{stream}"]=relative(filename)
        sizes[str(stream)]={"events":len(ev),"dependency_edges":sum(len(e.dependency_set) for e in ev.values())}
        if "hop" in spec["backends"]:
            chunks,chunk_stat=dstu2.chunk_by(ev,hp,{"size":256*1024,"only_d2d":False})
            filename=folder/f"hop_{stream}.pkl.gz"
            write_pickle(filename,chunks);files[f"hop_{stream}"]=relative(filename)
            T,wall,n=d.that_hop(chunks,hp)
            that[str(stream)]=int(T)
            sizes[str(stream)].update(chunked_events=len(chunks),chunk=chunk_stat,hop_activities=n,that_seconds=wall)
    result=dict(cache_key=digest,files=files,that=that,kv_tags=[list(t) for t in sorted(kv)],kv=kv_info,
                shared=shared,chain=CHAIN,chain_edges=edges,sizes=sizes,fault=fault,platform=identity,
                platform_sha256=platform_signature(hp),source_exports=references)
    write_json(meta_file,result)
    return result


def load_graph(spec,out):
    import f_config
    graph=read_json(result_path(out,spec["graph_job"]))
    s=dict(spec,phase="prefill",op=spec.get("p_op","mha"),method="A3h",g=0,x=123)
    _,c,_=f_config.prepare(s,out)
    d=d_modules()
    hp=f_config.build_platform(s,out,faults=graph["fault"],shape_override=shape_joint(s["shape"]))
    assert platform_signature(hp)==graph["platform_sha256"]
    return d,c,hp,graph


def kv_window(result,tags,d):
    kv={tuple(t) for t in tags}
    selected=[t for t in result["record"] if t[1]==10 and d.base_of(t) in kv]
    window=d.window(result["record"],selected)
    result.pop("record",None);result.pop("links",None)
    return window


def solo_job(spec,out):
    d,c,hp,graph=load_graph(spec,out)
    stream=spec["stream"]
    ev=read_pickle(graph["files"][f"{spec['backend']}_{stream}"])
    need_kv=stream==0 and bool(graph["kv_tags"])
    result=d.run_backend(c,pickle.dumps(ev,pickle.HIGHEST_PROTOCOL),hp,spec["backend"],{"name":"P1"},{stream:0},spec["seed"],record=need_kv)
    if need_kv:result["kv_window"]=kv_window(result,graph["kv_tags"],d)
    return dict(simulation=result,graph_job=spec["graph_job"],stream=stream,backend=spec["backend"],seed=spec["seed"])


def reference_job(spec,out):
    ps=[read_json(result_path(out,j))["simulation"] for j in spec["P_solos"]]
    ds=[read_json(result_path(out,j))["simulation"] for j in spec["D_solos"]]
    return dict(T_P=statistics.mean(r["streams"]["0"]["C"] for r in ps),
                T_D=statistics.mean(r["streams"]["1"]["C"] for r in ds),
                KV_start=statistics.mean(r["kv_window"][0] for r in ps),
                KV_end=statistics.mean(r["kv_window"][1] for r in ps),
                backend_seeds=list(SEEDS),reference="healthy A1, original backend, cross-group KV")


def joint_job(spec,out):
    d,c,hp,graph=load_graph(spec,out)
    reference=read_json(result_path(out,spec["reference_job"]))
    tau=int(round(reference["KV_start"]+spec["phase"]*(reference["KV_end"]-reference["KV_start"])))
    ev=read_pickle(graph["files"][spec["backend"]+"_0"])
    ev.update(read_pickle(graph["files"][spec["backend"]+"_1"]))
    policy={"name":spec["policy"]}
    if spec["policy"]=="C4":
        deadlines={0:1.5*graph["that"]["0"],1:tau+1.5*graph["that"]["1"]}
        order=sorted(deadlines,key=lambda k:(deadlines[k],k))
        policy.update(rank={k:order.index(k) for k in order},d_hat=deadlines)
    result=d.run_backend(c,pickle.dumps(ev,pickle.HIGHEST_PROTOCOL),hp,spec["backend"],policy,{0:0,1:tau},spec["seed"],record=bool(graph["kv_tags"]))
    if graph["kv_tags"]:result["kv_window"]=kv_window(result,graph["kv_tags"],d)
    psolo=read_json(result_path(out,spec["P_solo"]))["simulation"]["streams"]["0"]["C"]
    dsolo=read_json(result_path(out,spec["D_solo"]))["simulation"]["streams"]["1"]["C"]
    cp,cd=result["streams"][0]["C"],result["streams"][1]["C"]
    metrics=dict(P_completion=cp,D_latency=cd-tau,paired_delta_P=cp-psolo,P_solo=psolo,D_solo=dsolo,
                 P_self_amplification=cp/psolo,D_self_amplification=(cd-tau)/dsolo)
    if spec.get("p_op","mha")=="mha":
        metrics.update(Lhat_P=(cp-1.5*reference["T_P"])/reference["T_P"],
                       Lhat_D=(cd-tau-1.5*reference["T_D"])/reference["T_D"])
        metrics["Lhat_max"]=max(metrics["Lhat_P"],metrics["Lhat_D"])
    return dict(simulation=result,metrics=metrics,tau=tau,phase=spec["phase"],arm=spec["arm"],
                condition=spec["condition"],graph_job=spec["graph_job"],shared=graph["shared"],reference_job=spec["reference_job"])


def child(spec,out):
    out=path(out);start=time.perf_counter()
    fn={"joint_graph":graph_job,"joint_solo":solo_job,"joint_reference":reference_job,"joint":joint_job,"joint_ffn_check":joint_job}[spec["kind"]]
    result=fn(spec,out)
    import e2e_pilot_peak
    result.update(job_id=job_id(spec),kind=spec["kind"],status=result.get("status","ok"),
                  wall_seconds=time.perf_counter()-start,peak_rss_gb=e2e_pilot_peak.peak()/2**30,spec=spec)
    write_json(result_path(out,job_id(spec)),result)
    print(f"[fg_joint] {job_id(spec)} {result['wall_seconds']:.2f}s",flush=True)


def source_refs(cid,mapping,condition,p_op="mha"):
    arm="A1" if mapping=="A1" else "A3h"
    refs={"P":{"job":f"{cid}__pre_pf_{p_op}_{arm}","key":"exports.healthy"}}
    for op in DC:
        if condition!="D":ref={"job":f"{cid}__pre_{op}_{arm}","key":"exports.healthy"}
        elif mapping=="A3hEIR":ref={"job":f"{cid}__stop_{op}_PF4_300","key":"export"}
        else:ref={"job":f"{cid}__l1_{op}_{arm}_PF4","key":"exports.R0"}
        refs[op]=ref
    return refs


def jobs(configs,out):
    from fg_manifest import config_id, PROTOCOL
    result=[]
    configs=[c for c in configs if c["stage"]=="F" or c["split_mode"]=="growth"]
    watos=next(c for c in configs if c["hardware_profile"]=="watos")
    extra=[]
    for scale in (.5,2.):
        cfg=dict(watos,stage="F_bw",d2d_scale=scale)
        cfg["config_id"]=config_id(cfg);extra.append(cfg)
    for cfg in configs+extra:
        cid=config_id(cfg);sweep=cfg["stage"]=="F_bw"
        source_cid=config_id(watos) if sweep else cid
        base=dict(cfg,module="fg_joint",source_config_id=source_cid,mapping_reuse="watos_1x" if sweep else "matched",protocol_version=PROTOCOL)
        def add(suffix,kind,deps,**kw):
            name=cid+"__"+suffix
            result.append(dict(base,job_id=name,kind=kind,depends_on=list(dict.fromkeys(deps)),**kw))
            return name
        graphs={};solo={}
        for cond in (("healthy",) if sweep else ("healthy","D","cross")):
            for mapping in (("A1","A3hEIR","A3hR0") if cond=="D" else ("A1","A3hEIR")):
                refs=source_refs(source_cid,mapping,cond)
                deps=[r["job"] for r in refs.values()]
                if cond=="D":deps.append(source_cid+"__fault_select")
                if mapping=="A3hR0":deps.append(graphs[(cond,"A3hEIR")])
                backends=["orig"] if mapping=="A1" else ["hop"] if mapping=="A3hR0" else ["orig","hop"]
                graph=add(f"graph_{cond}_{mapping}","joint_graph",deps,sources=refs,condition=cond,backends=backends)
                graphs[(cond,mapping)]=graph
                for backend in backends:
                    for stream in (0,1):
                        for seed in SEEDS:
                            solo[(cond,mapping,backend,stream,seed)]=add(f"solo_{cond}_{mapping}_{backend}_{stream}_{seed}","joint_solo",[graph],
                                graph_job=graph,backend=backend,stream=stream,seed=seed,condition=cond)
        ps=[solo[("healthy","A1","orig",0,sd)] for sd in SEEDS]
        ds=[solo[("healthy","A1","orig",1,sd)] for sd in SEEDS]
        reference=add("joint_reference","joint_reference",ps+ds,P_solos=ps,D_solos=ds)
        for cond in (("healthy",) if sweep else ("healthy","D","cross")):
            arms=ARMS+(("-EIR",) if cond=="D" else ())
            for arm in arms:
                mapping="A1" if arm=="B-R0" else "A3hR0" if arm=="-EIR" else "A3hEIR"
                backend="orig" if arm in ("B-R0","-DSTU") else "hop"
                policy="C4" if arm in ("CRISP","-EIR") else "P1"
                for phase in ((.5,) if sweep else PHASES):
                    for seed in SEEDS:
                        psolo=solo[(cond,mapping,backend,0,seed)];dsolo=solo[(cond,mapping,backend,1,seed)]
                        add(f"joint_{cond}_{arm}_g{phase}_s{seed}","joint",[graphs[(cond,mapping)],reference,psolo,dsolo],
                            graph_job=graphs[(cond,mapping)],reference_job=reference,P_solo=psolo,D_solo=dsolo,
                            arm=arm,condition=cond,backend=backend,policy=policy,phase=phase,seed=seed)
        if not sweep:
            refs=source_refs(source_cid,"A3hEIR","healthy","ffn")
            graph=add("graph_ffn_check","joint_graph",[r["job"] for r in refs.values()],sources=refs,
                      p_op="ffn",condition="healthy",backends=["hop"])
            solos=[add(f"solo_ffn_check_{stream}","joint_solo",[graph],graph_job=graph,p_op="ffn",condition="healthy",backend="hop",stream=stream,seed=1) for stream in (0,1)]
            add("joint_ffn_check","joint_ffn_check",[graph,reference]+solos,graph_job=graph,p_op="ffn",condition="healthy",
                reference_job=reference,P_solo=solos[0],D_solo=solos[1],arm="CRISP",backend="hop",policy="C4",phase=.5,seed=1)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["child"])
    parser.add_argument("--spec",required=True);parser.add_argument("--out",required=True)
    args=parser.parse_args();child(read_json(args.spec),args.out)


if __name__=="__main__":main()
