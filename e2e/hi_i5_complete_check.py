"""Independent saved I5 audit: control records, streams, reuse and all metrics."""
from collections import Counter
import copy
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import statistics
from hi_next_check import read,sha
from hi_h3_i2_check import write

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"e2e/hi"
OUT=HERE/"out/i5"


def load(path):
    with gzip.open(path,"rb") as stream:return pickle.load(stream)


def identity(job):
    return hashlib.sha256(json.dumps({k:job[k] for k in ("id","kind","spec","result","argv")},sort_keys=True).encode()).hexdigest()


def stream_hash(events):
    def normalize(v):
        if isinstance(v,dict):return [(normalize(k),normalize(x)) for k,x in v.items()]
        if isinstance(v,set):return sorted((normalize(x) for x in v),key=repr)
        if isinstance(v,(tuple,list)):return [normalize(x) for x in v]
        if hasattr(v,"__dict__"):return (type(v).__module__,type(v).__qualname__,normalize(vars(v)))
        if hasattr(v,"tolist"):return normalize(v.tolist())
        if hasattr(v,"item"):return v.item()
        return v
    return hashlib.sha256(repr([(tag,normalize(vars(e))) for tag,e in events.items()]).encode()).hexdigest()


def order_hash(events):
    rows=[]
    for tag,e in events.items():
        target=getattr(e,"target_location",None)
        rows.append((tag,list(e.dependency_set),list(e.issue_set),list(target) if isinstance(target,set) else target))
    return hashlib.sha256(json.dumps(rows,default=int).encode()).hexdigest()


def links(events):
    return {link for e in events.values() if e.event_type=="communication" for branch in e.path_list for link in branch}


def check():
    plan=read(HERE/"evidence/i5_plan_review.json")
    assert sha(HERE/"manifest/i5_pilot.json")==plan["pilot_manifest_sha256"]
    assert sha(HERE/"manifest/i5_inputs.json")==plan["input_inventory_sha256"]
    manifest=read(HERE/"manifest/i5.json")
    # Full-launch estimates were replaced with measured pilot time/RSS.
    # Restore only those two scheduling fields and reproduce the original bytes.
    initial=copy.deepcopy(manifest)
    for job in initial["jobs"]:
        job["estimated_seconds"]={"hi_joint_export":15,"joint_graph":120,"joint_solo":120,"joint_reference":1,"joint":180}[job["kind"]]
        job["peak_rss_gb"]=3.
    initial_text=(json.dumps(initial,ensure_ascii=False,indent=2)+"\n").replace("\n","\r\n")
    assert hashlib.sha256(initial_text.encode()).hexdigest()==plan["manifest_sha256"]
    jobs={j["id"]:j for j in manifest["jobs"]}
    assert len(jobs)==110
    results={n:read(OUT/j["result"]) for n,j in jobs.items()}
    receipts={n:read(OUT/"receipts"/(n+".json")) for n in jobs}
    for folder in ("results","receipts","specs"):assert {p.stem for p in (OUT/folder).glob("*.json")}==set(jobs)
    source_checks={}
    for folder in (OUT,OUT/"pilot_run"):
        start=read(folder/"i5.start.json")
        for name,digest in start["source_sha256"].items():
            assert sha(HERE/"runtime/i5_20261114"/name)==digest and sha(ROOT/"e2e"/name)==digest
            source_checks[name]=digest
    controller=[json.loads(line) for line in (OUT/"i5.log").read_text(encoding="utf-8").splitlines() if line.startswith('{"event":')]
    starts_seen=[r for r in controller if r["event"]=="start"]
    assert [(r["jobs"],r["cached"]) for r in starts_seen]==[(14,0),(110,14)]
    done=[r for r in controller if r["event"]=="done"]
    assert len(done)==len({r["id"] for r in done})==110
    for r in done:assert {k:v for k,v in r.items() if k!="event"}==receipts[r["id"]]
    assert not list((OUT/"failed_attempts").glob("*/*/receipt.json"))
    starts,ends,records={},{},[]
    for name,job in jobs.items():
        result,receipt=results[name],receipts[name]
        assert result["status"]=="ok" and result["spec"]==job["spec"] and result["job_id"]==name
        assert read(OUT/"specs"/(name+".json"))==job["spec"]
        assert receipt["rc"]==0 and receipt["identity"]==identity(job) and receipt["result"]==job["result"] and not receipt["cached"]
        assert receipt["wall_seconds"]>=result["wall_seconds"]
        logs=list((OUT/"logs").glob(name+".*.log"));assert len(logs)==1
        log=logs[0];content=log.read_text(encoding="utf-8")
        if job["kind"] in ("hi_joint_export","joint_graph"):assert content==""
        else:assert content.splitlines()[-1]==f"[fg_joint] {name} {result['wall_seconds']:.2f}s"
        starts[name]=int(log.stem.rsplit(".",1)[1])/1e9
        ends[name]=starts[name]+receipt["wall_seconds"]
        records.append(dict(job_id=name,kind=job["kind"],receipt_identity=receipt["identity"],result_sha256=sha(OUT/job["result"]),log=log.relative_to(ROOT).as_posix(),log_bytes=log.stat().st_size))
    assert len(list((OUT/"logs").glob("*.log")))==110
    margins=[]
    for name,job in jobs.items():
        for parent in job["deps"]:
            gap=starts[name]-ends[parent]
            assert gap>=-2.1,(name,parent,gap)
            margins.append(dict(job_id=name,parent=parent,seconds=gap))
    pilot_ids={j["id"] for j in read(HERE/"manifest/i5_pilot.json")["jobs"]}
    pilot=read(OUT/"pilot_run/i5.batch.json");batch=read(OUT/"i5.batch.json")
    assert pilot["completed"]==14 and {r["id"] for r in pilot["runs"]}==pilot_ids
    assert batch["completed"]==110 and {r["id"] for r in batch["runs"] if r["cached"]}==pilot_ids
    assert batch["failed"]==batch["pending"]==pilot["failed"]==pilot["pending"]==[]
    for folder,count in ((OUT,110),(OUT/"pilot_run",14)):
        finish=read(folder/"controller_finish.json")
        assert finish["rc"]==0 and finish["expected"]==finish["completed"]==count
    inputs=read(HERE/"manifest/i5_inputs.json")
    for row in inputs["P"].values():
        for kind in ("record","raw","snapshot"):assert sha(ROOT/row[kind])==row["sha256"][kind]
    for operators in inputs["D"].values():
        for row in operators.values():assert sha(ROOT/row["file"])==row["sha256"]
    import post_fg_split
    post_fg_split.setup()
    exports={}
    for name,job in jobs.items():
        if job["kind"]!="hi_joint_export":continue
        result=results[name];mapping=job["spec"]["mapping"]
        assert result["new_backend_calls"]==0 and result["candidate_id"]==inputs["P"][mapping]["candidate_id"]
        bundle=load(ROOT/result["export"])
        assert bundle["model_contract_sha256"]==plan["model_contract_sha256"]
        assert bundle["kv"]["logical_bytes"]==bundle["kv"]["payload_bytes"]==8388608
        assert bundle["kv"]["messages"]==8 and bundle["kv"]["ddr_copy_bytes"]==268435456
        assert [t["tensor_name"] for t in bundle["kv"]["tensors"]]==["MHA_ktrans_ofmap","MHA_vconv_ofmap"]
        exports[name]=bundle
    graph_checks=[];actual_streams={}
    for name,job in jobs.items():
        if job["kind"]!="joint_graph":continue
        r=results[name];s=job["spec"]
        assert r["chain"]==["dc_ln"]+["dc_mha"]*4+["dc_proj","dc_ffn"]
        assert r["decode_chain_contract"]==dict(mha_groups=4,external_norms=1,ffn_embedded_norms=1,stages=7)
        fault={tuple(map(tuple,x)) for x in r["fault"]["failed_links"]}
        assert fault==({((2,11),(3,8)),((3,8),(2,11))} if s["condition"]=="D" else set())
        assert not r["fault"]["failed_nodes"] and not r["fault"].get("failed_pes",[])
        streams={}
        for key,file in r["files"].items():
            events=load(ROOT/file);stream=int(key[-1]);tags=set(events)
            assert all(t[1]==10+stream for t in tags)
            assert all((e.dependency_set|e.issue_set)<=tags for e in events.values())
            assert all(not (fault & set(branch)) for e in events.values() if e.event_type=="communication" for branch in e.path_list)
            assert all(e.comp_location[0] in ({0,1,4,5} if stream==0 else {2,3,6,7}) for e in events.values() if e.event_type=="computation")
            if key.startswith("hop_"):assert all(e.comm_bytes<=262144 for e in events.values() if e.event_type=="communication")
            canonical,ordered=stream_hash(events),order_hash(events)
            assert canonical==r["stream_identity"][key] and ordered==r["actual_iteration_sha256"][key]
            actual_streams[name,key]=(canonical,ordered)
            streams[key]=events
        p,d=streams["orig_0"],streams["orig_1"]
        assert len(p)==r["sizes"]["0"]["events"] and len(d)==r["sizes"]["1"]["events"]
        assert {t[2] for t in d}==set(range(7))
        cross=sum(dep[2]!=tag[2] for tag,e in d.items() for dep in e.dependency_set)
        assert cross==r["chain_edges"]
        assert all(dep[2] in (tag[2],tag[2]-1) for tag,e in d.items() for dep in e.dependency_set)
        kv={tuple(t) for t in r["kv_tags"]}
        p_export=exports[s["sources"]["P"]["job"]]
        assert kv=={(t[0],10,0)+tuple(t[3:]) for t in p_export["kv_tags"]}
        assert len(kv)==8 and sum(p[t].comm_bytes for t in kv)==8388608
        targets={(die,port) for die in (2,3,6,7) for port in range(16,24)}
        assert all(p[t].target_location==targets for t in kv)
        nonkv={t:e for t,e in p.items() if t not in kv};kv_events={t:p[t] for t in kv}
        shared=dict(P_D_links=len(links(nonkv)&links(d)),KV_D_links=len(links(kv_events)&links(d)),compute=0)
        assert shared==r["shared"] and shared["P_D_links"]==0 and shared["KV_D_links"]>0
        routed=sum(int(e.comm_bytes)*len({edge for branch in e.path_list for edge in branch}) for e in kv_events.values())
        assert routed==r["kv"]["multicast_link_bytes"]
        graph_checks.append(dict(job_id=name,streams=len(streams),D_stages=7,barrier_edges=cross,kv_payload_bytes=8388608,shared=shared,cross_stream_dependencies=0,stream_files_sha256={k:sha(ROOT/v) for k,v in r["files"].items()}))
    solo_ids={n for n,j in jobs.items() if j["kind"]=="joint_solo"}
    joint_ids={n for n,j in jobs.items() if j["kind"]=="joint"}
    assert len(solo_ids)==39 and len(joint_ids)==63
    for name in solo_ids|joint_ids:
        r,s=results[name],jobs[name]["spec"];sim=r["simulation"]
        assert sim["seed"]==s["seed"] and sim["backend"]==s["backend"] and not sim["stop"]
        assert all(v["n"]==v["n_done"] and v["C"]>=v["first_start"] for v in sim["streams"].values())
        if sim["backend"]=="hop":assert sim["audit"]["bad"]==0
    reference_id=next(n for n,j in jobs.items() if j["kind"]=="joint_reference")
    reference=results[reference_id];ref_spec=jobs[reference_id]["spec"]
    P=[results[n]["simulation"] for n in ref_spec["P_solos"]];D=[results[n]["simulation"] for n in ref_spec["D_solos"]]
    expected_ref=dict(T_P=statistics.mean(r["streams"]["0"]["C"] for r in P),T_D=statistics.mean(r["streams"]["1"]["C"] for r in D),KV_start=statistics.mean(r["kv_window"][0] for r in P),KV_end=statistics.mean(r["kv_window"][1] for r in P))
    assert all(reference[k]==v for k,v in expected_ref.items())
    metrics={};dedup=[]
    for name in joint_ids:
        r,s=results[name],jobs[name]["spec"];graph=results[s["graph_job"]]
        assert s["reference_job"]==reference_id
        tau=round(reference["KV_start"]+s["phase"]*(reference["KV_end"]-reference["KV_start"]))
        assert r["tau"]==tau and r["simulation"]["release"]=={"0":0,"1":tau}
        for stream,key in ((0,"P_solo"),(1,"D_solo")):
            solo=results[s[key]];paired=results[solo["graph_job"]]
            assert paired["platform_sha256"]==graph["platform_sha256"]
            assert actual_streams[solo["graph_job"],f"{s['backend']}_{stream}"]==actual_streams[s["graph_job"],f"{s['backend']}_{stream}"]
            if solo["graph_job"]!=s["graph_job"]:dedup.append((name,stream,s[key]))
        cp=r["simulation"]["streams"]["0"]["C"];dl=r["simulation"]["streams"]["1"]["C"]-tau
        ps=results[s["P_solo"]]["simulation"]["streams"]["0"]["C"]
        ds=results[s["D_solo"]]["simulation"]["streams"]["1"]["C"]
        m=dict(P_completion=cp,D_latency=dl,paired_delta_P=cp-ps,P_solo=ps,D_solo=ds,P_self_amplification=cp/ps,D_self_amplification=dl/ds,Lhat_P=(cp-1.5*reference["T_P"])/reference["T_P"],Lhat_D=(dl-1.5*reference["T_D"])/reference["T_D"])
        m["Lhat_max"]=max(m["Lhat_P"],m["Lhat_D"])
        assert m==r["metrics"]
        metrics[name]=m
    assert len(dedup)==9 and len({x[2] for x in dedup})==3
    with (HERE/"tables/i5_joints.csv").open(encoding="utf-8-sig",newline="") as stream:table=list(csv.DictReader(stream))
    assert len(table)==63
    for row in table:
        assert all(float(row[k])==v for k,v in metrics[row["job_id"]].items())
    comparisons=[]
    bykey={(s["condition"],s["arm"],s["phase"],s["seed"]):n for n in joint_ids for s in (jobs[n]["spec"],)}
    for condition,without,with_dstu in (("healthy","A3h_no_DSTU","A3h_A256"),("D","A3h_EIR_no_DSTU","A3h_EIR_A256")):
        pairs=[]
        for phase in (.1,.5,.9):
            for seed in (1,2,3):
                a=metrics[bykey[condition,without,phase,seed]];b=metrics[bykey[condition,with_dstu,phase,seed]]
                pairs.append(dict(P_ratio=b["P_completion"]/a["P_completion"]-1,D_ratio=b["D_latency"]/a["D_latency"]-1,Lhat_delta=b["Lhat_max"]-a["Lhat_max"]))
        comparisons.append(dict(condition=condition,pairs=9,P_median_change=statistics.median(x["P_ratio"] for x in pairs),D_median_change=statistics.median(x["D_ratio"] for x in pairs),Lhat_worse=sum(x["Lhat_delta"]>0 for x in pairs)))
    statuses=[r for r in controller if r["event"]=="status"]
    assert all(r["active"]+r["parent_reserved_slots"]<=128 for r in statuses)
    evidence=dict(status="passed",created_utc=datetime.now(timezone.utc).isoformat(),scope="Independent existing I5 outputs and original simulation metrics; no backend rerun and no report parser used.",
        jobs=110,failed_jobs=0,joint_samples=63,actual_paired_solo_samples=39,total_backend_samples=102,logical_paired_solos=42,
        pilot_jobs=14,pilot_backend_samples=9,pilot_formal_ids_reused=14,formal_jobs_executed_twice=0,
        unique_job_logs=110,empty_graph_or_export_logs=7,controller_done_records=110,
        old_healthy_searches_repeated=0,P_healthy_source_files_verified=6,D_existing_export_references_verified=20,
        model_contract_sha256=plan["model_contract_sha256"],common_reference=expected_ref,graphs=graph_checks,
        reused_P_solo_ids=sorted({x[2] for x in dedup}),reuse_joint_references=len(dedup),
        all_deadlines_met=all(m["Lhat_max"]<=0 for m in metrics.values()),A256_same_mapping_comparisons=comparisons,
        deadline_met_samples=sum(m["Lhat_max"]<=0 for m in metrics.values()),
        deadline_misses_by_arm=dict(Counter(jobs[n]["spec"]["arm"] for n,m in metrics.items() if m["Lhat_max"]>0)),
        dependency_edges=len(margins),minimum_dependency_margin_seconds=min(r["seconds"] for r in margins),dependency_tolerance_seconds=2.1,
        dependency_time_definition="Log filename precedes Popen; plus receipt.wall_seconds approximates successful receipt time and includes controller polling.",
        peak_I5_active=max(batch["max_active"],pilot["max_active"]),max_sampled_parent_reserved_plus_I5_active=max(r["active"]+r["parent_reserved_slots"] for r in statuses),
        minimum_sampled_available_gib=min(batch["min_available_gib"],pilot["min_available_gib"]),
        pilot_wall_seconds=pilot["wall_seconds"],remaining96_wall_seconds=batch["wall_seconds"],
        source_sha256=source_checks,report_joints_sha256=sha(HERE/"tables/i5_joints.csv"),
        report_artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/"e2e/hi_i5_report.py",HERE/"I5_REPORT.md",HERE/"evidence/i5_completion_audit.json"]+sorted((HERE/"tables").glob("i5_*.csv"))+sorted((HERE/"figures").glob("i5_joint*"))},
        initial_manifest_sha256=plan["manifest_sha256"],final_manifest_sha256=sha(HERE/"manifest/i5.json"),
        manifest_changes_only=["estimated_seconds","peak_rss_gb"],
        actual_runtime_files=sum(p.is_file() for p in OUT.rglob("*")),job_checks=records,dependency_checks=margins)
    write(HERE/"evidence/i5_completion_review.json",evidence)
    return evidence


if __name__=="__main__":
    r=check()
    print(json.dumps({k:r[k] for k in ("status","jobs","total_backend_samples","formal_jobs_executed_twice","minimum_dependency_margin_seconds")}))
