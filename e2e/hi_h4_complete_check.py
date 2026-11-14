"""Independently audit saved H4 PF4 outputs; do not call reporting or simulation code."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import statistics

from hi_next_check import read, sha, SEEDS
from hi_h3_i2_check import write

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"e2e/hi"
OUT=HERE/"out/h4_pf4"
PF4=[[[0,11],[1,8]],[[1,8],[0,11]]]


def load(path):
    with gzip.open(path,"rb") as stream:
        return pickle.load(stream)


def absolute(value):
    p=Path(value.replace("\\","/"))
    return p if p.is_absolute() else ROOT/p


def receipt_identity(job):
    payload={k:job[k] for k in ("id","kind","spec","result")}
    payload["argv"]=job["argv"]
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def check():
    plan=read(HERE/"evidence/h4_pf4_plan_review.json")
    manifest=HERE/"manifest/h4_pf4.json"
    assert sha(manifest)==plan["manifest_sha256"]
    input_path=HERE/"manifest/h4_healthy_inputs.json"
    assert sha(input_path)==plan["input_inventory_sha256"]
    inputs=read(input_path)["candidates"]
    assert len(inputs)==224
    healthy_records={}
    for name,row in inputs.items():
        for kind in ("record","raw","snapshot"):
            assert sha(absolute(row[kind]))==row["sha256"][kind],(name,kind)
        healthy_records[name]=read(absolute(row["record"]))
    launch=read(OUT/"h4_pf4.start.json")
    source_evidence={}
    for name,digest in launch["source_sha256"].items():
        archived=HERE/"runtime/h4_pf4_20261114"/name
        assert sha(archived)==digest and sha(ROOT/"e2e"/name)==digest
        assert next(v for k,v in plan["source_sha256"].items() if Path(k).name==name)==digest
        source_evidence[name]=dict(sha256=digest,archived_path=archived.relative_to(ROOT).as_posix())
    assert sha(ROOT/"eir/eir.py")==plan["source_sha256"]["eir\\eir.py"]
    jobs={j["id"]:j for j in read(manifest)["jobs"]}
    assert len(jobs)==264
    for folder in ("results","receipts","specs"):
        assert {p.stem for p in (OUT/folder).glob("*.json")}==set(jobs)
    results={n:read(OUT/j["result"]) for n,j in jobs.items()}
    receipts={n:read(OUT/"receipts"/(n+".json")) for n in jobs}
    assert not list((OUT/"failed_attempts").glob("*/*/receipt.json"))
    controller_lines=[json.loads(line) for line in (OUT/"h4_pf4.log").read_text(encoding="utf-8").splitlines() if line.startswith('{"event":')]
    done=[r for r in controller_lines if r["event"]=="done"]
    assert len(done)==len({r["id"] for r in done})==264
    assert {r["id"] for r in done}==set(jobs)
    for row in done:assert {k:v for k,v in row.items() if k!="event"}==receipts[row["id"]]
    starts,ends,log_evidence={}, {}, []
    for name,job in jobs.items():
        spec,result,receipt=job["spec"],results[name],receipts[name]
        assert read(OUT/"specs"/(name+".json"))==spec
        assert result["job_id"]==name and result["kind"]==job["kind"]
        assert receipt["id"]==name and receipt["rc"]==0 and receipt["identity"]==receipt_identity(job)
        assert receipt["result"]==job["result"] and receipt["wall_seconds"]>=result["wall_seconds"]
        assert not receipt["cached"]
        paths=list((OUT/"logs").glob(name+".*.log"))
        assert len(paths)==1,(name,len(paths))
        path=paths[0];content=path.read_text(encoding="utf-8")
        if result["status"]=="ok":
            assert result["spec"]==spec
            assert content.splitlines()[-1]==f"[fg_fault] {name} ok {result['wall_seconds']:.2f}s"
        else:
            assert result["status"]=="reused" and job["kind"]=="fault_rep"
            previous=results[spec["previous_stop"]]
            assert previous["stop"] and result["previous_stop"]==spec["previous_stop"]
            assert result["export"]==previous["export"] and absolute(result["export"]).exists()
            assert content=="" and not (OUT/"raw"/(name+".pkl.gz")).exists()
            assert not list((OUT/"exports").glob(name+"__*.pkl.gz"))
            assert "backend" not in result and "sink" not in result
        starts[name]=int(path.stem.rsplit(".",1)[1])/1e9
        ends[name]=starts[name]+receipt["wall_seconds"]
        log_evidence.append(dict(job_id=name,status=result["status"],receipt_identity=receipt["identity"],log=path.relative_to(ROOT).as_posix(),log_bytes=path.stat().st_size,result_sha256=sha(OUT/job["result"])))
    assert len(list((OUT/"logs").glob("*.log")))==264
    dependency_rows=[]
    for name,job in jobs.items():
        for parent in job["deps"]:
            margin=starts[name]-ends[parent]
            assert margin>=-2.1,(name,parent,margin)
            dependency_rows.append(dict(job_id=name,parent=parent,start_minus_parent_receipt_seconds=margin))

    selected={}
    for job in jobs.values():
        if job["kind"]!="fault_selection":continue
        rows=defaultdict(list)
        for candidate in job["spec"]["candidate_ids"]:
            r=healthy_records[candidate]
            rows[r["phase"],r["op"],r["method"]].append(r)
        saved=read(OUT/"selection"/(job["spec"]["config_id"]+".json"))
        assert saved["selection"]=="backend_mean"
        for (phase,op,arm),items in rows.items():
            assert len(items)==14
            mean=lambda r:statistics.mean(v["cycles"] for v in r["backend"])
            winner=min(items,key=lambda r:(mean(r),r["candidate_order"]))
            alternative=min(items,key=lambda r:(r["T_hat_rand_mean"],r["candidate_order"]))
            phase_op=("pf_" if phase=="prefill" else "dc_")+op
            value=saved["operators"][phase_op][arm]
            assert value["candidate_id"]==winner["candidate_id"] and value["random_CPCM_pick"]==alternative["candidate_id"]
            assert value["backend_mean"]==mean(winner)
            assert all(value[k]==inputs[winner["candidate_id"]][k] for k in ("record","raw","snapshot"))
            selected[job["spec"]["config_id"],phase_op,arm]=winner["candidate_id"]
    assert len(selected)==16
    for job in jobs.values():
        if job["kind"]!="fault_select":continue
        r=results[job["id"]]
        assert r["anchor_backend_calls"]==0 and set(r["faults"])=={"PF4"}
        f=r["faults"]["PF4"]
        assert f["failed_links"]==PF4 and f["failed_nodes"]==f["failed_pes"]==[]
        assert r["anchor"]["candidate_id"]==selected[job["spec"]["config_id"],"dc_mha","A3h"]
    import post_fg_split
    post_fg_split.setup()  # Register the classes needed by existing pickle files.
    states,samples,raw_evidence,export_paths={}, {}, [], set()
    actual_rep,reused_rep=0,0
    for name,job in jobs.items():
        kind,s,r=job["kind"],job["spec"],results[name]
        if kind not in ("fault_pre","fault_l1","fault_rep"):continue
        if r["status"]=="reused":reused_rep+=1;continue
        source=selected[s["config_id"],s["phase_op"],s["start_arm"]]
        assert r["source_candidate"]==source
        path=OUT/"raw"/(name+".pkl.gz")
        assert absolute(r["sink"])==path
        raw=load(path)
        assert raw["backend_seeds"]==SEEDS and raw["spec"]["checks"] is False
        assert raw["spec"]["name"]==name and raw["spec"]["model"]=="llama"
        if kind=="fault_pre":
            assert raw["faults"]=={"failed_nodes":[],"failed_links":[]}
            entries={"healthy":raw["record"]}
            assert r["T"]==raw["T"]==healthy_records[source]["T_hat"]
        else:
            assert raw["faults"]["failed_links"]==PF4
            assert raw["faults"]["failed_nodes"]==raw["faults"]["failed_pes"]==[]
            if kind=="fault_l1":
                entries={"R0":raw["R0"]}
                assert r["T_R0"]==raw["T_R0"]==raw["R0"]["T"]
                if s["start_arm"]=="A3h":
                    entries["L1"]=raw["L1"]
                    assert r["L1_evaluated"] and r["T_L1"]==raw["T_L1"]==raw["L1"]["T"]
                else:
                    assert not r["L1_evaluated"] and "L1" not in raw and set(r["exports"])=={"R0"}
            else:
                actual_rep+=1
                entries={"final":raw["record"]}
                assert not results[s["previous_stop"]]["stop"]
                assert raw["spec"]["method"]=="L2Gsync" and raw["spec"]["seed"]==s["seed"]
                assert [v["step"] for v in raw["search"]["rows"]]==list(range(1,s["steps"]+1))
                l1_name=Path(s["l1_sink"]).name.removesuffix(".pkl.gz")
                assert raw["T_L1_check"]==raw["search"]["T_start"]==results[l1_name]["T_L1"]
                assert raw["search"]["best_T"]==r["T"]==raw["record"]["T"]
                assert raw["proposals"]["steps"]==s["steps"] and raw["search"]["check_revert"]["n"]==0
        assert raw["exports"]==r["exports"] and set(entries)==set(r["exports"])
        for label,state in entries.items():
            assert set(state["backend"])==set(SEEDS) and len(state["backend_wall"])==10
            assert not state["fault_violations"]
            values=[state["backend"][seed][0] for seed in SEEDS]
            mean=statistics.mean(values)
            field=label+"_backend_mean" if label in ("R0","L1") else "backend_mean"
            assert mean==r[field]
            export_path=absolute(r["exports"][label])
            exp=load(export_path)
            assert exp["spec"]==s and exp["T"]==state["T"] and exp["n_events"]>0
            assert isinstance(exp["events_blob"],bytes) and len(pickle.loads(exp["events_blob"]))==exp["n_events"]
            export_paths.add(export_path)
            states[name,label]=dict(T=state["T"],backend_mean=mean,backend_min=min(values),backend_max=max(values),export=r["exports"][label],sha256=sha(export_path))
            for seed in SEEDS:samples[name,label,seed]=tuple(state["backend"][seed])
        raw_evidence.append(dict(job_id=name,raw_sha256=sha(path),states=list(entries),backend_samples=10*len(entries),search_steps=s["steps"] if kind=="fault_rep" else 0))
    assert len(states)==79 and len(samples)==790 and actual_rep==35 and reused_rep==145
    assert {p for p in (OUT/"exports").glob("*.pkl.gz")}==export_paths
    assert {p.name for p in (OUT/"raw").glob("*.pkl.gz")}=={r["job_id"]+".pkl.gz" for r in raw_evidence}
    stop_rows=[]
    for name,job in jobs.items():
        if job["kind"]!="fault_stop":continue
        s,r=job["spec"],results[name]
        propagated=False
        if s["steps"]==0:
            base=results[s["l1_job"]]
            level="R0" if base["T_R0"]<=1.02*base["T_pre"] else "L1"
            t=base["T_"+level]
            expected=dict(stop=t<=1.02*base["T_pre"],level=level,T=t,T_pre=base["T_pre"],export=base["exports"][level],source_job=s["l1_job"],reason="within_epsilon" if t<=1.02*base["T_pre"] else "continue")
        else:
            prior=results[s["previous_stop"]]
            members=[results[n] for n in s["rep_jobs"]]
            if prior["stop"]:
                propagated=True
                assert all(v["status"]=="reused" for v in members)
                expected={k:prior[k] for k in ("stop","level","T","T_pre","export","source_job","reason")}
                for key in ("seed","backend_mean","gain_from_previous"):
                    if key in prior:expected[key]=prior[key]
            else:
                assert len(members)==5 and {m["seed"] for m in members}=={123,1,2,3,4}
                assert all(m["status"]=="ok" for m in members)
                t=statistics.mean(m["T"] for m in members)
                gain=None if s["steps"]==30 else (prior["T"]-t)/prior["T"]
                epsilon=t<=prior["T_pre"]*1.02
                plateau=gain is not None and gain<.02
                median=statistics.median(m["backend_mean"] for m in members)
                winner=min((m for m in members if m["backend_mean"]==median),key=lambda m:m["seed"])
                expected=dict(stop=epsilon or plateau or s["steps"]==300,level="s"+str(s["steps"]),T=t,T_pre=prior["T_pre"],export=winner["exports"]["final"],source_job=winner["job_id"],seed=winner["seed"],backend_mean=winner["backend_mean"],gain_from_previous=gain,reason="within_epsilon" if epsilon else "tier_gain" if plateau else "budget" if s["steps"]==300 else "continue")
        assert all(r[k]==v for k,v in expected.items()),name
        stop_rows.append(dict(job_id=name,config_id=s["config_id"],phase_op=s["phase_op"],budget=s["steps"],propagated=propagated,**expected))
    assert len(stop_rows)==48 and sum(r["propagated"] for r in stop_rows)==29
    finals=[]
    for name,job in jobs.items():
        if job["kind"]!="fault_l1":continue
        s=job["spec"]
        if s["start_arm"]=="A1":
            state=states[name,"R0"]
            finals.append(dict(config_id=s["config_id"],phase_op=s["phase_op"],start_arm="A1",level="R0",source_job=name,**state))
            continue
        terminal=[r for r in stop_rows if r["config_id"]==s["config_id"] and r["phase_op"]==s["phase_op"] and r["stop"] and not r["propagated"]]
        assert len(terminal)==1
        final=terminal[0]
        label=final["level"] if final["level"] in ("R0","L1") else "final"
        state=states[final["source_job"],label]
        assert final["export"]==state["export"]
        finals.append(dict(config_id=s["config_id"],phase_op=s["phase_op"],start_arm="A3h",level=final["level"],source_job=final["source_job"],decision_mean_T=final["T"],repair_seed=final.get("seed"),**state))
    assert Counter(r["start_arm"] for r in finals)=={"A3h":12,"A1":4}
    # Compare report output only after all expected values have been derived above.
    with (HERE/"tables/h4_backend_samples.csv").open(encoding="utf-8-sig",newline="") as stream:
        reported=list(csv.DictReader(stream))
    assert len(reported)==790
    for row in reported:
        label="final" if row["state"]=="L2" else row["state"]
        expected=samples[row["job_id"],label,int(row["backend_seed"])]
        assert tuple(float(row[k]) for k in ("backend_ns","unhidden_compute_ns","unhidden_communication_ns"))==expected
    with (HERE/"tables/h4_units.csv").open(encoding="utf-8-sig",newline="") as stream:
        units={(r["config_id"],r["phase_op"],r["start_arm"]):r for r in csv.DictReader(stream)}
    assert len(units)==16
    for final in finals:
        row=units[final["config_id"],final["phase_op"],final["start_arm"]]
        if final["start_arm"]=="A1":
            assert float(row["R0_backend_mean_ns"])==final["backend_mean"]
        else:
            assert row["EIR_source_job"]==final["source_job"] and row["EIR_export"]==final["export"]
            assert float(row["EIR_stop_mean_T_hat_ns"])==final["decision_mean_T"]
            assert float(row["EIR_selected_T_hat_ns"])==final["T"] and float(row["EIR_backend_mean_ns"])==final["backend_mean"]
    batch=read(OUT/"h4_pf4.batch.json")
    finish=read(OUT/"controller_finish.json")
    assert batch["completed"]==264 and batch["failed"]==batch["pending"]==[]
    assert len(batch["runs"])==264 and {r["id"] for r in batch["runs"]}==set(jobs)
    assert finish["rc"]==0 and finish["completed"]==finish["expected"]==264
    statuses=[r for r in controller_lines if r["event"]=="status"]
    assert all(r["active"]+r["parent_reserved_slots"]<=128 for r in statuses)
    evidence=dict(status="passed",created_utc=datetime.now(timezone.utc).isoformat(),scope="Independent saved-output audit; no report parser reused and no simulation.",
        jobs=264,failed_jobs=0,actual_repair_trajectories=actual_rep,reused_repair_jobs=reused_rep,
        actual_backend_samples=len(samples),evaluated_states=len(states),raw_files=len(raw_evidence),export_files=len(export_paths),
        unique_successful_job_logs=264,nonempty_evaluated_job_logs=119,empty_stopped_job_logs=145,
        empty_log_explanation="Stopped fault_rep jobs execute the frozen early-return branch: write a reused result and return before its print. Each has one rc=0 matching receipt and controller done record, no raw file and no new export/backend samples.",
        original_healthy_inputs=224,original_healthy_files_sha256_verified=672,healthy_groups=16,
        a3h_final_states=12,a1_r0_states=4,stop_records=48,evaluated_stop_records=19,propagated_stop_records=29,
        checked_dependency_edges=len(dependency_rows),minimum_dependency_margin_seconds=min(r["start_minus_parent_receipt_seconds"] for r in dependency_rows),
        dependency_violations=0,dependency_tolerance_seconds=2.1,
        time_definition="Filename Unix_ns is sampled just before Popen; receipt.wall_seconds starts just after Popen and includes controller polling. Their sum approximates the successful receipt time; 2.1 seconds permits polling/bookkeeping differences.",
        source_identity=source_evidence,model_contract_sha256=next(iter(jobs.values()))["spec"]["model_contract_sha256"],
        peak_own_active=batch["max_active"],minimum_sampled_available_gib=batch["min_available_gib"],
        parent_slot_samples=len(statuses),max_parent_reserved_plus_own_active=max(r["active"]+r["parent_reserved_slots"] for r in statuses),
        run_wall_seconds=batch["wall_seconds"],runtime_files=sum(p.is_file() for p in OUT.rglob("*")),
        report_checks=dict(backend_sample_rows=790,unit_rows=16,final_representative_states=12),
        remains="MF6/Split is not part of this completed PF4 batch.",
        final_states=finals,stops=stop_rows,candidates=log_evidence,raw_outputs=raw_evidence,dependency_checks=dependency_rows)
    write(HERE/"evidence/h4_completion_review.json",evidence)
    return evidence


if __name__=="__main__":
    e=check()
    print(json.dumps({k:e[k] for k in ("status","jobs","actual_repair_trajectories","reused_repair_jobs","actual_backend_samples","minimum_dependency_margin_seconds")}))
