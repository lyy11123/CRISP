"""Independently review I5 inputs, 63 joint units and the formal pilot closure."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import inspect
import json
from pathlib import Path
import pickle
import statistics
from hi_next_check import read,sha
from hi_h3_i2_check import write

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"e2e/hi"


def load(path):
    with gzip.open(path,"rb") as stream:return pickle.load(stream)


def check():
    import hi_i5
    import hi_joint
    import hi_config
    import fg_joint
    manifest=read(hi_i5.MANIFEST)
    jobs={j["id"]:j for j in manifest["jobs"]}
    assert len(jobs)==len(manifest["jobs"])==110
    assert Counter(j["kind"] for j in jobs.values())=={"hi_joint_export":2,"joint_graph":5,"joint_solo":39,"joint_reference":1,"joint":63}
    inputs=read(hi_i5.INPUTS)
    assert inputs["model"]=="qwen3moe" and inputs["new_healthy_searches"]==0
    contract=hi_config.model_contract("qwen3moe")
    digest=hi_config.digest(contract)
    assert contract["head_groups"]==4 and contract["decode_ffn_includes_rmsnorm"]
    assert hi_joint.decode_chain_order("qwen3moe")==["dc_ln"]+["dc_mha"]*4+["dc_proj","dc_ffn"]
    previous={r["job_id"]:r for r in read(HERE/"evidence/i4_completion_review.json")["candidates"]}
    healthy=[]
    for folder in ("h1_main","next_344","h3_i2_440","i3_108"):
        for p in (HERE/"out"/folder/"results").glob("I_qwen3moe_busybarn_prefill_mha_*.json"):
            r=read(p)
            if r["kind"]=="search" and r["method"] in ("A1","A3h"):healthy.append(r)
    assert len(healthy)==28
    import post_fg_split
    post_fg_split.setup()  # Existing pickle classes only, no model construction.
    p_checks=[]
    for arm,source in inputs["P"].items():
        group=[r for r in healthy if r["method"]==arm]
        assert len(group)==14
        chosen=min(group,key=lambda r:(statistics.mean(x["cycles"] for x in r["backend"]),r["candidate_order"]))
        assert chosen["candidate_id"]==source["candidate_id"]
        for kind in ("record","raw","snapshot"):assert sha(ROOT/source[kind])==source["sha256"][kind]
        raw=load(ROOT/source["raw"])
        blob,_=load(ROOT/source["snapshot"])
        behaviours,data,events=pickle.loads(blob)
        tags=raw["meta"]["ctx"]["broadcast_datatags"]
        assert len(tags)==2
        tensors=[data[tuple(t)] for t in tags]
        assert [t.data_name for t in tensors]==["MHA_ktrans_ofmap","MHA_vconv_ofmap"]
        assert [t.data_shape for t in tensors]==[[16,64,2048],[16,2048,64]]
        assert sum(t.data_size for t in tensors)==8388608
        p_checks.append(dict(arm=arm,candidate_id=source["candidate_id"],candidates=14,kv_logical_bytes=8388608,tensor_names=[t.data_name for t in tensors]))
    assert set(inputs["D"])=={"healthy_A1","healthy_A3h","D_A1","D_A3hR0","D_A3hEIR"}
    d_checks=[]
    for state,operators in inputs["D"].items():
        assert set(operators)=={"dc_ln","dc_mha","dc_proj","dc_ffn"}
        for op,source in operators.items():
            path=ROOT/source["source_job"];job_id=path.stem
            assert sha(path)==previous[job_id]["result_sha256"]
            record=read(path)
            assert record["status"]=="ok"
            if state.startswith("healthy_"):
                assert record["kind"]=="fault_pre" and record["spec"]["start_arm"]==state.split("_")[1]
                assert source["file"]==record["exports"]["healthy"]
            elif state=="D_A3hEIR":
                assert record["kind"]=="fault_stop" and record["stop"] and record["spec"]["steps"]==300
                assert source["file"]==record["export"]
            else:
                assert record["kind"]=="fault_l1" and source["file"]==record["exports"]["R0"]
                assert record["spec"]["start_arm"]==("A1" if state=="D_A1" else "A3h")
            assert sha(ROOT/source["file"])==source["sha256"]
            export=load(ROOT/source["file"])
            assert export["spec"]["model"]=="qwen3moe" and export["spec"]["model_contract_sha256"]==digest
            assert export["spec"]["phase_op"]==op and export["n_events"]>0
            d_checks.append(dict(state=state,operator=op,file=source["file"],source_job=job_id,sha256=source["sha256"]))
    local_fault=read(ROOT/inputs["fault_file"])["faults"]["PF4"]
    assert local_fault["failed_links"]==[[[0,11],[1,8]],[[1,8],[0,11]]]
    maps=fg_joint.maps("2x2")
    assert maps=={"L":{0:0,1:1,2:4,3:5},"R":{0:2,1:3,2:6,3:7}}
    assert set(maps["L"].values()).isdisjoint(maps["R"].values())
    remapped=[[[maps["R"][a[0]],a[1]],[maps["R"][b[0]],b[1]]] for a,b in local_fault["failed_links"]]
    assert remapped==[[[2,11],[3,8]],[[3,8],[2,11]]]
    graphs={j["spec"]["mapping_state"]:j for j in jobs.values() if j["kind"]=="joint_graph"}
    assert set(graphs)==set(inputs["D"])
    for name,j in jobs.items():
        s=j["spec"]
        assert s["model"]=="qwen3moe" and s["model_contract_sha256"]==digest
        assert s["shape"]=="2x2" and s["hardware_profile"]=="busybarn"
        assert s["job_id"]==name and s["kind"]==j["kind"] and j["deps"]==s["depends_on"]
        assert set(j["deps"])<=set(jobs)
        assert j["argv"]==["{python}","-u","-B","e2e/hi_i5.py","child","--spec","{spec}","--out","{out}"]
        assert j["result"]==f"results/{name}.json"
        if j["kind"]=="joint_graph":
            state=s["mapping_state"]
            assert s["condition"]==("D" if state.startswith("D_") else "healthy")
            assert set(s["sources"])=={"P","dc_ln","dc_mha","dc_proj","dc_ffn"}
            for op in fg_joint.DC:assert s["sources"][op]["file"]==inputs["D"][state][op]["file"]
            exporter=jobs[s["sources"]["P"]["job"]]
            assert exporter["kind"]=="hi_joint_export" and exporter["spec"]["mapping"]==("A1" if state.endswith("A1") else "A3h")
            assert j["deps"]==[exporter["id"]]
    reference=next(j for j in jobs.values() if j["kind"]=="joint_reference")
    for stream,key in ((0,"P_solos"),(1,"D_solos")):
        members=[jobs[n] for n in reference["spec"][key]]
        assert len(members)==3 and {m["spec"]["seed"] for m in members}=={1,2,3}
        assert all(m["spec"]["graph_job"]==graphs["healthy_A1"]["id"] and m["spec"]["backend"]=="orig" and m["spec"]["stream"]==stream for m in members)
    expected_arms={"A1_no_DSTU":("healthy_A1","orig"),"A3h_no_DSTU":("healthy_A3h","orig"),"A3h_A256":("healthy_A3h","hop"),"A1_R0_no_DSTU":("D_A1","orig"),"A3h_R0_no_DSTU":("D_A3hR0","orig"),"A3h_EIR_no_DSTU":("D_A3hEIR","orig"),"A3h_EIR_A256":("D_A3hEIR","hop")}
    joint_specs=[j["spec"] for j in jobs.values() if j["kind"]=="joint"]
    assert {(s["arm"],s["phase"],s["seed"]) for s in joint_specs}=={(a,g,seed) for a in expected_arms for g in (.1,.5,.9) for seed in (1,2,3)}
    for s in joint_specs:
        state,backend=expected_arms[s["arm"]]
        assert s["graph_job"]==graphs[state]["id"] and s["reference_job"]==reference["id"]
        assert s["backend"]==backend and s["policy"]==("C4" if backend=="hop" else "P1")
        for stream,key in ((0,"P_solo"),(1,"D_solo")):
            solo=jobs[s[key]]["spec"]
            assert solo["backend"]==backend and solo["stream"]==stream and solo["seed"]==s["seed"]
            if solo["graph_job"]!=s["graph_job"]:
                assert stream==0 and state=="D_A3hEIR" and backend=="orig"
                assert solo["graph_job"]==graphs["D_A3hR0"]["id"]
    assert len(manifest["solo_aliases"])==3 and manifest["logical_paired_solos"]==42 and manifest["actual_paired_solos"]==39
    done=set()
    while len(done)<110:
        ready={n for n,j in jobs.items() if n not in done and set(j["deps"])<=done}
        assert ready
        done.update(ready)
    pilot=read(hi_i5.PILOT)
    target=pilot["target"]
    assert jobs[target]["spec"]["arm"]=="A3h_A256" and jobs[target]["spec"]["phase"]==.5 and jobs[target]["spec"]["seed"]==1
    closure=set()
    def visit(n):
        if n in closure:return
        closure.add(n)
        for dep in jobs[n]["deps"]:visit(dep)
    visit(target)
    assert len(pilot["jobs"])==len(closure)==14 and {j["id"] for j in pilot["jobs"]}==closure
    assert all(j==jobs[j["id"]] for j in pilot["jobs"])
    assert Counter(j["kind"] for j in pilot["jobs"])=={"hi_joint_export":2,"joint_graph":2,"joint_solo":8,"joint_reference":1,"joint":1}
    assert [len(read(m)["jobs"]) for m,_,_ in hi_i5.PARENTS]==[440,108,264,201,23,110]
    controller=hi_i5.Controller(hi_i5.OUT,128,hi_i5.PILOT)
    assert len(controller.jobs)==14 and set(controller.by_id)==closure and controller.out==hi_i5.OUT
    assert all(p["out"]!=hi_i5.OUT for p in controller.parents)
    source=(ROOT/"e2e/hi_i5.py").read_text(encoding="utf-8")
    assert 'result["actual_iteration_sha256"]' in source and 'paired["actual_iteration_sha256"]' in source
    assert "model_joint.redirect_kv" in source and 'bundle["kv_tags"]' in source
    assert "frozen.child(spec,out)" in source and "frozen.graph_job(spec,out)" in source
    assert 'archive=args.out/"pilot_run"' in source and 'source.rename(target)' in source
    assert '"controller_finish.json"' in source and '"i5.batch.json"' in source
    runtime=HERE/"dist/server_runtime/e2e"
    frozen_sha={}
    for name in ("fg_joint.py","d_joint.py"):
        assert sha(ROOT/"e2e"/name)==sha(runtime/name)
        frozen_sha[name]=sha(runtime/name)
    result=dict(status="passed",created_utc=datetime.now(timezone.utc).isoformat(),scope="Static entry and existing input review; no graph construction or backend run.",
        jobs=110,counts={"P_exports":2,"graphs":5,"actual_paired_solos":39,"common_reference":1,"joint_units":63},
        logical_paired_solos=42,aliased_identical_P_solos=3,phases=[.1,.5,.9],backend_seeds=[1,2,3],
        model_contract_sha256=digest,decode_chain=hi_joint.decode_chain_order("qwen3moe"),decode_norms="one separate RMSNorm plus FFN's embedded second RMSNorm",
        P_inputs=p_checks,D_inputs=d_checks,P_group=maps["L"],D_group=maps["R"],D_fault_links=remapped,
        cross_stream_causality="No P-to-D dependency inserted: frozen graph_job builds two separate streams; only D operator barriers are added.",
        common_reference="Healthy A1/original, mean of three P and three D solos; phase release, public deadlines and normalization retained for all arms.",
        C4_internal_priority="Frozen C4 ranking still uses each graph's own hop CPCM estimates; public Lhat reference remains the common A1/original values.",
        DSTU="All-link 256 KiB plus frozen C4 hop backend; no-DSTU retains original whole-message backend.",
        pilot_jobs=14,pilot_actual_backend_calls=9,pilot_target=target,pilot_formal_identity_equal=True,
        pilot_and_full_out_same=True,pilot_control_records_archived_on_full_launch=True,
        pilot_receipts_results_preserved_for_full_reuse=True,parent_job_counts=[440,108,264,201,23,110],workers_limit=128,reserve_gib=3,
        solo_reuse_runtime_check="platform hash, canonical stream fields and actual dependency/issue/target-set iteration hash must all match",
        frozen_joint_backend_sha256=frozen_sha,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"e2e/hi_i5.py",ROOT/"e2e/hi_joint.py",ROOT/"e2e/hi_i4.py",ROOT/"e2e/hi_h4.py")},
        manifest_sha256=sha(hi_i5.MANIFEST),pilot_manifest_sha256=sha(hi_i5.PILOT),input_inventory_sha256=sha(hi_i5.INPUTS),
        pending="One formal healthy A256 pilot checks actual graph/KV/order equality and timing before the remaining formal jobs.")
    write(HERE/"evidence/i5_plan_review.json",result)
    return result


if __name__=="__main__":
    r=check()
    print(json.dumps({k:r[k] for k in ("status","jobs","pilot_jobs","pilot_actual_backend_calls")}))
