"""Read-only H4 PF4 input, repair protocol and shared-pool review."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

from hi_next_check import read, sha, KEYS, SEEDS
from hi_h3_i2_check import write

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "e2e/hi"


def stop_semantics():
    """Call the frozen arithmetic stop function only; no model or backend."""
    import fg_fault
    records = {}
    original = fg_fault.read_json
    fg_fault.read_json = lambda path: records[Path(path).stem]
    rows = []
    try:
        for name,r0,l1,expected in (("zero_L1_gain",120,120,False),("small_L1_gain",120,119,False),("R0_within_epsilon",101,100,True),("L1_within_epsilon",120,101,True)):
            records["base"] = dict(T_pre=100,T_R0=r0,T_L1=l1,exports={"R0":"r0","L1":"l1"})
            result = fg_fault.stop_job(dict(steps=0,l1_job="base"),HERE)
            assert result["stop"] is expected
            rows.append(dict(case=name,result=result))
        for steps,previous_T,current_T,expected,reason in ((30,120,119,False,"continue"),(100,110,108.5,True,"tier_gain"),(100,110,101,True,"within_epsilon"),(300,110,105,True,"budget")):
            records["previous"] = dict(stop=False,T=previous_T,T_pre=100)
            names = []
            for index,seed in enumerate((123,1,2,3,4)):
                name=f"r{seed}"
                records[name]=dict(job_id=name,T=current_T,seed=seed,backend_mean=10+index,exports={"final":name})
                names.append(name)
            result=fg_fault.stop_job(dict(steps=steps,previous_stop="previous",rep_jobs=names),HERE)
            assert result["stop"] is expected and result["reason"]==reason and result["seed"]==2
            rows.append(dict(case=f"s{steps}_{reason}",result=result))
    finally:
        fg_fault.read_json=original
    return rows


def plan_check():
    import hi_h4
    import fg_fault
    jobs=read(HERE/"manifest/h4_pf4.json")["jobs"]
    by_id={j["id"]:j for j in jobs}
    counts=Counter(j["kind"] for j in jobs)
    assert len(jobs)==len(by_id)==264
    assert counts=={"fault_selection":2,"fault_select":2,"fault_pre":16,"fault_l1":16,"fault_stop":48,"fault_rep":180}
    all_specs={s["candidate_id"]:s for s in read(HERE/"manifest/all_candidates_planned.json")["jobs"]}
    inputs=read(HERE/"manifest/h4_healthy_inputs.json")["candidates"]
    wanted={n for j in jobs if j["kind"]=="fault_selection" for n in j["spec"]["candidate_ids"]}
    assert len(wanted)==224 and set(inputs)==wanted
    expected_inputs={n for n,s in all_specs.items() if s["model"]=="llama" and
        (s["method"]=="A3h" or (s["hardware_profile"]=="busybarn" and s["phase"]=="decode" and s["method"]=="A1"))}
    assert wanted==expected_inputs
    groups=defaultdict(list)
    for name,entry in inputs.items():
        record=read(ROOT/entry["record"])
        assert record["status"]=="ok" and all(record[k]==v for k,v in all_specs[name].items())
        assert [r["seed"] for r in record["backend"]]==SEEDS and len(record["T_hat_rand"])==10
        for kind in ("record","raw","snapshot"):
            assert sha(ROOT/entry[kind])==entry["sha256"][kind],(name,kind)
        groups[tuple(record[k] for k in KEYS)].append(record)
    assert len(groups)==16 and all(len(rows)==14 for rows in groups.values())
    selections=[]
    with hi_h4.healthy_routes():
        for key,rows in sorted(groups.items()):
            winner=min(rows,key=lambda r:(sum(s["cycles"] for s in r["backend"])/10,r["candidate_order"]))
            random_winner=min(rows,key=lambda r:(r["T_hat_rand_mean"],r["candidate_order"]))
            result=fg_fault.choose_candidate(rows,HERE/"out/h4_pf4")
            assert result["candidate_id"]==winner["candidate_id"] and result["random_CPCM_pick"]==random_winner["candidate_id"]
            assert all(result[k]==inputs[winner["candidate_id"]][k] for k in ("record","raw","snapshot"))
            selections.append(dict(category=list(key),candidate_id=winner["candidate_id"],random_CPCM_pick=random_winner["candidate_id"]))
    a3h,a1=[],[]
    for job in jobs:
        spec=job["spec"]
        assert spec["job_id"]==job["id"] and spec["kind"]==job["kind"]
        assert spec["model"]=="llama" and spec.get("fault_id") in (None,"PF4")
        assert spec["backend_seeds"]==SEEDS and job["deps"]==spec["depends_on"]
        assert set(job["deps"])<=set(by_id) and job["id"] not in job["deps"]
        assert job["argv"]==["{python}","-u","-B","e2e/hi_h4.py","child","--spec","{spec}","--out","{out}"]
        assert job["result"]==f"results/{job['id']}.json"
        if job["kind"]=="fault_selection":assert not job["deps"]
        if job["kind"]=="fault_l1":
            (a3h if spec["start_arm"]=="A3h" else a1).append(job)
        if job["kind"]=="fault_rep":
            assert spec["start_arm"]=="A3h" and spec["steps"] in (30,100,300) and spec["seed"] in (123,1,2,3,4)
            assert job["deps"]==[spec["previous_stop"]]
            assert spec["l1_sink"].endswith(f"l1_{spec['phase_op']}_A3h_PF4.pkl.gz")
    assert len(a3h)==12 and len(a1)==4
    assert {(j["spec"]["phase"],j["spec"]["op"]) for j in a1}=={("decode",op) for op in ("ln","mha","proj","ffn")}
    assert all(j["spec"]["hardware_profile"]=="busybarn" for j in a1)
    for start in a3h:
        s=start["spec"]
        repairs=[j for j in jobs if j["kind"]=="fault_rep" and j["spec"]["config_id"]==s["config_id"] and j["spec"]["phase_op"]==s["phase_op"]]
        assert len(repairs)==15 and {(j["spec"]["steps"],j["spec"]["seed"]) for j in repairs}=={(n,r) for n in (30,100,300) for r in (123,1,2,3,4)}
        assert len({j["spec"]["l1_sink"] for j in repairs})==1
        for steps in (0,30,100,300):
            stops=[j for j in jobs if j["kind"]=="fault_stop" and j["spec"]["config_id"]==s["config_id"] and j["spec"]["phase_op"]==s["phase_op"] and j["spec"]["steps"]==steps]
            assert len(stops)==1
            if steps==0:assert stops[0]["deps"]==[start["id"]]
            else:assert set(stops[0]["deps"])=={j["id"] for j in repairs if j["spec"]["steps"]==steps}
    # Kahn traversal independently rules out cycles in the repair dependencies.
    done=set()
    while len(done)<264:
        ready={n for n,j in by_id.items() if n not in done and set(j["deps"])<=done}
        assert ready
        done.update(ready)
    source=(ROOT/"e2e/hi_h4.py").read_text(encoding="utf-8")
    wrapped=hi_h4.wrapped_source()
    compile(wrapped,"<H4 independent review>","exec")
    assert 'spec["job_id"]' in inspect.getsource(hi_h4.Controller.spawn)
    assert "STDOUT=subprocess.STDOUT" in source
    assert 'namespace["execute"](self.manifest,self.out,self.workers,3,NAME,4,self.release)' in source
    assert "external_slots = _hi.reserve_parents()" in wrapped and "len(running) + external_slots >= workers" in wrapped
    assert "budget = free - reserve_gib - reserved_growth - _hi.parent_growth_reserve_gib" in wrapped
    assert [len(read(m)["jobs"]) for m,_,_ in hi_h4.PARENTS]==[440,108]
    assert all(o!=HERE/"out/h4_pf4" for _,o,_ in hi_h4.PARENTS)
    assert "[[[0, 11], [1, 8]], [[1, 8], [0, 11]]]" in (ROOT/"e2e/hi_fault.py").read_text(encoding="utf-8")
    assert 'anchor_backend_calls=0' in inspect.getsource(hi_h4.fixed_pf4)
    assert 'if spec["start_arm"]=="A1"' in inspect.getsource(fg_fault.engine_job)
    assert 'inner.update(method="L2Gsync"' in inspect.getsource(fg_fault.engine_job)
    stop_checks=stop_semantics()
    evidence=read(HERE/"evidence/h4_pf4_plan.json")
    assert evidence["maximum_new_backend_calls"]==160+240+40+1800==2240
    assert evidence["stop_rule_sha256"]==hashlib.sha256(inspect.getsource(fg_fault.stop_job).encode()).hexdigest()
    result=dict(status="passed",created_utc=datetime.now(timezone.utc).isoformat(),scope="Read-only H4 PF4 plan and existing inputs; arithmetic stop-function checks only, no model simulation.",
        jobs=264,counts=dict(counts),healthy_inputs=224,complete_14_candidate_input_groups=16,
        new_healthy_searches=0,a3h_pf4_units=12,a1_decode_r0_only_units=4,maximum_l2_jobs=180,
        maximum_backend_calls=2240,repair_seeds=[123,1,2,3,4],independent_budgets=[30,100,300],backend_seeds=SEEDS,
        fixed_physical_links=[[[0,11],[1,8]],[[1,8],[0,11]]],cross_operator_and_phase_fault_selection="same physical bidirectional pair; no per-operator reselection",
        selection_protocol="backend mean across ten seeds, then candidate_order; random-CPCM selection retained as a separate field",
        repair_start="Each 30/100/300 budget starts from the same saved L1 state; no continuation between budgets.",
        stop_semantics=stop_checks,representative_selection="Five repair-seed CPCM estimates determine stopping; median backend-mean seed supplies the exported representative.",
        mf6_split_status="Not included in this PF4 batch; H4 has a later MF6/Split point.",
        combined_slots="H4 active + all unfinished H3/I2 and I3 candidate IDs <= 128; unsuccessful IDs retain slots for OOM retries.",
        memory="System available minus 3 GiB and each pool's future-growth budget; existing pool status supplies parent budgets.",
        fixed_spawn_job_id=True,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"e2e/hi_h4.py",ROOT/"e2e/hi_fault.py",ROOT/"e2e/hi_next.py",ROOT/"e2e/fg_batch.py",ROOT/"e2e/fg_fault.py",ROOT/"eir/eir.py")},
        manifest_sha256=sha(HERE/"manifest/h4_pf4.json"),input_inventory_sha256=sha(HERE/"manifest/h4_healthy_inputs.json"),selections=selections)
    write(HERE/"evidence/h4_pf4_plan_review.json",result)
    return result


if __name__=="__main__":
    result=plan_check()
    print(json.dumps({k:result[k] for k in ("status","jobs","healthy_inputs","a3h_pf4_units","maximum_l2_jobs")}))
