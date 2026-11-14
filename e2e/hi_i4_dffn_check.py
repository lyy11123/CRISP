"""Read-only review of the added Qwen WATOS decode-FFN PF4 unit."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import inspect
import json
from pathlib import Path
import statistics
from hi_next_check import read,sha,SEEDS
from hi_h3_i2_check import write

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"e2e/hi"


def check():
    import hi_i4_dffn as entry
    import hi_h4
    import hi_i4
    import fg_fault
    jobs=read(entry.MANIFEST)["jobs"]
    by_id={j["id"]:j for j in jobs}
    assert len(jobs)==len(by_id)==23
    assert Counter(j["kind"] for j in jobs)=={"fault_selection":1,"fault_select":1,"fault_pre":1,"fault_l1":1,"fault_stop":4,"fault_rep":15}
    inputs=read(entry.INPUTS)["candidates"]
    frozen={s["candidate_id"]:s for s in read(HERE/"manifest/all_candidates_planned.json")["jobs"]}
    expected={n for n,s in frozen.items() if s["model"]=="qwen3moe" and s["hardware_profile"]=="watos" and s["phase"]=="decode" and s["op"] in ("mha","ffn") and s["method"]=="A3h"}
    assert len(inputs)==len(expected)==28 and set(inputs)==expected
    grouped=defaultdict(list)
    for name,paths in inputs.items():
        for kind in ("record","raw","snapshot"):
            assert sha(ROOT/paths[kind])==paths["sha256"][kind]
        r=read(ROOT/paths["record"])
        assert r["status"]=="ok" and all(r[k]==v for k,v in frozen[name].items())
        assert [v["seed"] for v in r["backend"]]==SEEDS and len(r["T_hat_rand"])==10
        grouped[r["op"]].append(r)
    assert set(grouped)=={"mha","ffn"} and all(len(v)==14 for v in grouped.values())
    for job in jobs:
        s=job["spec"]
        assert s["model"]=="qwen3moe" and s["hardware_profile"]=="watos" and s.get("fault_id") in (None,"PF4")
        assert s["backend_seeds"]==SEEDS and s["job_id"]==job["id"] and job["deps"]==s["depends_on"]
        assert set(job["deps"])<=set(by_id)
        assert job["argv"]==["{python}","-u","-B","e2e/hi_i4_dffn.py","child","--spec","{spec}","--out","{out}"]
        if "phase_op" in s:assert s["phase_op"]=="dc_ffn"
        if "start_arm" in s:assert s["start_arm"]=="A3h"
        if job["kind"]=="fault_selection":assert not job["deps"] and set(s["candidate_ids"])==expected
        if job["kind"]=="fault_rep":
            assert s["steps"] in (30,100,300) and s["seed"] in (123,1,2,3,4)
            assert job["deps"]==[s["previous_stop"]]
            assert "i4_watos_dffn/raw/" in s["l1_sink"] and "i4_watos_dffn/raw/" in s["pre_sink"]
    reps=[j for j in jobs if j["kind"]=="fault_rep"]
    assert {(j["spec"]["steps"],j["spec"]["seed"]) for j in reps}=={(n,r) for n in (30,100,300) for r in (123,1,2,3,4)}
    assert len({j["spec"]["l1_sink"] for j in reps})==1
    done=set()
    while len(done)<23:
        ready={n for n,j in by_id.items() if n not in done and set(j["deps"])<=done}
        assert ready
        done.update(ready)
    prior=read(HERE/"evidence/i4_light_plan_review.json")["source_sha256"]
    for name in ("hi_i4.py","hi_h4.py","hi_fault.py","hi_next.py","fg_batch.py","fg_fault.py"):
        assert sha(ROOT/"e2e"/name)==next(v for k,v in prior.items() if Path(k).name==name)
    assert [len(read(m)["jobs"]) for m,_,_ in entry.PARENTS]==[440,108,264,201]
    entry.bind()
    controller=hi_i4.Controller(entry.OUT,128)
    assert controller.manifest==entry.MANIFEST and controller.out==entry.OUT and len(controller.jobs)==23
    assert set(controller.by_id)==set(by_id) and all(p["out"]!=controller.out for p in controller.parents)
    assert hi_h4.INPUTS==entry.INPUTS and hi_h4.NAME=="i4_watos_dffn"
    winners={}
    with hi_h4.healthy_routes():
        for op,rows in grouped.items():
            winner=min(rows,key=lambda r:(statistics.mean(x["cycles"] for x in r["backend"]),r["candidate_order"]))
            selected=fg_fault.choose_candidate(rows,entry.OUT)
            assert selected["candidate_id"]==winner["candidate_id"]
            assert all(selected[k]==inputs[winner["candidate_id"]][k] for k in ("record","raw","snapshot"))
            winners[op]=winner["candidate_id"]
    assert winners["ffn"]==read(HERE/"evidence/i4_watos_dffn_plan.json")["selected_candidate"]
    assert "anchor_backend_calls=0" in inspect.getsource(hi_i4.fixed_pf4)
    wrapped=hi_h4.wrapped_source()
    assert "external_slots = _hi.reserve_parents()" in wrapped and "_hi.parent_growth_reserve_gib" in wrapped
    assert 'namespace["execute"](self.manifest,self.out,self.workers,3,NAME,4,self.release)' in inspect.getsource(hi_h4.Controller.run)
    compile(wrapped,"<DFFN shared pool review>","exec")
    result=dict(status="passed",created_utc=datetime.now(timezone.utc).isoformat(),scope="Read-only added decode-FFN PF4 unit; no smoke or simulation.",
        jobs=23,healthy_inputs=28,healthy_files_sha256_checked=84,complete_candidate_groups=2,new_healthy_searches=0,
        fault_units=1,operator="qwen3moe/watos/decode/ffn/A3h/PF4",maximum_L2_trajectories=15,maximum_backend_samples=180,
        additional_MHA_backend_samples=0,selected_candidates=winners,repair_seeds=[123,1,2,3,4],independent_budgets=[30,100,300],backend_seeds=SEEDS,
        parent_job_counts=[440,108,264,201],combined_slots="Current active plus all unfinished parent IDs <=128",reserve_gib=3,
        original_running_sources_unchanged=True,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"e2e/hi_i4_dffn.py",ROOT/"e2e/hi_i4.py",ROOT/"e2e/hi_h4.py",ROOT/"e2e/hi_fault.py")},
        manifest_sha256=sha(entry.MANIFEST),input_inventory_sha256=sha(entry.INPUTS))
    write(HERE/"evidence/i4_watos_dffn_plan_review.json",result)
    return result


if __name__=="__main__":
    r=check()
    print(json.dumps({k:r[k] for k in ("status","jobs","healthy_inputs","maximum_L2_trajectories")}))
