"""Read-only review of the ready Qwen PF4 subset, with no fault simulation."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

from hi_next_check import read, sha, KEYS, SEEDS
from hi_h3_i2_check import write
from hi_h4_check import stop_semantics

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"e2e/hi"


def plan_check():
    import hi_i4
    import hi_h4
    import fg_fault
    import hi_config
    jobs=read(HERE/"manifest/i4_light.json")["jobs"]
    by_id={j["id"]:j for j in jobs}
    counts=Counter(j["kind"] for j in jobs)
    assert len(jobs)==len(by_id)==201
    assert counts=={"fault_selection":2,"fault_select":2,"fault_pre":13,"fault_l1":13,"fault_stop":36,"fault_rep":135}
    all_specs={s["candidate_id"]:s for s in read(HERE/"manifest/all_candidates_planned.json")["jobs"]}
    wanted={n for j in jobs if j["kind"]=="fault_selection" for n in j["spec"]["candidate_ids"]}
    inputs=read(HERE/"manifest/i4_light_healthy_inputs.json")["candidates"]
    expected={n for n,s in all_specs.items() if s["model"]=="qwen3moe" and (
        s["hardware_profile"]=="busybarn" and (s["phase"],s["op"])!=("prefill","ffn") and
        (s["method"]=="A3h" or s["phase"]=="decode" and s["method"]=="A1") or
        s["hardware_profile"]=="watos" and s["op"]=="mha" and s["method"]=="A3h")}
    assert len(wanted)==len(inputs)==182 and wanted==set(inputs)==expected
    groups=defaultdict(list)
    for name,entry in inputs.items():
        record=read(ROOT/entry["record"])
        assert record["status"]=="ok" and all(record[k]==v for k,v in all_specs[name].items())
        assert [x["seed"] for x in record["backend"]]==SEEDS and len(record["T_hat_rand"])==10
        for kind in ("record","raw","snapshot"):
            assert sha(ROOT/entry[kind])==entry["sha256"][kind],(name,kind)
        groups[tuple(record[k] for k in KEYS)].append(record)
    assert len(groups)==13 and all(len(rows)==14 for rows in groups.values())
    contract=hi_config.model_contract("qwen3moe")
    assert contract["head_groups"]==4 and contract["decode_chain"]["mha"]==4
    contract_sha=hi_config.digest(contract)
    a3h,a1=[],[]
    for job in jobs:
        spec=job["spec"]
        assert spec["model"]=="qwen3moe" and spec["model_contract_sha256"]==contract_sha
        assert spec["job_id"]==job["id"] and spec["kind"]==job["kind"]
        assert spec.get("fault_id") in (None,"PF4") and spec["backend_seeds"]==SEEDS
        assert job["deps"]==spec["depends_on"] and set(job["deps"])<=set(by_id)
        assert job["argv"]==["{python}","-u","-B","e2e/hi_i4.py","child","--spec","{spec}","--out","{out}"]
        assert job["result"]==f"results/{job['id']}.json"
        if job["kind"]=="fault_selection":assert not job["deps"]
        if "phase_op" in spec:
            assert spec["phase_op"]!="pf_ffn"
            if spec["hardware_profile"]=="watos":assert spec["phase_op"] in ("pf_mha","dc_mha")
        if job["kind"]=="fault_l1":(a3h if spec["start_arm"]=="A3h" else a1).append(job)
        if job["kind"]=="fault_rep":
            assert spec["start_arm"]=="A3h" and job["deps"]==[spec["previous_stop"]]
            assert spec["l1_sink"].endswith(f"l1_{spec['phase_op']}_A3h_PF4.pkl.gz")
            assert "i4_light/raw/" in spec["l1_sink"].replace("\\","/") and "i4_light/raw/" in spec["pre_sink"].replace("\\","/")
    assert len(a3h)==9 and len(a1)==4
    assert {(j["spec"]["hardware_profile"],j["spec"]["phase_op"]) for j in a3h}=={
        ("busybarn",p) for p in ("pf_ln","pf_mha","pf_proj","dc_ln","dc_mha","dc_proj","dc_ffn")} | {("watos","pf_mha"),("watos","dc_mha")}
    assert {(j["spec"]["hardware_profile"],j["spec"]["phase_op"]) for j in a1}=={("busybarn",p) for p in ("dc_ln","dc_mha","dc_proj","dc_ffn")}
    for start in a3h:
        s=start["spec"]
        repairs=[j for j in jobs if j["kind"]=="fault_rep" and j["spec"]["config_id"]==s["config_id"] and j["spec"]["phase_op"]==s["phase_op"]]
        assert len(repairs)==15 and {(j["spec"]["steps"],j["spec"]["seed"]) for j in repairs}=={(n,r) for n in (30,100,300) for r in (123,1,2,3,4)}
        assert len({j["spec"]["l1_sink"] for j in repairs})==1
        for steps in (0,30,100,300):
            stops=[j for j in jobs if j["kind"]=="fault_stop" and j["spec"]["config_id"]==s["config_id"] and j["spec"]["phase_op"]==s["phase_op"] and j["spec"]["steps"]==steps]
            assert len(stops)==1
            assert set(stops[0]["deps"])==({start["id"]} if steps==0 else {j["id"] for j in repairs if j["spec"]["steps"]==steps})
    done=set()
    while len(done)<201:
        ready={n for n,j in by_id.items() if n not in done and set(j["deps"])<=done}
        assert ready
        done.update(ready)
    assert [len(read(m)["jobs"]) for m,_,_ in hi_i4.PARENTS]==[440,108,264]
    assert sha(ROOT/"e2e/hi_h4.py")=="76825ffd39262faf3469e158b50cb015c743e5067fd48ba03133b53c1b5a7b35"
    hi_i4.bind_shared()  # Process-local name binding; no files or subprocesses.
    assert hi_h4.NAME=="i4_light" and hi_h4.MANIFEST==HERE/"manifest/i4_light.json"
    assert hi_h4.INPUTS==HERE/"manifest/i4_light_healthy_inputs.json" and hi_h4.OUT==HERE/"out/i4_light"
    controller=hi_i4.Controller(HERE/"out/i4_light",128)
    assert len(controller.jobs)==201 and set(controller.by_id)==set(by_id)
    assert [len(p["jobs"]) for p in controller.parents]==[440,108,264]
    assert all(p["out"]!=controller.out for p in controller.parents)
    selections=[]
    with hi_h4.healthy_routes():
        for key,rows in sorted(groups.items()):
            winner=min(rows,key=lambda r:(sum(x["cycles"] for x in r["backend"])/10,r["candidate_order"]))
            result=fg_fault.choose_candidate(rows,controller.out)
            assert result["candidate_id"]==winner["candidate_id"]
            assert all(result[k]==inputs[winner["candidate_id"]][k] for k in ("record","raw","snapshot"))
            selections.append(dict(category=list(key),candidate_id=winner["candidate_id"],random_CPCM_pick=result["random_CPCM_pick"]))
    wrapped=hi_h4.wrapped_source()
    compile(wrapped,"<I4 reviewed shared pool>","exec")
    assert "external_slots = _hi.reserve_parents()" in wrapped and "len(running) + external_slots >= workers" in wrapped
    assert "budget = free - reserve_gib - reserved_growth - _hi.parent_growth_reserve_gib" in wrapped
    assert 'namespace["execute"](self.manifest,self.out,self.workers,3,NAME,4,self.release)' in inspect.getsource(hi_h4.Controller.run)
    assert "STDOUT=subprocess.STDOUT" in inspect.getsource(hi_h4.Controller.run)
    assert 'spec["job_id"]' in inspect.getsource(hi_h4.Controller.spawn)
    assert "anchor_backend_calls=0" in inspect.getsource(hi_i4.fixed_pf4)
    result=dict(status="passed",created_utc=datetime.now(timezone.utc).isoformat(),scope="Read-only ready I4 PF4 subset; no model construction or backend execution.",
        jobs=201,counts=dict(counts),control_jobs=40,healthy_inputs=182,complete_14_candidate_groups=13,
        a3h_pf4_units=9,a1_decode_r0_only_units=4,maximum_l2_jobs=135,maximum_backend_samples=130+180+40+1350,
        new_healthy_searches=0,model="qwen3moe",model_contract_sha256=contract_sha,decode_head_groups=4,
        excludes=["all prefill FFN","WATOS decode FFN","MF6/Split"],
        fixed_physical_links=[[[0,11],[1,8]],[[1,8],[0,11]]],selection_protocol="backend mean over ten seeds, candidate_order tie; random CPCM choice retained separately",
        dependency_semantics="Only selected complete groups enter selection; no dependency on excluded operators. Each L2 budget starts from the same L1 snapshot.",
        repair_seeds=[123,1,2,3,4],independent_budgets=[30,100,300],backend_seeds=SEEDS,stop_checks=stop_semantics(),
        parent_job_counts=[440,108,264],combined_slots="I4 active plus every unfinished parent job <=128, including parent retries and not-yet-ready repair jobs",
        memory="Available minus 3 GiB, own growth reserve and the three parent pools' growth reserves",
        frozen_h4_source_unchanged=True,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"e2e/hi_i4.py",ROOT/"e2e/hi_h4.py",ROOT/"e2e/hi_fault.py",ROOT/"e2e/hi_config.py",ROOT/"e2e/fg_fault.py",ROOT/"e2e/hi_next.py",ROOT/"e2e/fg_batch.py")},
        manifest_sha256=sha(HERE/"manifest/i4_light.json"),input_inventory_sha256=sha(HERE/"manifest/i4_light_healthy_inputs.json"),selections=selections)
    assert result["maximum_backend_samples"]==1700
    write(HERE/"evidence/i4_light_plan_review.json",result)
    return result


if __name__=="__main__":
    r=plan_check()
    print(json.dumps({k:r[k] for k in ("status","jobs","healthy_inputs","complete_14_candidate_groups","maximum_l2_jobs")}))
