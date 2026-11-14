"""J3: isolated EIR timing; three cases, three repeats, all independent budgets."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from contextlib import contextmanager
import copy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import sys
import time

import fg_fault as fault
from e_run import ROOT, write_json

BASE = ROOT / "e2e/hi/closeout"
OUT = ROOT / "e2e/hi/out/closeout_j_20261115"
INPUTS = BASE / "manifests/j3_inputs.json"
MANIFEST = BASE / "manifests/j3.json"
SEEDS = (123,1,2)
BUDGETS = (30,100,300)


def sha(path):
    return hashlib.sha256(fault.path(path).read_bytes()).hexdigest()


def plan(out=OUT):
    import post_fg_split
    post_fg_split.setup()
    gpt_mapping = "e2e/c/inputs/a3h_dc_mha.pkl.gz"
    mapping = fault.read_pickle(gpt_mapping)
    virtual = BASE / "j3_inputs"
    raw_path = virtual / "gpt64_mapping.pkl.gz"
    record_path = virtual / "gpt64_record.json"
    gpt64 = dict(model="gpt",hardware_profile="busybarn",shape="2x2",split_mode="fixed",d2d_scale=1.0,
        phase="decode",op="mha",phase_op="dc_mha",method="A3h",g=0,x=2,start_arm="A3h",fault_id="PF4",
        config_id="J3_GPT64_busybarn_2x2_fixed",workload_version="W1",backend_seeds=list(range(1,11)),
        source_mapping=gpt_mapping,source_mapping_sha256=sha(gpt_mapping))
    virtual.mkdir(parents=True,exist_ok=True)
    packed=pickle.dumps(dict(selected=dict(e3map=mapping),source_mapping=gpt_mapping,source_mapping_sha256=sha(gpt_mapping)),protocol=pickle.HIGHEST_PROTOCOL)
    temporary=raw_path.with_suffix(raw_path.suffix+".part")
    temporary.write_bytes(gzip.compress(packed,mtime=0));temporary.replace(raw_path)
    write_json(record_path,gpt64)
    oldpre="e2e/c/out/prep/pre/pre_a3h_dc_mha.pkl.gz"
    oldl1="e2e/c/out/prep/l1/l1_a3h_dc_mha_PF4.pkl.gz"
    cases={"gpt64":dict(base=gpt64,source=dict(candidate_id="C_A3h_dc_mha_g0_s2",record=fault.relative(record_path),
        raw=fault.relative(raw_path)),pre_sink=oldpre,old_l1_sink=oldl1,
        physical_fault=fault.read_json("e2e/c/faults.json")["dc_mha"]["PF4"],
        original_mapping=gpt_mapping)}
    for name,folder,cid in (("gpt144","e2e/fg/out/main","G_busybarn_3x3_growth_d2d1"),
                            ("qwen","e2e/hi/out/i4_light","qwen3moe_busybarn_2x2_fixed_668547c78787_49668fb8bf")):
        op="dc_ffn" if name=="qwen" else "dc_mha"
        root=ROOT/folder
        l1=fault.read_json(root/"results"/f"{cid}__l1_{op}_A3h_PF4.json")
        measured_budget=300 if name=="qwen" else 30
        rep=fault.read_json(root/"results"/f"{cid}__rep_{op}_PF4_s{measured_budget}_r123.json")
        pre=fault.read_json(root/"results"/f"{cid}__pre_{op}_A3h.json")
        source=fault.read_json(root/"selection"/(cid+".json"))["operators"][op]["A3h"]
        base=dict(l1["spec"],model="qwen3moe" if name=="qwen" else "gpt")
        cases[name]=dict(base=base,source=source,pre_sink=pre["sink"],old_l1_sink=l1["sink"],
            physical_fault=fault.read_json(root/"faults"/(cid+".json"))["faults"]["PF4"],
            estimate_l1_seconds=l1["wall_seconds"],estimate_search_step_seconds=rep["timing"]["t_L2_search"]/measured_budget,
            estimate_backend_seconds=rep["timing"]["t_backend_offline"],
            estimate_source=fault.relative(root/"results"/f"{cid}__rep_{op}_PF4_s{measured_budget}_r123.json"))
    cases["gpt64"].update({key:cases["gpt144"][key] for key in ("estimate_l1_seconds","estimate_search_step_seconds","estimate_backend_seconds")})
    cases["gpt64"]["estimate_source"]="GPT144 measured cost proxy until the first GPT64 serial measurement"
    for case in cases.values():
        files={"pre_sink":case["pre_sink"],"old_l1_sink":case["old_l1_sink"],**case["source"]}
        case["input_sha256"]={p:sha(p) for k,p in files.items() if k in ("pre_sink","old_l1_sink","record","raw","snapshot")}
        if "original_mapping" in case:
            case["input_sha256"][case["original_mapping"]]=sha(case["original_mapping"])
    write_json(INPUTS,dict(status="planned",cases=cases,source_note="GPT64 wraps the existing C mapping; no new healthy search"))
    jobs=[]
    for case_name,case in cases.items():
        for seed in SEEDS:
            prefix=f"J3__{case_name}_r{seed}"
            clean={k:v for k,v in case["base"].items() if k not in ("job_id","depends_on","kind","pre_sink","l1_sink","previous_stop")}
            common=dict(clean,case=case_name,module="hi_closeout_j_eir",protocol_version="HI_closeout_J3_v1",repair_seed=seed,
                timing_repeat=SEEDS.index(seed)+1,source_inputs=fault.relative(INPUTS),input_sha256=case["input_sha256"],
                backend_seeds=list(range(1,11)),pre_sink=case["pre_sink"],start_arm="A3h",fault_id="PF4",
                checks=False,launch_status="authorized_after_quality_pool_idle")
            l1_name=prefix+"__l1"
            l1_spec=dict(common,job_id=l1_name,kind="fault_l1",depends_on=[])
            sequence=[(l1_spec,case["estimate_l1_seconds"])]
            for budget in BUDGETS:
                name=prefix+f"__s{budget}"
                spec=dict(common,job_id=name,kind="fault_rep",steps=budget,seed=seed,
                    depends_on=[l1_name],l1_sink=(out/"j3_work"/l1_name/"raw"/(l1_name+".pkl.gz")).relative_to(ROOT).as_posix(),
                    independent_budget=True)
                sequence.append((spec,case["estimate_search_step_seconds"]*budget+case["estimate_backend_seconds"]+30))
            for spec,seconds in sequence:
                jobs.append(dict(id=spec["job_id"],kind=spec["kind"],spec=spec,deps=spec["depends_on"],
                    result=f"results/{spec['job_id']}.json",estimated_seconds=seconds,peak_rss_gb=1.5,
                    argv=["{python}","-u","-B","e2e/hi_closeout_j_eir.py","child","--spec","{spec}","--out","{out}"]))
    assert len(jobs)==36 and sum(j["kind"]=="fault_l1" for j in jobs)==9
    write_json(MANIFEST,dict(status="planned_serial_after_quality_pool",jobs=jobs,required_workers=1,
        estimated_serial_hours=sum(j["estimated_seconds"] for j in jobs)/3600,
        new_L1_jobs=9,new_L2_jobs=27,backend_validation_calls=450,
        stop_rule="independent timing budgets all executed; quality main-table stop rules unchanged"))
    return dict(jobs=36,R0_L1=9,L2=27,backend_validation_calls=450,
        estimated_serial_hours=sum(j["estimated_seconds"] for j in jobs)/3600,required_workers=1)


@contextmanager
def adapter(spec,out):
    case=fault.read_json(spec["source_inputs"])["cases"][spec["case"]]
    assert spec["input_sha256"]==case["input_sha256"]
    assert all(sha(p)==value for p,value in case["input_sha256"].items())
    work=Path(out)/"j3_work"/spec["job_id"]
    old_source,old_read=fault.selected_source,fault.read_json
    fault.selected_source=lambda inner,root:case["source"]
    def redirected(path):
        if Path(path)==work/"faults"/(spec["config_id"]+".json"):
            return dict(faults=dict(PF4=case["physical_fault"]))
        return old_read(path)
    fault.read_json=redirected
    try:
        if spec["model"]=="qwen3moe":
            import hi_fault
            with hi_fault.model_adapter(spec,work) as protocol:
                yield work,case,protocol
        else:
            import f_config
            _,c,_=f_config.prepare(dict(case["base"],method="A1",x=0),work)
            sys.path.insert(0,str(ROOT/"eir"))
            import eir
            import post_fg_bb as bb
            original_child,original_finalize,original_sim=eir.child,eir.Engine.finalize,c.cpcm.simulate
            def no_checks(inner):return original_child(dict(inner,checks=False))
            def original_final(eng,seeds=None):
                active=c.cpcm.simulate;c.cpcm.simulate=bb.ORIGINAL
                try:return original_finalize(eng,seeds)
                finally:c.cpcm.simulate=active
            eir.child=no_checks;eir.Engine.finalize=original_final;c.cpcm.simulate=bb.simulate
            try:
                yield work,case,dict(search_simulate="post_fg_bb.simulate",final_simulate="original cpcm.simulate",
                    debug_checks=False,exact_rollback=c.EXACT_REVERT,backend_seeds=list(range(1,11)))
            finally:
                eir.child,eir.Engine.finalize,c.cpcm.simulate=original_child,original_finalize,original_sim
    finally:
        fault.selected_source,fault.read_json=old_source,old_read


def child(spec,out):
    assert spec["repair_seed"] in SEEDS and spec["kind"] in ("fault_l1","fault_rep")
    assert "previous_stop" not in spec
    if spec["kind"]=="fault_rep":assert spec["steps"] in BUDGETS and spec["seed"]==spec["repair_seed"]
    started=time.perf_counter()
    with adapter(spec,out) as (work,case,protocol):
        import post_fg_bb as bb
        t=time.perf_counter()
        bb.simulate(bb.cpcm.build([dict(tag="warm",kind="comp",deps=[],res="cpu",dur=1)]))
        first_call=time.perf_counter()-t
        export_seconds=[]
        original_export=fault.export_engine
        def timed_export(*args,**kwargs):
            t=time.perf_counter();value=original_export(*args,**kwargs)
            export_seconds.append(time.perf_counter()-t)
            return value
        fault.export_engine=timed_export
        try:result=fault.engine_job(spec,work)
        finally:fault.export_engine=original_export
        raw=fault.read_pickle(result["sink"])
        assert raw["spec"]["checks"] is False
        timing=dict(numba_warmup_or_cache_seconds=first_call,state_export_seconds=sum(export_seconds),state_exports=len(export_seconds))
        if spec["kind"]=="fault_l1":
            assert raw["T_pre"]==fault.read_pickle(case["pre_sink"])["T"]
            assert raw["T_R0"]==fault.read_pickle(case["old_l1_sink"])["T_R0"]
            records=[raw[k] for k in ("R0","L1")]
            timing.update(R0_prepare_seconds=raw["t_R0"],L1_search_seconds=raw["t_L1"],
                new_control_seconds=raw["t_R0"]+raw["t_L1"],calibration_seconds=0,search_seconds=0,restore_best_seconds=0)
        else:
            records=[raw["record"]];search=raw["search"]
            assert [r["step"] for r in search["rows"]]==list(range(1,spec["steps"]+1))
            assert raw["spec"]["seed"]==spec["repair_seed"] and search["check_revert"]["n"]==0
            assert raw["T_L1_check"]==fault.read_pickle(spec["l1_sink"])["T_L1"]==search["T_start"]
            timing.update(R0_prepare_seconds=0,L1_search_seconds=0,
                calibration_seconds=search["t_calibration"],search_seconds=search["t_L2_search"],
                restore_best_seconds=search["t_restore_best"],
                new_control_seconds=search["t_calibration"]+search["t_L2_search"]+search["t_restore_best"],
                inherited_R0_L1_seconds=raw["timing"]["t_R0"]+raw["timing"]["t_L1"],
                search_proposals=len(search["rows"]),calibration_proposals=raw["proposals"]["calibration"],
                accepted_proposals=sum(r["accepted"] for r in search["rows"]),best_step=search["best_step"])
        assert all(set(r["backend"])==set(range(1,11)) and not r["fault_violations"] for r in records)
        timing.update(final_CPCM_seconds=sum(r["t_finalize"] for r in records),
            backend_validation_seconds=sum(r["t_backend"] for r in records),
            backend_actual_calls_seconds=sum(sum(r["backend_wall"]) for r in records),backend_calls=10*len(records))
        import j_timing,e2e_pilot_peak
        env=j_timing.environment()
        env["timing_protocol"]="one otherwise-idle worker; one independent EIR trajectory per process; no stop skipping"
        env["code_sha256"].update({p.name:sha(p) for p in (Path(__file__),ROOT/"e2e/fg_fault.py",ROOT/"eir/eir.py")})
        result.update(job_id=spec["job_id"],kind=spec["kind"],spec=spec,status="ok",
            wall_seconds=time.perf_counter()-started,peak_rss_gb=e2e_pilot_peak.peak()/2**30,
            J3_timing=timing,environment=env,protocol=protocol,input_sha256=case["input_sha256"],
            final_backend_calls=timing["backend_calls"],quality_table_update=False,
            repetition_semantics="R0/L1 repeats deterministic rule; repair_seed selects L2 trajectory")
    write_json(Path(out)/"results"/(spec["job_id"]+".json"),result)
    print(json.dumps(dict(job_id=spec["job_id"],status="ok",seconds=result["wall_seconds"])),flush=True)


class Inspected(Exception):pass


def smoke(case_name,out):
    manifest=fault.read_json(MANIFEST)
    job=next(j for j in manifest["jobs"] if j["spec"]["case"]==case_name and j["kind"]=="fault_l1")
    spec=dict(job["spec"],job_id="J3_smoke_"+case_name)
    began=time.perf_counter(); evidence={}
    with adapter(spec,out) as (work,case,protocol):
        import eir
        original=eir.child_l1
        def inspect(eng,inner,sink,*args):
            base=fault.read_pickle(case["old_l1_sink"])
            eng.seed_broadcast()
            injected_R0=eng.t_eval()
            assert injected_R0==base["T_R0"]
            values={}
            for level in ("R0","L1"):
                eng.load(base[level+"_snap"])
                values[level]=eng.t_eval()
                assert values[level]==base["T_"+level]
            evidence.update(status="passed",case=case_name,T=values,input_sha256=case["input_sha256"],
                injected_R0_T=injected_R0,injected_R0_matches_saved=True,
                reload_matches_saved_R0_and_L1=True,backend_calls=0,searches=0,protocol=protocol)
            raise Inspected()
        eir.child_l1=inspect
        try:fault.engine_job(spec,work)
        except Inspected:pass
        finally:eir.child_l1=original
    evidence["wall_seconds"]=time.perf_counter()-began
    write_json(BASE/"evidence"/("j3_smoke_"+case_name+".json"),evidence)
    return evidence


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","child","smoke"))
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--case",choices=("gpt64","gpt144","qwen"))
    args=parser.parse_args()
    if args.command=="plan":print(json.dumps(plan(args.out),indent=2))
    elif args.command=="smoke":print(json.dumps(smoke(args.case,args.out),indent=2))
    else:child(fault.read_json(args.spec),args.out)
