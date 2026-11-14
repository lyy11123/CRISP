"""Read-only review of the Qwen I5 adaptation before I4 inputs arrive."""
import ast
import copy
from datetime import datetime,timezone
import gzip
import hashlib
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
    import hi_i5 as new
    import hi_h5 as old
    import hi_config
    import hi_joint
    import hi_h4
    import post_fg_split
    contract=hi_config.model_contract("qwen3moe")
    digest=hi_config.digest(contract)
    chain=["dc_ln"]+["dc_mha"]*4+["dc_proj","dc_ffn"]
    assert contract["head_groups"]==4 and contract["decode_ffn_includes_rmsnorm"]
    assert hi_joint.decode_chain_order("qwen3moe")==chain
    assert new.NAME=="i5" and new.I4==HERE/"out/i4_light" and new.OUT==HERE/"out/i5"
    assert new.PHASES==(.1,.5,.9) and new.SEEDS==(1,2,3)
    unchanged=[]
    for name in ("stream_signature","input_record","bind"):
        assert ast.dump(ast.parse(inspect.getsource(getattr(old,name))))==ast.dump(ast.parse(inspect.getsource(getattr(new,name))))
        unchanged.append(name)
    for name in ("adapter","child"):
        source=inspect.getsource(getattr(old,name)).replace('"llama"','"qwen3moe"').replace('*8','*4').replace('mha_groups=8','mha_groups=4').replace('stages=11','stages=7')
        assert ast.dump(ast.parse(source))==ast.dump(ast.parse(inspect.getsource(getattr(new,name))))
        unchanged.append(name+": only model and head-group/stage counts changed")
    old_plan=ast.parse(inspect.getsource(old.plan)).body[0].body
    new_plan=ast.parse(inspect.getsource(new.plan)).body[0].body
    assert ast.unparse(new_plan[0])=="readiness = source_readiness()"
    assert ast.unparse(new_plan[1])=="if readiness['status'] != 'passed':\n    return readiness"
    class Translate(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):
                node.value=node.value.replace('H5_llama','I5_qwen3moe').replace('llama','qwen3moe').replace('hi_h5','hi_i5').replace('h5_','i5_')
            return node
        def visit_Name(self,node):
            if node.id=="H4":node.id="I4"
            return node
    original=Translate().visit(ast.Module(body=copy.deepcopy(old_plan),type_ignores=[]))
    assert ast.dump(original)==ast.dump(ast.Module(body=new_plan[2:],type_ignores=[]))
    # This is source equivalence, not a claim that the missing D files exist.
    template=read(HERE/"manifest/i5_template.json")
    assert template["status"]=="template_not_runnable" and template["decode_chain"]==chain
    refs=template["references"]
    assert len(refs)==20 and {r["operator"] for r in refs}=={"dc_ln","dc_mha","dc_proj","dc_ffn"}
    assert len({(r["state"],r["operator"]) for r in refs})==20
    for r in refs:
        assert r["result"].startswith("e2e/hi/out/i4_light/results/"+new.CID)
        if r["state"]=="D_A3hEIR":assert r["key"]=="export" and r["result"].endswith("_PF4_300.json")
        elif r["state"].startswith("healthy"):assert r["key"]=="exports.healthy"
        else:assert r["key"]=="exports.R0"
    P_records=[]
    for folder in ("h1_main","next_344","h3_i2_440","i3_108"):
        for path in (HERE/"out"/folder/"results").glob("I_qwen3moe_busybarn_prefill_mha_*.json"):
            r=read(path)
            if r["kind"]=="search" and r["method"] in ("A1","A3h"):P_records.append((r,path))
    assert len(P_records)==28
    stored=read(HERE/"evidence/i5_P_healthy_selection.json")
    post_fg_split.setup()  # Pickle classes only; no construction or backend evaluation.
    p_checks=[]
    for method in ("A1","A3h"):
        group=[x for x in P_records if x[0]["method"]==method]
        assert len(group)==14
        record,path=min(group,key=lambda x:(statistics.mean(b["cycles"] for b in x[0]["backend"]),x[0]["candidate_order"]))
        check=next(x for x in stored["groups"] if x["method"]==method)
        assert check["selected_candidate"]==record["candidate_id"]
        assert check["backend_mean_ns"]==statistics.mean(b["cycles"] for b in record["backend"])
        for relative,digest_file in check["source_sha256"].items():assert sha(ROOT/relative)==digest_file
        name=record["candidate_id"];folder=path.parent.parent
        raw=load(folder/"raw"/(name+".pkl.gz"));blob,_=load(folder/"snapshots"/(name+".pkl.gz"))
        _,data,_=pickle.loads(blob)
        tensors=[data[tuple(t)] for t in raw["meta"]["ctx"]["broadcast_datatags"]]
        assert [t.data_name for t in tensors]==["MHA_ktrans_ofmap","MHA_vconv_ofmap"]
        assert [t.data_shape for t in tensors]==[[16,64,2048],[16,2048,64]]
        assert sum(t.data_size for t in tensors)==8388608
        p_checks.append(dict(method=method,complete_candidates=14,candidate_id=name,backend_mean_ns=check["backend_mean_ns"],kv_bytes=8388608,source_sha256=check["source_sha256"]))
    parent_counts=[len(read(m)["jobs"]) for m,_,_ in new.PARENTS]
    assert parent_counts==[440,108,264,201,23,110]
    reserve=inspect.getsource(new.Controller.reserve_parents)
    assert 'remaining=len(parent["jobs"])-len(parent["success"])' in reserve
    assert 'if remaining:growth+=' in reserve
    assert 'result["job_id"]==name and result["spec"]==job["spec"]' in reserve
    assert '_hi.reserve_parents()' in hi_h4.wrapped_source()
    assert '_hi.parent_growth_reserve_gib' in hi_h4.wrapped_source()
    assert 'self.workers,3,NAME,4,self.release' in inspect.getsource(hi_h4.Controller.run)
    main=inspect.getsource(new.main)
    assert 'h5["rc"]==0 and h5["expected"]==h5["completed"]==110' in main
    assert 'assert source_readiness()["status"]=="passed"' in main
    assert 'stderr=subprocess.STDOUT' in main and 'assert 1<=args.workers<=128' in main
    assert 'archive=args.out/"pilot_run"' in main
    report=dict(status="passed_static_only",created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Existing 28 healthy P candidates and static I5 adaptation; no graph construction, search, backend evaluation or launch.",
        model="qwen3moe",model_contract_sha256=digest,decode_chain=chain,head_groups=4,external_norms=1,ffn_embedded_norms=1,
        source_equivalence_checks=unchanged,job_generation_equivalent_to_H5_after_model_and_I4_input_substitution=True,
        planned_jobs=110,planned_joint_samples=63,planned_actual_paired_solos=39,logical_paired_solos=42,
        pilot_jobs=14,pilot_backend_samples=9,pilot_formal_ids_reused_on_full_launch=True,
        P_groups=p_checks,D_reference_count=20,D_references=refs,
        missing_local_inputs=list(template["missing"]),formal_inputs_ready=False,
        public_reference="healthy A1/original three-seed P and D means plus P KV window",
        C4_internal_reference="per-graph hop CPCM estimates retained",
        causal_P_to_D_dependencies=False,PF4_applies_only_to_D_group=True,
        parent_job_counts=parent_counts,parent_reservation="Every unfinished parent ID, including pending work and possible retries",workers_limit=128,reserve_gib=3,
        launch_requires_H5_completed=110,launch_requires_all_I4_exports=True,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"e2e/hi_i5.py",ROOT/"e2e/hi_joint.py",ROOT/"e2e/fg_joint.py",ROOT/"e2e/d_joint.py")},
        remaining="Recheck actual I4 export hashes, formal manifests and pilot runtime once I4 final inputs are restored.")
    write(HERE/"evidence/i5_static_review.json",report)
    return report


if __name__=="__main__":
    r=check();print(json.dumps({k:r[k] for k in ("status","planned_jobs","D_reference_count","formal_inputs_ready")}))
