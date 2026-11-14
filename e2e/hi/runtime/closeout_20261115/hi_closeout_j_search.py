"""J2 full-search timings; fixed candidates, fresh processes, otherwise idle host."""
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
import ast
import copy
import csv
import hashlib
import inspect
import json
from pathlib import Path
import time

import f_config
import hi_config
import hi_manifest
import hi_run
import j_timing
from e_run import ROOT,write_json

BASE=ROOT/"e2e/hi/closeout"
OUT=ROOT/"e2e/hi/out/closeout_j_20261115"
MANIFEST=BASE/"manifests/j_search.json"
OBSERVATIONS={}


def instrument_j2_anneal(c,mark,progress):
    """Time existing A1 diagnostics and progress without changing proposals."""
    source=inspect.getsource(c.make_anneal)
    original=ast.parse(source);tree=ast.parse(source)
    state=dict(initial_A1_monitor_seconds=0.,loop_A1_monitor_seconds=0.,loop_A1_monitor_calls=0,
        progress_callback_seconds=0.,progress_callback_calls=0,anneal_mark_seconds=0.)
    OBSERVATIONS.clear();OBSERVATIONS.update(state)
    def monitor(evaluate,step):
        started=time.perf_counter();value=evaluate()[1].T;elapsed=time.perf_counter()-started
        OBSERVATIONS["initial_A1_monitor_seconds" if step==0 else "loop_A1_monitor_seconds"]+=elapsed
        if step:OBSERVATIONS["loop_A1_monitor_calls"]+=1
        return value
    def measured_mark(label):
        started=time.perf_counter();mark(label);OBSERVATIONS["anneal_mark_seconds"]+=time.perf_counter()-started
    def measured_progress(step):
        started=time.perf_counter();progress(step)
        OBSERVATIONS["progress_callback_seconds"]+=time.perf_counter()-started
        OBSERVATIONS["progress_callback_calls"]+=1
    class Monitor(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=="mon" for t in node.targets):
                assert ast.unparse(node.value)=="t_eval()[1].T"
                node.value=ast.Call(ast.Name("_j_monitor",ast.Load()),[ast.Name("t_eval",ast.Load()),copy.deepcopy(node.targets[0].slice)],[])
            return self.generic_visit(node)
    tree=Monitor().visit(tree)
    anneal=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="anneal")
    def call(name,arg):return ast.Expr(ast.Call(ast.Name(name,ast.Load()),[arg],[]))
    body=[call("_j_mark",ast.Constant("anneal_enter"))]
    for node in anneal.body:
        assigned=[n.id for n in node.targets if isinstance(n,ast.Name)] if isinstance(node,ast.Assign) else []
        if "steps" in assigned:body.append(call("_j_mark",ast.Constant("normalization_enter")))
        if "wall0" in assigned:body.append(call("_j_mark",ast.Constant("search_begin")))
        body.append(node)
        if "sa_wall" in assigned:body.append(call("_j_mark",ast.Constant("search_complete")))
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=="step":
            node.body.append(call("_j_progress",ast.Name("step",ast.Load())))
    anneal.body=body
    class Restore(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in ("_j_mark","_j_progress"):return None
            return self.generic_visit(node)
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=="_j_monitor":return ast.parse("t_eval()[1].T",mode="eval").body
            return self.generic_visit(node)
    restored=Restore().visit(copy.deepcopy(tree))
    assert ast.dump(original,include_attributes=False)==ast.dump(restored,include_attributes=False)
    namespace=dict(vars(c),_j_monitor=monitor,_j_mark=measured_mark,_j_progress=measured_progress)
    exec(compile(ast.fix_missing_locations(tree),"<J2 measured observations>","exec"),namespace)
    return namespace["make_anneal"],dict(original_algorithm_ast_identical_after_removing_observations=True,
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),changes="timestamps only; original A1 diagnostic evaluations retained and separately timed")


def gpt_child():
    """Use the already instrumented full loop, changing only model preparation."""
    source=inspect.getsource(hi_run.child)
    replacements={
        '    pilot, c, cfg = hi_config.prepare(spec, stage)':'    pilot, c, cfg = f_config.prepare(spec, stage)',
        "    assert spec['model_contract_sha256'] == hi_config.digest(hi_config.model_contract(spec['model']))":
        "    assert spec['model'] == 'gpt' and spec['workload_version'] == 'W1'",
    }
    for old,new in replacements.items():
        assert source.count(old)==1
        source=source.replace(old,new)
    namespace=dict(vars(hi_run),f_config=f_config)
    exec(compile(source,"<J2 GPT model preparation>","exec"),namespace)
    return namespace["child"],hashlib.sha256(source.encode()).hexdigest()


def plan():
    definitions=[("GPT64_decode_ffn","gpt","decode","ffn","2x2","fixed",3),
        ("Llama_prefill_mha","llama","prefill","mha","2x2","fixed",3),
        ("Qwen_decode_ffn","qwen3moe","decode","ffn","2x2","fixed",3),
        ("GPT144_prefill_mha","gpt","prefill","mha","3x3","growth",1)]
    all_hi=hi_manifest.candidates();jobs=[]
    with (ROOT/"e2e/hi/tables/h3_i2_candidates.csv").open(encoding="utf-8-sig",newline="") as stream:
        hi_sources={r["candidate_id"]:r["source"].replace("\\","/") for r in csv.DictReader(stream)}
    for label,model,phase,op,shape,split,reps in definitions:
        for method in ("A1","A3h"):
            x=1 if method=="A1" else 123
            if model=="gpt":
                base=dict(kind="search",model=model,phase=phase,op=op,hardware_profile="busybarn",shape=shape,
                    split_mode=split,d2d_scale=1,method=method,g=0,x=x,search_steps=1000,
                    workload_version="W1",backend_seeds=list(range(1,11)),random_cpcm_seeds=list(range(1,11)))
            else:
                base=dict(next(s for s in all_hi if s["model"]==model and s["hardware_profile"]=="busybarn"
                    and s["phase"]==phase and s["op"]==op and s["method"]==method and s["g"]==0 and s["x"]==x))
            if model!="gpt":
                source=hi_sources[base["candidate_id"]]
            elif shape=="3x3":
                suffix="l1" if method=="A1" else "s123"
                source=f"e2e/fg/out/main/results/G_busybarn_3x3_growth_d2d1__healthy_healthy_prefill_mha_{method}_g0_{suffix}.json"
            else:source=None
            measured=j_timing.read(ROOT/source) if source else None
            seconds=measured["wall_seconds"] if measured else 180
            rss=max(1.,1.5*measured["peak_rss_gb"]) if measured else 2.
            for repetition in range(1,reps+1):
                name=f"J2_{label}_{method}_r{repetition}"
                spec=dict(base,candidate_id=name,job_id=name,module="hi_closeout_j_search",timing_only=True,
                    input_id=label,repetition=repetition,quality_table_replacement=False,depends_on=[])
                jobs.append(dict(id=name,kind="search",spec=spec,deps=[],result=f"results/{name}.json",
                    argv=["{python}","-u","-B","e2e/hi_closeout_j_search.py","child","--spec","{spec}","--out","{out}"],
                    estimated_seconds=seconds,peak_rss_gb=rss,
                    estimate_source=source,estimate_note="historical parallel full candidate; GPT144 preceded bb acceleration; GPT64 180-second planning placeholder",
                    priority=-100 if reps==1 else 0))
    assert len(jobs)==20 and sum(j["spec"]["input_id"]=="GPT144_prefill_mha" for j in jobs)==2
    write_json(MANIFEST,dict(status="authorized_pending_idle_window",jobs=jobs,full_searches=20,
        historical_cost_sum_hours=sum(j["estimated_seconds"] for j in jobs)/3600,
        quality_backend_calls=200,random_cpcm_evaluations=200,protocol="one process; 1000 steps; fixed g0 A1/l1 and A3h/seed123",
        excludes="All J2 outputs are timing repeats and never replace healthy winners"))
    _,digest=gpt_child()
    write_json(BASE/"evidence/j_search_entry_check.json",dict(status="passed",jobs=20,
        gpt_adapter_sha256=digest,changes="model preparation and GPT workload assertion only",
        formal_steps=1000,backend_seeds=list(range(1,11)),random_cpcm_seeds=list(range(1,11)),
        extra_validation_searches=0))
    return dict(jobs=20,full_searches=20,backend_calls=200,status="ready_for_idle_window")


def child(spec,out):
    assert spec["timing_only"] and spec["search_steps"]==1000 and not spec["quality_table_replacement"]
    started=time.perf_counter()
    environment=j_timing.environment()
    hi_run.instrument_anneal=instrument_j2_anneal
    if spec["model"]=="gpt":
        run,adapter_hash=gpt_child()
        run(spec,out)
    else:
        adapter_hash=None
        hi_run.child(spec,out)
    path=Path(out)/"results"/(spec["job_id"]+".json")
    result=j_timing.read(path)
    raw_sa=result["timing"]["sa_seconds"]
    result["j_diagnostic_timing"]=dict(OBSERVATIONS,raw_sa_seconds=raw_sa,
        algorithm_search_seconds=raw_sa-OBSERVATIONS["loop_A1_monitor_seconds"]-OBSERVATIONS["progress_callback_seconds"],
        note="A1 diagnostic CPCM and loop progress callback time removed from search only; full observed costs retained")
    result.update(job_id=spec["job_id"],spec=spec,j_environment=environment,
        timing_entry_seconds=time.perf_counter()-started,gpt_adapter_sha256=adapter_hash,
        j_protocol="fresh process; 1000 proposals; fixed candidate; numerical libraries single-threaded",
        timing_scope="offline mapping; ten backends and random selectors separately measured")
    write_json(path,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("plan","child"));p.add_argument("--spec",type=Path);p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    if a.command=="plan":print(json.dumps(plan()))
    else:child(j_timing.read(a.spec),a.out)
