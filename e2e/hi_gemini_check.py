"""Two-step native Gemini interface probes; these are not quality candidates."""
import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import hi_config as H
from hi_model_check import write


def seed_check(spec,out,seed):
    begin=time.perf_counter()
    pilot,c,cfg=H.prepare(dict(spec,method="Gemini",x=seed),Path(out).parent)
    import e_gemini
    import Stream_Mapping
    import event_driver
    from post_fg_split import graph_digest
    cls=Stream_Mapping.StreamOptimization
    original_move=cls.move
    trace=[]
    def move(self):
        result=original_move(self)
        trace.append(dict(kind=e_gemini.move_type(self),energy=float(self.energy()),
                          event_graph_sha256=graph_digest(self.state[7])))
        return result
    cls.move=move
    skipped=[]
    def skip_backend(**kwargs):
        skipped.append(len(kwargs["events_dict"]))
        return (0,0,0)
    event_driver.event_driver=skip_backend
    task=H.generate(dict(spec,method="Gemini",x=seed))
    ctx={"__file__":str(task),"__name__":"__main__"}
    with e_gemini.install(cls,seed=seed,steps=2,validate_rollback=True,
                          capture_initial=True,find_context=c._find_context) as capture:
        try:
            exec(compile(task.read_text(encoding="utf-8"),str(task),"exec"),ctx)
        except e_gemini.SearchComplete:
            pass
    assert capture["meta"]["steps"] == 2 and len(trace)==2
    assert capture["meta"]["LP_flag"] and capture["meta"]["hops_only"] and not capture["meta"]["dijkstra_routing"]
    row=dict(status="ok",spec=spec,seed=seed,meta=capture["meta"],trace=trace,
             initial_graph_sha256=graph_digest(capture["initial"][2]),
             best_graph_sha256=graph_digest(capture["best"][2]),
             diagnostic_backend_calls_skipped=len(skipped),backend_evaluations=0,
             quality_candidates=0,wall_seconds=time.perf_counter()-begin,peak_rss_gib=pilot.peak()/2**30)
    write(out,row)


def child(spec,out):
    begin=time.perf_counter()
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    parts=out.parent/"gemini_interface_parts"/out.stem
    parts.mkdir(parents=True,exist_ok=True)
    sp=parts/"spec.json"
    write(sp,spec)
    rows=[]
    for seed in (123,1):
        result=parts/f"seed{seed}.json"
        with (parts/f"seed{seed}.log").open("w",encoding="utf-8") as log:
            subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),"seed", "--spec",str(sp),
                            "--out",str(result),"--seed",str(seed)],stdout=log,stderr=subprocess.STDOUT,check=True)
        rows.append(json.loads(result.read_text(encoding="utf-8")))
    same_start=rows[0]["initial_graph_sha256"]==rows[1]["initial_graph_sha256"]
    different_trace=rows[0]["trace"]!=rows[1]["trace"]
    assert different_trace, "seed123 and seed1 produced identical two-proposal traces"
    write(out,dict(status="ok",spec=spec,rows=rows,steps_per_seed=2,backend_evaluations=0,quality_candidates=0,
                   fixed_initializer_same_graph=same_start,independent_proposal_traces=different_trace,
                   wall_seconds=time.perf_counter()-begin,peak_rss_gb=max(r["peak_rss_gib"] for r in rows),
                   scope="native LP/XY entry and two seed traces only; full 14-seed quality search is not performed"))


def manifest():
    jobs=[]
    for model in H.MODELS:
        for phase in ("decode","prefill"):
            for op in ("ln","mha","proj","ffn"):
                name=f"gemini_interface_{model}_{phase}_{op}"
                spec=dict(model=model,phase=phase,op=op,hardware_profile="busybarn",shape="2x2",method="Gemini",g=0,x=123)
                jobs.append(dict(id=name,kind="gemini_interface",deps=[],spec=spec,result=f"results/{name}.json",
                                 argv=["{python}","-u","-B","e2e/hi_gemini_check.py","child","--spec","{spec}","--out",f"{{out}}/results/{name}.json"],
                                 estimated_seconds=30,peak_rss_gb=2))
    write(H.HERE/"hi/manifest/gemini_interface_checks.json",dict(status="authorized",jobs=jobs,
          counts=dict(model_operator_checks=16,independent_processes=32,proposals=64,backend_evaluations=0,quality_candidates=0)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("manifest")
    for name in ("child","seed"):
        p=sub.add_parser(name);p.add_argument("--spec",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
        if name=="seed": p.add_argument("--seed",type=int,required=True)
    args=parser.parse_args()
    if args.command=="manifest": manifest()
    elif args.command=="seed": seed_check(json.loads(args.spec.read_text(encoding="utf-8")),args.out,args.seed)
    else: child(json.loads(args.spec.read_text(encoding="utf-8")),args.out)
