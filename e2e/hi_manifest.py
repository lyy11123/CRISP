"""The frozen 896 H/I candidates and the four currently authorized H1 pilots."""
import collections
import hashlib
import json
from pathlib import Path
import hi_config
from e_run import ROOT, write_json


def candidates():
    rows = []
    for model in hi_config.MODELS:
        contract = hi_config.model_contract(model)
        for hardware in ("busybarn", "watos"):
            for phase in ("decode", "prefill"):
                for op in ("ln", "mha", "proj", "ffn") if hardware == "busybarn" else ("mha", "ffn"):
                    for method in ("A1", "A3h", "Gemini") if hardware == "busybarn" else ("A1", "A3h"):
                        variants = [(0,x) for x in (123,*range(1,14))] if method == "Gemini" else [
                            (g,x) for g in (0,1) for x in (range(7) if method == "A1" else (123,1,2,3,4,5,6))]
                        for order,(g,x) in enumerate(variants):
                            prefix="H" if model=="llama" else "I"
                            first = method=="A3h" and g==0 and x==123 and op in ("mha","ffn")
                            if hardware=="watos":
                                batch=prefix+("1c" if first and prefix=="H" else "1W" if first else "3")
                            else:
                                batch=prefix+("1a" if phase=="decode" else "1b") if first else prefix+("2D" if phase=="decode" else "2P")
                            row=dict(candidate_id=f"{prefix}_{model}_{hardware}_{phase}_{op}_{method}_g{g}_{'l' if method=='A1' else 's'}{x}",
                                     module="hi_run",kind="search",model=model,hardware_profile=hardware,
                                     shape="2x2",split_mode="fixed",d2d_scale=1.0,phase=phase,op=op,method=method,g=g,x=x,
                                     candidate_order=order,batch=batch,search_steps=1000,
                                     backend_seeds=list(range(1,11)),random_cpcm_seeds=list(range(1,11)),
                                     model_contract_version=hi_config.VERSION,graph_order_version=hi_config.GRAPH_ORDER_VERSION,
                                     workload_version=hi_config.WORKLOAD_VERSION,
                                     model_contract_sha256=hi_config.digest(contract),
                                     fault_id=None,repair_seed=None,repair_steps=None,backend_seed=None,strategy=None)
                            row["config_id"]=hi_config.identity(row)
                            rows.append(row)
    assert len(rows)==896 and len({r['candidate_id'] for r in rows})==896
    return rows


if __name__=="__main__":
    rows=candidates()
    folder=ROOT/"e2e/hi/manifest"
    write_json(folder/"all_candidates_planned.json",dict(status="planned_only",jobs=rows))
    pilots=[r for r in rows if r['batch'] in ('H1a','H1b')]
    assert len(pilots)==4
    write_json(folder/"h1_main.json",dict(status="authorized",jobs=pilots))
    precheck=[]
    for model in hi_config.MODELS:
        for hardware in ('busybarn','watos'):
            for phase in ('decode','prefill'):
                for op in ('ln','mha','proj','ffn') if hardware=='busybarn' else ('mha','ffn'):
                    for method in ('A3h','Gemini') if hardware=='busybarn' else ('A3h',):
                        name=f'check_{model}_{hardware}_{phase}_{op}_{method}'
                        spec=dict(model=model,hardware_profile=hardware,phase=phase,op=op,method=method,g=0,x=123)
                        argv=['{python}','-u','-B','e2e/hi_model_check.py','child','--spec','{spec}',
                              '--out','{out}/results/'+name+'.json']
                        if model=='llama' and hardware=='busybarn' and phase=='decode' and op in ('mha','ffn') and method=='A3h':
                            argv.append('--fast')
                        precheck.append(dict(id=name,kind='model_check',deps=[],spec=spec,
                                             result='results/'+name+'.json',argv=argv,
                                             estimated_seconds=600 if phase=='prefill' else 60,
                                             peak_rss_gb=6 if phase=='prefill' else 2))
    assert len(precheck)==40
    write_json(folder/'model_checks.json',dict(status='authorized',jobs=precheck))
    write_json(folder/"batch_counts.json",dict(total=896,per_model=dict(collections.Counter(r['model'] for r in rows)),
               per_batch=dict(collections.Counter(r['batch'] for r in rows)),authorized_searches=4,
               pilots_count_toward_formal_candidates=True))
    print(json.dumps(dict(total=len(rows),authorized_pilots=len(pilots))))
