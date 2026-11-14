"""Read-only H1 snapshot replay: exact saved mapping/routes and measured activities."""
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

import hi_config as H
import e_run


def worker(out,name,evidence):
    begin=time.perf_counter()
    record=json.loads((out/"results"/(name+".json")).read_text(encoding="utf-8"))
    pilot,c,cfg=H.prepare(record,out)
    with gzip.open(out/"raw"/(name+".pkl.gz"),"rb") as f:
        raw=pickle.load(f)
    snapshot=e_run.load_snapshot(out/"snapshots"/(name+".pkl.gz"),c,cfg)
    assert e_run.signature(snapshot)==record["workload_signature"]
    beha,data,events,hp=snapshot
    pre=c.cpcm_sim(events,hp)[1]
    assert int(pre.T)==record["T_hat_pre"]
    import numpy as np
    from add_communication import add_broadcast
    ctx=raw["meta"]["ctx"]
    random.seed(c.BCAST_SEED);np.random.seed(c.BCAST_SEED)
    add_broadcast(data_tags=ctx["broadcast_datatags"],ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]],
                  beha_dict=beha,data_dict=data,hardware_platform=hp,event_dict=events,
                  dijkstra_routing=ctx["dijkstra_routing"],alpha=ctx["alpha"],beta=ctx["beta"],gamma=ctx["gamma"])
    before=c.e3._e2()._snapshot_pre(events)
    actual=c.e3.mapping_of(dict(beha_dict=beha,data_dict=data))
    actual["routes"]=c.e3.routes_of(events,before)
    actual["layout"]=c.e3.event_layout_of(events)
    expected=raw["selected"]["e3map"]
    assert set(actual)==set(expected)
    assert actual==expected, "saved source/route/layout differs after snapshot reload and BCAST_SEED"
    graph=c.cpcm.compile_bb(events,hp)
    schedule=c.cpcm.simulate(graph,"bb")
    assert int(schedule.T)==record["T_hat"]
    comps=sum(e.event_type=="computation" for e in events.values())
    result=dict(status="passed",candidate_id=name,T_hat_pre=int(pre.T),T_hat=int(schedule.T),
                original_cpcm=True,exact_mapping_routes_and_layout=True,workload_signature_equal=True,
                measured_schedule_activities=len(schedule.a_s),
                branch_hop_activity_references=comps+len(graph.hop_link),
                original_result_activities_field=record["graph"]["activities"],
                activity_field_correction="original graph.activities counts duplicated branch-hop references; use measured_schedule_activities",
                events=graph.n,dependency_edges=len(graph.pred),backend_calls=0,sa_steps=0,
                elapsed_seconds=time.perf_counter()-begin,
                input_sha256={folder:hashlib.sha256((out/folder/(name+ext)).read_bytes()).hexdigest()
                              for folder,ext in (("raw",".pkl.gz"),("snapshots",".pkl.gz"),("results",".json"))})
    e_run.write_json(evidence,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=H.HERE/"hi/out/h1_main")
    p.add_argument("--name")
    p.add_argument("--evidence",type=Path,default=H.HERE/"hi/evidence/h1_graph_measured.json")
    a=p.parse_args()
    if a.name:
        worker(a.out,a.name,a.evidence)
    else:
        rows=[]
        for record in sorted((a.out/"results").glob("*.json")):
            part=a.evidence.parent/"h1_graph_parts"/(record.stem+".json")
            subprocess.run([sys.executable,str(Path(__file__).resolve()),"--out",str(a.out),"--name",record.stem,"--evidence",str(part)],check=True)
            rows.append(json.loads(part.read_text(encoding="utf-8")))
        e_run.write_json(a.evidence,dict(status="passed",rows=rows,backend_calls=0,sa_steps=0,
            note="Original results remain unchanged; these are measured activities from exact saved full broadcast graphs."))
        print(json.dumps(dict(status="passed",n=len(rows),activities={r["candidate_id"]:r["measured_schedule_activities"] for r in rows})))
