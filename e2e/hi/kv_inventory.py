"""Record actual native K/V messages in four fresh, non-search graph builds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hi_config as H
from hi_model_check import source_prefix,write


def worker(model,phase):
    spec=dict(model=model,phase=phase,op="mha",method="A3h",g=0,x=123)
    pilot,c,cfg=H.prepare(spec,H.HERE/"hi/out/kv")
    task=H.generate(spec)
    source=task.read_text(encoding="utf-8")
    ctx={"__file__":str(task),"__name__":"__main__"}
    exec(compile(source_prefix(source,True),str(task),"exec"),ctx)
    events=ctx["initial_event_dict"]
    added=set(events)-set(ctx["event_dict"])
    messages=[]
    for tag in sorted(added):
        e=events[tag]
        owners=[d for d in ctx["cache_list"] if set(e.dependency_set)&set(ctx["data_dict"][d].generated_tag_splitted_dict)]
        assert len(owners)==1
        targets=sorted(e.target_location)
        edges={edge for path in e.path_list for edge in path}
        messages.append(dict(tag=tag,cache_tensor_tag=owners[0],name=e.event_name,source=e.source_location,
                             targets=targets,multicast_receivers=len(targets),payload_bytes=e.comm_bytes,
                             destination_payload_bytes=e.comm_bytes*len(targets),
                             unique_directed_link_bytes=e.comm_bytes*len(edges),
                             branch_path_link_bytes=e.comm_bytes*sum(map(len,e.path_list))))
    cache=[]
    for tag in ctx["cache_list"]:
        d=ctx["data_dict"][tag]
        cache.append(dict(tag=tag,name=d.data_name,shape=d.data_shape,logical_bytes=d.data_size,
                          producer_tags=list(d.generated_tag_splitted_dict)))
    assert sum(d["logical_bytes"] for d in cache)==sum(m["payload_bytes"] for m in messages)
    value=dict(status="passed",spec=spec,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
               cache_outputs=cache,messages=messages,
               totals={k:sum(m[k] for m in messages) for k in ("payload_bytes","destination_payload_bytes","unique_directed_link_bytes","branch_path_link_bytes")},
               native_target="32 DDR ports on the same 2x2 group; cross-group joint transfer not executed here",
               backend_evaluations=0,searches=0)
    path=H.HERE/f"hi/evidence/kv_{model}_{phase}.json"
    write(path,value)
    print(json.dumps(dict(path=str(path),totals=value["totals"])))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=list(H.MODELS))
    p.add_argument("--phase",choices=["decode","prefill"])
    a=p.parse_args()
    if a.model:
        worker(a.model,a.phase)
    else:
        for model in H.MODELS:
            for phase in ("decode","prefill"):
                subprocess.run([sys.executable,str(Path(__file__).resolve()),"--model",model,"--phase",phase],check=True)
