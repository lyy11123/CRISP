"""Recompute the original C4 rank rule on a selected strategy's actual graph.

This is a separate CPCM check after strategy selection, not a backend rerun.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import post_fg_dstu as dstu


def run(manifest,arm,out):
    jobs=dstu.read_json(manifest)["jobs"]
    representatives={}
    for spec in jobs:
        if spec["arm"]==arm and spec["run_type"]=="joint":
            representatives.setdefault((spec["hardware_profile"],spec["condition"]),spec)
    assert representatives,arm
    rows=[];costs=[]
    for key,spec in representatives.items():
        started=time.perf_counter()
        d,hp,g,streams=dstu.graph_inputs(spec,out)
        events,_,_,stats=dstu.transform(streams,hp,arm,spec["rank"])
        estimates={}
        for stream in (0,1):
            ev={tag:e for tag,e in events.items() if tag[1]-10==stream}
            total,seconds,n=d.that_hop(ev,copy.deepcopy(hp))
            estimates[stream]=int(total)
            costs.append(dict(hardware=key[0],condition=key[1],arm=arm,stream=stream,
                              T_hat_ns=int(total),CPCM_seconds=seconds,events=n))
        for phase in (.1,.5,.9):
            reference=spec["reference"]
            tau=int(round(reference["KV_start"]+phase*(reference["KV_end"]-reference["KV_start"])))
            deadline={s:1.5*estimates[s]+(tau if s else 0) for s in (0,1)}
            order=sorted(deadline,key=lambda s:(deadline[s],s))
            rank={str(s):order.index(s) for s in (0,1)}
            old=next(s for s in jobs if s["hardware_profile"]==key[0] and s["condition"]==key[1]
                     and s["arm"]==arm and s["run_type"]=="joint" and s["phase"]==phase)["rank"]
            rows.append(dict(hardware=key[0],condition=key[1],arm=arm,phase=phase,tau_ns=tau,
                             original_rule_deadlines_ns=deadline,frozen_rank=old,recomputed_rank=rank,
                             same_rank=rank==old,graph_sha256=dstu.graph_digest(events),
                             shared_mask_frozen_rank=spec["rank"],selected_messages=stats["selected_messages"]))
        print(f"[C4 rank] {key} {arm}: {time.perf_counter()-started:.2f}s",flush=True)
    result=dict(status="passed" if all(r["same_rank"] for r in rows) else "rank_changed",
                rule="d_P=1.5*that(P), d_D=tau+1.5*that(D); stable stream-ID tie",
                original_policy_rule="CPCM hop on each transformed stream; no MsgCont-specific estimator",
                backend_calls=0,CPCM_calls=len(costs),rows=rows,timing=costs,
                action="Rank changes require a separately specified small confirmation before adopting normal rank generation")
    dstu.write_json(Path(out)/f"rank_{arm}.json",result)
    print(json.dumps(result))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--arm",required=True)
    p.add_argument("--out",default="e2e/post_fg/out/rankcheck")
    a=p.parse_args();run(a.manifest,a.arm,a.out)
