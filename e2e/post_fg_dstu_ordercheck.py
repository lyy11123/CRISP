"""Compare complete graph semantics and actual dependency iteration order."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import post_fg_dstu as runner


def ordered_digest(events):
    rows=[]
    for tag,event in events.items():
        rows.append(dict(tag=tag,dependencies=list(event.dependency_set),
                         successors=list(event.issue_set),
                         targets=list(event.target_location) if event.event_type=="communication" and isinstance(event.target_location,set) else None))
    value=dict(semantics=runner.graph_digest(events),iteration=rows)
    return hashlib.sha256(json.dumps(runner.plain(value),separators=(",",":"),sort_keys=True).encode()).hexdigest()


def comparison(left,right):
    return dict(semantic_equal=runner.graph_digest(left)==runner.graph_digest(right),
                event_order_equal=list(left)==list(right),
                dependency_order_differences=sum(list(left[t].dependency_set)!=list(right[t].dependency_set) for t in left),
                issue_order_differences=sum(list(left[t].issue_set)!=list(right[t].issue_set) for t in left),
                ordered_left=ordered_digest(left),ordered_right=ordered_digest(right))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware",choices=("wsc","face"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    spec=next(s for s in runner.read_json(runner.ROOT/"e2e/post_fg/manifest/dstu_health.json")["jobs"]
              if s["hardware_profile"]==args.hardware and s["arm"]=="A256-M" and s["run_type"]=="joint")
    _,hp,graph,_=runner.graph_inputs(spec,args.out.parent.parent/"out/order_checks"/args.hardware)
    graphs={}
    for arm in ("A256","A256-M","H0","S256","BB0"):
        streams={}
        for stream in (0,1):
            with gzip.open(runner.source(spec,graph["files"][f"orig_{stream}"]),"rb") as handle:
                streams[stream]=pickle.load(handle)
        events,_,_,_=runner.transform(streams,hp,arm,spec["rank"])
        graphs[arm]=events
    all_pair=comparison(graphs["A256"],graphs["A256-M"])
    raw_pair=comparison(graphs["BB0"],graphs["H0"])
    decode={arm:{t:e for t,e in graphs[arm].items() if t[1]==11} for arm in ("H0","S256")}
    solo_pair=comparison(decode["H0"],decode["S256"])
    checks=dict(All256_C4_vs_MsgCont=all_pair,BB0_vs_H0=raw_pair,H0_D_vs_S256_D=solo_pair)
    for row in checks.values():
        assert row["semantic_equal"] and row["event_order_equal"]
        assert row["dependency_order_differences"]==row["issue_order_differences"]==0
        assert row["ordered_left"]==row["ordered_right"]
    runner.write_json(args.out,dict(status="passed",hardware=args.hardware,checks=checks,
                                   protocol="fresh_transform_v1",new_backend_calls=0,
                                   construction="Fresh gzip load of each original stream for each arm; no deepcopy or pickle normalization"))
    print(json.dumps(dict(status="passed",hardware=args.hardware,output=str(args.out),checks=checks)))


if __name__=="__main__":main()
