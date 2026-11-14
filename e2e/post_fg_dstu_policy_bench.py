"""Original/fast policy comparison on the same objects through fast_hop."""
import argparse
import copy
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import random
import time
from types import SimpleNamespace

import post_fg_dstu as runner
from post_fg_dstu_policy import MessagePolicy as Original
from post_fg_dstu_policy_fast import MessagePolicy as Accelerated
from post_fg_dstu_check import Event, Link
import fast_hop


class PrefixFinished(Exception):
    pass


def compare(events, hp, metadata, rank, release, mode, seed, limit=None, traced=True, profile=False):
    saved = {tag: copy.deepcopy(event.__dict__) for tag, event in events.items()}
    devices = list(hp.links_dict.values()) + [d for group in hp.modules_dict.values() for d in group.values()]
    device_state = [(d, d.work_flag, d.work_endtime, list(d.work_record)) for d in devices]
    runs = []
    outputs = []
    for cls in (Original, Accelerated):
        for tag, state in saved.items():
            events[tag].__dict__.clear()
            events[tag].__dict__.update(copy.deepcopy(state))
        for device, flag, end, record in device_state:
            device.work_flag, device.work_endtime, device.work_record = flag, end, list(record)
        trace = []
        def emit(row):
            trace.append(row)
            if limit is not None and len(trace) == limit:
                raise PrefixFinished
        policy = cls(rank, metadata, mode, emit if traced else None)
        random.seed(seed)
        released = {}
        audit = []
        prof = cProfile.Profile() if profile else None
        if prof:
            prof.enable()
        begin = time.perf_counter()
        try:
            result = fast_hop.event_driver_fast(events, hp, release, policy, released, audit)
        except PrefixFinished:
            result = "prefix"
        seconds = time.perf_counter()-begin
        if prof:
            prof.disable()
        value = dict(result=result, trace=trace, released=released, audit=audit, stats=dict(policy.stats),
                     owner=dict(policy.owner), preferred=dict(policy.preferred), ready_since=dict(policy.ready_since),
                     events={tag:(e.start_time,e.end_time,e.path_list,e.dependency_set) for tag,e in events.items()},
                     resources=[copy.deepcopy(d.work_record) for d in devices], rng=random.getstate())
        value = runner.plain(value)
        outputs.append(value)
        row = dict(implementation=cls.__module__, seconds=seconds, issues=policy.stats["issues"],
                   trace_rows=len(trace), stopped_at_simulated_time=trace[-1]["t"] if trace else None,
                   digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest())
        if prof:
            text = io.StringIO()
            pstats.Stats(prof,stream=text).sort_stats("cumulative").print_stats(18)
            row["profile"] = text.getvalue()
        runs.append(row)
    fields = {key: outputs[0][key] == outputs[1][key] for key in outputs[0]}
    assert all(fields.values()), fields
    for tag, state in saved.items():
        events[tag].__dict__.clear()
        events[tag].__dict__.update(copy.deepcopy(state))
    for device, flag, end, record in device_state:
        device.work_flag, device.work_endtime, device.work_record = flag, end, list(record)
    return dict(status="passed",fields=fields,mode=mode,seed=seed,traced=traced,
                issue_limit=limit,profiled=profile,speedup=runs[0]["seconds"]/runs[1]["seconds"],runs=runs)


def synthetic():
    cases = []
    for seed in range(8):
        rng = random.Random(seed)
        links = [(i,i+1) for i in range(8)]
        events, metadata = {}, {}
        for parent in range(14):
            stream = parent%2
            count = rng.randrange(1,5)
            parent_tag = (1,10+stream,0,parent)
            paths = [rng.sample(links,rng.randrange(1,6)) for _ in range(rng.randrange(1,4))]
            dependencies = set(rng.sample(list(events), min(len(events), rng.randrange(3))))
            for index in range(count):
                tag = (1,10+stream,0,parent+index*1_000_000)
                event = Event(tag,paths,rng.randrange(1,9))
                event.dependency_set = set(dependencies)
                events[tag] = event
                metadata[tag] = dict(parent=parent_tag,index=index,count=count)
                for predecessor in dependencies:
                    events[predecessor].issue_set.add(tag)
        hp = SimpleNamespace(links_dict={link:Link() for link in links}, modules_dict={})
        for mode in ("C4","TieFIFO","MsgCont"):
            for traced in (False,True):
                cases.append(compare(events,hp,metadata,{0:1,1:0},{0:0,1:3},mode,seed,traced=traced))
    return dict(status="passed",cases=cases,count=len(cases),scope="synthetic graphs, actual fast_hop backend; same object identities")


def contention(profile):
    links = [(i,i+1) for i in range(7)]
    events, metadata = {}, {}
    for parent in range(24):
        parent_tag = (1,10+parent%2,0,parent)
        for index in range(12):
            tag = parent_tag[:-1]+(parent+index*1_000_000,)
            events[tag] = Event(tag,[links[:5],links[:3]+links[5:]],3+parent%3)
            metadata[tag] = dict(parent=parent_tag,index=index,count=12)
    hp = SimpleNamespace(links_dict={link:Link() for link in links},modules_dict={})
    result = compare(events,hp,metadata,{0:1,1:0},{0:0,1:3},"MsgCont",123,traced=True,profile=profile)
    result["scope"] = "constructed 288-chunk contention graph, real fast_hop; not a WSC run"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("smoke","contention","real-prefix"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--issues",type=int,default=10000)
    parser.add_argument("--source-root",type=Path)
    parser.add_argument("--profile",action="store_true")
    args = parser.parse_args()
    if args.command == "smoke":
        result = synthetic()
    elif args.command == "contention":
        result = contention(args.profile)
    else:
        manifest = runner.read_json(runner.ROOT/"e2e/post_fg/manifest/dstu_smoke.json")
        spec = next(s for s in manifest["jobs"] if s["arm"]=="A256-M" and s["run_type"]=="joint" and s["phase"]==.1)
        if args.source_root is not None:
            spec = dict(spec,source_root=str(args.source_root))
        _, hp, _, streams = runner.graph_inputs(spec,args.out.parent)
        events, metadata, _, _ = runner.transform(streams,hp,spec["arm"],spec["rank"])
        result = compare(events,hp,metadata,spec["rank"],{0:0,1:spec["tau"]},"MsgCont",spec["seed"],args.issues,profile=args.profile)
        result.update(source_spec=spec,scope="prefix on the complete WSC All256 joint graph; no complete-job speed claim")
    runner.write_json(args.out,result)
    print(json.dumps(dict(status=result["status"],output=str(args.out),count=result.get("count"),speedup=result.get("speedup"))))


if __name__ == "__main__":
    main()
