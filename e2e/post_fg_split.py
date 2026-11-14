"""MF6 orphan-only output tiling on saved event graphs, independent of EIR SA.

prepare: build and audit m1/m2/m4 graphs; no backend evaluations.
child: evaluate one prepared graph with one backend seed (fg_batch interface).
"""

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter, defaultdict
import configparser
import ast
import copy
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import pickle
import random
import statistics
import sys
import time

import e_gemini_analysis as static
import f_config
from post_fg_diag import CASES, read, write, table

ROOT = Path(__file__).resolve().parents[1]
VERSION = "post_fg_split_v1"


def setup():
    static.ARTIFACT = f_config.artifact_root()
    static.pickle_types()
    for path in (static.ARTIFACT / "utils", static.ARTIFACT / "src/backend/analytical", ROOT / "cpcm"):
        sys.path.insert(0, str(path))


def platform(spec, dead=None):
    setup()
    import WAMIS_HD
    f_config.install_hardware()
    parser = configparser.ConfigParser()
    parser.read_string(f_config.config_text(spec))
    cfg = {section: {k: ast.literal_eval(v) if v[:1].isdigit() or v.startswith("[") else v
                     for k, v in parser[section].items()} for section in parser.sections()}
    cfg["failures"] = dict(failed_nodes=[], failed_links=[], failed_pes=[tuple(dead)] if dead else [])
    return WAMIS_HD.wamis_hdc(cfg)


def write_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with gzip.open(temporary, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def volume(region):
    return math.prod(b-a+1 for a,b in region)


def overlap(left, right):
    assert len(left) == len(right)
    out = tuple((max(a,c), min(b,d)) for (a,b),(c,d) in zip(left,right))
    return out if all(a <= b for a,b in out) else None


def bounds(regions):
    return tuple((min(r[i][0] for r in regions), max(r[i][1] for r in regions))
                 for i in range(len(next(iter(regions)))))


def clipped(regions, axis, start, end):
    result = []
    for region in sorted(regions):
        box = list(region)
        box[axis] = (max(box[axis][0],start), min(box[axis][1],end))
        if box[axis][0] <= box[axis][1]:
            result.append(tuple(box))
    return result


def service(event, hp):
    return int(hp.modules_dict[event.comp_device][event.comp_location].working_time(
        event.comp_datashape, event.comp_type, hp.frequency))


def tensor_work(shape):
    a,b = shape[:2]
    k,outputs = a[-1],math.prod(a[:-1])*b[-1]
    return 2*k*outputs, 2*math.ceil(k/128)*128*math.ceil(outputs/64)*64


class Router:
    """Deterministic shortest service-time paths, without SA history updates."""
    def __init__(self, hp):
        self.hp, self.adj, self.cache = hp, defaultdict(list), {}
        for edge in sorted(hp.links_dict, key=str):
            self.adj[tuple(edge[0])].append((tuple(edge[1]),edge))

    def routes(self, source, nbytes):
        source = tuple(source)
        key = source,int(nbytes)
        if key in self.cache:
            return self.cache[key]
        distances = {source:0}
        paths = {source:()}
        queue = [(0,source)]
        while queue:
            cost,node = heapq.heappop(queue)
            if cost != distances[node]:
                continue
            for target,edge in self.adj[node]:
                alt = cost + int(self.hp.links_dict[edge].working_time(nbytes))
                if alt < distances.get(target,math.inf):
                    distances[target] = alt
                    paths[target] = paths[node] + (edge,)
                    heapq.heappush(queue,(alt,target))
        self.cache[key] = paths
        return paths

    def path(self, source, target, nbytes):
        return self.routes(source,nbytes)[tuple(target)]

    def link_bytes(self, source, target, nbytes):
        return len(self.path(source,target,nbytes))*nbytes


def read_case(run_out, case):
    _,cid,op = next(row for row in CASES if row[0] == case)
    selected = read(run_out/"selection"/(cid+".json"))["operators"][op]["A3h"]
    blob,_ = static.read_pickle(ROOT/selected["snapshot"])
    behaviours,data,_ = pickle.loads(blob)
    pre = read(run_out/"results"/f"{cid}__pre_{op}_A3h.json")
    exported = static.read_pickle(ROOT/pre["exports"]["healthy"])
    events = pickle.loads(exported["events_blob"])
    dead = tuple(read(run_out/"faults"/(cid+".json"))["faults"]["MF6"]["failed_pes"][0])
    assert all(tuple(events[tag].comp_location) == tuple(b.location) for tag,b in behaviours.items())
    return dict(case=case,config_id=cid,op=op,source_candidate=selected["candidate_id"],
                snapshot=selected["snapshot"],healthy_export=pre["exports"]["healthy"],
                spec=pre["spec"],dead=dead,behaviours=behaviours,data=data,events=events,
                T_pre=pre["T"],backend_pre=pre["backend_mean"])


def input_flows(state):
    """Locate every affected operand region in the original live graph."""
    from data_notation import type_bytes
    ev,behaviours,data,dead = (state[k] for k in ("events","behaviours","data","dead"))
    orphan = {tag for tag,e in ev.items() if e.event_type == "computation" and tuple(e.comp_location[:2]) == dead}
    touched = set(orphan)
    for tag in orphan:
        touched.update(behaviours[tag].consumer_tags)
    flows = []
    for consumer in sorted(touched):
        behaviour = behaviours[consumer]
        for operand,dt in enumerate(behaviour.beha_datatag):
            tensor = data[dt]
            for region in sorted(behaviour.needed_data_split_dict[dt]):
                producers = tensor.generated_splitted_tag_dict[region] - {(-1,)}
                assert len(producers) <= 1, (consumer,dt,region,producers)
                producer = next(iter(producers)) if producers else None
                if consumer not in orphan and producer not in orphan:
                    continue
                nbytes = volume(region)*type_bytes[tensor.data_type]
                tags = tensor.used_splitted_tag_dict[region] & ev[consumer].dependency_set
                comms = [tag for tag in tags if ev[tag].event_type == "communication"
                         and ev[tag].comm_bytes == nbytes and ev[tag].issue_set == {consumer}
                         and ev[tag].dependency_set == producers]
                if comms:
                    assert len(comms) == 1, (consumer,dt,region,comms)
                    original = comms[0]
                    source = tuple(ev[original].source_location)
                    assert tuple(ev[original].target_location) == tuple(ev[consumer].comp_location[:2])
                else:
                    assert producer in ev[consumer].dependency_set, (consumer,dt,region,"missing live operand")
                    original = None
                    source = tuple(ev[producer].comp_location[:2])
                    assert source == tuple(ev[consumer].comp_location[:2])
                if producer:
                    assert source == tuple(ev[producer].comp_location[:2]), (consumer,producer,source)
                flows.append(dict(consumer=consumer,producer=producer,data=dt,operand=operand,region=region,
                                  original_comm=original,source=source,nbytes=nbytes,element_bytes=type_bytes[tensor.data_type],
                                  data_name=tensor.data_name,offline=dt[0] == 1))
    # Each original dependency of an orphan is explained by an operand flow.
    for tag in orphan:
        explained = {f["original_comm"] or f["producer"] for f in flows if f["consumer"] == tag}
        assert explained == ev[tag].dependency_set, (tag,explained,ev[tag].dependency_set)
    return orphan,flows


def pieces_for(tag, event, behaviour, factor, axis):
    split = event.comp_device == "tensorcore" and factor > 1
    count = factor if split else 1
    if split:
        dt = behaviour.beha_datatag[1 if axis == "N" else 0]
        dim = -1 if axis == "N" else 0
        start,end = bounds(behaviour.needed_data_split_dict[dt])[dim]
        assert (end-start+1) % count == 0, (tag,axis,count)
        width = (end-start+1)//count
    pieces = []
    for index in range(count):
        need = {d:list(sorted(s)) for d,s in behaviour.needed_data_split_dict.items()}
        output = {d:list(sorted(s)) for d,s in behaviour.produced_data_split_dict.items()}
        shape = copy.deepcopy(event.comp_datashape)
        if split:
            lo,hi = start+index*width,start+(index+1)*width-1
            if axis == "N":
                for d in behaviour.beha_datatag[1:]:
                    need[d] = clipped(need[d],-1,lo,hi)
                shape[1][-1] //= count
                if len(shape) == 3:
                    shape[2][-1] //= count
            else:
                d = behaviour.beha_datatag[0]
                need[d] = clipped(need[d],0,lo,hi)
                shape[0][0] //= count
            output = {d:clipped(regions,-1 if axis == "N" else 0,lo,hi) for d,regions in output.items()}
        newtag = tag+(90,index) if split else tag
        pieces.append(dict(tag=newtag,parent=tag,index=index,needed=need,produced=output,shape=shape,
                           device=event.comp_device,kind=event.comp_type,old_location=tuple(event.comp_location)))
    for dt,regions in behaviour.produced_data_split_dict.items():
        assert sum(volume(r) for p in pieces for r in p["produced"][dt]) == sum(map(volume,regions))
    return pieces


def event_metrics(events, hp):
    compute = defaultdict(Counter)
    communication = Counter()
    effective,rounded = 0,0
    for e in events.values():
        if e.event_type == "computation":
            compute[e.comp_device][tuple(e.comp_location)] += service(e,hp)
            if e.comp_device == "tensorcore":
                a,b = tensor_work(e.comp_datashape)
                effective += a; rounded += b
        else:
            edges = set(edge for path in e.path_list for edge in path)
            communication["logical_network_bytes"] += e.comm_bytes
            communication["route_link_bytes"] += len(edges)*e.comm_bytes
            communication["branch_path_link_bytes"] += sum(map(len,e.path_list))*e.comm_bytes
            if e.source_location[1] >= 16:
                communication["ddr_read_bytes"] += e.comm_bytes
            targets=e.target_location if isinstance(e.target_location,set) else {e.target_location}
            communication["ddr_write_bytes"] += sum(tuple(t)[1]>=16 for t in targets)*e.comm_bytes
            for edge in edges:
                kind = hp.links_dict[edge].link_type
                communication[kind+"_link_bytes"] += e.comm_bytes
                communication[kind+"_service_ns"] += int(hp.links_dict[edge].working_time(e.comm_bytes))
    return dict(events=len(events),dependencies=sum(len(e.dependency_set) for e in events.values()),
                computation_events=sum(e.event_type == "computation" for e in events.values()),
                tensor_effective_flops=effective,tensor_rounded_flops=rounded,
                compute_peak_ns={k:max(v.values()) for k,v in compute.items()},
                compute_total_ns={k:sum(v.values()) for k,v in compute.items()},
                communication=dict(communication))


def tensor_bounds(events,hp):
    times=[service(e,hp) for e in events.values() if e.event_type=="computation" and e.comp_device=="tensorcore"]
    live=len(hp.tensorcore_dict)
    return dict(live_units=live,blocks=len(times),basic_ns=max(max(times),sum(times)/live),
                count_ns=math.ceil(len(times)/live)*min(times),
                lower_bound_ns=max(max(times),sum(times)/live,math.ceil(len(times)/live)*min(times)))


def canonical(events):
    rows=[]
    for tag,e in events.items():
        common=(tag,e.event_type,tuple(sorted(e.dependency_set)),tuple(sorted(e.issue_set)))
        if e.event_type == "computation":
            rows.append(common+(e.comp_device,tuple(e.comp_location),e.comp_type,e.comp_datashape))
        else:
            target=("multicast",tuple(sorted(e.target_location))) if isinstance(e.target_location,set) else ("unicast",tuple(e.target_location))
            rows.append(common+(tuple(e.source_location),target,e.comm_bytes,e.path_list))
    return rows


def graph_digest(events):
    encoded=json.dumps(canonical(events),ensure_ascii=True,separators=(",",":"),default=int).encode()
    return hashlib.sha256(encoded).hexdigest()


def adapt(state, hp, factor, axis, identity=False):
    from event_notation import communication_notation
    from net import TupleIdx
    started=time.perf_counter()
    original=state["events"]
    orphan,flows=input_flows(state)
    router=Router(hp)
    pieces={tag:pieces_for(tag,original[tag],state["behaviours"][tag],factor,axis) for tag in sorted(orphan)}
    loads=defaultdict(Counter)
    for tag,e in original.items():
        if e.event_type == "computation" and tag not in orphan:
            loads[e.comp_device][tuple(e.comp_location)]+=service(e,hp)
    placements=[]
    for tag,children in pieces.items():
        used=set()
        for child in children:
            example=copy.copy(original[tag]);example.comp_datashape=child["shape"]
            choices=sorted(hp.modules_dict[child["device"]])
            # Estimates use the other endpoints' original locations; complete
            # reconstructed path bytes decide the M/N comparison below.
            def score(loc):
                example.comp_location=loc
                own=service(example,hp)
                peak=max(max(loads[child["device"]].values(),default=0),loads[child["device"]][loc]+own)
                cost=0
                for f in flows:
                    if f["consumer"] == tag:
                        for need in child["needed"][f["data"]]:
                            cut=overlap(f["region"],need)
                            if cut:
                                amount=volume(cut)*f["element_bytes"]
                                cost+=router.link_bytes(f["source"],loc[:2],amount)
                    if f["producer"] == tag:
                        for produced in child["produced"].get(f["data"],[]):
                            cut=overlap(f["region"],produced)
                            if cut:
                                amount=volume(cut)*f["element_bytes"]
                                cost+=router.link_bytes(loc[:2],original[f["consumer"]].comp_location[:2],amount)
                return peak,loc[:2] in used,cost,loc
            loc=child["old_location"] if identity else min(choices,key=score)
            child["location"]=loc
            example.comp_location=loc
            loads[child["device"]][loc]+=service(example,hp)
            used.add(loc[:2])
            placements.append(dict(parent=list(tag),child=list(child["tag"]),device=child["device"],
                                   shape=child["shape"],old_location=list(child["old_location"]),location=list(loc),
                                   service_ns=service(example,hp),receiver_hops=len(router.path(state["dead"],loc[:2],1))))
    graph=copy.deepcopy(original)
    for tag,children in pieces.items():
        if len(children)>1:
            del graph[tag]
        for child in children:
            event=copy.deepcopy(original[tag])
            event.event_tag=child["tag"]
            event.comp_location=child["location"]
            event.comp_datashape=child["shape"]
            event.dependency_set=set()
            event.issue_set=set()
            graph[child["tag"]]=event
    transfers=[]
    next_tag=0
    def add_transfer(producer,consumer,source,target,nbytes,old,region,dt,offline,name):
        nonlocal next_tag
        source,target=tuple(source),tuple(target)
        unchanged=(old is not None and old in original and
                   source==tuple(original[old].source_location) and target==tuple(original[old].target_location)
                   and nbytes==original[old].comm_bytes and
                   original[old].dependency_set==({producer} if producer else set()) and
                   original[old].issue_set=={consumer})
        if source==target:
            assert producer is not None
            graph[consumer].dependency_set.add(producer)
            communication_tag=None
        else:
            if unchanged:
                communication_tag=old
                graph[old]=copy.deepcopy(original[old])
            else:
                communication_tag=(1,999,1,next_tag);next_tag+=1
                event=communication_notation(name,communication_tag,source,target,nbytes)
                event.path_list=[list(router.path(source,target,nbytes))]
                event.paths={edge[0]:[TupleIdx(edge[1],edge[2]) if len(edge)==3 else TupleIdx(edge[1])]
                             for edge in event.path_list[0]}
                event.dependency_set={producer} if producer else set()
                event.issue_set={consumer}
                graph[communication_tag]=event
            graph[consumer].dependency_set.add(communication_tag)
        transfers.append(dict(producer=producer,consumer=consumer,data=dt,region=region,bytes=nbytes,
                              source=source,target=target,communication_tag=communication_tag,
                              offline=offline,original_comm=old,original_path_preserved=unchanged))
    for f in flows:
        consumer,producer=f["consumer"],f["producer"]
        old=f["original_comm"]
        if old:
            graph.pop(old,None)
        if consumer not in orphan:
            graph[consumer].dependency_set.discard(old or producer)
        consumers=pieces.get(consumer,[dict(tag=consumer,needed={f["data"]:[f["region"]]},
                        location=original[consumer].comp_location)])
        for child in consumers:
            for need in child["needed"][f["data"]]:
                requested=overlap(f["region"],need)
                if not requested:
                    continue
                producers=pieces.get(producer,[dict(tag=producer,produced={f["data"]:[requested]},
                                                    location=f["source"]+(0,))])
                coverage=0
                for parent in producers:
                    for produced in parent["produced"].get(f["data"],[]):
                        cut=overlap(requested,produced)
                        if cut:
                            amount=volume(cut)*f["element_bytes"]
                            coverage+=amount
                            add_transfer(parent["tag"],child["tag"],parent["location"][:2],child["location"][:2],
                                         amount,old,cut,f["data"],f["offline"],original[consumer].event_name)
                assert coverage==volume(requested)*f["element_bytes"], (consumer,producer,requested,coverage)
    # Terminal output broadcasts belong to an orphan whole-vector event in these
    # three examples. Preserve their bytes/targets, rerouting only a moved source.
    broadcast_records=[]
    for tag in sorted(orphan):
        terminal=[(t,e) for t,e in original.items() if e.event_type=="communication" and
                  tag in e.dependency_set and not e.issue_set]
        for t,e in terminal:
            assert len(pieces[tag])==1, (tag,"split terminal broadcast requires output-region partition")
            new_source=pieces[tag][0]["location"][:2]
            event=copy.deepcopy(e)
            if tuple(new_source)!=tuple(e.source_location):
                event.source_location=tuple(new_source)
                targets=sorted(e.target_location) if isinstance(e.target_location,set) else [e.target_location]
                event.path_list=[list(router.path(new_source,target,e.comm_bytes)) for target in targets if tuple(target)!=tuple(new_source)]
                event.paths={}
                for path in event.path_list:
                    for edge in path:
                        item=TupleIdx(edge[1],edge[2]) if len(edge)==3 else TupleIdx(edge[1])
                        event.paths.setdefault(edge[0],[])
                        if item not in event.paths[edge[0]]:event.paths[edge[0]].append(item)
            graph[t]=event
            broadcast_records.append(dict(tag=t,bytes=e.comm_bytes,source_before=e.source_location,source_after=new_source,
                                          targets_unchanged=True))
    # issue_set is the exact reverse of the final dependency relation.
    for e in graph.values():e.issue_set=set();e.start_time=e.end_time=0
    for tag,e in graph.items():
        for dep in e.dependency_set:
            assert dep in graph,(tag,dep)
            graph[dep].issue_set.add(tag)
    for tag,e in original.items():
        if e.event_type=="computation" and tag not in orphan:
            assert tuple(graph[tag].comp_location)==tuple(e.comp_location)
            assert graph[tag].comp_datashape==e.comp_datashape
    # Every required region remains present, including all unchanged inputs of
    # healthy consumers and each original K-partial-sum operand.
    for tag,children in pieces.items():
        for child in children:
            for dt,regions in child["needed"].items():
                expected=sum(map(volume,regions))*__import__("data_notation").type_bytes[state["data"][dt].data_type]
                received=sum(t["bytes"] for t in transfers if t["consumer"]==child["tag"] and t["data"]==dt)
                assert received==expected,(tag,child["tag"],dt,received,expected)
    import cpcm
    compiled=cpcm.compile_bb(graph,hp) # structural DAG check only; no simulation
    metrics=event_metrics(graph,hp)
    assert metrics["tensor_effective_flops"]==sum(tensor_work(e.comp_datashape)[0] for e in original.values()
                                                if e.event_type=="computation" and e.comp_device=="tensorcore")
    if identity:
        graph={tag:graph[tag] for tag in original}
        assert canonical(graph)==canonical(original),"m1 identity changed events, bytes, deps or routes"
    else:
        assert all(tuple(e.comp_location[:2])!=state["dead"] for e in graph.values() if e.event_type=="computation")
    serial_pieces=[dict(parent=list(tag),children=[dict(tag=list(p["tag"]),shape=p["shape"],
                            needed={str(d):rs for d,rs in p["needed"].items()},
                            produced={str(d):rs for d,rs in p["produced"].items()},location=list(p["location"])) for p in ps])
                   for tag,ps in pieces.items()]
    arithmetic=[]
    from data_notation import type_bytes
    for tag,children in pieces.items():
        parent=original[tag]
        if parent.comp_device!="tensorcore":continue
        old=copy.copy(parent);old.comp_location=next(iter(hp.tensorcore_dict))
        eflop,rflop=tensor_work(parent.comp_datashape)
        child_flops=[tensor_work(p["shape"]) for p in children]
        b=state["behaviours"][tag]
        parent_inputs=[sum(map(volume,b.needed_data_split_dict[dt]))*type_bytes[state["data"][dt].data_type]
                       for dt in b.beha_datatag]
        child_inputs=[[sum(map(volume,p["needed"][dt]))*type_bytes[state["data"][dt].data_type]
                       for dt in b.beha_datatag] for p in children]
        arithmetic.append(dict(parent=list(tag),kind=parent.comp_type,parent_shape=parent.comp_datashape,
                          child_shapes=[p["shape"] for p in children],effective_flops=eflop,
                          child_effective_flops=sum(f[0] for f in child_flops),rounded_flops=rflop,
                          child_rounded_flops=sum(f[1] for f in child_flops),parent_service_ns=service(old,hp),
                          children_service_ns=[service(graph[p["tag"]],hp) for p in children],
                          parent_operand_logical_bytes=parent_inputs,child_operand_logical_bytes=child_inputs,
                          output_bytes=sum(volume(r)*type_bytes[state["data"][dt].data_type]
                                           for dt,rs in b.produced_data_split_dict.items() for r in rs)))
    original_communications={f["original_comm"] for f in flows if f["original_comm"] is not None}
    before_weight=sum(original[t].comm_bytes for t in original_communications
                      if any(f["original_comm"]==t and f["offline"] for f in flows))
    new_weight=sum(t["bytes"] for t in transfers if t["offline"] and t["communication_tag"] is not None)
    new_activation=sum(t["bytes"] for t in transfers if not t["offline"] and t["communication_tag"] is not None)
    resident=defaultdict(set)
    for t in transfers:resident[t["target"]].add((t["data"],t["region"],t["offline"]))
    residency={str(loc):sum(volume(region)*type_bytes[state["data"][dt].data_type] for dt,region,_ in ranges)
               for loc,ranges in resident.items()}
    info=dict(version=VERSION,factor=factor,axis=axis,case=state["case"],failed_pe=state["dead"],
              original_orphans=[list(t) for t in sorted(orphan)],parent_children=serial_pieces,placements=placements,
              transfers=transfers,terminal_broadcasts=broadcast_records,metrics=metrics,
              healthy_computation_moved=0,coverage_checked=True,DAG_events=compiled.n,
              original_reduction_events_preserved=True,identity=identity,
              parent_arithmetic=arithmetic,tensor_load_bound=tensor_bounds(graph,hp),
              affected_weight_network_bytes_before=before_weight,affected_weight_network_bytes_after=new_weight,
              affected_activation_network_bytes_after=new_activation,
              receiver_input_residency_upper_bytes=residency,
              residency_interpretation="sum of distinct received operand slices; lifetime overlap not assumed measured",
              weights_source="original live DDR read sources retained; no assumption that failed PE SRAM is destroyed",
              build_seconds=time.perf_counter()-started)
    return graph,info


def numerical_checks():
    import numpy as np
    rng=np.random.default_rng(417)
    checks=[]
    a=rng.integers(-3,4,size=(8,12,16));b=rng.integers(-3,4,size=(16,12));bias=rng.integers(-3,4,size=12)
    target=a@b+bias
    for factor in (2,4):
        for axis in ("N","M"):
            if axis=="N":
                result=np.concatenate([a@bb+cc for bb,cc in zip(np.split(b,factor,axis=-1),np.split(bias,factor))],axis=-1)
            else:
                result=np.concatenate([aa@b+bias for aa in np.split(a,factor,axis=0)],axis=0)
            assert np.array_equal(result,target)
            # Keep K partials distinct; split each output and retain the sum.
            aa=np.split(a,4,axis=-1);bb=np.split(b,4,axis=0)
            parts=[]
            for ak,bk in zip(aa,bb):
                if axis=="N":parts.append(np.concatenate([ak@bj for bj in np.split(bk,factor,axis=-1)],axis=-1))
                else:parts.append(np.concatenate([aj@bk for aj in np.split(ak,factor,axis=0)],axis=0))
            assert np.array_equal(sum(parts),a@b)
            checks.append(dict(factor=factor,axis=axis,bias_gemm_equal=True,K_partials_sum_equal=True))
    return checks


def prepare(run_out, out, cases):
    setup()
    numerical=numerical_checks()
    choices,graphs,jobs,identities=[],[],[],[]
    for case in cases:
        print("[split prepare]",case,flush=True)
        state=read_case(run_out,case)
        healthy_hp=platform(state["spec"])
        _,identity=adapt(state,healthy_hp,1,"N",identity=True)
        identities.append(dict(case=case,identity=True,events=identity["DAG_events"]))
        baseline=event_metrics(state["events"],healthy_hp)
        hp=platform(state["spec"],state["dead"])
        assert all(tuple(loc[:2])!=state["dead"] for kind in ("tensorcore","vectorunit") for loc in hp.modules_dict[kind])
        assert set(hp.links_dict)==set(healthy_hp.links_dict),"MF6 retains all router links"
        for factor in (1,2,4):
            candidates=[]
            for axis in (("N",) if factor==1 else ("N","M")):
                graph,info=adapt(state,hp,factor,axis)
                score=(info["metrics"]["tensor_rounded_flops"]-baseline["tensor_rounded_flops"],
                       info["metrics"]["communication"]["route_link_bytes"],0 if axis=="N" else 1)
                candidates.append((score,axis,graph,info))
            score,axis,graph,info=min(candidates,key=lambda row:row[0])
            for rank,other_axis,_,other in candidates:
                choices.append(dict(case=case,factor=factor,axis=other_axis,selected=axis==other_axis,
                                    rounded_flop_delta=rank[0],route_link_bytes=rank[1],
                                    build_seconds=other["build_seconds"],tensor_peak_ns=other["metrics"]["compute_peak_ns"]["tensorcore"],
                                    input_logical_bytes=sum(t["bytes"] for t in other["transfers"])))
            filename=out/"graphs"/f"{case}_m{factor}.pkl.gz"
            info["baseline_metrics"]=baseline
            info["axis_selection_seconds"]=sum(c[3]["build_seconds"] for c in candidates)
            all_original_times=[service(e,healthy_hp) for e in state["events"].values()
                                if e.event_type=="computation" and e.comp_device=="tensorcore"]
            alive=len(hp.tensorcore_dict)
            info["rigid_tensor_load_bound"]=dict(live_units=alive,blocks=len(all_original_times),
                lower_bound_ns=max(max(all_original_times),sum(all_original_times)/alive,
                                   math.ceil(len(all_original_times)/alive)*min(all_original_times)))
            payload=dict(events_blob=pickle.dumps(graph,pickle.HIGHEST_PROTOCOL),info=info,source_spec=state["spec"],
                         source_candidate=state["source_candidate"],source_export=state["healthy_export"],dead=state["dead"])
            write_pickle(filename,payload)
            fingerprint=graph_digest(graph)
            graphs.append(dict(case=case,factor=factor,axis=axis,graph=filename.relative_to(ROOT).as_posix(),
                               graph_sha256=fingerprint,metrics=info["metrics"],build_seconds=info["build_seconds"]))
            write(out/"metadata"/f"{case}_m{factor}.json",info)
            for seed in (1,2,3):
                name=f"split_{case}_m{factor}_s{seed}"
                jobs.append(dict(job_id=name,module="post_fg_split",kind="split_eval",case=case,
                                 config_id=state["config_id"],hardware_profile=state["spec"]["hardware_profile"],
                                 shape=state["spec"]["shape"],split_mode="growth" if case.startswith("G") else "fixed",
                                 phase="prefill",op="proj" if case=="FACE_proj" else "ffn",method=f"Split-m{factor}",
                                 factor=factor,axis=axis,seed=seed,graph=filename.relative_to(ROOT).as_posix(),
                                 graph_sha256=fingerprint,protocol_version=VERSION))
    dest=ROOT/"e2e/post_fg"
    table(dest/"tables/split_axis_choices.csv",choices)
    write(dest/"evidence/split_graph_checks.json",dict(status="passed",numerical=numerical,identity=identities,
                                                     graphs=graphs,new_backend_evaluations=0))
    write(dest/"manifest/split.json",dict(protocol_version=VERSION,jobs=jobs))
    print(json.dumps(dict(status="passed",prepared_graphs=len(graphs),jobs=len(jobs))),flush=True)


def report(run_out, out, dest):
    manifest=read(dest/"manifest/split.json")
    rows=[read(out/"results"/(s["job_id"]+".json")) for s in manifest["jobs"]]
    assert len(rows)==27 and all(r["status"]=="ok" for r in rows)
    samples=[]
    summary=[]
    for case,cid,op in CASES:
        l1=static.read_pickle(run_out/"raw"/f"{cid}__l1_{op}_A3h_MF6.pkl.gz")
        stop=read(run_out/"results"/f"{cid}__stop_{op}_MF6_300.json")
        rep=static.read_pickle(run_out/"raw"/(stop["source_job"]+".pkl.gz"))
        old={"R0":l1["R0"]["backend"],"EIR-representative":rep["record"]["backend"]}
        for label,seeds in old.items():
            for seed in (1,2,3):
                value=seeds[seed]
                samples.append(dict(case=case,method=label,seed=seed,backend_ns=value[0],T_hat_ns=None,
                                    source="saved F/G evaluations; no new run"))
        for factor in (1,2,4):
            selected=[r for r in rows if r["case"]==case and r["factor"]==factor]
            graph=static.read_pickle(ROOT/selected[0]["graph"])
            info=graph["info"]
            for row in selected:
                samples.append(dict(case=case,method=f"Local-m{factor}" if factor==1 else f"Split-m{factor}",
                                    seed=row["seed"],backend_ns=row["backend_triple_ns"][0],T_hat_ns=row["T_hat_ns"],
                                    backend_seconds=row["backend_seconds"],cpcm_seconds=row["cpcm_seconds"],
                                    source=row["job_id"]))
            values=[r["backend_triple_ns"][0] for r in selected]
            r0=statistics.mean(old["R0"][seed][0] for seed in (1,2,3))
            eir=statistics.mean(old["EIR-representative"][seed][0] for seed in (1,2,3))
            newmean=statistics.mean(values)
            bound=info["rigid_tensor_load_bound"]["lower_bound_ns"]
            summary.append(dict(case=case,factor=factor,axis=selected[0]["axis"],backend_mean_ns=newmean,
                backend_min_ns=min(values),backend_max_ns=max(values),R0_mean_seeds123_ns=r0,EIR_rep_mean_seeds123_ns=eir,
                relative_to_R0=newmean/r0-1,relative_to_EIR_representative=newmean/eir-1,
                rigid_tensor_lower_bound_ns=bound,all_backend_below_rigid_bound=max(values)<bound,
                tensor_peak_ns=info["metrics"]["compute_peak_ns"]["tensorcore"],
                new_tensor_lower_bound_ns=info["tensor_load_bound"]["lower_bound_ns"],
                effective_flops=info["metrics"]["tensor_effective_flops"],rounded_flops=info["metrics"]["tensor_rounded_flops"],
                route_link_bytes=info["metrics"]["communication"]["route_link_bytes"],
                branch_path_link_bytes=info["metrics"]["communication"]["branch_path_link_bytes"],
                ddr_read_bytes=info["metrics"]["communication"]["ddr_read_bytes"],
                ddr_write_bytes=info["metrics"]["communication"]["ddr_write_bytes"],
                extra_weight_input_bytes=info["affected_weight_network_bytes_after"]-info["affected_weight_network_bytes_before"],
                affected_activation_input_bytes=info["affected_activation_network_bytes_after"],
                healthy_computation_moved=info["healthy_computation_moved"],
                decision_seconds=info["axis_selection_seconds"],
                backend_wall_seconds=statistics.mean(r["backend_seconds"] for r in selected),
                peak_rss_gib=max(r["peak_rss_gb"] for r in selected)))
    table(dest/"tables/split_runs.csv",samples)
    table(dest/"tables/split_summary.csv",summary)
    render_report(dest,summary)


def render_report(dest,summary=None):
    text=["# EIR-Split：故障PE遗留计算块的局部拆分", "", "2026-11-13，Codex。新增实现为 `post_fg_split.py`，协议 `post_fg_split_v1`。", "",
          "## 方法", "", "从A3h健康完整图识别MF6失效PE原有的全部Tensor/Vector任务。健康计算落点保持不变，Vector整体迁移，Tensor只沿输出N或batch-M轴拆成2/4片。原有FFN第二层的K部分和仍进入原来的16项向量归约，不新增K切分，也不省略归约。", "",
          "输入片段来自行为/张量元数据，实际源位置取保存图中的活通信；这避免使用SA后未同步的DDR位置字段。N切分保留每个子块的完整A、拆B与bias；batch-M切分拆A、保留完整B与bias。消费者按需要的真实区域接收，每个区域覆盖检查通过；计算依赖在落点分离时变成真实数据通信。输出仍按照原布局收集，Vector等待全部所需子片。", "",
          "新落点先最小化加任务后的最大计算负载，同父优先不同核，再按预估输入输出链路字节和稳定核ID取舍。改动通信使用确定性最短服务时间路径，保留全部不受影响的原路径。MF6仅删除Tensor/Vector设备，原路由器链路全部保留。权重继续从原活读取对应的DDR源恢复，不假设PE故障同时清空SRAM。", "",
          "每案例分别构造N、M的合法输出划分，先比较实际grain取整FLOP，再比较完整图独占有向链字节，最后按N/M稳定顺序选轴。选择发生在后端运行前，代价记录同时计入两个候选的构造时间。m1使用相同放置与路由规则，是不拆分对照。", "",
          "## 正确性与静态结果", "", "小矩阵的M/N重组、bias与已有K部分和求和逐元素相等；三例m1指定原合法落点时，原事件顺序、字节、依赖、落点与路径完全一致。真实图核对了所有新消费者输入区域覆盖、有效FLOP守恒、健康计算移动数0、失败PE无计算，以及DAG无环。", "",
          "| 案例 | 新图m1/m2/m4选轴 | Tensor峰值m1→m2→m4，百万ns |", "|---|---|---|",
          "| FACE投影 | — / M / M | 9.437184 → 7.077888 → 5.898240 |",
          "| 2×3 FFN | — / M / M | 37.748736 → 31.444992 → 28.293120 |",
          "| 3×3 FFN | — / N / N | 25.178112 → 20.975616 → 18.874368 |", "",
          "三例两轴均合法，m2/m4的实际Tensor取整工作与父块之和相同。M选择指batch维16按2/4分片，不要求序列片段341或227可整除。Vector不按GEMM规则拆分。", "",
          "统计同时保存消息逻辑字节、实际独占有向链字节（同一多播消息的共享前缀计一次）与按分支路径逐条求和字节。轴选择使用前者的链路占用表示；后者用于与采用分支和的历史表核对。DDR写入按实际接收端口计数。输入驻留量仅作为各接收核不同输入片段的总量上界，未把它写成实际峰值SRAM。", ""]
    if summary is None:
        text += ["## 后端评估", "", "已准备三个案例×m1/m2/m4×种子1/2/3，共27次新后端评估。真实后端安排在服务器，结果待本轮运行。计算峰值降低目前不是总延迟收益。", ""]
    else:
        text += ["## 后端评估", "", "27次新后端评估全部完成。下表旧R0/EIR使用F/G保存的后端种子1–3；EIR取原停止档的后端中位代表，和F/G正文五修复种子平均列分别定义。独立后端存在已知跨进程平局波动，逐种子数值和范围完整保留。", "",
                 "| 案例 | 方案 | 均值ns [最小,最大] | 相对R0 | 相对原EIR代表 | 所有种子低于刚性Tensor下界 |", "|---|---|---|---:|---:|---|"]
        for r in summary:
            text.append(f"| {r['case']} | m{r['factor']} | {r['backend_mean_ns']:,.1f} [{r['backend_min_ns']:,}, {r['backend_max_ns']:,}] | {r['relative_to_R0']:+.2%} | {r['relative_to_EIR_representative']:+.2%} | {'是' if r['all_backend_below_rigid_bound'] else '否'} |")
        text += ["", "总延迟低于有效刚性计算下界时，可说明局部粒度调整达到固定块放置无法达到的时间；仅计算峰值下降的情况仍按实际后端收益报告。构图/选轴决策时间与本轮三个种子的评估成本分开列，后端墙钟来自这次服务器环境。", ""]
    text += ["## 保存与复算", "", "- [选轴比较](tables/split_axis_choices.csv)",
             "- [数值、identity与完整图核对](evidence/split_graph_checks.json)",
             "- [27项运行清单](manifest/split.json)",
             "- 原始图、逐片映射、输入输出区域与服务时间：`out/split_prepare/graphs`、`out/split_prepare/metadata`。", "",
             "```powershell", "python -B e2e/post_fg_split.py prepare", "python -B e2e/post_fg_split.py report --out e2e/post_fg/out/main", "```", ""]
    (dest/"EIR_SPLIT.md").write_text("\n".join(text),encoding="utf-8")


def child(spec,out):
    setup()
    started=time.perf_counter()
    data=static.read_pickle(ROOT/spec["graph"])
    events=pickle.loads(data["events_blob"])
    fingerprint=graph_digest(events)
    assert fingerprint==spec["graph_sha256"]
    hp=platform(data["source_spec"],data["dead"])
    import cpcm
    import numpy as np
    from event_driver import event_driver
    t=time.perf_counter();schedule=cpcm.simulate(cpcm.compile_bb(events,hp),"bb");cpcm_seconds=time.perf_counter()-t
    random.seed(spec["seed"]);np.random.seed(spec["seed"])
    evaluated=copy.deepcopy(events)
    t=time.perf_counter();triple=tuple(int(x) for x in event_driver(evaluated,copy.deepcopy(hp)));backend_seconds=time.perf_counter()-t
    trace=[dict(tag=list(tag),name=e.event_name,kind=e.event_type,start_ns=int(e.start_time),end_ns=int(e.end_time))
           for tag,e in evaluated.items()]
    write(out/"traces"/(spec["job_id"]+".json"),trace)
    import e2e_pilot_peak
    result=dict(spec,job_id=spec["job_id"],status="ok",T_hat_ns=int(schedule.T),backend_triple_ns=triple,
                cpcm_seconds=cpcm_seconds,backend_seconds=backend_seconds,wall_seconds=time.perf_counter()-started,
                peak_rss_gb=e2e_pilot_peak.peak()/2**30,graph_metrics=data["info"]["metrics"],
                build_seconds=data["info"]["build_seconds"],healthy_computation_moved=0,
                decision_seconds=data["info"]["axis_selection_seconds"],
                source_export=data["source_export"],source_candidate=data["source_candidate"])
    write(out/"results"/(spec["job_id"]+".json"),result)
    print(json.dumps(dict(job=spec["job_id"],T_hat_ns=int(schedule.T),backend=triple,wall_seconds=result["wall_seconds"])),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("prepare","child","report"))
    parser.add_argument("--run-out",type=Path,default=ROOT/"e2e/fg/out/main")
    parser.add_argument("--out",type=Path,default=ROOT/"e2e/post_fg/out/split_prepare")
    parser.add_argument("--cases",nargs="+",default=[row[0] for row in CASES])
    parser.add_argument("--spec",type=Path)
    args=parser.parse_args()
    args.out=args.out.resolve()
    if args.command=="prepare":
        prepare(args.run_out.resolve(),args.out,args.cases)
        render_report(ROOT/"e2e/post_fg")
    elif args.command=="report":report(args.run_out.resolve(),args.out,ROOT/"e2e/post_fg")
    else:child(read(args.spec),args.out)


if __name__=="__main__":main()
