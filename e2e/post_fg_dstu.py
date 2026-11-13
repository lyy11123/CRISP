"""Post-F/G DSTU: saved graphs, six strategies, paired solos and resource traces."""
import argparse
import collections
import copy
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import statistics
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

from fg_fault import ROOT, plain, write_json, result_path, platform_signature
from post_fg_dstu_policy import MessagePolicy

ARMS = ("BB0", "H0", "A256", "A256-M", "S256", "S256-M")
CHUNK = 256 * 1024
VERSION = "post_fg_dstu_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source(spec, rel):
    return Path(spec["source_root"]) / rel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_result(spec, name):
    return source(spec, f"e2e/fg/out/main/results/{name}.json")


def graph_inputs(spec, out):
    import f_config
    import fg_joint
    g = read_json(source_result(spec, spec["graph_job"]))
    assert sha(source_result(spec, spec["graph_job"])) == spec["graph_sha256"]
    setup = dict(spec, phase="prefill", op="mha", method="A3h", g=0, x=123)
    f_config.prepare(setup, out)
    d = fg_joint.d_modules()
    hp = f_config.build_platform(setup, out, faults=g["fault"], shape_override="2x4")
    assert platform_signature(hp) == g["platform_sha256"]
    streams = {}
    for stream in (0, 1):
        filename = source(spec, g["files"][f"orig_{stream}"])
        assert sha(filename) == spec["graph_files_sha256"][str(stream)]
        with gzip.open(filename, "rb") as handle:
            streams[stream] = pickle.load(handle)
    return d, hp, g, streams


def resource_set(event):
    return {link for branch in event.path_list for link in branch}


def transform(streams, hp, arm, rank):
    """Start from unchunked saved graphs; retain the joint-derived solo mask."""
    import dstu2
    rank = {int(k): v for k, v in rank.items()}
    events = {tag: event for stream in streams.values() for tag, event in stream.items()}
    resources = {s: {link for e in ev.values() if e.event_type == "communication" for link in resource_set(e)}
                 for s, ev in streams.items()}
    selected = set()
    for stream, ev in streams.items():
        higher = set().union(*(resources[s] for s in resources if rank[s] < rank[stream]))
        for tag, event in ev.items():
            if event.event_type == "communication" and event.comm_bytes > CHUNK and resource_set(event) & higher:
                selected.add(tag)
    if arm.startswith("A256") or arm == "TieFIFO":
        cut = {t for t, e in events.items() if e.event_type == "communication" and e.comm_bytes > CHUNK}
    elif arm.startswith("S256"):
        cut = selected
    else:
        cut = set()
    metadata = {}
    original = {}
    total_links = collections.Counter()
    for tag, event in events.items():
        if event.event_type != "communication":
            continue
        count = (int(event.comm_bytes) + CHUNK - 1) // CHUNK if tag in cut else 1
        for index in range(count):
            child = tuple(tag[:-1]) + (tag[-1] + index * 1_000_000,)
            metadata[child] = dict(parent=tag, index=index, count=count)
        links = resource_set(event)
        stream = tag[1] - 10
        shared = links & set().union(*(resources[s] for s in resources if s != stream))
        original[tag] = dict(bytes=int(event.comm_bytes), stream=stream, consumers=sorted(event.issue_set),
                             links=len(links), shared_links=len(shared), chunks=count)
        total_links["shared_link_bytes"] += int(event.comm_bytes) * len(shared)
        total_links["private_link_bytes"] += int(event.comm_bytes) * len(links - shared)
    plan = dict(size=max((int(e.comm_bytes) for e in events.values() if e.event_type == "communication"), default=0),
                only_d2d=False, override={dstu2.gid(t): CHUNK for t in cut})
    assert len({dstu2.gid(t) for t, e in events.items() if e.event_type == "communication"}) == len(original)
    chunked, stats = dstu2.chunk_by(events, hp, plan)
    assert stats["bytes_conserved"] and stats["tree_conserved"]
    assert set(metadata) == {t for t, e in chunked.items() if e.event_type == "communication"}
    stats.update(selected_messages=len(cut), shared_candidate_messages=len(selected),
                 selected_payload_bytes=sum(original[t]["bytes"] for t in cut),
                 shared_candidate_payload_bytes=sum(original[t]["bytes"] for t in selected),
                 shared_resource_count=len(resources[0] & resources[1]), **total_links)
    return chunked, metadata, original, stats


class Audit:
    def __init__(self): self.n = self.bad = 0
    def append(self, record):
        self.n += 1
        self.bad += record[2] != record[3]


def graph_digest(events):
    """Canonical complete initial service graph, preserving event iteration order."""
    rows=[]
    for tag,e in events.items():
        row=dict(tag=tag,type=e.event_type,name=e.event_name,
                 dependencies=sorted(e.dependency_set),successors=sorted(e.issue_set),
                 start=e.start_time,end=e.end_time)
        if e.event_type=="computation":
            row.update(device=e.comp_device,location=e.comp_location,kind=e.comp_type,shape=e.comp_datashape)
        else:
            target=sorted(e.target_location) if isinstance(e.target_location,set) else e.target_location
            row.update(source=e.source_location,target=target,target_is_set=isinstance(e.target_location,set),
                       bytes=e.comm_bytes,paths=e.path_list)
        rows.append(row)
    return hashlib.sha256(json.dumps(plain(rows),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def evaluation_identity(spec, events, hp, release):
    mode="MsgCont" if spec["arm"].endswith("-M") else "TieFIFO" if spec["arm"]=="TieFIFO" else "C4"
    return plain(dict(graph_sha256=graph_digest(events),platform_sha256=platform_signature(hp),
                      backend="BB0" if spec["arm"]=="BB0" else "Hop",policy=mode,
                      rank=spec["rank"],seed=spec["seed"],release=release))


def save_transform(spec, out, metadata, stats):
    path=Path(out)/"transforms"/(spec["job_id"]+".json.gz")
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(str(path)+".part","wt",encoding="utf-8") as handle:
        json.dump(plain(dict(input_sha256=spec["graph_sha256"],rank=spec["rank"],arm=spec["arm"],stats=stats,
                             messages=[dict(tag=t,**m) for t,m in metadata.items()])),handle,separators=(",",":"))
    Path(str(path)+".part").replace(path)
    return path


def simulate(spec, out):
    import numpy as np
    import fast_hop
    d, hp, graph, streams = graph_inputs(spec, out)
    rank = {int(k): v for k, v in spec["rank"].items()}
    t_transform = time.perf_counter()
    events, metadata, originals, stats = transform(streams, hp, spec["arm"], rank)
    transform_seconds = time.perf_counter() - t_transform
    transform_path=save_transform(spec,out,metadata,stats)
    if spec["run_type"] == "solo":
        events = {t: e for t, e in events.items() if t[1] - 10 == spec["stream"]}
        release = {spec["stream"]: 0}
    else:
        release = {0: 0, 1: spec["tau"]}
    evaluated_identity=evaluation_identity(spec,events,hp,release)
    deps = {t: set(e.dependency_set) for t, e in events.items()}
    trace_path = Path(out) / "traces" / (spec["job_id"] + ".issues.jsonl.gz")
    trace_handle = None
    if spec.get("trace"):
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = gzip.open(str(trace_path) + ".part", "wt", encoding="utf-8")
    writer = None if trace_handle is None else lambda row: trace_handle.write(json.dumps(plain(row), separators=(",", ":")) + "\n")
    mode = "MsgCont" if spec["arm"].endswith("-M") else "TieFIFO" if spec["arm"] == "TieFIFO" else "C4"
    policy = MessagePolicy(rank, metadata, mode=mode, trace=writer)
    random.seed(spec["seed"]); np.random.seed(spec["seed"])
    released = {}; audit = Audit()
    started = time.perf_counter()
    if spec["arm"] == "BB0":
        from e4_backend import event_driver_e4
        triple = event_driver_e4(events, hp, release=release, released=released)
    else:
        triple = fast_hop.event_driver_fast(events, hp, release=release, policy=policy, released=released, audit=audit)
    sim_seconds = time.perf_counter() - started
    if trace_handle is not None:
        if spec["arm"] == "BB0":
            for link, resource in hp.links_dict.items():
                for begin, end, tag in resource.work_record:
                    writer(dict(t=begin,end=end,resource=link,stream=tag[1]-10,parent=metadata[tag]["parent"],
                                chunk=metadata[tag]["index"],backend="BB0",original_key=None,
                                candidate_information="BB backend work record; C4 arbitration is not applied"))
        trace_handle.close()
        Path(str(trace_path) + ".part").replace(trace_path)
    assert audit.bad == 0
    finish = {s: max(e.end_time for t, e in events.items() if t[1] - 10 == s) for s in release}
    starts = {s: min(e.start_time for t, e in events.items() if t[1] - 10 == s) for s in release}
    kv_tags = {tuple(t) for t in graph["kv_tags"]}
    kv_events = [e for t, e in events.items() if t in metadata and metadata[t]["parent"] in kv_tags]
    kv = [min(e.start_time for e in kv_events), max(e.end_time for e in kv_events)] if kv_events else None
    trace_summary = None
    if spec.get("trace"):
        first = {}; parent_service = collections.Counter(); occupancy = collections.Counter()
        for link, resource in hp.links_dict.items():
            for begin, end, tag in resource.work_record:
                parent = metadata[tag]["parent"]
                first[parent] = min(first.get(parent, begin), begin)
                parent_service[parent] += end - begin
                occupancy[resource.link_type] += end - begin
        parent_ends = collections.defaultdict(list)
        for tag, event in events.items():
            if tag in metadata:
                parent_ends[metadata[tag]["parent"]].append(event.end_time)
        parents = [dict(parent=t, first_start=first[t], last_finish=max(parent_ends[t]),
                        link_service=parent_service[t], **info) for t, info in originals.items() if t in parent_ends]
        consumers = []
        for tag, event in events.items():
            if event.event_type == "computation":
                ready = max((events[p].end_time for p in deps[tag]), default=release[tag[1]-10])
                consumers.append(dict(tag=tag, ready=ready, start=event.start_time, end=event.end_time,
                                      device=event.comp_device, location=event.comp_location))
        trace_summary = Path(out) / "traces" / (spec["job_id"] + ".messages.json.gz")
        tmp = Path(str(trace_summary) + ".part")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(plain(dict(parents=parents, consumers=consumers, occupancy=occupancy)), handle)
        tmp.replace(trace_summary)
    result = dict(completion=finish, first_start=starts, kv_window=kv, release=release,
                  simulated_total=int(triple[0]), unhidden_compute=int(triple[1]), unhidden_communication=int(triple[2]),
                  simulation_seconds=sim_seconds, transform_seconds=transform_seconds, chunk_stats=stats,
                  policy_counters=dict(policy.stats) if spec.get("trace") or mode=="MsgCont" else {"issues":policy.stats["issues"]},
                  candidate_counters_available=bool(spec.get("trace") or mode=="MsgCont"), audit=dict(n=audit.n, bad=audit.bad),
                  input_graph_sha256=spec["graph_sha256"], rank=rank,
                  evaluation_identity=evaluated_identity,backend_calls=1,
                  message_transform=str(transform_path),
                  issue_trace=str(trace_path) if spec.get("trace") else None,
                  message_trace=str(trace_summary) if trace_summary else None,
                  timing_includes_resource_trace=bool(spec.get("trace")))
    if spec["run_type"] == "joint":
        cp, cd = finish[0], finish[1] - spec["tau"]
        rp, rd = spec["reference"]["T_P"], spec["reference"]["T_D"]
        result["metrics"] = dict(P_completion=cp, D_latency=cd,
                                 Lhat_P=(cp - 1.5*rp)/rp, Lhat_D=(cd - 1.5*rd)/rd)
        result["metrics"]["Lhat_max"] = max(result["metrics"]["Lhat_P"], result["metrics"]["Lhat_D"])
    return result


def batch_alias(spec, out):
    """Reuse one matched D solo, retaining S256's joint-derived mask metadata."""
    assert spec["arm"]=="S256" and spec["run_type"]=="solo" and spec["stream"]==1
    filename=result_path(out,spec["alias_job"])
    other=read_json(filename)
    assert other["status"]=="ok" and other["spec"]["arm"]=="H0"
    _,hp,graph,streams=graph_inputs(spec,out)
    started=time.perf_counter()
    events,metadata,original,stats=transform(streams,hp,"S256",spec["rank"])
    seconds=time.perf_counter()-started
    transform_path=save_transform(spec,out,metadata,stats)
    actual={t:e for t,e in events.items() if t[1]-10==spec["stream"]}
    release={spec["stream"]:0}
    identity=evaluation_identity(spec,actual,hp,release)
    assert identity==other["evaluation_identity"],(identity,other["evaluation_identity"])
    keys=("completion","first_start","kv_window","release","simulated_total","unhidden_compute",
          "unhidden_communication","policy_counters","candidate_counters_available","audit")
    result={key:other[key] for key in keys}
    result.update(batch_alias_from=spec["alias_job"],batch_alias_result_sha256=sha(filename),
                  evaluation_identity=identity,alias_identity_checked=True,backend_calls=0,
                  simulation_seconds=0,source_simulation_seconds=other["simulation_seconds"],
                  transform_seconds=seconds,chunk_stats=stats,message_transform=str(transform_path),
                  input_graph_sha256=spec["graph_sha256"],rank=spec["rank"],
                  issue_trace=None,message_trace=None,timing_includes_resource_trace=False)
    return result


def reuse(spec):
    filename = source_result(spec, spec["reuse_job"])
    assert sha(filename) == spec["reuse_sha256"]
    old = read_json(filename)
    for key in ("hardware_profile", "condition", "seed"):
        assert old["spec"][key] == spec[key], key
    assert old["graph_job"] == spec["graph_job"]
    sim = old["simulation"]
    r = dict(completion={k: v["C"] for k, v in sim["streams"].items()},
             first_start={k: v["first_start"] for k, v in sim["streams"].items()},
             kv_window=sim.get("kv_window"), release=sim["release"], simulation_seconds=0,
             original_simulation_seconds=sim["wall"], simulated_total=sim["out"][0],
             unhidden_compute=sim["out"][1], unhidden_communication=sim["out"][2],
             reused_from=str(filename), original_job=spec["reuse_job"], original_sha256=spec["reuse_sha256"],
             rank=spec["rank"], input_graph_sha256=spec["graph_sha256"],backend_calls=0)
    if spec["run_type"] == "joint":
        assert old["tau"] == spec["tau"]
        r["metrics"] = {k: old["metrics"][k] for k in ("P_completion", "D_latency", "Lhat_P", "Lhat_D", "Lhat_max")}
    return r


def graph_check(spec, out):
    """Real saved inputs and graph transformation only; no backend evaluation."""
    _, hp, graph, streams = graph_inputs(spec, out)
    def signature(events):
        rows=[]
        for tag, event in events.items():
            row=[tag,event.event_type,sorted(event.dependency_set),sorted(event.issue_set)]
            if event.event_type=="communication":row.extend([int(event.comm_bytes),event.path_list])
            else:row.extend([event.comp_device,event.comp_location,event.comp_type,event.comp_datashape])
            rows.append(row)
        return plain(rows)
    full, meta, originals, stats = transform(copy.deepcopy(streams), hp, "A256", spec["rank"])
    saved={}
    for stream in (0,1):
        with gzip.open(source(spec,graph["files"][f"hop_{stream}"]),"rb") as handle:saved.update(pickle.load(handle))
    assert signature(full)==signature(saved)
    shared, metadata, original, selected = transform(copy.deepcopy(streams), hp, "S256", spec["rank"])
    assert selected["shared_candidate_messages"]==128 and selected["selected_payload_bytes"]==100663296
    assert selected["blocks_added"]==256
    assert all(metadata[t]["count"]==1 for t in metadata if t[1]==11)
    for tag,event in shared.items():
        assert all(tag in shared[d].issue_set for d in event.dependency_set)
        assert all(tag in shared[c].dependency_set for c in event.issue_set)
    by_parent=collections.defaultdict(list)
    for tag,meta in metadata.items():by_parent[meta["parent"]].append(tag)
    for parent, info in original.items():
        children=by_parent[parent]
        assert sum(shared[t].comm_bytes for t in children)==info["bytes"]
        assert all(shared[t].path_list==shared[children[0]].path_list for t in children)
    result=dict(status="passed",graph_job=spec["graph_job"],source_sha256=spec["graph_sha256"],
                all256_identical_to_saved=True,shared=selected,
                messages=len(metadata),events=len(shared),new_backend_calls=0)
    write_json(Path(out)/"checks"/(spec["hardware_profile"]+"_"+spec["condition"]+"_graph.json"),result)
    return result


def child(spec, out):
    out = Path(out); started = time.perf_counter()
    result = reuse(spec) if spec.get("reuse_job") else batch_alias(spec,out) if spec.get("alias_job") else simulate(spec, out)
    if spec.get("compare_old_job"):
        assert sha(source_result(spec,spec["compare_old_job"]))==spec["compare_old_sha256"]
        old=read_json(source_result(spec,spec["compare_old_job"]))
        fields=("P_completion","D_latency","Lhat_P","Lhat_D","Lhat_max")
        result["old_result_comparison"]={k:dict(old=old["metrics"][k],new=result["metrics"][k],
                                               delta=result["metrics"][k]-old["metrics"][k],
                                               equal=old["metrics"][k]==result["metrics"][k]) for k in fields}
        result["trace_comparison_rule"]=("BB0 historical single-run difference retained; mechanism interpretation uses this trace run"
                                          if spec["arm"]=="BB0" else "same Hop graph, policy and seed must match")
        if spec["arm"]!="BB0":assert all(r["equal"] for r in result["old_result_comparison"].values())
    import e2e_pilot_peak
    result.update(job_id=spec["job_id"], kind=spec["kind"], status="ok", spec=spec,
                  wall_seconds=time.perf_counter()-started, peak_rss_gb=e2e_pilot_peak.peak()/2**30)
    write_json(result_path(out, spec["job_id"]), result)
    print(f"[post_fg_dstu] {spec['job_id']} {result['wall_seconds']:.2f}s", flush=True)


def manifest(source_root, output, condition="healthy", selected=None):
    root = Path(source_root).resolve(); jobs = []
    arms=ARMS if selected is None else tuple(dict.fromkeys(("BB0","A256",selected)))
    for hw in ("wsc", "face"):
        cid = f"F_{hw}_2x2_fixed_d2d1"
        graph_job = cid + f"__graph_{condition}_A3hEIR"
        graph_path = root / "e2e/fg/out/main/results" / (graph_job + ".json")
        graph = read_json(graph_path)
        reference = read_json(root / "e2e/fg/out/main/results" / (cid + "__joint_reference.json"))
        base = dict(module="post_fg_dstu", version=VERSION, source_root=str(root), config_id=cid,
                    hardware_profile=hw, shape="2x2", split_mode="fixed", d2d_scale=1,
                    graph_job=graph_job, graph_sha256=sha(graph_path), condition=condition,
                    graph_files_sha256={str(s): sha(root / graph["files"][f"orig_{s}"]) for s in (0,1)},
                    reference={k: reference[k] for k in ("T_P", "T_D", "KV_start", "KV_end")})
        ranks = {}
        for phase in (.1,.5,.9):
            tau = int(round(reference["KV_start"]+phase*(reference["KV_end"]-reference["KV_start"])))
            deadlines = {s: 1.5*graph["that"][str(s)] + (tau if s else 0) for s in (0,1)}
            order = sorted(deadlines, key=lambda s: (deadlines[s],s))
            ranks[phase] = {s: order.index(s) for s in order}
            for arm in arms:
                for seed in (1,2,3):
                    spec = dict(base, job_id=f"dstu_{hw}_{condition}_{arm}_g{phase}_s{seed}", kind="dstu_joint",
                                run_type="joint", arm=arm, phase=phase, seed=seed, tau=tau, rank=ranks[phase],
                                frozen_C4_deadlines=deadlines, rank_source=f"{graph_job}:All256_CPCM",
                                trace=False)
                    jobs.append(spec)
        assert ranks[.1] == ranks[.5] == ranks[.9]
        for arm in arms:
            for stream in (0,1):
                for seed in (1,2,3):
                    jobs.append(dict(base, job_id=f"dstu_{hw}_{condition}_{arm}_solo{stream}_s{seed}", kind="dstu_solo",
                                     run_type="solo", arm=arm, stream=stream, seed=seed, rank=ranks[.1],
                                     mask_rank_phases=[.1,.5,.9], rank_source=f"{graph_job}:All256_CPCM",trace=False))
    for spec in jobs:
        if spec["arm"] in ("BB0", "A256"):
            cid, cond = spec["config_id"], spec["condition"]
            if spec["run_type"] == "joint":
                arm = "-DSTU" if spec["arm"] == "BB0" else "CRISP"
                name = f"{cid}__joint_{cond}_{arm}_g{spec['phase']}_s{spec['seed']}"
            else:
                backend = "orig" if spec["arm"] == "BB0" else "hop"
                name = f"{cid}__solo_{cond}_A3hEIR_{backend}_{spec['stream']}_{spec['seed']}"
            spec.update(reuse_job=name,reuse_sha256=sha(source_result(spec,name)))
        if "H0" in arms and spec["arm"]=="S256" and spec["run_type"]=="solo" and spec["stream"]==1:
            name=f"dstu_{spec['hardware_profile']}_{condition}_H0_solo1_s{spec['seed']}"
            spec.update(alias_job=name,depends_on=[name],alias_rule="identical filtered D graph, C4, platform, release and seed")
        if condition=="healthy" and not spec.get("reuse_job") and spec["run_type"]=="joint" and spec["seed"]==1:
            spec["trace"]=(spec["hardware_profile"]=="wsc" and spec["phase"] in (.1,.9)) or (spec["hardware_profile"]=="face" and spec["phase"]==.9)
    payload = dict(version=VERSION, jobs=jobs, logical_units=len(jobs),
                   old_result_reuses=sum("reuse_job" in s for s in jobs),
                   new_batch_aliases=sum("alias_job" in s for s in jobs),
                   new_backend_calls=sum(not s.get("reuse_job") and not s.get("alias_job") for s in jobs))
    write_json(output,payload)
    smoke = [s for s in jobs if s["hardware_profile"]=="wsc" and s["seed"]==1
             and (s["run_type"]=="solo" or s["phase"] in (.1,.9))]
    if condition=="healthy":
        write_json(Path(output).with_name("dstu_smoke.json"),dict(version=VERSION,jobs=smoke,logical_units=len(smoke),
                   old_result_reuses=sum("reuse_job" in s for s in smoke),
                   new_batch_aliases=sum("alias_job" in s for s in smoke),
                   new_backend_calls=sum(not s.get("reuse_job") and not s.get("alias_job") for s in smoke)))
    if condition=="healthy":
        trace_jobs=[]
        for spec in jobs:
            point=spec["run_type"]=="joint" and spec["seed"]==1 and ((spec["hardware_profile"]=="wsc" and spec["phase"] in (.1,.9)) or (spec["hardware_profile"]=="face" and spec["phase"]==.9))
            if point and spec.get("reuse_job"):
                trace=dict(spec,job_id=spec["job_id"]+"_trace",kind="dstu_trace",trace=True)
                trace["compare_old_job"]=trace.pop("reuse_job")
                trace["compare_old_sha256"]=trace.pop("reuse_sha256")
                trace_jobs.append(trace)
        write_json(Path(output).with_name("dstu_trace_baselines.json"),dict(version=VERSION,jobs=trace_jobs,new_backend_calls=len(trace_jobs)))
    return payload


def report(manifest_path, out, destination):
    specs = read_json(manifest_path)["jobs"]; rows=[]; solos={}
    for spec in specs:
        r=read_json(result_path(out,spec["job_id"]))
        if spec["run_type"]=="solo":
            solos[(spec["hardware_profile"],spec["condition"],spec["arm"],spec["stream"],spec["seed"])]=r["completion"][str(spec["stream"])]
    for spec in specs:
        r=read_json(result_path(out,spec["job_id"]))
        row={k:spec.get(k) for k in ("job_id","hardware_profile","condition","arm","run_type","phase","stream","seed")}
        row.update(wall_seconds=r["wall_seconds"],peak_rss_gb=r["peak_rss_gb"],reused=bool(spec.get("reuse_job") or spec.get("alias_job")),
                   historical_reuse=bool(spec.get("reuse_job")),batch_alias=spec.get("alias_job"),
                   simulation_seconds=r["simulation_seconds"],kv_window=r.get("kv_window"),
                   issue_trace=r.get("issue_trace"),message_trace=r.get("message_trace"),
                   audit_bad=r.get("audit",{}).get("bad"),chunk_stats=r.get("chunk_stats"),
                   policy_counters=r.get("policy_counters"))
        row["timing_includes_resource_trace"]=r.get("timing_includes_resource_trace",False)
        if spec["run_type"]=="solo": row["completion_ns"]=r["completion"][str(spec["stream"])]
        else:
            row.update(r["metrics"])
            for stream,label in ((0,"P"),(1,"D")):
                key=(spec["hardware_profile"],spec["condition"],spec["arm"],stream,spec["seed"])
                solo=solos[key];bb=solos[(key[0],key[1],"BB0",stream,key[-1])]
                joint=row["P_completion" if stream==0 else "D_latency"]
                row.update({f"{label}_solo":solo,f"{label}_self_cost_O":solo-bb,f"{label}_interference_J":joint-solo})
            row["deadline_violation"]=row["Lhat_max"]>0
        rows.append(row)
    joints={(r["hardware_profile"],r["condition"],r["arm"],r["phase"],r["seed"]):r for r in rows if r["run_type"]=="joint"}
    identities=[]
    for key,row in joints.items():
        bb=joints[(key[0],key[1],"BB0",key[3],key[4])]
        current=joints[(key[0],key[1],"A256",key[3],key[4])]
        for stream,metric in (("P","P_completion"),("D","D_latency")):
            lhs=row[metric]-bb[metric]
            rhs=row[f"{stream}_self_cost_O"]+row[f"{stream}_interference_J"]-bb[f"{stream}_interference_J"]
            assert abs(lhs-rhs)<1e-7
            identities.append(dict(job=row["job_id"],stream=stream,lhs=lhs,rhs=rhs))
            row[f"{stream}_change_vs_BB0"]=row[metric]/bb[metric]-1
            row[f"{stream}_change_vs_A256"]=row[metric]/current[metric]-1
        row["Lhat_change_vs_A256"]=row["Lhat_max"]-current["Lhat_max"]
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/"dstu_runs.json",rows)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (dest/"dstu_runs.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        w=csv.DictWriter(handle,fieldnames=fields);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
    write_json(dest/"dstu_paired_checks.json",dict(status="passed",identities=len(identities),
                                                 max_error=max(abs(x["lhs"]-x["rhs"]) for x in identities)))
    summary=[]
    for hw in sorted({r["hardware_profile"] for r in rows}):
        for arm in ARMS:
            chosen=[r for r in rows if r["run_type"]=="joint" and r["hardware_profile"]==hw and r["arm"]==arm]
            if not chosen:continue
            record=dict(hardware=hw,arm=arm,n=len(chosen),deadline_violations=sum(r["deadline_violation"] for r in chosen))
            for name in ("P_change_vs_A256","D_change_vs_A256","Lhat_change_vs_A256","P_self_cost_O","D_self_cost_O"):
                values=[r[name] for r in chosen]
                record[name]=dict(median=statistics.median(values),minimum=min(values),maximum=max(values),
                                  lower=sum(v<0 for v in values),equal=sum(v==0 for v in values),higher=sum(v>0 for v in values))
            summary.append(record)
    write_json(dest/"dstu_summary.json",summary)
    lines=["# DSTU 机制与策略试点", "", "本报告读取独立 post_fg 结果；输入为已完成 F/G 的固定映射和路径。",
           "", "C4 排名取旧 All256 的估值与公共释放时刻；全部方案共用。Shared-only 只切与高优先流共享有向链路的低优先大消息。",
           "消息连续优先保留每个资源、每条流最近服务的 parent：有就绪子块时继续它；否则按原长路径优先和本轮 shuffle 顺序选择新 parent。新 parent 实际发放后才成为 owner。",
           "跨流抢占不清除低优先流的 owner；同 parent 的就绪块按 chunk_index 取最小者。key 只查询，不改变状态或随机数。", "",
           f"本次清单 {len(rows)} 个逻辑项，旧结果复用 {sum(r['historical_reuse'] for r in rows)}，本批等价单跑复用 {sum(bool(r['batch_alias']) for r in rows)}，新后端调用 {sum(not r['reused'] for r in rows)}。",
           "", "| 硬件 | 方案 | 联合次数 | P 相对 A256：中位 [最小,最大] | D 相对 A256：中位 [最小,最大] | 超期次数 |",
           "|---|---|---:|---|---|---:|"]
    for row in summary:
        def fmt(name):
            v=row[name];return f"{v['median']:+.2%} [{v['minimum']:+.2%}, {v['maximum']:+.2%}]"
        lines.append(f"| {row['hardware']} | {row['arm']} | {row['n']} | {fmt('P_change_vs_A256')} | {fmt('D_change_vs_A256')} | {row['deadline_violations']} |")
    lines += ["", "逐种子完成时间、相位、单跑 O、联合 J、公共期限、实际 KV 包络和主机成本见 [dstu_runs.csv](tables/dstu_runs.csv)。",
              f"共 {len(identities)} 个逐流等式 `joint差 = O + J新 − J基线` 已复算。",
              "", "O 表示相对 BB0 的单跑自身变化；J 是同方案联合减单跑。D 联合值使用完成时刻减公共释放时刻。单跑保留联合生成的 Shared-only 掩码。",
              "同 seed 用于配对，不声称不同切块图消耗相同随机序列。H0 与 BB0 同时包含后端与仲裁差异；模拟服务时间和 Python 墙钟分别记账。",
              "", "策略选择在两硬件全部相位与种子的结果齐备后统一做出，不能逐行选择赢家。代表资源日志用于检查同流插入、消息末块与消费者启动的因果顺序。"]
    (dest.parent/"DSTU_REFINEMENT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("child");p.add_argument("--spec",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("graph-check");p.add_argument("--spec",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("manifest");p.add_argument("--source-root",default=str(ROOT));p.add_argument("--output",default="e2e/post_fg/manifest/dstu_health.json");p.add_argument("--condition",default="healthy");p.add_argument("--selected")
    p=sub.add_parser("report");p.add_argument("--manifest",required=True);p.add_argument("--out",required=True);p.add_argument("--tables",default="e2e/post_fg/tables")
    args=parser.parse_args()
    if args.command=="child":child(read_json(args.spec),args.out)
    elif args.command=="graph-check":print(graph_check(read_json(args.spec),args.out))
    elif args.command=="manifest":print({k:v for k,v in manifest(args.source_root,args.output,args.condition,args.selected).items() if k!="jobs"})
    else:print("rows",len(report(args.manifest,args.out,args.tables)))


if __name__=="__main__":main()
