"""J0/J1/J4/J5 measurements in fresh, single-threaded processes.

The batch runner must use one worker in an otherwise idle server window.
Graph preparation is separate from the repeated J1 measurements.
"""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import collections
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import tempfile
import time

ENTRY_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
VERSION = "HIJ_timing_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    from hi_split_supplement import write as atomic_write
    atomic_write(path, value)


def environment():
    import numpy
    import numba
    import e2e_pilot_peak
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    return dict(host=platform.node(), system=platform.platform(), cpu=cpu, cpu_count=os.cpu_count(),
                python=sys.version, python_executable=sys.executable, numpy=numpy.__version__, numba=numba.__version__,
                available_gib=e2e_pilot_peak.free_gb(), command=sys.argv,
                thread_settings={key: os.environ[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
                code_sha256={p.name: sha(p) for p in (Path(__file__), ROOT / "e2e/post_fg_bb.py", ROOT / "cpcm/cpcm.py",
                                                     ROOT / "e2e/post_fg_split.py", ROOT / "e2e/post_fg_dstu.py")},
                idle_sample=Path("/proc/loadavg").read_text().strip() if Path("/proc/loadavg").exists() else None,
                timing_protocol="one worker; otherwise idle machine; five steady repetitions after one warmup")


def prepare_graph(spec, out):
    import f_config
    import e_run
    from post_fg_split import write_pickle
    started = time.perf_counter()
    t = time.perf_counter()
    _, c, cfg = f_config.prepare(spec, out)
    b, d, events, hp = e_run.load_snapshot(Path(spec["snapshot"]), c, cfg)
    with gzip.open(spec["source_raw"], "rb") as handle:
        raw = pickle.load(handle)
    loaded = time.perf_counter() - t
    t = time.perf_counter()
    complete = c.BcCache(raw["meta"]["ctx"], hp).graph(b, d, events, hp)
    broadcast = time.perf_counter() - t
    target = Path(spec["prepared_graph"])
    write_pickle(target, dict(events=complete, source_spec=spec))
    return dict(input_sha256={"snapshot": sha(spec["snapshot"]), "raw": sha(spec["source_raw"])},
                graph_file=str(target), graph_file_sha256=sha(target), pre_events=len(events), full_events=len(complete),
                load_seconds=loaded, initial_broadcast_seconds=broadcast,
                prepare_total_seconds=time.perf_counter() - started,
                measurement_scope="graph preparation only; no search or backend; concurrent preparation is not formal J timing")


def j1(spec, out):
    import post_fg_split as split
    split.setup()
    import cpcm
    import post_fg_bb as bb
    t = time.perf_counter()
    path = Path(spec["prepared_graph"])
    assert sha(path) == spec["prepared_graph_sha256"]
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    load_seconds = time.perf_counter() - t
    import f_config
    t = time.perf_counter()
    f_config.prepare(payload["source_spec"], out)
    hp = f_config.build_platform(payload["source_spec"], out)
    platform_seconds = time.perf_counter() - t
    events = payload["events"]
    rows = []
    for repeat in range(1 if spec.get("startup_mode") else 6):
        started = time.perf_counter()
        t = time.perf_counter()
        graph = cpcm.compile_bb(events, hp)
        compiled = time.perf_counter() - t
        t = time.perf_counter()
        schedule = bb.ORIGINAL(graph, "bb") if spec["implementation"] == "original" else bb.simulate(graph, "bb")
        forward = time.perf_counter() - t
        t = time.perf_counter()
        analysis = cpcm.Analysis(schedule)
        critical_path = analysis.critical_path()
        blocking = analysis.blocking()
        analyzed = time.perf_counter() - t
        row = dict(repetition=repeat, warmup=repeat == 0, compile_seconds=compiled,
                   forward_complete_seconds=forward, analysis_seconds=analyzed,
                   full_call_seconds=time.perf_counter() - started, T=int(schedule.T),
                   events=graph.n, activities=len(schedule.a_s), hops=len(graph.hop_link),
                   analysis_sha256=hashlib.sha256(pickle.dumps((critical_path, blocking))).hexdigest())
        if spec["implementation"] == "fast":
            row.update(bb.TIMING[-1])
        rows.append(row)
    result = dict(load_seconds=load_seconds, platform_seconds=platform_seconds, input_sha256=spec["prepared_graph_sha256"],
                warmup=rows[0], steady=rows[1:], graph_scope="full_W1_including_terminal_broadcast",
                implementation=spec["implementation"], backend_calls=0, search_runs=0)
    if spec.get("startup_mode"):
        result.update(startup_mode=spec["startup_mode"], numba_cache_dir=os.environ["NUMBA_CACHE_DIR"],
                      numba_signatures=[str(signature) for signature in bb.forward.signatures],
                      cache_files=[p.name for p in Path(os.environ["NUMBA_CACHE_DIR"]).rglob("*.nb*")])
    return result


def j4(spec, out):
    import post_fg_dstu as dstu
    started = time.perf_counter()
    t = time.perf_counter()
    d, hp, saved, streams = dstu.graph_inputs(spec, out)
    loaded = time.perf_counter() - t
    t = time.perf_counter()
    all_events, all_metadata, _, all_stats = dstu.transform(streams, hp, "A256", spec["rank"])
    all_chunk = time.perf_counter() - t
    t = time.perf_counter()
    estimates = {}
    for stream in (0, 1):
        selected = {tag: e for tag, e in all_events.items() if tag[1] - 10 == stream}
        estimates[stream] = int(d.that_hop(selected, hp)[0])
        assert estimates[stream] == saved["that"][str(stream)]
    deadlines = {0: 1.5 * estimates[0], 1: spec["tau"] + 1.5 * estimates[1]}
    order = sorted(deadlines, key=lambda key: (deadlines[key], key))
    rank = {key: order.index(key) for key in order}
    assert rank == {int(k): v for k, v in spec["rank"].items()}
    estimate_seconds = time.perf_counter() - t
    t = time.perf_counter()
    if spec["arm"] == "A256":
        events, metadata, stats = all_events, all_metadata, all_stats
    else:
        events, metadata, _, stats = dstu.transform(streams, hp, spec["arm"], rank)
    selected_seconds = time.perf_counter() - t
    t = time.perf_counter()
    policy = dstu.MessagePolicy(rank, metadata, mode="C4")
    ready = [tag for tag, event in events.items() if not event.dependency_set]
    policy_seconds = time.perf_counter() - t
    return dict(input_load_platform_seconds=loaded, shared_all256_chunk_seconds=all_chunk,
                cpcm_deadline_rank_seconds=estimate_seconds, selected_strategy_transform_seconds=selected_seconds,
                policy_and_initial_ready_list_seconds=policy_seconds, total_prepare_seconds=time.perf_counter() - started,
                events=len(events), ready_events=len(ready), policy_parents=len(policy.parent_order),
                cpcm_estimates=estimates, deadlines=deadlines, rank=rank, chunk_stats=stats,
                backend_calls=0, search_runs=0,
                scope="control preparation from frozen P/D graphs; no backend queue execution or per-packet hardware timing")


def instrumented_split(split):
    """Add eight clocks between unchanged transformation blocks, in memory only."""
    source = inspect.getsource(split.adapt)
    source = source.replace("    original=state[\"events\"]", "    jt=time.perf_counter(); phases={}\n    original=state[\"events\"]")
    markers = [
        ("    loads=defaultdict(Counter)", "orphan_and_legal_axes_seconds"),
        ("    graph=copy.deepcopy(original)", "placement_and_routing_score_seconds"),
        ("    broadcast_records=[]", "graph_and_input_transfers_seconds"),
        ("    for e in graph.values():e.issue_set=set();e.start_time=e.end_time=0", "terminal_broadcast_seconds"),
        ("    serial_pieces=", "semantic_checks_and_metrics_seconds"),
        ("    info=dict(version=VERSION", "arithmetic_and_transfer_accounting_seconds"),
    ]
    for marker, name in markers:
        assert source.count(marker) == 1
        source = source.replace(marker, f"    phases['{name}']=time.perf_counter()-jt; jt=time.perf_counter()\n" + marker)
    source = source.replace("    return graph,info", "    info['j_phases']=phases\n    return graph,info")
    namespace = dict(vars(split))
    exec(compile(source, "<J5 instrumented frozen Split>", "exec"), namespace)
    return namespace["adapt"], hashlib.sha256(source.encode()).hexdigest()


def j5(spec, out):
    import post_fg_split as split
    started = time.perf_counter()
    split.setup()
    t = time.perf_counter()
    old_root = split.ROOT
    run_out = Path(spec["run_out"])
    split.ROOT = run_out.parents[3]
    state = split.read_case(run_out, spec["case"])
    split.ROOT = old_root
    load_seconds = time.perf_counter() - t
    t = time.perf_counter()
    healthy = split.platform(state["spec"])
    hp = split.platform(state["spec"], state["dead"])
    platforms = time.perf_counter() - t
    t = time.perf_counter()
    baseline = split.event_metrics(state["events"], healthy)
    baseline_seconds = time.perf_counter() - t
    adapt, instrument_sha = instrumented_split(split)
    t = time.perf_counter()
    candidates = []
    for axis in ("N", "M"):
        graph, info = adapt(state, hp, 2, axis)
        score = (info["metrics"]["tensor_rounded_flops"] - baseline["tensor_rounded_flops"],
                 info["metrics"]["communication"]["route_link_bytes"], 0 if axis == "N" else 1)
        candidates.append((score, axis, graph, info))
    score, axis, graph, info = min(candidates, key=lambda row: row[0])
    decision = time.perf_counter() - t
    t = time.perf_counter()
    digest = split.graph_digest(graph)
    assert digest == spec["graph_sha256"] and axis == spec["axis"]
    verification_seconds = time.perf_counter() - t
    t = time.perf_counter()
    split.write_pickle(out / "graphs" / (spec["job_id"] + ".pkl.gz"),
                       dict(events_blob=pickle.dumps(graph, pickle.HIGHEST_PROTOCOL), info=info))
    exported = time.perf_counter() - t
    return dict(case=spec["case"], factor=2, axis=axis, graph_sha256=digest,
                load_state_seconds=load_seconds, platform_seconds=platforms,
                baseline_metrics_seconds=baseline_seconds, resident_decision_seconds=decision,
                frozen_graph_verification_seconds=verification_seconds, export_seconds=exported,
                complete_case_seconds=time.perf_counter() - started,
                candidates=[dict(axis=a, score=s, phases=i["j_phases"], build_seconds=i["build_seconds"])
                            for s, a, _, i in candidates],
                instrumented_function_sha256=instrument_sha, backend_calls=0, search_runs=0,
                scope="m2 decision includes both legal N/M constructions, unchanged placement, routing and checks")


def child(spec, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Numba reads its cache path on import, before any kernel module is loaded.
    if spec.get("startup_mode") == "first_jit":
        cache_parent = out / "numba_caches"
        cache_parent.mkdir(exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix=spec["job_id"] + "_", dir=cache_parent)
    elif spec.get("startup_mode") == "disk_cache":
        previous = read(out / "results" / (spec["cache_source_job"] + ".json"))
        assert previous["status"] == "ok" and previous["startup_mode"] == "first_jit"
        os.environ["NUMBA_CACHE_DIR"] = previous["numba_cache_dir"]
    elif "numba_cache_dir" in spec:
        os.environ["NUMBA_CACHE_DIR"] = spec["numba_cache_dir"]
    fn = {"j1_prepare": prepare_graph, "j1": j1, "j1_startup": j1, "j4": j4, "j5": j5}[spec["kind"]]
    result = fn(spec, out)
    import e2e_pilot_peak
    result.update(job_id=spec["job_id"], spec=spec, status="ok", protocol_version=VERSION,
                  wall_seconds=time.perf_counter() - ENTRY_START, peak_rss_gb=e2e_pilot_peak.peak() / 2**30,
                  environment=environment())
    write(out / "results" / (spec["job_id"] + ".json"), result)
    print(json.dumps(dict(job_id=spec["job_id"], status="ok", wall_seconds=result["wall_seconds"])), flush=True)


def manifests(source_root, graph_dir, destination, legacy64_root=None):
    source_root, graph_dir = Path(source_root), Path(graph_dir)
    legacy64_root = Path(legacy64_root) if legacy64_root is not None else source_root
    preparation = []
    for name, shape, mode, phase, op in (("GPT64_decode_ffn", "2x2", "fixed", "decode", "ffn"),
                                         ("GPT64_prefill_mha", "2x2", "fixed", "prefill", "mha"),
                                         ("GPT144_prefill_mha", "3x3", "growth", "prefill", "mha")):
        cid = f"{'F' if shape == '2x2' else 'G'}_busybarn_{shape}_{mode}_d2d1"
        candidate = f"{cid}__healthy_healthy_{phase}_{op}_A3h_g0_s123"
        if shape == "2x2":
            prefix = "cpcm_sa/out/s2" if phase == "decode" else "e2e/out/b4"
            old_name = "ffn_A3h_g0_s123" if phase == "decode" else "pf_mha_w1_A3h_g0_s123"
            snapshot = legacy64_root / prefix / "snaps" / (old_name + ".pkl.gz")
            raw = legacy64_root / prefix / "runs" / (old_name + ".pkl.gz")
        else:
            snapshot = source_root / "e2e/fg/out/main/snapshots" / (candidate + ".pkl.gz")
            raw = source_root / "e2e/fg/out/main/raw" / (candidate + ".pkl.gz")
        preparation.append(dict(job_id="J1_prepare_" + name, module="j_timing", kind="j1_prepare",
            input_id=name, hardware_profile="busybarn", shape=shape, split_mode=mode, d2d_scale=1,
            phase=phase, op=op, method="A3h", g=0, x=123, search_steps=1000,
            snapshot=snapshot.as_posix(), source_raw=raw.as_posix(),
            prepared_graph=(graph_dir / (name + ".pkl.gz")).as_posix(), protocol_version=VERSION))
    write(destination / "J1_prepare.json", dict(scope="input preparation, not formal timing", jobs=preparation))
    health = read(ROOT / "e2e/post_fg/manifest/quality_health_fresh_cloud.json")["jobs"]
    j4_jobs = []
    for hardware in ("wsc", "face"):
        for arm in ("A256", "S256"):
            src = next(s for s in health if s["hardware_profile"] == hardware and s.get("arm") == arm
                       and s.get("run_type") == "joint" and s.get("phase") == 0.5 and s.get("seed") == 1)
            for repeat in range(1, 4):
                j4_jobs.append(dict(src, kind="j4", module="j_timing", source_root=source_root.as_posix(),
                    job_id=f"J4_{hardware}_{arm}_r{repeat}", repetition=repeat, protocol_version=VERSION))
    write(destination / "J4.json", dict(scope="serial; no concurrent experiment workers", jobs=j4_jobs))
    old = read(ROOT / "e2e/post_fg/manifest/split.json")["jobs"]
    j5_jobs = []
    for src in (s for s in old if s["factor"] == 2 and s["seed"] == 1):
        for repeat in range(1, 4):
            j5_jobs.append(dict(job_id=f"J5_{src['case']}_m2_r{repeat}", module="j_timing", kind="j5",
                case=src["case"], axis=src["axis"], graph_sha256=src["graph_sha256"], repetition=repeat,
                run_out=(source_root / "e2e/fg/out/main").as_posix(), protocol_version=VERSION))
    write(destination / "J5.json", dict(scope="serial; existing three cases, m2 only", jobs=j5_jobs))
    write(destination / "J_INPUTS.json", dict(protocol_version=VERSION, first_batch=dict(
        J0="ready", J1_GPT="prepare inputs then finalize six-method/repeat jobs", J4="12 ready", J5="9 ready"),
        J1_preparation=preparation, J4=j4_jobs, J5=j5_jobs,
        deferred=["J1 new model graphs", "J2", "J3", "J5 H/I"]))


def finalize_j1(preparation_out, destination):
    rows = sorted(preparation_out.glob("results/J1_prepare_*.json"))
    assert len(rows) == 3
    jobs = []
    for filename in rows:
        row = read(filename)
        assert row["status"] == "ok"
        for implementation in ("original", "fast"):
            for repeat in range(1, 4):
                jobs.append(dict(job_id=f"J1_{row['spec']['input_id']}_{implementation}_r{repeat}",
                    module="j_timing", kind="j1", implementation=implementation, repetition=repeat,
                    prepared_graph=row["graph_file"], prepared_graph_sha256=row["graph_file_sha256"],
                    protocol_version=VERSION))
    write(destination, dict(scope="serial; 18 fresh processes; each warmup plus five steady calls", jobs=jobs))
    representative = next(row for row in jobs if row["job_id"] == "J1_GPT64_decode_ffn_fast_r1")
    startup = []
    for repeat in range(1, 4):
        cold_id = f"J1_first_jit_r{repeat}"
        startup.append(dict(representative, job_id=cold_id, kind="j1_startup", startup_mode="first_jit", repetition=repeat))
        startup.append(dict(representative, job_id=f"J1_disk_cache_r{repeat}", kind="j1_startup",
                            startup_mode="disk_cache", repetition=repeat, depends_on=[cold_id], cache_source_job=cold_id))
    write(destination.with_name("J1_startup.json"), dict(scope="serial; three private empty caches then their new-process loads", jobs=startup))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("environment", "manifest", "child", "finalize-j1"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "e2e/hi/j")
    parser.add_argument("--source-root", default="/home/ubuntu/crisp_fg_20261113")
    parser.add_argument("--legacy64-root")
    parser.add_argument("--graph-dir", default="/home/ubuntu/crisp_hi_20261114/e2e/hi/out/j_graphs")
    parser.add_argument("--preparation-out", type=Path)
    args = parser.parse_args()
    if args.command == "environment":
        write(args.out / "J_ENVIRONMENT.json", environment())
    elif args.command == "manifest":
        manifests(args.source_root, args.graph_dir, args.out, args.legacy64_root)
    elif args.command == "finalize-j1":
        finalize_j1(args.preparation_out, args.out / "J1.json")
    else:
        child(read(args.spec), args.out)
