"""E4a: link conflicts of several streams overlapping on one 2x2 die group (BusyBarn analytical backend).

Steps implemented here (handoff E4_HANDOFF.md, task list 1-2):
  capture   `python <task>.py` from the E4 work copy with _inject/sitecustomize.py on PYTHONPATH. At the
            template's 2nd (final) backend call the event objects are pickled before simulation (placements,
            sources, routes, tags, event order: the original objects), together with E3's mapping/route/layout
            capture (for the later fault steps) and a digest of the platform state the backend reads. The
            template's own result is checked against results/<task>.txt; a 2nd process re-simulates the same
            final state with backend seeds (deepcopy per seed), as E2/E3 do.
  simulate  child process: a fresh platform from the task cfg; the captured streams are unpickled, tags
            renamed (tag[1] = 10 + stream id) and merged into one event dict (stream 0's events first, each
            in its captured order); e4_backend.event_driver_e4 runs it with release times (approach (b)),
            the serial gate or a stream priority, once per backend seed on fresh objects.
  step2     a1 with 2 streams, 2x2, no fault, no chunking: singles, 5 staggers x 3 compositions, stream
            priority orders, E2 analysis with release times, both bounds, self-checks 1, 2, 3, 5, 6.

  python e4_overlap.py capture --streams mha512 mha2048 mha8192 ffn
  python e4_overlap.py step2 --jobs 12
"""
import argparse
import collections
import copy
import csv
import gzip
import hashlib
import itertools
import json
import os
import pickle
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E2_DIR = os.path.join(HERE, "..", "e2_attribution")
E3_DIR = os.path.join(HERE, "..", "e3_warmstart")
COMMIT_FILE = os.path.join(HERE, "..", "third_party", "isca2026-busybarn-artifact.COMMIT")
WORK = os.path.join(HERE, "work", "artifact")
MODEL = os.path.join(WORK, "models", "intra_mapping_gpt_decode")
PY_DIR = os.path.join(MODEL, "py")
CFG = os.path.join(MODEL, "cfg", "config_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0.cfg")
OUT = os.path.join(HERE, "e4_out")
STREAM_DIR = os.path.join(OUT, "streams")
TAIL = "ch2x2_bw256_co4x4_bw256_t128x64_failpattern0"
# winners of the 14 variants, verified in step 1 (e4_out/variants.json)
STREAMS = {"mha512": f"mha_sq512_greedy0_lossratio0_{TAIL}", "mha2048": f"mha_sq2048_greedy0_lossratio4_{TAIL}",
           "mha8192": f"mha_sq8192_greedy0_lossratio5_{TAIL}", "ffn": f"ffn_sq2048_greedy0_lossratio6_{TAIL}"}
SEEDS = [1, 2, 3]
KS = (1.1, 1.5, 2.0)
STAGGERS = [0.0, 0.25, 0.5, 1.0, "serial"]
A1_2 = {"512-8192": ["mha512", "mha8192"], "2048-8192": ["mha2048", "mha8192"], "2048-2048": ["mha2048", "mha2048"]}
TOL = 0.01  # handoff's latency tolerance: recorded only (user decision 2026-11-01)
NOISE_STOP = 0.04  # user decision 2026-11-01: latency differences stop the run only beyond the known ~4% noise
COMPUTE_STOP = 0.70  # handoff 7: stop if critical wait is mostly on compute units


class E4Error(RuntimeError):
    pass


def _mod(dirname, name):
    sys.path.insert(0, dirname)
    m = __import__(name)
    sys.path.pop(0)
    return m


def _e2():
    return _mod(E2_DIR, "e2_attribution")


def _e3():
    return _mod(E3_DIR, "e3_warmstart")


def _artifact_commit():
    try:
        with open(COMMIT_FILE, encoding="utf-8-sig") as f:
            return f.read().strip().split()[0]
    except OSError:
        return None


def platform_state(hp):
    """What the backend reads from the platform: link and device set, link timing, idle state."""
    links = sorted((str(k), l.link_type, l.bandwidth, l.latency, l.timeunit, bool(l.work_flag), l.work_endtime,
                    len(l.work_record)) for k, l in hp.links_dict.items())
    mods = sorted((t, str(loc), bool(getattr(m, "work_flag", False)), getattr(m, "work_endtime", None),
                   len(getattr(m, "work_record", []) or [])) for t, ms in hp.modules_dict.items() for loc, m in ms.items())
    blob = repr((links, mods, hp.frequency))
    return {"digest": hashlib.sha1(blob.encode()).hexdigest(), "n_links": len(links), "n_modules": len(mods),
            "busy_or_recorded": sum(1 for x in links if x[5] or x[7]) + sum(1 for x in mods if x[2] or x[4])}


# ---------------------------------------------------------------- capture (child)
def capture_final(orig, events_dict, hp, call_no, g):
    e2, e3 = _e2(), _e3()
    seeds = [int(s) for s in os.environ.get("E4_SEEDS", "").split()]
    sink = {"calls": call_no, "reseeded": []}
    pre = e2._snapshot_pre(events_dict)
    canon = sorted((str(t), sorted(map(str, p["deps"])), sorted(map(str, p["issue"])), str(p["path_list"]))
                   for t, p in pre.items())
    sink["pre_digest"] = hashlib.sha1(repr(canon).encode()).hexdigest()
    sink["events_blob"] = pickle.dumps(events_dict, protocol=pickle.HIGHEST_PROTOCOL)  # read only
    sink["platform_state"] = platform_state(hp)
    sink["empty_path_comms"] = sum(1 for p in pre.values() if p["path_list"] == [])
    digest = e3._rng_digest()
    reruns = copy.deepcopy((events_dict, hp)) if seeds else None
    t0 = time.time()
    out = orig(events_dict=events_dict, hardware_platform=hp)
    sink["eval_wall"] = time.time() - t0
    sink["last"] = e2._extract(events_dict, hp, pre, out, digest)
    sink["mapping"] = e3.mapping_of(g)
    sink["mapping"]["routes"] = e3.routes_of(events_dict, pre)
    sink["mapping"]["layout"] = e3.event_layout_of(events_dict)
    sink["platform"] = e3.platform_info(hp)
    if os.environ.get("E4_MODE") == "inject":
        sink["inject_log"] = e3._spec()["_log"]
    if reruns is not None:
        state = random.getstate(), np.random.get_state()
        try:
            for seed in seeds:
                ev_k, hp_k = copy.deepcopy(reruns)
                random.seed(seed)
                t0 = time.time()
                out_k = orig(events_dict=ev_k, hardware_platform=hp_k)
                sink["reseeded"].append((seed, e2._extract(ev_k, hp_k, pre, out_k, f"seed{seed}"), time.time() - t0))
        finally:
            random.setstate(state[0])
            np.random.set_state(state[1])
    with open(os.environ["E4_SINK"], "wb") as f:
        pickle.dump(sink, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def _run_capture_child(task, seeds, tag, e3_spec=None):
    """capture (template unchanged) or, with e3_spec, inject: E3's spec dict (mapping, faults, steps=None for
    R0, preserve_routes=True)."""
    os.makedirs(os.path.join(OUT, "_tmp"), exist_ok=True)
    os.makedirs(os.path.join(MODEL, "results"), exist_ok=True)
    sink = os.path.join(OUT, "_tmp", f"sink_{task[:10]}_{tag}_{os.getpid()}.pkl")
    env = dict(os.environ, E4_MODE="capture", E4_HOME=HERE, E4_SINK=sink, E4_SEEDS=" ".join(map(str, seeds)),
               E3_HOME=E3_DIR,
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "_inject")] +
                                          ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    spec_path = None
    if e3_spec is not None:
        spec_path = os.path.join(OUT, "_tmp", f"e3spec_{tag}_{os.getpid()}.pkl")
        with open(spec_path, "wb") as f:
            pickle.dump(e3_spec, f, protocol=pickle.HIGHEST_PROTOCOL)
        env.update(E4_MODE="inject", E3_SPEC=spec_path)
    log = os.path.join(OUT, "logs", f"{task}_{tag}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        rc = subprocess.call([sys.executable, os.path.join(PY_DIR, task + ".py")], env=env, cwd=HERE, stdout=lf,
                             stderr=subprocess.STDOUT)
    if spec_path:
        os.remove(spec_path)
    if rc != 0 or not os.path.exists(sink):
        raise E4Error(f"{task} {tag}: task script failed (exit {rc}) or never reached a 2nd backend call; see {log}")
    with open(sink, "rb") as f:
        out = pickle.load(f)
    os.remove(sink)
    if out["calls"] != 2:
        raise E4Error(f"{task} {tag}: expected 2 backend calls, saw {out['calls']}")
    out["process_wall"] = time.time() - t0
    return out


def capture(short):
    import re
    task = STREAMS[short]
    with ThreadPoolExecutor(2) as ex:
        fp = ex.submit(_run_capture_child, task, [], "capture")
        fs = ex.submit(_run_capture_child, task, SEEDS, "capture_seeds")
        prim, rs = fp.result(), fs.result()
    if rs["pre_digest"] != prim["pre_digest"]:
        raise E4Error(f"{short}: re-seed process saw a different pre-backend state than the primary capture")
    with open(os.path.join(MODEL, "results", f"{task}.txt")) as f:
        m = re.search(r"SA time cost:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", f.read())
    reported = tuple(map(int, m.groups()))
    rec = {"short": short, "task": task, "events_blob": prim["events_blob"], "pre_digest": prim["pre_digest"],
           "platform_state": prim["platform_state"], "empty_path_comms": prim["empty_path_comms"],
           "template_result": prim["last"]["result"], "reported_result": reported, "final": prim["last"],
           "mapping": prim["mapping"], "platform": prim["platform"],
           "reseeded": [(s, c) for s, c, _ in rs["reseeded"]], "reseed_walls": [w for _, _, w in rs["reseeded"]],
           "blob_identical_across_processes": prim["events_blob"] == rs["events_blob"],
           "artifact_commit": _artifact_commit(), "template_python_seed": 123, "backend_seeds": SEEDS,
           "python": sys.executable}
    os.makedirs(STREAM_DIR, exist_ok=True)
    with gzip.open(os.path.join(STREAM_DIR, f"{short}.pkl.gz"), "wb") as f:
        pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
    lat = {s: c["result"][0] for s, c in rec["reseeded"]}
    info = {"short": short, "template": rec["template_result"], "reported": reported,
            "template_matches_reported": tuple(rec["template_result"]) == reported, "seeds": lat,
            "seed_mean": float(np.mean(list(lat.values()))), "n_events": len(rec["final"]["events"]),
            "empty_path_comms": rec["empty_path_comms"], "platform_state": rec["platform_state"],
            "blob_identical_across_processes": rec["blob_identical_across_processes"]}
    print(f"[e4] capture {short}: {json.dumps(info, default=str)}", flush=True)
    return info


def load_stream(short):
    with gzip.open(os.path.join(STREAM_DIR, f"{short}.pkl.gz"), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- tags
def tag_facts(events_blob_loader):
    """Handoff 2: computation tags 5-tuples, communication tags 6-tuples, tag[0] = 0/1, tag[1] = tag[2] = 0,
    a communication's owner (0,) + tag[1:-1] is a computation event."""
    ev = events_blob_loader
    comp = {t for t, e in ev.items() if e.event_type == "computation"}
    comm = {t for t, e in ev.items() if e.event_type == "communication"}
    return {"n_comp": len(comp), "n_comm": len(comm),
            "comp_len": sorted({len(t) for t in comp}), "comm_len": sorted({len(t) for t in comm}),
            "tag0_comp": sorted({t[0] for t in comp}), "tag0_comm": sorted({t[0] for t in comm}),
            "tag1": sorted({t[1] for t in ev}), "tag2": sorted({t[2] for t in ev}),
            "comm_owner_is_comp": all((0,) + tuple(t[1:-1]) in comp for t in comm),
            "event_tag_attr_matches_key": all(tuple(e.event_tag) == tuple(t) for t, e in ev.items())}


def rename_stream(ev, i):
    """tag[1] = 10 + i on every event and in every dependency / issue set (objects modified in place)."""
    m = {t: (t[0], 10 + i) + tuple(t[2:]) for t in ev}
    out = {}
    for t, e in ev.items():
        if t[1] != 0 or t[2] != 0:
            raise E4Error(f"tag {t}: positions 1 and 2 are not 0; the renaming rule of handoff 2 does not apply")
        e.event_tag = m[t]
        e.dependency_set = {m[x] for x in e.dependency_set}
        e.issue_set = {m[x] for x in e.issue_set}
        out[m[t]] = e
    comp = {t for t, e in out.items() if e.event_type == "computation"}
    bad = [t for t, e in out.items() if e.event_type == "communication" and (0,) + t[1:-1] not in comp]
    if bad:
        raise E4Error(f"stream {i}: communication owner not found after renaming: {bad[:3]}")
    return out


def merged_events(blobs):
    ev = {}
    for i, b in enumerate(blobs):
        ev.update(rename_stream(pickle.loads(b), i))
    return ev


CHUNK_ID_BASE = 1_000_000  # block b >= 1 of communication (..., id) gets id + b * CHUNK_ID_BASE (ids are small)


def chunk_events(ev, hp, size, only_d2d=False):
    """Handoff 3.2: a message larger than `size` bytes becomes ceil(B/size) communications with the same source,
    target and routing tree (blocks of `size` bytes, the last one the remainder). Every block depends on the
    original producers, every consumer depends on all blocks, blocks do not depend on each other. Block 0
    keeps the original tag; block b >= 1 gets id + b * CHUNK_ID_BASE, so (0,) + tag[1:-1] still names the owner
    computation. only_d2d: only messages whose tree uses a ch2ch link. Returns (new event dict in the original
    order with blocks after their message, stats incl. the extra link occupancy of the per-block latency)."""
    out, stats = {}, {"messages_chunked": 0, "blocks_added": 0, "bytes_conserved": True, "tree_conserved": True,
                      "extra_occupancy": 0, "extra_occupancy_d2d": 0, "base_occupancy": 0, "extra_by_kind": {}}
    blocks_of = {}
    for t, e in ev.items():
        if e.event_type != "communication":
            continue
        links = [(r, l) for r, leaves in e.paths.items() for l in leaves]
        kinds = [hp.links_dict[k].link_type for k in links]
        stats["base_occupancy"] += sum(int(hp.links_dict[k].working_time(data_bytes=e.comm_bytes)) for k in links)
        B = int(e.comm_bytes)
        if B <= size or (only_d2d and "ch2ch" not in kinds):
            continue
        n = -(-B // size)
        sizes = [size] * (n - 1) + [B - size * (n - 1)]
        tags = [t] + [tuple(t[:-1]) + (t[-1] + b * CHUNK_ID_BASE,) for b in range(1, n)]
        blocks_of[t] = (tags, sizes, links, kinds)
        stats["messages_chunked"] += 1
        stats["blocks_added"] += n - 1
        if sum(sizes) != B:
            stats["bytes_conserved"] = False
        for k, kind in zip(links, kinds):
            w = hp.links_dict[k]
            extra = sum(int(w.working_time(data_bytes=x)) for x in sizes) - int(w.working_time(data_bytes=B))
            stats["extra_occupancy"] += extra
            stats["extra_by_kind"][kind] = stats["extra_by_kind"].get(kind, 0) + extra
            if kind == "ch2ch":
                stats["extra_occupancy_d2d"] += extra
    for t, e in ev.items():
        if t not in blocks_of:
            out[t] = e
            continue
        tags, sizes, links, _ = blocks_of[t]
        tree0 = sorted((tuple(r), tuple(map(tuple, ls))) for r, ls in e.paths.items())
        objs = [e] + [copy.deepcopy(e) for _ in tags[1:]]
        for o, tg, sz in zip(objs, tags, sizes):
            o.event_tag, o.comm_bytes = tg, sz
            o.dependency_set, o.issue_set = set(e.dependency_set), set(e.issue_set)
            if sorted((tuple(r), tuple(map(tuple, ls))) for r, ls in o.paths.items()) != tree0:
                stats["tree_conserved"] = False
            out[tg] = o
    # producers issue to every block, consumers wait for every block
    expand = lambda x: blocks_of[x][0] if x in blocks_of else [x]
    orig_sets = {t: (set(ev[t].dependency_set), set(ev[t].issue_set)) for t in blocks_of}
    for t, (tags, _, _, _) in blocks_of.items():
        deps, issue = orig_sets[t]
        for d in deps:
            for dd in expand(d):
                out[dd].issue_set.discard(t)
                out[dd].issue_set.update(tags)
        for c in issue:
            for cc in expand(c):
                out[cc].dependency_set.discard(t)
                out[cc].dependency_set.update(tags)
    stats["n_events"] = len(out)
    stats["extra_over_base"] = stats["extra_occupancy"] / stats["base_occupancy"] if stats["base_occupancy"] else 0
    return out, stats


def strip_cap(cap):
    """Cap of a 1-stream merged run with tags mapped back to the stream's own (tag[1] = 0)."""
    f = lambda t: (t[0], 0) + tuple(t[2:])
    ev = {}
    for t, e in cap["events"].items():
        e = dict(e)
        e["tag"] = f(t)
        e["deps"] = {f(x) for x in e["deps"]}
        e["issue"] = {f(x) for x in e["issue"]}
        ev[f(t)] = e
    return dict(cap, events=ev)


# ---------------------------------------------------------------- simulate (child)
def _artifact_path():
    for d in ["", "../../../utils/", "../../../src/partition/oper/", "../../../src/partition/func/",
              "../../../src/partition/", "../../../src/mapping/", "../../../src/scheduling/",
              "../../../src/backend/analytical/", "../../../endtoend/",
              "../../../src/scheduling/communication/topology/", "../../../tool/"]:
        sys.path.append(os.path.normpath(os.path.join(PY_DIR, d)))


def build_platform(cfg_path, faults=None):
    """Fresh platform from the task cfg with the [failures] of `faults`, as E3's read_cfg patch sets them, and
    E3's in-memory fix of WAMIS_HD.xy_multicast_path (DDR branch uses xy_pointopoint_path's result before its
    None check, so a failed on-chip link raises TypeError while the platform is built; return None instead,
    as the non-DDR branch does). Call inside a simulate child (artifact on sys.path)."""
    import WAMIS_HD
    from read_cfg import cfg_to_dict
    cls = WAMIS_HD.wamis_hdc
    if not getattr(cls, "_e4_xy_fix", False):
        orig_xy = cls.xy_multicast_path

        def xy_multicast_path(self, source_node_idx, target_nodes_set):
            try:
                return orig_xy(self, source_node_idx=source_node_idx, target_nodes_set=target_nodes_set)
            except TypeError as exc:
                if "NoneType" not in str(exc):
                    raise
                return None
        cls.xy_multicast_path = xy_multicast_path
        cls._e4_xy_fix = True
    cfg = cfg_to_dict(cfg_path)
    if faults and (faults.get("failed_nodes") or faults.get("failed_links")):
        cfg["failures"]["failed_nodes"] = [tuple(n) for n in faults.get("failed_nodes", [])]
        cfg["failures"]["failed_links"] = [(tuple(a), tuple(b)) for a, b in faults.get("failed_links", [])]
    return cls(cfg)


def simulate_child(spec_path):
    with open(spec_path, "rb") as f:
        spec = pickle.load(f)
    _artifact_path()
    from e4_backend import event_driver_e4
    e2 = _e2()
    hp0 = build_platform(spec.get("cfg", CFG), spec.get("faults"))
    ps = platform_state(hp0)
    recs = {s: load_stream(s) for s in set(spec["streams"])}
    for s, r in recs.items():
        if r["platform_state"]["digest"] != ps["digest"]:
            raise E4Error(f"fresh platform differs from the one {s}'s template simulated on")
    blobs = [recs[s]["events_blob"] for s in spec["streams"]]
    sink = {"spec": {k: v for k, v in spec.items()}, "platform_state": ps, "runs": []}
    state = random.getstate()
    for var in spec["variants"]:
        for seed in spec["seeds"]:
            ev = merged_events(blobs)
            cst = None
            if var.get("chunk"):
                t0 = time.time()
                ev, cst = chunk_events(ev, hp0, var["chunk"]["size"], var["chunk"].get("only_d2d", False))
                cst["chunk_wall"] = time.time() - t0
            hp = copy.deepcopy(hp0)
            pre = e2._snapshot_pre(ev)
            released = {}
            random.seed(seed)
            t0 = time.time()
            out = event_driver_e4(ev, hp, release=var.get("release"), gate=var.get("gate"), prio=var.get("prio"),
                                  released=released)
            wall = time.time() - t0
            cap = e2._extract(ev, hp, pre, out, f"seed{seed}")
            cap["release"] = released
            sink["runs"].append({"variant": var["name"], "seed": seed, "wall": wall, "cap": cap, "chunk_stats": cst})
    random.setstate(state)
    with open(spec["sink"], "wb") as f:
        pickle.dump(sink, f, protocol=pickle.HIGHEST_PROTOCOL)


def simulate(name, streams, variants, seeds=SEEDS, timeout=None, cfg=CFG, faults=None):
    tmp = os.path.join(OUT, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    spec_path = os.path.join(tmp, f"spec_{name}.pkl")
    sink = os.path.join(tmp, f"sink_{name}.pkl")
    with open(spec_path, "wb") as f:
        pickle.dump({"name": name, "streams": streams, "variants": variants, "seeds": list(seeds), "sink": sink,
                     "cfg": cfg, "faults": faults}, f)
    log = os.path.join(OUT, "logs", f"sim_{name}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        try:
            rc = subprocess.call([sys.executable, os.path.abspath(__file__), "simulate", "--spec", spec_path],
                                 cwd=HERE, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            os.remove(spec_path)
            return {"timeout": timeout, "process_wall": time.time() - t0, "runs": []}
    if rc != 0 or not os.path.exists(sink):
        raise E4Error(f"simulate {name} failed (exit {rc}); see {log}")
    with open(sink, "rb") as f:
        out = pickle.load(f)
    os.remove(sink)
    os.remove(spec_path)
    out["process_wall"] = time.time() - t0
    return out


# ---------------------------------------------------------------- analysis (worker processes)
def analyse_cap(cap, T, single=False):
    import e4_attr
    res = {"ok": True}
    try:
        summ, rows, segs, reorder, ddr_rows, sim = e4_attr.analyse(cap)
    except e4_attr.e2.E2Error as exc:
        return {"ok": False, "error": str(exc)}
    rel = {i: cap["release"][i] for i in sorted(cap["release"])}
    meas = e4_attr.stream_measures(cap, T, KS)
    res["checks"] = {k: summ[k] for k in ["check1_chain_end_equals_a", "check2_chain_intervals_sum_to_a",
                                          "check3_all_s_ge_r", "check4_activity_graph_longest_path_equals_a"]}
    res["release_row_on_chain"] = rows[0]["edge"] == "release"
    res["release_in_segs"] = sum(1 for s in segs if s["resource"] == "release" or s["blocker"] is None)
    tot = sum(s["wait_excl"] for s in segs)
    cross = sum(s["wait_excl"] for s in segs if e4_attr.stream_of(s["blocked"]) != e4_attr.stream_of(s["blocker"]))
    by_class = collections.Counter()
    cross_by_class = collections.Counter()
    for s in segs:
        c = e4_attr.kind_class(s["resource_kind"])
        by_class[c] += s["wait_excl"]
        if e4_attr.stream_of(s["blocked"]) != e4_attr.stream_of(s["blocker"]):
            cross_by_class[c] += s["wait_excl"]
    res.update({
        "a": summ["latency_a"], "critical_wait_over_a": summ["critical_wait_over_a"],
        "critical_wait_total": tot, "cross_stream_wait": cross, "cross_stream_share": cross / tot if tot else 0.0,
        "wait_share_by_class": {k: v / tot for k, v in by_class.items()} if tot else {},
        "cross_wait_share_by_class": {k: v / tot for k, v in cross_by_class.items()} if tot else {},
        "wait_share_by_resource_kind": summ["wait_share_by_resource_kind"],
        "chain_time_by_resource_kind_over_a": summ["chain_time_by_resource_kind_over_a"],
        "wait_top_resources": summ["wait_top_resources"], "measures": meas})
    # makespan objective (E2): best single-resource exact reorder and "delete the resource", top 5 by wait
    top5 = reorder[:5]
    a = summ["latency_a"]
    res["makespan_bound"] = {
        "reorder_gain_over_a_top5_max": max((r["reorder_gain_over_a"] for r in top5), default=0.0),
        "reorder_gain_ub_over_a_top5_max": max((r["reorder_gain_ub_over_a"] for r in top5), default=0.0),
        "delete_gain_over_a_top5_max": max(((a - r["longest_path_avoiding_resource"]) / a for r in top5), default=0.0),
        "top5": [{k: r[k] for k in ["resource", "wait_excl", "n_items", "current_max_f_plus_q", "schrage_edf",
                                    "exact_nonpreemptive", "jackson_preemptive_lb", "longest_path_avoiding_resource",
                                    "reorder_gain_over_a", "reorder_gain_ub_over_a"]} for r in top5]}
    # S_max objective, single resource, same top 5
    keys = []
    w = collections.Counter()
    for s in segs:
        w[s["res_key"]] += s["wait_excl"]
    keys = [k for k, _ in w.most_common(5)]
    bounds = [e4_attr.smax_resource(sim, rk, rel, T, KS) for rk in keys]
    S = meas["S_max"]
    for b in bounds:
        b["model_matches_S_max"] = abs(b["S_cur_model"] - S) <= 1e-9 * max(1.0, S)
        b["S_reduction"] = (S - b["S_opt_ub"]) / S
        b["S_reduction_ub"] = (S - b["S_opt_lb"]) / S
    res["smax_bound"] = {"top5": bounds,
                         "S_reduction_top5_max": max((b["S_reduction"] for b in bounds), default=0.0),
                         "S_reduction_ub_top5_max": max((b["S_reduction_ub"] for b in bounds), default=0.0),
                         "all_models_match": all(b["model_matches_S_max"] for b in bounds),
                         "Lmax_current": {str(k): max(meas["C"][i] - rel[i] - k * T[i] for i in rel) for k in KS},
                         "Lmax_best_top5_min": {str(k): min((b["Lmax"][str(k)]["best"] for b in bounds), default=None)
                                                for k in KS}}
    res["slack"] = e4_attr.slack_spread(rel, meas["C"], T, KS)
    if single:  # self-check 6: S_max bound x T == E2's exact single-resource reorder optimum
        by_name = {r["resource"]: r for r in reorder}
        rows6 = []
        for b in bounds:
            r = by_name[b["resource"]]
            opt = r["exact_nonpreemptive"]
            e2_best = max(r["longest_path_avoiding_resource"], min(r["current_max_f_plus_q"], r["schrage_edf"])
                          if opt is None else opt)
            Ti = T[0]
            rows6.append({"resource": b["resource"], "e2_best": e2_best, "smax_times_T": b["S_opt_ub"] * Ti,
                          "e2_exact_known": opt is not None, "smax_exact_everywhere": b["exact_everywhere"],
                          "match": abs(b["S_opt_ub"] * Ti - e2_best) <= 1.0})
        res["check6"] = {"rows": rows6, "pass": all(x["match"] for x in rows6)}
    return res


# ---------------------------------------------------------------- step 2 driver
def _mean(xs):
    return float(np.mean(xs)) if xs else None


def step2(jobs):
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    needed = sorted({s for v in A1_2.values() for s in v})
    for s in needed:
        if not os.path.exists(os.path.join(STREAM_DIR, f"{s}.pkl.gz")):
            raise E4Error(f"stream {s} not captured; run `capture` first")
    recs = {s: load_stream(s) for s in needed}
    # ---- singles (self-check 1) and T_i
    print("[e4] singles", flush=True)
    with ThreadPoolExecutor(min(jobs, len(needed))) as ex:
        singles = dict(zip(needed, ex.map(lambda s: simulate(f"single_{s}", [s], [{"name": "default",
                                                                                  "release": {0: 0}}]), needed)))
    T, check1 = {}, {}
    for s in needed:
        runs = singles[s]["runs"]
        lat = [r["cap"]["result"][0] for r in runs]
        orig = {sd: c for sd, c in recs[s]["reseeded"]}
        olat = [orig[sd]["result"][0] for sd in SEEDS]
        diffs = [_e3().graph_diff(strip_cap(r["cap"]), orig[r["seed"]]) for r in runs]
        T[s] = _mean(lat)
        rel_err = abs(T[s] - _mean(olat)) / _mean(olat)
        check1[s] = {"merged_latency": dict(zip(SEEDS, lat)), "original_latency": dict(zip(SEEDS, olat)),
                     "merged_mean": T[s], "original_mean": _mean(olat), "rel_diff": rel_err,
                     "graph_diff": diffs, "within_1pct": rel_err <= TOL,
                     "pass": all(d == 0 for d in diffs) and rel_err <= NOISE_STOP}
        print(f"[e4] check1 {s}: {json.dumps(check1[s])}", flush=True)
    if not all(c["pass"] for c in check1.values()):
        _write_json("step2_check1_fail.json", {"check1": check1})
        raise E4Error(f"self-check 1 failed: {check1}")

    # ---- configs
    configs = a1_configs(T)
    print(f"[e4] {len(configs)} configs, {sum(len(c['variants']) * len(c['seeds']) for c in configs)} simulations",
          flush=True)
    sims = simulate_configs(configs, jobs)
    print(f"[e4] simulations done ({time.time() - t_start:.0f} s)", flush=True)

    # ---- analysis (default arbitration graphs, plus singles for self-check 6)
    tasks = []
    for s in needed:
        for r in singles[s]["runs"]:
            tasks.append((("single", s, r["seed"]), r["cap"], {0: T[s]}, True))
    for c in configs:
        for r in sims[c["name"]]["runs"]:
            if r["variant"] == "default":
                tasks.append(((c["name"], "default", r["seed"]), r["cap"], c["T"], False))
    with ProcessPoolExecutor(jobs) as ex:
        futs = {key: ex.submit(analyse_cap, cap, Tm, single) for key, cap, Tm, single in tasks}
        an = {k: f.result() for k, f in futs.items()}
    print(f"[e4] analysis done ({time.time() - t_start:.0f} s)", flush=True)
    bad5 = {str(k): v.get("error") or v["checks"] for k, v in an.items()
            if not v["ok"] or not all(v["checks"].values()) or v["release_in_segs"]}
    check6 = {f"{k[1]}_seed{k[2]}": an[k]["check6"] for k in an if k[0] == "single"}

    # ---- per config
    table = []
    check2_all, check3 = [], {}
    stop_compute = []
    for c in configs:
        summ, c2, c3, comp_share = summarize_config(c, sims[c["name"]]["runs"], an)
        check2_all += c2
        if c3 is not None:
            check3[c["name"]] = c3
        if comp_share > COMPUTE_STOP:
            stop_compute.append((c["name"], comp_share))
        table.append(_table_row(summ))
    _write_csv("e4_table.csv", table)

    checks = {
        "check1_single_stream": {"pass": all(v["pass"] for v in check1.values()), "detail": check1},
        "check2_release_respected": {"pass": all(check2_all) and not any(an[k]["release_in_segs"] for k in an),
                                     "n_runs": len(check2_all), "impl": "(b): no virtual events exist",
                                     "release_rows_on_chain": sum(1 for v in an.values() if v.get("release_row_on_chain"))},
        "check3_serial": {"pass": all(v["pass"] for v in check3.values()), "detail": check3},
        "check5_e2_analysis": {"pass": not bad5, "n_graphs": len(an), "failures": bad5,
                               "smax_model_matches_actual": all(v["smax_bound"]["all_models_match"] for v in an.values()
                                                                if v["ok"])},
        "check6_smax_vs_e2": {"pass": all(v["pass"] for v in check6.values()), "detail": check6},
    }
    out = {"checks": checks, "T": T, "stop_compute_dominant": stop_compute,
           "wall_seconds": time.time() - t_start, "artifact_commit": _artifact_commit(), "command": sys.argv,
           "python": sys.executable}
    _write_json("step2_checks.json", out)
    line = {"checks": {k: v["pass"] for k, v in checks.items()}, "T": T,
            "S_max": {r["config"]: round(r["S_max_mean"], 4) for r in table},
            "cross": {r["config"]: round(r["cross_stream_share_mean"], 4) for r in table},
            "smax_red": {r["config"]: round(r["smax_bound_reduction_mean"], 4) for r in table},
            "prio": {r["config"]: round(r["prio_best_over_default"], 4) for r in table},
            "stop_compute": stop_compute}
    print("[e4] " + json.dumps(line, default=str), flush=True)


def a1_configs(T, only=None, d1_seeds=SEEDS):
    """a1 with 2 streams: 3 compositions x 5 staggers; every variant = default arbitration + each priority order."""
    configs = []
    for comp, streams in A1_2.items():
        Ts = [T[s] for s in streams]
        Tmin = min(Ts)
        for st in STAGGERS:
            n = len(streams)
            if st == "serial":
                base = {"gate": {i: i - 1 for i in range(1, n)}, "release": {0: 0}}
                label = "serial"
            else:
                delta = int(round(st * Tmin))
                base = {"release": {i: i * delta for i in range(n)}}
                label = f"d{st:g}"
            name = f"a1_2s_{comp}_{label}"
            if only is not None and name not in only:
                continue
            variants = [dict(base, name="default")]
            for perm in itertools.permutations(range(n)):
                variants.append(dict(base, name="prio_" + "".join(map(str, perm)),
                                     prio={sidx: rank for rank, sidx in enumerate(perm)}))
            configs.append({"name": name, "composition": "a1", "n_streams": n, "comp": comp,
                            "streams": streams, "stagger": st, "T": {i: T[s] for i, s in enumerate(streams)},
                            "Tmin": Tmin, "variants": variants,
                            "seeds": list(d1_seeds) if st == 1.0 else list(SEEDS)})
    return configs


def simulate_configs(configs, jobs):
    """One process per (config, variant); returns {config: {"runs": [...]}} with runs in variant order."""
    work = [(c, v) for c in configs for v in c["variants"]]
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(lambda cv: simulate(f"{cv[0]['name']}_{cv[1]['name']}", cv[0]["streams"], [cv[1]],
                                              cv[0]["seeds"]), work))
    out = {c["name"]: {"runs": []} for c in configs}
    for (c, v), r in zip(work, res):
        out[c["name"]]["runs"] += r["runs"]
    return out


def prio_detail(rows, n):
    """Default arbitration vs the best priority order (lowest mean S_max), from per-run rows (variant, seed,
    stretch_i, makespan, beat): per-stream stretch, makespan and beat means and their ratios."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)

    def agg(rs):
        return {"S_max": _mean([float(r["S_max"]) for r in rs]),
                "stretch": {i: _mean([float(r[f"stretch_{i}"]) for r in rs]) for i in range(n)},
                "makespan": _mean([float(r["makespan"]) for r in rs]),
                "beat": _mean([float(r["beat"]) for r in rs]), "n_seeds": len(rs)}
    d = agg(by["default"])
    orders = {v: agg(rs) for v, rs in by.items() if v != "default"}
    best = min(orders, key=lambda v: orders[v]["S_max"])
    b = orders[best]
    return {"default": d, "orders": orders, "best_order": best, "best": b,
            "best_over_default": {"S_max": b["S_max"] / d["S_max"],
                                  "stretch": {i: b["stretch"][i] / d["stretch"][i] for i in range(n)},
                                  "makespan": b["makespan"] / d["makespan"], "beat": b["beat"] / d["beat"]}}


def _table_row(summ):
    return {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in summ.items()
            if k not in ("per_seed_analysis", "tasks", "command")}


def summarize_config(c, runs, an):
    """Writes <config>_runs.csv and <config>_summary.json; returns (summary, release-respected flags, check 3
    entry or None, mean compute share of critical wait)."""
    import e4_attr
    seeds = c.get("seeds", SEEDS)
    if True:
        rows, per = [], collections.defaultdict(list)
        check2_all = []
        for r in runs:
            m = e4_attr.stream_measures(r["cap"], c["T"], KS)
            check2_all.append(m["release_respected"])
            per[r["variant"]].append(m)
            rows.append({"config": c["name"], "composition": c["composition"], "n_streams": c["n_streams"],
                         "streams": "+".join(c["streams"]), "stagger": c["stagger"], "hardware": c.get("hardware", "nofault"),
                         "chunking": c.get("chunking", "none"), "variant": r["variant"], "seed": r["seed"],
                         **{f"release_{i}": m["release"][i] for i in sorted(m["release"])},
                         **{f"latency_{i}": m["latency"][i] for i in sorted(m["latency"])},
                         **{f"stretch_{i}": m["stretch"][i] for i in sorted(m["stretch"])},
                         "S_max": m["S_max"], "makespan": m["makespan"], "beat": m["beat"],
                         "overlap_ge2": m["overlap_ge2"],
                         **{f"all_met_k{k}": m["all_met"][str(k)] for k in KS},
                         "release_respected": m["release_respected"], "sim_wall": r["wall"]})
        _write_csv(f"{c['name']}_runs.csv", rows)
        d = per["default"]
        A = [an[(c["name"], "default", sd)] for sd in seeds]
        smax_def = _mean([m["S_max"] for m in d])
        prio = {v: _mean([m["S_max"] for m in ms]) for v, ms in per.items() if v != "default"}
        best_v = min(prio, key=prio.get)
        cls = collections.Counter()
        for a in A:
            for k, v in a["wait_share_by_class"].items():
                cls[k] += v / len(A)
        slack_rows = [a["slack"] for a in A]
        slack_range = {str(k): _mean([max([row[str(k)]["range"] for row in sl if row[str(k)]["range"] is not None]
                                          or [0.0]) for sl in slack_rows]) for k in KS}
        slack_cv = {str(k): _mean([max([row[str(k)]["cv"] for row in sl if row[str(k)]["cv"] is not None] or [0.0])
                                   for sl in slack_rows]) for k in KS}
        summ = {
            "config": c["name"], "composition": c["composition"], "n_streams": c["n_streams"],
            "streams": c["streams"], "tasks": c.get("tasks") or [STREAMS.get(s, s) for s in c["streams"]], "stagger": c["stagger"],
            "T": c["T"], "Tmin": c["Tmin"], "T_ratio": max(c["T"].values()) / min(c["T"].values()),
            "hardware": c.get("hardware", "nofault"), "faults": c.get("faults"), "chunking": c.get("chunking", "none"), "backend_seeds": seeds,
            "release_mean": {i: _mean([m["release"][i] for m in d]) for i in d[0]["release"]},
            "latency_mean": {i: _mean([m["latency"][i] for m in d]) for i in d[0]["latency"]},
            "stretch_mean": {i: _mean([m["stretch"][i] for m in d]) for i in d[0]["stretch"]},
            "S_max_mean": smax_def, "S_max_seeds": [m["S_max"] for m in d],
            "makespan_mean": _mean([m["makespan"] for m in d]), "beat_mean": _mean([m["beat"] for m in d]),
            "overlap_ge2_mean": _mean([m["overlap_ge2"] for m in d]),
            "pair_overlap_mean": {k: _mean([m["pair_overlap"][k] for m in d]) for k in d[0]["pair_overlap"]},
            "all_met_share": {str(k): _mean([float(m["all_met"][str(k)]) for m in d]) for k in KS},
            "critical_wait_over_a_mean": _mean([a["critical_wait_over_a"] for a in A]),
            "wait_share_by_class_mean": dict(cls),
            "cross_stream_share_mean": _mean([a["cross_stream_share"] for a in A]),
            "cross_stream_share_seeds": [a["cross_stream_share"] for a in A],
            "smax_bound_reduction_mean": _mean([a["smax_bound"]["S_reduction_top5_max"] for a in A]),
            "smax_bound_reduction_ub_mean": _mean([a["smax_bound"]["S_reduction_ub_top5_max"] for a in A]),
            "Lmax_current_mean": {str(k): _mean([a["smax_bound"]["Lmax_current"][str(k)] for a in A]) for k in KS},
            "Lmax_best_top5_mean": {str(k): _mean([a["smax_bound"]["Lmax_best_top5_min"][str(k)] for a in A])
                                    for k in KS},
            "makespan_reorder_gain_mean": _mean([a["makespan_bound"]["reorder_gain_over_a_top5_max"] for a in A]),
            "makespan_reorder_gain_ub_mean": _mean([a["makespan_bound"]["reorder_gain_ub_over_a_top5_max"] for a in A]),
            "makespan_delete_gain_mean": _mean([a["makespan_bound"]["delete_gain_over_a_top5_max"] for a in A]),
            "slack_range_max_mean": slack_range, "slack_cv_max_mean": slack_cv,
            "prio_S_max_mean": prio, "prio_best_order": best_v, "prio_best_over_default": prio[best_v] / smax_def,
            "per_seed_analysis": A, "smax_models_match": all(a["smax_bound"]["all_models_match"] for a in A),
            "prio_detail": prio_detail(rows, c["n_streams"]),
            "artifact_commit": _artifact_commit(), "template_python_seed": 123,
            "release_impl": "(b) in-memory backend copy e4_backend.event_driver_e4", "command": sys.argv,
            "assumption_no_sram_capacity_check": True}
        check3 = None
        if c["stagger"] == "serial":
            lat_err = {i: abs(summ["latency_mean"][i] - c["T"][i]) / c["T"][i] for i in c["T"]}
            check3 = {"latency_rel_diff": lat_err, "cross_stream_share": summ["cross_stream_share_seeds"],
                      "latency_seeds": {i: [m["latency"][i] for m in d] for i in c["T"]},
                      "within_1pct": all(v <= TOL for v in lat_err.values()),
                      "within_noise_stop": all(v <= NOISE_STOP for v in lat_err.values()),
                      # user decision 2026-11-02: only cross-stream blocking = 0 is a stop condition; latency
                      # differences are recorded (the serial reference is analytic, see e4_serialref.py)
                      "pass": all(x == 0 for x in summ["cross_stream_share_seeds"])}
            summ["check3"] = check3
        _write_json(f"{c['name']}_summary.json", summ)
    return summ, check2_all, check3, cls.get("compute", 0.0)


def supplement_a1():
    """User decision 2026-11-01: add T_ratio and the priority detail to the existing a1 summaries (from their
    runs.csv; no re-simulation), then rebuild e4_table.csv from every a1 summary."""
    import glob
    for p in sorted(glob.glob(os.path.join(OUT, "a1_2s_*_summary.json"))):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        with open(p.replace("_summary.json", "_runs.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        s["T_ratio"] = max(s["T"].values()) / min(s["T"].values())
        s["prio_detail"] = prio_detail(rows, s["n_streams"])
        s.setdefault("backend_seeds", SEEDS)
        _write_json(os.path.basename(p), s)
    rebuild_table()


def rebuild_table():
    import glob
    table = []
    for p in sorted(glob.glob(os.path.join(OUT, "a*_summary.json"))):
        with open(p, encoding="utf-8") as f:
            table.append(_table_row(json.load(f)))
    _write_csv("e4_table.csv", table)
    return table


def redo_a1(names, n_seeds, jobs):
    """Re-run a1 configs with seeds 1..n_seeds and the step-2 T_i (user decision: Delta = 1T gets 6 seeds)."""
    with open(os.path.join(OUT, "step2_checks.json"), encoding="utf-8") as f:
        T = json.load(f)["T"]
    configs = a1_configs(T, only=set(names), d1_seeds=list(range(1, n_seeds + 1)))
    for c in configs:
        c["seeds"] = list(range(1, n_seeds + 1))
    t0 = time.time()
    sims = simulate_configs(configs, jobs)
    print(f"[e4] simulations done ({time.time() - t0:.0f} s)", flush=True)
    tasks = [((c["name"], "default", r["seed"]), r["cap"], c["T"]) for c in configs for r in sims[c["name"]]["runs"]
             if r["variant"] == "default"]
    with ProcessPoolExecutor(jobs) as ex:
        futs = {k: ex.submit(analyse_cap, cap, Tm) for k, cap, Tm in tasks}
        an = {k: f.result() for k, f in futs.items()}
    bad5 = {str(k): v.get("error") or v["checks"] for k, v in an.items()
            if not v["ok"] or not all(v["checks"].values()) or v["release_in_segs"]}
    if bad5:
        raise E4Error(f"self-check 5 failed on re-run: {bad5}")
    c2 = []
    for c in configs:
        summ, f2, _, _ = summarize_config(c, sims[c["name"]]["runs"], an)
        c2 += f2
        print(f"[e4] {c['name']}: S_max {summ['S_max_mean']:.4f} seeds {[round(x, 3) for x in summ['S_max_seeds']]} "
              f"prio {summ['prio_detail']['best_over_default']}", flush=True)
    if not all(c2):
        raise E4Error("self-check 2 failed on re-run")
    rebuild_table()
    print(f"[e4] redo done ({time.time() - t0:.0f} s), check2 ok, check5 ok on {len(an)} graphs", flush=True)


def _write_json(name, obj):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _write_csv(name, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture")
    p.add_argument("--streams", nargs="+", default=list(STREAMS))
    p.add_argument("--jobs", type=int, default=4)
    p = sub.add_parser("simulate")
    p.add_argument("--spec", required=True)
    p = sub.add_parser("step2")
    p.add_argument("--jobs", type=int, default=12)
    p = sub.add_parser("supplement")
    p = sub.add_parser("redo")
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    if args.cmd == "supplement":
        supplement_a1()
        return
    if args.cmd == "redo":
        redo_a1(args.configs, args.seeds, args.jobs)
        return
    if args.cmd == "capture":
        with ThreadPoolExecutor(args.jobs) as ex:
            infos = list(ex.map(capture, args.streams))
        facts = {}
        for s in args.streams:
            _artifact_path()
            facts[s] = tag_facts(pickle.loads(load_stream(s)["events_blob"]))
        _write_json("capture_info.json", {"streams": infos, "tag_facts": facts, "command": sys.argv,
                                          "artifact_commit": _artifact_commit(), "python": sys.executable})
        print("[e4] tag facts " + json.dumps(facts), flush=True)
    elif args.cmd == "simulate":
        simulate_child(args.spec)
    else:
        step2(args.jobs)


if __name__ == "__main__":
    try:
        main()
    except E4Error as exc:
        print(f"[e4] STOP: {exc}", flush=True)
        sys.exit(2)
