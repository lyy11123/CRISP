"""E3: post-fault warm-start SA, migration volume and lower-bound tightness on BusyBarn.

Every run is `python <task>.py` from the E3 artifact work copy in a child process, with
_inject/sitecustomize.py on PYTHONPATH (E2's approach; BusyBarn's results depend on object addresses, so
nothing is run in-process).

capture  The template runs unchanged; at its 2nd (final) event_driver call the final mapping is read from
         the template's globals: beha_dict[t].location for every behaviour and
         data_dict[d].generated_split_location for every data split (weights: the DDR port node they sit on;
         online data: producer nodes; the input activation (0, 0): every DDR port). A second process
         re-simulates the same final state with backend seeds (deepcopy per seed), as E2 does.
inject   read_cfg.cfg_to_dict adds the [failures] of the spec; Pre_Mapping.initialized_mapping runs as usual
         and then has every location overwritten from the spec's mapping, so the template's own build_event
         rebuilds computation and communication events and Dijkstra routes on the (faulty) platform;
         Stream_Mapping.stream_mapping runs the original StreamOptimization with the spec's steps and t_max
         (steps 0 = no move: R0); the 1st backend call (initial mapping, result unused by the template) is
         skipped; the 2nd is evaluated once per backend seed on a deepcopy and finally once with the
         template's own random state.

Usage (one task per process):
  python e3_warmstart.py step1 --task ffn --seeds 1 2 3
"""
import argparse
import collections
import copy
import gzip
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E2_DIR = os.path.join(HERE, "..", "e2_attribution")
COMMIT_FILE = os.path.join(HERE, "..", "third_party", "isca2026-busybarn-artifact.COMMIT")
WORK = os.path.join(HERE, "work", "artifact")
PY_DIR = os.path.join(WORK, "models", "intra_mapping_gpt_decode", "py")
OUT = os.path.join(HERE, "e3_out")
TASKS = {
    "ffn": "ffn_sq2048_greedy0_lossratio6_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0",
    "mha": "mha_sq2048_greedy0_lossratio4_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0",
    # E3 extension (user, 2026-10-31): 1x4 die group; winners of the 14 variants in E2's sweeps
    "ffn14": "ffn_sq2048_greedy1_lossratio4_ch1x4_bw256_co4x4_bw256_t128x64_failpattern0",
    "mha14": "mha_sq2048_greedy0_lossratio6_ch1x4_bw256_co4x4_bw256_t128x64_failpattern0",
    # E3 extension (user, 2026-10-31): Gemini baseline (one run, XY routing, no Dijkstra)
    "ffng": "ffn_sq2048_gemini_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0",
    "mhag": "mha_sq2048_gemini_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0",
}
# E2 (third_party copy, same machine): template-RNG result and backend seeds 1, 2, 3 on the final state
E2_REF = {"ffn": {"template": 203603, "seeds": {1: 205432, 2: 203603, 3: 205432}},
          "mha": {"template": 20153, "seeds": {1: 20892, 2: 19697, 3: 20836}},
          "ffn14": {"template": 233206, "seeds": {}}, "mha14": {"template": 24279, "seeds": {}},
          "ffng": {"template": 313695, "seeds": {}}, "mhag": {"template": 31495, "seeds": {}}}
TEMPLATE_SEED = 123
CHECK1_TOL = 0.01


class E3Error(RuntimeError):
    pass


def _e2():
    sys.path.insert(0, E2_DIR)
    import e2_attribution
    sys.path.pop(0)
    return e2_attribution


def _plain(x):
    if isinstance(x, tuple):
        return tuple(_plain(v) for v in x)
    if isinstance(x, (list, set)):
        return type(x)(_plain(v) for v in x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _artifact_commit():
    try:
        with open(COMMIT_FILE, encoding="utf-8-sig") as f:
            return f.read().strip().split()[0]
    except OSError:
        return None


def _rng_digest():
    return hashlib.sha1(repr(random.getstate()).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- read state inside the child
def split_bytes(split, data_type):
    from data_notation import type_bytes  # the artifact's own table (on sys.path inside the child)
    return int(np.prod([end - start + 1 for (start, end) in split]) * type_bytes[data_type])


def mapping_of(g):
    """Final mapping from the template's globals."""
    bd, dd = g["beha_dict"], g["data_dict"]
    m = {"beha_loc": {_plain(t): _plain(b.location) for t, b in bd.items()},
         "beha_dev": {_plain(t): b.device for t, b in bd.items()},
         "data_loc": {_plain(d): {_plain(sp): _plain(list(locs)) for sp, locs in t.generated_split_location.items()}
                      for d, t in dd.items()},
         # weights the SA may relocate (StreamOptimization.offlinedata_split_dict) and their sizes
         "weight_bytes": {(_plain(d), _plain(sp)): split_bytes(sp, t.data_type)
                          for d, t in dd.items() if d[0] == 1 for sp in t.used_splitted_tag_dict}}
    return m


def route_key(target, nbytes, consumers, producers):
    """Identity of a communication across processes (comm tags are allocated in build order, behaviour tags
    are stable). The source is NOT part of the key: the original SA's update_offlinedata moves a weight by
    rewriting the source_location of the communications listed in used_splitted_tag_dict and never updates
    data_dict's generated_split_location, so after SA the source actually read from lives only on the event
    (and stale, recycled comm tags in used_splitted_tag_dict can even give an online-data transfer a DDR
    source). Communications sharing a key have the same target, size, producers and consumers, so
    exchanging their (source, route) pairs changes nothing."""
    tg = ("set",) + tuple(sorted(map(_plain, target))) if isinstance(target, set) else _plain(target)
    return (tg, int(nbytes), tuple(sorted(map(str, consumers))), tuple(sorted(map(str, producers))))


def routes_of(events_dict, pre):
    """key -> list of (source, routing tree [(root, [leaf, ...]), ...]) of every communication. `pre` holds
    the dependency/issue sets snapshotted before simulation (the backend consumes them, never paths)."""
    out = {}
    for t, e in events_dict.items():
        if e.event_type != "communication":
            continue
        p = pre[_plain(t)]
        k = route_key(e.target_location, e.comm_bytes, p["issue"], p["deps"])
        out.setdefault(k, []).append((_plain(e.source_location),
                                      [(_plain(r), [_plain(l) for l in leaves]) for r, leaves in e.paths.items()],
                                      _plain(t)))
    return out


def event_layout_of(events_dict):
    """Insertion order of event_dict (the backend builds its first ready list by iterating it) and every
    computation's comm-id allocator state (comm tags are (1,) + owner[1:] + (id,))."""
    alloc = {_plain(t): (sorted(e.commid_allocator.used_ids), sorted(e.commid_allocator.recycled),
                         int(e.commid_allocator.next_id))
             for t, e in events_dict.items() if e.event_type == "computation"}
    return {"order": [_plain(t) for t in events_dict], "allocators": alloc}


def relabel_events(event_dict, data_dict, rename, layout, exact_alloc):
    """Rename communications (old tag -> captured tag), put event_dict in the captured insertion order and
    set the comm-id allocators: exact_alloc -> the captured state (plus any id still in use); otherwise the
    ids now in use, with every other captured id below next_id left recyclable."""
    import heapq
    rename = {o: n for o, n in rename.items() if o != n}
    if len(set(rename.values())) != len(rename):
        raise E3Error("comm relabelling maps two communications to one tag")
    keep = set(event_dict) - set(rename)
    # a communication that was not restored (rerouted) may hold a tag a restored one takes back: fresh id
    taken = set(rename.values()) | keep
    for k in sorted(set(rename.values()) & keep):
        owner_ids = [t[-1] for t in taken if t[0] == 1 and t[:-1] == k[:-1]]
        new = k[:-1] + (max(owner_ids) + 1,)
        rename[k] = new
        taken.add(new)
    keep = set(event_dict) - set(rename)
    if set(rename.values()) & keep:
        raise E3Error("comm relabelling would collide with an existing tag")
    objs = {o: event_dict.pop(o) for o in rename}
    for o, e in objs.items():
        e.event_tag = rename[o]
        event_dict[rename[o]] = e
    for e in event_dict.values():
        e.dependency_set = {rename.get(x, x) for x in e.dependency_set}
        e.issue_set = {rename.get(x, x) for x in e.issue_set}
    for d in data_dict.values():
        for sp, s in d.used_splitted_tag_dict.items():
            d.used_splitted_tag_dict[sp] = {rename.get(x, x) for x in s}
    items = dict(event_dict)
    event_dict.clear()
    for t in layout["order"]:
        if t in items:
            event_dict[t] = items.pop(t)
    event_dict.update(items)
    in_use = collections.defaultdict(set)
    for t, e in event_dict.items():
        if e.event_type == "communication":
            in_use[(0,) + tuple(t[1:-1])].add(t[-1])
    for t, e in event_dict.items():
        if e.event_type != "computation":
            continue
        used0, rec0, nxt0 = layout["allocators"].get(_plain(t), ([], [], 0))
        cur = in_use.get(_plain(t), set())
        a = e.commid_allocator
        if exact_alloc:
            a.used_ids = set(used0) | cur
            a.recycled = sorted(set(rec0) - cur)
        else:
            a.used_ids = set(cur)
            a.recycled = sorted((set(rec0) | set(used0)) - cur)
        heapq.heapify(a.recycled)
        a.next_id = max([nxt0] + [i + 1 for i in cur])
    return len(rename)


def _route_links(paths):
    return [(r, l) for r, leaves in paths.items() for l in leaves]


def _add_loads(hp, e, sign):
    """Platform link-load counters of event e's current tree, as record_dijkstra_multicast_path keeps them."""
    for link in _route_links(e.paths):
        hp.link_loads_dict[link] += sign * hp.current_link_loads(e.comm_bytes, hp.links_dict[link])
        hp.link_loads_count[link] += sign


def restore_routes(event_dict, hp, tags, routes, alpha, beta, gamma, banned_sources=frozenset(), src_replace=None,
                   xy=False):
    """Give each communication in `tags` its captured (source, route) when one with the same key exists and
    the source node and every link of the route exist on this platform; route the rest with the artifact's
    Dijkstra from their own source against the link loads of the restored routes. The platform's link-load
    counters end up equal to what the current routes imply (on top of whatever they held for communications
    outside `tags`). Returns the change of the hops / communication-distance totals build_event reports."""
    from net import TupleIdx
    # a source must still exist and still hold the data: a PE-failed node keeps routing but its producers moved
    alive = lambda n: tuple(n) in hp.nodes_set and tuple(n) not in banned_sources
    src_replace = src_replace or {}
    eff = lambda n: src_replace.get(tuple(n), tuple(n))  # a failed DDR port's reads come from its replacement
    pool = {k: list(v) for k, v in routes.items()}
    # entries by key without the target: a consumer moved by the repair keeps its pre-fault source
    pool_nt = collections.defaultdict(list)
    for k, v in routes.items():
        pool_nt[k[1:]] += [(k, x) for x in v]
    keep, redo = [], []
    for t in tags:
        e = event_dict[t]
        k = route_key(e.target_location, e.comm_bytes, e.issue_set, e.dependency_set)
        cand = pool.get(k) or []
        if not cand:  # target changed: borrow the source of an unused entry with the same size/consumer/producers
            for i, (k0, x) in enumerate(pool_nt.get(k[1:], [])):
                if x in pool.get(k0, []) and alive(eff(x[0])) and eff(x[0]) != _plain(e.target_location):
                    pool[k0].remove(x)
                    pool_nt[k[1:]].pop(i)
                    redo.append((t, (eff(x[0]), None, None)))
                    break
            else:
                redo.append((t, None))
            continue
        got = None
        for i, (src, c, tag0) in enumerate(cand):  # prefer an entry whose whole route survived
            if alive(src) and \
                    all((tuple(r), tuple(l)) in hp.links_dict for r, leaves in c for l in leaves):
                got = (tuple(src), c, tuple(tag0))
                cand.pop(i)
                break
        if got is None:  # route broken: keep the pre-fault source (if alive) and tag, re-route only the path
            for i, (src, c, tag0) in enumerate(cand):
                if alive(eff(src)):
                    got = (eff(src), None, tuple(tag0))
                    cand.pop(i)
                    break
        (keep if got is not None and got[1] is not None else redo).append((t, got))
    # Dijkstra keeps platform link-load counters (and routes by them); XY neither records nor reads them
    upd = (lambda e, s: None) if xy else (lambda e, s: _add_loads(hp, e, s))
    for t in tags:
        upd(event_dict[t], -1)
    d_hops = d_dist = 0
    moved_src = 0
    for t, got in keep + [x for x in redo if x[1] is not None]:
        e = event_dict[t]
        src = got[0]
        if src != _plain(e.source_location):
            moved_src += 1
            if not isinstance(e.target_location, set):  # unicast: hops / distance fields depend on the source
                tg = e.target_location
                d_hops -= e.hops * e.comm_bytes
                d_dist -= e.communication_distances
                e.source_location = src
                # as build_event fills them for each routing mode
                hop_d = hp.node_to_node_manhattan_hops_dict if xy else hp.node_to_node_hop_dict
                fun_d = hp.node_to_node_manhattan_distance_function_dict if xy else hp.node_to_node_distance_function_dict
                if hop_d[src][tg] is None:
                    raise E3Error(f"no {'XY' if xy else 'Dijkstra'} path from {src} to {tg}")
                e.hops = hop_d[src][tg]
                e.communication_distances = fun_d[src][tg](e.comm_bytes)
                d_hops += e.hops * e.comm_bytes
                d_dist += e.communication_distances
            else:
                e.source_location = src
    for t, (src, tree, _) in keep:
        e = event_dict[t]
        # as in the Dijkstra trees: source key plain, other nodes TupleIdx
        e.paths = {tuple(r) if tuple(r) == src else TupleIdx(tuple(r), None): [TupleIdx(tuple(l), None) for l in leaves]
                   for r, leaves in tree}
        e.path_list = e.get_paths(e.paths)
        upd(e, +1)
    for t, _ in redo:
        e = event_dict[t]
        tgt = e.target_location if isinstance(e.target_location, set) else {e.target_location}
        if xy:  # deterministic XY / XY-YX tables, as build_event and add_broadcast use without Dijkstra
            paths = hp.xy_paths[e.source_location][e.target_location] if not isinstance(e.target_location, set) \
                else hp.xy_multicast_path(source_node_idx=e.source_location, target_nodes_set=tgt)
            if paths is None:
                raise E3Error(f"no XY path for {t} from {e.source_location} to {e.target_location}")
            e.paths = paths
        else:
            e.paths = hp.record_dijkstra_multicast_path(comm_pairs=[[t, e.source_location, tgt, e.comm_bytes]],
                                                        alpha=alpha, beta=beta, gamma=gamma)[2][t]
        e.path_list = e.get_paths(e.paths)
    neg = [k for k, v in hp.link_loads_count.items() if v < 0]
    if neg:
        raise E3Error(f"negative link-load count after restoring routes on {neg[:3]}")
    return {"restored": len(keep), "rerouted": len(redo), "source_restored": moved_src,
            "rerouted_without_any_match": sum(1 for _, g in redo if g is None),
            "rename": {t: got[2] for t, got in keep + redo if got is not None and got[2] is not None},
            "rerouted_tags": [_plain(t) for t, _ in redo], "d_hops": d_hops, "d_dist": d_dist}


def platform_info(hp):
    def cfg(kind):
        l = next(l for l in hp.links_dict.values() if l.link_type == kind)
        return {"bandwidth": l.bandwidth, "latency": l.latency, "timeunit": l.timeunit}
    seen, stack = set(), [next(iter(hp.nodes_set))]
    while stack:  # every node reachable from every other one (links are directed: check both ways)
        u = stack.pop()
        if u not in seen:
            seen.add(u)
            stack += [v for v in hp.available_neighbors.get(u, ()) if v not in seen]
    rev = collections.defaultdict(set)
    for u, vs in hp.available_neighbors.items():
        for v in vs:
            rev[v].add(u)
    seen_r, stack = set(), [next(iter(hp.nodes_set))]
    while stack:
        u = stack.pop()
        if u not in seen_r:
            seen_r.add(u)
            stack += [v for v in rev.get(u, ()) if v not in seen_r]
    return {"n_tensorcore": len(hp.modules_dict["tensorcore"]), "n_vectorunit": len(hp.modules_dict["vectorunit"]),
            "ddr_nodes": sorted(_plain(n) for n in hp.ddr_set), "n_nodes": len(hp.nodes_set),
            "strongly_connected": seen == set(hp.nodes_set) == seen_r,
            "links": sorted(_plain(k) for k in hp.links_dict),
            "ddr2co": cfg("ddr2co"), "co2ddr": cfg("co2ddr"),
            "failed_nodes": sorted(_plain(n) for n in hp.failed_nodes),
            "failed_links": sorted(_plain(l) for l in hp.failed_links)}


def analytic_workload(g, hp):
    """Mapping-independent workload read from beha/data dicts (no events): per device type the sum of
    working times, every DDR read (one per consumer behaviour and needed offline split: weights d[0]==1 and
    the input activation (0, 0) live only on DDR nodes, so they are never local), every broadcast write."""
    bd, dd = g["beha_dict"], g["data_dict"]
    comp = {"tensorcore": 0, "vectorunit": 0}
    for b in bd.values():
        dev = next(iter(hp.modules_dict[b.device].values()))  # cores are homogeneous
        comp[b.device] += int(dev.working_time(source_datashape=b.beha_datashape, beha_type=b.beha_type,
                                               frequency=hp.frequency))
    reads = []
    for b in bd.values():
        for d, splits in b.needed_data_split_dict.items():
            if d[0] == 1 or d == (0, 0):
                reads += [split_bytes(sp, dd[d].data_type) for sp in splits]
    writes = [split_bytes(sp, dd[d].data_type) for d in g.get("broadcast_datatags", [])
              for _, sps in dd[d].generated_tag_splitted_dict.items() for sp in sps]
    return {"comp_sum": comp, "ddr_read_bytes": sorted(reads), "ddr_broadcast_bytes": sorted(writes)}


# ---------------------------------------------------------------- lower bounds
def _link_time(nbytes, c):
    """Same formula as src/platform/device/link/*.py working_time."""
    return int(math.ceil(math.ceil(nbytes / c["bandwidth"]) * c["timeunit"]) + c["latency"])


def lower_bounds(cap, plat, analytic):
    """LB_comp, LB_mem, LB_dep. DDR traffic comes from the mapping-free workload (analytic_workload): every
    consumer x needed offline split is one DDR read, every broadcast split reaches every port. A simulated
    graph can carry MORE DDR reads than that (see route_key: SA can give an online-data transfer a DDR
    source), which only makes the bound weaker, never invalid. Computation sums and the dependency chain
    come from the simulated graph and are compared with the analytic sums by the caller."""
    ev = cap["events"]
    ddr = set(map(tuple, plat["ddr_nodes"]))
    comp = {"tensorcore": 0, "vectorunit": 0}
    for e in ev.values():
        if e["type"] == "computation":
            comp[e["device"]] += e["f"] - e["s"]
    lb_comp_by = {k: math.ceil(v / plat["n_" + k]) for k, v in comp.items()}
    lb_comp = max(lb_comp_by.values())

    ev_reads, ev_writes = [], {n: [] for n in ddr}
    for e in ev.values():
        if e["type"] != "communication":
            continue
        if e["source"] in ddr:
            ev_reads.append(e["bytes"])
        tg = e["target"] if isinstance(e["target"], set) else {e["target"]}
        for n in tg & ddr:
            ev_writes[n].append(e["bytes"])
    reads = list(analytic["ddr_read_bytes"])
    writes_per_port = {n: list(analytic["ddr_broadcast_bytes"]) for n in ddr}
    P = len(ddr)
    rc, wc = plat["ddr2co"], plat["co2ddr"]
    # handoff definition: total bytes / (ports x bandwidth), converted with the link formula. Reads and
    # writes use different directed links (ddr2co / co2ddr), so they are bounded separately: reads are
    # spread over the P ports, while a broadcast write reaches every port, so each port's co2ddr link
    # carries all of it.
    lb_read = _link_time(sum(reads) / P, rc) if reads else 0
    lb_write = max((_link_time(sum(w), wc) for w in writes_per_port.values() if w), default=0)
    # tighter, still mapping-free: a link holds a transfer for its whole working_time (latency included,
    # event_driver.py), so the busiest ddr2co link holds at least the mean total occupancy
    lb_read_occ = math.ceil(sum(_link_time(b, rc) for b in reads) / P) if reads else 0
    lb_write_occ = max((sum(_link_time(b, wc) for b in w) for w in writes_per_port.values() if w), default=0)

    # longest path with every communication at zero and computations at their nominal (simulated) duration
    memo = {}

    def L(t):
        if t in memo:
            return memo[t]
        stack = [(t, False)]
        while stack:
            u, done = stack.pop()
            if u in memo:
                continue
            deps = ev[u]["deps"]
            if done:
                w = ev[u]["f"] - ev[u]["s"] if ev[u]["type"] == "computation" else 0
                memo[u] = w + max((memo[d] for d in deps), default=0)
            else:
                stack.append((u, True))
                stack += [(d, False) for d in deps if d not in memo]
        return memo[t]

    lb_dep = max(L(t) for t in ev)
    lb_mem, lb_mem_occ = max(lb_read, lb_write), max(lb_read_occ, lb_write_occ)
    comps = {"LB_comp": lb_comp, "LB_mem": lb_mem, "LB_dep": lb_dep}
    lb = max(comps.values())
    comps_occ = {"LB_comp": lb_comp, "LB_mem_occ": lb_mem_occ, "LB_dep": lb_dep}
    return {"LB": lb, "dominant": max(comps, key=comps.get), **comps,
            "LB_comp_by_device": lb_comp_by, "comp_sum": comp,
            "LB_mem_read": lb_read, "LB_mem_write": lb_write, "LB_mem_read_occ": lb_read_occ,
            "LB_mem_write_occ": lb_write_occ, "LB_mem_occ": lb_mem_occ,
            "LB_occ": max(comps_occ.values()), "dominant_occ": max(comps_occ, key=comps_occ.get),
            "n_ddr_ports": P, "ddr_read_bytes": sorted(reads),
            "ddr_broadcast_bytes_per_port": {str(n): sorted(w) for n, w in writes_per_port.items()},
            "graph_ddr_read_bytes": sorted(ev_reads),
            "graph_ddr_broadcast_bytes_per_port": {str(n): sorted(w) for n, w in ev_writes.items()}}


def check_comp_nominal(cap, events_dict, hp):
    """Every computation's simulated duration equals working_time on its device."""
    bad = []
    for t, e in cap["events"].items():
        if e["type"] == "computation":
            o = events_dict[t]
            wt = int(hp.modules_dict[o.comp_device][o.comp_location].working_time(
                source_datashape=o.comp_datashape, beha_type=o.comp_type, frequency=hp.frequency))
            if wt != e["f"] - e["s"]:
                bad.append((t, wt, e["f"] - e["s"]))
    return bad


# ---------------------------------------------------------------- capture mode (child)
def capture_final(orig, events_dict, hp, call_no, g):
    e2 = _e2()
    seeds = [int(s) for s in os.environ.get("E3_SEEDS", "").split()]
    sink = {"calls": call_no, "reseeded": []}
    pre = e2._snapshot_pre(events_dict)
    canon = sorted((str(t), sorted(map(str, p["deps"])), sorted(map(str, p["issue"])), str(p["path_list"]))
                   for t, p in pre.items())
    sink["pre_digest"] = hashlib.sha1(repr(canon).encode()).hexdigest()
    digest = _rng_digest()
    reruns = copy.deepcopy((events_dict, hp)) if seeds else None
    t0 = time.time()
    out = orig(events_dict=events_dict, hardware_platform=hp)
    sink["eval_wall"] = time.time() - t0
    sink["last"] = e2._extract(events_dict, hp, pre, out, digest)
    sink["comp_nominal_mismatch"] = check_comp_nominal(sink["last"], events_dict, hp)
    sink["mapping"] = mapping_of(g)
    sink["mapping"]["routes"] = routes_of(events_dict, pre)
    sink["mapping"]["layout"] = event_layout_of(events_dict)
    sink["platform"] = platform_info(hp)
    sink["analytic"] = analytic_workload(g, hp)
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
    with open(os.environ["E3_SINK"], "wb") as f:
        pickle.dump(sink, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


# ---------------------------------------------------------------- inject mode (child)
_SPEC = {}


def _spec():
    if not _SPEC:
        with open(os.environ["E3_SPEC"], "rb") as f:
            _SPEC.update(pickle.load(f))
        _SPEC["_log"] = {"sa_wall": None, "inject": None}
    return _SPEC


def apply_mapping(beha_dict, data_dict, m):
    """Overwrite every location with the mapping m; return the number of fields written."""
    if set(map(_plain, beha_dict)) != set(m["beha_loc"]):
        raise E3Error("behaviour set of the task differs from the mapping's")
    n = 0
    for t, b in beha_dict.items():
        if b.device != m["beha_dev"][_plain(t)]:
            raise E3Error(f"device type of {t} differs: {b.device} vs {m['beha_dev'][_plain(t)]}")
        b.location = tuple(m["beha_loc"][_plain(t)])
        n += 1
    for d, splits in m["data_loc"].items():
        gsl = data_dict[d].generated_split_location
        if set(map(_plain, gsl)) != set(splits):
            raise E3Error(f"split set of data {d} differs from the mapping's")
        for sp, locs in splits.items():
            gsl[sp] = [tuple(x) for x in locs]
            n += 1
    # consistency: every produced split lists its producer's node
    for t, b in beha_dict.items():
        for d, sps in b.produced_data_split_dict.items():
            for sp in sps:
                if b.location[:-1] not in data_dict[d].generated_split_location[sp]:
                    raise E3Error(f"mapping inconsistent: producer {t} at {b.location} not in locations of {d}{sp}")
    return n


CORNER_CORES = {0, 3, 12, 15}  # each feeds two DDR ports; WAMIS_HD cannot fail them


def _grid_xy(node, hp):
    """Global (y, x) of a core node on the 2-level mesh, also for a node that is not on the platform."""
    die, core = node
    n, m = divmod(die, hp.l2_width)
    j, i = divmod(core, hp.l1_width)
    return n * hp.l1_height + j, m * hp.l1_width + i


def ddr_replacement(hp, failed_nodes):
    """Failed DDR port -> the live DDR port fewest hops from its corner core (the sibling port of the same
    corner when alive), ties -> smallest id. DDR port k of a die (node (die, 16 + k)) hangs off corner k // 2,
    in WAMIS_HD.init_mem's order: top-left, top-right, bottom-left, bottom-right."""
    w, h = hp.l1_width, hp.l1_height
    corner_core = [0, 0, w - 1, w - 1, (h - 1) * w, (h - 1) * w, h * w - 1, h * w - 1]
    out = {}
    for n in map(tuple, failed_nodes):
        if n[1] < h * w:
            continue  # a core, not a DDR port
        corner = (n[0], corner_core[n[1] - h * w])
        out[n] = min(sorted(hp.ddr_set), key=lambda d: (hp.node_to_node_hop_dict[corner][d], d))
    return out


def relocate_ddr_data(data_dict, repl):
    """Data on a failed DDR port: drop it where the split also lives elsewhere (the input activation sits
    on every port), otherwise move it to the replacement port."""
    moved = []
    for d, t in data_dict.items():
        for sp, locs in t.generated_split_location.items():
            dead = [n for n in locs if tuple(n) in repl]
            if not dead:
                continue
            rest = [n for n in locs if tuple(n) not in repl]
            new = rest if rest else [repl[tuple(dead[0])]]
            t.generated_split_location[sp] = new
            moved.append((_plain(d), _plain(sp), [_plain(x) for x in dead], [_plain(x) for x in new]))
    return moved


def relocate_failed(beha_dict, data_dict, hp, failed_nodes):
    """R0's repair for failed cores: every behaviour on a failed node moves to the nearest healthy device of
    its own type (Manhattan distance on the global core grid; all dies of the group are candidates), the
    least loaded one (sum of working times placed on it) among equally near ones, then the smallest id;
    behaviours are handled in tag order and loads are updated as they move. Each produced split's location
    list gets the new node in place of the failed one."""
    failed = set(map(tuple, failed_nodes))
    load = collections.Counter()
    wt = {}
    for t, b in beha_dict.items():
        dev = next(iter(hp.modules_dict[b.device].values()))
        wt[t] = int(dev.working_time(source_datashape=b.beha_datashape, beha_type=b.beha_type,
                                     frequency=hp.frequency))
        if tuple(b.location[:-1]) not in failed:
            load[(b.device, tuple(b.location))] += wt[t]
    moves = []
    for t in sorted(beha_dict, key=str):
        b = beha_dict[t]
        old = tuple(b.location)
        if old[:-1] not in failed:
            continue
        fy, fx = _grid_xy(old[:-1], hp)
        cands = sorted(hp.modules_dict[b.device])
        best = min(cands, key=lambda d: (abs(_grid_xy(d[:-1], hp)[0] - fy) + abs(_grid_xy(d[:-1], hp)[1] - fx),
                                         load[(b.device, tuple(d))], tuple(d)))
        b.location = tuple(best)
        load[(b.device, tuple(best))] += wt[t]
        for d, sps in b.produced_data_split_dict.items():
            for sp in sps:
                gsl = data_dict[d].generated_split_location[sp]
                if old[:-1] in gsl:
                    gsl.remove(old[:-1])
                gsl.append(tuple(best[:-1]))
        moves.append((_plain(t), old, tuple(best)))
    left = [(d, sp) for d, t in data_dict.items() for sp, locs in t.generated_split_location.items()
            if any(tuple(n) in failed for n in locs)]
    if left:
        raise E3Error(f"data still located on a failed node after relocation: {left[:3]}")
    return moves


def patch_module(name, module):
    spec = _spec()
    if name == "WAMIS_HD":
        # WAMIS_HD.xy_multicast_path: for a DDR endpoint it uses xy_pointopoint_path's result (line ~898)
        # before its own None check (line ~906), so a pair the fault-tolerant XY router gives up on raises
        # TypeError while building the platform. The non-DDR branch returns None in that case and
        # xy_manhattan_distance stores None ("no XY path"); do the same here. XY tables are only used without
        # Dijkstra routing or for a communication that reaches the backend with no path.
        cls = module.wamis_hdc
        orig_xy = cls.xy_multicast_path

        def xy_multicast_path(self, source_node_idx, target_nodes_set):
            try:
                return orig_xy(self, source_node_idx=source_node_idx, target_nodes_set=target_nodes_set)
            except TypeError as exc:
                if "NoneType" not in str(exc):
                    raise
                spec["_log"].setdefault("xy_none_ddr_pairs", 0)
                spec["_log"]["xy_none_ddr_pairs"] += 1
                return None

        cls.xy_multicast_path = xy_multicast_path

        # PE-only fault (not in BusyBarn's fault model, which deletes core and router together): the node stays
        # in the topology and keeps forwarding, but its tensor and vector units are removed from tensorcore_dict
        # / vectorunit_dict (modules_dict holds the same dicts), which is where initial mapping, SA candidates
        # and load accounting take compute locations from.
        orig_init = cls.__init__

        def __init__(self, platform_cfg):
            orig_init(self, platform_cfg=platform_cfg)
            pes = set(map(tuple, spec["faults"].get("failed_pes", [])))
            if pes:
                gone = [loc for d in (self.tensorcore_dict, self.vectorunit_dict) for loc in list(d)
                        if tuple(loc[:-1]) in pes]
                for d in (self.tensorcore_dict, self.vectorunit_dict):
                    for loc in list(d):
                        if tuple(loc[:-1]) in pes:
                            del d[loc]
                spec["_log"]["removed_pe_devices"] = [_plain(x) for x in gone]

        cls.__init__ = __init__
    elif name == "read_cfg":
        orig = module.cfg_to_dict

        def cfg_to_dict(cfg_path):
            cfg = orig(cfg_path)
            if spec["faults"]["failed_nodes"] or spec["faults"]["failed_links"]:
                cfg["failures"]["failed_nodes"] = [tuple(n) for n in spec["faults"]["failed_nodes"]]
                cfg["failures"]["failed_links"] = [(tuple(a), tuple(b)) for a, b in spec["faults"]["failed_links"]]
            return cfg

        module.cfg_to_dict = cfg_to_dict
    elif name == "Pre_Mapping":
        orig = module.initialized_mapping

        def initialized_mapping(*args, **kw):
            orig(*args, **kw)
            if spec["mapping"] is None:  # FULL: the template's own initial mapping on the faulty platform
                return
            spec["_log"]["inject"] = apply_mapping(kw["beha_dict"], kw["data_dict"], spec["mapping"])
            hp = kw["hardware_platform"]
            repl = ddr_replacement(hp, spec["faults"]["failed_nodes"])
            spec["_ddr_repl"] = repl
            if repl:
                spec["_log"]["ddr_replacement"] = {str(k): v for k, v in repl.items()}
                spec["_log"]["ddr_data_moved"] = relocate_ddr_data(kw["data_dict"], repl)
            dead = [n for n in spec["faults"]["failed_nodes"] if tuple(n) not in repl] + \
                list(spec["faults"].get("failed_pes", []))
            if dead:
                spec["_log"]["relocated"] = relocate_failed(kw["beha_dict"], kw["data_dict"], hp, dead)

        module.initialized_mapping = initialized_mapping
    elif name == "Stream_Mapping":
        orig = module.stream_mapping

        def stream_mapping(*args, **kw):
            if spec["steps"] is None:  # R0: no SA at all; seed as anneal() would, so later Dijkstra calls match W(0)
                random.seed(123)
                np.random.seed(123)
                spec["_log"]["sa_args"] = None
                spec["_log"]["sa_wall"] = 0.0
                return None
            if spec["steps"] != "template":  # FULL keeps the template's steps and t_max
                kw["steps"] = float(spec["steps"])
                if spec.get("t_max") is not None:
                    kw["t_max"] = spec["t_max"]
            spec["_log"]["sa_args"] = {k: kw[k] for k in ["steps", "t_max", "t_min", "loss_ratio"]}
            t0 = time.time()
            out = orig(*args, **kw)
            spec["_log"]["sa_wall"] = time.time() - t0
            return out

        module.stream_mapping = stream_mapping
    elif name == "add_communication" and spec.get("preserve_routes"):
        orig_build, orig_bcast = module.build_event, module.add_broadcast
        spec["_log"]["routes"] = []
        banned = frozenset(map(tuple, spec["faults"].get("failed_pes", [])))

        def build_event(*args, **kw):
            hops, dist, _, tc, vu = orig_build(*args, **kw)
            spec["_xy"] = not kw.get("dijkstra_routing")  # Gemini routes with fixed XY tables
            ev, hp = kw["event_dict"], kw["hardware_platform"]
            tags = [t for t, e in ev.items() if e.event_type == "communication"]
            st = restore_routes(ev, hp, tags, spec["mapping"]["routes"], kw["alpha"], kw["beta"], kw["gamma"],
                                banned, spec.get("_ddr_repl"), spec["_xy"])
            st["relabelled"] = relabel_events(ev, kw["data_dict"], st.pop("rename"), spec["mapping"]["layout"],
                                              exact_alloc=False)
            spec["_log"]["routes"].append({"where": "build_event", **st})
            hops, dist = hops + st["d_hops"], dist + st["d_dist"]
            # communication_loads_dict exactly as build_event accumulates it, for the routes now in place
            comm = {link: 0 for link in hp.links_set}
            tags = [t for t, e in ev.items() if e.event_type == "communication"]
            for t in tags:
                for link in _route_links(ev[t].paths):
                    comm[link] += np.ceil(ev[t].comm_bytes / hp.links_dict[link].bandwidth)
            return hops, dist, comm, tc, vu

        def add_broadcast(*args, **kw):
            ev, hp = kw["event_dict"], kw["hardware_platform"]
            before = set(ev)
            out = orig_bcast(*args, **kw)
            tags = [t for t in ev if t not in before]
            st = restore_routes(ev, hp, tags, spec["mapping"]["routes"], kw["alpha"], kw["beta"], kw["gamma"],
                                banned, spec.get("_ddr_repl"), not kw.get("dijkstra_routing"))
            st["relabelled"] = relabel_events(ev, kw["data_dict"], st.pop("rename"), spec["mapping"]["layout"],
                                              exact_alloc=True)
            spec["_log"]["routes"].append({"where": "add_broadcast", **st})
            return out

        module.build_event, module.add_broadcast = build_event, add_broadcast
    elif name == "event_driver":
        orig, calls = module.event_driver, [0]

        def event_driver(events_dict, hardware_platform):
            calls[0] += 1
            if calls[0] == 1:  # initial mapping: the template discards the result
                return (0, 0, 0)
            g = sys._getframe(1).f_globals
            return _inject_eval(orig, events_dict, hardware_platform, calls[0], g)

        module.event_driver = event_driver


def _inject_eval(orig, events_dict, hp, call_no, g):
    e2 = _e2()
    spec = _spec()
    pre = e2._snapshot_pre(events_dict)
    sink = {"calls": call_no, "log": spec["_log"], "mapping": mapping_of(g), "platform": platform_info(hp),
            "analytic": analytic_workload(g, hp), "runs": []}
    sink["mapping"]["routes"] = routes_of(events_dict, pre)
    state = random.getstate(), np.random.get_state()
    digest = _rng_digest()
    base = copy.deepcopy((events_dict, hp))
    for seed in spec["seeds"]:
        ev_k, hp_k = copy.deepcopy(base)
        random.seed(seed)
        t0 = time.time()
        out_k = orig(events_dict=ev_k, hardware_platform=hp_k)
        sink["runs"].append({"seed": seed, "wall": time.time() - t0,
                             "cap": e2._extract(ev_k, hp_k, pre, out_k, f"seed{seed}")})
    random.setstate(state[0])
    np.random.set_state(state[1])
    t0 = time.time()
    out = orig(events_dict=events_dict, hardware_platform=hp)
    cap = e2._extract(events_dict, hp, pre, out, digest)
    sink["runs"].append({"seed": "template", "wall": time.time() - t0, "cap": cap})
    sink["comp_nominal_mismatch"] = check_comp_nominal(cap, events_dict, hp)
    with open(os.environ["E3_SINK"], "wb") as f:
        pickle.dump(sink, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


# ---------------------------------------------------------------- parent side
def run_child(task, env_extra, tag):
    path = os.path.join(PY_DIR, task + ".py")
    os.makedirs(os.path.join(PY_DIR, "..", "results"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "_tmp"), exist_ok=True)
    sink = os.path.join(OUT, "_tmp", f"sink_{task[:3]}_{tag}_{os.getpid()}.pkl")
    env = dict(os.environ, E3_SINK=sink, E3_HOME=HERE,
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "_inject")] +
                                          ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
               **env_extra)
    log = os.path.join(OUT, f"{task}_{tag}.log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        rc = subprocess.call([sys.executable, path], env=env, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if rc != 0 or not os.path.exists(sink):
        raise E3Error(f"{tag}: task script failed (exit {rc}) or never reached its 2nd backend call; see {log}")
    with open(sink, "rb") as f:
        out = pickle.load(f)
    os.remove(sink)
    if out["calls"] != 2:
        raise E3Error(f"{tag}: expected 2 backend calls, saw {out['calls']}")
    out["process_wall"] = wall
    return out


def capture(task, seeds, repeats=1):
    """Primary capture, then `repeats` independent re-seed processes (in parallel) on the same final state:
    BusyBarn's backend iterates sets of objects, so the same state and seed can give a different latency
    in another process."""
    from concurrent.futures import ThreadPoolExecutor
    prim = run_child(task, {"E3_MODE": "capture", "E3_SEEDS": ""}, "capture")
    env = {"E3_MODE": "capture", "E3_SEEDS": " ".join(map(str, seeds))}
    with ThreadPoolExecutor(repeats) as ex:
        rss = list(ex.map(lambda r: run_child(task, env, f"capture_seeds{r}"), range(repeats)))
    for rs in rss:
        if rs["pre_digest"] != prim["pre_digest"]:
            raise E3Error("re-seed process saw a different pre-backend state than the primary capture run")
    prim["reseeded"] = rss[0]["reseeded"]
    prim["reseeded_repeats"] = [rs["reseeded"] for rs in rss]
    return prim


def inject(task, mapping, faults, steps, t_max, seeds, tag, preserve_routes=True):
    os.makedirs(os.path.join(OUT, "_tmp"), exist_ok=True)
    spec_path = os.path.join(OUT, "_tmp", f"spec_{task[:3]}_{tag}_{os.getpid()}.pkl")
    with open(spec_path, "wb") as f:
        pickle.dump({"mapping": mapping, "faults": faults, "steps": steps, "t_max": t_max, "seeds": list(seeds),
                     "preserve_routes": preserve_routes}, f, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        return run_child(task, {"E3_MODE": "inject", "E3_SPEC": spec_path}, tag)
    finally:
        os.remove(spec_path)


def migration(m0, m1):
    """Behaviours whose computation location changed; weight bytes whose DDR location changed."""
    moved = [t for t in m0["beha_loc"] if tuple(m0["beha_loc"][t]) != tuple(m1["beha_loc"][t])]
    wb = m0["weight_bytes"]
    moved_w = [k for k in wb if [tuple(x) for x in m0["data_loc"][k[0]][k[1]]] !=
               [tuple(x) for x in m1["data_loc"][k[0]][k[1]]]]
    tot_w = sum(wb.values())
    return {"moved_behas": len(moved), "n_behas": len(m0["beha_loc"]), "moved_behas_share": len(moved) / len(m0["beha_loc"]),
            "moved_weight_bytes": sum(wb[k] for k in moved_w), "weight_bytes": tot_w,
            "moved_weight_share": sum(wb[k] for k in moved_w) / tot_w if tot_w else 0,
            "moved_weight_splits": len(moved_w)}


def route_diff(capA, capB):
    """How many communications got a different route (matched with route_key; tags may differ because comm
    ids are allocated in a different order)."""
    def key(e):  # here the source is part of the comparison
        return (e["source"],) + route_key(e["target"], e["bytes"], e["issue"], e["deps"])

    def routes(cap):
        r = {}
        for e in cap["events"].values():
            if e["type"] == "communication":
                r.setdefault(key(e), []).append(sorted(tuple(map(tuple, br)) for br in (e["branches"] or [])))
        return r
    ra, rb = routes(capA), routes(capB)
    same = diff = unmatched = 0
    for k in set(ra) | set(rb):
        a, b = sorted(ra.get(k, [])), sorted(rb.get(k, []))
        if len(a) != len(b):
            unmatched += max(len(a), len(b))
            continue
        for x, y in zip(a, b):
            same += x == y
            diff += x != y
    return {"same_route": same, "different_route": diff, "unmatched": unmatched,
            "n_comm_A": sum(map(len, ra.values())), "n_comm_B": sum(map(len, rb.values()))}


def _lb_consistent(lbs, analytic, plat):
    """(LB components and computation sums identical across graphs and equal to the analytic sums;
    per graph: DDR traffic in the graph minus the analytic one, as [extra read bytes...])."""
    keys = ["LB_comp", "LB_mem", "LB_dep", "LB_mem_occ", "comp_sum"]
    same = all(all(l[k] == lbs[0][k] for k in keys) for l in lbs[1:]) and analytic["comp_sum"] == lbs[0]["comp_sum"]
    extra = []
    for l in lbs:
        c = collections.Counter(l["graph_ddr_read_bytes"])
        c.subtract(collections.Counter(l["ddr_read_bytes"]))
        w_ok = all(w == analytic["ddr_broadcast_bytes"] for w in l["graph_ddr_broadcast_bytes_per_port"].values())
        extra.append({"extra_reads": sorted(c.elements()),
                      "missing_reads": sorted(b for b, v in c.items() for _ in range(-v)),
                      "broadcast_matches": w_ok})
    return same, extra


def _short(lb):
    return {k: lb[k] for k in ["LB", "dominant", "LB_comp", "LB_mem", "LB_dep", "LB_comp_by_device",
                               "LB_mem_read", "LB_mem_write", "LB_mem_occ", "LB_occ", "dominant_occ", "n_ddr_ports"]}


def step1(short, seeds, repeats):
    from concurrent.futures import ThreadPoolExecutor
    task = TASKS[short]
    os.makedirs(OUT, exist_ok=True)
    print(f"[e3] {short}: capture {task} ({repeats} re-seed processes)", flush=True)
    pre = capture(task, seeds, repeats)
    a_tmpl = pre["last"]["result"]
    pre_seed = {s: c["result"] for s, c, _ in pre["reseeded"]}
    print(f"[e3] {short}: pre-fault template {a_tmpl}, seeds {pre_seed}", flush=True)
    with gzip.open(os.path.join(OUT, f"{task}_prefault.pkl.gz"), "wb") as f:
        pickle.dump({"task": task, "mapping": pre["mapping"], "platform": pre["platform"], "final": pre["last"],
                     "reseeded": [(s, c) for s, c, _ in pre["reseeded"]], "analytic": pre["analytic"]},
                    f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[e3] {short}: inject the pre-fault mapping and routes, no fault, no SA step ({repeats} processes)",
          flush=True)
    nofault = {"failed_nodes": [], "failed_links": []}
    with ThreadPoolExecutor(repeats) as ex:
        injs = list(ex.map(lambda r: inject(task, pre["mapping"], nofault, 0, None, seeds, f"check1_{r}"),
                           range(repeats)))
    inj = injs[0]
    with gzip.open(os.path.join(OUT, f"{task}_check1_injected.pkl.gz"), "wb") as f:
        pickle.dump({"route_log": inj["log"].get("routes"), "runs": [(r["seed"], r["cap"]) for r in inj["runs"]]},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    # cross-process spread of the same evaluation (same state, same backend seed, another process)
    pre_rep = {s: [dict((k, c["result"][0]) for k, c, _ in rs)[s] for rs in pre["reseeded_repeats"]] for s in seeds}
    inj_rep = {s: [dict((r["seed"], r["cap"]["result"][0]) for r in i["runs"])[s] for i in injs] for s in seeds}
    inj_tmpl_rep = [dict((r["seed"], r["cap"]["result"][0]) for r in i["runs"])["template"] for i in injs]
    all0 = [v for s in seeds for v in pre_rep[s]]
    all1 = [v for s in seeds for v in inj_rep[s]]
    noise = {"prefault_per_seed": {str(s): pre_rep[s] for s in seeds},
             "injected_per_seed": {str(s): inj_rep[s] for s in seeds},
             "injected_template_rng": inj_tmpl_rep,
             "prefault_cross_process_spread_max": max((max(v) - min(v)) / min(v) for v in pre_rep.values()),
             "injected_cross_process_spread_max": max((max(v) - min(v)) / min(v) for v in inj_rep.values()),
             "mean_prefault_all": float(np.mean(all0)), "mean_injected_all": float(np.mean(all1)),
             "mean_rel_diff_all": float(np.mean(all1) / np.mean(all0) - 1),
             "min_prefault_all": min(all0), "min_injected_all": min(all1),
             "n_per_side": len(all0)}
    # check 1 as the user accepted it on 2026-10-31: the injected graph is isomorphic to the pre-fault one
    # and the mean over seeds x processes is within 1% (per-seed values carry cross-process noise)
    gdiff = graph_diff(pre["last"], inj["runs"][-1]["cap"])
    ok = gdiff == 0 and abs(noise["mean_rel_diff_all"]) <= CHECK1_TOL
    route_log = inj["log"].get("routes") or []
    rerouted = sum(r["rerouted"] for r in route_log)
    inj_res = {r["seed"]: r["cap"]["result"] for r in inj["runs"]}
    mig = migration(pre["mapping"], inj["mapping"])

    rows, ok_strict = [], True
    for s in seeds:
        a0, a1 = pre_seed[s][0], inj_res[s][0]
        rel = a1 / a0 - 1
        ok_strict &= abs(rel) <= CHECK1_TOL
        rows.append({"seed": s, "prefault": pre_seed[s], "injected": inj_res[s], "rel_diff": rel})
    rel_t = inj_res["template"][0] / a_tmpl[0] - 1
    mean0 = np.mean([pre_seed[s][0] for s in seeds])
    mean1 = np.mean([inj_res[s][0] for s in seeds])

    an = pre["analytic"]
    lbs = [lower_bounds(pre["last"], pre["platform"], an)] + \
        [lower_bounds(c, pre["platform"], an) for _, c, _ in pre["reseeded"]] + \
        [lower_bounds(r["cap"], inj["platform"], inj["analytic"]) for r in inj["runs"]]
    same, extra = _lb_consistent(lbs, an, pre["platform"])
    an_ok = inj["analytic"] == an
    lb = lbs[0]
    evals = [a_tmpl[0]] + all0 + all1 + inj_tmpl_rep
    best = min(evals)
    tight = {k: lb[k] / best for k in ["LB", "LB_comp", "LB_mem", "LB_dep", "LB_mem_occ", "LB_occ"]}
    check3_pre = lb["LB"] <= best and lb["LB_occ"] <= best

    summ = {
        "task": task, "step": 1,
        "prefault": {"template_rng": a_tmpl, "seeds": {str(s): pre_seed[s] for s in seeds},
                     "seed_mean": float(mean0), "e2_reference": E2_REF[short],
                     "equals_e2_template": a_tmpl[0] == E2_REF[short]["template"],
                     "equals_e2_seeds": all(pre_seed[s][0] == E2_REF[short]["seeds"].get(s) for s in seeds),
                     "eval_wall": pre["eval_wall"], "process_wall": pre["process_wall"]},
        "injected_no_fault": {"template_rng": inj_res["template"], "seeds": {str(s): inj_res[s] for s in seeds},
                              "seed_mean": float(mean1), "fields_written": inj["log"]["inject"],
                              "sa_args": inj["log"].get("sa_args"), "sa_wall": inj["log"]["sa_wall"],
                              "eval_walls": {str(r["seed"]): r["wall"] for r in inj["runs"]},
                              "process_wall": inj["process_wall"], "migration_vs_prefault": mig,
                              "route_restore": [{k: v for k, v in r.items() if k != "rerouted_tags"} for r in route_log],
                              "rerouted_without_fault": rerouted,
                              "route_diff_vs_prefault_template": route_diff(pre["last"], inj["runs"][-1]["cap"])},
        "check1": {"pass": bool(ok), "criterion": "graph isomorphic and |mean over seeds x processes| <= 1%",
                   "graph_diff": gdiff, "per_seed_strict_pass": bool(ok_strict),
                   "tolerance": CHECK1_TOL, "per_seed": rows, "template_rng_rel_diff": rel_t,
                   "seed_mean_rel_diff": float(mean1 / mean0 - 1), "repeats": repeats, "noise": noise},
        "LB_pre": _short(lb), "LB_pre_ddr_read_transfers": len(lb["ddr_read_bytes"]),
        "LB_pre_ddr_read_bytes": sum(lb["ddr_read_bytes"]),
        "LB_pre_ddr_broadcast_bytes_per_port": sum(next(iter(lb["ddr_broadcast_bytes_per_port"].values()), [])),
        "LB_identical_across_graphs_and_comp_equals_analytic": same,
        "analytic_workload_same_in_capture_and_inject": an_ok,
        "graph_ddr_traffic_minus_analytic": {"prefault_template": extra[0],
                                             "prefault_seeds": extra[1:1 + len(pre["reseeded"])],
                                             "injected": extra[1 + len(pre["reseeded"]):]},
        "comp_duration_equals_working_time": not pre["comp_nominal_mismatch"] and not inj["comp_nominal_mismatch"],
        "best_latency_prefault_platform": best, "n_evaluations": len(evals), "tightness_pre": tight,
        "check3_LB_pre_le_best": check3_pre,
        "platform": {k: pre["platform"][k] for k in ["n_tensorcore", "n_vectorunit", "n_nodes", "ddr2co", "co2ddr"]},
        "artifact_commit": _artifact_commit(), "artifact_work_copy": WORK, "template_python_seed": TEMPLATE_SEED,
        "backend_seeds": list(seeds), "command": sys.argv,
        "notes": ("check1 (option A, user 2026-10-31): the pre-fault final mapping (behaviour locations + every data "
                  "split location) is written over initialized_mapping's result; the template's build_event and "
                  "add_broadcast rebuild events, then every communication gets its captured pre-fault source and "
                  "route back (matched by route_key, which excludes the source because the original SA moves "
                  "weights only on the events; only communications whose key is missing or whose route uses a "
                  "missing node/link are re-run through the artifact's Dijkstra) and the platform link-load "
                  "counters are set to what the routes "
                  "imply; SA runs 0 steps; the final state is evaluated with the same backend seeds as the pre-fault "
                  "capture. LB_mem (from the mapping-free workload: consumer x needed offline split reads, "
                  "broadcast splits to every port): reads bounded by total DDR-read bytes / (ports x bandwidth); "
                  "broadcast writes reach every port, so bounded per port; both via the link formula; LB_mem_occ "
                  "sums whole per-transfer link occupancy instead."),
    }
    with open(os.path.join(OUT, f"{task}_step1.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, default=str)
    line = {"task": short, "check1": summ["check1"]["pass"], "pre_seeds": [pre_seed[s][0] for s in seeds],
            "inj_seeds": [inj_res[s][0] for s in seeds], "rel": [round(r["rel_diff"], 5) for r in rows],
            "pre_template": a_tmpl[0], "inj_template": inj_res["template"][0], "LB_pre": lb["LB"],
            "dominant": lb["dominant"], "tight": round(tight["LB"], 4), "LB_comp": lb["LB_comp"],
            "LB_mem": lb["LB_mem"], "LB_dep": lb["LB_dep"], "LB_occ": lb["LB_occ"],
            "lb_consistent": same and an_ok, "check3_pre": check3_pre,
            "extra_ddr_reads_prefault": extra[0]["extra_reads"], "extra_ddr_reads_injected": extra[-1]["extra_reads"],
            "moved_behas": mig["moved_behas"], "moved_weight_bytes": mig["moved_weight_bytes"],
            "route_restore": [(r["where"], r["restored"], r["rerouted"], r["source_restored"]) for r in route_log],
            "routes": summ["injected_no_fault"]["route_diff_vs_prefault_template"],
            "pre_rep": pre_rep, "inj_rep": inj_rep, "inj_tmpl_rep": inj_tmpl_rep,
            "mean_rel_all": round(noise["mean_rel_diff_all"], 5),
            "spread_pre": round(noise["prefault_cross_process_spread_max"], 4),
            "spread_inj": round(noise["injected_cross_process_spread_max"], 4)}
    print("[e3] " + json.dumps(line, default=str), flush=True)
    if not ok:
        raise E3Error(f"self-check 1 failed: {rows}")
    return summ


# ---------------------------------------------------------------- step 2+: faults, R0, W
def busiest_link(cap, kind):
    """Directed link of type `kind` with the largest busy time in a pre-fault final evaluation."""
    busy = {k: sum(f - s for s, f, _ in rec) for k, rec in cap["links"].items() if cap["link_types"][k] == kind}
    k = max(busy, key=lambda x: (busy[x], str(x)))
    return k, busy[k], busy


F2_SEED = 2026


def choose_fault(pre, fid):
    cap = pre["final"]
    a = cap["result"][0]
    all_cores = sorted({(d, c) for d in range(4) for c in range(16)})  # 2x2 dies x 4x4 cores
    non_corner = [n for n in all_cores if n[1] not in CORNER_CORES]
    if fid == "F1":
        busy = collections.Counter()
        for (_, _, loc), rec in cap["devices"].items():
            busy[tuple(loc[:-1])] += sum(f - s for s, f, _ in rec)
        top = max(busy[n] for n in non_corner)
        ties = [n for n in non_corner if busy[n] == top]
        node = ties[0]
        return {"id": fid, "kind": "node (core + router)", "failed_nodes": [node], "failed_links": [],
                "rule": "non-corner core with the largest computation busy time (tensor + vector unit) in the "
                        "pre-fault final evaluation; ties -> smallest (die, core)",
                "compute_busy": top, "compute_busy_over_a": top / a, "ties": ties}
    if fid == "F2":
        node = random.Random(F2_SEED).choice(non_corner)
        busy = sum(sum(f - s for s, f, _ in rec) for (_, _, loc), rec in cap["devices"].items()
                   if tuple(loc[:-1]) == node)
        return {"id": fid, "kind": "node (core + router)", "failed_nodes": [node], "failed_links": [],
                "rule": f"random non-corner core, random.Random({F2_SEED}).choice(sorted non-corner cores)",
                "seed": F2_SEED, "compute_busy": busy, "compute_busy_over_a": busy / a}
    if fid == "F8":  # E3 extension chosen by the user 2026-10-31: the busiest DDR port fails
        busy = collections.Counter()
        for k, rec in cap["links"].items():
            if cap["link_types"][k] == "ddr2co":
                busy[k[0]] += sum(f - s for s, f, _ in rec)
        node = max(busy, key=lambda n: (busy[n], str(n)))
        return {"id": fid, "kind": "DDR port node", "failed_nodes": [node], "failed_links": [],
                "rule": "DDR port whose ddr2co (read) link has the largest busy time in the pre-fault final "
                        "evaluation; R0 moves its data to the live port fewest hops from its corner core",
                "busy": busy[node], "busy_over_a": busy[node] / a,
                "top_ports": [[str(n), v] for n, v in busy.most_common(4)]}
    if fid in ("F6", "F7"):  # added by the user 2026-10-31: PE dead, router still forwards; F1's / F2's node
        base = choose_fault(pre, "F1" if fid == "F6" else "F2")
        return {**base, "id": fid, "kind": "PE only (tensor + vector unit); router keeps forwarding",
                "failed_nodes": [], "failed_links": [], "failed_pes": base["failed_nodes"],
                "rule": f"same node as {base['id']} ({base['rule']}); emulated in memory: the node and its links "
                        "stay, its compute devices are removed from tensorcore_dict / vectorunit_dict",
                "paired_with": base["id"]}
    if fid in ("F3", "F4"):
        kind = "co2co" if fid == "F3" else "ch2ch"
        (u, v), b, busy = busiest_link(cap, kind)
        return {"id": fid, "kind": f"link {kind}, both directions",
                "failed_nodes": [], "failed_links": [(u, v), (v, u)],
                "rule": f"busiest {kind} link in the pre-fault final evaluation (template RNG), busy time",
                "busy": b, "busy_over_a": b / a, "busy_reverse": busy.get((v, u), 0)}
    if fid == "F5":  # added by the user 2026-10-31: the most CRITICAL on-chip link, to contrast with F3/F4
        _, _, segs, _, _ = _e2().analyse(cap)
        wait = collections.Counter()
        for sg in segs:
            if sg["res_key"][0] == "link" and cap["link_types"][sg["res_key"][1]] in ("co2co", "ch2ch"):
                wait[sg["res_key"][1]] += sg["wait_excl"]
        if not wait:
            raise E3Error("F5: no critical wait on any co2co/ch2ch link")
        (u, v), w = max(wait.items(), key=lambda x: (x[1], str(x[0])))
        busy = {k: sum(f - s for s, f, _ in rec) for k, rec in cap["links"].items()}
        return {"id": fid, "kind": f"link {cap['link_types'][(u, v)]}, both directions",
                "failed_nodes": [], "failed_links": [(u, v), (v, u)],
                "rule": "on-chip link (co2co/ch2ch) with the largest exclusive critical wait (E2 attribution) in "
                        "the pre-fault final evaluation (template RNG); DDR port links cannot be failed in the artifact",
                "critical_wait": w, "critical_wait_over_a": w / a, "busy": busy.get((u, v), 0),
                "busy_over_a": busy.get((u, v), 0) / a, "busy_reverse": busy.get((v, u), 0),
                "top_critical_links": [[str(k), x] for k, x in wait.most_common(5)]}
    raise E3Error(f"fault {fid} not implemented yet (F1/F2 need task relocation: step 3)")


def canon_graph(cap):
    """Event graph with communications relabelled by (source, route_key, occurrence): equal iff two graphs
    have the same placements, dependencies, sources and routes."""
    lab, cnt = {}, collections.Counter()
    ev = cap["events"]
    for t in sorted(ev, key=str):
        x = ev[t]
        if x["type"] == "communication":
            k = (x["source"],) + route_key(x["target"], x["bytes"], x["issue"], x["deps"])
            lab[t] = ("c", k, cnt[k])
            cnt[k] += 1
        else:
            lab[t] = ("p", t, x["location"], x["device"])
    out = {}
    for t, x in ev.items():
        br = tuple(sorted(tuple(map(tuple, b)) for b in (x.get("branches") or []))) \
            if x["type"] == "communication" else None
        out[lab[t]] = (frozenset(lab[d] for d in x["deps"]), frozenset(lab[i] for i in x["issue"]), br)
    return out


def graph_diff(capA, capB):
    a, b = canon_graph(capA), canon_graph(capB)
    return sum(1 for k in set(a) | set(b) if a.get(k) != b.get(k))


def weight_migration(routes0, routes1, ddr_nodes):
    """Primary weight-migration measure: DDR reads of offline data (DDR source, no producer) whose DDR
    changed, matched by (bytes, consumers) so that a moved consumer is not counted as a moved weight."""
    ddr = set(map(tuple, ddr_nodes))

    def reads(routes):
        r = collections.defaultdict(list)
        for (tg, nbytes, cons, prods), lst in routes.items():
            for src, _, _ in lst:
                if tuple(src) in ddr and not prods:
                    r[(nbytes, cons)].append(tuple(src))
        return r
    r0, r1 = reads(routes0), reads(routes1)
    moved = moved_bytes = total = n = 0
    for k in set(r0) | set(r1):
        c0, c1 = collections.Counter(r0.get(k, [])), collections.Counter(r1.get(k, []))
        m = sum(c0.values()) - sum((c0 & c1).values())
        moved += m
        moved_bytes += m * k[0]
        total += sum(c0.values()) * k[0]
        n += sum(c0.values())
    return {"moved_ddr_reads": moved, "n_ddr_reads": n, "moved_read_bytes": moved_bytes, "read_bytes": total,
            "moved_read_share": moved_bytes / total if total else 0}


def check2(cap, mapping, fault):
    """No computation or data on a failed node, no route through a failed link or node."""
    fn = set(map(tuple, fault["failed_nodes"]))
    fp = set(map(tuple, fault.get("failed_pes", [])))  # PE-only: no compute or data there, routing through is fine
    fl = {(tuple(a), tuple(b)) for a, b in fault["failed_links"]}
    bad = []
    for t, loc in mapping["beha_loc"].items():
        if tuple(loc[:-1]) in fn | fp:
            bad.append(("computation_on_failed_node", t, loc))
    for d, sps in mapping["data_loc"].items():
        for sp, locs in sps.items():
            for n in locs:
                if tuple(n) in fn | fp:
                    bad.append(("data_on_failed_node", d, sp, n))
    for t, e in cap["events"].items():
        if e["type"] == "computation" and tuple(e["location"][:-1]) in fp:
            bad.append(("computation_event_on_failed_pe", t))
        if e["type"] == "communication" and e["source"] in fp:
            bad.append(("comm_source_on_failed_pe", t))
    for t, e in cap["events"].items():
        if e["type"] != "communication":
            continue
        tg = e["target"] if isinstance(e["target"], set) else {e["target"]}
        if e["source"] in fn or tg & fn:
            bad.append(("comm_endpoint_failed", t))
        for br in e["branches"] or []:
            for k in map(tuple, br):
                if k in fl or k[0] in fn or k[1] in fn:
                    bad.append(("route_through_failure", t, k))
    for k in cap["links"]:
        if k in fl:
            bad.append(("link_record_on_failed_link", k))
    for k in cap["devices"]:
        if tuple(k[2][:-1]) in fn | fp:
            bad.append(("device_record_on_failed_node", k))
    return bad


RUN_FIELDS = ["task", "fault", "method", "steps", "t_max", "seed", "a", "b", "c", "sa_wall", "eval_wall",
              "moved_behas", "moved_behas_share", "moved_read_bytes", "moved_read_share",
              "moved_weight_bytes_datadict", "rerouted_comms", "restored_comms", "process"]


FULL_VARIANTS = [(g, l) for g in (0, 1) for l in range(7)]  # greedy0/1 x lossratio0-6, as BusyBarn's end-to-end


def run_fault(short, fid, budgets, tmaxes, seeds, ex, full=True):
    """R0, W(budget, t_max) and FULL for one task x fault; child processes go to the shared executor ex."""
    task = TASKS[short]
    with gzip.open(os.path.join(OUT, f"{task}_prefault.pkl.gz"), "rb") as f:
        pre = pickle.load(f)
    fault = choose_fault(pre, fid)
    failed_desc = fault["failed_links"] or fault["failed_nodes"] or fault.get("failed_pes")
    print(f"[e3] {short} {fid}: {fault['kind']} {failed_desc}", flush=True)
    m0 = pre["mapping"]
    runs = [("R0", None, None)] + [("W", 0, 10)] + [("W", s, t) for t in tmaxes for s in budgets]
    gemini = "gemini" in task
    if full:  # Gemini is run once (no variants), as in BusyBarn's end-to-end comparison
        runs += [("FULL", "gemini", None)] if gemini else [("FULL", g, l) for g, l in FULL_VARIANTS]

    def tag(m, s, t):
        if m == "FULL":
            return f"{fid}_FULL_gemini" if s == "gemini" else f"{fid}_FULL_g{s}_l{t}"
        return f"{fid}_{m}" + ("" if s is None else f"_s{s}_t{t:g}")

    def launch(r):
        m, s, t = r
        if m == "FULL":
            import re
            variant = task if s == "gemini" else re.sub(r"greedy\d_lossratio\d", f"greedy{s}_lossratio{t}", task)
            return inject(variant, None, fault, "template", None, seeds, tag(*r), preserve_routes=False)
        return inject(task, m0, fault, s, t, seeds, tag(*r))
    sinks = list(ex.map(launch, runs))

    rows, res = [], {}
    for (meth, s, t), sk in zip(runs, sinks):
        if not sk["platform"]["strongly_connected"]:
            raise E3Error(f"{fid}: the faulty platform is not strongly connected")
        mig = migration(m0, sk["mapping"])
        # DDR nodes of the pre-fault platform too: reads from a failed port must be counted
        wm = weight_migration(m0["routes"], sk["mapping"]["routes"],
                              list(pre["platform"]["ddr_nodes"]) + list(sk["platform"]["ddr_nodes"]))
        rl = sk["log"].get("routes") or []
        for r in sk["runs"]:
            a, b, c = r["cap"]["result"]
            rows.append({"task": short, "fault": fid, "method": meth, "steps": s, "t_max": t, "seed": r["seed"],
                         "a": a, "b": b, "c": c, "sa_wall": sk["log"]["sa_wall"], "eval_wall": r["wall"],
                         "moved_behas": mig["moved_behas"], "moved_behas_share": mig["moved_behas_share"],
                         "moved_read_bytes": wm["moved_read_bytes"], "moved_read_share": wm["moved_read_share"],
                         "moved_weight_bytes_datadict": mig["moved_weight_bytes"],
                         "rerouted_comms": sum(x["rerouted"] for x in rl),
                         "restored_comms": sum(x["restored"] for x in rl), "process": tag(meth, s, t)})
        lat = [r["cap"]["result"][0] for r in sk["runs"] if r["seed"] != "template"]
        res[tag(meth, s, t)] = {"method": meth, "steps": s, "t_max": t, "seed_lat": lat,
                                "process_wall": sk["process_wall"], "relocated": sk["log"].get("relocated"),
                                "mean": float(np.mean(lat)), "min": min(lat), "max": max(lat),
                                "template_rng": [r["cap"]["result"] for r in sk["runs"] if r["seed"] == "template"][0],
                                "sa_wall": sk["log"]["sa_wall"], "eval_wall_mean": float(np.mean([r["wall"] for r in sk["runs"]])),
                                "migration": mig, "weight_migration": wm, "route_log":
                                    [{k: v for k, v in x.items() if k != "rerouted_tags"} for x in rl],
                                "check2_violations": [],
                                "xy_none_ddr_pairs": sk["log"].get("xy_none_ddr_pairs", 0),
                                "sa_args": sk["log"].get("sa_args")}
        for r in sk["runs"]:
            res[tag(meth, s, t)]["check2_violations"] += check2(r["cap"], sk["mapping"], fault)

    k_r0, k_w0 = tag("R0", None, None), tag("W", 0, 10)
    r0, w0 = sinks[0], sinks[1]
    # check 4: W at 0 steps has the same mapping and the same event graph as R0
    gd = graph_diff(r0["runs"][-1]["cap"], w0["runs"][-1]["cap"])
    mig04 = migration(r0["mapping"], w0["mapping"])
    check4 = gd == 0 and mig04["moved_behas"] == 0 and mig04["moved_weight_bytes"] == 0 and \
        weight_migration(r0["mapping"]["routes"], w0["mapping"]["routes"],
                         list(pre["platform"]["ddr_nodes"]) + list(r0["platform"]["ddr_nodes"]))["moved_ddr_reads"] == 0
    check2_ok = all(not v["check2_violations"] for v in res.values())
    # lower bound on the faulty platform
    plat = r0["platform"]
    lbs = [lower_bounds(r["cap"], sk["platform"], sk["analytic"]) for sk in sinks for r in sk["runs"]]
    same, _ = _lb_consistent(lbs, r0["analytic"], plat)
    lb = lbs[0]
    all_lat = [x["a"] for x in rows]
    best = min(all_lat)
    check3 = lb["LB"] <= best and lb["LB_occ"] <= best
    pre_mean = float(np.mean([c["result"][0] for _, c in pre["reseeded"]]))
    r0_mean = res[k_r0]["mean"]
    w_keys = [k for k, v in res.items() if v["method"] == "W" and v["steps"]]
    best_w = min(w_keys, key=lambda k: res[k]["mean"])
    f_keys = [k for k, v in res.items() if v["method"] == "FULL"]
    full_sum = None
    if f_keys:
        fb = min(f_keys, key=lambda k: res[k]["mean"])
        fm = res[fb]["mean"]
        gap = {k: res[k]["mean"] / fm - 1 for k in [k_r0] + w_keys}

        def reach(thr):
            """Fewest SA steps (R0 = 0) whose seed-mean is within thr of FULL's best; ties -> less SA time."""
            ok = [(0 if res[k]["method"] == "R0" else res[k]["steps"], res[k]["sa_wall"], k)
                  for k in [k_r0] + w_keys if gap[k] <= thr]
            if not ok:
                return None
            s_, w_, k_ = min(ok)
            per_t = {}
            for tm in tmaxes:
                c = sorted((res[k]["steps"], res[k]["sa_wall"]) for k in w_keys
                           if res[k]["t_max"] == tm and gap[k] <= thr)
                per_t[f"t{tm:g}"] = c[0] if c else None
            return {"steps": s_, "sa_wall": w_, "run": k_, "per_t_max": per_t}
        full_sum = {"best_variant": fb, "best_mean": fm, "best_seed_lat": res[fb]["seed_lat"],
                    "best_migration": res[fb]["migration"], "best_weight_migration": res[fb]["weight_migration"],
                    "variants": {k: {x: res[k][x] for x in ["seed_lat", "mean", "sa_wall", "process_wall"]} |
                                 {"moved_behas": res[k]["migration"]["moved_behas"]} for k in f_keys},
                    "total_sa_wall_14": sum(res[k]["sa_wall"] for k in f_keys),
                    "total_process_wall_14": sum(res[k]["process_wall"] for k in f_keys),
                    "gap_over_FULL": gap, "reach_5pct": reach(0.05), "reach_2pct": reach(0.02)}
    summ = {
        "task": task, "fault": fault, "prefault_seed_mean": pre_mean,
        "prefault_seeds": {str(s): c["result"] for s, c in pre["reseeded"]},
        "R0": res[k_r0], "fault_cost_R0_over_prefault_minus_1": r0_mean / pre_mean - 1,
        "W": {k: res[k] for k in [k_w0] + w_keys},
        "W_over_R0_minus_1": {k: res[k]["mean"] / r0_mean - 1 for k in w_keys},
        "W_over_prefault_minus_1": {k: res[k]["mean"] / pre_mean - 1 for k in w_keys},
        "best_W": best_w, "FULL": full_sum,
        "LB_post": _short(lb), "LB_post_identical_across_graphs": same,
        "best_latency_postfault_platform": best, "n_evaluations": len(all_lat),
        "tightness_post": {k: lb[k] / best for k in ["LB", "LB_comp", "LB_mem", "LB_dep", "LB_mem_occ", "LB_occ"]},
        "platform_postfault": {k: plat[k] for k in ["n_tensorcore", "n_vectorunit", "n_nodes", "strongly_connected"]},
        "n_links_postfault": len(plat["links"]),
        "checks": {"check2_fault_respected": check2_ok, "check3_LB_post_le_best": check3,
                   "check4_W0_equals_R0": check4, "check4_graph_diff": gd, "check4_migration": mig04},
        "artifact_commit": _artifact_commit(), "template_python_seed": TEMPLATE_SEED, "backend_seeds": list(seeds),
        "command": sys.argv,
        "notes": ("R0: pre-fault mapping, sources, routes, comm ids and event order injected on the faulty platform; "
                  "communications whose route uses a failed link/node are re-routed by the artifact's Dijkstra "
                  "(against the link loads of the kept routes); no SA (random/np.random seeded 123 where SA would "
                  "start, as anneal() does). W(steps, t_max): same start, then the original StreamOptimization with "
                  "that budget; t_min and loss ratios as in the task. Migration vs the pre-fault mapping: behaviours "
                  "whose computation location changed; weight reads whose DDR source changed (primary; matched by "
                  "bytes and consumer) and data_dict-based weight bytes (secondary; data_dict is stale after SA). "
                  "Node faults: R0 moves each behaviour of the failed core to the nearest healthy device of its "
                  "type (Manhattan on the global core grid), least loaded first, then smallest id. FULL: each of the "
                  "14 variants (greedy0/1 x lossratio0-6) run unchanged on the faulty platform (template steps and "
                  "t_max); best = smallest 3-seed mean. Wall times: sa_wall = stream_mapping only; eval_wall = one "
                  "backend evaluation; processes ran in parallel, so walls include contention.")}
    with open(os.path.join(OUT, f"{task}_{fid}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, default=str)
    _write_runs_csv(os.path.join(OUT, f"{task}_{fid}_runs.csv"), rows)
    with gzip.open(os.path.join(OUT, f"{task}_{fid}_caps.pkl.gz"), "wb") as f:
        pickle.dump({"fault": fault, "runs": {tag(*r): {"mapping": sk["mapping"], "platform": sk["platform"],
                                                         "caps": [(x["seed"], x["cap"]) for x in sk["runs"]]}
                                               for r, sk in zip(runs, sinks)}}, f, protocol=pickle.HIGHEST_PROTOCOL)
    short_k = lambda k: k.split(f"{fid}_W_")[1]
    row = {"task": short, "fault": fid, "fault_kind": fault["kind"],
           "failed": str(failed_desc),
           "prefault_mean": round(pre_mean), "R0_mean": round(r0_mean), "fault_cost": r0_mean / pre_mean - 1,
           "R0_moved_behas": res[k_r0]["migration"]["moved_behas"],
           "R0_rerouted": sum(x["rerouted"] for x in res[k_r0]["route_log"]),
           "best_W": short_k(best_w), "best_W_mean": round(res[best_w]["mean"]),
           "best_W_over_R0": res[best_w]["mean"] / r0_mean - 1, "best_W_sa_wall": res[best_w]["sa_wall"],
           "best_W_moved_behas": res[best_w]["migration"]["moved_behas"],
           "best_W_moved_read_share": res[best_w]["weight_migration"]["moved_read_share"],
           "LB_post": lb["LB"], "LB_dominant": lb["dominant"], "tightness": lb["LB"] / best,
           "tight_comp": lb["LB_comp"] / best, "tight_mem": lb["LB_mem"] / best, "tight_dep": lb["LB_dep"] / best,
           "tightness_occ": lb["LB_occ"] / best, "best_latency": best,
           "check2": check2_ok, "check3": check3, "check4": check4}
    if full_sum:
        row.update({"FULL_mean": round(full_sum["best_mean"]), "FULL_variant": full_sum["best_variant"].split("FULL_")[1],
                    "FULL_moved_behas": full_sum["best_migration"]["moved_behas"],
                    "FULL_moved_read_share": full_sum["best_weight_migration"]["moved_read_share"],
                    "FULL_total_sa_wall": full_sum["total_sa_wall_14"],
                    "FULL_total_process_wall": full_sum["total_process_wall_14"],
                    "R0_over_FULL": full_sum["gap_over_FULL"][k_r0],
                    "best_W_over_FULL": full_sum["gap_over_FULL"][best_w],
                    "reach5_steps": (full_sum["reach_5pct"] or {}).get("steps"),
                    "reach5_sa_wall": (full_sum["reach_5pct"] or {}).get("sa_wall"),
                    "reach2_steps": (full_sum["reach_2pct"] or {}).get("steps"),
                    "reach2_sa_wall": (full_sum["reach_2pct"] or {}).get("sa_wall")})
        row.update({f"gap_{short_k(k)}": full_sum["gap_over_FULL"][k] for k in w_keys})
    summ["table_row"] = row
    with open(os.path.join(OUT, f"{task}_{fid}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, default=str)
    print("[e3] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                               default=str), flush=True)
    if not (check2_ok and check3 and check4):
        raise E3Error(f"{short} {fid}: self-check failed: check2={check2_ok} check3={check3} check4={check4} "
                      f"(graph diff {gd})")
    return summ


def _write_runs_csv(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("step1")
    p1.add_argument("--task", choices=sorted(TASKS), required=True)
    p1.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p1.add_argument("--repeats", type=int, default=4, help="independent processes per side (cross-process noise)")
    p2 = sub.add_parser("fault", help="R0 and warm-start W for one fault")
    p2.add_argument("--task", choices=sorted(TASKS), required=True)
    p2.add_argument("--fault", choices=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"], required=True)
    p2.add_argument("--budgets", type=int, nargs="+", default=[30, 100, 300, 1000])
    p2.add_argument("--tmax", type=float, nargs="+", default=[10, 1])
    p2.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p2.add_argument("--jobs", type=int, default=10)
    p2.add_argument("--no-full", action="store_true")
    p3 = sub.add_parser("matrix", help="every task x fault incl. FULL, child processes on one shared pool")
    p3.add_argument("--cases", nargs="+", default=["ffn:F1", "ffn:F2", "ffn:F3", "ffn:F4", "ffn:F5",
                                                    "mha:F1", "mha:F2", "mha:F3", "mha:F4"])
    p3.add_argument("--budgets", type=int, nargs="+", default=[30, 100, 300, 1000])
    p3.add_argument("--tmax", type=float, nargs="+", default=[10, 1])
    p3.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p3.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    from concurrent.futures import ThreadPoolExecutor
    if args.cmd == "step1":
        step1(args.task, args.seeds, args.repeats)
    elif args.cmd == "fault":
        with ThreadPoolExecutor(args.jobs) as ex:
            run_fault(args.task, args.fault, args.budgets, args.tmax, args.seeds, ex, full=not args.no_full)
    elif args.cmd == "matrix":
        run_matrix(args.cases, args.budgets, args.tmax, args.seeds, args.jobs)


def run_matrix(cases, budgets, tmaxes, seeds, jobs):
    from concurrent.futures import ThreadPoolExecutor
    import csv
    t0 = time.time()
    out, errors = {}, {}
    with ThreadPoolExecutor(jobs) as ex, ThreadPoolExecutor(len(cases)) as outer:
        futs = {c: outer.submit(run_fault, c.split(":")[0], c.split(":")[1], budgets, tmaxes, seeds, ex)
                for c in cases}
        for c, fu in futs.items():
            try:
                out[c] = fu.result()
            except E3Error as exc:
                errors[c] = str(exc)
                print(f"[e3] STOP {c}: {exc}", flush=True)
    rows = [out[c]["table_row"] for c in cases if c in out]
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(OUT, "e3_table.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    keys = ["fault_cost", "R0_over_FULL", "best_W", "best_W_over_FULL", "reach5_steps", "reach5_sa_wall",
            "reach2_steps", "FULL_total_process_wall", "tightness", "LB_dominant", "check2", "check3", "check4"]
    final = {c: {k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k)) for k in keys}
             for c, r in ((c, out[c]["table_row"]) for c in cases if c in out)}
    final["_wall_seconds"] = round(time.time() - t0)
    final["_errors"] = errors
    print("[e3] " + json.dumps(final, default=str), flush=True)
    if errors:
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except E3Error as exc:
        print(f"[e3] STOP: {exc}", flush=True)
        sys.exit(2)
