"""EIR main experiment: tiered incremental fault repair (eir/EIR_HANDOFF.md).

One task script per process, as cpcm_sa / E3: the child imports the artifact's modules, applies E3's in-memory patches
(fault injection, mapping and route restoration, R0's relocation) and replaces StreamOptimization.anneal; the template
then runs up to stream_mapping, where the annealer holds the injected state. The child exits there (a repair never
reaches the template's own final evaluation).

Jobs (kind):
  pre    start mapping on the fault-free platform: T, link busy / CB ranking, reference for the migration metrics
  l1     R0 + L1 for one start x fault; saves both states; everything later starts from this file, so W, L2-Global and
         L2-Local get the very same L1 output
  rep    W / L2G / L2L from the L1 output: calibration, search, best state, backend seeds
  full_e3  E3's best FULL state rebuilt (verified identical, PROGRESS step 2), backend seeds = FULL-BusyBarn
  full   search from scratch on the faulty platform (cpcm_sa's A3h = FULL-A3h)
  verify E3's R0 / FULL states rebuilt and compared with E3's records (handoff 3.1)
  select E3's choose_fault on the A3h start itself: F4a, F5a (handoff 12.1)
  ctrl   L2-Global's rules on the healthy platform from the fault-free start (handoff 12.2)
  reload check: a saved final state loaded in a new process gives the same T and state (smoke3.py)

Next stage (EIR_NEXT_HANDOFF.md, driver eir_next.py): methods L2Lsync, L2Gsync, L2Impact in `rep` use the corrected DDR rule
(section 3) and the impact-ranked set (4.1). The methods L2G, L2L, W do not reach any of that code.

Usage (from eir/):
  python eir.py prepare                  faults.json and inputs/ (needs E3's and stage 2's files; local only)
  python eir.py verify --jobs 12
  python eir.py run --part ffn|mha|ffn14|a3h --out out/main --workers 14
  python eir.py report --out out/main
"""
import argparse
import collections
import copy
import gzip
import importlib
import json
import math
import os
import pickle
import random
import re
import runpy
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ.setdefault("CPCM_SA_STAGE", "s2")     # stage 2 rules: exact revert + T with the cached broadcast
sys.path.insert(0, os.path.join(ROOT, "cpcm_sa"))
import cpcm_sa as cs  # noqa: E402

e3, cpcm = cs.e3, cs.cpcm
WORK = os.path.join(HERE, "work", "artifact")
cs.WORK = WORK
e3.WORK = WORK
INPUTS = os.path.join(HERE, "inputs")
FAULTS_JSON = os.path.join(HERE, "faults.json")
GPT, SUF = cs.GPT, cs.SUF

# ---------------------------------------------------------------- experiment definition (handoff sections 2-6)
# A3h mappings are injected into the same template file as the E3 start of that operator: the files differ only in
# loss_ratio (checked with diff, 2026-11-05), which matters only for W's energy, so W uses BusyBarn's winning ratio
# on both starts.
TASKS = {"ffn": "ffn_sq2048_greedy0_lossratio6_ch2x2" + SUF, "mha": "mha_sq2048_greedy0_lossratio4_ch2x2" + SUF,
         "ffn14": "ffn_sq2048_greedy1_lossratio4_ch1x4" + SUF}
A3H_RUNS = {"ffn": ("ffn", 0, 5), "mha": ("mha", 0, 2)}       # backend-best A3h run of the fault-free config
FIDS = ["F1", "F2", "F4", "F5", "F6", "F7", "F8"]
COMBOS = [("e3", op, f) for op in ("ffn", "mha", "ffn14") for f in FIDS] + \
         [("a3h", op, f) for op in ("ffn", "mha") for f in ("F4", "F5")]
METHODS = ["W", "L2G", "L2L"]
SA_SEEDS = [123, 1, 2, 3, 4]
MAIN_STEPS, HARD_STEPS, HARD_FAULTS = (30, 100), (300,), ("F4", "F5", "F4a", "F5a")
CTRL_STEPS = (30, 100, 300)
CTRL_STARTS = [("a3h", "ffn"), ("a3h", "mha"), ("e3", "ffn"), ("e3", "mha"), ("e3", "ffn14")]
L1_N, L1_MAX_ATTEMPTS, L1_ALTS = 10, 300, 3
EPS = 0.02
CAL_PROPOSALS, CAL_ACCEPT, TMIN_RATIO, CAL_SEED_OFFSET = 15, 0.10, 1e-4, 100003
P, K = cs.P, cs.K
# local-set rule, fixed by the user 2026-11-05 ("本地开跑，C、R=2", PROGRESS.md): seeds = failed parts + relocations
LOCAL_R = 2
SEED_RULE = "fault_relocated"
# next stage: which methods use the corrected DDR rule, and which local set a method has (no entry: no set)
SYNC_METHODS = ("L2Lsync", "L2Gsync", "L2Impact")
LOCAL_METHODS = {"L2L": "geo", "L2Lsync": "geo", "L2Impact": "impact"}
BACKEND_SEEDS = list(range(1, int(os.environ.get("EIR_BACKEND_SEEDS", "10")) + 1))
cs.BACKEND_SEEDS = BACKEND_SEEDS
BCAST_SEED = cs.BCAST_SEED
E3_OUT = os.path.join(ROOT, "e3_warmstart", "e3_out")
S2_OUT = os.path.join(ROOT, "cpcm_sa", "out", "s2")


def load_faults():
    with open(FAULTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def fault_of(op, fid):
    return load_faults()[op][fid]


def a_combos():
    """Handoff 12.1: F4a, F5a of the A3h starts that differ from E3's coordinates (after `eir.py select`)."""
    t = load_faults()
    return [("a3h", op, f) for op in A3H_RUNS for f in ("F4a", "F5a") if f in t[op] and not t[op][f].get("same_as")]


def full_cfg(op, fid):
    return f"{op}_{fid.lower()}"


def combo_name(start, op, fid):
    return f"{start}_{op}_{fid}"


def rep_name(start, op, fid, method, steps, seed):
    return f"{combo_name(start, op, fid)}_{method}_s{steps}_r{seed}"


def steps_of(fid):
    return MAIN_STEPS + (HARD_STEPS if fid in HARD_FAULTS else ())


def _plain(x):
    return e3._plain(x)


# ---------------------------------------------------------------- engine on the annealer
class Engine:
    """Everything the repair levels share: T with the cached broadcast, exact revert, finalisation."""

    def __init__(self, ann, add_broadcast_e3, backend, faults):
        self.a = ann
        self.ctx = cs._find_context()
        self.hp = ann.hardware_platform
        self.bc = cs.BcCache(self.ctx, self.hp)
        self.add_broadcast_e3 = add_broadcast_e3
        self.backend = backend
        self.faults = faults
        self.g = self.sc = self.an = self.ev = None
        self.ncpcm = 0
        self.sync = False       # corrected DDR rule on: the fingerprint then also covers generated_split_location

    # ---- evaluation
    def t_eval(self):
        s = self.a.state
        ev = self.bc.graph(s[5], s[6], s[7], self.hp)
        g, sc = cs.cpcm_sim(ev, self.hp)
        self.ncpcm += 1
        self.g, self.sc, self.an, self.ev = g, sc, None, ev
        return sc.T

    def analysis(self):
        if self.an is None:
            self.an = cpcm.Analysis(self.sc)
        return self.an

    def W(self, e):
        return self.sc.S_last[e] - self.sc.r[e]

    # ---- exact revert and snapshots (cs.anneal's before_move / exact_revert, with the cache)
    def before(self):
        hp = self.hp
        return {"blob": pickle.dumps(self.a.state, protocol=pickle.HIGHEST_PROTOCOL),
                "loads": dict(hp.link_loads_dict), "counts": dict(hp.link_loads_count), "bc": dict(self.bc.routes),
                "st": (self.g, self.sc, self.an, self.ev)}

    def revert(self, pre):
        hp = self.hp
        self.a.state = pickle.loads(pre["blob"])
        hp.link_loads_dict.update(pre["loads"])
        hp.link_loads_count.update(pre["counts"])
        self.bc.routes = dict(pre["bc"])
        self.g, self.sc, self.an, self.ev = pre["st"]

    def snapshot(self):
        hp = self.hp
        return {"blob": pickle.dumps(self.a.state, protocol=pickle.HIGHEST_PROTOCOL),
                "loads": dict(hp.link_loads_dict), "counts": dict(hp.link_loads_count), "bc": dict(self.bc.routes)}

    def load(self, snap):
        hp = self.hp
        self.a.state = pickle.loads(snap["blob"])
        hp.link_loads_dict.clear()
        hp.link_loads_dict.update(snap["loads"])
        hp.link_loads_count.clear()
        hp.link_loads_count.update(snap["counts"])
        self.bc.routes = dict(snap["bc"])

    def fingerprint(self):
        hp, s = self.hp, self.a.state
        evs = []
        for t, e in s[7].items():
            row = [t, sorted(map(str, e.dependency_set)), sorted(map(str, e.issue_set))]
            if e.event_type == "communication":
                row += [e.source_location, str(e.target_location), e.comm_bytes, str(e.path_list), str(e.paths),
                        e.hops, e.communication_distances]
            else:
                a = e.commid_allocator
                row += [e.comp_location, sorted(a.used_ids), sorted(a.recycled), a.next_id]
            evs.append(row)
        fp = (evs, {t: b.location for t, b in s[5].items()}, s[0], s[1], dict(s[2]), dict(s[3]), dict(s[4]),
              dict(hp.link_loads_dict), dict(hp.link_loads_count), {k: (r[0], str(r[1])) for k, r in self.bc.routes.items()})
        if self.sync:
            fp += ({d: {sp: list(x) for sp, x in t.generated_split_location.items()} for d, t in s[6].items()},)
        return fp

    # ---- broadcast routes of the injected state
    def seed_broadcast(self):
        """Fill the cache with the broadcast routes E3's add_broadcast wrapper gives (captured routes where they
        survive, Dijkstra otherwise, RNG seeded 123 as E3's R0 does where SA would start)."""
        from data_notation import type_bytes
        s, ctx = self.a.state, self.ctx
        rng = random.getstate(), np.random.get_state()
        random.seed(123)
        np.random.seed(123)
        beha, data, ev, hp2 = copy.deepcopy((s[5], s[6], s[7], self.hp))
        before = set(ev)
        self.add_broadcast_e3(data_tags=ctx["broadcast_datatags"],
                              ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]], beha_dict=beha,
                              data_dict=data, hardware_platform=hp2, event_dict=ev,
                              dijkstra_routing=ctx["dijkstra_routing"], alpha=ctx["alpha"], beta=ctx["beta"],
                              gamma=ctx["gamma"])
        T_wrapped = cs.cpcm_sim(ev, hp2)[1].T
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        bucket = collections.defaultdict(list)
        for t in ev:
            if t not in before:
                e = ev[t]
                bucket[(next(iter(e.dependency_set)), e.comm_bytes)].append(e)
        mismatch = 0
        for dt in ctx["broadcast_datatags"]:
            for prod, splits in s[6][dt].generated_tag_splitted_dict.items():
                loc = s[7][prod].comp_location[:-1]
                for sp in splits:
                    nb = int(np.prod([e_ - s_ + 1 for (s_, e_) in sp]) * type_bytes[s[6][dt].data_type])
                    e = bucket[(prod, nb)].pop(0)
                    if tuple(e.source_location) != tuple(loc):
                        mismatch += 1
                        continue
                    w = [((r_, l_), self.hp.current_link_loads(e.comm_bytes, self.hp.links_dict[(r_, l_)]))
                         for r_, leaves in e.paths.items() for l_ in leaves]
                    self.bc.routes[(dt, prod, sp)] = (loc, e.paths, e.path_list, w)
        return {"T_wrapped": T_wrapped, "source_mismatch": mismatch, "n_routes": len(self.bc.routes)}

    # ---- final evaluation of the current state
    def finalize(self, seeds=None):
        s, hp = self.a.state, self.hp
        t0 = time.perf_counter()
        ev = self.bc.graph(s[5], s[6], s[7], hp)
        g, sc = cs.cpcm_sim(ev, hp)
        self.ncpcm += 1
        t_final = time.perf_counter() - t0
        pre = e3._e2()._snapshot_pre(ev)
        rec = {"T": sc.T, "t_finalize": t_final, "routes": e3.routes_of(ev, pre),
               "beha_loc": {_plain(t): _plain(b.location) for t, b in s[5].items()},
               "placement": placement(self.a), "fault_violations": fault_violations(self, ev),
               "sum_comm_bytes_hops": s[0], "dist": s[1]}
        rng = random.getstate(), np.random.get_state()
        bk, walls = {}, []
        t0 = time.perf_counter()
        base = copy.deepcopy((ev, hp))
        for sd in (BACKEND_SEEDS if seeds is None else seeds):
            ev_k, hp_k = copy.deepcopy(base)
            random.seed(sd)
            np.random.seed(sd)
            t1 = time.time()
            bk[sd] = tuple(int(v) for v in self.backend(events_dict=ev_k, hardware_platform=hp_k))
            walls.append(time.time() - t1)
            self.last_cap = (ev_k, hp_k, pre, bk[sd])
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        rec.update(backend=bk, backend_wall=walls, t_backend=time.perf_counter() - t0)
        return rec


def placement(a):
    """(weight data, split) -> DDR nodes it is actually read from: the sources of the live read communications
    (generated_split_location is stale after SA, CLAUDE.md), with cs.weight_split_of's rule for recycled tags."""
    s = a.state
    ddr = {k[:-1] for k in a.hardware_platform.ddr_dict}
    out = collections.defaultdict(set)
    for t, e in s[7].items():
        if e.event_type == "communication" and tuple(e.source_location) in ddr:
            ws = cs.weight_split_of(a, t)
            if ws is not None:
                out[ws].add(tuple(e.source_location))
    return dict(out)


def sync_locations(a):
    """Corrected DDR rule, entry (next handoff section 3): generated_split_location of a weight becomes where it is really
    read from. A split whose sources are all among its locations is left alone: its other locations are real copies (the
    input activation sits on every port). T does not read these lists (asserted by search)."""
    out = {"splits": 0, "synced": 0, "multi_copy": 0}
    for (d, sp), srcs in placement(a).items():
        t = a.state[6][d]
        locs = {tuple(x) for x in t.generated_split_location[sp]}
        out["splits"] += 1
        out["multi_copy"] += len(locs) > 1
        if not set(srcs) <= locs:
            t.generated_split_location[sp] = sorted(srcs)
            out["synced"] += 1
    return out


def sync_after_weight(a, d, sp, dest):
    """After a weight move: all reads of the split now come from dest. A single copy moved; on a multi-copy split dest is
    one of its real copies and the location list stays (a port not listed yet gets added)."""
    t, node = a.state[6][d], tuple(dest[:-1])
    locs = t.generated_split_location[sp]
    if len(locs) > 1:
        if node not in {tuple(x) for x in locs}:
            locs.append(node)
    else:
        t.generated_split_location[sp] = [node]


def stale_locations(a):
    """Weights read from a DDR that their location list does not name (0 under the corrected rule)."""
    return sum(not set(srcs) <= {tuple(x) for x in a.state[6][d].generated_split_location[sp]} for (d, sp), srcs in placement(a).items())


def split_bytes_of(a, d, sp):
    from data_notation import type_bytes
    return int(np.prod([e - s + 1 for (s, e) in sp]) * type_bytes[a.state[6][d].data_type])


def fault_violations(eng, ev):
    """Check 2: no computation, data or route on a failed core, PE or link, also in the data locations."""
    f = eng.faults
    fn = set(map(tuple, f["failed_nodes"]))
    fp = set(map(tuple, f.get("failed_pes", [])))
    fl = {(tuple(a), tuple(b)) for a, b in f["failed_links"]}
    s, bad = eng.a.state, []
    for t, b in s[5].items():
        if tuple(b.location[:-1]) in fn | fp:
            bad.append(("beha", str(t)))
    for d, t_ in s[6].items():
        for sp, locs in t_.generated_split_location.items():
            if any(tuple(n) in fn | fp for n in locs):
                bad.append(("data", str(d), str(sp)))
    for t, e in ev.items():
        if e.event_type == "computation":
            if tuple(e.comp_location[:-1]) in fn | fp:
                bad.append(("comp_event", str(t)))
            continue
        tg = e.target_location if isinstance(e.target_location, set) else {e.target_location}
        if tuple(e.source_location) in fn | fp or {tuple(x) for x in tg} & fn:
            bad.append(("comm_endpoint", str(t)))
        for r, leaves in e.paths.items():
            for l in leaves:
                if (tuple(r), tuple(l)) in fl or tuple(r) in fn or tuple(l) in fn:
                    bad.append(("route", str(t), str(r), str(l)))
    return bad


# ---------------------------------------------------------------- migration metrics (handoff section 7)
def migration_metrics(ref, now, ddr_nodes, sizes):
    moved = [t for t, loc in ref["beha_loc"].items() if tuple(loc[:-1]) != tuple(now["beha_loc"][t][:-1])]
    wm = e3.weight_migration(ref["routes"], now["routes"], ddr_nodes)
    uniq = moved_blocks = 0
    for k, locs in now["placement"].items():
        new = set(locs) - set(ref["placement"].get(k, ()))
        if new:
            uniq += sizes[k] * len(new)
            moved_blocks += len(new)
    return {"moved_behas": len(moved), "n_behas": len(ref["beha_loc"]), "moved_behas_share": len(moved) / len(ref["beha_loc"]),
            "ddr_read_moved_bytes": wm["moved_read_bytes"], "ddr_read_total_bytes": wm["read_bytes"],
            "ddr_read_moved_share": wm["moved_read_share"], "unique_block_bytes": uniq, "unique_blocks": moved_blocks}


# ---------------------------------------------------------------- local set (handoff 4.2)
def ddr_corner(node, hp):
    """Core a DDR port hangs off (node (die, 16 + k), order as e3.ddr_replacement); a core maps to itself."""
    w, h = hp.l1_width, hp.l1_height
    if node[1] < h * w:
        return tuple(node)
    return node[0], [0, 0, w - 1, w - 1, (h - 1) * w, (h - 1) * w, h * w - 1, h * w - 1][node[1] - h * w]


def _leaf_paths(src, tree):
    kids = {tuple(r): [tuple(l) for l in leaves] for r, leaves in tree}
    out, stack = {}, [(tuple(src), ())]
    while stack:
        u, p = stack.pop()
        if not kids.get(u):
            if p:
                out[u] = p
            continue
        stack += [(v, p + ((u, v),)) for v in kids[u]]
    return out


def changed_endpoints(ref_routes, now_routes):
    """Nodes at the ends of communications whose (source, route) differ from the reference. Multicast: the source and
    only the targets whose own path changed (a broadcast reaches every DDR port of the group; counting all of them
    would make every link fault cover the whole group)."""
    canon = lambda src, tree: (tuple(src), tuple(sorted((tuple(r), tuple(sorted(map(tuple, ls)))) for r, ls in tree)))
    pool = {k: [canon(x[0], x[1]) for x in v] for k, v in ref_routes.items()}
    by_rest = collections.defaultdict(list)
    for k, v in ref_routes.items():
        by_rest[k[1:]] += v
    nodes, n = set(), 0
    for k, v in now_routes.items():
        for src, tree, _ in v:
            c = canon(src, tree)
            if c in pool.get(k, []):
                pool[k].remove(c)
                continue
            n += 1
            nodes.add(tuple(src))
            if isinstance(k[0], tuple) and k[0] and k[0][0] == "set":
                olds = by_rest.get(k[1:]) or []
                old = _leaf_paths(olds[0][0], olds[0][1]) if olds else {}
                for leaf, p in _leaf_paths(src, tree).items():
                    if old.get(leaf) != p:
                        nodes.add(leaf)
            else:
                nodes.add(tuple(k[0]))
    return nodes, n


class LocalSet:
    """Fixed local neighbourhood, computed once at L2 entry; no expansion during the search."""

    def __init__(self, ann, seeds, R, nodes=None):
        """nodes: the cores given directly (L2-Impact, seeds and R unused); the DDR ports, behaviours and splits follow
        from them by the same rules as for a geometric set."""
        hp = ann.hardware_platform
        s = ann.state
        self.R = R
        cores = sorted({tuple(k[:-1]) for d in ("tensorcore", "vectorunit") for k in hp.modules_dict[d]})
        if nodes is not None:
            self.nodes = {tuple(n) for n in nodes}
        else:
            seeds = {ddr_corner(tuple(n), hp) for n in seeds}
            xy = [e3._grid_xy(n, hp) for n in seeds]
            self.nodes = {n for n in cores
                          if any(abs(e3._grid_xy(n, hp)[0] - y) + abs(e3._grid_xy(n, hp)[1] - x) <= R for y, x in xy)}
        self.behas = {t for t, b in s[5].items() if tuple(b.location[:-1]) in self.nodes}
        self.ddr_ports = {tuple(k[:-1]) for k in hp.ddr_dict if ddr_corner(tuple(k[:-1]), hp) in self.nodes}
        self.splits = set()
        ev7 = s[7]
        ddr = {tuple(k[:-1]) for k in hp.ddr_dict}
        for d in ann.offlinedata_list:
            for sp in ann.offlinedata_split_dict[d]:
                srcs = {tuple(ev7[t].source_location) for t in s[6][d].used_splitted_tag_dict[sp]
                        if t in ev7 and t[0] == 1 and tuple(ev7[t].source_location) in ddr}
                if srcs and srcs <= self.ddr_ports:
                    self.splits.add((d, sp))
        self.n_cores = len(cores)
        self.n_behas = len(s[5])
        self.n_splits = sum(len(ann.offlinedata_split_dict[d]) for d in ann.offlinedata_list)

    def coverage(self):
        return {"R": self.R, "beha_share": len(self.behas) / self.n_behas, "core_share": len(self.nodes) / self.n_cores,
                "n_behas": len(self.behas), "n_cores": len(self.nodes), "n_ddr_ports": len(self.ddr_ports),
                "n_splits": len(self.splits), "splits_share": len(self.splits) / max(1, self.n_splits)}


def changed_tags(ref_routes, now_routes):
    """Tags of the communications changed_endpoints() counts as changed (same matching)."""
    canon = lambda src, tree: (tuple(src), tuple(sorted((tuple(r), tuple(sorted(map(tuple, ls)))) for r, ls in tree)))
    pool = {k: [canon(x[0], x[1]) for x in v] for k, v in ref_routes.items()}
    out = []
    for k, v in now_routes.items():
        for src, tree, tag in v:
            c = canon(src, tree)
            if c in pool.get(k, []):
                pool[k].remove(c)
            else:
                out.append(tag)
    return out


def impact_select(eng, base, pre, pre_w):
    """L2-Impact's cores (next handoff 4.1), once, at the L1 state, no parameter.

    Candidates: endpoints of the communications changed between the fault-free mapping and the L1 state, the relocated
    objects, the behaviours producing and consuming all of these, and the DDR reads into those behaviours, as cores.
    Group 1 = the fault-direct cores (rule C's seeds), group 2 = the other candidates. Within a group by score: over the
    events of the core, max(0, W_L1 - W_pre) + phi (phi summed over resources). Ties: CB of the links ending at the core,
    core id. A computation belongs to its core; a communication to its source and, if unicast, its target (a DDR port
    counts as its corner core; a broadcast reaches every port, so only the source, the same reason as changed_endpoints).
    The first K cores, K = the core count of the geometric L2-Local of this combination."""
    a, hp = eng.a, eng.hp
    if pre_w["T"] != pre["T"]:
        raise RuntimeError(f"pre rerun T {pre_w['T']}, reference pre T {pre['T']}")
    eng.t_eval()
    an, g, ev = eng.analysis(), eng.g, eng.ev
    an.critical_path()
    cores = {tuple(k[:-1]) for d in ("tensorcore", "vectorunit") for k in hp.modules_dict[d]}
    ddr = {tuple(k[:-1]) for k in hp.ddr_dict}
    to_core = lambda n: ddr_corner(tuple(n), hp)
    geo = LocalSet(a, base["seed_nodes"], LOCAL_R)
    K = len(geo.nodes)
    g1 = {to_core(n) for n in base["seed_parts"]["fault_and_relocated"]} & cores
    now_routes = base["L1"]["routes"]
    ends, n_changed = changed_endpoints(pre["record"]["routes"], now_routes)
    cand = {to_core(n) for n in ends}
    tags = changed_tags(pre["record"]["routes"], now_routes)
    missing = 0
    behas = set()
    is_comp = lambda x: x in ev and ev[x].event_type == "computation"

    def neighbours(t):
        """Computation behaviours around an event: directly, or through a communication."""
        e, out = ev[t], set()
        for x in list(e.dependency_set) + list(e.issue_set):
            if is_comp(x):
                out.add(x)
            elif x in ev:
                out |= {y for y in list(ev[x].dependency_set) + list(ev[x].issue_set) if is_comp(y) and y != t}
        return out

    for t in tags:
        if t not in ev:
            missing += 1
            continue
        behas |= {x for x in list(ev[t].dependency_set) + list(ev[t].issue_set) if is_comp(x)}
    for t, old, new in base["log"].get("relocated") or []:
        cand |= {to_core(old[:-1]), to_core(new[:-1])}
        behas.add(t)
        behas |= neighbours(t)
    for d, sp, dead, new in base["log"].get("ddr_data_moved") or []:
        cand |= {to_core(n) for n in dead}
        if len(new) == 1:      # really moved; several = a copy dropped (the same reading as child_l1's seeds)
            cand.add(to_core(new[0]))
        for t in a.state[6][d].used_splitted_tag_dict.get(sp, ()):
            if t in ev and ev[t].event_type == "communication":
                behas |= {x for x in ev[t].issue_set if is_comp(x)}
    for b in behas:
        cand.add(to_core(ev[b].comp_location[:-1]))
        for x in ev[b].dependency_set:
            if x in ev and ev[x].event_type == "communication" and tuple(ev[x].source_location) in ddr:
                cand.add(to_core(ev[x].source_location))
    cand = (cand | g1) & cores
    g2 = cand - g1
    phi = collections.Counter()
    for dq in an.phi.values():
        for b, v in dq.items():
            phi[b] += v
    W_pre = pre_w["W_by_tag"]
    score, unaligned = collections.Counter(), 0
    for e in range(g.n):
        tag = g.tag[e]
        x = ev[tag]
        if tag not in W_pre:
            unaligned += 1
        v = max(0, eng.W(e) - W_pre.get(tag, 0)) + phi[e]
        if x.event_type == "computation":
            where = {to_core(x.comp_location[:-1])}
        else:
            where = {to_core(x.source_location)}
            if not isinstance(x.target_location, set):
                where.add(to_core(x.target_location))
        for c in where:
            score[c] += v
    cb = collections.Counter()
    for q, v in an.blocking().items():
        if g.res_name[q][0] == "link":
            for n in g.res_name[q][1][:2]:
                cb[to_core(n)] += v
    order = lambda grp: sorted(grp, key=lambda c: (-score[c], -cb[c], c))
    rank = order(g1) + order(g2)
    nodes = rank[:K]
    return {"K": K, "nodes": sorted(nodes), "rank": rank, "n_cand": len(cand), "n_g1": len(g1), "n_g2": len(g2),
            "chosen_g1": sum(c in g1 for c in nodes), "chosen_g2": sum(c in g2 for c in nodes),
            "overlap_geo": len(set(nodes) & geo.nodes), "geo_nodes": sorted(geo.nodes), "geo_cov": geo.coverage(),
            "score": {c: score[c] for c in rank}, "cb": {c: cb[c] for c in rank}, "n_changed": n_changed, "tags_changed": len(tags),
            "tags_missing_in_graph": missing, "n_events": g.n, "n_unaligned_W": unaligned, "n_behas_around": len(behas)}


class Heur(cs.Heur):
    """cs.Heur with candidates limited to the local set when there is one; with none it is cs.Heur."""

    def __init__(self, ann, local):
        super().__init__(ann)
        self.local = local
        self.cand_log = None       # sync methods: one row per comp_dest call (read only, for the diagnosis)

    def comp_dest(self, beha, primary=None):
        if self.local is None:
            return super().comp_dest(beha, primary)
        kind = self.a.state[7][beha].comp_device
        load = lambda c: self.dev_load(kind, c)
        cands = [c for c in cs.comp_cands(self.a, beha) if tuple(c[:-1]) in self.local.nodes]
        if not cands:
            if self.cand_log is not None:
                self.cand_log.append({"n": 0, "idle": 0, "load": None})
            return None
        half = self.light_half(cands, load)
        key = lambda c: ((primary(c[:-1]) if primary else 0), self.nbr_cost(beha, c[:-1]), load(c), c)
        chosen = random.choice(sorted(half, key=key)[:self.TOP_MOVE])
        if self.cand_log is not None:
            self.cand_log.append({"n": len(set(cands)), "idle": len({c for c in cands if load(c) == 0}), "load": float(load(chosen))})
        return chosen

    def ddr_dest(self, d, sp):
        if self.local is None:
            return super().ddr_dest(d, sp)
        ev7 = self.a.state[7]
        readers = [ev7[t] for t in self.a.state[6][d].used_splitted_tag_dict[sp]
                   if t in ev7 and t[0] == 1 and ev7[t].event_type == "communication"]
        rd = collections.Counter()
        for e in ev7.values():
            if e.event_type == "communication":
                rd[tuple(e.source_location)] += e.comm_bytes
        load = lambda c: rd[tuple(c[:-1])]
        cands = [c for c in cs.ddr_cands(self.a, d) if tuple(c[:-1]) in self.local.ddr_ports]
        if not cands:
            return None
        half = self.light_half(cands, load)
        dist = lambda c: sum(self.hop(c[:-1], r.target_location) for r in readers
                             if not isinstance(r.target_location, set))
        return random.choice(sorted(half, key=lambda c: (dist(c), load(c), c))[:self.TOP_MOVE])

    def swap_partner(self, b1, cp_tags, W):
        if self.local is None:
            return super().swap_partner(b1, cp_tags, W)
        a = self.a
        pool = [b for b in a.comp_tags_list if b != b1 and abs(b[2] - b1[2]) <= 1 and cs.region_ok(a, b)
                and b in self.local.behas]
        if not pool:
            return None
        non = [b for b in pool if b not in cp_tags]
        pool = non or pool
        ev7 = a.state[7]
        ld = {b: self.dev_load(ev7[b].comp_device, ev7[b].comp_location) for b in pool}
        rw = {b: i for i, b in enumerate(sorted(pool, key=lambda b: (W(b), b)))}
        rl = {b: i for i, b in enumerate(sorted(pool, key=lambda b: (ld[b], b)))}
        return random.choice(sorted(pool, key=lambda b: (rw[b] + rl[b], b))[:self.TOP_SWAP])


class Moves:
    """A3h's proposal (cs.make_anneal's targeted + heuristic_random). With local None the random-number use and every
    choice are the same as there; with a local set every branch draws only inside it."""

    def __init__(self, eng, local, sync=False):
        self.eng, self.local, self.sync = eng, local, sync
        self.a = eng.a
        self.heur = Heur(eng.a, local)
        self.ddr_nodes = {tuple(k[:-1]) for k in eng.hp.ddr_dict}
        self.diag = None         # sync methods: what this proposal's branches did (read only, no random numbers)

    def _comp_move(self, b, primary=None, action="comp"):
        dest = self.heur.comp_dest(b, primary)
        if dest is None:
            return None
        cs.mv_relocate(self.a, b, dest)
        return {"action": action, "beha": b, "dest": dest}

    def _weight_move(self, d, sp, action="weight"):
        dest = self.heur.ddr_dest(d, sp)
        if dest is None:
            return None
        cs.mv_weight(self.a, d, sp, dest)
        if self.sync:
            sync_after_weight(self.a, d, sp, dest)
            # update_offlinedata moves every tag listed under the split, also a recycled tag now carrying another split's read
            # (CLAUDE.md, used_splitted_tag_dict): that split's list follows too
            sync_locations(self.a)
        return {"action": action, "split": (d, sp), "dest": dest}

    def _eligible(self, tag):
        """Local: can this critical event move anything inside the set?"""
        loc, s = self.local, self.a.state
        ev = s[7][tag] if tag in s[7] else self.eng.ev[tag]
        if ev.event_type == "computation":
            return tag in loc.behas
        ws = cs.weight_split_of(self.a, tag) if tuple(ev.source_location) in self.ddr_nodes else None
        if ws is not None and ws in loc.splits:
            return True
        return any(t in loc.behas for t in ev.dependency_set) or any(t in loc.behas for t in ev.issue_set)

    def _classify(self, top):
        """Diagnosis: the kinds of the events the targeted draw picks from (no random numbers, no change to the state)."""
        a, g, eng = self.a, self.eng.g, self.eng
        out = collections.Counter(n=len(top))
        for e in top:
            tag = g.tag[e]
            ev = a.state[7][tag] if tag in a.state[7] else eng.ev[tag]
            if ev.event_type == "computation":
                out["comp"] += 1
            elif tuple(ev.source_location) in self.ddr_nodes:
                out["ddr_read"] += 1
                ws = cs.weight_split_of(a, tag)
                if ws is not None and (self.local is None or ws in self.local.splits):
                    out["ddr_read_ws_in_set"] += 1
            else:
                out["core_comm"] += 1
        return dict(out)

    def targeted(self):
        eng, a = self.eng, self.a
        an, g = eng.analysis(), eng.g
        evs = {r["event"] for r in an.critical_path()}
        if self.diag is not None:
            cp = collections.Counter()
            for e in evs:
                tag = g.tag[e]
                ev = a.state[7][tag] if tag in a.state[7] else eng.ev[tag]
                cp["comp" if ev.event_type == "computation" else "ddr_read" if tuple(ev.source_location) in self.ddr_nodes else "core_comm"] += 1
            self.diag["cp"] = dict(cp)
        if self.local is not None:
            evs = {e for e in evs if self._eligible(g.tag[e])}
            if self.diag is not None:
                self.diag["eligible"] = len(evs)
            if not evs:
                return None
        top = sorted(evs, key=lambda e: (-eng.W(e), e))[:K]
        if self.diag is not None:
            self.diag["top"] = self._classify(top)
        e = random.choice(top)
        tag = g.tag[e]
        ev = a.state[7][tag] if tag in a.state[7] else eng.ev[tag]
        info = {"event": str(tag), "W": eng.W(e), "n": len(evs)}
        if ev.event_type == "computation":
            if self.diag is not None:
                self.diag["chosen"] = "comp"
            r = self._comp_move(tag)
            return None if r is None else {**info, **r}
        ws = cs.weight_split_of(a, tag) if tuple(ev.source_location) in self.ddr_nodes else None
        if self.diag is not None:
            self.diag["chosen"] = "ddr_read_ws_in_set" if ws is not None and (self.local is None or ws in self.local.splits) else "ddr_read_ws_out" if ws is not None else "ddr_read_no_ws" if tuple(ev.source_location) in self.ddr_nodes else "core_comm"
        if ws is not None and (self.local is None or ws in self.local.splits):
            r = self._weight_move(*ws)
            if r is not None or self.local is None:
                if self.diag is not None:
                    self.diag["weight_returned"] = r is not None
                return {**info, **(r or {})}
        ok = (lambda t: True) if self.local is None else (lambda t: t in self.local.behas)
        prods = sorted((t for t in ev.dependency_set if t in a.state[5] and ok(t)), key=str)
        cons = sorted((t for t in ev.issue_set if t in a.state[5] and ok(t)), key=str)
        sides = [x for x in (("producer", prods), ("consumer", cons)) if x[1]]
        if not sides:
            return None
        side, behas = random.choice(sides)
        b = random.choice(behas)
        if side == "consumer":
            src = ev.source_location
            prim = lambda c: self.heur.hop(src, c)
        else:
            tgt = ev.target_location
            prim = (lambda c: sum(self.heur.hop(c, n) for n in tgt)) if isinstance(tgt, set) \
                else (lambda c: self.heur.hop(c, tgt))
        r = self._comp_move(b, prim, side)
        return None if r is None else {**info, **r}

    def rand(self):
        a, eng, loc = self.a, self.eng, self.local
        change_flag = random.random() < a.random_threshold[0]
        change_mem = random.random() < a.mem_threshold[0]
        if change_flag:
            if change_mem and a.mem_enable:
                if loc is None:
                    d = random.choice(a.offlinedata_list)
                    sp = random.choice(a.offlinedata_split_dict[d])
                    return self._weight_move(d, sp, "rand_weight")
                if loc.splits:
                    d, sp = random.choice(sorted(loc.splits, key=str))
                    r = self._weight_move(d, sp, "rand_weight")
                    if r is not None:
                        return r
            pool = a.comp_tags_list if loc is None else sorted(loc.behas, key=str)
            if not pool:
                return None
            return self._comp_move(random.choice(pool), None, "rand_comp")
        if loc is None:
            b1 = random.choice(a.comp_tags_list)
            while not cs.region_ok(a, b1):
                b1 = random.choice(a.comp_tags_list)
        else:
            pool = [b for b in sorted(loc.behas, key=str) if cs.region_ok(a, b)]
            if not pool:
                return None
            b1 = random.choice(pool)
        an, g = eng.analysis(), eng.g
        cp_tags = {g.tag[r["event"]] for r in an.critical_path()}
        b2 = self.heur.swap_partner(b1, cp_tags, lambda b: eng.W(g.id_of[b]))
        if b2 is None:           # no partner inside the set: relocate instead
            return self._comp_move(b1, None, "rand_comp")
        cs.mv_swap(a, b1, b2)
        return {"action": "rand_swap", "beha": b1, "beha2": b2}

    def propose(self):
        if self.sync:
            self.diag = {}
            self.heur.cand_log = self.diag["cands"] = []
        if random.random() < P:
            if self.diag is not None:
                self.diag["drawn"] = True
            r = self.targeted()
            if r is not None:
                r["targeted"] = True
                return r
            if self.local is None:
                raise RuntimeError("targeted move failed without a local set")
            if self.diag is not None:
                self.diag["fallback"] = True
        return self.rand()


def read_table(a):
    """(consumer behaviours, bytes) -> DDR sources of the live reads (multiset), to see what a move changed."""
    ddr = {tuple(k[:-1]) for k in a.hardware_platform.ddr_dict}
    c = collections.Counter()
    for e in a.state[7].values():
        if e.event_type == "communication" and tuple(e.source_location) in ddr:
            c[(tuple(sorted(map(str, e.issue_set))), e.comm_bytes, tuple(e.source_location))] += 1
    return c


def reads_by_key(a):
    """read_table's key -> tags."""
    ddr = {tuple(k[:-1]) for k in a.hardware_platform.ddr_dict}
    out = collections.defaultdict(list)
    for t, e in a.state[7].items():
        if e.event_type == "communication" and tuple(e.source_location) in ddr:
            out[(tuple(sorted(map(str, e.issue_set))), e.comm_bytes, tuple(e.source_location))].append(t)
    return out


def has_real_copy(a, tag, node):
    """(the data split this read carries, or None; whether its location list names the DDR). Under the corrected rule the
    list holds the real copies. An offline weight by cs.weight_split_of; otherwise (the input activation, which is on every
    port and is not in offlinedata_list) any data listing the tag, a split of the read's size first."""
    ws = cs.weight_split_of(a, tag)
    if ws is None:
        nb = a.state[7][tag].comm_bytes
        hits = [(split_bytes_of(a, d, sp) != nb, str(d), str(sp), d, sp) for d, t in a.state[6].items()
                for sp, tags in t.used_splitted_tag_dict.items() if tag in tags and sp in t.generated_split_location]
        if hits:
            ws = min(hits, key=lambda h: h[:3])[3:]
    return ws, ws is not None and tuple(node) in {tuple(x) for x in a.state[6][ws[0]].generated_split_location[ws[1]]}


def rebound_reads(a, before):
    """After a behaviour move: live reads of the same consumers and size that changed DDR. rebound_stale = the new source
    is not among the split's locations; rebound_other = no data split found for the read."""
    now = read_table(a)
    added, removed = now - before, before - now
    gone = {k[:2] for k in removed}
    out = collections.Counter()
    hits = [k for k in added if k[:2] in gone]
    if hits:
        idx = reads_by_key(a)
        for k in hits:
            ws, real = has_real_copy(a, idx[k][0], k[2])
            out["rebound"] += added[k]
            out["rebound_real_copy" if real else "rebound_other" if ws is None else "rebound_stale"] += added[k]
    return out


class LocalAudit:
    """State-level check of every proposal under a local set (check 3): whatever branch produced it, behaviours that
    changed place must be movable and land on allowed devices, and DDR sources that newly appear must be allowed.
    sync (corrected rule, next handoff section 3): a new source outside the set is allowed, and counted, when every consumer
    of that read is in the set and the data has a real copy on that DDR; the rest stays ddr_source_outside."""

    def __init__(self, a, local, sync=False):
        self.a, self.local, self.sync = a, local, sync
        self.n = 0
        self.count = collections.Counter()

    def _allowed(self, k, idx):
        s = self.a.state
        for t in idx[k]:
            if not set(s[7][t].issue_set) <= self.local.behas or not has_real_copy(self.a, t, k[2])[1]:
                return False
        return True

    def before(self):
        s = self.a.state
        return {t: b.location for t, b in s[5].items()}, read_table(self.a)

    def check(self, pre):
        locs0, reads0 = pre
        s, loc = self.a.state, self.local
        self.n += 1
        out = collections.Counter()
        for t, b in s[5].items():
            if b.location != locs0[t]:
                self.count["behas_changed"] += 1
                if t not in loc.behas:
                    out["beha_not_movable"] += 1
                if tuple(b.location[:-1]) not in loc.nodes:
                    out["beha_dest_outside"] += 1
        idx = None
        for k, v in read_table(self.a).items():
            if v > reads0.get(k, 0) and k[2] not in loc.ddr_ports:
                if self.sync:
                    idx = idx or reads_by_key(self.a)
                    if self._allowed(k, idx):
                        self.count["ddr_outside_allowed"] += 1
                        continue
                out["ddr_source_outside"] += 1
        self.count.update(out)
        return out


# ---------------------------------------------------------------- L1 (handoff section 6)
class L1:
    def __init__(self, eng, n=L1_N, max_attempts=L1_MAX_ATTEMPTS, checks=False):
        self.eng, self.n, self.max_attempts, self.checks = eng, n, max_attempts, checks
        hp = eng.hp
        self.ddr = set(map(tuple, hp.ddr_set))
        self.adj = collections.defaultdict(list)
        self.radj = collections.defaultdict(list)
        for k in hp.links_dict:
            if len(k) != 2:
                raise RuntimeError(f"parallel-link key {k}: not covered")
            self.adj[tuple(k[0])].append(tuple(k[1]))
            self.radj[tuple(k[1])].append(tuple(k[0]))
        self.log, self.rev = [], {"n": 0, "state_mismatch": 0, "T_mismatch": 0}

    # ---- candidate routes
    def _dist_to(self, tgt, avoid):
        d, q = {tgt: 0}, collections.deque([tgt])
        while q:
            v = q.popleft()
            for u in self.radj[v]:
                if u in d or (u, v) == avoid or (u in self.ddr):       # DDR ports do not forward
                    continue
                d[u] = d[v] + 1
                q.append(u)
        return d

    def alt_unicast(self, src, tgt, avoid, busy, current):
        """Up to L1_ALTS shortest routes avoiding the link, least busy first (ties: lexicographic)."""
        src, tgt = tuple(src), tuple(tgt)
        d = self._dist_to(tgt, avoid)
        if src in self.ddr:           # a DDR source reaches the mesh through its only link
            ds = [(d[v] + 1, v) for v in self.adj[src] if v in d and (src, v) != avoid]
            if not ds:
                return []
            best = min(x[0] for x in ds)
            firsts = [v for h, v in ds if h == best]
        else:
            if src not in d:
                return []
            firsts = None
        paths = []

        def walk(u, path):
            if len(paths) >= 256:
                return
            if u == tgt:
                paths.append(list(path))
                return
            for v in sorted(self.adj[u]):
                if (u, v) != avoid and v in d and d[v] == d[u] - 1:
                    path.append((u, v))
                    walk(v, path)
                    path.pop()
        if firsts is None:
            walk(src, [])
        else:
            for v in sorted(firsts):
                walk(v, [(src, v)])
        scored = sorted(paths, key=lambda p: (sum(busy.get(k, 0) for k in p), p))
        return [p for p in scored if p != current][:L1_ALTS]

    def alt_multicast(self, e, avoid):
        """Tree without the link: keep what still reaches a target, attach each cut-off target by a shortest path
        from the tree."""
        src = tuple(e.source_location)
        targets = {tuple(x) for x in e.target_location}
        kids = {tuple(r): [tuple(l) for l in ls] for r, ls in e.paths.items()}
        kids[avoid[0]] = [v for v in kids.get(avoid[0], []) if v != avoid[1]]
        seen, stack = set(), [src]
        while stack:
            u = stack.pop()
            if u not in seen:
                seen.add(u)
                stack += kids.get(u, [])
        kids = {u: [v for v in vs if v in seen] for u, vs in kids.items() if u in seen}

        def prune():
            changed = True
            while changed:
                changed = False
                for u in list(kids):
                    keep = [v for v in kids[u] if v in targets or kids.get(v)]
                    if keep != kids[u]:
                        kids[u] = keep
                        changed = True
        prune()
        tree_nodes = {src} | {v for vs in kids.values() for v in vs}
        for tgt in sorted(targets - tree_nodes):
            d = self._dist_to(tgt, avoid)
            starts = [n for n in tree_nodes if n in d and n not in self.ddr]
            if not starts:
                ds_ok = src in self.ddr and any(v in d for v in self.adj[src])
                if not ds_ok:
                    return None
                starts = [v for v in self.adj[src] if v in d]
                u = min(starts, key=lambda n: (d[n], n))
                kids.setdefault(src, []).append(u)
                tree_nodes.add(u)
            else:
                u = min(starts, key=lambda n: (d[n], n))
            while u != tgt:
                v = min((v for v in self.adj[u] if v in d and d[v] == d[u] - 1 and (u, v) != avoid), key=lambda x: x)
                kids.setdefault(u, []).append(v)
                tree_nodes.add(v)
                u = v
        return kids

    # ---- applying a route
    def _tree_from_path(self, p):
        return {k[0]: [k[1]] for k in p}

    def _as_paths(self, kids, src):
        from net import TupleIdx
        src = tuple(src)
        return {(u if u == src else TupleIdx(u, None)): [TupleIdx(v, None) for v in vs] for u, vs in kids.items() if vs}

    def apply(self, tag, ev_obj, kids):
        eng, hp, s = self.eng, self.eng.hp, self.eng.a.state
        paths = self._as_paths(kids, ev_obj.source_location)
        if tag in s[7]:
            e = s[7][tag]
            e3._add_loads(hp, e, -1)
            for link in e3._route_links(e.paths):
                s[2][link] -= np.ceil(e.comm_bytes / hp.links_dict[link].bandwidth)
            e.paths = paths
            e.path_list = e.get_paths(paths)
            e3._add_loads(hp, e, +1)
            for link in e3._route_links(paths):
                s[2][link] += np.ceil(e.comm_bytes / hp.links_dict[link].bandwidth)
            if not isinstance(e.target_location, set):     # the totals build_event reports, for this route
                links = e3._route_links(paths)
                base = sum(hp.links_dict[l].latency for l in links)
                slope = sum(hp.links_dict[l].bandwidth / hp.links_dict[l].timeunit for l in links)
                hops, dist = len(links), base + math.ceil(e.comm_bytes / slope)
                s[0] += (hops - e.hops) * e.comm_bytes
                s[1] += dist - e.communication_distances
                e.hops, e.communication_distances = hops, dist
            return "state"
        key = next((k for k, r in eng.bc.routes.items() if r[1] is ev_obj.paths), None)
        if key is None:
            raise RuntimeError(f"broadcast {tag} is not in the route cache")
        path_list = ev_obj.get_paths(paths)
        w = [((r_, l_), hp.current_link_loads(ev_obj.comm_bytes, hp.links_dict[(r_, l_)]))
             for r_, leaves in paths.items() for l_ in leaves]
        eng.bc.routes[key] = (eng.bc.routes[key][0], paths, path_list, w)
        return "broadcast"

    # ---- the loop
    def candidates(self, tried):
        """Next (link, blocker, route) not yet tried in this state: links by CB, blockers by phi, routes by busy."""
        eng = self.eng
        an, g, sc = eng.analysis(), eng.g, eng.sc
        busy = {}
        for q, seq in enumerate(sc.res_seq):
            if seq and g.res_name[q][0] == "link":
                busy[g.res_name[q][1]] = sum(sc.a_f[a] - sc.a_s[a] for a in seq)
        cb = an.blocking()
        links = [q for q, v in cb.items() if v > 0 and g.res_name[q][0] == "link"]
        for q in sorted(links, key=lambda x: (-cb[x], x)):
            link = tuple(map(tuple, g.res_name[q][1]))
            _, phi = an.blocking(q)
            for b in sorted(phi, key=lambda x: (-phi[x], x)):
                if phi[b] <= 0 or g.kind[b] != cpcm.COMM:
                    continue
                tag = g.tag[b]
                obj = eng.a.state[7][tag] if tag in eng.a.state[7] else eng.ev[tag]
                if isinstance(obj.target_location, set):
                    kids = self.alt_multicast(obj, link)
                    routes = [kids] if kids else []
                else:
                    cur = [tuple(map(tuple, k)) for br in obj.path_list for k in br]
                    routes = [self._tree_from_path(p)
                              for p in self.alt_unicast(obj.source_location, obj.target_location, link, busy, cur)]
                for kids in routes:
                    key = (tag, str(sorted((u, sorted(v)) for u, v in kids.items())))
                    if key not in tried:
                        yield link, cb[q], phi[b], tag, obj, kids, key

    def run(self):
        eng = self.eng
        T = eng.t_eval()
        T_start = T
        fails, attempts = 0, 0
        tried = set()
        reason = "n_failures"
        while True:
            if fails >= self.n:
                break
            if attempts >= self.max_attempts:
                reason = "max_attempts"
                break
            cand = next(self.candidates(tried), None)
            if cand is None:
                reason = "no_candidates"
                break
            link, cbv, phiv, tag, obj, kids, key = cand
            tried.add(key)
            pre = eng.before()
            fp0 = eng.fingerprint() if self.checks else None
            where = self.apply(tag, obj, kids)
            T2 = eng.t_eval()
            attempts += 1
            row = {"attempt": attempts, "link": str(link), "cb": cbv, "phi": phiv, "event": str(tag), "where": where,
                   "T_new": T2, "T_before": T}
            if T2 < T:
                row["accepted"] = True
                T = T2
                fails = 0
                tried = set()
            else:
                row["accepted"] = False
                eng.revert(pre)
                fails += 1
                if self.checks:
                    self.rev["n"] += 1
                    self.rev["state_mismatch"] += eng.fingerprint() != fp0
                    keep = (eng.g, eng.sc, eng.an, eng.ev)
                    self.rev["T_mismatch"] += eng.t_eval() != T
                    eng.g, eng.sc, eng.an, eng.ev = keep
            self.log.append(row)
        return {"T_start": T_start, "T_end": T, "attempts": attempts, "accepted": sum(r["accepted"] for r in self.log),
                "reason": reason, "log": self.log, "check_revert": self.rev}


# ---------------------------------------------------------------- L2 / W search (handoff sections 3.2, 4, 5)
def search(eng, method, steps, seed, seeds_nodes, checks, checkpoints=None, impact=None):
    """From the current state (the L1 output). L2G / L2L: E = T/T0 + (0.01/T0) dist/dist0, kept state = smallest E.
    W: BusyBarn's energy with the artifact's random move, kept state = smallest energy. Same exact revert, same 10%
    calibration, same seeds.

    checkpoints: optional step numbers inside this one cooling schedule (`steps` unchanged). At the end of each, the best
    state so far is recorded (best_E, best_T, best_step, snap, t_search = search seconds so far, no calibration). It
    only reads `best`: no random numbers, no change to the state, and the result has the key only when asked for.

    Methods in SYNC_METHODS: the corrected DDR rule. impact: impact_select's result, for L2Impact's set. Their rows and result
    carry more keys (landing cores, counters); the other methods take none of these branches."""
    if checkpoints and not all(0 < c <= steps for c in checkpoints):
        raise ValueError(f"checkpoints {checkpoints} outside 1..{steps}")
    a, hp = eng.a, eng.hp
    sync = method in SYNC_METHODS
    sync_entry = None
    if sync:
        T_a = eng.t_eval()
        sync_entry = sync_locations(a)
        T_b = eng.t_eval()
        if T_a != T_b:
            raise RuntimeError(f"syncing the locations changed T: {T_a} -> {T_b}")
        sync_entry.update(T=T_b, stale_after=stale_locations(a))
        eng.sync = True
    kind = LOCAL_METHODS.get(method)
    if kind == "impact":
        local = LocalSet(a, None, "impact", nodes=impact["nodes"])
    else:
        local = LocalSet(a, seeds_nodes, LOCAL_R) if kind == "geo" else None
    moves = Moves(eng, local, sync) if method != "W" else None
    audit = LocalAudit(a, local, sync) if local is not None else None
    watch = collections.Counter()
    orig_move = type(a).move
    st = {"T0": None, "d0": None}
    tm = collections.Counter()
    crev = {"n": 0, "E_mismatch": 0, "T_mismatch": 0, "state_mismatch": 0}
    rows = []
    if method == "W":      # energy normalised at the start state, as a run started here would
        s = a.state
        a.origin = [s[0], s[1] or 1, max(s[2].values()) or 1, max(s[3].values()) or 1, max(s[4].values()) or 1]

    def energy():
        if method == "W":
            return a.energy(), None
        T = eng.t_eval()
        return T / st["T0"] + (0.01 / st["T0"]) * a.state[1] / st["d0"], T

    def propose():
        """Returns (info, illegal): a proposal whose rebuild leaves the local set is undone by the caller."""
        if method == "W":
            orig_move(a)
            return {"action": "rand"}, False
        pre_a = audit.before() if audit else None
        reads0 = read_table(a) if sync else None
        info = moves.propose()
        if info is None:
            return None, False
        if sync:
            watch["proposals"] += 1
            watch["stale_after_proposal"] += stale_locations(a)
            if "split" in info:
                info["landing_ddr"] = tuple(info["dest"][:-1])
            else:
                watch.update(rebound_reads(a, reads0))
                info["landing"] = [tuple(a.state[7][b].comp_location[:-1]) for b in (info.get("beha"), info.get("beha2")) if b is not None]
        if audit:
            bad = audit.check(pre_a)
            if bad:
                return {**info, "illegal": dict(bad)}, True
        return info, False

    if method != "W":
        st["T0"] = eng.t_eval()
        st["d0"] = a.state[1] or 1
    E0, T0 = energy()
    # ---- calibration
    t0 = time.perf_counter()
    random.seed(seed + CAL_SEED_OFFSET)
    np.random.seed(seed + CAL_SEED_OFFSET)
    worse, illegal_cal = [], 0
    for _ in range(CAL_PROPOSALS):
        pre = eng.before()
        fp0 = eng.fingerprint() if checks else None
        info, illegal = propose()
        if info is not None and not illegal:
            E1, T1 = energy()
            if E1 - E0 > 0 and (T1 is None or T1 > T0):
                worse.append(E1 - E0)
        illegal_cal += illegal
        eng.revert(pre)
        if checks:
            crev["n"] += 1
            crev["state_mismatch"] += eng.fingerprint() != fp0
    d50 = statistics.median(worse) if worse else None
    Tmax = -d50 / math.log(CAL_ACCEPT) if d50 else 0.0
    Tmin = Tmax * TMIN_RATIO
    tm["t_calibration"] = time.perf_counter() - t0
    if method != "W":
        energy()      # schedule of the start state (the revert restored it; recomputed for the counters' sake)
    # ---- search
    random.seed(seed)
    np.random.seed(seed)
    E, Tcur = E0, T0
    best = {"E": E, "T": Tcur, "step": 0, "snap": eng.snapshot()}
    t0 = time.perf_counter()
    last_accept = 0
    n_illegal = 0
    ck_steps, ck_rows = set(checkpoints or ()), []

    def mark(step):
        if step in ck_steps:
            ck_rows.append({"step": step, "best_E": best["E"], "best_T": best["T"], "best_step": best["step"], "snap": best["snap"],
                            "t_search": time.perf_counter() - t0})

    for step in range(1, steps + 1):
        temp = Tmax * math.exp(-math.log(Tmax / Tmin) * step / steps) if Tmax > 0 else 0.0
        pre = eng.before()
        fp0 = eng.fingerprint() if checks else None
        info, illegal = propose()
        row = {"step": step, "temp": temp, "action": (info or {}).get("action"), "targeted": bool((info or {}).get("targeted"))}
        if sync and info is not None:
            row.update({k: info[k] for k in ("landing", "landing_ddr") if k in info})
        if sync:
            row["diag"] = moves.diag
        if info is None or illegal:   # nothing proposed or proposal out of bounds: undone, the step is spent
            if info is not None:
                eng.revert(pre)
                n_illegal += 1
                row["illegal"] = info["illegal"]
            row["accepted"] = False
            rows.append(row)
            mark(step)
            continue
        E_new, T_new = energy()
        dE = E_new - E
        row.update(E_new=E_new, T_new=T_new, dE=dE)
        if dE > 0.0 and (temp <= 0.0 or math.exp(-dE / temp) < random.random()):
            eng.revert(pre)
            row["accepted"] = False
            if checks:
                keep = (eng.g, eng.sc, eng.an, eng.ev)
                E_re, T_re = energy()
                eng.g, eng.sc, eng.an, eng.ev = keep
                crev["n"] += 1
                crev["E_mismatch"] += E_re != E
                crev["T_mismatch"] += T_re != Tcur
                crev["state_mismatch"] += eng.fingerprint() != fp0
        else:
            row["accepted"] = True
            E, Tcur = E_new, T_new
            last_accept = step
            if E < best["E"]:
                best.update(E=E, T=Tcur, step=step, snap=eng.snapshot())
        rows.append(row)
        mark(step)
    tm["t_L2_search"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    eng.load(best["snap"])
    tm["t_restore_best"] = time.perf_counter() - t0
    return {"rows": rows, "Tmax": Tmax, "Tmin": Tmin, "cal_worse": worse, "cal_illegal": illegal_cal,
            "best_E": best["E"], "best_T": best["T"], "best_step": best["step"], "last_accept": last_accept,
            "E_start": E0, "T_start": T0, "n_illegal": n_illegal, "check_revert": crev, **tm,
            "local": local.coverage() if local else None,
            "local_behas": set(local.behas) if local else None, "local_nodes": set(local.nodes) if local else None,
            "local_audit": ({"n": audit.n, **audit.count} if audit else None),
            **({"checkpoints": ck_rows} if checkpoints else {}),
            **({"sync_entry": sync_entry, "read_watch": dict(watch), "stale_final": stale_locations(a), "impact": impact} if sync else {})}


# ---------------------------------------------------------------- child
def write_sink(path, d):
    d["done"] = True
    tmp = path + ".part"
    with gzip.open(tmp, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def read_sink(path):
    cs._artifact_path()
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_input(name):
    with gzip.open(os.path.join(INPUTS, name + ".pkl.gz"), "rb") as f:
        return pickle.load(f)


def start_mapping(start, op):
    return load_input(f"{start}_{op}")


def setup_modules(mapping, faults):
    """E3's inject mode applied directly (e3's sitecustomize does the same from an import hook). Returns the raw
    backend. E3 reads its spec from the file named by E3_SPEC."""
    fd, spec_path = tempfile.mkstemp(suffix=".pkl", prefix="eirspec_")
    with os.fdopen(fd, "wb") as f:
        pickle.dump({"mapping": mapping, "faults": faults, "steps": "template", "t_max": None, "seeds": [],
                     "preserve_routes": mapping is not None}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.environ["E3_SPEC"] = spec_path
    os.environ["E3_SINK"] = spec_path + ".unused"
    e3._spec()
    os.remove(spec_path)
    cs._artifact_path()
    raw = {}
    for name in ("read_cfg", "WAMIS_HD", "Pre_Mapping", "add_communication", "Stream_Mapping", "event_driver"):
        mod = importlib.import_module(name)
        if name == "event_driver":
            raw["backend"] = mod.event_driver
        e3.patch_module(name, mod)
    return raw["backend"]


class Timers:
    """Decision time of R0: E3's relocation and route restoration."""

    def __init__(self):
        self.t = 0.0
        for name in ("ddr_replacement", "relocate_ddr_data", "relocate_failed", "restore_routes", "relabel_events"):
            setattr(e3, name, self._wrap(getattr(e3, name)))

    def _wrap(self, fn):
        def inner(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.t += time.perf_counter() - t0
        return inner


def run_template(task, anneal, finish_sink=None):
    import Stream_Mapping
    Stream_Mapping.StreamOptimization.anneal = anneal
    path = os.path.join(WORK, "models", GPT, "py", task + ".py")
    os.makedirs(os.path.join(os.path.dirname(path), "..", "results"), exist_ok=True)
    runpy.run_path(path, run_name="__main__")


def seeds_of_fault(f, hp):
    n = {tuple(x) for x in f["failed_nodes"]} | {tuple(x) for x in f.get("failed_pes", [])}
    for a, b in f["failed_links"]:
        n |= {tuple(a), tuple(b)}
    return n


def child(spec):
    kind = spec["kind"]
    if kind == "full":
        return child_full(spec)
    if kind in ("verify_r0", "verify_full"):
        return child_verify(spec)
    start, op, fid = spec["start"], spec["op"], spec.get("fault")
    mapping = start_mapping(start, op)
    task = TASKS[op]
    if kind == "full_e3":
        fu = load_input(f"full_{op}_{fid}")
        mapping = dict(fu["mapping"], layout=mapping["layout"])
        task = re.sub(r"greedy\d_lossratio\d", f"greedy{fu['g']}_lossratio{fu['l']}", task)
    faults = fault_of(op, fid) if fid else {"failed_nodes": [], "failed_links": []}
    timers = Timers()
    backend = setup_modules(mapping, faults)
    add_comm = sys.modules["add_communication"]

    def anneal(self):
        if self.LP_flag or self.mem_datatags or self.related:
            raise RuntimeError("template uses LP_flag / mem_datatags / related: not covered")
        t_r0_inject = timers.t
        eng = Engine(self, add_comm.add_broadcast, backend, faults)
        sink = {"spec": spec, "faults": faults, "backend_seeds": BACKEND_SEEDS, "bcast_seed": BCAST_SEED,
                "log": {k: v for k, v in e3._spec()["_log"].items() if k in ("relocated", "ddr_data_moved", "ddr_replacement",
                                                                           "removed_pe_devices", "xy_none_ddr_pairs")}}
        route_log = [{k: v for k, v in x.items() if k != "rerouted_tags"} for x in e3._spec()["_log"].get("routes", [])]
        sink["route_log"] = route_log
        if kind == "pre":
            child_pre(eng, spec, sink)
        elif kind == "l1":
            child_l1(eng, spec, sink, t_r0_inject)
        elif kind == "rep":
            child_rep(eng, spec, sink)
        elif kind == "select":
            child_select(eng, sink, spec.get("fids", ("F4", "F5")))
        elif kind == "ctrl":
            child_ctrl(eng, spec, sink)
        elif kind == "reload":
            child_reload(eng, spec, sink)
        elif kind == "full_e3":
            pre = read_sink(spec["pre_sink"])
            sink["bc_seed"] = eng.seed_broadcast()
            sink["record"] = eng.finalize()
            sink["record"]["migration_vs_start"] = migration_metrics(pre["record"], sink["record"], pre["ddr_nodes"], pre["sizes"])
            sink["e3"] = {k: fu[k] for k in ("variant", "e3_seed_lat", "e3_mean", "e3_total_sa_wall_14")}
        write_sink(spec["sink"], sink)
        raise SystemExit(0)

    run_template(task, anneal)


def child_pre(eng, spec, sink):
    a, hp = eng.a, eng.hp
    sink["bc_seed"] = eng.seed_broadcast()
    T = eng.t_eval()
    an, g, sc = eng.analysis(), eng.g, eng.sc
    an.critical_path()
    busy = {}
    for q, seq in enumerate(sc.res_seq):
        if seq and g.res_name[q][0] == "link":
            busy[g.res_name[q][1]] = sum(sc.a_f[x] - sc.a_s[x] for x in seq)
    cb = an.blocking()
    cb_links = {g.res_name[q][1]: v for q, v in cb.items() if g.res_name[q][0] == "link"}
    if spec.get("store_w"):       # L2-Impact aligns the waits by tag
        sink["W_by_tag"] = {g.tag[e]: eng.W(e) for e in range(g.n)}
    rec = eng.finalize()
    sizes = {(d, sp): split_bytes_of(a, d, sp) for d in a.offlinedata_list for sp in a.offlinedata_split_dict[d]}
    sink.update(T=T, record=rec, busy=busy, cb_links=cb_links, sizes=sizes, ddr_nodes=sorted(map(tuple, hp.ddr_set)))


def child_select(eng, sink, fids=("F4", "F5")):
    """E3's rule on this start. E3 used the template RNG's final evaluation; here backend seeds 1, 2, 3, seed 1 decides
    and the other two show how stable the choice is."""
    sink["bc_seed"] = eng.seed_broadcast()
    e2 = e3._e2()
    sink["choice"] = {}
    for sd in (1, 2, 3):
        eng.finalize(seeds=[sd])
        ev_k, hp_k, pre, out = eng.last_cap
        cap = e2._extract(ev_k, hp_k, pre, out, f"seed{sd}")
        p = {"final": cap}
        sink["choice"][sd] = {f: {k: v for k, v in e3.choose_fault(p, f).items() if k in ("failed_nodes", "failed_links", "kind", "busy", "busy_over_a", "critical_wait", "critical_wait_over_a", "top_ports")}
                              for f in fids}
        sink["choice"][sd]["latency"] = out[0]


def keep_final(eng, spec, sink):
    """The deployed mapping, complete: load it into an engine on the same platform and T comes back (child_reload)."""
    sink["final_snap"] = eng.snapshot()
    if spec.get("checks"):
        sink["final_fp"] = eng.fingerprint()


def child_reload(eng, spec, sink):
    """Check: a saved final state, loaded in a new process on the same faulty platform, gives the same T and state."""
    d = read_sink(spec["from_sink"])
    eng.sync = d["spec"].get("method") in SYNC_METHODS
    eng.load(d["final_snap"])
    sink["T"] = eng.t_eval()
    sink["T_saved"] = d["record"]["T"]
    sink["T_equal"] = sink["T"] == sink["T_saved"]
    if "final_fp" in d:
        sink["fp_equal"] = eng.fingerprint() == d["final_fp"]


def child_ctrl(eng, spec, sink):
    pre = read_sink(spec["pre_sink"])
    sink["bc_seed"] = eng.seed_broadcast()
    T = eng.t_eval()
    if T != pre["T"]:
        raise RuntimeError(f"fault-free state rebuilt with T {T}, pre job had {pre['T']}")
    fp0 = eng.fingerprint() if spec["steps"] == 0 else None
    n0 = eng.ncpcm
    res = search(eng, "L2G", spec["steps"], spec["seed"], [], spec.get("checks", False), spec.get("checkpoints"))
    sink["search"] = res
    keep_final(eng, spec, sink)
    if spec["steps"] == 0:
        sink["s0_state_equal"] = eng.fingerprint() == fp0
    rec = eng.finalize()
    sink["record"] = {**rec, "migration_vs_start": migration_metrics(pre["record"], rec, pre["ddr_nodes"], pre["sizes"])}
    sink["timing"] = {"t_calibration": res["t_calibration"], "t_L2_search": res["t_L2_search"], "t_finalize": res["t_restore_best"] + rec["t_finalize"],
                      "t_backend_offline": rec["t_backend"]}
    sink["timing"]["t_decision"] = sum(sink["timing"][k] for k in ("t_calibration", "t_L2_search", "t_finalize"))
    sink["cpcm_calls"] = {"L2": eng.ncpcm - n0}


def link_stats(pre, faults):
    """Busy share and CB rank, in the fault-free state of this start, of the failed links (handoff 2.2)."""
    out = []
    ranked = sorted(pre["cb_links"], key=lambda k: -pre["cb_links"][k])
    for a_, b_ in faults["failed_links"]:
        k = (tuple(a_), tuple(b_))
        out.append({"link": str(k), "busy": pre["busy"].get(k, 0), "busy_share": pre["busy"].get(k, 0) / pre["T"],
                    "cb": pre["cb_links"].get(k, 0), "cb_rank": ranked.index(k) + 1 if k in ranked else None,
                    "n_links_with_cb": len(ranked)})
    return out


def child_l1(eng, spec, sink, t_r0_inject):
    a, hp = eng.a, eng.hp
    pre = read_sink(spec["pre_sink"])
    faults = eng.faults
    ddr_nodes = pre["ddr_nodes"]
    t0 = time.perf_counter()
    sink["bc_seed"] = eng.seed_broadcast()
    t_r0 = t_r0_inject + (time.perf_counter() - t0)
    T_R0 = eng.t_eval()
    sink["R0_snap"] = eng.snapshot()
    n0 = eng.ncpcm
    r0 = eng.finalize()
    sink["R0"] = {**r0, "migration_vs_start": migration_metrics(pre["record"], r0, ddr_nodes, pre["sizes"])}
    t0 = time.perf_counter()
    res = L1(eng, checks=spec.get("checks", False)).run()
    t_l1 = time.perf_counter() - t0
    sink["L1_snap"] = eng.snapshot()
    rec = eng.finalize()
    sink["L1"] = {**rec, "migration_vs_start": migration_metrics(pre["record"], rec, ddr_nodes, pre["sizes"]),
                  "migration_vs_R0": migration_metrics(r0, rec, ddr_nodes, pre["sizes"])}
    sink["L1_run"] = res
    sink.update(T_pre=pre["T"], T_R0=T_R0, T_L1=res["T_end"], t_R0=t_r0, t_L1=t_l1, cpcm_L1=eng.ncpcm - n0,
                link_stats=link_stats(pre, faults))
    # local-set seeds: fault, R0's relocations, every event whose route changed in R0 or L1
    seeds = seeds_of_fault(faults, hp)
    for t, old, new in sink["log"].get("relocated") or []:
        seeds |= {tuple(old[:-1]), tuple(new[:-1])}
    for _, _, dead, new in sink["log"].get("ddr_data_moved") or []:
        seeds |= {tuple(x) for x in dead}
        if len(new) == 1:     # really moved; several = a copy dropped (the input activation is on every port)
            seeds |= {tuple(new[0])}
    e_r0, n_r0 = changed_endpoints(pre["record"]["routes"], r0["routes"])
    e_l1, n_l1 = changed_endpoints(r0["routes"], rec["routes"])
    rule = {"handoff": seeds | e_r0 | e_l1, "no_R0_rerouted": seeds | e_l1, "fault_relocated": seeds}[SEED_RULE]
    sink.update(seed_nodes=sorted(rule), seed_rule=SEED_RULE, n_changed_routes_R0=n_r0, n_changed_routes_L1=n_l1,
                seed_parts={"fault_and_relocated": sorted(seeds), "R0_rerouted": sorted(e_r0), "L1_rerouted": sorted(e_l1)})
    sink["coverage"] = {R: LocalSet(a, sorted(seeds | e_r0 | e_l1), R).coverage() for R in range(0, 7)}
    sink["coverage_no_R0_rerouted"] = {R: LocalSet(a, sorted(seeds | e_l1), R).coverage() for R in range(0, 7)}
    sink["coverage_fault_only"] = {R: LocalSet(a, sorted(seeds), R).coverage() for R in range(0, 7)}


def child_rep(eng, spec, sink):
    a, hp = eng.a, eng.hp
    pre = read_sink(spec["pre_sink"])
    base = read_sink(spec["l1_sink"])
    ddr_nodes = pre["ddr_nodes"]
    eng.load(base["L1_snap"])
    eng.bc.recomputed = 0
    sink["T_L1_check"] = eng.t_eval()
    if sink["T_L1_check"] != base["T_L1"]:
        raise RuntimeError(f"L1 state reloaded with T {sink['T_L1_check']}, expected {base['T_L1']}")
    fp_l1 = eng.fingerprint() if spec["steps"] == 0 else None
    n0 = eng.ncpcm
    impact = impact_select(eng, base, pre, read_sink(spec["pre_w_sink"])) if spec["method"] == "L2Impact" else None
    res = search(eng, spec["method"], spec["steps"], spec["seed"], base["seed_nodes"], spec.get("checks", False), spec.get("checkpoints"), impact)
    sink["search"] = res
    keep_final(eng, spec, sink)
    if spec["steps"] == 0:
        sink["s0_state_equal"] = eng.fingerprint() == fp_l1
    rec = eng.finalize()
    sink["record"] = {**rec, "migration_vs_start": migration_metrics(pre["record"], rec, ddr_nodes, pre["sizes"]),
                      "migration_vs_R0": migration_metrics(base["R0"], rec, ddr_nodes, pre["sizes"])}
    sink["timing"] = {"t_R0": base["t_R0"], "t_L1": base["t_L1"], "t_calibration": res["t_calibration"],
                      "t_L2_search": res["t_L2_search"], "t_finalize": res["t_restore_best"] + rec["t_finalize"],
                      "t_backend_offline": rec["t_backend"]}
    sink["timing"]["t_decision"] = sum(sink["timing"][k] for k in ("t_R0", "t_L1", "t_calibration", "t_L2_search", "t_finalize"))
    sink["cpcm_calls"] = {"L1": base["cpcm_L1"], "L2": eng.ncpcm - n0}
    sink["proposals"] = {"calibration": CAL_PROPOSALS, "steps": spec["steps"]}
    if spec.get("ck_backend"):
        ck_backend(eng, base, pre, res, sink)


def handoff_view(a):
    """Phase B's stage-to-stage endpoints, read from the state. ifmap_reads: (target, bytes) of the DDR reads that carry no
    offline weight (the input activation). output_locs: where the last MHA_p behaviour runs (empty in FFN, whose output
    goes through add_broadcast)."""
    ddr = {tuple(k[:-1]) for k in a.hardware_platform.ddr_dict}
    reads = [(_plain(e.target_location), e.comm_bytes) for t, e in a.state[7].items()
             if e.event_type == "communication" and tuple(e.source_location) in ddr and cs.weight_split_of(a, t) is None]
    tags = [t for t, b in a.state[5].items() if b.beha_name.startswith("MHA_p_")]
    return {"ifmap_reads": reads, "output_locs": [_plain(a.state[5][max(tags)].location)] if tags else []}


def ck_backend(eng, base, pre, res, sink):
    """Phase B: the backend at every checkpoint, after the run's own record. Each checkpoint's best state is loaded and
    finalized as the end of a run is; search's rows and result were written before and do not change."""
    ddr_nodes = pre["ddr_nodes"]
    rows = []
    for c in res["checkpoints"]:
        eng.load(c["snap"])
        rec = eng.finalize()
        rows.append({"step": c["step"], "best_T": c["best_T"], "best_step": c["best_step"], "t_search": c["t_search"], "T": rec["T"],
                     "backend": rec["backend"], "t_backend": rec["t_backend"], "t_finalize": rec["t_finalize"],
                     "fault_violations": len(rec["fault_violations"]), "stale": stale_locations(eng.a),
                     "migration_vs_start": migration_metrics(pre["record"], rec, ddr_nodes, pre["sizes"]),
                     "migration_vs_R0": migration_metrics(base["R0"], rec, ddr_nodes, pre["sizes"]),
                     "handoff": handoff_view(eng.a)})
    sink["ck"] = rows
    sink["link_time"] = {_plain(k): (float(v.working_time(data_bytes=0)), float(v.working_time(data_bytes=256))) for k, v in eng.hp.links_dict.items()}


def child_full(spec):
    """cpcm_sa's child with this experiment's fault table; PE-only faults patched as E3 does."""
    op, fid, arm, g, x = spec["op"], spec["fault"], spec["arm"], spec["g"], spec["x"]
    f = fault_of(op, fid)
    cfg = full_cfg(op, fid)
    ch = "ch1x4" if op == "ffn14" else "ch2x2"
    cs.CONFIGS[cfg] = (GPT, op.replace("14", ""), {"failed_nodes": f["failed_nodes"], "failed_links": f["failed_links"]}, ch)
    cs._artifact_path()
    if f.get("failed_pes"):
        import WAMIS_HD
        pes = set(map(tuple, f["failed_pes"]))
        orig = WAMIS_HD.wamis_hdc.__init__

        def __init__(self, platform_cfg):
            orig(self, platform_cfg=platform_cfg)
            for d in (self.tensorcore_dict, self.vectorunit_dict):
                for loc in list(d):
                    if tuple(loc[:-1]) in pes:
                        del d[loc]
        WAMIS_HD.wamis_hdc.__init__ = __init__
    cs.child(cfg, arm, g, x, spec["sink"], spec["snap"], False)


# ---------------------------------------------------------------- step 2: E3's R0 / FULL against E3's records
def T_of_cap(cap, hp):
    """T of a graph recorded by E3 (tags, dependencies, routes, computation places as simulated)."""
    def plain_key(t):
        return t if isinstance(t, tuple) else (t,)
    res = []
    for dev in sorted(hp.modules_dict, key=str):
        for loc in sorted(hp.modules_dict[dev], key=str):
            res.append((("dev", dev, loc), "dev:" + dev, False))
    for k in sorted(hp.links_dict, key=str):
        res.append((("link", k), hp.links_dict[k].link_type, len(k) == 3))
    events = []
    for tag in sorted(cap["events"], key=lambda t: (str(type(t)), plain_key(t))):
        x = cap["events"][tag]
        d = {"tag": tag, "name": str(tag), "deps": sorted(x["deps"], key=plain_key)}
        if x["type"] == "computation":
            d.update(kind="comp", res=("dev", x["device"], x["location"]), dur=x["f"] - x["s"])
        else:
            brs = [[("link", tuple(k)) for k in map(tuple, br)] for br in x["branches"]]
            d.update(kind="comm", branches=brs, dur={l: int(hp.links_dict[l[1]].working_time(data_bytes=x["bytes"]))
                                                     for br in brs for l in br})
        events.append(d)
    return cpcm.simulate(cpcm.build(events, res), "bb").T


def child_verify(spec):
    """Rebuild E3's state and compare: graph (placements, dependencies, sources, routes) against the recorded one, T
    against the T of the recorded graph, backend seeds against E3's values."""
    op, fid = spec["op"], spec["e3_fault"]
    with gzip.open(os.path.join(E3_OUT, f"{TASKS[op]}_{fid}_caps.pkl.gz"), "rb") as f:
        caps = pickle.load(f)
    faults = {k: caps["fault"].get(k, []) for k in ("failed_nodes", "failed_links", "failed_pes")}
    key = spec["e3_run"]
    run = caps["runs"][key]
    if spec["kind"] == "verify_r0":     # R0 recomputed from the fault-free start, as the l1 job does
        m = start_mapping("e3", op)
    else:
        m = copy.deepcopy(run["mapping"])  # E3 did not save FULL's event layout: the start's (cpcm V4's approximation)
        m["layout"] = start_mapping("e3", op)["layout"]
    backend = setup_modules(m, faults)
    add_comm = sys.modules["add_communication"]

    def anneal(self):
        eng = Engine(self, add_comm.add_broadcast, backend, faults)
        sb = eng.seed_broadcast()
        T = eng.t_eval()
        T_rec = {s: T_of_cap(c, eng.hp) for s, c in run["caps"] if s == "template"}
        e2 = e3._e2()
        rec = eng.finalize(seeds=[1, 2, 3])
        ev_k, hp_k, pre, out = eng.last_cap
        cap = e2._extract(ev_k, hp_k, pre, out, "verify")
        gd = {str(s): e3.graph_diff(c, cap) for s, c in run["caps"]}
        lat = [c["result"][0] for s, c in run["caps"] if s != "template"]
        sink = {"T": T, "T_wrapped": sb["T_wrapped"], "T_recorded_graph": T_rec.get("template"), "graph_diff": gd,
                "backend": rec["backend"], "E3_seed_lat": lat, "fault_violations": rec["fault_violations"],
                "route_log": [{k: v for k, v in x.items() if k != "rerouted_tags"} for x in e3._spec()["_log"].get("routes", [])]}
        write_sink(spec["sink"], sink)
        raise SystemExit(0)

    task = re.sub(r"greedy\d_lossratio\d", f"greedy{spec['g']}_lossratio{spec['l']}", TASKS[op]) if spec.get("g") is not None else TASKS[op]
    run_template(task, anneal)


# ---------------------------------------------------------------- parent
def sink_ok(path, spec):
    try:
        d = read_sink(path)
    except Exception:
        return False
    if spec["kind"] == "full":
        return "selected" in d and (spec["arm"] != "A0" or "template_out" in d)
    return d.get("done") is True


def run_job(spec, out):
    """Finished = the sink loads completely. Anything else is removed first; children write to .part and rename."""
    os.makedirs(os.path.dirname(spec["sink"]), exist_ok=True)
    if spec.get("snap"):
        os.makedirs(os.path.dirname(spec["snap"]), exist_ok=True)
    name = spec["name"]
    if os.path.exists(spec["sink"]):
        if sink_ok(spec["sink"], spec):
            return name, "cached", 0.0
        os.remove(spec["sink"])
        print(f"[eir] removed unfinished {name}", flush=True)
    for p in (spec["sink"] + ".part", spec["sink"] + ".tmp"):
        if os.path.exists(p):
            os.remove(p)
    logs = os.path.join(out, "logs")
    os.makedirs(logs, exist_ok=True)
    spec_dir = os.path.join(out, "_tmp")
    os.makedirs(spec_dir, exist_ok=True)
    spec_path = os.path.join(spec_dir, name + ".json")
    final = spec["sink"]
    sp = dict(spec)
    if spec["kind"] == "full":       # cpcm_sa writes the sink directly: give it a temporary name
        sp["sink"] = final + ".tmp"
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(sp, f)
    t0 = time.time()
    with open(os.path.join(logs, name + ".log"), "w", encoding="utf-8") as lf:
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), "child", spec_path], cwd=HERE, stdout=lf,
                             stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    if spec["kind"] == "full" and rc == 0 and os.path.exists(sp["sink"]):
        os.replace(sp["sink"], final)
    ok = rc == 0 and os.path.exists(final) and sink_ok(final, spec)
    wall = time.time() - t0
    print(f"[eir] {name}: exit {rc}, {wall:.0f} s, {'ok' if ok else 'FAILED'}", flush=True)
    return name, "ok" if ok else "failed", wall


def job_pre(out, start, op):
    name = f"pre_{start}_{op}"
    return {"kind": "pre", "name": name, "start": start, "op": op, "sink": os.path.join(out, "pre", name + ".pkl.gz")}


def job_l1(out, start, op, fid, checks=False):
    name = "l1_" + combo_name(start, op, fid)
    return {"kind": "l1", "name": name, "start": start, "op": op, "fault": fid, "checks": checks,
            "pre_sink": job_pre(out, start, op)["sink"], "sink": os.path.join(out, "l1", name + ".pkl.gz")}


def job_rep(out, start, op, fid, method, steps, seed, checks=False):
    name = rep_name(start, op, fid, method, steps, seed)
    return {"kind": "rep", "name": name, "start": start, "op": op, "fault": fid, "method": method, "steps": steps,
            "seed": seed, "checks": checks, "pre_sink": job_pre(out, start, op)["sink"],
            "l1_sink": job_l1(out, start, op, fid)["sink"], "sink": os.path.join(out, "rep", name + ".pkl.gz")}


def job_select(out, op):
    name = f"select_a3h_{op}"
    return {"kind": "select", "name": name, "start": "a3h", "op": op, "sink": os.path.join(out, "select", name + ".pkl.gz")}


def job_ctrl(out, start, op, steps, seed, checks=False):
    name = f"ctrl_{start}_{op}_s{steps}_r{seed}"
    return {"kind": "ctrl", "name": name, "start": start, "op": op, "steps": steps, "seed": seed, "checks": checks,
            "pre_sink": job_pre(out, start, op)["sink"], "sink": os.path.join(out, "ctrl", name + ".pkl.gz")}


def job_full_e3(out, op, fid):
    name = f"fulle3_{op}_{fid}"
    return {"kind": "full_e3", "name": name, "start": "e3", "op": op, "fault": fid,
            "pre_sink": job_pre(out, "e3", op)["sink"], "sink": os.path.join(out, "full_e3", name + ".pkl.gz")}


def job_full(out, op, fid, arm, g, x):
    name = f"full_{op}_{fid}_{arm}_g{g}_{x}"
    return {"kind": "full", "name": name, "op": op, "fault": fid, "arm": arm, "g": g, "x": x,
            "sink": os.path.join(out, "full", "runs", name + ".pkl.gz"), "snap": os.path.join(out, "full", "snaps", name + ".pkl.gz")}


def full_todo():
    """Searches from scratch still to run: FULL-A3h on 1x4 FFN at F4, F5 (handoff 3.1). FULL-BusyBarn needs none: every
    E3 FULL state was rebuilt exactly (PROGRESS step 2), so it only gets its backend seeds (job_full_e3)."""
    return [("ffn14", "F4", "A3h"), ("ffn14", "F5", "A3h")]


# batches of the main run, reported one by one
PARTS = {"ffn": {"ops": ["ffn"], "starts": ["e3"]}, "mha": {"ops": ["mha"], "starts": ["e3"]},
         "ffn14": {"ops": ["ffn14"], "starts": ["e3"]}, "a3h": {"ops": ["ffn", "mha"], "starts": ["a3h"]},
         "all": {"ops": ["ffn", "mha", "ffn14"], "starts": ["e3", "a3h"]}}


def plan(out, part):
    if part == "afault":
        combos = a_combos()
        return [[job_pre(out, s, o) for s, o in sorted({c[:2] for c in combos})],
                [job_l1(out, *c) for c in combos],
                [job_rep(out, *c, m, st, sd) for c in combos for m in METHODS for st in steps_of(c[2]) for sd in SA_SEEDS]]
    if part == "ctrl":
        return [[job_pre(out, s, o) for s, o in CTRL_STARTS],
                [job_ctrl(out, s, o, st, sd) for s, o in CTRL_STARTS for st in CTRL_STEPS for sd in SA_SEEDS]]
    p = PARTS[part]
    combos = [c for c in COMBOS if c[1] in p["ops"] and c[0] in p["starts"]]
    pres = sorted({(s, o) for s, o, _ in combos})
    phases = [[job_pre(out, s, o) for s, o in pres],
              [job_l1(out, s, o, f) for s, o, f in combos] +
              [job_full_e3(out, o, f) for o in p["ops"] for f in FIDS if "e3" in p["starts"]],
              [job_rep(out, s, o, f, m, st, sd) for s, o, f in combos for m in METHODS for st in steps_of(f) for sd in SA_SEEDS]
              + [job_full(out, o, f, arm, g, x) for o, f, arm in full_todo() if o in p["ops"] and "e3" in p["starts"]
                 for g, x in cs.variants(arm)]]
    return phases


def run_phases(phases, out, workers):
    total_failed = []
    for i, jobs in enumerate(phases):
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda j: run_job(j, out), jobs))
        bad = [r[0] for r in res if r[1] == "failed"]
        total_failed += bad
        print(f"[eir] phase {i}: {len(res)} jobs, {sum(r[1] == 'cached' for r in res)} cached, {len(bad)} failed, "
              f"{time.time() - t0:.0f} s", flush=True)
        if bad:
            print(f"[eir] FAILED {bad}; later phases need these, stopping here (rerun to retry)", flush=True)
            break
    return total_failed


# ---------------------------------------------------------------- prepare and verify
def prepare():
    """faults.json (the physical coordinates E3 recorded; MHA 2x2's F5 is E3's F3 link) and inputs/ (start mappings)."""
    table = {}
    for op in TASKS:
        table[op] = {}
        for fid in FIDS:
            ef = "F3" if (op, fid) == ("mha", "F5") else fid
            with open(os.path.join(E3_OUT, f"{TASKS[op]}_{ef}_summary.json"), encoding="utf-8") as f:
                x = json.load(f)["fault"]
            table[op][fid] = {"failed_nodes": x["failed_nodes"], "failed_links": x["failed_links"],
                              "failed_pes": x.get("failed_pes", []), "kind": x["kind"], "e3_fault": ef}
    with open(FAULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=1, ensure_ascii=False)
    os.makedirs(INPUTS, exist_ok=True)
    cs._artifact_path()
    for op in TASKS:
        with gzip.open(os.path.join(E3_OUT, f"{TASKS[op]}_prefault.pkl.gz"), "rb") as f:
            m = pickle.load(f)["mapping"]
        with gzip.open(os.path.join(INPUTS, f"e3_{op}.pkl.gz"), "wb") as f:
            pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
    for op, (cfg, g, x) in A3H_RUNS.items():
        d = cs.load_run(cfg, "A3h", g, x, S2_OUT)
        with gzip.open(os.path.join(INPUTS, f"a3h_{op}.pkl.gz"), "wb") as f:
            pickle.dump(d["selected"]["e3map"], f, protocol=pickle.HIGHEST_PROTOCOL)
    for op in TASKS:
        for fid in FIDS:
            ef = table[op][fid]["e3_fault"]
            with open(os.path.join(E3_OUT, f"{TASKS[op]}_{ef}_summary.json"), encoding="utf-8") as f:
                fu = json.load(f)["FULL"]
            v = fu["best_variant"]
            with gzip.open(os.path.join(E3_OUT, f"{TASKS[op]}_{ef}_caps.pkl.gz"), "rb") as f:
                m = pickle.load(f)["runs"][v]["mapping"]
            g_, l_ = re.search(r"g(\d)_l(\d)", v).groups()
            with gzip.open(os.path.join(INPUTS, f"full_{op}_{fid}.pkl.gz"), "wb") as f:
                pickle.dump({"mapping": m, "g": int(g_), "l": int(l_), "variant": v, "e3_seed_lat": fu["best_seed_lat"],
                             "e3_mean": fu["best_mean"], "e3_total_sa_wall_14": fu["total_sa_wall_14"]}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
    print("faults.json and inputs/ written")


def select(out, workers):
    """F4a, F5a into faults.json, marked same_as when E3's F4 / F5 are the same link (handoff 12.1)."""
    jobs = [job_select(out, op) for op in A3H_RUNS]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: run_job(j, out), jobs))
    if any(r[1] == "failed" for r in res):
        raise SystemExit("select failed")
    t = load_faults()
    report = {}
    links = lambda x: {frozenset((tuple(a), tuple(b))) for a, b in x}
    for op in A3H_RUNS:
        d = read_sink(job_select(out, op)["sink"])["choice"]
        report[op] = d
        for f in ("F4", "F5"):
            c = d[1][f]
            fl = [[list(a), list(b)] for a, b in c["failed_links"]]
            same = next((g for g in FIDS if t[op][g]["failed_links"] and links(t[op][g]["failed_links"]) == links(fl)), None)
            t[op][f + "a"] = {"failed_nodes": [], "failed_links": fl, "failed_pes": [], "kind": c["kind"], "e3_fault": None,
                              "chosen_on": "a3h", "same_as": same, "stable_over_seeds_123": all(links(d[s][f]["failed_links"]) == links(fl) for s in (2, 3)),
                              **{k: c[k] for k in ("busy_over_a", "critical_wait_over_a") if k in c}}
    tmp = FAULTS_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(t, f, indent=1, ensure_ascii=False)
    os.replace(tmp, FAULTS_JSON)
    print(json.dumps({op: {f: t[op][f + "a"] for f in ("F4", "F5")} for op in A3H_RUNS}, indent=0, ensure_ascii=False, default=str))


def verify(jobs_n):
    out = os.path.join(HERE, "out", "verify")
    table = load_faults()
    jobs = []
    for op in TASKS:
        for fid in FIDS:
            ef = table[op][fid]["e3_fault"]
            with open(os.path.join(E3_OUT, f"{TASKS[op]}_{ef}_summary.json"), encoding="utf-8") as f:
                fv = json.load(f)["FULL"]["best_variant"]
            m_ = re.search(r"g(\d)_l(\d)", fv)
            for kind, run, g, l in (("verify_r0", f"{ef}_R0", None, None), ("verify_full", fv, int(m_.group(1)), int(m_.group(2)))):
                name = f"{kind}_{op}_{fid}"
                jobs.append({"kind": kind, "name": name, "op": op, "fault": fid, "e3_fault": ef, "e3_run": run, "g": g, "l": l,
                             "sink": os.path.join(out, name + ".pkl.gz")})
    with ThreadPoolExecutor(jobs_n) as ex:
        list(ex.map(lambda j: run_job(j, out), jobs))
    rows = []
    for j in jobs:
        try:
            d = read_sink(j["sink"])
        except Exception as ex_:
            rows.append({"job": j["name"], "error": repr(ex_)})
            continue
        lat, E3 = [d["backend"][s][0] for s in (1, 2, 3)], d["E3_seed_lat"]
        rows.append({"job": j["name"], "e3_run": j["e3_run"], "T": d["T"], "T_wrapped": d["T_wrapped"], "T_recorded_graph": d["T_recorded_graph"],
                     "graph_diff": d["graph_diff"], "backend_123": lat, "E3_123": E3,
                     "backend_mean": statistics.mean(lat), "E3_mean": statistics.mean(E3),
                     "mean_in_E3_range": min(E3) <= statistics.mean(lat) <= max(E3),
                     "T_equal": d["T"] == d["T_recorded_graph"], "graph_equal": min(d["graph_diff"].values()) == 0,
                     "violations": len(d["fault_violations"])})
    with open(os.path.join(out, "verify.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, default=str)
    for r in rows:
        print(r)


# ---------------------------------------------------------------- report
def bmean(b):
    return statistics.mean(v[0] for v in b.values())


def triple(b):
    return [round(statistics.mean(v[i] for v in b.values()), 1) for i in range(3)]


def report(out):
    rows = []
    for name in sorted(os.listdir(os.path.join(out, "l1"))):
        d = read_sink(os.path.join(out, "l1", name))
        s = d["spec"]
        for lev in ("R0", "L1"):
            r = d[lev]
            rows.append({"start": s["start"], "op": s["op"], "fault": s["fault"], "method": lev, "steps": 0, "seed": None, "T": r["T"],
                         "T_pre": d["T_pre"], "backend": round(bmean(r["backend"]), 1), "min": min(v[0] for v in r["backend"].values()),
                         "max": max(v[0] for v in r["backend"].values()), "triple": triple(r["backend"]),
                         "mig_start": r.get("migration_vs_start"), "mig_R0": r.get("migration_vs_R0"),
                         "t_R0": d["t_R0"], "t_L1": d["t_L1"] if lev == "L1" else 0, "t_decision": d["t_R0"] + (d["t_L1"] if lev == "L1" else 0),
                         "violations": len(r["fault_violations"]), "L1_attempts": d["L1_run"]["attempts"], "L1_accepted": d["L1_run"]["accepted"],
                         "link_stats": d["link_stats"]})
    if os.path.isdir(os.path.join(out, "rep")):
        for name in sorted(os.listdir(os.path.join(out, "rep"))):
            if not name.endswith(".pkl.gz"):
                continue
            d = read_sink(os.path.join(out, "rep", name))
            s, r, x = d["spec"], d["record"], d["search"]
            rows.append({"start": s["start"], "op": s["op"], "fault": s["fault"], "method": s["method"], "steps": s["steps"], "seed": s["seed"],
                         "T": r["T"], "backend": round(bmean(r["backend"]), 1), "min": min(v[0] for v in r["backend"].values()),
                         "max": max(v[0] for v in r["backend"].values()), "triple": triple(r["backend"]),
                         "mig_start": r["migration_vs_start"], "mig_R0": r["migration_vs_R0"], **d["timing"], "cpcm": d["cpcm_calls"],
                         "Tmax": x["Tmax"], "best_step": x["best_step"], "n_illegal": x["n_illegal"], "local": x["local"],
                         "local_audit": x["local_audit"], "violations": len(r["fault_violations"])})
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=0, default=str)
    print(f"{out}/report.json: {len(rows)} rows")


def _log_peak_memory(name, path):
    """Next stage only (env EIR_MEM_LOG): at exit, append this process's peak working set to a file (Windows)."""
    import atexit
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    def log():
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"name": name, "peak_wset": pmc.PeakWorkingSetSize, "peak_commit": pmc.PeakPagefileUsage}) + "\n")
    atexit.register(log)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        with open(sys.argv[2], encoding="utf-8") as f:
            sp_ = json.load(f)
        if os.environ.get("EIR_MEM_LOG"):
            _log_peak_memory(sp_["name"], os.environ["EIR_MEM_LOG"])
        return child(sp_)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "verify", "select", "run", "report"])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--part", default="all", choices=sorted(PARTS) + ["afault", "ctrl"])
    ap.add_argument("--out", default=os.path.join(HERE, "out", "main"))
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "verify":
        verify(a.jobs)
    elif a.cmd == "select":
        select(os.path.abspath(a.out), a.workers)
    elif a.cmd == "run":
        out = os.path.abspath(a.out)
        t0 = time.time()
        bad = run_phases(plan(out, a.part), out, a.workers)
        print(f"[eir] part {a.part}: {'all ok' if not bad else str(len(bad)) + ' failed'}, {(time.time() - t0) / 3600:.2f} h", flush=True)
        if bad:
            raise SystemExit(1)
    elif a.cmd == "report":
        report(os.path.abspath(a.out))


if __name__ == "__main__":
    main()
