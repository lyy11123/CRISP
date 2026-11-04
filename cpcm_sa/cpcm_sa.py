"""CPCM in BusyBarn's SA mapping search, stage-1 pilot (cpcm_sa/CPCM_SA_HANDOFF.md).

Child (one task script per process, runpy as cpcm_v3 / cpcm_v7sa): the template runs from cpcm_sa/work/artifact;
StreamOptimization.anneal is replaced by the loop of the arm (the artifact's own move / revert / update_event /
update_offlinedata are reused; only the energy, the move source / destination, the kept state and the temperature
change):
  A0    BusyBarn energy, artifact moves, last state, template temperature (the artifact loop, unchanged)
  A1    BusyBarn energy, artifact moves, best state over the run, calibrated temperature
  A2    E = T/T0 + (0.01/T0) * dist/dist0 (CPCM T, comm distance as tie key), artifact moves, best state, calibrated
  A3    as A2; with probability P a targeted move (source: one of the K largest-W events on the CPCM critical path)
  A3CB  as A3; source: critical events blocked on the resource with the largest CB (exclusive)
  A3h   as A3; every move destination drawn by the fixed heuristic (PROGRESS.md, 2026-11-03 rules entry)
The selected state (last for A0, best otherwise) is deep-copied when it is reached; at the end it is evaluated as
cpcm_v3 does: a deep copy gets the template's final add_broadcast (random / numpy seeded BCAST_SEED), then the
original event_driver runs once per backend seed 1..5 on a deep copy. RNG state is saved / restored around every
evaluation, so the SA trajectory is the arm's own. A0 then returns to the template, whose own final evaluation is
recorded (self-check 1); the other arms exit after writing the sink.
Fault configs: read_cfg.cfg_to_dict gets E3's F5 links; WAMIS_HD.xy_multicast_path gets E3's None fix.

Usage:
  python cpcm_sa.py run --configs ffn mha ffn_f5 --arms A0 A1 ... --jobs 12
  python cpcm_sa.py check3          (inject the selected state of run 0 per config x arm with E3's code)
  python cpcm_sa.py diag --jobs 6   (routing-history diagnostic on A1 / A3 backend-best mappings)
  python cpcm_sa.py analyse
"""
import argparse
import collections
import copy
import csv
import gzip
import json
import math
import os
import pickle
import random
import runpy
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "cpcm"))
import cpcm  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "e1_fidelity"))
from e1_fidelity import SRC_SUBDIRS, _find_context  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "e3_warmstart"))
import e3_warmstart as e3  # noqa: E402

WORK = os.path.join(HERE, "work", "artifact")
# stage 1.5 (handoff section 9), from the environment so children inherit it:
#   ""     stage 1 as run (artifact revert, T without the final broadcast)
#   drift  stage 1 rules, plus E / T of the reverted state recorded after every rejection (measurement only)
#   b1     exact revert                 b2  exact revert + T, CP, W(e) on the graph with the final broadcast
STAGE = os.environ.get("CPCM_SA_STAGE", "")
EXACT_REVERT = STAGE in ("b1", "b2")
BROADCAST_T = STAGE == "b2"
OUT_ROOT = os.path.join(HERE, "out")
OUT = os.path.join(OUT_ROOT, STAGE) if STAGE else OUT_ROOT
COMMIT_FILE = os.path.join(ROOT, "third_party", "isca2026-busybarn-artifact.COMMIT")
SUF = "_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0"
F5 = {"failed_nodes": [], "failed_links": [[[3, 15], [3, 14]], [[3, 14], [3, 15]]]}  # E3 ffn F5 (summary json)
CONFIGS = {"ffn": ("intra_mapping_gpt_decode", "ffn", None),
           "mha": ("intra_mapping_gpt_decode", "mha", None),
           "ffn_f5": ("intra_mapping_gpt_decode", "ffn", F5)}
ARMS = ["A0", "A1", "A2", "A3", "A3CB", "A3h"]
CPCM_ARMS = {"A2", "A3", "A3CB", "A3h"}
TARGETED = {"A3", "A3CB", "A3h"}
SA_SEEDS = [123, 1, 2, 3, 4, 5, 6]       # A2+: g0/g1 x these 7 seeds (A0/A1: g0/g1 x lossratio0-6, seed 123)
BACKEND_SEEDS = list(range(1, 11))     # user 2026-11-04: 10 seeds (was 1-5; A0/A1 re-evaluated, see reeval)
BCAST_SEED = 0
P, K = 0.5, 5
CAL_PROBES, CAL_SEED_OFFSET = 50, 100003
CAL_MIN_WORSE, CAL_MAX_PROBES = 3, 200    # fewer than 3 worsening probes in 50: keep probing, at most 200
MONITOR_EVERY = 50                        # T (CPCM) of the current state for A0 / A1 (no effect on the trajectory)
CHECK2_STEPS = (200, 400, 600, 800, 1000)
CHECK4_N = 100
# self-check 1 points exempted by the user 2026-11-04 (backend multimodality; PROGRESS step 3)
CHECK1_EXEMPT = {("ffn", 0, 4), ("ffn", 1, 2), ("mha", 1, 2), ("ffn_f5", 1, 4)}
E3_F5 = {"R0_mean_3seeds": 270742.0, "FULL_best_mean_3seeds": 232402.0, "FULL_best_variant": "g1_l6"}


def variants(arm):
    return [(g, x) for g in (0, 1) for x in (range(7) if arm in ("A0", "A1") else SA_SEEDS)]


def run_name(cfg, arm, g, x):
    return f"{cfg}_{arm}_g{g}_{'l' if arm in ('A0', 'A1') else 's'}{x}"


def task_of(cfg, arm, g, x):
    model, op, _ = CONFIGS[cfg]
    l = x if arm in ("A0", "A1") else 0           # A2+ do not use the loss ratio: lossratio0 file
    return model, f"{op}_sq2048_greedy{g}_lossratio{l}{SUF}"


# ------------------------------------------------------------------ child: helpers on the annealer
def _upd_event(self, tags, devs):
    from add_communication import update_event
    s = self.state
    s[0], s[1], s[2], s[3], s[4] = update_event(
        target_event_tags=tags, target_devices=devs, hops=s[0], communication_distances=s[1],
        communication_loads_dict=s[2], tensorcore_loads_dict=s[3], vectorunit_loads_dict=s[4], beha_dict=s[5],
        data_dict=s[6], hardware_platform=self.hardware_platform, event_dict=s[7],
        dijkstra_routing=self.dijkstra_routing, alpha=self.alpha, beta=self.beta, gamma=self.gamma)


def mv_relocate(self, beha, dev):
    """The artifact's move type B with a given behaviour and device (LP_flag is False in every template)."""
    from Pre_Mapping import find_exclusive_dependents
    self.change_flag, self.change_mem = True, False
    tags = [beha] + (find_exclusive_dependents(beha_tag=beha, beha_dict=self.state[5]) if self.related else [])
    self.last_behas = tags
    self.last_position = [self.state[7][beha].comp_location] * len(tags)
    _upd_event(self, tags, [dev] * len(tags))
    self.movements += 1


def mv_weight(self, d, sp, loc):
    """The artifact's move type A with a given weight split and DDR (last_position as the artifact computes it)."""
    from add_communication import update_offlinedata
    self.change_flag, self.change_mem = True, True
    self.last_data, self.last_split = d, sp
    self.last_position = self.state[6][d].generated_split_location[sp][0] + (0,)
    s = self.state
    s[0], s[1], s[2], s[3], s[4] = update_offlinedata(
        target_data_tags=d, target_data_split=sp, target_devices=loc, hops=s[0], communication_distances=s[1],
        communication_loads_dict=s[2], tensorcore_loads_dict=s[3], vectorunit_loads_dict=s[4], beha_dict=s[5],
        data_dict=s[6], hardware_platform=self.hardware_platform, event_dict=s[7],
        dijkstra_routing=self.dijkstra_routing, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
    self.movements += 1


def mv_swap(self, b1, b2):
    """The artifact's move type C with two given behaviours (update_event gets [b1, b2] as in the artifact)."""
    from Pre_Mapping import find_exclusive_dependents
    self.change_flag = False
    d1, d2 = self.state[7][b1].comp_location, self.state[7][b2].comp_location
    if self.related:
        r1 = [b1] + find_exclusive_dependents(beha_tag=b1, beha_dict=self.state[5])
        r2 = [b2] + find_exclusive_dependents(beha_tag=b2, beha_dict=self.state[5])
        self.last_behas = r1 + r2
        self.last_position = [d1] * len(r1) + [d2] * len(r2)
    else:
        self.last_behas, self.last_position = [b1, b2], [d1, d2]
    _upd_event(self, [b1, b2], [d2, d1])
    self.movements += 1


def comp_cands(self, beha):
    return self.comp_position_dict[beha[2]][self.state[7][beha].comp_device]


def ddr_cands(self, d):
    out = []
    for bl in self.state[6][d].belong_layer_set:
        out.extend(self.ddr_list[bl[-1]])
    return out


def region_ok(self, b):
    return self.state[5][b].location[0] in self.layers_regions[b[2]]


def weight_split_of(self, tag):
    """(data, split) of the weight a DDR-read communication carries, from used_splitted_tag_dict (stale recycled
    tags can list a communication under several splits: prefer offline data whose split size equals comm_bytes)."""
    from data_notation import type_bytes
    ev = self.state[7][tag]
    hits = []
    for d in self.offlinedata_list:
        t = self.state[6][d]
        for sp in self.offlinedata_split_dict[d]:
            if tag in t.used_splitted_tag_dict[sp]:
                nb = int(np.prod([e - s + 1 for (s, e) in sp]) * type_bytes[t.data_type])
                hits.append((nb != ev.comm_bytes, str(d), str(sp), d, sp))
    if not hits:
        return None
    hits.sort(key=lambda h: h[:3])
    return hits[0][3], hits[0][4]


class Heur:
    """A3h destination heuristic (rules fixed in PROGRESS.md before the runs)."""
    TOP_MOVE, TOP_SWAP = 3, 5

    def __init__(self, self_):
        self.a = self_

    def hop(self, u, v):
        u, v = tuple(u), tuple(v)
        return 0 if u == v else self.a.hardware_platform.node_to_node_hop_dict[u][v]

    def dev_load(self, dev_kind, loc):
        return self.a.state[3][loc] if dev_kind == "tensorcore" else self.a.state[4][loc]

    def nbr_cost(self, beha, c):
        ev7 = self.a.state[7]
        e = ev7[beha]
        tot = 0
        for d in e.dependency_set:
            x = ev7[d]
            tot += self.hop(x.source_location if x.event_type == "communication" else x.comp_location[:-1], c)
        for d in e.issue_set:
            x = ev7[d]
            if x.event_type == "communication":
                tg = x.target_location
                tot += sum(self.hop(c, n) for n in tg) if isinstance(tg, set) else self.hop(c, tg)
            else:
                tot += self.hop(c, x.comp_location[:-1])
        return tot

    def light_half(self, cands, load):
        cs = sorted(set(cands), key=lambda c: (load(c), c))
        return cs[:max(1, math.ceil(len(cs) / 2))]

    def comp_dest(self, beha, primary=None):
        kind = self.a.state[7][beha].comp_device
        load = lambda c: self.dev_load(kind, c)
        half = self.light_half(comp_cands(self.a, beha), load)
        key = lambda c: ((primary(c[:-1]) if primary else 0), self.nbr_cost(beha, c[:-1]), load(c), c)
        return random.choice(sorted(half, key=key)[:self.TOP_MOVE])

    def ddr_dest(self, d, sp):
        ev7 = self.a.state[7]
        readers = [ev7[t] for t in self.a.state[6][d].used_splitted_tag_dict[sp]
                   if t in ev7 and t[0] == 1 and ev7[t].event_type == "communication"]
        rd = collections.Counter()
        for e in ev7.values():
            if e.event_type == "communication":
                rd[tuple(e.source_location)] += e.comm_bytes
        load = lambda c: rd[tuple(c[:-1])]
        half = self.light_half(ddr_cands(self.a, d), load)
        dist = lambda c: sum(self.hop(c[:-1], r.target_location) for r in readers
                             if not isinstance(r.target_location, set))
        return random.choice(sorted(half, key=lambda c: (dist(c), load(c), c))[:self.TOP_MOVE])

    def swap_partner(self, b1, cp_tags, W):
        pool = [b for b in self.a.comp_tags_list if b != b1 and abs(b[2] - b1[2]) <= 1 and region_ok(self.a, b)]
        non = [b for b in pool if b not in cp_tags]
        pool = non or pool
        ev7 = self.a.state[7]
        ld = {b: self.dev_load(ev7[b].comp_device, ev7[b].comp_location) for b in pool}
        rw = {b: i for i, b in enumerate(sorted(pool, key=lambda b: (W(b), b)))}
        rl = {b: i for i, b in enumerate(sorted(pool, key=lambda b: (ld[b], b)))}
        return random.choice(sorted(pool, key=lambda b: (rw[b] + rl[b], b))[:self.TOP_SWAP])


# ------------------------------------------------------------------ child: the run
def child(cfg, arm, g, x, sink_path, snap_path, checks):
    model, task = task_of(cfg, arm, g, x)
    faults = CONFIGS[cfg][2]
    seed = 123 if arm in ("A0", "A1") else x
    task_path = os.path.join(WORK, "models", model, "py", task + ".py")
    for sub in SRC_SUBDIRS:
        p = os.path.join(WORK, sub)
        if p not in sys.path:
            sys.path.append(p)
    import Stream_Mapping
    import event_driver as ed_mod
    from add_communication import add_broadcast
    orig_ed = ed_mod.event_driver
    if faults:
        import read_cfg
        import WAMIS_HD
        orig_cfg = read_cfg.cfg_to_dict

        def cfg_to_dict(cfg_path):
            c = orig_cfg(cfg_path)
            c["failures"]["failed_nodes"] = [tuple(n) for n in faults["failed_nodes"]]
            c["failures"]["failed_links"] = [(tuple(a), tuple(b)) for a, b in faults["failed_links"]]
            return c
        read_cfg.cfg_to_dict = cfg_to_dict
        orig_xy = WAMIS_HD.wamis_hdc.xy_multicast_path

        def xy_multicast_path(self, source_node_idx, target_nodes_set):   # E3's fix (e3_warmstart.patch_module)
            try:
                return orig_xy(self, source_node_idx=source_node_idx, target_nodes_set=target_nodes_set)
            except TypeError as exc:
                if "NoneType" not in str(exc):
                    raise
                return None
        WAMIS_HD.wamis_hdc.xy_multicast_path = xy_multicast_path

    sink = {"meta": {"cfg": cfg, "arm": arm, "g": g, "x": x, "sa_seed": seed, "task": task, "faults": faults,
                     "P": P, "K": K, "backend_seeds": BACKEND_SEEDS, "bcast_seed": BCAST_SEED}}
    calls = [0]

    def event_driver(events_dict, hardware_platform):
        calls[0] += 1
        if calls[0] == 1:          # initial mapping: the template discards the result
            return (0, 0, 0)
        out = orig_ed(events_dict=events_dict, hardware_platform=hardware_platform)
        sink["template_out"] = tuple(int(v) for v in out)   # A0 only: the template's own final evaluation
        with gzip.open(sink_path, "wb") as f:
            pickle.dump(sink, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out
    ed_mod.event_driver = event_driver

    Stream_Mapping.StreamOptimization.anneal = make_anneal(arm, seed, sink, sink_path, snap_path, checks,
                                                           add_broadcast, orig_ed)
    os.makedirs(os.path.join(os.path.dirname(task_path), "..", "results"), exist_ok=True)
    runpy.run_path(task_path, run_name="__main__")


def cpcm_sim(ev, hp):
    g = cpcm.compile_bb(ev, hp)
    return g, cpcm.simulate(g, "bb")


def final_eval(snap, ctx, add_broadcast, event_driver):
    """cpcm_v3's evaluation of one state (snap: own copies of beha, data, events, platform): broadcast seeded
    BCAST_SEED, CPCM before / after the broadcast, backend seeds on deep copies; also the E3 mapping record
    (mapping, routes, layout) for check 3."""
    beha, data, events, hp = snap
    rng = random.getstate(), np.random.get_state()
    try:
        t_pre = cpcm_sim(events, hp)[1].T
        random.seed(BCAST_SEED)
        np.random.seed(BCAST_SEED)
        add_broadcast(data_tags=ctx["broadcast_datatags"],
                      ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]],
                      beha_dict=beha, data_dict=data, hardware_platform=hp, event_dict=events,
                      dijkstra_routing=ctx["dijkstra_routing"], alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])
        pre = e3._e2()._snapshot_pre(events)
        deps = {t: set(e.dependency_set) for t, e in events.items()}
        paths = {t: [list(b) for b in e.path_list] for t, e in events.items() if e.event_type == "communication"}
        t_bc = cpcm.simulate(cpcm.compile_bb(events, hp, path_lists=paths, deps=deps), "bb").T
        m = e3.mapping_of({"beha_dict": beha, "data_dict": data})
        m["routes"] = e3.routes_of(events, pre)
        m["layout"] = e3.event_layout_of(events)
        base = copy.deepcopy((events, hp))
        bk, wall = {}, []
        for s in BACKEND_SEEDS:
            ev_k, hp_k = copy.deepcopy(base)
            random.seed(s)
            np.random.seed(s)
            t0 = time.time()
            bk[s] = tuple(int(v) for v in event_driver(events_dict=ev_k, hardware_platform=hp_k))
            wall.append(time.time() - t0)
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
    return {"T_pre": t_pre, "T_bc": t_bc, "backend": bk, "backend_wall": wall, "e3map": m}


def make_anneal(arm, seed, sink, sink_path, snap_path, checks, add_broadcast, event_driver):
    cp_arm = arm in CPCM_ARMS

    def anneal(self):
        ctx = _find_context()
        orig_move = type(self).move
        rows, sink["rows"] = [], None
        st = {"g": None, "sc": None, "an": None, "T0": None, "d0": None}
        heur = Heur(self)
        tm = collections.Counter()
        c2, c4 = [], []
        crev = {"n": 0, "E_mismatch": 0, "T_mismatch": 0, "state_mismatch": 0}
        exact = EXACT_REVERT and arm != "A0"
        if self.LP_flag or self.mem_datatags:
            raise RuntimeError("template uses LP_flag / mem_datatags: not covered")

        def t_eval():
            """CPCM on the current SA state: (graph, schedule, event dict evaluated). With BROADCAST_T the template's
            final add_broadcast is applied as final_eval applies it (seed BCAST_SEED) to a temporary event dict: it
            only adds communications, each producer's issue_set entry and comm id, and platform link counters, so
            the producers are shallow-copied and the counters and RNG restored afterwards."""
            ev0, hp = self.state[7], self.hardware_platform
            if not BROADCAST_T:
                return (*cpcm_sim(ev0, hp), ev0)
            loads, counts = dict(hp.link_loads_dict), dict(hp.link_loads_count)
            rng = random.getstate(), np.random.get_state()
            ev = dict(ev0)
            for dt in ctx["broadcast_datatags"]:
                for prod in self.state[6][dt].generated_tag_splitted_dict:
                    c = copy.copy(ev0[prod])
                    c.issue_set = set(c.issue_set)
                    c.commid_allocator = copy.deepcopy(c.commid_allocator)
                    ev[prod] = c
            random.seed(BCAST_SEED)
            np.random.seed(BCAST_SEED)
            add_broadcast(data_tags=ctx["broadcast_datatags"],
                          ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]],
                          beha_dict=self.state[5], data_dict=self.state[6], hardware_platform=hp, event_dict=ev,
                          dijkstra_routing=ctx["dijkstra_routing"], alpha=ctx["alpha"], beta=ctx["beta"],
                          gamma=ctx["gamma"])
            g, sc = cpcm_sim(ev, hp)
            hp.link_loads_dict.update(loads)
            hp.link_loads_count.update(counts)
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            return g, sc, ev

        def evaluate():
            """CPCM on the current SA state; sets st (graph, schedule; analysis lazily)."""
            g, sc, ev = t_eval()
            st.update(g=g, sc=sc, an=None, ev=ev)
            return sc.T

        def before_move():
            """Exact revert: everything a rejected move can change, and the CPCM result of the current state."""
            hp = self.hardware_platform
            return {"blob": pickle.dumps(self.state, protocol=pickle.HIGHEST_PROTOCOL),
                    "loads": dict(hp.link_loads_dict), "counts": dict(hp.link_loads_count),
                    "st": {k: st[k] for k in ("g", "sc", "an", "ev") if k in st}}

        def exact_revert(pre):
            hp = self.hardware_platform
            self.state = pickle.loads(pre["blob"])
            hp.link_loads_dict.update(pre["loads"])
            hp.link_loads_count.update(pre["counts"])
            st.update(pre["st"])

        def fingerprint():
            """Check 9.3: tags and order, dependencies, issues, sources, routes, comm-id allocators, locations, the
            SA totals and the platform's counters."""
            hp, s_ = self.hardware_platform, self.state
            evs = []
            for t, e in s_[7].items():
                row = [t, sorted(map(str, e.dependency_set)), sorted(map(str, e.issue_set))]
                if e.event_type == "communication":
                    row += [e.source_location, str(e.target_location), e.comm_bytes, str(e.path_list), str(e.paths)]
                else:
                    a = e.commid_allocator
                    row += [e.comp_location, sorted(a.used_ids), sorted(a.recycled), a.next_id]
                evs.append(row)
            return (evs, {t: b.location for t, b in s_[5].items()}, s_[0], s_[1], dict(s_[2]), dict(s_[3]), dict(s_[4]),
                    dict(hp.link_loads_dict), dict(hp.link_loads_count))

        def energy_cp():
            T = evaluate()
            return T / st["T0"] + (0.01 / st["T0"]) * self.state[1] / st["d0"], T

        def energy():
            if cp_arm:
                return energy_cp()
            return self.energy(), None

        def analysis():
            if st["an"] is None:
                st["an"] = cpcm.Analysis(st["sc"])
            return st["an"]

        def W(e):
            sc = st["sc"]
            return sc.S_last[e] - sc.r[e]

        def pick(variant):
            an, g = analysis(), st["g"]
            rows_cp = an.critical_path()
            if variant == "cb":
                cb = {q: v for q, v in an.blocking().items() if v > 0}
                if cb:
                    q = max(cb, key=lambda q: (cb[q], -q))
                    evs = {r["event"] for r in rows_cp if r["edge"] == "resource" and r["res"] == q}
                    if evs:
                        top = sorted(evs, key=lambda e: (-W(e), e))[:K]
                        return random.choice(top), {"res": str(g.res_name[q]), "cb": cb[q], "n": len(evs)}
            evs = {r["event"] for r in rows_cp}
            top = sorted(evs, key=lambda e: (-W(e), e))[:K]
            return random.choice(top), {"n": len(evs)}

        def targeted(variant, heuristic, check_now=False):
            fresh = None
            if check_now:      # check 4: a fresh CPCM pass of the state the pick is made in (not timed)
                fg, fsc, _ = t_eval()
                fresh = (fsc.T, {fg.tag[r["event"]] for r in cpcm.Analysis(fsc).critical_path()})
            t0 = time.perf_counter()
            e, info = pick(variant)
            g = st["g"]
            tag = g.tag[e]
            if fresh is not None:
                info["check4"] = {"T_fresh": fresh[0], "T_cached": st["sc"].T, "on_fresh_cp": tag in fresh[1],
                                  "n_cp_events": len(fresh[1])}
            ev = self.state[7][tag] if tag in self.state[7] else st["ev"][tag]   # broadcast: evaluated dict only
            info.update(event=str(tag), W=W(e))
            ddr_nodes = {k[:-1] for k in self.hardware_platform.ddr_dict}
            if ev.event_type == "computation":
                info["action"] = "comp"
                dest = heur.comp_dest(tag) if heuristic else random.choice(comp_cands(self, tag))
                tm["target"] += time.perf_counter() - t0
                mv_relocate(self, tag, dest)
            else:
                ws = weight_split_of(self, tag) if tuple(ev.source_location) in ddr_nodes else None
                if ws is not None:
                    info["action"] = "weight"
                    dest = heur.ddr_dest(*ws) if heuristic else random.choice(ddr_cands(self, ws[0]))
                    tm["target"] += time.perf_counter() - t0
                    mv_weight(self, ws[0], ws[1], dest)
                else:
                    ev7 = self.state[7]
                    prods = sorted((t for t in ev.dependency_set if t in self.state[5]), key=str)
                    cons = sorted((t for t in ev.issue_set if t in self.state[5]), key=str)
                    sides = [s for s in (("producer", prods), ("consumer", cons)) if s[1]]
                    side, behas = random.choice(sides)
                    b = random.choice(behas)
                    info["action"] = side
                    if heuristic:
                        if side == "consumer":
                            src = ev.source_location
                            prim = lambda c: heur.hop(src, c)
                        else:
                            tgt = ev.target_location
                            prim = (lambda c: sum(heur.hop(c, n) for n in tgt)) if isinstance(tgt, set) \
                                else (lambda c: heur.hop(c, tgt))
                        dest = heur.comp_dest(b, primary=prim)
                    else:
                        dest = random.choice(comp_cands(self, b))
                    tm["target"] += time.perf_counter() - t0
                    mv_relocate(self, b, dest)
            info["tag"] = tag
            return info

        def heuristic_random():
            """The artifact's move with the A3h destination rule (sources drawn as the artifact draws them)."""
            t0 = time.perf_counter()
            change_flag = random.random() < self.random_threshold[0]
            change_mem = random.random() < self.mem_threshold[0]
            if change_flag:
                if change_mem and self.mem_enable:
                    d = random.choice(self.offlinedata_list)
                    sp = random.choice(self.offlinedata_split_dict[d])
                    dest = heur.ddr_dest(d, sp)
                    tm["target"] += time.perf_counter() - t0
                    mv_weight(self, d, sp, dest)
                    return {"action": "rand_weight"}
                b = random.choice(self.comp_tags_list)
                dest = heur.comp_dest(b)
                tm["target"] += time.perf_counter() - t0
                mv_relocate(self, b, dest)
                return {"action": "rand_comp"}
            b1 = random.choice(self.comp_tags_list)
            while not region_ok(self, b1):
                b1 = random.choice(self.comp_tags_list)
            an, g = analysis(), st["g"]
            cp_tags = {g.tag[r["event"]] for r in an.critical_path()}
            b2 = heur.swap_partner(b1, cp_tags, lambda b: W(g.id_of[b]))
            tm["target"] += time.perf_counter() - t0
            mv_swap(self, b1, b2)
            return {"action": "rand_swap"}

        def move(check_now=False):
            if arm in ("A0", "A1", "A2"):
                orig_move(self)
                return {"action": "rand"}
            if random.random() < P:
                info = targeted("cb" if arm == "A3CB" else "cp", arm == "A3h", check_now)
                info["targeted"] = True
                return info
            if arm == "A3h":
                return heuristic_random()
            orig_move(self)
            return {"action": "rand"}

        def snapshot():
            """(beha, data, events) pickled together (shared objects kept) + the platform's routing-history
            counters, the only platform state the SA changes (record_dijkstra_multicast_path); a deep copy of the
            platform costs ~1.4 s, so it is rebuilt once at the end (restore_snapshot)."""
            hp = self.hardware_platform
            return (pickle.dumps((self.state[5], self.state[6], self.state[7]), protocol=pickle.HIGHEST_PROTOCOL),
                    dict(hp.link_loads_dict), dict(hp.link_loads_count))

        def restore_snapshot(sn):
            b_, d_, e_ = pickle.loads(sn[0])
            hp = copy.deepcopy(self.hardware_platform)
            hp.link_loads_dict.clear()
            hp.link_loads_dict.update(sn[1])
            hp.link_loads_count.clear()
            hp.link_loads_count.update(sn[2])
            return b_, d_, e_, hp

        # ---- normalisation (CPCM arms) and temperature calibration (A1+)
        steps = int(os.environ.get("CPCM_SA_STEPS", self.steps))   # override for smoke tests only
        if cp_arm:
            st["d0"] = self.state[1] if self.state[1] else 1
            st["T0"] = 1          # placeholder for the first evaluation
            st["T0"] = evaluate()
        Tmax, Tmin = self.Tmax, self.Tmin
        cal = None
        if arm != "A0":
            saved = (self.state, self.hardware_platform, self.movements)
            self.state, self.hardware_platform = copy.deepcopy((self.state, self.hardware_platform))
            heur.a = self
            random.seed(seed + CAL_SEED_OFFSET)
            np.random.seed(seed + CAL_SEED_OFFSET)
            t0 = time.perf_counter()
            Ec, Tc = energy()
            dEs, worse = [], []
            while len(dEs) < CAL_PROBES or (len(worse) < CAL_MIN_WORSE and len(dEs) < CAL_MAX_PROBES):
                pre = before_move() if exact else None
                move()
                E1, T1 = energy()
                dEs.append(E1 - Ec)
                # worsening: dE > 0; CPCM arms: T increases (a change of the tie key alone is not a worsening)
                if E1 - Ec > 0 and (not cp_arm or T1 > Tc):
                    worse.append(abs(E1 - Ec))
                if exact:
                    exact_revert(pre)
                else:
                    self.revert()
                    Ec, Tc = energy()
            dE50 = statistics.median(worse) if worse else None
            if dE50:
                Tmax = -dE50 / math.log(0.5)
                Tmin = Tmax * 1e-4
            cal = {"dE": dEs, "n_worse": len(worse), "dE50": dE50, "Tmax": Tmax, "Tmin": Tmin,
                   "wall": time.perf_counter() - t0}
            self.state, self.hardware_platform, self.movements = saved
            heur.a = self
            if cp_arm:
                st["T0"] = st["T0"]   # unchanged
                evaluate()            # schedule of the real initial state
            tm.clear()
        sink["meta"]["ctx"] = {k: ctx[k] for k in ctx}
        sink["meta"].update(Tmax=Tmax, Tmin=Tmin, calibration=cal, T0=st["T0"], d0=st["d0"],
                            template_Tmax=self.Tmax, template_Tmin=self.Tmin, steps=steps,
                            loss_ratio=list(self.loss_ratio))

        # ---- the loop (the artifact's anneal, with the arm's energy / move / kept state)
        random.seed(seed)
        np.random.seed(seed)
        Tfactor = -math.log(Tmax / Tmin)
        E, Tcur = energy()
        prevE, prevT = E, Tcur
        best = {"E": E, "T": Tcur, "step": 0, "snap": snapshot() if arm != "A0" else None}
        mon = {}
        if not cp_arm:
            mon[0] = t_eval()[1].T
        wall0 = time.time()
        n4 = 0
        for step in range(1, steps + 1):
            T = Tmax * math.exp(Tfactor * step / steps)
            t0 = time.perf_counter()
            pre = before_move() if exact else None
            t_pre = time.perf_counter() - t0
            fp0 = fingerprint() if exact and checks else None
            t0 = time.perf_counter()
            tgt0 = tm["target"]
            info = move(checks and n4 < CHECK4_N)
            t1 = time.perf_counter()
            if "check4" in info:
                c4.append({"step": step, "tag": str(info["tag"]), "action": info["action"], **info.pop("check4")})
                n4 += 1
            E_new, T_new = energy()
            t2 = time.perf_counter()
            dE = E_new - prevE
            row = {"step": step, "temp": T, "E_new": E_new, "T_new": T_new, "dE": dE,
                   "t_move": (t1 - t0) - (tm["target"] - tgt0), "t_target": tm["target"] - tgt0,
                   "t_energy": t2 - t1, "action": info.get("action"), "targeted": bool(info.get("targeted"))}
            if info.get("targeted"):
                row.update(pick_event=info["event"], pick_W=info["W"], pick_n=info.get("n"), pick_res=info.get("res"))
            row["t_pre"] = t_pre
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                t3 = time.perf_counter()
                exact_revert(pre) if exact else self.revert()
                row["t_revert"] = time.perf_counter() - t3
                row["accepted"] = False
                if exact and checks:    # check 9.3: E of the restored state, recomputed, against the recorded value
                    keep = {k: st[k] for k in ("g", "sc", "an", "ev") if k in st}
                    E_re, T_re = energy()
                    st.update(keep)
                    crev["n"] += 1
                    crev["E_mismatch"] += E_re != prevE
                    crev["T_mismatch"] += T_re != prevT
                    crev["state_mismatch"] += fingerprint() != fp0
                if STAGE == "drift":    # measurement only: E / T of the reverted state; the loop keeps the old value
                    E_re, T_re = energy()
                    row.update(E_reverted=E_re, T_reverted=T_re)
                elif not exact and arm in TARGETED:     # targeting reads the schedule of the current (reverted, re-routed) state
                    t3 = time.perf_counter()
                    row["T_reverted"] = evaluate()
                    row["t_refresh"] = time.perf_counter() - t3
                E, Tcur = prevE, prevT
            else:
                row["accepted"] = True
                E, Tcur = E_new, T_new
                prevE, prevT = E, Tcur
                if arm != "A0" and E < best["E"]:
                    t3 = time.perf_counter()
                    best.update(E=E, T=Tcur, step=step, snap=snapshot())
                    row["t_snapshot"] = time.perf_counter() - t3
                c2_due = (step >= c2[-1]["step"] + 10 if c2 else True) if (STAGE or os.environ.get("CPCM_SA_CHECK2_ANY"))                     else step >= CHECK2_STEPS[min(len(c2), len(CHECK2_STEPS) - 1)]
                if checks and len(c2) < len(CHECK2_STEPS) and c2_due:   # CHECK2_ANY: first accepted states, >= 10 apart
                    ev_c2 = st["ev"] if cp_arm else t_eval()[2]
                    T_rec = Tcur if cp_arm else t_eval()[1].T
                    c2.append({"step": step, "T_rec": T_rec,
                               "state": pickle.dumps(ev_c2, protocol=pickle.HIGHEST_PROTOCOL)})
            if not cp_arm and step % MONITOR_EVERY == 0:
                mon[step] = t_eval()[1].T
            row["best_E"] = best["E"]
            rows.append(row)
        sa_wall = time.time() - wall0
        if arm == "A0":
            sel = {"E": E, "T": None, "step": steps, "snap": snapshot()}
        else:
            sel = best
        t0 = time.time()
        fe = final_eval(restore_snapshot(sel["snap"]), ctx, add_broadcast, event_driver)
        fe["eval_wall"] = time.time() - t0
        if os.environ.get("CPCM_SA_SNAPTEST") and arm == "A0":   # smoke test: rebuilt snapshot == full deep copy
            full = final_eval(copy.deepcopy((self.state[5], self.state[6], self.state[7], self.hardware_platform)),
                              ctx, add_broadcast, event_driver)
            sink["snaptest"] = {"T_pre": (fe["T_pre"], full["T_pre"]), "T_bc": (fe["T_bc"], full["T_bc"]),
                                "routes_equal": fe["e3map"]["routes"] == full["e3map"]["routes"],
                                "backend": (fe["backend"], full["backend"])}
        sink.update(rows=rows, monitor=mon, sa_wall=sa_wall, check2=c2, check4=c4, check_revert=crev,
                    selected={"own_cost": sel["E"], "T_sel": sel["T"], "step": sel["step"], **fe})
        with gzip.open(snap_path, "wb") as f:     # the platform holds lambdas: only its routing-history counters
            pickle.dump((sel["snap"][0], {"link_loads_dict": sel["snap"][1], "link_loads_count": sel["snap"][2]}),
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open(sink_path, "wb") as f:
            pickle.dump(sink, f, protocol=pickle.HIGHEST_PROTOCOL)
        if arm != "A0":
            raise SystemExit(0)
        return self.state, E      # A0: back to the template's own final evaluation (self-check 1)

    return anneal


# ------------------------------------------------------------------ parent
def run_one(cfg, arm, g, x, checks):
    name = run_name(cfg, arm, g, x)
    sink = os.path.join(OUT, "runs", name + ".pkl.gz")
    snap = os.path.join(OUT, "snaps", name + ".pkl.gz")
    if os.path.exists(sink):
        return name, "cached", 0.0
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "snaps"), exist_ok=True)
    t0 = time.time()
    with open(os.path.join(OUT, "logs", name + ".log"), "w", encoding="utf-8") as lf:
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), "child", cfg, arm, str(g), str(x),
                              sink, snap, "1" if checks else "0"], cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
    ok = rc == 0 and os.path.exists(sink)
    print(f"[cpcm_sa] {name}: exit {rc}, {time.time() - t0:.0f} s, {'ok' if ok else 'FAILED'}", flush=True)
    return name, "ok" if ok else "failed", time.time() - t0


def run(configs, arms, jobs, only_first=False):
    todo = []
    for cfg in configs:
        for arm in arms:
            vs = variants(arm)
            for i, (g, x) in enumerate(vs[:(2 if STAGE else 1)] if only_first else vs):
                todo.append((cfg, arm, g, x, i < (2 if STAGE else 1)))   # stage 1.5: checks on 2 runs per arm
    todo.sort(key=lambda t: (t[1] not in CPCM_ARMS, t[0] != "mha"))   # long ones first
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(lambda t: run_one(*t), todo))
    bad = [r for r in res if r[1] == "failed"]
    print(f"[cpcm_sa] {len(res)} runs, {len(bad)} failed: {[b[0] for b in bad]}", flush=True)


_PLAT = {}


def platform(cfg):
    """A fresh platform of the config (the template's cfg plus the config's faults; E3's xy fix), for CPCM
    durations / topology outside the run process."""
    if cfg not in _PLAT:
        _artifact_path()
        import read_cfg
        import WAMIS_HD
        faults = CONFIGS[cfg][2]
        c = read_cfg.cfg_to_dict(os.path.join(WORK, "models", CONFIGS[cfg][0], "cfg", "config" + SUF + ".cfg"))
        if faults:
            c["failures"]["failed_nodes"] = [tuple(n) for n in faults["failed_nodes"]]
            c["failures"]["failed_links"] = [(tuple(a), tuple(b)) for a, b in faults["failed_links"]]
            orig_xy = WAMIS_HD.wamis_hdc.xy_multicast_path

            def xy_multicast_path(self, source_node_idx, target_nodes_set):
                try:
                    return orig_xy(self, source_node_idx=source_node_idx, target_nodes_set=target_nodes_set)
                except TypeError as exc:
                    if "NoneType" not in str(exc):
                        raise
                    return None
            WAMIS_HD.wamis_hdc.xy_multicast_path = xy_multicast_path
        _PLAT[cfg] = WAMIS_HD.wamis_hdc(c)
    return _PLAT[cfg]


def load_run(cfg, arm, g, x, base=None):
    _artifact_path()
    with gzip.open(os.path.join(base or OUT, "runs", run_name(cfg, arm, g, x) + ".pkl.gz"), "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ check 3 (E3 injection of a selected state)
def check3_child_hook():
    """Called from _inject/sitecustomize.py inside an E3 inject-mode process (R0: no SA): CPCM on the injected
    pre-broadcast state, written to CPCM_SA_SINK, then exit."""
    import Stream_Mapping
    inner = Stream_Mapping.stream_mapping

    def stream_mapping(*args, **kw):
        g, sc = cpcm_sim(kw["event_dict"], kw["hardware_platform"])
        ctx = _find_context()
        ev, hp = copy.deepcopy((kw["event_dict"], kw["hardware_platform"]))
        random.seed(BCAST_SEED)
        np.random.seed(BCAST_SEED)
        sys.modules["add_communication"].add_broadcast(   # E3's wrapper: the captured broadcast routes come back
            data_tags=ctx["broadcast_datatags"], ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]],
            beha_dict=kw["beha_dict"], data_dict=kw["data_dict"], hardware_platform=hp, event_dict=ev,
            dijkstra_routing=ctx["dijkstra_routing"], alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])
        with open(os.environ["CPCM_SA_SINK"], "wb") as f:
            pickle.dump({"T": sc.T, "T_bc": cpcm_sim(ev, hp)[1].T, "n": g.n, "log": e3._spec()["_log"]}, f)
        raise SystemExit(0)
    Stream_Mapping.stream_mapping = stream_mapping
    return inner


def check3(configs, arms, jobs):
    tmp = os.path.join(OUT, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    def one(ca):
        cfg, arm = ca
        g, x = variants(arm)[0]
        d = load_run(cfg, arm, g, x)
        model, task = task_of(cfg, arm, g, x)
        faults = CONFIGS[cfg][2] or {"failed_nodes": [], "failed_links": []}
        spec = os.path.join(tmp, f"spec_{cfg}_{arm}.pkl")
        out = os.path.join(tmp, f"c3_{cfg}_{arm}.pkl")
        with open(spec, "wb") as f:
            pickle.dump({"mapping": d["selected"]["e3map"], "faults": faults, "steps": None, "t_max": None,
                         "seeds": [], "preserve_routes": True}, f)
        env = dict(os.environ, E3_MODE="inject", E3_SPEC=spec, E3_HOME=os.path.join(ROOT, "e3_warmstart"),
                   E3_SINK=os.path.join(tmp, f"e3sink_{cfg}_{arm}.pkl"), CPCM_SA_SINK=out, CPCM_SA_HOME=HERE,
                   PYTHONPATH=os.path.join(HERE, "_inject"))
        path = os.path.join(WORK, "models", model, "py", task + ".py")
        with open(os.path.join(OUT, "logs", f"check3_{cfg}_{arm}.log"), "w", encoding="utf-8") as lf:
            rc = subprocess.call([sys.executable, path], env=env, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
        if not os.path.exists(out):
            return {"cfg": cfg, "arm": arm, "rc": rc, "ok": False}
        with open(out, "rb") as f:
            r = pickle.load(f)
        rec = d["selected"]
        rt = r["log"].get("routes", [{}])[0]
        T_sa = r["T_bc"] if BROADCAST_T else r["T"]      # what the SA recorded as its best T
        return {"cfg": cfg, "arm": arm, "T_inject": r["T"], "T_pre_final_eval": rec["T_pre"], "T_sel_in_SA": rec["T_sel"],
                "T_bc_inject": r["T_bc"], "T_bc_final_eval": rec["T_bc"],
                "equal": r["T"] == rec["T_pre"] and r["T_bc"] == rec["T_bc"] and (rec["T_sel"] is None or T_sa == rec["T_sel"]),
                "restored": rt.get("restored"), "rerouted": rt.get("rerouted")}

    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(one, [(c, a) for c in configs for a in arms]))
    for r in res:
        print(r)
    with open(os.path.join(OUT, "check3.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)


# ------------------------------------------------------------------ routing-history diagnostic
def diag_child(cfg, arm, g, x, out_path):
    _artifact_path()
    from add_communication import add_broadcast
    from event_driver import event_driver
    ctx = load_run(cfg, arm, g, x)["meta"]["ctx"]     # the template's broadcast arguments, saved by the run
    snap = load_snap(cfg, arm, g, x)
    res = {}
    for mode in ("i_history", "ii_fresh_loads", "iii_cb_weights"):
        beha, data, events, hp = copy.deepcopy(snap)
        random.seed(BCAST_SEED)
        np.random.seed(BCAST_SEED)
        g0, sc0 = cpcm_sim(events, hp)
        comms = [t for t, e in events.items() if e.event_type == "communication"]
        old = {t: [list(b) for b in events[t].path_list] for t in comms}
        cbw = None
        if mode != "i_history":
            if mode == "iii_cb_weights":
                cb = cpcm.Analysis(sc0).blocking()
                cbw = {g0.res_name[q][1]: v for q, v in cb.items() if g0.res_name[q][0] == "link"}
                base_loads = {k: cbw.get(k, 0) for k in hp.link_loads_dict}
            else:
                base_loads = {k: 0 for k in hp.link_loads_dict}
            for k in hp.link_loads_dict:
                hp.link_loads_dict[k] = base_loads[k]
                hp.link_loads_count[k] = 0
            for t in comms:
                e = events[t]
                tgt = e.target_location if isinstance(e.target_location, set) else {e.target_location}
                e.paths = hp.record_dijkstra_multicast_path(comm_pairs=[[t, e.source_location, tgt, e.comm_bytes]],
                                                            alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])[2][t]
                e.path_list = e.get_paths(e.paths)
                if mode == "iii_cb_weights":     # fixed weights: every message sees CB, not earlier messages
                    for k in hp.link_loads_dict:
                        hp.link_loads_dict[k] = base_loads[k]
        changed = sum(1 for t in comms if [list(b) for b in events[t].path_list] != old[t])
        T_pre = cpcm_sim(events, hp)[1].T
        add_broadcast(data_tags=ctx["broadcast_datatags"],
                      ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]],
                      beha_dict=beha, data_dict=data, hardware_platform=hp, event_dict=events,
                      dijkstra_routing=ctx["dijkstra_routing"], alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])
        deps = {t: set(e.dependency_set) for t, e in events.items()}
        paths = {t: [list(b) for b in e.path_list] for t, e in events.items() if e.event_type == "communication"}
        T_bc = cpcm.simulate(cpcm.compile_bb(events, hp, path_lists=paths, deps=deps), "bb").T
        base = copy.deepcopy((events, hp))
        bk = {}
        for s in BACKEND_SEEDS:
            ev_k, hp_k = copy.deepcopy(base)
            random.seed(s)
            np.random.seed(s)
            bk[s] = tuple(int(v) for v in event_driver(events_dict=ev_k, hardware_platform=hp_k))
        res[mode] = {"comms": len(comms), "routes_changed": changed, "T_pre": T_pre, "T_bc": T_bc, "backend": bk,
                     "cb_links": len(cbw) if cbw is not None else None}
        print(mode, res[mode], flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)


def load_snap(cfg, arm, g, x):
    """Saved selected state with a rebuilt platform carrying the run's routing-history counters."""
    with gzip.open(os.path.join(OUT, "snaps", run_name(cfg, arm, g, x) + ".pkl.gz"), "rb") as f:
        blob, hist = pickle.load(f)
    b_, d_, e_ = pickle.loads(blob)
    hp_ = copy.deepcopy(platform(cfg))
    for k in ("link_loads_dict", "link_loads_count"):
        if set(getattr(hp_, k)) != set(hist[k]):
            raise RuntimeError(f"{k}: link set of the rebuilt platform differs from the run's")
        getattr(hp_, k).update(hist[k])
    return b_, d_, e_, hp_


def reeval_child(cfg, arm, g, x):
    """Re-evaluate a finished run's selected state with the current BACKEND_SEEDS in this (fresh) process; the
    in-run 5-seed result is kept as backend_inrun_5."""
    _artifact_path()
    from add_communication import add_broadcast
    from event_driver import event_driver
    path = os.path.join(OUT, "runs", run_name(cfg, arm, g, x) + ".pkl.gz")
    d = load_run(cfg, arm, g, x)
    sel = d["selected"]
    if sel.get("reeval"):
        return
    t0 = time.time()
    fe = final_eval(load_snap(cfg, arm, g, x), d["meta"]["ctx"], add_broadcast, event_driver)
    if (fe["T_pre"], fe["T_bc"]) != (sel["T_pre"], sel["T_bc"]):
        raise RuntimeError(f"rebuilt state differs: T {fe['T_pre']}/{fe['T_bc']} vs {sel['T_pre']}/{sel['T_bc']}")
    sel.update(backend_inrun_5=sel["backend"], backend=fe["backend"], backend_wall=fe["backend_wall"], reeval=True,
               reeval_wall=time.time() - t0, reeval_routes_equal=fe["e3map"]["routes"] == sel["e3map"]["routes"])
    d["meta"]["backend_seeds"] = BACKEND_SEEDS
    with gzip.open(path + ".tmp", "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    print(f"{cfg} {arm} g{g} {x}: 10-seed mean {bmean(fe['backend']):.0f}, in-run 5-seed {bmean(sel['backend_inrun_5']):.0f}")


def reeval(configs, arms, jobs):
    todo = [(c, a, g, x) for c in configs for a in arms for g, x in variants(a)]

    def one(t):
        name = run_name(*t)
        with open(os.path.join(OUT, "logs", f"reeval_{name}.log"), "w", encoding="utf-8") as lf:
            rc = subprocess.call([sys.executable, os.path.abspath(__file__), "reeval_child", *map(str, t)],
                                 cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
        print(f"[reeval] {name}: exit {rc}", flush=True)
        return rc
    with ThreadPoolExecutor(jobs) as ex:
        rcs = list(ex.map(one, todo))
    print(f"[reeval] {len(rcs)} runs, {sum(1 for r in rcs if r)} failed", flush=True)


def diag(configs, jobs):
    rows = summarise(configs, ["A1", "A3"])
    todo = []
    for cfg in configs:
        for arm in ("A1", "A3"):
            r = rows[(cfg, arm)]
            todo.append((cfg, arm, r["best_backend_run"]))
    os.makedirs(os.path.join(OUT, "diag"), exist_ok=True)

    def one(t):
        cfg, arm, (g, x) = t
        out = os.path.join(OUT, "diag", f"{cfg}_{arm}.json")
        with open(os.path.join(OUT, "logs", f"diag_{cfg}_{arm}.log"), "w", encoding="utf-8") as lf:
            rc = subprocess.call([sys.executable, os.path.abspath(__file__), "diag_child", cfg, arm, str(g), str(x), out],
                                 cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
        print(f"[diag] {cfg} {arm}: exit {rc}", flush=True)
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(one, todo))


# ------------------------------------------------------------------ analysis
def bmean(bk):
    return statistics.mean(v[0] for v in bk.values())


def bmean_json(bk):
    return statistics.mean(v[0] for v in bk.values())


def spread(bk):
    v = [x[0] for x in bk.values()]
    return min(v), max(v)


def summarise(configs, arms):
    out = {}
    for cfg in configs:
        for arm in arms:
            runs = []
            for g, x in variants(arm):
                try:
                    d = load_run(cfg, arm, g, x)
                except FileNotFoundError:
                    continue
                s = d["selected"]
                runs.append({"g": g, "x": x, "mean": bmean(s["backend"]), "mn": spread(s["backend"])[0],
                             "mx": spread(s["backend"])[1], "own": s["own_cost"], "T_pre": s["T_pre"],
                             "T_bc": s["T_bc"], "d": d})
            if not runs:
                continue
            bb = min(runs, key=lambda r: (r["mean"], r["g"], r["x"]))
            oc = min(runs, key=lambda r: (r["own"], r["g"], r["x"]))
            out[(cfg, arm)] = {"n": len(runs), "runs": runs, "best_backend": bb, "best_own": oc,
                               "best_backend_run": (bb["g"], bb["x"]), "median": statistics.median(r["mean"] for r in runs)}
    return out


def _artifact_path():
    for sub in SRC_SUBDIRS:
        p = os.path.join(WORK, sub)
        if p not in sys.path:
            sys.path.append(p)


def last_steps(d):
    rows = d["rows"]
    return max((r["step"] for r in rows if r["accepted"]), default=0), d["selected"]["step"]


def triple(bk):
    return [round(statistics.mean(v[i] for v in bk.values()), 1) for i in range(3)]


def report15(configs, arms):
    """Stage 1.5 table: selected mappings with the backend triple, change vs A1 of this stage and vs the same arm of
    stage 1 (10 seeds there too), last accepted / last best-refreshing step, check 9.3 counters."""
    S = summarise(configs, arms)
    base = OUT_ROOT
    tab, noise = [], []
    for (cfg, arm), s in S.items():
        bb, oc = s["best_backend"], s["best_own"]
        noise += [(r["mx"] - r["mn"]) / r["mean"] for r in (bb, oc)]
        s1 = [load_run(cfg, arm, g, x, base) for g, x in variants(arm)]
        m1 = [bmean(d["selected"]["backend"]) for d in s1]
        o1 = min(s1, key=lambda d: d["selected"]["own_cost"])
        l1, l15 = [last_steps(d) for d in s1], [last_steps(r["d"]) for r in s["runs"]]
        rv = collections.Counter()
        for r in s["runs"]:
            rv.update(r["d"].get("check_revert") or {})
        tab.append({"cfg": cfg, "arm": arm, "n": s["n"],
                    "best_backend": round(bb["mean"], 1), "min": bb["mn"], "max": bb["mx"], "run": f"g{bb['g']}_{bb['x']}",
                    "best_backend_triple": triple(bb["d"]["selected"]["backend"]),
                    "best_own": round(oc["mean"], 1), "best_own_run": f"g{oc['g']}_{oc['x']}",
                    "best_own_triple": triple(oc["d"]["selected"]["backend"]),
                    "median": round(s["median"], 1), "worst": round(max(r["mean"] for r in s["runs"]), 1),
                    "stage1_best_backend": round(min(m1), 1), "stage1_best_own": round(bmean(o1["selected"]["backend"]), 1),
                    "stage1_median": round(statistics.median(m1), 1),
                    "last_accept_median": statistics.median(x[0] for x in l15),
                    "last_best_median": statistics.median(x[1] for x in l15),
                    "last_best_max": max(x[1] for x in l15),
                    "stage1_last_accept_median": statistics.median(x[0] for x in l1),
                    "stage1_last_best_median": statistics.median(x[1] for x in l1),
                    "Tmax_median": statistics.median(r["d"]["meta"]["Tmax"] for r in s["runs"]),
                    "stage1_Tmax_median": statistics.median(d["meta"]["Tmax"] for d in s1),
                    "accept_per_100_median": [round(statistics.median(
                        sum(1 for x in r["d"]["rows"][i:i + 100] if x["accepted"]) for r in s["runs"]), 1)
                        for i in range(0, 1000, 100)],
                    "Tbc_over_Tpre_median": round(statistics.median(r["T_bc"] / r["T_pre"] for r in s["runs"]), 3),
                    "Tsel_eq_Tbc": sum(1 for r in s["runs"] if r["d"]["selected"]["T_sel"] == r["T_bc"]),
                    "Tsel_eq_Tpre": sum(1 for r in s["runs"] if r["d"]["selected"]["T_sel"] == r["T_pre"]),
                    "err_Tbc_vs_backend_median": round(statistics.median((r["T_bc"] - r["mean"]) / r["mean"] for r in s["runs"]), 4),
                    "check_revert": dict(rv),
                    "sa_wall_median": round(statistics.median(r["d"]["sa_wall"] for r in s["runs"]), 1)})
    for t in tab:
        a1 = next(u for u in tab if u["cfg"] == t["cfg"] and u["arm"] == "A1")
        for k in ("best_backend", "best_own", "median"):
            t[k + "_vs_A1"] = round(t[k] / a1[k] - 1, 4)
            t[k + "_vs_stage1"] = round(t[k] / t["stage1_" + k] - 1, 4)
    res = {"stage": STAGE, "noise_median": statistics.median(noise), "table": tab}
    with open(os.path.join(OUT, "report15.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print("noise", res["noise_median"])
    for t in tab:
        print(t)
    return res


def drift_cmd(jobs):
    """Pre-fix drift of A1 / A2 (handoff 9.4): 2 runs per config x arm under CPCM_SA_STAGE=drift."""
    todo = [(c, a, *variants(a)[i], False) for c in ("mha", "ffn") for a in ("A1", "A2") for i in (0, 7)]
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda t: run_one(*t), todo))
    out = []
    for c, a, g, x, _ in todo:
        d = load_run(c, a, g, x)
        recE = trueE = None
        n = diff = fake = 0
        rel = []
        last = 0
        for r in d["rows"]:
            if recE is None:
                recE = trueE = r["E_new"] - r["dE"]
            if r["accepted"]:
                recE = trueE = r["E_new"]
                last = r["step"]
            else:
                n += 1
                fake += r["E_new"] < trueE
                diff += r["E_reverted"] != recE
                rel.append((r["E_reverted"] - recE) / recE)
                trueE = r["E_reverted"]
        out.append({"cfg": c, "arm": a, "g": g, "x": x, "rejected": n, "reverted_differs": diff,
                    "better_than_true_but_rejected": fake, "E_rel_median": statistics.median(rel),
                    "E_rel_max": max(rel), "E_rel_end": rel[-1], "last_accept": last})
        print(out[-1])
    with open(os.path.join(OUT, "drift.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def analyse(configs, arms):
    _artifact_path()
    S = summarise(configs, arms)
    table, proc, chk = [], [], {"check1": [], "check2": [], "check4": []}
    noise = []
    for (cfg, arm), s in S.items():
        bb, oc = s["best_backend"], s["best_own"]
        noise += [(r["mx"] - r["mn"]) / r["mean"] for r in (bb, oc)]
        table.append({"cfg": cfg, "arm": arm, "n": s["n"],
                      "sel_backend_mean": round(bb["mean"], 1), "sel_backend_min": bb["mn"], "sel_backend_max": bb["mx"],
                      "sel_backend_run": f"g{bb['g']}_{bb['x']}",
                      "sel_own_mean": round(oc["mean"], 1), "sel_own_min": oc["mn"], "sel_own_max": oc["mx"],
                      "sel_own_run": f"g{oc['g']}_{oc['x']}",
                      "median_of_runs": round(s["median"], 1),
                      "best_of_runs": round(min(r["mean"] for r in s["runs"]), 1),
                      "worst_of_runs": round(max(r["mean"] for r in s["runs"]), 1)})
        # process quantities
        acc_bins, t_parts, best_T_curve = [], collections.defaultdict(list), []
        for r in s["runs"]:
            rows = r["d"]["rows"]
            acc_bins.append([statistics.mean(1.0 if x["accepted"] else 0.0 for x in rows[i:i + 100])
                             for i in range(0, len(rows), 100)])
            for k in ("t_move", "t_target", "t_energy", "t_revert", "t_refresh", "t_snapshot"):
                t_parts[k].append(sum(x.get(k, 0.0) for x in rows))
            t_parts["sa_wall"].append(r["d"]["sa_wall"])
            t_parts["eval_wall"].append(r["d"]["selected"]["eval_wall"])
            cal = r["d"]["meta"]["calibration"]
            t_parts["cal_wall"].append(cal["wall"] if cal else 0.0)
            if arm in CPCM_ARMS:
                T0 = r["d"]["meta"]["T0"]
                cur, bt = T0, []
                for x in rows:
                    if x["accepted"] and x["T_new"] < cur:
                        cur = x["T_new"]
                    if x["step"] % 100 == 0:
                        bt.append(cur / T0)
                best_T_curve.append(bt)
            else:
                m = r["d"]["monitor"]
                best_T_curve.append([m[k] / m[0] for k in sorted(m) if k % 100 == 0 and k > 0])
        med = lambda v: round(statistics.median(v), 4)
        proc.append({"cfg": cfg, "arm": arm,
                     "acc_rate_per_100_steps_median": [med([a[i] for a in acc_bins]) for i in range(len(acc_bins[0]))],
                     "T_curve_median": [med([b[i] for b in best_T_curve]) for i in range(len(best_T_curve[0]))],
                     "per_run_s_median": {k: med(v) for k, v in t_parts.items()},
                     "Tmax_median": med([r["d"]["meta"]["Tmax"] for r in s["runs"]]),
                     "targeted_actions": dict(collections.Counter(x["action"] for r in s["runs"] for x in r["d"]["rows"]
                                                                  if x["targeted"])) if arm in TARGETED else None,
                     "best_step_median": statistics.median(r["d"]["selected"]["step"] for r in s["runs"])})
        # self-checks
        for r in s["runs"]:
            d = r["d"]
            if arm == "A0" and "template_out" in d:
                t = d["template_out"][0]
                m5 = bmean(d["selected"].get("backend_inrun_5", d["selected"]["backend"]))
                chk["check1"].append({"cfg": cfg, "run": f"g{r['g']}_{r['x']}", "inrun_5seed_mean": m5,
                                      "reeval_10seed_mean": r["mean"], "template": t, "rel": (m5 - t) / t,
                                      "rel_10": (r["mean"] - t) / t, "exempt": (cfg, r["g"], r["x"]) in CHECK1_EXEMPT})
            for c in d.get("check2") or []:
                ev, hp = pickle.loads(c["state"]), platform(cfg)
                chk["check2"].append({"cfg": cfg, "arm": arm, "step": c["step"], "T_rec": c["T_rec"],
                                      "T_offline": cpcm_sim(ev, hp)[1].T})
            for c in d.get("check4") or []:
                chk["check4"].append({"cfg": cfg, "arm": arm, **c})
    nz = statistics.median(noise) if noise else None
    for t in table:
        a1 = next((u for u in table if u["cfg"] == t["cfg"] and u["arm"] == "A1"), None)
        if a1:
            for k in ("sel_backend_mean", "sel_own_mean", "median_of_runs", "best_of_runs"):
                t[k + "_vs_A1"] = round(t[k] / a1[k] - 1, 4)
    diagres = {}
    for fn in sorted(os.listdir(os.path.join(OUT, "diag"))) if os.path.isdir(os.path.join(OUT, "diag")) else []:
        with open(os.path.join(OUT, "diag", fn), encoding="utf-8") as f:
            dd = json.load(f)
        diagres[fn[:-5]] = {m: {"mean": bmean(v["backend"]), "min": spread(v["backend"])[0],
                                "max": spread(v["backend"])[1], "T_pre": v["T_pre"], "T_bc": v["T_bc"],
                                "routes_changed": v["routes_changed"], "comms": v["comms"]} for m, v in dd.items()}
    c3 = None
    if os.path.exists(os.path.join(OUT, "check3.json")):
        with open(os.path.join(OUT, "check3.json"), encoding="utf-8") as f:
            c3 = json.load(f)
    res = {"table": table, "process": proc, "backend_noise_median": nz, "diag": diagres, "check3": c3,
           "e3_f5_reference": E3_F5,
           "check1_nonexempt_max_abs": max((abs(c["rel"]) for c in chk["check1"] if not c["exempt"]), default=None),
           "check4_on_cp": sum(1 for c in chk["check4"] if c["on_fresh_cp"]),
           "check4_T_equal": sum(1 for c in chk["check4"] if c["T_fresh"] == c["T_cached"]),
           "check1_max_abs": max((abs(c["rel"]) for c in chk["check1"]), default=None),
           "check1": chk["check1"],
           "check2_mismatch": sum(1 for c in chk["check2"] if c["T_rec"] != c["T_offline"]), "check2_n": len(chk["check2"]),
           "check4": chk["check4"], "artifact_commit": open(COMMIT_FILE, encoding="utf-8-sig").readline().strip()}
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    with open(os.path.join(OUT, "table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for t in table for k in t)))
        w.writeheader()
        w.writerows(table)
    for t in table:
        print(t)
    print("noise median", nz, "check1 max", res["check1_max_abs"], "check2", res["check2_mismatch"], "/", res["check2_n"])
    return res


def main():
    if sys.argv[1] == "child":
        _, _, cfg, arm, g, x, sink, snap, checks = sys.argv
        child(cfg, arm, int(g), int(x), sink, snap, checks == "1")
        return
    if sys.argv[1] == "reeval_child":
        _, _, cfg, arm, g, x = sys.argv
        reeval_child(cfg, arm, int(g), int(x))
        return
    if sys.argv[1] == "diag_child":
        _, _, cfg, arm, g, x, out = sys.argv
        diag_child(cfg, arm, int(g), int(x), out)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "reeval", "check3", "diag", "analyse", "report15", "drift"])
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--first", action="store_true", help="only the first variant of each config x arm")
    a = ap.parse_args()
    t0 = time.time()
    if a.cmd == "run":
        run(a.configs, a.arms, a.jobs, a.first)
    elif a.cmd == "reeval":
        reeval(a.configs, a.arms, a.jobs)
    elif a.cmd == "report15":
        report15(a.configs, a.arms)
    elif a.cmd == "drift":
        drift_cmd(a.jobs)
    elif a.cmd == "check3":
        check3(a.configs, a.arms, a.jobs)
    elif a.cmd == "diag":
        diag(a.configs, a.jobs)
    else:
        analyse(a.configs, a.arms)
    print(f"[cpcm_sa] {a.cmd} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
