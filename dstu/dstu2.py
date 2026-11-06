"""DSTU stage 2 (dstu/DSTU_HANDOFF.md section 9, dstu/STAGE2_EVAL.md). Nothing in cpcm/, e4_overlap/, third_party/ is changed.

2A  chain MHA -> FFN (zero-lag edges from MHA's sinks to FFN's sources on the merged graph) against a competitor X.
    Every method only sets D_M, the key of the MHA stage; FFN's key is d_req. Each row reduces to two binary orders, so
    the methods map to fixed rankings: only the rankings that occur are run, the rows are post-processing.
2B  selective chunking of a2 MHA+FFN with the fixed stream order C4; every run rebuilds from the unchunked events.
CM  C-M mechanism ablation: backward pass on the merged schedule with the cross-stream resource edges masked.

Commands (cwd dstu/):
  python dstu2.py check             chunk_by against E.chunk_events (isomorphic); masked backward pass unit check
  python dstu2.py enum              2A keys only (no simulation)
  python dstu2.py 2a --jobs 14
  python dstu2.py pilot --jobs 14
  python dstu2.py 2b --jobs 14      [--pool-b] (decided by the pilot, rule in PROGRESS)
  python dstu2.py greedy --jobs 14  [--pool-b]
  python dstu2.py cm --jobs 14
Writes dstu/out/stage2/ (sims/, logs/, _tmp/ stay out of git).
"""
import argparse
import collections
import copy
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import dstu as D
from dstu import B, E, cpcm

OUT = os.path.join(D.OUT, "stage2")
SIMS = os.path.join(OUT, "sims")
LIMIT = 3600  # section 9.5: one process over 1 h stops the run
TH = 0.04
K256, K64 = 256 * 1024, 64 * 1024
A2 = "a2_2s_mha-ffn"
BASE = {"size": K256, "only_d2d": True}
HWS = ("nofault", "F4", "F5")
KR = [1.1, 1.2, 1.5, 2, 3]
KX = [1.1, 1.2, 1.5, 2, 3]
RX = (0.0, 0.5)
BUDGETS = (540, 1620)
SEL_SEED = 20261107
METHODS_2A = ["UD", "HALF", "R03", "AGG_ED", "BYTES", "PRE_ED", "PRE_EQF", "CP_ED", "CP_EQF", "CP_EQS", "SOLO_ED", "SOLO_EQF"]
# a1 pilots (STAGE2_EVAL), budget = 10% of the blocks uniform 64 KB D2D adds (540 / 5,487 on a2)
A1 = "a1_2s_512-8192"
CM_ROWS = [("a2_2s_mha-ffn_F4_d0", "k1.5"), ("a2_2s_mha-ffn_F5_d0", "k1.5"), ("a1_2s_2048-8192_F4_d0", "k1.5"),
           ("a1_3s_512-2048-8192_F4_long_first", "k1.5"), ("a1_2s_2048-2048_F4_d0", "k1.5"),
           ("a1_3s_512-2048-8192_nofault_long_first", "k1.1")]


def log(m):
    print(f"[dstu2] {m}", flush=True)


def write_json(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


def write_csv(name, rows):
    os.makedirs(OUT, exist_ok=True)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ chunking per message
def gid(tag):
    return (tag[1],) + B.group_of(B.rt(tag))


def chunk_by(ev, hp, plan):
    """E.chunk_events with the block size chosen per message. plan = {"size", "only_d2d", "override": {gid: size}}:
    an overridden message uses its own size; the others follow (size, only_d2d). Only the choice of the size differs
    from E.chunk_events; dependency rewriting, tags and statistics are the same."""
    over = plan.get("override") or {}
    CB = E.CHUNK_ID_BASE
    out, stats = {}, {"messages_chunked": 0, "blocks_added": 0, "bytes_conserved": True, "tree_conserved": True,
                      "extra_occupancy": 0, "extra_occupancy_d2d": 0, "base_occupancy": 0, "extra_by_kind": {}}
    blocks_of = {}
    for t, e in ev.items():
        if e.event_type != "communication":
            continue
        links = [(r, l) for r, leaves in e.paths.items() for l in leaves]
        kinds = [hp.links_dict[k].link_type for k in links]
        stats["base_occupancy"] += sum(int(hp.links_dict[k].working_time(data_bytes=e.comm_bytes)) for k in links)
        Bn = int(e.comm_bytes)
        g = gid(t)
        if g in over:
            size = over[g]
        else:
            size = plan["size"]
            if plan["only_d2d"] and "ch2ch" not in kinds:
                continue
        if Bn <= size:
            continue
        n = -(-Bn // size)
        sizes = [size] * (n - 1) + [Bn - size * (n - 1)]
        tags = [t] + [tuple(t[:-1]) + (t[-1] + b * CB,) for b in range(1, n)]
        blocks_of[t] = (tags, sizes, links, kinds)
        stats["messages_chunked"] += 1
        stats["blocks_added"] += n - 1
        if sum(sizes) != Bn:
            stats["bytes_conserved"] = False
        for k, kind in zip(links, kinds):
            w = hp.links_dict[k]
            extra = sum(int(w.working_time(data_bytes=x)) for x in sizes) - int(w.working_time(data_bytes=Bn))
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


def link_chain(ev, a=10, b=11):
    """Zero-lag edges from every sink of stream a to every source of stream b (STAGE2_EVAL 2A)."""
    sinks = [t for t, e in ev.items() if t[1] == a and not e.issue_set]
    srcs = [t for t, e in ev.items() if t[1] == b and not e.dependency_set]
    for s in sinks:
        ev[s].issue_set.update(srcs)
    for t in srcs:
        ev[t].dependency_set.update(sinks)
    return len(sinks), len(srcs)


def blocks(Bn, size):
    return max(1, -(-Bn // size))


def msg_table(raw, hp):
    """{gid: (bytes, uses D2D)} of the unchunked events"""
    out = {}
    for t, e in raw.items():
        if e.event_type == "communication":
            kinds = {hp.links_dict[(r, l)].link_type for r, leaves in e.paths.items() for l in leaves}
            out[gid(t)] = (int(e.comm_bytes), "ch2ch" in kinds)
    return out


def cur_blocks(m, msgs, over):
    Bn, d2d = msgs[m]
    if m in over:
        return blocks(Bn, over[m])
    return blocks(Bn, K256) if d2d else 1


def new_blocks(msgs, over):
    return sum(cur_blocks(m, msgs, over) - cur_blocks(m, msgs, {}) for m in over)


def action(m, msgs, over, pool_b, level=None):
    """Next size for message m. D2D: 64 KB. Non-D2D (pool B only): level if given (fixed-selection variants), else the
    ladder none -> 256 KB -> 64 KB (Greedy). None if nothing is left."""
    Bn, d2d = msgs[m]
    cur = over.get(m, K256 if d2d else None)
    if d2d:
        return K64 if cur != K64 and Bn > K64 else None
    if not pool_b:
        return None
    if level:
        return level if cur is None and Bn > level else None
    if cur is None and Bn > K256:
        return K256
    if cur != K64 and Bn > K64:
        return K64
    return None


# ------------------------------------------------------------------ attribution
def phi_cpcm(g, sc, rank, istar):
    """phi_wait, phi_res of stream index istar on a CPCM schedule, per original message (handoff 9.2)."""
    stream = [g.tag[e][1] - 10 for e in range(g.n)]
    wait, res = collections.Counter(), collections.Counter()
    a_s, a_f, a_ev, a_par, a_res = sc.a_s, sc.a_f, sc.a_ev, sc.a_par, sc.a_res
    for a in range(len(a_s)):
        e = a_ev[a]
        if stream[e] != istar or g.kind[e] != cpcm.COMM:
            continue
        p = a_par[a]
        lo, hi = (a_s[p] if p >= 0 else sc.r[e]), a_s[a]
        if hi <= lo:
            continue
        for u in sc.res_seq[a_res[a]]:
            eu = a_ev[u]
            if stream[eu] == istar or a_f[u] <= lo or a_s[u] >= hi:
                continue
            m = gid(g.tag[eu])
            wait[m] += min(a_f[u], hi) - max(a_s[u], lo)
            if rank[stream[eu]] > rank[istar] and a_s[u] < lo < a_f[u]:
                res[m] += min(a_f[u], hi) - lo
    return wait, res


def phi_backend(cap, rank, istar):
    import e4_attr as A
    sim = A.SimR(cap)
    wait, res = collections.Counter(), collections.Counter()
    tot = 0
    for k, X in sim.act.items():
        if X["tag"][1] - 10 != istar or X["res"][0] != "link" or X["s"] <= X["can"]:
            continue
        lo, hi = X["can"], X["s"]
        for ss, ff, tag in cap["links"][X["res"][1]]:
            if tag[1] - 10 == istar or ff <= lo or ss >= hi:
                continue
            m = gid(tag)
            ov = min(ff, hi) - max(ss, lo)
            wait[m] += ov
            tot += ov
            if rank[tag[1] - 10] > rank[istar] and ss < lo < ff:
                res[m] += min(ff, hi) - lo
    return wait, res


def top(phi, n):
    return {m for m, _ in sorted(phi.items(), key=lambda x: (-x[1], str(x[0])))[:n]}


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def cpcm_run(blobs, hp0, plan, rank, release, n_streams=None):
    ev, cst = chunk_by(E.merged_events(blobs), hp0, plan)
    g = D._graph(ev, hp0)
    st = [g.tag[e][1] - 10 for e in range(g.n)]
    rel = {e: release[st[e]] for e in range(g.n) if g.pred_ptr[e + 1] == g.pred_ptr[e]}
    sc = cpcm.simulate(g, "hop", prio=[rank[s] for s in st], release=rel)
    C = collections.defaultdict(int)
    for e in range(g.n):
        C[st[e]] = max(C[st[e]], sc.F[e])
    return g, sc, dict(C), cst


def J_of(C, T, d, release):
    norm = {i: (C[i] - d[i]) / T[i] for i in T}
    istar = max(norm, key=lambda i: (norm[i], -i))
    return (norm[istar], sum((C[i] - release[i]) / T[i] for i in T)), istar


# ------------------------------------------------------------------ children
def child(spec):
    E._artifact_path()
    from e4b_backend import P1, event_driver_e4b
    hp0 = E.build_platform(E.CFG, spec["faults"])
    blobs = [B.load_stream(s)["events_blob"] for s in spec["streams"]]
    ev, cst = chunk_by(E.merged_events(blobs), hp0, spec["plan"])
    if spec.get("chain"):
        spec["chain_links"] = link_chain(ev)
    hp = copy.deepcopy(hp0)
    pol = B.Fixed(spec["rank"]) if spec.get("rank") else P1()
    e2 = E._e2() if spec.get("measure") is not None else None
    pre = e2._snapshot_pre(ev) if e2 else None
    audit, released = [], {}
    random.seed(spec["seed"])
    t0 = time.time()
    res = event_driver_e4b(ev, hp, release=spec["release"], policy=pol, released=released, audit=audit)
    wall = time.time() - t0
    C, first = collections.defaultdict(int), {}
    for t, e in ev.items():
        i = t[1] - 10
        C[i] = max(C[i], e.end_time)
        first[i] = min(first.get(i, e.start_time), e.start_time)
    out = {"seed": spec["seed"], "wall": wall, "C": dict(C), "first_start": first, "chunk_stats": cst, "result": res,
           "audit": {"n_issues": len(audit), "n_mismatch": sum(1 for a in audit if a[2] != a[3])}, "chain_links": spec.get("chain_links")}
    if e2:
        cap = e2._extract(ev, hp, pre, res, f"seed{spec['seed']}")
        cap["release"] = released
        w, r = phi_backend(cap, spec["rank"], spec["measure"])
        out["phi_wait"], out["phi_res"] = dict(w), dict(r)
    return out


def score_child(spec):
    """2B: CPCM forward on the uniform 256 KB D2D base, i*, phi_wait, phi_res."""
    E._artifact_path()
    hp0 = E.build_platform(E.CFG, spec["faults"])
    blobs = [B.load_stream(s)["events_blob"] for s in spec["streams"]]
    msgs = msg_table(E.merged_events(blobs), hp0)
    t0 = time.perf_counter()
    g, sc, C, cst = cpcm_run(blobs, hp0, BASE, spec["rank"], spec["release"])
    J, istar = J_of(C, spec["T"], spec["d"], spec["release"])
    w, r = phi_cpcm(g, sc, spec["rank"], istar)
    return {"msgs": msgs, "C": C, "J": J, "istar": istar, "phi_wait": dict(w), "phi_res": dict(r), "wall": time.perf_counter() - t0, "n_events": g.n}


def greedy_child(spec):
    """Handoff 9.2 Greedy: each round the top 10 candidates by phi_res, CPCM forward trial, keep the largest J improvement."""
    E._artifact_path()
    hp0 = E.build_platform(E.CFG, spec["faults"])
    blobs = [B.load_stream(s)["events_blob"] for s in spec["streams"]]
    msgs = msg_table(E.merged_events(blobs), hp0)
    rank, rel, T, d = spec["rank"], spec["release"], spec["T"], spec["d"]
    over = {}
    g, sc, C, _ = cpcm_run(blobs, hp0, BASE, rank, rel)
    J, istar = J_of(C, T, d, rel)
    J0, C0 = J, dict(C)
    rounds = []
    t00 = time.time()
    while True:
        _, res = phi_cpcm(g, sc, rank, istar)
        left = spec["budget"] - new_blocks(msgs, over)
        cand = []
        for m, v in sorted(res.items(), key=lambda x: (-x[1], str(x[0]))):
            a = action(m, msgs, over, spec["pool_b"])
            if a is None:
                continue
            cost = blocks(msgs[m][0], a) - cur_blocks(m, msgs, over)
            if cost <= left:
                cand.append((m, a))
            if len(cand) == 10:
                break
        best = None
        for m, a in cand:
            o2 = dict(over)
            o2[m] = a
            g2, sc2, C2, _ = cpcm_run(blobs, hp0, dict(BASE, override=o2), rank, rel)
            J2, i2 = J_of(C2, T, d, rel)
            if J2 < J and (best is None or J2 < best[0]):
                best = (J2, i2, m, a, g2, sc2, C2)
        if best is None:
            rounds.append({"stop": "no improvement" if cand else "no candidate", "candidates": len(cand), "t": time.time() - t00})
            break
        J, istar, m, a, g, sc, C = best
        over[m] = a
        rounds.append({"msg": m, "size": a, "J": J, "blocks": new_blocks(msgs, over), "istar": istar, "candidates": len(cand), "t": time.time() - t00})
        log(f"round {len(rounds)}: {rounds[-1]}")
    return {"override": over, "J0": J0, "J": J, "C0": C0, "C": C, "rounds": rounds, "new_blocks": new_blocks(msgs, over), "wall": time.time() - t00}


def agg_child(spec):
    """Aggregate cost of the handoff: the busiest resource's cumulative occupancy on this platform (and the byte sum)."""
    E._artifact_path()
    hp0 = E.build_platform(E.CFG, spec["faults"])
    out = {}
    for name, k in spec["streams"].items():
        ev, _ = D._events([k], BASE, hp0)
        by = sum(int(e.comm_bytes) for e in ev.values() if e.event_type == "communication")
        g = D._graph(ev, hp0)
        sc = cpcm.simulate(g, "hop")
        occ = collections.Counter()
        for a in range(len(sc.a_s)):
            occ[sc.a_res[a]] += sc.a_f[a] - sc.a_s[a]
        kinds = collections.defaultdict(int)
        for q, v in occ.items():
            kinds[g.res_kind[q]] = max(kinds[g.res_kind[q]], v)
        out[name] = {"max_occ": max(occ.values()), "by_kind": dict(kinds), "bytes": by, "T_hat": sc.T}
    return out


class MaskedAnalysis(cpcm.Analysis):
    """cpcm.Analysis on a schedule with the cross-stream resource edges removed: an activity's resource predecessor is the
    previous activity of its own stream on that resource (handoff 9.3). Same loop as Analysis.__init__ otherwise."""

    def __init__(self, sc, sink_deadline, stream):
        self.sc = sc
        g = sc.g
        na = len(sc.a_s)
        a_s, a_f, a_par, a_ev = sc.a_s, sc.a_f, sc.a_par, sc.a_ev
        rprev = [-1] * na
        for seq in sc.res_seq:
            last = {}
            for a in seq:
                s = stream[a_ev[a]]
                if s in last:
                    rprev[a] = last[s]
                last[s] = a
        self.rprev = rprev
        self.T = sc.T
        self.T_end = max([sc.T] + a_f)
        lst = [sink_deadline[a_ev[a]] - (a_f[a] - a_s[a]) for a in range(na)]
        pred, pred_ptr, term = g.pred, g.pred_ptr, sc.term
        for a in range(na - 1, -1, -1):
            v = lst[a]
            p = a_par[a]
            if p >= 0:
                if v < lst[p]:
                    lst[p] = v
            else:
                e = a_ev[a]
                for j in range(pred_ptr[e], pred_ptr[e + 1]):
                    u = term[pred[j]]
                    w = v - (a_f[u] - a_s[u])
                    if w < lst[u]:
                        lst[u] = w
            u = rprev[a]
            if u >= 0:
                w = v - (a_f[u] - a_s[u])
                if w < lst[u]:
                    lst[u] = w
        self.lst = lst
        self.slack = [lst[a] - a_s[a] for a in range(na)]
        self.lst_ev = [min((lst[a] for a in sc.ev_acts[e] if a_par[a] < 0), default=sink_deadline[e]) for e in range(g.n)]
        self.D = [min((self.lst_ev[g.succ[k]] for k in range(g.succ_ptr[b], g.succ_ptr[b + 1])), default=sink_deadline[b])
                  for b in range(g.n)]
        self._cp = None


def key_flips(g, sc, stream, keyf, idealf):
    """Over adjacent activities of different streams on one link of the schedule: how often the static key order
    (d - tail, block tail own + rest) differs from the ideal key's."""
    n = flips = 0
    for q, seq in enumerate(sc.res_seq):
        if not str(g.res_name[q][0]).startswith("link"):
            continue
        for u, v in zip(seq, seq[1:]):
            eu, ev_ = sc.a_ev[u], sc.a_ev[v]
            if stream[eu] == stream[ev_]:
                continue
            a, b = keyf(eu), keyf(ev_)
            c, x = idealf(eu), idealf(ev_)
            if None in (a, b, c, x) or c == x:
                continue
            n += 1
            if (a < b) != (c < x):
                flips += 1
    return {"n": n, "flips": flips, "rate": flips / n if n else None}


def cm_child(spec):
    E._artifact_path()
    from e4b_backend import event_driver_e4b
    hp0 = E.build_platform(E.CFG, spec["faults"])
    blobs = [B.load_stream(s)["events_blob"] for s in spec["streams"]]
    chunk = spec["chunk"]
    ev0 = E.merged_events(blobs)
    if chunk:
        ev0, _ = E.chunk_events(ev0, hp0, chunk["size"], chunk["only_d2d"])
    g = D._graph(ev0, hp0)
    stream = [g.tag[e][1] - 10 for e in range(g.n)]
    rel = {e: spec["release"][stream[e]] for e in range(g.n) if g.pred_ptr[e + 1] == g.pred_ptr[e]}
    dhat, d, T, ideal, solo = spec["dhat"], spec["d"], spec["T"], spec["ideal"], spec["solo"]
    dl = [dhat[stream[e]] for e in range(g.n)]
    sc = cpcm.simulate(g, "bb", release=rel)
    full = cpcm.Analysis(sc, sink_deadline=dl)
    mask = MaskedAnalysis(sc, dl, stream)
    same = MaskedAnalysis(sc, dl, [0] * g.n)
    by = collections.defaultdict(list)
    for e in range(g.n):
        by[stream[e]].append(e)
    data_full = {i: D.tail_data(g, sc, full, dhat[i], es) for i, es in by.items()}
    data_mask = {i: D.tail_data(g, sc, mask, dhat[i], es) for i, es in by.items()}
    checks = {"unit_same_as_full_lst": same.lst == full.lst, "unit_same_as_full_D": same.D == full.D,
              "D_violations": sum(1 for e in range(g.n) if full.D[e] > mask.D[e]),
              "lst_violations": sum(1 for a in range(len(full.lst)) if full.lst[a] > mask.lst[a]),
              "D_strictly_later": sum(1 for e in range(g.n) if full.D[e] < mask.D[e]), "n_events": g.n}
    est = {"full": D.est_stats(data_full, lambda i, r: data_full[i]["D"][r], dhat, d, T, ideal),
           "mask": D.est_stats(data_mask, lambda i, r: data_mask[i]["D"][r], dhat, d, T, ideal)}
    sh = {i: dhat[i] - spec["T_hat"][i] for i in solo}
    est["C-S"] = D.est_stats(solo, lambda i, r: sh[i] + solo[i]["D"][r], dhat, d, T, ideal)

    def kf(dd, data):
        def f(e):
            i, r = stream[e], B.rt(g.tag[e])
            x = data[i]
            return dd[i] - x["own"][r] - x["rest"][r] if r in x["own"] and r in x["rest"] else None
        return f
    idl = kf(d, ideal)
    flips = {"full": key_flips(g, sc, stream, kf(dhat, data_full), idl), "mask": key_flips(g, sc, stream, kf(dhat, data_mask), idl),
             "C-S": key_flips(g, sc, stream, kf(dhat, solo), idl)}
    runs = []
    for seed in spec["seeds"]:
        ev = E.merged_events(blobs)
        if chunk:
            ev, _ = E.chunk_events(ev, hp0, chunk["size"], chunk["only_d2d"])
        hp = copy.deepcopy(hp0)
        pol = B.DstuE(data_mask, dhat, ev, hp)
        shadow = B.DstuE(ideal, d, ev, hp)
        audit, released, slog = [], {}, {}
        random.seed(seed)
        t0 = time.time()
        event_driver_e4b(ev, hp, release=spec["release"], policy=D.Two(pol, shadow), released=released, audit=audit, shadow=shadow, shadow_log=slog)
        C = collections.defaultdict(int)
        for t, e in ev.items():
            C[t[1] - 10] = max(C[t[1] - 10], e.end_time)
        runs.append({"seed": seed, "wall": time.time() - t0, "C": dict(C), "release": released, "shadow": slog,
                     "audit": {"n_issues": len(audit), "n_mismatch": sum(1 for a in audit if a[2] != a[3])}})
    return {"runs": runs, "checks": checks, "est": est, "flips": flips}


CHILDREN = {"child": child, "score-child": score_child, "greedy-child": greedy_child, "agg-child": agg_child, "cm-child": cm_child}


# ------------------------------------------------------------------ parent: running
def _call(name, spec, cmd):
    path = os.path.join(SIMS, f"{name}.pkl.gz")
    if os.path.exists(path):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    tmp, logs = os.path.join(OUT, "_tmp"), os.path.join(OUT, "logs")
    for x in (SIMS, tmp, logs):
        os.makedirs(x, exist_ok=True)
    sp, sink = os.path.join(tmp, f"spec_{name}.pkl"), os.path.join(tmp, f"sink_{name}.pkl")
    with open(sp, "wb") as f:
        pickle.dump(dict(spec, sink=sink), f, protocol=pickle.HIGHEST_PROTOCOL)
    t0 = time.time()
    with open(os.path.join(logs, f"{name}.log"), "w", encoding="utf-8") as lf:
        try:
            rc = subprocess.call([sys.executable, os.path.abspath(__file__), cmd, sp], cwd=D.HERE, stdout=lf, stderr=subprocess.STDOUT, timeout=LIMIT)
        except subprocess.TimeoutExpired:
            raise D.DstuError(f"{name}: over {LIMIT} s (section 9.5)")
    if rc != 0 or not os.path.exists(sink):
        raise D.DstuError(f"{name} failed (exit {rc}); see dstu/out/stage2/logs/{name}.log")
    with open(sink, "rb") as f:
        r = pickle.load(f)
    r["process_wall"] = time.time() - t0
    os.remove(sink)
    os.remove(sp)
    with gzip.open(path, "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    return r


def run_all(jobs, items, cmd="child"):
    t0 = time.time()
    done = [0]

    def one(it):
        r = _call(it[0], it[1], cmd)
        done[0] += 1
        if done[0] % 25 == 0:
            log(f"  {done[0]}/{len(items)} ({time.time() - t0:.0f} s)")
        return r
    with ThreadPoolExecutor(jobs) as ex:
        res = dict(zip([i[0] for i in items], ex.map(one, items)))
    log(f"{len(items)} processes, {time.time() - t0:.0f} s")
    return res


def seeded(name, spec, seeds=None):
    return [(f"{name}_s{s}", dict(spec, seed=s)) for s in (seeds or B.SEEDS)]


def runs_of(res, name, seeds=None):
    return [res[f"{name}_s{s}"] for s in (seeds or B.SEEDS)]


def fh(o):
    return hashlib.md5(repr(sorted(o.items(), key=str)).encode()).hexdigest()[:8]


def plats_all():
    S1, plats, T = D.load_platforms()
    solo = {}
    for g, hw, cz, keys, fa, alias in plats:
        for k in keys:
            if (k, cz) not in solo:
                with gzip.open(os.path.join(D.SOLO, f"{k}_{cz or 'none'}.pkl.gz"), "rb") as f:
                    solo[(k, cz)] = pickle.load(f)
    return plats, T, solo


def a2_info():
    plats, T, solo = plats_all()
    out = {}
    for g, hw, cz, keys, fa, alias in plats:
        if g == A2:
            out[hw] = {"keys": keys, "faults": fa, "cz": cz, "T": [T[(k, cz)] for k in keys], "That": [solo[(k, cz)]["T_hat"] for k in keys]}
    return out


# ------------------------------------------------------------------ 2A
def cmp3(a, b):
    return 0 if a < b else (1 if a == b else 2)


def rows_2a(info, agg):
    """One row per hardware x X x release x k_req x k_x x d_req basis. Everything external uses T_basis; D_M per method."""
    T0M, T0F = info["nofault"]["T"]
    rows = []
    for hw in HWS:
        TM, TF = info[hw]["T"]
        HM, HF = info[hw]["That"]
        a = agg[hw]
        for xn, rx, kr, kx, basis in itertools.product(("MHA", "FFN"), RX, KR, KX, ("cur", "pre")):
            if hw == "nofault" and basis == "pre":
                continue
            BM, BF = (TM, TF) if basis == "cur" else (T0M, T0F)
            dreq = kr * (BM + BF)
            r_x = rx * TM
            BX = BM if xn == "MHA" else BF
            dx = r_x + kx * BX
            eqs = lambda m, f: m + 0.5 * (dreq - m - f)
            Dm = {"UD": dreq, "HALF": dreq / 2, "R03": 0.3 * dreq, "AGG_ED": dreq - a["FFN"]["max_occ"],
                  "BYTES": dreq * a["MHA"]["bytes"] / (a["MHA"]["bytes"] + a["FFN"]["bytes"]),
                  "PRE_ED": dreq - T0F, "PRE_EQF": T0M * dreq / (T0M + T0F),
                  "CP_ED": dreq - HF, "CP_EQF": HM * dreq / (HM + HF), "CP_EQS": eqs(HM, HF),
                  "SOLO_ED": dreq - TF, "SOLO_EQF": TM * dreq / (TM + TF)}
            ideal = {"CP_ED": Dm["SOLO_ED"], "CP_EQF": Dm["SOLO_EQF"], "CP_EQS": eqs(TM, TF), "PRE_ED": Dm["SOLO_ED"], "PRE_EQF": Dm["SOLO_EQF"]}
            rows.append({"hw": hw, "X": xn, "rx": rx, "k_req": kr, "k_x": kx, "basis": basis, "d_req": dreq, "d_x": dx, "r_x": r_x, "T_M": TM, "T_F": TF,
                         "T_X": TM if xn == "MHA" else TF, "D": Dm, "ideal": ideal,
                         "F": cmp3(dreq, dx), "M": {m: cmp3(v, dx) for m, v in Dm.items()}})
    return rows


def rank_name(m, f):
    return f"M{m}F{f}"


def enum_2a(rows):
    """Keys only: orders against CP_ED, ties, gaps, estimate error, flips."""
    out = {}
    for hw in ("all",) + HWS:
        sub = [r for r in rows if hw in ("all", r["hw"])]
        d = {"n": len(sub)}
        for m in METHODS_2A:
            d[m] = {"differ_from_CP_ED": sum(1 for r in sub if r["M"][m] != r["M"]["CP_ED"]), "ties": sum(1 for r in sub if r["M"][m] == 1),
                    "M_first": sum(1 for r in sub if r["M"][m] == 0)}
        gaps = {}
        for m in ("CP_ED", "CP_EQF", "CP_EQS", "PRE_ED", "PRE_EQF"):
            g, safe, flip, flip_inside = [], 0, 0, 0
            for r in sub:
                gp = abs(r["D"][m] - r["d_x"])
                e = abs(r["D"][m] - r["ideal"][m])
                g.append(gp / (r["T_M"] + r["T_F"]))
                f = cmp3(r["ideal"][m], r["d_x"]) != r["M"][m]
                flip += f
                if gp > e:
                    safe += 1
                else:
                    flip_inside += f
            s = sorted(g)
            d[f"gap_{m}"] = {"median_over_Tchain": statistics.median(g), "p10": s[len(s) // 10], "gap_gt_err": safe, "gap_le_err": len(sub) - safe, "flips_vs_ideal_profile": flip, "flips_with_gap_le_err": flip_inside}
        out[hw] = d
    return out


def stage_2a(jobs):
    info = a2_info()
    t0 = time.time()
    specs = [(f"agg_{hw}", {"faults": info[hw]["faults"], "streams": {"MHA": info[hw]["keys"][0], "FFN": info[hw]["keys"][1]}}) for hw in HWS]
    a = run_all(jobs, specs, cmd="agg-child")
    agg = {hw: a[f"agg_{hw}"] for hw in HWS}
    write_json("agg_cost.json", agg)
    rows = rows_2a(info, agg)
    en = enum_2a(rows)
    write_json("stage2a_enum.json", {"T": {h: info[h]["T"] for h in HWS}, "That": {h: info[h]["That"] for h in HWS}, "agg": agg, "enum": en})
    needed = collections.defaultdict(set)
    for r in rows:
        for m in METHODS_2A:
            needed[(r["hw"], r["X"], r["rx"])].add((r["M"][m], r["F"]))
    items = []
    for (hw, xn, rx), ranks in needed.items():
        i = info[hw]
        x = i["keys"][0] if xn == "MHA" else i["keys"][1]
        base = {"faults": i["faults"], "streams": [i["keys"][0], i["keys"][1], x], "plan": BASE, "chain": True,
                "release": {0: 0, 1: 0, 2: rx * i["T"][0]}}
        for m, f in ranks:
            items += seeded(f"a_{hw}_{xn}_{rx}_{rank_name(m, f)}", dict(base, rank={0: m, 1: f, 2: 1}))
        items += seeded(f"a_{hw}_{xn}_{rx}_P1", dict(base, rank=None))
    for hw in HWS:
        i = info[hw]
        items += seeded(f"a_{hw}_chain", {"faults": i["faults"], "streams": i["keys"], "plan": BASE, "chain": True, "release": {0: 0, 1: 0}, "rank": {0: 0, 1: 1}})
    log(f"2A: {len(rows)} rows, {len(items)} processes")
    res = run_all(jobs, items)
    # checks
    ck = []
    audit = collections.Counter()
    for hw in HWS:
        TM, TF = info[hw]["T"]
        for x in runs_of(res, f"a_{hw}_chain"):
            ck.append({"hw": hw, "seed": x["seed"], "C_M": x["C"][0], "C_F": x["C"][1], "first_F": x["first_start"][1], "T_M": TM, "T_F": TF,
                       "rel_chain": x["C"][1] / (TM + TF) - 1, "links": x["chain_links"]})
    bad_dep = 0
    for n, x in res.items():
        audit["n_issues"] += x["audit"]["n_issues"]
        audit["n_mismatch"] += x["audit"]["n_mismatch"]
        if x["first_start"][1] < x["C"][0]:
            bad_dep += 1
        if not (x["chunk_stats"]["bytes_conserved"] and x["chunk_stats"]["tree_conserved"]):
            audit["chunk_bad"] += 1
    checks = {"chain_alone": ck, "audit": dict(audit), "ffn_before_mha_end": bad_dep, "max_rel_chain": max(abs(c["rel_chain"]) for c in ck)}
    log("checks " + json.dumps(D.round_all(checks)))
    if bad_dep or audit["n_mismatch"] or audit["chunk_bad"] or checks["max_rel_chain"] > 0.04:
        write_json("stage2a_checks.json", checks)
        raise D.DstuError("2A self-check")

    def lv(r, nm):
        L, Lc, Lx, Cs = [], [], [], []
        for x in runs_of(res, nm):
            a_, b_ = (x["C"][1] - r["d_req"]) / (r["T_M"] + r["T_F"]), (x["C"][2] - r["d_x"]) / r["T_X"]
            L.append(max(a_, b_))
            Lc.append(a_)
            Lx.append(b_)
            Cs.append((x["C"][0], x["C"][1], x["C"][2]))
        return {"L": statistics.mean(L), "range": max(L) - min(L), "Lchain": statistics.mean(Lc), "LX": statistics.mean(Lx),
                "C_M": statistics.mean(c[0] for c in Cs), "C_F": statistics.mean(c[1] for c in Cs), "C_X": statistics.mean(c[2] for c in Cs)}
    table = []
    for r in rows:
        o = {k: r[k] for k in ("hw", "X", "rx", "k_req", "k_x", "basis", "d_req", "d_x")}
        pre = f"a_{r['hw']}_{r['X']}_{r['rx']}_"
        for m in METHODS_2A:
            v = lv(r, pre + rank_name(r["M"][m], r["F"]))
            o[f"D_{m}"], o[f"o_{m}"], o[f"L_{m}"], o[f"R_{m}"] = r["D"][m], rank_name(r["M"][m], r["F"]), v["L"], v["range"]
            o[f"Lc_{m}"], o[f"Lx_{m}"] = v["Lchain"], v["LX"]
        o["L_P1"] = lv(r, pre + "P1")["L"]
        table.append(o)
    write_csv("stage2a_rows.csv", table)
    S = summary_2a(table)
    write_json("stage2a.json", {"summary": S, "checks": checks, "enum": en, "wall": time.time() - t0, "command": sys.argv, "artifact_commit": E._artifact_commit(), "seeds": B.SEEDS})
    log(f"2A done {time.time() - t0:.0f} s")
    log(json.dumps(D.round_all(S["all"]), ensure_ascii=False))


def summary_2a(table):
    def block(sub):
        out = {"n": len(sub)}
        for m in METHODS_2A:
            L = [r[f"L_{m}"] for r in sub]
            diff = [r for r in sub if r[f"o_{m}"] != r["o_CP_ED"]]
            d = [r[f"L_{m}"] - r["L_CP_ED"] for r in diff]
            out[m] = {"L_median": statistics.median(L), "L_worst": max(L), "n_diff": len(diff),
                      "n_better_than_CP_ED": sum(x < -TH for x in d), "n_worse_than_CP_ED": sum(x > TH for x in d), "n_tie": sum(abs(x) <= TH for x in d)}
            if diff:
                out[m].update(diff_dL_median=statistics.median(d), diff_dL_min=min(d), diff_dL_max=max(d), diff_L_m_median=statistics.median(r[f"L_{m}"] for r in diff),
                              diff_L_CP_median=statistics.median(r["L_CP_ED"] for r in diff), diff_worst_m=max(r[f"L_{m}"] for r in diff), diff_worst_CP=max(r["L_CP_ED"] for r in diff))
        out["P1_L_median"] = statistics.median(r["L_P1"] for r in sub)
        return out
    S = {"all": block(table)}
    for hw in HWS:
        S[hw] = block([r for r in table if r["hw"] == hw])
        for xn in ("MHA", "FFN"):
            S[f"{hw}_{xn}"] = block([r for r in table if r["hw"] == hw and r["X"] == xn])
    for hw in ("F4", "F5"):
        for bs in ("cur", "pre"):
            S[f"{hw}_{bs}"] = block([r for r in table if r["hw"] == hw and r["basis"] == bs])
    pair = {}
    for hw in ("F4", "F5"):
        for a_, b_ in (("PRE_ED", "CP_ED"), ("PRE_EQF", "CP_EQF")):
            sub = [r for r in table if r["hw"] == hw and r[f"o_{a_}"] != r[f"o_{b_}"]]
            d = [r[f"L_{a_}"] - r[f"L_{b_}"] for r in sub]
            pair[f"{hw}_{a_}_vs_{b_}"] = {"n_rows": sum(1 for r in table if r["hw"] == hw), "n_diff": len(sub), "median": statistics.median(d) if d else None,
                                          "pre_better": sum(x < -TH for x in d), "cp_better": sum(x > TH for x in d), "worst_pre": max((r[f"L_{a_}"] for r in sub), default=None), "worst_cp": max((r[f"L_{b_}"] for r in sub), default=None)}
    S["pre_vs_cp_same_alpha"] = pair
    return S


# ------------------------------------------------------------------ 2B
def a2_cfgs():
    plats, T, solo = plats_all()
    sub = [p for p in plats if p[0] == A2]
    ideal = D.load_ideal(sub)
    cfgs = D.build_cfgs(1, sub, T, solo, ideal, only=lambda g, hw, o, tg: tg == "k1.5")
    for c in cfgs:
        c["rank"] = D.rank_of(D.order_of(c["dhat"], c["n"]))
    return cfgs


def run_spec(c, plan, **kw):
    return dict({"faults": c["faults"], "streams": c["streams"], "release": c["release"], "rank": c["rank"], "plan": plan}, **kw)


def stat_b(runs, c):
    Tc, d = c["T"], c["d"]
    L = [max((r["C"][i] - d[i]) / Tc[i] for i in Tc) for r in runs]
    cst = runs[0]["chunk_stats"]
    return {"L": statistics.mean(L), "L_range": max(L) - min(L), "C0": statistics.mean(r["C"][0] for r in runs), "C1": statistics.mean(r["C"][1] for r in runs),
            "M": statistics.mean(max(r["C"].values()) for r in runs), "blocks_total": cst["blocks_added"], "extra_occupancy": cst["extra_occupancy"], "n_events": cst["n_events"],
            "wall": statistics.mean(r["wall"] for r in runs), "seeds_L": L}


UNI = {"256d2d": BASE, "64d2d": {"size": K64, "only_d2d": True}, "256all": {"size": K256, "only_d2d": False}, "64all": {"size": K64, "only_d2d": False}}


def pilot(jobs):
    t0 = time.time()
    cfgs = [c for c in a2_cfgs() if c["order"] == "d0" and c["hw"] in ("nofault", "F4")]
    items = []
    for c in cfgs:
        for n, p in UNI.items():
            items += seeded(f"b_{c['name']}_{n}", run_spec(c, p))
    res = run_all(jobs, items)
    rows = []
    keep = False
    for c in cfgs:
        s = {n: stat_b(runs_of(res, f"b_{c['name']}_{n}"), c) for n in UNI}
        d256 = s["256d2d"]["L"] - s["256all"]["L"]
        d64 = s["64d2d"]["L"] - s["64all"]["L"]
        keep = keep or d256 >= TH or d64 >= TH
        rows.append({"config": c["name"], **{n: D.round_all(v) for n, v in s.items()}, "256d2d_minus_256all": d256, "64d2d_minus_64all": d64})
    audit = sum(x["audit"]["n_mismatch"] for x in res.values())
    out = {"rows": rows, "pool_B_kept": keep, "wall": time.time() - t0, "audit_mismatch": audit, "command": sys.argv}
    write_json("pilot.json", out)
    log(json.dumps(D.round_all(out)))


def select(phi, msgs, budget, pool_b, level):
    over = {}
    used = 0
    for m, v in sorted(phi.items(), key=lambda x: (-x[1], str(x[0]))):
        if m not in msgs or v <= 0:
            continue
        a = action(m, msgs, over, pool_b, level)
        if a is None:
            continue
        cost = blocks(msgs[m][0], a) - cur_blocks(m, msgs, over)
        if used + cost > budget:
            continue
        over[m] = a
        used += cost
    return over, used


def select_random(msgs, budget, pool_b, level, seed):
    rng = random.Random(seed)
    ms = list(sorted(msgs, key=str))
    rng.shuffle(ms)
    ms.sort(key=lambda m: -msgs[m][0])  # stable: same size in random order
    over = {}
    used = 0
    for m in ms:
        a = action(m, msgs, over, pool_b, level)
        if a is None:
            continue
        cost = blocks(msgs[m][0], a) - cur_blocks(m, msgs, over)
        if used + cost > budget:
            continue
        over[m] = a
        used += cost
    return over, used


def stage_2b(jobs, pool_b):
    t0 = time.time()
    cfgs = a2_cfgs()
    items = [(f"score_{c['name']}", {"faults": c["faults"], "streams": c["streams"], "release": c["release"], "rank": c["rank"], "T": c["T"], "d": c["d"]}) for c in cfgs]
    sc = run_all(jobs, items, cmd="score-child")
    plan_items = []
    meth = {}
    jac_rows = []
    levels = [("", None)] if not pool_b else [("B256", K256), ("B64", K64)]
    for c in cfgs:
        s = sc[f"score_{c['name']}"]
        J = jac(top(s["phi_wait"], 30), top(s["phi_res"], 30))
        msgs = s["msgs"]
        i_ = s["istar"]
        jac_rows.append({"config": c["name"], "istar": i_, "J_top30": J, "n_wait": len(s["phi_wait"]), "n_res": len(s["phi_res"]),
                         "sum_wait": sum(s["phi_wait"].values()), "sum_res": sum(s["phi_res"].values()),
                         "nonD2D_share_wait": sum(v for m, v in s["phi_wait"].items() if not msgs[m][1]) / max(1, sum(s["phi_wait"].values())),
                         "nonD2D_share_res": sum(v for m, v in s["phi_res"].items() if not msgs[m][1]) / max(1, sum(s["phi_res"].values()))})
        # uniform references, measured for attribution checks
        plan_items += seeded(f"b_{c['name']}_256d2d", run_spec(c, UNI["256d2d"], measure=None))
        plan_items += seeded(f"b_{c['name']}_64d2d", run_spec(c, UNI["64d2d"]))
        plan_items += seeded(f"b_{c['name']}_256all", run_spec(c, UNI["256all"]))
        plan_items += seeded(f"m_{c['name']}_256d2d", run_spec(c, UNI["256d2d"], measure=i_))
        for bud in BUDGETS:
            for lvn, lv in levels:
                tags = [("res", s["phi_res"])] + ([] if J >= 0.9 else [("wait", s["phi_wait"])])
                seen = {}
                for tg, phi in tags:
                    o, u = select(phi, msgs, bud, pool_b, lv)
                    h = fh(o)
                    if h in seen:
                        meth[(c["name"], bud, lvn, tg)] = {**meth[(c["name"], bud, lvn, seen[h])], "same_as": seen[h]}
                        continue
                    seen[h] = tg
                    nm = f"s_{c['name']}_{h}"
                    meth[(c["name"], bud, lvn, tg)] = {"name": nm, "blocks": u, "n_msgs": len(o)}
                    plan_items += seeded(nm, run_spec(c, dict(BASE, override=o), measure=(i_ if tg == "res" and bud == 540 else None)))
                for j in range(3):
                    o, u = select_random(msgs, bud, pool_b, lv, SEL_SEED + j)
                    nm = f"s_{c['name']}_{fh(o)}"
                    meth[(c["name"], bud, lvn, f"rand{j}")] = {"name": nm, "blocks": u, "n_msgs": len(o)}
                    plan_items += seeded(nm, run_spec(c, dict(BASE, override=o)))
    uniq = {}
    for n, s_ in plan_items:
        uniq[n] = s_
    plan_items = list(uniq.items())
    # a1 pilots
    plats, T, solo = plats_all()
    sub = [p for p in plats if p[0] == A1 and p[1] in ("nofault", "F4")]
    a1 = D.build_cfgs(1, sub, T, solo, D.load_ideal(sub), only=lambda g, hw, o, tg: o == "d0" and tg == "k1.5")
    for c in a1:
        c["rank"] = D.rank_of(D.order_of(c["dhat"], c["n"]))
    sa1 = run_all(jobs, [(f"score_{c['name']}", {"faults": c["faults"], "streams": c["streams"], "release": c["release"], "rank": c["rank"], "T": c["T"], "d": c["d"]}) for c in a1], cmd="score-child")
    a1m = {}
    for c in a1:
        s = sa1[f"score_{c['name']}"]
        msgs = s["msgs"]
        u64 = sum(blocks(b, K64) - blocks(b, K256) for b, d2 in msgs.values() if d2)
        bud = max(1, round(0.1 * u64))
        plan_items += seeded(f"b_{c['name']}_256d2d", run_spec(c, UNI["256d2d"]))
        plan_items += seeded(f"b_{c['name']}_64d2d", run_spec(c, UNI["64d2d"]))
        o, u = select(s["phi_res"], msgs, bud, False, None)
        a1m[(c["name"], "res")] = {"name": f"s_{c['name']}_{fh(o)}", "blocks": u, "n_msgs": len(o), "budget": bud, "u64": u64}
        plan_items += seeded(a1m[(c["name"], "res")]["name"], run_spec(c, dict(BASE, override=o)))
        o, u = select_random(msgs, bud, False, None, SEL_SEED)
        a1m[(c["name"], "rand0")] = {"name": f"s_{c['name']}_{fh(o)}", "blocks": u, "n_msgs": len(o), "budget": bud, "u64": u64}
        plan_items += seeded(a1m[(c["name"], "rand0")]["name"], run_spec(c, dict(BASE, override=o)))
    log(f"2B: {len(plan_items)} processes; jaccard {json.dumps(D.round_all(jac_rows))}")
    res = run_all(jobs, plan_items)
    table = []
    for c in cfgs:
        for n in ("256d2d", "64d2d", "256all"):
            table.append({"config": c["name"], "hw": c["hw"], "order": c["order"], "method": n, "budget": "", "level": "", "n_msgs": "", "new_blocks": "", **stat_b(runs_of(res, f"b_{c['name']}_{n}"), c)})
    for (nm, bud, lvn, tg), x in meth.items():
        c = next(c for c in cfgs if c["name"] == nm)
        table.append({"config": nm, "hw": c["hw"], "order": c["order"], "method": tg, "budget": bud, "level": lvn, "n_msgs": x["n_msgs"], "new_blocks": x["blocks"], "same_as": x.get("same_as", ""), **stat_b(runs_of(res, x["name"]), c)})
    for c in cfgs:
        base = table[[t["config"] == c["name"] and t["method"] == "256d2d" for t in table].index(True)]
        for t in table:
            if t["config"] == c["name"]:
                t["dL_vs_base"] = t["L"] - base["L"]
                t["extra_vs_base"] = t["extra_occupancy"] - base["extra_occupancy"]
                if t["new_blocks"] != "" and t["blocks_total"] - base["blocks_total"] != t["new_blocks"]:
                    raise D.DstuError(f"block accounting: {t['config']} {t['method']} {t['budget']}")
    write_csv("stage2b_rows.csv", [{k: v for k, v in t.items() if k != "seeds_L"} for t in table])
    a1rows = []
    for c in a1:
        base = stat_b(runs_of(res, f"b_{c['name']}_256d2d"), c)
        u = stat_b(runs_of(res, f"b_{c['name']}_64d2d"), c)
        for tg, s_ in (("256d2d", base), ("64d2d", u), ("res", stat_b(runs_of(res, a1m[(c["name"], "res")]["name"]), c)), ("rand0", stat_b(runs_of(res, a1m[(c["name"], "rand0")]["name"]), c))):
            a1rows.append({"config": c["name"], "method": tg, "budget": a1m[(c["name"], "res")]["budget"], "u64": a1m[(c["name"], "res")]["u64"], "new_blocks": a1m.get((c["name"], tg), {}).get("blocks", ""), **{k: v for k, v in s_.items() if k != "seeds_L"}})
    write_csv("stage2b_a1.csv", a1rows)
    # self-checks: Jaccard (CPCM vs backend vs backend), counterfactual
    jr, cf = [], []
    for c in cfgs:
        s = sc[f"score_{c['name']}"]
        bk = runs_of(res, f"m_{c['name']}_256d2d")
        for phi in ("phi_wait", "phi_res"):
            for n in (10, 30):
                jr.append({"config": c["name"], "phi": phi, "top": n, "cpcm_vs_backend": statistics.mean(jac(top(s[phi], n), top(b[phi], n)) for b in bk),
                           "backend_pairs": statistics.mean(jac(top(a_[phi], n), top(b_[phi], n)) for a_, b_ in itertools.combinations(bk, 2))})
        x = meth[(c["name"], 540, levels[0][0], "res")]
        af = runs_of(res, x["name"])
        if "phi_wait" in af[0]:
            i_ = s["istar"]
            cf.append({"config": c["name"], "istar": i_, "overlap_base": statistics.mean(sum(b["phi_wait"].values()) for b in bk), "overlap_after": statistics.mean(sum(b["phi_wait"].values()) for b in af),
                       "res_base": statistics.mean(sum(b["phi_res"].values()) for b in bk), "res_after": statistics.mean(sum(b["phi_res"].values()) for b in af),
                       "C_base": statistics.mean(b["C"][i_] for b in bk), "C_after": statistics.mean(b["C"][i_] for b in af)})
    write_csv("stage2b_jaccard.csv", jr)
    audit = sum(x["audit"]["n_mismatch"] for x in res.values())
    chunk_bad = sum(1 for x in res.values() if not (x["chunk_stats"]["bytes_conserved"] and x["chunk_stats"]["tree_conserved"]))
    pairs = pair_2b(table, cfgs, levels)
    # rep: uniform 256 KB against E4b's P4
    tab = D.ref_table(1)
    rep = []
    for c in cfgs:
        rf = tab[(c["name"], "k1.5", "P4")]
        L = [t for t in table if t["config"] == c["name"] and t["method"] == "256d2d"][0]
        rep.append({"config": c["name"], "L": L["L"], "L_e4b_P4": rf["L"], "M": L["M"], "M_e4b": rf["M"]})
    out = {"pool_b": pool_b, "jaccard_top30": jac_rows, "jaccard_checks": jr, "counterfactual": cf, "audit_mismatch": audit, "chunk_bad": chunk_bad, "pairs": pairs, "rep": rep, "wall": time.time() - t0, "command": sys.argv, "artifact_commit": E._artifact_commit()}
    write_json("stage2b.json", out)
    log(json.dumps(D.round_all({k: out[k] for k in ("audit_mismatch", "chunk_bad", "pairs", "counterfactual", "rep", "wall")}), ensure_ascii=False))
    if audit or chunk_bad:
        raise D.DstuError("2B self-check")


def pair_2b(table, cfgs, levels):
    P = {}
    for lvn, _ in levels:
        for bud in BUDGETS:
            rows = []
            for c in cfgs:
                g = lambda m: next((t for t in table if t["config"] == c["name"] and t["method"] == m and t["budget"] in (bud, "") and t.get("level", "") in (lvn, "")), None)
                res = g("res")
                rnd = [g(f"rand{j}") for j in range(3)]
                x = {"config": c["name"], "L_res": res["L"], "L_rand": statistics.mean(r["L"] for r in rnd), "L_rand_best": min(r["L"] for r in rnd), "L_rand_range": max(r["L"] for r in rnd) - min(r["L"] for r in rnd),
                     "L_base": g("256d2d")["L"], "L_64d2d": g("64d2d")["L"], "L_256all": g("256all")["L"], "blocks_res": res["new_blocks"], "blocks_rand": rnd[0]["new_blocks"],
                     "C0_res": res["C0"], "C0_base": g("256d2d")["C0"], "C1_res": res["C1"], "C1_base": g("256d2d")["C1"], "seed_range_res": res["L_range"]}
                w = g("wait")
                if w:
                    x["L_wait"] = w["L"]
                rows.append(x)
            d = lambda a_, b_: [r[a_] - r[b_] for r in rows]
            sm = lambda v: {"median": statistics.median(v), "min": min(v), "max": max(v), "better": sum(z < -TH for z in v), "worse": sum(z > TH for z in v)}
            P[f"{lvn or 'A'}_{bud}"] = {"rows": rows, "res_minus_rand": sm(d("L_res", "L_rand")), "res_minus_base": sm(d("L_res", "L_base")), "res_minus_256all": sm(d("L_res", "L_256all")), "res_minus_64d2d": sm(d("L_res", "L_64d2d")), "rand_minus_base": sm(d("L_rand", "L_base"))}
    return P


def stage_greedy(jobs, pool_b):
    t0 = time.time()
    cfgs = [c for c in a2_cfgs() if c["order"] == "d0"]
    items = [(f"greedy_{c['name']}_{'B' if pool_b else 'A'}", {"faults": c["faults"], "streams": c["streams"], "release": c["release"], "rank": c["rank"], "T": c["T"], "d": c["d"], "budget": 540, "pool_b": pool_b}) for c in cfgs]
    g = run_all(jobs, items, cmd="greedy-child")
    its = []
    for c, (n, _) in zip(cfgs, items):
        its += seeded(f"s_{c['name']}_{fh(g[n]['override'])}", run_spec(c, dict(BASE, override=g[n]["override"])))
        its += seeded(f"b_{c['name']}_256d2d", run_spec(c, UNI["256d2d"]))
    res = run_all(jobs, list({n: s for n, s in its}.items()))
    rows = []
    for c, (n, _) in zip(cfgs, items):
        x = g[n]
        s0 = stat_b(runs_of(res, f"b_{c['name']}_256d2d"), c)
        s1 = stat_b(runs_of(res, f"s_{c['name']}_{fh(x['override'])}"), c)
        rows.append({"config": c["name"], "rounds": len(x["rounds"]), "stop": x["rounds"][-1].get("stop"), "new_blocks": x["new_blocks"], "pred_L0": x["J0"][0], "pred_L": x["J"][0], "pred_gain": x["J0"][0] - x["J"][0],
                     "pred_J2_0": x["J0"][1], "pred_J2": x["J"][1], "pred_C0": x["C0"], "pred_C": x["C"], "be_L0": s0["L"], "be_L": s1["L"], "be_gain": s0["L"] - s1["L"], "be_seeds0": s0["seeds_L"], "be_seeds": s1["seeds_L"],
                     "be_C0_base": s0["C0"], "be_C0": s1["C0"], "be_C1_base": s0["C1"], "be_C1": s1["C1"], "extra_vs_base": s1["extra_occupancy"] - s0["extra_occupancy"], "wall": x["wall"]})
    write_json("greedy.json", {"rows": rows, "pool_b": pool_b, "wall": time.time() - t0, "command": sys.argv})
    log(json.dumps(D.round_all(rows)))


def stage_cm(jobs):
    t0 = time.time()
    plats, T, solo = plats_all()
    ideal = D.load_ideal(plats)
    cfgs = D.build_cfgs(1, plats, T, solo, ideal)
    rep = {(r["config"], r["dl"]): r for r in csv.DictReader(open(os.path.join(D.OUT, "setting1_report.csv"), encoding="utf-8"))}
    items, chosen = [], []
    for nm, tg in CM_ROWS:
        c = next(x for x in cfgs if x["name"] == nm and x["tg"] == tg)
        chosen.append(c)
        items.append((f"cm_{nm}_{tg}", {"faults": c["faults"], "streams": c["streams"], "release": c["release"], "chunk": D.H.CHUNKS[c["chunk"]] if c["chunk"] else None,
                                         "seeds": B.SEEDS, "dhat": c["dhat"], "d": c["d"], "T": c["T"], "T_hat": c["That"],
                                         "ideal": {i: ideal[(s, c["chunk"])] for i, s in enumerate(c["streams"])},
                                         "solo": {i: D.trim(solo[(s, c["chunk"])]) for i, s in enumerate(c["streams"])}}))
    res = run_all(jobs, items, cmd="cm-child")
    rows = []
    bad = 0
    for c, (n, _) in zip(chosen, items):
        x = res[n]
        s = D.stat(x["runs"], c)
        r = rep[(c["name"], c["tg"])]
        sh = {k: sum(q["shadow"].get(k, 0) for q in x["runs"]) for k in ("n", "differ", "differ_stream")}
        ck = x["checks"]
        bad += ck["D_violations"] + ck["lst_violations"] + (not ck["unit_same_as_full_lst"]) + (not ck["unit_same_as_full_D"])
        row = {"config": c["name"], "dl": c["tg"], "L_P1": float(r["L_P1"]), "L_P7": float(r["L_P7"]), "L_C-M": float(r["L_C-M"]), "L_mask": s["L"], "L_mask_range": s["L_range"], "L_C-S": float(r["L_C-S"]),
               "differ_stream_C-M": float(r["differ_stream_C-M"]), "differ_stream_mask": sh["differ_stream"] / sh["n"] if sh["n"] else None, "differ_stream_C-S": float(r["differ_stream_C-S"]), **ck}
        for m in ("full", "mask", "C-S"):
            row[f"flip_{m}"] = x["flips"][m]["rate"]
            row[f"flip_n_{m}"] = x["flips"][m]["n"]
            for i, e in x["est"][m].items():
                row[f"Ddev_{m}_{i}"] = e["D_vs_ideal"]["median"]
                row[f"Ddev_p05_{m}_{i}"] = e["D_vs_ideal"]["p05"]
        rows.append(row)
    write_csv("cm_ablation.csv", rows)
    write_json("cm_ablation.json", {"rows": rows, "wall": time.time() - t0, "command": sys.argv, "seeds": B.SEEDS, "audit_mismatch": sum(q["audit"]["n_mismatch"] for v in res.values() for q in v["runs"])})
    log(json.dumps(D.round_all(rows)))
    if bad:
        raise D.DstuError(f"C-M self-check (property): {bad}")


# ------------------------------------------------------------------ checks
def check():
    E._artifact_path()
    info = a2_info()
    res = []
    for hw in HWS:
        i = info[hw]
        hp0 = E.build_platform(E.CFG, i["faults"])
        blobs = [B.load_stream(s)["events_blob"] for s in i["keys"]]
        for sz, od in ((K256, True), (K64, True), (K256, False), (K64, False)):
            a, sa = E.chunk_events(E.merged_events(blobs), hp0, sz, od)
            b, sb = chunk_by(E.merged_events(blobs), hp0, {"size": sz, "only_d2d": od})
            same = list(a) == list(b) and sa == sb and all(a[t].comm_bytes == b[t].comm_bytes and a[t].dependency_set == b[t].dependency_set and a[t].issue_set == b[t].issue_set
                                                          for t in a if a[t].event_type == "communication") and all(a[t].dependency_set == b[t].dependency_set and a[t].issue_set == b[t].issue_set for t in a)
            res.append({"hw": hw, "size": sz, "only_d2d": od, "isomorphic": same, "blocks": sb["blocks_added"], "extra": sb["extra_occupancy"], "n": sb["n_events"], "bytes": sb["bytes_conserved"], "tree": sb["tree_conserved"]})
        msgs = msg_table(E.merged_events(blobs), hp0)
        nd = [b for b, d in msgs.values() if not d]
        dd = [b for b, d in msgs.values() if d]
        res.append({"hw": hw, "msgs": len(msgs), "d2d": len(dd), "nond2d": len(nd), "cost_d2d_64": sum(blocks(b, K64) - blocks(b, K256) for b in dd), "cost_nond2d_256": sum(blocks(b, K256) - 1 for b in nd), "cost_nond2d_64": sum(blocks(b, K64) - 1 for b in nd),
                    "max_b": max(b for b, _ in msgs.values())})
    for r in res:
        log(json.dumps(r))
    write_json("check.json", res)
    if any(r.get("isomorphic") is False or r.get("bytes") is False or r.get("tree") is False for r in res):
        raise D.DstuError("chunk_by is not isomorphic to E.chunk_events")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] in CHILDREN:
        with open(sys.argv[2], "rb") as f:
            spec = pickle.load(f)
        r = CHILDREN[sys.argv[1]](spec)
        with open(spec["sink"], "wb") as f:
            pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "enum", "2a", "pilot", "2b", "greedy", "cm"])
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--pool-b", action="store_true")
    a = ap.parse_args()
    if a.cmd == "check":
        check()
    elif a.cmd == "enum":
        info = a2_info()
        specs = [(f"agg_{hw}", {"faults": info[hw]["faults"], "streams": {"MHA": info[hw]["keys"][0], "FFN": info[hw]["keys"][1]}}) for hw in HWS]
        x = run_all(a.jobs, specs, cmd="agg-child")
        agg = {hw: x[f"agg_{hw}"] for hw in HWS}
        rows = rows_2a(info, agg)
        ranks = collections.Counter()
        for r in rows:
            for m in METHODS_2A:
                ranks[(r["hw"], r["X"], r["rx"], r["M"][m], r["F"])] += 1
        en = enum_2a(rows)
        log(json.dumps(D.round_all({"agg": agg, "T": {h: info[h]["T"] for h in HWS}, "That": {h: info[h]["That"] for h in HWS}, "rows": len(rows), "ranks_to_run": len(ranks), "ties": sorted(str(k) for k in ranks if k[3] == 1 or k[4] == 1), "enum": en})))
    elif a.cmd == "2a":
        stage_2a(a.jobs)
    elif a.cmd == "pilot":
        pilot(a.jobs)
    elif a.cmd == "2b":
        stage_2b(a.jobs, a.pool_b)
    elif a.cmd == "greedy":
        stage_greedy(a.jobs, a.pool_b)
    else:
        stage_cm(a.jobs)


if __name__ == "__main__":
    try:
        main()
    except (D.DstuError, B.E4bError, E.E4Error, cpcm.CPCMError) as exc:
        print(f"[dstu2] STOP: {exc}", flush=True)
        sys.exit(2)
