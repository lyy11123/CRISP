"""DSTU stage 2C (dstu/DSTU_HANDOFF.md section 10): retest on all-link 256 KB. Nothing in cpcm/, e4_overlap/, third_party/ changes.

  python dstu2c.py solo --jobs 14    backend and CPCM solo data on all-link 256 KB for every stream; self-checks
  python dstu2c.py a --jobs 14       10.1 2A on all-link 256 KB + 10.2 CP_SIM on both chunkings
  python dstu2c.py b --jobs 14       10.3 2B with the static selection, groups A and B
  python dstu2c.py s1 --jobs 14      10.4 setting 1 on all-link 256 KB
Results in dstu/out/stage2c/; simulations cached in dstu/out/stage2/sims (names start with c_, bs_, ds_, sim_, sc2, sB_, sA_, s1all_).
"""
import argparse
import collections
import copy
import gzip
import itertools
import json
import os
import pickle
import random
import statistics
import sys
import time

import dstu as D
import dstu2 as S
import dstu_pair as DP
from dstu import B, E, cpcm

OUT = os.path.join(D.OUT, "stage2c")
ALL = S.UNI["256all"]
CZ = "256all"
TH = S.TH
RANKS4 = [(0, 0), (0, 2), (2, 0), (2, 2)]
STATIC_SEED = S.SEL_SEED + 100
S.SCRIPT = os.path.abspath(__file__)


def log(m):
    print(f"[dstu2c] {m}", flush=True)


def wj(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


def wc(name, rows):
    import csv
    os.makedirs(OUT, exist_ok=True)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def rcsv(path):
    import csv
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fl(x):
    return float(x)


# ------------------------------------------------------------------ children
def bsolo_child(spec):
    """Backend alone, P1, seeds 1-3 (E4b's solo reference): T = mean C; ideal tail data from seed 1."""
    E._artifact_path()
    from e4b_backend import P1, event_driver_e4b
    e2 = E._e2()
    hp0 = E.build_platform(E.CFG, spec["faults"])
    blob = [B.load_stream(spec["stream"])["events_blob"]]
    Cs, sd, audit = [], None, 0
    for seed in B.SEEDS:
        ev = E.merged_events(blob)
        ev, cst = E.chunk_events(ev, hp0, spec["chunk"]["size"], spec["chunk"]["only_d2d"])
        hp = copy.deepcopy(hp0)
        pre = e2._snapshot_pre(ev)
        released = {}
        random.seed(seed)
        res = event_driver_e4b(ev, hp, release={0: 0}, policy=P1(), released=released)
        Cs.append(max(e.end_time for e in ev.values()))
        if seed == 1:
            cap = e2._extract(ev, hp, pre, res, "seed1")
            cap["release"] = released
            x = B.solo_data(cap)
            sd = {k: x[k] for k in ("own", "rest", "group_n", "group_rest")}
    return {"T": statistics.mean(float(c) for c in Cs), "Cs": Cs, "ideal": sd, "chunk_stats": cst}


def dsolo_child(spec):
    D.solo_child(spec["_path"])
    return None


def dchild(spec):
    D.child(spec["_path"])
    return None


def sim_child(spec):
    """CPCM forward on the chain + X graph for each of the 4 rankings."""
    E._artifact_path()
    hp0 = E.build_platform(E.CFG, spec["faults"])
    blobs = [B.load_stream(s)["events_blob"] for s in spec["streams"]]
    t0 = time.perf_counter()
    ev, _ = S.chunk_by(E.merged_events(blobs), hp0, spec["plan"])
    S.link_chain(ev)
    g = D._graph(ev, hp0)
    st = [g.tag[e][1] - 10 for e in range(g.n)]
    rel = {e: spec["release"][st[e]] for e in range(g.n) if g.pred_ptr[e + 1] == g.pred_ptr[e]}
    t_compile = time.perf_counter() - t0
    out, tm = {}, []
    for m, f in RANKS4:
        rk = {0: m, 1: f, 2: 1}
        t1 = time.perf_counter()
        sc = cpcm.simulate(g, "hop", prio=[rk[s] for s in st], release=rel)
        tm.append(time.perf_counter() - t1)
        C = collections.defaultdict(int)
        for e in range(g.n):
            C[st[e]] = max(C[st[e]], sc.F[e])
        out[S.rank_name(m, f)] = dict(C)
    return {"C": out, "compile_s": t_compile, "forward_s": tm, "n_events": g.n}


def score2_child(spec):
    """CPCM forward on the given plan, i*, phi_wait, phi_res, links shared with i* per message (routes only)."""
    E._artifact_path()
    hp0 = E.build_platform(E.CFG, spec["faults"])
    blobs = [B.load_stream(s)["events_blob"] for s in spec["streams"]]
    raw = E.merged_events(blobs)
    msgs = S.msg_table(raw, hp0)
    links = {}
    for t, e in raw.items():
        if e.event_type == "communication":
            links[S.gid(t)] = {(r, l) for r, leaves in e.paths.items() for l in leaves}
    g, sc, C, cst = S.cpcm_run(blobs, hp0, spec["plan"], spec["rank"], spec["release"])
    J, istar = S.J_of(C, spec["T"], spec["d"], spec["release"])
    w, r = S.phi_cpcm(g, sc, spec["rank"], istar)
    mine = set().union(*[v for m, v in links.items() if m[0] - 10 == istar])
    shared = {m: len(v & mine) for m, v in links.items() if m[0] - 10 != istar}
    return {"msgs": msgs, "C": C, "J": J, "istar": istar, "phi_wait": dict(w), "phi_res": dict(r), "shared": shared, "n_events": g.n}


CHILDREN = dict(S.CHILDREN, **{"bsolo-child": bsolo_child, "sim-child": sim_child, "score2-child": score2_child})


# ------------------------------------------------------------------ solo data
def solo_all(jobs):
    plats, Tref, solo0 = S.plats_all()
    items_b, items_d, seen = [], [], {}
    for g, hw, cz, keys, fa, alias in plats:
        for k in keys:
            if k in seen:
                continue
            seen[k] = fa
            items_b.append((f"bs_{k}", {"stream": k, "faults": fa, "chunk": ALL}))
            items_d.append((f"ds_{k}", {"stream": k, "faults": fa, "chunk": ALL}))
    rep = [(f"bsrep_{k}", {"stream": k, "faults": fa, "chunk": S.BASE}) for k, fa in seen.items() if (k, "256d2d") in Tref]
    rb = S.run_all(jobs, items_b + rep, cmd="bsolo-child")
    rd = S.run_all(jobs, items_d, cmd="dsolo")
    T, solo, ideal = {}, {}, {}
    rows = []
    for k in seen:
        b, d = rb[f"bs_{k}"], rd[f"ds_{k}"]
        T[(k, CZ)] = b["T"]
        ideal[(k, CZ)] = b["ideal"]
        solo[(k, CZ)] = d
        ck = d["check4"]
        rows.append({"stream": k, "T": b["T"], "T_hat": d["T_hat"], "err": d["T_hat"] / b["T"] - 1, "T_rand_mean": d["T_rand_mean"], "err_rand": d["T_rand_mean"] / b["T"] - 1, "events": d["n_events"],
                     "blocks": b["chunk_stats"]["blocks_added"], "extra_over_base": b["chunk_stats"]["extra_over_base"], **ck})
    wc("solo_256all.csv", rows)
    repdev = {k: rb[f"bsrep_{k}"]["T"] / Tref[(k, "256d2d")] - 1 for k, fa in seen.items() if (k, "256d2d") in Tref}
    errs = [abs(r["err"]) for r in rows]
    summ = {"n": len(rows), "err_abs_median": statistics.median(errs), "err_abs_max": max(errs), "rep_256d2d_vs_e4b": repdev,
            "check4_bad": [r["stream"] for r in rows if r["bad_monotone"] or r["bad_sink"] or r["bad_block_D"] or r["occupancy_mismatch"]]}
    wj("solo_256all.json", summ)
    log(S.dumps(D.round_all(summ)))
    if summ["check4_bad"] or summ["err_abs_median"] > 0.05 or max(abs(v) for v in repdev.values()) > 0.04:
        raise D.DstuError(f"solo self-check: {summ}")
    return plats, T, solo, ideal


# ------------------------------------------------------------------ 10.1, 10.2
def info_all(T, solo):
    info = S.a2_info()
    for hw, i in info.items():
        i["T"] = [T[(k, CZ)] for k in i["keys"]]
        i["That"] = [solo[(k, CZ)]["T_hat"] for k in i["keys"]]
    return info


def run_chunking(jobs, tag, plan, info, agg):
    rows = S.rows_2a(info, agg)
    en = S.enum_2a(rows)
    needed = collections.defaultdict(set)
    for r in rows:
        for m in S.METHODS_BASE:
            needed[(r["hw"], r["X"], r["rx"])].add((r["M"][m], r["F"]))
    items, sims = [], []
    for (hw, xn, rx), ranks in needed.items():
        i = info[hw]
        x = i["keys"][0] if xn == "MHA" else i["keys"][1]
        base = {"faults": i["faults"], "streams": [i["keys"][0], i["keys"][1], x], "plan": plan, "chain": True, "release": {0: 0, 1: 0, 2: rx * i["T"][0]}}
        for m, f in set(ranks) | set(RANKS4):
            items += S.seeded(f"{tag}_{hw}_{xn}_{rx}_{S.rank_name(m, f)}", dict(base, rank={0: m, 1: f, 2: 1}))
        items += S.seeded(f"{tag}_{hw}_{xn}_{rx}_P1", dict(base, rank=None))
        sims.append((f"sim{tag}_{hw}_{xn}_{rx}", {k: base[k] for k in ("faults", "streams", "plan", "release")}))
    for hw in S.HWS:
        i = info[hw]
        items += S.seeded(f"{tag}_{hw}_chain", {"faults": i["faults"], "streams": i["keys"], "plan": plan, "chain": True, "release": {0: 0, 1: 0}, "rank": {0: 0, 1: 1}})
    log(f"{tag}: {len(rows)} rows, {len(items)} backend processes, {len(sims)} CPCM processes")
    res = S.run_all(jobs, items)
    sim = S.run_all(jobs, sims, cmd="sim-child")
    ck, audit, bad_dep, chunk_bad = [], 0, 0, 0
    for hw in S.HWS:
        TM, TF = info[hw]["T"]
        for x in S.runs_of(res, f"{tag}_{hw}_chain"):
            ck.append(float(x["C"][1]) / (TM + TF) - 1)
    for x in res.values():
        audit += x["audit"]["n_mismatch"]
        bad_dep += x["first_start"][1] < x["C"][0]
        chunk_bad += not (x["chunk_stats"]["bytes_conserved"] and x["chunk_stats"]["tree_conserved"])
    checks = {"chain_rel_max": max(abs(c) for c in ck), "audit_mismatch": audit, "ffn_before_mha": bad_dep, "chunk_bad": chunk_bad}
    log(tag + " checks " + S.dumps(D.round_all(checks)))
    if audit or bad_dep or chunk_bad or checks["chain_rel_max"] > 0.04:
        raise D.DstuError(f"{tag}: self-check {checks}")

    def lv(r, nm):
        L, Lc, Lx, Cs = [], [], [], []
        for x in S.runs_of(res, nm):
            a_, b_ = (float(x["C"][1]) - r["d_req"]) / (r["T_M"] + r["T_F"]), (float(x["C"][2]) - r["d_x"]) / r["T_X"]
            L.append(max(a_, b_))
            Lc.append(a_)
            Lx.append(b_)
            Cs.append([float(x["C"][j]) for j in range(3)])
        return {"L": statistics.mean(L), "range": max(L) - min(L), "Lc": statistics.mean(Lc), "Lx": statistics.mean(Lx), "C": [statistics.mean(c[j] for c in Cs) for j in range(3)]}
    table = []
    for r in rows:
        o = {k: r[k] for k in ("hw", "X", "rx", "k_req", "k_x", "basis", "d_req", "d_x")}
        pre = f"{tag}_{r['hw']}_{r['X']}_{r['rx']}_"
        for m in S.METHODS_BASE:
            v = lv(r, pre + S.rank_name(r["M"][m], r["F"]))
            o[f"D_{m}"], o[f"o_{m}"], o[f"L_{m}"], o[f"R_{m}"] = r["D"][m], S.rank_name(r["M"][m], r["F"]), v["L"], v["range"]
            o[f"Lc_{m}"], o[f"Lx_{m}"] = v["Lc"], v["Lx"]
            o[f"CM_{m}"], o[f"CF_{m}"], o[f"CX_{m}"] = v["C"]
        o["L_P1"] = lv(r, pre + "P1")["L"]
        # CP_SIM and oracle over the 4 strict rankings
        pred = {}
        C4 = sim[f"sim{tag}_{r['hw']}_{r['X']}_{r['rx']}"]["C"]
        for m, f in RANKS4:
            n = S.rank_name(m, f)
            C = C4[n]
            pred[n] = max((C[1] - r["d_req"]) / (r["T_M"] + r["T_F"]), (C[2] - r["d_x"]) / r["T_X"])
        best = min(pred.values())
        tied = [n for n, v in pred.items() if v == best]
        ch = o["o_CP_ED"] if o["o_CP_ED"] in tied else sorted(tied)[0]
        back = {S.rank_name(m, f): lv(r, pre + S.rank_name(m, f)) for m, f in RANKS4}
        orc = min(back, key=lambda n: back[n]["L"])
        v = back[ch]
        o.update({"o_CP_SIM": ch, "L_CP_SIM": v["L"], "R_CP_SIM": v["range"], "Lc_CP_SIM": v["Lc"], "Lx_CP_SIM": v["Lx"], "CM_CP_SIM": v["C"][0], "CF_CP_SIM": v["C"][1], "CX_CP_SIM": v["C"][2],
                  "pred_CP_SIM": best, "n_tied_pred": len(tied), "o_oracle": orc, "L_oracle": back[orc]["L"], "R_oracle": back[orc]["range"],
                  **{f"L4_{n}": back[n]["L"] for n in back}, **{f"P4_{n}": pred[n] for n in pred}})
        table.append(o)
    tt = [t for s in sim.values() for t in s["forward_s"]]
    return table, en, checks, {"cpcm_runs": len(tt), "forward_s_median": statistics.median(tt), "forward_s_max": max(tt), "compile_s_median": statistics.median(s["compile_s"] for s in sim.values()), "n_events": sorted({s["n_events"] for s in sim.values()}), "per_row_group": 4}


def regret(table, m):
    d = [r[f"L_{m}"] - r["L_oracle"] for r in table]
    same = sum(1 for r in table if r[f"o_{m}"] == r["o_oracle"])
    return {"n": len(table), "L_median": statistics.median(r[f"L_{m}"] for r in table), "L_worst": max(r[f"L_{m}"] for r in table), "regret_median": statistics.median(d), "regret_mean": statistics.mean(d), "regret_max": max(d),
            "regret_le_TH": sum(x <= TH for x in d), "same_as_oracle": same, "same_rate": same / len(table)}


def sim_block(table):
    out = {m: regret(table, m) for m in ("CP_SIM", "CP_ED", "UD", "HALF", "AGG_ED", "oracle")} if False else {}
    for m in ("CP_SIM", "CP_ED", "UD", "HALF", "AGG_ED"):
        out[m] = regret(table, m)
    out["oracle"] = {"L_median": statistics.median(r["L_oracle"] for r in table), "L_worst": max(r["L_oracle"] for r in table)}
    pairs = {}
    for m in ("CP_ED", "UD", "HALF", "AGG_ED"):
        sub = [r for r in table if r["o_CP_SIM"] != r[f"o_{m}"]]
        d = [r["L_CP_SIM"] - r[f"L_{m}"] for r in sub]
        pairs[m] = {"n_diff": len(sub), "sim_better": sum(x < -TH for x in d), "sim_worse": sum(x > TH for x in d), "tie": sum(abs(x) <= TH for x in d)}
        if d:
            pairs[m].update(median=statistics.median(d), min=min(d), max=max(d), worst_sim=max(r["L_CP_SIM"] for r in sub), worst_m=max(r[f"L_{m}"] for r in sub))
    out["pairs"] = pairs
    e = [abs(r["pred_CP_SIM"] - r["L_CP_SIM"]) for r in table]
    out["pred_vs_backend"] = {"abs_median": statistics.median(e), "abs_max": max(e)}
    out["rank_counts"] = dict(collections.Counter(r["o_CP_SIM"] for r in table))
    out["oracle_rank_counts"] = dict(collections.Counter(r["o_oracle"] for r in table))
    out["n_tied_pred_rows"] = sum(1 for r in table if int(r["n_tied_pred"]) > 1)
    return out


def cuts(table):
    c = {"all": table}
    for hw in S.HWS:
        c[hw] = [r for r in table if r["hw"] == hw]
    for x in ("MHA", "FFN"):
        c[x] = [r for r in table if r["X"] == x]
        for hw in S.HWS:
            c[f"{hw}_{x}"] = [r for r in table if r["hw"] == hw and r["X"] == x]
    return c


def stage_a(jobs):
    t0 = time.time()
    plats, T, solo, ideal = solo_all(jobs)
    infoA = info_all(T, solo)
    x = S.run_all(jobs, [(f"agg256_{hw}", {"faults": infoA[hw]["faults"], "plan": ALL, "streams": {"MHA": infoA[hw]["keys"][0], "FFN": infoA[hw]["keys"][1]}}) for hw in S.HWS], cmd="agg-child")
    aggA = {hw: x[f"agg256_{hw}"] for hw in S.HWS}
    infoD = S.a2_info()
    y = S.run_all(jobs, [(f"agg_{hw}", {"faults": infoD[hw]["faults"], "streams": {"MHA": infoD[hw]["keys"][0], "FFN": infoD[hw]["keys"][1]}}) for hw in S.HWS], cmd="agg-child")
    aggD = {hw: y[f"agg_{hw}"] for hw in S.HWS}
    S.METHODS_BASE = list(S.METHODS_2A)
    tA, enA, ckA, tmA = run_chunking(jobs, "c", ALL, infoA, aggA)
    tD, enD, ckD, tmD = run_chunking(jobs, "a", S.BASE, infoD, aggD)
    wc("a2_rows_256all.csv", tA)
    wc("a2_rows_256d2d.csv", tD)
    S.METHODS_2A = S.METHODS_BASE
    sA, sD = S.summary_2a(tA), S.summary_2a(tD)
    simA = {k: sim_block(v) for k, v in cuts(tA).items()}
    simD = {k: sim_block(v) for k, v in cuts(tD).items()}
    pair = {}
    key = lambda r: (r["hw"], r["X"], r["rx"], r["k_req"], r["k_x"], r["basis"])
    dD = {key(r): r for r in tD}
    for m in S.METHODS_BASE + ["CP_SIM"]:
        d = [r[f"L_{m}"] - dD[key(r)][f"L_{m}"] for r in tA]
        same = sum(1 for r in tA if r[f"o_{m}"] == dD[key(r)][f"o_{m}"])
        pair[m] = {"dL_median": statistics.median(d), "dL_min": min(d), "dL_max": max(d), "all_better": sum(x < -TH for x in d), "all_worse": sum(x > TH for x in d), "same_order": same,
                   "CX_all_median": statistics.median(r[f"CX_{m}"] for r in tA), "CX_d2d_median": statistics.median(dD[key(r)][f"CX_{m}"] for r in tA),
                   "CF_all_median": statistics.median(r[f"CF_{m}"] for r in tA), "CF_d2d_median": statistics.median(dD[key(r)][f"CF_{m}"] for r in tA),
                   "CM_all_median": statistics.median(r[f"CM_{m}"] for r in tA), "CM_d2d_median": statistics.median(dD[key(r)][f"CM_{m}"] for r in tA),
                   "L_all_median": statistics.median(r[f"L_{m}"] for r in tA), "L_d2d_median": statistics.median(dD[key(r)][f"L_{m}"] for r in tA)}
    wj("a.json", {"summary_256all": sA, "summary_256d2d": sD, "enum_256all": enA, "enum_256d2d": enD, "checks": {"256all": ckA, "256d2d": ckD}, "sim_256all": simA, "sim_256d2d": simD, "pair_vs_d2d": pair,
                  "cpcm_time": {"256all": tmA, "256d2d": tmD}, "T": {h: infoA[h]["T"] for h in S.HWS}, "That": {h: infoA[h]["That"] for h in S.HWS}, "agg": aggA, "wall": time.time() - t0, "command": sys.argv, "artifact_commit": E._artifact_commit()})
    log(f"a done {time.time() - t0:.0f} s")


# ------------------------------------------------------------------ 10.3
def fill(order, msgs, budget, act, cost):
    over, used = {}, 0
    for m in order:
        a = act(m, over)
        if a is None:
            continue
        c = cost(m, a, over)
        if used + c > budget:
            continue
        over[m] = a
        used += c
    return over, used


def static_order(s):
    ms = sorted(s["shared"], key=str)
    random.Random(STATIC_SEED).shuffle(ms)
    ms = [m for m in ms if s["shared"][m] > 0]
    ms.sort(key=lambda m: -s["shared"][m] * s["msgs"][m][0])
    return ms


def jac_over(a, b):
    A, B_ = set(a), set(b)
    return {"jaccard": len(A & B_) / len(A | B_) if A | B_ else 1.0, "common": len(A & B_), "n_a": len(A), "n_b": len(B_), "same_size": sum(1 for m in A & B_ if a[m] == b[m])}


def sm(v):
    return {"median": statistics.median(v), "min": min(v), "max": max(v), "better": sum(z < -TH for z in v), "worse": sum(z > TH for z in v)}


def stage_b(jobs):
    t0 = time.time()
    cfgs = S.a2_cfgs()
    sp = lambda c, plan: {"faults": c["faults"], "streams": c["streams"], "release": c["release"], "rank": c["rank"], "T": c["T"], "d": c["d"], "plan": plan}
    sc = S.run_all(jobs, [(f"sc2A_{c['name']}", sp(c, S.BASE)) for c in cfgs] + [(f"sc2B_{c['name']}", sp(c, ALL)) for c in cfgs], cmd="score2-child")
    items, plan_a, plan_b = [], {}, {}
    for c in cfgs:
        nm = c["name"]
        items += S.seeded(f"b_{nm}_256d2d", S.run_spec(c, S.UNI["256d2d"]))
        items += S.seeded(f"b_{nm}_256all", S.run_spec(c, ALL))
        items += S.seeded(f"b_{nm}_64all", S.run_spec(c, S.UNI["64all"]))
        # group A
        s = sc[f"sc2A_{nm}"]
        msgs = s["msgs"]
        act = lambda m, ov, msgs=msgs: S.action(m, msgs, ov, True, S.K256)
        cost = lambda m, a, ov, msgs=msgs: S.blocks(msgs[m][0], a) - S.cur_blocks(m, msgs, ov)
        o, u = fill(static_order(s), msgs, 540, act, cost)
        pl = {"static": (o, u, f"sA_{nm}_{S.fh(o)}")}
        items += S.seeded(pl["static"][2], S.run_spec(c, dict(S.BASE, override=o)))
        o, u = S.select(s["phi_res"], msgs, 540, True, S.K256)
        pl["res"] = (o, u, f"s_{nm}_{S.fh(o)}")
        items += S.seeded(pl["res"][2], S.run_spec(c, dict(S.BASE, override=o), measure=s["istar"]))
        for j in range(3):
            o, u = S.select_random(msgs, 540, True, S.K256, S.SEL_SEED + j)
            pl[f"rand{j}"] = (o, u, f"s_{nm}_{S.fh(o)}")
            items += S.seeded(pl[f"rand{j}"][2], S.run_spec(c, dict(S.BASE, override=o)))
        plan_a[nm] = pl
        # group B
        s = sc[f"sc2B_{nm}"]
        msgs = s["msgs"]
        actb = lambda m, ov, msgs=msgs: S.K64 if msgs[m][0] > S.K64 and m not in ov else None
        costb = lambda m, a, ov, msgs=msgs: S.blocks(msgs[m][0], S.K64) - S.blocks(msgs[m][0], S.K256)
        for bud in S.BUDGETS:
            pb = {}
            order = [m for m, v in sorted(s["phi_res"].items(), key=lambda x: (-x[1], str(x[0]))) if v > 0]
            pb["res"] = fill(order, msgs, bud, actb, costb)
            pb["static"] = fill(static_order(s), msgs, bud, actb, costb)
            for j in range(3):
                rng = random.Random(S.SEL_SEED + j)
                ms = sorted(msgs, key=str)
                rng.shuffle(ms)
                ms.sort(key=lambda m: -msgs[m][0])
                pb[f"rand{j}"] = fill(ms, msgs, bud, actb, costb)
            for tg, (o, u) in pb.items():
                n = f"sB_{nm}_{S.fh(o)}"
                items += S.seeded(n, S.run_spec(c, dict(ALL, override=o)))
                plan_b[(nm, bud, tg)] = (o, u, n)
    items = list({n: x for n, x in items}.items())
    log(f"10.3: {len(items)} processes")
    res = S.run_all(jobs, items)
    rows = []
    for c in cfgs:
        nm = c["name"]
        ref = {n: S.stat_b(S.runs_of(res, f"b_{nm}_{n}"), c) for n in ("256d2d", "256all", "64all")}
        base = lambda grp: ref["256d2d"] if grp == "A" else ref["256all"]
        for n, st_ in ref.items():
            rows.append({"config": nm, "hw": c["hw"], "order": c["order"], "group": "ref", "method": n, "budget": "", "new_blocks": "", "n_msgs": "", **st_})
        for tg, (o, u, n) in plan_a[nm].items():
            st_ = S.stat_b(S.runs_of(res, n), c)
            if st_["blocks_total"] - ref["256d2d"]["blocks_total"] != u:
                raise D.DstuError(f"block accounting A {nm} {tg}")
            rows.append({"config": nm, "hw": c["hw"], "order": c["order"], "group": "A", "method": tg, "budget": 540, "new_blocks": u, "n_msgs": len(o), **st_})
        for bud in S.BUDGETS:
            for tg in ("res", "static", "rand0", "rand1", "rand2"):
                o, u, n = plan_b[(nm, bud, tg)]
                st_ = S.stat_b(S.runs_of(res, n), c)
                if st_["blocks_total"] - ref["256all"]["blocks_total"] != u:
                    raise D.DstuError(f"block accounting B {nm} {tg}")
                rows.append({"config": nm, "hw": c["hw"], "order": c["order"], "group": "B", "method": tg, "budget": bud, "new_blocks": u, "n_msgs": len(o), **st_})
    for r in rows:
        ref = next(x for x in rows if x["config"] == r["config"] and x["method"] == ("256all" if r["group"] == "B" else "256d2d") and x["group"] == "ref")
        r["extra_vs_start"] = r["extra_occupancy"] - ref["extra_occupancy"]
        r["dL_vs_start"] = r["L"] - ref["L"]
    wc("b_rows.csv", [{k: v for k, v in r.items() if k != "seeds_L"} for r in rows])

    def g(nm, grp, m, bud=""):
        return next(x for x in rows if x["config"] == nm and x["method"] == m and x["group"] == grp and x["budget"] == bud)
    out = {}
    names = [c["name"] for c in cfgs]
    # A
    p = []
    for nm in names:
        s_, r_ = g(nm, "A", "static", 540), g(nm, "A", "res", 540)
        rnd = statistics.mean(g(nm, "A", f"rand{j}", 540)["L"] for j in range(3))
        p.append({"config": nm, "L_static": s_["L"], "L_res": r_["L"], "L_rand": rnd, "L_256all": g(nm, "ref", "256all")["L"], "L_base": g(nm, "ref", "256d2d")["L"], "blocks_static": s_["new_blocks"], "blocks_res": r_["new_blocks"], "extra_static": s_["extra_vs_start"], "extra_res": r_["extra_vs_start"],
                  "C0_static": s_["C0"], "C0_res": r_["C0"], "C1_static": s_["C1"], "C1_res": r_["C1"], "M_static": s_["M"], "M_res": r_["M"], "range_static": s_["L_range"],
                  **{f"ov_{k}": v for k, v in jac_over(plan_a[nm]["static"][0], plan_a[nm]["res"][0]).items()}})
    out["A"] = {"rows": p, "median": {k: statistics.median(x[k] for x in p) for k in ("L_static", "L_res", "L_rand", "L_256all", "L_base", "ov_jaccard")}, "worst": {k: max(x[k] for x in p) for k in ("L_static", "L_res", "L_rand", "L_256all", "L_base")},
                "res_minus_static": sm([x["L_res"] - x["L_static"] for x in p]), "res_minus_rand": sm([x["L_res"] - x["L_rand"] for x in p]), "res_minus_256all": sm([x["L_res"] - x["L_256all"] for x in p]),
                "static_minus_256all": sm([x["L_static"] - x["L_256all"] for x in p]), "static_minus_rand": sm([x["L_static"] - x["L_rand"] for x in p])}
    # B
    for bud in S.BUDGETS:
        p = []
        for nm in names:
            s_, r_ = g(nm, "B", "static", bud), g(nm, "B", "res", bud)
            rnd = statistics.mean(g(nm, "B", f"rand{j}", bud)["L"] for j in range(3))
            p.append({"config": nm, "L_static": s_["L"], "L_res": r_["L"], "L_rand": rnd, "L_256all": g(nm, "ref", "256all")["L"], "L_64all": g(nm, "ref", "64all")["L"], "blocks_static": s_["new_blocks"], "blocks_res": r_["new_blocks"], "blocks_rand": g(nm, "B", "rand0", bud)["new_blocks"],
                      "n_res": r_["n_msgs"], "extra_static": s_["extra_vs_start"], "extra_res": r_["extra_vs_start"], "C0_static": s_["C0"], "C0_res": r_["C0"], "C0_256all": g(nm, "ref", "256all")["C0"], "C1_static": s_["C1"], "C1_res": r_["C1"], "C1_256all": g(nm, "ref", "256all")["C1"], "M_res": r_["M"], "range_res": r_["L_range"],
                      **{f"ov_{k}": v for k, v in jac_over(plan_b[(nm, bud, "static")][0], plan_b[(nm, bud, "res")][0]).items()}})
        out[f"B{bud}"] = {"rows": p, "median": {k: statistics.median(x[k] for x in p) for k in ("L_static", "L_res", "L_rand", "L_256all", "L_64all", "ov_jaccard")}, "worst": {k: max(x[k] for x in p) for k in ("L_static", "L_res", "L_rand", "L_256all", "L_64all")},
                          "res_minus_static": sm([x["L_res"] - x["L_static"] for x in p]), "res_minus_rand": sm([x["L_res"] - x["L_rand"] for x in p]), "res_minus_256all": sm([x["L_res"] - x["L_256all"] for x in p]), "res_minus_64all": sm([x["L_res"] - x["L_64all"] for x in p]),
                          "static_minus_256all": sm([x["L_static"] - x["L_256all"] for x in p]), "static_minus_rand": sm([x["L_static"] - x["L_rand"] for x in p])}
    out["istar"] = {f"{k[:4]}{k[5:]}": v["istar"] for k, v in sc.items()}
    out["cpcm_J"] = {k: v["J"] for k, v in sc.items()}
    out["audit_mismatch"] = sum(x["audit"]["n_mismatch"] for x in res.values())
    out["chunk_bad"] = sum(not (x["chunk_stats"]["bytes_conserved"] and x["chunk_stats"]["tree_conserved"]) for x in res.values())
    out.update(wall=time.time() - t0, command=sys.argv, artifact_commit=E._artifact_commit())
    wj("b.json", out)
    log(f"b done {time.time() - t0:.0f} s, audit {out['audit_mismatch']}, chunk_bad {out['chunk_bad']}")
    if out["audit_mismatch"] or out["chunk_bad"]:
        raise D.DstuError("10.3 self-check")


# ------------------------------------------------------------------ 10.4
def stage_s1(jobs):
    t0 = time.time()
    plats0, T, solo, ideal = solo_all(jobs)
    plats = [(g, hw, CZ, keys, fa, alias) for g, hw, cz, keys, fa, alias in plats0]
    cfgs = D.build_cfgs(1, plats, T, solo, ideal)
    log(f"10.4: {len(cfgs)} rows")
    items = {}
    for c in cfgs:
        nm, n, spb = c["name"], c["n"], D.spec_base(c)
        items[f"s1all_{nm}_P1"] = dict(spb, policy={"name": "P1"})
        for p in itertools.permutations(range(n)):
            o = "".join(map(str, p))
            items[f"s1all_{nm}_fixed_{o}"] = dict(spb, policy={"name": "fixed", "rank": D.rank_of(o)})
        sd = {i: ideal[(s, CZ)] for i, s in enumerate(c["streams"])}
        tl = {i: D.trim(solo[(s, CZ)]) for i, s in enumerate(c["streams"])}
        com = {"ideal": sd, "d_ideal": c["d"], "T_ideal": c["T"], "T_hat": c["That"]}
        items[f"s1all_{nm}_P7_{c['tg']}"] = dict(spb, policy={"name": "IDEAL", "ideal": sd, "d_ideal": c["d"]})
        items[f"s1all_{nm}_CS_{c['tg']}"] = dict(spb, policy=dict(com, name="C-S", tail=tl, d=c["dhat"]))
    log(f"10.4: {len(items)} processes")
    res = S.run_all(jobs, list(items.items()), cmd="dchild")
    aud = collections.Counter()
    for x in res.values():
        for q in x["runs"]:
            aud["n_issues"] += q["audit"]["n_issues"]
            aud["n_mismatch"] += q["audit"]["n_mismatch"]
            aud["early"] += q["early_events"]
            if q["chunk_stats"] and not (q["chunk_stats"]["bytes_conserved"] and q["chunk_stats"]["tree_conserved"]):
                aud["chunk_bad"] += 1
    rows, rng = [], {}
    for c in cfgs:
        nm, tg, n, Tc, rel = c["name"], c["tg"], c["n"], c["T"], c["release"]
        fx = {"".join(map(str, p)): f"s1all_{nm}_fixed_{''.join(map(str, p))}" for p in itertools.permutations(range(n))}
        st_ = {o: D.stat(res[v]["runs"], c) for o, v in fx.items()}
        best = min(st_, key=lambda o: st_[o]["L"])
        fcfs = "".join(map(str, sorted(range(n), key=lambda i: (rel[i], i))))
        sjf = "".join(map(str, sorted(range(n), key=lambda i: (Tc[i], i))))
        p4 = D.order_of(c["d"], n)
        c4 = D.order_of(c["dhat"], n)
        pol = {"P1": D.stat(res[f"s1all_{nm}_P1"]["runs"], c), "P3": st_[sjf], "P4": st_[p4], "P5": st_[best], "C4": st_[c4],
               "P7": D.stat(res[f"s1all_{nm}_P7_{tg}"]["runs"], c), "C-S": D.stat(res[f"s1all_{nm}_CS_{tg}"]["runs"], c)}
        pol["P2"] = pol["P1"] if c["order"] == "d0" else st_[fcfs]
        o = {"config": nm, "group": c["group"], "hw": c["hw"], "order": c["order"], "dl": tg, "P5_member": "fixed_" + best, "C4_order": c4, "P4_order": p4, "seeds": len(c["seeds"])}
        for p, s in pol.items():
            o[f"L_{p}"], o[f"Lraw_{p}"], o[f"S_{p}"], o[f"V_{p}"], o[f"M_{p}"], o[f"Lrange_{p}"] = s["L"], s["Lraw"], s["S"], s["V"], s["M"], s["L_range"]
        o["errT_max"] = max(abs(c["That"][i] / Tc[i] - 1) for i in Tc)
        tau = max(D.NOISE_L, D.NOISE_S * pol["P1"]["S"])
        den = pol["P1"]["L"] - pol["P7"]["L"]
        o["tau"], o["den_C-S"], o["den_noise"] = tau, den, den < tau
        o["ret_C-S"] = (pol["P1"]["L"] - pol["C-S"]["L"]) / den if abs(den) > 1e-12 else ""
        o["_k"] = (1, nm, tg)
        rng[o["_k"]] = pol["P5"]["L_range"]
        for i, t in Tc.items():
            o[f"T_{i}"] = t
        rows.append(o)
    wc("s1_report.csv", [{k: v for k, v in r.items() if k != "_k"} for r in rows])
    old = {(r["config"], r["dl"]): r for r in rcsv(os.path.join(D.OUT, "setting1_report.csv"))}
    POLS = ["P1", "P2", "P4", "P5", "P7", "C4", "C-S"]

    def stats(sub, get):
        out = {}
        for p in POLS:
            L = [get(r, f"L_{p}") for r in sub]
            out[p] = {"L_median": statistics.median(L), "L_worst": max(L), "V_mean": statistics.mean(get(r, f"V_{p}") for r in sub), "S_median": statistics.median(get(r, f"S_{p}") for r in sub)}
        for p in ("P4", "P5", "P7", "C4", "C-S", "P3"):
            for base in ("P2", "P1"):
                g_ = [1 - get(r, f"S_{p}") / get(r, f"S_{base}") for r in sub]
                out[f"gain_{p}_vs_{base}"] = {"median": statistics.median(g_), "min": min(g_), "max": max(g_)}
        return out
    newg = lambda r, k: float(r[k])
    S_ = {"n": len(rows)}
    for fam in ("a1", "a2"):
        sub = [r for r in rows if r["group"].startswith(fam)]
        oldsub = [old[(r["config"], r["dl"])] for r in sub]
        S_[fam] = {"all_link": stats(sub, newg), "stage1": stats(oldsub, newg), "M_median_all_link": {p: statistics.median(r[f"M_{p}"] for r in sub) for p in POLS}}
        # the "20-66%" formula: 1 - S(short first) / S(FCFS) on the arrival orders where FCFS is not short first (E4a section 3.3)
        for od in ("d0", "long_first"):
            for hw in S.HWS:
                x = [r for r in sub if r["order"] == od and r["hw"] == hw]
                if x:
                    S_[fam][f"e4a_{od}_{hw}"] = {"all_link": {"median": statistics.median(1 - r["S_P3"] / r["S_P2"] for r in x), "min": min(1 - r["S_P3"] / r["S_P2"] for r in x), "max": max(1 - r["S_P3"] / r["S_P2"] for r in x)},
                                                 "stage1": {"median": statistics.median(1 - float(old[(r["config"], r["dl"])]["S_P3"]) / float(old[(r["config"], r["dl"])]["S_P2"]) for r in x),
                                                            "min": min(1 - float(old[(r["config"], r["dl"])]["S_P3"]) / float(old[(r["config"], r["dl"])]["S_P2"]) for r in x),
                                                            "max": max(1 - float(old[(r["config"], r["dl"])]["S_P3"]) / float(old[(r["config"], r["dl"])]["S_P2"]) for r in x)}}
        inc = [r for r in sub if not r["den_noise"]]
        v = [r["ret_C-S"] for r in inc]
        S_[fam]["retention_C-S"] = {"n_in": len(inc), "n_out": len(sub) - len(inc), "median": statistics.median(v) if v else None, "min": min(v) if v else None, "max": max(v) if v else None}
    # per group E4a formula
    S_["e4a_by_group"] = {}
    for g in sorted({r["group"] for r in rows}):
        for hw in S.HWS:
            for od in ("d0", "long_first"):
                x = {r["config"]: r for r in rows if r["group"] == g and r["hw"] == hw and r["order"] == od}
                for r in x.values():
                    q = old[(r["config"], r["dl"])]
                    S_["e4a_by_group"][r["config"]] = {"all_link": 1 - r["S_P3"] / r["S_P2"], "stage1": 1 - float(q["S_P3"]) / float(q["S_P2"]), "S_P2_all": r["S_P2"], "S_P3_all": r["S_P3"], "S_P2_stage1": float(q["S_P2"]), "S_P3_stage1": float(q["S_P3"])}
                    break
    # C4 - C-S pairing, format of stage 1
    pair = {}
    cc = {"all": rows, "a1": [r for r in rows if r["group"].startswith("a1")], "a2": [r for r in rows if r["group"].startswith("a2")]}
    for hw in S.HWS:
        cc[hw] = [r for r in rows if r["hw"] == hw]
        for f in ("a1", "a2"):
            cc[f"{f}_{hw}"] = [r for r in rows if r["group"].startswith(f) and r["hw"] == hw]
    for g in sorted({r["group"] for r in rows}):
        cc[g] = [r for r in rows if r["group"] == g]
    for od in ("d0", "short_first", "long_first"):
        for f in ("a1", "a2"):
            cc[f"{f}_{od}"] = [r for r in rows if r["group"].startswith(f) and r["order"] == od]
    for k, v in cc.items():
        if v:
            pair[k] = DP.block(v, rng)
    S_["pair_c4_cs"] = pair
    S_["checks"] = dict(aud)
    S_["err_T_hat_max"] = max(r["errT_max"] for r in rows)
    S_.update(wall=time.time() - t0, command=sys.argv, seeds=B.SEEDS, ext6=sorted(D.EXT6), artifact_commit=E._artifact_commit())
    wj("s1.json", S_)
    log(f"s1 done {time.time() - t0:.0f} s; checks {dict(aud)}")
    if aud["n_mismatch"] or aud["chunk_bad"] or aud["early"]:
        raise D.DstuError(f"10.4 self-check {dict(aud)}")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] in ("dsolo", "dchild"):
        (D.solo_child if sys.argv[1] == "dsolo" else D.child)(sys.argv[2])
        return
    if len(sys.argv) >= 3 and sys.argv[1] in CHILDREN:
        with open(sys.argv[2], "rb") as f:
            spec = pickle.load(f)
        r = CHILDREN[sys.argv[1]](spec)
        with open(spec["sink"], "wb") as f:
            pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["solo", "a", "b", "s1"])
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    {"solo": solo_all, "a": stage_a, "b": stage_b, "s1": stage_s1}[a.cmd](a.jobs)


if __name__ == "__main__":
    try:
        main()
    except (D.DstuError, B.E4bError, E.E4Error, cpcm.CPCMError) as exc:
        print(f"[dstu2c] STOP: {exc}", flush=True)
        sys.exit(2)
