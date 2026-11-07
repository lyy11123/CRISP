"""DSTU final runs (dstu/DSTU_HANDOFF.md section 13). Sources elsewhere unchanged.

  python dstu_final.py check --jobs 14   13.1 preparation and 13.2 plans, self-checks that need no runs
  python dstu_final.py probe --jobs 14   a sample of the runs, walls (reused by the full runs)
  python dstu_final.py s2 --jobs 14      13.1
  python dstu_final.py sel --jobs 14     13.2
Results in dstu/out/final/; simulations cached in dstu/out/stage2/sims (names start with s2all_, fin).
"""
import argparse
import collections
import csv
import gzip
import json
import os
import pickle
import random
import statistics
import sys
import time

import dstu as D
import dstu2 as S
import dstu2c as C
import dstu_oh as OH
from dstu import B, E, cpcm

S.SCRIPT = os.path.abspath(__file__)
OUT = os.path.join(D.OUT, "final")
ALL = S.UNI["256all"]
CZ = "256all"
TH = S.TH
BUD = 540
OS2 = (50, 200)
ORD2 = ["01", "10"]
med = statistics.median
mean = statistics.mean


def log(m):
    print(f"[dstu_final] {m}", flush=True)


def wj(n, o):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, n), "w", encoding="utf-8") as f:
        json.dump(o, f, indent=1, default=str)


def wc(n, rows):
    os.makedirs(OUT, exist_ok=True)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(os.path.join(OUT, n), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def rcsv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ child
def shared_child(spec):
    """Links each message of the other streams shares with stream istar (routes only, as score2-child in 10.3)."""
    E._artifact_path()
    hp0 = E.build_platform(E.CFG, spec["faults"])
    blobs = [B.load_stream(s)["events_blob"] for s in spec["streams"]]
    raw = E.merged_events(blobs)
    links = {}
    for t, e in raw.items():
        if e.event_type == "communication":
            links[S.gid(t)] = {(r, l) for r, leaves in e.paths.items() for l in leaves}
    istar = spec["istar"]
    mine = set().union(*[v for m, v in links.items() if m[0] - 10 == istar])
    return {"shared": {m: len(v & mine) for m, v in links.items() if m[0] - 10 != istar}}


CHILDREN = dict(OH.CHILDREN, **{"fin-shared": shared_child})


# ------------------------------------------------------------------ 13.1 configuration
def keys_order(c, val):
    return D.order_of({i: c["release"][i] + c["k"][i] * val[i] for i in c["T"]}, c["n"])


def prepare(jobs):
    plats, T0, solo0 = S.plats_all()
    c0 = D.build_cfgs(2, plats, T0, solo0, {})
    if len(c0) != 156:
        raise D.DstuError(f"{len(c0)} rows")
    seen = {}
    for c in c0:
        for k in c["streams"]:
            seen[k] = c["faults"]
    rb = S.run_all(jobs, [(f"bs_{k}", {"stream": k, "faults": fa, "chunk": ALL}) for k, fa in seen.items()], cmd="bsolo-child")
    rd = S.run_all(jobs, [(f"ds_{k}", {"stream": k, "faults": fa, "chunk": ALL}) for k, fa in seen.items()], cmd="dsolo")
    T = {(k, CZ): rb[f"bs_{k}"]["T"] for k in seen}
    ideal = {(k, CZ): rb[f"bs_{k}"]["ideal"] for k in seen}
    solo = {(k, CZ): rd[f"ds_{k}"] for k in seen}
    plats2 = [(g, hw, CZ, keys, fa, alias) for g, hw, cz, keys, fa, alias in plats]
    keep = lambda g, hw: g in ("a1_2s_512-8192", "a1_2s_2048-8192", "a2_2s_mha-ffn") and hw in ("nofault", "F4")
    plats2 = [p for p in plats2 if not p[5] and keep(p[0], p[1])]
    cfgs = D.build_cfgs(2, plats2, T, solo, ideal)
    old = {(c["name"], c["tg"]): c for c in c0}
    for c in cfgs:
        o = old[(c["name"], c["tg"])]
        if c["k"] != o["k"]:
            raise D.DstuError(f"k assignment differs on {c['name']} {c['tg']}")
        c["release"] = dict(o["release"])
        rel, kk = c["release"], c["k"]
        c["d"] = {i: rel[i] + kk[i] * c["T"][i] for i in c["T"]}
        c["dhat"] = {i: rel[i] + kk[i] * c["That"][i] for i in c["T"]}
        c["dr"] = {i: rel[i] + kk[i] * c["Trnd"][i] for i in c["T"]}
    ag = S.run_all(jobs, [(f"ohagg_{k}_{CZ}", {"faults": fa, "plan": ALL, "streams": {"X": k}}) for k, fa in seen.items()], cmd="agg-child")
    agg = {k: ag[f"ohagg_{k}_{CZ}"]["X"] for k in seen}
    # checks
    chk = {}
    tb = {(r["config"], f"kl{float(r['k_long']):g}_ks{float(r['k_short']):g}"): r for r in rcsv(os.path.join(B.OUT, "setting2_table.csv")) if r["policy"] == "P1"}
    bad = 0
    for c in c0:
        r = tb[(c["name"], c["tg"])]
        rr, kk = json.loads(r["release"]), json.loads(r["k"])
        if any(int(rr[str(i)]) != c["release"][i] or abs(float(kk[str(i)]) - c["k"][i]) > 1e-12 for i in c["T"]):
            bad += 1
    chk["stage1_draws_vs_e4b_bad"] = bad
    rep = {(r["config"], r["dl"]): r for r in rcsv(os.path.join(D.OUT, "setting2_report.csv"))}
    hard = sorted(k for k, r in rep.items() if float(r["L_C4"]) - float(r["L_P5"]) > TH)
    chk["hard_forward"] = sum(rep[k]["kind"] == "forward" for k in hard)
    chk["hard_reverse"] = sum(rep[k]["kind"] != "forward" for k in hard)
    chk["stage1_C4_order_bad"] = sum(D.order_of(c["dhat"], c["n"]) != rep[(c["name"], c["tg"])]["C4_order"] for c in c0)
    chk["n_forward"] = sum(r["kind"] == "forward" for r in rep.values())
    chk["n_reverse"] = sum(r["kind"] != "forward" for r in rep.values())
    ac = {(r["stream"], r["chunk"]): r for r in rcsv(os.path.join(D.OUT, "overhead", "agg_cost.csv"))}
    kr = {(r["config"], r["dl"]): r for r in rcsv(os.path.join(D.OUT, "overhead", "key_rows.csv")) if r["set"] == "S2"}
    bad = 0
    for c in c0:
        A = {i: float(ac[(s, c["chunk"] or "none")]["max_occ"]) for i, s in enumerate(c["streams"])}
        By = {i: float(ac[(s, c["chunk"] or "none")]["bytes"]) / 256 for i, s in enumerate(c["streams"])}
        q = kr[(c["name"], c["tg"])]
        if keys_order(c, A) != q["order_agg"] or keys_order(c, By) != q["order_bytes"]:
            bad += 1
    chk["key_order_vs_11_2_bad"] = bad
    chk["agg_vs_11_2_compared"] = sum((k, CZ) in ac for k in agg)
    chk["agg_vs_11_2_bad"] = sum((k, CZ) in ac and (float(ac[(k, CZ)]["max_occ"]) != agg[k]["max_occ"] or float(ac[(k, CZ)]["bytes"]) != agg[k]["bytes"]) for k in agg)
    errs = [abs(solo[(k, CZ)]["T_hat"] / T[(k, CZ)] - 1) for k in seen]
    chk["T_hat_err_median"], chk["T_hat_err_max"] = med(errs), max(errs)
    chk4 = [k for k in seen if any(rd[f"ds_{k}"]["check4"][x] for x in ("bad_monotone", "bad_sink", "bad_block_D", "occupancy_mismatch"))]
    chk["check4_bad"] = chk4
    wc("solo.csv", [{"stream": k, "T": T[(k, CZ)], "T_hat": solo[(k, CZ)]["T_hat"], "err": solo[(k, CZ)]["T_hat"] / T[(k, CZ)] - 1, "max_occ": agg[k]["max_occ"], "bytes": agg[k]["bytes"], "blocks": rb[f"bs_{k}"]["chunk_stats"]["blocks_added"]} for k in seen])
    log(json.dumps(D.round_all(chk), default=str))
    if chk["stage1_draws_vs_e4b_bad"] or chk["hard_forward"] != 1 or chk["hard_reverse"] != 0 or chk["n_forward"] != 108 or chk["n_reverse"] != 48 or chk["stage1_C4_order_bad"] or chk["key_order_vs_11_2_bad"] or chk["agg_vs_11_2_bad"] or chk4 or chk["T_hat_err_median"] > 0.05:
        raise D.DstuError(f"preparation self-check: {chk}")
    return cfgs, agg, solo, ideal, rep, hard, chk


def items_for(cfgs, solo, ideal):
    items = {}
    for c in cfgs:
        nm, tg, spb = c["name"], c["tg"], D.spec_base(c)
        items[f"s2all_{nm}_P1"] = dict(spb, policy={"name": "P1"})
        for o in ORD2:
            items[f"s2all_{nm}_fixed_{o}"] = dict(spb, policy={"name": "fixed", "rank": D.rank_of(o)})
        sd = {i: ideal[(s, CZ)] for i, s in enumerate(c["streams"])}
        tl = {i: D.trim(solo[(s, CZ)]) for i, s in enumerate(c["streams"])}
        items[f"s2all_{nm}_CS_{tg}"] = dict(spb, policy={"ideal": sd, "d_ideal": c["d"], "T_ideal": c["T"], "T_hat": c["That"], "name": "C-S", "tail": tl, "d": c["dhat"]})
    return items


# ------------------------------------------------------------------ 13.1
def pair_block(rows, a, b, by_order):
    sub = [r for r in rows if not by_order or r[f"o_{a}"] != r[f"o_{b}"]]
    d = [r[f"L_{a}"] - r[f"L_{b}"] for r in sub]
    dm = [(r[f"M_{a}"] - r[f"M_{b}"]) / max(r["T_0"], r["T_1"]) for r in sub]
    out = {"n_rows": len(rows), "differ": len(sub) if by_order else "n/a", "better": sum(x < -TH for x in d), "worse": sum(x > TH for x in d), "tie": sum(abs(x) <= TH for x in d)}
    if d:
        out.update(median=med(d), min=min(d), max=max(d), worst_a=max(r[f"L_{a}"] for r in sub), worst_b=max(r[f"L_{b}"] for r in sub), dM_median=med(dm))
    return out


def regret_block(rows, m, fixed=True):
    d = [r[f"L_{m}"] - r["L_P5"] for r in rows]
    out = {"n": len(rows), "median": med(d), "max": max(d), "over": sum(x > TH for x in d), "under": sum(x < -TH for x in d)}
    if fixed:
        out["same_as_best"] = sum(r[f"o_{m}"] == r["o_P5"] for r in rows) / len(rows)
    return out


POLS = ["P1", "P2", "P3", "P4", "P5", "C4", "C-S", "AGG", "BYTES"]


def s2(jobs):
    t0 = time.time()
    cfgs, agg, solo, ideal, rep, hard, chk = prepare(jobs)
    items = items_for(cfgs, solo, ideal)
    log(f"{len(items)} backend processes")
    res = S.run_all(jobs, list(items.items()), cmd="dchild")
    aud = collections.Counter()
    for x in res.values():
        for q in x["runs"]:
            aud["n_issues"] += q["audit"]["n_issues"]
            aud["n_mismatch"] += q["audit"]["n_mismatch"]
            aud["early"] += q["early_events"]
            if q["chunk_stats"] and not (q["chunk_stats"]["bytes_conserved"] and q["chunk_stats"]["tree_conserved"]):
                aud["chunk_bad"] += 1
    tab = D.ref_table(2)
    kr = {(r["config"], r["dl"]): r for r in rcsv(os.path.join(D.OUT, "overhead", "key_rows.csv")) if r["set"] == "S2"}
    rows = []
    for c in cfgs:
        nm, tg, n, Tc, rel, d = c["name"], c["tg"], c["n"], c["T"], c["release"], c["d"]
        st_ = {o: D.stat(res[f"s2all_{nm}_fixed_{o}"]["runs"], c) for o in ORD2}
        best = min(ORD2, key=lambda o: (st_[o]["L"], o))
        o3 = "".join(map(str, sorted(range(n), key=lambda i: (Tc[i], c["streams"][i], i))))
        o4p = D.order_of(d, n)
        o4 = D.order_of(c["dhat"], n)
        A = {i: agg[s]["max_occ"] for i, s in enumerate(c["streams"])}
        By = {i: agg[s]["bytes"] / 256 for i, s in enumerate(c["streams"])}
        oa, ob = keys_order(c, A), keys_order(c, By)
        p1 = D.stat(res[f"s2all_{nm}_P1"]["runs"], c)
        # FCFS as in E4b: at d0 every key ties, so it is P1; otherwise the fixed order of the releases
        o2 = "" if c["order"] == "d0" else "".join(map(str, sorted(range(n), key=lambda i: (rel[i], i))))
        pol = {"P1": p1, "P2": p1 if not o2 else st_[o2], "P3": st_[o3], "P4": st_[o4p], "P5": st_[best], "C4": st_[o4], "C-S": D.stat(res[f"s2all_{nm}_CS_{tg}"]["runs"], c), "AGG": st_[oa], "BYTES": st_[ob]}
        ods = {"P2": o2, "P3": o3, "P4": o4p, "P5": best, "C4": o4, "AGG": oa, "BYTES": ob}
        q = rep[(nm, tg)]
        r = {"config": nm, "dl": tg, "group": c["group"], "fam": "a1" if c["group"].startswith("a1") else "a2", "hw": c["hw"], "order": c["order"], "kind": q["kind"], "k_long": q["k_long"], "k_short": q["k_short"]}
        for p, s in pol.items():
            r[f"L_{p}"], r[f"V_{p}"], r[f"M_{p}"], r[f"Lrange_{p}"] = s["L"], s["V"], s["M"], s["L_range"]
        for p, o in ods.items():
            r[f"o_{p}"] = o
        for i in Tc:
            r[f"T_{i}"], r[f"That_{i}"], r[f"r_{i}"], r[f"k_{i}"], r[f"d_{i}"] = Tc[i], c["That"][i], rel[i], c["k"][i], d[i]
        for o in ORD2:
            r[f"fix_L_{o}"] = st_[o]["L"]
        for p in ("P1", "P2", "P3", "P4", "P5", "C4", "C-S"):
            r[f"s1_L_{p}"], r[f"s1_V_{p}"] = float(q[f"L_{p}"]), float(q[f"V_{p}"])
        r["s1_o_C4"] = q["C4_order"]
        r["s1_o_P5"] = q["P5_member"].replace("fixed_", "")
        for p in ("P1", "P2", "P3", "P4", "P5"):
            r[f"s1_M_{p}"] = tab[(nm, tg, p)]["M"]
        r["s1_M_C4"] = tab[(nm, tg, "fixed_" + q["C4_order"])]["M"]
        with gzip.open(os.path.join(D.SIMS, f"{nm}_CS_{tg}.pkl.gz"), "rb") as f:
            cs = pickle.load(f)["runs"]
        r["s1_M_C-S"] = mean(float(max(x["C"].values())) for x in cs)
        for t2, nmz in (("agg", "AGG"), ("bytes", "BYTES")):
            o = kr[(nm, tg)][f"order_{t2}"]
            r[f"s1_o_{nmz}"] = o
            v = tab[(nm, tg, "fixed_" + o)]
            r[f"s1_L_{nmz}"], r[f"s1_V_{nmz}"], r[f"s1_M_{nmz}"] = v["L"], v["V"], v["M"]
        rows.append(r)
    wc("s2_rows.csv", rows)
    # recheck from the rows
    enum_bad = 0
    for r in rows:
        if abs(min(r["fix_L_01"], r["fix_L_10"]) - r["L_P5"]) > 1e-9:
            enum_bad += 1
        for p in ("P3", "P4", "P5", "C4", "AGG", "BYTES"):
            if abs(r[f"fix_L_{r[f'o_{p}']}"] - r[f"L_{p}"]) > 1e-9:
                enum_bad += 1
    hard_all = [(r["config"], r["dl"]) for r in rows if r["L_C4"] - r["L_P5"] > TH]
    subs = {}
    for kn, kind in (("fwd", "forward"), ("rev", "reverse")):
        base = [r for r in rows if r["kind"] == kind]
        subs[kn] = base
        for fam in ("a1", "a2"):
            subs[f"{kn}_{fam}"] = [r for r in base if r["fam"] == fam]
            for hw in ("nofault", "F4"):
                subs[f"{kn}_{fam}_{hw}"] = [r for r in base if r["fam"] == fam and r["hw"] == hw]
        for g in ("a1_2s_512-8192", "a1_2s_2048-8192"):
            subs[f"{kn}_{g}"] = [r for r in base if r["group"] == g]

    def pol_stats(sub, pre=""):
        out = {}
        for p in POLS:
            L = [r[f"{pre}L_{p}"] for r in sub]
            out[p] = {"L_median": med(L), "L_worst": max(L), "viol_rate": mean(r[f"{pre}V_{p}"] for r in sub) / 2, "rows_any_viol": sum(r[f"{pre}V_{p}"] > 0 for r in sub) / len(sub), "M_median": med(r[f"{pre}M_{p}"] for r in sub)}
        return out
    PR = [("C-S", "C4", False), ("C4", "P3", True), ("C4", "AGG", True), ("C4", "BYTES", True)]
    Sm = {"n": len(rows), "sizes": {k: len(v) for k, v in subs.items()},
          "policies": {k: pol_stats(v) for k, v in subs.items() if v}, "policies_stage1": {k: pol_stats(v, "s1_") for k, v in subs.items() if v},
          "regret": {k: {m: regret_block(v, m, m != "C-S") for m in ("C4", "C-S", "P3", "AGG", "BYTES")} for k, v in subs.items() if v},
          "pairs": {k: {f"{a} - {b}": pair_block(v, a, b, dk) for a, b, dk in PR} for k, v in subs.items() if v},
          "hard_stage1": [list(x) for x in hard], "hard_all_link": [list(x) for x in hard_all],
          "order_counts": {k: {m: dict(collections.Counter(r[f"o_{m}"] for r in v)) for m in ("P3", "P5", "C4", "AGG", "BYTES")} for k, v in subs.items() if k in ("fwd_a1", "fwd_a2", "rev_a1", "rev_a2")},
          "L_range": {p: {"median": med(r[f"Lrange_{p}"] for r in rows), "max": max(r[f"Lrange_{p}"] for r in rows)} for p in ("C4", "C-S", "P5")},
          "checks": dict(chk, **aud, enum_bad=enum_bad),
          "wall": time.time() - t0, "command": sys.argv, "seeds": B.SEEDS, "artifact_commit": E._artifact_commit(), "process_wall_max": max(x["process_wall"] for x in res.values())}
    wj("s2.json", Sm)
    log(f"done {time.time() - t0:.0f} s; checks {Sm['checks']}")
    if aud["n_mismatch"] or aud["early"] or aud["chunk_bad"] or enum_bad:
        raise D.DstuError(f"13.1 self-check {dict(aud)} {enum_bad}")


# ------------------------------------------------------------------ 13.2 plans
def fill_b(order, msgs):
    actb = lambda m, x: S.K64 if msgs[m][0] > S.K64 and m not in x else None
    costb = lambda m, a, x: S.blocks(msgs[m][0], S.K64) - S.blocks(msgs[m][0], S.K256)
    return C.fill(order, msgs, BUD, actb, costb)


def rand_plan(msgs, j):
    rng = random.Random(S.SEL_SEED + j)
    ms = sorted(msgs, key=str)
    rng.shuffle(ms)
    ms.sort(key=lambda m: -msgs[m][0])
    return fill_b(ms, msgs)


def plans_for(jobs):
    cf, solo, sc, plans = OH.prep(jobs)
    istar = {(hw, o): sc[f"ohscore_{hw}_{o}_all256"]["istar"] for hw in cf for o in (0,) + OS2}
    its = {(hw, i) for (hw, o), i in istar.items()}
    sh = S.run_all(jobs, [(f"finshared_{hw}_{i}", {"faults": cf[hw]["faults"], "streams": cf[hw]["streams"], "istar": i}) for hw, i in sorted(its)], cmd="fin-shared")
    P = {}
    for hw in cf:
        for o in (0,) + OS2:
            s = sc[f"ohscore_{hw}_{o}_all256"]
            msgs = s["msgs"]
            P[(hw, o, "static")] = fill_b(C.static_order({"shared": sh[f"finshared_{hw}_{istar[(hw, o)]}"]["shared"], "msgs": msgs}), msgs)
            for j in range(3):
                P[(hw, o, f"rand{j}")] = rand_plan(msgs, j)
            P[(hw, o, "res")] = (plans[(hw, o, "res540_all")][1], plans[(hw, o, "res540_all")][2])
    # o = 0 against 10.3 (plans only; the scores of 10.3 are cached)
    sc2 = S.run_all(jobs, [(f"sc2B_{c['name']}", {"faults": c["faults"], "streams": c["streams"], "release": c["release"], "rank": c["rank"], "T": c["T"], "d": c["d"], "plan": ALL}) for c in cf.values()], cmd="score2-child")
    chk = []
    for hw, c in cf.items():
        s = sc2[f"sc2B_{c['name']}"]
        ref = {"static": fill_b(C.static_order(s), s["msgs"])[0]}
        for j in range(3):
            ref[f"rand{j}"] = rand_plan(s["msgs"], j)[0]
        for tg, ov in ref.items():
            mine = P[(hw, 0, tg)][0]
            chk.append({"hw": hw, "method": tg, "same_plan": mine == ov, "fh": S.fh(mine), "sim_10_3_exists": all(os.path.exists(os.path.join(S.SIMS, f"sB_{c['name']}_{S.fh(mine)}_s{x}.pkl.gz")) for x in B.SEEDS)})
    return cf, solo, sc, plans, P, istar, chk


def sel_items(cf, solo, P):
    items = []
    for hw, c in cf.items():
        base = {"faults": c["faults"], "streams": c["streams"], "release": c["release"]}
        for o in OS2:
            rk = OH.rank_for(c, solo, hw, o, "all256")
            for tg in ("static", "rand0", "rand1", "rand2"):
                items += S.seeded(f"fin_{hw}_{o}_{tg}", dict(base, plan=dict(ALL, override=P[(hw, o, tg)][0]), o=o, policy={"kind": "fixed", "rank": rk}))
    return items


def plan_check(chk):
    log(f"plans vs 10.3: {sum(x['same_plan'] for x in chk)} / {len(chk)} same, sims of 10.3 found {sum(x['sim_10_3_exists'] for x in chk)}")
    if not all(x["same_plan"] and x["sim_10_3_exists"] for x in chk):
        raise D.DstuError(f"plans vs 10.3: {chk}")


def sel(jobs):
    t0 = time.time()
    cf, solo, sc, plans, P, istar, chk = plans_for(jobs)
    plan_check(chk)
    items = sel_items(cf, solo, P)
    log(f"{len(items)} runs")
    if len(items) != 72:
        raise D.DstuError("72 runs expected")
    res = S.run_all(jobs, items, cmd="oh-run")
    t1 = time.time()
    # section 11's runs, exactly as dstu_oh.all_items names and specifies them; cached
    it11, _ = OH.all_items(cf, solo, plans)
    want = [f"ohrun_{hw}_{o}_{m}_C4" for hw in cf for o in OS2 for m in ("all256", "res540_all")]
    keep = [x for x in it11 if x[0].rsplit("_s", 1)[0] in want]
    missing = [x[0] for x in keep if not os.path.exists(os.path.join(S.SIMS, f"{x[0]}.pkl.gz"))]
    if missing:
        raise D.DstuError(f"section 11 runs not cached: {missing}")
    r11 = S.run_all(jobs, keep, cmd="oh-run")
    aud, cons, occ_bad, bud_bad = 0, 0, [], []
    rows, by = [], {}
    for hw, c in cf.items():
        for o in OS2:
            bs = S.runs_of(r11, f"ohrun_{hw}_{o}_all256_C4")
            b0 = OH.stat(bs, c)
            start_extra = b0["extra_chunk"] + o * b0["oh_links"]
            for m in ("all256", "res", "static", "rand0", "rand1", "rand2"):
                nm = {"all256": f"ohrun_{hw}_{o}_all256_C4", "res": f"ohrun_{hw}_{o}_res540_all_C4"}.get(m, f"fin_{hw}_{o}_{m}")
                src = r11 if m in ("all256", "res") else res
                rs = S.runs_of(src, nm)
                for r in rs:
                    aud += r["audit"]["n_mismatch"]
                    cons += not (r["chunk_stats"]["bytes_conserved"] and r["chunk_stats"]["tree_conserved"])
                    dd = r["occ"] - r["occ0"] - o * r["oh_links"]
                    if dd:
                        occ_bad.append((nm, r["seed"], dd))
                st_ = OH.stat(rs, c)
                nb = st_["blocks_total"] - b0["blocks_total"]
                row = {"hw": hw, "o": o, "method": m, "L": st_["L"], "L_range": st_["L_range"], "C_MHA": st_["C_MHA"], "C_FFN": st_["C_FFN"], "M": st_["M"], "new_blocks": nb, "extra_occupancy": st_["extra_chunk"] + o * st_["oh_links"], "extra_vs_all256": st_["extra_chunk"] + o * st_["oh_links"] - start_extra,
                       "L_seeds": [max((r["C"][i] - c["d"][i]) / c["T"][i] for i in c["T"]) for r in rs], "istar": istar[(hw, o)]}
                if m != "all256":
                    ov, u = P[(hw, o, m)]
                    if nb != u or u > BUD:
                        bud_bad.append((hw, o, m, nb, u))
                    j0 = C.jac_over(ov, P[(hw, 0, m)][0])
                    row.update(n_msgs=len(ov), plan_blocks=u, ov_jaccard_o0=j0["jaccard"], ov_common_o0=j0["common"], ov_n_o0=j0["n_b"])
                    if m == "static":
                        jr = C.jac_over(ov, P[(hw, o, "res")][0])
                        row.update(ov_jaccard_res=jr["jaccard"], ov_common_res=jr["common"], ov_n_res=jr["n_b"])
                rows.append(row)
                by[(hw, o, m)] = row
    wc("sel_rows.csv", rows)
    # pairs
    MET = ("L", "M", "C_FFN", "C_MHA")

    def val(hw, o, m):
        if m == "randmean":
            return {k: mean(by[(hw, o, f"rand{j}")][k] for j in range(3)) for k in MET}
        return by[(hw, o, m)]
    PAIRS = [("res", "static"), ("res", "randmean"), ("res", "rand0"), ("res", "rand1"), ("res", "rand2"), ("static", "randmean"), ("static", "rand0"), ("static", "rand1"), ("static", "rand2"),
             ("static", "all256"), ("randmean", "all256"), ("rand0", "all256"), ("rand1", "all256"), ("rand2", "all256"), ("res", "all256")]
    pr = []
    for hw, c in cf.items():
        for o in OS2:
            for a, b in PAIRS:
                x, y = val(hw, o, a), val(hw, o, b)
                dL, dM = x["L"] - y["L"], (x["M"] - y["M"]) / c["T"][1]
                pr.append({"hw": hw, "o": o, "pair": f"{a} - {b}", "dL": dL, "dL_sign": OH.sign(dL), "dM_over_TF": dM, "dM_sign": OH.sign(dM), "dCF_over_TF": (x["C_FFN"] - y["C_FFN"]) / c["T"][1], "dCM_over_TM": (x["C_MHA"] - y["C_MHA"]) / c["T"][0], "L_a": x["L"], "L_b": y["L"], "M_a": x["M"], "M_b": y["M"]})
    wc("sel_pairs.csv", pr)
    out = {"audit_mismatch": aud, "conservation_bad": cons, "occupancy_bad": occ_bad[:20], "occupancy_bad_n": len(occ_bad), "budget_bad": bud_bad, "plans_vs_10_3": chk,
           "istar": {f"{hw}_{o}": istar[(hw, o)] for hw in cf for o in (0,) + OS2}, "rank": {f"{hw}_{o}": OH.rank_for(cf[hw], solo, hw, o, "all256") for hw in cf for o in OS2},
           "T": {hw: cf[hw]["T"] for hw in cf}, "d": {hw: cf[hw]["d"] for hw in cf}, "wall": time.time() - t0, "wall_runs": t1 - t0, "process_wall_max": max(x["process_wall"] for x in res.values()),
           "command": sys.argv, "seeds": B.SEEDS, "artifact_commit": E._artifact_commit()}
    wj("sel.json", out)
    log(json.dumps({k: out[k] for k in ("audit_mismatch", "conservation_bad", "occupancy_bad_n", "budget_bad", "istar", "wall")}, default=str))
    if aud or cons or occ_bad or bud_bad:
        raise D.DstuError("13.2 self-check")


# ------------------------------------------------------------------ probe
def check(jobs):
    prepare(jobs)
    cf, solo, sc, plans, P, istar, chk = plans_for(jobs)
    plan_check(chk)
    log(f"i*: { {f'{h}_{o}': v for (h, o), v in istar.items()} }")
    for (hw, o, m), (ov, u) in sorted(P.items(), key=str):
        log(f"{hw} {o} {m}: {len(ov)} messages, {u} blocks, fh {S.fh(ov)}")


def probe(jobs):
    t0 = time.time()
    cfgs, agg, solo, ideal, rep, hard, chk = prepare(jobs)
    its = items_for(cfgs, solo, ideal)
    cf, solo2, sc, plans, P, istar, pchk = plans_for(jobs)
    plan_check(pchk)
    t1 = time.time()
    first = {}
    for c in cfgs:
        first.setdefault(c["name"], c["tg"])
    pick = {}
    for nm, tg in first.items():
        if ("2048-8192" in nm or "mha-ffn" in nm):
            pick[f"s2all_{nm}_CS_{tg}"] = its[f"s2all_{nm}_CS_{tg}"]
            for o in ORD2:
                pick[f"s2all_{nm}_fixed_{o}"] = its[f"s2all_{nm}_fixed_{o}"]
            pick[f"s2all_{nm}_P1"] = its[f"s2all_{nm}_P1"]
    r = S.run_all(jobs, list(pick.items()), cmd="dchild")
    t2 = time.time()
    sit = sel_items(cf, solo2, P)
    sp = [x for x in sit if x[0].endswith("_s1") and ("_200_static" in x[0] or "_50_rand0" in x[0])]
    q = S.run_all(jobs, sp, cmd="oh-run")
    t3 = time.time()
    by = collections.defaultdict(list)
    for n, x in r.items():
        by[("a1" if "_a1_" in n else "a2", "CS" if "_CS_" in n else "fixed")].append(x["process_wall"])
    tot = {("a1", "CS"): 8 * 13, ("a1", "fixed"): 8 * 3, ("a2", "CS"): 4 * 13, ("a2", "fixed"): 4 * 3}
    est = sum(mean(by[k]) * n for k, n in tot.items())
    sw = [x["process_wall"] for x in q.values()]
    out = {"prep_wall": t1 - t0, "backend_wall": t2 - t1, "sel_wall": t3 - t2, "processes": len(pick), "by": {str(k): {"mean": mean(v), "max": max(v), "n": len(v)} for k, v in by.items()},
           "sel_process_mean": mean(sw), "sel_process_max": max(sw), "sel_n": len(sw), "est_s2_process_h": est / 3600, "est_s2_wall_min": est / jobs / 60, "est_sel_process_h": mean(sw) * 72 / 3600, "est_sel_wall_min": mean(sw) * 72 / jobs / 60}
    wj("probe.json", out)
    log(json.dumps(D.round_all(out), default=str))


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
    ap.add_argument("cmd", choices=["check", "probe", "s2", "sel"])
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    {"check": check, "probe": probe, "s2": s2, "sel": sel}[a.cmd](a.jobs)


if __name__ == "__main__":
    try:
        main()
    except (D.DstuError, B.E4bError, E.E4Error, cpcm.CPCMError) as exc:
        print(f"[dstu_final] STOP: {exc}", flush=True)
        sys.exit(2)
