"""DSTU all-link setting 3 and the unchunked reference (dstu/DSTU_HANDOFF.md section 12). Sources elsewhere unchanged.

  python dstu_s3all.py solo --jobs 14    all-link solo data and aggregate keys of the setting 3 streams; self-checks that need no runs
  python dstu_s3all.py probe --jobs 14   draw 0 of each group x hardware, walls (reused by the full run)
  python dstu_s3all.py run --jobs 14     12.1
  python dstu_s3all.py ns --jobs 14      12.2
Results in dstu/out/setting3_all/; simulations cached in dstu/out/stage2/sims (names start with s3all_, s3sim_, ns).
"""
import argparse
import collections
import csv
import gzip
import itertools
import json
import os
import pickle
import statistics
import sys
import time

import dstu as D
import dstu2 as S
import dstu2c as C
import dstu_oh as OH
from dstu import B, E, cpcm

S.SCRIPT = os.path.abspath(__file__)
OUT = os.path.join(D.OUT, "setting3_all")
ALL = S.UNI["256all"]
CZ = "256all"
TH = S.TH
NONE_PLAN = {"size": 10 ** 18, "only_d2d": False}
E4A_HOP = os.path.join(D.HERE, "..", "e4_overlap", "e4_out", "hop")
med = statistics.median


def log(m):
    print(f"[dstu_s3all] {m}", flush=True)


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


def orders(n):
    return ["".join(map(str, p)) for p in itertools.permutations(range(n))]


# ------------------------------------------------------------------ child
def sim_child(spec):
    """CPCM forward on the merged graph for every fixed order of one row."""
    E._artifact_path()
    hp0 = E.build_platform(E.CFG, spec["faults"])
    t0 = time.perf_counter()
    ev, _ = D._events(spec["streams"], ALL, hp0)
    g = D._graph(ev, hp0)
    st = [g.tag[e][1] - 10 for e in range(g.n)]
    rel = {e: spec["release"][st[e]] for e in range(g.n) if g.pred_ptr[e + 1] == g.pred_ptr[e]}
    t_compile = time.perf_counter() - t0
    out, tm = {}, []
    for o in orders(len(spec["streams"])):
        rk = D.rank_of(o)
        t1 = time.perf_counter()
        sc = cpcm.simulate(g, "hop", prio=[rk[s] for s in st], release=rel)
        tm.append(time.perf_counter() - t1)
        Cc = collections.defaultdict(int)
        for e in range(g.n):
            Cc[st[e]] = max(Cc[st[e]], sc.F[e])
        out[o] = {int(k): float(v) for k, v in Cc.items()}
    return {"C": out, "compile_s": t_compile, "forward_s": tm, "n_events": g.n}


CHILDREN = dict(OH.CHILDREN, **{"s3sim": sim_child})


# ------------------------------------------------------------------ configuration
def keys_order(c, val):
    return D.order_of({i: c["release"][i] + c["k"][i] * val[i] for i in c["T"]}, c["n"])


def prepare(jobs):
    """Stage 1 rows, all-link rows with stage 1's releases, aggregate keys; self-checks that need no runs."""
    _, plats3, T3 = D.load_s3()
    solo3 = {}
    for g, hw, cz, keys, fa, alias in plats3:
        for k in keys:
            with gzip.open(os.path.join(D.SOLO, f"{k}_{cz or 'none'}.pkl.gz"), "rb") as f:
                solo3[(k, cz)] = pickle.load(f)
    c0 = D.build_cfgs(3, plats3, T3, solo3, {})
    seen = {}
    for g, hw, cz, keys, fa, alias in plats3:
        for k in keys:
            seen[k] = fa
    rb = S.run_all(jobs, [(f"bs_{k}", {"stream": k, "faults": fa, "chunk": ALL}) for k, fa in seen.items()], cmd="bsolo-child")
    rd = S.run_all(jobs, [(f"ds_{k}", {"stream": k, "faults": fa, "chunk": ALL}) for k, fa in seen.items()], cmd="dsolo")
    T = {(k, CZ): rb[f"bs_{k}"]["T"] for k in seen}
    ideal = {(k, CZ): rb[f"bs_{k}"]["ideal"] for k in seen}
    solo = {(k, CZ): rd[f"ds_{k}"] for k in seen}
    plats = [(g, hw, CZ, keys, fa, alias) for g, hw, cz, keys, fa, alias in plats3]
    cfgs = D.build_cfgs(3, plats, T, solo, ideal)
    old = {c["name"]: c for c in c0}
    for c in cfgs:
        o = old[c["name"]]
        if c["k"] != o["k"]:
            raise D.DstuError(f"draws differ on {c['name']}")
        c["release"] = dict(o["release"])
        rel, kk = c["release"], c["k"]
        c["d"] = {i: rel[i] + kk[i] * c["T"][i] for i in c["T"]}
        c["dhat"] = {i: rel[i] + kk[i] * c["That"][i] for i in c["T"]}
        c["dr"] = {i: rel[i] + kk[i] * c["Trnd"][i] for i in c["T"]}
    ag = S.run_all(jobs, [(f"ohagg_{k}_{CZ}", {"faults": fa, "plan": ALL, "streams": {"X": k}}) for k, fa in seen.items()], cmd="agg-child")
    agg = {k: ag[f"ohagg_{k}_{CZ}"]["X"] for k in seen}
    # checks
    chk = {}
    tb = {r["config"]: r for r in rcsv(os.path.join(B.OUT, "setting3_table.csv")) if r["policy"] == "P1"}
    bad = 0
    for c in c0:
        r = tb[c["name"]]
        rr, kk = json.loads(r["release"]), json.loads(r["k"])
        if any(int(rr[str(i)]) != c["release"][i] or abs(float(kk[str(i)]) - c["k"][i]) > 1e-12 for i in c["T"]):
            bad += 1
    chk["stage1_draws_vs_e4b_bad"] = bad
    rep = {r["config"]: r for r in rcsv(os.path.join(D.OUT, "setting3_report.csv"))}
    hard_b = sorted(n for n, r in rep.items() if float(r["L_C4"]) - float(r["L_P5"]) > TH)
    chk["hard_b_n"] = len(hard_b)
    chk["stage1_C4_order_bad"] = sum(D.order_of(c["dhat"], c["n"]) != rep[c["name"]]["C4_order"] for c in c0)
    # aggregate keys against 11.2
    ac = {(r["stream"], r["chunk"]): r for r in rcsv(os.path.join(D.OUT, "overhead", "agg_cost.csv"))}
    kr = {r["config"]: r for r in rcsv(os.path.join(D.OUT, "overhead", "key_rows.csv")) if r["set"] == "S3"}
    bad = 0
    for c in c0:
        A = {i: float(ac[(s, c["chunk"] or "none")]["max_occ"]) for i, s in enumerate(c["streams"])}
        By = {i: float(ac[(s, c["chunk"] or "none")]["bytes"]) / 256 for i, s in enumerate(c["streams"])}
        if keys_order(c, A) != kr[c["name"]]["order_agg"] or keys_order(c, By) != kr[c["name"]]["order_bytes"]:
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
    if chk["stage1_draws_vs_e4b_bad"] or chk["hard_b_n"] != 13 or chk["stage1_C4_order_bad"] or chk["key_order_vs_11_2_bad"] or chk["agg_vs_11_2_bad"] or chk4 or chk["T_hat_err_median"] > 0.05:
        raise D.DstuError(f"preparation self-check: {chk}")
    return c0, cfgs, agg, solo, ideal, hard_b, chk


def items_for(cfgs, solo, ideal):
    items, sims = {}, {}
    for c in cfgs:
        nm, spb = c["name"], D.spec_base(c)
        items[f"s3all_{nm}_P1"] = dict(spb, policy={"name": "P1"})
        for o in orders(c["n"]):
            items[f"s3all_{nm}_fixed_{o}"] = dict(spb, policy={"name": "fixed", "rank": D.rank_of(o)})
        sd = {i: ideal[(s, CZ)] for i, s in enumerate(c["streams"])}
        tl = {i: D.trim(solo[(s, CZ)]) for i, s in enumerate(c["streams"])}
        items[f"s3all_{nm}_CS"] = dict(spb, policy={"ideal": sd, "d_ideal": c["d"], "T_ideal": c["T"], "T_hat": c["That"], "name": "C-S", "tail": tl, "d": c["dhat"]})
        sims[f"s3sim_{nm}"] = {"faults": c["faults"], "streams": c["streams"], "release": c["release"]}
    return items, sims


def probe(jobs):
    t0 = time.time()
    c0, cfgs, agg, solo, ideal, hb, chk = prepare(jobs)
    t1 = time.time()
    sub = [c for c in cfgs if c["label"]["draw"] == 0]
    items, sims = items_for(sub, solo, ideal)
    r = S.run_all(jobs, list(items.items()), cmd="dchild")
    t2 = time.time()
    q = S.run_all(jobs, list(sims.items()), cmd="s3sim")
    t3 = time.time()
    pw = {n: x["process_wall"] for n, x in r.items()}
    by = collections.defaultdict(list)
    for n, v in pw.items():
        by[("a1" if "_a1_" in n else "a2", "C-S" if n.endswith("_CS") else "fixed/P1")].append(v)
    out = {"prep_wall": t1 - t0, "probe_backend_wall": t2 - t1, "probe_sim_wall": t3 - t2, "processes": len(items), "sum_process_wall": sum(pw.values()), "max_process_wall": max(pw.values()), "max_at": max(pw, key=pw.get),
           "process_wall_by": {str(k): {"mean": statistics.mean(v), "max": max(v)} for k, v in by.items()}, "sim_process_wall": {n: x["process_wall"] for n, x in q.items()}, "forward_s_max": max(t for x in q.values() for t in x["forward_s"])}
    per_draw = (out["sum_process_wall"] + sum(x["process_wall"] for x in q.values())) / 4
    out["est_full_process_h"] = per_draw * 80 / 3600
    out["est_wall_min_14jobs"] = per_draw * 80 / 14 / 60
    wj("probe.json", out)
    log(json.dumps(D.round_all(out), default=str))


# ------------------------------------------------------------------ 12.1
def pair_block(rows, a, b, by_order):
    """dL = a - b on the rows where the orders differ (all rows when the pair has no order to compare)."""
    sub = [r for r in rows if not by_order or r[f"o_{a}"] != r[f"o_{b}"]]
    d = [r[f"L_{a}"] - r[f"L_{b}"] for r in sub]
    dm = [(r[f"M_{a}"] - r[f"M_{b}"]) / max(r[f"T_{i}"] for i in range(3)) for r in sub]
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


POLS = ["P1", "P2", "P3", "P4", "P5", "C4", "C-S", "CP_SIM", "AGG", "BYTES"]


def run(jobs):
    t0 = time.time()
    c0, cfgs, agg, solo, ideal, hard_b, chk = prepare(jobs)
    items, sims = items_for(cfgs, solo, ideal)
    log(f"{len(items)} backend processes, {len(sims)} CPCM processes")
    res = S.run_all(jobs, list(items.items()), cmd="dchild")
    t1 = time.time()
    sim = S.run_all(jobs, list(sims.items()), cmd="s3sim")
    aud = collections.Counter()
    for x in res.values():
        for q in x["runs"]:
            aud["n_issues"] += q["audit"]["n_issues"]
            aud["n_mismatch"] += q["audit"]["n_mismatch"]
            aud["early"] += q["early_events"]
            if q["chunk_stats"] and not (q["chunk_stats"]["bytes_conserved"] and q["chunk_stats"]["tree_conserved"]):
                aud["chunk_bad"] += 1
    rows, pred_all, order_bad, tied_rows, same_T = [], [], 0, 0, 0
    for c in cfgs:
        nm, n, Tc, rel, d = c["name"], c["n"], c["T"], c["release"], c["d"]
        ks = orders(n)
        st_ = {o: D.stat(res[f"s3all_{nm}_fixed_{o}"]["runs"], c) for o in ks}
        best = min(ks, key=lambda o: (st_[o]["L"], o))
        o2 = "".join(map(str, sorted(range(n), key=lambda i: (rel[i], i))))
        o3 = "".join(map(str, sorted(range(n), key=lambda i: (Tc[i], c["streams"][i], i))))
        tie_T = len(set(Tc.values())) < n
        same_T += tie_T
        o4p = D.order_of(d, n)
        o4 = D.order_of(c["dhat"], n)
        A = {i: agg[s]["max_occ"] for i, s in enumerate(c["streams"])}
        By = {i: agg[s]["bytes"] / 256 for i, s in enumerate(c["streams"])}
        oa, ob = keys_order(c, A), keys_order(c, By)
        sm = sim[f"s3sim_{nm}"]
        pred = {o: max((sm["C"][o][i] - d[i]) / Tc[i] for i in Tc) for o in ks}
        bp = min(pred.values())
        tied = sorted(o for o, v in pred.items() if v == bp)
        tied_rows += len(tied) > 1
        ocp = o4 if o4 in tied else tied[0]
        if ocp not in ks:
            order_bad += 1
        for o in ks:
            pred_all.append(abs(pred[o] - st_[o]["L"]))
        pol = {"P1": D.stat(res[f"s3all_{nm}_P1"]["runs"], c), "P2": st_[o2], "P3": st_[o3], "P4": st_[o4p], "P5": st_[best], "C4": st_[o4], "C-S": D.stat(res[f"s3all_{nm}_CS"]["runs"], c), "CP_SIM": st_[ocp], "AGG": st_[oa], "BYTES": st_[ob]}
        ods = {"P2": o2, "P3": o3, "P4": o4p, "P5": best, "C4": o4, "CP_SIM": ocp, "AGG": oa, "BYTES": ob}
        r = {"config": nm, "group": c["group"], "fam": "a1" if c["group"].startswith("a1") else "a2", "hw": c["hw"], "draw": c["label"]["draw"], "tie_T": tie_T, "pred_CP_SIM": bp, "n_tied_pred": len(tied), "err_pred": abs(bp - st_[ocp]["L"]),
             "compile_s": sm["compile_s"], "forward_s": sum(sm["forward_s"]), "forward_s_max": max(sm["forward_s"]), "n_events": sm["n_events"]}
        for p, s in pol.items():
            r[f"L_{p}"], r[f"V_{p}"], r[f"M_{p}"], r[f"Lrange_{p}"] = s["L"], s["V"], s["M"], s["L_range"]
        for p, o in ods.items():
            r[f"o_{p}"] = o
        for i in Tc:
            r[f"T_{i}"], r[f"That_{i}"], r[f"r_{i}"], r[f"k_{i}"], r[f"d_{i}"] = Tc[i], c["That"][i], rel[i], c["k"][i], d[i]
        for o in ks:
            r[f"fix_L_{o}"], r[f"pred_L_{o}"] = st_[o]["L"], pred[o]
        rows.append(r)
    # stage 1 values
    rep = {r["config"]: r for r in rcsv(os.path.join(D.OUT, "setting3_report.csv"))}
    tab = D.ref_table(3)
    kr = {r["config"]: r for r in rcsv(os.path.join(D.OUT, "overhead", "key_rows.csv")) if r["set"] == "S3"}
    for r in rows:
        q, nm = rep[r["config"]], r["config"]
        for p in ("P1", "P2", "P3", "P4", "P5", "C4", "C-S"):
            r[f"s1_L_{p}"], r[f"s1_V_{p}"] = float(q[f"L_{p}"]), float(q[f"V_{p}"])
        r["s1_o_C4"] = q["C4_order"]
        r["s1_o_P5"] = q["P5_member"].replace("fixed_", "")
        for p in ("P1", "P2", "P3", "P4", "P5"):
            r[f"s1_M_{p}"] = tab[(nm, "s3", p)]["M"]
        r["s1_M_C4"] = tab[(nm, "s3", "fixed_" + q["C4_order"])]["M"]
        with gzip.open(os.path.join(D.SIMS, f"{nm}_CS_s3.pkl.gz"), "rb") as f:
            cs = pickle.load(f)["runs"]
        r["s1_M_C-S"] = statistics.mean(float(max(x["C"].values())) for x in cs)
        for tg, nmz in (("agg", "AGG"), ("bytes", "BYTES")):
            o = kr[nm][f"order_{tg}"]
            r[f"s1_o_{nmz}"] = o
            v = tab[(nm, "s3", "fixed_" + o)]
            r[f"s1_L_{nmz}"], r[f"s1_V_{nmz}"], r[f"s1_M_{nmz}"] = v["L"], v["V"], v["M"]
    wc("rows.csv", rows)
    hard_c = [r["config"] for r in rows if r["L_C4"] - r["L_P5"] > TH]
    subsets = {"all": rows, "a1": [r for r in rows if r["fam"] == "a1"], "a2": [r for r in rows if r["fam"] == "a2"], "a": [r for r in rows if r["fam"] == "a2"], "b": [r for r in rows if r["config"] in hard_b], "c": [r for r in rows if r["config"] in hard_c]}
    for hw in ("nofault", "F4"):
        subsets[f"a1_{hw}"] = [r for r in rows if r["fam"] == "a1" and r["hw"] == hw]
        subsets[f"a2_{hw}"] = [r for r in rows if r["fam"] == "a2" and r["hw"] == hw]

    def pol_stats(sub, pre=""):
        out = {}
        for p in POLS:
            if pre and p == "CP_SIM":
                continue
            L = [r[f"{pre}L_{p}"] for r in sub]
            out[p] = {"L_median": med(L), "L_worst": max(L), "viol_rate": statistics.mean(r[f"{pre}V_{p}"] for r in sub) / 3, "rows_any_viol": sum(r[f"{pre}V_{p}"] > 0 for r in sub) / len(sub), "M_median": med(r[f"{pre}M_{p}"] for r in sub)}
        return out
    Sm = {"n": len(rows), "policies": {k: pol_stats(subsets[k]) for k in ("all", "a1", "a2")}, "policies_stage1": {k: pol_stats(subsets[k], "s1_") for k in ("all", "a1", "a2")}}
    Sm["policies_hw"] = {k: pol_stats(subsets[k]) for k in ("a1_nofault", "a1_F4", "a2_nofault", "a2_F4")}
    Sm["regret"] = {k: {m: regret_block(v, m, m != "C-S") for m in ("CP_SIM", "C4", "C-S", "P3", "AGG", "BYTES")} for k, v in subsets.items() if v}
    PR = [("CP_SIM", "C4", True), ("CP_SIM", "P3", True), ("CP_SIM", "AGG", True), ("CP_SIM", "BYTES", True), ("C-S", "C4", False), ("C-S", "P3", False), ("C-S", "CP_SIM", False)]
    Sm["pairs"] = {k: {f"{a} - {b}": pair_block(v, a, b, dk) for a, b, dk in PR} for k, v in subsets.items() if v}
    Sm["subsets"] = {"b": hard_b, "c": hard_c, "b_and_c": sorted(set(hard_b) & set(hard_c)), "sizes": {k: len(v) for k, v in subsets.items()}}
    Sm["pred"] = {"chosen": {"median": med(r["err_pred"] for r in rows), "max": max(r["err_pred"] for r in rows)}, "all_orders": {"n": len(pred_all), "median": med(pred_all), "max": max(pred_all)}, "tied_rows": tied_rows,
                  "a1": {"median": med(r["err_pred"] for r in subsets["a1"]), "max": max(r["err_pred"] for r in subsets["a1"])}, "a2": {"median": med(r["err_pred"] for r in subsets["a2"]), "max": max(r["err_pred"] for r in subsets["a2"])},
                  "cpcm_per_row": {"compile": 1, "forward": 6}, "compile_s": {k: {"median": med(r["compile_s"] for r in subsets[k]), "max": max(r["compile_s"] for r in subsets[k])} for k in ("a1", "a2")},
                  "forward_6_s": {k: {"median": med(r["forward_s"] for r in subsets[k]), "max": max(r["forward_s"] for r in subsets[k])} for k in ("a1", "a2")}, "n_events": {k: sorted({r["n_events"] for r in subsets[k]}) for k in ("a1", "a2")}}
    Sm["same_T_rows"] = same_T
    Sm["same_T_rows_by_fam"] = {k: sum(r["tie_T"] for r in subsets[k]) for k in ("a1", "a2")}
    Sm["order_counts_a2"] = {m: dict(collections.Counter(r[f"o_{m}"] for r in subsets["a2"])) for m in ("P3", "AGG", "BYTES", "C4", "CP_SIM", "P5")}
    Sm["order_counts_a1"] = {m: dict(collections.Counter(r[f"o_{m}"] for r in subsets["a1"])) for m in ("P3", "AGG", "BYTES", "C4", "CP_SIM", "P5")}
    Sm["checks"] = dict(chk, **aud, cp_sim_order_bad=order_bad)
    Sm.update(wall=time.time() - t0, wall_backend=t1 - t0, command=sys.argv, seeds=B.SEEDS, sample_seed="20261103 + 100 g + j", artifact_commit=E._artifact_commit(), process_wall_max=max(x["process_wall"] for x in res.values()))
    wj("setting3_all.json", Sm)
    log(f"done {time.time() - t0:.0f} s; checks {Sm['checks']}")
    if aud["n_mismatch"] or aud["early"] or aud["chunk_bad"] or order_bad:
        raise D.DstuError(f"12.1 self-check {dict(aud)} {order_bad}")


# ------------------------------------------------------------------ 12.2
def ns(jobs):
    t0 = time.time()
    cf = OH.cfgs_d0()
    so = S.run_all(jobs, [(f"nssolo_{hw}_{i}", {"faults": c["faults"], "stream": k, "plan": NONE_PLAN, "o": 0}) for hw, c in cf.items() for i, k in enumerate(c["streams"])], cmd="oh-solo")
    items, rk = [], {}
    for hw, c in cf.items():
        rk[hw] = D.rank_of(D.order_of({i: c["release"][i] + OH.K_ * so[f"nssolo_{hw}_{i}"]["T_hat"] for i in range(2)}, 2))
        items += S.seeded(f"nsrun_{hw}", {"faults": c["faults"], "streams": c["streams"], "release": c["release"], "plan": NONE_PLAN, "o": 0, "policy": {"kind": "fixed", "rank": rk[hw]}})
    res = S.run_all(jobs, items, cmd="oh-run")
    aud, bad, ref = 0, [], []
    NS = {}
    for hw, c in cf.items():
        rs = S.runs_of(res, f"nsrun_{hw}")
        for r in rs:
            aud += r["audit"]["n_mismatch"]
            if r["oh_links"] or r["chunk_stats"]["blocks_added"] or r["occ"] != r["occ0"]:
                bad.append((hw, r["seed"], r["oh_links"], r["chunk_stats"]["blocks_added"], r["occ"], r["occ0"]))
        NS[hw] = OH.stat(rs, c)
        NS[hw].update(T_hat=[so[f"nssolo_{hw}_{i}"]["T_hat"] for i in range(2)], rank=rk[hw], T=c["T"], d=c["d"], L_seeds=[max((r["C"][i] - c["d"][i]) / c["T"][i] for i in c["T"]) for r in rs])
        # E4a per-hop result, same configuration
        fn = "a2_2s_mha-ffn_d0_hop.json" if hw == "nofault" else f"a2_2s_mha-ffn_{hw}_d0_hop.json"
        with open(os.path.join(E4A_HOP, fn), encoding="utf-8") as f:
            h = json.load(f)
        o = "".join(map(str, sorted(range(2), key=lambda i: rk[hw][i])))
        v = h["per_variant"][f"prio_{o}"]["latency_seeds"]
        ref.append({"hw": hw, "order": o, "rank": str(rk[hw]), "same_seeds": sum(all(float(v[str(i)][j]) == rs[j]["C"][i] for i in range(2)) for j in range(3)),
                    **{f"rel_C{i}": statistics.mean(rs[j]["C"][i] for j in range(3)) / statistics.mean(v[str(i)]) - 1 for i in range(2)},
                    **{f"new_C{i}": statistics.mean(rs[j]["C"][i] for j in range(3)) for i in range(2)}, **{f"e4a_C{i}": statistics.mean(v[str(i)]) for i in range(2)}})
    wc("ns_vs_e4a.csv", ref)
    idx = {(r["hw"], int(r["o"]), r["scheme"], r["policy"]): r for r in rcsv(os.path.join(D.OUT, "overhead", "rows.csv"))}
    SCH = ("d2d256", "all256", "all64", "res540_d2d", "res540_all", "res1620_all")
    pairs, flips = [], []
    for hw, c in cf.items():
        TF = c["T"][1]
        n = NS[hw]
        for sn in SCH:
            sq = {"L": [], "M": [], "CF": []}
            for o in OH.OS_:
                x = idx[(hw, o, sn, "C4")]
                dL, dM, dCF = float(x["L"]) - n["L"], (float(x["M"]) - n["M"]) / TF, (float(x["C_FFN"]) - n["C_FFN"]) / TF
                sq["L"].append(dL)
                sq["M"].append(dM)
                sq["CF"].append(dCF)
                pairs.append({"hw": hw, "scheme": sn, "o": o, "L": float(x["L"]), "C_MHA": float(x["C_MHA"]), "C_FFN": float(x["C_FFN"]), "M": float(x["M"]), "dL": dL, "dL_sign": OH.sign(dL), "dM_over_TF": dM, "dM_sign": OH.sign(dM), "dCF_over_TF": dCF, "dCMHA_over_TMHA": (float(x["C_MHA"]) - n["C_MHA"]) / c["T"][0]})
            for m, s in sq.items():
                sg = [OH.sign(v) for v in s]
                for i in range(3):
                    if sg[i] != sg[i + 1]:
                        flips.append({"hw": hw, "scheme": sn, "metric": m, "o_low": OH.OS_[i], "o_high": OH.OS_[i + 1], "val_low": s[i], "val_high": s[i + 1], "sign_low": sg[i], "sign_high": sg[i + 1], "full_reversal": sg[i] * sg[i + 1] < 0})
    wc("ns_pairs.csv", pairs)
    wc("ns_flips.csv", flips)
    out = {"unchunked": NS, "audit_mismatch": aud, "occupancy_or_block_bad": bad, "vs_e4a": ref, "wall": time.time() - t0, "command": sys.argv, "seeds": B.SEEDS, "artifact_commit": E._artifact_commit()}
    wj("ns.json", out)
    log(json.dumps(D.round_all(out), default=str))
    if aud or bad or any(abs(r["rel_C0"]) > 0.04 or abs(r["rel_C1"]) > 0.04 for r in ref):
        raise D.DstuError("12.2 self-check")


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
    ap.add_argument("cmd", choices=["solo", "probe", "run", "ns"])
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    {"solo": prepare, "probe": probe, "run": run, "ns": ns}[a.cmd](a.jobs)


if __name__ == "__main__":
    try:
        main()
    except (D.DstuError, B.E4bError, E.E4Error, cpcm.CPCMError) as exc:
        print(f"[dstu_s3all] STOP: {exc}", flush=True)
        sys.exit(2)
