"""DSTU block overhead sensitivity and aggregate key check (dstu/DSTU_HANDOFF.md section 11). Sources elsewhere unchanged.

Overhead: from the 2nd block of an original message on (tag id >= CHUNK_ID_BASE), every link the block crosses is occupied o
cycles longer. In memory: such a block's comm_bytes is an int subclass carrying .oh, and the link classes' working_time adds
it, so the backend (record_issue), CPCM (compile_bb) and DstuE.occ all use the same accounting. o = 0 applies nothing.

  python dstu_oh.py prep --jobs 14   CPCM solo and scores on every o; selected plans; overlap with o = 0
  python dstu_oh.py probe --jobs 14  walls of the slowest runs
  python dstu_oh.py run --jobs 14    11.1
  python dstu_oh.py key --jobs 14    11.2 (no backend runs)
Results in dstu/out/overhead/; simulations cached in dstu/out/stage2/sims (names start with oh).
"""
import argparse
import collections
import copy
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
from dstu import B, E, cpcm

S.SCRIPT = os.path.abspath(__file__)
OUT = os.path.join(D.OUT, "overhead")
OS_ = (0, 50, 200, 1000)
HWS = S.HWS
TH = S.TH
K_ = 1.5
CB = E.CHUNK_ID_BASE
SCHEMES = {"d2d256": S.UNI["256d2d"], "all256": S.UNI["256all"], "all64": S.UNI["64all"]}


def log(m):
    print(f"[dstu_oh] {m}", flush=True)


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


class OhInt(int):
    def __new__(cls, v, oh):
        x = super().__new__(cls, v)
        x.oh = oh
        return x


def patch_links(hp):
    for cls in {type(l) for l in hp.links_dict.values()}:
        if "_oh_patched" in cls.__dict__:
            continue
        orig = cls.working_time

        def wt(self, data_bytes, _orig=orig):
            return _orig(self, data_bytes) + getattr(data_bytes, "oh", 0)
        cls.working_time = wt
        cls._oh_patched = True


def build(blobs, hp0, plan, o):
    """Unchunked -> chunked -> overhead. Returns events, chunk stats, number of (block from the 2nd on x link)."""
    ev, cst = S.chunk_by(E.merged_events(blobs), hp0, plan)
    n = 0
    for t, e in ev.items():
        if e.event_type == "communication" and t[-1] >= CB:
            n += sum(len(v) for v in e.paths.values())
            if o:
                e.comm_bytes = OhInt(int(e.comm_bytes), o)
    return ev, cst, n


# ------------------------------------------------------------------ children
def solo_child(spec):
    """T^ and tail^ of one stream on a chunking and o (CPCM solo, stage 1's C-S data)."""
    E._artifact_path()
    hp0 = E.build_platform(E.CFG, spec["faults"])
    patch_links(hp0)
    ev, cst, n = build([B.load_stream(spec["stream"])["events_blob"]], hp0, spec["plan"], spec["o"])
    g = D._graph(ev, hp0)
    sc = cpcm.simulate(g, "hop")
    an = cpcm.Analysis(sc, sink_deadline=[sc.T] * g.n)
    data = D.tail_data(g, sc, an, sc.T, range(g.n))
    return dict(D.trim(data), T_hat=sc.T, n_events=g.n, oh_links=n)


def score_child(spec):
    E._artifact_path()
    hp0 = E.build_platform(E.CFG, spec["faults"])
    patch_links(hp0)
    blobs = [B.load_stream(s)["events_blob"] for s in spec["streams"]]
    msgs = S.msg_table(E.merged_events(blobs), hp0)
    ev, cst, n = build(blobs, hp0, spec["plan"], spec["o"])
    g = D._graph(ev, hp0)
    st = [g.tag[e][1] - 10 for e in range(g.n)]
    rel = {e: spec["release"][st[e]] for e in range(g.n) if g.pred_ptr[e + 1] == g.pred_ptr[e]}
    rank = spec["rank"]
    sc = cpcm.simulate(g, "hop", prio=[rank[s] for s in st], release=rel)
    Cc = collections.defaultdict(int)
    for e in range(g.n):
        Cc[st[e]] = max(Cc[st[e]], sc.F[e])
    J, istar = S.J_of(dict(Cc), spec["T"], spec["d"], spec["release"])
    w, r = S.phi_cpcm(g, sc, rank, istar)
    return {"msgs": msgs, "C": dict(Cc), "J": J, "istar": istar, "phi_res": dict(r), "phi_wait": dict(w), "n_events": g.n}


def run_child(spec):
    E._artifact_path()
    from e4b_backend import P1, event_driver_e4b
    hp0 = E.build_platform(E.CFG, spec["faults"])
    patch_links(hp0)
    blobs = [B.load_stream(s)["events_blob"] for s in spec["streams"]]
    ev, cst, n = build(blobs, hp0, spec["plan"], spec["o"])
    occ0 = sum(int(hp0.links_dict[(r, l)].working_time(data_bytes=int(e.comm_bytes))) for e in ev.values() if e.event_type == "communication" for r, ls in e.paths.items() for l in ls)
    hp = copy.deepcopy(hp0)
    p = spec["policy"]
    if p["kind"] == "P1":
        pol = P1()
    elif p["kind"] == "fixed":
        pol = B.Fixed(p["rank"])
    else:
        pol = B.DstuE(p["tail"], p["d"], ev, hp)
    audit, released = [], {}
    random.seed(spec["seed"])
    t0 = time.time()
    event_driver_e4b(ev, hp, release=spec["release"], policy=pol, released=released, audit=audit)
    wall = time.time() - t0
    Cc = collections.defaultdict(int)
    for t, e in ev.items():
        Cc[t[1] - 10] = max(Cc[t[1] - 10], e.end_time)
    occ = sum(f - s for l in hp.links_dict.values() for s, f, _ in l.work_record)
    return {"seed": spec["seed"], "wall": wall, "C": {k: float(v) for k, v in Cc.items()}, "chunk_stats": cst, "oh_links": n, "occ": float(occ), "occ0": float(occ0),
            "audit": {"n_issues": len(audit), "n_mismatch": sum(1 for a in audit if a[2] != a[3])}}


CHILDREN = dict(C.CHILDREN, **{"oh-solo": solo_child, "oh-score": score_child, "oh-run": run_child})


# ------------------------------------------------------------------ parent
def cfgs_d0():
    return {c["hw"]: c for c in S.a2_cfgs() if c["order"] == "d0"}


def hs(plan_name):
    return plan_name


def prep(jobs):
    cf = cfgs_d0()
    items = []
    for hw, c in cf.items():
        for o in OS_:
            for sn, pl in SCHEMES.items():
                for i, k in enumerate(c["streams"]):
                    items.append((f"ohsolo_{hw}_{o}_{sn}_{i}", {"faults": c["faults"], "stream": k, "plan": pl, "o": o}))
    so = S.run_all(jobs, items, cmd="oh-solo")
    solo = {}
    for hw, c in cf.items():
        for o in OS_:
            for sn in SCHEMES:
                solo[(hw, o, sn)] = {i: so[f"ohsolo_{hw}_{o}_{sn}_{i}"] for i in range(2)}
    items = []
    for hw, c in cf.items():
        for o in OS_:
            for sn in ("d2d256", "all256"):
                That = {i: solo[(hw, o, sn)][i]["T_hat"] for i in range(2)}
                rank = D.rank_of(D.order_of({i: c["release"][i] + K_ * That[i] for i in range(2)}, 2))
                items.append((f"ohscore_{hw}_{o}_{sn}", {"faults": c["faults"], "streams": c["streams"], "plan": SCHEMES[sn], "o": o, "rank": rank, "release": c["release"], "T": c["T"], "d": c["d"]}))
    sc = S.run_all(jobs, items, cmd="oh-score")
    plans = {}
    for hw, c in cf.items():
        for o in OS_:
            s = sc[f"ohscore_{hw}_{o}_d2d256"]
            ov, u = S.select(s["phi_res"], s["msgs"], 540, True, S.K256)
            plans[(hw, o, "res540_d2d")] = (dict(S.BASE, override=ov), ov, u, s["istar"])
            s = sc[f"ohscore_{hw}_{o}_all256"]
            msgs = s["msgs"]
            actb = lambda m, x, msgs=msgs: S.K64 if msgs[m][0] > S.K64 and m not in x else None
            costb = lambda m, a, x, msgs=msgs: S.blocks(msgs[m][0], S.K64) - S.blocks(msgs[m][0], S.K256)
            order = [m for m, v in sorted(s["phi_res"].items(), key=lambda x: (-x[1], str(x[0]))) if v > 0]
            for bud in S.BUDGETS:
                ov, u = C.fill(order, msgs, bud, actb, costb)
                plans[(hw, o, f"res{bud}_all")] = (dict(S.UNI["256all"], override=ov), ov, u, s["istar"])
    return cf, solo, sc, plans


def rank_for(c, solo, hw, o, sn):
    return D.rank_of(D.order_of({i: c["release"][i] + K_ * solo[(hw, o, sn)][i]["T_hat"] for i in range(2)}, 2))


def all_items(cf, solo, plans):
    items, meta = [], []
    for hw, c in cf.items():
        base = {"faults": c["faults"], "streams": c["streams"], "release": c["release"]}
        for o in OS_:
            for sn, pl in SCHEMES.items():
                rk = rank_for(c, solo, hw, o, sn)
                tl = {i: solo[(hw, o, sn)][i] for i in range(2)}
                dh = {i: c["release"][i] + K_ * tl[i]["T_hat"] for i in range(2)}
                for pn, pol in (("P1", {"kind": "P1"}), ("C4", {"kind": "fixed", "rank": rk}), ("C-S", {"kind": "dstu", "tail": tl, "d": dh})):
                    nm = f"ohrun_{hw}_{o}_{sn}_{pn}"
                    items += S.seeded(nm, dict(base, plan=pl, o=o, policy=pol))
                    meta.append({"hw": hw, "o": o, "scheme": sn, "policy": pn, "name": nm})
            for pn, sn in (("res540_d2d", "d2d256"), ("res540_all", "all256"), ("res1620_all", "all256")):
                rk = rank_for(c, solo, hw, o, sn)
                nm = f"ohrun_{hw}_{o}_{pn}_C4"
                items += S.seeded(nm, dict(base, plan=plans[(hw, o, pn)][0], o=o, policy={"kind": "fixed", "rank": rk}))
                meta.append({"hw": hw, "o": o, "scheme": pn, "policy": "C4", "name": nm})
    return items, meta


def probe(jobs):
    t0 = time.time()
    cf, solo, sc, plans = prep(jobs)
    t1 = time.time()
    items, meta = all_items(cf, solo, plans)
    pick = [x for x in items if any(x[0].startswith(f"ohrun_F4_{o}_{s}") and x[0].endswith("_s1") for o in (0, 1000) for s in ("all64_C-S", "all64_C4", "all256_C-S", "res1620_all"))]
    r = S.run_all(jobs, pick, cmd="oh-run")
    log(f"prep {t1 - t0:.0f} s; {len(items)} runs in all")
    for n, x in r.items():
        log(f"{n}: wall {x['wall']:.0f} s, process {x['process_wall']:.0f} s, C {x['C']}")


def stat(rs, c):
    Tc, d = c["T"], c["d"]
    L = [max((r["C"][i] - d[i]) / Tc[i] for i in Tc) for r in rs]
    cst = rs[0]["chunk_stats"]
    o = rs[0].get("_o", 0)
    return {"L": statistics.mean(L), "L_range": max(L) - min(L), "C_MHA": statistics.mean(r["C"][0] for r in rs), "C_FFN": statistics.mean(r["C"][1] for r in rs), "M": statistics.mean(max(r["C"].values()) for r in rs),
            "blocks_total": cst["blocks_added"], "extra_chunk": cst["extra_occupancy"], "oh_links": rs[0]["oh_links"], "wall": statistics.mean(r["wall"] for r in rs)}


def sign(x, th=TH):
    return 0 if abs(x) <= th else (1 if x > 0 else -1)


def run(jobs):
    t0 = time.time()
    cf, solo, sc, plans = prep(jobs)
    items, meta = all_items(cf, solo, plans)
    log(f"{len(items)} runs")
    res = S.run_all(jobs, items, cmd="oh-run")
    aud, cons, occ_bad = 0, 0, []
    for n, x in res.items():
        aud += x["audit"]["n_mismatch"]
        cons += not (x["chunk_stats"]["bytes_conserved"] and x["chunk_stats"]["tree_conserved"])
    rows = []
    blocks0 = {hw: None for hw in HWS}
    for m in meta:
        c = cf[m["hw"]]
        rs = S.runs_of(res, m["name"])
        for r in rs:
            d = r["occ"] - r["occ0"] - m["o"] * r["oh_links"]
            if d:
                occ_bad.append((m["name"], r["seed"], d))
        s = stat(rs, c)
        base = stat(S.runs_of(res, f"ohrun_{m['hw']}_0_d2d256_C4"), c)
        o = m["o"]
        row = {**{k: m[k] for k in ("hw", "o", "scheme", "policy")}, **s, "new_blocks": s["blocks_total"] - base["blocks_total"], "extra_occupancy": s["extra_chunk"] + o * s["oh_links"],
               "extra_vs_d2d256": s["extra_chunk"] + o * s["oh_links"] - base["extra_chunk"], "dM_vs_o0": None}
        if m["scheme"] in plans_names():
            ov0 = plans[(m["hw"], 0, m["scheme"])][1]
            ov = plans[(m["hw"], o, m["scheme"])][1]
            j = C.jac_over(ov, ov0)
            row.update(ov_jaccard_o0=j["jaccard"], ov_common_o0=j["common"], n_msgs=j["n_a"], same_size_o0=j["same_size"], plan_blocks=plans[(m["hw"], o, m["scheme"])][2], istar=plans[(m["hw"], o, m["scheme"])][3])
        rows.append(row)
    idx = {(r["hw"], r["o"], r["scheme"], r["policy"]): r for r in rows}
    for r in rows:
        r["dM_vs_o0"] = r["M"] - idx[(r["hw"], 0, r["scheme"], r["policy"])]["M"]
        r["dCF_vs_o0"] = r["C_FFN"] - idx[(r["hw"], 0, r["scheme"], r["policy"])]["C_FFN"]
    wc("rows.csv", rows)
    # pairings
    PAIRS = [("all256 - d2d256", ("all256", "C4"), ("d2d256", "C4")), ("all64 - all256", ("all64", "C4"), ("all256", "C4")), ("res540_all - all256", ("res540_all", "C4"), ("all256", "C4")),
             ("res1620_all - all64", ("res1620_all", "C4"), ("all64", "C4")), ("res540_d2d - all256", ("res540_d2d", "C4"), ("all256", "C4")), ("res1620_all - res540_all", ("res1620_all", "C4"), ("res540_all", "C4")),
             ("C4 - C-S d2d256", ("d2d256", "C4"), ("d2d256", "C-S")), ("C4 - C-S all256", ("all256", "C4"), ("all256", "C-S")), ("C4 - C-S all64", ("all64", "C4"), ("all64", "C-S")),
             ("C4 - P1 d2d256", ("d2d256", "C4"), ("d2d256", "P1")), ("C4 - P1 all256", ("all256", "C4"), ("all256", "P1")), ("C4 - P1 all64", ("all64", "C4"), ("all64", "P1"))]
    pair, flips = [], []
    for hw in HWS:
        c = cf[hw]
        TF = c["T"][1]
        for pn, a, b in PAIRS:
            seqL, seqM = [], []
            for o in OS_:
                x, y = idx[(hw, o, *a)], idx[(hw, o, *b)]
                d = x["L"] - y["L"]
                dm = (x["M"] - y["M"]) / TF
                dc = (x["C_FFN"] - y["C_FFN"]) / TF
                seqL.append(d)
                seqM.append(dm)
                pair.append({"hw": hw, "pair": pn, "o": o, "dL": d, "dL_sign": sign(d), "dM_over_TF": dm, "dM_sign": sign(dm), "dCF_over_TF": dc, "L_a": x["L"], "L_b": y["L"], "M_a": x["M"], "M_b": y["M"], "CF_a": x["C_FFN"], "CF_b": y["C_FFN"], "CM_a": x["C_MHA"], "CM_b": y["C_MHA"], "range_a": x["L_range"], "range_b": y["L_range"]})
            for metric, seq in (("L", seqL), ("M", seqM)):
                sg = [sign(v) for v in seq]
                for i in range(3):
                    if sg[i] != sg[i + 1] and sg[i] * sg[i + 1] <= 0:
                        flips.append({"hw": hw, "pair": pn, "metric": metric, "o_low": OS_[i], "o_high": OS_[i + 1], "val_low": seq[i], "val_high": seq[i + 1], "sign_low": sg[i], "sign_high": sg[i + 1], "full_reversal": sg[i] * sg[i + 1] < 0})
    wc("pairs.csv", pair)
    wc("flips.csv", flips)
    # o = 0 against existing results
    nm = {hw: cf[hw]["name"] for hw in HWS}
    refs = []

    def load(path):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    missing = []
    for hw in HWS:
        c = cf[hw]
        name = nm[hw]
        opts = [("d2d256", "C4", [os.path.join(S.SIMS, f"b_{name}_256d2d_s{s}.pkl.gz") for s in B.SEEDS]), ("all256", "C4", [os.path.join(S.SIMS, f"b_{name}_256all_s{s}.pkl.gz") for s in B.SEEDS]), ("all64", "C4", [os.path.join(S.SIMS, f"b_{name}_64all_s{s}.pkl.gz") for s in B.SEEDS])]
        p0 = plans[(hw, 0, "res540_d2d")][1]
        opts.append(("res540_d2d", "C4", [os.path.join(S.SIMS, f"s_{name}_{S.fh(p0)}_s{s}.pkl.gz") for s in B.SEEDS]))
        for bud in S.BUDGETS:
            p0 = plans[(hw, 0, f"res{bud}_all")][1]
            opts.append((f"res{bud}_all", "C4", [os.path.join(S.SIMS, f"sB_{name}_{S.fh(p0)}_s{s}.pkl.gz") for s in B.SEEDS]))
        for sn, pn, paths in opts:
            if not all(os.path.exists(p) for p in paths):
                missing.append((hw, sn))
                continue
            old = [load(p) for p in paths]
            new = S.runs_of(res, f"ohrun_{hw}_0_{sn}_{pn}")
            refs.append({"hw": hw, "scheme": sn, "policy": pn, "same_C_seeds": sum(1 for a, b in zip(old, new) if {k: float(v) for k, v in a["C"].items()} == b["C"]), "rel_M": statistics.mean(float(max(b["C"].values())) for b in new) / statistics.mean(float(max(a["C"].values())) for a in old) - 1,
                         "rel_CM": statistics.mean(b["C"][0] for b in new) / statistics.mean(float(a["C"][0]) for a in old) - 1, "dL": idx[(hw, 0, sn, pn)]["L"] - statistics.mean(max((float(a["C"][i]) - c["d"][i]) / c["T"][i] for i in c["T"]) for a in old)})
        for sn, pn, tag in (("all256", "C-S", f"s1all_{name}_CS_k1.5"), ("all256", "P1", f"s1all_{name}_P1"), ("d2d256", "C-S", None), ("d2d256", "P1", None)):
            if tag:
                old = load(os.path.join(S.SIMS, f"{tag}.pkl.gz"))["runs"]
            else:
                p = os.path.join(D.SIMS, f"{name}_CS_k1.5.pkl.gz") if pn == "C-S" else os.path.join(D.SIMS, f"rep_{name}_P1.pkl.gz")
                if not os.path.exists(p):
                    missing.append((hw, sn, pn))
                    continue
                old = load(p)["runs"]
            old = [a for a in old if a["seed"] in B.SEEDS]
            new = S.runs_of(res, f"ohrun_{hw}_0_{sn}_{pn}")
            refs.append({"hw": hw, "scheme": sn, "policy": pn, "same_C_seeds": sum(1 for a, b in zip(old, new) if {k: float(v) for k, v in a["C"].items()} == b["C"]), "rel_M": statistics.mean(float(max(b["C"].values())) for b in new) / statistics.mean(float(max(a["C"].values())) for a in old) - 1,
                         "rel_CM": statistics.mean(b["C"][0] for b in new) / statistics.mean(float(a["C"][0]) for a in old) - 1, "dL": idx[(hw, 0, sn, pn)]["L"] - statistics.mean(max((float(a["C"][i]) - c["d"][i]) / c["T"][i] for i in c["T"]) for a in old)})
    wc("check_o0.csv", refs)
    out = {"audit_mismatch": aud, "conservation_bad": cons, "occupancy_bad": occ_bad[:20], "occupancy_bad_n": len(occ_bad), "missing_refs": missing,
           "o0_max_rel_M": max(abs(r["rel_M"]) for r in refs), "o0_max_abs_dL": max(abs(r["dL"]) for r in refs), "o0_all_seeds_same": sum(r["same_C_seeds"] == 3 for r in refs), "o0_refs": len(refs),
           "T": {hw: cf[hw]["T"] for hw in HWS}, "d": {hw: cf[hw]["d"] for hw in HWS}, "rank": {f"{hw}_{o}_{sn}": rank_for(cf[hw], solo, hw, o, sn) for hw in HWS for o in OS_ for sn in SCHEMES},
           "That": {f"{hw}_{o}_{sn}": [solo[(hw, o, sn)][i]["T_hat"] for i in range(2)] for hw in HWS for o in OS_ for sn in SCHEMES}, "wall": time.time() - t0, "command": sys.argv, "seeds": B.SEEDS, "artifact_commit": E._artifact_commit()}
    wj("overhead.json", out)
    log(json.dumps({k: out[k] for k in ("audit_mismatch", "conservation_bad", "occupancy_bad_n", "missing_refs", "o0_max_rel_M", "o0_max_abs_dL", "o0_all_seeds_same", "o0_refs", "wall")}, default=str))
    if aud or cons or occ_bad or out["o0_max_rel_M"] > 0.04:
        raise D.DstuError("overhead self-check")


def plans_names():
    return ("res540_d2d", "res540_all", "res1620_all")


# ------------------------------------------------------------------ 11.2
def key_check(jobs):
    t0 = time.time()
    plats, T, solo = S.plats_all()
    S13, plats3, T3 = D.load_s3()
    solo3 = {}
    for g, hw, cz, keys, fa, alias in plats3:
        for k in keys:
            with gzip.open(os.path.join(D.SOLO, f"{k}_{cz or 'none'}.pkl.gz"), "rb") as f:
                solo3[(k, cz)] = pickle.load(f)
    plats_c, T_c, solo_c, ideal_c = C.solo_all(jobs)
    plats_c = [(g, hw, C.CZ, keys, fa, alias) for g, hw, cz, keys, fa, alias in plats_c]
    # aggregate cost per stream x chunking
    seen = {}
    for pl in (plats, plats3, plats_c):
        for g, hw, cz, keys, fa, alias in pl:
            for k in keys:
                seen[(k, cz)] = fa
    plan_of = lambda cz: {"none": None, None: None, "256d2d": S.BASE, "256all": S.UNI["256all"]}[cz]
    its = [(f"ohagg_{k}_{cz or 'none'}", {"faults": fa, "plan": plan_of(cz), "streams": {"X": k}}) for (k, cz), fa in seen.items()]
    ag = S.run_all(jobs, its, cmd="agg-child")
    agg = {kk: ag[f"ohagg_{kk[0]}_{kk[1] or 'none'}"]["X"] for kk in seen}
    wc("agg_cost.csv", [{"stream": k[0], "chunk": k[1] or "none", "max_occ": v["max_occ"], "bytes": v["bytes"], "bytes_over_256": v["bytes"] / 256, "T_hat": v["T_hat"]} for k, v in agg.items()])
    out = {}
    all_rows = []
    sets = [("S1", 1, plats, T, solo, os.path.join(D.OUT, "setting1_report.csv")), ("S2", 2, plats, T, solo, os.path.join(D.OUT, "setting2_report.csv")), ("S3", 3, plats3, T3, solo3, os.path.join(D.OUT, "setting3_report.csv")), ("S1all", 1, plats_c, T_c, solo_c, os.path.join(C.OUT, "s1_report.csv"))]
    for sname, n, pl, Tb, sl, rep in sets:
        ideal = {}
        cfgs = D.build_cfgs(n, pl, Tb, sl, ideal)
        with open(rep, encoding="utf-8") as f:
            R = {(r["config"], r["dl"]): r for r in csv.DictReader(f)}
        if sname == "S1all":
            def fx(c, o):
                p = os.path.join(S.SIMS, f"s1all_{c['name']}_fixed_{o}.pkl.gz")
                with gzip.open(p, "rb") as f:
                    return D.stat(pickle.load(f)["runs"], c)["L"]
        else:
            tab = D.ref_table(n)

            def fx(c, o, tab=tab):
                v = tab.get((c["name"], c["tg"], f"fixed_{o}"))
                return v["L"] if v else None
        bad_c4 = 0
        for c in cfgs:
            r = R[(c["name"], c["tg"])]
            nst = c["n"]
            ks = c["k"]
            rel = c["release"]
            o4 = D.order_of(c["dhat"], nst)
            if o4 != r["C4_order"]:
                bad_c4 += 1
            A = {i: agg[(s, c["chunk"])]["max_occ"] for i, s in enumerate(c["streams"])}
            By = {i: agg[(s, c["chunk"])]["bytes"] / 256 for i, s in enumerate(c["streams"])}
            oa = D.order_of({i: rel[i] + ks[i] * A[i] for i in A}, nst)
            ob = D.order_of({i: rel[i] + ks[i] * By[i] for i in A}, nst)
            row = {"set": sname, "config": c["name"], "dl": c["tg"], "fam": "a1" if c["group"].startswith("a1") else "a2", "hw": c["hw"], "order_C4": o4, "order_agg": oa, "order_bytes": ob, "L_C4": float(r["L_C4"])}
            for tg, oo in (("agg", oa), ("bytes", ob)):
                row[f"differ_{tg}"] = oo != o4
                v = fx(c, oo) if oo != o4 else float(r["L_C4"])
                row[f"L_{tg}"] = v if v is not None else ""
                row[f"dL_{tg}"] = (v - float(r["L_C4"])) if v is not None else ""
            all_rows.append(row)
        out[f"{sname}_C4_order_mismatch"] = bad_c4
    wc("key_rows.csv", all_rows)
    S_ = {}
    for sname in ("S1", "S2", "S3", "S1all"):
        for fam in ("all", "a1", "a2"):
            sub = [r for r in all_rows if r["set"] == sname and fam in ("all", r["fam"])]
            if not sub:
                continue
            for tg in ("agg", "bytes"):
                d = [r for r in sub if r[f"differ_{tg}"]]
                v = [r[f"dL_{tg}"] for r in d if r[f"dL_{tg}"] != ""]
                S_[f"{sname}_{fam}_{tg}"] = {"n": len(sub), "differ": len(d), "with_L": len(v), "no_L": len(d) - len(v), "median": statistics.median(v) if v else None, "min": min(v) if v else None, "max": max(v) if v else None,
                                             "worse": sum(x > TH for x in v), "better": sum(x < -TH for x in v), "tie": sum(abs(x) <= TH for x in v), "worst_L_C4": max(r["L_C4"] for r in sub), "worst_L_key": max(float(r[f"L_{tg}"]) for r in sub if r[f"L_{tg}"] != "") if any(r[f"L_{tg}"] != "" for r in sub) else None}
    out["summary"] = S_
    out.update(wall=time.time() - t0, command=sys.argv)
    wj("key.json", out)
    log(json.dumps(D.round_all(out), default=str, ensure_ascii=False))
    if any(out[f"{s}_C4_order_mismatch"] for s in ("S1", "S2", "S3", "S1all")):
        raise D.DstuError("C4 order mismatch against the stage 1 reports")


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
    ap.add_argument("cmd", choices=["probe", "run", "key"])
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    {"probe": probe, "run": run, "key": key_check}[a.cmd](a.jobs)


if __name__ == "__main__":
    try:
        main()
    except (D.DstuError, B.E4bError, E.E4Error, cpcm.CPCMError) as exc:
        print(f"[dstu_oh] STOP: {exc}", flush=True)
        sys.exit(2)
