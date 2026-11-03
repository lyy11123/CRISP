"""E4b step 3: setting 2 (heterogeneous deadlines, two streams), handoff 2.3. Per-hop backend e4b_backend; policies
of e4b.py. User: "开始设定 2"; "超过一小时也没事，正常跑".

Groups: a1 512+8192, a1 2048+8192 (unchunked), a2 MHA+FFN (256 KB D2D-only). Hardware: no fault, F4. Releases: d0,
  long_first (D = round(0.5 T_min)). T_i, solo references, seeds (1-3) and releases as setting 1 (e4b_s1.py).
Deadlines d_i = r_i + k_i T_i (long = the stream with the larger T_i):
  forward  k_long in {1.1, 1.2, 1.5} x k_short in {2, 3, 5}
  reverse  k_short in {1.1, 1.5} x k_long in {2, 3}   (PD-style control)
Policies: P1 and every fixed order read from the setting-1 cache (deadline-independent); P2 / P3 / P4 = the fixed
  order their key induces (P2 at d0 = P1); P5 = best fixed order by mean L_hat_max; P6, P7, P6' (P6n), P7m run per
  deadline pair.
Self-check 3 (user decision 2026-11-02): reference order = increasing initial slack r_i + (k_i - 1) T_i; when P6's
  slack order equals it at every tick, shadow differ = 0 and S_max within 4% of that fixed order. Checks 4 and 7 on
  the new runs. L_hat_max comparisons count |difference| > 0.04 (T_i units); S_max comparisons are relative (4%).

  python e4b_s2.py --jobs 12
Writes e4b_out/setting2_runs.csv, setting2_table.csv, setting2_report.csv, setting2.json.
"""
import argparse
import collections
import itertools
import json
import os
import sys
import time

import e4_hop as H
import e4_overlap as E
import e4b as B
import e4b_s1 as S1
import e4b_s1b as S1b

GROUPS = ["a1_2s_512-8192", "a1_2s_2048-8192", "a2_2s_mha-ffn"]
HW = ["nofault", "F4"]
ORDERS = ["d0", "long_first"]
COMBOS = [(kl, ks, "forward") for kl in (1.1, 1.2, 1.5) for ks in (2.0, 3.0, 5.0)] + \
         [(kl, ks, "reverse") for ks in (1.1, 1.5) for kl in (2.0, 3.0)]
RUN = ["P6", "P7", "P6n", "P7m"]
POL = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P6n", "P7m"]


def tag(kl, ks):
    return f"kl{kl:g}_ks{ks:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()
    st, _ = S1.load()
    plats = [p for p in S1.platforms(st) if p[0] in GROUPS and p[1] in HW]
    with open(os.path.join(B.OUT, "setting1.json"), encoding="utf-8") as f:
        s1 = json.load(f)
    T = {}
    for kk, v in s1["T"].items():
        k, cz = kk.rsplit("|", 1)
        T[(k, None if cz == "None" else cz)] = v
    solo_items = {}
    for g, hw, cz, keys, fa, alias in plats:
        for k in keys:
            solo_items[(k, cz)] = (f"solo_{k}_{cz or 'none'}", {"cfg": E.CFG, "faults": fa, "streams": [k], "release": {0: 0},
                                                              "seeds": B.SEEDS, "policy": {"name": "P1"},
                                                              "chunk": H.CHUNKS[cz] if cz else None, "keep_cap": True})
    sres = B.run_all(args.jobs, list(solo_items.values()))
    solo = {}
    for (k, cz), (n, _) in solo_items.items():
        solo[(k, cz)] = B.solo_data(next(r["cap"] for r in sres[n]["runs"] if r["seed"] == 1))
    del sres
    t1 = time.time()
    cfgs, items = [], []
    for g, hw, cz, keys, fa, alias in plats:
        if alias:
            raise B.E4bError(f"unexpected alias {g} {hw}")
        n = len(keys)
        Tc = {i: T[(k, cz)] for i, k in enumerate(keys)}
        D = int(round(0.5 * min(Tc.values())))
        lng = max(Tc, key=lambda i: Tc[i])
        for order in ORDERS:
            rel = S1.releases(Tc, order, D)
            name = f"{g}_{hw}_{order}"
            base = {"cfg": E.CFG, "faults": fa, "streams": keys, "release": rel, "seeds": B.SEEDS,
                    "chunk": H.CHUNKS[cz] if cz else None, "audit": True}
            sd = {i: solo[(k, cz)] for i, k in enumerate(keys)}
            for kl, ks, kind in COMBOS:
                kk = {i: (kl if i == lng else ks) for i in Tc}
                d = {i: rel[i] + kk[i] * Tc[i] for i in Tc}
                ref_rank = {i: r for r, i in enumerate(sorted(Tc, key=lambda i: (rel[i] + (kk[i] - 1) * Tc[i], i)))}
                cfgs.append({"name": name, "group": g, "hw": hw, "chunk": cz, "order": order, "T": Tc, "release": rel,
                             "n": n, "kl": kl, "ks": ks, "kind": kind, "k": kk, "d": d, "ref_rank": ref_rank, "long": lng})
                pp = {"solo": sd, "d": d, "T": Tc, "release": rel, "ref_rank": ref_rank}
                for p in RUN:
                    items.append((f"{name}_{p}_{tag(kl, ks)}", dict(base, policy=dict(pp, name=p), shadow_rank=ref_rank)))
    B.log(f"{len(cfgs)} configuration x deadline rows, {len(items)} processes")
    res = B.run_all(args.jobs, items)
    t2 = time.time()
    old = {}

    def runs_of(rn):
        if rn in res:
            return res[rn]["runs"]
        if rn not in old:
            old[rn] = S1b.cached(rn)["runs"]
        return old[rn]

    runs_rows, table, rows_rep = [], [], []
    chk = {"check3": {}, "check4": collections.Counter(), "check7": {}}
    for c in cfgs:
        src, Tc, rel, n, d = c["name"], c["T"], c["release"], c["n"], c["d"]
        tg = tag(c["kl"], c["ks"])
        fixed = {"".join(map(str, p)): f"{src}_fixed_{''.join(map(str, p))}" for p in itertools.permutations(range(n))}
        sjf = "".join(map(str, sorted(range(n), key=lambda i: (Tc[i], i))))
        fcfs = "".join(map(str, sorted(range(n), key=lambda i: (rel[i], i))))
        edf = "".join(map(str, sorted(range(n), key=lambda i: (d[i], i))))
        refo = "".join(map(str, sorted(range(n), key=lambda i: c["ref_rank"][i])))
        pol = {"P1": f"{src}_P1", "P2": f"{src}_P1" if c["order"] == "d0" else fixed[fcfs], "P3": fixed[sjf],
               "P4": fixed[edf], **{p: f"{src}_{p}_{tg}" for p in RUN}}
        pol.update({f"fixed_{o}": v for o, v in fixed.items()})
        stat = {}
        for pn, rn in pol.items():
            rs = runs_of(rn)
            L = [B.lhat(r, Tc, d) for r in rs]
            Lr = [max(r["C"][i] - d[i] for i in Tc) for r in rs]
            S = [B.smax(r, Tc) for r in rs]
            V = [sum(1 for i in Tc if r["C"][i] > d[i]) for r in rs]
            M = [max(r["C"].values()) for r in rs]
            stat[pn] = {"Lhat": E._mean(L), "Lhat_range": max(L) - min(L), "L_max": E._mean(Lr),
                        "L_max_range": max(Lr) - min(Lr), "S_max": E._mean(S), "S_range": max(S) - min(S),
                        "makespan": E._mean(M), "violations": E._mean(V),
                        "stretch": {i: E._mean([(r["C"][i] - r["release"][i]) / Tc[i] for r in rs]) for i in Tc}, "run": rn}
            if not pn.startswith("fixed_"):
                for r, l, lr, s_, m, v in zip(rs, L, Lr, S, M, V):
                    runs_rows.append({"group": c["group"], "hw": c["hw"], "chunk": c["chunk"] or "none", "order": c["order"],
                                      "k_long": c["kl"], "k_short": c["ks"], "kind": c["kind"], "policy": pn, "run": rn,
                                      "seed": r["seed"], **{f"r_{i}": rel[i] for i in Tc}, **{f"k_{i}": c["k"][i] for i in Tc},
                                      **{f"C_{i}": r["C"][i] for i in Tc}, "Lhat_max": l, "L_max": lr, "S_max": s_,
                                      "makespan": m, "violations": v, "wall": r["wall"]})
        best = min((p for p in stat if p.startswith("fixed_")), key=lambda p: stat[p]["Lhat"])
        stat["P5"] = dict(stat[best], member=best)
        for pn, s in stat.items():
            table.append({"config": src, "group": c["group"], "hw": c["hw"], "chunk": c["chunk"] or "none",
                          "order": c["order"], "k_long": c["kl"], "k_short": c["ks"], "kind": c["kind"], "policy": pn,
                          "member": s.get("member", ""), "Lhat_max": s["Lhat"], "Lhat_range": s["Lhat_range"],
                          "L_max": s["L_max"], "L_max_range": s["L_max_range"], "S_max": s["S_max"], "S_range": s["S_range"],
                          "makespan": s["makespan"], "violations": s["violations"],
                          **{f"stretch_{i}": v for i, v in s["stretch"].items()}, "T": json.dumps(Tc),
                          "release": json.dumps(rel), "k": json.dumps(c["k"])})
        o = {"config": src, "group": c["group"], "hw": c["hw"], "order": c["order"], "k_long": c["kl"], "k_short": c["ks"],
             "kind": c["kind"], "P5_member": best, "ref_order": refo, "edf_order": edf}
        P = POL
        for p in P:
            o[f"L_{p}"] = stat[p]["Lhat"]
            o[f"Lraw_{p}"] = stat[p]["L_max"]
            o[f"S_{p}"] = stat[p]["S_max"]
            o[f"V_{p}"] = stat[p]["violations"]
            o[f"M_{p}"] = stat[p]["makespan"]
        for p in RUN:
            for ref in ("P3", "P4", "P5"):
                o[f"rS_{p}_{ref}"] = o[f"S_{p}"] / o[f"S_{ref}"] - 1
                o[f"dL_{p}_{ref}"] = o[f"L_{p}"] - o[f"L_{ref}"]
        o["rS_P6n_P6"] = o["S_P6n"] / o["S_P6"] - 1
        o["dL_P6n_P6"] = o["L_P6n"] - o["L_P6"]
        o["rS_P7m_P7"] = o["S_P7m"] / o["S_P7"] - 1
        o["dL_P7m_P7"] = o["L_P7m"] - o["L_P7"]
        o["best_Lhat"] = min(P, key=lambda p: (round(stat[p]["Lhat"], 9), p))
        o["best_L_max"] = min(P, key=lambda p: (round(stat[p]["L_max"], 6), p))
        rows_rep.append(o)
        r6 = res[pol["P6"]]["runs"]
        sl = {x: sum(r["slack_log"][x] for r in r6) for x in ["ticks", "rank_changes", "ticks_not_ref"]}
        sh = {x: sum(r["shadow"].get(x, 0) for r in r6) for x in ["n", "differ", "differ_stream"]}
        ent = {"slack_log": sl, "shadow": sh, "S_P6": stat["P6"]["S_max"], "ref_order": refo,
               "S_ref": stat[f"fixed_{refo}"]["S_max"]}
        if sl["ticks_not_ref"] == 0:
            ent["pass"] = sh["differ"] == 0 and abs(stat["P6"]["S_max"] / stat[f"fixed_{refo}"]["S_max"] - 1) <= 0.04
        chk["check3"][f"{src}_{tg}"] = ent
        for p in RUN:
            for r in res[pol[p]]["runs"]:
                for x in ("n_issues", "n_multi", "n_mismatch"):
                    chk["check4"][x] += r["audit"][x]
        w1 = E._mean([r["wall"] for r in runs_of(f"{src}_P1")])
        chk["check7"][f"{src}_{tg}"] = {p: E._mean([r["wall"] for r in res[pol[p]]["runs"]]) / w1 for p in RUN}
    B.write_csv("setting2_runs.csv", runs_rows)
    B.write_csv("setting2_table.csv", table)
    B.write_csv("setting2_report.csv", rows_rep)
    stop = []
    if any(v.get("pass") is False for v in chk["check3"].values()):
        stop.append("check 3: " + ", ".join(n for n, v in chk["check3"].items() if v.get("pass") is False))
    if chk["check4"]["n_mismatch"] or chk["check4"]["n_issues"] < 1000:
        stop.append("check 4")
    if any(x > 10 for v in chk["check7"].values() for x in v.values()):
        stop.append("handoff 6: DSTU > 10x P1")
    out = {"checks": {**chk, "check4": dict(chk["check4"])}, "combos": COMBOS, "stop": stop,
           "walls": {"solo": t1 - t0, "runs": t2 - t1, "total": time.time() - t0}, "seeds": B.SEEDS, "command": sys.argv,
           "artifact_commit": E._artifact_commit(), "processes": len(items),
           "reused": "P1 and fixed orders from the setting-1 cache (e4b_out/sims)"}
    B.write_json("setting2.json", out)
    B.log(json.dumps({"rows": len(cfgs), "processes": len(items), "wall_s": round(time.time() - t0),
                      "check3_applicable": sum(1 for v in chk["check3"].values() if "pass" in v),
                      "check3_fail": sum(1 for v in chk["check3"].values() if v.get("pass") is False),
                      "check4": dict(chk["check4"]),
                      "check7_max": round(max(x for v in chk["check7"].values() for x in v.values()), 2), "stop": stop}))
    if stop:
        raise B.E4bError("; ".join(stop))


if __name__ == "__main__":
    try:
        main()
    except (B.E4bError, E.E4Error) as exc:
        print(f"[e4b] STOP: {exc}", flush=True)
        sys.exit(2)
