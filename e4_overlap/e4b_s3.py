"""E4b step 4: setting 3 (random deadlines and releases, three streams), handoff 2.3. Per-hop backend e4b_backend;
policies of e4b.py. User: "估算设定 3 的耗时（策略包含 P1–P7、P6′、P7m，再顺手加一个不除以 Tᵢ 的原始时间托底版 P7m_raw：
排序键为 dᵢ − t − max(R̂ᵢ(t), tailₑ)），没问题就可以跑了".

Groups: a1 512/2048/8192 (unchunked), a2 MHA/FFN/MHA (256 KB D2D-only). Hardware: no fault, F4 (E4a batch-1 faults
  and R0 graphs). T_i as setting 1 (unchunked: E4a per-hop singles; chunked: E4a diag chunk singles, missing ones
  run here with seeds 1-6, as e4b_s1.py).
Draws: N = 20 per group; draw j of group g uses random.Random(SAMPLE_SEED + 100 * g_index + j): for stream i
  (0-based) u_i ~ U[0, 1], k_i ~ U[1.1, 3.0]; r_i = round(u_i * 0.5 * T_min * i) (stream 0 always at 0), T_min = the
  shortest T_i on that hardware. The same (u_i, k_i) are used for no fault and F4 (paired). d_i = r_i + k_i T_i.
Policies per draw: P1, the 6 fixed orders, P6, P7, P6' (P6n), P7m, P7m_raw. P2 / P3 / P4 = the fixed order of
  increasing r_i / T_i / d_i; P5 = best fixed order by mean L_hat_max. Backend seeds 1-3.
Self-check 3 (as setting 2): reference = increasing initial slack r_i + (k_i - 1) T_i. Checks 4, 5 (solo R_hat), 7.
Comparisons: L_hat_max by absolute difference > 0.04, S_max relative 4%; "strictly better than BestFixed" = mean
  L_hat_max lower by more than max(P5's seed range, the policy's seed range).

  python e4b_s3.py --probe --jobs 12   # draw 0 of every group x hardware (48 processes; reused by the full run)
  python e4b_s3.py --jobs 12
Writes e4b_out/setting3_runs.csv, setting3_table.csv, setting3_report.csv, setting3_draws.csv, setting3.json.
"""
import argparse
import collections
import gzip
import itertools
import json
import os
import pickle
import random
import statistics
import sys
import time

import e4_grid as G
import e4_hop as H
import e4_overlap as E
import e4b as B
import e4b_s1 as S1

GRP = {"a1_3s_512-2048-8192": None, "a2_3s_mha-ffn-mha": "256d2d"}
HW = ["nofault", "F4"]
N = 20
SAMPLE_SEED = 20261103
RUN = ["P6", "P7", "P6n", "P7m", "P7m_raw"]
POL = ["P1", "P2", "P3", "P4", "P5"] + RUN


def platforms(st):
    out = []
    for g, cz in GRP.items():
        keys = G.GROUPS[g][1]
        for hw in HW:
            if hw == "nofault":
                out.append((g, hw, cz, keys, None))
            else:
                fl = st["faults"][g][hw]
                out.append((g, hw, cz, [G.fault_key(k, fl) for k in keys],
                            {"failed_nodes": [], "failed_links": fl["failed_links"]}))
    return out


def draws():
    out = {}
    for gi, g in enumerate(GRP):
        n = len(G.GROUPS[g][1])
        for j in range(N):
            rng = random.Random(SAMPLE_SEED + 100 * gi + j)
            out[(g, j)] = {"seed": SAMPLE_SEED + 100 * gi + j, "u": [rng.random() for _ in range(n)],
                           "k": [rng.uniform(1.1, 3.0) for _ in range(n)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    st, Thop = S1.load()
    plats = platforms(st)
    T, need = {}, []
    for g, hw, cz, keys, fa in plats:
        for k in keys:
            if cz is None:
                T[(k, cz)] = Thop[k]
                continue
            p = os.path.join(E.OUT, "diag", "sims", f"chunk_single_{k}_{cz}.pkl.gz")
            if os.path.exists(p):
                with gzip.open(p, "rb") as f:
                    T[(k, cz)] = E._mean([r["C"][0] for r in pickle.load(f)["runs"]])
            elif (k, cz) not in T and not any(n == f"Tsingle_{k}_{cz}" for n, _ in need):
                need.append((f"Tsingle_{k}_{cz}", {"cfg": E.CFG, "faults": fa, "streams": [k], "release": {0: 0},
                                                   "seeds": G.D1_SEEDS, "policy": {"name": "P1"}, "chunk": H.CHUNKS[cz]}))
    if need:
        r = B.run_all(args.jobs, need)
        for n, _ in need:
            k, cz = n[len("Tsingle_"):].rsplit("_", 1)
            T[(k, cz)] = E._mean([x["C"][0] for x in r[n]["runs"]])
    solo_items = {}
    for g, hw, cz, keys, fa in plats:
        for k in keys:
            solo_items[(k, cz)] = (f"solo_{k}_{cz or 'none'}", {"cfg": E.CFG, "faults": fa, "streams": [k], "release": {0: 0},
                                                              "seeds": B.SEEDS, "policy": {"name": "P1"},
                                                              "chunk": H.CHUNKS[cz] if cz else None, "keep_cap": True})
    sres = B.run_all(args.jobs, list(solo_items.values()))
    solo, chk5 = {}, {}
    for (k, cz), (n, _) in solo_items.items():
        caps = {r["seed"]: r["cap"] for r in sres[n]["runs"]}
        solo[(k, cz)] = B.solo_data(caps[1])
        chk5[f"{k}|{cz or 'none'}"] = B.rhat_check(caps, solo[(k, cz)], T[(k, cz)])
    del sres
    bad5 = [k for k, x in chk5.items() if statistics.median([v["median"] for v in x.values()]) > 0.05]
    if bad5:
        raise B.E4bError(f"self-check 5: median R_hat deviation > 5%: {bad5}")
    t1 = time.time()
    dr = draws()
    cfgs, items = [], []
    for g, hw, cz, keys, fa in plats:
        n = len(keys)
        Tc = {i: T[(k, cz)] for i, k in enumerate(keys)}
        Tmin = min(Tc.values())
        sd = {i: solo[(k, cz)] for i, k in enumerate(keys)}
        for j in range(1 if args.probe else N):
            x = dr[(g, j)]
            rel = {i: int(round(x["u"][i] * 0.5 * Tmin * i)) for i in Tc}
            kk = {i: x["k"][i] for i in Tc}
            d = {i: rel[i] + kk[i] * Tc[i] for i in Tc}
            ref_rank = {i: r for r, i in enumerate(sorted(Tc, key=lambda i: (rel[i] + (kk[i] - 1) * Tc[i], i)))}
            name = f"s3_{g}_{hw}_j{j:02d}"
            cfgs.append({"name": name, "group": g, "hw": hw, "chunk": cz, "draw": j, "sample_seed": x["seed"], "T": Tc,
                         "release": rel, "k": kk, "d": d, "ref_rank": ref_rank, "n": n})
            base = {"cfg": E.CFG, "faults": fa, "streams": keys, "release": rel, "seeds": B.SEEDS,
                    "chunk": H.CHUNKS[cz] if cz else None, "audit": True}
            items.append((f"{name}_P1", dict(base, policy={"name": "P1"})))
            for perm in itertools.permutations(range(n)):
                items.append((f"{name}_fixed_{''.join(map(str, perm))}",
                              dict(base, policy={"name": "fixed", "rank": {s: r for r, s in enumerate(perm)}})))
            pp = {"solo": sd, "d": d, "T": Tc, "release": rel, "ref_rank": ref_rank}
            for p in RUN:
                items.append((f"{name}_{p}", dict(base, policy=dict(pp, name=p), shadow_rank=ref_rank)))
    B.log(f"{len(cfgs)} draws x hardware, {len(items)} processes")
    res = B.run_all(args.jobs, items)
    t2 = time.time()
    if args.probe:
        pw = {c["name"]: sum(res[n]["process_wall"] for n, _ in items if n.startswith(c["name"] + "_")) for c in cfgs}
        est = sum(v * N for v in pw.values()) / args.jobs / 3600
        B.write_json("setting3_probe.json", {"process_wall_per_draw": pw, "est_total_h": est, "probe_wall": t2 - t1,
                                             "max_eval": max(r["wall"] for v in res.values() for r in v["runs"])})
        B.log(json.dumps({"process_wall_per_draw": {k: round(v) for k, v in pw.items()}, "probe_wall_s": round(t2 - t1),
                          "est_total_h": round(est, 2)}))
        return
    runs_rows, table, rows_rep, drows = [], [], [], []
    chk = {"check3": {}, "check4": collections.Counter(), "check7": {}}
    for c in cfgs:
        src, Tc, rel, n, d = c["name"], c["T"], c["release"], c["n"], c["d"]
        drows.append({"config": src, "group": c["group"], "hw": c["hw"], "draw": c["draw"], "sample_seed": c["sample_seed"],
                      **{f"r_{i}": rel[i] for i in Tc}, **{f"k_{i}": c["k"][i] for i in Tc}, **{f"T_{i}": Tc[i] for i in Tc},
                      **{f"d_{i}": d[i] for i in Tc}})
        fixed = {"".join(map(str, p)): f"{src}_fixed_{''.join(map(str, p))}" for p in itertools.permutations(range(n))}
        order_of = lambda key: "".join(map(str, sorted(range(n), key=lambda i: (key(i), i))))
        refo = order_of(lambda i: c["ref_rank"][i])
        pol = {"P1": f"{src}_P1", "P2": fixed[order_of(lambda i: rel[i])], "P3": fixed[order_of(lambda i: Tc[i])],
               "P4": fixed[order_of(lambda i: d[i])], **{p: f"{src}_{p}" for p in RUN}}
        pol.update({f"fixed_{o}": v for o, v in fixed.items()})
        stat = {}
        for pn, rn in pol.items():
            rs = res[rn]["runs"]
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
                    runs_rows.append({"group": c["group"], "hw": c["hw"], "chunk": c["chunk"] or "none", "draw": c["draw"],
                                      "policy": pn, "run": rn, "seed": r["seed"], **{f"r_{i}": rel[i] for i in Tc},
                                      **{f"k_{i}": c["k"][i] for i in Tc}, **{f"C_{i}": r["C"][i] for i in Tc},
                                      "Lhat_max": l, "L_max": lr, "S_max": s_, "makespan": m, "violations": v,
                                      "wall": r["wall"]})
        best = min((p for p in stat if p.startswith("fixed_")), key=lambda p: stat[p]["Lhat"])
        stat["P5"] = dict(stat[best], member=best)
        for pn, s in stat.items():
            table.append({"config": src, "group": c["group"], "hw": c["hw"], "chunk": c["chunk"] or "none", "draw": c["draw"],
                          "policy": pn, "member": s.get("member", ""), "Lhat_max": s["Lhat"], "Lhat_range": s["Lhat_range"],
                          "L_max": s["L_max"], "L_max_range": s["L_max_range"], "S_max": s["S_max"], "S_range": s["S_range"],
                          "makespan": s["makespan"], "violations": s["violations"],
                          **{f"stretch_{i}": v for i, v in s["stretch"].items()}, "T": json.dumps(Tc),
                          "release": json.dumps(rel), "k": json.dumps(c["k"])})
        o = {"config": src, "group": c["group"], "hw": c["hw"], "draw": c["draw"], "P5_member": best, "ref_order": refo}
        for p in POL:
            o[f"L_{p}"] = stat[p]["Lhat"]
            o[f"Lrange_{p}"] = stat[p]["Lhat_range"]
            o[f"Lraw_{p}"] = stat[p]["L_max"]
            o[f"S_{p}"] = stat[p]["S_max"]
            o[f"V_{p}"] = stat[p]["violations"]
            o[f"M_{p}"] = stat[p]["makespan"]
        for p in RUN:
            for ref in ("P3", "P4", "P5"):
                o[f"rS_{p}_{ref}"] = o[f"S_{p}"] / o[f"S_{ref}"] - 1
                o[f"dL_{p}_{ref}"] = o[f"L_{p}"] - o[f"L_{ref}"]
            o[f"strict_better_P5_{p}"] = o[f"L_{p}"] < o["L_P5"] - max(o["Lrange_P5"], o[f"Lrange_{p}"])
        for a, b in (("P7m_raw", "P7"), ("P7m_raw", "P7m"), ("P7m", "P7")):
            o[f"rS_{a}_{b}"] = o[f"S_{a}"] / o[f"S_{b}"] - 1
            o[f"dL_{a}_{b}"] = o[f"L_{a}"] - o[f"L_{b}"]
            o[f"dLraw_{a}_{b}"] = o[f"Lraw_{a}"] - o[f"Lraw_{b}"]
        o["best_Lhat"] = min(POL, key=lambda p: (round(stat[p]["Lhat"], 9), p))
        o["best_L_max"] = min(POL, key=lambda p: (round(stat[p]["L_max"], 6), p))
        for p in ("P6", "P6n"):
            lg = [r["slack_log"] for r in res[pol[p]]["runs"]]
            o[f"flips_{p}"] = E._mean([x["flips"] for x in lg])
            o[f"rank_changes_{p}"] = E._mean([x["rank_changes"] for x in lg])
        rows_rep.append(o)
        r6 = res[pol["P6"]]["runs"]
        sl = {x: sum(r["slack_log"][x] for r in r6) for x in ["ticks", "rank_changes", "ticks_not_ref"]}
        sh = {x: sum(r["shadow"].get(x, 0) for r in r6) for x in ["n", "differ", "differ_stream"]}
        ent = {"slack_log": sl, "shadow": sh, "S_P6": stat["P6"]["S_max"], "ref_order": refo,
               "S_ref": stat[f"fixed_{refo}"]["S_max"]}
        if sl["ticks_not_ref"] == 0:
            ent["pass"] = sh["differ"] == 0 and abs(stat["P6"]["S_max"] / stat[f"fixed_{refo}"]["S_max"] - 1) <= 0.04
        chk["check3"][src] = ent
        for rn in [pol["P1"]] + list(fixed.values()) + [pol[p] for p in RUN]:
            for r in res[rn]["runs"]:
                for x in ("n_issues", "n_multi", "n_mismatch"):
                    chk["check4"][x] += r["audit"][x]
        w1 = E._mean([r["wall"] for r in res[pol["P1"]]["runs"]])
        chk["check7"][src] = {p: E._mean([r["wall"] for r in res[pol[p]]["runs"]]) / w1 for p in RUN}
    B.write_csv("setting3_runs.csv", runs_rows)
    B.write_csv("setting3_table.csv", table)
    B.write_csv("setting3_report.csv", rows_rep)
    B.write_csv("setting3_draws.csv", drows)
    stop = []
    if any(v.get("pass") is False for v in chk["check3"].values()):
        stop.append("check 3: " + ", ".join(n for n, v in chk["check3"].items() if v.get("pass") is False))
    if chk["check4"]["n_mismatch"] or chk["check4"]["n_issues"] < 1000:
        stop.append("check 4")
    if any(x > 10 for v in chk["check7"].values() for x in v.values()):
        stop.append("handoff 6: DSTU > 10x P1")
    for p in RUN:  # handoff 6: DSTU never differs from / always worse than BestFixed (report and ask)
        dl = [o[f"dL_{p}_P5"] for o in rows_rep]
        if all(abs(x) <= 1e-9 for x in dl):
            stop.append(f"handoff 6: {p} never differs from BestFixed")
        elif all(x > 0 for x in dl):
            stop.append(f"handoff 6: {p} always worse than BestFixed")
    out = {"T": {f"{k}|{cz}": v for (k, cz), v in T.items()}, "checks": {**chk, "check4": dict(chk["check4"]), "check5": chk5},
           "stop": stop, "walls": {"solo": t1 - t0, "runs": t2 - t1, "total": time.time() - t0}, "seeds": B.SEEDS,
           "sample_seed": SAMPLE_SEED, "N": N, "command": sys.argv, "artifact_commit": E._artifact_commit(),
           "processes": len(items)}
    B.write_json("setting3.json", out)
    B.log(json.dumps({"rows": len(cfgs), "processes": len(items), "wall_s": round(time.time() - t0),
                      "check3_applicable": sum(1 for v in chk["check3"].values() if "pass" in v),
                      "check3_fail": sum(1 for v in chk["check3"].values() if v.get("pass") is False),
                      "check4": dict(chk["check4"]),
                      "check5_median_max": round(max(max(v["median"] for v in x.values()) for x in chk5.values()), 4),
                      "check7_max": round(max(x for v in chk["check7"].values() for x in v.values()), 2), "stop": stop}))
    if stop:
        raise B.E4bError("; ".join(stop))


if __name__ == "__main__":
    try:
        main()
    except (B.E4bError, E.E4Error) as exc:
        print(f"[e4b] STOP: {exc}", flush=True)
        sys.exit(2)
