"""E4b setting 1 supplement (user decision 2026-11-02: normalized slack variants and the raw lateness measure on
setting 1 before setting 2). Same configurations, seeds (1-3; a2 MHA+FFN F4 short_first 1-6), T_i, releases, k,
solo references and chunking as e4b_s1.py; only the sort key changes:
  P6' (P6n)  ((d_i - t - R_i(t)) / T_i, -ln)
  P7' (P7n)  ((d_i - t - tail_e) / T_i, -ln)    tail as P7 (chunked: unissued sibling blocks included, 2026-11-02)
Every policy (P1-P7, P6', P7') also gets L_max = max_i (C_i - d_i) in cycles, from the stored C_i of each run.
Existing setting-1 runs are read from the e4b_out/sims cache (never re-run); only P6'/P7' are simulated.
Diagnosis (a1 groups, every hardware state): tail_e / T_i at the stream's release, over all events and over the
  events ready at release (no data dependency); tail_e as used by P7 (E2 DAG of the solo schedule: data edges and
  the solo run's resource-order edges) and, for comparison, the same longest path over data edges only.
Self-checks: 3 (P6' with constant normalized-slack order: shadow differ = 0 and S_max within 4% of short first),
  4 (audit), 7 (walls vs P1).

  python e4b_s1b.py --jobs 12
Writes e4b_out/setting1b_table.csv, setting1b_report.csv, setting1b_tail_diag.csv, setting1b.json.
"""
import argparse
import collections
import gzip
import itertools
import json
import os
import pickle
import statistics
import sys
import time

import e4_attr as A
import e4_hop as H
import e4_overlap as E
import e4b as B
import e4b_s1 as S1

KS = S1.KS
NEW = ["P6n", "P7n"]
POL = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P6n", "P7n"]


def cached(name):
    p = os.path.join(B.OUT, "sims", f"{name}.pkl.gz")
    if not os.path.exists(p):
        raise B.E4bError(f"setting-1 run {name} missing from the cache")
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def tail_diag(cap, sd, T):
    """tail_e / T at release: P7's tail (all DAG edges) and the data-edge-only longest path."""
    sim = A.SimR(cap)
    succ, topo = sim.dag()
    Ld = {}
    for u in reversed(topo):
        Aa = sim.act[u]
        p = Aa["f"] - Aa["s"]
        best = p
        for v, kind, rk in succ[u]:
            if rk is not None:
                continue
            c = p + Ld[v] if kind == "fs" else Ld[v]
            best = max(best, c)
        Ld[u] = best
    full, data, ready_full, ready_data = [], [], [], []
    for t, e in cap["events"].items():
        r = B.rt(t)
        term = sim.terminal[t]
        Aa = sim.act[term]
        tf = (sd["own"][r] + sd["rest"][r]) / T
        td = (sd["own"][r] + Ld[term] - (Aa["f"] - Aa["s"])) / T
        full.append(tf)
        data.append(td)
        if not e["deps"]:
            ready_full.append(tf)
            ready_data.append(td)
    q = lambda v: {"max": max(v), "median": statistics.median(v)}
    return {"n_events": len(full), "n_ready": len(ready_full), "all": q(full), "all_data_only": q(data),
            "ready": q(ready_full), "ready_data_only": q(ready_data), "T_solo_seed1_over_T": sd["T_ref_run"] / T}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()
    st, _ = S1.load()
    plats = S1.platforms(st)
    with open(os.path.join(B.OUT, "setting1.json"), encoding="utf-8") as f:
        s1 = json.load(f)
    T = {}
    for kk, v in s1["T"].items():
        k, cz = kk.rsplit("|", 1)
        T[(k, None if cz == "None" else cz)] = v
    # ---- solo references (cached setting-1 runs)
    solo_items = {}
    for g, hw, cz, keys, fa, alias in plats:
        for k in keys:
            solo_items[(k, cz)] = (f"solo_{k}_{cz or 'none'}", {"cfg": E.CFG, "faults": fa, "streams": [k], "release": {0: 0},
                                                              "seeds": B.SEEDS, "policy": {"name": "P1"},
                                                              "chunk": H.CHUNKS[cz] if cz else None, "keep_cap": True})
    sres = B.run_all(args.jobs, list(solo_items.values()))
    solo, diag = {}, []
    for (k, cz), (n, _) in solo_items.items():
        cap = next(r["cap"] for r in sres[n]["runs"] if r["seed"] == 1)
        solo[(k, cz)] = B.solo_data(cap)
    for g, hw, cz, keys, fa, alias in plats:
        if not g.startswith("a1") or alias:
            continue
        for i, k in enumerate(keys):
            cap = next(r["cap"] for r in sres[solo_items[(k, cz)][0]]["runs"] if r["seed"] == 1)
            dd = tail_diag(cap, solo[(k, cz)], T[(k, cz)])
            diag.append({"group": g, "hw": hw, "stream": i, "key": k, "T": T[(k, cz)], **{
                f"{a}_{b}": dd[a][b] for a in ("all", "all_data_only", "ready", "ready_data_only") for b in ("max", "median")},
                "n_events": dd["n_events"], "n_ready": dd["n_ready"], "T_solo_seed1_over_T": dd["T_solo_seed1_over_T"]})
    del sres
    t1 = time.time()
    # ---- configurations and the new runs
    cfgs, items = [], []
    for g, hw, cz, keys, fa, alias in plats:
        n = len(keys)
        Tc = {i: T[(k, cz)] for i, k in enumerate(keys)}
        D = int(round(0.5 * min(Tc.values())))
        for order in ["d0", "short_first", "long_first"]:
            rel = S1.releases(Tc, order, D)
            name = f"{g}_{hw}_{order}"
            cfgs.append({"name": name, "group": g, "hw": hw, "chunk": cz, "order": order, "T": Tc, "release": rel,
                         "alias_of_hw": alias, "n": n})
            if alias:
                continue
            base = {"cfg": E.CFG, "faults": fa, "streams": keys, "release": rel, "seeds": B.SEEDS,
                    "chunk": H.CHUNKS[cz] if cz else None, "audit": True}
            sd = {i: solo[(k, cz)] for i, k in enumerate(keys)}
            ref_rank = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: (Tc[i], i)))}
            for k in KS:
                d = {i: rel[i] + k * Tc[i] for i in Tc}
                pp = {"solo": sd, "d": d, "T": Tc, "release": rel, "ref_rank": ref_rank}
                for p in NEW:
                    items.append((f"{name}_{p}_k{k}", dict(base, policy=dict(pp, name=p), shadow_rank=ref_rank)))
    ext = [(f"{n}_s456", dict(s, seeds=S1.EXT_SEEDS)) for n, s in items if any(n.startswith(c + "_") for c in S1.EXT6)]
    res = B.run_all(args.jobs, items + ext)
    for n, _ in ext:
        res[n[:-len("_s456")]] = {"runs": res[n[:-len("_s456")]]["runs"] + res[n]["runs"]}
    t2 = time.time()
    # ---- tables
    table, rows_rep = [], []
    chk = {"check3": {}, "check4": collections.Counter(), "check7": {}, "k_identical": {}}
    old = {}

    def runs_of(rn):
        if rn in res:
            return res[rn]["runs"]
        if rn not in old:
            rs = cached(rn)["runs"]
            if any(rn.startswith(c + "_") for c in S1.EXT6):
                rs = rs + cached(rn + "_s456")["runs"]
            old[rn] = rs
        return old[rn]

    for c in cfgs:
        src = c["name"] if not c["alias_of_hw"] else c["name"].replace(f"_{c['hw']}_", f"_{c['alias_of_hw']}_")
        Tc, rel, n = c["T"], c["release"], c["n"]
        fixed = {"".join(map(str, p)): f"{src}_fixed_{''.join(map(str, p))}" for p in itertools.permutations(range(n))}
        sjf = "".join(map(str, sorted(range(n), key=lambda i: (Tc[i], i))))
        fcfs = "".join(map(str, sorted(range(n), key=lambda i: (rel[i], i))))
        for k in KS:
            d = {i: rel[i] + k * Tc[i] for i in Tc}
            edf = "".join(map(str, sorted(range(n), key=lambda i: (d[i], i))))
            pol = {"P1": f"{src}_P1", "P2": f"{src}_P1" if c["order"] == "d0" else fixed[fcfs], "P3": fixed[sjf],
                   "P4": fixed[edf], "P6": f"{src}_P6_k{k}", "P7": f"{src}_P7_k{k}",
                   "P6n": f"{src}_P6n_k{k}", "P7n": f"{src}_P7n_k{k}"}
            pol.update({f"fixed_{o}": v for o, v in fixed.items()})
            stat = {}
            for pn, rn in pol.items():
                rs = runs_of(rn)
                L = [B.lhat(r, Tc, d) for r in rs]
                Lr = [max(r["C"][i] - d[i] for i in Tc) for r in rs]
                S = [B.smax(r, Tc) for r in rs]
                stat[pn] = {"Lhat": E._mean(L), "Lhat_range": max(L) - min(L), "L_max": E._mean(Lr),
                            "L_max_range": max(Lr) - min(Lr), "S_max": E._mean(S), "S_range": max(S) - min(S),
                            "makespan": E._mean([max(r["C"].values()) for r in rs]),
                            "violations": E._mean([sum(1 for i in Tc if r["C"][i] > d[i]) for r in rs]),
                            "stretch": {i: E._mean([(r["C"][i] - r["release"][i]) / Tc[i] for r in rs]) for i in Tc},
                            "n_seeds": len(rs), "run": rn}
            fx = [p for p in stat if p.startswith("fixed_")]
            best = min(fx, key=lambda p: stat[p]["Lhat"])
            best_raw = min(fx, key=lambda p: stat[p]["L_max"])
            stat["P5"] = dict(stat[best], member=best, member_by_L_max=best_raw)
            for pn, s in stat.items():
                table.append({"config": c["name"], "group": c["group"], "hw": c["hw"], "chunk": c["chunk"] or "none",
                              "order": c["order"], "k": k, "policy": pn, "member": s.get("member", ""),
                              "Lhat_max": s["Lhat"], "Lhat_range": s["Lhat_range"], "L_max": s["L_max"],
                              "L_max_range": s["L_max_range"], "S_max": s["S_max"], "S_range": s["S_range"],
                              "makespan": s["makespan"], "violations": s["violations"], "n_seeds": s["n_seeds"],
                              **{f"stretch_{i}": v for i, v in s["stretch"].items()},
                              "alias_of_hw": c["alias_of_hw"] or "", "T": json.dumps(Tc), "release": json.dumps(rel)})
            o = {"config": c["name"], "group": c["group"], "hw": c["hw"], "order": c["order"], "k": k,
                 "alias_of_hw": c["alias_of_hw"] or "", "P5_member": best, "P5_member_by_L_max": best_raw}
            for p in POL:
                o[f"L_{p}"] = stat[p]["Lhat"]
                o[f"Lraw_{p}"] = stat[p]["L_max"]
                o[f"S_{p}"] = stat[p]["S_max"]
                o[f"M_{p}"] = stat[p]["makespan"]
                o[f"V_{p}"] = stat[p]["violations"]
            for p in ("P6n", "P7n", "P6", "P7"):
                for ref in ("P3", "P4", "P5", "P6", "P7"):
                    if ref != p:
                        o[f"rS_{p}_{ref}"] = o[f"S_{p}"] / o[f"S_{ref}"] - 1
            rk_hat = sorted(POL, key=lambda p: (round(stat[p]["Lhat"], 9), p))
            rk_raw = sorted(POL, key=lambda p: (round(stat[p]["L_max"], 6), p))
            o["rank_Lhat"] = " ".join(rk_hat)
            o["rank_L_max"] = " ".join(rk_raw)
            o["rank_changes"] = rk_hat != rk_raw
            o["best_Lhat"] = rk_hat[0]
            o["best_L_max"] = rk_raw[0]
            rows_rep.append(o)
            if c["alias_of_hw"]:
                continue
            for p in NEW:
                rs = res[pol[p]]["runs"]
                if p == "P6n":
                    sl = {x: sum(r["slack_log"][x] for r in rs) for x in ["ticks", "rank_changes", "ticks_not_ref"]}
                    sh = {x: sum(r["shadow"].get(x, 0) for r in rs) for x in ["n", "differ", "differ_stream"]}
                    ent = {"slack_log": sl, "shadow": sh, "S_P6n": stat["P6n"]["S_max"], "S_sjf": stat["P3"]["S_max"]}
                    if sl["ticks_not_ref"] == 0:
                        ent["pass"] = sh["differ"] == 0 and abs(stat["P6n"]["S_max"] / stat["P3"]["S_max"] - 1) <= 0.04
                    chk["check3"][f"{c['name']}_k{k}"] = ent
                for r in rs:
                    for x in ("n_issues", "n_multi", "n_mismatch"):
                        chk["check4"][x] += r["audit"][x]
        if not c["alias_of_hw"]:
            w1 = E._mean([r["wall"] for r in runs_of(f"{src}_P1")])
            chk["check7"][c["name"]] = {f"{p}_k{k}": E._mean([r["wall"] for r in res[f"{src}_{p}_k{k}"]["runs"]]) / w1
                                        for p in NEW for k in KS}
            for p in NEW:
                Cs = [[r["C"] for r in res[f"{src}_{p}_k{k}"]["runs"]] for k in KS]
                chk["k_identical"][f"{c['name']}_{p}"] = all(x == Cs[0] for x in Cs[1:])
    B.write_csv("setting1b_table.csv", table)
    B.write_csv("setting1b_report.csv", rows_rep)
    B.write_csv("setting1b_tail_diag.csv", diag)
    stop = []
    if any(v.get("pass") is False for v in chk["check3"].values()):
        stop.append("check 3")
    if chk["check4"]["n_mismatch"] or chk["check4"]["n_issues"] < 1000:
        stop.append("check 4")
    if any(x > 10 for v in chk["check7"].values() for x in v.values()):
        stop.append("handoff 6: DSTU > 10x P1")
    out = {"checks": {**chk, "check4": dict(chk["check4"])}, "tail_diag": diag, "stop": stop,
           "walls": {"solo": t1 - t0, "runs": t2 - t1, "total": time.time() - t0}, "seeds": B.SEEDS,
           "ext6": sorted(S1.EXT6), "command": sys.argv, "artifact_commit": E._artifact_commit(), "processes": len(items + ext)}
    B.write_json("setting1b.json", out)
    B.log(json.dumps({"configs": len(cfgs), "processes": len(items + ext), "wall_s": round(time.time() - t0),
                      "check3_applicable": sum(1 for v in chk["check3"].values() if "pass" in v),
                      "check3_fail": [n for n, v in chk["check3"].items() if v.get("pass") is False],
                      "check4": dict(chk["check4"]), "check7_max": round(max(x for v in chk["check7"].values() for x in v.values()), 2),
                      "k_identical": sum(chk["k_identical"].values()), "k_total": len(chk["k_identical"]), "stop": stop}))
    if stop:
        raise B.E4bError("; ".join(stop))


if __name__ == "__main__":
    try:
        main()
    except (B.E4bError, E.E4Error) as exc:
        print(f"[e4b] STOP: {exc}", flush=True)
        sys.exit(2)
