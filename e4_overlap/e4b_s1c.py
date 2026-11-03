"""E4b setting 1, second supplement (user decision 2026-11-02): the corrected variant P7m and stream-order flips.
Same configurations, seeds (1-3; a2 MHA+FFN F4 short_first 1-6), T_i, releases, k, solo references and chunking as
e4b_s1.py / e4b_s1b.py.
  P7m  ((d_i - t - max(R_i(t), tail_e)) / T_i, -ln)    R_i as P6', tail_e as P7' (chunked correction included)
Flips (e4b.DstuS.flip_count): on arbitration ticks where >= 2 streams have a ready communication, the streams are
  ordered by the policy's stream key (P6: slack, P6': slack / T); a pair of streams whose order swaps between two
  consecutive such ticks is one flip. Reported per run: flips, such ticks, first / last flip time, and how many
  flips had both / one slack already negative. P6 and P6' are re-run (names *_flip) only on
  group A: k = 1.1 configurations where P6 is worse than BestFixed by > 4% (setting1b_report.csv);
  group B: k = 1.5 configurations where P6 is within 4% of BestFixed.
Existing runs are read from the e4b_out/sims cache.

  python e4b_s1c.py --jobs 12
Writes e4b_out/setting1c_table.csv, setting1c_report.csv, setting1c_flips.csv, setting1c.json.
"""
import argparse
import collections
import csv
import itertools
import json
import os
import statistics
import sys
import time

import e4_hop as H
import e4_overlap as E
import e4b as B
import e4b_s1 as S1
import e4b_s1b as S1b

KS = S1.KS
POL = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P6n", "P7n", "P7m"]


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
    with open(os.path.join(B.OUT, "setting1b_report.csv"), encoding="utf-8") as f:
        rep = [r for r in csv.DictReader(f) if not r["alias_of_hw"]]
    grpA = {r["config"] for r in rep if r["k"] == "1.1" and float(r["rS_P6_P5"]) > 0.04}
    grpB = {r["config"] for r in rep if r["k"] == "1.5" and abs(float(r["rS_P6_P5"])) <= 0.04}
    flipk = [(c, 1.1, "A") for c in sorted(grpA)] + [(c, 1.5, "B") for c in sorted(grpB)]
    # ---- solo references (cached)
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
                items.append((f"{name}_P7m_k{k}", dict(base, policy=dict(pp, name="P7m"), shadow_rank=ref_rank)))
                if (name, k) in {(c, kk) for c, kk, _ in flipk}:
                    for p in ("P6", "P6n"):
                        items.append((f"{name}_{p}_k{k}_flip", dict(base, policy=dict(pp, name=p), shadow_rank=ref_rank)))
    ext = [(f"{n}_s456", dict(s, seeds=S1.EXT_SEEDS)) for n, s in items if any(n.startswith(c + "_") for c in S1.EXT6)]
    res = B.run_all(args.jobs, items + ext)
    for n, _ in ext:
        res[n[:-len("_s456")]] = {"runs": res[n[:-len("_s456")]]["runs"] + res[n]["runs"]}
    t2 = time.time()
    old = {}

    def runs_of(rn):
        if rn in res:
            return res[rn]["runs"]
        if rn not in old:
            rs = S1b.cached(rn)["runs"]
            if any(rn.startswith(c + "_") for c in S1.EXT6):
                rs = rs + S1b.cached(rn + "_s456")["runs"]
            old[rn] = rs
        return old[rn]

    table, rows_rep, flips = [], [], []
    chk = {"check4": collections.Counter(), "check7": {}, "flip_rerun_identical": {}}
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
                   "P4": fixed[edf], **{p: f"{src}_{p}_k{k}" for p in ("P6", "P7", "P6n", "P7n", "P7m")}}
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
            stat["P5"] = dict(stat[best], member=best)
            for pn, s in stat.items():
                table.append({"config": c["name"], "group": c["group"], "hw": c["hw"], "chunk": c["chunk"] or "none",
                              "order": c["order"], "k": k, "policy": pn, "member": s.get("member", ""),
                              "Lhat_max": s["Lhat"], "Lhat_range": s["Lhat_range"], "L_max": s["L_max"],
                              "L_max_range": s["L_max_range"], "S_max": s["S_max"], "S_range": s["S_range"],
                              "makespan": s["makespan"], "violations": s["violations"], "n_seeds": s["n_seeds"],
                              **{f"stretch_{i}": v for i, v in s["stretch"].items()},
                              "alias_of_hw": c["alias_of_hw"] or "", "T": json.dumps(Tc), "release": json.dumps(rel)})
            o = {"config": c["name"], "group": c["group"], "hw": c["hw"], "order": c["order"], "k": k,
                 "alias_of_hw": c["alias_of_hw"] or "", "P5_member": best}
            for p in POL:
                o[f"L_{p}"] = stat[p]["Lhat"]
                o[f"Lraw_{p}"] = stat[p]["L_max"]
                o[f"S_{p}"] = stat[p]["S_max"]
                o[f"V_{p}"] = stat[p]["violations"]
                o[f"M_{p}"] = stat[p]["makespan"]
            for ref in ("P3", "P5", "P6", "P6n", "P7", "P7n"):
                o[f"rS_P7m_{ref}"] = o["S_P7m"] / o[f"S_{ref}"] - 1
            rows_rep.append(o)
            if c["alias_of_hw"]:
                continue
            for r in res[pol["P7m"]]["runs"]:
                for x in ("n_issues", "n_multi", "n_mismatch"):
                    chk["check4"][x] += r["audit"][x]
            for cn, kk, grp in flipk:
                if cn != c["name"] or kk != k:
                    continue
                for p in ("P6", "P6n"):
                    rs = res[f"{src}_{p}_k{k}_flip"]["runs"]
                    chk["flip_rerun_identical"][f"{cn}_{p}_k{k}"] = [r["C"] for r in rs] == [r["C"] for r in runs_of(pol[p])]
                    for r in rs:
                        lg = r["slack_log"]
                        mk = max(r["C"].values())
                        for x in ("n_issues", "n_multi", "n_mismatch"):
                            chk["check4"][x] += r["audit"][x]
                        flips.append({"config": cn, "group_flip": grp, "k": k, "policy": p, "seed": r["seed"],
                                      "flips": lg["flips"], "flip_ticks": lg["flip_ticks"],
                                      "flips_both_neg": lg["flips_both_neg"], "flips_one_neg": lg["flips_one_neg"],
                                      "first_flip_frac": None if lg["first_flip_t"] is None else lg["first_flip_t"] / mk,
                                      "last_flip_frac": None if lg["last_flip_t"] is None else lg["last_flip_t"] / mk,
                                      "makespan": mk, "S_max": B.smax(r, Tc), "Lhat_max": B.lhat(r, Tc, d),
                                      "rS_P6_P5_k": float(next(x["rS_P6_P5"] for x in rep if x["config"] == cn and float(x["k"]) == k))})
        if not c["alias_of_hw"]:
            w1 = E._mean([r["wall"] for r in runs_of(f"{src}_P1")])
            chk["check7"][c["name"]] = {f"P7m_k{k}": E._mean([r["wall"] for r in res[f"{src}_P7m_k{k}"]["runs"]]) / w1 for k in KS}
    B.write_csv("setting1c_table.csv", table)
    B.write_csv("setting1c_report.csv", rows_rep)
    B.write_csv("setting1c_flips.csv", flips)
    stop = []
    if chk["check4"]["n_mismatch"] or chk["check4"]["n_issues"] < 1000:
        stop.append("check 4")
    if any(x > 10 for v in chk["check7"].values() for x in v.values()):
        stop.append("handoff 6: DSTU > 10x P1")
    out = {"checks": {**chk, "check4": dict(chk["check4"])}, "group_A": sorted(grpA), "group_B": sorted(grpB), "stop": stop,
           "walls": {"solo": t1 - t0, "runs": t2 - t1, "total": time.time() - t0}, "seeds": B.SEEDS,
           "ext6": sorted(S1.EXT6), "command": sys.argv, "artifact_commit": E._artifact_commit(), "processes": len(items + ext)}
    B.write_json("setting1c.json", out)
    B.log(json.dumps({"processes": len(items + ext), "groupA": len(grpA), "groupB": len(grpB), "wall_s": round(time.time() - t0),
                      "check4": dict(chk["check4"]), "check7_max": round(max(x for v in chk["check7"].values() for x in v.values()), 2),
                      "flip_rerun_identical": sum(chk["flip_rerun_identical"].values()),
                      "flip_rerun_total": len(chk["flip_rerun_identical"]), "stop": stop}))
    if stop:
        raise B.E4bError("; ".join(stop))


if __name__ == "__main__":
    try:
        main()
    except (B.E4bError, E.E4Error) as exc:
        print(f"[e4b] STOP: {exc}", flush=True)
        sys.exit(2)
