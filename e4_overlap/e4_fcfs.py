"""E4: slack gain vs first-come-first-served gain (user request 2026-11-02, "加这组测量"; measurement only, per-hop
arbitration backend e4_backend_diag mode "perhop", in memory).

Groups (two streams; stream 0 = the short one, stream 1 = the long one in every run):
  a1_2s_512-8192, a1_2s_2048-8192 (unchunked, plus 256 KB D2D-only as the optional extra), a2_2s_mha-ffn
  (256 KB D2D-only). Hardware: no fault, F4 (batch-1 fault and R0 graphs).
Release orders: short_first (r0 = 0, r1 = Delta) and long_first (r1 = 0, r0 = Delta), Delta in {0.25T, 0.5T},
  T = the short stream alone (per hop, same chunking, same platform, 6 seeds); Delta = 0 once as reference.
Arbitration (stream priority first key, remaining branch length second): default (no stream priority),
  short first (prio_01), long first (prio_10). FCFS = the order of release, i.e. prio_01 for short_first and
  prio_10 for long_first (the same run, not simulated twice); at Delta = 0 FCFS degenerates to default.
Static slack order (handoff 3.3): at the overlap start t = the later release, slack_j = r_j + k T_j - t -
  max(0, T_j - (t - r_j)), k in {1.1, 1.5, 2.0}; the smaller slack goes first; its S_max = that fixed order's.
Backend seeds 1-3. Writes e4_out/fcfs/.

  python e4_fcfs.py --jobs 12
"""
import argparse
import json
import os
import sys
import time

import e4_grid as G
import e4_hop as H
import e4_overlap as E

FCFS = os.path.join(E.OUT, "fcfs")
KS_SLACK = [1.1, 1.5, 2.0]
GROUPS = {"a1_2s_512-8192": ["none", "256d2d"], "a1_2s_2048-8192": ["none", "256d2d"], "a2_2s_mha-ffn": ["256d2d"]}
VARS = [("default", None), ("prio_01", {0: 0, 1: 1}), ("prio_10", {0: 1, 1: 0})]


def slack_order(rel, T, k):
    t = max(rel.values())
    sl = {j: rel[j] + k * T[j] - t - max(0.0, T[j] - (t - rel[j])) for j in rel}
    first = min(sl, key=lambda j: (sl[j], j))
    return ("prio_01" if first == 0 else "prio_10"), sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(FCFS, exist_ok=True)
    state, Torig = H.load_state()
    t0 = time.time()
    plat = {}
    for g in GROUPS:
        keys = G.GROUPS[g][1]
        fl = state["faults"][g]["F4"]
        plat[(g, "nofault")] = (keys, None)
        plat[(g, "F4")] = ([G.fault_key(k, fl) for k in keys], {"failed_nodes": [], "failed_links": fl["failed_links"]})
    # T per (key, chunking): unchunked = e4_out/hop singles, chunked = e4_out/diag chunk singles (6 seeds each)
    singles = {}
    for (g, hw), (keys, fa) in plat.items():
        for cz in GROUPS[g]:
            for k in keys:
                if cz == "none":
                    singles[(k, cz)] = (f"single_{k}_hop", H.HOP, {"streams": [k], "cfg": E.CFG, "faults": fa,
                                                                  "release": {0: 0}, "seeds": G.D1_SEEDS, "mode": "perhop"})
                else:
                    singles[(k, cz)] = (f"chunk_single_{k}_{cz}", H.DIAG, {"streams": [k], "cfg": E.CFG, "faults": fa,
                                                                          "release": {0: 0}, "seeds": G.D1_SEEDS,
                                                                          "mode": "perhop", "chunk": H.CHUNKS[cz]})
    sres = H.run_all(args.jobs, list(singles.values()))
    T = {kc: E._mean([r["C"][0] for r in sres[it[0]]["runs"]]) for kc, it in singles.items()}
    Tseeds = {f"{k}|{cz}": [r["C"][0] for r in sres[it[0]]["runs"]] for (k, cz), it in singles.items()}
    t1 = time.time()
    cfgs = []
    for (g, hw), (keys, fa) in plat.items():
        for cz in GROUPS[g]:
            Ts = {i: T[(k, cz)] for i, k in enumerate(keys)}
            Tmin = min(Ts.values())
            for order, st in [("both", 0.0)] + [(o, s) for o in ["short_first", "long_first"] for s in [0.25, 0.5]]:
                d = int(round(st * Tmin))
                rel = {0: 0, 1: d} if order in ("both", "short_first") else {0: d, 1: 0}
                name = f"{g}_{hw}_{cz}_{order}_d{st:g}"
                cfgs.append({"name": name, "group": g, "hw": hw, "chunk": cz, "order": order, "stagger": st, "Delta": d,
                             "streams": keys, "faults": fa, "release": rel, "T": Ts})
    items = [(f"{c['name']}_{vn}", FCFS, {"streams": c["streams"], "cfg": E.CFG, "faults": c["faults"],
                                          "release": c["release"], "prio": pr, "seeds": E.SEEDS, "mode": "perhop",
                                          "chunk": H.CHUNKS[c["chunk"]]}) for c in cfgs for vn, pr in VARS]
    res = H.run_all(args.jobs, items)
    t2 = time.time()
    rows = []
    for c in cfgs:
        per = {}
        for vn, _ in VARS:
            ms = [H.measures(r, c["T"]) for r in res[f"{c['name']}_{vn}"]["runs"]]
            per[vn] = {"S_max": E._mean([m["S_max"] for m in ms]), "S_max_seeds": [m["S_max"] for m in ms],
                       "stretch": {i: E._mean([m["stretch"][i] for m in ms]) for i in range(2)},
                       "makespan": E._mean([m["makespan"] for m in ms]),
                       "eval_wall": [r["wall"] for r in res[f"{c['name']}_{vn}"]["runs"]]}
        fc = "default" if c["order"] == "both" else ("prio_01" if c["order"] == "short_first" else "prio_10")
        sl = {str(k): slack_order(c["release"], c["T"], k) for k in KS_SLACK}
        s = dict({k: v for k, v in c.items() if k != "faults"}, per_variant=per, fcfs_variant=fc,
                 fcfs_note="Delta = 0: FCFS degenerates to default" if c["order"] == "both" else "FCFS = release order",
                 slack={k: {"order": o, "slack": v, "S_max": per[o]["S_max"],
                            "extra_gain_over_fcfs": 1 - per[o]["S_max"] / per[fc]["S_max"]} for k, (o, v) in sl.items()})
        E._write_json(os.path.join("fcfs", f"{c['name']}.json"), s)
        row = {"config": c["name"], "group": c["group"], "hw": c["hw"], "chunk": c["chunk"], "order": c["order"],
               "stagger": c["stagger"], "Delta": c["Delta"], "T_0": c["T"][0], "T_1": c["T"][1],
               "S_default": per["default"]["S_max"], "S_fcfs": per[fc]["S_max"], "S_short_first": per["prio_01"]["S_max"],
               "S_long_first": per["prio_10"]["S_max"], "fcfs_variant": fc}
        for k in KS_SLACK:
            row[f"slack_k{k}_order"] = s["slack"][str(k)]["order"]
            row[f"S_slack_k{k}"] = s["slack"][str(k)]["S_max"]
            row[f"slack_extra_gain_k{k}"] = s["slack"][str(k)]["extra_gain_over_fcfs"]
        for vn, _ in VARS:
            row[f"stretch0_{vn}"] = per[vn]["stretch"][0]
            row[f"stretch1_{vn}"] = per[vn]["stretch"][1]
            row[f"makespan_{vn}"] = per[vn]["makespan"]
        rows.append(row)
    E._write_csv(os.path.join("fcfs", "fcfs_table.csv"), rows)
    E._write_json(os.path.join("fcfs", "fcfs_state.json"), {
        "T": {f"{k}|{cz}": v for (k, cz), v in T.items()}, "T_seeds": Tseeds, "singles_wall": t1 - t0,
        "configs_wall": t2 - t1, "n_processes": len(items), "seeds": E.SEEDS, "command": sys.argv,
        "eval_wall_max": max(r["wall"] for v in res.values() for r in v["runs"]), "artifact_commit": E._artifact_commit()})
    G.log(f"fcfs done: {len(cfgs)} configs, {len(items)} processes ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    try:
        main()
    except E.E4Error as exc:
        print(f"[e4] STOP: {exc}", flush=True)
        sys.exit(2)
