"""E4 user decision 2026-11-02: the "fully serial" reference is analytic and T_i uses 6 backend seeds.

  - T_i = mean of seeds 1-6 of the stream alone (release 0) on its platform: seeds 1-3 are the existing runs
    (step 2 / step 3 for the 2x2 no-fault streams, e4_out/sims/single_<key>_default for the rest), seeds 4-6 are
    run here (e4_out/sims/single_<key>_default_s456).
  - Every unchunked summary: stretch, S_max, deadline met per k and the priority detail are recomputed from the
    measured latencies in <config>_runs.csv divided by the 6-seed T_i (the 3-seed values are kept with suffix _T3);
    graph-level quantities (single-resource S_max bound, slack spread, makespan bounds) and the Delta release
    offsets are not recomputed (they used the 3-seed T_i).
  - Every summary gets serial_ref = {latency T_i, stretch 1, S_max 1, makespan sum T_i, beat sum T_i / n}; a
    simulated serial configuration keeps its measured values under serial_simulated and its main fields become
    the analytic values. Self-check 3 = cross-stream blocking 0 only.
  - Chunked summaries (batch 2) already use 6-seed chunked T_i: only serial_ref and the unchunked same-platform
    T_i are updated.

  python e4_serialref.py --jobs 12      # top up seeds 4-6, then rewrite the summaries, e4_table.csv, core plot
  python e4_serialref.py --no-topup     # rewrite only (fails if a key lacks seeds 4-6)
"""
import argparse
import csv
import glob
import gzip
import json
import os
import pickle
import sys
import time

import e4_grid as G
import e4_overlap as E

TOPUP = [4, 5, 6]


def _sim(name):
    p = os.path.join(G.SIM_DIR, f"{name}.pkl.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rb") as f:
        return {r["seed"]: r["cap"]["result"][0] for r in pickle.load(f)["runs"]}


def seeds_1_3(k):
    s = _sim(f"single_{k}_default")
    if s:
        return s
    with open(os.path.join(E.OUT, "step2_checks.json"), encoding="utf-8") as f:
        d = json.load(f)["checks"]["check1_single_stream"]["detail"]
    if k in d:
        return {int(sd): v for sd, v in d[k]["merged_latency"].items()}
    if k == "ffn":
        with open(os.path.join(E.OUT, "step3_chunk.json"), encoding="utf-8") as f:
            return dict(zip(E.SEEDS, json.load(f)["T_ffn_seeds"]))
    raise E.E4Error(f"no seeds 1-3 for {k}")


def platform_of(k):
    cfg = G.CFG_1X4 if "_1x4" in k else E.CFG
    faults = None
    if "__" in k:
        fl = E.load_stream(k)["faults"]
        faults = {"failed_nodes": [], "failed_links": [tuple(map(tuple, l)) for l in fl["failed_links"]]}
    return cfg, faults


def summaries():
    for p in sorted(glob.glob(os.path.join(E.OUT, "a*_summary.json"))):
        with open(p, encoding="utf-8") as f:
            yield p, json.load(f)


def keys_needed():
    ks = set()
    for _, s in summaries():
        if s.get("chunking", "none") != "none":
            continue
        for k in s["streams"]:
            ks.add(k)
            ks.add(k.split("__")[0])
    return sorted(ks)


def topup(keys, jobs):
    units = []
    for k in keys:
        cfg, fa = platform_of(k)
        units.append((f"single_{k}_default_s456", [k], {"name": "default", "release": {0: 0}}, TOPUP, cfg, fa))
    G.run_units(units, jobs)


def T6_table(keys):
    out = {}
    for k in keys:
        s = dict(seeds_1_3(k))
        s4 = _sim(f"single_{k}_default_s456")
        if not s4:
            raise E.E4Error(f"seeds 4-6 missing for {k}")
        s.update(s4)
        if sorted(s) != [1, 2, 3, 4, 5, 6]:
            raise E.E4Error(f"{k}: seeds {sorted(s)}")
        out[k] = {"seeds": {sd: s[sd] for sd in sorted(s)}, "T3": E._mean([s[i] for i in (1, 2, 3)]),
                  "T6": E._mean(list(s.values()))}
    return out


def serial_ref(T):
    tot = sum(T.values())
    return {"latency": dict(T), "stretch": {i: 1.0 for i in T}, "S_max": 1.0, "makespan": tot, "beat": tot / len(T),
            "rule": "analytic: each stream alone on the same hardware state (6-seed T_i), back to back"}


def rewrite(p, s, T6):
    n = s["n_streams"]
    name = os.path.basename(p)[:-len("_summary.json")]
    if s.get("chunking", "none") != "none":
        T = {str(i): v for i, v in s["T"].items()}
        s["serial_ref"] = serial_ref(T)
        s["T_unchunked_same_platform"] = {str(i): T6[k]["T6"] for i, k in enumerate(s["streams"])}
        E._write_json(os.path.basename(p), s)
        return s
    T = {str(i): T6[k]["T6"] for i, k in enumerate(s["streams"])}
    with open(os.path.join(E.OUT, f"{name}_runs.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if "S_max_T3" not in r:
            r["S_max_T3"] = r["S_max"]
        st = {i: float(r[f"latency_{i}"]) / T[str(i)] for i in range(n)}
        for i in range(n):
            r[f"stretch_{i}"] = st[i]
        r["S_max"] = max(st.values())
        for k in E.KS:
            r[f"all_met_k{k}"] = all(v <= k for v in st.values())
        r["T_basis"] = "6 seeds"
    E._write_csv(f"{name}_runs.csv", rows)
    d = [r for r in rows if r["variant"] == "default"]
    if "T_T3" not in s:
        for f in ["T", "Tmin", "T_ratio", "stretch_mean", "S_max_mean", "S_max_seeds", "all_met_share",
                  "prio_S_max_mean", "prio_best_order", "prio_best_over_default", "prio_detail",
                  "S_max_vs_nofault_T_mean", "T_nofault"]:
            if f in s:
                s[f + "_T3"] = s[f]
    s["T"] = T
    s["T_seeds"] = {str(i): T6[k]["seeds"] for i, k in enumerate(s["streams"])}
    s["T_basis"] = "6 seeds (user decision 2026-11-02); Delta releases and graph-level bounds used the 3-seed T"
    s["Tmin"] = min(T.values())
    s["T_ratio"] = max(T.values()) / min(T.values())
    meas = {"latency_mean": {str(i): E._mean([float(r[f"latency_{i}"]) for r in d]) for i in range(n)},
            "stretch_mean": {str(i): E._mean([float(r[f"stretch_{i}"]) for r in d]) for i in range(n)},
            "S_max_mean": E._mean([r["S_max"] for r in d]), "S_max_seeds": [r["S_max"] for r in d],
            "makespan_mean": E._mean([float(r["makespan"]) for r in d]),
            "beat_mean": E._mean([float(r["beat"]) for r in d]),
            "all_met_share": {str(k): E._mean([float(r[f"all_met_k{k}"]) for r in d]) for k in E.KS}}
    pd = E.prio_detail(rows, n)
    prio = {v: o["S_max"] for v, o in pd["orders"].items()}
    if s.get("hardware", "nofault") != "nofault":
        Tn = {str(i): T6[k.split("__")[0]]["T6"] for i, k in enumerate(s["streams"])}
        s["T_nofault"] = Tn
        s["S_max_vs_nofault_T_mean"] = E._mean([max(float(r[f"latency_{i}"]) / Tn[str(i)] for i in range(n))
                                                for r in d])
    ref = serial_ref(T)
    s["serial_ref"] = ref
    if s["stagger"] == "serial":
        s["serial_simulated"] = dict(meas, prio_S_max_mean=prio, prio_detail=pd,
                                     latency_rel_diff={str(i): abs(meas["latency_mean"][str(i)] - T[str(i)]) / T[str(i)]
                                                       for i in range(n)})
        s.update(latency_mean=ref["latency"], stretch_mean=ref["stretch"], S_max_mean=1.0, S_max_seeds=[1.0],
                 makespan_mean=ref["makespan"], beat_mean=ref["beat"],
                 all_met_share={str(k): float(k >= 1) for k in E.KS},
                 prio_S_max_mean=None, prio_best_order=None, prio_best_over_default=None, prio_detail=None)
        if "T_nofault" in s and s.get("hardware", "nofault") != "nofault":
            s["serial_simulated"]["S_max_vs_nofault_T_mean"] = s["S_max_vs_nofault_T_mean"]
            s["S_max_vs_nofault_T_mean"] = max(T[i] / s["T_nofault"][i] for i in T)
        c3 = s.get("check3", {})
        c3.update(latency_rel_diff=s["serial_simulated"]["latency_rel_diff"],
                  within_noise_stop=all(v <= E.NOISE_STOP for v in s["serial_simulated"]["latency_rel_diff"].values()))
        c3["pass"] = all(x == 0 for x in s["cross_stream_share_seeds"])
        c3["rule"] = "user decision 2026-11-02: pass = cross-stream blocking 0; latency differences recorded only"
        s["check3"] = c3
        if not c3["pass"]:
            raise E.E4Error(f"self-check 3 (cross-stream blocking) failed: {s['config']}")
    else:
        best = min(prio, key=prio.get)
        s.update(meas, prio_S_max_mean=prio, prio_best_order=best, prio_best_over_default=prio[best] / meas["S_max_mean"],
                 prio_detail=pd)
    E._write_json(os.path.basename(p), s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--no-topup", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    keys = keys_needed()
    if not args.no_topup:
        topup(keys, args.jobs)
    T6 = T6_table(keys)
    E._write_json("T6.json", {"T": T6, "command": sys.argv, "artifact_commit": E._artifact_commit()})
    n = 0
    for p, s in summaries():
        rewrite(p, s, T6)
        n += 1
    E.rebuild_table()
    G.write_core_plot()
    G.log(f"serialref: {len(keys)} keys, {n} summaries rewritten ({time.time() - t0:.0f} s)")
    G.log(json.dumps({k: [round(v["T3"]), round(v["T6"]), round(v["T6"] / v["T3"] - 1, 4)] for k, v in T6.items()}))


if __name__ == "__main__":
    try:
        main()
    except E.E4Error as exc:
        print(f"[e4] STOP: {exc}", flush=True)
        sys.exit(2)
