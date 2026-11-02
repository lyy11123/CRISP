"""E4 step 4, batch 2 (user decision 2026-11-01): chunking spot check. 256 KB blocks, only messages whose route
uses a D2D (ch2ch) link; 2-stream asymmetric compositions plus the symmetric control (a1 and a2); Delta = 0 and
0.5T; no fault and F4 (the batch-1 faults and R0 graphs); 3 backend seeds; default arbitration and every stream
priority order. T_i = the stream alone with the same chunking on the same platform (3 seeds).

  python e4_grid2.py --estimate      # time estimate from batch 1's measured unit times, no simulation
  python e4_grid2.py --jobs 12
"""
import argparse
import glob
import gzip
import json
import os
import pickle
import shutil
import sys
import time

import e4_grid as G
import e4_overlap as E

CHUNK = {"size": 256 * 1024, "only_d2d": True}
CTAG = "c256d2d"
GROUPS2 = ["a1_2s_512-8192", "a1_2s_2048-8192", "a1_2s_2048-2048", "a2_2s_mha-ffn"]
STAG2 = [0.0, 0.5]
# measured in step 3 (seed 1, Delta 0): chunked / unchunked single-evaluation wall
FACTOR = {"ffn": 205.2 / 17.1, "mha": None}


def configs(state, T):
    out = []
    for g in GROUPS2:
        comp, keys = G.GROUPS[g]
        for hw in ["nofault", "F4"]:
            if hw == "nofault":
                ks, fa = keys, None
            else:
                fl = state["faults"][g]["F4"]
                ks, fa = [G.fault_key(k, fl) for k in keys], {"failed_nodes": [], "failed_links": fl["failed_links"]}
            for c in G.make_configs(g, comp, ks, T, hw, fa, E.CFG, staggers=STAG2, seeds_override=list(E.SEEDS)):
                c["name"] = c["name"] + f"_{CTAG}"
                c["chunking"] = CTAG
                for v in c["variants"]:
                    v["chunk"] = dict(CHUNK)
                out.append(c)
    return out


def estimate(state):
    """Process-seconds of batch 2 from the measured batch-1 unit times of the same configurations (unchunked),
    scaled by the step-3 chunked/unchunked factor (FFN-containing: measured 12.0; MHA-only: 68.0 s chunked at
    4-way vs 152-161 s unchunked at 12-way in step 2, not comparable, so 1.0-1.5 is assumed)."""
    tot_lo = tot_hi = 0
    rows = []
    for g in GROUPS2:
        for hw in ["", "_F4"]:
            for st in STAG2:
                base = f"{g}{hw}_{G.label_of(st)}"
                walls = []
                for p in glob.glob(os.path.join(G.SIM_DIR, f"{base}_*.pkl.gz")):
                    if p.endswith("_faultsel_default.pkl.gz"):
                        continue
                    with gzip.open(p, "rb") as f:
                        walls += [r["wall"] for r in pickle.load(f)["runs"]]
                if not walls:  # a1 2-stream no fault: step 2's runs.csv
                    import csv
                    with open(os.path.join(E.OUT, f"{base}_runs.csv"), encoding="utf-8") as f:
                        walls = [float(r["sim_wall"]) for r in csv.DictReader(f)]
                ffn = "ffn" in g
                lo, hi = (FACTOR["ffn"], FACTOR["ffn"]) if ffn else (1.0, 1.5)
                s = sum(walls)
                rows.append({"config": base, "n_runs": len(walls), "unchunked_s": s, "lo": s * lo, "hi": s * hi})
                tot_lo += s * lo
                tot_hi += s * hi
    singles = 4 * 2 * 6 * 250  # 4 streams x 2 states x 6 seeds, up to ~250 s each (FFN chunked)
    return {"rows": rows, "process_s_lo": tot_lo + singles, "process_s_hi": tot_hi + singles,
            "wall_h_lo_12": (tot_lo + singles) / 12 / 3600, "wall_h_hi_12": (tot_hi + singles) / 12 / 3600}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()
    with open(os.path.join(E.OUT, "grid1_state.json"), encoding="utf-8") as f:
        state = json.load(f)
    # faults back to tuples (JSON lists)
    for g, fs in state["faults"].items():
        for fl in fs.values():
            fl["failed_links"] = [tuple(map(tuple, l)) for l in fl["failed_links"]]
    if args.estimate:
        est = estimate(state)
        E._write_json("grid2_estimate.json", est)
        print("[e4] " + json.dumps({k: v for k, v in est.items() if k != "rows"}), flush=True)
        return
    t0 = time.time()
    keys = sorted({k for g in GROUPS2 for k in G.GROUPS[g][1]})
    fkeys = sorted({G.fault_key(k, state["faults"][g]["F4"]) for g in GROUPS2 for k in G.GROUPS[g][1]})
    faults_of = {k: None for k in keys}
    for g in GROUPS2:
        fl = state["faults"][g]["F4"]
        for k in G.GROUPS[g][1]:
            faults_of[G.fault_key(k, fl)] = {"failed_nodes": [], "failed_links": fl["failed_links"]}
    # user decision 2026-11-02: T_i from 6 seeds; no serial configuration (analytic reference, e4_serialref.py)
    units = [(f"single_{k}_{CTAG}", [k], {"name": "default", "release": {0: 0}, "chunk": dict(CHUNK)},
              list(G.D1_SEEDS), E.CFG, faults_of[k]) for k in keys + fkeys]
    res = G.run_units(units, args.jobs)
    T = {k: E._mean([r["cap"]["result"][0] for r in res[f"single_{k}_{CTAG}"]["runs"]]) for k in keys + fkeys}
    T_seeds = {k: [r["cap"]["result"][0] for r in res[f"single_{k}_{CTAG}"]["runs"]] for k in keys + fkeys}
    chunk_stats = {k: res[f"single_{k}_{CTAG}"]["runs"][0]["chunk_stats"] for k in keys + fkeys}
    cfgs = configs(state, T)
    G.log(f"batch 2: {len(cfgs)} configs")
    sims = G.simulate_configs(cfgs, args.jobs)
    an = G.analyse_all(cfgs, sims, args.jobs)
    with open(os.path.join(E.OUT, "T6.json"), encoding="utf-8") as f:  # written by e4_serialref.py
        Tn = {k: v["T6"] for k, v in json.load(f)["T"].items()}  # unchunked 6-seed T on the same platform
    G.summarize(cfgs, sims, an, Tn)
    # chunk overhead per configuration (default arbitration, seed 1) into the summary
    for c in cfgs:
        r = next(x for x in sims[c["name"]]["runs"] if x["variant"] == "default" and x["seed"] == 1)
        p = os.path.join(E.OUT, f"{c['name']}_summary.json")
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        s["chunk"] = CHUNK
        s["chunk_stats_seed1"] = r["chunk_stats"]
        s["T_unchunked_same_platform"] = {i: Tn[k] for i, k in enumerate(c["streams"])}
        E._write_json(os.path.basename(p), s)
    # a2 symmetric control = a1 symmetric (same graphs)
    for hw in ["", "_F4"]:
        for st in STAG2:
            src = os.path.join(E.OUT, f"a1_2s_2048-2048{hw}_{G.label_of(st)}_{CTAG}_summary.json")
            with open(src, encoding="utf-8") as f:
                s = json.load(f)
            name = f"a2_2s_sym{hw}_{G.label_of(st)}_{CTAG}"
            s.update(config=name, composition="a2", alias_of=s["config"])
            E._write_json(f"{name}_summary.json", s)
            shutil.copyfile(src.replace("_summary.json", "_runs.csv"), os.path.join(E.OUT, f"{name}_runs.csv"))
    E.rebuild_table()
    G.write_core_plot()
    E._write_json("grid2_state.json", {"T_chunked": T, "T_chunked_seeds": T_seeds, "single_chunk_stats": chunk_stats,
                                       "configs": [c["name"] for c in cfgs], "wall": time.time() - t0,
                                       "command": sys.argv, "artifact_commit": E._artifact_commit()})
    G.log(f"batch 2 done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    try:
        main()
    except E.E4Error as exc:
        print(f"[e4] STOP: {exc}", flush=True)
        sys.exit(2)
