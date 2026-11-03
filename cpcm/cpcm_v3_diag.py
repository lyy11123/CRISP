"""V3 diagnostic (user 2026-11-03; NOT a CPCM mode): on the V3 states of opt, mha, ffn14, run bb mode with random
ties 8 times (cpcm.simulate(rng=...)), and compare tau of the 8-run mean with the deterministic CPCM.

graphs  re-runs each V3 trajectory (same SA, same add_broadcast seed 0) with no backend, keeping the compiled
        graphs (V3 did not store them); checks the deterministic T of every regenerated graph against V3's.
analyse taus vs the V3 backend mean / median / min; for opt, discrete levels per state: the state's 4 backend
        values + CPCM + 8 random runs, sorted, split where consecutive values differ by more than LEVEL_GAP of
        their median; reports the level of CPCM (0 = lowest), and how many levels the backend seeds and the
        random runs occupy.
Usage: python cpcm_v3_diag.py graphs ; python cpcm_v3_diag.py analyse
"""
import gzip, json, os, pickle, random, statistics as st, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpcm  # noqa: E402
import cpcm_v3  # noqa: E402
from cpcm_v3 import kendall_tau_b  # noqa: E402

TASKS = ["opt", "mha", "ffn14"]
K_RANDOM = 8
RNG_SEEDS = list(range(101, 101 + K_RANDOM))
LEVEL_GAP = 0.08
OUT = cpcm_v3.OUT


def graphs_one(name):
    every = cpcm_v3.PLAN[name][2]
    sink = os.path.join(OUT, f"{name}_graphs.pkl.gz")
    log = os.path.join(OUT, "logs", f"{name}_graphs.log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        rc = subprocess.call([sys.executable, os.path.join(HERE, "cpcm_v3.py"), "child", "--name", name, "--part", "0",
                              "--nparts", "1", "--every", str(every), "--sink", sink, "--keep-graph"],
                             cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[v3diag] graphs {name}: exit {rc}, {time.time() - t0:.0f} s", flush=True)


def load_v3(name):
    P = cpcm_v3.PLAN[name][4]
    rows = [r for p in range(P) for r in pickle.load(gzip.open(os.path.join(OUT, f"{name}_part{p}.pkl.gz")))["rows"]]
    return {r["step"]: r for r in rows}


def levels(vals, gap):
    s = sorted(vals)
    out, cur = [], [s[0]]
    for x in s[1:]:
        if x - cur[-1] > gap * st.median(s):
            out.append(cur)
            cur = [x]
        else:
            cur.append(x)
    out.append(cur)
    return out


def level_of(x, lv):
    return next(i for i, c in enumerate(lv) if c[0] <= x <= c[-1])


def analyse():
    res = {}
    for name in TASKS:
        v3 = load_v3(name)
        gs = pickle.load(gzip.open(os.path.join(OUT, f"{name}_graphs.pkl.gz")))["rows"]
        seeds = cpcm_v3.PLAN[name][3]
        same = sum(1 for r in gs if r["cpcm_T"] == v3[r["step"]]["cpcm_T"])
        rows = []
        t0 = time.time()
        for r in gs:
            g = r["graph"]
            rnd = [cpcm.simulate(g, "bb", rng=random.Random(s)).T for s in RNG_SEEDS]
            b = [v3[r["step"]]["backend"][s][0] for s in seeds]
            rows.append({"step": r["step"], "energy": v3[r["step"]]["energy"], "cpcm": v3[r["step"]]["cpcm_T"],
                         "cpcm_regen": r["cpcm_T"], "rand": rnd, "backend": b})
        t_rand = time.time() - t0
        mean = [st.mean(x["backend"]) for x in rows]
        med = [st.median(x["backend"]) for x in rows]
        mn = [min(x["backend"]) for x in rows]
        cp = [x["cpcm"] for x in rows]
        rm = [st.mean(x["rand"]) for x in rows]
        en = [x["energy"] for x in rows]
        d = {"n": len(rows), "regen_T_equal_v3": same, "random_runs_s": round(t_rand, 1),
             "tau_energy_mean": kendall_tau_b(en, mean), "tau_cpcm_mean": kendall_tau_b(cp, mean),
             "tau_rand8_mean": kendall_tau_b(rm, mean),
             "tau_energy_median": kendall_tau_b(en, med), "tau_cpcm_median": kendall_tau_b(cp, med),
             "tau_rand8_median": kendall_tau_b(rm, med),
             "tau_energy_min": kendall_tau_b(en, mn), "tau_cpcm_min": kendall_tau_b(cp, mn),
             "tau_rand8_min": kendall_tau_b(rm, mn),
             "tau_rand8min_min": kendall_tau_b([min(x["rand"]) for x in rows], mn),
             "err_rand8_abs_median": st.median(abs(a - b) / b for a, b in zip(rm, mean)),
             "err_rand8_abs_max": max(abs(a - b) / b for a, b in zip(rm, mean)),
             "err_cpcm_abs_median": st.median(abs(a - b) / b for a, b in zip(cp, mean)),
             "err_cpcm_abs_max": max(abs(a - b) / b for a, b in zip(cp, mean)),
             "rand_range_over_mean_median": st.median((max(x["rand"]) - min(x["rand"])) / st.mean(x["rand"]) for x in rows),
             "rand_range_over_mean_max": max((max(x["rand"]) - min(x["rand"])) / st.mean(x["rand"]) for x in rows)}
        if name == "opt":
            per = []
            for x in rows:
                lv = levels(x["backend"] + [x["cpcm"]] + x["rand"], LEVEL_GAP)
                per.append({"step": x["step"], "n_levels": len(lv), "cpcm_level": level_of(x["cpcm"], lv),
                            "backend_levels": sorted({level_of(v, lv) for v in x["backend"]}),
                            "rand_levels": sorted({level_of(v, lv) for v in x["rand"]}),
                            "level_means": [round(st.mean(c)) for c in lv]})
            multi = [p for p in per if p["n_levels"] > 1]
            d["opt_levels"] = {
                "states": len(per), "states_with_more_than_one_level": len(multi),
                "cpcm_level_counts_in_multi": {str(k): sum(p["cpcm_level"] == k for p in multi) for k in range(4)},
                "cpcm_on_lowest_level_in_multi": sum(p["cpcm_level"] == 0 for p in multi),
                "cpcm_level_shared_with_a_backend_seed_in_multi": sum(p["cpcm_level"] in p["backend_levels"] for p in multi),
                "rand_levels_count_hist_all": {str(k): sum(len(p["rand_levels"]) == k for p in per) for k in range(1, 5)},
                "backend_levels_count_hist_all": {str(k): sum(len(p["backend_levels"]) == k for p in per) for k in range(1, 5)},
                "level_gap_rule": f"split where consecutive sorted values differ by > {LEVEL_GAP} x median of the state's values"}
            d["opt_per_state"] = per
        res[name] = d
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "opt_per_state"})
    with open(os.path.join(HERE, "cpcm_out", "v3_diag.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "graphs":
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(graphs_one, TASKS))
    else:
        analyse()
