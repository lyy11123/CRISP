"""Handoff 10.2 validations of the broadcast-route cache.
  init     python validate10.py init        saved initial states of mha g0 / g1 (0-step A2 children, stage c)
  val1     python validate10.py val1        on every selected state saved by B2: T with the cache (aged from the initial
                                            state of the same g; moved producers recomputed) vs T with the full broadcast
  val2     python validate10.py val2        stage c reruns vs B2 (best by backend, median) and cached vs exact T along
                                            the search, plus T of the selected state vs the full broadcast
  time     python validate10.py time        per-step split of the serial stage s2 children (run by this script)
"""
import gzip
import json
import os
import pickle
import statistics
import subprocess
import sys

os.environ["CPCM_SA_STAGE"] = "c"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpcm_sa as C  # noqa: E402

ARMS = ["A1", "A2", "A3", "A3h"]
B2 = os.path.join(C.OUT_ROOT, "b2")
INIT = os.path.join(C.OUT, "init")
SERIAL = os.path.join(C.OUT_ROOT, "c", "serial")


def init():
    os.makedirs(INIT, exist_ok=True)
    env = dict(os.environ, CPCM_SA_STEPS="0")
    procs = [subprocess.Popen([sys.executable, os.path.join(HERE, "cpcm_sa.py"), "child", "mha", "A2", str(g), "123",
                               os.path.join(INIT, f"g{g}.pkl.gz"), os.path.join(INIT, f"g{g}_snap.pkl.gz"), "0"],
                              env=env, cwd=HERE) for g in (0, 1)]
    print([p.wait() for p in procs])


def load(path):
    with gzip.open(path, "rb") as f:
        blob, hist = pickle.load(f)
    return pickle.loads(blob), hist


def val1():
    C._artifact_path()
    hp = C.platform("mha")
    ctx = C.load_run("mha", "A2", 0, 123, B2)["meta"]["ctx"]
    inits = {g: load(os.path.join(INIT, f"g{g}_snap.pkl.gz")) for g in (0, 1)}
    rows = []
    for arm in ARMS:
        for g, x in C.variants(arm):
            exact = C.load_run("mha", arm, g, x, B2)["selected"]["T_bc"]
            cache = C.BcCache(ctx, hp)
            (b0, d0, e0), h0 = inits[g]
            hp.link_loads_dict.update(h0["link_loads_dict"])
            hp.link_loads_count.update(h0["link_loads_count"])
            cache.graph(b0, d0, e0, hp)
            (b1, d1, e1), h1 = load(os.path.join(B2, "snaps", C.run_name("mha", arm, g, x) + ".pkl.gz"))
            hp.link_loads_dict.update(h1["link_loads_dict"])
            hp.link_loads_count.update(h1["link_loads_count"])
            n0 = cache.recomputed
            T = C.cpcm_sim(cache.graph(b1, d1, e1, hp), hp)[1].T
            rows.append(dict(arm=arm, g=g, x=x, T_exact=exact, T_cached=T, err=(T - exact) / exact,
                             recomputed=cache.recomputed - n0, messages=len(cache.routes)))
            print(rows[-1], flush=True)
    abs_err = [abs(r["err"]) for r in rows]
    res = {"n": len(rows), "abs_err_median": statistics.median(abs_err), "abs_err_max": max(abs_err),
           "recomputed_median": statistics.median(r["recomputed"] for r in rows), "rows": rows}
    with open(os.path.join(C.OUT, "val1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print({k: v for k, v in res.items() if k != "rows"})


def val2():
    res = {"noise_b2": 0.0382, "arms": {}}
    for arm in ("A2", "A3h"):
        c = [C.load_run("mha", arm, g, x) for g, x in C.variants(arm)]
        b = [C.load_run("mha", arm, g, x, B2) for g, x in C.variants(arm)]
        mc, mb = [C.bmean(d["selected"]["backend"]) for d in c], [C.bmean(d["selected"]["backend"]) for d in b]
        live = [abs(r["T_cached"] - r["T_exact"]) / r["T_exact"] for d in c for r in d["live_exact"]]
        sel = [abs(d["selected"]["T_sel"] - d["selected"]["T_bc"]) / d["selected"]["T_bc"] for d in c]
        res["arms"][arm] = {
            "best_backend_c": min(mc), "best_backend_b2": min(mb), "best_rel": min(mc) / min(mb) - 1,
            "median_c": statistics.median(mc), "median_b2": statistics.median(mb),
            "median_rel": statistics.median(mc) / statistics.median(mb) - 1,
            "live_samples": len(live), "live_err_median": statistics.median(live), "live_err_max": max(live),
            "selected_err_median": statistics.median(sel), "selected_err_max": max(sel),
            "revert_check": {k: int(sum(d["check_revert"][k] for d in c)) for k in c[0]["check_revert"]},
            "bc_recomputed_median": statistics.median(d["bc_recomputed"] for d in c),
            "sa_wall_median": statistics.median(d["sa_wall"] for d in c)}
    print(json.dumps(res, indent=1))
    with open(os.path.join(C.OUT, "val2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


def time_():
    """Serial, one process at a time, stage s2 rules, no checks."""
    os.makedirs(SERIAL, exist_ok=True)
    env = dict(os.environ, CPCM_SA_STAGE="s2")
    res = {}
    for arm, x in (("A2", 123), ("A3h", 123)):
        sink = os.path.join(SERIAL, f"{arm}.pkl.gz")
        rc = subprocess.call([sys.executable, os.path.join(HERE, "cpcm_sa.py"), "child", "mha", arm, "0", str(x), sink,
                              os.path.join(SERIAL, f"{arm}_snap.pkl.gz"), "0"], env=env, cwd=HERE,
                             stdout=subprocess.DEVNULL)
        with gzip.open(sink, "rb") as f:
            d = pickle.load(f)
        rows = d["rows"]
        keys = ["t_pre", "t_move", "t_target", "t_bcast", "t_energy", "t_revert", "t_snapshot"]
        res[arm] = {"rc": rc, "steps": len(rows), "sa_wall": d["sa_wall"],
                    "per_step_ms": 1000 * d["sa_wall"] / len(rows),
                    "split_ms_per_step": {k: 1000 * sum(r.get(k, 0.0) for r in rows) / len(rows) for k in keys},
                    "bcast_median_ms": 1000 * statistics.median(r["t_bcast"] for r in rows),
                    "bc_recomputed": d["bc_recomputed"]}
        print(arm, res[arm], flush=True)
    with open(os.path.join(C.OUT_ROOT, "c", "val3.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    {"init": init, "val1": val1, "val2": val2, "time": time_}[sys.argv[1]]()
