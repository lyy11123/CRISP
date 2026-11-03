"""V5: multi-stream and hop mode (cpcm/CPCM_HANDOFF.md section 3). Record, do not stop.

Configurations: E4b setting 1 (e4_overlap/e4b_s1.py, imported read-only): groups a1 512+8192, a1 2048+8192,
a1 512/2048/8192, a1 2048x2 (unchunked) and a2 MHA+FFN (256 KB, D2D links only), hardware no fault / F4 / F5
(aliases skipped), releases d0 / short_first / long_first (0.5 T); k = 1.5 only (d_i = r_i + 1.5 T_i, T_i as E4b).
Per configuration, one child process (E4's merged streams on E4's platform; stream i = tag[1] - 10, released at r_i):
  pass 1  CPCM bb -> Analysis with every event's base bound = d_(its stream) -> D0(b)
  pass 2  CPCM hop, prio = D0                       (the two-pass estimate)
  pass 3  Analysis of pass 2 (same d) -> D1, hop with prio D1   (one more iteration: does it change?)
  also    CPCM hop with no prio (= E4b P1), compared with E4b's cached P1 backend runs of the same configuration
  backend e4b_backend.event_driver_e4b with a policy whose key is (D0 of the event, -ln) (the same static key as
          pass 2; ties: the backend's shuffle), seeds 1-3.
Ideal comparison (E4b's solo reference, e4b.solo_data on the seed-1 solo cap of every stream): D0(b) against the
ideal latest finish d_i - rest(b); R_hat_i at t = 0 (= T_i) against CPCM's solo estimate (hop, no prio).
Metrics: C_i (stream completion = max end over its events), makespan, L_hat_max = max_i (C_i - d_i) / T_i.

Usage: python cpcm_v5.py run --jobs 12 ; python cpcm_v5.py analyse
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
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
E4 = os.path.join(HERE, "..", "e4_overlap")
sys.path.insert(0, HERE)
sys.path.insert(0, E4)
import cpcm  # noqa: E402
import e4_overlap as E  # noqa: E402
import e4_hop as H  # noqa: E402
import e4b as B  # noqa: E402
import e4b_s1 as S1  # noqa: E402

OUT = os.path.join(HERE, "cpcm_out", "v5")
K = 1.5
SEEDS = [1, 2, 3]


# ------------------------------------------------------------------ child
class KeyPolicy:
    def __init__(self, key):
        self.k = key

    def key(self, e, link, ln, t):
        return (self.k[e.event_tag], -ln)

    def on_issue(self, e, link, t):
        pass

    def on_finish(self, e, t):
        pass

    def on_tick(self, t, ready):
        pass


def _events(spec, hp0, streams):
    ev = E.merged_events([B.load_stream(s)["events_blob"] for s in streams])
    if spec.get("chunk"):
        ev, _ = E.chunk_events(ev, hp0, spec["chunk"]["size"], spec["chunk"]["only_d2d"])
    return ev


def _graph(ev, hp0):
    deps = {t: set(e.dependency_set) for t, e in ev.items()}
    paths = {t: [list(b) for b in e.path_list] for t, e in ev.items() if e.event_type == "communication"}
    return cpcm.compile_bb(ev, hp0, path_lists=paths, deps=deps)


def _C(g, sc, n):
    C = [0] * n
    for e in range(g.n):
        i = g.tag[e][1] - 10
        C[i] = max(C[i], sc.F[e])
    return C


def child(spec_path):
    with open(spec_path, "rb") as f:
        spec = pickle.load(f)
    E._artifact_path()
    from e4b_backend import event_driver_e4b
    n = len(spec["streams"])
    hp0 = E.build_platform(spec["cfg"], spec["faults"])
    ev = _events(spec, hp0, spec["streams"])
    t0 = time.perf_counter()
    g = _graph(ev, hp0)
    t_comp = time.perf_counter() - t0
    stream = [g.tag[e][1] - 10 for e in range(g.n)]
    rel = {e: spec["release"][stream[e]] for e in range(g.n) if g.pred_ptr[e + 1] == g.pred_ptr[e]}
    dl = [spec["d"][stream[e]] for e in range(g.n)]
    tm = {}
    t0 = time.perf_counter()
    sc1 = cpcm.simulate(g, "bb", release=rel)
    an1 = cpcm.Analysis(sc1, sink_deadline=dl)
    D0 = an1.D
    t1 = time.perf_counter()
    sc2 = cpcm.simulate(g, "hop", prio=D0, release=rel)
    t2 = time.perf_counter()
    an2 = cpcm.Analysis(sc2, sink_deadline=dl)
    sc3 = cpcm.simulate(g, "hop", prio=an2.D, release=rel)
    t3 = time.perf_counter()
    scp = cpcm.simulate(g, "hop", release=rel)
    tm.update(compile=t_comp, pass1=t1 - t0, pass2=t2 - t1, pass3=t3 - t2)
    solo = []
    for i, s in enumerate(spec["streams"]):
        gs = _graph(_events(spec, hp0, [s]), hp0)
        solo.append(cpcm.simulate(gs, "hop").T)
    key = {g.tag[e]: D0[e] for e in range(g.n)}
    runs = []
    base = copy.deepcopy((ev, hp0))
    for seed in SEEDS:
        ev_k, hp_k = copy.deepcopy(base)
        random.seed(seed)
        tb = time.time()
        res = event_driver_e4b(ev_k, hp_k, release=spec["release"], policy=KeyPolicy(key))
        C = collections.defaultdict(int)
        for t, e in ev_k.items():
            C[t[1] - 10] = max(C[t[1] - 10], e.end_time)
        runs.append({"seed": seed, "wall": time.time() - tb, "result": tuple(int(x) for x in res),
                     "C": [int(C[i]) for i in range(n)]})
    out = {"n_events": g.n, "n_acts_pass2": len(sc2.a_s), "C1_bb": _C(g, sc1, n), "C2": _C(g, sc2, n),
           "C3": _C(g, sc3, n), "C_P1": _C(g, scp, n), "T2": sc2.T, "T3": sc3.T,
           "D0": {B.rt(g.tag[e]) + (stream[e],): D0[e] for e in range(g.n)},
           "D_changed_pass3": sum(1 for e in range(g.n) if an2.D[e] != D0[e]),
           "solo_T": solo, "backend": runs, "times": tm}
    with open(spec["sink"], "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


# ------------------------------------------------------------------ parent
def configs():
    st, Thop = S1.load()
    out = []
    for g, hw, cz, keys, fa, alias in S1.platforms(st):
        if alias:
            continue
        T = []
        for k in keys:
            if cz is None:
                T.append(Thop[k])
            else:
                p = os.path.join(E.OUT, "diag", "sims", f"chunk_single_{k}_{cz}.pkl.gz")
                if not os.path.exists(p):
                    p = os.path.join(B.OUT, "sims", f"Tsingle_{k}_{cz}.pkl.gz")
                with gzip.open(p, "rb") as f:
                    T.append(E._mean([r["C"][0] for r in pickle.load(f)["runs"]]))
        D = int(round(0.5 * min(T)))
        for order in ["d0", "short_first", "long_first"]:
            rel = S1.releases(dict(enumerate(T)), order, D)
            out.append({"name": f"{g}_{hw}_{order}", "group": g, "hw": hw, "chunk": cz, "order": order,
                        "streams": keys, "T": T, "release": rel, "d": {i: rel[i] + K * T[i] for i in range(len(T))},
                        "faults": fa, "cfg": E.CFG, "chunk_spec": H.CHUNKS[cz] if cz else None})
    return out


def run_one(c):
    sink = os.path.join(OUT, f"{c['name']}.pkl.gz")
    if os.path.exists(sink):
        return True
    os.makedirs(os.path.join(OUT, "_tmp"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    sp = os.path.join(OUT, "_tmp", f"{c['name']}.spec")
    tmp = os.path.join(OUT, "_tmp", f"{c['name']}.sink")
    with open(sp, "wb") as f:
        pickle.dump({"cfg": c["cfg"], "faults": c["faults"], "streams": c["streams"], "release": c["release"],
                     "d": c["d"], "chunk": c["chunk_spec"], "sink": tmp}, f)
    t0 = time.time()
    with open(os.path.join(OUT, "logs", f"{c['name']}.log"), "w", encoding="utf-8") as lf:
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), "child", sp], cwd=HERE, stdout=lf,
                             stderr=subprocess.STDOUT)
    ok = rc == 0 and os.path.exists(tmp)
    if ok:
        with open(tmp, "rb") as f:
            r = pickle.load(f)
        r["process_wall"] = time.time() - t0
        with gzip.open(sink, "wb") as f:
            pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.remove(tmp)
    os.remove(sp)
    print(f"[v5] {c['name']}: exit {rc}, {time.time() - t0:.0f} s", flush=True)
    return ok


def lhat(C, c):
    return max((C[i] - c["d"][i]) / c["T"][i] for i in range(len(C)))


def analyse():
    cfgs = configs()
    rows = []
    dev_D, dev_R = collections.defaultdict(list), []
    for c in cfgs:
        p = os.path.join(OUT, f"{c['name']}.pkl.gz")
        if not os.path.exists(p):
            rows.append({"config": c["name"], "status": "missing"})
            continue
        with gzip.open(p, "rb") as f:
            r = pickle.load(f)
        n = len(c["streams"])
        bC = [statistics.mean(x["C"][i] for x in r["backend"]) for i in range(n)]
        bT = statistics.mean(max(x["C"]) for x in r["backend"])
        bL = statistics.mean(lhat(x["C"], c) for x in r["backend"])
        bLs = [lhat(x["C"], c) for x in r["backend"]]
        p1 = os.path.join(B.OUT, "sims", f"{c['name']}_P1.pkl.gz")
        p1T = None
        if os.path.exists(p1):
            with gzip.open(p1, "rb") as f:
                p1T = statistics.mean(max(x["C"].values()) for x in pickle.load(f)["runs"])
        T2 = max(r["C2"])
        row = {"config": c["name"], "status": "ok", "group": c["group"], "hw": c["hw"], "order": c["order"],
               "chunk": c["chunk"] or "", "events": r["n_events"],
               "cpcm_T2": T2, "backend_T_mean": round(bT, 1), "err_T2": round((T2 - bT) / bT, 5),
               "err_C_max_abs": round(max(abs(r["C2"][i] - bC[i]) / bC[i] for i in range(n)), 5),
               "cpcm_Lhat2": round(lhat(r["C2"], c), 4), "backend_Lhat_mean": round(bL, 4),
               "backend_Lhat_min": round(min(bLs), 4), "backend_Lhat_max": round(max(bLs), 4),
               "Lhat_abs_diff": round(abs(lhat(r["C2"], c) - bL), 4),
               "backend_T_range_over_mean": round((max(max(x["C"]) for x in r["backend"]) -
                                                  min(max(x["C"]) for x in r["backend"])) / bT, 5),
               "iter_C_changed": r["C3"] != r["C2"], "iter_T_rel_change": round((max(r["C3"]) - T2) / T2, 5),
               "iter_Lhat3": round(lhat(r["C3"], c), 4), "iter_D_changed_events": r["D_changed_pass3"],
               "cpcm_T_P1": max(r["C_P1"]), "e4b_P1_T_mean": round(p1T, 1) if p1T else "",
               "err_T_P1": round((max(r["C_P1"]) - p1T) / p1T, 5) if p1T else "",
               "cpcm_T_bb_pass1": max(r["C1_bb"]),
               "t_pass1_s": round(r["times"]["pass1"], 3), "t_pass2_s": round(r["times"]["pass2"], 3),
               "backend_s_mean": round(statistics.mean(x["wall"] for x in r["backend"]), 1)}
        rows.append(row)
        # ideal comparisons
        for i, s in enumerate(c["streams"]):
            dev_R.append(abs(r["solo_T"][i] - c["T"][i]) / c["T"][i])
            sp = os.path.join(B.OUT, "sims", f"solo_{s}_{c['chunk'] or 'none'}.pkl.gz")
            if not os.path.exists(sp):
                continue
            with gzip.open(sp, "rb") as f:
                caps = {x["seed"]: x["cap"] for x in pickle.load(f)["runs"]}
            sd = B.solo_data(caps[1])
            for rk, rest in sd["rest"].items():
                d0 = r["D0"].get(rk + (i,))
                if d0 is not None:
                    dev_D[c["group"]].append((d0 - (c["d"][i] - rest)) / c["T"][i])
    ok = [x for x in rows if x["status"] == "ok"]

    def st(v):
        v = list(v)
        return {"n": len(v), "median": round(statistics.median(v), 5), "max": round(max(v), 5),
                "mean": round(statistics.mean(v), 5)} if v else {"n": 0}

    summ = {"k": K, "seeds": SEEDS, "configs": len(rows), "ok": len(ok),
            "err_T2_abs": st(abs(x["err_T2"]) for x in ok), "err_C_max_abs": st(x["err_C_max_abs"] for x in ok),
            "Lhat_abs_diff": st(x["Lhat_abs_diff"] for x in ok),
            "backend_T_range_over_mean": st(x["backend_T_range_over_mean"] for x in ok),
            "by_group_err_T2_abs": {gname: st(abs(x["err_T2"]) for x in ok if x["group"] == gname)
                                    for gname in dict.fromkeys(x["group"] for x in ok)},
            "iteration": {"configs_C_changed": sum(x["iter_C_changed"] for x in ok),
                          "T_rel_change": st(abs(x["iter_T_rel_change"]) for x in ok),
                          "Lhat_change": st(abs(x["iter_Lhat3"] - x["cpcm_Lhat2"]) for x in ok),
                          "D_changed_events": st(x["iter_D_changed_events"] for x in ok)},
            "P1_check_err_abs": st(abs(x["err_T_P1"]) for x in ok if x["err_T_P1"] != ""),
            "D0_minus_ideal_over_T": {gname: {"n": len(v), "median": round(statistics.median(v), 4),
                                              "median_abs": round(statistics.median(abs(x) for x in v), 4),
                                              "p05": round(sorted(v)[int(0.05 * (len(v) - 1))], 4),
                                              "p95": round(sorted(v)[int(0.95 * (len(v) - 1))], 4)}
                                      for gname, v in dev_D.items()},
            "Rhat0_solo_err_abs": st(dev_R)}
    with open(os.path.join(HERE, "cpcm_out", "v5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    keys = list(dict.fromkeys(k for x in rows for k in x))
    with open(os.path.join(HERE, "cpcm_out", "v5_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyse", "child"])
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.cmd == "child":
        child(a.spec)
        return
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if a.cmd == "run":
        cs = configs()
        if a.only:
            cs = [c for c in cs if c["name"] in a.only]
        cs.sort(key=lambda c: (c["group"] != "a1_3s_512-2048-8192", c["chunk"] is None))
        with ThreadPoolExecutor(a.jobs) as ex:
            ok = list(ex.map(run_one, cs))
        print(f"[v5] run: {sum(ok)}/{len(ok)} ok in {time.time() - t0:.0f} s", flush=True)
    else:
        analyse()


if __name__ == "__main__":
    main()
