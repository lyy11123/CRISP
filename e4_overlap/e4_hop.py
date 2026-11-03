"""E4 per-hop arbitration runs (user request 2026-11-02, "两项都做"; measurement only; backend =
e4_backend_diag.event_driver_diag mode "perhop", in memory).

Item 1 (grid): 7 simulated stream groups (+ the 2 a2 symmetric aliases) x Delta {0, 0.5T} x {no fault, F4, F5}
  (faults and R0 graphs of batch 1; F5 = F4 groups listed from F4) x unchunked; default + every priority order;
  backend seeds 1-3. Releases = those of the original-backend configuration (batch 1 / step 2 summaries), so the
  two backends see the same arrivals. T_i (hop) = the stream alone under per-hop arbitration, seeds 1-6; stretch
  reported against T_i (hop) and against the original backend's 6-seed T_i. Side by side with the original
  configuration's summary. Writes e4_out/hop/.
Item 2 (chunk): a2 MHA+FFN, per hop, Delta 0, no fault and F4, default and prio_01, seeds 1-3, chunking none /
  256 KB D2D-only / 64 KB D2D-only (and 256 KB on every link if asked by --all-link); full caps kept for the
  chain analysis. Writes e4_out/diag/hop_chunk*.

  python e4_hop.py grid --probe       # singles + one 3-stream configuration, then a time estimate
  python e4_hop.py grid --jobs 12
  python e4_hop.py chunk --jobs 12 [--sizes none,256d2d,64d2d]
"""
import argparse
import collections
import copy
import gzip
import itertools
import json
import os
import pickle
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import e4_attr as A
import e4_grid as G
import e4_overlap as E

HERE = os.path.dirname(os.path.abspath(__file__))
HOP = os.path.join(E.OUT, "hop")
DIAG = os.path.join(E.OUT, "diag")
LIMIT = 1200  # one evaluation > 20 min: stop and report
CHUNKS = {"none": None, "256d2d": {"size": 256 * 1024, "only_d2d": True}, "64d2d": {"size": 64 * 1024, "only_d2d": True},
          "256all": {"size": 256 * 1024, "only_d2d": False}}


def child(spec_path):
    with open(spec_path, "rb") as f:
        spec = pickle.load(f)
    E._artifact_path()
    from e4_backend_diag import event_driver_diag
    e2 = E._e2()
    hp0 = E.build_platform(spec["cfg"], spec["faults"])
    blobs = [E.load_stream(s)["events_blob"] for s in spec["streams"]]
    out = []
    for seed in spec["seeds"]:
        ev = E.merged_events(blobs)
        cst = None
        if spec.get("chunk"):
            ev, cst = E.chunk_events(ev, hp0, spec["chunk"]["size"], spec["chunk"]["only_d2d"])
        hp = copy.deepcopy(hp0)
        pre = e2._snapshot_pre(ev) if spec.get("keep_cap") else None
        released = {}
        random.seed(seed)
        t0 = time.time()
        res = event_driver_diag(ev, hp, release=spec["release"], prio=spec.get("prio"), released=released,
                                mode=spec.get("mode", "perhop"), watch=None, log={})
        wall = time.time() - t0
        C = collections.defaultdict(int)
        for t, e in ev.items():
            C[t[1] - 10] = max(C[t[1] - 10], e.end_time)
        r = {"seed": seed, "wall": wall, "result": res, "C": dict(C), "release": released, "chunk_stats": cst}
        if spec.get("keep_cap"):
            cap = e2._extract(ev, hp, pre, res, f"seed{seed}")
            cap["release"] = released
            r["cap"] = cap
        out.append(r)
    with open(spec["sink"], "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def run(name, d, spec):
    """One process; cached in <d>/sims/<name>.pkl.gz."""
    path = os.path.join(d, "sims", f"{name}.pkl.gz")
    if os.path.exists(path):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = os.path.join(E.OUT, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    sp = os.path.join(tmp, f"spec_hop_{name}.pkl")
    sink = os.path.join(tmp, f"sink_hop_{name}.pkl")
    with open(sp, "wb") as f:
        pickle.dump(dict(spec, sink=sink), f)
    t0 = time.time()
    if not os.path.exists(sink):
        with open(os.path.join(E.OUT, "logs", f"hop_{name}.log"), "w", encoding="utf-8") as lf:
            try:
                rc = subprocess.call([sys.executable, os.path.abspath(__file__), "child", sp], cwd=HERE, stdout=lf,
                                     stderr=subprocess.STDOUT, timeout=LIMIT * len(spec["seeds"]))
            except subprocess.TimeoutExpired:
                return {"timeout": True, "wall": time.time() - t0}
        if rc != 0:
            raise E.E4Error(f"{name} failed (exit {rc})")
    E._artifact_path()
    with open(sink, "rb") as f:
        runs = pickle.load(f)
    os.remove(sink)
    os.remove(sp)
    r = {"runs": runs, "process_wall": time.time() - t0}
    with gzip.open(path, "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    return r


def run_all(jobs, items):
    """items: [(name, dir, spec)]"""
    t0 = time.time()
    with ThreadPoolExecutor(jobs) as ex:
        res = dict(zip([i[0] for i in items], ex.map(lambda i: run(*i), items)))
    G.log(f"{len(items)} hop processes ({time.time() - t0:.0f} s)")
    bad = [k for k, v in res.items() if v.get("timeout")]
    if bad:
        raise E.E4Error(f"evaluation over {LIMIT} s: {bad}")
    return res


def measures(r, T):
    rel, C = r["release"], r["C"]
    lat = {i: C[i] - rel[i] for i in sorted(C)}
    st = {i: lat[i] / T[i] for i in lat}
    return {"latency": lat, "stretch": st, "S_max": max(st.values()), "makespan": max(C.values())}


# ---------------------------------------------------------------- item 1
def load_state():
    with open(os.path.join(E.OUT, "grid1_state.json"), encoding="utf-8") as f:
        st = json.load(f)
    for g, fs in st["faults"].items():
        for fl in fs.values():
            fl["failed_links"] = [tuple(map(tuple, l)) for l in fl["failed_links"]]
    with open(os.path.join(E.OUT, "T6.json"), encoding="utf-8") as f:
        T6 = {k: v["T6"] for k, v in json.load(f)["T"].items()}
    return st, T6


def grid_configs(state):
    same = set(state["F5_same_as_F4"])
    out = []
    for g, (comp, keys) in G.GROUPS.items():
        for hw in ["nofault", "F4", "F5"]:
            if hw == "F5" and g in same:
                continue
            fa = None if hw == "nofault" else {"failed_nodes": [], "failed_links": state["faults"][g][hw]["failed_links"]}
            ks = keys if hw == "nofault" else [G.fault_key(k, state["faults"][g][hw]) for k in keys]
            for st in [0.0, 0.5]:
                oname = f"{g}{'' if hw == 'nofault' else '_' + hw}_{G.label_of(st)}"
                with open(os.path.join(E.OUT, f"{oname}_summary.json"), encoding="utf-8") as f:
                    osum = json.load(f)
                rel = {int(i): int(round(v)) for i, v in osum["release_mean"].items()}
                n = len(keys)
                variants = [("default", None)] + [("prio_" + "".join(map(str, p)), {s: r for r, s in enumerate(p)})
                                                  for p in itertools.permutations(range(n))]
                out.append({"name": oname, "group": g, "composition": comp, "hw": hw, "stagger": st, "streams": ks,
                            "faults": fa, "release": rel, "variants": variants, "orig": osum})
    return out


def single_items(keys, faults_of):
    return [(f"single_{k}_hop", HOP, {"streams": [k], "cfg": E.CFG, "faults": faults_of[k], "release": {0: 0},
                                      "seeds": G.D1_SEEDS, "mode": "perhop"}) for k in keys]


def config_items(c):
    return [(f"{c['name']}_hop_{vn}", HOP, {"streams": c["streams"], "cfg": E.CFG, "faults": c["faults"],
                                             "release": c["release"], "prio": pr, "seeds": E.SEEDS, "mode": "perhop"})
            for vn, pr in c["variants"]]


def summarize_grid(cfgs, res, Thop, Torig):
    rows = []
    for c in cfgs:
        n = len(c["streams"])
        Th = {i: Thop[k] for i, k in enumerate(c["streams"])}
        To = {i: Torig[k] for i, k in enumerate(c["streams"])}
        per = {}
        for vn, _ in c["variants"]:
            rs = res[f"{c['name']}_hop_{vn}"]["runs"]
            mh = [measures(r, Th) for r in rs]
            mo = [measures(r, To) for r in rs]
            per[vn] = {"S_max_hopT": E._mean([m["S_max"] for m in mh]), "S_max_origT": E._mean([m["S_max"] for m in mo]),
                       "stretch_hopT": {i: E._mean([m["stretch"][i] for m in mh]) for i in range(n)},
                       "stretch_origT": {i: E._mean([m["stretch"][i] for m in mo]) for i in range(n)},
                       "makespan": E._mean([m["makespan"] for m in mh]),
                       "latency_seeds": {i: [m["latency"][i] for m in mh] for i in range(n)},
                       "eval_wall": [r["wall"] for r in rs]}
        orders = {v: p for v, p in per.items() if v != "default"}
        best = min(orders, key=lambda v: orders[v]["S_max_hopT"])
        best_o = min(orders, key=lambda v: orders[v]["S_max_origT"])
        o = c["orig"]
        opd = o.get("prio_detail") or {}
        s = {"config": c["name"], "group": c["group"], "composition": c["composition"], "hw": c["hw"],
             "stagger": c["stagger"], "streams": c["streams"], "release": c["release"],
             "T_hop": Th, "T_orig": To, "per_variant": per,
             "hop_default_S_max": per["default"]["S_max_hopT"], "hop_best_order": best,
             "hop_best_S_max": per[best]["S_max_hopT"], "hop_gain": 1 - per[best]["S_max_hopT"] / per["default"]["S_max_hopT"],
             "hop_default_S_max_origT": per["default"]["S_max_origT"], "hop_best_order_origT": best_o,
             "hop_best_S_max_origT": per[best_o]["S_max_origT"],
             "hop_gain_origT": 1 - per[best_o]["S_max_origT"] / per["default"]["S_max_origT"],
             "hop_default_makespan": per["default"]["makespan"], "hop_best_makespan": per[best]["makespan"],
             "orig_default_S_max": o["S_max_mean"], "orig_best_order": o["prio_best_order"],
             "orig_best_S_max": o["S_max_mean"] * o["prio_best_over_default"],
             "orig_gain": 1 - o["prio_best_over_default"], "orig_default_makespan": o["makespan_mean"],
             "orig_best_makespan": (opd.get("best") or {}).get("makespan"),
             "orig_stretch_default": o["stretch_mean"], "orig_stretch_best": (opd.get("best") or {}).get("stretch")}
        E._write_json(os.path.join("hop", f"{c['name']}_hop.json"), s)
        rows.append({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in s.items() if k != "per_variant"})
        rows[-1].update({f"hop_stretch_default_{i}": per["default"]["stretch_hopT"][i] for i in range(n)})
        rows[-1].update({f"hop_stretch_best_{i}": per[best]["stretch_hopT"][i] for i in range(n)})
    # a2 symmetric aliases (same graphs as the a1 symmetric groups)
    for a2, a1 in G.ALIAS.items():
        for r in [dict(x) for x in rows if x["group"] == a1]:
            r.update(config=r["config"].replace(a1, a2), group=a2, composition="a2", alias_of=r["config"])
            rows.append(r)
    E._write_csv(os.path.join("hop", "hop_grid.csv"), rows)
    return rows


def grid(args):
    state, Torig = load_state()
    cfgs = grid_configs(state)
    if args.probe:
        cfgs = [c for c in cfgs if c["name"] == "a1_3s_512-2048-8192_F4_d0"]
    keys = sorted({k for c in cfgs for k in c["streams"]})
    faults_of = {k: next(c["faults"] for c in cfgs if k in c["streams"]) for k in keys}
    t0 = time.time()
    res = run_all(args.jobs, single_items(keys, faults_of))
    Thop = {k: E._mean([r["C"][0] - r["release"][0] for r in res[f"single_{k}_hop"]["runs"]]) for k in keys}
    Thop_seeds = {k: [r["C"][0] - r["release"][0] for r in res[f"single_{k}_hop"]["runs"]] for k in keys}
    t1 = time.time()
    items = [it for c in cfgs for it in config_items(c)]
    res.update(run_all(args.jobs, items))
    t2 = time.time()
    walls = [w for k, v in res.items() if not k.startswith("single_") for w in [r["wall"] for r in v["runs"]]]
    info = {"n_configs": len(cfgs), "n_processes": len(items), "singles_wall": t1 - t0, "configs_wall": t2 - t1,
            "eval_wall_max": max(walls), "eval_wall_mean": E._mean(walls),
            "T_hop": Thop, "T_hop_seeds": Thop_seeds,
            "T_hop_vs_orig": {k: Thop[k] / Torig[k] - 1 for k in keys}}
    if args.probe:
        allc = grid_configs(state)
        n3 = sum(len(c["variants"]) for c in allc if len(c["streams"]) == 3)
        n2 = sum(len(c["variants"]) for c in allc if len(c["streams"]) == 2)
        per3 = max(v["process_wall"] for k, v in res.items() if not k.startswith("single_"))
        info.update(full_processes_3s=n3, full_processes_2s=n2, probe_process_wall_max=per3,
                    est_wall_h_12=(n3 * per3 + n2 * per3 * 0.5) / 12 / 3600)
        E._write_json(os.path.join("hop", "hop_probe.json"), info)
        G.log(json.dumps({k: v for k, v in info.items() if not k.startswith("T_hop")}))
        return
    rows = summarize_grid(cfgs, res, Thop, Torig)
    E._write_json(os.path.join("hop", "hop_state.json"), dict(info, command=sys.argv, artifact_commit=E._artifact_commit(),
                                                              seeds=E.SEEDS, single_seeds=G.D1_SEEDS))
    G.log(f"grid done: {len(rows)} rows ({time.time() - t0:.0f} s)")


# ---------------------------------------------------------------- item 2
def chunk(args):
    state, Torig = load_state()
    fl = state["faults"]["a2_2s_mha-ffn"]["F4"]
    cases = {"nofault": (["mha2048", "ffn"], None),
             "F4": ([G.fault_key(k, fl) for k in ["mha2048", "ffn"]], {"failed_nodes": [], "failed_links": fl["failed_links"]})}
    items = []
    for hw, (keys, fa) in cases.items():
        for cz in args.sizes.split(","):
            if cz != "none":  # T_i alone with the same chunking (6 seeds), per hop
                for k in keys:
                    items.append((f"chunk_single_{k}_{cz}", DIAG, {"streams": [k], "cfg": E.CFG, "faults": fa, "release": {0: 0},
                                                                   "seeds": G.D1_SEEDS, "mode": "perhop", "chunk": CHUNKS[cz]}))
            for vn, pr in [("default", None), ("prio_01", {0: 0, 1: 1})]:
                items.append((f"chunk_a2_{hw}_{cz}_{vn}", DIAG, {"streams": keys, "cfg": E.CFG, "faults": fa,
                                                                 "release": {0: 0, 1: 0}, "prio": pr, "seeds": E.SEEDS,
                                                                 "mode": "perhop", "chunk": CHUNKS[cz], "keep_cap": True}))
    res = run_all(args.jobs, items)
    with open(os.path.join(HOP, "hop_state.json"), encoding="utf-8") as f:
        Thop = json.load(f)["T_hop"]
    out = {}
    for hw, (keys, fa) in cases.items():
        for cz in args.sizes.split(","):
            if cz == "none":
                T = {i: Thop[k] for i, k in enumerate(keys)}
            else:
                T = {i: E._mean([r["C"][0] for r in res[f"chunk_single_{k}_{cz}"]["runs"]]) for i, k in enumerate(keys)}
            To = {i: Torig[k] for i, k in enumerate(keys)}
            d = {"T": T, "T_orig_unchunked": To}
            for vn in ["default", "prio_01"]:
                rs = res[f"chunk_a2_{hw}_{cz}_{vn}"]["runs"]
                mh = [measures(r, T) for r in rs]
                mo = [measures(r, To) for r in rs]
                d[vn] = {"S_max": E._mean([m["S_max"] for m in mh]), "S_max_origT": E._mean([m["S_max"] for m in mo]),
                         "stretch": {i: E._mean([m["stretch"][i] for m in mh]) for i in range(2)},
                         "latency_seeds": {i: [m["latency"][i] for m in mh] for i in range(2)},
                         "makespan": E._mean([m["makespan"] for m in mh]), "eval_wall": [r["wall"] for r in rs],
                         "chunk_stats_seed1": rs[0]["chunk_stats"]}
            d["prio_over_default"] = d["prio_01"]["S_max"] / d["default"]["S_max"]
            d["short_stretch_prio_over_default"] = d["prio_01"]["stretch"][0] / d["default"]["stretch"][0]
            out[f"{hw}_{cz}"] = d
            G.log(f"{hw} {cz}: S {d['default']['S_max']:.3f} -> {d['prio_01']['S_max']:.3f} "
                  f"({d['prio_over_default']:.3f}), walls {d['default']['eval_wall']}")
    prev = {}
    p = os.path.join(DIAG, "hop_chunk.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            prev = json.load(f)["results"]
    prev.update(out)
    E._write_json(os.path.join("diag", "hop_chunk.json"), {"results": prev, "command": sys.argv,
                                                          "artifact_commit": E._artifact_commit()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--sizes", default="none,256d2d,64d2d")
    args = ap.parse_args()
    if args.cmd == "child":
        child(args.spec)
        return
    os.makedirs(HOP, exist_ok=True)
    {"grid": grid, "chunk": chunk}[args.cmd](args)


if __name__ == "__main__":
    try:
        main()
    except E.E4Error as exc:
        print(f"[e4] STOP: {exc}", flush=True)
        sys.exit(2)
