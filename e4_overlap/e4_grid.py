"""E4 step 4, batch 1 (user decision 2026-11-01): the unchunked grid.

9 stream compositions x 5 staggers x {no fault, F4, F5}; 3 backend seeds (6 for Delta = 1T); every configuration
with default arbitration and every stream priority order; plus the 1x4 spot check (a1 and a2, 2 streams,
asymmetric, Delta = 0.5T, no fault and F4).

  - a1 2-stream, no fault: the step-2 configurations (Delta = 1T re-run with 6 seeds) are reused as they are.
  - a2's symmetric controls (2 / 3 x MHA 2048) are the same event graphs as a1's symmetric ones: they are
    simulated once (under the a1 name) and listed again under the a2 name (summary field alias_of).
  - F4 / F5 per composition: E3's rules (F4 busiest ch2ch link by busy time, F5 co2co/ch2ch link with the largest
    exclusive critical wait, both directions) on the merged, Delta = 0, no-fault, default-arbitration graph of
    backend seed 1. Each stream then gets E3's R0 for that fault (template re-run with the stream's pre-fault
    mapping injected, routes kept, only cut communications re-routed, no SA), captured as event objects.
  - T_i on a faulty platform = the stream's own R0-repaired graph alone on that platform (3 seeds); stretch and
    deadlines use it (S_max against the no-fault T_i is reported too).
  - Every (configuration, arbitration variant) is one process; results are cached in e4_out/sims/ so a re-run
    resumes.

  python e4_grid.py --jobs 12
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import e4_overlap as E

OUT = E.OUT
SIM_DIR = os.path.join(OUT, "sims")
CFG_1X4 = os.path.join(E.MODEL, "cfg", "config_ch1x4_bw256_co4x4_bw256_t128x64_failpattern0.cfg")
TAIL14 = "ch1x4_bw256_co4x4_bw256_t128x64_failpattern0"
GROUPS = {  # name -> (composition, stream keys); stream 0 released first
    "a1_2s_512-8192": ("a1", ["mha512", "mha8192"]),
    "a1_2s_2048-8192": ("a1", ["mha2048", "mha8192"]),
    "a1_2s_2048-2048": ("a1", ["mha2048", "mha2048"]),
    "a1_3s_512-2048-8192": ("a1", ["mha512", "mha2048", "mha8192"]),
    "a1_3s_2048x3": ("a1", ["mha2048", "mha2048", "mha2048"]),
    "a2_2s_mha-ffn": ("a2", ["mha2048", "ffn"]),
    "a2_3s_mha-ffn-mha": ("a2", ["mha2048", "ffn", "mha2048"]),
}
ALIAS = {"a2_2s_sym": "a1_2s_2048-2048", "a2_3s_sym": "a1_3s_2048x3"}
GROUPS_1X4 = {"a1_2s_512-8192_1x4": ("a1", ["mha512_1x4", "mha8192_1x4"]),
              "a2_2s_mha-ffn_1x4": ("a2", ["mha2048_1x4", "ffn_1x4"])}
D1_SEEDS = [1, 2, 3, 4, 5, 6]
TIMEOUT = 3600  # one process > 1 h: stop and ask (CLAUDE.md)


def log(msg):
    print(f"[e4] {msg}", flush=True)


def label_of(st):
    return "serial" if st == "serial" else f"d{st:g}"


def make_configs(group, comp, keys, T, hw, faults, cfg, staggers=E.STAGGERS, prio=True, seeds_override=None):
    n = len(keys)
    Tmin = min(T[k] for k in keys)
    out = []
    for st in staggers:
        if st == "serial":
            base = {"gate": {i: i - 1 for i in range(1, n)}, "release": {0: 0}}
        else:
            base = {"release": {i: i * int(round(st * Tmin)) for i in range(n)}}
        variants = [dict(base, name="default")]
        if prio:
            import itertools
            for perm in itertools.permutations(range(n)):
                variants.append(dict(base, name="prio_" + "".join(map(str, perm)),
                                     prio={s: r for r, s in enumerate(perm)}))
        hwtag = "" if hw == "nofault" else f"_{hw}"
        out.append({"name": f"{group}{hwtag}_{label_of(st)}", "group": group, "composition": comp,
                    "n_streams": n, "comp": group, "streams": keys, "stagger": st, "hardware": hw,
                    "faults": faults, "cfg": cfg, "T": {i: T[k] for i, k in enumerate(keys)}, "Tmin": Tmin,
                    "variants": variants,
                    "seeds": seeds_override or (D1_SEEDS if st == 1.0 else list(E.SEEDS)),
                    "tasks": [load_rec_task(k) for k in keys]})
    return out


_TASK = {}


def load_rec_task(key):
    if key not in _TASK:
        _TASK[key] = E.load_stream(key)["task"]
    return _TASK[key]


def sim_cached(name, keys, var, seeds, cfg, faults):
    os.makedirs(SIM_DIR, exist_ok=True)
    path = os.path.join(SIM_DIR, f"{name}.pkl.gz")
    if os.path.exists(path):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    r = E.simulate(name, keys, [var], seeds, timeout=TIMEOUT, cfg=cfg, faults=faults)
    if r.get("timeout"):
        raise E.E4Error(f"simulation {name} exceeded {TIMEOUT} s (one process > 1 h): stop and ask")
    with gzip.open(path, "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    return r


def run_units(units, jobs):
    """units: [(unit name, keys, variant, seeds, cfg, faults)] -> {unit name: result}"""
    t0 = time.time()
    with ThreadPoolExecutor(jobs) as ex:
        res = dict(zip([u[0] for u in units], ex.map(lambda u: sim_cached(*u), units)))
    log(f"{len(units)} simulation units done ({time.time() - t0:.0f} s)")
    return res


def simulate_configs(configs, jobs):
    units = [(f"{c['name']}_{v['name']}", c["streams"], v, c["seeds"], c["cfg"], c["faults"])
             for c in configs for v in c["variants"]]
    res = run_units(units, jobs)
    return {c["name"]: {"runs": [r for v in c["variants"] for r in res[f"{c['name']}_{v['name']}"]["runs"]]}
            for c in configs}


def analyse_all(configs, sims, jobs):
    tasks = [((c["name"], "default", r["seed"]), r["cap"], c["T"]) for c in configs for r in sims[c["name"]]["runs"]
             if r["variant"] == "default"]
    t0 = time.time()
    with ProcessPoolExecutor(jobs) as ex:
        futs = {k: ex.submit(E.analyse_cap, cap, Tm) for k, cap, Tm in tasks}
        an = {k: f.result() for k, f in futs.items()}
    log(f"{len(an)} graphs analysed ({time.time() - t0:.0f} s)")
    return an


def summarize(configs, sims, an, T_nofault):
    import e4_attr
    checks = {"check2": [], "check3": {}, "check5_fail": {}, "compute_dominant": []}
    for k, v in an.items():
        if not v["ok"] or not all(v["checks"].values()) or v["release_in_segs"]:
            checks["check5_fail"][str(k)] = v.get("error") or v["checks"]
    if checks["check5_fail"]:
        raise E.E4Error(f"self-check 5 failed: {list(checks['check5_fail'].items())[:3]}")
    for c in configs:
        summ, c2, c3, comp_share = E.summarize_config(c, sims[c["name"]]["runs"], an)
        checks["check2"] += c2
        if c3 is not None:
            checks["check3"][c["name"]] = c3
        if comp_share > E.COMPUTE_STOP:
            checks["compute_dominant"].append((c["name"], comp_share))
        if c["hardware"] != "nofault":  # S_max against the no-fault T_i as well
            Tn = {i: T_nofault[k] for i, k in enumerate(c["streams"])}
            sm = [e4_attr.stream_measures(r["cap"], Tn, E.KS)["S_max"] for r in sims[c["name"]]["runs"]
                  if r["variant"] == "default"]
            summ["T_nofault"] = Tn
            summ["S_max_vs_nofault_T_mean"] = E._mean(sm)
            E._write_json(f"{c['name']}_summary.json", summ)
    if not all(checks["check2"]):
        raise E.E4Error("self-check 2 failed (an event started before its stream's release)")
    bad3 = {k: v for k, v in checks["check3"].items() if not v["pass"]}
    if bad3:
        raise E.E4Error(f"self-check 3 failed: {bad3}")
    if checks["compute_dominant"]:
        raise E.E4Error(f"handoff 7: critical wait mostly on compute units: {checks['compute_dominant']}")
    return checks


# ---------------------------------------------------------------- faults
def choose_faults(group, cap):
    """F4 / F5 (E3's rules) on one merged Delta = 0 no-fault default-arbitration graph."""
    import e4_attr
    busy = {k: sum(f - s for s, f, _ in rec) for k, rec in cap["links"].items()}
    a = cap["result"][0]
    d2d = {k: v for k, v in busy.items() if cap["link_types"][k] == "ch2ch"}
    u, v = max(d2d, key=lambda x: (d2d[x], str(x)))
    f4 = {"id": "F4", "kind": "link ch2ch, both directions", "failed_nodes": [], "failed_links": [(u, v), (v, u)],
          "rule": "busiest ch2ch link (busy time) in the merged Delta=0 no-fault default graph, backend seed 1",
          "busy": d2d[(u, v)], "busy_over_a": d2d[(u, v)] / a, "busy_reverse": busy.get((v, u), 0),
          "top": [[str(k), x] for k, x in sorted(d2d.items(), key=lambda kv: -kv[1])[:4]]}
    _, _, segs, _, _, _ = e4_attr.analyse(cap)
    wait = collections.Counter()
    for sg in segs:
        if sg["res_key"][0] == "link" and cap["link_types"][sg["res_key"][1]] in ("co2co", "ch2ch"):
            wait[sg["res_key"][1]] += sg["wait_excl"]
    (u, v), w = max(wait.items(), key=lambda x: (x[1], str(x[0])))
    f5 = {"id": "F5", "kind": f"link {cap['link_types'][(u, v)]}, both directions", "failed_nodes": [],
          "failed_links": [(u, v), (v, u)],
          "rule": "co2co/ch2ch link with the largest exclusive critical wait (E2 attribution with release times) "
                  "in the merged Delta=0 no-fault default graph, backend seed 1",
          "critical_wait": w, "critical_wait_over_a": w / a, "busy": busy.get((u, v), 0),
          "busy_over_a": busy.get((u, v), 0) / a, "top_critical_links": [[str(k), x] for k, x in wait.most_common(5)]}
    return {"F4": f4, "F5": f5}


def fault_key(base, faults):
    h = hashlib.sha1(repr(sorted(map(str, faults["failed_links"]))).encode()).hexdigest()[:8]
    return f"{base}__{h}"


def capture_r0(key, base, faults):
    """E3's R0 for `faults` on stream `base`, captured as event objects (one template run)."""
    path = os.path.join(E.STREAM_DIR, f"{key}.pkl.gz")
    if os.path.exists(path):
        return key, "cached"
    b = E.load_stream(base)
    spec = {"mapping": b["mapping"], "steps": None, "t_max": None, "seeds": [], "preserve_routes": True,
            "faults": {"failed_nodes": [], "failed_links": [tuple(map(tuple, l)) for l in faults["failed_links"]]}}
    o = E._run_capture_child(b["task"], [], f"r0_{key}", e3_spec=spec)
    lg = o.get("inject_log", {})
    routes = [{k: r.get(k) for k in ["where", "restored", "rerouted", "source_restored", "rerouted_without_any_match"]}
              for r in lg.get("routes", [])]
    rec = {"short": key, "base": base, "task": b["task"], "faults": spec["faults"], "events_blob": o["events_blob"],
           "pre_digest": o["pre_digest"], "platform_state": o["platform_state"],
           "empty_path_comms": o["empty_path_comms"], "template_result": o["last"]["result"], "final": o["last"],
           "mapping": o["mapping"], "platform": o["platform"], "inject_routes": routes,
           "inject_log_keys": sorted(lg), "relocated": lg.get("relocated"), "artifact_commit": E._artifact_commit()}
    with gzip.open(path, "wb") as f:
        pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
    return key, {"template_result": rec["template_result"], "routes": routes,
                 "empty_path_comms": rec["empty_path_comms"]}


def singles_T(keys, cfg_of, faults_of, jobs):
    """T_i: the stream alone (release 0) on its platform, seeds 1-3."""
    units = [(f"single_{k}_default", [k], {"name": "default", "release": {0: 0}}, list(E.SEEDS), cfg_of[k],
              faults_of[k]) for k in keys]
    res = run_units(units, jobs)
    return {k: E._mean([r["cap"]["result"][0] for r in res[f"single_{k}_default"]["runs"]]) for k in keys}, \
        {k: [r["cap"]["result"][0] for r in res[f"single_{k}_default"]["runs"]] for k in keys}


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--skip-1x4", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    state = {"command": sys.argv, "artifact_commit": E._artifact_commit(), "phases": {}}

    def save_state():
        E._write_json("grid1_state.json", state)

    # ---- T (no fault)
    with open(os.path.join(OUT, "step2_checks.json"), encoding="utf-8") as f:
        T = dict(json.load(f)["T"])
    with open(os.path.join(OUT, "step3_chunk.json"), encoding="utf-8") as f:
        T["ffn"] = json.load(f)["T"]["1"]
    state["T_nofault"] = dict(T)
    cfg_of = collections.defaultdict(lambda: E.CFG)
    faults_of = collections.defaultdict(lambda: None)

    # ---- phase A: no fault (new groups) + Delta = 0 default re-runs of the a1 2-stream groups (fault choice)
    new_groups = [g for g in GROUPS if not g.startswith("a1_2s_")]
    cfgsA = [c for g in new_groups for c in make_configs(g, *GROUPS[g], T, "nofault", None, E.CFG)]
    sel = [c for g in GROUPS if g.startswith("a1_2s_")
           for c in make_configs(g, *GROUPS[g], T, "nofault", None, E.CFG, staggers=[0.0], prio=False)]
    for c in sel:
        c["name"] += "_faultsel"
    log(f"phase A: {len(cfgsA)} no-fault configs + {len(sel)} fault-selection re-runs")
    simsA = simulate_configs(cfgsA + sel, args.jobs)
    anA = analyse_all(cfgsA, simsA, args.jobs)
    chkA = summarize(cfgsA, simsA, anA, T)
    state["phases"]["A"] = {"configs": [c["name"] for c in cfgsA], "wall": time.time() - t_start,
                            "check3": {k: v["pass"] for k, v in chkA["check3"].items()}}
    save_state()

    # ---- phase B: faults per group from the Delta = 0 seed-1 default graph
    def d0_cap(g):
        name = f"{g}_d0_faultsel" if g.startswith("a1_2s_") else f"{g}_d0"
        return next(r["cap"] for r in simsA[name]["runs"] if r["variant"] == "default" and r["seed"] == 1)
    faults = {g: choose_faults(g, d0_cap(g)) for g in GROUPS}
    state["faults"] = faults
    save_state()
    log("faults: " + json.dumps({g: {f: [str(x) for x in v["failed_links"]] for f, v in fs.items()}
                                 for g, fs in faults.items()}))

    # ---- phase C: R0 captures per (stream, fault)
    need = {}
    for g, fs in faults.items():
        for fid, fl in fs.items():
            for k in set(GROUPS[g][1]):
                need[fault_key(k, fl)] = (k, fl)
    t0 = time.time()
    with ThreadPoolExecutor(args.jobs) as ex:
        r0 = dict(ex.map(lambda kv: capture_r0(kv[0], *kv[1]), need.items()))
    state["r0_captures"] = r0
    log(f"phase C: {len(need)} R0 captures ({time.time() - t0:.0f} s)")
    save_state()

    # ---- phase D: T on faulty platforms
    for fk, (k, fl) in need.items():
        faults_of[fk] = {"failed_nodes": [], "failed_links": fl["failed_links"]}
    Tf, Tf_seeds = singles_T(list(need), cfg_of, faults_of, args.jobs)
    T.update(Tf)
    state["T_fault"] = Tf
    state["T_fault_seeds"] = Tf_seeds
    save_state()

    # ---- phase E: fault grid, one fault id at a time (memory)
    del simsA, anA
    T_nofault_by_key = dict(state["T_nofault"])
    for fk, (k, fl) in need.items():
        T_nofault_by_key[fk] = state["T_nofault"][k]
    same = {g for g in GROUPS if set(map(str, faults[g]["F5"]["failed_links"])) ==
            set(map(str, faults[g]["F4"]["failed_links"]))}
    state["F5_same_as_F4"] = sorted(same)
    for fid in ["F4", "F5"]:
        cfgsE = []
        for g, (comp, keys) in GROUPS.items():
            if fid == "F5" and g in same:
                continue  # listed again from the F4 results below
            fl = faults[g][fid]
            fa = {"failed_nodes": [], "failed_links": fl["failed_links"]}
            cfgsE += make_configs(g, comp, [fault_key(k, fl) for k in keys], T, fid, fa, E.CFG)
        log(f"phase E {fid}: {len(cfgsE)} fault configs")
        simsE = simulate_configs(cfgsE, args.jobs)
        anE = analyse_all(cfgsE, simsE, args.jobs)
        chkE = summarize(cfgsE, simsE, anE, T_nofault_by_key)
        state["phases"][f"E_{fid}"] = {"configs": [c["name"] for c in cfgsE], "wall": time.time() - t_start,
                                       "check3": {k: v["pass"] for k, v in chkE["check3"].items()}}
        save_state()
        del simsE, anE
    for g in same:  # F5 picked the same link pair as F4: the same platform and graphs
        for st in E.STAGGERS:
            src = os.path.join(OUT, f"{g}_F4_{label_of(st)}_summary.json")
            with open(src, encoding="utf-8") as f:
                s = json.load(f)
            s.update(config=f"{g}_F5_{label_of(st)}", hardware="F5", alias_of=s["config"])
            E._write_json(f"{g}_F5_{label_of(st)}_summary.json", s)
            shutil.copyfile(src.replace("_summary.json", "_runs.csv"),
                            os.path.join(OUT, f"{g}_F5_{label_of(st)}_runs.csv"))

    # ---- aliases (a2 symmetric controls)
    for a2, a1 in ALIAS.items():
        for hw in ["", "_F4", "_F5"]:
            for st in E.STAGGERS:
                src = os.path.join(OUT, f"{a1}{hw}_{label_of(st)}_summary.json")
                if not os.path.exists(src):
                    continue
                with open(src, encoding="utf-8") as f:
                    s = json.load(f)
                s.update(config=f"{a2}{hw}_{label_of(st)}", composition="a2", alias_of=s["config"])
                E._write_json(f"{a2}{hw}_{label_of(st)}_summary.json", s)
                shutil.copyfile(src.replace("_summary.json", "_runs.csv"),
                                os.path.join(OUT, f"{a2}{hw}_{label_of(st)}_runs.csv"))

    # ---- 1x4 spot check
    if not args.skip_1x4:
        state["phases"]["1x4"] = run_1x4(args.jobs, state)
        save_state()

    table = E.rebuild_table()
    write_core_plot()
    state["wall"] = time.time() - t_start
    save_state()
    log(json.dumps({"configs": len(table), "wall_s": round(state["wall"]),
                    "faults": {g: {f: [str(x) for x in v["failed_links"]] for f, v in fs.items()}
                               for g, fs in faults.items()}}))


def run_1x4(jobs, state):
    with open(os.path.join(OUT, "variants_1x4.json"), encoding="utf-8") as f:
        v14 = json.load(f)
    best ={"mha512_1x4": v14["mha_sq512"]["best"], "mha8192_1x4": v14["mha_sq8192"]["best"],
            "mha2048_1x4": v14["mha_sq2048"]["best"], "ffn_1x4": v14["ffn_sq2048"]["best"]}
    for k, t in best.items():
        E.STREAMS[k] = t
    with ThreadPoolExecutor(4) as ex:
        caps = list(ex.map(lambda k: E.capture(k) if not os.path.exists(os.path.join(E.STREAM_DIR, f"{k}.pkl.gz"))
                           else {"short": k, "cached": True}, list(best)))
    cfg_of = collections.defaultdict(lambda: CFG_1X4)
    faults_of = collections.defaultdict(lambda: None)
    T, T_seeds = singles_T(list(best), cfg_of, faults_of, jobs)
    out = {"winners": best, "captures": caps, "T_nofault": T, "T_nofault_seeds": T_seeds}
    # Delta = 0 default (fault choice) and Delta = 0.5T full, no fault
    cfgs, sel = [], []
    for g, (comp, keys) in GROUPS_1X4.items():
        cfgs += make_configs(g, comp, keys, T, "nofault", None, CFG_1X4, staggers=[0.5])
        s = make_configs(g, comp, keys, T, "nofault", None, CFG_1X4, staggers=[0.0], prio=False)
        s[0]["name"] += "_faultsel"
        sel += s
    sims = simulate_configs(cfgs + sel, jobs)
    an = analyse_all(cfgs, sims, jobs)
    summarize(cfgs, sims, an, T)
    faults = {}
    need = {}
    for g, (comp, keys) in GROUPS_1X4.items():
        cap = next(r["cap"] for r in sims[f"{g}_d0_faultsel"]["runs"] if r["variant"] == "default" and r["seed"] == 1)
        faults[g] = {"F4": choose_faults(g, cap)["F4"]}
        for k in set(keys):
            need[fault_key(k, faults[g]["F4"])] = (k, faults[g]["F4"])
    with ThreadPoolExecutor(jobs) as ex:
        r0 = dict(ex.map(lambda kv: capture_r0(kv[0], *kv[1]), need.items()))
    for fk, (k, fl) in need.items():
        faults_of[fk] = {"failed_nodes": [], "failed_links": fl["failed_links"]}
    Tf, Tf_seeds = singles_T(list(need), cfg_of, faults_of, jobs)
    T.update(Tf)
    cfgsF = []
    for g, (comp, keys) in GROUPS_1X4.items():
        fl = faults[g]["F4"]
        cfgsF += make_configs(g, comp, [fault_key(k, fl) for k in keys], T, "F4",
                              {"failed_nodes": [], "failed_links": fl["failed_links"]}, CFG_1X4, staggers=[0.5])
    simsF = simulate_configs(cfgsF, jobs)
    anF = analyse_all(cfgsF, simsF, jobs)
    Tn = dict(out["T_nofault"])
    for fk, (k, fl) in need.items():
        Tn[fk] = out["T_nofault"][k]
    summarize(cfgsF, simsF, anF, Tn)
    out.update(faults=faults, r0=r0, T_fault=Tf, T_fault_seeds=Tf_seeds,
               configs=[c["name"] for c in cfgs + cfgsF])
    return out


def write_core_plot():
    """Core figure data (user decision): x = T_max/T_min; y = stream-priority gain (1 - best/default S_max) and the
    single-resource S_max bound; makespan bounds attached."""
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "a*_summary.json"))):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        pd = s.get("prio_detail") or {}
        bo = pd.get("best_over_default", {})
        rows.append({"config": s["config"], "composition": s["composition"], "n_streams": s["n_streams"],
                     "streams": "+".join(s["streams"]), "stagger": s["stagger"], "hardware": s.get("hardware"),
                     "chunking": s.get("chunking"), "alias_of": s.get("alias_of", ""),
                     "n_seeds": len(s.get("backend_seeds", [])), "T_ratio": s.get("T_ratio"),
                     "S_max_default": s["S_max_mean"], "prio_best_order": pd.get("best_order"),
                     "prio_gain": 1 - bo["S_max"] if bo else None,
                     "smax_bound_reduction": s["smax_bound_reduction_mean"],
                     "smax_bound_reduction_ub": s["smax_bound_reduction_ub_mean"],
                     "makespan_reorder_gain": s["makespan_reorder_gain_mean"],
                     "makespan_delete_gain": s["makespan_delete_gain_mean"],
                     "cross_stream_share": s["cross_stream_share_mean"],
                     "overlap_ge2": s["overlap_ge2_mean"],
                     "prio_makespan_ratio": bo.get("makespan"), "prio_beat_ratio": bo.get("beat")})
    E._write_csv("e4_core_plot.csv", rows)


if __name__ == "__main__":
    try:
        main()
    except E.E4Error as exc:
        print(f"[e4] STOP: {exc}", flush=True)
        sys.exit(2)
