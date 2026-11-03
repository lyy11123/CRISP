"""V2: CPCM bb mode vs the mean of 5 backend seeds (cpcm/CPCM_HANDOFF.md section 3), no fault + 5 E3 faults.

Every run is `python <task>.py` in the work copy cpcm/work/artifact, driven by E3's own machinery
(e3_warmstart.py, imported read-only) through _inject_v2/sitecustomize.py:
  capture   E3 capture mode, unchanged: the template runs as is; its final mapping (placements, data locations,
            routes, event layout) and the final backend record are read at the 2nd event_driver call.
  faults    e3_warmstart.choose_fault on that record: F1 node, F4 D2D (ch2ch) link, F5 most critical on-chip link,
            F6 PE only (F1's node), F8 busiest DDR port.
  inject    E3 inject mode, R0 (steps None: no SA): the mapping is injected, failed elements removed, R0
            relocation, routes restored (re-routed by Dijkstra where they cross a failure). At the final backend
            call inject_eval (below) replaces E3's _inject_eval: snapshot, backend seeds 1-5 each on a deepcopy,
            then cpcm.compile_bb on the untouched objects. The no-fault point is the same pipeline with no fault.
  analyse   bb mode on each compiled graph; error |T - mean| / mean against the 5 seeds.

Usage: python cpcm_v2.py capture --jobs 5 ; python cpcm_v2.py inject --jobs 12 ; python cpcm_v2.py analyse
"""
import argparse
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpcm  # noqa: E402

E3_DIR = os.path.join(HERE, "..", "e3_warmstart")
WORK = os.path.join(HERE, "work", "artifact")
OUT = os.path.join(HERE, "cpcm_out", "v2")
COMMIT_FILE = os.path.join(HERE, "..", "third_party", "isca2026-busybarn-artifact.COMMIT")
SUF = "_bw256_co4x4_bw256_t128x64_failpattern0"
# name -> (model folder, task, regime by the E2 extension's uncovered-communication share)
TASKS = {
    "ffn": ("intra_mapping_gpt_decode", "ffn_sq2048_greedy0_lossratio6_ch2x2" + SUF, "balanced"),
    "mha": ("intra_mapping_gpt_decode", "mha_sq2048_greedy0_lossratio4_ch2x2" + SUF, "balanced"),
    "moe": ("intra_mapping_qwen3moe_decode", "ffn_sq2048_greedy0_lossratio5_ch2x2" + SUF, "comm-dominated"),
    "ffn14": ("intra_mapping_gpt_decode", "ffn_sq2048_greedy1_lossratio4_ch1x4" + SUF, "balanced"),
    "opt": ("intra_mapping_ch", "ffn_sq2048_greedy0_lossratio3_ch2x2" + SUF, "compute-dominated"),
}
FAULTS = ["F1", "F4", "F5", "F6", "F8"]
SEEDS = [1, 2, 3, 4, 5]
NOFAULT = {"failed_nodes": [], "failed_links": []}


def _e3():
    if "e3_warmstart" not in sys.modules:
        sys.path.insert(0, E3_DIR)
        import e3_warmstart  # noqa: F401
        sys.path.pop(0)
    return sys.modules["e3_warmstart"]


# ------------------------------------------------------------------ child side (inside the task script)
def inject_eval(orig, events_dict, hp, call_no, g):
    e3 = _e3()
    e2 = e3._e2()
    spec = e3._spec()
    pre = e2._snapshot_pre(events_dict)
    deps = {t: set(e.dependency_set) for t, e in events_dict.items()}
    paths = {t: [list(br) for br in e.path_list] for t, e in events_dict.items() if e.event_type == "communication"}
    sink = {"calls": call_no, "log": spec["_log"], "faults": spec["faults"], "runs": []}
    state = random.getstate(), np.random.get_state()
    base = copy.deepcopy((events_dict, hp))
    out = None
    for seed in spec["seeds"]:
        ev_k, hp_k = copy.deepcopy(base)
        random.seed(seed)
        t0 = time.time()
        out = orig(events_dict=ev_k, hardware_platform=hp_k)
        sink["runs"].append({"seed": seed, "wall": time.time() - t0, "result": tuple(int(x) for x in out),
                             "cap": e2._extract(ev_k, hp_k, pre, out, f"seed{seed}")})
    random.setstate(state[0])
    np.random.set_state(state[1])
    t0 = time.time()
    sink["graph"] = cpcm.compile_bb(events_dict, hp, path_lists=paths, deps=deps)
    sink["compile_s"] = time.time() - t0
    rep = int(os.environ.get("CPCM_COMPILE_REPEAT", "0"))   # V7: serial compile timing, perf_counter
    sink["compile_times"] = []
    for _ in range(rep):
        t0 = time.perf_counter()
        cpcm.compile_bb(events_dict, hp, path_lists=paths, deps=deps)
        sink["compile_times"].append(time.perf_counter() - t0)
    with open(os.environ["E3_SINK"], "wb") as f:
        pickle.dump(sink, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


# ------------------------------------------------------------------ parent side
def run_child(name, env_extra, tag):
    model, task, _ = TASKS[name]
    path = os.path.join(WORK, "models", model, "py", task + ".py")
    os.makedirs(os.path.join(os.path.dirname(path), "..", "results"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "_tmp"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    sink = os.path.join(OUT, "_tmp", f"sink_{name}_{tag}.pkl")
    env = dict(os.environ, E3_SINK=sink, E3_HOME=E3_DIR, CPCM_HOME=HERE,
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "_inject_v2")] +
                                          ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
               **env_extra)
    log = os.path.join(OUT, "logs", f"{name}_{tag}.log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        rc = subprocess.call([sys.executable, path], env=env, cwd=E3_DIR, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if rc != 0 or not os.path.exists(sink):
        raise RuntimeError(f"{name} {tag}: exit {rc} or no sink; see {log}")
    with open(sink, "rb") as f:
        out = pickle.load(f)
    os.remove(sink)
    if out["calls"] != 2:
        raise RuntimeError(f"{name} {tag}: expected 2 backend calls, saw {out['calls']}")
    out["process_wall"] = wall
    return out


def capture_one(name):
    pre = run_child(name, {"E3_MODE": "capture", "E3_SEEDS": ""}, "capture")
    e3 = _e3()
    faults = {}
    for fid in FAULTS:
        try:
            faults[fid] = e3.choose_fault({"final": pre["last"]}, fid)
        except Exception as ex:  # noqa: BLE001  recorded, the point is skipped
            faults[fid] = {"error": repr(ex)}
    keep = {"mapping": pre["mapping"], "final_result": pre["last"]["result"], "faults": faults,
            "process_wall": pre["process_wall"], "eval_wall": pre.get("eval_wall")}
    with gzip.open(os.path.join(OUT, f"{name}_pre.pkl.gz"), "wb") as f:
        pickle.dump(keep, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[v2] capture {name}: template result {keep['final_result']}, {pre['process_wall']:.0f} s; faults "
          + ", ".join(f"{k}={v.get('failed_nodes') or v.get('failed_links') or v.get('failed_pes') or v.get('error')}"
                      for k, v in faults.items()), flush=True)
    return name


def inject_one(name, fid):
    with gzip.open(os.path.join(OUT, f"{name}_pre.pkl.gz"), "rb") as f:
        pre = pickle.load(f)
    fault = NOFAULT if fid == "F0" else pre["faults"][fid]
    if "error" in fault:
        print(f"[v2] {name} {fid}: skipped ({fault['error']})", flush=True)
        return None
    spec_path = os.path.join(OUT, "_tmp", f"spec_{name}_{fid}.pkl")
    os.makedirs(os.path.dirname(spec_path), exist_ok=True)
    with open(spec_path, "wb") as f:
        pickle.dump({"mapping": pre["mapping"], "faults": fault, "steps": None, "t_max": None, "seeds": SEEDS,
                     "preserve_routes": True}, f, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        sink = run_child(name, {"E3_MODE": "inject", "E3_SPEC": spec_path}, fid)
    except Exception as ex:  # noqa: BLE001
        print(f"[v2] {name} {fid}: FAILED {ex}", flush=True)
        return None
    finally:
        os.remove(spec_path)
    sink["fault_spec"] = fault
    with gzip.open(os.path.join(OUT, f"{name}_{fid}.pkl.gz"), "wb") as f:
        pickle.dump(sink, f, protocol=pickle.HIGHEST_PROTOCOL)
    ts = [r["result"][0] for r in sink["runs"]]
    print(f"[v2] {name} {fid}: backend {ts}, {sink['process_wall']:.0f} s "
          f"(backend {sum(r['wall'] for r in sink['runs']):.0f} s)", flush=True)
    return name, fid


def bimodal(ts):
    """Two clear clusters: split at the largest gap of the sorted values; both sides non-empty, the gap >= 2% of
    the mean and >= 3x the larger within-cluster spread."""
    s = sorted(ts)
    gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    if not gaps:
        return False
    i = max(range(len(gaps)), key=lambda k: gaps[k])
    lo, hi = s[:i + 1], s[i + 1:]
    spread = max(lo[-1] - lo[0], hi[-1] - hi[0])
    return gaps[i] >= 0.02 * statistics.mean(s) and gaps[i] >= 3 * spread


def analyse():
    rows = []
    for name, (model, task, regime) in TASKS.items():
        for fid in ["F0"] + FAULTS:
            p = os.path.join(OUT, f"{name}_{fid}.pkl.gz")
            if not os.path.exists(p):
                continue
            with gzip.open(p, "rb") as f:
                sink = pickle.load(f)
            g = sink["graph"]
            t0 = time.perf_counter()
            sc = cpcm.simulate(g, "bb")
            t_sim = time.perf_counter() - t0
            ts = [r["result"][0] for r in sink["runs"]]
            comm = [r["result"][2] / r["result"][0] for r in sink["runs"]]
            mean = statistics.mean(ts)
            lo, hi = min(ts), max(ts)
            err = (sc.T - mean) / mean
            dist_out = 0.0 if lo <= sc.T <= hi else min(abs(sc.T - lo), abs(sc.T - hi)) / mean
            rows.append({"task": name, "regime": regime, "fault": fid,
                         "fault_kind": sink["fault_spec"].get("kind", "none"),
                         "events": g.n, "cpcm_T": sc.T, "backend_mean": round(mean, 1), "backend_min": lo,
                         "backend_max": hi, "backend_std_over_mean": round(statistics.pstdev(ts) / mean, 5),
                         "backend_range_over_mean": round((hi - lo) / mean, 5), "backend_seeds": ts,
                         "err_signed": round(err, 5), "err_abs": round(abs(err), 5), "in_range": lo <= sc.T <= hi,
                         "dist_outside_range": round(dist_out, 5), "bimodal": bimodal(ts),
                         "comm_share": round(statistics.mean(comm), 3), "cpcm_s": round(t_sim, 4),
                         "backend_s_mean": round(statistics.mean(r["wall"] for r in sink["runs"]), 2),
                         "compile_s": round(sink["compile_s"], 3)})
    for r in rows:
        r["exempt"] = r["err_abs"] > 0.10 and r["bimodal"] and r["dist_outside_range"] <= 0.03
    errs = [r["err_abs"] for r in rows]
    nonexempt = [r["err_abs"] for r in rows if not r["exempt"]]

    def stats(v):
        return {"n": len(v), "median": round(statistics.median(v), 5), "max": round(max(v), 5),
                "mean": round(statistics.mean(v), 5)} if v else {"n": 0}

    groups = {}
    for key in ["regime", "task", "fault"]:
        groups[key] = {k: stats([r["err_abs"] for r in rows if r[key] == k]) for k in dict.fromkeys(r[key] for r in rows)}
    noise = [r["backend_range_over_mean"] for r in rows]
    summ = {"points": len(rows), "err_all": stats(errs), "err_nonexempt": stats(nonexempt),
            "exempt_points": [(r["task"], r["fault"]) for r in rows if r["exempt"]],
            "stop_median_gt_3pct": statistics.median(errs) > 0.03 if errs else None,
            "stop_max_gt_10pct_nonexempt": max(nonexempt) > 0.10 if nonexempt else None,
            "in_range": sum(r["in_range"] for r in rows), "bimodal_points": sum(r["bimodal"] for r in rows),
            "backend_range_over_mean": stats(noise),
            "backend_std_over_mean": stats([r["backend_std_over_mean"] for r in rows]),
            "groups": groups, "seeds": SEEDS,
            "artifact_commit": open(COMMIT_FILE, encoding="utf-8-sig").readline().strip()}
    with open(os.path.join(HERE, "cpcm_out", "v2_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, "cpcm_out", "v2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, default=str)
    for r in rows:
        print(f"{r['task']:6s} {r['fault']} {r['regime']:17s} T^={r['cpcm_T']:>11} mean={r['backend_mean']:>13} "
              f"[{r['backend_min']},{r['backend_max']}] err={r['err_signed']:+.4f} range/mean={r['backend_range_over_mean']:.4f}"
              f"{' BIMODAL' if r['bimodal'] else ''}{' EXEMPT' if r['exempt'] else ''}")
    print(json.dumps({k: v for k, v in summ.items() if k != "groups"}, default=str))
    print(json.dumps(summ["groups"], default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["capture", "inject", "analyse"])
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--faults", nargs="*", default=["F0"] + FAULTS)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if a.cmd == "capture":
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(capture_one, a.tasks))
    elif a.cmd == "inject":
        pts = [(n, f) for n in a.tasks for f in a.faults]
        pts.sort(key=lambda p: p[0] != "moe")       # longest first
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(lambda p: inject_one(*p), pts))
    else:
        analyse()
    print(f"[v2] {a.cmd} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
