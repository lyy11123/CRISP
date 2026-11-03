"""V4: faults (cpcm/CPCM_HANDOFF.md section 3). Record, do not stop.

errors  E3's 44 "task x fault" combinations (e3_warmstart/e3_out/e3_table.csv), two mappings each, read from E3's
        <task>_<F>_caps.pkl.gz: R0 and FULL (the best of E3's FULL runs, table column FULL_variant).
        R0 = the pre-fault mapping (E3's <task>_prefault.pkl.gz) injected onto the faulty platform with E3's
        inject mode, no SA (steps None: E3's own R0 relocation and rerouting). FULL = the FULL run's placements,
        data locations and routes with the pre-fault event layout (E3 stored no layout for FULL runs), injected
        the same way (nothing to relocate; routes restored, unmatched ones re-routed). Then at the final backend call cpcm_v2.inject_eval runs backend seeds 1-5 and compiles the graph; bb mode vs the
        5-seed mean, as V2.
tau     ffn (GPT decode FFN 2x2) with faults F1 node, F4 D2D, F5 critical link, F6 PE only, F8 DDR port (V2's): a
        post-fault warm-start SA trajectory from the pre-fault mapping (E3's W_s1000_t10: inject, 1000 steps,
        t_max 10), sampled every 25 steps (41 candidates) by cpcm_v3.make_anneal inside E3's inject mode; backend
        seeds 1-4 (groups {1,2} / {3,4}); three Kendall tau as V3.
Groups: node (F1, F2), busiest on-chip link (F3), D2D (F4), critical link (F5), PE only (F6, F7), DDR port (F8).

Usage: python cpcm_v4.py errors --jobs 14 ; python cpcm_v4.py tau --jobs 10 ; python cpcm_v4.py analyse
"""
import argparse
import csv
import gzip
import json
import os
import pickle
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpcm  # noqa: E402
import cpcm_v2  # noqa: E402
import cpcm_v3  # noqa: E402
from cpcm_v3 import kendall_tau_b  # noqa: E402

E3_OUT = os.path.join(HERE, "..", "e3_warmstart", "e3_out")
OUT = os.path.join(HERE, "cpcm_out", "v4")
SUF = "_bw256_co4x4_bw256_t128x64_failpattern0"
E3_TASKS = {   # e3_warmstart.TASKS, with the model folder
    "ffn": ("intra_mapping_gpt_decode", "ffn_sq2048_greedy0_lossratio6_ch2x2" + SUF),
    "mha": ("intra_mapping_gpt_decode", "mha_sq2048_greedy0_lossratio4_ch2x2" + SUF),
    "ffn14": ("intra_mapping_gpt_decode", "ffn_sq2048_greedy1_lossratio4_ch1x4" + SUF),
    "mha14": ("intra_mapping_gpt_decode", "mha_sq2048_greedy0_lossratio6_ch1x4" + SUF),
    "ffng": ("intra_mapping_gpt_decode", "ffn_sq2048_gemini_ch2x2" + SUF),
    "mhag": ("intra_mapping_gpt_decode", "mha_sq2048_gemini_ch2x2" + SUF),
}
GROUP = {"F1": "node", "F2": "node", "F3": "busiest on-chip link", "F4": "D2D link", "F5": "critical link",
         "F6": "PE only", "F7": "PE only", "F8": "DDR port"}
TAU_TASK, TAU_FAULTS, TAU_EVERY, TAU_SEEDS, TAU_PARTS = "ffn", ["F1", "F4", "F5", "F6", "F8"], 25, [1, 2, 3, 4], 2

cpcm_v2.OUT = OUT
for k, (m, t) in E3_TASKS.items():
    cpcm_v2.TASKS[k] = (m, t, "")


def combos():
    with open(os.path.join(E3_OUT, "e3_table.csv"), encoding="utf-8") as f:
        return [(r["task"], r["fault"], r["FULL_variant"]) for r in csv.DictReader(f)]


def load_caps(name, fid):
    with gzip.open(os.path.join(E3_OUT, f"{E3_TASKS[name][1]}_{fid}_caps.pkl.gz"), "rb") as f:
        return pickle.load(f)


def inject_point(name, fid, which, mapping, fault):
    tag = f"{fid}_{which}"
    dst = os.path.join(OUT, f"{name}_{tag}.pkl.gz")
    if os.path.exists(dst):
        return True
    spec_path = os.path.join(OUT, "_tmp", f"spec_{name}_{tag}.pkl")
    os.makedirs(os.path.dirname(spec_path), exist_ok=True)
    with open(spec_path, "wb") as f:
        pickle.dump({"mapping": mapping, "faults": fault, "steps": None, "t_max": None, "seeds": cpcm_v2.SEEDS,
                     "preserve_routes": True}, f, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        sink = cpcm_v2.run_child(name, {"E3_MODE": "inject", "E3_SPEC": spec_path}, tag)
    except Exception as ex:  # noqa: BLE001  recorded, not fatal (V4: record, do not stop)
        print(f"[v4] {name} {tag}: FAILED {ex}", flush=True)
        return False
    finally:
        os.remove(spec_path)
    sink["fault_spec"] = fault
    sink["log"] = {k: v for k, v in sink["log"].items() if k in ("relocated", "ddr_replacement", "removed_pe_devices")}
    with gzip.open(dst, "wb") as f:
        pickle.dump(sink, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[v4] {name} {tag}: backend {[r['result'][0] for r in sink['runs']]}, {sink['process_wall']:.0f} s", flush=True)
    return True


def errors(jobs):
    pts = []
    pre = {}
    for name, fid, variant in combos():
        if name not in pre:
            with gzip.open(os.path.join(E3_OUT, f"{E3_TASKS[name][1]}_prefault.pkl.gz"), "rb") as f:
                pre[name] = pickle.load(f)["mapping"]
        caps = load_caps(name, fid)
        full_key = f"{fid}_FULL_gemini" if variant == "gemini" else f"{fid}_FULL_{variant}"
        # R0 exactly as E3 built it: the pre-fault mapping (with its event layout) injected onto the faulty platform.
        # FULL: E3 stored no event layout for FULL runs; its placements, data locations and routes are injected
        # with the pre-fault layout (event order / comm-id allocators; unmatched tags are appended / relabelled).
        pts.append((name, fid, "R0", pre[name], caps["fault"]))
        pts.append((name, fid, "FULL", dict(caps["runs"][full_key]["mapping"], layout=pre[name]["layout"]), caps["fault"]))
    pts.sort(key=lambda p: not p[0].startswith("mha"))     # long ones first
    with ThreadPoolExecutor(jobs) as ex:
        ok = list(ex.map(lambda p: inject_point(*p), pts))
    print(f"[v4] errors: {sum(ok)}/{len(ok)} points ok", flush=True)


def tau_part(fid, part):
    name = TAU_TASK
    with gzip.open(os.path.join(HERE, "cpcm_out", "v2", f"{name}_pre.pkl.gz"), "rb") as f:
        pre = pickle.load(f)
    fault = pre["faults"][fid]
    sink = os.path.join(OUT, f"tau_{name}_{fid}_part{part}.pkl.gz")
    if os.path.exists(sink):
        return True
    spec_path = os.path.join(OUT, "_tmp", f"spec_tau_{fid}_{part}.pkl")
    os.makedirs(os.path.join(OUT, "_tmp"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    with open(spec_path, "wb") as f:
        pickle.dump({"mapping": pre["mapping"], "faults": fault, "steps": 1000, "t_max": 10, "seeds": [],
                     "preserve_routes": True}, f, protocol=pickle.HIGHEST_PROTOCOL)
    model, task = E3_TASKS[name]
    path = os.path.join(cpcm_v2.WORK, "models", model, "py", task + ".py")
    env = dict(os.environ, E3_MODE="inject", E3_SPEC=spec_path, E3_SINK=os.path.join(OUT, "_tmp", f"unused_{fid}_{part}"),
               E3_HOME=cpcm_v2.E3_DIR, CPCM_HOME=HERE,
               CPCM_V4_SAMPLE="|".join([f"{name}_{fid}", str(part), str(TAU_PARTS), str(TAU_EVERY), sink,
                                        " ".join(map(str, TAU_SEEDS))]),
               PYTHONPATH=os.path.join(HERE, "_inject_v2"))
    t0 = time.time()
    with open(os.path.join(OUT, "logs", f"tau_{fid}_part{part}.log"), "w", encoding="utf-8") as lf:
        rc = subprocess.call([sys.executable, path], env=env, cwd=cpcm_v2.E3_DIR, stdout=lf, stderr=subprocess.STDOUT)
    os.remove(spec_path)
    ok = rc == 0 and os.path.exists(sink)
    print(f"[v4] tau {fid} part {part}: exit {rc}, {time.time() - t0:.0f} s, {'ok' if ok else 'MISSING'}", flush=True)
    return ok


def tau(jobs):
    jl = [(f, p) for f in TAU_FAULTS for p in range(TAU_PARTS)]
    with ThreadPoolExecutor(jobs) as ex:
        ok = list(ex.map(lambda x: tau_part(*x), jl))
    print(f"[v4] tau: {sum(ok)}/{len(ok)} parts ok", flush=True)


def _stats(v):
    return {"n": len(v), "median": round(statistics.median(v), 5), "max": round(max(v), 5),
            "mean": round(statistics.mean(v), 5)} if v else {"n": 0}


def analyse():
    rows = []
    for name, fid, variant in combos():
        for which in ("R0", "FULL"):
            p = os.path.join(OUT, f"{name}_{fid}_{which}.pkl.gz")
            if not os.path.exists(p):
                rows.append({"task": name, "fault": fid, "group": GROUP[fid], "mapping": which, "status": "missing"})
                continue
            with gzip.open(p, "rb") as f:
                sink = pickle.load(f)
            sc = cpcm.simulate(sink["graph"], "bb")
            ts = [r["result"][0] for r in sink["runs"]]
            mean = statistics.mean(ts)
            lo, hi = min(ts), max(ts)
            err = (sc.T - mean) / mean
            rows.append({"task": name, "fault": fid, "group": GROUP[fid], "mapping": which, "status": "ok",
                         "full_variant": variant if which == "FULL" else "", "events": sink["graph"].n,
                         "cpcm_T": sc.T, "backend_mean": round(mean, 1), "backend_min": lo, "backend_max": hi,
                         "backend_seeds": ts, "err_signed": round(err, 5), "err_abs": round(abs(err), 5),
                         "in_range": lo <= sc.T <= hi,
                         "dist_outside_range": round(0.0 if lo <= sc.T <= hi else min(abs(sc.T - lo), abs(sc.T - hi)) / mean, 5),
                         "backend_range_over_mean": round((hi - lo) / mean, 5), "bimodal": cpcm_v2.bimodal(ts),
                         "comm_share": round(statistics.mean(r["result"][2] / r["result"][0] for r in sink["runs"]), 3)})
    ok = [r for r in rows if r["status"] == "ok"]
    for r in ok:
        r["exempt"] = r["err_abs"] > 0.10 and r["bimodal"] and r["dist_outside_range"] <= 0.03
    summ = {"points": len(rows), "ok": len(ok), "missing": [(r["task"], r["fault"], r["mapping"]) for r in rows if r["status"] != "ok"],
            "err_all": _stats([r["err_abs"] for r in ok]),
            "err_nonexempt": _stats([r["err_abs"] for r in ok if not r["exempt"]]),
            "exempt": [(r["task"], r["fault"], r["mapping"]) for r in ok if r["exempt"]],
            "over_10pct": [(r["task"], r["fault"], r["mapping"], r["err_signed"]) for r in ok if r["err_abs"] > 0.10],
            "in_range": sum(r["in_range"] for r in ok), "bimodal": sum(r["bimodal"] for r in ok),
            "backend_range_over_mean": _stats([r["backend_range_over_mean"] for r in ok])}
    for key in ("group", "mapping", "task"):
        summ[f"by_{key}"] = {k: _stats([r["err_abs"] for r in ok if r[key] == k]) for k in dict.fromkeys(r[key] for r in ok)}
    summ["by_group_mapping"] = {f"{g}|{m}": _stats([r["err_abs"] for r in ok if r["group"] == g and r["mapping"] == m])
                                for g in dict.fromkeys(r["group"] for r in ok) for m in ("R0", "FULL")}
    # tau part
    tau_res = {}
    for fid in TAU_FAULTS:
        parts = []
        for p in range(TAU_PARTS):
            fp = os.path.join(OUT, f"tau_{TAU_TASK}_{fid}_part{p}.pkl.gz")
            if os.path.exists(fp):
                with gzip.open(fp, "rb") as f:
                    parts.append(pickle.load(f))
        if len(parts) != TAU_PARTS:
            tau_res[fid] = {"status": "missing parts"}
            continue
        same = all(pt["energies"] == parts[0]["energies"] for pt in parts)
        rs = sorted((r for pt in parts for r in pt["rows"]), key=lambda r: r["step"])
        d = {"group": GROUP[fid], "trajectory_identical_across_parts": same, "all": cpcm_v3.taus(rs, TAU_SEEDS),
             "unique": cpcm_v3.taus(cpcm_v3._unique(rs), TAU_SEEDS)}
        m = [statistics.mean(r["backend"][s][0] for s in TAU_SEEDS) for r in rs]
        e = [abs(r["cpcm_T"] - x) / x for r, x in zip(rs, m)]
        d.update(abs_err_median=round(statistics.median(e), 5), abs_err_max=round(max(e), 5),
                 latency_range_over_min=round((max(m) - min(m)) / min(m), 4))
        tau_res[fid] = d
    summ["tau"] = {"task": TAU_TASK, "every": TAU_EVERY, "seeds": TAU_SEEDS, "sa": "inject pre-fault mapping, 1000 steps, t_max 10",
                   "faults": tau_res}
    with open(os.path.join(HERE, "cpcm_out", "v4_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, default=str)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(HERE, "cpcm_out", "v4_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: v for k, v in summ.items() if k != "tau"}, default=str))
    for fid, d in tau_res.items():
        if "all" in d:
            print(fid, d["group"], "n", d["all"]["n"], "tau energy %.3f A-B %.3f cpcm %.3f" % (
                d["all"]["tau_energy_vs_backend"], d["all"]["tau_backendA_vs_backendB"], d["all"]["tau_cpcm_vs_backend"]),
                "| unique n", d["unique"]["n"], "%.3f %.3f %.3f" % (d["unique"]["tau_energy_vs_backend"],
                d["unique"]["tau_backendA_vs_backendB"], d["unique"]["tau_cpcm_vs_backend"]),
                "| err", d["abs_err_median"], d["abs_err_max"], "| same", d["trajectory_identical_across_parts"])
        else:
            print(fid, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["errors", "tau", "analyse"])
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    {"errors": lambda: errors(a.jobs), "tau": lambda: tau(a.jobs), "analyse": analyse}[a.cmd]()
    print(f"[v4] {a.cmd} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
