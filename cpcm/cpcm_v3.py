"""V3: ranking fidelity on SA-trajectory mappings (cpcm/CPCM_HANDOFF.md section 3), sampled as E1 does.

Child (one task script per process, `runpy` as e1_fidelity.py): StreamOptimization.anneal is replaced in memory by a
copy of the artifact's loop (the same loop E1 replays) that, at steps 0, K, 2K, ..., 1000, records the SA energy
(BusyBarn's aggregate cost) and its parts; for the samples this process owns (sample index % nparts == part) it
  1. deep-copies (beha_dict, data_dict, event_dict, platform) and runs the template's final add_broadcast with
     random / numpy seeded BCAST_SEED (one fixed event graph per state; E1 instead reseeded add_broadcast per
     backend seed),
  2. compiles that graph (cpcm.compile_bb) and runs CPCM bb mode,
  3. runs the original event_driver once per backend seed, each on a deepcopy, random.seed(seed).
The RNG state is saved before and restored after every sample, so the SA trajectory is the unpatched one; every
process records the energy at every sample step and the parent checks the parts agree (same trajectory).
Parent: launches the parts in parallel, merges, and reports three Kendall tau-b (E1's kendall_tau_b):
  energy vs backend mean; backend group A mean vs group B mean (noise ceiling); CPCM vs backend mean,
on all samples and on distinct SA states (E1's state key without latency).

Usage: python cpcm_v3.py run --tasks ffn mha ... ; python cpcm_v3.py analyse
"""
import argparse
import copy
import csv
import gzip
import json
import math
import os
import pickle
import random
import runpy
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpcm  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "e1_fidelity"))
from e1_fidelity import SRC_SUBDIRS, _find_context, kendall_tau_b, spearman  # noqa: E402

sys.path.pop(0)

WORK = os.path.join(HERE, "work", "artifact")
OUT = os.path.join(HERE, "cpcm_out", "v3")
COMMIT_FILE = os.path.join(HERE, "..", "third_party", "isca2026-busybarn-artifact.COMMIT")
BCAST_SEED = 0
SUF = "_bw256_co4x4_bw256_t128x64_failpattern0"
# name -> (model folder, task, every K steps, backend seeds, processes)
PLAN = {
    "ffn": ("intra_mapping_gpt_decode", "ffn_sq2048_greedy0_lossratio6_ch2x2" + SUF, 10, [1, 2, 3, 4], 2),
    "mha": ("intra_mapping_gpt_decode", "mha_sq2048_greedy0_lossratio4_ch2x2" + SUF, 20, [1, 2, 3, 4], 4),
    "moe": ("intra_mapping_qwen3moe_decode", "ffn_sq2048_greedy0_lossratio5_ch2x2" + SUF, 20, [1, 2], 8),
    "ffn14": ("intra_mapping_gpt_decode", "ffn_sq2048_greedy1_lossratio4_ch1x4" + SUF, 10, [1, 2, 3, 4], 2),
    "opt": ("intra_mapping_ch", "ffn_sq2048_greedy0_lossratio3_ch2x2" + SUF, 10, [1, 2, 3, 4], 2),
}
STATE_KEY = ("energy", "hops", "comm_dist", "max_link_load", "max_tc_load", "max_vu_load")


# ------------------------------------------------------------------ child
def child(name, part, nparts, every, seeds, sink_path):
    model, task = PLAN[name][:2]
    task_path = os.path.join(WORK, "models", model, "py", task + ".py")
    for sub in SRC_SUBDIRS:
        p = os.path.join(WORK, sub)
        if p not in sys.path:
            sys.path.append(p)
    import Stream_Mapping
    from add_communication import add_broadcast
    from event_driver import event_driver

    sink = {"rows": [], "energies": {}, "meta": {}}

    def evaluate(self, st, step, E, T, ctx):
        rng = random.getstate(), np.random.get_state()
        try:
            t0 = time.time()
            beha, data, events, hp = copy.deepcopy((st[5], st[6], st[7], self.hardware_platform))
            random.seed(BCAST_SEED)
            np.random.seed(BCAST_SEED)
            add_broadcast(data_tags=ctx["broadcast_datatags"],
                          ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]],
                          beha_dict=beha, data_dict=data, hardware_platform=hp, event_dict=events,
                          dijkstra_routing=ctx["dijkstra_routing"],
                          alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])
            deps = {t: set(e.dependency_set) for t, e in events.items()}
            paths = {t: [list(b) for b in e.path_list] for t, e in events.items()
                     if e.event_type == "communication"}
            base = copy.deepcopy((events, hp))
            t1 = time.time()
            g = cpcm.compile_bb(events, hp, path_lists=paths, deps=deps)
            t2 = time.time()
            sc = cpcm.simulate(g, "bb")
            t3 = time.time()
            bk = {}
            bwall = []
            for s in seeds:
                ev_k, hp_k = copy.deepcopy(base)
                random.seed(s)
                np.random.seed(s)
                tb = time.time()
                out = event_driver(events_dict=ev_k, hardware_platform=hp_k)
                bwall.append(time.time() - tb)
                bk[s] = tuple(int(x) for x in out)
        finally:
            random.setstate(rng[0])
            np.random.set_state(rng[1])
        sink["rows"].append({
            "step": step, "temperature": T, "energy": float(E),
            "hops": float(st[0]), "comm_dist": float(st[1]),
            "max_link_load": float(max(st[2].values())) if st[2] else 0.0,
            "max_tc_load": float(max(st[3].values())) if st[3] else 0.0,
            "max_vu_load": float(max(st[4].values())) if st[4] else 0.0,
            "events": g.n, "cpcm_T": sc.T, "backend": bk,
            "prep_s": t1 - t0, "compile_s": t2 - t1, "cpcm_s": t3 - t2, "backend_s": bwall})

    def anneal(self):
        # identical to Stream_Mapping.StreamOptimization.anneal (as replayed by e1_fidelity.py), plus sampling
        random.seed(123)
        np.random.seed(123)
        ctx = _find_context()
        step = 0
        self.start = time.time()
        Tfactor = -math.log(self.Tmax / self.Tmin)
        T = self.Tmax
        E = self.energy()
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials = accepts = improves = 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)
        k = 0

        def sample(st, E, T):
            nonlocal k
            sink["energies"][step] = float(E)
            if k % nparts == part:
                evaluate(self, st, step, E, T, ctx)
                print(f"[v3] {name} part {part}: step {step} done", flush=True)
            k += 1

        sample(self.state, E, T)
        while step < self.steps and not self.user_exit:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                self.revert()
                E = prevEnergy
            else:
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(step, T, E, accepts / trials, improves / trials)
                    trials = accepts = improves = 0
            if step % every == 0:
                sample(self.state, E, T)
        sink["meta"] = {"steps": int(self.steps), "loss_ratio": list(self.loss_ratio)}
        with gzip.open(sink_path, "wb") as f:
            pickle.dump(sink, f, protocol=pickle.HIGHEST_PROTOCOL)
        raise SystemExit(0)    # samples taken: skip the template's own final evaluation

    Stream_Mapping.StreamOptimization.anneal = anneal
    os.makedirs(os.path.join(os.path.dirname(task_path), "..", "results"), exist_ok=True)
    runpy.run_path(task_path, run_name="__main__")


# ------------------------------------------------------------------ parent
def run_part(name, part):
    model, task, every, seeds, nparts = PLAN[name]
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    sink = os.path.join(OUT, f"{name}_part{part}.pkl.gz")
    log = os.path.join(OUT, "logs", f"{name}_part{part}.log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), "child", "--name", name, "--part", str(part),
                              "--nparts", str(nparts), "--every", str(every), "--seeds", *map(str, seeds),
                              "--sink", sink], cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
    ok = rc == 0 and os.path.exists(sink)
    print(f"[v3] {name} part {part}/{nparts}: exit {rc}, {time.time() - t0:.0f} s, sink {'ok' if ok else 'MISSING'}",
          flush=True)
    return ok


def run(tasks, jobs):
    jobs_list = [(n, p) for n in tasks for p in range(PLAN[n][4])]
    jobs_list.sort(key=lambda x: {"moe": 0, "mha": 1}.get(x[0], 2))
    with ThreadPoolExecutor(jobs or len(jobs_list)) as ex:
        return all(ex.map(lambda x: run_part(*x), jobs_list))


def _unique(rows):
    seen, out = set(), []
    for r in rows:
        key = tuple(round(r[k], 9) for k in STATE_KEY)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def taus(rows, seeds):
    a, b = seeds[:len(seeds) // 2], seeds[len(seeds) // 2:]
    mean = [statistics.mean(r["backend"][s][0] for s in seeds) for r in rows]
    ga = [statistics.mean(r["backend"][s][0] for s in a) for r in rows]
    gb = [statistics.mean(r["backend"][s][0] for s in b) for r in rows]
    en = [r["energy"] for r in rows]
    cp = [r["cpcm_T"] for r in rows]
    return {"n": len(rows), "tau_energy_vs_backend": kendall_tau_b(en, mean),
            "tau_backendA_vs_backendB": kendall_tau_b(ga, gb), "tau_cpcm_vs_backend": kendall_tau_b(cp, mean),
            "tau_cpcm_vs_backendA": kendall_tau_b(cp, ga), "tau_cpcm_vs_backendB": kendall_tau_b(cp, gb),
            "tau_energy_vs_cpcm": kendall_tau_b(en, cp),
            "spearman_energy_vs_backend": spearman(en, mean), "spearman_cpcm_vs_backend": spearman(cp, mean),
            "seeds_A": a, "seeds_B": b}


def analyse(tasks):
    summ, table = {}, []
    for name in tasks:
        model, task, every, seeds, nparts = PLAN[name]
        parts = []
        for p in range(nparts):
            with gzip.open(os.path.join(OUT, f"{name}_part{p}.pkl.gz"), "rb") as f:
                parts.append(pickle.load(f))
        en0 = parts[0]["energies"]
        same = all(pt["energies"] == en0 for pt in parts)
        rows = sorted((r for pt in parts for r in pt["rows"]), key=lambda r: r["step"])
        for r in rows:
            m = statistics.mean(r["backend"][s][0] for s in seeds)
            table.append({"task": name, "step": r["step"], "energy": r["energy"], "cpcm_T": r["cpcm_T"],
                          "backend_mean": round(m, 1), **{f"backend_s{s}": r["backend"][s][0] for s in seeds},
                          "comm_share": round(statistics.mean(r["backend"][s][2] / r["backend"][s][0] for s in seeds), 4),
                          "err_signed": round((r["cpcm_T"] - m) / m, 5), "events": r["events"],
                          "cpcm_s": round(r["cpcm_s"], 4), "backend_s_mean": round(statistics.mean(r["backend_s"]), 2)})
        uniq = _unique(rows)
        errs = [abs(t["err_signed"]) for t in table if t["task"] == name]
        summ[name] = {"task": task, "every": every, "seeds": seeds, "nparts": nparts,
                      "trajectory_identical_across_parts": same, "samples": len(rows),
                      "steps": [rows[0]["step"], rows[-1]["step"]] if rows else None,
                      "all": taus(rows, seeds), "unique": taus(uniq, seeds),
                      "abs_err_median": round(statistics.median(errs), 5), "abs_err_max": round(max(errs), 5),
                      "backend_s_mean": round(statistics.mean(t["backend_s_mean"] for t in table if t["task"] == name), 2),
                      "cpcm_s_mean": round(statistics.mean(t["cpcm_s"] for t in table if t["task"] == name), 4),
                      "meta": parts[0]["meta"], "template_sa_seed": 123, "bcast_seed": BCAST_SEED}
    summ["_artifact_commit"] = open(COMMIT_FILE, encoding="utf-8-sig").readline().strip()
    with open(os.path.join(HERE, "cpcm_out", "v3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    keys = list(dict.fromkeys(k for t in table for k in t))
    with open(os.path.join(HERE, "cpcm_out", "v3_samples.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    for name in tasks:
        s = summ[name]
        for k in ("all", "unique"):
            t = s[k]
            print(f"{name:6s} {k:6s} n={t['n']:3d} tau energy={t['tau_energy_vs_backend']:+.3f} "
                  f"backendA-B={t['tau_backendA_vs_backendB']:+.3f} cpcm={t['tau_cpcm_vs_backend']:+.3f}")
        print(f"       same trajectory across parts: {s['trajectory_identical_across_parts']}; |err| median "
              f"{s['abs_err_median']:.4f} max {s['abs_err_max']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyse", "child"])
    ap.add_argument("--tasks", nargs="*", default=list(PLAN))
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--name")
    ap.add_argument("--part", type=int)
    ap.add_argument("--nparts", type=int)
    ap.add_argument("--every", type=int)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--sink")
    a = ap.parse_args()
    if a.cmd == "child":
        child(a.name, a.part, a.nparts, a.every, a.seeds, a.sink)
        return
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if a.cmd == "run":
        ok = run(a.tasks, a.jobs)
        print(f"[v3] run {'ok' if ok else 'with failures'} in {time.time() - t0:.0f} s", flush=True)
    else:
        analyse(a.tasks)


if __name__ == "__main__":
    main()
