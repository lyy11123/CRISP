"""E1: does BusyBarn's aggregate cost rank mappings the way its own backend does?

For each BusyBarn/Gemini task script, this runner replays the artifact's SA loop
unchanged and, every K steps, evaluates the current mapping twice:
  - aggregate cost : the SA energy BusyBarn actually minimises (plus its raw parts)
  - backend latency: the artifact's own final evaluation (add_broadcast + event_driver)
It then reports Kendall tau / Spearman rho between the two and the share of
"near-tie" pairs (cost within 1%) whose latency differs by more than z%.

Nothing in the artifact is modified on disk; the SA loop is patched in memory.
RNG state is saved and restored around every backend evaluation, so the SA
trajectory is identical to an unpatched run.

The backend breaks ties randomly, so some states are re-evaluated under several
seeds; latency gaps are also reported against 3x that measured noise. Metrics
are given for all samples and for distinct states only (prefix "unique_"),
because SA often sits on one mapping for many steps.

Usage (from the artifact root, after `make generate_exp` in the model dir;
one task per process, because each task script builds module-level state):
  python e1_fidelity.py --task models/intra_mapping_gpt_decode/py/<task>.py --every 5 --out e1_out
"""
import argparse
import copy
import csv
import itertools
import json
import math
import os
import random
import re
import runpy
import sys
import time

import numpy as np

SRC_SUBDIRS = [
    "utils", "src/partition/oper", "src/partition/func", "src/partition",
    "src/mapping", "src/scheduling", "src/backend/analytical",
    "src/scheduling/communication/topology", "tool",
]
CONTEXT_NAMES = ["broadcast_datatags", "layers_regions", "initialization_layer_tag",
                 "dijkstra_routing", "alpha", "beta", "gamma"]


# ---------------------------------------------------------------- statistics
def _ranks(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    # average ranks over ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau_b(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    conc = disc = tx = ty = 0
    for i, j in itertools.combinations(range(n), 2):
        dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx == dy:
            conc += 1
        else:
            disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / denom) if denom else float("nan")


def pair_stats(cost, lat, tie_tol=0.01, zs=(0.05, 0.10)):
    """Near-tie pairs: |dC|/min(C) < tie_tol. Inversions: cost prefers one mapping
    by more than tie_tol while latency prefers the other by more than z."""
    cost, lat = np.asarray(cost, float), np.asarray(lat, float)
    n = len(cost)
    out = {"pairs": n * (n - 1) // 2, "near_tie_pairs": 0}
    hits = {z: 0 for z in zs}
    inv = {z: 0 for z in zs}
    decided = 0
    for i, j in itertools.combinations(range(n), 2):
        dc = abs(cost[i] - cost[j]) / max(min(cost[i], cost[j]), 1e-12)
        dl = abs(lat[i] - lat[j]) / max(min(lat[i], lat[j]), 1e-12)
        if dc < tie_tol:
            out["near_tie_pairs"] += 1
            for z in zs:
                if dl > z:
                    hits[z] += 1
        else:
            decided += 1
            cost_prefers_i = cost[i] < cost[j]
            lat_prefers_i = lat[i] < lat[j]
            if cost_prefers_i != lat_prefers_i:
                for z in zs:
                    if dl > z:
                        inv[z] += 1
    out["decided_pairs"] = decided
    for z in zs:
        tag = f"{int(round(z * 100))}pct"
        out[f"near_tie_latency_gap_gt_{tag}"] = hits[z]
        out[f"near_tie_share_gap_gt_{tag}"] = (hits[z] / out["near_tie_pairs"]) if out["near_tie_pairs"] else float("nan")
        out[f"inversions_gap_gt_{tag}"] = inv[z]
        out[f"inversion_share_gap_gt_{tag}"] = (inv[z] / decided) if decided else float("nan")
    return out


STATE_KEY = ("energy", "lat_total", "hops", "comm_dist", "max_link_load", "max_tc_load", "max_vu_load")


def unique_rows(rows):
    """SA often sits on the same mapping for many steps; keep one row per distinct state."""
    seen, out = set(), []
    for r in rows:
        key = tuple(round(r[k], 9) if isinstance(r[k], float) else r[k] for k in STATE_KEY)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def summarize(rows, tie_tol, noise_z=None):
    cost = [r["energy"] for r in rows]
    lat = [r["lat_total"] for r in rows]
    s = {
        "n_samples": len(rows),
        "kendall_tau_energy_vs_latency": kendall_tau_b(cost, lat),
        "spearman_energy_vs_latency": spearman(cost, lat),
    }
    for comp in ["hops", "comm_dist", "max_link_load", "max_tc_load", "max_vu_load"]:
        s[f"spearman_{comp}_vs_latency"] = spearman([r[comp] for r in rows], lat)
    s.update(pair_stats(cost, lat, tie_tol=tie_tol))
    if noise_z:
        ps = pair_stats(cost, lat, tie_tol=tie_tol, zs=(noise_z,))
        tag = f"{int(round(noise_z * 100))}pct"
        s["noise_floor_gap"] = noise_z
        s["near_tie_share_gap_gt_noise_floor"] = ps[f"near_tie_share_gap_gt_{tag}"]
        s["inversion_share_gap_gt_noise_floor"] = ps[f"inversion_share_gap_gt_{tag}"]
    half = rows[len(rows) // 2:]
    if len(half) >= 3:
        s["late_half_kendall_tau"] = kendall_tau_b([r["energy"] for r in half], [r["lat_total"] for r in half])
        s["late_half_spearman"] = spearman([r["energy"] for r in half], [r["lat_total"] for r in half])
    return s


def noise_summary(noise_rows):
    """Relative spread of backend latency for the same state under different seeds."""
    by_step = {}
    for r in noise_rows:
        by_step.setdefault(r["step"], []).append(r["lat_total"])
    cvs, ranges = [], []
    for vals in by_step.values():
        v = np.asarray(vals, float)
        if len(v) >= 2 and v.mean() > 0:
            cvs.append(v.std() / v.mean())
            ranges.append((v.max() - v.min()) / v.min())
    if not cvs:
        return {}
    return {"noise_states": len(cvs), "noise_mean_cv": float(np.mean(cvs)),
            "noise_max_cv": float(np.max(cvs)), "noise_max_rel_range": float(np.max(ranges))}


# ---------------------------------------------------------- artifact hooking
def _find_context():
    """Walk up the call stack to the task script's module frame and read the
    names its own final evaluation uses."""
    frame = sys._getframe(1)
    while frame is not None:
        g = frame.f_globals
        if all(name in g for name in CONTEXT_NAMES):
            return {name: g[name] for name in CONTEXT_NAMES}
        frame = frame.f_back
    raise RuntimeError("could not find the task script's globals "
                       f"({', '.join(CONTEXT_NAMES)}); is this a BusyBarn/Gemini template script?")


def _backend_latency(state, platform, ctx, add_broadcast, event_driver, eval_seed):
    """Mirror the template's final evaluation on deep copies of the given state.
    Returns (latency triple, seconds copying, seconds in add_broadcast, seconds in event_driver)."""
    t0 = time.time()
    beha, data, events, hp = copy.deepcopy((state[5], state[6], state[7], platform))
    t1 = time.time()
    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    try:
        add_broadcast(
            data_tags=ctx["broadcast_datatags"],
            ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]],
            beha_dict=beha, data_dict=data, hardware_platform=hp, event_dict=events,
            dijkstra_routing=ctx["dijkstra_routing"],
            alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"],
        )
        t2 = time.time()
        lat = event_driver(events_dict=events, hardware_platform=hp)
        t3 = time.time()
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return lat, t1 - t0, t2 - t1, t3 - t2


def install_patch(Stream_Mapping, add_broadcast, event_driver, every, eval_seed, sink,
                  noise_seeds=5, noise_every=100):
    Annealer = Stream_Mapping.StreamOptimization

    def anneal(self):
        # ---- identical to Stream_Mapping.StreamOptimization.anneal, plus sampling ----
        random.seed(123)
        np.random.seed(123)
        ctx = _find_context()
        step = 0
        self.start = time.time()
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
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

        eval_time = 0.0

        def evaluate(st, seed):
            nonlocal eval_time
            t0 = time.time()
            res = _backend_latency(st, self.hardware_platform, ctx, add_broadcast, event_driver, seed)
            eval_time += time.time() - t0
            return res

        def sample(tag, st, energy_value, temperature):
            lat, t_copy, t_bcast, t_drv = evaluate(st, eval_seed)
            sink["rows"].append({
                "step": tag, "temperature": temperature, "energy": float(energy_value),
                "hops": float(st[0]), "comm_dist": float(st[1]),
                "max_link_load": float(max(st[2].values())) if st[2] else 0.0,
                "max_tc_load": float(max(st[3].values())) if st[3] else 0.0,
                "max_vu_load": float(max(st[4].values())) if st[4] else 0.0,
                "lat_total": int(lat[0]), "lat_pure_comp": int(lat[1]), "lat_pure_comm": int(lat[2]),
                "eval_seconds": t_copy + t_bcast + t_drv,
                "copy_seconds": t_copy, "broadcast_seconds": t_bcast, "driver_seconds": t_drv,
            })
            # Backend tie-breaking is random: re-evaluate some states under other seeds.
            if noise_seeds > 1 and tag % noise_every == 0:
                sink["noise"].append({"step": tag, "seed": eval_seed, "lat_total": int(lat[0])})
                for k in range(1, noise_seeds):
                    lat_k = evaluate(st, eval_seed + k)[0]
                    sink["noise"].append({"step": tag, "seed": eval_seed + k, "lat_total": int(lat_k[0])})

        sample(0, self.state, E, T)
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
                sample(step, self.state, E, T)

        sa_wall = time.time() - self.start - eval_time
        # The template evaluates the dicts it passed in, which SA mutates in place,
        # i.e. the last trajectory state; SA itself returns best_state. Record both,
        # each under several seeds.
        seeds = range(eval_seed, eval_seed + max(noise_seeds, 1))
        final_lats = [int(evaluate(self.state, s)[0][0]) for s in seeds]
        best_lats = [int(evaluate(self.best_state, s)[0][0]) for s in seeds]
        sink["end"] = {
            "final_trajectory_energy": float(prevEnergy), "final_trajectory_latency": final_lats[0],
            "final_trajectory_latency_min": min(final_lats), "final_trajectory_latency_max": max(final_lats),
            "best_energy": float(self.best_energy), "best_state_latency": best_lats[0],
            "best_state_latency_min": min(best_lats), "best_state_latency_max": max(best_lats),
            "sa_wall_seconds_excluding_samples": sa_wall,
            "sa_seconds_per_step": sa_wall / max(int(self.steps), 1),
            "backend_eval_seconds_total": eval_time,
            "steps": int(self.steps), "hops_only": bool(self.hops_only),
            "loss_ratio": list(self.loss_ratio),
        }

        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()
        if self.verbose:
            print("Best energy: ", self.best_energy)
        return self.best_state, self.best_energy

    Annealer.anneal = anneal


# ------------------------------------------------------------------ driver
def _template_reported(task_path):
    """The task script writes its own final result to ../results/<name>.txt."""
    name = os.path.splitext(os.path.basename(task_path))[0]
    path = os.path.join(os.path.dirname(task_path), "../results", name + ".txt")
    if not os.path.isfile(path):
        return None
    m = None
    with open(path) as f:
        for line in f:
            hit = re.search(r"SA time cost:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", line)
            if hit:
                m = hit
    return int(m.group(1)) if m else None


def run_task(task_path, every, eval_seed, tie_tol, out_dir, noise_seeds=5, noise_every=100):
    task_path = os.path.abspath(task_path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(task_path), "../../.."))
    for sub in SRC_SUBDIRS:
        p = os.path.join(repo_root, sub)
        if p not in sys.path:
            sys.path.append(p)

    import Stream_Mapping  # noqa: E402  (same module object the task script will import)
    from add_communication import add_broadcast  # noqa: E402
    from event_driver import event_driver  # noqa: E402

    sink = {"rows": [], "noise": [], "end": None}
    install_patch(Stream_Mapping, add_broadcast, event_driver, every, eval_seed, sink,
                  noise_seeds=noise_seeds, noise_every=noise_every)

    os.makedirs(os.path.join(os.path.dirname(task_path), "../results"), exist_ok=True)
    t0 = time.time()
    runpy.run_path(task_path, run_name="__main__")
    wall = time.time() - t0

    if not sink["rows"]:
        raise RuntimeError("no samples captured: the task did not call StreamOptimization.anneal "
                           "(it may use the V2 optimizer, which this runner does not patch)")

    name = os.path.splitext(os.path.basename(task_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sink["rows"][0].keys()))
        w.writeheader()
        w.writerows(sink["rows"])

    if sink["noise"]:
        with open(os.path.join(out_dir, f"{name}_noise.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "seed", "lat_total"])
            w.writeheader()
            w.writerows(sink["noise"])

    rows = sink["rows"]
    summary = {"task": name, "task_wall_seconds": wall}
    for k in ["eval_seconds", "copy_seconds", "broadcast_seconds", "driver_seconds"]:
        summary[f"mean_{k}"] = float(np.mean([r[k] for r in rows]))
    summary.update(sink["end"] or {})
    rep = _template_reported(task_path)
    summary["template_reported_latency"] = rep
    if rep is not None and sink["end"]:
        e = sink["end"]
        summary["template_matches_final_state"] = e["final_trajectory_latency_min"] <= rep <= e["final_trajectory_latency_max"]
        summary["template_matches_best_state"] = e["best_state_latency_min"] <= rep <= e["best_state_latency_max"]
    ns = noise_summary(sink["noise"])
    summary.update(ns)
    # Count a latency gap only if it clears 3x the measured seed-to-seed noise.
    noise_z = 3 * ns["noise_mean_cv"] if ns.get("noise_mean_cv") else None
    summary.update(summarize(rows, tie_tol, noise_z))
    uniq = unique_rows(rows)
    for k, v in summarize(uniq, tie_tol, noise_z).items():
        summary[f"unique_{k}"] = v
    with open(os.path.join(out_dir, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e = np.array([r["energy"] for r in sink["rows"]])
        lat = np.array([r["lat_total"] for r in sink["rows"]])
        steps = np.array([r["step"] for r in sink["rows"]])
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(e, lat / lat.min(), c=steps, s=12, cmap="viridis")
        ax.set_xlabel("aggregate cost (SA energy)")
        ax.set_ylabel("backend latency / min")
        ax.set_title(name[:60], fontsize=8)
        fig.colorbar(sc, label="SA step")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150)
        plt.close(fig)
    except Exception as exc:  # plotting is optional
        print(f"[e1] plot skipped: {exc}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True,
                    help="path to one generated task script (run one task per process)")
    ap.add_argument("--every", type=int, default=5, help="sample every K SA steps (default 5)")
    ap.add_argument("--eval-seed", type=int, default=0, help="fixed seed for backend tie-breaking")
    ap.add_argument("--tie-tol", type=float, default=0.01, help="relative cost gap counted as a near tie")
    ap.add_argument("--noise-seeds", type=int, default=5,
                    help="seeds per noise check (1 disables; default 5)")
    ap.add_argument("--noise-every", type=int, default=100,
                    help="run a noise check every N SA steps (default 100)")
    ap.add_argument("--out", default="e1_out")
    args = ap.parse_args()

    print(f"[e1] running {args.task}", flush=True)
    s = run_task(args.task, args.every, args.eval_seed, args.tie_tol, args.out,
                 noise_seeds=args.noise_seeds, noise_every=args.noise_every)
    keys = ["n_samples", "unique_n_samples",
            "kendall_tau_energy_vs_latency", "unique_kendall_tau_energy_vs_latency",
            "unique_near_tie_pairs", "unique_near_tie_share_gap_gt_5pct",
            "unique_near_tie_share_gap_gt_noise_floor", "unique_inversion_share_gap_gt_noise_floor",
            "noise_mean_cv", "noise_max_rel_range",
            "mean_copy_seconds", "mean_broadcast_seconds", "mean_driver_seconds", "sa_seconds_per_step",
            "best_energy", "best_state_latency_min", "best_state_latency_max",
            "final_trajectory_energy", "final_trajectory_latency_min", "final_trajectory_latency_max",
            "template_reported_latency", "template_matches_final_state", "template_matches_best_state"]
    print("[e1] " + json.dumps({k: s.get(k) for k in keys}), flush=True)


if __name__ == "__main__":
    main()
