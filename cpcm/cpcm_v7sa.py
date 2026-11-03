"""V7 addendum (user 2026-11-03): CPCM full vs incremental per step under REAL BusyBarn SA moves (the artifact's own
StreamOptimization.move(): a behaviour / weight split moves and update_event / update_offlinedata re-route its
messages, possibly creating and recycling communication tags), and the projected cost of 1,000 SA steps.

Child (one task script per process, runpy as V3): StreamOptimization.anneal is replaced by the artifact loop in
which, after every move() (before the Metropolis test), the SA state's event_dict (state[7]; the template adds the
final broadcasts only after SA, so they are not part of the evaluated graph) is evaluated twice, all timed with
perf_counter, random / numpy RNG state saved and restored around the evaluation so the trajectory is the
unpatched one:
  full  snapshot dependency_set / path_list + cpcm.compile_bb + cpcm.simulate(bb)
  inc   diff the event dict against the previous evaluated one (per-event signature: kind, deps, device or
        (bytes, branches)); if the tag set and every dependency set are unchanged, copy the previous Graph, patch
        the changed events (compute: resource; message: branches and per-hop durations from working_time) and run
        cpcm.incremental on the previous checkpointed schedule; otherwise (tags or dependencies changed) a full
        compile + simulate. The new state becomes the base of the next step only through a checkpointed full
        simulate ("rebase"), timed separately: incremental() does not carry checkpoints forward.
  check incremental result == full result (event arrays, T).
Usage: python cpcm_v7sa.py run --tasks ffn mha moe --steps 1000 ; python cpcm_v7sa.py analyse
"""
import argparse, copy, gzip, json, math, os, pickle, random, runpy, statistics, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpcm  # noqa: E402
import cpcm_v3  # noqa: E402
from cpcm_v3 import SRC_SUBDIRS, WORK  # noqa: E402

OUT = os.path.join(HERE, "cpcm_out", "v7sa")
K = 32


def signature(ev):
    if ev.event_type == "computation":
        return ("c", frozenset(ev.dependency_set), ev.comp_device, tuple(ev.comp_location))
    return ("m", frozenset(ev.dependency_set), int(ev.comm_bytes), tuple(tuple(b) for b in ev.path_list))


def child(name, steps, sink_path):
    model, task = cpcm_v3.PLAN[name][:2]
    task_path = os.path.join(WORK, "models", model, "py", task + ".py")
    for sub in SRC_SUBDIRS:
        p = os.path.join(WORK, sub)
        if p not in sys.path:
            sys.path.append(p)
    import Stream_Mapping

    rows = []
    prev = {}

    def evaluate(self, step):
        ev, hp = self.state[7], self.hardware_platform
        rng = random.getstate(), np.random.get_state()
        try:
            r = {"step": step, "n": len(ev)}
            t0 = time.perf_counter()
            deps = {t: set(e.dependency_set) for t, e in ev.items()}
            paths = {t: [list(b) for b in e.path_list] for t, e in ev.items() if e.event_type == "communication"}
            g = cpcm.compile_bb(ev, hp, path_lists=paths, deps=deps)
            t1 = time.perf_counter()
            sf = cpcm.simulate(g, "bb")
            t2 = time.perf_counter()
            r.update(full_compile=t1 - t0, full_sim=t2 - t1, T=sf.T)
            # incremental path
            t0 = time.perf_counter()
            sig = {t: signature(e) for t, e in ev.items()}
            t1 = time.perf_counter()
            r["diff"] = t1 - t0
            if prev:
                same_tags = sig.keys() == prev["sig"].keys()
                changed = [t for t in sig if prev["sig"].get(t) != sig[t]] if same_tags else None
                deps_same = same_tags and all(sig[t][1] == prev["sig"][t][1] for t in changed)
                r.update(same_tags=same_tags, n_changed=len(changed) if changed is not None else None,
                         deps_same=deps_same)
                if same_tags and deps_same and changed:
                    t0 = time.perf_counter()
                    g0 = prev["g"]
                    g2 = g0.copy()
                    ids = []
                    for t in changed:
                        e = g2.id_of[t]
                        ids.append(e)
                        x = ev[t]
                        if x.event_type == "computation":
                            g2.res_of[e] = g2.res_id[("dev", x.comp_device, x.comp_location)]
                        else:
                            brs = [[g2.res_id[("link", k)] for k in b] for b in x.path_list]
                            ld = {g2.res_id[("link", k)]: int(hp.links_dict[k].working_time(data_bytes=x.comm_bytes))
                                  for b in x.path_list for k in b}
                            g2.set_branches(e, brs, ld)
                    t1 = time.perf_counter()
                    si = cpcm.incremental(g2, prev["sc"], ids)
                    t2 = time.perf_counter()
                    r.update(inc_patch=t1 - t0, inc_sim=t2 - t1, inc_ok=(si.T == sf.T and si.F == sf.F and si.S == sf.S),
                             spliced=si.inc["splice_t"] is not None, path="inc")
                elif changed == []:
                    r.update(path="unchanged", inc_patch=0.0, inc_sim=0.0, inc_ok=True, spliced=None)
                else:
                    r.update(path="fallback_full")
            t0 = time.perf_counter()
            base = cpcm.simulate(g, "bb", checkpoint_every=K)
            r["rebase"] = time.perf_counter() - t0
            prev.update(sig=sig, g=g, sc=base)
            rows.append(r)
        finally:
            random.setstate(rng[0])
            np.random.set_state(rng[1])

    def anneal(self):
        random.seed(123)
        np.random.seed(123)
        step = 0
        Tfactor = -math.log(self.Tmax / self.Tmin)
        T = self.Tmax
        E = self.energy()
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        evaluate(self, 0)
        acc = 0
        while step < min(self.steps, steps):
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            t0 = time.perf_counter()
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            t_move = time.perf_counter() - t0
            evaluate(self, step)
            rows[-1]["sa_move_energy_s"] = t_move
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                t0 = time.perf_counter()
                self.revert()
                rows[-1]["sa_revert_s"] = time.perf_counter() - t0
                rows[-1]["accepted"] = False
                E = prevEnergy
                evaluate(self, step)     # the reverted state becomes the next base
                rows[-1]["after_revert"] = True
            else:
                acc += 1
                rows[-1]["accepted"] = True
                prevEnergy = E
        with gzip.open(sink_path, "wb") as f:
            pickle.dump({"rows": rows, "steps": step, "accepted": acc}, f)
        raise SystemExit(0)

    Stream_Mapping.StreamOptimization.anneal = anneal
    os.makedirs(os.path.join(os.path.dirname(task_path), "..", "results"), exist_ok=True)
    runpy.run_path(task_path, run_name="__main__")


def run(tasks, steps):
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    for name in tasks:      # one after another: timing
        sink = os.path.join(OUT, f"{name}.pkl.gz")
        t0 = time.time()
        with open(os.path.join(OUT, "logs", f"{name}.log"), "w", encoding="utf-8") as lf:
            rc = subprocess.call([sys.executable, os.path.abspath(__file__), "child", name, str(steps), sink],
                                 cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
        print(f"[v7sa] {name}: exit {rc}, {time.time() - t0:.0f} s", flush=True)


def analyse(tasks, sa_steps=1000):
    res = {}
    for name in tasks:
        with gzip.open(os.path.join(OUT, f"{name}.pkl.gz"), "rb") as f:
            d = pickle.load(f)
        rows = [r for r in d["rows"] if r["step"] > 0 and not r.get("after_revert")]
        reverts = [r for r in d["rows"] if r.get("after_revert")]
        med = lambda v: statistics.median(v) if v else None
        mean = lambda v: statistics.mean(v) if v else None
        full = [r["full_compile"] + r["full_sim"] for r in rows]
        inc_rows = [r for r in rows if r.get("path") == "inc"]
        paths = {}
        for r in rows:
            paths[r.get("path")] = paths.get(r.get("path"), 0) + 1
        # per step, incremental route: diff + (patch + incremental) or (full when it falls back), + rebase of the
        # state that becomes the next base (the move, or the reverted state when rejected)
        def inc_cost(r):
            c = r["diff"]
            c += (r["inc_patch"] + r["inc_sim"]) if r.get("path") in ("inc", "unchanged") else (r["full_compile"] + r["full_sim"])
            return c
        rebase = [r["rebase"] for r in rows] + [x["rebase"] + x["diff"] for x in reverts]
        sa_move = [r.get("sa_move_energy_s", 0) + r.get("sa_revert_s", 0) for r in rows]
        n = len(rows)
        tot_full = sum(full) / n * sa_steps
        tot_inc_eval = sum(inc_cost(r) for r in rows) / n * sa_steps
        tot_rebase = sum(rebase) / n * sa_steps
        tot_sa = sum(sa_move) / n * sa_steps
        res[name] = {"steps_measured": n, "accepted": d["accepted"], "events": rows[0]["n"], "paths": paths,
                     "inc_mismatch": sum(1 for r in inc_rows if not r["inc_ok"]),
                     "spliced_share_of_inc": mean([1.0 if r["spliced"] else 0.0 for r in inc_rows]),
                     "n_changed_median": med([r["n_changed"] for r in rows if r.get("n_changed") is not None]),
                     "full_step_median_s": med(full), "full_compile_median_s": med([r["full_compile"] for r in rows]),
                     "full_sim_median_s": med([r["full_sim"] for r in rows]),
                     "inc_step_median_s": med([inc_cost(r) for r in rows]),
                     "inc_only_median_s": med([r["inc_patch"] + r["inc_sim"] for r in inc_rows]),
                     "diff_median_s": med([r["diff"] for r in rows]), "rebase_median_s": med(rebase),
                     "sa_move_energy_median_s": med(sa_move),
                     f"projected_{sa_steps}_steps_s": {"sa_alone": tot_sa, "sa_plus_full": tot_sa + tot_full,
                                                       "sa_plus_inc_eval_only": tot_sa + tot_inc_eval,
                                                       "sa_plus_inc_eval_plus_rebase": tot_sa + tot_inc_eval + tot_rebase}}
        print(name, json.dumps(res[name], indent=1, default=str))
    with open(os.path.join(HERE, "cpcm_out", "v7sa_summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--tasks", nargs="*", default=["ffn", "mha", "moe"])
        ap.add_argument("--steps", type=int, default=1000)
        a = ap.parse_args()
        run(a.tasks, a.steps) if a.cmd == "run" else analyse(a.tasks)
