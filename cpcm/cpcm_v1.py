"""V1 semantic replay (cpcm/CPCM_HANDOFF.md section 3): impose the backend's recorded service order on CPCM (order
mode) and compare every activity's start/end and every event's start_time/end_time with the backend record.

capture  runs `python <task>.py` unchanged in the work copy cpcm/work/artifact (via _inject/sitecustomize.py); at
         the template's 2nd (final) event_driver call it snapshots dependency_set / path_list, runs the original
         backend (template RNG state), then the same mapping again with random.seed(s) for each --seeds s on deep
         copies taken before that call, and compiles the graph with cpcm.compile_bb from the snapshots.
         Output cpcm_out/v1_<name>_cap.pkl.gz (not in git).
replay   order mode with: per resource, the backend's records sorted by start; per communication, the terminal hop =
         among hops ending at the backend end_time, the latest start (E2's rule; ties by resource name).

Usage (one task per process):
  python cpcm_v1.py capture --name ffn --seeds 1 2
  python cpcm_v1.py replay --name ffn
"""
import argparse
import copy
import gzip
import json
import os
import pickle
import random
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpcm  # noqa: E402

OUT = os.path.join(HERE, "cpcm_out")
WORK = os.path.join(HERE, "work", "artifact")
COMMIT_FILE = os.path.join(HERE, "..", "third_party", "isca2026-busybarn-artifact.COMMIT")
SUFFIX = "_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0"
TASKS = {
    "ffn": ("intra_mapping_gpt_decode", "ffn_sq2048_greedy0_lossratio6" + SUFFIX),   # E3 FFN winner
    "mha": ("intra_mapping_gpt_decode", "mha_sq2048_greedy0_lossratio4" + SUFFIX),   # E3 MHA winner
    "moe": ("intra_mapping_qwen3moe_decode", "ffn_sq2048_greedy0_lossratio5" + SUFFIX),  # E2 ext winner
}


# ------------------------------------------------------------------ child side
def _records(events_dict, hp):
    res = {}
    for dev, mods in hp.modules_dict.items():
        for loc, m in mods.items():
            if m.work_record:
                res[("dev", dev, loc)] = [(int(s), int(f), tag) for s, f, tag in m.work_record]
    for k, l in hp.links_dict.items():
        if l.work_record:
            res[("link", k)] = [(int(s), int(f), tag) for s, f, tag in l.work_record]
    ev = {tag: (int(e.start_time), int(e.end_time)) for tag, e in events_dict.items()}
    return {"res": res, "events": ev}


def child_capture(orig, events_dict, hp, call_no):
    seeds = [int(s) for s in os.environ.get("CPCM_SEEDS", "").split()]
    deps = {t: set(e.dependency_set) for t, e in events_dict.items()}
    paths = {t: [list(br) for br in e.path_list] for t, e in events_dict.items() if e.event_type == "communication"}
    reruns = copy.deepcopy((events_dict, hp)) if seeds else None
    t0 = time.perf_counter()
    out = orig(events_dict=events_dict, hardware_platform=hp)
    wall = time.perf_counter() - t0
    runs = [{"seed": "template", "result": tuple(int(x) for x in out), "wall": wall, **_records(events_dict, hp)}]
    if reruns is not None:
        state = random.getstate(), np.random.get_state()
        try:
            for s in seeds:
                ev_k, hp_k = copy.deepcopy(reruns)
                random.seed(s)
                t0 = time.perf_counter()
                out_k = orig(events_dict=ev_k, hardware_platform=hp_k)
                runs.append({"seed": s, "result": tuple(int(x) for x in out_k), "wall": time.perf_counter() - t0,
                             **_records(ev_k, hp_k)})
        finally:
            random.setstate(state[0])
            np.random.set_state(state[1])
    t0 = time.perf_counter()
    g = cpcm.compile_bb(events_dict, hp, path_lists=paths, deps=deps)
    compile_s = time.perf_counter() - t0
    sink = {"calls": call_no, "runs": runs, "graph": g, "compile_s": compile_s,
            "n_comp": sum(1 for k in g.kind if k == cpcm.COMP), "n_comm": sum(1 for k in g.kind if k == cpcm.COMM)}
    with gzip.open(os.environ["CPCM_SINK"], "wb") as f:
        pickle.dump(sink, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


# ------------------------------------------------------------------ parent side
def capture(name, seeds):
    model, task = TASKS[name]
    task_path = os.path.join(WORK, "models", model, "py", task + ".py")
    if not os.path.exists(task_path):
        raise SystemExit(f"missing {task_path} (work copy: copy third_party/isca2026-busybarn-artifact to {WORK})")
    os.makedirs(os.path.join(os.path.dirname(task_path), "..", "results"), exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    sink = os.path.join(OUT, f"v1_{name}_cap.pkl.gz")
    env = dict(os.environ, CPCM_SINK=sink, CPCM_HOME=HERE, CPCM_SEEDS=" ".join(map(str, seeds)),
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "_inject")] +
                                          ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    if os.path.exists(sink):
        os.remove(sink)
    t0 = time.time()
    rc = subprocess.call([sys.executable, task_path], env=env, cwd=HERE)
    print(f"[v1] capture {name}: exit {rc}, {time.time() - t0:.0f} s, sink {'ok' if os.path.exists(sink) else 'MISSING'}")
    return rc


def replay_inputs(g, run):
    """Backend record -> (imposed order per resource, terminal hop per communication, {(event, res): (s, f)})."""
    tid = g.id_of
    rq = g.res_id
    seq, term = {}, {}
    rec = {}
    for rname, lst in run["res"].items():
        q = rq[rname]
        lst = sorted(lst, key=lambda x: (x[0], x[1]))
        seq[q] = [tid[tag] for _, _, tag in lst]
        for s, f, tag in lst:
            key = (tid[tag], q)
            if key in rec:
                raise cpcm.CPCMError(f"event {tag} used {rname} twice")
            rec[key] = (s, f)
    by_ev = {}
    for (x, q), sf in rec.items():
        by_ev.setdefault(x, []).append((q, sf))
    for tag, (s_ev, f_ev) in run["events"].items():
        e = tid[tag]
        if g.kind[e] == cpcm.COMM:
            ends = [q for q, (s, f) in by_ev.get(e, []) if f == f_ev]
            if not ends:
                raise cpcm.CPCMError(f"communication {tag}: no hop ends at end_time {f_ev}")
            term[e] = max(ends, key=lambda q: (rec[(e, q)][0], str(g.res_name[q])))
    return seq, term, rec


def replay_run(g, run):
    tid = g.id_of
    seq, term, rec = replay_inputs(g, run)
    t0 = time.perf_counter()
    sc = cpcm.simulate_order(g, seq, term_res=term)
    t_sim = time.perf_counter() - t0
    mism = []
    mine = {(sc.a_ev[a], sc.a_res[a]): (sc.a_s[a], sc.a_f[a]) for a in range(len(sc.a_s))}
    for key in set(rec) | set(mine):
        if rec.get(key) != mine.get(key):
            e, q = key
            mism.append({"kind": "activity", "t": min(x[0] for x in (rec.get(key), mine.get(key)) if x),
                         "event": str(g.tag[e]), "name": g.name[e], "resource": str(g.res_name[q]),
                         "backend": rec.get(key), "cpcm": mine.get(key)})
    for tag, (s_ev, f_ev) in run["events"].items():
        e = tid[tag]
        got = (sc.S_last[e], sc.F[e])
        if got != (s_ev, f_ev):
            mism.append({"kind": "event", "t": min(s_ev, got[0]), "event": str(tag), "name": g.name[e],
                         "backend_start_end": (s_ev, f_ev), "cpcm_S_last_F": got, "r": sc.r[e]})
    mism.sort(key=lambda m: (m["t"], m["kind"]))
    return {"seed": run["seed"], "backend_T": run["result"][0], "cpcm_T": sc.T, "n_activities": len(rec),
            "n_events": len(run["events"]), "mismatches": len(mism), "first": mism[:5],
            "replay_s": round(t_sim, 4), "backend_s": round(run["wall"], 3)}


def replay(name):
    with gzip.open(os.path.join(OUT, f"v1_{name}_cap.pkl.gz"), "rb") as f:
        cap = pickle.load(f)
    g = cap["graph"]
    rows = [replay_run(g, run) for run in cap["runs"]]
    summ = {"name": name, "task": TASKS[name], "n_events": g.n, "n_comp": cap["n_comp"], "n_comm": cap["n_comm"],
            "n_hops_listed": sum(cpcm_hops(g, e) for e in range(g.n)), "compile_s": round(cap["compile_s"], 3),
            "artifact_commit": open(COMMIT_FILE, encoding="utf-8-sig").readline().strip(), "runs": rows}
    with open(os.path.join(OUT, f"v1_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, default=str)
    for r in rows:
        print(f"[v1] {name} seed={r['seed']}: backend T={r['backend_T']} cpcm T={r['cpcm_T']} "
              f"activities={r['n_activities']} events={r['n_events']} mismatches={r['mismatches']} "
              f"replay {r['replay_s']} s (backend {r['backend_s']} s)")
        for m in r["first"][:1]:
            print("      first:", m)
    return 0 if all(r["mismatches"] == 0 for r in rows) else 1


def cpcm_hops(g, e):
    return len(cpcm.hop_tree(g, e)) if g.kind[e] == cpcm.COMM else 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("capture")
    p.add_argument("--name", choices=sorted(TASKS), required=True)
    p.add_argument("--seeds", type=int, nargs="*", default=[])
    p = sp.add_parser("replay")
    p.add_argument("--name", choices=sorted(TASKS), required=True)
    a = ap.parse_args()
    return capture(a.name, a.seeds) if a.cmd == "capture" else replay(a.name)


if __name__ == "__main__":
    sys.exit(main())
