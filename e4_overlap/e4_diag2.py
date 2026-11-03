"""E4 diagnosis, round 2 (user request 2026-11-02, "两项都做吧"; measurement only, backend changes in memory in
e4_backend_diag.py). Short stream = stream 0, priority order 01 = short first.

Check 1 (inversion count, mode "orig" = the original Phase 4 with a deferral log): every deferral of a short-
stream event is classified by the occupant of the binding resource:
  own                    occupant belongs to the short stream
  a_same_tick            occupant (other stream) started on the link in the same tick (with prio: it won that tick
                         although of lower priority = length_count layering)
  b_hold_and_wait        occupant (other stream) was partially issued at that moment (later hops still waiting)
  c_after_ready          occupant (other stream) started after the short event became ready, before this tick;
                         sub-classified by where the short event was (earlier hops issued / deferred earlier on
                         another link / on the same link / first attempt)
  d_inflight_full        occupant (other stream), fully issued, started before the short event became ready
  compute_own/other      Phase 3 deferral on a busy device
with counts and summed deferral (new start - tick). Stall decomposition: on [r_0, C_0], instants with no short-
stream activity in service, labelled by the classes of the pending deferrals, cross-tabulated with the step-4
label (e4_diag.timeline: occupant at t started before / after the waiting activity's can).
Chain jumps: on the short stream's own critical chain (e4_diag.chain_for), each stretch on the other stream's
activities: entry activity, rows and time spent, and where it returns to the short stream.
Check 2 (mode "perhop"): per-hop arbitration without length layering; single-stream self-check first.

  python e4_diag2.py --jobs 10 [--only a1]
Writes e4_out/diag/diag2.json and diag2_runs.csv.
"""
import argparse
import collections
import copy
import gzip
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
DIAG = os.path.join(E.OUT, "diag")
LIMIT = 1200  # one evaluation > 20 min: stop and report
SEEDS = [1, 2, 3]


def child(spec_path):
    with open(spec_path, "rb") as f:
        spec = pickle.load(f)
    E._artifact_path()
    from e4_backend_diag import event_driver_diag
    e2 = E._e2()
    hp0 = E.build_platform(spec["cfg"], spec["faults"])
    recs = {s: E.load_stream(s) for s in set(spec["streams"])}
    blobs = [recs[s]["events_blob"] for s in spec["streams"]]
    out = []
    for seed in spec["seeds"]:
        ev = E.merged_events(blobs)
        hp = copy.deepcopy(hp0)
        pre = e2._snapshot_pre(ev)
        released, log = {}, {}
        random.seed(seed)
        t0 = time.time()
        res = event_driver_diag(ev, hp, release=spec["release"], prio=spec.get("prio"), released=released,
                                mode=spec["mode"], watch=0, log=log)
        wall = time.time() - t0
        cap = e2._extract(ev, hp, pre, res, f"seed{seed}")
        cap["release"] = released
        for d in log["deferrals"]:  # device keys can be artifact objects (not importable in the parent)
            if "res" in d:
                d["res"] = str(d["res"])
        out.append({"seed": seed, "wall": wall, "cap": cap, "log": log})
    with open(spec["sink"], "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def run(name, streams, mode, prio, release, cfg, faults):
    path = os.path.join(DIAG, "sims", f"{name}.pkl.gz")
    if os.path.exists(path):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = os.path.join(E.OUT, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    spec = os.path.join(tmp, f"spec_d2_{name}.pkl")
    sink = os.path.join(tmp, f"sink_d2_{name}.pkl")
    with open(spec, "wb") as f:
        pickle.dump({"streams": streams, "mode": mode, "prio": prio, "release": release, "cfg": cfg,
                     "faults": faults, "seeds": SEEDS, "sink": sink}, f)
    t0 = time.time()
    if not os.path.exists(sink):  # else: a finished child from an earlier attempt
        with open(os.path.join(E.OUT, "logs", f"d2_{name}.log"), "w", encoding="utf-8") as lf:
            try:  # 3 seeds per process: limit 3 x 20 min
                rc = subprocess.call([sys.executable, os.path.abspath(__file__), "child", spec], cwd=HERE, stdout=lf,
                                     stderr=subprocess.STDOUT, timeout=LIMIT * len(SEEDS))
            except subprocess.TimeoutExpired:
                return {"timeout": True, "wall": time.time() - t0}
        if rc != 0:
            raise E.E4Error(f"{name} failed (exit {rc})")
    E._artifact_path()  # the log can hold artifact objects (module 'net')
    with open(sink, "rb") as f:
        runs = pickle.load(f)
    os.remove(sink)
    os.remove(spec)
    r = {"runs": runs, "process_wall": time.time() - t0}
    with gzip.open(path, "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    return r


# ---------------------------------------------------------------- check 1
def classify(d):
    if d["kind"] == "compute":
        return "compute_own" if A.stream_of(d["occ"]) == 0 else "compute_other"
    if A.stream_of(d["occ"]) == 0:
        return "own"
    if d["occ_s"] == d["t"]:
        return "a_same_tick"
    if d["occ_partial"]:
        return "b_hold_and_wait"
    if d["ready"] is not None and d["occ_s"] >= d["ready"]:
        if d["hops_done"] > 0:
            return "c_after_ready|short_partial_earlier_hops_issued"
        if d["prev"] is not None:
            return "c_after_ready|short_deferred_earlier_" + ("same_link" if tuple(d["prev"][1]) == tuple(d["link"]) else "other_link")
        return "c_after_ready|first_attempt"
    return "d_inflight_full"


def stall_cross(cap, log):
    """[r_0, C_0]: no short activity in service -> (new class set, step-4 label) durations."""
    import e4_diag as D
    sim = A.SimR(cap)
    rel = cap["release"]
    C, _ = A.stream_times(cap)
    r0, C0 = rel[0], C[0]
    ev = []
    for k, X in sim.act.items():
        if A.stream_of(X["tag"]) != 0:
            continue
        ev += [(X["s"], ("svc",), 1), (X["f"], ("svc",), -1)]
        if X["s"] <= X["can"]:
            continue
        lo, hi = X["can"], X["s"]
        t = lo
        for s, f, tag in sim.res[X["res"]]:
            if f <= lo or s >= hi or tag == X["tag"]:
                continue
            a, b = max(s, lo), min(f, hi)
            if a > t:
                ev += [(t, ("old", "free"), 1), (a, ("old", "free"), -1)]
            lab = "own" if A.stream_of(tag) == 0 else ("other_inflight" if s < lo else "other_arbitration")
            ev += [(a, ("old", lab), 1), (b, ("old", lab), -1)]
            t = max(t, b)
        if t < hi:
            ev += [(t, ("old", "free"), 1), (hi, ("old", "free"), -1)]
    for d in log["deferrals"]:
        c = classify(d).split("|")[0]
        if d["new_start"] > d["t"]:
            ev += [(d["t"], ("new", c), 1), (d["new_start"], ("new", c), -1)]
    ev.sort(key=lambda x: x[0])
    cnt = collections.Counter()
    out = collections.Counter()
    last = r0
    for t, key, dd in ev + [(C0, None, 0)]:
        a, b = max(last, r0), min(t, C0)
        if b > a and cnt[("svc",)] <= 0:
            old = {k[1] for k, v in cnt.items() if v > 0 and k[0] == "old"}
            oth = {x for x in old if x.startswith("other")}
            if not old:
                ol = "nothing_pending"
            elif oth and "own" not in old:
                ol = "other_only|" + "+".join(sorted(x[6:] for x in oth))
            elif "own" in old and not oth:
                ol = "own_only"
            elif old <= {"free"}:
                ol = "free_only"
            else:
                ol = "own+other"
            new = "+".join(sorted({k[1] for k, v in cnt.items() if v > 0 and k[0] == "new"})) or "none"
            out[f"{ol} || {new}"] += b - a
        if key is not None:
            cnt[key] += dd
        last = t
    return {k: v / (C0 - r0) for k, v in out.items()}


def chain_jumps(cap):
    import e4_diag as D
    sim = A.SimR(cap)
    rel = cap["release"]
    C, _ = A.stream_times(cap)
    r0, C0 = rel[0], C[0]
    _, rows, _ = D.chain_for(sim, 0)  # latest first
    segs = []
    k = 0
    while k < len(rows):
        if rows[k]["owner"] == 0:
            k += 1
            continue
        j = k
        while j < len(rows) and rows[j]["owner"] != 0:
            j += 1
        entry = rows[k - 1] if k > 0 else None  # the short-stream activity that waited behind the other stream
        back = rows[j] if j < len(rows) else None  # where the chain is back on the short stream (earlier in time)
        span = sum(max(0, min(rw["t_out"], C0) - max(rw["s"], r0)) for rw in rows[k:j])
        segs.append({"entry_kind": entry["kind"] if entry else None, "entry_t": entry["s"] if entry else None,
                     "rows": j - k, "time_in_window": span,
                     "returns": back is not None, "return_kind": back["kind"] if back else None,
                     "return_t": back["t_out"] if back else None,
                     "return_pos": (back["t_out"] - r0) / (C0 - r0) if back else None,
                     "chain_start_in_other": back is None,
                     "other_kinds": dict(collections.Counter(rw["kind"] for rw in rows[k:j]))})
        k = j
    return {"n_rows": len(rows), "n_jumps": len(segs), "rows_in_other": sum(s["rows"] for s in segs),
            "time_in_other_share": sum(s["time_in_window"] for s in segs) / (C0 - r0), "segments": segs}


def analyse_run(r, T, check1):
    cap = r["cap"]
    m = A.stream_measures(cap, T, E.KS)
    out = {"seed": r["seed"], "wall": r["wall"], "latency": m["latency"], "stretch": m["stretch"], "S_max": m["S_max"],
           "makespan": m["makespan"], "result": cap["result"]}
    if check1:
        cls = collections.Counter()
        n = collections.Counter()
        for d in r["log"]["deferrals"]:
            c = classify(d)
            cls[c] += d["new_start"] - d["t"]
            n[c] += 1
        L = m["latency"][0]
        out.update(deferral_wait=dict(cls), deferral_n=dict(n), deferral_wait_over_latency={k: v / L for k, v in cls.items()},
                   stall_cross=stall_cross(cap, r["log"]), chain=chain_jumps(cap))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run")
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    if args.cmd == "child":
        child(args.spec)
        return
    os.makedirs(DIAG, exist_ok=True)
    with open(os.path.join(E.OUT, "grid1_state.json"), encoding="utf-8") as f:
        fl = json.load(f)["faults"]["a1_2s_512-8192"]["F4"]
    fl["failed_links"] = [tuple(map(tuple, l)) for l in fl["failed_links"]]
    with open(os.path.join(E.OUT, "T6.json"), encoding="utf-8") as f:
        T6 = json.load(f)["T"]
    cases = {"a1_512-8192_F4_d0": ([G.fault_key(k, fl) for k in ["mha512", "mha8192"]],
                                   {"failed_nodes": [], "failed_links": fl["failed_links"]}),
             "a2_mha-ffn_nofault_d0": (["mha2048", "ffn"], None)}
    if args.only:
        cases = {k: v for k, v in cases.items() if k.startswith(args.only)}
    jobs = []
    for cname, (keys, fa) in cases.items():
        for mode in ["orig", "perhop"]:
            for vname, pr in [("default", None), ("prio_01", {0: 0, 1: 1})]:
                jobs.append((f"{cname}_{mode}_{vname}", keys, mode, pr, {0: 0, 1: 0}, fa, cname, vname))
        for k in keys:  # single-stream self-check of perhop (no prio)
            jobs.append((f"single_{k}_perhop", [k], "perhop", None, {0: 0}, fa, cname, "single"))
    t0 = time.time()
    with ThreadPoolExecutor(args.jobs) as ex:
        res = dict(zip([j[0] for j in jobs], ex.map(lambda j: run(j[0], j[1], j[2], j[3], j[4], E.CFG, j[5]), jobs)))
    G.log(f"{len(jobs)} diag2 processes ({time.time() - t0:.0f} s)")
    summary, rows = {}, []
    for name, keys, mode, pr, rel, fa, cname, vname in jobs:
        r = res[name]
        if r.get("timeout"):
            summary[name] = {"timeout": True, "wall": r["wall"]}
            continue
        T = {i: T6[k]["T6"] for i, k in enumerate(keys)}
        an = [analyse_run(x, T, check1=(mode == "orig" and vname != "single")) for x in r["runs"]]
        s = {"case": cname, "mode": mode, "variant": vname, "streams": keys, "T6": T,
             "latency_seeds": {i: [a["latency"][i] for a in an] for i in T},
             "stretch_mean": {i: E._mean([a["stretch"][i] for a in an]) for i in T},
             "S_max_mean": E._mean([a["S_max"] for a in an]), "S_max_seeds": [a["S_max"] for a in an],
             "makespan_mean": E._mean([a["makespan"] for a in an]), "eval_wall": [a["wall"] for a in an],
             "process_wall": r["process_wall"], "per_seed": an}
        if vname == "single":
            k = keys[0]
            s["orig_T6_seeds"] = T6[k]["seeds"]
            s["rel_diff_vs_orig_T6"] = s["latency_seeds"][0] and E._mean(s["latency_seeds"][0]) / T6[k]["T6"] - 1
        if mode == "orig" and vname != "single":
            agg = collections.Counter()
            aggn = collections.Counter()
            st = collections.Counter()
            for a in an:
                for kk, v in a["deferral_wait_over_latency"].items():
                    agg[kk] += v / len(an)
                for kk, v in a["deferral_n"].items():
                    aggn[kk] += v / len(an)
                for kk, v in a["stall_cross"].items():
                    st[kk] += v / len(an)
            s.update(deferral_wait_over_latency_mean=dict(agg), deferral_n_mean=dict(aggn), stall_cross_mean=dict(st),
                     chain_summary=[{kk: a["chain"][kk] for kk in ["n_rows", "n_jumps", "rows_in_other", "time_in_other_share"]}
                                    for a in an])
        summary[name] = s
        for a in an:
            rows.append({"run": name, "case": cname, "mode": mode, "variant": vname, "seed": a["seed"], "wall": a["wall"],
                         "S_max": a["S_max"], "makespan": a["makespan"],
                         **{f"latency_{i}": v for i, v in a["latency"].items()},
                         **{f"stretch_{i}": v for i, v in a["stretch"].items()}})
    E._write_json(os.path.join("diag", "diag2.json"), {"summary": summary, "limit_s": LIMIT, "seeds": SEEDS,
                                                       "artifact_commit": E._artifact_commit(), "command": sys.argv})
    E._write_csv(os.path.join("diag", "diag2_runs.csv"), rows)
    G.log("diag2 done")


if __name__ == "__main__":
    try:
        main()
    except E.E4Error as exc:
        print(f"[e4] STOP: {exc}", flush=True)
        sys.exit(2)
