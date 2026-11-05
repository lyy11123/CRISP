"""EIR next stage, phase A driver (EIR_NEXT_HANDOFF.md sections 2-4). Uses eir.py's jobs; writes only out/next/ (and appends
F8a to faults.json). A3h start, 2x2 FFN and MHA.

  python eir_next.py select                     F8a by choose_fault(F8) on each A3h mapping, appended to faults.json
  python eir_next.py prep --workers 6           pre rerun (with the wait of every tag), l1 at F8a
  python eir_next.py prior --workers 4          section 4.2: FFN, MHA at F4a, s = 100, seed 123, L2Impact and L2Lsync
  python eir_next.py repair --share 1|2 --workers 14|18
  python eir_next.py repair --share all --workers N    whatever is left
  python eir_next.py full --workers N           FULL-A3h at F4a, F8a
  python eir_next.py seeds --workers 14         15 more SA seeds: FFN F4a, F5, F8a x 5 methods, MHA F4a x L2Impact, L2Lsync
  python eir_next.py check_log --workers 6      the step log changes nothing: reruns stored +sync runs, compares every step

Each run appends one row to out/next/runs.jsonl (workers, jobs, wall, memory). The pre and l1 of F4a and F5 are out/main's.
"""
import argparse
import collections
import ctypes
import json
import os
import random
import threading
import time

import eir

OUT = os.path.join(eir.HERE, "out", "next")
MAIN = os.path.join(eir.HERE, "out", "main")
OPS = ("ffn", "mha")
FAULTS = ("F4a", "F5", "F8a")
STEPS = 100
NEW_METHODS = ("L2Lsync", "L2Impact", "L2Gsync")
OLD_METHODS = ("L2L", "L2G")
EXTRA_SEEDS = list(range(5, 20))      # user 2026-11-05: 20 seeds in all
ALL_SEEDS = eir.SA_SEEDS + EXTRA_SEEDS
MEM_FLOOR_GB = 3.0      # handoff section 2: free memory must not go below this
START_GATE_GB = 4.5     # no new process below this (a process peaks around 1 GB, see PROGRESS)


def free_gb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def l1_out(fid):
    return MAIN if fid in ("F4a", "F5") else OUT


# ---------------------------------------------------------------- phase C (handoff sections 5, 8)
C_FAULTS = ("CRandom", "CCluster", "CEdge", "CScratch")
C_METHODS = ("L2Lsync", "L2Impact", "L2Gsync")
C_SEEDS = {"ffn": ALL_SEEDS, "mha": [123] + list(range(1, 10))}
C_RANDOM_SEED = 20261105
C_RULES = {"CRandom": f"each die: random.Random({C_RANDOM_SEED}).choice(sorted loaded cores), dies in order; loaded = tensor + vector load > 0 at the fault-free A3h mapping",
           "CCluster": "connected path of 3 loaded cores in one die with the largest load sum (ties: smallest ids); PE only; plus the busiest on-chip link (both directions summed) touching one of them, both directions",
           "CEdge": "3 consecutive loaded cores on one side of the group's outer perimeter (global grid) with the largest load sum (ties: smallest ids); PE only; plus the DDR port with the largest ddr2co busy time",
           "CScratch": "the D2D pair with the largest busy time (both directions summed), both directions; plus 3 consecutive loaded cores on the global row or column crossing that link, containing one of its ends, with the largest load sum (ties: smallest ids); PE only"}


def _grid(n):
    return (n[0] // 2) * 4 + n[1] // 4, (n[0] % 2) * 4 + n[1] % 4


def _node_at(y, x):
    return (y // 4) * 2 + x // 4, (y % 4) * 4 + x % 4


def choose_c(op):
    """The four instances of one task from its fault-free A3h mapping (F4a's L1 state has the same behaviour places, R0 and L1 move none,
    and the pre job's link busy times). Deterministic: rules in C_RULES, written with the instance."""
    import pickle
    s = pickle.loads(eir.read_sink(eir.job_l1(MAIN, "a3h", op, "F4a")["sink"])["L1_snap"]["blob"])
    load = collections.Counter()
    for k, v in list(s[3].items()) + list(s[4].items()):
        load[tuple(k[:-1])] += float(v)
    busy = eir.read_sink(job_pre_w(op)["sink"])["busy"]
    cores = [(d, c) for d in range(4) for c in range(16)]
    loaded = [n for n in cores if load[n] > 0]
    out = {}
    rng = random.Random(C_RANDOM_SEED)
    out["CRandom"] = {"failed_pes": [rng.choice([n for n in loaded if n[0] == d]) for d in range(4)], "failed_links": [], "failed_nodes": []}
    # ---- cluster
    def nbrs(n):
        y, x = _grid(n)
        return [m for m in (_node_at(y + dy, x + dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= y + dy < 8 and 0 <= x + dx < 8) if m[0] == n[0]]
    tri = set()
    for b in loaded:
        ns = [m for m in nbrs(b) if load[m] > 0]
        for i, a in enumerate(ns):
            for c in ns[i + 1:]:
                tri.add(tuple(sorted((a, b, c))))
    best = max(sorted(tri), key=lambda t: sum(load[n] for n in t))
    und = collections.Counter()
    for k, v in busy.items():
        und[tuple(sorted(map(tuple, k[:2])))] += v
    onchip = [k for k in sorted(und) if k[0][0] == k[1][0] and k[0][1] < 16 and k[1][1] < 16 and (k[0] in best or k[1] in best)]
    l = max(onchip, key=lambda k: und[k])
    out["CCluster"] = {"failed_pes": list(best), "failed_links": [[l[0], l[1]], [l[1], l[0]]], "failed_nodes": [], "link_busy_both": und[l]}
    # ---- edge
    lines = []
    for i in range(6):
        lines += [[_node_at(y, i + j) for j in range(3)] for y in (0, 7)] + [[_node_at(i + j, x) for j in range(3)] for x in (0, 7)]
    lines = sorted(tuple(sorted(t)) for t in lines if all(load[n] > 0 for n in t))
    best = max(lines, key=lambda t: sum(load[n] for n in t))
    ports = {tuple(k[0]): v for k, v in busy.items() if k[0][1] >= 16}
    port = max(sorted(ports), key=lambda p: ports[p])
    out["CEdge"] = {"failed_pes": list(best), "failed_links": [], "failed_nodes": [port], "port_busy": ports[port]}
    # ---- scratch
    d2d = [k for k in sorted(und) if k[0][0] != k[1][0] and k[0][1] < 16 and k[1][1] < 16]
    l = max(d2d, key=lambda k: und[k])
    (y0, x0), (y1, x1) = _grid(l[0]), _grid(l[1])
    if x0 == x1:
        lines = [tuple(_node_at(y + j, x0) for j in range(3)) for y in range(6)]
    else:
        lines = [tuple(_node_at(y0, x + j) for j in range(3)) for x in range(6)]
    lines = sorted(tuple(sorted(t)) for t in lines if all(load[n] > 0 for n in t) and (l[0] in t or l[1] in t))
    best = max(lines, key=lambda t: sum(load[n] for n in t))
    out["CScratch"] = {"failed_pes": list(best), "failed_links": [[l[0], l[1]], [l[1], l[0]]], "failed_nodes": [], "link_busy_both": und[l]}
    for fid, x in out.items():
        x.update(kind="k=4 composite: " + fid[1:], e3_fault=None, chosen_on="a3h", rule=C_RULES[fid], pe_load=[load[tuple(n)] for n in x["failed_pes"]],
                 random_seed=C_RANDOM_SEED if fid == "CRandom" else None, same_as=None)
    return out


def select_c():
    t = eir.load_faults()
    for op in OPS:
        for fid, x in choose_c(op).items():
            row = json.loads(json.dumps(x))
            if fid in t[op] and t[op][fid] != row:
                raise SystemExit(f"{op} {fid}: already in faults.json with other contents")
            t[op][fid] = row
            print(op, fid, row["failed_pes"], row["failed_links"], row["failed_nodes"], row["pe_load"])
    tmp = eir.FAULTS_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(t, f, indent=1, ensure_ascii=False)
    os.replace(tmp, eir.FAULTS_JSON)


def c_rep_jobs():
    return [job_rep(o, f, m, sd) for o in OPS for f in C_FAULTS for m in C_METHODS for sd in C_SEEDS[o]]


def c_full_jobs():
    return [eir.job_full(OUT, o, f, "A3h", g, x) for o in OPS for f in C_FAULTS for g, x in eir.cs.variants("A3h")]


B_FAULT = "F4a"       # the shared physical link (0,13)-(2,1), section 6
B_METHODS = ("L2Lsync", "L2Impact", "L2Gsync")
B_SEEDS = [123] + list(range(1, 10))
B_CK = (30, 100, 300)
B_OUT = os.path.join(OUT, "b")


def b_job(op, method, seed, out=B_OUT, steps=300, ck=B_CK, checks=False):
    j = job_rep(op, B_FAULT, method, seed, out=out, steps=steps, checks=checks)
    return dict(j, checkpoints=list(ck), ck_backend=True)


def b_jobs():
    return [b_job(o, m, sd) for o in OPS for m in B_METHODS for sd in B_SEEDS]


def b_smoke(workers):
    """s = 20 with checkpoints 5, 10, 20 and checks. T at a checkpoint must equal the best_T search saw; the last checkpoint
    must equal the run's own record in T."""
    out = os.path.join(eir.HERE, "out", "_smoke_b")
    jobs = [b_job(o, m, 123, out=out, steps=20, ck=(5, 10, 20), checks=True) for o in OPS for m in B_METHODS]
    bad = eir.run_phases([jobs], out, workers)
    ok = not bad
    for j in jobs:
        d = eir.read_sink(j["sink"])
        s = d["search"]
        rows = d["ck"]
        h = rows[-1]["handoff"]
        r = {"T_eq_best": [x["T"] == x["best_T"] for x in rows], "last_eq_record": rows[-1]["T"] == d["record"]["T"], "viol": [x["fault_violations"] for x in rows], "stale": [x["stale"] for x in rows],
             "revert_bad": sum(s["check_revert"][k] for k in ("E_mismatch", "T_mismatch", "state_mismatch")), "t_search": [round(x["t_search"], 2) for x in rows],
             "reads": len(h["ifmap_reads"]), "read_bytes": sum(b for _, b in h["ifmap_reads"]), "out_locs": h["output_locs"], "n_links": len(d["link_time"]), "link_sample": list(d["link_time"].items())[:2],
             "mig_keys": list(rows[-1]["migration_vs_R0"])[:12]}
        ok = ok and all(r["T_eq_best"]) and r["last_eq_record"] and not any(r["viol"]) and not any(r["stale"]) and r["revert_bad"] == 0
        print(j["name"], r, flush=True)
    print("B_SMOKE PASSED" if ok else "B_SMOKE FAILED", flush=True)


def c_smoke(workers):
    """Do the instances build? l1 (kept: the real one) and, isolated, s = 5 of every method with checks."""
    if run_logged("c_l1", [job_l1(o, f) for o in OPS for f in C_FAULTS], workers):
        raise SystemExit("l1 failed")
    out = os.path.join(eir.HERE, "out", "_smoke_c")
    jobs = []
    for o in OPS:
        for f in C_FAULTS:
            for m in C_METHODS:
                j = job_rep(o, f, m, 123, steps=5, checks=True)
                jobs.append(dict(j, name="c_" + j["name"], sink=os.path.join(out, os.path.basename(j["sink"]))))
    bad = eir.run_phases([jobs], out, workers)
    ok = not bad
    for j in jobs:
        d = eir.read_sink(j["sink"])
        s = d["search"]
        la = s["local_audit"] or {}
        r = {"T_L1": d["T_L1_check"], "viol": len(d["record"]["fault_violations"]), "revert_bad": sum(s["check_revert"][k] for k in ("E_mismatch", "T_mismatch", "state_mismatch")),
             "stale": s["stale_final"] + s["read_watch"].get("stale_after_proposal", 0) + s["read_watch"].get("rebound_stale", 0), "outside": la.get("beha_not_movable", 0) + la.get("beha_dest_outside", 0),
             "cov": s["local"] and (s["local"]["n_cores"], round(s["local"]["beha_share"], 2))}
        ok = ok and r["viol"] == 0 and r["revert_bad"] == 0 and r["stale"] == 0 and r["outside"] == 0
        print(j["name"], r, flush=True)
    print("C_SMOKE PASSED" if ok else "C_SMOKE FAILED", flush=True)
    if not ok:
        raise SystemExit(1)


def job_pre_w(op):
    return dict(eir.job_pre(OUT, "a3h", op), store_w=True)


def job_l1(op, fid, out=OUT):
    """F8a's l1, measured against out/main's pre (the reference of every migration number)."""
    return dict(eir.job_l1(out, "a3h", op, fid), pre_sink=eir.job_pre(MAIN, "a3h", op)["sink"])


def job_rep(op, fid, method, seed, out=OUT, steps=STEPS, checks=False):
    j = eir.job_rep(out, "a3h", op, fid, method, steps, seed, checks=checks)
    j["pre_sink"] = eir.job_pre(MAIN, "a3h", op)["sink"]
    j["l1_sink"] = eir.job_l1(l1_out(fid), "a3h", op, fid)["sink"]
    if method == "L2Impact":
        j["pre_w_sink"] = job_pre_w(op)["sink"]
    return j


def repair_jobs():
    """Handoff 4.4: the old methods only at F8a (F4a, F5 are in out/main), the new ones on 6 combinations."""
    jobs = [job_rep(o, "F8a", m, sd) for o in OPS for m in OLD_METHODS for sd in eir.SA_SEEDS]
    jobs += [job_rep(o, f, m, sd) for o in OPS for f in FAULTS for m in NEW_METHODS for sd in eir.SA_SEEDS]
    return jobs


def seed_jobs():
    jobs = [job_rep("ffn", f, m, sd) for f in FAULTS for m in OLD_METHODS + NEW_METHODS for sd in EXTRA_SEEDS]
    return jobs + [job_rep("mha", "F4a", m, sd) for m in ("L2Impact", "L2Lsync") for sd in EXTRA_SEEDS]


def strip_log(rows):
    return [{k: v for k, v in r.items() if k != "diag"} for r in rows]


def check_log(workers):
    """The read-only step log: stored runs (before the log) rerun with it. Rows without the log, T, places and routes must be equal."""
    base = [job_rep(o, "F4a", m, 123) for o, m in (("ffn", "L2Impact"), ("ffn", "L2Lsync"), ("ffn", "L2Gsync"), ("mha", "L2Impact"), ("mha", "L2Lsync"))] + [job_rep("ffn", "F8a", "L2Impact", 1)]
    out = os.path.join(eir.HERE, "out", "_smoke_next", "check_log")
    jobs = [dict(j, name="log_" + j["name"], sink=os.path.join(out, os.path.basename(j["sink"]))) for j in base]
    for j in jobs:
        if os.path.exists(j["sink"]):
            os.remove(j["sink"])
    bad = eir.run_phases([jobs], out, workers)
    ok = not bad
    for b, j in zip(base, jobs):
        o, n = eir.read_sink(b["sink"]), eir.read_sink(j["sink"])
        r = {"rows": strip_log(n["search"]["rows"]) == strip_log(o["search"]["rows"]), "T": n["record"]["T"] == o["record"]["T"], "beha": n["record"]["beha_loc"] == o["record"]["beha_loc"],
             "routes": n["record"]["routes"] == o["record"]["routes"], "has_log": all("diag" in x for x in n["search"]["rows"])}
        ok = ok and all(r.values())
        print(b["name"], r, flush=True)
    print("CHECK_LOG PASSED" if ok else "CHECK_LOG FAILED", flush=True)
    if not ok:
        raise SystemExit(1)


def full_jobs():
    return [eir.job_full(OUT, o, f, "A3h", g, x) for o in OPS for f in ("F4a", "F8a") for g, x in eir.cs.variants("A3h")]


def shares(jobs):
    """Two fixed shares of the repairs outside the prior (section 4.2), equal in size and in the FFN : MHA and method mix; a fixed
    shuffle mixes short and long jobs in both. Fixed = does not depend on what has finished, so each share can be resumed."""
    prior = {eir.rep_name("a3h", o, "F4a", m, STEPS, 123) for o in OPS for m in ("L2Impact", "L2Lsync")}
    by = collections.defaultdict(list)
    for j in jobs:
        if j["name"] in prior:
            continue
        by[(j["op"], j["method"], j["fault"])].append(j)
    a, b = [], []
    flip = 0
    for k in sorted(by):
        for j in by[k]:
            (a if flip % 2 == 0 else b).append(j)
            flip += 1
    rng = random.Random(20261105)
    rng.shuffle(a)
    rng.shuffle(b)
    return a, b


def run_logged(label, jobs, workers):
    """Runs the jobs with free-memory sampling; a new process starts only above START_GATE_GB."""
    os.makedirs(OUT, exist_ok=True)
    mem_log = os.path.join(OUT, f"mem_{label}.jsonl")
    os.environ["EIR_MEM_LOG"] = mem_log
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append((time.time(), free_gb()))
            stop.wait(3)

    gate_waits = [0]

    def one(j):
        while free_gb() < START_GATE_GB:
            gate_waits[0] += 1
            time.sleep(10)
        return eir.run_job(j, OUT)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    done_at, res = [], []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(workers) as ex:
        fut = [ex.submit(one, j) for j in jobs]
        for f in as_completed(fut):
            r = f.result()
            res.append(r)
            done_at.append(time.time() - t0)
    wall = time.time() - t0
    stop.set()
    th.join()
    bad = [r[0] for r in res if r[1] == "failed"]
    ran = [r for r in res if r[1] == "ok"]
    peaks = []
    if os.path.exists(mem_log):
        with open(mem_log, encoding="utf-8") as f:
            peaks = [json.loads(x) for x in f]
    # the time when every worker was still busy: last job start = the (n - workers + 1)-th finish
    n, w = len(ran), workers
    full_load = sorted(done_at)[n - w] if n > w else None
    row = {"label": label, "workers": workers, "jobs": len(res), "ran": n, "failed": bad, "wall_s": wall, "jobs_per_h": n / wall * 3600 if wall else 0,
           "jobs_done_at_full_load": n - w + 1 if full_load else None, "full_load_s": full_load,
           "jobs_per_h_full_load": (n - w + 1) / full_load * 3600 if full_load else None,
           "sum_job_wall_s": sum(r[2] for r in ran), "free_gb_min": min(s[1] for s in samples), "free_gb_start": samples[0][1], "samples": len(samples),
           "gate_waits": gate_waits[0], "peak_wset_gb_max": max((p["peak_wset"] for p in peaks), default=0) / 2 ** 30,
           "peak_commit_gb_max": max((p["peak_commit"] for p in peaks), default=0) / 2 ** 30,
           "peak_wset_gb_ffn": _mean_peak(peaks, "ffn"), "peak_wset_gb_mha": _mean_peak(peaks, "mha"), "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))}
    with open(os.path.join(OUT, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[next] {label}: {'all ok' if not bad else str(len(bad)) + ' failed: ' + str(bad)}", flush=True)
    print(json.dumps(row, indent=0), flush=True)
    return bad


def _mean_peak(peaks, op):
    v = [p["peak_wset"] for p in peaks if f"_{op}_" in p["name"]]
    return sum(v) / len(v) / 2 ** 30 if v else None


def select(workers):
    """F8 on each A3h mapping, appended (handoff 4.3). Decides on backend seed 1, seeds 2, 3 show the stability."""
    jobs = [dict(eir.job_select(OUT, op), fids=["F8"]) for op in OPS]
    bad = [j["name"] for j in jobs if eir.run_job(j, OUT)[1] == "failed"]
    if bad:
        raise SystemExit(f"select failed: {bad}")
    t = eir.load_faults()
    for op, j in zip(OPS, jobs):
        d = eir.read_sink(j["sink"])["choice"]
        c = d[1]["F8"]
        nodes = [list(n) for n in c["failed_nodes"]]
        row = {"failed_nodes": nodes, "failed_links": [], "failed_pes": [], "kind": c["kind"], "e3_fault": None, "chosen_on": "a3h",
               "same_as": next((g for g in eir.FIDS if t[op][g]["failed_nodes"] == nodes), None),
               "stable_over_seeds_123": all(d[s]["F8"]["failed_nodes"] == c["failed_nodes"] for s in (2, 3)),
               "busy_over_a": c["busy_over_a"], "busy_over_a_seeds_123": [d[s]["F8"]["busy_over_a"] for s in (1, 2, 3)],
               "top_ports": c["top_ports"]}
        if "F8a" in t[op] and t[op]["F8a"]["failed_nodes"] != nodes:
            raise SystemExit(f"{op}: F8a already in faults.json with another node: {t[op]['F8a']['failed_nodes']} vs {nodes}")
        t[op]["F8a"] = row
        print(op, json.dumps(row, ensure_ascii=False))
    tmp = eir.FAULTS_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(t, f, indent=1, ensure_ascii=False)
    os.replace(tmp, eir.FAULTS_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "prep", "prior", "repair", "full", "seeds", "check_log", "select_c", "c_smoke", "c_repair", "c_full", "b_smoke", "b_ck"])
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--share", default="all", choices=["1", "2", "all"])
    a = ap.parse_args()
    if a.cmd == "select":
        select(a.workers)
    elif a.cmd == "prep":
        bad = run_logged("prep_pre", [job_pre_w(o) for o in OPS], a.workers)
        bad = bad or run_logged("prep_l1", [job_l1(o, "F8a") for o in OPS], a.workers)
        for o in OPS:
            p, m = eir.read_sink(job_pre_w(o)["sink"]), eir.read_sink(eir.job_pre(MAIN, "a3h", o)["sink"])
            print(f"{o}: pre rerun T {p['T']}, out/main T {m['T']}, equal {p['T'] == m['T']}, W tags {len(p['W_by_tag'])}", flush=True)
        if bad:
            raise SystemExit(1)
    elif a.cmd == "prior":
        if run_logged("prior", [job_rep(o, "F4a", m, 123) for o in OPS for m in ("L2Impact", "L2Lsync")], a.workers):
            raise SystemExit(1)
    elif a.cmd == "repair":
        s1, s2 = shares(repair_jobs())
        jobs = {"1": s1, "2": s2, "all": s1 + s2}[a.share]
        print(f"[next] repair share {a.share}: {len(jobs)} jobs, FFN {sum(j['op'] == 'ffn' for j in jobs)}, MHA {sum(j['op'] == 'mha' for j in jobs)}", flush=True)
        if run_logged(f"repair_{a.share}_w{a.workers}", jobs, a.workers):
            raise SystemExit(1)
    elif a.cmd == "select_c":
        select_c()
    elif a.cmd == "b_smoke":
        b_smoke(a.workers)
    elif a.cmd == "b_ck":
        jobs = b_jobs()
        print(f"[next] b_ck: {len(jobs)} jobs", flush=True)
        if run_logged(f"b_ck_w{a.workers}", jobs, a.workers):
            raise SystemExit(1)
    elif a.cmd == "c_smoke":
        c_smoke(a.workers)
    elif a.cmd == "c_repair":
        jobs = c_rep_jobs()
        random.Random(C_RANDOM_SEED).shuffle(jobs)
        print(f"[next] c_repair: {len(jobs)} jobs", flush=True)
        if run_logged(f"c_repair_w{a.workers}", jobs, a.workers):
            raise SystemExit(1)
    elif a.cmd == "c_full":
        jobs = c_full_jobs()
        random.Random(C_RANDOM_SEED).shuffle(jobs)
        if run_logged(f"c_full_w{a.workers}", jobs, a.workers):
            raise SystemExit(1)
    elif a.cmd == "check_log":
        check_log(a.workers)
    elif a.cmd == "seeds":
        jobs = seed_jobs()
        random.Random(20261105).shuffle(jobs)
        print(f"[next] seeds: {len(jobs)} jobs, FFN {sum(j['op'] == 'ffn' for j in jobs)}, MHA {sum(j['op'] == 'mha' for j in jobs)}", flush=True)
        if run_logged(f"seeds_w{a.workers}", jobs, a.workers):
            raise SystemExit(1)
    elif a.cmd == "full":
        if run_logged(f"full_w{a.workers}", full_jobs(), a.workers):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
