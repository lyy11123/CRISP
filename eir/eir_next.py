"""EIR next stage, phase A driver (EIR_NEXT_HANDOFF.md sections 2-4). Uses eir.py's jobs; writes only out/next/ (and appends
F8a to faults.json). A3h start, 2x2 FFN and MHA.

  python eir_next.py select                     F8a by choose_fault(F8) on each A3h mapping, appended to faults.json
  python eir_next.py prep --workers 6           pre rerun (with the wait of every tag), l1 at F8a
  python eir_next.py prior --workers 4          section 4.2: FFN, MHA at F4a, s = 100, seed 123, L2Impact and L2Lsync
  python eir_next.py repair --share 1|2 --workers 14|18
  python eir_next.py repair --share all --workers N    whatever is left
  python eir_next.py full --workers N           FULL-A3h at F4a, F8a

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
    return OUT if fid == "F8a" else MAIN


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
    ap.add_argument("cmd", choices=["select", "prep", "prior", "repair", "full"])
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
    elif a.cmd == "full":
        if run_logged(f"full_w{a.workers}", full_jobs(), a.workers):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
