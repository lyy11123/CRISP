"""End-to-end stage A: GPT prefill pilot (e2e/E2E_HANDOFF.md section 1). Imports cpcm_sa (stage 2 rules: exact revert, T with the
cached broadcast, final evaluation with the template's full broadcast, backend seeds 1-10) without changing it; only its module
paths and config table are pointed here. Thread counts are fixed before numpy is imported.

  python e2e_pilot.py threadcheck               same state and seed with / without the thread settings
  python e2e_pilot.py run --ops mha ffn --jobs 14
  python e2e_pilot.py check3                    E3 injection of run 0 per op x arm
  python e2e_pilot.py report                    out/report2.json, csv; out/selfchecks.json
  python e2e_pilot.py readonly --jobs 4         out/readonly.json: sizes, 256 KB blocks, hop backend, T for C4 (one process per mapping)
  python e2e_pilot.py heads                     out/heads.json: heads in the templates
"""
import argparse
import atexit
import ctypes
import gzip
import json
import os
import pickle
import random
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
if os.environ.get("E2E_NO_THREAD_VARS") != "1":
    for v in THREAD_VARS:
        os.environ[v] = "1"
os.environ["CPCM_SA_STAGE"] = "s2"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "cpcm_sa"))
import cpcm_sa as C  # noqa: E402

PREFILL = "intra_mapping_gpt_prefill"
C.WORK = os.path.join(HERE, "work", "artifact")
C.OUT_ROOT = os.path.join(HERE, "out")
C.OUT = os.path.join(C.OUT_ROOT, os.environ["E2E_SUB"]) if os.environ.get("E2E_SUB") else C.OUT_ROOT   # smoke tests: a separate folder
C.CONFIGS["pf_mha"] = (PREFILL, "mha", None, "ch2x2")
C.CONFIGS["pf_ffn"] = (PREFILL, "ffn", None, "ch2x2")
C.REGIME["pf_mha"] = "prefill"
C.REGIME["pf_ffn"] = "prefill"
ARMS = ["A1", "A2", "A3h"]
OPS = {"mha": "pf_mha", "ffn": "pf_ffn"}
# B3: the other operators, reachable only with E2E_B3=1 so that OPS of the pilot's report and read-only stay as they were
C.CONFIGS["pf_ln"] = (PREFILL, "ln", None, "ch2x2")
C.CONFIGS["pf_proj"] = (PREFILL, "proj", None, "ch2x2")
C.CONFIGS["dc_ln"] = (C.GPT, "ln", None, "ch2x2")
C.CONFIGS["dc_proj"] = (C.GPT, "proj", None, "ch2x2")
for _c in ("pf_ln", "pf_proj", "dc_ln", "dc_proj"):
    C.REGIME[_c] = "b3"
if os.environ.get("E2E_B3"):
    OPS.update(ln="pf_ln", proj="pf_proj", dln="dc_ln", dproj="dc_proj")
DECODE_OUT = os.path.join(ROOT, "cpcm_sa", "out", "s2")
GATE_GB, GATE_RAMP_S = 4.0, 20   # a pf_mha run peaks near 0.8 GB, so free memory stays above 3 GB


def peak():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize


def free_gb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def write_peak(name):
    d = os.path.join(C.OUT, "mem")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "peak_wset": peak()}, f)


# ------------------------------------------------------------------ SA runs
def run_one(cfg, arm, g, x, checks, env=None):
    name = C.run_name(cfg, arm, g, x)
    sink = os.path.join(C.OUT, "runs", name + ".pkl.gz")
    snap = os.path.join(C.OUT, "snaps", name + ".pkl.gz")
    if os.path.exists(sink):
        return name, "cached", 0.0
    for d in ("runs", "snaps", "logs"):
        os.makedirs(os.path.join(C.OUT, d), exist_ok=True)
    t0 = time.time()
    with open(os.path.join(C.OUT, "logs", name + ".log"), "w", encoding="utf-8") as lf:
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), "child", cfg, arm, str(g), str(x), sink, snap, "1" if checks else "0"],
                             cwd=HERE, stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, **(env or {})))
    ok = rc == 0 and os.path.exists(sink)
    print(f"[e2e] {name}: exit {rc}, {time.time() - t0:.0f} s, {'ok' if ok else 'FAILED'}", flush=True)
    return name, "ok" if ok else "failed", time.time() - t0


def child():
    _, _, cfg, arm, g, x, sink, snap, checks = sys.argv
    atexit.register(write_peak, C.run_name(cfg, arm, int(g), int(x)))
    C.child(cfg, arm, int(g), int(x), sink, snap, checks == "1")


def run(ops, jobs, first=False, arms=ARMS, tag=""):
    todo = []
    for op in ops:
        for arm in arms:
            vs = C.variants(arm)
            for i, (g, x) in enumerate(vs[:2] if first else vs):
                todo.append((OPS[op], arm, g, x, i < 2))
    todo.sort(key=lambda t: t[1] == "A1")
    mins = [free_gb()]
    stop = threading.Event()

    def watch():
        while not stop.wait(5):
            mins.append(free_gb())
    threading.Thread(target=watch, daemon=True).start()
    t0 = time.time()
    if tag == "_mha_cpcm":      # parent session 2026-11-07: 14 processes with the memory gate instead of the 6 in run_pilot.sh
        jobs = 14
    lock, waits = threading.Lock(), [0]

    def gated(t):
        """Memory gate (as eir_next): start a run only with GATE_GB free, one start per GATE_RAMP_S so its memory shows up."""
        if not os.path.exists(os.path.join(C.OUT, "runs", C.run_name(*t[:4]) + ".pkl.gz")):
            with lock:
                while free_gb() < GATE_GB:
                    waits[0] += 1
                    time.sleep(30)
                time.sleep(GATE_RAMP_S)
        return run_one(*t)
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(gated, todo))
    stop.set()
    bad = [r[0] for r in res if r[1] == "failed"]
    print(f"[e2e] {len(res)} runs, {len(bad)} failed: {bad}; {time.time() - t0:.0f} s; min free {min(mins):.1f} GB", flush=True)
    os.makedirs(C.OUT, exist_ok=True)
    with open(os.path.join(C.OUT, f"run_meta{'_first' if first else ''}{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": len(res), "failed": bad, "wall_s": time.time() - t0, "jobs": jobs, "gate_waits_30s": waits[0], "min_free_gb": min(mins), "start_free_gb": mins[0]}, f, indent=1)


# ------------------------------------------------------------------ thread settings
def threadcheck_child(snap_path, cfg, ctx_run, out):
    """T (before / after the broadcast) and backend seeds 1-3 of one saved state, in this process."""
    C._artifact_path()
    from add_communication import add_broadcast
    from event_driver import event_driver
    with gzip.open(ctx_run, "rb") as f:
        ctx = pickle.load(f)["meta"]["ctx"]
    with gzip.open(snap_path, "rb") as f:
        blob, hist = pickle.load(f)
    b_, d_, e_ = pickle.loads(blob)
    hp = C.platform(cfg)
    hp.link_loads_dict.update(hist["link_loads_dict"])
    hp.link_loads_count.update(hist["link_loads_count"])
    keep = C.BACKEND_SEEDS
    C.BACKEND_SEEDS = [1, 2, 3]
    fe = C.final_eval((b_, d_, e_, hp), ctx, add_broadcast, event_driver)
    C.BACKEND_SEEDS = keep
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"T_pre": fe["T_pre"], "T_bc": fe["T_bc"], "backend": {str(k): list(v) for k, v in fe["backend"].items()},
                   "env": {v: os.environ.get(v) for v in THREAD_VARS}}, f)


def threadcheck():
    d = os.path.join(C.OUT, "threadcheck")
    os.makedirs(d, exist_ok=True)
    jobs = []
    for op in OPS:
        cfg = OPS[op]
        name = C.run_name(cfg, "A3h", *C.variants("A3h")[0])
        for mode in ("on", "off"):
            for rep in (1, 2):
                jobs.append((cfg, name, mode, rep))

    def one(j):
        cfg, name, mode, rep = j
        out = os.path.join(d, f"{cfg}_{mode}{rep}.json")
        env = dict(os.environ, E2E_NO_THREAD_VARS="1" if mode == "off" else "0")
        if mode == "off":
            for v in THREAD_VARS:
                env.pop(v, None)
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), "threadcheck_child", os.path.join(C.OUT, "snaps", name + ".pkl.gz"), cfg,
                              os.path.join(C.OUT, "runs", name + ".pkl.gz"), out], cwd=HERE, env=env)
        return rc
    with ThreadPoolExecutor(8) as ex:
        print("rc", list(ex.map(one, jobs)))
    res = {}
    for cfg, name, mode, rep in jobs:
        res[f"{cfg}_{mode}{rep}"] = json.load(open(os.path.join(d, f"{cfg}_{mode}{rep}.json"), encoding="utf-8"))
        print(cfg, mode, rep, {k: v for k, v in res[f"{cfg}_{mode}{rep}"].items() if k != "backend"} , [v[0] for v in res[f"{cfg}_{mode}{rep}"]["backend"].values()])
    with open(os.path.join(C.OUT, "threadcheck.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


# ------------------------------------------------------------------ report
def report():
    cfgs = list(OPS.values())
    res = C.report2(cfgs, ARMS)
    S = C.summarise(cfgs, ARMS)
    extra = {}
    for (cfg, arm), s in S.items():
        rows = {}
        peaks = []
        for r in s["runs"]:
            n = C.run_name(cfg, arm, r["g"], r["x"])
            p = os.path.join(C.OUT, "mem", n + ".json")
            if os.path.exists(p):
                peaks.append(json.load(open(p))["peak_wset"] / 2 ** 30)
        bw = [w for r in s["runs"] for w in r["d"]["selected"]["backend_wall"]]
        acc = [sum(1 for x in r["d"]["rows"] if x["accepted"]) for r in s["runs"]]
        extra[f"{cfg}_{arm}"] = {"backend_wall_median_s": statistics.median(bw), "backend_wall_max_s": max(bw), "peak_wset_gb_max": max(peaks) if peaks else None,
                                 "peak_wset_gb_median": statistics.median(peaks) if peaks else None, "accepted_min": min(acc), "accepted_median": statistics.median(acc),
                                 "comm_share_best_backend": s["best_backend"]["d"]["selected"]["backend"] and C.triple(s["best_backend"]["d"]["selected"]["backend"])[2] / s["best_backend"]["mean"],
                                 "comm_share_best_own": C.triple(s["best_own"]["d"]["selected"]["backend"])[2] / s["best_own"]["mean"],
                                 "comm_share_median": statistics.median(C.triple(r["d"]["selected"]["backend"])[2] / r["mean"] for r in s["runs"])}
    with open(os.path.join(C.OUT, "extra.json"), "w", encoding="utf-8") as f:
        json.dump(extra, f, indent=1)
    print(json.dumps(extra, indent=1))
    return res


# ------------------------------------------------------------------ read-only numbers
def readonly_child(spec_path, out):
    spec = json.load(open(spec_path, encoding="utf-8"))
    sys.path.insert(0, os.path.join(ROOT, "dstu"))
    C._artifact_path()
    import dstu as D
    import dstu2 as S
    from dstu import E, cpcm
    from add_communication import add_broadcast
    from e4b_backend import P1, event_driver_e4b
    with gzip.open(spec["run"], "rb") as f:
        ctx = pickle.load(f)["meta"]["ctx"]
    with gzip.open(spec["snap"], "rb") as f:
        blob, hist = pickle.load(f)
    beha, data, events = pickle.loads(blob)
    hp = C.platform(spec["cfg"])
    hp.link_loads_dict.update(hist["link_loads_dict"])
    hp.link_loads_count.update(hist["link_loads_count"])
    random.seed(C.BCAST_SEED)
    np.random.seed(C.BCAST_SEED)
    add_broadcast(data_tags=ctx["broadcast_datatags"], ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]], beha_dict=beha, data_dict=data, hardware_platform=hp,
                  event_dict=events, dijkstra_routing=ctx["dijkstra_routing"], alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])
    res = {k: spec[k] for k in ("op", "phase", "arm", "run_name")}
    print(f"[segment {time.strftime('%H:%M:%S')}] state and broadcast ready", flush=True)
    g = D._graph(events, hp)
    sc = cpcm.simulate(g, "bb")
    by = sorted(int(e.comm_bytes) for e in events.values() if e.event_type == "communication")
    q = lambda p: by[min(len(by) - 1, int(p * len(by)))]
    res.update(events=g.n, computations=sum(1 for e in events.values() if e.event_type == "computation"), messages=len(by), activities=len(sc.a_s), T_bb=sc.T,
               bytes_total=sum(by), bytes_min=by[0], bytes_p50=q(0.5), bytes_p90=q(0.9), bytes_max=by[-1], messages_over_256KB=sum(1 for b in by if b > 256 * 1024))
    ev, cst = S.chunk_by(E.merged_events([pickle.dumps(events, protocol=pickle.HIGHEST_PROTOCOL)]), hp, S.UNI["256all"])
    res.update(blocks_added=cst["blocks_added"], messages_chunked=cst["messages_chunked"], events_chunked=len(ev), comm_events_chunked=sum(1 for e in ev.values() if e.event_type == "communication"),
               bytes_conserved=cst["bytes_conserved"], tree_conserved=cst["tree_conserved"], extra_occupancy=cst["extra_occupancy"], base_occupancy=cst["base_occupancy"])
    print(f"[segment {time.strftime('%H:%M:%S')}] chunked, starting T for C4", flush=True)
    # T for C4 (dstu_oh.solo_child, o = 0): CPCM hop mode on the stream's own graph, tail data
    t0 = time.perf_counter()
    g2 = D._graph(ev, hp)
    t1 = time.perf_counter()
    s2 = cpcm.simulate(g2, "hop")
    t2 = time.perf_counter()
    an = cpcm.Analysis(s2, sink_deadline=[s2.T] * g2.n)
    D.tail_data(g2, s2, an, s2.T, range(g2.n))
    t3 = time.perf_counter()
    res.update(c4_compile_s=t1 - t0, c4_hop_sim_s=t2 - t1, c4_tail_s=t3 - t2, c4_total_s=t3 - t0, T_hop_chunked=s2.T, activities_chunked=len(s2.a_s))
    res["peak_wset_gb_before_backend"] = peak() / 2 ** 30
    print(f"[segment {time.strftime('%H:%M:%S')}] T for C4 done, starting hop backend", flush=True)
    if os.environ.get("E2E_RO_SKIP_BACKEND"):     # partial numbers while the full process runs: no hop backend
        res["peak_wset_gb"] = res["peak_wset_gb_before_backend"]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=float)
        return
    # backend per hop, policy P1, one seed
    hpb = C.copy.deepcopy(hp)
    audit, released = [], {}
    random.seed(1)
    t0 = time.time()
    event_driver_e4b(ev, hpb, release={0: 0}, policy=P1(), released=released, audit=audit)
    res["hop_backend_wall_s"] = time.time() - t0
    print(f"[segment {time.strftime('%H:%M:%S')}] hop backend done", flush=True)
    res["hop_backend_latency"] = max(e.end_time for e in ev.values())
    res["hop_audit_mismatch"] = sum(1 for a in audit if a[2] != a[3])
    res["peak_wset_gb"] = peak() / 2 ** 30
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)


RO_FLOOR_GB, RO_KILL_GB, RO_EST_GB, RO_RAMP_S, RO_LIMIT_S = 3.0, 3.3, 4.0, 30, 7200


def readonly(jobs):
    d = os.path.join(C.OUT, "readonly")
    os.makedirs(d, exist_ok=True)
    specs = []
    for op, cfg in OPS.items():
        S = C.summarise([cfg], ["A1", "A3h"])
        for arm in ("A1", "A3h"):
            bb = S[(cfg, arm)]["best_backend"]
            n = C.run_name(cfg, arm, bb["g"], bb["x"])
            specs.append(dict(op=op, phase="prefill", arm=arm, cfg=cfg, run_name=n, run=os.path.join(C.OUT, "runs", n + ".pkl.gz"), snap=os.path.join(C.OUT, "snaps", n + ".pkl.gz")))
    # decode: existing cpcm_sa stage 2 mappings (best by backend, out/s2/report_{ffn,mha}.json)
    for op in OPS:
        rep = json.load(open(os.path.join(DECODE_OUT, f"report_{op}.json"), encoding="utf-8"))
        for arm in ("A1", "A3h"):
            t = next(t for t in rep["table"] if t["cfg"] == op and t["arm"] == arm)
            g, x = t["run"].split("_")
            n = C.run_name(op, arm, int(g[1:]), int(x))
            specs.append(dict(op=op, phase="decode", arm=arm, cfg=op, run_name=n, run=os.path.join(DECODE_OUT, "runs", n + ".pkl.gz"), snap=os.path.join(DECODE_OUT, "snaps", n + ".pkl.gz")))

    # one process per mapping; free memory is watched during the run, not only at start
    logp = os.path.join(d, "driver.log")

    def say(msg):
        line = f"[readonly {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(line + chr(10))
    q = [s for s in sorted(specs, key=lambda s: s["op"] != "ffn") if not os.path.exists(os.path.join(d, f"{s['phase']}_{s['op']}_{s['arm']}.json"))]
    est = {"ffn": RO_EST_GB, "mha": RO_EST_GB}
    live, alerted, peak_seen, min_free = [], set(), {}, free_gb()
    say(f"{len(q)} mappings to do, free {min_free:.1f} GB, floor {RO_FLOOR_GB} GB, jobs at most {jobs}")
    while q or live:
        fr = free_gb()
        min_free = min(min_free, fr)
        for item in list(live):
            s, pr, t0, lf = item
            name = f"{s['phase']}_{s['op']}_{s['arm']}"
            if pr.poll() is not None:
                lf.close()
                live.remove(item)
                path = os.path.join(d, name + ".json")
                if os.path.exists(path):
                    r = json.load(open(path, encoding="utf-8"))
                    est[s["op"]] = max(r["peak_wset_gb"], est[s["op"]] if est[s["op"]] != RO_EST_GB else 0)
                    say(f"{name}: done, {time.time() - t0:.0f} s, peak {r['peak_wset_gb']:.2f} GB, min free so far {min_free:.1f} GB")
                else:
                    say(f"{name}: FAILED exit {pr.returncode}")
            elif time.time() - t0 > RO_LIMIT_S and name not in alerted:
                alerted.add(name)
                say(f"ALERT {name}: over 2 hours, still running")
        if fr < RO_KILL_GB and live:
            s, pr, t0, lf = live[-1]
            pr.kill()
            pr.wait()
            lf.close()
            live.pop()
            q.insert(0, s)
            est[s["op"]] = max(est[s["op"]], RO_EST_GB) * 1.5
            say(f"KILL {s['phase']}_{s['op']}_{s['arm']}: free {fr:.1f} GB below {RO_KILL_GB} GB; requeued, estimate now {est[s['op']]:.1f} GB")
        elif q and len(live) < jobs and fr - est[q[0]["op"]] >= RO_KILL_GB:
            # the first mapping of each op runs alone among same-op unknowns, so its peak is measured before others start
            if est[q[0]["op"]] != RO_EST_GB or not any(x[0]["op"] == q[0]["op"] for x in live):
                s = q.pop(0)
                name = f"{s['phase']}_{s['op']}_{s['arm']}"
                p = os.path.join(d, name)
                with open(p + "_spec.json", "w", encoding="utf-8") as f:
                    json.dump(s, f)
                lf = open(p + ".log", "w", encoding="utf-8")
                live.append((s, subprocess.Popen([sys.executable, os.path.abspath(__file__), "readonly_child", p + "_spec.json", p + ".json"], cwd=HERE, stdout=lf, stderr=subprocess.STDOUT), time.time(), lf))
                say(f"{name}: start, free {fr:.1f} GB, {len(live)} live")
                time.sleep(RO_RAMP_S)
        elif q and not live and fr - est[q[0]["op"]] < RO_KILL_GB:
            say(f"ALERT waiting: free {fr:.1f} GB, need {est[q[0]['op']] + RO_KILL_GB:.1f} GB for {q[0]['phase']}_{q[0]['op']}_{q[0]['arm']}")
            time.sleep(600)
        time.sleep(5)
    say(f"finished, min free {min_free:.2f} GB")
    missing = [s for s in specs if not os.path.exists(os.path.join(d, f"{s['phase']}_{s['op']}_{s['arm']}.json"))]
    if missing:
        say(f"ALERT missing: {[(s['phase'], s['op'], s['arm']) for s in missing]}")
        return
    res = [json.load(open(os.path.join(d, f"{s['phase']}_{s['op']}_{s['arm']}.json"), encoding="utf-8")) for s in specs]
    with open(os.path.join(C.OUT, "readonly.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for r in res:
        print(r)


def main():
    cmd = sys.argv[1]
    if cmd == "child":
        return child()
    if cmd == "threadcheck_child":
        return threadcheck_child(*sys.argv[2:])
    if cmd == "readonly_child":
        return readonly_child(*sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--ops", nargs="*", default=list(OPS))
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--first", action="store_true")
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if cmd == "run":
        run(a.ops, a.jobs, a.first, a.arms, a.tag)
    elif cmd == "threadcheck":
        threadcheck()
    elif cmd == "check3":
        C.check3(list(OPS.values()), ARMS, a.jobs)
    elif cmd == "report":
        report()
    elif cmd == "readonly":
        readonly(a.jobs)


if __name__ == "__main__":
    main()
