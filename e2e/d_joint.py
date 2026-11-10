"""D: A3h + EIR + DSTU joint experiment (E2E_HANDOFF.md section 6, user release 2026-11-10; plan e2e/D_PLAN.md).

Scenario S-A: prefill (P) and decode (D) share one 2x2 group. A unit = one prefill operator graph (stream 0, released at 0) and one
decode step as a serial chain (stream 1: LN, MHA x 8, projection, LN, FFN; released at tau = f x T_P).

Data come from C (nothing in eir/, e2e/c/ or C's outputs is written): the main checkout's e2e/c (work copy, inputs, faults.json,
c/out) and colab_results/c. This file may live in a worktree; MAIN is the folder that holds e2e/c/work/artifact.

Two kinds of child process (one task script per process, as everywhere in this repository):
  export  one operator state -> its event graph (W1-filtered, unchunked, the graph BusyBarn's backend evaluates), pickled.
          EIR states (start before the fault, R0, L1, a repair's final state) go through c_eir.child with kind "reload", whose
          handler is replaced in memory by d_export (eir.child_reload is not used): the state is loaded into the engine, T-hat
          is recomputed and compared with the one C saved, then the graph is taken as Engine.finalize takes it.
          FULL-BusyBarn winners (cpcm_sa snapshots) are rebuilt as cpcm_sa.final_eval does (broadcast seeded BCAST_SEED, W1 filter
          on the broadcast), and T-hat (bb, with the broadcast) is compared with the saved T_bc.
  sim     a fresh platform from the cfg and the fault (E4's build_platform; failed PEs removed as c_eir.child_full does); the
          structure must equal the exporting process's platform. Streams are renamed (tag[1] = 10 + stream), the decode chain is
          built, optionally chunked (dstu2.chunk_by, all links 256 KB, o = 0), and run on the original backend
          (e4_backend.event_driver_e4) or the per-hop backend (d_hop = fast_hop with a stop on the decode stream).
          Event hash = hash of the tag in every sim process (as the W1 rerun, option 2). T-hat for C4: fast_cpcm, mode hop, on
          the stream's own chunked graph (as dstu.solo_child).

  python e2e/d_joint.py d0 [--workers N] [--only export|stats|probe]      D0 (D_PLAN section 8), outputs in e2e/d/out/d0
  python e2e/d_joint.py export SPEC.json | sim SPEC.json                   children
"""
import argparse
import collections
import copy
import ctypes
import gzip
import hashlib
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

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
for _v in THREAD_VARS:
    os.environ[_v] = "1"
os.environ.setdefault("CPCM_SA_STAGE", "s2")

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_main():
    if os.environ.get("D_MAIN"):
        return os.environ["D_MAIN"]
    p = os.path.dirname(HERE)
    for _ in range(6):
        if os.path.isdir(os.path.join(p, "e2e", "c", "work", "artifact", "models")):
            return p
        p = os.path.dirname(p)
    raise SystemExit("d_joint: no folder with e2e/c/work/artifact found above this file (set D_MAIN)")


MAIN = _find_main()
ME2E = os.path.join(MAIN, "e2e")
for _p in (os.path.join(MAIN, "cpcm"), os.path.join(MAIN, "e4_overlap"), os.path.join(MAIN, "dstu"), HERE, ME2E):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)          # ends as [ME2E, HERE, dstu, e4_overlap, cpcm, ...]: c_eir, fast_hop from MAIN; d_hop from HERE
OUT = os.environ.get("D_OUT", os.path.join(HERE, "d", "out"))
SCRIPT = os.path.abspath(__file__)
C_REPORT = os.path.join(ME2E, "c", "out", "c_report.json")
ROOTS = [os.path.join(ME2E, "c", "out"), os.path.join(MAIN, "colab_results", "c")]
CFG_NAME = "config_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0.cfg"
K256 = 256 * 1024
CHUNK = {"size": K256, "only_d2d": False}
K = 1.5
FS = (0.1, 0.4, 0.7)
SEEDS = (1, 2, 3)
CHAIN = ["dc_ln", "dc_mha", "dc_mha", "dc_mha", "dc_mha", "dc_mha", "dc_mha", "dc_mha", "dc_mha", "dc_proj", "dc_ln", "dc_ffn"]
PF = ("pf_ln", "pf_mha", "pf_proj", "pf_ffn")
DC = ("dc_ln", "dc_mha", "dc_proj", "dc_ffn")
OPS8 = PF + DC
# memory stop lines (GB): per process working set and system free memory; set per machine (--peak-gb / --free-gb or D_PEAK_GB /
# D_FREE_GB). Defaults: this machine's rule (3 GB / 3 GB); a remote machine with more memory passes its own (e.g. 5 GB per process).
LOCAL_PEAK_GB = float(os.environ.get("D_PEAK_GB", "3.0"))
LOCAL_FREE_GB = float(os.environ.get("D_FREE_GB", "3.0"))


# ---------------------------------------------------------------- memory (Windows: working set of a child; Linux: /proc)
def ws_gb(pid):
    if os.name != "nt":
        try:
            with open(f"/proc/{pid}/status") as f:
                for ln in f:
                    if ln.startswith("VmRSS:"):
                        return int(ln.split()[1]) / 2 ** 20
        except OSError:
            return 0.0
        return 0.0
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                    ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.OpenProcess.restype = wintypes.HANDLE
    h = k32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not h:
        return 0.0
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    ok = psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    k32.CloseHandle(h)
    return pmc.WorkingSetSize / 2 ** 30 if ok else 0.0


def free_gb():
    import e2e_pilot_peak as PK
    return PK.free_gb()


def peak_gb():
    import e2e_pilot_peak as PK
    return PK.peak() / 2 ** 30


# ---------------------------------------------------------------- platform structure and tags
def plat_struct(hp):
    links = sorted((str(k), l.link_type, l.bandwidth, l.latency, l.timeunit) for k, l in hp.links_dict.items())
    mods = sorted((t, str(loc)) for t, ms in hp.modules_dict.items() for loc in ms)
    blob = repr((links, mods, hp.frequency))
    return {"sha1": hashlib.sha1(blob.encode()).hexdigest(), "n_links": len(links), "n_modules": len(mods)}


def tag_facts(ev):
    comp = {t for t, e in ev.items() if e.event_type == "computation"}
    comm = {t for t, e in ev.items() if e.event_type == "communication"}
    return {"n_comp": len(comp), "n_comm": len(comm), "comp_len": sorted({len(t) for t in comp}), "comm_len": sorted({len(t) for t in comm}),
            "tag0": sorted({t[0] for t in ev}), "tag1": sorted({t[1] for t in ev}), "tag2": sorted({t[2] for t in ev}),
            "comm_owner_is_comp": all((0,) + tuple(t[1:-1]) in comp for t in comm)}


# ---------------------------------------------------------------- export child
def d_export(eng, spec, sink):
    """Replaces eir.child_reload inside c_eir's child (kind "reload")."""
    import eir as E
    fr = spec["d_from"]
    t0 = time.perf_counter()
    if fr["type"] == "pre":
        sink["bc_seed"] = eng.seed_broadcast()
        ref = E.read_sink(fr["sink"])
    else:
        ref = E.read_sink(fr["sink"])
        eng.sync = ref["spec"].get("method") in E.SYNC_METHODS
        eng.load(ref[fr["key"]])
    T_ref = ref
    for k in fr["ref_key"]:
        T_ref = T_ref[k]
    T = eng.t_eval()
    s = eng.a.state
    ev = eng.bc.graph(s[5], s[6], s[7], eng.hp)        # as Engine.finalize builds the graph the backend evaluates
    sink["d_export"] = {"T": T, "T_ref": T_ref, "T_equal": T == T_ref, "events_blob": pickle.dumps(ev, protocol=pickle.HIGHEST_PROTOCOL),
                        "n_events": len(ev), "plat": plat_struct(eng.hp), "tag_facts": tag_facts(ev),
                        "sync": eng.sync, "wall": time.perf_counter() - t0, "peak_gb": peak_gb()}


def export_eir(spec_path):
    import c_eir as c
    c.eir.child_reload = d_export
    c.child(spec_path)


def export_full(spec_path):
    """A FULL run's selected state (cpcm_sa snapshot) -> the graph cpcm_sa.final_eval gives the backend."""
    import numpy as np
    import c_eir as c
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    E, cs = c.eir, c.cs
    op, fid = spec["op"], spec["fault"]
    c.redirect()
    E.GPT = c.OPS[op][0]
    E.TASKS[op] = c.task_of(op)
    if c.OPS[op][4]:
        c.install_w1("full", op)
    f = E.fault_of(op, fid)
    cfg = E.full_cfg(op, fid)
    cs.CONFIGS[cfg] = (c.OPS[op][0], c.OPS[op][1], {"failed_nodes": f["failed_nodes"], "failed_links": f["failed_links"]}, "ch2x2")
    cs.REGIME[cfg] = "c"
    cs._artifact_path()
    if f.get("failed_pes"):
        import WAMIS_HD
        pes = set(map(tuple, f["failed_pes"]))
        orig = WAMIS_HD.wamis_hdc.__init__

        def __init__(self, platform_cfg):
            orig(self, platform_cfg=platform_cfg)
            for d in (self.tensorcore_dict, self.vectorunit_dict):
                for loc in list(d):
                    if tuple(loc[:-1]) in pes:
                        del d[loc]
        WAMIS_HD.wamis_hdc.__init__ = __init__
    t0 = time.perf_counter()
    run = E.read_sink(spec["run_sink"])
    ctx, sel = run["meta"]["ctx"], run["selected"]
    with gzip.open(spec["snap"], "rb") as fh:
        blob, hist = pickle.load(fh)
    beha, data, events = pickle.loads(blob)
    hp = copy.deepcopy(cs.platform(cfg))
    for k in ("link_loads_dict", "link_loads_count"):
        if set(getattr(hp, k)) != set(hist[k]):
            raise RuntimeError(f"{k}: link set of the rebuilt platform differs from the run's")
        getattr(hp, k).update(hist[k])
    from add_communication import add_broadcast         # after install_w1: the filtered one for MHA
    cpcm = cs.cpcm
    rng = random.getstate(), np.random.get_state()
    random.seed(cs.BCAST_SEED)
    np.random.seed(cs.BCAST_SEED)
    add_broadcast(data_tags=ctx["broadcast_datatags"], ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]],
                  beha_dict=beha, data_dict=data, hardware_platform=hp, event_dict=events,
                  dijkstra_routing=ctx["dijkstra_routing"], alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])
    random.setstate(rng[0])
    np.random.set_state(rng[1])
    deps = {t: set(e.dependency_set) for t, e in events.items()}
    paths = {t: [list(b) for b in e.path_list] for t, e in events.items() if e.event_type == "communication"}
    T = cpcm.simulate(cpcm.compile_bb(events, hp, path_lists=paths, deps=deps), "bb").T
    sink = {"spec": spec, "w1": {f"{w}:{n}": k for (w, n), k in c.DROPS.items()},
            "d_export": {"T": T, "T_ref": sel["T_bc"], "T_equal": T == sel["T_bc"],
                         "events_blob": pickle.dumps(events, protocol=pickle.HIGHEST_PROTOCOL), "n_events": len(events),
                         "plat": plat_struct(hp), "tag_facts": tag_facts(events), "wall": time.perf_counter() - t0, "peak_gb": peak_gb()}}
    E.write_sink(spec["sink"], sink)


# ---------------------------------------------------------------- sim child
class Audit:
    """Self-check 3 of DSTU (chosen key = minimum key over the candidates), counted without keeping the records."""

    def __init__(self):
        self.n = self.bad = 0

    def append(self, rec):
        self.n += 1
        if rec[2] != rec[3]:
            self.bad += 1


def _sim_setup():
    import c_eir as c
    c.redirect()
    c.cs._artifact_path()
    import event_notation
    event_notation.event_notation.__hash__ = lambda self: hash(self.event_tag)
    import cpcm
    import fast_cpcm
    cpcm.simulate = fast_cpcm.simulate
    return c


def build_platform(c, fault):
    import e4_overlap as E4
    path = os.path.join(c.WORK, "models", c.PREFILL, "cfg", CFG_NAME)
    hp = E4.build_platform(path, fault or None)
    if fault and fault.get("failed_pes"):
        pes = set(map(tuple, fault["failed_pes"]))
        for d in (hp.tensorcore_dict, hp.vectorunit_dict):
            for loc in list(d):
                if tuple(loc[:-1]) in pes:
                    del d[loc]
    return hp


def retag(ev, sid, k=0):
    """tag[1] = 10 + sid, tag[2] = k (the operator's place in a chain); needs tag[1] = tag[2] = 0 (E4a's rule)."""
    m = {}
    for t in ev:
        if t[1] != 0 or t[2] != 0:
            raise RuntimeError(f"tag {t}: positions 1 and 2 are not 0")
        m[t] = (t[0], 10 + sid, k) + tuple(t[3:])
    out = {}
    for t, e in ev.items():
        e.event_tag = m[t]
        e.dependency_set = {m[x] for x in e.dependency_set}
        e.issue_set = {m[x] for x in e.issue_set}
        out[m[t]] = e
    comp = {t for t, e in out.items() if e.event_type == "computation"}
    bad = [t for t, e in out.items() if e.event_type == "communication" and (0,) + t[1:-1] not in comp]
    if bad:
        raise RuntimeError(f"stream {sid}: communication owner not found after renaming: {bad[:3]}")
    return out


def chain(blobs, sid):
    """Operators in CHAIN order; every source of operator k+1 waits for every sink of operator k."""
    out, prev_sinks, edges = {}, None, 0
    for k, op in enumerate(CHAIN):
        ev = retag(pickle.loads(blobs[op]), sid, k)
        sources = [t for t, e in ev.items() if not e.dependency_set]
        sinks = [t for t, e in ev.items() if not e.issue_set]
        if prev_sinks is not None:
            for t in sources:
                ev[t].dependency_set.update(prev_sinks)
            for t in prev_sinks:
                out[t].issue_set.update(sources)
            edges += len(sources) * len(prev_sinks)
        out.update(ev)
        prev_sinks = sinks
    return out, edges


def stream_of(t):
    return t[1] - 10


def that_hop(ev, hp):
    import cpcm
    import dstu as D
    t0 = time.perf_counter()
    g = D._graph(ev, hp)
    sc = cpcm.simulate(g, "hop")
    return sc.T, time.perf_counter() - t0, g.n


def run_backend(c, ev_blob, hp0, backend, policy, release, seed, stop=False, record=False, audit=True):
    import numpy as np
    import e4b_backend as B
    import e4b
    ev = pickle.loads(ev_blob)
    hp = copy.deepcopy(hp0)
    random.seed(seed)
    np.random.seed(seed)
    released, info, au = {}, {}, (Audit() if (audit and backend == "hop") else None)
    t0 = time.perf_counter()
    if backend == "orig":
        from e4_backend import event_driver_e4
        out = event_driver_e4(ev, hp, release=release, released=released)
    else:
        import d_hop
        pol = None if policy["name"] == "P1" else e4b.Fixed({int(i): v for i, v in policy["rank"].items()})
        out = d_hop.event_driver_stop(ev, hp, release=release, policy=pol, released=released, audit=au,
                                      stop_stream=1 if stop else None, stop_info=info)
    wall = time.perf_counter() - t0
    per = collections.defaultdict(lambda: [None, None, 0, 0])      # stream -> [first start, last end, n, n finished]
    for t, e in ev.items():
        s = stream_of(t)
        p = per[s]
        p[2] += 1
        if e.end_time and (not stop or e.end_time <= info.get("t_stop", float("inf"))):      # an event never run keeps end_time 0
            p[3] += 1
            p[1] = e.end_time if p[1] is None else max(p[1], e.end_time)
        if e.start_time is not None:
            p[0] = e.start_time if p[0] is None else min(p[0], e.start_time)
    res = {"backend": backend, "policy": policy["name"], "seed": seed, "stop": stop, "out": [int(x) for x in out], "wall": wall,
           "streams": {s: {"first_start": p[0], "C": p[1], "n": p[2], "n_done": p[3]} for s, p in per.items()},
           "release": {int(k): v for k, v in (release or {}).items()}, "t_stop": info.get("t_stop")}
    if au is not None:
        res["audit"] = {"n": au.n, "bad": au.bad}
    if record:
        res["record"] = {t: (e.start_time, e.end_time) for t, e in ev.items()}
        res["links"] = {str(k): [tuple(x[:2]) + (x[2],) for x in L.work_record] for k, L in hp.links_dict.items() if L.work_record}
    return res


def sim(spec_path):
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    c = _sim_setup()
    import dstu2
    t_start = time.perf_counter()
    hp0 = build_platform(c, spec.get("fault_def"))
    ps = plat_struct(hp0)
    sink = {"spec": spec, "plat": ps, "runs": [], "checks": {}}
    ex = {}
    for name, p in spec["exports"].items():
        with gzip.open(p, "rb") as fh:
            d = pickle.load(fh)["d_export"]
        if d["plat"]["sha1"] != ps["sha1"]:
            raise RuntimeError(f"{name}: platform structure differs from the exporting process's ({p})")
        ex[name] = d["events_blob"]
    mode = spec["mode"]
    if mode == "stats":
        rows = {}
        for name, blob in ex.items():
            try:
                ev = retag(pickle.loads(blob), 0)
            except RuntimeError as e:
                rows[name] = {"retag_error": str(e), "tag_facts": tag_facts(pickle.loads(blob))}
                continue
            t0 = time.perf_counter()
            ch, cst = dstu2.chunk_by(ev, hp0, CHUNK)
            rows[name] = {"n_events": len(ev), "tag_facts": tag_facts(pickle.loads(blob)), "chunk": cst, "chunk_wall": time.perf_counter() - t0}
            del ch, ev
        sink["stats"] = rows
    else:
        streams = {}
        if spec.get("P"):
            streams[0] = ("P", lambda: retag(pickle.loads(ex[spec["P"]]), 0))
        if spec.get("D"):
            streams[1] = ("D", lambda: chain({op: ex[spec["D"][op]] for op in DC}, 1)[0])
        # the per-hop backend runs on the 256 KB chunked graph, the original backend on the unchunked one (D_PLAN section 3;
        # 2026-11-10: the first D0 probes gave the original backend the chunked graph, which is not a configuration of D)
        built, raw_blob = {}, {}
        for s, (nm, fn) in streams.items():
            t0 = time.perf_counter()
            ev = fn()
            n0 = len(ev)
            raw_blob[s] = pickle.dumps(ev, protocol=pickle.HIGHEST_PROTOCOL)
            cst = None
            if spec["chunk"]:
                ev, cst = dstu2.chunk_by(ev, hp0, CHUNK)
            built[s] = ev
            sink["checks"][f"build_{nm}"] = {"n_events": n0, "n_chunked": len(ev), "chunk": cst, "wall": time.perf_counter() - t0}
        if spec.get("D"):
            _, edges = chain({op: ex[spec["D"][op]] for op in DC}, 1)
            sink["checks"]["chain_edges"] = edges
        solo_blob = {s: pickle.dumps(ev, protocol=pickle.HIGHEST_PROTOCOL) for s, ev in built.items()}
        that = {}
        for s in spec.get("that", []):
            T, w, n = that_hop(built[s], hp0)
            that[s] = T
            sink["checks"][f"that_{s}"] = {"T_hat": T, "wall": w, "n": n}
        del built
        merged_blob = {}
        if len(solo_blob) == 2:
            for nm_, bl in (("hop", solo_blob), ("orig", raw_blob)):
                merged = pickle.loads(bl[0])
                merged.update(pickle.loads(bl[1]))
                merged_blob[nm_] = pickle.dumps(merged, protocol=pickle.HIGHEST_PROTOCOL)
                del merged
        blob_of = lambda b, s: (raw_blob if b == "orig" else solo_blob)[s]  # noqa: E731
        T_solo = {}
        for r in spec["runs"]:
            pol = dict(r.get("policy", {"name": "P1"}))
            if r["kind"] == "solo":
                s = r["stream"]
                rel = {s: 0}
                if pol["name"] == "C4":
                    pol["rank"] = {s: 0}
                res = run_backend(c, blob_of(r["backend"], s), hp0, r["backend"], pol, rel, r["seed"], record=r.get("record", False))
                T_solo.setdefault((s, r["backend"]), []).append(res["out"][0])
            else:
                TP = r.get("T_P") or statistics.mean(T_solo[(0, r["backend"])])
                tau = int(round(r["f"] * TP))
                rel = {0: 0, 1: tau}
                if pol["name"] == "C4":
                    d_hat = {0: K * that[0], 1: tau + K * that[1]}
                    order = sorted(d_hat, key=lambda i: (d_hat[i], i))
                    pol["rank"] = {i: order.index(i) for i in d_hat}
                    pol["d_hat"] = d_hat
                elif pol["name"] == "PD":
                    pol = {"name": "fixed", "rank": {0: 0, 1: 1}}
                res = run_backend(c, merged_blob[r["backend"]], hp0, r["backend"], pol, rel, r["seed"], stop=r.get("stop", False),
                                  record=r.get("record", False))
                res.update(f=r["f"], tau=tau, T_P_used=TP, rank=pol.get("rank"), d_hat=pol.get("d_hat"))
            res["kind"] = r["kind"]
            res["label"] = r.get("label")
            sink["runs"].append(res)
            print(f"[d] {spec['name']} {r.get('label')} out {res['out']} wall {res['wall']:.1f}s", flush=True)
    sink["wall"] = time.perf_counter() - t_start
    sink["peak_gb"] = peak_gb()
    tmp = spec["sink"] + ".part"
    with gzip.open(tmp, "wb") as fh:
        pickle.dump(sink, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, spec["sink"])


# ---------------------------------------------------------------- driver
def launch(name, cmd, spec, folder, limit_gb, log_dir):
    os.makedirs(folder, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    sp = os.path.join(folder, name + ".spec.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    log = open(os.path.join(log_dir, name + ".log"), "w", encoding="utf-8")
    t0 = time.time()
    p = subprocess.Popen([sys.executable, SCRIPT, cmd, sp], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    peak, killed = 0.0, None
    while p.poll() is None:
        g = ws_gb(p.pid)
        peak = max(peak, g)
        if g > limit_gb:
            p.kill()
            killed = f"working set {g:.2f} GB > {limit_gb} GB"
        elif free_gb() < LOCAL_FREE_GB:
            p.kill()
            killed = f"system free memory < {LOCAL_FREE_GB} GB"
        time.sleep(2)
    log.close()
    return {"name": name, "rc": p.returncode, "wall": time.time() - t0, "ws_peak_gb_sampled": peak, "killed": killed}


def run_all(items, workers, limit_gb=None):
    log_dir = os.path.join(OUT, "logs")
    res = []

    def one(it):
        if os.path.exists(it["spec"]["sink"]):
            return {"name": it["name"], "rc": 0, "cached": True}
        r = launch(it["name"], it["cmd"], it["spec"], it["folder"], it.get("limit_gb") or limit_gb or LOCAL_PEAK_GB, log_dir)
        print(f"[d0] {r}", flush=True)
        return r
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, items))
    return res


def creport():
    with open(C_REPORT, encoding="utf-8") as f:
        return {x["fault"]: x for x in json.load(f)["faults"]}


def first_root(job_of):
    for r in ROOTS:
        j = job_of(r)
        if os.path.exists(j["sink"]):
            return j
    raise RuntimeError(f"no sink for {job_of(ROOTS[0])['sink']}")


def median_seed(c, op, fid, tier):
    """CRISP's repair state at an L2 tier: of the 5 SA seeds, the one whose backend mean (C's sink) is the median; ties: smaller seed."""
    E = c.eir
    i0 = c.STEPS.index(tier)
    rows = []
    for k, sd in enumerate(c.SEEDS):
        j = first_root(lambda r: c.jobs_rep(r, op, fid)[i0 * len(c.SEEDS) + k])
        d = E.read_sink(j["sink"])
        rows.append((statistics.mean(v[0] for v in d["record"]["backend"].values()), sd, j["sink"]))
        del d
    m = statistics.median(r[0] for r in rows)
    pick = min((r for r in rows if r[0] == m), key=lambda r: r[1])
    return pick, rows


def export_item(c, op, state, fid, rep, folder):
    """state: a1_pre, a3h_pre (fid None), a1_R0, a3h_R0, a3h_L1, a3h_rep, a1_fullbb."""
    E = c.eir
    name = f"x_{op}_{state}_{fid or 'none'}"
    sink = os.path.join(folder, name + ".pkl.gz")
    start = state.split("_")[0]
    if state.endswith("_pre"):
        j = first_root(lambda r: E.job_pre(os.path.join(r, "prep"), start, op))
        fr = {"type": "pre", "sink": j["sink"], "ref_key": ["T"]}
        spec = {"name": name, "kind": "reload", "start": start, "op": op, "fault": None, "sink": sink, "d_from": fr}
        return {"name": name, "cmd": "export", "spec": spec, "folder": folder}
    if state == "a1_fullbb":
        run = rep[fid]["ops"][op]["a1"]["FULL"]["run"]          # "g0_0"
        g, x = run[1:].split("_")
        i = c.cs.variants("A1").index((int(g), int(x)))
        j = first_root(lambda r: c.jobs_full(r, op, fid, "A1")[i])
        spec = {"name": name, "kind": "full", "op": op, "fault": fid, "sink": sink, "run_sink": j["sink"], "snap": j["snap"], "winner": run}
        return {"name": name, "cmd": "export", "spec": spec, "folder": folder}
    if state == "a1_rep":                          # -A3h: the A1-start L2Gsync of e2e/d_rep.py (levels.json)
        lv = json.load(open(os.path.join(OUT, "a1rep", "levels.json"), encoding="utf-8"))[fid][op]
        lvl = lv["level"]
        if lvl is None:
            raise SystemExit(f"-A3h repair of {op} {fid} not finished")
        if lvl in ("R0", "L1"):
            state = f"a1_{lvl}"
        else:
            rows = []
            for p, sd in zip(lv["sinks"], (123, 1, 2, 3, 4)):
                dd = E.read_sink(p)
                rows.append((statistics.mean(v[0] for v in dd["record"]["backend"].values()), sd, p))
            m = statistics.median(r[0] for r in rows)
            pick = min((r for r in rows if r[0] == m), key=lambda r: r[1])
            fr = {"type": "snap", "sink": pick[2], "key": "final_snap", "ref_key": ["record", "T"], "level": lvl, "seed": pick[1],
                  "seed_rows": [[r[0], r[1]] for r in rows]}
            spec = {"name": name, "kind": "reload", "start": "a1", "op": op, "fault": fid, "sink": sink, "d_from": fr}
            return {"name": name, "cmd": "export", "spec": spec, "folder": folder}
    if state == "a3h_rep":
        lvl = rep[fid]["steps"]["per_op"][op]["level"]
        if lvl in ("R0", "L1"):
            state = f"a3h_{lvl}"
        else:
            pick, rows = median_seed(c, op, fid, int(lvl[1:]))
            fr = {"type": "snap", "sink": pick[2], "key": "final_snap", "ref_key": ["record", "T"], "level": lvl, "seed": pick[1],
                  "seed_rows": [[r[0], r[1]] for r in rows]}
            spec = {"name": name, "kind": "reload", "start": "a3h", "op": op, "fault": fid, "sink": sink, "d_from": fr}
            return {"name": name, "cmd": "export", "spec": spec, "folder": folder}
    key = state.split("_")[1]                                    # R0 / L1
    j = first_root(lambda r: E.job_l1(os.path.join(r, "prep"), start, op, fid))
    fr = {"type": "snap", "sink": j["sink"], "key": f"{key}_snap", "ref_key": [key, "T"], "level": key}
    spec = {"name": name, "kind": "reload", "start": start, "op": op, "fault": fid, "sink": sink, "d_from": fr}
    return {"name": name, "cmd": "export", "spec": spec, "folder": folder}


def d0(workers, only=None):
    import c_eir as c
    c.redirect()
    rep = creport()
    faults = c.eir.load_faults()
    base = os.path.join(OUT, "d0")
    xf = os.path.join(base, "export")
    os.makedirs(xf, exist_ok=True)
    sf = os.path.join(base, "sims")
    # ---- item 1: exports
    items, xs = [], {}
    for op in OPS8:
        for st in ("a1_pre", "a3h_pre"):
            it = export_item(c, op, st, None, rep, xf)
            items.append(it)
            xs[(op, st, None)] = it
        for st in ("a1_R0", "a1_fullbb", "a3h_R0", "a3h_rep"):
            it = export_item(c, op, st, "PF4", rep, xf)
            items.append(it)
            xs[(op, st, "PF4")] = it
    with open(os.path.join(base, "export_plan.json"), "w", encoding="utf-8") as f:
        json.dump([it["spec"] for it in items], f, indent=1)
    if only in (None, "export"):
        r = run_all(items, workers)
        json.dump(r, open(os.path.join(base, "export_runs.json"), "w"), indent=1)
        if only == "export":
            return
    P = lambda op, st, fid=None: xs[(op, st, fid)]["spec"]["sink"]  # noqa: E731
    fdef = lambda fid: ({k: faults[PF[0]][fid][k] for k in ("failed_nodes", "failed_links", "failed_pes")} if fid else None)  # noqa: E731
    # ---- item 2: stats (tags, renaming, chunk counts) per platform
    st_items = []
    for fid, states in ((None, ("a1_pre", "a3h_pre")), ("PF4", ("a1_R0", "a1_fullbb", "a3h_R0", "a3h_rep"))):
        nm = f"stats_{fid or 'none'}"
        spec = {"name": nm, "mode": "stats", "fault_def": fdef(fid), "sink": os.path.join(sf, nm + ".pkl.gz"),
                "exports": {f"{op}:{st}": P(op, st, fid) for op in OPS8 for st in states}}
        st_items.append({"name": nm, "cmd": "sim", "spec": spec, "folder": sf})
    # ---- items 3-7: probes
    D_of = lambda st, fid=None: {op: f"{op}:{st}" for op in DC}  # noqa: E731

    def probe(nm, fid, pst, dst, runs, that=(), chunk=True, limit=None):
        ex = {}
        if pst:
            ex[f"{pst[0]}:{pst[1]}"] = P(pst[0], pst[1], fid)
        if dst:
            for op in DC:
                ex[f"{op}:{dst}"] = P(op, dst, fid)
        spec = {"name": nm, "mode": "probe", "fault_def": fdef(fid), "sink": os.path.join(sf, nm + ".pkl.gz"), "exports": ex,
                "P": f"{pst[0]}:{pst[1]}" if pst else None, "D": D_of(dst) if dst else None, "chunk": chunk, "that": list(that), "runs": runs}
        return {"name": nm, "cmd": "sim", "spec": spec, "folder": sf, "limit_gb": limit}
    solo = lambda s, b, sd=1, pol="P1", rec=False, lab=None: {"kind": "solo", "stream": s, "backend": b, "seed": sd,  # noqa: E731
                                                               "policy": {"name": pol}, "record": rec, "label": lab or f"solo{s}_{b}_{pol}_s{sd}"}
    ov = lambda f, pol, stop, sd=1, rec=True, b="hop": {"kind": "ov", "f": f, "backend": b, "seed": sd, "policy": {"name": pol},  # noqa: E731
                                                        "stop": stop, "record": rec, "label": f"ov_f{f}_{b}_{pol}_{'stop' if stop else 'full'}_s{sd}"}
    pr = [
        # item 4 and 5 on the decode chain (A1 and A3h before the fault): chain solo hop / orig, C4 vs P1
        probe("chain_a1_none", None, None, "a1_pre", [solo(1, "hop", sd) for sd in SEEDS] + [solo(1, "orig", sd) for sd in SEEDS] +
              [solo(1, "hop", 1, "P1", True, "solo1_hop_P1_rec"), solo(1, "hop", 1, "C4", True, "solo1_hop_C4_rec")], that=(1,)),
        probe("chain_a3h_none", None, None, "a3h_pre", [solo(1, "hop", sd) for sd in SEEDS] + [solo(1, "orig", sd) for sd in SEEDS], that=(1,)),
        # item 3 (+5 on pf_proj): pf_ln / pf_proj solo on the per-hop backend, T-hat
        probe("pf_proj_a1_none", None, ("pf_proj", "a1_pre"), None, [solo(0, "hop", 1, "P1", True, "solo0_hop_P1_rec"),
                                                                    solo(0, "hop", 1, "C4", True, "solo0_hop_C4_rec"), solo(0, "orig", 1)], that=(0,)),
        probe("pf_proj_a3h_none", None, ("pf_proj", "a3h_pre"), None, [solo(0, "hop", 1), solo(0, "orig", 1)], that=(0,)),
        probe("pf_ln_a3h_none", None, ("pf_ln", "a3h_pre"), None, [solo(0, "hop", 1), solo(0, "orig", 1)], that=(0,)),
        # item 6: stop mode against the full run (same seed), plus the audit / release checks of item 7
        probe("pf_ln_a1_none_ov", None, ("pf_ln", "a1_pre"), "a1_pre", [solo(0, "hop", 1), solo(0, "orig", 1), ov(0.1, "P1", False), ov(0.1, "P1", True)], that=(0, 1)),
        probe("pf_ffn_a3h_none_ov", None, ("pf_ffn", "a3h_pre"), "a3h_pre", [solo(0, "hop", 1), ov(0.4, "C4", False), ov(0.4, "C4", True)], that=(0, 1)),
        probe("pf_ffn_a3h_PF4_ov", "PF4", ("pf_ffn", "a3h_rep"), "a3h_rep", [solo(0, "hop", 1), ov(0.4, "C4", False), ov(0.4, "C4", True)], that=(0, 1)),
        probe("pf_mha_a3h_none_ov", None, ("pf_mha", "a3h_pre"), "a3h_pre", [solo(0, "hop", 1), ov(0.7, "C4", False), ov(0.7, "C4", True)], that=(0, 1)),
    ]
    # decode operators one by one on the same platform (item 4's sum), A1 and A3h before the fault
    for st in ("a1_pre", "a3h_pre"):
        for op in DC:
            pr.append(probe(f"op_{op}_{st}", None, (op, st), None, [solo(0, "hop", sd) for sd in SEEDS] + [solo(0, "orig", sd) for sd in SEEDS]))
    with open(os.path.join(base, "sim_plan.json"), "w", encoding="utf-8") as f:
        json.dump([it["spec"] for it in st_items + pr], f, indent=1)
    if only in (None, "stats"):
        r = run_all(st_items, workers)
        json.dump(r, open(os.path.join(base, "stats_runs.json"), "w"), indent=1)
    if only in (None, "probe"):
        r = run_all(pr, workers)
        json.dump(r, open(os.path.join(base, "probe_runs.json"), "w"), indent=1)


# ================================================================ S-B: P group (left 2x2) and D group (right 2x2) on a 2x4 platform
# User 2026-11-11 (E2E_HANDOFF.md section 6, "场景改定"). The 2x2 graphs are exported as before (C's states, unchanged) and
# remapped here: die n*2+m of the 2x2 -> n*4+m (left, P) or n*4+m+2 (right, D); core and DDR-port numbers stay. The K and V messages
# of prefill MHA's final broadcast (kconv, vconv) get the D group's DDR ports as targets and new multicast trees (the artifact's
# Dijkstra on the 2x4 platform, in tag order, RNG seeded BCAST_SEED, alpha / beta / gamma of the template); K^T and P stay in the P
# group. Faults: C's coordinates remapped into one group (that group's operators take C's state for the fault, the other group
# its fault-free state), or a link between the groups (both groups fault-free).
SB_MAP = {"L": {0: 0, 1: 1, 2: 4, 3: 5}, "R": {0: 2, 1: 3, 2: 6, 3: 7}}
CFG24 = "config_ch2x4_bw256_co4x4_bw256_t128x64_failpattern0.cfg"
KV_NAMES = ("kconv", "vconv")
# links between the groups (2x4 coordinates; D0 probe): dies 1-2 and 5-6, two per direction each
INTER = {"G1": [[[1, 11], [2, 8]], [[2, 8], [1, 11]]],
         "G2": [[[1, 11], [2, 8]], [[2, 8], [1, 11]], [[1, 7], [2, 4]], [[2, 4], [1, 7]]]}


def cfg_2x4(c):
    src = os.path.join(c.WORK, "models", c.PREFILL, "cfg", CFG_NAME)
    with open(src, encoding="utf-8") as f:
        txt = f.read()
    if txt.count("chiplet_number = 4") != 1 or txt.count("chiplet_columns = 2") != 1:
        raise RuntimeError("2x2 cfg: unexpected chiplet lines")
    new = txt.replace("chiplet_number = 4", "chiplet_number = 8").replace("chiplet_columns = 2", "chiplet_columns = 4")
    dst = os.path.join(OUT, "cfg", CFG24)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst) or open(dst, encoding="utf-8").read() != new:
        with open(dst + ".tmp", "w", encoding="utf-8") as f:
            f.write(new)
        os.replace(dst + ".tmp", dst)
    return dst


def rn(n, mp):
    return (mp[n[0]],) + tuple(n[1:])


def remap_fault(f, mp):
    if not f:
        return {"failed_nodes": [], "failed_links": [], "failed_pes": []}
    return {"failed_nodes": [list(rn(tuple(n), mp)) for n in f.get("failed_nodes", [])],
            "failed_links": [[list(rn(tuple(a), mp)), list(rn(tuple(b), mp))] for a, b in f.get("failed_links", [])],
            "failed_pes": [list(rn(tuple(n), mp)) for n in f.get("failed_pes", [])]}


def build_platform_sb(c, fault24):
    import e4_overlap as E4
    hp = E4.build_platform(cfg_2x4(c), fault24 if fault24 and (fault24["failed_nodes"] or fault24["failed_links"]) else None)
    pes = set(map(tuple, (fault24 or {}).get("failed_pes", [])))
    for d in (hp.tensorcore_dict, hp.vectorunit_dict):
        for loc in list(d):
            if tuple(loc[:-1]) in pes:
                del d[loc]
    return hp


def remap_events(ev, mp):
    for e in ev.values():
        if e.event_type == "computation":
            e.comp_location = rn(e.comp_location, mp)
        else:
            e.source_location = rn(e.source_location, mp)
            T = e.target_location
            e.target_location = {rn(x, mp) for x in T} if isinstance(T, set) else rn(T, mp)
            e.paths = {rn(k, mp): [rn(x, mp) for x in v] for k, v in e.paths.items()}
            e.path_list = [[(rn(a, mp), rn(b, mp)) for a, b in br] for br in e.path_list]
    return ev


def check_group(hp22, hp24, mp):
    """The 2x2 platform of a group, remapped, is exactly the 2x4 platform restricted to the group's dies."""
    dies = set(mp.values())
    bad = 0
    for (a, b), L in hp22.links_dict.items():
        L2 = hp24.links_dict.get((rn(a, mp), rn(b, mp)))
        if L2 is None or (L2.link_type, L2.bandwidth, L2.latency, L2.timeunit) != (L.link_type, L.bandwidth, L.latency, L.timeunit):
            bad += 1
    n24 = sum(1 for (a, b) in hp24.links_dict if a[0] in dies and b[0] in dies)
    m22 = {(t, rn(loc, mp)) for t, ms in hp22.modules_dict.items() for loc in ms}
    m24 = {(t, loc) for t, ms in hp24.modules_dict.items() for loc in ms if loc[0] in dies}
    return {"links_bad": bad, "links_22": len(hp22.links_dict), "links_24_in_group": n24, "modules_equal": m22 == m24}


def check_graph(ev, hp):
    miss_c = sum(1 for e in ev.values() if e.event_type == "computation" and e.comp_location not in hp.modules_dict[e.comp_device])
    miss_l = sum(1 for e in ev.values() if e.event_type == "communication" for br in e.path_list for k in br if k not in hp.links_dict)
    return {"missing_compute": miss_c, "missing_links": miss_l}


def links_of(ev):
    return {k for e in ev.values() if e.event_type == "communication" for br in e.path_list for k in br}


def build_kv(c, ev, hp, ctx):
    """K and V messages of prefill MHA's final broadcast -> the D group's DDR ports, new multicast trees."""
    import numpy as np
    right = set(SB_MAP["R"].values())
    d_ddrs = {d[:-1] for d in hp.ddr_dict if d[0] in right}
    kv = sorted(t for t, e in ev.items() if e.event_type == "communication" and any(k in e.event_name for k in KV_NAMES)
                and isinstance(e.target_location, set) and all(x[1] >= 16 for x in e.target_location))
    b0 = sum(int(ev[t].comm_bytes) for t in kv)
    rng = random.getstate(), np.random.get_state()
    random.seed(c.cs.BCAST_SEED)
    np.random.seed(c.cs.BCAST_SEED)
    for t in kv:
        e = ev[t]
        e.target_location = set(d_ddrs)
        e.paths = hp.record_dijkstra_multicast_path(comm_pairs=[[t, e.source_location, set(d_ddrs), e.comm_bytes]],
                                                    alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])[2][t]
        e.path_list = e.get_paths(e.paths)
    random.setstate(rng[0])
    np.random.set_state(rng[1])
    # checks: every target is a leaf, every branch ends at a target, links exist, inter-group crossings
    leaf_bad = tree_bad = cross = 0
    for t in kv:
        e = ev[t]
        leaves = {br[-1][1] for br in e.path_list if br}
        if leaves != d_ddrs:
            leaf_bad += 1
        if any(k not in hp.links_dict for br in e.path_list for k in br):
            tree_bad += 1
        cross += len({k for br in e.path_list for k in br if hp.links_dict[k].link_type == "ch2ch" and (k[0][0] % 4 < 2) != (k[1][0] % 4 < 2)})
    return kv, {"n_kv": len(kv), "bytes": b0, "bytes_after": sum(int(ev[t].comm_bytes) for t in kv), "targets": len(d_ddrs),
                "leaf_mismatch": leaf_bad, "missing_links": tree_bad, "inter_group_link_uses": cross}


def chain_sb(blobs, sid, mp, reps=1):
    out, prev_sinks, edges = {}, None, 0
    for k, op in enumerate(CHAIN * reps):
        ev = retag(remap_events(pickle.loads(blobs[op]), mp), sid, k)
        sources = [t for t, e in ev.items() if not e.dependency_set]
        sinks = [t for t, e in ev.items() if not e.issue_set]
        if prev_sinks is not None:
            for t in sources:
                ev[t].dependency_set.update(prev_sinks)
            for t in prev_sinks:
                out[t].issue_set.update(sources)
            edges += len(sources) * len(prev_sinks)
        out.update(ev)
        prev_sinks = sinks
    return out, edges


def window(rec, tags):
    v = [rec[t] for t in tags if t in rec]
    return (min(x[0] for x in v), max(x[1] for x in v)) if v else (None, None)


def base_of(t):
    return tuple(t[:-1]) + (t[-1] % 1_000_000,)


def sim_sb(spec_path):
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    c = _sim_setup()
    import dstu2
    t_start = time.perf_counter()
    hp0 = build_platform_sb(c, spec.get("fault24"))
    sink = {"spec": spec, "plat": plat_struct(hp0), "runs": [], "checks": {}}
    ex, hp22 = {}, {}
    for g in ("P", "D"):
        gs = spec["groups"].get(g)
        if not gs:
            continue
        hp22[g] = build_platform(c, gs.get("fault22"))
        ps = plat_struct(hp22[g])
        for name in gs["exports"]:
            d = _load(spec["exports"][name])["d_export"]
            if d["plat"]["sha1"] != ps["sha1"]:
                raise RuntimeError(f"{name}: 2x2 platform structure differs from the exporting process's")
            ex[name] = d["events_blob"]
        sink["checks"][f"group_{g}"] = check_group(hp22[g], hp0, SB_MAP[gs["map"]])
    built, raw, kv_base = {}, {}, set()
    if spec["groups"].get("P"):
        gs = spec["groups"]["P"]
        ev = retag(remap_events(pickle.loads(ex[gs["P"]]), SB_MAP[gs["map"]]), 0)
        if gs.get("kv"):
            kv, st = build_kv(c, ev, hp0, spec["kv_ctx"])
            kv_base = set(kv)
            sink["checks"]["kv"] = st
        built[0] = ev
    if spec["groups"].get("D"):
        gs = spec["groups"]["D"]
        ev, edges = chain_sb({op: ex[gs["D"][op]] for op in DC}, 1, SB_MAP[gs["map"]], spec.get("reps", 1))
        built[1] = ev
        sink["checks"]["chain_edges"] = edges
    for s, ev in built.items():
        sink["checks"][f"graph_{s}"] = check_graph(ev, hp0)
    if 0 in built and 1 in built:
        kvl = links_of({t: e for t, e in built[0].items() if t in kv_base})
        pl = links_of({t: e for t, e in built[0].items() if t not in kv_base})
        dl = links_of(built[1])
        typ = lambda ks: dict(collections.Counter(hp0.links_dict[k].link_type for k in ks))  # noqa: E731
        pm = {(e.comp_device, e.comp_location) for e in built[0].values() if e.event_type == "computation"}
        dm = {(e.comp_device, e.comp_location) for e in built[1].values() if e.event_type == "computation"}
        sink["checks"]["shared"] = {"P_links_and_D_links": typ(pl & dl), "KV_links_and_D_links": typ(kvl & dl),
                                    "KV_links_and_P_links": typ(kvl & pl), "compute_shared": len(pm & dm), "KV_links": typ(kvl)}
    blobs = {}
    for s, ev in built.items():
        raw[s] = pickle.dumps(ev, protocol=pickle.HIGHEST_PROTOCOL)
        t0 = time.perf_counter()
        ch, cst = dstu2.chunk_by(ev, hp0, CHUNK)
        sink["checks"][f"build_{s}"] = {"n_events": len(ev), "n_chunked": len(ch), "chunk": cst, "wall": time.perf_counter() - t0}
        blobs[s] = pickle.dumps(ch, protocol=pickle.HIGHEST_PROTOCOL)
        built[s] = ch
    that = {}
    for s in spec.get("that", []):
        T, w, n = that_hop(built[s], hp0)
        that[s] = T
        sink["checks"][f"that_{s}"] = {"T_hat": T, "wall": w, "n": n}
    del built
    merged = {}
    if len(blobs) == 2:
        for nm_, bl in (("hop", blobs), ("orig", raw)):
            m = pickle.loads(bl[0])
            m.update(pickle.loads(bl[1]))
            merged[nm_] = pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL)
            del m
    kv_win = solo_P = None
    for r in spec["runs"]:
        pol = dict(r.get("policy", {"name": "P1"}))
        rec_needed = r.get("record", False) or bool(kv_base)
        if r["kind"] == "solo":
            s = r["stream"]
            if pol["name"] == "C4":
                pol["rank"] = {s: 0}
            if r.get("on") == "2x2":                        # the same state on its own 2x2 platform (no remap, no KV)
                g = "P" if s == 0 else "D"
                gs = spec["groups"][g]
                ev = retag(pickle.loads(ex[gs["P"]]), 0) if s == 0 else chain({op: ex[gs["D"][op]] for op in DC}, 1)[0]
                if r["backend"] == "hop":
                    ev, _ = dstu2.chunk_by(ev, hp22[g], CHUNK)
                res = run_backend(c, pickle.dumps(ev), hp22[g], r["backend"], pol, {s: 0}, r["seed"], record=True)
            else:
                res = run_backend(c, (raw if r["backend"] == "orig" else blobs)[s], hp0, r["backend"], pol, {s: 0}, r["seed"], record=rec_needed)
        else:
            tau = r.get("tau")
            if tau is None and "f_p" in r:                    # f x this process's first solo P run
                tau = int(round(r["f_p"] * solo_P))
            elif tau is None:                                 # relative to the KV window of a solo run of this process
                lo, hi = kv_win
                tau = int(round(lo + r["f_kv"] * (hi - lo)))
            rel = {0: 0, 1: tau}
            if pol["name"] == "C4":
                d_hat = {0: K * that[0], 1: tau + K * that[1]}
                order = sorted(d_hat, key=lambda i: (d_hat[i], i))
                pol["rank"] = {i: order.index(i) for i in d_hat}
                pol["d_hat"] = d_hat
            elif pol["name"] == "PD":
                pol = {"name": "fixed", "rank": {0: 0, 1: 1}}
            res = run_backend(c, merged[r["backend"]], hp0, r["backend"], pol, rel, r["seed"], record=rec_needed)
            res.update(tau=tau, rank=pol.get("rank"), d_hat=pol.get("d_hat"))
        if kv_base and "record" in res:
            kt = [t for t in res["record"] if t[1] == 10 and base_of(t) in kv_base]
            res["kv_window"] = window(res["record"], kt)
            if r["kind"] == "solo" and r["stream"] == 0 and r.get("on") != "2x2" and kv_win is None:
                kv_win = res["kv_window"]
            if r.get("by_op") and 1 in res["streams"]:
                by = collections.defaultdict(list)
                for t, v in res["record"].items():
                    if t[1] == 11:
                        by[t[2]].append(v)
                res["d_ops"] = {k: (min(x[0] for x in v), max(x[1] for x in v)) for k, v in sorted(by.items())}
            if not r.get("record"):
                res.pop("record", None)
                res.pop("links", None)
        elif r.get("by_op") and "record" in res:
            by = collections.defaultdict(list)
            for t, v in res["record"].items():
                if t[1] == 11:
                    by[t[2]].append(v)
            res["d_ops"] = {k: (min(x[0] for x in v), max(x[1] for x in v)) for k, v in sorted(by.items())}
        if r["kind"] == "solo" and r["stream"] == 0 and r.get("on") != "2x2" and solo_P is None:
            solo_P = res["out"][0]
        res["kind"], res["label"], res["on"] = r["kind"], r.get("label"), r.get("on", "2x4")
        sink["runs"].append(res)
        print(f"[sb] {spec['name']} {r.get('label')} out {res['out']} wall {res['wall']:.1f}s kv {res.get('kv_window')}", flush=True)
    sink["wall"] = time.perf_counter() - t_start
    sink["peak_gb"] = peak_gb()
    tmp = spec["sink"] + ".part"
    with gzip.open(tmp, "wb") as fh:
        pickle.dump(sink, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, spec["sink"])


def kv_ctx(c):
    """alpha / beta / gamma of prefill MHA's template (cpcm_sa saved the template's broadcast arguments with every run)."""
    j = first_root(lambda r: c.jobs_full(r, "pf_mha", "PF4", "A3h")[0])
    ctx = c.eir.read_sink(j["sink"])["meta"]["ctx"]
    return {k: ctx[k] for k in ("alpha", "beta", "gamma", "dijkstra_routing")}


def sb_smoke(workers):
    import c_eir as c
    c.redirect()
    ctx = kv_ctx(c)
    faults = c.eir.load_faults()
    f22 = lambda fid: ({k: faults["pf_ln"][fid][k] for k in ("failed_nodes", "failed_links", "failed_pes")} if fid else None)  # noqa: E731
    X = lambda op, st, fid=None: os.path.join(OUT, "d0", "export", f"x_{op}_{st}_{fid or 'none'}.pkl.gz")  # noqa: E731
    base = os.path.join(OUT, "sb_smoke")
    sf = os.path.join(base, "sims")

    def spec(nm, P=None, D=None, fault24=None, reps=1, runs=(), that=(0, 1)):
        ex, groups = {}, {}
        if P:
            op, st, fid, kv = P
            ex[f"{op}:{st}"] = X(op, st, fid)
            groups["P"] = {"map": "L", "fault22": f22(fid), "exports": [f"{op}:{st}"], "P": f"{op}:{st}", "kv": kv}
        if D:
            st, fid = D
            for o in DC:
                ex[f"{o}:{st}"] = X(o, st, fid)
            groups["D"] = {"map": "R", "fault22": f22(fid), "exports": [f"{o}:{st}" for o in DC], "D": {o: f"{o}:{st}" for o in DC}}
        sp = {"name": nm, "sink": os.path.join(sf, nm + ".pkl.gz"), "exports": ex, "groups": groups, "fault24": fault24, "reps": reps,
              "kv_ctx": ctx, "that": [t for t in that if (t == 0 and P) or (t == 1 and D)], "runs": list(runs)}
        return {"name": nm, "cmd": "simsb", "spec": sp, "folder": sf}
    so = lambda s, b="hop", sd=1, on="2x4", pol="P1", by=False, rec=False: {"kind": "solo", "stream": s, "backend": b, "seed": sd, "on": on,  # noqa: E731
                                                                          "policy": {"name": pol}, "by_op": by, "record": rec,
                                                                          "label": f"solo{s}_{b}_{on}_{pol}_s{sd}"}
    ov = lambda pol="C4", sd=1, b="hop", **kw: dict({"kind": "ov", "backend": b, "seed": sd, "policy": {"name": pol}, "by_op": True,  # noqa: E731
                                                     "label": f"ov_{b}_{pol}_s{sd}_" + "_".join(f"{k}{v}" for k, v in kw.items())}, **kw)
    fl = lambda fid, g: remap_fault(f22(fid), SB_MAP[g])  # noqa: E731
    inter = lambda g: {"failed_nodes": [], "failed_links": INTER[g], "failed_pes": []}  # noqa: E731
    items = [
        # 2x4 against 2x2, same state, no KV retargeting (hop and original backend, seed 1, same process)
        spec("sb_nokv_mha_a3h", P=("pf_mha", "a3h_pre", None, False), D=("a3h_pre", None),
             runs=[so(0, rec=True), so(0, on="2x2"), so(0, "orig", rec=True), so(0, "orig", on="2x2"),
                   so(1, rec=True), so(1, on="2x2"), so(1, "orig", rec=True), so(1, "orig", on="2x2")]),
        # KV: solo P with KV, solo D, overlaps across the KV window (C4 and P1)
        spec("sb_mha_a3h_none", P=("pf_mha", "a3h_pre", None, True), D=("a3h_pre", None),
             runs=[so(0), so(1, by=True), so(1, sd=2), so(1, sd=3), so(0, "orig")] + [ov(f_kv=f) for f in FS] + [ov("P1", f_kv=0.4)]),
        # linearity: 8 decode steps in a row against 1 (same tau rule)
        spec("sb_mha_a3h_none_lin8", P=("pf_mha", "a3h_pre", None, True), D=("a3h_pre", None), reps=8,
             runs=[so(0), so(1, by=True)] + [ov(f_kv=f) for f in (0.0, 0.4)] + [ov("P1", f_kv=0.0)]),
        # faults: C's PF4 in the P group / in the D group; links between the groups
        spec("sb_mha_PF4_P", P=("pf_mha", "a3h_rep", "PF4", True), D=("a3h_pre", None), fault24=fl("PF4", "L"),
             runs=[so(0), so(1, by=True), ov(f_kv=0.4)]),
        spec("sb_mha_PF4_D", P=("pf_mha", "a3h_pre", None, True), D=("a3h_rep", "PF4"), fault24=fl("PF4", "R"),
             runs=[so(0), so(1, by=True), ov(f_kv=0.4)]),
        spec("sb_mha_G1", P=("pf_mha", "a3h_pre", None, True), D=("a3h_pre", None), fault24=inter("G1"),
             runs=[so(0), so(1, by=True), ov(f_kv=0.4)]),
        spec("sb_mha_G2", P=("pf_mha", "a3h_pre", None, True), D=("a3h_pre", None), fault24=inter("G2"),
             runs=[so(0), so(1, by=True), ov(f_kv=0.4)]),
        # a non-MHA prefill operator: no KV, so P and D share no resource (checked); one overlap as evidence
        spec("sb_ffn_a3h_none", P=("pf_ffn", "a3h_pre", None, False), D=("a3h_pre", None),
             runs=[so(0), so(1), ov(f_p=0.4)]),
    ]
    # common reference: healthy W1 baseline = A1 before the fault, original backend, seeds 1-3, on the S-B platform (P with KV)
    for op in PF:
        items.append(spec(f"sb_ref_{op}", P=(op, "a1_pre", None, op == "pf_mha"), D=("a1_pre", None) if op == "pf_ln" else None,
                          runs=[so(0, "orig", sd) for sd in SEEDS] + ([so(1, "orig", sd) for sd in SEEDS] if op == "pf_ln" else []), that=()))
    with open(os.path.join(base, "plan.json") if os.path.isdir(base) else _mk(base, "plan.json"), "w", encoding="utf-8") as f:
        json.dump([it["spec"] for it in items], f, indent=1)
    r = run_all(items, workers)
    json.dump(r, open(os.path.join(base, "runs.json"), "w"), indent=1)


def _mk(d, name):
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def _load(p):
    with gzip.open(p, "rb") as fh:
        return pickle.load(fh)


def d0_report():
    """D0's seven checks from the sinks (numbers only) -> e2e/d/out/d0/d0_report.json."""
    base = os.path.join(OUT, "d0")
    out = {"main": MAIN, "out": OUT}
    # 1. exports
    plan = json.load(open(os.path.join(base, "export_plan.json"), encoding="utf-8"))
    ex, plat = [], collections.defaultdict(set)
    for s in plan:
        if not os.path.exists(s["sink"]):
            ex.append({"name": s["name"], "missing": True})
            continue
        d = _load(s["sink"])
        x = d["d_export"]
        w1 = d.get("w1") or {}
        ex.append({"name": s["name"], "T": x["T"], "T_ref": x["T_ref"], "T_equal": x["T_equal"], "n_events": x["n_events"],
                   "w1": w1.get("drops", w1), "plat": x["plat"]["sha1"][:10], "wall": round(x["wall"], 2), "peak_gb": round(x["peak_gb"], 3),
                   "from": {k: v for k, v in s.get("d_from", {}).items() if k in ("level", "seed", "seed_rows")} or {"winner": s.get("winner")}})
        plat[s.get("fault") or "none"].add(x["plat"]["sha1"])
    out["1_exports"] = {"n": len(ex), "missing": sum(1 for e in ex if e.get("missing")), "T_equal": sum(1 for e in ex if e.get("T_equal")),
                        "platforms_per_fault": {k: len(v) for k, v in plat.items()}, "rows": ex}
    sims = {s["name"]: s for s in json.load(open(os.path.join(base, "sim_plan.json"), encoding="utf-8"))}
    S = {n: _load(s["sink"]) for n, s in sims.items() if os.path.exists(s["sink"])}
    out["missing_sims"] = sorted(set(sims) - set(S))
    # 2. stats
    st = {}
    for n in ("stats_none", "stats_PF4"):
        if n in S:
            st[n] = {k: ({"retag_error": v["retag_error"]} if "retag_error" in v else
                         {"n_events": v["n_events"], "n_chunked": v["chunk"]["n_events"], "blocks_added": v["chunk"]["blocks_added"],
                          "bytes_ok": v["chunk"]["bytes_conserved"], "tree_ok": v["chunk"]["tree_conserved"], "chunk_wall": round(v["chunk_wall"], 1),
                          "tag1": v["tag_facts"]["tag1"], "tag2": v["tag_facts"]["tag2"], "owner_ok": v["tag_facts"]["comm_owner_is_comp"]})
                     for k, v in S[n]["stats"].items()}
    out["2_stats"] = st
    # 3. timing and memory per probe
    tm = {}
    for n, d in S.items():
        if d["spec"]["mode"] == "stats":
            tm[n] = {"wall": round(d["wall"], 1), "peak_gb": round(d["peak_gb"], 3)}
            continue
        tm[n] = {"wall": round(d["wall"], 1), "peak_gb": round(d["peak_gb"], 3),
                 "build": {k: {"n": v["n_events"], "n_chunked": v["n_chunked"], "wall": round(v["wall"], 1)} for k, v in d["checks"].items() if k.startswith("build")},
                 "that": {k: {"T_hat": v["T_hat"], "wall": round(v["wall"], 1)} for k, v in d["checks"].items() if k.startswith("that")},
                 "runs": [{"label": r["label"], "out": r["out"], "wall": round(r["wall"], 1), "t_stop": r.get("t_stop"),
                           "C": {s: v["C"] for s, v in r["streams"].items()}, "n_done": {s: v["n_done"] for s, v in r["streams"].items()},
                           "n": {s: v["n"] for s, v in r["streams"].items()}} for r in d["runs"]]}
    out["3_timing"] = tm
    # 4. chain against the sum of its operators
    ch = {}
    W = collections.Counter(CHAIN)
    for st_ in ("a1_pre", "a3h_pre"):
        cn = f"chain_{st_.split('_')[0]}_none"
        if cn not in S:
            continue
        row = {}
        for b in ("hop", "orig"):
            chain_m = statistics.mean(r["out"][0] for r in S[cn]["runs"] if r["kind"] == "solo" and r["backend"] == b and r["label"].endswith(f"_s{r['seed']}") and "_rec" not in r["label"])
            ops = {}
            for op in DC:
                n = f"op_{op}_{st_}"
                if n in S:
                    ops[op] = statistics.mean(r["out"][0] for r in S[n]["runs"] if r["backend"] == b)
            tot = sum(W[op] * v for op, v in ops.items()) if len(ops) == 4 else None
            row[b] = {"chain_mean": chain_m, "ops_mean": ops, "sum_2ln_8mha_proj_ffn": tot, "chain_over_sum": chain_m / tot - 1 if tot else None}
        rec = next((r for r in S[cn]["runs"] if r["label"] == "solo1_hop_P1_rec"), None)
        if rec:
            by = collections.defaultdict(list)
            for t, (s0, e0) in rec["record"].items():
                by[t[2]].append((s0, e0))
            viol = sum(1 for k in range(1, len(CHAIN)) if min(x[0] for x in by[k]) < max(x[1] for x in by[k - 1]))
            row["order_violations"] = viol
        row["chain_edges"] = S[cn]["checks"].get("chain_edges")
        ch[cn] = row
    out["4_chain"] = ch
    # 5. one stream: C4 (Fixed rank) against P1, bit for bit
    eq = {}
    for n, a, b in (("chain_a1_none", "solo1_hop_P1_rec", "solo1_hop_C4_rec"), ("pf_proj_a1_none", "solo0_hop_P1_rec", "solo0_hop_C4_rec")):
        if n in S:
            ra = next(r for r in S[n]["runs"] if r["label"] == a)
            rb = next(r for r in S[n]["runs"] if r["label"] == b)
            eq[n] = {"out_equal": ra["out"] == rb["out"], "events_equal": ra["record"] == rb["record"], "links_equal": ra["links"] == rb["links"],
                     "n_events": len(ra["record"])}
    out["5_c4_vs_p1_solo"] = eq
    # 6. stop mode against the full run
    sm = {}
    for n, d in S.items():
        fulls = [r for r in d.get("runs", []) if r["kind"] == "ov" and not r["stop"]]
        for rf in fulls:
            rs = next((r for r in d["runs"] if r["kind"] == "ov" and r["stop"] and r["f"] == rf["f"] and r["seed"] == rf["seed"] and r["policy"] == rf["policy"]), None)
            if rs is None:
                continue
            ts = rs["t_stop"]
            fin_s = {t: v for t, v in rs["record"].items() if v[1] and v[1] <= ts}
            fin_f = {t: v for t, v in rf["record"].items() if v[1] and v[1] <= ts}
            lk_s = {k: [x for x in v if x[0] <= ts] for k, v in rs["links"].items()}
            lk_f = {k: [x for x in v if x[0] <= ts] for k, v in rf["links"].items()}
            lk_s = {k: v for k, v in lk_s.items() if v}
            lk_f = {k: v for k, v in lk_f.items() if v}
            sm[n] = {"t_stop": ts, "C_D_full": rf["streams"][1]["C"], "C_D_stop": rs["streams"][1]["C"], "C_P_full": rf["streams"][0]["C"],
                     "events_done_by_t_stop": len(fin_f), "events_equal": fin_s == fin_f, "links_equal": lk_s == lk_f,
                     "wall_full": round(rf["wall"], 1), "wall_stop": round(rs["wall"], 1), "stop_over_full": rs["wall"] / rf["wall"],
                     "P_done_share_at_stop": sum(1 for t, v in rs["record"].items() if t[1] == 10 and v[1] and v[1] <= ts) / rs["streams"][0]["n"]}
    out["6_stop_mode"] = sm
    # 7. audit, release, chunk conservation
    au = {"issues": 0, "bad": 0, "release_violations": 0, "chunk_bytes_bad": 0, "chunk_tree_bad": 0}
    for n, d in S.items():
        for k, v in d.get("checks", {}).items():
            if k.startswith("build") and v["chunk"]:
                au["chunk_bytes_bad"] += 0 if v["chunk"]["bytes_conserved"] else 1
                au["chunk_tree_bad"] += 0 if v["chunk"]["tree_conserved"] else 1
        for r in d.get("runs", []):
            if "audit" in r:
                au["issues"] += r["audit"]["n"]
                au["bad"] += r["audit"]["bad"]
            for s, v in r["streams"].items():
                if v["first_start"] is not None and v["first_start"] < r["release"].get(s, 0):
                    au["release_violations"] += 1
    out["7_audit"] = au
    with open(os.path.join(base, "d0_report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only")
    ap.add_argument("--peak-gb", type=float)
    ap.add_argument("--free-gb", type=float)
    ap.add_argument("--out", help="output root (default e2e/d/out or D_OUT)")
    a = ap.parse_args()
    global LOCAL_PEAK_GB, LOCAL_FREE_GB, OUT
    if a.peak_gb:
        LOCAL_PEAK_GB = a.peak_gb
    if a.free_gb:
        LOCAL_FREE_GB = a.free_gb
    if a.out:
        OUT = os.path.abspath(a.out)
    if a.cmd == "export":
        with open(a.arg, encoding="utf-8") as f:
            kind = json.load(f)["kind"]
        return export_full(a.arg) if kind == "full" else export_eir(a.arg)
    if a.cmd == "sim":
        return sim(a.arg)
    if a.cmd == "simsb":
        return sim_sb(a.arg)
    if a.cmd == "sbsmoke":
        return sb_smoke(a.workers)
    if a.cmd == "d0":
        return d0(a.workers, a.only)
    if a.cmd == "d0report":
        r = d0_report()
        print(json.dumps({k: v for k, v in r.items() if k not in ("1_exports", "3_timing")}, indent=1, default=str))
        return
    raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
