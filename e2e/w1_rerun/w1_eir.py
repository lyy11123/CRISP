"""W1 rerun of EIR's MHA runs (E2E_HANDOFF.md section 4, user release 2026-11-08; inventory W1_RERUN_INVENTORY.md section 1).

Same jobs, rules, seeds, stop rule and evaluation as EIR (eir.py / eir_next.py, BusyBarn backend with the address hash, 10 backend
seeds); the only difference is the workload W1: the final broadcast without the messages of QK^T and S. Nothing in eir/ is changed
or written: eir.py and eir_next.py are imported and redirected in memory to this folder's inputs, fault table, artifact work copy
and output root (--out); the children are this script ("child"), which redirects and installs W1 before calling eir.child.

W1 in memory:
  - every job except `full`: cpcm_sa.BcCache.graph (the broadcast EIR's T and final evaluation see) builds W0's broadcast with W0's
    routes and RNG use, then drops QK^T and S (the B1 / B4 filter). Engine.seed_broadcast keeps the W0 add_broadcast: it pairs the
    template's broadcast with the producers' splits, and its T_wrapped is a W0 diagnostic only.
  - `full` (cpcm_sa's search from scratch, FULL-A3h and FULL-BusyBarn): as B4 (e2e_pilot.install_w1), the template's add_broadcast
    and BcCache.graph both drop.
  - every drop is checked: MHA 72 messages, FFN 0; anything else stops the process.
Starts (same source as EIR): a3h_mha = B4's backend-best W1 A3h run dc_mha_w1_A3h_g0_s2 (selected state), e3_mha = E3's start
  (unchanged; the A1 trajectory does not depend on the final broadcast). Fault coordinates: eir/faults.json unchanged (user: same
  coordinates; the report `faults` says whether they are still what the rules pick on the W1 start).
FULL-BusyBarn: user 2026-11-08, rerun as the original process (E3's FULL): the 14 A1 variants on each of the 7 faulty platforms.

Shards (one Colab notebook each; prep first, the other four at the same time; every job one process, finished sinks are skipped):
  prep          pre (a3h, e3, a3h with waits), l1 (16), select (F4, F5, F8 rules on the W1 A3h start) and 2 reference jobs
  eir_main_a3h  main repairs from the A3h start (F4, F5, F4a, F5a: 180) and its control (15)
  eir_main_e3   main repairs from the E3 start (7 faults: 240) and its control (15)
  eir_full      FULL-A3h (main F4, F5: 28; next F4a, F8a, C x 4: 84) and FULL-BusyBarn (7 faults x 14 A1 variants: 98)
  eir_next      next repairs (F8a old methods 10, new methods 45, extra seeds 30, C 120) and phase B checkpoints (30)

  python e2e/w1_rerun/w1_eir.py setup                       local only: work copy, eir_inputs/, eir_faults.json
  python e2e/w1_rerun/w1_eir.py probe --out OUT             pre of the W1 A3h start in OUT/_probe, compared with local_ref.json
  python e2e/w1_rerun/w1_eir.py smoke --out OUT             every job kind once, short, in OUT/_smoke (used by the bundle check)
  python e2e/w1_rerun/w1_eir.py run <shard> --out OUT --workers 7
  python e2e/w1_rerun/w1_eir.py localref --out OUT          local only, after `run prep`: writes local_ref.json
  python e2e/w1_rerun/w1_eir.py compare --out OUT           prep sinks and the 2 references vs local_ref.json
  python e2e/w1_rerun/w1_eir.py faults --out OUT            fault-coordinate report (after prep)
  python e2e/w1_rerun/w1_eir.py report <shard> --out OUT    failures, walls, memory peaks
  python e2e/w1_rerun/w1_eir.py pack <shard> --out OUT      OUT/results_w1_<shard>.tar.gz (sinks, logs, mem; no _tmp)
  python e2e/w1_rerun/w1_eir.py ffncheck                    local self-check: the W1 filter drops 0 messages on the FFN starts
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import pickle
import shutil
import statistics
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.dirname(HERE)
ROOT = os.path.dirname(E2E)
os.environ.setdefault("CPCM_SA_STAGE", "s2")
sys.path.insert(0, os.path.join(ROOT, "eir"))
import eir  # noqa: E402
import eir_next as EN  # noqa: E402

cs = eir.cs
WORK = os.path.join(HERE, "work", "artifact")
INPUTS = os.path.join(HERE, "eir_inputs")
FAULTS = os.path.join(HERE, "eir_faults.json")
LOCAL_REF = os.path.join(HERE, "local_ref.json")
OP = "mha"
A3H_W1 = ("dc_mha_w1", "A3h", 0, 2)          # backend best of B4's 14 W1 A3h runs (11,737.8 cycles)
W1_DROP = ("qkt", "s_softmax")
EXPECT = {"mha": 72, "ffn": 0, "ffn14": 0}
ALERT_S = 3600                                # single process over 1 h: stop and report
SHARDS = ["prep", "eir_main_a3h", "eir_main_e3", "eir_full", "eir_next"]
REF_JOBS = ("a3h_mha_F4_L2G_s30_r123", "full_mha_F1_A1_g0_4")


def redirect():
    eir.WORK = cs.WORK = cs.e3.WORK = WORK
    eir.INPUTS, eir.FAULTS_JSON = INPUTS, FAULTS
    eir.__file__ = os.path.abspath(__file__)      # eir.run_job launches `<this file> child <spec>`


# ---------------------------------------------------------------- W1 filter (B1's filter, e2e_pilot.w1_drop, copied: importing
# e2e_pilot would reconfigure cpcm_sa for the e2e work copy)
def w1_drop(events, added):
    drop = {t for t in added if any(k in events[t].event_name for k in W1_DROP)}
    for t in drop:
        for dep in events[t].dependency_set:
            if events.get(dep) is not None:
                events[dep].issue_set.discard(t)
        del events[t]
    tags = set(events)
    bad = sum(1 for t in added - drop for x in events[t].dependency_set if x not in tags) + sum(1 for t in added - drop for x in events[t].issue_set if x not in tags)
    bad += sum(1 for e in events.values() for x in e.issue_set if x in drop)
    if bad:
        raise RuntimeError(f"W1 filter left {bad} dangling references")
    return drop


DROPS = collections.Counter()


def install_w1(kind, op):
    want = EXPECT[op]

    def check(n, where):
        DROPS[(where, n)] += 1
        if n != want:
            raise RuntimeError(f"W1 filter dropped {n} messages in {where}, expected {want} ({op})")
    graph0 = cs.BcCache.graph

    def graph(self, beha, data, ev0, hp):
        ev = graph0(self, beha, data, ev0, hp)
        check(len(w1_drop(ev, set(ev) - set(ev0))), "graph")
        return ev
    cs.BcCache.graph = graph
    if kind == "full":
        cs._artifact_path()
        import add_communication
        orig = add_communication.add_broadcast

        def add_broadcast(*a, **kw):
            ev = kw["event_dict"]
            before = set(ev)
            out = orig(*a, **kw)
            check(len(w1_drop(ev, set(ev) - before)), "add_broadcast")
            return out
        add_communication.add_broadcast = add_broadcast
    # EIR sinks carry the drop counts (cpcm_sa's own sink, `full`, does not: its log has them)
    ws = eir.write_sink

    def write_sink(path, d):
        d["w1"] = {"drops": {f"{w}:{n}": c for (w, n), c in DROPS.items()}, "expect": want}
        return ws(path, d)
    eir.write_sink = write_sink


def mem_logger(name):
    path = os.environ.get("W1_MEM_LOG")
    if not path:
        return
    import atexit

    def log():
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

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
            peak = pmc.PeakWorkingSetSize
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"name": name, "peak_gb": peak / 2 ** 30, "drops": {f"{w}:{n}": c for (w, n), c in DROPS.items()}}) + "\n")
    atexit.register(log)


def child(spec_path):
    redirect()
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    mem_logger(spec["name"])
    install_w1(spec["kind"], spec["op"])
    cs._artifact_path()
    import Stream_Mapping
    if not os.path.abspath(Stream_Mapping.__file__).startswith(WORK):
        raise RuntimeError(f"artifact imported from {Stream_Mapping.__file__}, not from {WORK}")
    print(f"[w1] child {spec['name']} ({spec['kind']}, {spec['op']}), expect {EXPECT[spec['op']]} dropped per broadcast", flush=True)
    try:
        return eir.child(spec)
    finally:
        print(f"[w1] drops {dict(DROPS)}", flush=True)


# ---------------------------------------------------------------- jobs
def dirs(out):
    M, N = os.path.join(out, "main"), os.path.join(out, "next")
    EN.OUT, EN.MAIN, EN.B_OUT = N, M, os.path.join(N, "b")
    return M, N, EN.B_OUT


def prep_jobs(out):
    M, N, _ = dirs(out)
    pre = [eir.job_pre(M, "a3h", OP), eir.job_pre(M, "e3", OP), dict(EN.job_pre_w(OP), name="prew_a3h_mha")]   # unique name: logs and spec files go by name
    l1 = [eir.job_l1(M, "a3h", OP, f) for f in ("F4", "F5", "F4a", "F5a")] + [eir.job_l1(M, "e3", OP, f) for f in eir.FIDS] + \
         [EN.job_l1(OP, f, out=N) for f in ("F8a",) + EN.C_FAULTS]
    sel = [dict(eir.job_select(M, OP), fids=["F4", "F5", "F8"])]
    return pre, l1, sel


def shard_jobs(shard, out):
    M, N, B = dirs(out)
    S = eir.SA_SEEDS
    if shard == "prep":
        pre, l1, sel = prep_jobs(out)
        refs = [j for p in shard_jobs("eir_main_a3h", out) + shard_jobs("eir_full", out) for j in p if j["name"] in REF_JOBS]
        return [pre, l1 + sel, refs]
    if shard == "eir_main_a3h":
        return [[eir.job_rep(M, "a3h", OP, f, m, st, sd) for f in ("F4", "F5", "F4a", "F5a") for m in eir.METHODS for st in eir.steps_of(f) for sd in S]
                + [eir.job_ctrl(M, "a3h", OP, st, sd) for st in eir.CTRL_STEPS for sd in S]]
    if shard == "eir_main_e3":
        return [[eir.job_rep(M, "e3", OP, f, m, st, sd) for f in eir.FIDS for m in eir.METHODS for st in eir.steps_of(f) for sd in S]
                + [eir.job_ctrl(M, "e3", OP, st, sd) for st in eir.CTRL_STEPS for sd in S]]
    if shard == "eir_full":
        return [[eir.job_full(M, OP, f, "A3h", g, x) for f in ("F4", "F5") for g, x in cs.variants("A3h")]
                + [eir.job_full(N, OP, f, "A3h", g, x) for f in ("F4a", "F8a") + EN.C_FAULTS for g, x in cs.variants("A3h")]
                + [eir.job_full(M, OP, f, "A1", g, x) for f in eir.FIDS for g, x in cs.variants("A1")]]
    if shard == "eir_next":
        jr = lambda f, m, sd, **k: EN.job_rep(OP, f, m, sd, out=N, **k)
        jobs = [jr("F8a", m, sd) for m in EN.OLD_METHODS for sd in S]
        jobs += [jr(f, m, sd) for f in EN.FAULTS for m in EN.NEW_METHODS for sd in S]
        jobs += [jr("F4a", m, sd) for m in ("L2Impact", "L2Lsync") for sd in EN.EXTRA_SEEDS]
        jobs += [jr(f, m, sd) for f in EN.C_FAULTS for m in EN.C_METHODS for sd in EN.C_SEEDS[OP]]
        jobs += [dict(EN.job_rep(OP, EN.B_FAULT, m, sd, out=B, steps=300), checkpoints=list(EN.B_CK), ck_backend=True) for m in EN.B_METHODS for sd in EN.B_SEEDS]
        return [jobs]
    raise SystemExit(f"unknown shard {shard}")


def guard(phases, out):
    """No job may read or write outside OUT (eir/out is never touched)."""
    o = os.path.abspath(out)
    names = [j["name"] for jobs in phases for j in jobs]
    dup = [n for n, c in collections.Counter(names).items() if c > 1]
    if dup:
        raise SystemExit(f"job names not unique (logs and specs go by name): {dup[:5]}")
    for jobs in phases:
        for j in jobs:
            for k in ("sink", "snap", "pre_sink", "l1_sink", "pre_w_sink"):
                if j.get(k) and not os.path.abspath(j[k]).startswith(o):
                    raise SystemExit(f"{j['name']}: {k} outside {o}: {j[k]}")
            j.setdefault("op", OP)


def prep_done(out):
    pre, l1, sel = prep_jobs(out)
    return [j["name"] for j in pre + l1 + sel if not eir.sink_ok(j["sink"], j)]


def run(shard, out, workers):
    out = os.path.abspath(out)
    phases = shard_jobs(shard, out)
    guard(phases, out)
    if shard != "prep":
        miss = prep_done(out)
        if miss:
            raise SystemExit(f"prep is not finished in {out} ({len(miss)} missing, e.g. {miss[:3]}): run the prep shard first")
    os.makedirs(os.path.join(out, "mem"), exist_ok=True)
    os.environ["W1_MEM_LOG"] = os.path.join(out, "mem", f"{shard}.jsonl")
    runs_log = os.path.join(out, f"runs_{shard}.jsonl")
    alert = []
    t_all = time.time()
    for i, jobs in enumerate(phases):
        t0 = time.time()
        live = {}

        def one(j):
            live[j["name"]] = time.time()
            try:
                r = eir.run_job(j, out)
            finally:
                live.pop(j["name"], None)
            with open(runs_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"name": r[0], "status": r[1], "wall_s": r[2], "workers": workers, "t": time.strftime("%H:%M:%S")}) + "\n")
            if r[2] > ALERT_S:
                alert.append(r[0])
            return r
        stop = threading.Event()

        def watch():
            while not stop.wait(60):
                over = [n for n, t in list(live.items()) if time.time() - t > ALERT_S]
                if over:
                    print(f"[w1] ALERT: over 1 h single process: {over}", flush=True)
        threading.Thread(target=watch, daemon=True).start()
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(one, jobs))
        stop.set()
        bad = [r[0] for r in res if r[1] == "failed"]
        print(f"[w1] {shard} phase {i}: {len(res)} jobs, {sum(r[1] == 'cached' for r in res)} cached, {len(bad)} failed, {time.time() - t0:.0f} s", flush=True)
        if bad:
            print(f"[w1] FAILED {bad}; stopping (rerun to retry)", flush=True)
            raise SystemExit(1)
        if alert:
            print(f"[w1] STOP: single process over 1 h: {alert}", flush=True)
            raise SystemExit(1)
    print(f"[w1] {shard}: all ok, {(time.time() - t_all) / 3600:.2f} h", flush=True)


# ---------------------------------------------------------------- setup (local)
def setup():
    if not os.path.isdir(WORK):
        src = os.path.join(ROOT, "eir", "work", "artifact")
        shutil.copytree(src, WORK, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "results"))
        os.makedirs(os.path.join(WORK, "models", eir.GPT, "results"), exist_ok=True)
        print(f"work copy: {WORK} (from {src}, without results)")
    os.makedirs(INPUTS, exist_ok=True)
    for n in ("e3_mha", "e3_ffn", "a3h_ffn"):        # FFN starts: only for ffncheck
        shutil.copy(os.path.join(ROOT, "eir", "inputs", n + ".pkl.gz"), os.path.join(INPUTS, n + ".pkl.gz"))
    cfg, arm, g, x = A3H_W1
    d = cs.load_run(cfg, arm, g, x, os.path.join(E2E, "out", "b4"))
    with gzip.open(os.path.join(INPUTS, "a3h_mha.pkl.gz"), "wb") as f:
        pickle.dump(d["selected"]["e3map"], f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copy(os.path.join(ROOT, "eir", "faults.json"), FAULTS)
    bk = statistics.mean(v[0] for v in d["selected"]["backend"].values())
    print(f"inputs: a3h_mha from {cs.run_name(cfg, arm, g, x)} (backend mean {bk:.1f}); e3_mha, e3_ffn, a3h_ffn copied; faults.json copied")


def ffncheck():
    """Self-check (inventory 0.1): the W1 filter drops nothing on the FFN starts, so the FFN results stand. pre of e3_ffn and a3h_ffn
    under W1 into a scratch folder; the drop check inside raises if a message is dropped."""
    out = os.path.join(HERE, "out", "_ffncheck")
    jobs = [dict(eir.job_pre(out, s, "ffn"), op="ffn") for s in ("e3", "a3h")]
    os.environ["W1_MEM_LOG"] = os.path.join(out, "mem.jsonl")
    res = [eir.run_job(j, out) for j in jobs]
    rows = []
    for j, r in zip(jobs, res):
        d = eir.read_sink(j["sink"]) if r[1] != "failed" else {}
        w0 = eir.read_sink(os.path.join(ROOT, "eir", "out", "main", "pre", os.path.basename(j["sink"])))
        rows.append({"job": j["name"], "status": r[1], "drops": d.get("w1"), "T_w1": d.get("T"), "T_w0": w0["T"], "T_same": d.get("T") == w0["T"]})
        print(json.dumps(rows[-1]), flush=True)
    ok = all(r["status"] != "failed" and r["T_same"] for r in rows)
    print("FFNCHECK PASSED" if ok else "FFNCHECK FAILED", flush=True)
    return ok


# ---------------------------------------------------------------- comparison with this machine
def sha(x):
    return hashlib.sha256(repr(x).encode()).hexdigest()[:16]


def fingerprint(d, kind):
    """Address-independent parts must be identical; the backend (address hash) only by its mean."""
    iv = lambda v: None if v is None else int(v)
    if kind == "full":
        s = d["selected"]
        rows = [(None if r.get("E") is None else float(r["E"]), None if r.get("accepted") is None else bool(r["accepted"]), iv(r.get("T"))) for r in d["rows"]]
        return {"exact": dict(T_pre=iv(s["T_pre"]), T_bc=iv(s["T_bc"]), T_sel=iv(s["T_sel"]), step=iv(s["step"]), rows=sha(rows)),
                "backend_mean": statistics.mean(v[0] for v in s["backend"].values())}
    ex, bk = {}, None
    for k in ("T", "T_pre", "T_R0", "T_L1", "T_L1_check"):
        if k in d:
            ex[k] = iv(d[k])
    for part in ("record", "R0", "L1"):
        if part in d:
            r = d[part]
            ex[part] = dict(T=iv(r["T"]), routes=sha(sorted(r["routes"].items(), key=repr)), beha=sha(sorted(r["beha_loc"].items(), key=repr)))
            if part in ("record", "L1"):
                bk = statistics.mean(v[0] for v in r["backend"].values())
    if "search" in d:
        ex["search_rows"] = sha([(r.get("step"), r.get("action"), iv(r.get("T_new")), r.get("accepted")) for r in d["search"]["rows"]])
        ex["best_T"] = iv(d["search"].get("best_T"))
    if "w1" in d:
        ex["drops"] = d["w1"]["drops"]
    if kind == "select":       # chosen on the backend: compare only what it saw as latency
        return {"exact": {}, "backend_mean": statistics.mean(v["latency"] for v in d["choice"].values())}
    return {"exact": ex, "backend_mean": bk}


def ref_jobs(out):
    phases = shard_jobs("prep", out)
    return [j for p in phases for j in p]


def localref(out):
    out = os.path.abspath(out)
    ref = {}
    for j in ref_jobs(out):
        ref[j["name"]] = fingerprint(eir.read_sink(j["sink"]), j["kind"])
    probe = os.path.join(out, "main", "pre", "pre_a3h_mha.pkl.gz")
    json.dump({"jobs": ref, "probe": ref["pre_a3h_mha"], "built": time.strftime("%Y-%m-%d %H:%M:%S")}, open(LOCAL_REF, "w"), indent=1)
    print(f"{LOCAL_REF}: {len(ref)} jobs (probe = {probe})")


def compare(out):
    out = os.path.abspath(out)
    ref = json.load(open(LOCAL_REF, encoding="utf-8"))["jobs"]
    rows, ok = [], True
    for j in ref_jobs(out):
        f, r = fingerprint(eir.read_sink(j["sink"]), j["kind"]), ref[j["name"]]
        same = json.loads(json.dumps(f["exact"])) == r["exact"]
        rel = (f["backend_mean"] / r["backend_mean"] - 1) if r["backend_mean"] else None
        ok &= same
        rows.append({"job": j["name"], "identical": same, "backend_rel": rel})
        print(f"[w1] {j['name']}: {'IDENTICAL' if same else 'DIFFERS'}" + (f", backend mean {100 * rel:+.2f}%" if rel is not None else ""), flush=True)
        if not same:
            a, b = json.loads(json.dumps(f["exact"])), r["exact"]
            print(f"[w1]   differs in {[k for k in set(a) | set(b) if a.get(k) != b.get(k)]}", flush=True)
    json.dump(rows, open(os.path.join(out, "compare_prep.json"), "w"), indent=1)
    print("PREP: reference identical" if ok else "PREP: REFERENCE DIFFERS (do not start the other notebooks)", flush=True)
    return ok


def probe(out):
    """The W1 A3h start's pre job in OUT/_probe (a fresh folder each time), compared with this machine's."""
    import platform
    out = os.path.join(os.path.abspath(out), "_probe")
    shutil.rmtree(out, ignore_errors=True)
    j = eir.job_pre(out, "a3h", OP)
    j["op"] = OP
    t0 = time.time()
    r = eir.run_job(j, out)
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "nproc": os.cpu_count(), "status": r[1], "wall_s": time.time() - t0}
    try:
        import networkx
        import numpy
        info.update(numpy=numpy.__version__, networkx=networkx.__version__)
    except Exception:
        pass
    print(f"[w1] probe env {json.dumps(info)}", flush=True)
    if r[1] == "failed":
        log = os.path.join(out, "logs", j["name"] + ".log")
        print("".join(open(log, encoding="utf-8", errors="replace").readlines()[-80:]) if os.path.exists(log) else "(no log)")
        print("PROBE FAILED", flush=True)
        return False
    f = fingerprint(eir.read_sink(j["sink"]), "pre")
    ref = json.load(open(LOCAL_REF, encoding="utf-8"))["probe"]
    same = json.loads(json.dumps(f["exact"])) == ref["exact"]
    print(f"[w1] probe T {f['exact'].get('T')} (local {ref['exact'].get('T')}), drops {f['exact'].get('drops')}, backend mean {f['backend_mean']:.1f} (local {ref['backend_mean']:.1f})", flush=True)
    print("PROBE PASSED" if same else "PROBE FAILED: differs from local_ref " + str([k for k in ref['exact'] if ref['exact'][k] != json.loads(json.dumps(f['exact'])).get(k)]), flush=True)
    return same


# ---------------------------------------------------------------- fault-coordinate report (release item 2)
def links_of(x):
    return {frozenset((tuple(a), tuple(b))) for a, b in x}


def link_rank(table, link):
    """Rank (1 = largest) of an undirected link by the sum of both directions' values in table {(a, b, ...): v}."""
    und = collections.Counter()
    for k, v in table.items():
        und[frozenset((tuple(k[0]), tuple(k[1])))] += v
    order = sorted(und, key=lambda l: -und[l])
    return (order.index(link) + 1 if link in und else None), und.get(link, 0.0), len(order)


def faults_report(out):
    out = os.path.abspath(out)
    M, N, _ = dirs(out)
    redirect()
    t = eir.load_faults()[OP]
    w1_pre = eir.read_sink(eir.job_pre(M, "a3h", OP)["sink"])
    w0_pre = eir.read_sink(os.path.join(ROOT, "eir", "out", "main", "pre", "pre_a3h_mha.pkl.gz"))
    sel = eir.read_sink(dict(eir.job_select(M, OP))["sink"])["choice"]
    rep = {"T_pre": {"W1": w1_pre["T"], "W0": w0_pre["T"]}, "faults": {}}
    for fid, rule in (("F4a", "F4"), ("F5a", "F5"), ("F8a", "F8")):
        row = {"table": {k: t[fid][k] for k in ("failed_links", "failed_nodes")}, "rule": rule}
        picks = {}
        for sd in (1, 2, 3):
            c = sel[sd][rule]
            picks[sd] = {"failed_links": c["failed_links"], "failed_nodes": c["failed_nodes"], "kind": c.get("kind"),
                         "busy_over_a": c.get("busy_over_a"), "critical_wait_over_a": c.get("critical_wait_over_a")}
        if t[fid]["failed_links"]:
            same = {sd: links_of(picks[sd]["failed_links"]) == links_of(t[fid]["failed_links"]) for sd in picks}
            l = next(iter(links_of(t[fid]["failed_links"])))
            for nm, pre in (("W1", w1_pre), ("W0", w0_pre)):
                rk, v, n = link_rank(pre["busy"], l)
                ck, cv, cn = link_rank(pre["cb_links"], l)
                row[f"busy_rank_{nm}"] = f"{rk}/{n}"
                row[f"busy_{nm}"] = v
                row[f"cb_rank_{nm}"] = f"{ck}/{cn}"
                row[f"cb_{nm}"] = cv
        else:
            same = {sd: sorted(map(tuple, picks[sd]["failed_nodes"])) == sorted(map(tuple, t[fid]["failed_nodes"])) for sd in picks}
            row["top_ports_W1_seed1"] = sel[1][rule].get("top_ports")
        row["W1_pick"] = picks
        row["same_as_table_seed123"] = same
        rep["faults"][fid] = row
    # the four composite faults: the same rules (eir_next.choose_c) on the W1 start
    cw = EN.choose_c(OP)
    for fid, x in cw.items():
        tab = t[fid]
        same = all(sorted(map(tuple, map(tuple, x[k]))) == sorted(map(tuple, map(tuple, tab[k]))) for k in ("failed_pes", "failed_nodes")) and \
            links_of(x["failed_links"]) == links_of(tab["failed_links"])
        rep["faults"][fid] = {"table": {k: tab[k] for k in ("failed_pes", "failed_links", "failed_nodes")},
                              "W1_pick": {k: x[k] for k in ("failed_pes", "failed_links", "failed_nodes")}, "same_as_table": same,
                              "pe_load_W1": x["pe_load"], "pe_load_table": tab.get("pe_load")}
    json.dump(rep, open(os.path.join(out, "faults_report.json"), "w"), indent=1, default=str)
    for fid, r in rep["faults"].items():
        print(fid, json.dumps({k: v for k, v in r.items() if k not in ("W1_pick",)}, default=str), flush=True)
    return rep


def smoke(out):
    """Every code path once, short (bundle check): pre, l1 at F4, rep L2Gsync 5 steps, FULL-BusyBarn A1 15 SA steps; 2 backend
    seeds in EIR jobs. Written to OUT/_smoke (a fresh folder), never into the results."""
    out = os.path.join(os.path.abspath(out), "_smoke")
    shutil.rmtree(out, ignore_errors=True)
    os.environ.update(EIR_BACKEND_SEEDS="2", CPCM_SA_STEPS="15")
    jobs = [[eir.job_pre(out, "a3h", OP)], [eir.job_l1(out, "a3h", OP, "F4")],
            [eir.job_rep(out, "a3h", OP, "F4", "L2Gsync", 5, 123), eir.job_full(out, OP, "F1", "A1", 0, 4)]]
    ok = True
    for phase in jobs:
        for j in phase:
            j["op"] = OP
            r = eir.run_job(j, out)
            good = r[1] != "failed"
            if good:
                d = eir.read_sink(j["sink"])
                if j["kind"] != "full":
                    good = d.get("w1", {}).get("drops") and all(k.endswith(":72") for k in d["w1"]["drops"])
            print(f"[w1] smoke {j['name']}: {'ok' if good else 'FAILED'} ({r[2]:.0f} s)", flush=True)
            if not good:
                ok = False
                log = os.path.join(out, "logs", j["name"] + ".log")
                if os.path.exists(log):
                    print("".join(open(log, encoding="utf-8", errors="replace").readlines()[-80:]), flush=True)
                break
        if not ok:
            break
    if ok:          # FULL: the drop counts are in its log
        log = open(os.path.join(out, "logs", jobs[2][1]["name"] + ".log"), encoding="utf-8", errors="replace").read()
        line = [x for x in log.splitlines() if x.startswith("[w1] drops")]
        import re
        got = re.findall(r"'(\w+)', (\d+)\)", line[-1]) if line else []
        ok = bool(got) and {w for w, _ in got} == {"graph", "add_broadcast"} and all(n == "72" for _, n in got)
        print(f"[w1] smoke full drops: {line[-1] if line else 'none'}", flush=True)
    print("SMOKE PASSED" if ok else "SMOKE FAILED", flush=True)
    return ok


# ---------------------------------------------------------------- report and pack
def report(shard, out):
    out = os.path.abspath(out)
    phases = shard_jobs(shard, out)
    jobs = [j for p in phases for j in p]
    done = [j for j in jobs if eir.sink_ok(j["sink"], j)]
    walls = {}
    p = os.path.join(out, f"runs_{shard}.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["status"] == "ok":
                walls[r["name"]] = r["wall_s"]
    peaks = []
    mp = os.path.join(out, "mem", f"{shard}.jsonl")
    if os.path.exists(mp):
        peaks = [json.loads(x) for x in open(mp, encoding="utf-8")]
    w = list(walls.values())
    rep = {"shard": shard, "jobs": len(jobs), "done": len(done), "missing": [j["name"] for j in jobs if j not in done][:20],
           "wall_median_s": statistics.median(w) if w else None, "wall_max_s": max(w) if w else None, "process_hours": sum(w) / 3600,
           "peak_gb_max": max((x["peak_gb"] for x in peaks), default=None), "n_mem": len(peaks),
           "drop_counts": sorted({k for x in peaks for k in x["drops"]})}
    json.dump(rep, open(os.path.join(out, f"report_{shard}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


def pack(shard, out):
    out = os.path.abspath(out)
    jobs = [j for p in shard_jobs(shard, out) for j in p]
    dst = os.path.join(out, f"results_w1_{shard}.tar.gz")
    n = 0
    with tarfile.open(dst + ".part", "w:gz") as tar:
        for j in jobs:
            for k in ("sink", "snap"):
                if j.get(k) and os.path.exists(j[k]):
                    tar.add(j[k], os.path.relpath(j[k], out))
                    n += 1
            lg = os.path.join(out, "logs", j["name"] + ".log")
            if os.path.exists(lg):
                tar.add(lg, os.path.relpath(lg, out))
        for f in (f"runs_{shard}.jsonl", f"report_{shard}.json", os.path.join("mem", f"{shard}.jsonl"), "compare_prep.json", "faults_report.json"):
            p = os.path.join(out, f)
            if os.path.exists(p):
                tar.add(p, f)
    os.replace(dst + ".part", dst)
    print(f"{dst}: {n} files, {os.path.getsize(dst) / 1e6:.1f} MB", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        return child(sys.argv[2])
    redirect()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "probe", "smoke", "run", "localref", "compare", "faults", "report", "pack", "ffncheck"])
    ap.add_argument("shard", nargs="?", choices=SHARDS)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "eir"))
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    if a.cmd in ("run", "report", "pack") and not a.shard:
        raise SystemExit("shard needed")
    if a.cmd == "setup":
        setup()
    elif a.cmd == "probe":
        sys.exit(0 if probe(a.out) else 1)
    elif a.cmd == "smoke":
        sys.exit(0 if smoke(a.out) else 1)
    elif a.cmd == "run":
        run(a.shard, a.out, a.workers)
        if a.shard == "prep":
            if os.path.exists(LOCAL_REF):
                if not compare(a.out):
                    raise SystemExit(1)
            faults_report(a.out)
    elif a.cmd == "localref":
        localref(a.out)
    elif a.cmd == "compare":
        sys.exit(0 if compare(a.out) else 1)
    elif a.cmd == "faults":
        faults_report(a.out)
    elif a.cmd == "report":
        report(a.shard, a.out)
    elif a.cmd == "pack":
        pack(a.shard, a.out)
    elif a.cmd == "ffncheck":
        sys.exit(0 if ffncheck() else 1)


if __name__ == "__main__":
    main()
