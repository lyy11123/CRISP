"""C: GPT, one physical fault on every operator of a layer (W1), E2E_HANDOFF.md section 5 (user 2026-11-09), plan C_PLAN.md.

EIR (eir/eir.py) is imported and redirected in memory, nothing in eir/ is changed or written: the module globals GPT (model folder) and
TASKS[op] (template file) are set per operator in the child, the start mappings, fault table, artifact work copy and output root are
this folder's (c/). FULL goes through cpcm_sa's child with this experiment's fault, as eir.child_full does, with the operator's model.

Operators (8): prefill and decode LN, MHA, projection, FFN, 2x2, sequence 2048, W1 (MHA: the final broadcast without QK^T and S; prefill
drops 128 messages per broadcast, decode 72, every drop checked as in w1_rerun/w1_eir.py; LN, projection, FFN have no such message).
Starts per operator: B4's W1 backend-best A1 and A3h runs (b4.json), injected as EIR injects its starts. The template file of an
operator is the A1 winner's (g, lossratio) file (only W's energy reads the loss ratio, and W is not run).

Jobs per fault and operator: l1 (R0 + L1, both starts; the post-fault latency before repair), L2Gsync x s{30, 100, 300} x 5 SA seeds
(A3h start), FULL-A3h (14 = g0/g1 x 7 seeds), FULL-BusyBarn (A1 arm, 14 = g0/g1 x lossratio 0-6). Winners by backend (analysis).
Fault set (a) fixed by physical position, set (b) by EIR's rules on decode MHA's W1 A3h start; the table is c/faults.json (`coords`),
written before any fault result is seen (PROGRESS.md).

  python e2e/c_eir.py setup                         work copy c/work, inputs c/inputs (starts), checks against b4.json
  python e2e/c_eir.py select --out OUT              decode MHA A3h start: E3's F1 rule (and F4, F5, F8 for the record)
  python e2e/c_eir.py coords --out OUT              c/faults.json: set (a) and set (b)
  python e2e/c_eir.py regress --out OUT             W0 decode: this driver vs eir/out sinks (exact parts bit for bit)
  python e2e/c_eir.py smoke --out OUT               every operator x job kind once, short
  python e2e/c_eir.py run <shard> --out OUT --workers N
  python e2e/c_eir.py shards                        shard names, job counts and estimated process hours
  python e2e/c_eir.py localref / compare / probe / report / pack   as w1_rerun/w1_eir.py
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import pickle
import random
import shutil
import statistics
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
for _v in THREAD_VARS:
    os.environ[_v] = "1"
os.environ.setdefault("CPCM_SA_STAGE", "s2")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CDIR = os.path.join(HERE, "c")
sys.path.insert(0, os.path.join(ROOT, "eir"))
import eir  # noqa: E402

cs = eir.cs
WORK = os.path.join(CDIR, "work", "artifact")
INPUTS = os.path.join(CDIR, "inputs")
FAULTS = os.path.join(CDIR, "faults.json")
LOCAL_REF = os.path.join(CDIR, "local_ref.json")
PREFILL, DECODE = "intra_mapping_gpt_prefill", "intra_mapping_gpt_decode"
SUF = cs.SUF
B4 = os.path.join(HERE, "out", "b4")
# op -> (model, template operator, B4 cfg, runs folder, W1 drops per broadcast, A1 run, A3h run); runs = B4's W1 backend best (b4.json)
OPS = {
    "pf_ln": (PREFILL, "ln", "pf_ln", B4, 0, ("A1", 0, 4), ("A3h", 0, 123)),
    "pf_mha": (PREFILL, "mha", "pf_mha_w1", B4, 128, ("A1", 0, 6), ("A3h", 0, 1)),
    "pf_proj": (PREFILL, "proj", "pf_proj", B4, 0, ("A1", 0, 0), ("A3h", 0, 1)),
    "pf_ffn": (PREFILL, "ffn", "pf_ffn", os.path.join(HERE, "out"), 0, ("A1", 0, 0), ("A3h", 0, 123)),
    "dc_ln": (DECODE, "ln", "dc_ln", B4, 0, ("A1", 0, 1), ("A3h", 0, 123)),
    "dc_mha": (DECODE, "mha", "dc_mha_w1", B4, 72, ("A1", 0, 4), ("A3h", 0, 2)),
    "dc_proj": (DECODE, "proj", "dc_proj", B4, 0, ("A1", 0, 4), ("A3h", 1, 3)),
    "dc_ffn": (DECODE, "ffn", "ffn", os.path.join(ROOT, "cpcm_sa", "out", "s2"), 0, ("A1", 0, 4), ("A3h", 0, 5)),
}
B4_CFG = {"pf_ln": ("prefill", "ln"), "pf_mha": ("prefill", "mha"), "pf_proj": ("prefill", "proj"), "pf_ffn": ("prefill", "ffn"),
          "dc_ln": ("decode", "ln"), "dc_mha": ("decode", "mha"), "dc_proj": ("decode", "proj"), "dc_ffn": ("decode", "ffn")}
STARTS = {"a1": 5, "a3h": 6}            # index into OPS rows
W1_DROP = ("qkt", "s_softmax")
STEPS = (30, 100, 300)
SEEDS = eir.SA_SEEDS                     # 123, 1, 2, 3, 4
SET_A = ["PF1", "PF2", "PF4", "PF5", "PF6", "PF7", "PF8", "PCRandom", "PCCluster", "PCEdge", "PCScratch"]
SET_B = ["MF1", "MF2", "MF4", "MF5", "MF6", "MF7", "MF8", "MCRandom", "MCCluster", "MCEdge", "MCScratch"]
LONG_OPS = ("pf_mha", "pf_ln")           # handoff 5.6: 3 h, alert only
ALERT_S = {op: (3 * 3600 if op in LONG_OPS else 3600) for op in OPS}
MEM_STOP_GB, PEAK_STOP_GB = 3.0, 3.0
# prep jobs that failed and wait for a decision (2026-11-09: seed_broadcast IndexError, a broadcast event of producer 20 depends on 19);
# jobs reading their sinks are held back, everything else runs
HELD_PREP = ["l1_a3h_pf_mha_MCRandom"]
C_RANDOM_SEED = 20261105


def task_of(op):
    model, top, _, _, _, (arm, g, l), _ = OPS[op]
    return f"{top}_sq2048_greedy{g}_lossratio{l}_ch2x2{SUF}"


def run_file(op, arm):
    model, top, cfg, base, _, a1, a3h = OPS[op]
    a, g, x = a1 if arm == "a1" else a3h
    return os.path.join(base, "runs", cs.run_name(cfg, a, g, x) + ".pkl.gz")


def redirect(reg=False):
    eir.WORK = cs.WORK = cs.e3.WORK = WORK
    if not reg:
        eir.INPUTS, eir.FAULTS_JSON = INPUTS, FAULTS
    eir.__file__ = os.path.abspath(__file__)      # eir.run_job launches `<this file> child <spec>`


# ---------------------------------------------------------------- W1 (as w1_rerun/w1_eir.py, the expected count per operator)
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
    want = OPS[op][4]

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
    ws = eir.write_sink

    def write_sink(path, d):
        d["w1"] = {"drops": {f"{w}:{n}": c for (w, n), c in DROPS.items()}, "expect": want}
        return ws(path, d)
    eir.write_sink = write_sink


def mem_logger(name):
    path = os.environ.get("C_MEM_LOG")
    if not path:
        return
    import atexit
    sys.path.insert(0, HERE)
    import e2e_pilot_peak as PK  # noqa: F401  (tiny helper below, kept out of e2e_pilot's module-level config)

    def log():
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"name": name, "peak_gb": PK.peak() / 2 ** 30, "drops": {f"{w}:{n}": c for (w, n), c in DROPS.items()}}) + "\n")
    atexit.register(log)


# ---------------------------------------------------------------- child
def child_full(spec):
    """eir.child_full with the operator's model and template operator (eir's version takes both from the decode GPT table)."""
    op, fid, arm, g, x = spec["op"], spec["fault"], spec["arm"], spec["g"], spec["x"]
    f = eir.fault_of(op, fid)
    cfg = eir.full_cfg(op, fid)
    model, top = OPS[op][0], OPS[op][1]
    cs.CONFIGS[cfg] = (model, top, {"failed_nodes": f["failed_nodes"], "failed_links": f["failed_links"]}, "ch2x2")
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
    cs.child(cfg, arm, g, x, spec["sink"], spec["snap"], False)


def child(spec_path):
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    reg = spec.get("reg", False)
    redirect(reg)
    mem_logger(spec["name"])
    if os.environ.get("C_BACKEND_SEEDS"):
        eir.BACKEND_SEEDS = cs.BACKEND_SEEDS = list(range(1, int(os.environ["C_BACKEND_SEEDS"]) + 1))
    op = spec["op"]
    if not reg:
        eir.GPT = OPS[op][0]
        eir.TASKS[op] = task_of(op)
        if OPS[op][4]:
            install_w1(spec["kind"], op)
    cs._artifact_path()
    import Stream_Mapping
    if not os.path.abspath(Stream_Mapping.__file__).startswith(WORK):
        raise RuntimeError(f"artifact imported from {Stream_Mapping.__file__}, not from {WORK}")
    print(f"[c] child {spec['name']} ({spec['kind']}, {op}, model {eir.GPT}, task {eir.TASKS.get(op)}, reg {reg})", flush=True)
    try:
        if spec["kind"] == "full" and not reg:
            return child_full(spec)
        return eir.child(spec)
    finally:
        print(f"[c] drops {dict(DROPS)}", flush=True)


# ---------------------------------------------------------------- setup and coordinates
def setup():
    if not os.path.isdir(WORK):
        src = os.path.join(HERE, "work", "artifact")
        os.makedirs(os.path.join(WORK, "models"), exist_ok=True)
        for sub in os.listdir(src):
            if sub != "models":
                p = os.path.join(src, sub)
                (shutil.copytree if os.path.isdir(p) else shutil.copy)(p, os.path.join(WORK, sub), **({"ignore": shutil.ignore_patterns("__pycache__", "*.pyc")} if os.path.isdir(p) else {}))
        for m in (PREFILL, DECODE):
            shutil.copytree(os.path.join(src, "models", m), os.path.join(WORK, "models", m), ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "results"))
            os.makedirs(os.path.join(WORK, "models", m, "results"), exist_ok=True)
        print(f"work copy {WORK} (from {src}: {PREFILL}, {DECODE}, without results)")
    os.makedirs(INPUTS, exist_ok=True)
    b4 = json.load(open(os.path.join(B4, "b4.json"), encoding="utf-8"))
    sel = {(r["phase"], r["op"], r["arm"]): r["backend"] for r in b4["ops"] if r["workload"] == "W1"}
    info = {}
    for op, row in OPS.items():
        ph, o = B4_CFG[op]
        for start, i in STARTS.items():
            arm, g, x = row[i]
            name = cs.run_name(row[2], arm, g, x)
            ref = sel[(ph, o, arm)]
            if ref["run"] != name:
                raise SystemExit(f"{op} {start}: b4.json's W1 backend best is {ref['run']}, not {name}")
            with gzip.open(run_file(op, start), "rb") as f:
                d = pickle.load(f)
            mean = statistics.mean(v[0] for v in d["selected"]["backend"].values())
            if abs(mean - ref["mean"]) > 0.5:
                raise SystemExit(f"{op} {start}: backend mean {mean} vs b4.json {ref['mean']}")
            with gzip.open(os.path.join(INPUTS, f"{start}_{op}.pkl.gz"), "wb") as f:
                pickle.dump(d["selected"]["e3map"], f, protocol=pickle.HIGHEST_PROTOCOL)
            if not os.path.exists(os.path.join(WORK, "models", row[0], "py", task_of(op) + ".py")):
                raise SystemExit(f"{op}: template {task_of(op)} missing")
            info[f"{start}_{op}"] = {"run": name, "b4_backend_mean": ref["mean"], "task": task_of(op)}
    json.dump(info, open(os.path.join(CDIR, "starts.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


def _node_at(y, x):
    return (y // 4) * 2 + x // 4, (y % 4) * 4 + x % 4


def set_a():
    """Fixed by physical position, independent of any mapping (rules written with each row)."""
    rnd = random.Random(C_RANDOM_SEED)
    noncorner = lambda d: [(d, c) for c in range(16) if c not in (0, 3, 12, 15)]
    rows = {
        "PF1": dict(failed_nodes=[[0, 10]], kind="node (core + router)",
                    rule="interior core closest to the 2x2 group's centre (die 0 core 10, grid (2, 2)); ties over dies: lowest die"),
        "PF2": dict(failed_nodes=[[0, 9]], kind="node (core + router)",
                    rule="E3's F2 rule, already mapping-independent: random.Random(F2_SEED).choice(sorted non-corner cores) = (0, 9)"),
        "PF4": dict(failed_links=[[[0, 11], [1, 8]], [[1, 8], [0, 11]]], kind="link ch2ch, both directions",
                    rule="D2D link of die pair (0, 1) nearer to the group centre (the pair has (0,7)-(1,4) and (0,11)-(1,8))"),
        "PF5": dict(failed_links=[[[0, 10], [0, 11]], [[0, 11], [0, 10]]], kind="link co2co, both directions",
                    rule="on-chip link from PF1's core (0, 10) to its neighbour on the die (0, 1) edge (0, 11), which feeds PF4's D2D link"),
        "PF6": dict(failed_pes=[[0, 10]], kind="PE only (tensor + vector unit); router keeps forwarding", rule="PF1's core, PE only (as E3's F6)"),
        "PF7": dict(failed_pes=[[0, 9]], kind="PE only (tensor + vector unit); router keeps forwarding", rule="PF2's core, PE only (as E3's F7)"),
        "PF8": dict(failed_nodes=[[0, 22]], kind="DDR port node",
                    rule="lower-numbered DDR port of die 0's core 15, the corner core at the group centre (ports 22, 23)"),
        "PCRandom": dict(failed_pes=[rnd.choice(noncorner(d)) for d in range(4)], kind="k=4 composite: Random",
                         rule=f"per die one non-corner core, random.Random({C_RANDOM_SEED}).choice(sorted non-corner cores), dies 0-3 in order; PE only"),
        "PCCluster": dict(failed_pes=[[0, 5], [0, 6], [0, 10]], failed_links=[[[0, 6], [0, 10]], [[0, 10], [0, 6]]], kind="k=4 composite: Cluster",
                          rule="die 0 interior path 5-6-10, PE only; plus the on-chip link 6-10, both directions"),
        "PCEdge": dict(failed_pes=[[0, 0], [0, 1], [0, 2]], failed_nodes=[[0, 16]], kind="k=4 composite: Edge",
                       rule="3 consecutive cores on the group's outer top edge from the outer corner (die 0 cores 0, 1, 2), PE only; plus DDR port 16 of core 0"),
        "PCScratch": dict(failed_pes=[[0, 10], [0, 11], [1, 8]], failed_links=[[[0, 11], [1, 8]], [[1, 8], [0, 11]]], kind="k=4 composite: Scratch",
                          rule="PF4's D2D link, both directions; plus the 3 cores of global row 2 across it, x = 2..4: (0,10), (0,11), (1,8); PE only"),
    }
    out = {}
    for fid, r in rows.items():
        out[fid] = {"failed_nodes": [list(x) for x in r.get("failed_nodes", [])], "failed_links": r.get("failed_links", []),
                    "failed_pes": [list(x) for x in r.get("failed_pes", [])], "kind": r["kind"], "rule": r["rule"], "set": "a", "same_as": None}
    return out


def select_job(out):
    return dict(eir.job_select(os.path.join(out, "prep"), "dc_mha"), start="a3h", fids=["F1", "F2", "F4", "F5", "F8"])


def set_b(out):
    """EIR's rules on decode MHA's W1 A3h start. F1 (and F6 on its core): E3's F1 rule here (`select`, backend seed 1 decides, as
    eir.select). F2 / F7: E3's random rule (mapping-independent). F4, F5, F8 and the four composite: the W1 EIR reselection on the same
    start (w1_rerun/eir_faults.json F4aW, F5aW, F8aW, C*W; same rules, same start, frozen 2026-11-09)."""
    d = eir.read_sink(select_job(out)["sink"])["choice"]
    w1 = json.load(open(os.path.join(HERE, "w1_rerun", "eir_faults.json"), encoding="utf-8"))["mha"]
    c1 = d[1]["F1"]
    rows = {"MF1": {"failed_nodes": [list(n) for n in c1["failed_nodes"]], "failed_links": [], "failed_pes": [], "kind": c1["kind"],
                    "rule": "E3's F1 rule (non-corner core with the largest computation busy time) on decode MHA's W1 A3h start, backend seed 1",
                    "stable_over_seeds_123": all(d[s]["F1"]["failed_nodes"] == c1["failed_nodes"] for s in (2, 3))},
            "MF2": {"failed_nodes": [list(n) for n in d[1]["F2"]["failed_nodes"]], "failed_links": [], "failed_pes": [], "kind": "node (core + router)",
                    "rule": "E3's F2 rule (random, mapping-independent)"}}
    rows["MF6"] = {"failed_nodes": [], "failed_links": [], "failed_pes": rows["MF1"]["failed_nodes"], "kind": "PE only (tensor + vector unit); router keeps forwarding", "rule": "MF1's core, PE only"}
    rows["MF7"] = {"failed_nodes": [], "failed_links": [], "failed_pes": rows["MF2"]["failed_nodes"], "kind": "PE only (tensor + vector unit); router keeps forwarding", "rule": "MF2's core, PE only"}
    for new, old in (("MF4", "F4aW"), ("MF5", "F5aW"), ("MF8", "F8aW"), ("MCRandom", "CRandomW"), ("MCCluster", "CClusterW"), ("MCEdge", "CEdgeW"), ("MCScratch", "CScratchW")):
        x = w1[old]
        rows[new] = {"failed_nodes": x["failed_nodes"], "failed_links": x["failed_links"], "failed_pes": x.get("failed_pes", []), "kind": x["kind"],
                     "rule": f"w1_rerun eir_faults.json {old}: {x.get('rule', '')}".strip(), "stable_over_seeds_123": x.get("stable_over_seeds_123")}
    for fid in SET_B:
        rows[fid]["set"] = "b"
    rows["select_record"] = {s: {f: d[s][f] for f in ("F1", "F2", "F4", "F5", "F8")} for s in (1, 2, 3)}
    return rows


def same_fault(x, y):
    links = lambda r: sorted({frozenset((tuple(u), tuple(v))) for u, v in r["failed_links"]}, key=repr)
    key = lambda r: (sorted(map(tuple, r["failed_nodes"])), links(r), sorted(map(tuple, r["failed_pes"])))
    return key(x) == key(y)


def coords(out):
    if os.path.exists(FAULTS):
        raise SystemExit(f"{FAULTS} exists: the coordinates are fixed (handoff 5.1); remove it by hand only before any fault result")
    a = set_a()
    b = set_b(out)
    rec = b.pop("select_record")
    for fid in SET_B:
        b[fid]["same_as"] = next((g for g in SET_A if same_fault(b[fid], a[g])), None)
    table = {op: {**a, **b} for op in OPS}
    table["_select_record"] = rec
    with open(FAULTS, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=1, ensure_ascii=False)
    for fid, r in list(a.items()) + list(b.items()):
        print(f"[c] {fid}: nodes {r['failed_nodes']} links {r['failed_links']} pes {r['failed_pes']} same_as {r.get('same_as')}", flush=True)


def faults_run():
    """Fault ids that get runs: set (a), and set (b) except rows identical to a set (a) row."""
    t = json.load(open(FAULTS, encoding="utf-8"))["dc_mha"]
    return SET_A + [f for f in SET_B if not t[f].get("same_as")]


# ---------------------------------------------------------------- jobs and shards
def jobs_l1(out, op, fid):
    P = os.path.join(out, "prep")
    return [eir.job_l1(P, s, op, fid) for s in STARTS]


def jobs_rep(out, op, fid):
    P = os.path.join(out, "prep")
    R = os.path.join(out, "rep")
    res = []
    for st in STEPS:
        for sd in SEEDS:
            j = eir.job_rep(R, "a3h", op, fid, "L2Gsync", st, sd)
            j["pre_sink"], j["l1_sink"] = eir.job_pre(P, "a3h", op)["sink"], eir.job_l1(P, "a3h", op, fid)["sink"]
            res.append(j)
    return res


def jobs_full(out, op, fid, arm):
    return [eir.job_full(os.path.join(out, "full"), op, fid, arm, g, x) for g, x in cs.variants(arm)]


def prep_jobs(out):
    P = os.path.join(out, "prep")
    pre = [eir.job_pre(P, s, op) for op in OPS for s in STARTS]
    l1 = [j for op in OPS for fid in faults_run() for j in jobs_l1(out, op, fid)]
    return pre, l1


# local estimate per single run (s). Revised 2026-11-09 from this machine's measured C runs (prep 304 l1, FULL pf_mha A3h 56 runs,
# 12-19 processes): l1 medians by operator (L1_MEAS); FULL pf_mha A3h 1,644 s against B4's W 1,855 s, so the other FULL = B4 W x 0.886;
# a repair = l1 x REP_X (EIR W1 decode MHA on Colab: L2Gsync 64 / 79 / 121 s at 30 / 100 / 300 steps against l1 about 107 s locally,
# Colab factor 1.3, scaled by the measured 300-step prefill MHA). COLAB_X: Colab single run / this machine's (EIR W1: FULL decode MHA A3h 275-340 s on Colab
# against 240 s here in B4, FULL A1 152 s against 103 s), 1.15-1.5, planned at 1.3.
W_EST = {"pf_ln": (1503, 1110), "pf_mha": (1855, 660), "pf_proj": (588, 285), "pf_ffn": (252, 258),
         "dc_ln": (38, 35), "dc_mha": (240, 125), "dc_proj": (70, 50), "dc_ffn": (100, 60)}
L1_MEAS = {"pf_ln": 1290, "pf_mha": 1000, "pf_proj": 332, "pf_ffn": 109, "dc_ln": 66, "dc_mha": 123, "dc_proj": 76, "dc_ffn": 90}
FULL_X, FULL_PF_MHA_A3H = 1644 / 1855, 1644
REP_X = {30: 0.55, 100: 0.65, 300: 1.0}      # 300: pf_mha measured 1,005-1,018 s (l1 1,000 s)
COLAB_X = 1.3


def est(j):
    op = j["op"]
    if j["kind"] == "full":
        if op == "pf_mha" and j["arm"] == "A3h":
            return FULL_PF_MHA_A3H
        a3h, a1 = W_EST[op]
        return FULL_X * (a1 if j["arm"] == "A1" else a3h)
    if j["kind"] == "rep":
        return REP_X[j["steps"]] * L1_MEAS[op]
    return L1_MEAS.get(op, 300) if j["kind"] == "l1" else 200


# Allocation. This machine (14 parallel): the longest single jobs (the 300-step repairs of prefill MHA and LN, prefill MHA's FULL-A3h)
# of the 16 Colab faults, and the 3 whole faults LOCAL_FAULTS (shards localfull + localrep, running since 2026-11-09 morning / 14:46).
# Colab (user 2026-11-09, revised: at most 5 high-RAM notebooks at once, 7 parallel each): the rest, packed by `plan` into shards of about
# 5 h (SHARD_H at COLAB_X) and frozen in c/shards.json, 5 shards per batch; jobs reading a HELD_PREP sink are left out.
def local_jobs(out, fid):
    return [j for op in ("pf_mha", "pf_ln") for j in jobs_rep(out, op, fid) if j["steps"] == 300] + jobs_full(out, "pf_mha", fid, "A3h")


def colab_h2(out, fid):
    jobs = [j for j in jobs_rep(out, "pf_ln", fid) if j["steps"] != 300] + jobs_full(out, "pf_ln", fid, "A3h") + jobs_full(out, "pf_ln", fid, "A1")
    return jobs + [j for op in ("pf_proj", "pf_ffn") for j in jobs_rep(out, op, fid) + jobs_full(out, op, fid, "A3h") + jobs_full(out, op, fid, "A1")]


def colab_h1(out, fid):
    jobs = [j for j in jobs_rep(out, "pf_mha", fid) if j["steps"] != 300] + jobs_full(out, "pf_mha", fid, "A1")
    return jobs + [j for op in ("dc_ln", "dc_mha", "dc_proj", "dc_ffn") for j in jobs_rep(out, op, fid) + jobs_full(out, op, fid, "A3h") + jobs_full(out, op, fid, "A1")]


# user 2026-11-09 (via the main session): this machine also takes whole faults (all their jobs), sized so both sides end together
# (14 parallel here from now; 6 notebooks x 7 parallel on Colab from about 4.5 h later). Total 1,004 local process h for 19 faults:
# here 291 h of long jobs + 3 whole faults (112 h), Colab 24 shards = 4 batches of 6.
LOCAL_FAULTS = ["MCCluster", "MCEdge", "MCScratch"]


def colab_faults():
    return [f for f in faults_run() if f not in LOCAL_FAULTS]


SHARDS_JSON = os.path.join(CDIR, "shards.json")
SHARD_H, PER_BATCH = 5.0, 5


def colab_pool(out):
    """Every Colab job (the old h1 / h2 split: everything of the 16 Colab faults except local_jobs), minus jobs reading a held prep."""
    held = {os.path.abspath(eir.job_l1(os.path.join(out, "prep"), n.split("_")[1], "_".join(n.split("_")[2:4]), "_".join(n.split("_")[4:]))["sink"])
            for n in HELD_PREP}
    jobs = [j for f in colab_faults() for j in colab_h1(out, f) + colab_h2(out, f)]
    return [j for j in jobs if not any(j.get(k) and os.path.abspath(j[k]) in held for k in ("pre_sink", "l1_sink"))]


def plan(out):
    """Pack the Colab pool into shards of about SHARD_H Colab hours at 7 parallel: groups (fault, operator, kind, arm or steps) by LPT,
    largest first; frozen in shards.json (refuses to overwrite)."""
    if os.path.exists(SHARDS_JSON):
        raise SystemExit(f"{SHARDS_JSON} exists; it is frozen")
    jobs = colab_pool(out)
    groups = collections.defaultdict(list)
    for j in jobs:
        groups[(j["fault"] if "fault" in j else j["name"].split("_")[3], j["op"], j["kind"], j.get("arm") or j.get("steps"))].append(j)
    total = sum(est(j) for j in jobs) * COLAB_X / 7 / 3600
    n = PER_BATCH * max(1, round(total / SHARD_H / PER_BATCH))
    bins = [[0.0, []] for _ in range(n)]
    for k, g in sorted(groups.items(), key=lambda kv: (-sum(est(j) for j in kv[1]), str(kv[0]))):
        b = min(bins, key=lambda x: x[0])
        b[0] += sum(est(j) for j in g) * COLAB_X / 7 / 3600
        b[1] += [j["name"] for j in g]
    bins.sort(key=lambda x: -x[0])
    shards = {f"c{i + 1:02d}": sorted(b[1]) for i, b in enumerate(bins)}
    hours = {f"c{i + 1:02d}": round(b[0], 2) for i, b in enumerate(bins)}
    names = list(shards)
    rec = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "model": {"L1_MEAS": L1_MEAS, "FULL_X": FULL_X, "REP_X": REP_X, "COLAB_X": COLAB_X},
           "held_prep": HELD_PREP, "jobs": len(jobs), "colab_hours_est": hours,
           "batches": [names[i:i + PER_BATCH] for i in range(0, n, PER_BATCH)], "shards": shards}
    json.dump(rec, open(SHARDS_JSON, "w", encoding="utf-8"), indent=1)
    print(f"{SHARDS_JSON}: {len(jobs)} jobs, {n} shards, Colab h per shard {min(hours.values())}-{max(hours.values())}, total {total:.1f} h at 7 parallel")


def frozen():
    return json.load(open(SHARDS_JSON, encoding="utf-8")) if os.path.exists(SHARDS_JSON) else None


def colab_shards():
    f = frozen()
    return list(f["shards"]) if f else []


def batches():
    """Batch 1: notebook 1 runs the check and then its shard; the other 4 open once the check passes. 5 shards per batch."""
    b = [list(x) for x in frozen()["batches"]]
    b[0][0] = "check+" + b[0][0]
    return b


def check(out, workers):
    """Colab batch 1, first notebook: the EIR decode regression, then the start evaluations (16) and the l1 of PF4 (16) recomputed in
    OUT/_check and compared with this machine's (local_ref.json). Nothing is written into the results."""
    sub = os.path.join(os.path.abspath(out), "_check")
    ok = regress(sub, workers)
    P = os.path.join(sub, "prep")
    phases = [[eir.job_pre(P, s, op) for op in OPS for s in STARTS], [eir.job_l1(P, s, op, "PF4") for op in OPS for s in STARTS]]
    for ph in phases:
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda j: eir.run_job(j, sub), ph))
        if any(r[1] == "failed" for r in res):
            print(f"[c] check: failed {[r[0] for r in res if r[1] == 'failed']}", flush=True)
            print("CHECK FAILED", flush=True)
            return False
    ref = json.load(open(LOCAL_REF, encoding="utf-8"))["jobs"]
    n = 0
    for j in phases[0] + phases[1]:
        f = fingerprint(eir.read_sink(j["sink"]), j["kind"])
        r = ref[j["name"]]
        same = json.loads(json.dumps(f["exact"])) == r["exact"]
        n += same
        ok &= same
        rel = (f["backend_mean"] / r["backend_mean"] - 1) if r["backend_mean"] else 0
        print(f"[c] check {j['name']}: {'IDENTICAL' if same else 'DIFFERS'}, backend mean {100 * rel:+.2f}%", flush=True)
    print(f"[c] check: {n}/{len(phases[0]) + len(phases[1])} identical with this machine", flush=True)
    print("CHECK PASSED" if ok else "CHECK FAILED", flush=True)
    return ok


def shard_names():
    return ["prep", "local", "localfull", "localrep", "check"] + colab_shards()


def shard_jobs(shard, out):
    if shard == "prep":
        pre, l1 = prep_jobs(out)
        return [pre, l1]
    if shard == "regress":
        return [regress_jobs(out)]
    if shard in ("local", "localfull", "localrep"):
        jobs = [j for fid in faults_run() if fid not in LOCAL_FAULTS for j in local_jobs(out, fid)]
        jobs += [j for fid in LOCAL_FAULTS for op in OPS for j in jobs_rep(out, op, fid) + jobs_full(out, op, fid, "A3h") + jobs_full(out, op, fid, "A1")]
        if shard == "localfull":          # the local FULL jobs need no prep: they run while prep runs
            jobs = [j for j in jobs if j["kind"] == "full"]
        elif shard == "localrep":         # the rest of local, beside a running localfull (no job in both)
            jobs = [j for j in jobs if j["kind"] != "full"]
    elif shard in colab_shards():
        want = set(frozen()["shards"][shard])
        jobs = [j for j in colab_pool(out) if j["name"] in want]
        if len(jobs) != len(want):
            raise SystemExit(f"{shard}: {len(want) - len(jobs)} frozen jobs not in the pool")
    else:
        raise SystemExit(f"unknown shard {shard}")
    return [sorted(jobs, key=lambda j: -est(j))]       # longest first


def guard(phases, out):
    o = os.path.abspath(out)
    names = [j["name"] for jobs in phases for j in jobs]
    dup = [n for n, c in collections.Counter(names).items() if c > 1]
    if dup:
        raise SystemExit(f"job names not unique: {dup[:5]}")
    for jobs in phases:
        for j in jobs:
            for k in ("sink", "snap", "pre_sink", "l1_sink", "pre_w_sink"):
                if j.get(k) and not j.get("reg") and not os.path.abspath(j[k]).startswith(o):
                    raise SystemExit(f"{j['name']}: {k} outside {o}: {j[k]}")


def free_gb():
    sys.path.insert(0, HERE)
    import e2e_pilot_peak as PK
    return PK.free_gb()


CLAIM_STALE_S = 4 * 3600
LOCAL_SHARDS = ("local", "localfull", "localrep")   # claims only here (Colab shards are disjoint; a reconnect must resume at once)


def claim(j, out):
    """Two runners on the same OUT may share a job list (2026-11-09: a second `run localfull` once localrep ends): a job starts only
    if this runner creates OUT/claims/<name> first. A claim older than CLAIM_STALE_S of a job without a sink (a killed runner) is taken over;
    to resume at once after killing a runner, delete OUT/claims."""
    if eir.sink_ok(j["sink"], j):
        return True                              # cached: run_job returns at once
    d = os.path.join(out, "claims")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, j["name"])
    try:
        os.close(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        if time.time() - os.path.getmtime(p) > CLAIM_STALE_S:
            os.utime(p)
            return True
        return False


def run(shard, out, workers):
    out = os.path.abspath(out)
    phases = shard_jobs(shard, out)
    guard(phases, out)
    if shard not in ("prep", "regress", "localfull"):
        pre, l1 = prep_jobs(out)
        miss = [j for j in pre + l1 if not eir.sink_ok(j["sink"], j)]
        if len(miss) > len(HELD_PREP):
            raise SystemExit(f"prep is not finished in {out} ({len(miss)} missing, e.g. {[j['name'] for j in miss[:3]]}): run the prep shard first")
        gone = {os.path.abspath(j["sink"]) for j in miss}
        if {j["name"] for j in miss} - set(HELD_PREP):
            raise SystemExit(f"missing prep {[j['name'] for j in miss]} not in HELD_PREP {HELD_PREP}")
        if gone:      # hold back the jobs that read a missing (failed, under decision) prep sink; the rest run
            held = [j["name"] for p in phases for j in p if any(j.get(k) and os.path.abspath(j[k]) in gone for k in ("pre_sink", "l1_sink", "pre_w_sink"))]
            phases = [[j for j in p if j["name"] not in set(held)] for p in phases]
            print(f"[c] held back {len(held)} jobs reading failed prep {HELD_PREP}: {held}", flush=True)
    os.makedirs(os.path.join(out, "mem"), exist_ok=True)
    os.environ["C_MEM_LOG"] = os.path.join(out, "mem", f"{shard}.jsonl")
    runs_log = os.path.join(out, f"runs_{shard}.jsonl")
    stop = []
    t_all = time.time()
    for i, jobs in enumerate(phases):
        t0 = time.time()
        live = {}

        def one(j):
            while not stop and free_gb() < MEM_STOP_GB + 1.5:     # no new process close to the memory line
                time.sleep(20)
            if stop:
                return j["name"], "skipped", 0.0
            if shard in LOCAL_SHARDS and not claim(j, out):
                return j["name"], "claimed", 0.0
            live[j["name"]] = (time.time(), j["op"])
            try:
                r = eir.run_job(j, out)
            finally:
                live.pop(j["name"], None)
            with open(runs_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"name": r[0], "status": r[1], "wall_s": r[2], "op": j["op"], "workers": workers, "t": time.strftime("%H:%M:%S")}) + "\n")
            return r
        done = threading.Event()

        def watch():
            warned = set()
            while not done.wait(60):
                for n, (t, op) in list(live.items()):
                    if time.time() - t > ALERT_S[op] and n not in warned:
                        warned.add(n)
                        if op in LONG_OPS:
                            print(f"[c] ALERT: {n} over 3 h (alert only, handoff 5.6)", flush=True)
                        else:
                            print(f"[c] STOP: {n} over 1 h; no new jobs (running ones finish)", flush=True)
                            stop.append(n)
                fg = free_gb()
                if fg < MEM_STOP_GB:
                    print(f"[c] STOP: free memory {fg:.1f} GB below {MEM_STOP_GB} GB; no new jobs", flush=True)
                    stop.append("memory")
        threading.Thread(target=watch, daemon=True).start()
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(one, jobs))
        done.set()
        bad = [r[0] for r in res if r[1] == "failed"]
        print(f"[c] {shard} phase {i}: {len(res)} jobs, {sum(r[1] == 'cached' for r in res)} cached, {len(bad)} failed, "
              f"{sum(r[1] == 'skipped' for r in res)} skipped, {time.time() - t0:.0f} s", flush=True)
        peaks = [json.loads(x) for x in open(os.environ["C_MEM_LOG"], encoding="utf-8")] if os.path.exists(os.environ["C_MEM_LOG"]) else []
        hi = [p["name"] for p in peaks if p["peak_gb"] > PEAK_STOP_GB]
        if bad or stop or hi:
            print(f"[c] STOP: failed {bad}, stop {stop}, peak over {PEAK_STOP_GB} GB {hi}", flush=True)
            raise SystemExit(1)
    print(f"[c] {shard}: all ok, {(time.time() - t_all) / 3600:.2f} h", flush=True)


def shards_info(out):
    rows = []
    for s in shard_names():
        if s == "check":
            continue
        jobs = [j for p in shard_jobs(s, out) for j in p]
        e = sum(est(j) for j in jobs)
        rows.append((s, len(jobs), e / 3600))
    for s, n, h in rows:
        print(f"{s:28s} {n:5d} jobs  {h:6.1f} local process h  Colab 7 parallel {h * 1.15 / 7:4.1f}-{h * 1.5 / 7:4.1f} h  local 14 parallel {h / 14:4.1f} h")
    print(f"total {sum(r[1] for r in rows)} jobs, {sum(r[2] for r in rows):.0f} local process h")
    for i, b in enumerate(batches(), 1):
        print(f"batch {i}: {b}")
    return rows


# ---------------------------------------------------------------- regression against EIR (W0 decode) and comparisons
def sha(x):
    return hashlib.sha256(repr(x).encode()).hexdigest()[:16]


def fingerprint(d, kind):
    """Address-independent parts must be identical; the backend (address hash) only by its mean (w1_rerun/w1_eir.py)."""
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
    if kind == "select":
        return {"exact": {}, "backend_mean": statistics.mean(v["latency"] for v in d["choice"].values())}
    return {"exact": ex, "backend_mean": bk}


EIR_OUT = os.path.join(ROOT, "eir", "out")
REG = [  # (kind, eir sink, spec builder): EIR W0 decode jobs of every kind C uses, incl. PE-only, DDR port, composite faults
    ("pre", "main/pre/pre_a3h_mha.pkl.gz", lambda R: eir.job_pre(R, "a3h", "mha")),
    ("pre", "main/pre/pre_a3h_ffn.pkl.gz", lambda R: eir.job_pre(R, "a3h", "ffn")),
    ("pre", "main/pre/pre_e3_mha.pkl.gz", lambda R: eir.job_pre(R, "e3", "mha")),
    ("pre", "main/pre/pre_e3_ffn.pkl.gz", lambda R: eir.job_pre(R, "e3", "ffn")),
    ("l1", "main/l1/l1_a3h_mha_F4.pkl.gz", lambda R: eir.job_l1(R, "a3h", "mha", "F4")),
    ("l1", "main/l1/l1_e3_mha_F6.pkl.gz", lambda R: eir.job_l1(R, "e3", "mha", "F6")),
    ("l1", "main/l1/l1_e3_ffn_F1.pkl.gz", lambda R: eir.job_l1(R, "e3", "ffn", "F1")),
    ("l1", "next/l1/l1_a3h_mha_CEdge.pkl.gz", lambda R: eir.job_l1(R, "a3h", "mha", "CEdge")),
    ("l1", "next/l1/l1_a3h_ffn_F8a.pkl.gz", lambda R: eir.job_l1(R, "a3h", "ffn", "F8a")),
    ("rep", "next/rep/a3h_mha_CEdge_L2Gsync_s100_r123.pkl.gz", None),
    ("rep", "next/rep/a3h_ffn_F8a_L2Gsync_s100_r123.pkl.gz", None),
    ("full", "next/full/runs/full_mha_F4a_A3h_g0_123.pkl.gz", lambda R: eir.job_full(R, "mha", "F4a", "A3h", 0, 123)),
]


def regress_jobs(out):
    R = os.path.join(os.path.abspath(out), "regress")
    pre = {op: eir.job_pre(R, "a3h", op) for op in ("mha", "ffn")}
    jobs = []
    for kind, ref, mk in REG:
        if mk:
            j = mk(R)
        else:
            op, fid = ("mha", "CEdge") if "mha" in ref else ("ffn", "F8a")
            j = eir.job_rep(R, "a3h", op, fid, "L2Gsync", 100, 123)
            j["pre_sink"] = pre[op]["sink"]
            j["l1_sink"] = eir.job_l1(R, "a3h", op, fid)["sink"]
        j = dict(j, reg=True, ref=os.path.join(EIR_OUT, ref))
        j["name"] = "reg_" + j["name"]
        jobs.append(j)
    return jobs


def regress(out, workers):
    out = os.path.abspath(out)
    jobs = regress_jobs(out)
    # dependency order: pre, l1, then rep and full
    order = [[j for j in jobs if j["kind"] == "pre"], [j for j in jobs if j["kind"] == "l1"], [j for j in jobs if j["kind"] in ("rep", "full")]]
    for ph in order:
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda j: eir.run_job(j, out), ph))
        if any(r[1] == "failed" for r in res):
            raise SystemExit(f"STOP: regression job failed: {[r[0] for r in res if r[1] == 'failed']}")
    rows, ok = [], True
    for j in jobs:
        a, b = fingerprint(eir.read_sink(j["sink"]), j["kind"]), fingerprint(eir.read_sink(j["ref"]), j["kind"])
        same = json.loads(json.dumps(a["exact"])) == json.loads(json.dumps(b["exact"]))
        ok &= same
        rel = (a["backend_mean"] / b["backend_mean"] - 1) if b["backend_mean"] else None
        rows.append({"job": j["name"], "ref": os.path.relpath(j["ref"], ROOT), "identical": same, "backend_rel": rel,
                     "differs_in": [k for k in set(a["exact"]) | set(b["exact"]) if a["exact"].get(k) != b["exact"].get(k)]})
        print(f"[c] regress {j['name']}: {'IDENTICAL' if same else 'DIFFERS ' + str(rows[-1]['differs_in'])}, backend mean {100 * (rel or 0):+.2f}%", flush=True)
    json.dump(rows, open(os.path.join(out, "regress.json"), "w"), indent=1)
    print("REGRESSION PASSED" if ok else "REGRESSION FAILED", flush=True)
    return ok


def smoke(out):
    """Every operator once per job kind, short: pre (a1, a3h), l1 (a3h) at PF4, rep L2Gsync 5 steps, FULL A3h and A1 15 SA steps;
    2 backend seeds. Written to OUT/_smoke, never into the results."""
    out = os.path.join(os.path.abspath(out), "_smoke")
    shutil.rmtree(out, ignore_errors=True)
    os.environ.update(C_BACKEND_SEEDS="2", CPCM_SA_STEPS="15")
    P = os.path.join(out, "prep")
    ok = True
    for op in OPS:
        phases = [[eir.job_pre(P, "a3h", op), eir.job_pre(P, "a1", op)], [eir.job_l1(P, "a3h", op, "PF4")]]
        r = dict(eir.job_rep(os.path.join(out, "rep"), "a3h", op, "PF4", "L2Gsync", 5, 123), pre_sink=phases[0][0]["sink"], l1_sink=phases[1][0]["sink"])
        phases.append([r, eir.job_full(os.path.join(out, "full"), op, "PF4", "A3h", 0, 123), eir.job_full(os.path.join(out, "full"), op, "PF4", "A1", 0, OPS[op][5][2])])
        for ph in phases:
            for j in ph:
                res = eir.run_job(j, out)
                good = res[1] != "failed"
                if good and j["kind"] != "full" and OPS[op][4]:
                    d = eir.read_sink(j["sink"])
                    good = bool(d.get("w1", {}).get("drops")) and all(k.endswith(f":{OPS[op][4]}") for k in d["w1"]["drops"])
                print(f"[c] smoke {j['name']}: {'ok' if good else 'FAILED'} ({res[2]:.0f} s)", flush=True)
                if not good:
                    ok = False
                    log = os.path.join(out, "logs", j["name"] + ".log")
                    if os.path.exists(log):
                        print("".join(open(log, encoding="utf-8", errors="replace").readlines()[-60:]), flush=True)
    print("SMOKE PASSED" if ok else "SMOKE FAILED", flush=True)
    return ok


def localref(out):
    out = os.path.abspath(out)
    pre, l1 = prep_jobs(out)
    ref = {j["name"]: fingerprint(eir.read_sink(j["sink"]), j["kind"]) for j in pre + l1 if j["name"] not in HELD_PREP}
    json.dump({"jobs": ref, "probe": ref["pre_a3h_dc_mha"], "built": time.strftime("%Y-%m-%d %H:%M:%S")}, open(LOCAL_REF, "w"), indent=1)
    print(f"{LOCAL_REF}: {len(ref)} jobs")


def compare(out):
    out = os.path.abspath(out)
    ref = json.load(open(LOCAL_REF, encoding="utf-8"))["jobs"]
    pre, l1 = prep_jobs(out)
    ok, rows = True, []
    for j in pre + l1:
        f, r = fingerprint(eir.read_sink(j["sink"]), j["kind"]), ref[j["name"]]
        same = json.loads(json.dumps(f["exact"])) == r["exact"]
        ok &= same
        rel = (f["backend_mean"] / r["backend_mean"] - 1) if r["backend_mean"] else None
        rows.append({"job": j["name"], "identical": same, "backend_rel": rel})
        if not same:
            print(f"[c] {j['name']}: DIFFERS in {[k for k in r['exact'] if r['exact'][k] != json.loads(json.dumps(f['exact'])).get(k)]}", flush=True)
    json.dump(rows, open(os.path.join(out, "compare_prep.json"), "w"), indent=1)
    print(f"[c] {sum(r['identical'] for r in rows)}/{len(rows)} identical")
    print("PREP: reference identical" if ok else "PREP: REFERENCE DIFFERS (do not start the other notebooks)", flush=True)
    return ok


def probe(out):
    import platform
    out = os.path.join(os.path.abspath(out), "_probe")
    shutil.rmtree(out, ignore_errors=True)
    j = eir.job_pre(os.path.join(out, "prep"), "a3h", "dc_mha")
    t0 = time.time()
    r = eir.run_job(j, out)
    print(f"[c] probe env {json.dumps({'python': sys.version.split()[0], 'platform': platform.platform(), 'nproc': os.cpu_count(), 'status': r[1], 'wall_s': time.time() - t0})}", flush=True)
    if r[1] == "failed":
        log = os.path.join(out, "logs", j["name"] + ".log")
        print("".join(open(log, encoding="utf-8", errors="replace").readlines()[-80:]) if os.path.exists(log) else "(no log)")
        print("PROBE FAILED", flush=True)
        return False
    f = fingerprint(eir.read_sink(j["sink"]), "pre")
    ref = json.load(open(LOCAL_REF, encoding="utf-8"))["probe"]
    same = json.loads(json.dumps(f["exact"])) == ref["exact"]
    print(f"[c] probe T {f['exact'].get('T')} (local {ref['exact'].get('T')}), drops {f['exact'].get('drops')}", flush=True)
    print("PROBE PASSED" if same else "PROBE FAILED", flush=True)
    return same


def report(shard, out):
    out = os.path.abspath(out)
    jobs = [j for p in shard_jobs(shard, out) for j in p]
    done = [j for j in jobs if eir.sink_ok(j["sink"], j)]
    walls = {}
    p = os.path.join(out, f"runs_{shard}.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["status"] == "ok":
                walls[r["name"]] = r["wall_s"]
    mp = os.path.join(out, "mem", f"{shard}.jsonl")
    peaks = [json.loads(x) for x in open(mp, encoding="utf-8")] if os.path.exists(mp) else []
    w = list(walls.values())
    rep = {"shard": shard, "jobs": len(jobs), "done": len(done), "missing": [j["name"] for j in jobs if j not in done][:20],
           "wall_median_s": statistics.median(w) if w else None, "wall_max_s": max(w) if w else None, "process_hours": sum(w) / 3600,
           "peak_gb_max": max((x["peak_gb"] for x in peaks), default=None), "drop_counts": sorted({k for x in peaks for k in x["drops"]})}
    json.dump(rep, open(os.path.join(out, f"report_{shard}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


def pack(shard, out):
    out = os.path.abspath(out)
    jobs = [j for p in shard_jobs(shard, out) for j in p]
    dst = os.path.join(out, f"results_c_{shard}.tar.gz")
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
        for f in (f"runs_{shard}.jsonl", f"report_{shard}.json", os.path.join("mem", f"{shard}.jsonl"), "compare_prep.json"):
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
    ap.add_argument("cmd", choices=["setup", "select", "coords", "regress", "smoke", "run", "shards", "plan", "localref", "compare", "probe", "check", "report", "pack"])
    ap.add_argument("shard", nargs="?")
    ap.add_argument("--out", default=os.path.join(CDIR, "out"))
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    if a.cmd == "setup":
        setup()
    elif a.cmd == "select":
        j = select_job(os.path.abspath(a.out))
        r = eir.run_job(j, os.path.abspath(a.out))
        if r[1] == "failed":
            raise SystemExit("select failed")
        d = eir.read_sink(j["sink"])["choice"]
        print(json.dumps({s: {f: d[s][f]["failed_nodes"] or d[s][f]["failed_links"] for f in ("F1", "F2", "F4", "F5", "F8")} for s in (1, 2, 3)}, default=str))
    elif a.cmd == "coords":
        coords(a.out)
    elif a.cmd == "regress":
        sys.exit(0 if regress(a.out, a.workers) else 1)
    elif a.cmd == "smoke":
        sys.exit(0 if smoke(a.out) else 1)
    elif a.cmd == "run":
        run(a.shard, a.out, a.workers)
        if a.shard == "prep" and os.path.exists(LOCAL_REF):
            if not compare(a.out):
                raise SystemExit(1)
    elif a.cmd == "shards":
        shards_info(a.out)
    elif a.cmd == "plan":
        plan(a.out)
    elif a.cmd == "localref":
        localref(a.out)
    elif a.cmd == "compare":
        sys.exit(0 if compare(a.out) else 1)
    elif a.cmd == "probe":
        sys.exit(0 if probe(a.out) else 1)
    elif a.cmd == "check":
        sys.exit(0 if check(a.out, a.workers) else 1)
    elif a.cmd == "report":
        report(a.shard, a.out)
    elif a.cmd == "pack":
        pack(a.shard, a.out)


if __name__ == "__main__":
    main()
