"""DSTU stage 1 (dstu/DSTU_HANDOFF.md): E4b's per-hop arbitration with the ideal estimates replaced by CPCM's.

New policies (key smaller first; same key and chunk handling as e4b.py P7: class B.DstuE is reused, only its data change):
  C4    (d^_i, -ln)                     fixed stream order by d^_i = r_i + k_i T^_i (ties by index, as E4b's P4)
  C-S   (d^_i - t - tail^_e, -ln)       tail^ from CPCM on each stream's own graph
  C-M   (d^_i - t - tail^_e, -ln)       tail^ from CPCM on the merged graph (one backward pass, see below)
  C-S'  (d_i - t - tail^_e, -ln)        tail^ as C-S, ideal d_i
  C4r, C-Sr  as C4, C-S with T^_i = mean of 40 random-tie CPCM runs (diagnostic)
Solo CPCM (once per stream x hardware x chunking): simulate(hop, no prio) -> T^; Analysis(sink_deadline = T^):
  own^(e) = F - S_last (backend: end_time - start_time, start_time = last issue tick)
  rest^(e) = T^ - lst[term(e)] - dur(term(e))  (= E2's L[term] - dur; independent of the sink level)
  group_rest^(m) = rest^ of the last-finishing block of m in the CPCM schedule
  D_T(b) = Analysis.D with sink T^; for a deadline d^, D_S(b) = d^ - T^ + D_T(b)
Merged CPCM (C-M; per configuration x deadlines, inside the child): V5's two-pass fixed before the runs: pass 1
  simulate(bb, release r) -> Analysis(sink_deadline = d^ of each event's stream) -> own^, rest^ = d^_i - lst[term] -
  dur(term), group_rest^, D_M = Analysis.D. These are the only values the policy uses. Pass 2 (hop, prio = D_M) and its
  Analysis are diagnostics of the oscillation: never used by the policy.
Random ties: cpcm.simulate(hop, rng) only shuffles the ready order; hop's candidate key stays (-k, event id). So each of
  the 40 runs rebuilds the graph in memory with a random event order (random.Random(20261106 + j)), which makes the
  (-k, id) tie random per event, with cpcm.py unchanged.

Commands (cwd dstu/):
  python dstu.py solo --jobs 14     CPCM on every stream of settings 1-2, self-checks 4 and 6
  python dstu.py smoke --jobs 14    a1 512+8192, no fault, d0, k in {1.1, 1.5, 2.0}: self-checks 1-3, 5; wall estimate
  python dstu.py s1 --jobs 14       setting 1
  python dstu.py s2 --jobs 14       setting 2
  python dstu.py s3 --jobs 14       setting 3 (E4b's draws; self-check 1 on draw 0 of each group x hardware)
Writes dstu/out/ (sims/, solo/, logs/ and _tmp/ stay out of git). E4b's files are only read.
"""
import argparse
import collections
import copy
import csv
import gzip
import itertools
import json
import os
import pickle
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
E4 = os.path.normpath(os.path.join(HERE, "..", "e4_overlap"))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "cpcm")))
sys.path.insert(0, E4)
import cpcm  # noqa: E402
import e4_hop as H  # noqa: E402
import e4_overlap as E  # noqa: E402
import e4b as B  # noqa: E402

OUT = os.path.join(HERE, "out")
SIMS = os.path.join(OUT, "sims")
SOLO = os.path.join(OUT, "solo")
KS = [1.1, 1.5, 2.0]
N_RAND = 40
RAND_SEED = 20261106
LIMIT = 1200  # s per evaluation
NOISE_L = 0.04      # E4b: |difference of L_hat| counted as a difference (T_i units)
NOISE_S = 0.0268    # E4b section 7: largest S_max change of a repeat run with the same decisions
EXT6 = {"a2_2s_mha-ffn_F4_short_first"}  # E4b: this configuration has seeds 1-6
NEW = ["C-S", "C-M", "C-S'", "C-Sr"]
ALLNEW = ["C4", "C4r"] + NEW


class DstuError(Exception):
    pass


def log(msg):
    print(f"[dstu] {msg}", flush=True)


# ------------------------------------------------------------------ CPCM data
def _graph(ev, hp0):
    deps = {t: set(e.dependency_set) for t, e in ev.items()}
    paths = {t: [list(b) for b in e.path_list] for t, e in ev.items() if e.event_type == "communication"}
    return cpcm.compile_bb(ev, hp0, path_lists=paths, deps=deps)


def _events(streams, chunk, hp0):
    ev = E.merged_events([B.load_stream(s)["events_blob"] for s in streams])
    cst = None
    if chunk:
        ev, cst = E.chunk_events(ev, hp0, chunk["size"], chunk["only_d2d"])
    return ev, cst


def _groups(g, tags_of):
    groups = collections.defaultdict(list)
    for e in tags_of:
        if g.kind[e] == cpcm.COMM:
            groups[B.group_of(B.rt(g.tag[e]))].append(e)
    return groups


def tail_data(g, sc, an, sink, events):
    """own^, rest^, group data of the events (ids) of one stream on schedule sc with its Analysis an;
    sink = the stream's sink level used by an."""
    own, rest, D = {}, {}, {}
    for e in events:
        r = B.rt(g.tag[e])
        t = sc.term[e]
        own[r] = sc.F[e] - sc.S_last[e]
        rest[r] = sink - an.lst[t] - (sc.a_f[t] - sc.a_s[t])
        D[r] = an.D[e]
    groups = _groups(g, events)
    gn = {m: len(v) for m, v in groups.items()}
    grest = {m: rest[B.rt(g.tag[max(v, key=lambda x: (sc.F[x], str(g.tag[x])))])] for m, v in groups.items() if len(v) >= 2}
    return {"own": own, "rest": rest, "group_n": gn, "group_rest": grest, "D": D}


def solo_child(spec_path):
    with open(spec_path, "rb") as f:
        spec = pickle.load(f)
    E._artifact_path()
    hp0 = E.build_platform(E.CFG, spec["faults"])
    ev, cst = _events([spec["stream"]], spec["chunk"], hp0)
    tm = {}
    t0 = time.perf_counter()
    g = _graph(ev, hp0)
    tm["compile"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sc = cpcm.simulate(g, "hop")
    tm["forward"] = time.perf_counter() - t0
    T = sc.T
    t0 = time.perf_counter()
    an = cpcm.Analysis(sc, sink_deadline=[T] * g.n)
    data = tail_data(g, sc, an, T, range(g.n))
    tm["backward"] = time.perf_counter() - t0
    # self-check 4
    bad_mono = bad_sink = 0
    for b in range(g.n):
        sk = g.succ[g.succ_ptr[b]:g.succ_ptr[b + 1]]
        if not sk and an.D[b] != T:
            bad_sink += 1
        for x in sk:
            if an.D[x] < an.D[b]:
                bad_mono += 1
    groups = _groups(g, range(g.n))
    bad_blocks = sum(1 for v in groups.values() if len({an.D[e] for e in v}) > 1)
    occ_bad = 0
    n_occ = 0
    for e in range(g.n):
        if g.kind[e] != cpcm.COMM:
            continue
        for a in sc.ev_acts[e]:
            lk = g.res_name[sc.a_res[a]][1] if hasattr(g, "res_name") else None
            if lk is not None:
                n_occ += 1
                if int(hp0.links_dict[lk].working_time(data_bytes=ev[g.tag[e]].comm_bytes)) != sc.a_f[a] - sc.a_s[a]:
                    occ_bad += 1
    # random ties
    t0 = time.perf_counter()
    cap = {}
    real = cpcm.build

    def grab(events, res):
        cap["e"], cap["r"] = events, res
        return real(events, res)
    cpcm.build = grab
    try:
        _graph(ev, hp0)
    finally:
        cpcm.build = real
    Tr = []
    for j in range(N_RAND):
        rng = random.Random(RAND_SEED + j)
        gj = real(rng.sample(cap["e"], len(cap["e"])), cap["r"])
        Tr.append(cpcm.simulate(gj, "hop", rng=rng).T)
    tm["rand40"] = time.perf_counter() - t0
    out = dict(data, stream=spec["stream"], T_hat=T, T_rand=Tr, T_rand_mean=statistics.mean(Tr), times=tm,
               n_events=g.n, chunk_stats=cst,
               check4={"bad_monotone": bad_mono, "bad_sink": bad_sink, "bad_block_D": bad_blocks,
                       "chunked_messages": sum(1 for v in groups.values() if len(v) >= 2),
                       "occupancy_checked": n_occ, "occupancy_mismatch": occ_bad})
    with open(spec["sink"], "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def merged_cpcm(ev, hp0, release, dhat, T_ideal, ideal):
    """C-M's data (module docstring). Returns per-stream tail data and diagnostics."""
    tm = {}
    t0 = time.perf_counter()
    g = _graph(ev, hp0)
    tm["compile"] = time.perf_counter() - t0
    stream = [g.tag[e][1] - 10 for e in range(g.n)]
    rel = {e: release[stream[e]] for e in range(g.n) if g.pred_ptr[e + 1] == g.pred_ptr[e]}
    dl = [dhat[stream[e]] for e in range(g.n)]
    t0 = time.perf_counter()
    sc1 = cpcm.simulate(g, "bb", release=rel)
    tm["forward"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    an1 = cpcm.Analysis(sc1, sink_deadline=dl)
    by = collections.defaultdict(list)
    for e in range(g.n):
        by[stream[e]].append(e)
    data = {i: tail_data(g, sc1, an1, dhat[i], es) for i, es in by.items()}
    tm["backward"] = time.perf_counter() - t0
    bad_mono = sum(1 for b in range(g.n) for x in g.succ[g.succ_ptr[b]:g.succ_ptr[b + 1]] if an1.D[x] < an1.D[b])
    bad_sink = sum(1 for b in range(g.n) if g.succ_ptr[b + 1] == g.succ_ptr[b] and an1.D[b] != dl[b])
    # oscillation diagnostic (pass 2, not used by the policy)
    sc2 = cpcm.simulate(g, "hop", prio=an1.D, release=rel)
    an2 = cpcm.Analysis(sc2, sink_deadline=dl)
    dD = [abs(an2.D[e] - an1.D[e]) / T_ideal[stream[e]] for e in range(g.n)]
    C2 = {i: max(sc2.F[e] for e in es) for i, es in by.items()}
    C1 = {i: max(sc1.F[e] for e in es) for i, es in by.items()}
    diag = {"D_changed_events": sum(1 for e in range(g.n) if an2.D[e] != an1.D[e]), "n_events": g.n,
            "D_change_median_over_T": statistics.median(dD), "D_change_max_over_T": max(dD),
            "C_pass1_bb": C1, "C_pass2_hop": C2, "times": tm, "check4": {"bad_monotone": bad_mono, "bad_sink": bad_sink}}
    return data, diag


def pct(v, q):
    v = sorted(v)
    return v[int(q * (len(v) - 1))]


def dev_stats(v):
    v = list(v)
    return {"n": len(v), "median": statistics.median(v), "p05": pct(v, 0.05), "p95": pct(v, 0.95), "min": min(v), "max": max(v)} \
        if v else {"n": 0}


def est_stats(data, Dfun, dhat, d, T, ideal):
    """Per stream: D(b) against the ideal latest finish d_i - rest(b) (V5's measure, T_i units); D(b) against
    d^ - tail^(b) and against d^ - tail^(b) + own^(b) (T_i units); tail^ against the ideal tail (T_i units)."""
    out = {}
    for i, x in data.items():
        id_ = ideal[i]
        rs = [r for r in x["own"] if r in id_["rest"]]
        out[i] = {
            "D_vs_ideal": dev_stats((Dfun(i, r) - (d[i] - id_["rest"][r])) / T[i] for r in rs),
            "D_vs_d_minus_tail": dev_stats((Dfun(i, r) - (dhat[i] - x["own"][r] - x["rest"][r])) / T[i] for r in rs),
            "D_vs_d_minus_rest": dev_stats((Dfun(i, r) - (dhat[i] - x["rest"][r])) / T[i] for r in rs),
            "tail_vs_ideal": dev_stats((x["own"][r] + x["rest"][r] - id_["own"][r] - id_["rest"][r]) / T[i] for r in rs),
            "n_missing_in_ideal": len(x["own"]) - len(rs)}
    return out


# ------------------------------------------------------------------ child: backend runs
class Two:
    """Keeps a shadow policy's bookkeeping in step (the backend only calls the deciding policy)."""

    def __init__(self, pol, shadow):
        self.p, self.s = pol, shadow

    def key(self, e, link, ln, t):
        return self.p.key(e, link, ln, t)

    def on_issue(self, e, link, t):
        self.p.on_issue(e, link, t)
        self.s.on_issue(e, link, t)

    def on_finish(self, e, t):
        self.p.on_finish(e, t)

    def on_tick(self, t, ready):
        self.p.on_tick(t, ready)


class StaticP7:
    """Unchunked P7 without the -t: self-check 2's static rule (d_i - tail_e, -ln)."""

    def __init__(self, solo, d):
        self.solo, self.d = solo, d

    def key(self, e, link, ln, t):
        i = e.event_tag[1] - 10
        r = B.rt(e.event_tag)
        return (self.d[i] - self.solo[i]["own"][r] - self.solo[i]["rest"][r], -ln)

    def on_issue(self, e, link, t):
        pass


def make(spec, ev, hp, hp0):
    p = spec["policy"]
    n = p["name"]
    if n == "P1":
        return B.P1() if hasattr(B, "P1") else None, None
    if n == "fixed":
        return B.Fixed(p["rank"]), None
    if n == "C4":
        return B.Fixed(p["rank"]), B.Fixed(p["shadow_rank"])
    d_id = p["d_ideal"]
    shadow = B.DstuE(p["ideal"], d_id, ev, hp)
    if n == "IDEAL":
        if p.get("static_shadow"):
            return B.DstuE(p["ideal"], d_id, ev, hp), StaticP7(p["ideal"], d_id)
        return B.DstuE(p["ideal"], d_id, ev, hp), shadow
    return B.DstuE(p["tail"], p["d"], ev, hp), shadow


def child(spec_path):
    with open(spec_path, "rb") as f:
        spec = pickle.load(f)
    E._artifact_path()
    from e4b_backend import event_driver_e4b, P1
    hp0 = E.build_platform(E.CFG, spec["faults"])
    blobs = [B.load_stream(s)["events_blob"] for s in spec["streams"]]
    p = spec["policy"]
    extra = {}
    if p["name"] == "C-M":
        ev0 = E.merged_events(blobs)
        if spec.get("chunk"):
            ev0, _ = E.chunk_events(ev0, hp0, spec["chunk"]["size"], spec["chunk"]["only_d2d"])
        data, diag = merged_cpcm(ev0, hp0, spec["release"], p["d"], p["T_ideal"], p["ideal"])
        p["tail"] = data
        extra["merged"] = diag
        extra["est"] = est_stats(data, lambda i, r: data[i]["D"][r], p["d"], p["d_ideal"], p["T_ideal"], p["ideal"])
    elif p["name"] in ("C-S", "C-Sr", "C-S'"):
        tl = p["tail"]
        sh = {i: p["d"][i] - p["T_hat"][i] for i in tl}
        extra["est"] = est_stats(tl, lambda i, r: sh[i] + tl[i]["D"][r], p["d"], p["d_ideal"], p["T_ideal"], p["ideal"])
    out = []
    for seed in spec["seeds"]:
        ev = E.merged_events(blobs)
        cst = None
        if spec.get("chunk"):
            ev, cst = E.chunk_events(ev, hp0, spec["chunk"]["size"], spec["chunk"]["only_d2d"])
        hp = copy.deepcopy(hp0)
        if p["name"] == "P1":
            pol, sh = P1(), None
        else:
            pol, sh = make(spec, ev, hp, hp0)
        audit = []
        slog = {}
        released = {}
        random.seed(seed)
        t0 = time.time()
        res = event_driver_e4b(ev, hp, release=spec["release"], policy=Two(pol, sh) if sh is not None else pol,
                               released=released, audit=audit, shadow=sh, shadow_log=slog)
        wall = time.time() - t0
        C = collections.defaultdict(int)
        early = 0
        for t, e in ev.items():
            i = t[1] - 10
            C[i] = max(C[i], e.end_time)
            if e.start_time < spec["release"][i]:
                early += 1
        bad = [a for a in audit if a[2] != a[3]]
        rng = random.Random(1000 + seed)
        r = {"seed": seed, "wall": wall, "result": res, "C": dict(C), "release": released, "chunk_stats": cst,
             "shadow": slog, "early_events": early,
             "audit": {"n_issues": len(audit), "n_multi": sum(1 for a in audit if a[4] > 1), "n_mismatch": len(bad),
                       "sample": [(a[0], str(a[1]), a[2], a[3], a[4]) for a in rng.sample(audit, min(5, len(audit)))]}}
        out.append(r)
    with open(spec["sink"], "wb") as f:
        pickle.dump({"runs": out, **extra}, f, protocol=pickle.HIGHEST_PROTOCOL)


# ------------------------------------------------------------------ parent: running
def _call(name, spec, cmd, folder):
    path = os.path.join(folder, f"{name}.pkl.gz")
    if os.path.exists(path):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    for d in (folder, os.path.join(OUT, "_tmp"), os.path.join(OUT, "logs")):
        os.makedirs(d, exist_ok=True)
    sp = os.path.join(OUT, "_tmp", f"spec_{name}.pkl")
    sink = os.path.join(OUT, "_tmp", f"sink_{name}.pkl")
    with open(sp, "wb") as f:
        pickle.dump(dict(spec, sink=sink), f, protocol=pickle.HIGHEST_PROTOCOL)
    t0 = time.time()
    with open(os.path.join(OUT, "logs", f"{name}.log"), "w", encoding="utf-8") as lf:
        try:
            rc = subprocess.call([sys.executable, os.path.abspath(__file__), cmd, sp], cwd=HERE, stdout=lf,
                                 stderr=subprocess.STDOUT, timeout=LIMIT * max(1, len(spec.get("seeds", [1]))) + 600)
        except subprocess.TimeoutExpired:
            raise DstuError(f"{name}: over the limit")
    if rc != 0 or not os.path.exists(sink):
        raise DstuError(f"{name} failed (exit {rc}); see dstu/out/logs/{name}.log")
    with open(sink, "rb") as f:
        r = pickle.load(f)
    r["process_wall"] = time.time() - t0
    os.remove(sink)
    os.remove(sp)
    with gzip.open(path, "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    return r


def run_all(jobs, items, cmd="child", folder=SIMS):
    t0 = time.time()
    done = [0]
    total = len(items)

    def one(it):
        r = _call(it[0], it[1], cmd, folder)
        done[0] += 1
        if done[0] % 25 == 0:
            log(f"  {done[0]}/{total} ({time.time() - t0:.0f} s)")
        return r
    with ThreadPoolExecutor(jobs) as ex:
        res = dict(zip([i[0] for i in items], ex.map(one, items)))
    log(f"{total} processes ({time.time() - t0:.0f} s)")
    return res


def load_platforms():
    import e4b_s1 as S1
    st, _ = S1.load()
    with open(os.path.join(B.OUT, "setting1.json"), encoding="utf-8") as f:
        s1 = json.load(f)
    T = {}
    for kk, v in s1["T"].items():
        k, cz = kk.rsplit("|", 1)
        T[(k, None if cz == "None" else cz)] = v
    return S1, S1.platforms(st), T


def load_s3():
    import e4b_s1 as S1
    import e4b_s3 as S3
    st, _ = S1.load()
    with open(os.path.join(B.OUT, "setting3.json"), encoding="utf-8") as f:
        s3 = json.load(f)
    T = {}
    for kk, v in s3["T"].items():
        k, cz = kk.rsplit("|", 1)
        T[(k, None if cz == "None" else cz)] = v
    return S1, [p + (None,) for p in S3.platforms(st)], T


def solo_step(jobs, s3=False):
    """CPCM on every stream of settings 1-2 (s3: setting 3; cached), self-checks 4 and 6."""
    S1, plats, T = load_s3() if s3 else load_platforms()
    items = {}
    for g, hw, cz, keys, fa, alias in plats:
        for k in keys:
            items[(k, cz)] = (f"{k}_{cz or 'none'}", {"stream": k, "chunk": H.CHUNKS[cz] if cz else None, "faults": fa})
    res = run_all(jobs, list(items.values()), cmd="solo-child", folder=SOLO)
    solo = {kk: res[v[0]] for kk, v in items.items()}
    rows = []
    for (k, cz), s in solo.items():
        t = T[(k, cz)]
        Tr = s["T_rand"]
        rows.append({"stream": k, "chunk": cz or "none", "events": s["n_events"], "T_ideal": t, "T_hat": s["T_hat"],
                     "err_T_hat": s["T_hat"] / t - 1, "T_rand_mean": s["T_rand_mean"], "err_T_rand": s["T_rand_mean"] / t - 1,
                     "T_rand_min": min(Tr), "T_rand_max": max(Tr),
                     "compile_s": s["times"]["compile"], "forward_s": s["times"]["forward"],
                     "backward_s": s["times"]["backward"], "rand40_s": s["times"]["rand40"], **s["check4"]})
    write_csv("solo_table_s3.csv" if s3 else "solo_table.csv", rows)
    errs = [abs(r["err_T_hat"]) for r in rows]
    bad4 = [r["stream"] for r in rows if r["bad_monotone"] or r["bad_sink"] or r["bad_block_D"] or r["occupancy_mismatch"]]
    summ = {"n_streams": len(rows), "err_T_hat_abs_median": statistics.median(errs), "err_T_hat_abs_max": max(errs),
            "err_T_hat_median": statistics.median(r["err_T_hat"] for r in rows),
            "err_T_rand_abs_median": statistics.median(abs(r["err_T_rand"]) for r in rows),
            "err_T_rand_abs_max": max(abs(r["err_T_rand"]) for r in rows),
            "err_T_rand_median": statistics.median(r["err_T_rand"] for r in rows),
            "check4_bad": bad4, "occupancy_checked": sum(r["occupancy_checked"] for r in rows),
            "chunked_messages": sum(r["chunked_messages"] for r in rows)}
    write_json("solo_summary_s3.json" if s3 else "solo_summary.json", {**summ, "rows": rows, "rand_seed": RAND_SEED, "n_rand": N_RAND,
                                     "artifact_commit": E._artifact_commit(), "command": sys.argv})
    log("solo " + json.dumps({k: v if not isinstance(v, float) else round(v, 5) for k, v in summ.items()}))
    if bad4:
        raise DstuError(f"self-check 4: {bad4}")
    if summ["err_T_hat_abs_median"] > 0.05:
        raise DstuError("self-check 6: median |T_hat - T| > 5%")
    return S1, plats, T, solo


def load_ideal(plats):
    out = {}
    for g, hw, cz, keys, fa, alias in plats:
        for k in keys:
            if (k, cz) in out:
                continue
            p = os.path.join(B.OUT, "sims", f"solo_{k}_{cz or 'none'}.pkl.gz")
            if not os.path.exists(p):
                raise DstuError(f"E4b solo cache missing: {p}")
            with gzip.open(p, "rb") as f:
                caps = {x["seed"]: x["cap"] for x in pickle.load(f)["runs"]}
            sd = B.solo_data(caps[1])
            out[(k, cz)] = {x: sd[x] for x in ("own", "rest", "group_n", "group_rest")}
    return out


def order_of(d, n):
    return "".join(map(str, sorted(range(n), key=lambda i: (d[i], i))))


def rank_of(o):
    return {int(s): r for r, s in enumerate(o)}


def trim(s):
    return {x: s[x] for x in ("own", "rest", "group_n", "group_rest", "D")}


def build_cfgs(setting, plats, T, solo, ideal, only=None):
    """One entry per (configuration, deadlines) with everything the runs need."""
    import e4b_s1 as S1
    out = []
    if setting == 3:
        import e4b_s3 as S3
        dr = S3.draws()
        for g, hw, cz, keys, fa, alias in plats:
            n = len(keys)
            Tc = {i: T[(k, cz)] for i, k in enumerate(keys)}
            Tmin = min(Tc.values())
            for j in range(S3.N):
                x = dr[(g, j)]
                rel = {i: int(round(x["u"][i] * 0.5 * Tmin * i)) for i in Tc}
                kk = {i: x["k"][i] for i in Tc}
                That = {i: solo[(s, cz)]["T_hat"] for i, s in enumerate(keys)}
                Trnd = {i: solo[(s, cz)]["T_rand_mean"] for i, s in enumerate(keys)}
                out.append({"name": f"s3_{g}_{hw}_j{j:02d}", "tg": "s3", "label": {"draw": j}, "group": g, "hw": hw, "chunk": cz,
                            "order": "random", "streams": keys, "faults": fa, "n": n, "T": Tc, "release": rel, "k": kk,
                            "d": {i: rel[i] + kk[i] * Tc[i] for i in Tc}, "That": That, "Trnd": Trnd,
                            "dhat": {i: rel[i] + kk[i] * That[i] for i in Tc}, "dr": {i: rel[i] + kk[i] * Trnd[i] for i in Tc},
                            "seeds": B.SEEDS})
        return out
    for g, hw, cz, keys, fa, alias in plats:
        if alias:
            continue
        if setting == 2 and (g not in ("a1_2s_512-8192", "a1_2s_2048-8192", "a2_2s_mha-ffn") or hw not in ("nofault", "F4")):
            continue
        n = len(keys)
        Tc = {i: T[(k, cz)] for i, k in enumerate(keys)}
        D = int(round(0.5 * min(Tc.values())))
        lng = max(Tc, key=lambda i: Tc[i])
        orders = ["d0", "short_first", "long_first"] if setting == 1 else ["d0", "long_first"]
        for order in orders:
            rel = S1.releases(Tc, order, D)
            name = f"{g}_{hw}_{order}"
            if setting == 1:
                dls = [(f"k{k}", {i: k for i in Tc}, {"k": k}) for k in KS]
            else:
                import e4b_s2 as S2
                dls = [(S2.tag(kl, ks), {i: (kl if i == lng else ks) for i in Tc}, {"k_long": kl, "k_short": ks, "kind": kind})
                       for kl, ks, kind in S2.COMBOS]
            for tg, kk, label in dls:
                if only and not only(g, hw, order, tg):
                    continue
                d = {i: rel[i] + kk[i] * Tc[i] for i in Tc}
                That = {i: solo[(s, cz)]["T_hat"] for i, s in enumerate(keys)}
                Trnd = {i: solo[(s, cz)]["T_rand_mean"] for i, s in enumerate(keys)}
                dhat = {i: rel[i] + kk[i] * That[i] for i in Tc}
                dr = {i: rel[i] + kk[i] * Trnd[i] for i in Tc}
                out.append({"name": name, "tg": tg, "label": label, "group": g, "hw": hw, "chunk": cz, "order": order,
                            "streams": keys, "faults": fa, "n": n, "T": Tc, "release": rel, "k": kk, "d": d,
                            "That": That, "Trnd": Trnd, "dhat": dhat, "dr": dr,
                            "seeds": (B.SEEDS + [4, 5, 6]) if name in EXT6 else B.SEEDS})
    return out


def spec_base(c):
    return {"cfg": E.CFG, "faults": c["faults"], "streams": c["streams"], "release": c["release"], "seeds": c["seeds"],
            "chunk": H.CHUNKS[c["chunk"]] if c["chunk"] else None}


def rep_items(cfgs):
    """Self-check 1: P1 and the fixed order P4 uses, run again through this script."""
    items = {}
    for c in cfgs:
        items[f"rep_{c['name']}_P1"] = dict(spec_base(c), policy={"name": "P1"})
        o = order_of(c["d"], c["n"])
        items[f"rep_{c['name']}_fixed_{o}"] = dict(spec_base(c), policy={"name": "fixed", "rank": rank_of(o)})
    return list(items.items())


def new_items(cfgs, solo, ideal, which=NEW):
    items = {}
    for c in cfgs:
        sd = {i: ideal[(s, c["chunk"])] for i, s in enumerate(c["streams"])}
        tl = {i: trim(solo[(s, c["chunk"])]) for i, s in enumerate(c["streams"])}
        com = {"ideal": sd, "d_ideal": c["d"], "T_ideal": c["T"], "T_hat": c["That"]}
        pn = f"{c['name']}_{{}}_{c['tg']}"
        spb = spec_base(c)
        if "C-S" in which:
            items[pn.format("CS")] = dict(spb, policy=dict(com, name="C-S", tail=tl, d=c["dhat"]))
        if "C-M" in which:
            items[pn.format("CM")] = dict(spb, policy=dict(com, name="C-M", d=c["dhat"]))
        if "C-S'" in which:
            items[pn.format("CSp")] = dict(spb, policy=dict(com, name="C-S'", tail=tl, d=c["d"]))
        if "C-Sr" in which:
            items[pn.format("CSr")] = dict(spb, policy=dict(com, name="C-Sr", tail=tl, d=c["dr"], T_hat=c["Trnd"]))
        p4 = order_of(c["d"], c["n"])
        for o in {order_of(c["dhat"], c["n"]), order_of(c["dr"], c["n"])}:
            if o != p4:
                items[f"{c['name']}_C4_{o}_vs{p4}"] = dict(spb, policy={"name": "C4", "rank": rank_of(o), "shadow_rank": rank_of(p4)})
    return list(items.items())


# ------------------------------------------------------------------ parent: summaries
def stat(rs, c):
    Tc, d = c["T"], c["d"]
    L = [B.lhat(r, Tc, d) for r in rs]
    Lr = [max(r["C"][i] - d[i] for i in Tc) for r in rs]
    S = [B.smax(r, Tc) for r in rs]
    V = [sum(1 for i in Tc if r["C"][i] > d[i]) for r in rs]
    M = [max(r["C"].values()) for r in rs]
    return {"L": E._mean(L), "L_range": max(L) - min(L), "Lraw": E._mean(Lr), "S": E._mean(S), "V": E._mean(V), "M": E._mean(M)}


def ref_table(setting):
    fn = {1: "setting1c_table.csv", 2: "setting2_table.csv", 3: "setting3_table.csv"}[setting]
    t = {}
    with open(os.path.join(B.OUT, fn), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if setting == 3:
                key = (r["config"], "s3", r["policy"])
            elif setting == 1:
                key = (r["config"], f"k{float(r['k']):g}" if float(r["k"]) != 2.0 else "k2.0", r["policy"])
            else:
                key = (r["config"], f"kl{float(r['k_long']):g}_ks{float(r['k_short']):g}", r["policy"])
            t[key] = {"L": float(r["Lhat_max"]), "L_range": float(r["Lhat_range"]), "Lraw": float(r["L_max"]), "S": float(r["S_max"]),
                      "V": float(r["violations"]), "M": float(r["makespan"]), "member": r["member"]}
    return t


def check1(cfgs, res, tab):
    """P1 and P4's fixed order vs E4b's tables: S_max relative (stop > 4%), L_hat absolute reported."""
    rows = []
    seen = set()
    for c in cfgs:
        o = order_of(c["d"], c["n"])
        for p, rn in (("P1", f"rep_{c['name']}_P1"), ("P4", f"rep_{c['name']}_fixed_{o}")):
            if (rn, c["tg"]) in seen:
                continue
            seen.add((rn, c["tg"]))
            s = stat(res[rn]["runs"], c)
            rf = tab[(c["name"], c["tg"], p)]
            rows.append({"config": c["name"], "dl": c["tg"], "policy": p, "S_new": s["S"], "S_e4b": rf["S"],
                         "rel_S": s["S"] / rf["S"] - 1, "L_new": s["L"], "L_e4b": rf["L"], "dL": s["L"] - rf["L"]})
    return rows


def report(setting, cfgs, res, tab, solo):
    rows, ckaudit = [], collections.Counter()
    early = 0
    chunk_bad = 0
    wall_rows = []
    REF = ["P1", "P2", "P3", "P4", "P5", "P7", "P7m", "P6n"]
    for c in cfgs:
        nm, tg, n = c["name"], c["tg"], c["n"]
        ref = {p: tab[(nm, tg, p)] for p in REF}
        o = {"config": nm, "group": c["group"], "hw": c["hw"], "order": c["order"], "dl": tg, **c["label"], "P5_member": ref["P5"]["member"],
             "seeds": len(c["seeds"])}
        s_new = {}
        for p, rn in (("C-S", f"{nm}_CS_{tg}"), ("C-M", f"{nm}_CM_{tg}"), ("C-S'", f"{nm}_CSp_{tg}"), ("C-Sr", f"{nm}_CSr_{tg}")):
            r = res[rn]
            s_new[p] = stat(r["runs"], c)
            sh = {x: sum(q["shadow"].get(x, 0) for q in r["runs"]) for x in ("n", "differ", "differ_stream")}
            o[f"differ_{p}"] = sh["differ"] / sh["n"] if sh["n"] else ""
            o[f"differ_stream_{p}"] = sh["differ_stream"] / sh["n"] if sh["n"] else ""
            for q in r["runs"]:
                for x in ("n_issues", "n_multi", "n_mismatch"):
                    ckaudit[x] += q["audit"][x]
                early += q["early_events"]
                if q["chunk_stats"] and not (q["chunk_stats"]["bytes_conserved"] and q["chunk_stats"]["tree_conserved"]):
                    chunk_bad += 1
                if sorted(q["release"].items()) != sorted(c["release"].items()):
                    early += 1
            if "est" in r:
                for i, e in r["est"].items():
                    o[f"Ddev_{p}_{i}"] = e["D_vs_ideal"]["median"]
                    o[f"Dtail_{p}_{i}"] = e["D_vs_d_minus_tail"]["median"]
                    o[f"Drest_{p}_{i}"] = e["D_vs_d_minus_rest"]["median"]
                    o[f"taildev_{p}_{i}"] = e["tail_vs_ideal"]["median"]
            if "merged" in r:
                dg = r["merged"]
                o["osc_D_changed"] = dg["D_changed_events"] / dg["n_events"]
                o["osc_D_median"] = dg["D_change_median_over_T"]
                o["osc_D_max"] = dg["D_change_max_over_T"]
                o["cm_compile_s"] = dg["times"]["compile"]
                o["cm_forward_s"] = dg["times"]["forward"]
                o["cm_backward_s"] = dg["times"]["backward"]
                o["cm_backend_s"] = E._mean([q["wall"] for q in r["runs"]])
        p4o = order_of(c["d"], n)
        for p, dd in (("C4", c["dhat"]), ("C4r", c["dr"])):
            oo = order_of(dd, n)
            o[f"{p}_order"] = oo
            o[f"{p}_same_as_P4"] = oo == p4o
            if oo == p4o:
                s_new[p] = {x: ref["P4"][x] for x in ("L", "L_range", "Lraw", "S", "V", "M")}
                o[f"differ_{p}"] = 0.0
                o[f"differ_stream_{p}"] = 0.0
            else:
                r = res[f"{nm}_C4_{oo}_vs{p4o}"]
                s_new[p] = stat(r["runs"], c)
                sh = {x: sum(q["shadow"].get(x, 0) for q in r["runs"]) for x in ("n", "differ", "differ_stream")}
                o[f"differ_{p}"] = sh["differ"] / sh["n"] if sh["n"] else ""
                o[f"differ_stream_{p}"] = sh["differ_stream"] / sh["n"] if sh["n"] else ""
                for q in r["runs"]:
                    for x in ("n_issues", "n_multi", "n_mismatch"):
                        ckaudit[x] += q["audit"][x]
        for p in REF:
            o[f"L_{p}"], o[f"Lraw_{p}"], o[f"S_{p}"], o[f"V_{p}"] = ref[p]["L"], ref[p]["Lraw"], ref[p]["S"], ref[p]["V"]
        for p in ALLNEW:
            o[f"L_{p}"], o[f"Lraw_{p}"], o[f"S_{p}"], o[f"V_{p}"] = s_new[p]["L"], s_new[p]["Lraw"], s_new[p]["S"], s_new[p]["V"]
            o[f"Lrange_{p}"] = s_new[p]["L_range"]
        tau = max(NOISE_L, NOISE_S * ref["P1"]["S"])
        o["tau"] = tau
        for p, idl in (("C4", "P4"), ("C4r", "P4"), ("C-S", "P7"), ("C-M", "P7"), ("C-S'", "P7"), ("C-Sr", "P7")):
            den = ref["P1"]["L"] - ref[idl]["L"]
            o[f"den_{p}"] = den
            o[f"den_noise_{p}"] = den < tau
            o[f"ret_{p}"] = (ref["P1"]["L"] - s_new[p]["L"]) / den if abs(den) > 1e-12 else ""
        for p in ALLNEW:
            for r_ in ("P1", "P2", "P4", "P5", "P7", "P7m", "P6n"):
                o[f"dL_{p}_{r_}"] = s_new[p]["L"] - ref[r_]["L"]
                o[f"rS_{p}_{r_}"] = s_new[p]["S"] / ref[r_]["S"] - 1
        o["errT_max"] = max(abs(c["That"][i] / c["T"][i] - 1) for i in c["T"])
        rows.append(o)
    return rows, {"check3_audit": dict(ckaudit), "check5_early_or_release_diff": early, "check5_chunk_bad": chunk_bad}


def med(v):
    v = [x for x in v if x != ""]
    return statistics.median(v) if v else None


def summary(rows):
    S = {}
    S["n_rows"] = len(rows)
    pol = {}
    for p in ["P1", "P2", "P3", "P4", "P5", "P7", "P7m", "P6n"] + ALLNEW:
        pol[p] = {"L_median": med(r[f"L_{p}"] for r in rows), "L_worst": max(r[f"L_{p}"] for r in rows),
                  "Lraw_median": med(r[f"Lraw_{p}"] for r in rows), "V_mean": E._mean([r[f"V_{p}"] for r in rows])}
    S["policies"] = pol
    cmp_ = {}
    for p in ALLNEW:
        for r_ in ("P1", "P2", "P4", "P5", "P7", "P7m", "P6n"):
            ds = [r[f"dL_{p}_{r_}"] for r in rows]
            rs = [r[f"rS_{p}_{r_}"] for r in rows]
            cmp_[f"{p} vs {r_}"] = {"dL_median": med(ds), "better": sum(x < -NOISE_L for x in ds), "worse": sum(x > NOISE_L for x in ds),
                                    "dL_min": min(ds), "dL_max": max(ds), "rS_median": med(rs),
                                    "S_better": sum(x < -0.04 for x in rs), "S_worse": sum(x > 0.04 for x in rs)}
    S["compare"] = cmp_
    ret = {}
    for p in ("C4", "C4r", "C-S", "C-M", "C-S'", "C-Sr"):
        inc = [r for r in rows if not r[f"den_noise_{p}"]]
        v = [r[f"ret_{p}"] for r in inc]
        ret[p] = {"n_in": len(inc), "n_out_noise": len(rows) - len(inc), "median": med(v),
                  "p25": pct(v, 0.25) if v else None, "p75": pct(v, 0.75) if v else None,
                  "min": min(v) if v else None, "max": max(v) if v else None}
        if p in ("C4", "C4r"):
            ret[p]["same_order_as_P4"] = sum(1 for r in rows if r[f"{p}_same_as_P4"])
            dif = [r[f"ret_{p}"] for r in inc if not r[f"{p}_same_as_P4"]]
            ret[p]["n_in_different_order"] = len(dif)
            ret[p]["median_different_order"] = med(dif)
    S["retention"] = ret
    gr = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for p in ("C-S", "C-M", "C-S'"):
            if not r[f"den_noise_{p}"]:
                gr[r["group"]][p].append(r[f"ret_{p}"])
    S["retention_by_group"] = {g: {p: {"n": len(v), "median": med(v), "min": min(v), "max": max(v)} for p, v in x.items()} for g, x in gr.items()}
    S["differ"] = {p: {"differ_median": med(r[f"differ_{p}"] for r in rows), "differ_max": max(x for x in (r[f"differ_{p}"] for r in rows) if x != ""),
                       "differ_stream_median": med(r[f"differ_stream_{p}"] for r in rows),
                       "differ_stream_max": max(x for x in (r[f"differ_stream_{p}"] for r in rows) if x != "")} for p in ALLNEW}
    S["oscillation"] = {"D_changed_frac_median": med(r["osc_D_changed"] for r in rows), "D_changed_frac_max": max(r["osc_D_changed"] for r in rows),
                        "D_change_median_over_T_median": med(r["osc_D_median"] for r in rows), "D_change_max_over_T_max": max(r["osc_D_max"] for r in rows)}
    gD = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for x, f in (("Ddev", "D_vs_ideal"), ("Dtail", "D_vs_d_minus_tail"), ("Drest", "D_vs_d_minus_rest"), ("taildev", "tail_vs_ideal")):
            for p in ("C-S", "C-M"):
                for i in range(3):
                    if f"{x}_{p}_{i}" in r:
                        gD[r["group"]][f"{p} {f}"].append(r[f"{x}_{p}_{i}"])
    S["estimate_by_group"] = {g: {k: {"median": med(v), "min": min(v), "max": max(v)} for k, v in x.items()} for g, x in gD.items()}
    S["overhead"] = {"cm_backward_s_median": med(r["cm_backward_s"] for r in rows), "cm_forward_s_median": med(r["cm_forward_s"] for r in rows),
                     "cm_compile_s_median": med(r["cm_compile_s"] for r in rows), "backend_s_median": med(r["cm_backend_s"] for r in rows),
                     "backward_over_backend_median": med(r["cm_backward_s"] / r["cm_backend_s"] for r in rows),
                     "all_over_backend_median": med((r["cm_backward_s"] + r["cm_forward_s"] + r["cm_compile_s"]) / r["cm_backend_s"] for r in rows)}
    return S


def write_json(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


def write_csv(name, rows):
    os.makedirs(OUT, exist_ok=True)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def round_all(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: round_all(v) for k, v in x.items()}
    if isinstance(x, list):
        return [round_all(v) for v in x]
    return x


def setting(n, jobs):
    t0 = time.time()
    S1, plats, T, solo = solo_step(jobs, s3=n == 3)
    ideal = load_ideal(plats)
    cfgs = build_cfgs(n, plats, T, solo, ideal)
    tab = ref_table(n)
    log(f"setting {n}: {len(cfgs)} rows")
    cfgs1 = [c for c in cfgs if c["label"].get("draw") in (None, 0)]  # setting 3: draw 0 only (PROGRESS)
    rep = rep_items(cfgs1)
    res = run_all(jobs, rep)
    ck1 = check1(cfgs1, res, tab)
    write_csv(f"setting{n}_check1.csv", ck1)
    mx = max(ck1, key=lambda r: abs(r["rel_S"]))
    ckl = {"processes": len(rep), "rows": len(ck1), "max_abs_rel_S": abs(mx["rel_S"]), "at": f"{mx['config']} {mx['policy']}",
           "median_abs_rel_S": med(abs(r["rel_S"]) for r in ck1), "max_abs_dL": max(abs(r["dL"]) for r in ck1)}
    log("check 1 " + json.dumps(round_all(ckl)))
    if abs(mx["rel_S"]) > 0.04:
        over = [(r["config"], r["policy"], round(r["rel_S"], 4)) for r in ck1 if abs(r["rel_S"]) > 0.04]
        write_json(f"setting{n}_stop.json", {"check1": ckl, "over": over})
        raise DstuError(f"self-check 1: {over}")
    t1 = time.time()
    its = new_items(cfgs, solo, ideal)
    log(f"setting {n}: {len(its)} new-policy processes")
    res2 = run_all(jobs, its)
    res.update(res2)
    t2 = time.time()
    rows, cks = report(n, cfgs, res, tab, solo)
    write_csv(f"setting{n}_report.csv", rows)
    S = summary(rows)
    S.update(checks={"check1": ckl, **cks}, walls={"rep": t1 - t0, "new": t2 - t1}, processes=len(rep) + len(its),
             seeds=B.SEEDS, ext6=sorted(EXT6), command=sys.argv, artifact_commit=E._artifact_commit(), noise={"L": NOISE_L, "S": NOISE_S})
    write_json(f"setting{n}.json", S)
    log("summary " + json.dumps(round_all(S)))
    if cks["check3_audit"].get("n_mismatch"):
        raise DstuError("self-check 3: audit mismatch")
    if cks["check5_early_or_release_diff"] or cks["check5_chunk_bad"]:
        raise DstuError("self-check 5")


def smoke(jobs):
    t0 = time.time()
    S1, plats, T, solo = solo_step(jobs)
    ideal = load_ideal(plats)
    t_solo = time.time() - t0
    cfgs = build_cfgs(1, plats, T, solo, ideal, only=lambda g, hw, o, tg: g == "a1_2s_512-8192" and hw == "nofault" and o == "d0")
    tab = ref_table(1)
    rep = rep_items(cfgs)
    res = run_all(jobs, rep)
    ck1 = check1(cfgs, res, tab)
    # self-check 2: ideal tail in this framework, a1 static shadow; a2 chunked, P7 shadow
    its = []
    for c in cfgs:
        sd = {i: ideal[(s, c["chunk"])] for i, s in enumerate(c["streams"])}
        its.append((f"ideal_{c['name']}_{c['tg']}", dict(spec_base(c), policy={"name": "IDEAL", "ideal": sd, "d_ideal": c["d"], "static_shadow": True})))
    a2 = build_cfgs(1, plats, T, solo, ideal, only=lambda g, hw, o, tg: g == "a2_2s_mha-ffn" and hw == "nofault" and o == "d0" and tg == "k1.5")
    for c in a2:
        sd = {i: ideal[(s, c["chunk"])] for i, s in enumerate(c["streams"])}
        its.append((f"ideal_{c['name']}_{c['tg']}", dict(spec_base(c), policy={"name": "IDEAL", "ideal": sd, "d_ideal": c["d"]})))
    new = new_items(cfgs, solo, ideal)
    t1 = time.time()
    res.update(run_all(jobs, its + new))
    t2 = time.time()
    ck2 = []
    for c in cfgs + a2:
        r = res[f"ideal_{c['name']}_{c['tg']}"]
        s = stat(r["runs"], c)
        rf = tab[(c["name"], c["tg"], "P7")]
        sh = {x: sum(q["shadow"].get(x, 0) for q in r["runs"]) for x in ("n", "differ", "differ_stream")}
        ck2.append({"config": c["name"], "dl": c["tg"], "S_new": s["S"], "S_e4b_P7": rf["S"], "rel": s["S"] / rf["S"] - 1, "L_new": s["L"], "L_e4b_P7": rf["L"], **sh,
                    "shadow": "static d-tail" if c in cfgs else "P7"})
    rows, cks = report(1, cfgs, res, tab, solo)
    write_csv("smoke_report.csv", rows)
    # walls -> estimate
    sec = [res[n]["process_wall"] for n, _ in new]
    tt = {"smoke_new_process_s_mean": E._mean(sec), "smoke_new_process_s_max": max(sec), "seconds": {"solo": t_solo, "new": t2 - t1, "total": time.time() - t0}}
    out = {"check1": ck1, "check2": ck2, **cks, "walls": tt, "rows": rows, "command": sys.argv, "artifact_commit": E._artifact_commit()}
    write_json("smoke.json", out)
    log("check1 " + json.dumps(round_all(ck1)))
    log("check2 " + json.dumps(round_all(ck2)))
    log("checks " + json.dumps(cks))
    log("walls " + json.dumps(round_all(tt)))
    keep = ["dl", "L_P1", "L_P4", "L_P5", "L_P7", "L_P7m", "L_C4", "L_C-S", "L_C-M", "L_C-S'", "L_C-Sr", "ret_C-S", "ret_C-M", "ret_C-S'", "den_C-S", "tau",
            "differ_C-S", "differ_stream_C-S", "differ_C-M", "differ_stream_C-M", "differ_C-S'", "differ_stream_C-S'", "C4_same_as_P4",
            "Ddev_C-S_0", "Ddev_C-S_1", "Ddev_C-M_0", "Ddev_C-M_1", "Dtail_C-S_0", "Dtail_C-S_1", "Drest_C-S_0", "Drest_C-S_1", "taildev_C-S_0", "taildev_C-S_1", "taildev_C-M_0", "taildev_C-M_1",
            "osc_D_changed", "osc_D_median", "osc_D_max", "cm_compile_s", "cm_forward_s", "cm_backward_s", "cm_backend_s"]
    for r in rows:
        log("row " + json.dumps(round_all({k: r.get(k) for k in keep})))
    if any(abs(r["rel_S"]) > 0.04 for r in ck1):
        raise DstuError("self-check 1")
    if any(abs(r["rel"]) > 0.04 or r["differ"] for r in ck2):
        raise DstuError("self-check 2")
    if cks["check3_audit"].get("n_mismatch") or cks["check3_audit"].get("n_issues", 0) < 1000:
        raise DstuError("self-check 3")
    if cks["check5_early_or_release_diff"] or cks["check5_chunk_bad"]:
        raise DstuError("self-check 5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["solo", "smoke", "s1", "s2", "s3", "child", "solo-child"])
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    if a.cmd == "child":
        child(a.spec)
    elif a.cmd == "solo-child":
        solo_child(a.spec)
    elif a.cmd == "solo":
        solo_step(a.jobs)
    elif a.cmd == "smoke":
        smoke(a.jobs)
    elif a.cmd == "s1":
        setting(1, a.jobs)
    elif a.cmd == "s3":
        setting(3, a.jobs)
    else:
        setting(2, a.jobs)


if __name__ == "__main__":
    try:
        main()
    except (DstuError, B.E4bError, E.E4Error, cpcm.CPCMError) as exc:
        print(f"[dstu] STOP: {exc}", flush=True)
        sys.exit(2)
