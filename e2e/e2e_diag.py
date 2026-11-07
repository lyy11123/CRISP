"""T-hat error diagnostic of the GPT prefill pilot (user's release through the main session, 2026-11-07).

No search, no change to cpcm / cpcm_sa. For each saved mapping: T-hat before / after the template's final broadcast
(deterministic and with cpcm's diagnostic random ties, 10 seeds), backend seeds 1-10 before / after the broadcast,
critical chain by category on both sides, and what the broadcast is (bytes, destinations, times).

Chain (same rule on both sides, event level): from the latest-finishing event go back through the latest-finishing
data predecessor. For an event, r = max F of its predecessors (0 if none); residence = F - r; nominal = its own
duration (computation: dur; message: longest hop duration, hops of a message issue together, no store-and-forward);
wait = residence - nominal. Sum of residence over the chain = T, so the gap per category sums exactly to the gap in T.
The chains of the two sides need not contain the same events: this is a comparison of composition, not of one chain.

  python e2e_diag.py run --jobs 8     out/diag/*.json, out/diag.json
  python e2e_diag.py corr             out/diag_corr.json (the 84 existing results, no new backend)
  python e2e_diag.py ties84 8         out/diag_ties.json (random-tie T-hat of all 84, against the recorded backend means)
"""
import copy
import gzip
import json
import os
import pickle
import random
import statistics
import subprocess
import sys
import time

import e2e_pilot as P

C = P.C
import numpy as np  # noqa: E402

HERE = P.HERE
SEEDS = list(range(1, 11))
KILL_GB, EST_GB, RAMP_S, LIMIT_S = 3.3, 3.0, 30, 7200


def classify(g, hp, events, added):
    ddr = {k[:-1] for k in hp.ddr_dict}
    cat, nom = [], []
    for e in range(g.n):
        ev = events[g.tag[e]]
        if ev.event_type == "computation":
            cat.append("comp:" + str(ev.comp_device))
            nom.append(g.dur[e])
            continue
        hops = [h for b in g.ev_br[e] for h in range(g.br_lo[b], g.br_hi[b])]
        nom.append(max(g.hop_dur[h] for h in hops))
        kinds = {g.res_kind[g.hop_link[h]] for h in hops}
        if g.tag[e] in added:
            cat.append("broadcast")
        elif tuple(ev.source_location) in ddr:
            cat.append("ddr_read")
        elif "ch2ch" in kinds:
            cat.append("comm_d2d")
        else:
            cat.append("comm_onchip")
    return cat, nom


def chain(g, F, cat, nom):
    """{category: [nominal, wait]} on the critical chain; sum of both = max F."""
    e = max(range(g.n), key=lambda x: (F[x], x))
    res, n = {}, 0
    while True:
        ps = g.pred[g.pred_ptr[e]:g.pred_ptr[e + 1]]
        p = max(ps, key=lambda x: (F[x], x)) if len(ps) else None
        r = F[p] if p is not None else 0
        d = res.setdefault(cat[e], [0, 0])
        d[0] += nom[e]
        d[1] += F[e] - r - nom[e]
        n += 1
        if p is None:
            break
        e = p
    res["_events"] = [n, 0]
    return res


def waits(g, F, cat, nom):
    """{category: total wait over ALL events} (residence - nominal)."""
    res = {}
    for e in range(g.n):
        ps = g.pred[g.pred_ptr[e]:g.pred_ptr[e + 1]]
        r = max((F[x] for x in ps), default=0)
        res[cat[e]] = res.get(cat[e], 0) + F[e] - r - nom[e]
    return res


def mean_dicts(ds):
    keys = {k for d in ds for k in d}
    out = {}
    for k in keys:
        v = [d.get(k, 0) for d in ds]
        if any(isinstance(x, list) for x in v):
            w = [x if isinstance(x, list) else [0, 0] for x in v]
            out[k] = [sum(x[0] for x in w) / len(w), sum(x[1] for x in w) / len(w)]
        else:
            out[k] = sum(v) / len(v)
    return out


def side(g, sims, cat, nom, added_idx):
    """Aggregate of several (T, F) of one graph."""
    chains = [chain(g, F, cat, nom) for _, F in sims]
    w = [waits(g, F, cat, nom) for _, F in sims]
    res = dict(T=[t for t, _ in sims], chain=mean_dicts(chains), waits_all=mean_dicts(w))
    if added_idx:
        main = [i for i in range(g.n) if i not in added_idx]
        res["main_end"] = statistics.mean(max(F[i] for i in main) for _, F in sims)
        res["bc_first_ready"] = statistics.mean(min(max((F[x] for x in g.pred[g.pred_ptr[i]:g.pred_ptr[i + 1]]), default=0) for i in added_idx) for _, F in sims)
        res["bc_end"] = statistics.mean(max(F[i] for i in added_idx) for _, F in sims)
        res["chain_ends_in_broadcast"] = sum(1 for _, F in sims if max(range(g.n), key=lambda x: (F[x], x)) in added_idx)
    return res


def child(spec_path, out):
    spec = json.load(open(spec_path, encoding="utf-8"))
    C._artifact_path()
    from add_communication import add_broadcast
    from event_driver import event_driver
    cpcm = C.cpcm
    with gzip.open(spec["run"], "rb") as f:
        ctx = pickle.load(f)["meta"]["ctx"]
    with gzip.open(spec["snap"], "rb") as f:
        blob, hist = pickle.load(f)
    beha, data, events = pickle.loads(blob)
    hp = C.platform(spec["cfg"])
    hp.link_loads_dict.update(hist["link_loads_dict"])
    hp.link_loads_count.update(hist["link_loads_count"])

    def compiled(evs):
        deps = {t: set(e.dependency_set) for t, e in evs.items()}
        paths = {t: [list(b) for b in e.path_list] for t, e in evs.items() if e.event_type == "communication"}
        return cpcm.compile_bb(evs, hp, path_lists=paths, deps=deps)

    def backend(base, g):
        sims, triples, wall = [], [], []
        for s in SEEDS:
            ev_k, hp_k = copy.deepcopy(base)
            random.seed(s)
            np.random.seed(s)
            t0 = time.time()
            tri = tuple(int(v) for v in event_driver(events_dict=ev_k, hardware_platform=hp_k))
            wall.append(time.time() - t0)
            sims.append((tri[0], [int(ev_k[g.tag[e]].end_time) for e in range(g.n)]))
            triples.append(tri)
            del ev_k, hp_k
        return sims, triples, wall

    def tie_sims(g):
        out_ = []
        for s in SEEDS:
            sc_ = cpcm.simulate(g, "bb", rng=random.Random(s))
            out_.append((sc_.T, list(sc_.F)))
        return out_

    res = {k: spec[k] for k in ("label", "cfg", "arm", "run_name", "err_recorded", "why")}
    # ---- before the broadcast
    gp = compiled(events)
    cp, np_ = classify(gp, hp, events, set())
    sp = cpcm.simulate(gp, "bb")
    res["pre"] = dict(det=side(gp, [(sp.T, list(sp.F))], cp, np_, None), ties=side(gp, tie_sims(gp), cp, np_, None))
    base = copy.deepcopy((events, hp))
    sims, tri, wall = backend(base, gp)
    res["pre"]["backend"] = side(gp, sims, cp, np_, None)
    res["pre"]["backend"]["triples"] = tri
    res["pre"]["backend_wall_median_s"] = statistics.median(wall)
    del base, sims
    # ---- after the broadcast (as final_eval)
    before = set(events)
    random.seed(C.BCAST_SEED)
    np.random.seed(C.BCAST_SEED)
    add_broadcast(data_tags=ctx["broadcast_datatags"], ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]], beha_dict=beha, data_dict=data, hardware_platform=hp,
                  event_dict=events, dijkstra_routing=ctx["dijkstra_routing"], alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])
    added = set(events) - before
    g = compiled(events)
    cat, nom = classify(g, hp, events, added)
    aidx = {e for e in range(g.n) if g.tag[e] in added}
    sc = cpcm.simulate(g, "bb")
    res["post"] = dict(det=side(g, [(sc.T, list(sc.F))], cat, nom, aidx), ties=side(g, tie_sims(g), cat, nom, aidx))
    base = copy.deepcopy((events, hp))
    sims, tri, wall = backend(base, g)
    res["post"]["backend"] = side(g, sims, cat, nom, aidx)
    res["post"]["backend"]["triples"] = tri
    res["post"]["backend_wall_median_s"] = statistics.median(wall)
    # ---- what the broadcast is
    names = {}
    for e in aidx:
        ev = events[g.tag[e]]
        d = names.setdefault(ev.event_name, dict(events=0, bytes=0, nominal_sum=0, targets=set(), sources=set(), hops=0, links=set()))
        d["events"] += 1
        d["bytes"] += int(ev.comm_bytes)
        d["nominal_sum"] += nom[e]
        d["targets"].add(len(ev.target_location))
        d["sources"].add(str(ev.source_location))
        for b in g.ev_br[e]:
            for h in range(g.br_lo[b], g.br_hi[b]):
                d["hops"] += 1
                d["links"].add(g.hop_link[h])
    bys = sorted(int(events[g.tag[e]].comm_bytes) for e in aidx)
    tg = next(iter(events[g.tag[e]].target_location for e in aidx)) if aidx else []
    ms = [int(ev.comm_bytes) for t, ev in events.items() if ev.event_type == "communication" and t not in added]
    res["broadcast"] = dict(
        data_tags=len(ctx["broadcast_datatags"]), events=len(aidx), bytes_total=sum(bys), bytes_min=bys[0] if bys else 0, bytes_median=statistics.median(bys) if bys else 0, bytes_max=bys[-1] if bys else 0,
        targets=sorted(str(x) for x in tg), nominal_sum=sum(nom[e] for e in aidx), successors=sum(g.succ_ptr[e + 1] - g.succ_ptr[e] for e in aidx),
        link_kinds=sorted({str(g.res_kind[g.hop_link[h]]) for e in aidx for b in g.ev_br[e] for h in range(g.br_lo[b], g.br_hi[b])}),
        by_name={k: dict(events=v["events"], bytes=v["bytes"], nominal_sum=v["nominal_sum"], targets=sorted(v["targets"]), sources=len(v["sources"]), hops=v["hops"], links=len(v["links"])) for k, v in names.items()},
        other_comm_bytes_total=sum(ms), other_comm_events=len(ms))
    res["categories"] = {c: cat.count(c) for c in set(cat)}
    res["link_kinds_all"] = sorted({str(k) for k in g.res_kind})
    res["peak_wset_gb"] = P.peak() / 2 ** 30
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)


def specs():
    S = C.summarise(["pf_mha", "pf_ffn"], ["A1", "A2", "A3h"])
    out = []

    def add(cfg, arm, g, x, why):
        r = next(r for r in S[(cfg, arm)]["runs"] if r["g"] == g and r["x"] == x)
        n = C.run_name(cfg, arm, g, x)
        out.append(dict(label=n, cfg=cfg, arm=arm, run_name=n, why=why, err_recorded=(r["T_bc"] - r["mean"]) / r["mean"], run=os.path.join(C.OUT, "runs", n + ".pkl.gz"), snap=os.path.join(C.OUT, "snaps", n + ".pkl.gz")))

    def extreme(cfg, arm, largest):
        rs = S[(cfg, arm)]["runs"]
        f = lambda r: abs((r["T_bc"] - r["mean"]) / r["mean"])
        r = max(rs, key=f) if largest else min(rs, key=f)
        return r["g"], r["x"]
    add("pf_ffn", "A3h", 1, 123, "named by the user")
    add("pf_ffn", "A3h", *extreme("pf_ffn", "A3h", False), "A3h smallest error")
    rep = json.load(open(os.path.join(P.DECODE_OUT, "report_mha.json"), encoding="utf-8"))
    for arm in ("A1", "A3h"):
        t = next(t for t in rep["table"] if t["cfg"] == "mha" and t["arm"] == arm)
        g, x = t["run"].split("_")
        n = C.run_name("mha", arm, int(g[1:]), int(x))
        out.append(dict(label="decode_" + n, cfg="mha", arm=arm, run_name=n, why="decode MHA, same measures", err_recorded=t["err_Tbc_vs_backend_median"], run=os.path.join(P.DECODE_OUT, "runs", n + ".pkl.gz"), snap=os.path.join(P.DECODE_OUT, "snaps", n + ".pkl.gz")))
    add("pf_mha", "A2", 0, 1, "named by the user")
    add("pf_mha", "A3h", 0, 4, "named by the user")
    add("pf_mha", "A1", *extreme("pf_mha", "A1", True), "A1 largest error")
    add("pf_mha", "A1", *extreme("pf_mha", "A1", False), "A1 smallest error")
    add("pf_mha", "A2", *extreme("pf_mha", "A2", False), "A2 smallest error")
    add("pf_mha", "A3h", *extreme("pf_mha", "A3h", False), "A3h smallest error")
    return out


def run(jobs, only=None):
    d = os.path.join(C.OUT, "diag")
    os.makedirs(d, exist_ok=True)
    logp = os.path.join(d, "driver.log")

    def say(msg):
        line = f"[diag {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(line + chr(10))
    sp = [s for s in specs() if not only or s["label"] in only]
    q = [s for s in sp if not os.path.exists(os.path.join(d, s["label"] + ".json"))]
    est, live, alerted, min_free = {}, [], set(), P.free_gb()
    say(f"{len(q)} mappings to do, free {min_free:.1f} GB, jobs at most {jobs}")
    while q or live:
        fr = P.free_gb()
        min_free = min(min_free, fr)
        for item in list(live):
            s, pr, t0, lf = item
            if pr.poll() is not None:
                lf.close()
                live.remove(item)
                path = os.path.join(d, s["label"] + ".json")
                if os.path.exists(path):
                    r = json.load(open(path, encoding="utf-8"))
                    est[s["cfg"]] = max(est.get(s["cfg"], 0), r["peak_wset_gb"])
                    say(f"{s['label']}: done, {time.time() - t0:.0f} s, peak {r['peak_wset_gb']:.2f} GB, min free so far {min_free:.1f} GB")
                else:
                    say(f"{s['label']}: FAILED exit {pr.returncode}")
            elif time.time() - t0 > LIMIT_S and s["label"] not in alerted:
                alerted.add(s["label"])
                say(f"ALERT {s['label']}: over 2 hours, still running")
        need = est.get(q[0]["cfg"], EST_GB) if q else 0
        if fr < KILL_GB and live:
            s, pr, t0, lf = live.pop()
            pr.kill()
            pr.wait()
            lf.close()
            q.insert(0, s)
            jobs = max(1, len(live))
            say(f"KILL {s['label']}: free {fr:.1f} GB below {KILL_GB} GB; requeued, jobs now {jobs}")
        elif q and len(live) < jobs and fr - need >= KILL_GB:
            s = q.pop(0)
            p = os.path.join(d, s["label"])
            with open(p + "_spec.json", "w", encoding="utf-8") as f:
                json.dump(s, f)
            lf = open(p + ".log", "w", encoding="utf-8")
            live.append((s, subprocess.Popen([sys.executable, os.path.abspath(__file__), "child", p + "_spec.json", p + ".json"], cwd=HERE, stdout=lf, stderr=subprocess.STDOUT), time.time(), lf))
            say(f"{s['label']}: start, free {fr:.1f} GB, {len(live)} live")
            time.sleep(RAMP_S)
        elif q and not live:
            say(f"ALERT waiting: free {fr:.1f} GB, need {need + KILL_GB:.1f} GB")
            time.sleep(600)
        time.sleep(5)
    say(f"finished, min free {min_free:.2f} GB")
    res = [json.load(open(os.path.join(d, s["label"] + ".json"), encoding="utf-8")) for s in sp if os.path.exists(os.path.join(d, s["label"] + ".json"))]
    with open(os.path.join(C.OUT, "diag.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


def corr():
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            for k in range(i, j + 1):
                r[idx[k]] = (i + j) / 2
            i = j + 1
        return r

    def pear(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(set(a)) > 1 and len(set(b)) > 1 else None
    rows = []
    for cfg in ("pf_mha", "pf_ffn"):
        S = C.summarise([cfg], ["A1", "A2", "A3h"])
        for arm in ("A1", "A2", "A3h"):
            for r in S[(cfg, arm)]["runs"]:
                n = C.run_name(cfg, arm, r["g"], r["x"])
                with gzip.open(os.path.join(C.OUT, "runs", n + ".pkl.gz"), "rb") as f:
                    fe = pickle.load(f)
                bk = fe["selected"]["backend"]
                T = [v[0] for v in bk.values()]
                m = statistics.mean(T)
                comm = statistics.mean(v[2] for v in bk.values()) / m
                rows.append(dict(cfg=cfg, arm=arm, g=r["g"], x=r["x"], err=(r["T_bc"] - m) / m, abs_err=abs(r["T_bc"] - m) / m, noise=(max(T) - min(T)) / m, comm_share=comm, mean=m, T_bc=r["T_bc"], T_pre=r["T_pre"], ratio_bc=r["T_bc"] / r["T_pre"], seeds=sorted(T)))
    out = dict(rows=rows, corr={})
    groups = {"all": rows, "pf_mha": [r for r in rows if r["cfg"] == "pf_mha"], "pf_ffn": [r for r in rows if r["cfg"] == "pf_ffn"]}
    for c in ("pf_mha", "pf_ffn"):
        for g in (0, 1):
            groups[f"{c}_g{g}"] = [r for r in rows if r["cfg"] == c and r["g"] == g]
        for a in ("A1", "A2", "A3h"):
            groups[f"{c}_{a}"] = [r for r in rows if r["cfg"] == c and r["arm"] == a]
    for k, rs in groups.items():
        e = [r["abs_err"] for r in rs]
        out["corr"][k] = {"n": len(rs)}
        for f in ("comm_share", "noise", "ratio_bc", "mean"):
            v = [r[f] for r in rs]
            out["corr"][k][f] = dict(pearson=pear(e, v), spearman=pear(rank(e), rank(v)))
    out["ratio_bc"] = {f"{c}_{a}": [min(r["ratio_bc"] for r in rows if r["cfg"] == c and r["arm"] == a), statistics.median(r["ratio_bc"] for r in rows if r["cfg"] == c and r["arm"] == a), max(r["ratio_bc"] for r in rows if r["cfg"] == c and r["arm"] == a)] for c in ("pf_mha", "pf_ffn") for a in ("A1", "A2", "A3h")}
    with open(os.path.join(C.OUT, "diag_corr.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["corr"])[:4000])
    print(out["ratio_bc"])


def ties_child(spec_path, out):
    """Deterministic and random-tie T-hat of one saved mapping, after the final broadcast. No backend."""
    spec = json.load(open(spec_path, encoding="utf-8"))
    C._artifact_path()
    from add_communication import add_broadcast
    cpcm = C.cpcm
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
    deps = {t: set(e.dependency_set) for t, e in events.items()}
    paths = {t: [list(b) for b in e.path_list] for t, e in events.items() if e.event_type == "communication"}
    g = cpcm.compile_bb(events, hp, path_lists=paths, deps=deps)
    det = cpcm.simulate(g, "bb").T
    ties = [cpcm.simulate(g, "bb", rng=random.Random(s)).T for s in SEEDS]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(spec, det=det, ties=ties), f)


def ties84(jobs):
    """All 84 mappings: random-tie T-hat (seeds 1-10) against the backend means already recorded."""
    from concurrent.futures import ThreadPoolExecutor
    d = os.path.join(C.OUT, "diag_ties")
    os.makedirs(d, exist_ok=True)
    todo = []
    for cfg in ("pf_mha", "pf_ffn"):
        S = C.summarise([cfg], ["A1", "A2", "A3h"])
        for arm in ("A1", "A2", "A3h"):
            for r in S[(cfg, arm)]["runs"]:
                n = C.run_name(cfg, arm, r["g"], r["x"])
                todo.append(dict(cfg=cfg, arm=arm, g=r["g"], x=r["x"], name=n, mean=r["mean"], T_bc=r["T_bc"], run=os.path.join(C.OUT, "runs", n + ".pkl.gz"), snap=os.path.join(C.OUT, "snaps", n + ".pkl.gz")))
    mins = [P.free_gb()]

    def one(s):
        p = os.path.join(d, s["name"])
        if os.path.exists(p + ".json"):
            return
        while P.free_gb() < 4.0:
            time.sleep(30)
        mins.append(P.free_gb())
        with open(p + "_spec.json", "w", encoding="utf-8") as f:
            json.dump(s, f)
        with open(p + ".log", "w", encoding="utf-8") as lf:
            subprocess.call([sys.executable, os.path.abspath(__file__), "ties_child", p + "_spec.json", p + ".json"], cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
        mins.append(P.free_gb())
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(one, todo))
    rows = [json.load(open(os.path.join(d, s["name"] + ".json"), encoding="utf-8")) for s in todo]
    out = {"min_free_gb": min(mins), "det_equals_recorded": sum(r["det"] == r["T_bc"] for r in rows), "n": len(rows), "by": {}}
    for cfg in ("pf_mha", "pf_ffn"):
        for arm in ("A1", "A2", "A3h"):
            rs = [r for r in rows if r["cfg"] == cfg and r["arm"] == arm]
            ed = [(r["det"] - r["mean"]) / r["mean"] for r in rs]
            et = [(statistics.mean(r["ties"]) - r["mean"]) / r["mean"] for r in rs]
            out["by"][f"{cfg}_{arm}"] = dict(det_median=statistics.median(ed), det_maxabs=max(map(abs, ed)), det_over5=sum(abs(x) > .05 for x in ed), det_neg=sum(x < 0 for x in ed),
                                             ties_median=statistics.median(et), ties_maxabs=max(map(abs, et)), ties_over5=sum(abs(x) > .05 for x in et), ties_neg=sum(x < 0 for x in et))
    out["rows"] = rows
    with open(os.path.join(C.OUT, "diag_ties.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "child":
        child(*sys.argv[2:])
    elif cmd == "run":
        only = sys.argv[4:] if len(sys.argv) > 4 else None
        run(int(sys.argv[3]), only)
    elif cmd == "corr":
        corr()
    elif cmd == "ties_child":
        ties_child(*sys.argv[2:])
    elif cmd == "ties84":
        ties84(int(sys.argv[2]))
