"""Stage B preparation steps B0, B1, B2 (E2E_HANDOFF.md section 2). No search; cpcm / cpcm_sa / dstu are imported, not changed.

  python e2e_b.py b0 --jobs 8      out/b0/*.json, out/b0.json          random-tie T-hat of the candidate pools (no backend)
  python e2e_b.py b1 --jobs 6      out/b1/*.json, out/b1.json          W0 / W1 / no broadcast on fixed mappings, same process
  python e2e_b.py b2               out/b2/*                            cProfile of T-hat for C4 and of the hop backend
Rules: PROGRESS.md (B steps).
"""
import copy
import cProfile
import gzip
import json
import os
import pickle
import pstats
import random
import statistics
import subprocess
import sys
import time

import e2e_diag as D
import e2e_pilot as P

C = P.C
import numpy as np  # noqa: E402

HERE = P.HERE
SEEDS = list(range(1, 11))
KILL_GB, EST_GB, RAMP_S, LIMIT_S = 3.3, 1.5, 3, 7200
OUT_CFGS = ["mha", "ffn", "ffn14", "opt", "moe", "ffn_f4", "ffn_f5", "mha_f4", "mha_f5"]
ARMS = ["A1", "A2", "A3h"]
# B1 mappings named in the handoff: (cfg, arm, g, x)
B1_MAPS = [("pf_mha", "A1", 0, 3), ("pf_mha", "A1", 1, 4), ("pf_mha", "A1", 0, 5), ("pf_mha", "A2", 0, 1), ("pf_mha", "A2", 1, 1), ("pf_mha", "A3h", 0, 4), ("pf_mha", "A3h", 1, 6), ("pf_mha", "A3h", 0, 2),
           ("mha", "A1", 0, 4), ("mha", "A3h", 0, 2)]


def paths(cfg, name):
    base = C.OUT if cfg.startswith("pf_") else P.DECODE_OUT
    return os.path.join(base, "runs", name + ".pkl.gz"), os.path.join(base, "snaps", name + ".pkl.gz")


def load_state(spec):
    C._artifact_path()
    with gzip.open(spec["run"], "rb") as f:
        ctx = pickle.load(f)["meta"]["ctx"]
    with gzip.open(spec["snap"], "rb") as f:
        blob, hist = pickle.load(f)
    beha, data, events = pickle.loads(blob)
    hp = C.platform(spec["cfg"])
    hp.link_loads_dict.update(hist["link_loads_dict"])
    hp.link_loads_count.update(hist["link_loads_count"])
    return ctx, beha, data, events, hp


def broadcast(ctx, beha, data, events, hp):
    from add_communication import add_broadcast
    before = set(events)
    random.seed(C.BCAST_SEED)
    np.random.seed(C.BCAST_SEED)
    add_broadcast(data_tags=ctx["broadcast_datatags"], ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]], beha_dict=beha, data_dict=data, hardware_platform=hp,
                  event_dict=events, dijkstra_routing=ctx["dijkstra_routing"], alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])
    return set(events) - before


def compiled(evs, hp):
    deps = {t: set(e.dependency_set) for t, e in evs.items()}
    pth = {t: [list(b) for b in e.path_list] for t, e in evs.items() if e.event_type == "communication"}
    return C.cpcm.compile_bb(evs, hp, path_lists=pth, deps=deps)


# ------------------------------------------------------------------ driver: free memory watched during the run
def drive(tag, specs, jobs, child, name=lambda s: s["name"]):
    d = os.path.join(C.OUT, tag)
    os.makedirs(d, exist_ok=True)

    def say(msg):
        line = f"[{tag} {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(os.path.join(d, "driver.log"), "a", encoding="utf-8") as f:
            f.write(line + chr(10))
    q = [s for s in specs if not os.path.exists(os.path.join(d, name(s) + ".json"))]
    live, alerted, min_free, t_all, peak_seen = [], set(), P.free_gb(), time.time(), 0.0
    say(f"{len(q)} to do of {len(specs)}, free {min_free:.1f} GB, jobs at most {jobs}")
    while q or live:
        fr = P.free_gb()
        min_free = min(min_free, fr)
        for item in list(live):
            s, pr, t0, lf = item
            if pr.poll() is not None:
                lf.close()
                live.remove(item)
                ok = os.path.exists(os.path.join(d, name(s) + ".json"))
                if ok:
                    if tag != "b0":
                        say(f"{name(s)}: done, {time.time() - t0:.0f} s, min free so far {min_free:.1f} GB")
                else:
                    say(f"{name(s)}: FAILED exit {pr.returncode}")
            elif time.time() - t0 > LIMIT_S and name(s) not in alerted:
                alerted.add(name(s))
                say(f"ALERT {name(s)}: over 2 hours, still running")
        if fr < KILL_GB and live:
            s, pr, t0, lf = live.pop()
            pr.kill()
            pr.wait()
            lf.close()
            q.insert(0, s)
            jobs = max(1, len(live))
            say(f"KILL {name(s)}: free {fr:.1f} GB below {KILL_GB} GB; requeued, jobs now {jobs}")
        elif q and len(live) < jobs and fr - EST_GB >= KILL_GB:
            s = q.pop(0)
            p = os.path.join(d, name(s))
            with open(p + "_spec.json", "w", encoding="utf-8") as f:
                json.dump(s, f)
            lf = open(p + ".log", "w", encoding="utf-8")
            live.append((s, subprocess.Popen([sys.executable, os.path.abspath(__file__), child, p + "_spec.json", p + ".json"], cwd=HERE, stdout=lf, stderr=subprocess.STDOUT), time.time(), lf))
            if tag != "b0":
                say(f"{name(s)}: start, free {fr:.1f} GB, {len(live)} live")
            time.sleep(RAMP_S)
            continue
        elif q and not live:
            say(f"ALERT waiting: free {fr:.1f} GB")
            time.sleep(300)
        time.sleep(2)
    say(f"finished, {time.time() - t_all:.0f} s, min free {min_free:.2f} GB")
    return dict(wall_s=time.time() - t_all, min_free_gb=min_free)


# ------------------------------------------------------------------ B0
def b0_child(spec_path, out):
    spec = json.load(open(spec_path, encoding="utf-8"))
    ctx, beha, data, events, hp = load_state(spec)
    broadcast(ctx, beha, data, events, hp)
    t0 = time.perf_counter()
    g = compiled(events, hp)
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    det = C.cpcm.simulate(g, "bb").T
    t_det = time.perf_counter() - t0
    ties, wall = [], []
    for s in SEEDS:
        t0 = time.perf_counter()
        ties.append(C.cpcm.simulate(g, "bb", rng=random.Random(s)).T)
        wall.append(time.perf_counter() - t0)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(spec, det=det, ties=ties, t_compile=t_compile, t_det=t_det, t_tie_each=wall, n_events=g.n), f)


def b0_specs():
    out = []
    for cfg in ["pf_mha", "pf_ffn"] + OUT_CFGS:
        for arm in ARMS:
            for i, (g, x) in enumerate(C.variants(arm)):
                n = C.run_name(cfg, arm, g, x)
                run, snap = paths(cfg, n)
                with gzip.open(run, "rb") as f:
                    sel = pickle.load(f)["selected"]
                bk = sel["backend"]
                out.append(dict(name=n, cfg=cfg, arm=arm, g=g, x=x, idx=i, run=run, snap=snap, mean=C.bmean(bk), T_bc=sel["T_bc"], seeds=sorted(int(v[0]) for v in bk.values())))
    return out


def b0(jobs):
    sp = b0_specs()
    meta = drive("b0", sp, jobs, "b0_child")
    d = os.path.join(C.OUT, "b0")
    rows, missing = [], []
    for s in sp:
        p = os.path.join(d, s["name"] + ".json")
        if os.path.exists(p):
            rows.append(json.load(open(p, encoding="utf-8")))
        else:
            missing.append(s["name"])
    prior = {r["name"]: r for r in json.load(open(os.path.join(C.OUT, "diag_ties.json"), encoding="utf-8"))["rows"]}
    checks = dict(n=len(rows), missing=missing, det_differs=[r["name"] for r in rows if r["det"] != r["T_bc"]],
                  ties_differ_from_diag=[r["name"] for r in rows if r["name"] in prior and r["ties"] != prior[r["name"]]["ties"]], pilot_compared=sum(r["name"] in prior for r in rows))
    table = []
    for cfg in ["pf_mha", "pf_ffn"] + OUT_CFGS:
        by = {}
        for arm in ARMS:
            rs = sorted([r for r in rows if r["cfg"] == cfg and r["arm"] == arm], key=lambda r: r["idx"])
            if not rs:
                continue
            pick = lambda f: min(rs, key=lambda r: (f(r), r["idx"]))
            best = pick(lambda r: r["mean"])
            det = pick(lambda r: r["det"])
            tie = pick(lambda r: statistics.mean(r["ties"]))
            first, last = pick(lambda r: statistics.mean(r["ties"][:5])), pick(lambda r: statistics.mean(r["ties"][5:]))
            by[arm] = dict(best=best, det=det, tie=tie)
            walls = [w for r in rs for w in r["t_tie_each"]]
            table.append(dict(cfg=cfg, arm=arm, sample="in" if cfg.startswith("pf_") else "out", n=len(rs),
                              backend_best=best["mean"], backend_best_run=best["name"], det_run=det["name"], det_backend=det["mean"], det_gap=det["mean"] / best["mean"] - 1,
                              tie_run=tie["name"], tie_backend=tie["mean"], tie_gap=tie["mean"] / best["mean"] - 1, same_as_det=tie["name"] == det["name"], same_as_best=tie["name"] == best["name"],
                              first5_run=first["name"], last5_run=last["name"], first_last_same=first["name"] == last["name"], first5_is_all=first["name"] == tie["name"], last5_is_all=last["name"] == tie["name"],
                              first_gap=first["mean"] / best["mean"] - 1, last_gap=last["mean"] / best["mean"] - 1,
                              tie_eval_median_s=statistics.median(walls), tie_eval_max_s=max(walls), ten_seed_median_s=statistics.median(sum(r["t_tie_each"]) for r in rs), compile_median_s=statistics.median(r["t_compile"] for r in rs),
                              det_eval_median_s=statistics.median(r["t_det"] for r in rs), n_events=rs[0]["n_events"]))
        if "A1" in by:
            for row in [t for t in table if t["cfg"] == cfg]:
                for sel, key in (("best", "best"), ("det", "det"), ("tie", "tie")):
                    row[f"{sel}_vs_A1"] = by[row["arm"]][key]["mean"] / by["A1"][key]["mean"] - 1
    res = dict(meta=meta, checks=checks, table=table)
    with open(os.path.join(C.OUT, "b0.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(dict(meta=meta, checks=checks)))


# ------------------------------------------------------------------ B1
def b1_child(spec_path, out):
    spec = json.load(open(spec_path, encoding="utf-8"))
    sys.path.insert(0, os.path.join(P.ROOT, "dstu"))
    ctx, beha, data, events, hp = load_state(spec)
    import dstu2 as S
    from dstu import E
    from event_driver import event_driver
    cpcm = C.cpcm

    def backend(base, g):
        sims, tri = [], []
        for s in SEEDS:
            ev_k, hp_k = copy.deepcopy(base)
            random.seed(s)
            np.random.seed(s)
            t = tuple(int(v) for v in event_driver(events_dict=ev_k, hardware_platform=hp_k))
            sims.append((t[0], [int(ev_k[g.tag[e]].end_time) for e in range(g.n)]))
            tri.append(t)
            del ev_k, hp_k
        return sims, tri

    def ties(g):
        return [(lambda sc: (sc.T, list(sc.F)))(cpcm.simulate(g, "bb", rng=random.Random(s))) for s in SEEDS]

    def blocks(evs):
        ev, st = S.chunk_by(E.merged_events([pickle.dumps(evs, protocol=pickle.HIGHEST_PROTOCOL)]), hp, S.UNI["256all"])
        return dict(blocks_added=st["blocks_added"], messages_chunked=st["messages_chunked"], events_chunked=len(ev), bytes_conserved=st["bytes_conserved"], tree_conserved=st["tree_conserved"])

    def measure(evs, added, g):
        cat, nom = D.classify(g, hp, evs, added)
        aidx = {e for e in range(g.n) if g.tag[e] in added}
        sc = cpcm.simulate(g, "bb")
        base = copy.deepcopy((evs, hp))
        sims, tri = backend(base, g)
        m = dict(det=D.side(g, [(sc.T, list(sc.F))], cat, nom, aidx), ties=D.side(g, ties(g), cat, nom, aidx), backend=D.side(g, sims, cat, nom, aidx))
        m["triples"] = tri
        m["bc_events"] = len(aidx)
        m["bc_bytes"] = sum(int(evs[g.tag[e]].comm_bytes) for e in aidx)
        m["n_events"] = g.n
        return m

    res = {k: spec[k] for k in ("name", "cfg", "arm", "g", "x")}
    t00 = time.time()
    g = compiled(events, hp)
    res["none"] = measure(events, set(), g)
    res["none"]["chunk"] = blocks(events)
    main = {t: (set(e.dependency_set), set(e.issue_set)) for t, e in events.items()}
    print(f"[segment {time.strftime('%H:%M:%S')}] no broadcast done", flush=True)
    added = broadcast(ctx, beha, data, events, hp)
    g = compiled(events, hp)
    res["W0"] = measure(events, added, g)
    res["W0"]["chunk"] = blocks(events)
    print(f"[segment {time.strftime('%H:%M:%S')}] W0 done", flush=True)
    # ---- W1: the same W0 events (same routes) without the broadcast messages of QK^T and S
    drop = {t for t in added if "qkt" in events[t].event_name or "s_softmax" in events[t].event_name}
    names = {}
    for t in added:
        k = "dropped" if t in drop else "kept"
        d = names.setdefault(k, {})
        d[events[t].event_name] = d.get(events[t].event_name, 0) + int(events[t].comm_bytes)
    for t in drop:
        for dep in events[t].dependency_set:
            events[dep].issue_set.discard(t)
        del events[t]
    all_tags = set(events)
    dangling_dep = sum(1 for e in events.values() for x in e.dependency_set if x not in all_tags)
    dangling_issue = sum(1 for e in events.values() for x in e.issue_set if x not in all_tags)
    main_same = all((set(events[t].dependency_set), set(events[t].issue_set) - added) == (main[t][0], main[t][1]) for t in main)
    kept_issue_ok = all(t in events[next(iter(events[t].dependency_set))].issue_set for t in added - drop)
    res["filter"] = dict(dropped_events=len(drop), kept_events=len(added - drop), dropped_bytes=sum(names.get("dropped", {}).values()), kept_bytes=sum(names.get("kept", {}).values()),
                         dropped_names=len(names.get("dropped", {})), kept_names=sorted(names.get("kept", {})), dropped_name_list=sorted(names.get("dropped", {})),
                         dangling_dep=dangling_dep, dangling_issue=dangling_issue, main_events_same=main_same, kept_issue_ok=kept_issue_ok)
    if dangling_dep or dangling_issue or not main_same or not kept_issue_ok:
        res["ABORTED"] = "filter check failed; no W1 backend"
    else:
        g = compiled(events, hp)
        res["W1"] = measure(events, added - drop, g)
        res["W1"]["chunk"] = blocks(events)
    res["wall_s"] = time.time() - t00
    res["peak_wset_gb"] = P.peak() / 2 ** 30
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)


def b1(jobs):
    sp = []
    for cfg, arm, g, x in sorted(B1_MAPS, key=lambda m: m[0].startswith("pf_")):
        n = C.run_name(cfg, arm, g, x)
        run, snap = paths(cfg, n)
        sp.append(dict(name=("decode_" if not cfg.startswith("pf_") else "") + n, cfg=cfg, arm=arm, g=g, x=x, run=run, snap=snap))
    meta = drive("b1", sp, jobs, "b1_child")
    res = [json.load(open(os.path.join(C.OUT, "b1", s["name"] + ".json"), encoding="utf-8")) for s in sp if os.path.exists(os.path.join(C.OUT, "b1", s["name"] + ".json"))]
    with open(os.path.join(C.OUT, "b1.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, rows=res), f, indent=1)


# ------------------------------------------------------------------ B2
def top(prof, n=15):
    st = pstats.Stats(prof).stats
    total = sum(v[2] for v in st.values())
    def row(k, v):
        return dict(func=f"{os.path.basename(k[0])}:{k[1]}:{k[2]}", calls=v[1], tottime_s=v[2], tottime_share=v[2] / total, cumtime_s=v[3], cum_share=v[3] / total)
    return dict(total_s=total, by_tottime=[row(k, v) for k, v in sorted(st.items(), key=lambda kv: -kv[1][2])[:n]], by_cumtime=[row(k, v) for k, v in sorted(st.items(), key=lambda kv: -kv[1][3])[:n]])


def b2():
    """prefill FFN A3h, the read-only mapping (best by backend). Segments are written as soon as they end."""
    d = os.path.join(C.OUT, "b2")
    os.makedirs(d, exist_ok=True)
    sys.path.insert(0, os.path.join(P.ROOT, "dstu"))
    S0 = C.summarise(["pf_ffn"], ["A3h"])[("pf_ffn", "A3h")]["best_backend"]
    n = C.run_name("pf_ffn", "A3h", S0["g"], S0["x"])
    run, snap = paths("pf_ffn", n)
    spec = dict(name=n, cfg="pf_ffn", run=run, snap=snap)
    ctx, beha, data, events, hp = load_state(spec)
    import dstu as DS
    import dstu2 as S
    from dstu import E, cpcm
    from e4b_backend import P1, event_driver_e4b
    broadcast(ctx, beha, data, events, hp)
    res = dict(mapping=n)
    out = os.path.join(C.OUT, "b2.json")

    def flush():
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=float)
    t0 = time.perf_counter()
    ev, cst = S.chunk_by(E.merged_events([pickle.dumps(events, protocol=pickle.HIGHEST_PROTOCOL)]), hp, S.UNI["256all"])
    res["chunk_s"] = time.perf_counter() - t0
    res["blocks_added"], res["events_chunked"] = cst["blocks_added"], len(ev)
    flush()
    print("chunked", res["chunk_s"], flush=True)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    g2 = DS._graph(ev, hp)
    s2 = cpcm.simulate(g2, "hop")
    an = cpcm.Analysis(s2, sink_deadline=[s2.T] * g2.n)
    DS.tail_data(g2, s2, an, s2.T, range(g2.n))
    pr.disable()
    res["c4_profiled_wall_s"] = time.perf_counter() - t0
    res["c4_T"] = s2.T
    res["c4"] = top(pr)
    flush()
    print("c4", res["c4_profiled_wall_s"], flush=True)
    del an
    hpb = copy.deepcopy(hp)
    audit, released = [], {}
    random.seed(1)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    event_driver_e4b(ev, hpb, release={0: 0}, policy=P1(), released=released, audit=audit)
    pr.disable()
    res["hop_profiled_wall_s"] = time.perf_counter() - t0
    res["hop_latency"] = max(e.end_time for e in ev.values())
    res["hop"] = top(pr)
    res["peak_wset_gb"] = P.peak() / 2 ** 30
    flush()
    print("hop", res["hop_profiled_wall_s"], flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "b0_child":
        b0_child(*sys.argv[2:])
    elif cmd == "b1_child":
        b1_child(*sys.argv[2:])
    elif cmd == "b0":
        b0(int(sys.argv[3]))
    elif cmd == "b1":
        b1(int(sys.argv[3]))
    elif cmd == "b2":
        b2()
