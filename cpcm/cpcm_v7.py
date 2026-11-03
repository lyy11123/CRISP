"""V7: timing (cpcm/CPCM_HANDOFF.md section 3; user 2026-11-03: serial, single process, three-part breakdown, profile
hot lines if the pure forward pass exceeds 50 ms and propose a speed-up without installing anything).

Graphs: V2's no-fault points (cpcm_out/v2/<task>_F0.pkl.gz): ffn, mha, moe, ffn14, opt.
compile   one task process per task, one after another (E3 inject of the V2 pre-fault mapping, no backend seeds):
          cpcm.compile_bb timed REPEAT times with perf_counter on the live objects (cpcm_v2.inject_eval,
          CPCM_COMPILE_REPEAT).
time      in this process, nothing else running: per task, median over R repeats of
            forward   cpcm.simulate(g, "bb")                         (replay only)
            backward  cpcm.Analysis(sc) + critical_path() (LST, slack, D, CP, CB, phi)
            ckpt      simulate with checkpoint_every=K (overhead of checkpoints)
moves     N random moves per task on the checkpointed base schedule, each evaluated by cpcm.incremental and by a full
          simulate; the two must agree on every event's (r, S, S_last, F), on T and on the activity multiset.
            computation move: a random computation to another device of the same kind (same duration);
            route move: a random single-branch message whose links are all co2co, re-routed by BFS over co2co links
              around one random link of its route (at most 2 hops longer), same per-hop duration.
          (A synthetic stand-in for an SA move: BusyBarn's moves also re-route the moved behaviour's messages.)
          Reported: incremental vs full time, truncation rate (spliced), re-simulated window share.
reroute   M single-message route moves through cpcm.reroute (incremental) vs a full recompute.
profile   if the forward median > 50 ms: cProfile of one forward pass, and per-line hit counts of _loop (sys.settrace).
Usage: python cpcm_v7.py compile ; python cpcm_v7.py time ; python cpcm_v7.py profile
"""
import collections
import cProfile
import gzip
import io
import json
import os
import pickle
import pstats
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpcm  # noqa: E402

V2 = os.path.join(HERE, "cpcm_out", "v2")
OUT = os.path.join(HERE, "cpcm_out")
TASKS = ["ffn", "mha", "ffn14", "opt", "moe"]
REPEAT_COMPILE = 5
R = {"ffn": 30, "mha": 15, "ffn14": 30, "opt": 30, "moe": 7}
K = 32
N_MOVES = {"ffn": 1000, "mha": 1000, "ffn14": 1000, "opt": 1000, "moe": 1000}
N_REROUTE = 200
SEED = 20261103


def load(name):
    with gzip.open(os.path.join(V2, f"{name}_F0.pkl.gz"), "rb") as f:
        return pickle.load(f)["graph"]


def med(f, n):
    ts = []
    out = None
    for _ in range(n):
        t0 = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), out


# ------------------------------------------------------------------ compile (serial task processes)
def compile_times():
    import cpcm_v2
    res = {}
    for name in TASKS:
        with gzip.open(os.path.join(V2, f"{name}_pre.pkl.gz"), "rb") as f:
            pre = pickle.load(f)
        spec_path = os.path.join(V2, "_tmp", f"spec_v7_{name}.pkl")
        os.makedirs(os.path.dirname(spec_path), exist_ok=True)
        with open(spec_path, "wb") as f:
            pickle.dump({"mapping": pre["mapping"], "faults": cpcm_v2.NOFAULT, "steps": None, "t_max": None,
                         "seeds": [], "preserve_routes": True}, f)
        os.environ["CPCM_COMPILE_REPEAT"] = str(REPEAT_COMPILE)
        sink = cpcm_v2.run_child(name, {"E3_MODE": "inject", "E3_SPEC": spec_path}, "v7compile")
        os.remove(spec_path)
        ts = sink["compile_times"]
        res[name] = {"events": sink["graph"].n, "compile_median_s": statistics.median(ts), "compile_min_s": min(ts),
                     "times": ts}
        print(f"[v7] compile {name}: {res[name]}", flush=True)
    with open(os.path.join(OUT, "v7_compile.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


# ------------------------------------------------------------------ moves
def _co2co_adj(g):
    adj = collections.defaultdict(list)
    for q, (nm, kd) in enumerate(zip(g.res_name, g.res_kind)):
        if kd == "co2co" and nm[0] == "link" and len(nm[1]) == 2:
            u, v = nm[1]
            adj[u].append((v, q))
    return adj


def _bfs(adj, src, dst, banned, maxlen):
    prev = {src: None}
    frontier = [src]
    depth = 0
    while frontier and depth < maxlen:
        depth += 1
        nxt = []
        for u in frontier:
            for v, q in adj.get(u, ()):
                if q in banned or v in prev:
                    continue
                prev[v] = (u, q)
                if v == dst:
                    path = []
                    x = v
                    while prev[x] is not None:
                        x, q2 = prev[x]
                        path.append(q2)
                    return path[::-1]
                nxt.append(v)
        frontier = nxt
    return None


def make_moves(g, n, rng):
    """List of (kind, event, payload); payload: new resource id, or (branches, link_dur)."""
    by_kind = collections.defaultdict(list)
    for q, kd in enumerate(g.res_kind):
        if kd.startswith("dev"):
            by_kind[kd].append(q)
    comps = [e for e in range(g.n) if g.kind[e] == cpcm.COMP and len(by_kind[g.res_kind[g.res_of[e]]]) > 1]
    adj = _co2co_adj(g)
    routable = []
    for e in range(g.n):
        if g.kind[e] != cpcm.COMM or len(g.ev_br[e]) != 1:
            continue
        b = g.ev_br[e][0]
        hops = list(range(g.br_lo[b], g.br_hi[b]))
        if hops and all(g.res_kind[g.hop_link[h]] == "co2co" and len(g.res_name[g.hop_link[h]][1]) == 2 for h in hops):
            routable.append(e)
    moves, tries = [], 0
    while len(moves) < n and tries < 50 * n:
        tries += 1
        if rng.random() < 0.5 and comps:
            e = rng.choice(comps)
            q = g.res_of[e]
            q2 = rng.choice([x for x in by_kind[g.res_kind[q]] if x != q])
            moves.append(("comp", e, q2))
        elif routable:
            e = rng.choice(routable)
            b = g.ev_br[e][0]
            hops = list(range(g.br_lo[b], g.br_hi[b]))
            links = [g.hop_link[h] for h in hops]
            src = g.res_name[links[0]][1][0]
            dst = g.res_name[links[-1]][1][1]
            ban = rng.choice(links)
            path = _bfs(adj, src, dst, {ban}, len(links) + 2)
            if path is None or path == links:
                continue
            d = g.hop_dur[hops[0]]
            moves.append(("route", e, ([path], {q: d for q in path})))
    return moves, {"comp_candidates": len(comps), "route_candidates": len(routable), "tries": tries}


def apply_move(g, mv):
    g2 = g.copy()
    kind, e, p = mv
    if kind == "comp":
        g2.res_of[e] = p
    else:
        g2.set_branches(e, p[0], p[1])
    return g2


def same(a, b):
    if a.T != b.T or a.r != b.r or a.S != b.S or a.S_last != b.S_last or a.F != b.F:
        return False
    acts = lambda s: sorted(zip(s.a_ev, s.a_res, s.a_s, s.a_f))
    return acts(a) == acts(b)


def timing():
    res = {}
    for name in TASKS:
        g = load(name)
        tf, tf_min, sc = med(lambda: cpcm.simulate(g, "bb"), R[name])
        tb, tb_min, an = med(lambda: _backward(sc), R[name])
        tck, _, base = med(lambda: cpcm.simulate(g, "bb", checkpoint_every=K), max(3, R[name] // 3))
        na = len(sc.a_s)
        d = {"events": g.n, "activities": na, "hops_listed": len(g.hop_link), "T": sc.T,
             "forward_median_s": tf, "forward_min_s": tf_min, "backward_median_s": tb, "backward_min_s": tb_min,
             "forward_us_per_event": tf / g.n * 1e6, "forward_us_per_activity": tf / na * 1e6,
             "backward_us_per_activity": tb / na * 1e6,
             "forward_with_checkpoints_median_s": tck, "checkpoints": len(base.ckpts), "checkpoint_every_ticks": K}
        # moves
        rng = random.Random(SEED)
        moves, mstat = make_moves(g, N_MOVES[name], rng)
        d["moves"] = mstat
        inc_t, full_t, spliced, window, mism, kinds = [], [], 0, [], 0, collections.Counter()
        copy_t = []
        for mv in moves:
            t0 = time.perf_counter()
            g2 = apply_move(g, mv)
            copy_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            si = cpcm.incremental(g2, base, [mv[1]])
            inc_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            sf = cpcm.simulate(g2, "bb")
            full_t.append(time.perf_counter() - t0)
            if not same(si, sf):
                mism += 1
            kinds[mv[0]] += 1
            if si.inc["splice_t"] is not None:
                spliced += 1
            end = si.inc["splice_t"] if si.inc["splice_t"] is not None else sf.T
            window.append((end - si.inc["restart_t"]) / sf.T if sf.T else 0)
        d.update({"n_moves": len(moves), "move_kinds": dict(kinds), "inc_mismatch": mism,
                  "inc_median_s": statistics.median(inc_t), "inc_mean_s": statistics.mean(inc_t),
                  "full_median_s": statistics.median(full_t), "full_mean_s": statistics.mean(full_t),
                  "inc_over_full_mean": statistics.mean(i / f for i, f in zip(inc_t, full_t)),
                  "inc_faster_share": sum(i < f for i, f in zip(inc_t, full_t)) / len(inc_t),
                  "truncation_rate": spliced / len(moves),
                  "resim_window_share_median": statistics.median(window), "resim_window_share_mean": statistics.mean(window),
                  "graph_copy_median_s": statistics.median(copy_t)})
        # single-message reroute through cpcm.reroute
        anb = cpcm.Analysis(base)
        rr = [m for m in moves if m[0] == "route"][:N_REROUTE]
        rt, rf, rmis, ok = [], [], 0, 0
        for kind, e, (brs, ld) in rr:
            t0 = time.perf_counter()
            out = cpcm.reroute(g, e, brs, ld, base=anb)
            rt.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            sf = cpcm.simulate(apply_move(g, (kind, e, (brs, ld))), "bb")
            rf.append(time.perf_counter() - t0)
            rmis += not same(out["schedule"], sf)
            ok += out["ok"]
        if rr:
            d.update({"reroute_n": len(rr), "reroute_median_s": statistics.median(rt),
                      "reroute_full_median_s": statistics.median(rf), "reroute_mismatch": rmis, "reroute_ok_share": ok / len(rr)})
        res[name] = d
        print(f"[v7] {name}: " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)
    with open(os.path.join(OUT, "v7_time.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


def _backward(sc):
    an = cpcm.Analysis(sc)
    an.critical_path()
    return an


# ------------------------------------------------------------------ profile
def profile():
    with open(os.path.join(OUT, "v7_time.json"), encoding="utf-8") as f:
        tm = json.load(f)
    out = {}
    for name in TASKS:
        if tm[name]["forward_median_s"] <= 0.050:
            continue
        g = load(name)
        pr = cProfile.Profile()
        pr.enable()
        cpcm.simulate(g, "bb")
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
        code = cpcm._loop.__code__
        hits = collections.Counter()

        def tracer(frame, event, arg):
            if frame.f_code is code or frame.f_code.co_name in ("at", "<lambda>", "_act"):
                if event == "line":
                    hits[(frame.f_code.co_name, frame.f_lineno)] += 1
                return tracer
            return None

        sys.settrace(tracer)
        cpcm.simulate(g, "bb")
        sys.settrace(None)
        lines = open(cpcm.__file__, encoding="utf-8").read().splitlines()
        total = sum(hits.values())
        top = [{"func": fn, "line": ln, "hits": h, "share": round(h / total, 4), "src": lines[ln - 1].strip()}
               for (fn, ln), h in hits.most_common(15)]
        out[name] = {"cprofile_top": s.getvalue(), "line_hits_total": total, "top_lines": top}
        print(f"[v7] profile {name}: total line events {total}")
        for t in top:
            print(f"   {t['share']:.3f} {t['hits']:>9} {t['func']}:{t['line']}  {t['src'][:90]}")
        print(s.getvalue()[:2500])
    with open(os.path.join(OUT, "v7_profile.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    {"compile": compile_times, "time": timing, "profile": profile}[sys.argv[1]]()
