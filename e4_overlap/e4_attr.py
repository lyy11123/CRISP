"""E4 analysis on a merged multi-stream event graph: E2's attribution with release times, cross-stream
blocking, the single-resource S_max bound (handoff 3.3) and the per-stream measures.

E2's `analyse` is reused unchanged (e2_attribution.py is imported, never edited); only its `Sim` is replaced
in memory, for the duration of the call, by SimR:
  - the earliest start of a dependency-free event is its stream's release time (E2: 0);
  - when the critical chain ends at such an event with s > 0, a "release" row covering [0, s] is put in
    front, so the rows still tile [0, a] (E2 check 2). It is not a wait, never a blocker, and carries no
    resource, so it is excluded from attribution, cross-stream shares and every bound;
  - longest paths "from time 0" (E2 check 4 and the longest path avoiding a resource) add the release time
    of the head activity they start from. Tails (longest path from an activity's finish, skip_res given)
    are unchanged.
With every release 0 SimR is E2's Sim.
"""
import collections
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e2_attribution"))
import e2_attribution as e2  # noqa: E402
sys.path.pop(0)

NEG = float("-inf")
KIND_CLASS = {"tensorcore": "compute", "vectorunit": "compute", "ch2ch": "d2d", "ddr2co": "ddr", "co2ddr": "ddr"}


def stream_of(tag):
    return tag[1] - 10


def kind_class(kind):
    return KIND_CLASS.get(kind, "onchip" if kind else None)


class SimR(e2.Sim):
    last = None

    def __init__(self, cap):
        super().__init__(cap)
        rel = cap.get("release", {})
        for t, e in self.ev.items():
            if not e["deps"]:
                self.r[t] = rel.get(stream_of(t), 0)
        self.rel_act = {}
        for k, A in self.act.items():
            if A["prev"] is None:
                A["can"] = self.r[A["tag"]]
                if not self.ev[A["tag"]]["deps"]:
                    self.rel_act[k] = self.r[A["tag"]]
        SimR.last = self

    def data_class(self, t):
        return "release" if t is None else super().data_class(t)

    def chain(self):
        last, rows, segs = super().chain()
        s0 = rows[0]["s"]
        if rows[0]["edge"] == "start" and s0 > 0:
            rows.insert(0, {"activity": "release", "tag": None, "name": "(release)", "type": "release",
                            "resource": "release", "resource_kind": "release", "r": 0, "can_start": 0, "s": 0,
                            "f": s0, "t_out": s0, "event_s": 0, "event_f": s0, "wait": 0, "pred": "",
                            "edge": "release", "critical_wait": 0, "contribution": s0})
        return last, rows, segs

    def longest_from_start(self, skip_res=None, removed=frozenset(), raw=False):
        L = super().longest_from_start(skip_res=skip_res, removed=removed)
        if raw or skip_res is not None:
            return L  # tails
        # "from time 0": only max(values) is used by E2 (check 4, longest path avoiding a resource)
        return {u: v + self.rel_act.get(u, 0) for u, v in L.items()}

    # ---- per-stream longest paths
    def stream_L(self, skip_res=None, removed=frozenset()):
        """Li[i][u] = longest time from u's start to the end of an activity of stream i (u's duration
        included), NEG when no activity of stream i is reachable; same edges as E2's longest_from_start."""
        succ, topo = self.dag()
        streams = sorted({stream_of(t) for t in self.ev})
        Li = {i: {} for i in streams}
        for u in reversed(topo):
            if u in removed:
                continue
            A = self.act[u]
            p = A["f"] - A["s"]
            su = stream_of(A["tag"])
            for i in streams:
                best = p if su == i else NEG
                Lv = Li[i]
                for v, kind, rk in succ[u]:
                    if v in removed or (rk is not None and rk == skip_res):
                        continue
                    x = Lv[v]
                    if x == NEG:
                        continue
                    c = p + x if kind == "fs" else x
                    if c > best:
                        best = c
                Lv[u] = best
        return Li

    def stream_ends(self, Li):
        return {i: max((self.rel_act.get(u, 0) + v for u, v in L.items() if v != NEG), default=NEG)
                for i, L in Li.items()}


def analyse(cap):
    """E2's analyse with SimR; returns (summary, rows, segs, reorder, ddr_rows, sim)."""
    orig = e2.Sim
    e2.Sim = SimR
    try:
        out = e2.analyse(cap)
    finally:
        e2.Sim = orig
    return out + (SimR.last,)


# ---------------------------------------------------------------- per-stream measures
def stream_times(cap):
    """Completion C_i = latest end of any event of stream i; release r_i as used by the backend."""
    C = collections.defaultdict(int)
    first = {}
    for t, e in cap["events"].items():
        s = stream_of(t)
        C[s] = max(C[s], e["f"])
    acts = [(s, f, t) for rec in list(cap["devices"].values()) + list(cap["links"].values()) for s, f, t in rec]
    for s, f, t in acts:
        i = stream_of(t)
        first[i] = min(first.get(i, s), s)
    return dict(C), first


def stream_measures(cap, T, ks):
    """Per stream latency / stretch, S_max, makespan, pipeline beat, deadline met per k, overlap."""
    rel = cap["release"]
    C, first = stream_times(cap)
    streams = sorted(C)
    lat = {i: C[i] - rel[i] for i in streams}
    stretch = {i: lat[i] / T[i] for i in streams}
    smax = max(stretch.values())
    make = max(C.values())
    beat = (make - min(rel.values())) / len(streams)
    met = {str(k): {i: stretch[i] <= k for i in streams} for k in ks}
    pair_overlap = {f"{i}-{j}": max(0, min(C[i], C[j]) - max(rel[i], rel[j]))
                    for a, i in enumerate(streams) for j in streams[a + 1:]}
    # time during which >= 2 streams are active ([r_i, C_i))
    pts = sorted([(rel[i], 1) for i in streams] + [(C[i], -1) for i in streams])
    act, last, multi = 0, None, 0
    for t, d in pts:
        if act >= 2 and last is not None:
            multi += t - last
        act += d
        last = t
    return {"C": C, "release": dict(rel), "first_activity": first, "latency": lat, "stretch": stretch,
            "S_max": smax, "argmax_stream": max(stretch, key=stretch.get), "makespan": make, "beat": beat,
            "met": met, "all_met": {k: all(v.values()) for k, v in met.items()},
            "pair_overlap": pair_overlap, "overlap_ge2": multi,
            "release_respected": all(first[i] >= rel[i] for i in streams)}


def slack_spread(rel, C, T, ks):
    """At each overlap start t (release of every stream after the first): slack_j = d_j - t - remaining_j for
    released, unfinished streams; remaining_j = max(0, T_j - (t - r_j)). Range and CV per k."""
    streams = sorted(rel, key=lambda i: (rel[i], i))
    out = []
    for i in streams[1:]:
        t = rel[i]
        live = [j for j in streams if rel[j] <= t < C[j]]
        row = {"t": t, "stream": i, "live": live}
        for k in ks:
            sl = [rel[j] + k * T[j] - t - max(0.0, T[j] - (t - rel[j])) for j in live]
            if len(sl) >= 2:
                m = sum(sl) / len(sl)
                sd = math.sqrt(sum((x - m) ** 2 for x in sl) / len(sl))
                row[str(k)] = {"slack": sl, "range": max(sl) - min(sl), "cv": sd / abs(m) if m else None}
            else:
                row[str(k)] = {"slack": sl, "range": None, "cv": None}
        out.append(row)
    return out


# ---------------------------------------------------------------- S_max bound on one resource
def _feas_parts(jobs, D):
    """jobs [(can, p, f)], deadlines D. Returns (Dmax, jobs4) with qq = Dmax - D >= 0, so that an order meets
    every deadline iff max(C + qq) <= Dmax (E2's _schrage/_jackson/_exact work on max(C + q))."""
    Dmax = max(D)
    return Dmax, [(c, p, Dmax - d, f) for (c, p, f), d in zip(jobs, D)]


def smax_resource(sim, rk, rel, T, ks, bisect_iter=50):
    """Single-resource S_max bound on resource rk (handoff 3.3). Every other resource's order and all durations
    stay fixed; release of an item = its current earliest start (as in E2). An item j reaches stream i with tail
    q_ji (longest path from j's finish to the end of stream i, without rk's own order edges); it must finish by
    D_j(k) = min_i (r_i + k T_i - q_ji). This is the handoff's 'deadline of its own stream' generalised to every
    stream the item can delay (other resources' order edges can lead into another stream); off_i = longest
    path from 0 to stream i's end avoiding rk's items. The optimal S_max on rk is the least k for which an
    order with all C_j <= D_j(k) exists and k >= max_i (off_i - r_i)/T_i; found by bisection on k with
    feasibility from: current order or Schrage EDD (achievable, k_ub), E2's exact branch and bound (<= 14
    items), preemptive Jackson EDF (necessary, k_lb)."""
    items = [sim._act_of(rk, t) for _, _, t in sim.res[rk]]
    streams = sorted(rel)
    Ls = sim.stream_L(skip_res=rk)
    Loff = sim.stream_L(removed=frozenset(items))
    off = sim.stream_ends(Loff)
    jobs, Q = [], []
    for k_ in items:
        A = sim.act[k_]
        p = A["f"] - A["s"]
        jobs.append((A["can"], p, A["f"]))
        Q.append({i: Ls[i][k_] - p for i in streams if Ls[i][k_] != NEG})
    k_floor = max(((off[i] - rel[i]) / T[i] for i in streams if off[i] != NEG), default=0.0)
    cur = max([k_floor] + [(f + q - rel[i]) / T[i] for (c, p, f), qd in zip(jobs, Q) for i, q in qd.items()])
    own = sum(1 for k_, qd in zip(items, Q) if set(qd) - {stream_of(sim.act[k_]["tag"])})

    def D_of(k):
        return [min(rel[i] + k * T[i] - q for i, q in qd.items()) if qd else float("inf") for qd in Q]

    def test(k):
        """(achievable, possible, exact_known) for deadlines at k."""
        if k < k_floor - 1e-12:
            return False, False, True
        D = D_of(k)
        if all(d == float("inf") for d in D):
            return True, True, True
        D = [d if d != float("inf") else max(x for x in D if x != float("inf")) + 1e18 for d in D]
        Dmax, j4 = _feas_parts(jobs, D)
        cur_v = max(f + q for _, _, q, f in j4)
        sch = e2._schrage(j4)
        if min(cur_v, sch) <= Dmax + 1e-9:
            return True, True, True
        if e2._jackson(j4) > Dmax + 1e-9:
            return False, False, True
        opt = e2._exact(j4, min(cur_v, sch))
        if opt is not None:
            return opt <= Dmax + 1e-9, opt <= Dmax + 1e-9, True
        return False, True, False

    def least(pred, lo, hi):
        if pred(lo):
            return lo
        for _ in range(bisect_iter):
            mid = (lo + hi) / 2
            if pred(mid):
                hi = mid
            else:
                lo = mid
        return hi

    unknown = [False]

    def ach(k):
        a, _, ex = test(k)
        unknown[0] |= not ex
        return a

    k_ub = least(ach, k_floor, cur)
    k_lb = least(lambda k: test(k)[1], k_floor, k_ub)

    def lmax(k):
        d = {i: rel[i] + k * T[i] for i in streams}
        off_l = max((off[i] - d[i] for i in streams if off[i] != NEG), default=NEG)
        D = D_of(k)
        fin = [x for x in D if x != float("inf")]
        if not fin:
            return {"best": off_l, "lb": off_l, "exact": True}
        D = [x if x != float("inf") else max(fin) + 1e18 for x in D]
        Dmax, j4 = _feas_parts(jobs, D)
        cur_v = max(f + q for _, _, q, f in j4)
        sch = e2._schrage(j4)
        opt = e2._exact(j4, min(cur_v, sch))
        best = (opt if opt is not None else min(cur_v, sch)) - Dmax
        lb = (opt - Dmax) if opt is not None else e2._jackson(j4) - Dmax
        return {"current": max(off_l, cur_v - Dmax), "best": max(off_l, best), "lb": max(off_l, lb),
                "exact": opt is not None}

    return {"resource": sim.res_name(rk), "resource_kind": sim.res_kind(rk), "n_items": len(items),
            "items_by_stream": dict(collections.Counter(stream_of(sim.act[k_]["tag"]) for k_ in items)),
            "items_reaching_other_streams": own,
            "off": off, "k_floor": k_floor, "S_cur_model": cur, "S_opt_ub": k_ub, "S_opt_lb": k_lb,
            "exact_everywhere": not unknown[0],
            "Lmax": {str(k): lmax(k) for k in ks}}
