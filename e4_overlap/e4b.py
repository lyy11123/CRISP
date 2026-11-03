"""E4b (handoff E4B_HANDOFF.md): stream policies on the per-hop arbitration backend (e4b_backend.py), step 1:
policy framework, P8 partition routing, self-checks 1-5 and 7 on a1 512+8192 no fault Delta 0 (check 3 also on a2
MHA+FFN 256 KB D2D-only, same k, Delta 0).

Policies (key, smaller first; ln = longest remaining branch through the link):
  P1 (-ln)                       per-hop default
  P2 (r_i, -ln)  FCFS            P3 (T_i, -ln)  SJF            P4 (d_i, -ln)  static EDF
  P5 member (rank_i, -ln)        fixed order; P5 = best member afterwards by mean L_hat_max
  P6 (d_i - t - R_i(t), -ln)     DSTU-s, R_i(t) ideal remaining work (below)
  P7 (d_i - t - tail_e, -ln)     DSTU-e, tail from the solo graph (below)
  P8 P1 on partitioned routes    2 streams only; each direction of each adjacent Die pair has 2 D2D links, the
                                 sorted-first goes to stream 0, the other to stream 1; each stream is re-routed by
                                 E3's R0 with the other stream's links failed; run on the full platform.
Solo reference (per stream, same hardware state and chunking, per-hop P1, release 0): backend seed 1 run with its
event graph (cap).
  R_i(t): work weight w_e = computation duration, or the sum of a communication's link-record durations, in the
    solo run; W(tau) = sum of w_e over events finished by tau in the solo run (non-decreasing step function).
    done_i(t) = sum of w_e over stream i's events retired by t in the run being simulated; tau_i(t) = the
    earliest solo finish instant with W >= done_i(t) (0 when done = 0); R_i(t) = max(0, T_i - tau_i(t)).
  tail: E2 DAG of the solo cap (SimR, resource-order edges of the solo schedule included). rest(e) = longest path
    from e's terminal activity's end to the stream's end = L[term(e)] - dur(term(e)) (e4_attr.stream_L).
    Unchunked message: tail_e = (f - s of e in the solo run) + rest(e).
    Chunked message m (>= 2 blocks; handoff 2.2): on link l, tail_b(t) = sum over m's blocks not yet issued on l
    (b included) of their occupancy on l - b's own occupancy on l + rest(last-finishing block of m in the solo run).
  T_i, d_i: T_i = the 6-seed per-hop solo mean measured in E4a (e4_out/hop/hop_state.json; chunked:
    e4_out/diag/sims/chunk_single_<key>_<chunk>), d_i = r_i + k_i T_i.

  python e4b.py step1 --jobs 12
Writes e4b_out/.
"""
import argparse
import bisect
import collections
import copy
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

import e4_attr as A
import e4_grid as G
import e4_hop as H
import e4_overlap as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e4b_out")
STREAMS = os.path.join(OUT, "streams")
LIMIT = 1200  # one evaluation > 20 min: stop
SEEDS = [1, 2, 3]
CB = E.CHUNK_ID_BASE


class E4bError(Exception):
    pass


def rt(tag):
    """stream-relative tag (stream position dropped)"""
    return (tag[0],) + tuple(tag[2:])


def group_of(r):
    return r[:-1] + (r[-1] % CB,)


def load_stream(key):
    p = os.path.join(STREAMS, f"{key}.pkl.gz")
    if os.path.exists(p):
        with gzip.open(p, "rb") as f:
            return pickle.load(f)
    return E.load_stream(key)


# ---------------------------------------------------------------- policies (child side)
class Fixed:
    def __init__(self, rank):
        self.rank = rank

    def key(self, e, link, ln, t):
        return (self.rank[e.event_tag[1] - 10], -ln)

    def on_issue(self, e, link, t):
        pass

    def on_finish(self, e, t):
        pass

    def on_tick(self, t, ready):
        pass


class DstuS(Fixed):
    def __init__(self, solo, d, T, release, ref_rank):
        self.d, self.T, self.rel = d, T, release
        self.w = {i: s["w"] for i, s in solo.items()}
        self.Wf = {i: s["W_f"] for i, s in solo.items()}
        self.Wc = {i: s["W_c"] for i, s in solo.items()}
        self.n = {i: s["n_events"] for i, s in solo.items()}
        self.done = collections.Counter()
        self.ndone = collections.Counter()
        self.ref = tuple(sorted(T, key=lambda i: ref_rank[i]))
        self.log = {"ticks": 0, "rank_changes": 0, "ticks_not_ref": 0, "last_t": None, "last_rank": None,
                    "flip_ticks": 0, "flips": 0, "flips_both_neg": 0, "flips_one_neg": 0, "last_flip_t": None,
                    "first_flip_t": None}
        self.prev_ready_rank = None

    def R(self, i):
        x = self.done[i]
        if x <= 0:
            tau = 0
        else:
            j = bisect.bisect_left(self.Wc[i], x - 1e-9)
            tau = self.Wf[i][min(j, len(self.Wf[i]) - 1)]
        return max(0.0, self.T[i] - tau)

    def slack(self, i, t):
        return self.d[i] - t - self.R(i)

    def score(self, i, t):
        return self.slack(i, t)

    def stream_key(self, i, t):
        """stream-level order used by the slack log and the flip count"""
        return self.score(i, t)

    def key(self, e, link, ln, t):
        return (self.score(e.event_tag[1] - 10, t), -ln)

    def on_finish(self, e, t):
        i = e.event_tag[1] - 10
        r = rt(e.event_tag)
        if r not in self.w[i]:
            raise E4bError(f"event {e.event_tag} missing from the solo trajectory of stream {i}")
        self.done[i] += self.w[i][r]
        self.ndone[i] += 1

    def on_tick(self, t, ready):
        if t == self.log["last_t"]:
            return
        self.log["last_t"] = t
        self.flip_count(t, ready)
        live =[i for i in self.T if self.rel[i] <= t and self.ndone[i] < self.n[i]]
        if len(live) < 2:
            return
        rank = tuple(sorted(live, key=lambda i: (self.stream_key(i, t), i)))
        ref = tuple(i for i in self.ref if i in live)
        self.log["ticks"] += 1
        if rank != ref:
            self.log["ticks_not_ref"] += 1
        if self.log["last_rank"] is not None and rank != self.log["last_rank"] and \
                set(rank) == set(self.log["last_rank"]):
            self.log["rank_changes"] += 1
        self.log["last_rank"] = rank

    def flip_count(self, t, ready):
        """user request 2026-11-02: on arbitration ticks where >= 2 streams have a ready communication, order the
        streams by their key; every pair of streams present in two consecutive such ticks whose order swaps is one
        flip (both / one slack already negative at the flip recorded)."""
        P = {e.event_tag[1] - 10 for e in ready}
        if len(P) < 2:
            return
        self.log["flip_ticks"] += 1
        sc = {i: self.stream_key(i, t) for i in P}
        cur = {i: (sc[i], i) for i in P}
        prev = self.prev_ready_rank
        if prev is not None:
            common = sorted(set(prev) & P)
            for a, b in itertools.combinations(common, 2):
                if (prev[a] < prev[b]) != (cur[a] < cur[b]):
                    self.log["flips"] += 1
                    neg = (self.slack(a, t) < 0) + (self.slack(b, t) < 0)
                    if neg == 2:
                        self.log["flips_both_neg"] += 1
                    elif neg == 1:
                        self.log["flips_one_neg"] += 1
                    self.log["last_flip_t"] = t
                    if self.log["first_flip_t"] is None:
                        self.log["first_flip_t"] = t
        self.prev_ready_rank = cur


class DstuE(Fixed):
    def __init__(self, solo, d, events, hp):
        self.d = d
        self.own = {i: s["own"] for i, s in solo.items()}
        self.rest = {i: s["rest"] for i, s in solo.items()}
        self.grest = {i: s["group_rest"] for i, s in solo.items()}
        self.nblk = {i: s["group_n"] for i, s in solo.items()}
        self.hp = hp
        self.unissued = collections.Counter()
        for tag, e in events.items():
            if e.event_type != "communication":
                continue
            i = tag[1] - 10
            g = group_of(rt(tag))
            if self.nblk[i].get(g, 1) < 2:
                continue
            for link in {l for br in e.path_list for l in br}:
                self.unissued[(i, g, link)] += self.occ(e, link)

    def occ(self, e, link):
        return int(self.hp.links_dict[link].working_time(data_bytes=e.comm_bytes))

    def tail(self, e, link):
        i = e.event_tag[1] - 10
        r = rt(e.event_tag)
        g = group_of(r)
        if self.nblk[i].get(g, 1) >= 2:  # handoff 2.2 as corrected 2026-11-02 (no "- own occupancy of b")
            return self.unissued[(i, g, link)] + self.grest[i][g]
        return self.own[i][r] + self.rest[i][r]

    def key(self, e, link, ln, t):
        return (self.d[e.event_tag[1] - 10] - t - self.tail(e, link), -ln)

    def on_issue(self, e, link, t):
        i = e.event_tag[1] - 10
        g = group_of(rt(e.event_tag))
        if self.nblk[i].get(g, 1) >= 2:
            self.unissued[(i, g, link)] -= self.occ(e, link)


class DstuSn(DstuS):
    """P6' (user decision 2026-11-02): slack_i(t) / T_i"""

    def score(self, i, t):
        return self.slack(i, t) / self.T[i]


class DstuEn(DstuE):
    """P7' (user decision 2026-11-02): (d_i - t - tail_e) / T_i, tail as in P7"""

    def __init__(self, solo, d, events, hp, T):
        super().__init__(solo, d, events, hp)
        self.T = T

    def key(self, e, link, ln, t):
        i = e.event_tag[1] - 10
        return ((self.d[i] - t - self.tail(e, link)) / self.T[i], -ln)


class DstuSh(DstuSn):
    """P6h (user decision 2026-11-02): P6' with hysteresis. A stream order is kept; a stream enters it at its
    release, placed by its P6' key at that instant; at every new arbitration instant adjacent pairs are compared and a
    challenger moves ahead of the stream in front of it only when its P6' key is smaller by more than theta (passes
    repeated until no swap). Event key = (position in the order, -ln)."""

    def __init__(self, solo, d, T, release, ref_rank, theta):
        super().__init__(solo, d, T, release, ref_rank)
        self.theta = theta
        self.order = []
        self.pos = {}
        self.t_upd = None

    def update(self, t):
        if t == self.t_upd:
            return
        self.t_upd = t
        live = [i for i in self.T if self.rel[i] <= t and self.ndone[i] < self.n[i]]
        sc = {i: self.score(i, t) for i in live}
        self.order = [i for i in self.order if i in sc]
        for i in sorted((i for i in live if i not in self.order), key=lambda i: (sc[i], i)):
            j = 0
            while j < len(self.order) and (sc[self.order[j]], self.order[j]) <= (sc[i], i):
                j += 1
            self.order.insert(j, i)
        for _ in range(len(self.order) ** 2 + 1):
            sw = False
            for j in range(len(self.order) - 1):
                a, b = self.order[j], self.order[j + 1]
                if sc[b] < sc[a] - self.theta:
                    self.order[j], self.order[j + 1] = b, a
                    sw = True
            if not sw:
                break
        self.pos = {i: j for j, i in enumerate(self.order)}

    def stream_key(self, i, t):
        self.update(t)
        return self.pos.get(i, len(self.T))

    def key(self, e, link, ln, t):
        return (self.stream_key(e.event_tag[1] - 10, t), -ln)

    def on_tick(self, t, ready):
        self.update(t)
        super().on_tick(t, ready)


class DstuSv(DstuSn):
    """P6v (user decision 2026-11-02): streams with slack_i(t) < 0 after all others; P6' key inside each class."""

    def stream_key(self, i, t):
        return (self.slack(i, t) < 0, self.score(i, t))

    def key(self, e, link, ln, t):
        i = e.event_tag[1] - 10
        return (self.slack(i, t) < 0, self.score(i, t), -ln)


class DstuEm(DstuSn):
    """P7m (user decision 2026-11-02): ((d_i - t - max(R_i(t), tail_e)) / T_i, -ln); R_i as P6', tail_e as P7'.
    The stream-level slack log and flip count use P6''s score."""

    def __init__(self, solo, d, T, release, ref_rank, events, hp):
        super().__init__(solo, d, T, release, ref_rank)
        self.te = DstuE(solo, d, events, hp)

    def key(self, e, link, ln, t):
        i = e.event_tag[1] - 10
        return ((self.d[i] - t - max(self.R(i), self.te.tail(e, link))) / self.T[i], -ln)

    def on_issue(self, e, link, t):
        self.te.on_issue(e, link, t)


class DstuEmr(DstuEm):
    """P7m_raw (user decision 2026-11-03): (d_i - t - max(R_i(t), tail_e), -ln), P7m without the division by T_i.
    Stream-level order for the log / flips: raw slack (P6's score)."""

    def score(self, i, t):
        return self.slack(i, t)

    def key(self, e, link, ln, t):
        i = e.event_tag[1] - 10
        return (self.d[i] - t - max(self.R(i), self.te.tail(e, link)), -ln)


class DstuEmv(DstuEm):
    """P7mv (user decision 2026-11-02): P7m with P6v's demotion; class = stream-level slack_i(t) < 0, P7m's event key
    inside each class. Stream-level order for the log / flips: (slack_i(t) < 0, P6' key)."""

    def stream_key(self, i, t):
        return (self.slack(i, t) < 0, self.score(i, t))

    def key(self, e, link, ln, t):
        i = e.event_tag[1] - 10
        return (self.slack(i, t) < 0,) + super().key(e, link, ln, t)


def make_policy(p, ev, hp):
    n = p["name"]
    if n == "P1":
        return None
    if n in ("P2", "P3", "P4", "fixed"):
        return Fixed({int(i): v for i, v in p["rank"].items()})
    if n == "P6":
        return DstuS(p["solo"], p["d"], p["T"], p["release"], p["ref_rank"])
    if n == "P7":
        return DstuE(p["solo"], p["d"], ev, hp)
    if n == "P6n":
        return DstuSn(p["solo"], p["d"], p["T"], p["release"], p["ref_rank"])
    if n == "P7n":
        return DstuEn(p["solo"], p["d"], ev, hp, p["T"])
    if n == "P7m":
        return DstuEm(p["solo"], p["d"], p["T"], p["release"], p["ref_rank"], ev, hp)
    if n == "P7m_raw":
        return DstuEmr(p["solo"], p["d"], p["T"], p["release"], p["ref_rank"], ev, hp)
    if n == "P6h":
        return DstuSh(p["solo"], p["d"], p["T"], p["release"], p["ref_rank"], p["theta"])
    if n == "P6v":
        return DstuSv(p["solo"], p["d"], p["T"], p["release"], p["ref_rank"])
    if n == "P7mv":
        return DstuEmv(p["solo"], p["d"], p["T"], p["release"], p["ref_rank"], ev, hp)
    raise E4bError(f"unknown policy {n}")


# ---------------------------------------------------------------- child
def child(spec_path):
    with open(spec_path, "rb") as f:
        spec = pickle.load(f)
    E._artifact_path()
    from e4b_backend import event_driver_e4b
    e2 = E._e2()
    hp0 = E.build_platform(spec["cfg"], spec["faults"])
    blobs = [load_stream(s)["events_blob"] for s in spec["streams"]]
    out = []
    for seed in spec["seeds"]:
        ev = E.merged_events(blobs)
        cst = None
        if spec.get("chunk"):
            ev, cst = E.chunk_events(ev, hp0, spec["chunk"]["size"], spec["chunk"]["only_d2d"])
        hp = copy.deepcopy(hp0)
        pre = e2._snapshot_pre(ev) if spec.get("keep_cap") else None
        pol = make_policy(spec["policy"], ev, hp)
        shadow = Fixed({int(i): v for i, v in spec["shadow_rank"].items()}) if spec.get("shadow_rank") else None
        audit = [] if spec.get("audit") else None
        slog = {}
        released = {}
        random.seed(seed)
        t0 = time.time()
        res = event_driver_e4b(ev, hp, release=spec["release"], policy=pol, released=released, audit=audit,
                               shadow=shadow, shadow_log=slog)
        wall = time.time() - t0
        C = collections.defaultdict(int)
        for t, e in ev.items():
            C[t[1] - 10] = max(C[t[1] - 10], e.end_time)
        r = {"seed": seed, "wall": wall, "result": res, "C": dict(C), "release": released, "chunk_stats": cst,
             "shadow": slog}
        if audit is not None:
            bad = [a for a in audit if a[2] != a[3]]
            rng = random.Random(1000 + seed)
            r["audit"] = {"n_issues": len(audit), "n_multi": sum(1 for a in audit if a[4] > 1), "n_mismatch": len(bad),
                          "sample": [(a[0], str(a[1]), a[2], a[3], a[4]) for a in rng.sample(audit, min(5, len(audit)))]}
        if isinstance(pol, DstuS):
            r["slack_log"] = {k: v for k, v in pol.log.items() if k not in ("last_t", "last_rank")}
        if spec.get("keep_cap"):
            cap = e2._extract(ev, hp, pre, res, f"seed{seed}")
            cap["release"] = released
            r["cap"] = cap
        out.append(r)
    with open(spec["sink"], "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def run(name, spec):
    path = os.path.join(OUT, "sims", f"{name}.pkl.gz")
    if os.path.exists(path):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = os.path.join(OUT, "_tmp")
    logs = os.path.join(OUT, "logs")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    sp = os.path.join(tmp, f"spec_{name}.pkl")
    sink = os.path.join(tmp, f"sink_{name}.pkl")
    with open(sp, "wb") as f:
        pickle.dump(dict(spec, sink=sink), f)
    t0 = time.time()
    if not os.path.exists(sink):
        with open(os.path.join(logs, f"{name}.log"), "w", encoding="utf-8") as lf:
            try:
                rc = subprocess.call([sys.executable, os.path.abspath(__file__), "child", sp], cwd=HERE, stdout=lf,
                                     stderr=subprocess.STDOUT, timeout=LIMIT * len(spec["seeds"]))
            except subprocess.TimeoutExpired:
                raise E4bError(f"{name}: over {LIMIT} s per evaluation (handoff 6)")
        if rc != 0:
            raise E4bError(f"{name} failed (exit {rc}); see e4b_out/logs/{name}.log")
    E._artifact_path()
    with open(sink, "rb") as f:
        runs = pickle.load(f)
    os.remove(sink)
    os.remove(sp)
    r = {"runs": runs, "process_wall": time.time() - t0}
    with gzip.open(path, "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    return r


def run_all(jobs, items):
    t0 = time.time()
    with ThreadPoolExecutor(jobs) as ex:
        res = dict(zip([i[0] for i in items], ex.map(lambda i: run(*i), items)))
    log(f"{len(items)} processes ({time.time() - t0:.0f} s)")
    return res


def log(msg):
    print(f"[e4b] {msg}", flush=True)


def write_json(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def write_csv(name, rows):
    import csv
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------- solo reference data (parent side)
def solo_data(cap):
    sim = A.SimR(cap)
    Li = sim.stream_L()
    L = Li[0]
    ev = cap["events"]
    hop = collections.defaultdict(int)
    for k, rec in cap["links"].items():
        for s, f, tag in rec:
            hop[tag] += f - s
    w, own, rest = {}, {}, {}
    for t, e in ev.items():
        r = rt(t)
        w[r] = (e["f"] - e["s"]) if e["type"] == "computation" else hop[t]
        own[r] = e["f"] - e["s"]
        term = sim.terminal[t]
        Aa = sim.act[term]
        rest[r] = L[term] - (Aa["f"] - Aa["s"])
    fin = sorted((e["f"], w[rt(t)]) for t, e in ev.items())
    Wf, Wc, c = [], [], 0
    for f, x in fin:
        c += x
        Wf.append(f)
        Wc.append(c)
    groups = collections.defaultdict(list)
    for t, e in ev.items():
        if e["type"] == "communication":
            groups[group_of(rt(t))].append(t)
    gn = {g: len(ts) for g, ts in groups.items()}
    grest = {g: rest[rt(max(ts, key=lambda x: (ev[x]["f"], str(x))))] for g, ts in groups.items() if len(ts) >= 2}
    return {"w": w, "own": own, "rest": rest, "W_f": Wf, "W_c": Wc, "n_events": len(ev), "group_n": gn,
            "group_rest": grest, "T_ref_run": max(e["f"] for e in ev.values())}


def rhat_check(caps, sd, T):
    """self-check 5: along each solo run, t - r + R(t) vs T at every event-finish instant."""
    out = {}
    for seed, cap in caps.items():
        fin = sorted((e["f"], sd["w"][rt(t)]) for t, e in cap["events"].items())
        devs, done = [], 0
        k = 0
        while k < len(fin):
            t = fin[k][0]
            while k < len(fin) and fin[k][0] == t:
                done += fin[k][1]
                k += 1
            j = bisect.bisect_left(sd["W_c"], done - 1e-9)
            tau = sd["W_f"][min(j, len(sd["W_f"]) - 1)]
            R = max(0.0, T - tau)
            devs.append(abs(t + R - T) / T)
        out[seed] = {"max": max(devs), "median": statistics.median(devs), "n_points": len(devs),
                     "C_run": fin[-1][0]}
    return out


# ---------------------------------------------------------------- P8 partition
def d2d_partition(link_types):
    groups = collections.defaultdict(list)
    for k, t in link_types.items():
        if t == "ch2ch":
            groups[(k[0][0], k[1][0])].append(k)
    asg = {}
    for g, ls in sorted(groups.items()):
        if len(ls) != 2:
            return None, {"reason": f"Die pair/direction {g} has {len(ls)} D2D links", "groups": {str(a): [str(x) for x in b]
                                                                                                     for a, b in groups.items()}}
        ls = sorted(ls)
        asg[g] = {0: ls[0], 1: ls[1]}
    return asg, {"groups": {f"{a[0]}->{a[1]}": {"stream0": str(v[0]), "stream1": str(v[1])} for a, v in asg.items()}}


def capture_r0_b(key, base, failed_links):
    """E4a's capture_r0 (E3 R0: mapping injected, routes kept, only cut communications re-routed, no SA) with the
    output in e4b_out/streams."""
    path = os.path.join(STREAMS, f"{key}.pkl.gz")
    if os.path.exists(path):
        with gzip.open(path, "rb") as f:
            rec = pickle.load(f)
        return {"cached": True, "inject_routes": rec["inject_routes"], "empty_path_comms": rec["empty_path_comms"]}
    os.makedirs(STREAMS, exist_ok=True)
    b = E.load_stream(base)
    spec = {"mapping": b["mapping"], "steps": None, "t_max": None, "seeds": [], "preserve_routes": True,
            "faults": {"failed_nodes": [], "failed_links": [tuple(map(tuple, l)) for l in failed_links]}}
    o = E._run_capture_child(b["task"], [], f"e4b_r0_{key}", e3_spec=spec)
    lg = o.get("inject_log", {})
    routes = [{k: r.get(k) for k in ["where", "restored", "rerouted", "source_restored", "rerouted_without_any_match"]}
              for r in lg.get("routes", [])]
    rec = {"short": key, "base": base, "task": b["task"], "faults": spec["faults"], "events_blob": o["events_blob"],
           "platform_state": o["platform_state"], "empty_path_comms": o["empty_path_comms"],
           "template_result": o["last"]["result"], "mapping": o["mapping"], "inject_routes": routes,
           "artifact_commit": E._artifact_commit()}
    with gzip.open(path, "wb") as f:
        pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {"cached": False, "inject_routes": routes, "empty_path_comms": rec["empty_path_comms"],
            "template_result": rec["template_result"]}


# ---------------------------------------------------------------- step 1
def lhat(r, T, d):
    return max((r["C"][i] - d[i]) / T[i] for i in T)


def smax(r, T):
    return max((r["C"][i] - r["release"][i]) / T[i] for i in T)


def summarize(res, name, T, d=None):
    rs = res[name]["runs"]
    s = {"S_max": E._mean([smax(r, T) for r in rs]), "S_max_seeds": [smax(r, T) for r in rs],
         "stretch": {i: E._mean([(r["C"][i] - r["release"][i]) / T[i] for r in rs]) for i in T},
         "makespan": E._mean([max(r["C"].values()) for r in rs]), "eval_wall": [r["wall"] for r in rs],
         "C_seeds": {i: [r["C"][i] for r in rs] for i in T}}
    if d:
        s["Lhat_max"] = E._mean([lhat(r, T, d) for r in rs])
        s["Lhat_max_seeds"] = [lhat(r, T, d) for r in rs]
    if "audit" in rs[0]:
        s["audit"] = {k: sum(r["audit"][k] for r in rs) for k in ["n_issues", "n_multi", "n_mismatch"]}
    if rs[0].get("shadow"):
        s["shadow_vs_short_first"] = {k: sum(r["shadow"].get(k, 0) for r in rs) for k in ["n", "differ", "differ_stream"]}
    if "slack_log" in rs[0]:
        s["slack_log"] = {k: sum(r["slack_log"][k] for r in rs) for k in ["ticks", "rank_changes", "ticks_not_ref"]}
    return s


def step1(jobs):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(E.OUT, "hop", "hop_state.json"), encoding="utf-8") as f:
        Thop = json.load(f)["T_hop"]
    Tch = {}
    for k in ["mha2048", "ffn"]:
        with gzip.open(os.path.join(E.OUT, "diag", "sims", f"chunk_single_{k}_256d2d.pkl.gz"), "rb") as f:
            Tch[k] = E._mean([r["C"][0] for r in pickle.load(f)["runs"]])
    cfgA = {"keys": ["mha512", "mha8192"], "chunk": None, "T": {0: Thop["mha512"], 1: Thop["mha8192"]}}
    cfgB = {"keys": ["mha2048", "ffn"], "chunk": H.CHUNKS["256d2d"], "T": {0: Tch["mha2048"], 1: Tch["ffn"]}}
    base = {"cfg": E.CFG, "faults": None, "seeds": SEEDS}
    # ---- solo reference runs (per-hop P1, release 0, caps kept)
    solo_items = []
    for c in (cfgA, cfgB):
        for k in c["keys"]:
            solo_items.append((f"solo_{k}_{'256d2d' if c['chunk'] else 'none'}",
                               dict(base, streams=[k], release={0: 0}, policy={"name": "P1"}, chunk=c["chunk"],
                                    keep_cap=True)))
    sres = run_all(jobs, solo_items)
    solo, chk5 = {}, {}
    for c in (cfgA, cfgB):
        cz = "256d2d" if c["chunk"] else "none"
        c["solo"] = {}
        for i, k in enumerate(c["keys"]):
            caps = {r["seed"]: r["cap"] for r in sres[f"solo_{k}_{cz}"]["runs"]}
            sd = solo_data(caps[1])
            c["solo"][i] = sd
            chk5[f"{k}|{cz}"] = rhat_check(caps, sd, c["T"][i])
            solo[f"{k}|{cz}"] = {"T_ref_run_seed1": sd["T_ref_run"], "n_events": sd["n_events"],
                                 "chunked_groups": sum(1 for n in sd["group_n"].values() if n >= 2)}
    log("solo reference data ready; check 5: " + json.dumps({k: {s: [round(v["median"], 4), round(v["max"], 4)]
                                                              for s, v in x.items()} for k, x in chk5.items()}))
    bad5 = {k: x for k, x in chk5.items() if statistics.median([v["median"] for v in x.values()]) > 0.05}
    if bad5:
        raise E4bError(f"self-check 5: median deviation of R_hat > 5%: {list(bad5)}")
    # ---- P8 partition routing (a1 512+8192, no fault)
    lt = sres["solo_mha512_none"]["runs"][0]["cap"]["link_types"]
    asg, pinfo = d2d_partition(lt)
    p8 = {"partition": pinfo}
    if asg is None:
        p8["applicable"] = False
    else:
        p8["applicable"] = True
        caps8 = {}
        for i, k in enumerate(cfgA["keys"]):
            other = [v[1 - i] for v in asg.values()]
            caps8[i] = capture_r0_b(f"{k}__p8s{i}", k, other)
        p8["r0"] = caps8
        bad = sum(r.get("rerouted_without_any_match") or 0 for c in caps8.values() for r in c["inject_routes"])
        p8["rerouted_without_any_match"] = bad
        p8["failed_links_per_stream"] = {i: [str(v[1 - i]) for v in asg.values()] for i in (0, 1)}
    # ---- runs
    TA, TB = cfgA["T"], cfgB["T"]
    ref_rank = {0: 0, 1: 1}  # short first (stream 0 is the short one in both configurations)
    items = []

    def pol(name, c, k=None, rel=None, rank=None):
        rel = rel or {0: 0, 1: 0}
        p = {"name": name}
        if rank is not None:
            p["rank"] = rank
        if name in ("P6", "P7"):
            d = {i: rel[i] + k * c["T"][i] for i in c["T"]}
            p.update(solo=c["solo"], d=d, T=c["T"], release=rel, ref_rank=ref_rank)
        return p

    relA = {0: 0, 1: 0}
    A_ = dict(base, streams=cfgA["keys"], chunk=None, release=relA)
    B_ = dict(base, streams=cfgB["keys"], chunk=cfgB["chunk"], release=relA)
    items.append(("a1_d0_P1", dict(A_, policy={"name": "P1"}, audit=True)))
    items.append(("a1_d0_P2", dict(A_, policy={"name": "P2", "rank": relA}, audit=True)))
    items.append(("a1_d0_P3", dict(A_, policy={"name": "P3", "rank": TA}, audit=True)))
    items.append(("a1_d0_P4_k1.5", dict(A_, policy={"name": "P4", "rank": {i: 1.5 * TA[i] for i in TA}}, audit=True)))
    for perm in itertools.permutations(range(2)):
        items.append((f"a1_d0_fixed_{''.join(map(str, perm))}",
                      dict(A_, policy={"name": "fixed", "rank": {s: r for r, s in enumerate(perm)}}, audit=True)))
    for k in (1.1, 1.5, 2.0):
        items.append((f"a1_d0_P6_k{k}", dict(A_, policy=pol("P6", cfgA, k), audit=True, shadow_rank=ref_rank)))
        items.append((f"a1_d0_P7_k{k}", dict(A_, policy=pol("P7", cfgA, k), audit=True, shadow_rank=ref_rank)))
    items.append(("a2c_d0_P1", dict(B_, policy={"name": "P1"}, audit=True)))
    items.append(("a2c_d0_fixed_01", dict(B_, policy={"name": "fixed", "rank": ref_rank}, audit=True)))
    for k in (1.1, 1.5, 2.0):
        items.append((f"a2c_d0_P6_k{k}", dict(B_, policy=pol("P6", cfgB, k), audit=True, shadow_rank=ref_rank)))
        items.append((f"a2c_d0_P7_k{k}", dict(B_, policy=pol("P7", cfgB, k), audit=True, shadow_rank=ref_rank)))
    # check 3b: distinct releases, equal k_i T_i -> P4 (EDF) == P2 (FCFS)
    dl = int(round(0.5 * min(TA.values())))
    relS = {0: 0, 1: dl}
    K = 1.5 * max(TA.values())
    items.append(("a1_s05_P2", dict(A_, release=relS, policy={"name": "P2", "rank": relS})))
    items.append(("a1_s05_P4_eqK", dict(A_, release=relS, policy={"name": "P4", "rank": {i: relS[i] + K for i in TA}})))
    if p8["applicable"]:
        items.append(("a1_d0_P8", dict(A_, streams=[f"{k}__p8s{i}" for i, k in enumerate(cfgA["keys"])],
                                       policy={"name": "P1"}, audit=True)))
        for i, k in enumerate(cfgA["keys"]):
            items.append((f"solo_{k}__p8s{i}", dict(base, streams=[f"{k}__p8s{i}"], release={0: 0},
                                                    policy={"name": "P1"}, seeds=G.D1_SEEDS)))
    t1 = time.time()
    res = run_all(jobs, items)
    t2 = time.time()
    # ---- checks
    with open(os.path.join(E.OUT, "hop", "a1_2s_512-8192_d0_hop.json"), encoding="utf-8") as f:
        hj = json.load(f)
    with open(os.path.join(E.OUT, "diag", "hop_chunk.json"), encoding="utf-8") as f:
        hc = json.load(f)["results"]["nofault_256d2d"]
    S = {n: summarize(res, n, TA if n.startswith("a1") else TB) for n, _ in items if not n.startswith("solo_")}
    for n in list(S):
        if "_P6_" in n or "_P7_" in n:
            k = float(n.split("_k")[-1])
            T = TA if n.startswith("a1") else TB
            S[n] = summarize(res, n, T, {i: k * T[i] for i in T})
    e4a = {"P1": hj["per_variant"]["default"]["S_max_hopT"], "fixed_01": hj["per_variant"]["prio_01"]["S_max_hopT"],
           "fixed_10": hj["per_variant"]["prio_10"]["S_max_hopT"],
           "a2c_P1": hc["default"]["S_max"], "a2c_fixed_01": hc["prio_01"]["S_max"]}
    c1 = {"new": S["a1_d0_P1"]["S_max"], "e4a": e4a["P1"], "rel_diff": S["a1_d0_P1"]["S_max"] / e4a["P1"] - 1,
          "a2c_new": S["a2c_d0_P1"]["S_max"], "a2c_e4a": e4a["a2c_P1"],
          "a2c_rel_diff": S["a2c_d0_P1"]["S_max"] / e4a["a2c_P1"] - 1}
    c2 = {o: {"new": S[f"a1_d0_fixed_{o}"]["S_max"], "e4a": e4a[f"fixed_{o}"],
              "rel_diff": S[f"a1_d0_fixed_{o}"]["S_max"] / e4a[f"fixed_{o}"] - 1} for o in ("01", "10")}
    c2["a2c_01"] = {"new": S["a2c_d0_fixed_01"]["S_max"], "e4a": e4a["a2c_fixed_01"],
                    "rel_diff": S["a2c_d0_fixed_01"]["S_max"] / e4a["a2c_fixed_01"] - 1}
    c2["P3_SJF_vs_fixed01_identical"] = S["a1_d0_P3"]["C_seeds"] == S["a1_d0_fixed_01"]["C_seeds"]
    c2["P4_EDF_vs_fixed01_identical"] = S["a1_d0_P4_k1.5"]["C_seeds"] == S["a1_d0_fixed_01"]["C_seeds"]
    c2["P2_FCFS_d0_vs_P1_identical"] = S["a1_d0_P2"]["C_seeds"] == S["a1_d0_P1"]["C_seeds"]
    c3 = {}
    for pre, ref in (("a1_d0", "a1_d0_fixed_01"), ("a2c_d0", "a2c_d0_fixed_01")):
        for p in ("P6", "P7"):
            for k in (1.1, 1.5, 2.0):
                n = f"{pre}_{p}_k{k}"
                c3[n] = {"S_max": S[n]["S_max"], "S_max_short_first": S[ref]["S_max"],
                         "identical_to_short_first": S[n]["C_seeds"] == S[ref]["C_seeds"],
                         "rel_diff": S[n]["S_max"] / S[ref]["S_max"] - 1,
                         "shadow_vs_short_first": S[n].get("shadow_vs_short_first"),
                         "slack_log": S[n].get("slack_log")}
    c3["P4_vs_P2_distinct_release_equal_kT"] = {
        "release": relS, "K": K, "identical": S["a1_s05_P2"]["C_seeds"] == S["a1_s05_P4_eqK"]["C_seeds"],
        "S_max_P2": S["a1_s05_P2"]["S_max"], "S_max_P4": S["a1_s05_P4_eqK"]["S_max"]}
    aud = collections.Counter()
    for n, s in S.items():
        for k, v in (s.get("audit") or {}).items():
            aud[k] += v
    c4 = dict(aud)
    w1 = {"a1": E._mean(S["a1_d0_P1"]["eval_wall"]), "a2c": E._mean(S["a2c_d0_P1"]["eval_wall"])}
    c7 = {n: E._mean(S[n]["eval_wall"]) / w1["a1" if n.startswith("a1") else "a2c"]
          for n in S if "_P6_" in n or "_P7_" in n}
    if p8["applicable"]:
        Ts8 = {i: E._mean([r["C"][0] for r in res[f"solo_{k}__p8s{i}"]["runs"]]) for i, k in enumerate(cfgA["keys"])}
        p8.update(solo_partitioned_T=Ts8, solo_partitioned_seeds={i: [r["C"][0] for r in res[f"solo_{k}__p8s{i}"]["runs"]]
                                                                   for i, k in enumerate(cfgA["keys"])},
                  solo_partitioned_over_T={i: Ts8[i] / TA[i] for i in TA}, run=S["a1_d0_P8"])
    out = {"T": {"a1": TA, "a2c": TB}, "solo": solo, "check1": c1, "check2": c2, "check3": c3, "check4": c4,
           "check5": chk5, "check7_wall_over_P1": c7, "P1_wall": w1, "P8": p8, "summaries": S,
           "walls": {"solo": t1 - t0, "runs": t2 - t1}, "seeds": SEEDS, "command": sys.argv,
           "artifact_commit": E._artifact_commit()}
    write_json("step1.json", out)
    rows = [dict(run=n, S_max=s["S_max"], Lhat_max=s.get("Lhat_max"), makespan=s["makespan"],
                 stretch_0=s["stretch"][0], stretch_1=s["stretch"][1], eval_wall=E._mean(s["eval_wall"]),
                 S_max_seed_range=max(s["S_max_seeds"]) - min(s["S_max_seeds"])) for n, s in S.items()]
    write_csv("step1_table.csv", rows)
    stop = []
    if abs(c1["rel_diff"]) > 0.04 or abs(c1["a2c_rel_diff"]) > 0.04:
        stop.append("check 1 > 4%")
    if any(abs(v["rel_diff"]) > 0.04 for k, v in c2.items() if isinstance(v, dict)):
        stop.append("check 2 > 4%")
    if not c3["P4_vs_P2_distinct_release_equal_kT"]["identical"]:
        stop.append("check 3 (P4 vs P2) not identical")
    for n, v in c3.items():  # criterion (a), user decision 2026-11-02
        if n.startswith("a") and "_P6_" in n and v["slack_log"]["ticks_not_ref"] == 0:
            v["pass_a"] = v["shadow_vs_short_first"].get("differ", 0) == 0 and abs(v["rel_diff"]) <= 0.04
            if not v["pass_a"]:
                stop.append(f"check 3: {n} slack order constant but shadow differs or S_max > 4% from short first")
    if c4.get("n_mismatch", 0) or c4.get("n_issues", 0) < 1000:
        stop.append("check 4")
    if any(v > 10 for v in c7.values()):
        stop.append("handoff 6: DSTU > 10x slower than P1")
    line = {"check1": round(c1["rel_diff"], 4), "check1_a2c": round(c1["a2c_rel_diff"], 4),
            "check2": {k: round(v["rel_diff"], 4) for k, v in c2.items() if isinstance(v, dict)},
            "check3_identical": {n: v["identical_to_short_first"] for n, v in c3.items() if n.startswith("a")},
            "check3_P4_P2": c3["P4_vs_P2_distinct_release_equal_kT"]["identical"],
            "check4": c4, "check5_median_max": {k: [round(max(v["median"] for v in x.values()), 4),
                                                    round(max(v["max"] for v in x.values()), 4)] for k, x in chk5.items()},
            "check7_max": round(max(c7.values()), 2), "P8": p8.get("applicable"), "stop": stop}
    log(json.dumps(line, default=str))
    if stop:
        raise E4bError("; ".join(stop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    if args.cmd == "child":
        child(args.spec)
        return
    step1(args.jobs)


if __name__ == "__main__":
    try:
        main()
    except (E4bError, E.E4Error) as exc:
        print(f"[e4b] STOP: {exc}", flush=True)
        sys.exit(2)
