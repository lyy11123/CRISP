"""E2: critical-wait attribution on BusyBarn's analytical backend.

Runs one generated task script unchanged and captures the template's final
backend evaluation (the 2nd and last call to event_driver; SA itself never
calls the backend). From that one simulation it:
  - walks the critical chain back from the latest-finishing event,
  - attributes every positive wait on the chain to the device/link occupant
    that released the resource exactly when the waiting activity started,
  - asks whether each blocker could yield: defer (data-deadline slack) or,
    for communication, reroute (a path <= current+2 hops that avoids the
    contended link and is idle over the blocker's window).

Backend semantics this relies on (read from src/backend/analytical/event_driver.py):
  - hops are NOT store-and-forward: every free head link of a message is
    issued at the same instant, so consecutive hops usually overlap; a hop
    waits only for its own busy link, or for the previous hop on its branch to
    have been issued (it becomes the branch head only then);
  - a communication's start_time ends up as the time its LAST hops were issued
    (each deferral overwrites it); end_time is the end of the last-issued hop.
    A message's hops can be spread over a long time, so one hop's occupancy
    ending says nothing about the whole message ending;
  - dependency_set and path_list are consumed during simulation, so both are
    snapshotted before calling the original event_driver.

Hence the chain is built over ACTIVITIES: one computation execution, or one
hop (event, link). For an activity X with start s, end f and earliest possible
start c (comp: r(e); first hop: r(e); later hop: start of the previous hop on
its branch):
  s > c            -> resource edge to the occupant of X's device/link whose
                      record ends exactly at s (critical wait = s - c);
  s == c, later hop -> hop edge to the previous hop (zero length on the chain);
  s == c, otherwise -> data edge to the dependency that finished last (its
                      terminal activity, i.e. the hop/execution that ends at r).
Every chain row owns [s, t_out], t_out = a for the last row and = the start it
explains otherwise, so the rows tile [0, a] (self-check 2).

Usage (one task per process):
  python e2_attribution.py --task <artifact>/models/intra_mapping_gpt_decode/py/<task>.py --seeds 1 2 3
  python e2_attribution.py --from-dump e2_out/<task>_dump.pkl.gz   # re-analyse only
"""
import argparse
import bisect
import collections
import copy
import csv
import gzip
import hashlib
import json
import os
import pickle
import random
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMIT_FILE = os.path.join(HERE, "..", "third_party", "isca2026-busybarn-artifact.COMMIT")
DDR_LINK_TYPES = {"co2ddr", "ddr2co"}
TEMPLATE_SEED = 123  # random.seed(123) at the top of every template script


def _plain(x):
    """TupleIdx and friends -> plain nested tuples (picklable, hashable)."""
    if isinstance(x, tuple):
        return tuple(_plain(v) for v in x)
    if isinstance(x, (list, set)):
        return type(x)(_plain(v) for v in x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _artifact_commit():
    try:
        with open(COMMIT_FILE, encoding="utf-8-sig") as f:
            return f.read().strip().split()[0]
    except OSError:
        return None


# ---------------------------------------------------------------- capture
def _snapshot_pre(events_dict):
    pre = {}
    for tag, ev in events_dict.items():
        pre[_plain(tag)] = {
            "deps": _plain(set(ev.dependency_set)),
            "issue": _plain(set(ev.issue_set)),
            "path_list": _plain([list(p) for p in ev.path_list]) if ev.event_type == "communication" else None,
        }
    return pre


def _extract(events_dict, hp, pre, result, rng_digest):
    """Plain-data view of one finished simulation."""
    events = {}
    for tag, ev in events_dict.items():
        t = _plain(tag)
        d = {"tag": t, "type": ev.event_type, "name": ev.event_name,
             "deps": pre[t]["deps"], "issue": pre[t]["issue"],
             "s": int(ev.start_time), "f": int(ev.end_time)}
        if ev.event_type == "computation":
            d.update(device=ev.comp_device, location=_plain(ev.comp_location), comp_type=ev.comp_type)
        else:
            d.update(source=_plain(ev.source_location), target=_plain(ev.target_location),
                     bytes=int(ev.comm_bytes), branches=pre[t]["path_list"])
        events[t] = d
    devices = {}
    for dev_type, mods in hp.modules_dict.items():
        for loc, m in mods.items():
            rec = getattr(m, "work_record", None)
            if rec:
                devices[("dev", dev_type, _plain(loc))] = [(int(a), int(b), _plain(c)) for a, b, c in rec]
    links = {}
    link_types = {}
    for key, l in hp.links_dict.items():
        k = _plain(key)
        link_types[k] = l.link_type
        if l.work_record:
            links[k] = [(int(a), int(b), _plain(c)) for a, b, c in l.work_record]
    return {"events": events, "devices": devices, "links": links, "link_types": link_types,
            "result": tuple(int(v) for v in result), "rng_digest": rng_digest}


def _environment():
    import importlib
    vers = {}
    for m in ["numpy", "networkx", "matplotlib", "yaml", "simanneal", "tqdm"]:
        try:
            vers[m] = getattr(importlib.import_module(m), "__version__", "?")
        except ImportError:
            vers[m] = None
    return {"python": sys.executable, "python_version": sys.version.split()[0], "packages": vers}


def _rng_digest():
    return hashlib.sha1(repr(random.getstate()).encode()).hexdigest()[:16]


def child_capture(orig, events_dict, hardware_platform, call_no):
    """Runs inside `python <task>.py` (via _inject/sitecustomize.py) at the 2nd backend call.
    BusyBarn's results depend on object addresses (iteration order of sets of objects), so anything
    allocated before SA (extra sys.path entries, earlier imports, a different argv) or a deepcopy before
    the final call changes the latency; e.g. GPT-decode MHA sq8192 g0 l5 gave 36617 in-process instead
    of the plain script's 38608. Hence: the script runs exactly as `python <task>.py`, nothing is touched
    before the 2nd call, and re-seeded runs (which need a deepcopy) happen in a separate process."""
    seeds = [int(s) for s in os.environ.get("E2_SEEDS", "").split()]
    sink = {"calls": call_no, "reseeded": []}
    pre = _snapshot_pre(events_dict)
    canon = sorted((str(t), sorted(map(str, p["deps"])), sorted(map(str, p["issue"])), str(p["path_list"]))
                   for t, p in pre.items())  # sets sorted: string hashes differ between processes
    sink["pre_digest"] = hashlib.sha1(repr(canon).encode()).hexdigest()
    digest = _rng_digest()
    reruns = copy.deepcopy((events_dict, hardware_platform)) if seeds else None
    out = orig(events_dict=events_dict, hardware_platform=hardware_platform)
    sink["last"] = _extract(events_dict, hardware_platform, pre, out, digest)
    if reruns is not None:
        state = random.getstate(), np.random.get_state()
        try:
            for seed in seeds:
                ev_k, hp_k = copy.deepcopy(reruns)
                random.seed(seed)
                out_k = orig(events_dict=ev_k, hardware_platform=hp_k)
                sink["reseeded"].append((seed, _extract(ev_k, hp_k, pre, out_k, f"seed{seed}")))
        finally:
            random.setstate(state[0])
            np.random.set_state(state[1])
    with open(os.environ["E2_SINK"], "wb") as f:  # a 3rd call would overwrite: checked by the parent
        pickle.dump(sink, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def run_task(task_path, extra_seeds):
    """`python <task>.py` in a child process, captured at its 2nd (final) event_driver call."""
    task_path = os.path.abspath(task_path)
    os.makedirs(os.path.join(os.path.dirname(task_path), "../results"), exist_ok=True)
    tmp = os.path.join(HERE, "e2_out", f"_sink_{os.getpid()}_{len(extra_seeds)}.pkl")
    env = dict(os.environ, E2_SINK=tmp, E2_HOME=HERE, E2_SEEDS=" ".join(map(str, extra_seeds)),
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "_inject")] +
                                          ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    t0 = time.time()
    rc = subprocess.call([sys.executable, task_path], env=env, cwd=HERE)
    wall = time.time() - t0
    if rc != 0 or not os.path.exists(tmp):
        raise RuntimeError(f"task script failed (exit {rc}) or never reached a 2nd backend call")
    with open(tmp, "rb") as f:
        sink = pickle.load(f)
    os.remove(tmp)
    if sink["calls"] != 2:
        raise RuntimeError(f"expected 2 backend calls (initial + final), saw {sink['calls']}")
    return sink, wall


# ---------------------------------------------------------------- analysis
class E2Error(RuntimeError):
    pass


def _op_of(name):
    """Operator label: event name without its trailing index (MLPmlp1_conv1d_matadd_74 -> MLPmlp1_conv1d_matadd)."""
    return re.sub(r"_\d+$", "", str(name))


def _layer_op(name):
    """Coarse operator: LN / MLPmlp1 / MLP_act / ... (first token, or first two for MLP_act style)."""
    parts = str(name).split("_")
    return parts[0] if parts[0] != "MLP" or len(parts) < 2 else parts[0] + "_" + parts[1]


class Sim:
    def __init__(self, cap):
        self.cap = cap
        self.ev = cap["events"]
        self.a = cap["result"][0]
        self.link_types = cap["link_types"]
        self.ddr_nodes = {k[0] for k, t in self.link_types.items() if t == "ddr2co"}
        # resource -> sorted records (records on one resource never overlap)
        self.res = {k: sorted(rec) for k, rec in cap["devices"].items()}
        for k, rec in cap["links"].items():
            self.res[("link", k)] = sorted(rec)
        self.res_starts = {k: [r[0] for r in v] for k, v in self.res.items()}
        # ready times
        self.r = {t: max((self.ev[d]["f"] for d in e["deps"]), default=0) for t, e in self.ev.items()}
        # activities: key -> dict(s, f, can, res, tag, prev)
        self.act = {}
        self.terminal = {}
        hops = collections.defaultdict(dict)
        for k, rec in cap["links"].items():
            for s, f, tag in rec:
                if k in hops[tag]:
                    raise E2Error(f"event {tag} used link {k} twice")
                hops[tag][k] = (s, f)
        for t, e in self.ev.items():
            if e["type"] == "computation":
                key = ("comp", t)
                self.act[key] = {"s": e["s"], "f": e["f"], "can": self.r[t], "prev": None, "tag": t,
                                 "res": ("dev", e["device"], e["location"])}
                self.terminal[t] = key
                continue
            h = hops.get(t)
            if not h:
                raise E2Error(f"communication {t} has no link records")
            prev_of = {}
            for br in e["branches"] or []:
                prev = None
                for k in map(tuple, br):
                    if k not in h:
                        raise E2Error(f"communication {t}: hop {k} on its path has no record")
                    prev_of.setdefault(k, prev)  # shared multicast prefixes agree
                    prev = k
            if set(prev_of) != set(h):
                raise E2Error(f"communication {t}: recorded links differ from its path list")
            for k, (s, f) in h.items():
                p = prev_of[k]
                self.act[("hop", t, k)] = {"s": s, "f": f, "tag": t, "res": ("link", k),
                                           "prev": ("hop", t, p) if p else None,
                                           "can": h[p][0] if p else self.r[t]}
            ends = [k for k, (s, f) in h.items() if f == e["f"]]
            if not ends:
                raise E2Error(f"communication {t}: no hop ends at its end_time {e['f']}")
            self.terminal[t] = ("hop", t, max(ends, key=lambda k: (h[k][0], str(k))))
        self.first = {t: min(self.act[("hop", t, k)]["s"] for k in hops[t]) for t in hops}
        # graph for rerouting
        self.adj = collections.defaultdict(list)
        for k in self.link_types:
            self.adj[k[0]].append((k[1], k))

    def start(self, t):
        """Earliest activity of an event (first hop for communication)."""
        return self.first.get(t, self.ev[t]["s"])

    def res_name(self, rk):
        return f"{rk[1]}@{rk[2]}" if rk[0] == "dev" else f"{self.link_types[rk[1]]}:{rk[1]}"

    def res_kind(self, rk):
        return rk[1] if rk[0] == "dev" else self.link_types[rk[1]]

    def data_class(self, t):
        e = self.ev[t]
        if e["type"] == "computation":
            return "compute"
        tg = e["target"] if isinstance(e["target"], set) else {e["target"]}
        if e["source"] in self.ddr_nodes:
            return "ddr_read"
        if tg & self.ddr_nodes:
            return "ddr_write"
        return "onchip"

    def occupant_ending(self, rk, time_, exclude):
        for s, f, tag in self.res.get(rk, []):
            if f == time_ and tag != exclude:
                return tag
        return None

    def busy(self, rk, lo, hi, exclude):
        rec = self.res.get(rk)
        if not rec:
            return False
        j = bisect.bisect_left(self.res_starts[rk], hi) - 1
        while j >= 0:
            s, f, tag = rec[j]
            if f <= lo:
                break
            if tag != exclude:
                return True
            j -= 1
        return False

    def _act_of(self, rk, tag):
        return ("comp", tag) if rk[0] == "dev" else ("hop", tag, rk[1])

    # -- critical chain
    def chain(self):
        last = max(self.ev, key=lambda t: (self.ev[t]["f"], str(t)))
        cur, t_out = self.terminal[last], self.ev[last]["f"]
        rows, segs = [], []
        while cur is not None:
            if len(rows) > len(self.act) + 5:
                raise E2Error("critical chain does not terminate")
            A = self.act[cur]
            t = A["tag"]
            e = self.ev[t]
            s, can = A["s"], A["can"]
            row = {"activity": cur[0], "tag": t, "name": e["name"], "type": e["type"],
                   "resource": self.res_name(A["res"]), "resource_kind": self.res_kind(A["res"]),
                   "r": self.r[t], "can_start": can, "s": s, "f": A["f"], "t_out": t_out,
                   "event_s": e["s"], "event_f": e["f"], "wait": s - can, "contribution": t_out - s}
            if s > can:
                b = self.occupant_ending(A["res"], s, t)
                if b is None:
                    raise E2Error(f"positive wait on {cur} (can={can}, s={s}) but no occupant of "
                                  f"{self.res_name(A['res'])} ends at {s}")
                nxt = self._act_of(A["res"], b)
                row.update(pred=b, edge="resource", critical_wait=s - can)
                segs.append(self._blocker(cur, nxt))
            elif A["prev"] is not None:
                nxt = A["prev"]
                row.update(pred=t, edge="hop", critical_wait=0)
            elif e["deps"]:
                p = max(e["deps"], key=lambda d: (self.ev[d]["f"], str(d)))
                nxt = self.terminal[p]
                row.update(pred=p, edge="data", critical_wait=0)
            else:
                nxt = None
                row.update(pred="", edge="start", critical_wait=0)
            rows.append(row)
            if nxt is not None and row["edge"] != "hop" and self.act[nxt]["f"] != s:
                raise E2Error(f"chain discontinuity at {cur}: predecessor {nxt} ends at "
                              f"{self.act[nxt]['f']}, activity starts at {s}")
            cur, t_out = nxt, s
        rows.reverse()
        # Waits of a queue overlap in time (each blocker itself waited behind the next one), so the raw
        # sum can exceed a. "wait_excl" credits each instant only to the latest wait covering it.
        claimed = []
        for sg in segs:  # latest blocked start first
            lo, hi = sg["blocked_can_start"], sg["blocked_s"]
            free = hi - lo
            for c0, c1 in claimed:
                free -= max(0, min(hi, c1) - max(lo, c0))
            sg["wait_excl"] = free
            claimed.append((lo, hi))
            claimed = _merge(claimed)
        return last, rows, segs

    def _blocker(self, x_key, b_key):
        X, B = self.act[x_key], self.act[b_key]
        e, b = X["tag"], B["tag"]
        be = self.ev[b]
        e_dur, b_dur = X["f"] - X["s"], B["f"] - B["s"]
        D = min((self.start(c) for c in be["issue"]), default=self.a)
        # let e use the resource first from when it could, then b; b must still beat its first consumer
        defer = X["can"] + e_dur + b_dur <= D
        cur_len = ""
        if b_key[0] == "hop":
            L = b_key[2]
            br = next((list(map(tuple, p)) for p in be["branches"] if L in map(tuple, p)), None)
            if br is None:
                raise E2Error(f"blocker {b} has no branch through {L}")
            src, dst, cur_len = br[0][0], br[-1][1], len(br)
            lo = self.r[b]
            hi = lo + (be["f"] - be["s"])
            alt = self._bfs(src, dst, cur_len + 2, banned={L}) is not None
            idle = self._bfs(src, dst, cur_len + 2, banned={L},
                             busy=lambda k: self.busy(("link", k), lo, hi, b)) is not None
            # Supplementary, not the handoff's rule: reroute only the remainder of b's branch from the
            # contended link's upstream node, idle over the time b actually held that link.
            rest = br[br.index(L):]
            lo2, hi2 = B["s"], B["s"] + b_dur
            idle_local = self._bfs(L[0], dst, len(rest) + 2, banned={L},
                                   busy=lambda k: self.busy(("link", k), lo2, hi2, b)) is not None
        else:
            alt = idle = idle_local = False
        cat = ("both" if defer and idle else "defer_only" if defer else
               "reroute_only" if idle else "neither")
        # primary from round 2: reroute = local (remainder of the branch from the contended link)
        cat_local = ("both" if defer and idle_local else "defer_only" if defer else
                     "reroute_only" if idle_local else "neither")
        return {"blocked": e, "blocked_name": self.ev[e]["name"], "blocked_type": self.ev[e]["type"],
                "blocked_data": self.data_class(e),
                "blocker": b, "blocker_name": be["name"], "blocker_type": be["type"],
                "blocker_op": _op_of(be["name"]), "blocker_layer_op": _layer_op(be["name"]),
                "blocker_data": self.data_class(b),
                "resource": self.res_name(X["res"]), "resource_kind": self.res_kind(X["res"]),
                "wait": X["s"] - X["can"], "blocked_can_start": X["can"], "blocked_s": X["s"],
                "blocked_res_dur": e_dur, "blocker_res_s": B["s"], "blocker_res_dur": b_dur,
                "blocker_r": self.r[b], "blocker_s": be["s"], "blocker_f": be["f"],
                "D_b": D, "slack": D - be["f"], "can_defer": defer,
                "blocker_path_len": cur_len, "alt_path_exists": alt, "idle_alt_path_exists": idle,
                "idle_local_alt_path_exists": idle_local, "category": cat, "category_local": cat_local,
                "res_key": X["res"]}

    # -- activity DAG of the current schedule (durations fixed, order on every resource fixed)
    def dag(self):
        """succ[u] = [(v, kind, rk)]: kind "fs" (v starts after u ends) or "ss" (v starts after u starts,
        next hop of a branch); rk = resource for resource-order edges, else None. Data edges leave the
        dependency's terminal activity (its f is what the consumer's r is computed from)."""
        if hasattr(self, "_succ"):
            return self._succ, self._topo
        succ = collections.defaultdict(list)
        heads = collections.defaultdict(list)
        for k, A in self.act.items():
            if A["prev"] is not None:
                succ[A["prev"]].append((k, "ss", None))
            else:
                heads[A["tag"]].append(k)
        for t, e in self.ev.items():
            for d in e["deps"]:
                for h in heads[t]:
                    succ[self.terminal[d]].append((h, "fs", None))
        for rk, rec in self.res.items():
            for (_, _, t0), (_, _, t1) in zip(rec, rec[1:]):
                succ[self._act_of(rk, t0)].append((self._act_of(rk, t1), "fs", rk))
        indeg = collections.Counter(v for u in list(succ) for v, _, _ in succ[u])
        stack = [k for k in self.act if indeg[k] == 0]
        topo = []
        while stack:
            u = stack.pop()
            topo.append(u)
            for v, _, _ in succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    stack.append(v)
        if len(topo) != len(self.act):
            raise E2Error("activity graph of the current schedule has a cycle; tails are ill-defined")
        self._succ, self._topo = succ, topo
        return succ, topo

    def longest_from_start(self, skip_res=None, removed=frozenset()):
        """L[u] = longest time from u's start to the end of the run (u's own duration included), ignoring
        the order edges of resource skip_res and the activities in `removed`. Tail q(u) = L[u] - p(u)."""
        succ, topo = self.dag()
        L = {}
        for u in reversed(topo):
            if u in removed:
                continue
            p = self.act[u]["f"] - self.act[u]["s"]
            best = p
            for v, kind, rk in succ[u]:
                if v in removed or (rk is not None and rk == skip_res):
                    continue
                best = max(best, p + L[v] if kind == "fs" else L[v])
            L[u] = best
        return L

    def reorder_bound(self, rk):
        """Single-resource reorder test on rk: durations and the order on every other resource fixed,
        release = the item's current earliest start (can), tail = longest downstream path without rk's
        own order edges. Returns current max(f+q), Schrage (ready item with the largest tail first),
        preemptive Jackson (lower bound for any order), busy_end + min tail, and the longest path that
        avoids rk entirely (a floor no reordering of rk can beat)."""
        rec = self.res[rk]
        items = [self._act_of(rk, t) for _, _, t in rec]
        L = self.longest_from_start(skip_res=rk)
        jobs = []
        for k in items:
            A = self.act[k]
            p = A["f"] - A["s"]
            jobs.append((A["can"], p, L[k] - p, A["f"]))
        cur = max(f + q for _, _, q, f in jobs)
        sch, jac, lb3, busy_end = _schrage(jobs), _jackson(jobs), None, 0
        for r, p, _, _ in sorted(jobs):
            busy_end = max(busy_end, r) + p
        lb3 = busy_end + min(q for _, _, q, _ in jobs)
        off = max(self.longest_from_start(removed=frozenset(items)).values(), default=0)
        opt = _exact(jobs, min(cur, sch))
        best = max(off, min(cur, sch) if opt is None else opt)
        floor = max(off, jac, lb3) if opt is None else best  # exact optimum known -> bound is tight
        return {"resource": self.res_name(rk), "resource_kind": self.res_kind(rk), "n_items": len(jobs),
                "busy_over_a": sum(p for _, p, _, _ in jobs) / self.a,
                "current_max_f_plus_q": cur, "schrage_edf": sch, "exact_nonpreemptive": opt,
                "jackson_preemptive_lb": jac,
                "busy_end_plus_min_tail_lb": lb3, "longest_path_avoiding_resource": off,
                "reorder_gain": max(0, cur - best), "reorder_gain_over_a": max(0, cur - best) / self.a,
                "reorder_gain_ub": max(0, cur - floor), "reorder_gain_ub_over_a": max(0, cur - floor) / self.a}

    def ddr_corner(self, segs):
        """Per DDR port (its ddr2co and co2ddr links) and per corner core's outgoing on-chip links:
        bytes carried, busy time / a, share of exclusive critical wait."""
        wait = collections.Counter()
        for s in segs:
            wait[s["resource"]] += s["wait_excl"]
        total = sum(wait.values()) or 1

        def stats(k):
            rec = self.res.get(("link", k), [])
            name = self.res_name(("link", k))
            return {"bytes": sum(self.ev[t]["bytes"] for _, _, t in rec), "n_transfers": len(rec),
                    "busy_over_a": sum(f - s for s, f, _ in rec) / self.a, "wait_share": wait[name] / total}

        rows = []
        corner_of = {}
        for k, typ in self.link_types.items():
            if typ == "ddr2co":
                corner_of[k[0]] = k[1]
        for n in sorted(self.ddr_nodes):
            c = corner_of.get(n)
            rd = stats(next(k for k, t in self.link_types.items() if t == "ddr2co" and k[0] == n))
            wr_key = next((k for k, t in self.link_types.items() if t == "co2ddr" and k[1] == n), None)
            wr = stats(wr_key) if wr_key else {"bytes": 0, "busy_over_a": 0, "wait_share": 0, "n_transfers": 0}
            rows.append({"kind": "ddr_port", "die": n[0], "node": n, "corner": c, "link": "",
                         "read_bytes": rd["bytes"], "read_transfers": rd["n_transfers"],
                         "read_busy_over_a": rd["busy_over_a"], "read_wait_share": rd["wait_share"],
                         "write_bytes": wr["bytes"], "write_busy_over_a": wr["busy_over_a"],
                         "write_wait_share": wr["wait_share"]})
        for c in sorted(set(corner_of.values())):
            for k, typ in sorted(self.link_types.items(), key=str):
                if k[0] == c and typ not in DDR_LINK_TYPES:
                    st = stats(k)
                    rows.append({"kind": "corner_out_link", "die": c[0], "node": c, "corner": c,
                                 "link": f"{typ}:{k}", "read_bytes": st["bytes"], "read_transfers": st["n_transfers"],
                                 "read_busy_over_a": st["busy_over_a"], "read_wait_share": st["wait_share"]})
        ports = [r for r in rows if r["kind"] == "ddr_port"]
        per_corner = {}
        for c in sorted(set(corner_of.values())):
            pr = [r for r in ports if r["corner"] == c]
            ol = [r for r in rows if r["kind"] == "corner_out_link" and r["corner"] == c]
            per_corner[str(c)] = {
                "ddr_read_bytes": sum(r["read_bytes"] for r in pr),
                "ddr_read_busy_over_a": [round(r["read_busy_over_a"], 3) for r in pr],
                "out_link_busy_over_a": [round(r["read_busy_over_a"], 3) for r in ol],
                "out_link_bytes": sum(r["read_bytes"] for r in ol),
                "wait_share_ddr_ports": sum(r["read_wait_share"] + r["write_wait_share"] for r in pr),
                "wait_share_out_links": sum(r["read_wait_share"] for r in ol)}
        rb = np.array([r["read_bytes"] for r in ports], dtype=float)
        busy = np.array([r["read_busy_over_a"] for r in ports])
        outb = [max(v["out_link_busy_over_a"] or [0]) for v in per_corner.values()]
        hot = max(per_corner, key=lambda c: per_corner[c]["wait_share_ddr_ports"] + per_corner[c]["wait_share_out_links"])
        if per_corner[hot]["wait_share_ddr_ports"] + per_corner[hot]["wait_share_out_links"] <= 0:
            hot = None  # no critical wait at any corner
        summ = {"ddr_port_read_bytes_max_over_mean": float(rb.max() / rb.mean()) if rb.mean() else None,
                "ddr_port_read_bytes_cv": float(rb.std() / rb.mean()) if rb.mean() else None,
                "ddr_port_read_busy_over_a": {"min": float(busy.min()), "mean": float(busy.mean()),
                                              "max": float(busy.max())},
                "ddr_ports_read_busy_over_50pct": int((busy > 0.5).sum()), "n_ddr_ports": len(ports),
                "corner_max_out_link_busy_over_a": {"min": min(outb), "mean": float(np.mean(outb)), "max": max(outb)},
                "corners_with_out_link_busy_over_50pct": sum(1 for v in outb if v > 0.5), "n_corners": len(outb),
                "hotspot_corner": hot, "hotspot": per_corner.get(hot),
                "wait_share_all_ddr_ports": sum(r["read_wait_share"] + r["write_wait_share"] for r in ports),
                "wait_share_all_corner_out_links": sum(r["read_wait_share"] for r in rows if r["kind"] == "corner_out_link"),
                "per_corner": per_corner}
        return rows, summ

    def _bfs(self, src, dst, max_len, banned, busy=None):
        if src == dst:
            return 0
        seen, frontier, depth = {src}, [src], 0
        while frontier and depth < max_len:
            depth += 1
            nxt = []
            for u in frontier:
                for v, k in self.adj[u]:
                    if k in banned or v in seen or (busy and busy(k)):
                        continue
                    if v == dst:
                        return depth
                    seen.add(v)
                    nxt.append(v)
            frontier = nxt
        return None


def _schrage(jobs):
    """1|r_j|max(C_j + q_j), non-preemptive: whenever the resource frees, run the released job with the
    largest tail (EDF on deadlines a - q). jobs = [(r, p, q, _)]."""
    import heapq
    jobs = sorted(jobs)
    heap, t, i, best = [], 0, 0, 0
    while i < len(jobs) or heap:
        if not heap:
            t = max(t, jobs[i][0])
        while i < len(jobs) and jobs[i][0] <= t:
            heapq.heappush(heap, (-jobs[i][2], i))
            i += 1
        nq, j = heapq.heappop(heap)
        t += jobs[j][1]
        best = max(best, t - nq)
    return best


def _jackson(jobs):
    """Preemptive largest-tail-first: optimal for 1|r_j,pmtn|max(C_j + q_j), so a lower bound for any
    non-preemptive order of the same jobs."""
    import heapq
    jobs = sorted(jobs)
    heap, t, i, best = [], 0, 0, 0
    while i < len(jobs) or heap:
        if not heap:
            t = max(t, jobs[i][0])
        while i < len(jobs) and jobs[i][0] <= t:
            heapq.heappush(heap, (-jobs[i][2], i, jobs[i][1]))
            i += 1
        nq, j, rem = heapq.heappop(heap)
        nxt = jobs[i][0] if i < len(jobs) else float("inf")
        run = min(rem, nxt - t)
        t += run
        if rem - run > 0:
            heapq.heappush(heap, (nq, j, rem - run))
        else:
            best = max(best, t - nq)
    return best


EXACT_MAX_JOBS = 14
EXACT_NODE_BUDGET = 300_000


def _exact(jobs, incumbent):
    """Optimal non-preemptive order (branch and bound, Jackson bound on the remaining jobs).
    Returns None when there are too many jobs."""
    if len(jobs) > EXACT_MAX_JOBS:
        return None
    best, nodes = [incumbent], [0]

    def dfs(t, val, rest):
        nodes[0] += 1
        if nodes[0] > EXACT_NODE_BUDGET:
            raise TimeoutError
        if not rest:
            best[0] = min(best[0], val)
            return
        if max(val, _jackson([(max(r, t), p, q, 0) for r, p, q, _ in rest])) >= best[0]:
            return
        for i, (r, p, q, _) in sorted(enumerate(rest), key=lambda x: -x[1][2]):
            c = max(t, r) + p
            dfs(c, max(val, c + q), rest[:i] + rest[i + 1:])

    try:
        dfs(0, 0, list(jobs))
    except TimeoutError:
        return None
    return best[0]


def _merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _shares(counter, total):
    return {k: (v / total if total else None) for k, v in counter.most_common()}


def analyse(cap):
    sim = Sim(cap)
    a, b_comp, c_comm = cap["result"]
    bad = [(k, A["can"], A["s"]) for k, A in sim.act.items() if A["s"] < A["can"]]
    bad += [(t, sim.r[t], sim.start(t)) for t in sim.ev if sim.start(t) < sim.r[t]]
    check3 = not bad
    last, rows, segs = sim.chain()
    check1 = sim.ev[last]["f"] == a
    chain_sum = sum(r["contribution"] for r in rows)
    check2 = abs(chain_sum - a) <= 1 and rows[0]["s"] == 0
    if not (check1 and check2 and check3):
        raise E2Error(f"self-check failed: check1={check1} (end {sim.ev[last]['f']} vs a {a}), "
                      f"check2={check2} (sum {chain_sum} vs a {a}, chain starts at {rows[0]['s']}), "
                      f"check3={check3} ({bad[:5]})")

    # check4: the activity graph (fixed durations, fixed order per resource) reproduces a, so tails are
    # consistent with the schedule the chain was read from
    L0 = sim.longest_from_start()
    dag_lp = max(L0.values())
    check4 = dag_lp == a
    if not check4:
        raise E2Error(f"self-check 4 failed: longest path of the schedule's activity graph {dag_lp} != a {a}")
    reorder = []
    for rk in {s["res_key"] for s in segs}:
        rb = sim.reorder_bound(rk)
        rb["wait_excl"] = sum(s["wait_excl"] for s in segs if s["res_key"] == rk)
        reorder.append(rb)
    reorder.sort(key=lambda r: -r["wait_excl"])
    gain_of = {r["resource"]: r["reorder_gain"] for r in reorder}
    for s in segs:
        s["resource_reorder_gain"] = gain_of[s["resource"]]
        s["category_local_or_reorder"] = (s["category_local"] if s["category_local"] != "neither" or
                                          s["resource_reorder_gain"] <= 0 else "reorder_only")
    ddr_rows, ddr_summ = sim.ddr_corner(segs)

    def weighted(wkey):
        total = sum(s[wkey] for s in segs)
        agg = {k: collections.Counter() for k in
               ["resource", "category", "category_local", "category_local_or_reorder", "resource_kind",
                "blocker_type", "blocker_layer_op", "blocker_op", "blocker_data", "blocked_data"]}
        for s in segs:
            for k, c in agg.items():
                c[s[k]] += s[wkey]
        share = lambda x: (x / total) if total else None
        top = [v for _, v in agg["resource"].most_common()]
        cats = agg["category"]
        return {
            "critical_wait_total": total, "critical_wait_over_a": total / a,
            "wait_top1_resource_share": share(sum(top[:1])), "wait_top5_resource_share": share(sum(top[:5])),
            "wait_top_resources": [[k, v] for k, v in agg["resource"].most_common(5)],
            "wait_share_by_category": {k: share(cats[k]) for k in ["defer_only", "reroute_only", "both", "neither"]},
            "wait_share_by_resource_kind": _shares(agg["resource_kind"], total),
            "wait_share_by_blocker_type": _shares(agg["blocker_type"], total),
            "wait_share_by_blocker_layer_op": _shares(agg["blocker_layer_op"], total),
            "wait_share_by_blocker_op": _shares(agg["blocker_op"], total),
            "wait_share_by_blocker_data": _shares(agg["blocker_data"], total),
            "wait_share_by_blocked_data": _shares(agg["blocked_data"], total),
            "wait_share_blocker_can_defer": share(sum(s[wkey] for s in segs if s["can_defer"])),
            "wait_share_alt_path_exists": share(sum(s[wkey] for s in segs if s["alt_path_exists"])),
            "wait_share_idle_alt_path_exists": share(sum(s[wkey] for s in segs if s["idle_alt_path_exists"])),
            "wait_share_idle_local_alt_path_exists": share(sum(s[wkey] for s in segs if s["idle_local_alt_path_exists"])),
            "wait_share_neither_even_with_local_reroute": share(sum(
                s[wkey] for s in segs if not s["can_defer"] and not s["idle_local_alt_path_exists"])),
            "wait_share_neither_over_70pct": (share(cats["neither"]) or 0) > 0.70,
            "dstu_gain_upper_bound_over_a": (cats["defer_only"] + cats["reroute_only"] + cats["both"]) / a,
            # round 2 primary: pairwise defer + LOCAL reroute
            "wait_share_by_category_local": {k: share(agg["category_local"][k])
                                             for k in ["defer_only", "reroute_only", "both", "neither"]},
            "dstu_gain_upper_bound_local_over_a": sum(agg["category_local"][k]
                                                      for k in ["defer_only", "reroute_only", "both"]) / a,
            # "neither" waits that sit on a resource whose reorder gain is > 0 move to reorder_only
            "wait_share_by_category_local_or_reorder": {
                k: share(agg["category_local_or_reorder"][k])
                for k in ["defer_only", "reroute_only", "both", "reorder_only", "neither"]},
            "wait_share_on_resources_with_reorder_gain": share(sum(s[wkey] for s in segs
                                                                   if s["resource_reorder_gain"] > 0)),
        }

    on_chain = collections.Counter()
    for r in rows:
        on_chain[r["resource_kind"]] += r["contribution"]
    ddr_time = sum(v for k, v in on_chain.items() if k in DDR_LINK_TYPES)
    # D2D (ch2ch): on the chain, in critical wait, and as the blocker's route
    d2d_links = [k for k, t in sim.link_types.items() if t == "ch2ch"]
    crosses = {t: any(sim.link_types[tuple(k)] == "ch2ch" for br in (e["branches"] or []) for k in map(tuple, br))
               for t, e in sim.ev.items() if e["type"] == "communication"}
    tot_w = sum(s["wait_excl"] for s in segs) or 1
    d2d_busy = [sum(f - s for s, f, _ in sim.res.get(("link", k), [])) / a for k in d2d_links]
    d2d = {
        "n_d2d_links": len(d2d_links),
        "d2d_link_time_on_chain_over_a": on_chain.get("ch2ch", 0) / a,
        "d2d_crossing_event_time_on_chain_over_a": sum(r["contribution"] for r in rows if crosses.get(r["tag"])) / a,
        "wait_share_on_d2d_links": sum(s["wait_excl"] for s in segs if s["resource_kind"] == "ch2ch") / tot_w,
        "wait_share_blocker_crosses_d2d": sum(s["wait_excl"] for s in segs if crosses.get(s["blocker"])) / tot_w,
        "wait_share_blocked_crosses_d2d": sum(s["wait_excl"] for s in segs if crosses.get(s["blocked"])) / tot_w,
        "d2d_link_busy_over_a": ({"min": min(d2d_busy), "mean": float(np.mean(d2d_busy)), "max": max(d2d_busy)}
                                 if d2d_busy else None),
        "d2d_bytes_share_of_all_comm_bytes": (
            sum(e["bytes"] for t, e in sim.ev.items() if crosses.get(t)) /
            (sum(e["bytes"] for e in sim.ev.values() if e["type"] == "communication") or 1)),
    }
    ddr_read_events = {r["tag"] for r in rows if r["contribution"] > 0 and sim.data_class(r["tag"]) == "ddr_read"}
    ddr_read_time = sum(r["contribution"] for r in rows if sim.data_class(r["tag"]) == "ddr_read")
    summ = {
        "latency_a": a, "pure_comp_b": b_comp, "pure_comm_c": c_comm, "overlap_a_minus_b_minus_c": a - b_comp - c_comm,
        "pure_comm_share_c_over_a": c_comm / a, "pure_comp_share_b_over_a": b_comp / a,
        "n_events": len(sim.ev), "n_activities": len(sim.act), "chain_rows": len(rows),
        "chain_events": len({r["tag"] for r in rows}),
        "chain_time_by_resource_kind_over_a": {k: v / a for k, v in on_chain.most_common()},
        "n_resource_waits_on_chain": len(segs),
        # primary: exclusive waits (each instant counted once, <= a); raw = per-activity s - can, overlapping
        **weighted("wait_excl"),
        "raw": weighted("wait"),
        "ddr_link_time_on_chain_over_a": ddr_time / a,
        "ddr_read_events_on_chain": len(ddr_read_events),
        "ddr_read_event_time_on_chain_over_a": ddr_read_time / a,
        "check1_chain_end_equals_a": check1, "check2_chain_intervals_sum_to_a": check2,
        "check2_chain_sum": chain_sum, "check3_all_s_ge_r": check3,
        "check4_activity_graph_longest_path_equals_a": check4,
        # reorder: per resource gains are each "everything else fixed"; they are NOT additive
        "reorder_gain_best_resource_over_a": max((r["reorder_gain_over_a"] for r in reorder), default=0),
        "reorder_gain_best_resource": max(reorder, key=lambda r: r["reorder_gain"])["resource"] if reorder else None,
        "reorder_gain_top5_sum_over_a_nonadditive": sum(r["reorder_gain_over_a"] for r in reorder[:5]),
        "reorder_gain_all_sum_over_a_nonadditive": sum(r["reorder_gain_over_a"] for r in reorder),
        "reorder_gain_ub_best_resource_over_a": max((r["reorder_gain_ub_over_a"] for r in reorder), default=0),
        "reorder_top5_by_wait": [{k: r[k] for k in ["resource", "wait_excl", "n_items", "busy_over_a",
                                                     "current_max_f_plus_q", "schrage_edf", "exact_nonpreemptive",
                                                     "jackson_preemptive_lb",
                                                     "busy_end_plus_min_tail_lb", "longest_path_avoiding_resource",
                                                     "reorder_gain_over_a", "reorder_gain_ub_over_a"]}
                                 for r in reorder[:5]],
        "ddr_corner": ddr_summ,
        "d2d": d2d,
        "backend_rng_digest": cap["rng_digest"],
    }
    return summ, rows, segs, reorder, ddr_rows


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (str(v) if isinstance(v, (tuple, set, list)) else v) for k, v in r.items()})


CHAIN_FIELDS = ["activity", "tag", "name", "type", "resource", "resource_kind", "r", "can_start", "s", "f",
                "t_out", "event_s", "event_f", "wait", "pred", "edge", "critical_wait", "contribution"]
BLOCKER_FIELDS = ["blocked", "blocked_name", "blocked_type", "blocked_data", "blocker", "blocker_name",
                  "blocker_type", "blocker_op", "blocker_data", "resource", "resource_kind", "wait", "wait_excl",
                  "blocked_can_start", "blocked_s", "blocked_res_dur", "blocker_res_s", "blocker_res_dur",
                  "blocker_r", "blocker_s", "blocker_f", "D_b", "slack", "can_defer", "blocker_path_len",
                  "alt_path_exists", "idle_alt_path_exists", "idle_local_alt_path_exists", "category",
                  "category_local", "resource_reorder_gain", "category_local_or_reorder"]
REORDER_FIELDS = ["resource", "resource_kind", "wait_excl", "n_items", "busy_over_a", "current_max_f_plus_q",
                  "schrage_edf", "exact_nonpreemptive", "jackson_preemptive_lb", "busy_end_plus_min_tail_lb",
                  "longest_path_avoiding_resource", "reorder_gain", "reorder_gain_over_a", "reorder_gain_ub",
                  "reorder_gain_ub_over_a"]
DDR_FIELDS = ["kind", "die", "node", "corner", "link", "read_bytes", "read_transfers", "read_busy_over_a",
              "read_wait_share", "write_bytes", "write_busy_over_a", "write_wait_share"]
PRINT_KEYS = ["latency_a", "pure_comm_share_c_over_a", "pure_comp_share_b_over_a", "chain_rows",
              "critical_wait_over_a", "wait_top1_resource_share", "wait_top5_resource_share",
              "wait_share_by_category", "wait_share_by_blocker_type", "wait_share_by_blocker_data",
              "wait_share_alt_path_exists", "wait_share_idle_local_alt_path_exists",
              "dstu_gain_upper_bound_over_a", "ddr_link_time_on_chain_over_a",
              "wait_share_by_category_local", "dstu_gain_upper_bound_local_over_a",
              "wait_share_by_category_local_or_reorder",
              "reorder_gain_best_resource_over_a", "reorder_gain_best_resource",
              "reorder_gain_ub_best_resource_over_a", "reorder_gain_top5_sum_over_a_nonadditive",
              "check1_chain_end_equals_a", "check2_chain_intervals_sum_to_a", "check3_all_s_ge_r",
              "check4_activity_graph_longest_path_equals_a"]
SEED_EXTRA_KEYS = ["ddr_corner", "d2d"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", help="one generated task script (one task per process)")
    g.add_argument("--from-dump", help="re-analyse a saved <task>_dump.pkl.gz")
    ap.add_argument("--out", default=os.path.join(HERE, "e2_out"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[],
                    help="extra backend seeds to re-simulate the same final state with (e.g. 1 2 3)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.task:
        name = os.path.splitext(os.path.basename(args.task))[0]
        print(f"[e2] running {args.task}", flush=True)
        # Primary run with no deepcopy (see child_capture); re-seeded runs in a second process.
        sink, wall = run_task(args.task, [])
        reseeded = []
        if args.seeds:
            child, _ = run_task(args.task, args.seeds)
            if child["pre_digest"] != sink["pre_digest"]:
                raise E2Error("re-seed process saw a different pre-backend state than the primary run")
            reseeded = child["reseeded"]
        dump = {"task": name, "task_path": os.path.abspath(args.task), "wall_seconds": wall,
                "argv": sys.argv, "environment": _environment(), "final": sink["last"], "reseeded": reseeded,
                "pre_backend_state_digest": sink["pre_digest"]}
        with gzip.open(os.path.join(args.out, f"{name}_dump.pkl.gz"), "wb") as f:
            pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with gzip.open(args.from_dump, "rb") as f:
            dump = pickle.load(f)
        name = dump["task"]

    summ, rows, segs, reorder, ddr_rows = analyse(dump["final"])
    _write_csv(os.path.join(args.out, f"{name}_critical.csv"), rows, CHAIN_FIELDS)
    _write_csv(os.path.join(args.out, f"{name}_blockers.csv"), segs, BLOCKER_FIELDS)
    _write_csv(os.path.join(args.out, f"{name}_reorder.csv"), reorder, REORDER_FIELDS)
    _write_csv(os.path.join(args.out, f"{name}_ddr_corner.csv"), ddr_rows, DDR_FIELDS)

    reseeded = []
    for seed, cap in dump.get("reseeded", []):
        try:
            s_k = analyse(cap)[0]
            reseeded.append({"backend_seed": seed, **{k: s_k[k] for k in PRINT_KEYS + SEED_EXTRA_KEYS}})
        except E2Error as exc:
            reseeded.append({"backend_seed": seed, "error": str(exc)})
    out = {"task": name, **summ,
           "reseeded_runs": reseeded,
           "artifact_commit": _artifact_commit(),
           "template_python_seed": TEMPLATE_SEED,
           "final_eval_rng": "template's own random state at the 2nd event_driver call (digest above)",
           "command": dump.get("argv"), "task_wall_seconds": dump.get("wall_seconds"),
           "environment": dump.get("environment"),
           "notes": ("Chain over activities (computation execution or one hop); critical wait = s - earliest "
                     "start of the blocked activity; check2 = rows tile [0, a]. Defer: blocked.can + "
                     "blocked.dur + blocker.dur (on the contended resource) <= D(b). Reroute window = "
                     "[r(b), r(b) + f(b) - s(b)] with s(b) the backend start_time (last-hop issue), "
                     "b's own records excluded; path search is BFS on directed links minus the contended one. "
                     "Round 2: PRIMARY categories are *_local (pairwise defer + local reroute = remainder of "
                     "b's branch from the contended link's upstream node, idle over b's hold window); the "
                     "whole-path categories are kept as secondary. Reorder: per resource, durations and the "
                     "order on all other resources fixed; release = current earliest start; tail = longest "
                     "path from the item's finish to the end without that resource's order edges; gain = "
                     "current max(f+q) - max(Schrage EDF, longest path avoiding the resource); gain_ub uses "
                     "the preemptive Jackson / busy_end+min tail lower bounds. Per-resource gains are not "
                     "additive; the headline is the best single resource. BusyBarn routing is already "
                     "congestion-aware static Dijkstra (gamma=100), so reroute measures runtime rerouting "
                     "on top of it.")}
    with open(os.path.join(args.out, f"{name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    line = {"task": name, **{k: summ[k] for k in PRINT_KEYS}, "d2d": summ["d2d"]}
    if reseeded:
        line["reseeded_latency"] = [r.get("latency_a") for r in reseeded]
        line["reseeded_dstu_ub"] = [r.get("dstu_gain_upper_bound_over_a") for r in reseeded]
        line["reseeded_neither"] = [(r.get("wait_share_by_category") or {}).get("neither") for r in reseeded]
    print("[e2] " + json.dumps(line, default=str), flush=True)


if __name__ == "__main__":
    try:
        main()
    except E2Error as exc:
        print(f"[e2] STOP: {exc}", flush=True)
        sys.exit(2)
