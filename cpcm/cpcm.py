"""CPCM: dependency-aware schedule-level latency estimation (cpcm/CPCM_HANDOFF.md).

Input: an event graph (computation events on one device each, communication events as lists of link branches)
compiled once into flat integer-indexed lists. Output of a forward pass: per-activity start/end (an activity is one
computation execution or one hop = (communication, link)), per-event ready r, first start S, last issue S_last and end
F, and the service order on every resource. The backward pass on the extended graph (data edges + hop edges + the
resource-order edges of this schedule) gives LST, slack, the critical path, waits, CB(rho), phi(b, e) and D(b).

Forward modes (same interface, all deterministic):
  bb     BusyBarn event_driver Phase 1-4 (src/backend/analytical/event_driver.py, event_driver at line 282):
         per tick compute events first, then link levels from the longest remaining branch down; at each level events
         in order (-longest remaining branch, id) issue every free head of their branches of exactly that length;
         all free hops of a message issue in the same tick (no store-and-forward); a blocked event is retried at the
         earliest work_endtime of its busy heads. The backend's random shuffles are replaced by the tie rule
         (remaining length, event id). Inside one event branches go in the backend's own sort order (a remaining
         3-tuple link key first), then branch index. F = end of the LAST ISSUED hop (backend: event.end_time =
         links_dict[issued_path].work_endtime), not the max over hops.
  hop    per-hop arbitration as e4_overlap/e4b_backend.py: per tick, rounds; every free head link goes to the
         candidate with the smallest key (prio[e], -ln, e) (prio None -> (-ln, e), E4b's P1). prio is a fixed
         input array (two-pass use: pass 1 bb -> D, pass 2 hop with prio = D).
  order  imposed service order per resource (replay, or a hand-made reordering): s = max(earliest, end of the
         previous activity in the imposed order on that resource). Not work-conserving.

Earliest start of an activity (ignoring its resource): r(e) for a computation or a root hop, the start of the
parent hop for any later hop (a hop may issue in the same tick as its parent).
"""
import heapq
from math import inf

COMP, COMM = 0, 1


class CPCMError(RuntimeError):
    pass


# ------------------------------------------------------------------ graph
class Graph:
    """Flat event graph. Event ids are a static topological order (Kahn, ties by the caller's tag order).

    events:  kind[e], tag[e], name[e]; pred/succ in CSR (pred_ptr/pred, succ_ptr/succ)
    comp:    res_of[e], dur[e]
    comm:    ev_br[e] -> list of branch ids; branch b covers hops br_lo[b] .. br_hi[b]-1
    hops:    hop_link[h], hop_dur[h], hop_suf3[h] (a link with a 3-tuple key at or after h on its branch)
    resources: res_name[q], res_kind[q], res_key3[q]; failed resources are simply absent.
    Rerouting appends new branches/hops (set_branches); old ones become unreachable, ids stay valid.
    """

    def __init__(self):
        self.n = 0
        self.kind, self.tag, self.name = [], [], []
        self.pred_ptr, self.pred, self.succ_ptr, self.succ = [0], [], [0], []
        self.res_of, self.dur = [], []
        self.ev_br = []
        self.br_lo, self.br_hi = [], []
        self.hop_link, self.hop_dur, self.hop_suf3 = [], [], []
        self.res_name, self.res_kind, self.res_key3 = [], [], []
        self.res_id = {}
        self.id_of = {}

    def add_resource(self, name, kind="link", key3=False):
        if name in self.res_id:
            return self.res_id[name]
        q = len(self.res_name)
        self.res_id[name] = q
        self.res_name.append(name)
        self.res_kind.append(kind)
        self.res_key3.append(bool(key3))
        return q

    @property
    def nres(self):
        return len(self.res_name)

    def _add_branch(self, links, durs):
        b = len(self.br_lo)
        lo = len(self.hop_link)
        suf = False
        flags = []
        for q in reversed(links):
            suf = suf or self.res_key3[q]
            flags.append(suf)
        flags.reverse()
        for q, d, f in zip(links, durs, flags):
            if not d > 0:
                raise CPCMError(f"non-positive hop duration {d} on {self.res_name[q]}")
            self.hop_link.append(q)
            self.hop_dur.append(d)
            self.hop_suf3.append(f)
        self.br_lo.append(lo)
        self.br_hi.append(len(self.hop_link))
        return b

    def set_branches(self, e, branches, link_dur):
        """Replace the route of communication e. branches: list of lists of resource ids; link_dur: q -> duration."""
        if self.kind[e] != COMM:
            raise CPCMError("set_branches on a computation")
        self.ev_br[e] = [self._add_branch(br, [link_dur[q] for q in br]) for br in branches]

    def branches(self, e):
        return [self.hop_link[self.br_lo[b]:self.br_hi[b]] for b in self.ev_br[e]]

    def copy(self):
        g = Graph.__new__(Graph)
        for k, v in self.__dict__.items():
            if isinstance(v, list):
                v = [list(x) for x in v] if k == "ev_br" else list(v)
            elif isinstance(v, dict):
                v = dict(v)
            setattr(g, k, v)
        return g


def build(events, resources=()):
    """events: list of dicts in tie order, each
         {"tag", "name"?, "deps": [tags], "kind": "comp", "res": name, "dur": int}
         {"tag", "name"?, "deps": [tags], "kind": "comm", "branches": [[link names]], "dur": {link: int} | int}
       resources: optional [(name, kind, key3)] registered first (e.g. the whole link table, for rerouting).
       A resource first named inside an event is registered as ("dev" for comp, "link" for comm, key3 False)."""
    g = Graph()
    for name, kind, key3 in resources:
        g.add_resource(name, kind, key3)
    order = {ev["tag"]: i for i, ev in enumerate(events)}
    if len(order) != len(events):
        raise CPCMError("duplicate tags")
    indeg = [0] * len(events)
    out = [[] for _ in events]
    for i, ev in enumerate(events):
        for d in ev.get("deps", ()):
            if d not in order:
                raise CPCMError(f"{ev['tag']} depends on unknown {d}")
            indeg[i] += 1
            out[order[d]].append(i)
    heap = [i for i in range(len(events)) if indeg[i] == 0]
    heapq.heapify(heap)
    topo = []
    while heap:
        i = heapq.heappop(heap)
        topo.append(i)
        for j in out[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, j)
    if len(topo) != len(events):
        raise CPCMError("dependency cycle")
    new = {i: k for k, i in enumerate(topo)}
    g.n = len(events)
    preds = [[] for _ in topo]
    succs = [[] for _ in topo]
    for i, ev in enumerate(events):
        for d in ev.get("deps", ()):
            preds[new[i]].append(new[order[d]])
            succs[new[order[d]]].append(new[i])
    for k, i in enumerate(topo):
        ev = events[i]
        g.tag.append(ev["tag"])
        g.name.append(ev.get("name", str(ev["tag"])))
        g.id_of[ev["tag"]] = k
        g.pred.extend(sorted(preds[k]))
        g.pred_ptr.append(len(g.pred))
        g.succ.extend(sorted(succs[k]))
        g.succ_ptr.append(len(g.succ))
        if ev["kind"] == "comp":
            g.kind.append(COMP)
            g.res_of.append(g.add_resource(ev["res"], "dev"))
            d = ev["dur"]
            if not d > 0:
                raise CPCMError(f"non-positive duration {d} for {ev['tag']}")
            g.dur.append(d)
            g.ev_br.append([])
        elif ev["kind"] == "comm":
            g.kind.append(COMM)
            g.res_of.append(-1)
            g.dur.append(0)
            dd = ev["dur"]
            brs = []
            for br in ev["branches"]:
                links = [g.add_resource(l, "link") for l in br]
                durs = [dd[l] if isinstance(dd, dict) else dd for l in br]
                brs.append(g._add_branch(links, durs))
            if not brs or not any(br for br in ev["branches"]):
                raise CPCMError(f"communication {ev['tag']} has no hops")
            g.ev_br.append(brs)
        else:
            raise CPCMError(f"unknown kind {ev['kind']}")
    return g


def _int_time(x, what):
    v = int(x)
    if v != x:
        raise CPCMError(f"non-integer duration {x!r} for {what}")
    return v


def compile_bb(events_dict, hp, path_lists=None, deps=None):
    """Compile a BusyBarn events_dict (after build_event / route restoration, BEFORE the backend consumes
    dependency_set and path_list) plus its platform. Durations come from the artifact's own working_time
    (pure functions of bytes / datashape / type / frequency, verified 2026-11-03). path_lists / deps: optional
    snapshots {tag: ...} when the objects were already consumed. Resource names: ("dev", device, location) and
    ("link", key). All links and modules of the (faulted) platform are registered, used or not."""
    def plain_key(t):
        return t if isinstance(t, tuple) else (t,)

    res = []
    for dev in sorted(hp.modules_dict, key=str):
        for loc in sorted(hp.modules_dict[dev], key=str):
            res.append((("dev", dev, loc), "dev:" + dev, False))
    for k in sorted(hp.links_dict, key=str):
        res.append((("link", k), hp.links_dict[k].link_type, len(k) == 3))
    events = []
    for tag in sorted(events_dict, key=lambda t: (str(type(t)), plain_key(t))):
        ev = events_dict[tag]
        dep = list(deps[tag]) if deps is not None else list(ev.dependency_set)
        d = {"tag": tag, "name": ev.event_name, "deps": sorted(dep, key=plain_key)}
        if ev.event_type == "computation":
            m = hp.modules_dict[ev.comp_device][ev.comp_location]
            t = m.working_time(source_datashape=ev.comp_datashape, beha_type=ev.comp_type, frequency=hp.frequency)
            d.update(kind="comp", res=("dev", ev.comp_device, ev.comp_location), dur=_int_time(t, tag))
        else:
            pl = path_lists[tag] if path_lists is not None else ev.path_list
            if not pl:
                pl = ev.get_paths(hp.xy_multicast_path(
                    source_node_idx=ev.source_location,
                    target_nodes_set=ev.target_location if type(ev.target_location) is set else {ev.target_location}))
            branches = [[("link", k) for k in br] for br in pl]
            durs = {("link", k): _int_time(hp.links_dict[k].working_time(data_bytes=ev.comm_bytes), (tag, k))
                    for br in pl for k in br}
            d.update(kind="comm", branches=branches, dur=durs)
        events.append(d)
    return build(events, res)


# ------------------------------------------------------------------ schedule
class Schedule:
    """Result of one forward pass. Activities are numbered in issue order, so every edge of the extended graph
    (data, hop, resource order) goes from a lower to a higher activity id."""

    def __init__(self, g, mode):
        self.g, self.mode = g, mode
        n = g.n
        self.r = [0] * n
        self.S = [-1] * n
        self.S_last = [-1] * n
        self.F = [-1] * n
        self.term = [-1] * n          # activity whose end is F(e)
        self.ev_acts = [[] for _ in range(n)]
        self.a_ev, self.a_res, self.a_s, self.a_f, self.a_par = [], [], [], [], []
        self.res_seq = [[] for _ in range(g.nres)]
        self.T = None

    def _act(self, e, q, s, f, par):
        a = len(self.a_s)
        self.a_ev.append(e)
        self.a_res.append(q)
        self.a_s.append(s)
        self.a_f.append(f)
        self.a_par.append(par)
        self.res_seq[q].append(a)
        self.ev_acts[e].append(a)
        if self.S[e] < 0:
            self.S[e] = s
        return a

    def _done(self):
        if min(self.F, default=0) < 0:
            raise CPCMError("unfinished events")
        self.T = max(self.F, default=0)
        return self


def simulate(g, mode="bb", prio=None, release=None):
    """Forward pass in mode "bb" or "hop". release: optional {event id: earliest ready time} for source events."""
    if mode not in ("bb", "hop"):
        raise CPCMError(f"unknown mode {mode}")
    sc = Schedule(g, mode)
    n = g.n
    kind, pred_ptr, succ_ptr, succ = g.kind, g.pred_ptr, g.succ_ptr, g.succ
    res_of, dur = g.res_of, g.dur
    ev_br, br_lo, br_hi = g.ev_br, g.br_lo, g.br_hi
    hop_link, hop_dur, hop_suf3 = g.hop_link, g.hop_dur, g.hop_suf3
    r, S_last, F, term = sc.r, sc.S_last, sc.F, sc.term
    act = sc._act
    a_f = sc.a_f
    cnt = [pred_ptr[e + 1] - pred_ptr[e] for e in range(n)]
    free = [0] * g.nres
    pos = list(br_lo)               # per branch: next unissued hop
    blast = [-1] * len(br_lo)       # per branch: activity of its last issued hop
    last = [-1] * n                 # per event: last issued activity
    todo, fin, heap = {}, {}, []

    def at(dct, t, e):
        lst = dct.get(t)
        if lst is None:
            dct[t] = [e]
            heapq.heappush(heap, t)
        else:
            lst.append(e)

    for e in range(n):
        if cnt[e] == 0:
            t0 = release.get(e, 0) if release else 0
            r[e] = t0
            at(todo, t0, e)
    prev_t = None
    while heap:
        t = heapq.heappop(heap)
        if t == prev_t:
            continue
        prev_t = t
        ready = todo.pop(t, [])
        for e in fin.pop(t, ()):
            for i in range(succ_ptr[e], succ_ptr[e + 1]):
                x = succ[i]
                cnt[x] -= 1
                if cnt[x] == 0:
                    r[x] = t
                    ready.append(x)
        if not ready:
            continue
        ready.sort()
        comms = []
        for e in ready:
            if kind[e] == COMP:
                q = res_of[e]
                if free[q] > t:
                    at(todo, free[q], e)
                else:
                    f = t + dur[e]
                    free[q] = f
                    term[e] = act(e, q, t, f, -1)
                    S_last[e] = t
                    F[e] = f
                    at(fin, f, e)
            else:
                comms.append(e)
        if not comms:
            continue
        cand = {}
        if mode == "bb":
            L = 0
            for e in comms:
                for b in ev_br[e]:
                    k = br_hi[b] - pos[b]
                    if k > L:
                        L = k
            while L > 0:
                keyed = []
                for e in comms:
                    m = 0
                    for b in ev_br[e]:
                        k = br_hi[b] - pos[b]
                        if k > m:
                            m = k
                    if m >= L:
                        keyed.append((-m, e))
                keyed.sort()
                for _, e in keyed:
                    brs = [b for b in ev_br[e] if br_hi[b] - pos[b] == L]
                    if len(brs) > 1:
                        brs.sort(key=lambda b: (not hop_suf3[pos[b]], b))
                    for b in brs:
                        h = pos[b]
                        if br_hi[b] - h != L:       # trimmed by a shared head issued just before
                            continue
                        q = hop_link[h]
                        if free[q] > t:
                            if free[q] < cand.get(e, inf):
                                cand[e] = free[q]
                            continue
                        f = t + hop_dur[h]
                        free[q] = f
                        a = act(e, q, t, f, blast[b])
                        last[e] = a
                        for c in ev_br[e]:
                            p = pos[c]
                            if p < br_hi[c] and hop_link[p] == q:
                                pos[c] = p + 1
                                blast[c] = a
                L -= 1
        else:
            live = list(comms)
            while True:
                cands = {}
                for e in live:
                    seen = {}
                    for b in ev_br[e]:
                        p = pos[b]
                        if p < br_hi[b]:
                            q = hop_link[p]
                            k = br_hi[b] - p
                            if q not in seen or k > seen[q][0]:
                                seen[q] = (k, b)
                    for q, (k, b) in seen.items():
                        key = ((prio[e], -k, e) if prio is not None else (-k, e))
                        cands.setdefault(q, []).append((key, e, b))
                if not cands:
                    break
                issued = False
                for q in sorted(cands):
                    cs = cands[q]
                    if free[q] > t:
                        for _, e, _b in cs:
                            if free[q] < cand.get(e, inf):
                                cand[e] = free[q]
                        continue
                    _, e, b = min(cs)
                    h = pos[b]
                    f = t + hop_dur[h]
                    free[q] = f
                    a = act(e, q, t, f, blast[b])
                    last[e] = a
                    for c in ev_br[e]:
                        p = pos[c]
                        if p < br_hi[c] and hop_link[p] == q:
                            pos[c] = p + 1
                            blast[c] = a
                    for _, x, _b in cs:
                        if x != e and f < cand.get(x, inf):
                            cand[x] = f
                    issued = True
                live = [e for e in live if any(pos[b] < br_hi[b] for b in ev_br[e])]
                if not issued or not live:
                    break
        for e in comms:
            if all(pos[b] == br_hi[b] for b in ev_br[e]):
                a = last[e]
                term[e] = a
                S_last[e] = t
                F[e] = a_f[a]
                at(fin, F[e], e)
            else:
                c = cand.get(e)
                if c is None:
                    raise CPCMError(f"event {g.tag[e]} blocked with no candidate time at {t}")
                at(todo, c, e)
    return sc._done()


def hop_tree(g, e):
    """Unique hops of communication e in branch order: list of (resource, duration, parent index or -1).
    Shared multicast prefixes are one hop (first occurrence wins, as E2)."""
    idx, out = {}, []
    for b in g.ev_br[e]:
        prev = -1
        for h in range(g.br_lo[b], g.br_hi[b]):
            q = g.hop_link[h]
            if q not in idx:
                idx[q] = len(out)
                out.append((q, g.hop_dur[h], prev))
            prev = idx[q]
    return out


def simulate_order(g, seq, term_res=None, release=None):
    """Imposed service order. seq: {resource id: [event ids in service order]} (resources absent from seq serve
    in readiness order of whoever asks, i.e. are uncontended only if one user). term_res: optional {event id:
    resource id of the hop whose end is F(e)}; default: the hop with the latest start, ties by latest end."""
    sc = Schedule(g, "order")
    n = g.n
    trees = [hop_tree(g, e) if g.kind[e] == COMM else [(g.res_of[e], g.dur[e], -1)] for e in range(n)]
    pos_in = {}
    for q, evs in seq.items():
        for i, e in enumerate(evs):
            if (q, e) in pos_in:
                raise CPCMError(f"event {g.tag[e]} twice in the order of {g.res_name[q]}")
            pos_in[(q, e)] = i
    for e in range(n):
        for q, _, _ in trees[e]:
            if q in seq and (q, e) not in pos_in:
                raise CPCMError(f"event {g.tag[e]} uses {g.res_name[q]} but is missing from its order")
    nxt = {q: 0 for q in seq}
    rfree = {}                         # resource -> end of its last activity (ordered resources)
    cnt = [g.pred_ptr[e + 1] - g.pred_ptr[e] for e in range(n)]
    ready_ev = [release.get(e, 0) if release else 0 for e in range(n)]
    done_hops = [0] * n
    hop_act = [[-1] * len(trees[e]) for e in range(n)]
    # waiting[(q)] = list of (e, i) whose turn on q has not come yet
    pending = set()                    # (e, i) whose non-resource predecessors are done
    for e in range(n):
        if cnt[e] == 0:
            for i, (q, d, p) in enumerate(trees[e]):
                if p < 0:
                    pending.add((e, i))
    unordered_free = {}
    while pending:
        # issue every pending activity whose resource turn has come; pick the smallest feasible start first
        best = None
        for (e, i) in pending:
            q, d, p = trees[e][i]
            if q in seq:
                if seq[q][nxt[q]] != e:
                    continue
                prevf = rfree.get(q, 0)
            else:
                prevf = unordered_free.get(q, 0)
            c = ready_ev[e] if p < 0 else sc.a_s[hop_act[e][p]]
            s = max(c, prevf)
            k = (s, e, i)
            if best is None or k < best:
                best = k
        if best is None:
            raise CPCMError("imposed order deadlocks against the data dependencies")
        s, e, i = best
        pending.discard((e, i))
        q, d, p = trees[e][i]
        a = sc._act(e, q, s, s + d, hop_act[e][p] if p >= 0 else -1)
        hop_act[e][i] = a
        if q in seq:
            nxt[q] += 1
            rfree[q] = s + d
        else:
            unordered_free[q] = s + d
        for j, (_, _, pj) in enumerate(trees[e]):
            if pj == i:
                pending.add((e, j))
        done_hops[e] += 1
        if done_hops[e] == len(trees[e]):
            acts = sc.ev_acts[e]
            if term_res is not None and e in term_res:
                ta = next(x for x in acts if sc.a_res[x] == term_res[e])
            else:
                ta = max(acts, key=lambda x: (sc.a_s[x], sc.a_f[x], x))
            sc.term[e] = ta
            sc.S_last[e] = max(sc.a_s[x] for x in acts)
            sc.F[e] = sc.a_f[ta]
            sc.r[e] = ready_ev[e]
            for k in range(g.succ_ptr[e], g.succ_ptr[e + 1]):
                x = g.succ[k]
                cnt[x] -= 1
                if cnt[x] == 0:
                    ready_ev[x] = max(sc.F[g.pred[j]] for j in range(g.pred_ptr[x], g.pred_ptr[x + 1]))
                    for j, (_, _, pj) in enumerate(trees[x]):
                        if pj < 0:
                            pending.add((x, j))
    for e in range(n):
        sc.r[e] = ready_ev[e]
    # activity ids above follow issue order by increasing start, but an activity issued later in the loop can
    # start earlier than one issued before it; the backward pass only needs every edge to go from a lower to a
    # higher id, which holds because an activity is created only after all its predecessors exist.
    return sc._done()


# ------------------------------------------------------------------ backward pass and attribution
class Analysis:
    """Backward pass on the extended graph of a Schedule.

    Edges (u -> v, lag): hop parent -> child (0); previous activity on the same resource -> next (d(u));
    terminal activity of a data predecessor -> each root activity of the successor (d(u)).
    LST(a) = min(T_end - d(a), min over successors (LST(v) - lag)), T_end = max(T, every activity end).
    Event quantities: LSTev(x) = latest ready time = min LST over x's root activities;
    D(b) = min over data successors x of LSTev(x) (T_end if none); W(e) = S_last(e) - r(e).
    """

    def __init__(self, sc):
        self.sc = sc
        g = sc.g
        na = len(sc.a_s)
        a_s, a_f, a_par, a_ev, a_res = sc.a_s, sc.a_f, sc.a_par, sc.a_ev, sc.a_res
        rprev = [-1] * na
        for seq in sc.res_seq:
            for i in range(1, len(seq)):
                rprev[seq[i]] = seq[i - 1]
        self.rprev = rprev
        self.T = sc.T
        self.T_end = max([sc.T] + a_f)
        T_end = self.T_end
        lst = [T_end - (a_f[a] - a_s[a]) for a in range(na)]
        pred, pred_ptr, term = g.pred, g.pred_ptr, sc.term
        for a in range(na - 1, -1, -1):
            v = lst[a]
            p = a_par[a]
            if p >= 0:
                if v < lst[p]:
                    lst[p] = v
            else:
                e = a_ev[a]
                for j in range(pred_ptr[e], pred_ptr[e + 1]):
                    u = term[pred[j]]
                    w = v - (a_f[u] - a_s[u])
                    if w < lst[u]:
                        lst[u] = w
            u = rprev[a]
            if u >= 0:
                w = v - (a_f[u] - a_s[u])
                if w < lst[u]:
                    lst[u] = w
        self.lst = lst
        self.slack = [lst[a] - a_s[a] for a in range(na)]
        if min(self.slack, default=0) < 0:
            raise CPCMError("negative slack: the schedule violates its own extended graph")
        self.lst_ev = [min((lst[a] for a in sc.ev_acts[e] if a_par[a] < 0), default=T_end) for e in range(g.n)]
        self.D = [min((self.lst_ev[g.succ[k]] for k in range(g.succ_ptr[b], g.succ_ptr[b + 1])), default=T_end)
                  for b in range(g.n)]
        self._cp = None

    # earliest start of an activity ignoring its resource
    def can(self, a):
        sc = self.sc
        p = sc.a_par[a]
        return sc.a_s[p] if p >= 0 else sc.r[sc.a_ev[a]]

    def critical_path(self):
        """Critical chain from the latest-finishing event back to time 0 (E2's rule). Rows in time order:
        dict(act, event, res, s, f, can, wait, edge, pred_act). edge: how this row's start is explained:
        "resource" (wait > 0: the previous activity on the resource ends at s), "hop" (parent issued at s),
        "data" (ready at s = F of the latest predecessor), "start"."""
        if self._cp is not None:
            return self._cp
        sc, g = self.sc, self.sc.g
        last = max(range(g.n), key=lambda e: (sc.F[e], e))
        cur = sc.term[last]
        rows = []
        self.gaps = 0
        while cur >= 0:
            s, c = sc.a_s[cur], self.can(cur)
            e = sc.a_ev[cur]
            row = {"act": cur, "event": e, "res": sc.a_res[cur], "s": s, "f": sc.a_f[cur], "can": c,
                   "wait": s - c}
            if s > c:
                u = self.rprev[cur]
                if u < 0 or sc.a_f[u] != s:
                    self.gaps += 1      # resource idle before s (order mode only); explain by the predecessor anyway
                    if u < 0:
                        raise CPCMError(f"positive wait with no resource predecessor at activity {cur}")
                row.update(edge="resource", pred_act=u)
                nxt = u
            elif sc.a_par[cur] >= 0:
                nxt = sc.a_par[cur]
                row.update(edge="hop", pred_act=nxt)
            elif g.pred_ptr[e + 1] > g.pred_ptr[e]:
                p = max((g.pred[j] for j in range(g.pred_ptr[e], g.pred_ptr[e + 1])), key=lambda x: (sc.F[x], x))
                if sc.F[p] != s:
                    raise CPCMError(f"chain discontinuity at activity {cur}")
                nxt = sc.term[p]
                row.update(edge="data", pred_act=nxt)
            else:
                nxt = -1
                row.update(edge="start", pred_act=-1)
            rows.append(row)
            cur = nxt
        rows.reverse()
        self._cp = rows
        self._attribute()
        return rows

    def _attribute(self):
        sc = self.sc
        cb, cb_excl, phi = {}, {}, {}
        claimed = []
        for row in reversed(self._cp):          # latest wait first (E2's exclusive crediting)
            if row["edge"] != "resource":
                continue
            q, lo, hi = row["res"], row["can"], row["s"]
            cb[q] = cb.get(q, 0) + hi - lo
            free = hi - lo
            for c0, c1 in claimed:
                free -= max(0, min(hi, c1) - max(lo, c0))
            cb_excl[q] = cb_excl.get(q, 0) + free
            claimed = _merge(claimed + [(lo, hi)])
            shares = {}
            u = self.rprev[row["act"]]
            while u >= 0 and sc.a_f[u] > lo:
                ov = min(sc.a_f[u], hi) - max(sc.a_s[u], lo)
                if ov > 0:
                    b = sc.a_ev[u]
                    shares[b] = shares.get(b, 0) + ov
                u = self.rprev[u]
            row["phi"] = shares
            dq = phi.setdefault(q, {})
            for b, v in shares.items():
                dq[b] = dq.get(b, 0) + v
        self.cb, self.cb_excl, self.phi = cb, cb_excl, phi

    # ---- interfaces used downstream
    def wait(self, e):
        return self.sc.S_last[e] - self.sc.r[e]

    def blocking(self, q=None):
        """CB(q) and {blocker event: phi} on resource q; with q None, {q: CB} for every resource with CB > 0."""
        self.critical_path()
        if q is None:
            return dict(self.cb)
        return self.cb.get(q, 0), dict(self.phi.get(q, {}))

    def deadline(self, b):
        return self.D[b]

    def critical_events(self):
        return sorted({row["event"] for row in self.critical_path()})


def _merge(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


# ------------------------------------------------------------------ single-message reroute (full recompute for now)
def reroute(g, b, branches, link_dur, mode="bb", prio=None, base=None):
    """Evaluate moving communication b to `branches` (lists of resource ids). Returns dict(T, F_b, D_b, ok)
    where D_b is the deadline of b in the CURRENT schedule (base Analysis, computed if not given).
    TODO(V7): route through the incremental update instead of a full recompute."""
    if base is None:
        base = Analysis(simulate(g, mode, prio))
    g2 = g.copy()
    g2.set_branches(b, branches, link_dur)
    sc2 = simulate(g2, mode, prio)
    return {"T": sc2.T, "F_b": sc2.F[b], "D_b": base.D[b], "ok": sc2.F[b] <= base.D[b], "schedule": sc2}
