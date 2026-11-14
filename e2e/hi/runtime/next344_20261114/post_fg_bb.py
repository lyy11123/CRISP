"""Deterministic BusyBarn CPCM forward pass, with a single-thread Numba kernel.

Only the deterministic bb forward pass is replaced. Graph construction and the
existing Schedule/Analysis interface are retained; each call packs its own graph.
"""
import heapq
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpcm"))
import cpcm

ORIGINAL = cpcm.simulate
TIMING = []


@njit(cache=True)
def forward(kind, succ_ptr, succ, res_of, dur, ev_ptr, ev_br, br_lo, br_hi,
            hop_link, hop_dur, hop_suf3, indegree, release, nres):
    n = len(kind)
    cnt = indegree.copy()
    free = np.zeros(nres, np.int64)
    pos = br_lo.copy()
    blast = np.full(len(br_lo), -1, np.int64)
    last = np.full(n, -1, np.int64)
    r = np.zeros(n, np.int64)
    S = np.full(n, -1, np.int64)
    S_last = np.full(n, -1, np.int64)
    F = np.full(n, -1, np.int64)
    term = np.full(n, -1, np.int64)
    cap = n + len(hop_link)
    a_ev = np.empty(cap, np.int64)
    a_res = np.empty(cap, np.int64)
    a_s = np.empty(cap, np.int64)
    a_f = np.empty(cap, np.int64)
    a_par = np.empty(cap, np.int64)
    na = 0
    heap = [(np.int64(0), np.int64(0), np.int64(0))]
    heap.pop()
    for e in range(n):
        if cnt[e] == 0:
            r[e] = release[e]
            heapq.heappush(heap, (release[e], np.int64(0), np.int64(e)))
    while heap:
        t = heap[0][0]
        ready = []
        while heap and heap[0][0] == t:
            _, typ, e = heapq.heappop(heap)
            if typ == 0:
                ready.append(e)
            else:
                for j in range(succ_ptr[e], succ_ptr[e + 1]):
                    x = succ[j]
                    cnt[x] -= 1
                    if cnt[x] == 0:
                        r[x] = t
                        ready.append(x)
        ready.sort()
        comms = []
        for e in ready:
            if kind[e] == 0:
                q = res_of[e]
                if free[q] > t:
                    heapq.heappush(heap, (free[q], np.int64(0), e))
                else:
                    f = t + dur[e]
                    free[q] = f
                    a_ev[na], a_res[na] = e, q
                    a_s[na], a_f[na], a_par[na] = t, f, -1
                    if S[e] < 0:
                        S[e] = t
                    term[e] = na
                    na += 1
                    S_last[e], F[e] = t, f
                    heapq.heappush(heap, (f, np.int64(1), e))
            else:
                comms.append(e)
        candidates = np.full(len(comms), np.iinfo(np.int64).max, np.int64)
        maxL = 0
        for e in comms:
            for j in range(ev_ptr[e], ev_ptr[e + 1]):
                b = ev_br[j]
                maxL = max(maxL, br_hi[b] - pos[b])
        for L in range(maxL, 0, -1):
            # Sort once at each length, as in the frozen bb implementation.
            keys = []
            for i in range(len(comms)):
                e = comms[i]
                maximum = 0
                at_level = False
                for j in range(ev_ptr[e], ev_ptr[e + 1]):
                    b = ev_br[j]
                    rem = br_hi[b] - pos[b]
                    maximum = max(maximum, rem)
                    at_level = at_level or rem == L
                if at_level:
                    keys.append((-maximum, e, i))
            keys.sort()
            for _, e, i in keys:
                branches = []
                for j in range(ev_ptr[e], ev_ptr[e + 1]):
                    b = ev_br[j]
                    if br_hi[b] - pos[b] == L:
                        branches.append((not hop_suf3[pos[b]], b))
                branches.sort()
                for _, b in branches:
                    h = pos[b]
                    if br_hi[b] - h != L:
                        continue
                    q = hop_link[h]
                    if free[q] > t:
                        candidates[i] = min(candidates[i], free[q])
                        continue
                    f = t + hop_dur[h]
                    free[q] = f
                    a_ev[na], a_res[na] = e, q
                    a_s[na], a_f[na], a_par[na] = t, f, blast[b]
                    if S[e] < 0:
                        S[e] = t
                    last[e] = na
                    for j in range(ev_ptr[e], ev_ptr[e + 1]):
                        c = ev_br[j]
                        p = pos[c]
                        if p < br_hi[c] and hop_link[p] == q:
                            pos[c] = p + 1
                            blast[c] = na
                    na += 1
        for i in range(len(comms)):
            e = comms[i]
            finished = True
            for j in range(ev_ptr[e], ev_ptr[e + 1]):
                b = ev_br[j]
                finished = finished and pos[b] == br_hi[b]
            if finished:
                a = last[e]
                term[e], S_last[e], F[e] = a, t, a_f[a]
                heapq.heappush(heap, (F[e], np.int64(1), e))
            else:
                assert candidates[i] != np.iinfo(np.int64).max
                heapq.heappush(heap, (candidates[i], np.int64(0), e))
    return r, S, S_last, F, term, a_ev[:na], a_res[:na], a_s[:na], a_f[:na], a_par[:na]


def packed(g, release):
    ev_ptr = [0]
    ev_br = []
    for branches in g.ev_br:
        ev_br.extend(branches)
        ev_ptr.append(len(ev_br))
    source_release = np.zeros(g.n, np.int64)
    for e, value in (release or {}).items():
        source_release[e] = value
    integer = lambda values: np.asarray(values, dtype=np.int64)
    return (integer(g.kind), integer(g.succ_ptr), integer(g.succ), integer(g.res_of),
            integer(g.dur), integer(ev_ptr), integer(ev_br), integer(g.br_lo), integer(g.br_hi),
            integer(g.hop_link), integer(g.hop_dur), np.asarray(g.hop_suf3, dtype=np.bool_),
            np.diff(integer(g.pred_ptr)), source_release, g.nres)


def simulate(g, mode="bb", prio=None, release=None, rng=None, checkpoint_every=0):
    if mode != "bb" or rng is not None or checkpoint_every:
        return ORIGINAL(g, mode, prio, release, rng, checkpoint_every)
    t0 = time.perf_counter()
    args = packed(g, release)
    t1 = time.perf_counter()
    values = forward(*args)
    t2 = time.perf_counter()
    sc = cpcm.Schedule(g, mode)
    sc.prio, sc.release = prio, release
    for key, value in zip(("r", "S", "S_last", "F", "term", "a_ev", "a_res", "a_s", "a_f", "a_par"), values):
        setattr(sc, key, value.tolist())
    for a, (e, q) in enumerate(zip(sc.a_ev, sc.a_res)):
        sc.ev_acts[e].append(a)
        sc.res_seq[q].append(a)
    sc._done()
    t3 = time.perf_counter()
    TIMING.append(dict(pack_seconds=t1-t0, kernel_seconds=t2-t1, unpack_seconds=t3-t2,
                       total_seconds=t3-t0, events=g.n, hops=len(g.hop_link)))
    return sc


def compare(g, release=None):
    original = ORIGINAL(g, "bb", release=release)
    accelerated = simulate(g, "bb", release=release)
    fields = ("r", "S", "S_last", "F", "term", "a_ev", "a_res", "a_s", "a_f", "a_par",
              "ev_acts", "res_seq", "T")
    for field in fields:
        assert getattr(original, field) == getattr(accelerated, field), field
    a, b = cpcm.Analysis(original), cpcm.Analysis(accelerated)
    for field in ("lst", "slack", "lst_ev", "D", "rprev"):
        assert getattr(a, field) == getattr(b, field), field
    assert a.critical_path() == b.critical_path()
    assert a.blocking() == b.blocking()
    return dict(events=g.n, activities=len(original.a_s), T=original.T,
                schedule_fields=list(fields), attribution_equal=True)
