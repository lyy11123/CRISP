"""Bit-identical speed-up of cpcm.simulate in mode "hop" (E2E_HANDOFF.md section 3). cpcm is imported, not changed; the Schedule it
returns is the same object type with the same contents. Other modes, checkpoints and stop callbacks go to cpcm.simulate unchanged.

Changes against cpcm._loop (hop branch), all with the same result:
- per event a count of unfinished branches replaces `all(pos[b] == br_hi[b] ...)` / `any(pos[b] < br_hi[b] ...)`;
- per event the branches are kept by their next link (head); a round reads the heads instead of rescanning every branch. For each
  head the candidate is the branch with the most hops left, the first in ev_br order on ties, as the original's `seen` gives;
- issuing on link q advances exactly the branches whose head is q (the original's loop over ev_br with hop_link[pos] == q).
- from the second round of a tick on, only the events that won a link in the previous round are offered again (the others'
  heads are all on busy links by then: same candidate times, no issue).
Candidate keys, the link order (sorted ids), min() over the candidates, the loser times and the bookkeeping order are the original's.
"""
import heapq
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "cpcm"))
import cpcm  # noqa: E402
from cpcm import COMP, CPCMError, Schedule, _State  # noqa: E402

inf = float("inf")
_orig_simulate = cpcm.simulate      # kept, so that cpcm.simulate may be replaced by this module's simulate


def simulate(g, mode="bb", prio=None, release=None, rng=None, checkpoint_every=0):
    if mode != "hop" or checkpoint_every:
        return _orig_simulate(g, mode, prio=prio, release=release, rng=rng, checkpoint_every=checkpoint_every)
    sc = Schedule(g, mode)
    sc.prio, sc.release = prio, release
    st = _State()
    n = g.n
    st.cnt = [g.pred_ptr[e + 1] - g.pred_ptr[e] for e in range(n)]
    st.free = [0] * g.nres
    st.pos = list(g.br_lo)
    st.blast = [-1] * len(g.br_lo)
    st.last = [-1] * n
    st.todo, st.fin, st.heap, st.ticks = {}, {}, [], 0
    for e in range(n):
        if st.cnt[e] == 0:
            t0 = release.get(e, 0) if release else 0
            sc.r[e] = t0
            if t0 not in st.todo:
                st.todo[t0] = []
                heapq.heappush(st.heap, t0)
            st.todo[t0].append(e)
    _loop_hop(g, sc, st, prio, rng)
    return sc._done()


def _loop_hop(g, sc, st, prio, rng):
    kind, succ_ptr, succ = g.kind, g.succ_ptr, g.succ
    res_of, dur = g.res_of, g.dur
    ev_br, br_hi = g.ev_br, g.br_hi
    hop_link, hop_dur = g.hop_link, g.hop_dur
    r, S_last, F, term = sc.r, sc.S_last, sc.F, sc.term
    act = sc._act
    a_f = sc.a_f
    cnt, free, pos, blast, last = st.cnt, st.free, st.pos, st.blast, st.last
    todo, fin, heap = st.todo, st.fin, st.heap
    heads = {}      # comm event -> {link: [branches whose next hop is on it]}
    nleft = {}      # comm event -> branches not finished
    rank = {}       # branch -> its index in ev_br of its event (the original's tie order)

    def init(e):
        h, k = {}, 0
        for i, b in enumerate(ev_br[e]):
            rank[b] = i
            p = pos[b]
            if p < br_hi[b]:
                k += 1
                q = hop_link[p]
                lst = h.get(q)
                if lst is None:
                    h[q] = [b]
                else:
                    lst.append(b)
        heads[e] = h
        nleft[e] = k
        return h

    def at(dct, t, e):
        lst = dct.get(t)
        if lst is None:
            dct[t] = [e]
            heapq.heappush(heap, t)
        else:
            lst.append(e)

    prev_t = None
    while heap:
        t = heapq.heappop(heap)
        if t == prev_t:
            continue
        prev_t = t
        st.ticks += 1
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
        if rng is not None:
            rng.shuffle(ready)
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
        live = comms
        while True:
            cands = {}
            for e in live:
                h = heads.get(e)
                if h is None:
                    h = init(e)
                for q, brs in h.items():
                    if len(brs) == 1:
                        b = brs[0]
                        k = br_hi[b] - pos[b]
                    else:
                        b, k, rb = -1, -1, None
                        for c in brs:
                            kc = br_hi[c] - pos[c]
                            if kc > k or (kc == k and rank[c] < rb):
                                b, k, rb = c, kc, rank[c]
                    key = ((prio[e], -k, e) if prio is not None else (-k, e))
                    cs = cands.get(q)
                    if cs is None:
                        cands[q] = [(key, e, b)]
                    else:
                        cs.append((key, e, b))
            if not cands:
                break
            issued = False
            won = set()
            for q in sorted(cands):
                cs = cands[q]
                fq = free[q]
                if fq > t:
                    for _, e, _b in cs:
                        if fq < cand.get(e, inf):
                            cand[e] = fq
                    continue
                _, e, b = min(cs)
                h = pos[b]
                f = t + hop_dur[h]
                free[q] = f
                a = act(e, q, t, f, blast[b])
                last[e] = a
                won.add(e)
                hd = heads[e]
                for c in hd.pop(q):
                    p = pos[c] + 1
                    pos[c] = p
                    blast[c] = a
                    if p < br_hi[c]:
                        q2 = hop_link[p]
                        lst = hd.get(q2)
                        if lst is None:
                            hd[q2] = [c]
                        else:
                            lst.append(c)
                    else:
                        nleft[e] -= 1
                for _, x, _b in cs:
                    if x != e and f < cand.get(x, inf):
                        cand[x] = f
                issued = True
            # the next round needs only the events that won a link now: the others keep their heads, all on links now busy
            # (busy before, or just issued), so they would only get the same candidate times again and issue nothing
            live = [e for e in live if nleft[e] and e in won]
            if not issued or not live:
                break
        for e in comms:
            if not nleft[e]:
                heads.pop(e, None)
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
    return False
