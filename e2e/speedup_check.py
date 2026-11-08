"""Regression of the hop-path speed-up (E2E_HANDOFF.md section 3): the original and the new implementation on the same input, outputs
compared bit for bit (no tolerance). One process per object (child), results in out/speedup/<name>.json.

  python speedup_check.py pilot <readonly spec json> <out json> [--seeds 1 2 3 4] [--parts hop c4] [--new-only]

Hashing: the original's issued_events are sets of event objects, iterated in hash order; the event class hashes by address, so
the backend result depends on where the objects sit in memory (known since E0; the same state run twice in one process can differ).
For a bit-for-bit comparison both implementations run with the event class hashing by its tag (patched in memory, for both, in
this process only). The new code performs the same set operations in the same order, so under any hash function it iterates the
sets as the original does; equality under the tag hash is the test of that.
"""
import copy
import hashlib
import json
import os
import pickle
import random
import sys
import time

import numpy as np

import e2e_pilot as P

C = P.C


def digest(x):
    return hashlib.sha256(pickle.dumps(x, protocol=4)).hexdigest()


def prepare(spec):
    """readonly_child up to the chunked events: state + broadcast (seed 0) + full-link 256 KB chunking (o = 0)."""
    sys.path.insert(0, os.path.join(P.ROOT, "dstu"))
    C._artifact_path()
    import dstu as D
    import dstu2 as S
    from dstu import E
    from add_communication import add_broadcast
    import gzip
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
    ev, cst = S.chunk_by(E.merged_events([pickle.dumps(events, protocol=pickle.HIGHEST_PROTOCOL)]), hp, S.UNI["256all"])
    return D, ev, hp


def tag_hash():
    from event_notation import event_notation
    event_notation.__hash__ = lambda self: hash(self.event_tag)


def backend_outputs(ev, hp, triple, audit, released):
    return dict(triple=list(triple),
                times=digest(sorted((repr(t), e.start_time, e.end_time) for t, e in ev.items())),
                latency=max(e.end_time for e in ev.values()),
                links=digest(sorted((repr(k), [list(map(repr, r)) for r in L.work_record]) for k, L in hp.links_dict.items())),
                devs=digest(sorted((repr((d, k)), [list(map(repr, r)) for r in m.work_record]) for d, mm in hp.modules_dict.items() for k, m in mm.items())),
                audit=digest([tuple(map(repr, a)) for a in audit]), n_audit=len(audit), released=repr(sorted(released.items())),
                rng_after=digest(random.getstate()))


def sched_outputs(sc, an=None, td=None):
    o = dict(T=sc.T, n_act=len(sc.a_s),
             sched=digest([sc.r, sc.S, sc.S_last, sc.F, sc.term, sc.ev_acts, sc.a_ev, sc.a_res, sc.a_s, sc.a_f, sc.a_par, sc.res_seq]))
    if an is not None:
        o["analysis"] = digest([an.lst, an.slack, an.lst_ev, an.D, an.rprev, an.T_end])
    if td is not None:
        o["tail"] = digest(sorted((repr(k), repr(v)) for k, v in td.items()))
    return o


def run_c4(sim, D, g):
    from dstu import cpcm
    t0 = time.perf_counter()
    sc = sim(g, "hop")
    t1 = time.perf_counter()
    an = cpcm.Analysis(sc, sink_deadline=[sc.T] * g.n)
    td = D.tail_data(g, sc, an, sc.T, range(g.n))
    o = sched_outputs(sc, an, td)
    o.update(sim_s=t1 - t0, total_s=time.perf_counter() - t0)
    return o


def run_backend(fn, base, seed, **kw):
    ev, hp = copy.deepcopy(base)
    audit, released = [], {}
    random.seed(seed)
    t0 = time.perf_counter()
    tri = fn(ev, hp, audit=audit, released=released, **kw)
    wall = time.perf_counter() - t0
    out = backend_outputs(ev, hp, tri, audit, released)
    out["wall_s"] = wall
    return out


def pilot(spec_path, out_path, seeds, parts, new_only):
    spec = json.load(open(spec_path, encoding="utf-8"))
    D, ev, hp = prepare(spec)
    tag_hash()
    from e4b_backend import P1, event_driver_e4b
    from fast_hop import event_driver_fast
    res = dict(name=os.path.basename(out_path)[:-5], spec=spec, seeds=seeds, parts=parts, events_chunked=len(ev))
    base = copy.deepcopy((ev, hp))
    if "c4" in parts:
        import fast_cpcm
        from dstu import cpcm
        g = D._graph(ev, hp)
        new = run_c4(fast_cpcm.simulate, D, g)
        old = None if new_only else run_c4(cpcm.simulate, D, g)
        same = None if old is None else all(old[k] == new[k] for k in new if not k.endswith("_s"))
        res["c4"] = dict(old=old, new=new, identical=same, diff=None if old is None else [k for k in new if not k.endswith("_s") and old[k] != new[k]])
        print(f"[c4] old {old and old['total_s']:.1f} s new {new['total_s']:.1f} s identical {same} T {new['T']}", flush=True)
    if "hop" in parts:
        rows = []
        for s in seeds:
            new = run_backend(event_driver_fast, base, s, release={0: 0}, policy=P1())
            old = None if new_only else run_backend(event_driver_e4b, base, s, release={0: 0}, policy=P1())
            same = None if old is None else all(old[k] == new[k] for k in new if k != "wall_s")
            rows.append(dict(seed=s, old=old, new=new, identical=same, diff=None if old is None else [k for k in new if k != "wall_s" and old[k] != new[k]]))
            print(f"[hop] seed {s}: old {old and old['wall_s']:.1f} s new {new['wall_s']:.1f} s identical {same} latency {new['latency']}", flush=True)
            json.dump(dict(res, hop=rows), open(out_path + ".part", "w"), indent=1, default=str)
        res["hop"] = rows
    res["peak_wset_gb"] = P.peak() / 2 ** 30
    json.dump(res, open(out_path, "w"), indent=1, default=str)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    ap.add_argument("--parts", nargs="*", default=["hop"])
    ap.add_argument("--new-only", action="store_true")
    a = ap.parse_args()
    if a.cmd == "pilot":
        pilot(a.spec, a.out, a.seeds, a.parts, a.new_only)
