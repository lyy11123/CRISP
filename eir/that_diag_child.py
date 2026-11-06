"""Child side of the read-only T̂ diagnosis (that_diag.py). No SA, no change to stored results: loads one saved state on its platform,
reruns the backend on deep copies and CPCM (deterministic and with the V3 diagnostic random ties), and decomposes each schedule with
E2's critical chain (e2_attribution.Sim). Imports nothing from eir: eir.child passes what it needs (eir may be __main__ there).
"""
import collections
import copy
import random
import statistics as st

import numpy as np

CLASSES = {"ch2ch": "D2D", "co2co": "片内链路", "ddr2co": "DDR 读", "co2ddr": "DDR 写", "tensorcore": "计算", "vectorunit": "计算"}
N_BACKEND = 10
N_RNG = 10
RNG_T_ONLY = 40


def cls(kind):
    return CLASSES.get(kind, "其他:" + str(kind))


def cpcm_cap(g, sc, cap0, plain):
    """E2's plain view of a CPCM schedule, so that the same chain code reads both."""
    events = {}
    for e in range(g.n):
        t = plain(g.tag[e])
        events[t] = dict(cap0["events"][t], s=int(sc.S[e]), f=int(sc.F[e]))
    devices, links = collections.defaultdict(list), collections.defaultdict(list)
    for a in range(len(sc.a_s)):
        name = g.res_name[sc.a_res[a]]
        rec = (int(sc.a_s[a]), int(sc.a_f[a]), plain(g.tag[sc.a_ev[a]]))
        if name[0] == "dev":
            devices[("dev", name[1], plain(name[2]))].append(rec)
        else:
            links[plain(name[1])].append(rec)
    return {"events": events, "devices": dict(devices), "links": dict(links), "link_types": cap0["link_types"],
            "result": (int(sc.T), 0, 0), "rng_digest": ""}


def metrics(e2, cap):
    """Chain occupancy (a partition of [first start, T] by the resource each chain activity holds), exclusive critical wait, and the wait
    of all activities, each by resource class; chain occupancy also by what the event carries."""
    sim = e2.Sim(cap)
    last, rows, segs = sim.chain()
    chain, chain_data, res = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        d = r["t_out"] - r["s"]
        chain[cls(r["resource_kind"])] += d
        chain_data[sim.data_class(r["tag"])] += d
        res[r["resource"]] += d
    excl = collections.Counter()
    for s in segs:
        excl[cls(s["resource_kind"])] += s["wait_excl"]
    wait = collections.Counter()
    for a in sim.act.values():
        wait[cls(sim.res_kind(a["res"]))] += a["s"] - a["can"]
    busy = collections.Counter()
    for a in sim.act.values():
        busy[cls(sim.res_kind(a["res"]))] += a["f"] - a["s"]
    return {"T": max(e["f"] for e in cap["events"].values()), "chain": dict(chain), "chain_data": dict(chain_data), "excl_wait": dict(excl), "wait": dict(wait), "busy": dict(busy),
            "top": dict(res.most_common(5)), "chain_len": len(rows), "chain_sum": sum(chain.values())}


def mean_rows(rows):
    out = {"T": st.mean(r["T"] for r in rows), "chain_len": st.mean(r["chain_len"] for r in rows), "chain_sum": st.mean(r["chain_sum"] for r in rows)}
    for k in ("chain", "chain_data", "excl_wait", "wait", "busy"):
        keys = {c for r in rows for c in r[k]}
        out[k] = {c: sum(r[k].get(c, 0) for r in rows) / len(rows) for c in keys}
    return out


def child(eng, spec, sink, read_sink, e2, cs, cpcm, sync_methods):
    src = spec.get("src")
    if src is None:
        sink["bc_seed"] = eng.seed_broadcast()
    else:
        d = read_sink(src["sink"])
        snap = d
        for k in src["path"]:
            snap = snap[k]
        eng.sync = d["spec"].get("method") in sync_methods
        eng.load(snap)
    s, hp = eng.a.state, eng.hp
    ev = eng.bc.graph(s[5], s[6], s[7], hp)
    g, sc = cs.cpcm_sim(ev, hp)
    sink["T_det"] = sc.T
    sink["T_stored"] = spec.get("T_stored")
    sink["T_equal"] = spec.get("T_stored") is None or sc.T == spec["T_stored"]
    pre = e2._snapshot_pre(ev)
    base = copy.deepcopy((ev, hp))
    rng = random.getstate(), np.random.get_state()
    caps = []
    for sd in range(1, N_BACKEND + 1):
        ev_k, hp_k = copy.deepcopy(base)
        random.seed(sd)
        np.random.seed(sd)
        res = tuple(int(v) for v in eng.backend(events_dict=ev_k, hardware_platform=hp_k))
        caps.append(e2._extract(ev_k, hp_k, pre, res, ""))
    random.setstate(rng[0])
    np.random.set_state(rng[1])
    cap0 = caps[0]
    sink["backend"] = [c["result"][0] for c in caps]
    sink["bk_mean"] = st.mean(sink["backend"])
    sink["m_backend"] = mean_rows([metrics(e2, c) for c in caps])
    sink["m_det"] = metrics(e2, cpcm_cap(g, sc, cap0, e2._plain))
    rows, ts = [], []
    for i in range(RNG_T_ONLY):
        s_r = cpcm.simulate(g, "bb", rng=random.Random(1000 + i))
        ts.append(s_r.T)
        if i < N_RNG:
            rows.append(metrics(e2, cpcm_cap(g, s_r, cap0, e2._plain)))
    sink["T_rng"] = ts
    sink["m_rng"] = mean_rows(rows)
    sink["backend_top"] = metrics(e2, caps[0])["top"]
