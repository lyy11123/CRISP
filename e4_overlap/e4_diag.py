"""E4 diagnosis (user request 2026-11-02): why stream priority barely helps after a fault and in a2. Measurement
only, on cached event graphs (e4_out/sims/; the a1 no-fault ones re-simulated by e4_diag_sims.py).

1. Per-stream critical chain of the short stream (a1: MHA 512, a2: MHA 2048 = stream 0): E2's chain walk
   (e2_attribution.Sim.chain, copied loop) started from the stream's own last-finishing event instead of the
   globally last one, with SimR's release handling. Rows clipped to [r_i, C_i]:
     - latency split = time on the chain per resource class (compute / onchip / d2d / ddr / release) and per
       owner of the activity in service (own stream / other stream);
     - critical wait (E2's exclusive rule, recomputed on the clipped intervals) split by blocked owner x blocker
       owner, and for other-stream blockers into "in flight" (blocker started on that resource before the
       blocked activity could start) vs "arbitration" (blocker started at or after);
     - path-free timeline (added because a chain that passes into the other stream runs in parallel with the
       short stream's own progress): share of the latency with some activity of the short stream in service,
       or stalled with every pending activity waiting behind own / other (in flight, arbitration) occupants;
     - all activities of the short stream (not only the chain): raw sum of positive waits by resource class and
       blocker owner (overlapping, not additive to the latency); compute waits behind the other stream.
2. Message sizes: communications crossing a ch2ch link per stream (bytes min / median / max / count) and the
   number of messages and blocks for 256 KB and 64 KB chunking of D2D messages only.
Writes e4_out/diag/diag.json and diag_rows.csv.

  python e4_diag.py --jobs 6
"""
import argparse
import collections
import gzip
import json
import os
import pickle
import statistics
from concurrent.futures import ProcessPoolExecutor

import e4_attr as A
import e4_overlap as E

SIM = os.path.join(E.OUT, "sims")
DIAG = os.path.join(E.OUT, "diag")
CONFIGS = {  # name -> {variant: sim file}; the short stream is stream 0 everywhere
    "a1_512-8192_nofault_d0": {"default": "a1_2s_512-8192_d0_faultsel_default", "prio_01": "a1_2s_512-8192_d0_prio_01"},
    "a1_512-8192_nofault_d0.5": {"default": "a1_2s_512-8192_d0.5_default", "prio_01": "a1_2s_512-8192_d0.5_prio_01"},
    "a1_512-8192_F4_d0": {"default": "a1_2s_512-8192_F4_d0_default", "prio_01": "a1_2s_512-8192_F4_d0_prio_01"},
    "a1_512-8192_F4_d0.5": {"default": "a1_2s_512-8192_F4_d0.5_default", "prio_01": "a1_2s_512-8192_F4_d0.5_prio_01"},
    "a2_mha-ffn_nofault_d0": {"default": "a2_2s_mha-ffn_d0_default", "prio_01": "a2_2s_mha-ffn_d0_prio_01"},
    "a2_mha-ffn_F4_d0": {"default": "a2_2s_mha-ffn_F4_d0_default", "prio_01": "a2_2s_mha-ffn_F4_d0_prio_01"},
}
SHORT = 0
COMPUTE_KINDS = {"tensorcore", "vectorunit"}


def chain_for(sim, i):
    """E2 Sim.chain's loop from stream i's last-finishing event (rows latest first, then reversed)."""
    last = max((t for t in sim.ev if A.stream_of(t) == i), key=lambda t: (sim.ev[t]["f"], str(t)))
    cur, t_out = sim.terminal[last], sim.ev[last]["f"]
    rows, segs = [], []
    while cur is not None:
        X = sim.act[cur]
        t = X["tag"]
        e = sim.ev[t]
        s, can = X["s"], X["can"]
        row = {"act": cur, "tag": t, "owner": A.stream_of(t), "kind": sim.res_kind(X["res"]), "s": s, "t_out": t_out}
        if s > can:
            b = sim.occupant_ending(X["res"], s, t)
            if b is None:
                raise E.E4Error(f"positive wait on {cur} without an occupant ending at {s}")
            nxt = sim._act_of(X["res"], b)
            B = sim.act[nxt]
            segs.append({"blocked": cur, "blocker": nxt, "blocked_owner": A.stream_of(t), "blocker_owner": A.stream_of(b),
                         "kind": sim.res_kind(X["res"]), "can": can, "s": s, "blocker_res_s": B["s"]})
        elif X["prev"] is not None:
            nxt = X["prev"]
        elif e["deps"]:
            p = max(e["deps"], key=lambda d: (sim.ev[d]["f"], str(d)))
            nxt = sim.terminal[p]
        else:
            nxt = None
        rows.append(row)
        if nxt is None and s > 0:  # rows stay latest first
            rows.append({"act": None, "tag": None, "owner": A.stream_of(t), "kind": "release", "s": 0, "t_out": s})
        cur, t_out = nxt, s
    return last, rows, segs


def clip(lo, hi, a, b):
    return max(0, min(hi, b) - max(lo, a))


def own(o, i):
    return "own" if o == i else "other"


def analyse_one(cap, i=SHORT):
    sim = A.SimR(cap)
    rel = cap["release"]
    C, _ = A.stream_times(cap)
    r_i, C_i = rel[i], C[i]
    last, rows, segs = chain_for(sim, i)
    lat = collections.Counter()
    for rw in rows:
        d = clip(rw["s"], rw["t_out"], r_i, C_i)
        if d:
            cls = "release" if rw["kind"] == "release" else A.kind_class(rw["kind"])
            lat[f"{cls}|{own(rw['owner'], i)}"] += d
    tiled = sum(lat.values())
    # exclusive critical wait on the clipped intervals (E2's rule: latest blocked start first)
    wait = collections.Counter()
    claimed = []
    for sg in segs:
        lo, hi = max(sg["can"], r_i), min(sg["s"], C_i)
        if hi <= lo:
            continue
        free = hi - lo
        for c0, c1 in claimed:
            free -= clip(c0, c1, lo, hi)
        claimed.append((lo, hi))
        claimed = E._e2()._merge(claimed)
        if free <= 0:
            continue
        cls = A.kind_class(sg["kind"])
        bo = own(sg["blocker_owner"], i)
        mode = "" if bo == "own" else ("|inflight" if sg["blocker_res_s"] < sg["can"] else "|arbitration")
        wait[f"{own(sg['blocked_owner'], i)}<-{bo}{mode}|{cls}"] += free
        if sg["kind"] in COMPUTE_KINDS and bo == "other":
            wait[f"compute_unit_behind_other|{sg['kind']}"] += free
    # every activity of stream i (raw, overlapping)
    allw = collections.Counter()
    nwait = collections.Counter()
    for k, X in sim.act.items():
        t = X["tag"]
        if A.stream_of(t) != i or X["s"] <= X["can"]:
            continue
        b = sim.occupant_ending(X["res"], X["s"], t)
        bo = "none" if b is None else own(A.stream_of(b), i)
        mode = ""
        if b is not None and bo == "other":
            mode = "|inflight" if sim.act[sim._act_of(X["res"], b)]["s"] < X["can"] else "|arbitration"
        key = f"{bo}{mode}|{A.kind_class(sim.res_kind(X['res']))}"
        allw[key] += X["s"] - X["can"]
        nwait[key] += 1
        if sim.res_kind(X["res"]) in COMPUTE_KINDS and bo == "other":
            allw["compute_unit_behind_other"] += X["s"] - X["can"]
            nwait["compute_unit_behind_other"] += 1
    return {"latency": C_i - r_i, "release": r_i, "C": C_i, "chain_tiled": tiled, "chain_rows": len(rows),
            "timeline": timeline(sim, i, r_i, C_i),
            "chain_last_event": str(last), "latency_split": dict(lat), "critical_wait": dict(wait),
            "critical_wait_total": sum(v for k, v in wait.items() if not k.startswith("compute_unit")),
            "all_waits": dict(allw), "all_waits_n": dict(nwait),
            "global_chain_ends_in_stream": A.stream_of(max(cap["events"], key=lambda t: (cap["events"][t]["f"], str(t))))}


def timeline(sim, i, r_i, C_i):
    """Path-free state of stream i at each instant of [r_i, C_i]: 'service' = some activity of stream i is on
    its resource; otherwise the waiting activities of stream i (can <= t < s) are labelled by the occupant of
    their resource at t: own / other_inflight (occupant started before the waiting activity's can) /
    other_arbitration / free (resource idle). Shares of the latency per class, plus the class of stream i's
    in-service resource and the stall time behind the other stream on compute units."""
    ev = []
    for k, X in sim.act.items():
        if A.stream_of(X["tag"]) != i:
            continue
        kind = sim.res_kind(X["res"])
        ev += [(X["s"], ("svc", A.kind_class(kind)), 1), (X["f"], ("svc", A.kind_class(kind)), -1)]
        if X["s"] <= X["can"]:
            continue
        lo, hi = X["can"], X["s"]
        t = lo
        comp = kind in COMPUTE_KINDS
        for s, f, tag in sim.res[X["res"]]:
            if f <= lo or s >= hi or tag == X["tag"]:
                continue
            a, b = max(s, lo), min(f, hi)
            if a > t:
                ev += [(t, ("wait", "free", comp), 1), (a, ("wait", "free", comp), -1)]
            lab = "own" if A.stream_of(tag) == i else ("other_inflight" if s < lo else "other_arbitration")
            ev += [(a, ("wait", lab, comp), 1), (b, ("wait", lab, comp), -1)]
            t = max(t, b)
        if t < hi:
            ev += [(t, ("wait", "free", comp), 1), (hi, ("wait", "free", comp), -1)]
    ev.sort(key=lambda x: x[0])
    cnt = collections.Counter()
    out = collections.Counter()
    last = r_i
    for t, key, d in ev + [(C_i, None, 0)]:
        a, b = max(last, r_i), min(t, C_i)
        if b > a:
            dur = b - a
            svc = [k[1] for k, v in cnt.items() if v > 0 and k[0] == "svc"]
            w = {k[1] for k, v in cnt.items() if v > 0 and k[0] == "wait"}
            if svc:
                out["service"] += dur
                for c in set(svc):
                    out[f"service_on|{c}"] += dur / len(set(svc))
            else:
                oth = {x for x in w if x.startswith("other")}
                if not w:
                    lab = "nothing_pending"
                elif w <= {"free"}:
                    lab = "stalled|resource_free_only"
                elif oth and "own" not in w:
                    lab = "stalled|other_only|" + ("+".join(sorted(x[6:] for x in oth)))
                elif "own" in w and not oth:
                    lab = "stalled|own_only"
                else:
                    lab = "stalled|own+other"
                out[lab] += dur
                if any(v > 0 and k[0] == "wait" and k[1].startswith("other") and k[2] for k, v in cnt.items()):
                    out["stalled_with_compute_wait_behind_other"] += dur
            if any(v > 0 and k[0] == "wait" and k[1].startswith("other") and k[2] for k, v in cnt.items()):
                out["any_time_compute_wait_behind_other"] += dur
        if key is not None:
            cnt[key] += d
        last = t
    return dict(out)


def msg_sizes(cap):
    lt = cap["link_types"]
    out = {}
    for i in sorted({A.stream_of(t) for t in cap["events"]}):
        d2d, allc = [], 0
        for t, e in cap["events"].items():
            if A.stream_of(t) != i or e["type"] != "communication":
                continue
            allc += 1
            if any(lt[tuple(k)] == "ch2ch" for br in (e["branches"] or []) for k in map(tuple, br)):
                d2d.append(e["bytes"])
        row = {"n_comm": allc, "n_d2d": len(d2d)}
        if d2d:
            row.update(min=min(d2d), median=statistics.median(d2d), max=max(d2d), total=sum(d2d))
        for kb in (256, 64):
            sz = kb * 1024
            big = [b for b in d2d if b > sz]
            row[f"chunk{kb}k_messages"] = len(big)
            row[f"chunk{kb}k_blocks"] = sum(-(-b // sz) for b in big)
            row[f"chunk{kb}k_blocks_added"] = sum(-(-b // sz) - 1 for b in big)
        out[i] = row
    return out


def load(name):
    with gzip.open(os.path.join(SIM, f"{name}.pkl.gz"), "rb") as f:
        return pickle.load(f)["runs"]


def work(item):
    cname, var, fname = item
    res = []
    for r in load(fname):
        a = analyse_one(r["cap"])
        a.update(config=cname, variant=var, seed=r["seed"], sim=fname)
        if var == "default" and r["seed"] == 1:
            a["msg_sizes"] = msg_sizes(r["cap"])
            a["n_events"] = len(r["cap"]["events"])
            a["sim_wall"] = r["wall"]
        res.append(a)
    return res


def mean_dicts(ds):
    keys = sorted({k for d in ds for k in d})
    return {k: sum(d.get(k, 0) for d in ds) / len(ds) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(DIAG, exist_ok=True)
    items = [(c, v, f) for c, vs in CONFIGS.items() for v, f in vs.items()]
    with ProcessPoolExecutor(args.jobs) as ex:
        allres = [a for rs in ex.map(work, items) for a in rs]
    summ = {}
    for c in CONFIGS:
        summ[c] = {}
        for v in CONFIGS[c]:
            rs = [a for a in allres if a["config"] == c and a["variant"] == v]
            L = sum(a["latency"] for a in rs) / len(rs)
            ls = mean_dicts([a["latency_split"] for a in rs])
            cw = mean_dicts([a["critical_wait"] for a in rs])
            aw = mean_dicts([a["all_waits"] for a in rs])
            summ[c][v] = {"n_seeds": len(rs), "latency_mean": L, "latency_seeds": [a["latency"] for a in rs],
                          "chain_tiled_over_latency": [a["chain_tiled"] / a["latency"] for a in rs],
                          "latency_split_share": {k: x / L for k, x in ls.items()},
                          "critical_wait_share_of_latency": {k: x / L for k, x in cw.items()},
                          "timeline_share": {k: x / L for k, x in mean_dicts([a["timeline"] for a in rs]).items()},
                          "critical_wait_total_share": sum(a["critical_wait_total"] for a in rs) / len(rs) / L,
                          "all_waits_over_latency": {k: x / L for k, x in aw.items()},
                          "all_waits_share": {k: x / sum(v for kk, v in aw.items() if kk != "compute_unit_behind_other")
                                              for k, x in aw.items()},
                          "all_waits_n_mean": mean_dicts([a["all_waits_n"] for a in rs]),
                          "global_chain_ends_in_stream": [a["global_chain_ends_in_stream"] for a in rs]}
            m = next((a for a in rs if "msg_sizes" in a), None)
            if m:
                summ[c][v].update(msg_sizes=m["msg_sizes"], n_events=m["n_events"], sim_wall_seed1=m["sim_wall"])
    with open(os.path.join(DIAG, "diag.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summ, "runs": allres, "configs": CONFIGS, "short_stream": SHORT,
                   "artifact_commit": E._artifact_commit()}, f, indent=2, default=str)
    rows = []
    for a in allres:
        r = {k: a[k] for k in ["config", "variant", "seed", "latency", "release", "chain_tiled", "global_chain_ends_in_stream"]}
        r.update({f"lat:{k}": v for k, v in a["latency_split"].items()})
        r.update({f"cw:{k}": v for k, v in a["critical_wait"].items()})
        r.update({f"all:{k}": v for k, v in a["all_waits"].items()})
        rows.append(r)
    E._write_csv(os.path.join("diag", "diag_rows.csv"), rows)
    print("[e4] diag done", flush=True)


if __name__ == "__main__":
    main()
