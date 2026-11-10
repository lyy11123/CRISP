"""-A3h arm's repairs: EIR L2Gsync on C's A1 starts (D_PLAN.md v2 section 4; PROGRESS 2026-11-11 rule 11).

Rules as C (c_report.stop): 5 SA seeds (123, 1, 2, 3, 4), eps = 2 %; R0, then L1, within eps (T-hat <= T_pre x 1.02) -> stop there,
no L2 run; else 30 -> 100 -> 300 steps, stop on entering eps or a tier improving the 5-seed mean T-hat by less than 2 %; else 300.
Tiers run in order per (fault, operator): the 5 seeds of a tier in parallel, the next tier only if the rule says so. Every job is
c_eir's child (kind rep, start a1), reading C's prep sinks (pre, l1) of this machine; C's files are only read.

  python e2e/d_rep.py plan                        pairs needing L2 (from c_report.json) and the estimate
  python e2e/d_rep.py run [--workers N] [--shard i --nshards M] [--out DIR]
  python e2e/d_rep.py levels [--out DIR]          the stopped level per (fault, operator) -> OUT/levels.json
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import d_joint as d  # noqa: E402  (puts the main checkout's e2e first: c_eir from there)
import c_eir as c  # noqa: E402

E = c.eir
EPS, TIER_GAIN = E.EPS, 0.02
OUT_DEFAULT = os.path.join(d.OUT, "a1rep")
PREP = os.path.join(d.ME2E, "c", "out", "prep")


def rep():
    return d.creport()


def pairs():
    out = []
    for fid, f in rep().items():
        for op, o in f["ops"].items():
            a = o["a1"]
            lim = a["pre"]["T"] * (1 + EPS)
            if a["R0"]["T"] <= lim:
                lvl = "R0"
            elif a["L1"]["T"] <= lim:
                lvl = "L1"
            else:
                lvl = None
            out.append({"fault": fid, "op": op, "T_pre": a["pre"]["T"], "T_R0": a["R0"]["T"], "T_L1": a["L1"]["T"], "level0": lvl})
    return out


def job(out, op, fid, steps, seed):
    j = E.job_rep(os.path.join(out, "rep"), "a1", op, fid, "L2Gsync", steps, seed)
    j["pre_sink"] = E.job_pre(PREP, "a1", op)["sink"]
    j["l1_sink"] = E.job_l1(PREP, "a1", op, fid)["sink"]
    for k in ("pre_sink", "l1_sink"):
        if not os.path.exists(j[k]):
            raise SystemExit(f"C's prep sink missing: {j[k]}")
    return j


def tier_T(out, op, fid, steps):
    T = []
    for sd in c.SEEDS:
        s = E.read_sink(job(out, op, fid, steps, sd)["sink"])
        T.append(s["record"]["T"])
    return statistics.mean(T)


def decide(p, tiers):
    """c_report.stop for the tiers seen so far: (level, why) or None (run the next tier)."""
    lim = p["T_pre"] * (1 + EPS)
    prev = None
    for s in c.STEPS:
        if s not in tiers:
            return None
        T = tiers[s]
        if T <= lim:
            return f"s{s}", "进 ε"
        if prev is not None and (prev - T) / prev < TIER_GAIN:
            return f"s{s}", f"档改进 {100 * (prev - T) / prev:+.1f}%"
        prev = T
    return f"s{c.STEPS[-1]}", "预算用尽"


def run(workers, out, shard=0, nshards=1):
    c.redirect()
    todo = [p for p in pairs() if p["level0"] is None]
    todo.sort(key=lambda p: (-c.L1_MEAS[p["op"]], p["fault"], p["op"]))
    todo = todo[shard::nshards]
    os.makedirs(out, exist_ok=True)
    print(f"[a1rep] {len(todo)} (fault, operator) pairs, {workers} workers", flush=True)
    pool = ThreadPoolExecutor(workers)
    log = os.path.join(out, f"runs_{shard}.jsonl")
    lock = threading.Lock()
    failed = []

    def one(j):
        while d.free_gb() < d.LOCAL_FREE_GB + 1.5:
            time.sleep(20)
        r = E.run_job(j, out)
        with lock, open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"name": r[0], "status": r[1], "wall_s": r[2], "op": j["op"], "t": time.strftime("%H:%M:%S")}) + "\n")
        if r[1] == "failed":
            failed.append(r[0])
        return r

    def pair(p):
        tiers = {}
        for s in c.STEPS:
            if failed:
                return
            rs = list(pool.map(one, [job(out, p["op"], p["fault"], s, sd) for sd in c.SEEDS]))
            if any(r[1] == "failed" for r in rs):
                return
            tiers[s] = tier_T(out, p["op"], p["fault"], s)
            dec = decide(p, tiers)
            print(f"[a1rep] {p['fault']} {p['op']} s{s}: mean T-hat {tiers[s]:.0f} (pre {p['T_pre']}) -> {dec or 'next tier'}", flush=True)
            if dec:
                return
    with ThreadPoolExecutor(max(1, len(todo))) as px:
        list(px.map(pair, todo))
    pool.shutdown()
    print(f"[a1rep] done; failed {failed}", flush=True)
    if failed:
        raise SystemExit(1)


def levels(out):
    c.redirect()
    res = {}
    for p in pairs():
        fid, op = p["fault"], p["op"]
        if p["level0"]:
            lvl, why, tiers = p["level0"], "进 ε", {}
        else:
            tiers = {}
            dec = None
            for s in c.STEPS:
                if not all(os.path.exists(job(out, op, fid, s, sd)["sink"]) for sd in c.SEEDS):
                    break
                tiers[s] = tier_T(out, op, fid, s)
                dec = decide(p, tiers)
                if dec:
                    break
            if not dec:
                res.setdefault(fid, {})[op] = {"level": None, "tiers": tiers}
                continue
            lvl, why = dec
        res.setdefault(fid, {})[op] = {"level": lvl, "why": why, "tiers": tiers,
                                       "steps": 0 if lvl in ("R0", "L1") else int(lvl[1:]),
                                       "sinks": [job(out, op, fid, int(lvl[1:]), sd)["sink"] for sd in c.SEEDS] if lvl not in ("R0", "L1") else None}
    with open(os.path.join(out, "levels.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--free-gb", type=float)
    a = ap.parse_args()
    if a.free_gb:
        d.LOCAL_FREE_GB = a.free_gb
    out = os.path.abspath(a.out)
    if a.cmd == "plan":
        ps = pairs()
        need = [p for p in ps if p["level0"] is None]
        up = sum(5 * sum(c.REP_X[s] for s in c.STEPS) * c.L1_MEAS[p["op"]] for p in need) / 3600
        lo = sum(5 * c.REP_X[30] * c.L1_MEAS[p["op"]] for p in need) / 3600
        print(f"{len(ps)} pairs; {len(need)} need L2; local process h: s30 only {lo:.0f}, all 3 tiers {up:.0f}")
    elif a.cmd == "run":
        run(a.workers, out, a.shard, a.nshards)
    elif a.cmd == "levels":
        r = levels(out)
        print(json.dumps({f: {o: v["level"] for o, v in ops.items()} for f, ops in r.items()}, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
