"""W1_RERUN_SUMMARY section 12: EIR (decode MHA) under W0 / W1 (old coordinates) / W1 (reselected coordinates), numbers only.
Rules: PROGRESS 2026-11-08 "W1 补跑比较规则（EIR 部分）" and 2026-11-09 (E3 start l1, 7 faults reselected). The same code reads the W0
sinks (eir/out, cpcm_sa stage 2, E3's full_e3) and the W1 sinks (out/eir_all = this machine's prep + colab_results/w1_eir).

  python e2e/w1_rerun/report_eir_w1.py            -> out/report_eir_w1.json, out/report_eir_w1.md
Backend: seeds 1-10 mean per run, then the mean over SA seeds (as report_batch). Stop level: make_summary.final_rule (eps = 2 %).
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
os.environ.setdefault("CPCM_SA_STAGE", "s2")
sys.path.insert(0, os.path.join(ROOT, "eir"))
import eir  # noqa: E402
import eir_next as en  # noqa: E402

cs = eir.cs
mean, median = statistics.mean, statistics.median
EPS, TIER_GAIN, TIERS = eir.EPS, 0.02, (30, 100, 300)
MIGF = ("moved_behas_share", "ddr_read_moved_bytes", "unique_block_bytes")
W1A = os.path.join(HERE, "out", "eir_all")
ENV = {"W0": {"main": os.path.join(ROOT, "eir", "out", "main"), "next": os.path.join(ROOT, "eir", "out", "next"), "w1sel": None},
       "W1": {"main": os.path.join(W1A, "main"), "next": os.path.join(W1A, "next"), "w1sel": os.path.join(W1A, "w1sel")}}
W1SEL = {"F4a": "F4aW", "F5a": "F5aW", "F8a": "F8aW", "CRandom": "CRandomW", "CCluster": "CClusterW", "CEdge": "CEdgeW", "CScratch": "CScratchW"}
OP = "mha"


def bk(b):
    return mean(v[0] for v in b.values())


def rd(p):
    return eir.read_sink(p) if p and os.path.exists(p) else None


def mig(r):
    m = r.get("migration_vs_start") if r else None
    return {f: m[f] for f in MIGF} if m else None


def lvl(runs):
    """Mean over SA seeds of T, backend mean and migration (report_batch)."""
    if not runs:
        return None
    ms = [mig(x["record"]) for x in runs]
    return {"n": len(runs), "T": mean(x["record"]["T"] for x in runs), "bk": mean(bk(x["record"]["backend"]) for x in runs),
            "bk_min_run": min(bk(x["record"]["backend"]) for x in runs), "bk_max_run": max(bk(x["record"]["backend"]) for x in runs),
            "mig": {f: mean(m[f] for m in ms) for f in MIGF} if all(ms) else None}


def fmig(pre, s):
    """FULL's migration against the start (EIR's migration_vs_start without the unique-block term: a FULL sink keeps no placement)."""
    if not (pre and s and s.get("e3map")):
        return None
    ref, now = pre["record"], s["e3map"]
    moved = sum(tuple(loc[:-1]) != tuple(now["beha_loc"][t][:-1]) for t, loc in ref["beha_loc"].items())
    wm = eir.e3.weight_migration(ref["routes"], now["routes"], pre["ddr_nodes"])
    return {"moved_behas_share": moved / len(ref["beha_loc"]), "ddr_read_moved_bytes": wm["moved_read_bytes"], "unique_block_bytes": None}


def full_rows(sinks, pre=None):
    rows = []
    for name, s in sinks:
        if s:
            rows.append({"run": name, "bk": bk(s["backend"]), "T": s["T_sel"], "own": s["own_cost"], "s": s})
    if not rows:
        return None
    b = min(rows, key=lambda r: r["bk"])
    t = min(rows, key=lambda r: (r["T"], r["own"]))
    return {"n": len(rows), "best": b["bk"], "best_run": b["run"], "median": median(r["bk"] for r in rows), "by_T": t["bk"], "by_T_run": t["run"],
            "best_mig": fmig(pre, b["s"]), "by_T_mig": fmig(pre, t["s"])}


def full_a3h(w, fid):
    if w == "W0" and fid in ("F4", "F5"):
        return full_rows([(f"g{g}_{x}", cs.load_run(f"mha_{fid.lower()}", "A3h", g, x, eir.S2_OUT)["selected"]) for g, x in cs.variants("A3h")], pre_of(w, "a3h"))
    e = ENV[w]
    d = e["w1sel"] if fid.endswith("W") else (e["main"] if fid in ("F4", "F5") else e["next"])
    return full_rows([(f"g{g}_{x}", (rd(eir.job_full(d, OP, fid, "A3h", g, x)["sink"]) or {}).get("selected")) for g, x in cs.variants("A3h")], pre_of(w, "a3h"))


def full_bb(w, fid):
    """W0: E3's FULL state (full_e3, 10 seeds). W1 (rules 2026-11-08): 14 A1 runs on the fault platform, the best 10-seed backend mean."""
    if w == "W0":
        s = rd(eir.job_full_e3(ENV[w]["main"], OP, fid)["sink"])
        return {"bk": bk(s["record"]["backend"]), "T": s["record"]["T"], "variant": s["e3"]["variant"], "mig": mig(s["record"])} if s else None
    r = full_rows([(f"g{g}_{x}", (rd(eir.job_full(ENV[w]["main"], OP, fid, "A1", g, x)["sink"]) or {}).get("selected")) for g, x in cs.variants("A1")],
                  pre_of(w, "e3"))
    return {"bk": r["best"], "variant": r["best_run"], "median14": r["median"], "by_T": r["by_T"], "by_T_run": r["by_T_run"], "n": r["n"], "mig": r["best_mig"]} if r else None


def pre_of(w, start):
    return rd(eir.job_pre(ENV[w]["main"], start, OP)["sink"])


def stop(c, m):
    """make_summary.final_rule on the levels present."""
    lim = c["T_pre"] * (1 + EPS)
    for k in ("R0", "L1"):
        if c["lv"][k]["T"] <= lim:
            return k, "进 ε"
    tiers = [s for s in TIERS if c["lv"].get(f"{m}_{s}")]
    if not tiers:
        return None, None
    prev = None
    for s in tiers:
        T = c["lv"][f"{m}_{s}"]["T"]
        if T <= lim:
            return f"{m}_{s}", "进 ε"
        if prev is not None and (prev - T) / prev < TIER_GAIN:
            return f"{m}_{s}", f"档改进 {100 * (prev - T) / prev:+.1f}%"
        prev = T
    return f"{m}_{tiers[-1]}", "预算用尽"


def combo(w, start, fid):
    e = ENV[w]
    d = e["main"]
    if start == "a3h" and (fid in en.C_FAULTS or fid == "F8a"):
        d = e["next"]
    if fid.endswith("W"):
        d = e["w1sel"]
    pre = rd(eir.job_pre(e["main"], start, OP)["sink"])
    l1 = rd(eir.job_l1(d, start, OP, fid)["sink"])
    if not (pre and l1):
        return None
    c = {"w": w, "start": start, "fault": fid, "T_pre": pre["T"], "bk_pre": bk(pre["record"]["backend"]), "lv": {}}
    for k in ("R0", "L1"):
        c["lv"][k] = {"n": 1, "T": l1[k]["T"], "bk": bk(l1[k]["backend"]), "mig": mig(l1[k])}
    # the W1 run set's seeds (w1_eir.shard_jobs), applied to W0 too: 5 SA seeds; composite faults 10 (EN.C_SEEDS)
    seeds = en.C_SEEDS[OP] if fid.rstrip("W") in en.C_FAULTS else eir.SA_SEEDS
    for m in eir.METHODS + list(en.NEW_METHODS):
        for st in TIERS:
            runs = []
            for sd in seeds:
                ps = [eir.job_rep(x, start, OP, fid, m, st, sd)["sink"] for x in {d, e["main"], e["next"]} if x]
                s = next((rd(p) for p in ps if os.path.exists(p)), None)
                if s:
                    runs.append(s)
            if runs:
                c["lv"][f"{m}_{st}"] = lvl(runs)
    c["FULL_A3h"] = full_a3h(w, fid) if start == "a3h" else None
    c["FULL_BB"] = full_bb(w, fid) if start == "e3" else None
    c["stop"] = {m: stop(c, m) for m in ("L2G", "L2L", "L2Gsync")}
    return c


def main():
    rows = []
    a3h_faults = ["F4", "F5", "F4a", "F5a", "F8a"] + list(en.C_FAULTS)
    for w in ("W0", "W1"):
        for fid in a3h_faults:
            rows.append(combo(w, "a3h", fid))
        for fid in eir.FIDS:
            rows.append(combo(w, "e3", fid))
    for fid in W1SEL.values():
        rows.append(combo("W1", "a3h", fid))
    rows = [r for r in rows if r]
    out = os.path.join(HERE, "out", "report_eir_w1.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=1, default=str)
    print(f"{out}: {len(rows)} combinations", flush=True)


if __name__ == "__main__":
    main()
