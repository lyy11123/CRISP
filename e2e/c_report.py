"""C summary (GPT, one physical fault on all 8 operators, W1): numbers only, by the rules of PROGRESS 2026-11-09 "C 第 0 步" and C_PLAN section 6.

  python e2e/c_report.py check c04 c08 ...   -> per Colab shard: jobs present, failed runs, W1 drop counts, peak memory, wall times
  python e2e/c_report.py report               -> e2e/c/out/c_report.json and e2e/c/out/c_report.md (faults with every job present)

Sinks are looked up in this machine's output (e2e/c/out) and the unpacked Colab results (colab_results/c), in that order.

Rules (fixed before results, PROGRESS "C 第 0 步"):
- Before the fault: each start's B4 W1 backend mean (starts.json b4_backend_mean, the same mapping).
- Every state: BusyBarn backend, mean over seeds 1-10.
  - R0: the R0 state of l1. L1: the L1 state of l1 (reported, not a scheme of its own).
  - L2Gsync: per operator the eps = 2 % stop rule after the fact (as EIR, make_summary.final_rule / report_eir_w1.stop):
    R0, then L1, enter eps (T-hat <= T_pre x 1.02) -> stop there; inside L2 30 -> 100 -> 300 steps, stop on entering eps or a tier
    improving the 5-seed mean T-hat by less than 2 %; else 300 ("预算用尽"). The stopped tier's 5 SA seeds: backend mean and median.
  - FULL-A3h, FULL-BusyBarn: the winner is the run with the smallest backend mean of 14; median and worst also given.
- Assembly as B4: T_block = 2 LN + 8 MHA + proj + FFN; TTFT = prefill T_block, TPOT = decode T_block, request = TTFT + n x TPOT,
  n = 128, 512, 2048. LN "asis" and "fixed" (decode one LN less: the decode FFN template holds an LN). Every operator of a layer
  takes the same scheme. Relative to the same start before the fault, and to A1 before the fault.
- Migration: EIR migration_vs_start per operator (moved behaviour share, DDR-read bytes moved); FULL's for its winner, computed as
  report_eir_w1.fmig (no unique-block term). Layer: bytes and moved behaviours summed with the T_block counts (asis), shares as
  moved / total over the layer.
- Repair steps: the stopped level per operator and its SA steps (tier steps; cumulative = 30 + 100 + ... up to the stopped tier),
  layer sum with the T_block counts.
- T-hat error: T-hat / backend mean - 1. Backend noise: (max - min) / mean over the 10 seeds.
- (a'): the combination (a3h, pf_mha, MCRandom) used the seed_broadcast fallback (a') (PROGRESS 2026-11-09); its rows are marked.
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import c_eir as c  # noqa: E402

E = c.eir
mean, median = statistics.mean, statistics.median
ROOTS = [os.path.join(HERE, "c", "out"), os.path.join(ROOT, "colab_results", "c")]
EPS, TIER_GAIN = E.EPS, 0.02
COUNT = {"ln": 2, "mha": 8, "proj": 1, "ffn": 1}
OUTS = (128, 512, 2048)
PH = {"prefill": "pf", "decode": "dc"}
A_PRIME = ("a3h", "pf_mha", "MCRandom")
STARTS = json.load(open(os.path.join(HERE, "c", "starts.json"), encoding="utf-8"))
_cache = {}


def rd(job_of):
    """job_of(root) -> job; the first root whose sink is complete."""
    for r in ROOTS:
        j = job_of(r)
        p = j["sink"]
        if p in _cache:
            return _cache[p]
        if os.path.exists(p) and E.sink_ok(p, j):
            _cache[p] = E.read_sink(p)
            return _cache[p]
    return None


def bk(b):
    return mean(v[0] for v in b.values())


def noise(b):
    v = [x[0] for x in b.values()]
    return (max(v) - min(v)) / mean(v)


def state(rec, T):
    m = bk(rec["backend"])
    mg = rec.get("migration_vs_start")
    return {"bk": m, "T": T, "err": T / m - 1 if T is not None else None, "noise": noise(rec["backend"]),
            "mig": {k: mg[k] for k in ("moved_behas", "n_behas", "ddr_read_moved_bytes", "ddr_read_total_bytes")} if mg else None}


def fmig(pre, s):
    if not (pre and s and s.get("e3map")):
        return None
    ref, now = pre["record"], s["e3map"]
    moved = sum(tuple(loc[:-1]) != tuple(now["beha_loc"][t][:-1]) for t, loc in ref["beha_loc"].items())
    wm = E.e3.weight_migration(ref["routes"], now["routes"], pre["ddr_nodes"])
    tot = sum(1 for _ in ref["beha_loc"])
    return {"moved_behas": moved, "n_behas": tot, "ddr_read_moved_bytes": wm["moved_read_bytes"], "ddr_read_total_bytes": wm["read_bytes"]}


def full(op, fid, arm, pre):
    rows = []
    for i, (g, x) in enumerate(c.cs.variants(arm)):
        s = rd(lambda r: c.jobs_full(r, op, fid, arm)[i])
        if not s:
            return None
        sel = s["selected"]
        T = sel["T_sel"] if sel.get("T_sel") is not None else sel["T_bc"]
        rows.append({"run": f"g{g}_{x}", "s": sel, "bk": bk(sel["backend"]), "T": T})
    w = min(rows, key=lambda r: r["bk"])
    st = state(w["s"], w["T"])
    st["mig"] = fmig(pre, w["s"])
    st.update(run=w["run"], median=median(r["bk"] for r in rows), worst=max(r["bk"] for r in rows), n=len(rows),
              err_median=median(r["T"] / r["bk"] - 1 for r in rows), noise_median=median(noise(r["s"]["backend"]) for r in rows))
    return st


def stop(T_pre, lv):
    lim = T_pre * (1 + EPS)
    for k in ("R0", "L1"):
        if lv[k]["T"] <= lim:
            return k, "进 ε"
    prev = None
    for s in c.STEPS:
        T = lv[f"s{s}"]["T"]
        if T <= lim:
            return f"s{s}", "进 ε"
        if prev is not None and (prev - T) / prev < TIER_GAIN:
            return f"s{s}", f"档改进 {100 * (prev - T) / prev:+.1f}%"
        prev = T
    return f"s{c.STEPS[-1]}", "预算用尽"


def op_row(op, fid):
    """Everything for one (fault, operator); None if a job is missing."""
    P = lambda r: os.path.join(r, "prep")  # noqa: E731
    o = {"op": op, "fault": fid}
    for st in ("a1", "a3h"):
        pre = rd(lambda r: E.job_pre(P(r), st, op))
        l1 = rd(lambda r: E.job_l1(P(r), st, op, fid))
        if not (pre and l1):
            return None
        b4 = STARTS[f"{st}_{op}"]["b4_backend_mean"]
        o[st] = {"pre": {"bk": b4, "bk_prep": bk(pre["record"]["backend"]), "T": pre["T"], "err": pre["T"] / b4 - 1, "noise": noise(pre["record"]["backend"])},
                 "R0": state(l1["R0"], l1["R0"]["T"]), "L1": state(l1["L1"], l1["L1"]["T"])}
        o[st]["_pre"] = pre
    lv = o["a3h"]
    for s in c.STEPS:
        runs = []
        for k, sd in enumerate(c.SEEDS):
            x = rd(lambda r: c.jobs_rep(r, op, fid)[c.STEPS.index(s) * len(c.SEEDS) + k])
            if not x:
                return None
            runs.append(x)
        sts = [state(x["record"], x["record"]["T"]) for x in runs]
        lv[f"s{s}"] = {"bk": mean(z["bk"] for z in sts), "bk_median": median(z["bk"] for z in sts), "T": mean(z["T"] for z in sts),
                       "err": median(z["err"] for z in sts), "noise": median(z["noise"] for z in sts), "n": len(sts),
                       "mig": {k: mean(z["mig"][k] for z in sts) for k in sts[0]["mig"]}}
    lvl, why = stop(lv["pre"]["T"], lv)
    steps = 0 if lvl in ("R0", "L1") else int(lvl[1:])
    cum = 0 if lvl in ("R0", "L1") else sum(s for s in c.STEPS if s <= steps)
    rep = dict(lv[lvl])
    rep.setdefault("bk_median", rep["bk"])      # stopped at R0 / L1: one deterministic state
    rep.update(level=lvl, why=why, steps=steps, steps_cum=cum)
    lv["REP"] = rep
    o["a3h"]["FULL"] = full(op, fid, "A3h", o["a3h"]["_pre"])
    o["a1"]["FULL"] = full(op, fid, "A1", o["a1"]["_pre"])
    if not (o["a3h"]["FULL"] and o["a1"]["FULL"]):
        return None
    for st in ("a1", "a3h"):
        del o[st]["_pre"]
    o["a_prime"] = (("a3h", op, fid) == A_PRIME)
    return o


# scheme -> (start, level, value key)
SCHEMES = [("A1 故障前", "a1", "pre", "bk"), ("A1 R0", "a1", "R0", "bk"), ("A1 FULL-BusyBarn", "a1", "FULL", "bk"),
           ("A3h 故障前", "a3h", "pre", "bk"), ("A3h R0", "a3h", "R0", "bk"), ("A3h L2Gsync 停止档（均值）", "a3h", "REP", "bk"),
           ("A3h L2Gsync 停止档（中位）", "a3h", "REP", "bk_median"), ("A3h L2Gsync 30", "a3h", "s30", "bk"), ("A3h L2Gsync 100", "a3h", "s100", "bk"),
           ("A3h L2Gsync 300", "a3h", "s300", "bk"), ("A3h FULL-A3h", "a3h", "FULL", "bk"), ("A3h FULL-A3h（中位）", "a3h", "FULL", "median"),
           ("A1 FULL-BusyBarn（中位）", "a1", "FULL", "median")]


def assemble(rows):
    """rows: {op: op_row} for one fault (all 8 operators)."""
    res = []
    for name, st, lvl, key in SCHEMES:
        blk = {}
        for ph, p in PH.items():
            v = {o: rows[f"{p}_{o}"][st][lvl][key] for o in COUNT}
            t = sum(COUNT[o] * v[o] for o in COUNT)
            blk[ph] = {"asis": t, "fixed": t - (v["ln"] if ph == "decode" else 0)}
        for ln in ("asis", "fixed"):
            ttft, tpot = blk["prefill"][ln], blk["decode"][ln]
            res.append({"scheme": name, "start": st, "level": lvl, "ln": ln, "TTFT": ttft, "TPOT": tpot, **{f"req{n}": ttft + n * tpot for n in OUTS}})
    keys = ("TTFT", "TPOT", *[f"req{n}" for n in OUTS])
    for r in res:
        same = next(x for x in res if x["start"] == r["start"] and x["level"] == "pre" and x["ln"] == r["ln"])
        a1 = next(x for x in res if x["start"] == "a1" and x["level"] == "pre" and x["ln"] == r["ln"])
        r["vs_pre"] = {k: r[k] / same[k] - 1 for k in keys}
        r["vs_A1pre"] = {k: r[k] / a1[k] - 1 for k in keys}
    return res


def layer_mig(rows, st, lvl):
    tot = {"moved_behas": 0, "n_behas": 0, "ddr_read_moved_bytes": 0, "ddr_read_total_bytes": 0}
    for p in PH.values():
        for o, n in COUNT.items():
            m = rows[f"{p}_{o}"][st][lvl]["mig"]
            if m is None:
                return None
            for k in tot:
                tot[k] += n * (m[k] or 0)
    tot["moved_behas_share"] = tot["moved_behas"] / tot["n_behas"] if tot["n_behas"] else None
    tot["ddr_read_moved_share"] = tot["ddr_read_moved_bytes"] / tot["ddr_read_total_bytes"] if tot["ddr_read_total_bytes"] else None
    return tot


def report():
    faults = c.faults_run()
    out, missing = [], {}
    for fid in faults:
        rows = {}
        for op in c.OPS:
            r = op_row(op, fid)
            if r is None:
                missing.setdefault(fid, []).append(op)
            else:
                rows[op] = r
        if fid in missing:
            continue
        steps = {"per_op": {op: {"level": rows[op]["a3h"]["REP"]["level"], "why": rows[op]["a3h"]["REP"]["why"], "steps": rows[op]["a3h"]["REP"]["steps"],
                                 "steps_cum": rows[op]["a3h"]["REP"]["steps_cum"]} for op in c.OPS}}
        steps["layer_steps"] = sum(COUNT[op[3:]] * steps["per_op"][op]["steps"] for op in c.OPS)
        steps["layer_steps_cum"] = sum(COUNT[op[3:]] * steps["per_op"][op]["steps_cum"] for op in c.OPS)
        mig = {name: layer_mig(rows, st, lvl) for name, st, lvl in (("A1 R0", "a1", "R0"), ("A1 FULL-BusyBarn", "a1", "FULL"), ("A3h R0", "a3h", "R0"),
                                                                    ("A3h L2Gsync 停止档", "a3h", "REP"), ("A3h FULL-A3h", "a3h", "FULL"))}
        out.append({"fault": fid, "ops": rows, "assembly": assemble(rows), "steps": steps, "migration": mig,
                    "a_prime": fid == A_PRIME[2]})
    dst = os.path.join(ROOTS[0], "c_report.json")
    json.dump({"faults": out, "missing": missing}, open(dst, "w", encoding="utf-8"), indent=1, default=float)
    print(f"{dst}: {len(out)} faults complete, missing {missing}", flush=True)
    md(out, missing)


def pct(x):
    return "—" if x is None else f"{100 * x:+.1f}%"


def md(faults, missing):
    L = ["# C 汇总（自动生成，`e2e\\c_report.py report`）", "",
         f"完整的故障 {len(faults)} 个；不完整：{', '.join(f'{k}（{len(v)} 个算子缺）' for k, v in missing.items()) or '无'}。", ""]
    L += ["## 1. 请求延迟（LN 原样）", "",
          "单位：周期。相对 = 相对同一起点的故障前 / 相对 A1 故障前。标 † 的故障含 (a′) 组合（A3h、prefill MHA、MCRandom）。", ""]
    for ln, title in (("asis", "原样"), ("fixed", "修正")):
        if ln == "fixed":
            L += ["## 2. 请求延迟（LN 修正）", ""]
        L += ["| 故障 | 方案 | TTFT | TPOT | 请求 128 | 请求 512 | 请求 2048 | 相对故障前：TTFT / TPOT / 2048 | 相对 A1 故障前：TTFT / TPOT / 2048 |", "|---|---|---|---|---|---|---|---|---|"]
        for f in faults:
            for r in f["assembly"]:
                if r["ln"] != ln:
                    continue
                v, a = r["vs_pre"], r["vs_A1pre"]
                L.append(f"| {f['fault']}{'†' if f['a_prime'] else ''} | {r['scheme']} | {r['TTFT']:,.0f} | {r['TPOT']:,.0f} | {r['req128']:,.0f} | {r['req512']:,.0f} | {r['req2048']:,.0f} | "
                         f"{pct(v['TTFT'])} / {pct(v['TPOT'])} / {pct(v['req2048'])} | {pct(a['TTFT'])} / {pct(a['TPOT'])} / {pct(a['req2048'])} |")
        L.append("")
    L += ["## 3. 各算子：后端均值相对同一起点故障前", "", "A3h 修复为停止档 5 个种子的均值；FULL 为 14 次中的胜者。", "",
          "| 故障 | 算子 | A1 R0 | A1 FULL-BB | A3h R0 | A3h L1 | A3h 停止档 | 停在 | 原因 | A3h FULL-A3h |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f in faults:
        for op, o in f["ops"].items():
            a1, a3 = o["a1"], o["a3h"]
            rel = lambda s, k: s[k]["bk"] / s["pre"]["bk"] - 1  # noqa: E731
            L.append(f"| {f['fault']} | {op}{'†' if o['a_prime'] else ''} | {pct(rel(a1, 'R0'))} | {pct(rel(a1, 'FULL'))} | {pct(rel(a3, 'R0'))} | {pct(rel(a3, 'L1'))} | "
                     f"{pct(rel(a3, 'REP'))} | {a3['REP']['level']} | {a3['REP']['why']} | {pct(rel(a3, 'FULL'))} |")
    L += ["", "## 4. 迁移量（一层，按 T_block 计数求和）与修复步数", "",
          "| 故障 | 方案 | 搬动行为 | 占比 | DDR 读搬动字节 | 占比 |", "|---|---|---|---|---|---|"]
    for f in faults:
        for name, m in f["migration"].items():
            if m is None:
                L.append(f"| {f['fault']} | {name} | — | — | — | — |")
                continue
            L.append(f"| {f['fault']} | {name} | {m['moved_behas']:,.1f} | {pct(m['moved_behas_share']).lstrip('+')} | {m['ddr_read_moved_bytes']:,.0f} | "
                     f"{pct(m['ddr_read_moved_share']).lstrip('+') if m['ddr_read_moved_share'] is not None else '—'} |")
    L += ["", "| 故障 | " + " | ".join(c.OPS) + " | 一层 SA 步数（档 / 累计） |", "|---|" + "---|" * len(c.OPS) + "---|"]
    for f in faults:
        s = f["steps"]
        L.append(f"| {f['fault']} | " + " | ".join(s["per_op"][op]["level"] for op in c.OPS) + f" | {s['layer_steps']:,} / {s['layer_steps_cum']:,} |")
    L += ["", "## 5. T̂ 误差与后端噪声（各算子，跨完整故障取中位 [最小, 最大]）", "",
          "误差 = T̂ / 后端 10 种子均值 − 1；噪声 = (最大 − 最小) / 均值，10 个种子。A3h 停止档与 FULL 取各自的中位。", "",
          "| 算子 | 起点 | 状态 | 误差 | 噪声 |", "|---|---|---|---|---|"]
    for op in c.OPS:
        for st, lvls in (("a1", ("pre", "R0", "FULL")), ("a3h", ("pre", "R0", "REP", "FULL"))):
            for lvl in lvls:
                e = [f["ops"][op][st][lvl]["err" if lvl != "FULL" else "err_median"] for f in faults]
                n = [f["ops"][op][st][lvl]["noise" if lvl != "FULL" else "noise_median"] for f in faults]
                if lvl == "pre":
                    e, n = e[:1], n[:1]
                fmt = lambda xs: f"{pct(median(xs))} [{pct(min(xs))}, {pct(max(xs))}]" if xs else "—"  # noqa: E731
                L.append(f"| {op} | {st.upper()} | {lvl} | {fmt(e)} | {fmt(n)} |")
    L += ["", "故障前一行的误差以 B4 后端均值为分母；本次 prep 重评估的后端均值与 B4 的差见 `c_report.json` 的 `bk_prep`。", ""]
    open(os.path.join(ROOTS[0], "c_report.md"), "w", encoding="utf-8").write("\n".join(L))
    print(os.path.join(ROOTS[0], "c_report.md"), flush=True)


# ---------------------------------------------------------------- Colab shard check
def check(shards):
    import json as _j
    res = {}
    for sh in shards:
        out = ROOTS[1]
        jobs = [j for p in c.shard_jobs(sh, out) for j in p]
        miss = [j["name"] for j in jobs if not (os.path.exists(j["sink"]) and E.sink_ok(j["sink"], j))]
        rp = os.path.join(out, f"runs_{sh}.jsonl")
        runs = [_j.loads(x) for x in open(rp, encoding="utf-8")] if os.path.exists(rp) else []
        bad = [r for r in runs if r["status"] not in ("ok", "cached")]
        last = {}
        for r in runs:
            last[r["name"]] = r
        bad_last = [n for n, r in last.items() if r["status"] not in ("ok", "cached")]
        mp = os.path.join(out, "mem", f"{sh}.jsonl")
        mem = [_j.loads(x) for x in open(mp, encoding="utf-8")] if os.path.exists(mp) else []
        byname = {j["name"]: j for j in jobs}
        drop_bad = []
        for m in mem:
            j = byname.get(m["name"])
            if not j:
                continue
            want = c.OPS[j["op"]][4]
            got = {int(k.split(":")[1]) for k in m["drops"]}
            if (want and got != {want}) or (not want and got):
                drop_bad.append((m["name"], m["drops"]))
        wall = {}
        for n, r in last.items():
            if r["status"] == "ok" and n in byname:
                wall.setdefault(byname[n]["kind"] + ":" + byname[n]["op"], []).append(r["wall_s"] / c.est(byname[n]))
        res[sh] = {"jobs": len(jobs), "missing": len(miss), "missing_names": miss[:5], "run_records": len(runs), "failed_records": len(bad),
                   "failed_last": bad_last, "mem_records": len(mem), "mem_jobs_unmatched": sum(1 for m in mem if m["name"] not in byname),
                   "peak_gb_max": max((m["peak_gb"] for m in mem), default=None), "drop_mismatch": drop_bad[:5], "n_drop_mismatch": len(drop_bad),
                   "mha_jobs_with_mem": sum(1 for m in mem if m["name"] in byname and c.OPS[byname[m["name"]]["op"]][4]),
                   "wall_over_local_est_median": {k: round(median(v), 2) for k, v in sorted(wall.items())},
                   "wall_max_s": max((r["wall_s"] for r in last.values() if r["status"] == "ok"), default=None),
                   "process_hours": sum(r["wall_s"] for r in last.values() if r["status"] == "ok") / 3600,
                   "alerts": [n for n, r in last.items() if r["status"] == "ok" and n in byname and r["wall_s"] > c.ALERT_S[byname[n]["op"]]]}
        print(sh, _j.dumps(res[sh], ensure_ascii=False), flush=True)
    dst = os.path.join(ROOTS[0], "c_colab_check.json")
    old = _j.load(open(dst, encoding="utf-8")) if os.path.exists(dst) else {}
    old.update(res)
    _j.dump(old, open(dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check(sys.argv[2:])
    elif cmd == "report":
        report()
