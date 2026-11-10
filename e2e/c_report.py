"""C summary (GPT, one physical fault on all 8 operators, W1): numbers only, by the rules of PROGRESS 2026-11-09 "C 第 0 步" and C_PLAN section 6.

  python e2e/c_report.py check c04 c08 ...   -> per Colab shard: jobs present, failed runs, W1 drop counts, peak memory, wall times
  python e2e/c_report.py xcheck [workers]      -> rerun a sample of Colab jobs here, compare fingerprints (c/out/c_xcheck.json)
  python e2e/c_report.py report               -> e2e/c/out/c_report.json and e2e/C_SUMMARY.md (faults with every job present)

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
import collections
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
def rd(job_of):
    """job_of(root) -> job; the first root whose sink is complete. Each sink is read once and not cached (2026-11-10: the earlier
    version read every sink twice, via eir.sink_ok, and kept all of them; the process reached about 24 GB with 0.0-0.9 GB free, and
    one run reported 2 groups of complete MCScratch sinks as missing, since sink_ok turns any read error into False).
    A sink that exists but cannot be read now stops the report instead of counting as missing. Sinks are written to .part / .tmp
    and renamed by eir.run_job, so a sink path never holds a half-written file from a finished runner."""
    for r in ROOTS:
        j = job_of(r)
        p = j["sink"]
        if not os.path.exists(p):
            continue
        try:
            d = E.read_sink(p)
        except Exception as e:
            raise RuntimeError(f"{p}: exists but cannot be read ({type(e).__name__}: {e})") from e
        if ("selected" in d) if j["kind"] == "full" else (d.get("done") is True):
            return d
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


MAIN = ["A1 故障前", "A1 R0", "A1 FULL-BusyBarn", "A3h 故障前", "A3h R0", "A3h L2Gsync 停止档（均值）", "A3h FULL-A3h"]
SUMMARY_MD = os.path.join(HERE, "C_SUMMARY.md")
FSET = {f: v.get("set") for f, v in json.load(open(c.FAULTS, encoding="utf-8"))["dc_mha"].items()}


def fmt3(xs):
    return f"{pct(median(xs))} [{pct(min(xs))}, {pct(max(xs))}]" if xs else "—"


def check_section():
    L = []
    p = os.path.join(ROOTS[0], "c_colab_check.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        L += ["| 分片 | 作业 | 缺 | 运行记录 | 失败记录 | 内存记录 | MHA 作业的内存记录 | 删去条数不符 | 单进程峰值 GB | 单次最长 s | 进程小时 | 超时 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in sorted(d):
            r = d[k]
            L.append(f"| {k} | {r['jobs']} | {r['missing']} | {r['run_records']} | {r['failed_records']} | {r['mem_records']} | {r['mha_jobs_with_mem']} | "
                     f"{r['n_drop_mismatch']} | {r['peak_gb_max']:.2f} | {r['wall_max_s']:,.0f} | {r['process_hours']:.1f} | {len(r['alerts'])} |")
        ks = sorted({x for r in d.values() for x in r["wall_over_local_est_median"]})
        L += ["", "Colab 单次墙钟 / 本机估时（`c_eir.est`）：先在每个分片内取中位，再列各分片的最小、中位、最大。", "",
              "| 种类：算子 | 最小 | 中位 | 最大 |", "|---|---|---|---|"]
        for x in ks:
            v = [r["wall_over_local_est_median"][x] for r in d.values() if x in r["wall_over_local_est_median"]]
            L.append(f"| {x} | {min(v):.2f} | {median(v):.2f} | {max(v):.2f} |")
        L.append("")
    p = os.path.join(ROOTS[0], "c_xcheck.json")
    if os.path.exists(p):
        r = json.load(open(p, encoding="utf-8"))
        b = [x["backend_rel"] for x in r]
        bad = [(x["job"], x["differs"]) for x in r if not x["identical"]]
        L += [f"本机重跑抽样（`c_report.py xcheck`：每个 Colab 分片取 1 个 decode 修复、1 个 decode FULL）：{sum(x['identical'] for x in r)}/{len(r)} 个"
              f"与地址无关的部分（T̂、路由、行为位置、搜索轨迹）逐位相同；后端均值相差 {pct(min(b))} 到 {pct(max(b))}。" + (f" 不同的：{bad}。" if bad else ""), ""]
    return L


def set_sections(L, faults, tag, title):
    L += [f"## {tag}. {title}", ""]
    for ln, lname in (("asis", "原样"), ("fixed", "修正")):
        L += [f"### {tag}.{1 if ln == 'asis' else 2} 请求延迟（LN {lname}）", "",
              "| 故障 | 方案 | TTFT | TPOT | 请求 128 | 请求 512 | 请求 2048 | 相对故障前 TTFT / TPOT / 128 / 512 / 2048 | 相对 A1 故障前 TTFT / TPOT / 2048 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for f in faults:
            for r in f["assembly"]:
                if r["ln"] != ln or r["scheme"] not in MAIN:
                    continue
                v, a = r["vs_pre"], r["vs_A1pre"]
                L.append(f"| {f['fault']}{'†' if f['a_prime'] else ''} | {r['scheme']} | {r['TTFT']:,.0f} | {r['TPOT']:,.0f} | {r['req128']:,.0f} | {r['req512']:,.0f} | "
                         f"{r['req2048']:,.0f} | {pct(v['TTFT'])} / {pct(v['TPOT'])} / {pct(v['req128'])} / {pct(v['req512'])} / {pct(v['req2048'])} | "
                         f"{pct(a['TTFT'])} / {pct(a['TPOT'])} / {pct(a['req2048'])} |")
        L.append("")
    L += [f"### {tag}.3 跨故障（本组 {len(faults)} 个）：中位 [最小, 最大]", "",
          "| 方案 | LN | 相对故障前 TTFT | TPOT | 请求 2048 | 相对 A1 故障前 TTFT | TPOT | 请求 2048 |", "|---|---|---|---|---|---|---|---|"]
    for ln, lname in (("asis", "原样"), ("fixed", "修正")):
        for name, *_ in SCHEMES:
            rs = [r for f in faults for r in f["assembly"] if r["ln"] == ln and r["scheme"] == name]
            if not rs or name.endswith("故障前"):
                continue
            L.append(f"| {name} | {lname} | " + " | ".join(fmt3([r[k][q] for r in rs]) for k in ("vs_pre", "vs_A1pre") for q in ("TTFT", "TPOT", "req2048")) + " |")
    extra = [n for n, *_ in SCHEMES if n not in MAIN]
    L += ["", f"### {tag}.4 补充：A3h 各档与中位、FULL 中位（请求 2048 相对故障前，LN 原样 / 修正）", "",
          "| 故障 | " + " | ".join(extra) + " |", "|---|" + "---|" * len(extra)]
    for f in faults:
        cells = []
        for n in extra:
            a = next(r for r in f["assembly"] if r["scheme"] == n and r["ln"] == "asis")["vs_pre"]["req2048"]
            b = next(r for r in f["assembly"] if r["scheme"] == n and r["ln"] == "fixed")["vs_pre"]["req2048"]
            cells.append(f"{pct(a)} / {pct(b)}")
        L.append(f"| {f['fault']}{'†' if f['a_prime'] else ''} | " + " | ".join(cells) + " |")
    L += ["", f"### {tag}.5 各算子：后端均值相对同一起点故障前", "",
          "A3h 停止档为该档 5 个种子的均值；FULL 为 14 次中的胜者，括号内为 14 次的中位。", "",
          "| 故障 | 算子 | A1 R0 | A1 FULL-BB | A3h R0 | A3h L1 | A3h 停止档 | 停在 | 原因 | A3h FULL-A3h |", "|---|---|---|---|---|---|---|---|---|---|"]
    rel = lambda s, k, q="bk": s[k][q] / s["pre"]["bk"] - 1  # noqa: E731
    for f in faults:
        for op, o in f["ops"].items():
            a1, a3 = o["a1"], o["a3h"]
            L.append(f"| {f['fault']} | {op}{'†' if o['a_prime'] else ''} | {pct(rel(a1, 'R0'))} | {pct(rel(a1, 'FULL'))} ({pct(rel(a1, 'FULL', 'median'))}) | "
                     f"{pct(rel(a3, 'R0'))} | {pct(rel(a3, 'L1'))} | {pct(rel(a3, 'REP'))} | {a3['REP']['level']} | {a3['REP']['why']} | "
                     f"{pct(rel(a3, 'FULL'))} ({pct(rel(a3, 'FULL', 'median'))}) |")
    L += ["", f"### {tag}.6 迁移量（一层，按 T_block 计数 2/8/1/1 求和）", "",
          "| 故障 | 方案 | 搬动行为 | 占比 | DDR 读搬动字节 | 占比 |", "|---|---|---|---|---|---|"]
    for f in faults:
        for name, m in f["migration"].items():
            if m is None:
                L.append(f"| {f['fault']} | {name} | — | — | — | — |")
                continue
            sh = m["ddr_read_moved_share"]
            shs = "—" if sh is None else f"{100 * sh:.1f}%"
            L.append(f"| {f['fault']} | {name} | {m['moved_behas']:,.1f} | {100 * m['moved_behas_share']:.1f}% | {m['ddr_read_moved_bytes']:,.0f} | {shs} |")
    L += ["", f"### {tag}.7 停止级别与 SA 步数（A3h L2Gsync，ε = 2%）", "",
          "| 故障 | " + " | ".join(c.OPS) + " | 一层 SA 步数（停止档 / 累计） |", "|---|" + "---|" * len(c.OPS) + "---|"]
    for f in faults:
        s = f["steps"]
        L.append(f"| {f['fault']} | " + " | ".join(s["per_op"][op]["level"] for op in c.OPS) + f" | {s['layer_steps']:,} / {s['layer_steps_cum']:,} |")
    lv = collections.Counter(f["steps"]["per_op"][op]["level"] for f in faults for op in c.OPS)
    why = collections.Counter(f["steps"]["per_op"][op]["why"].split(" ")[0] for f in faults for op in c.OPS)
    L += ["", f"停止级别计数（{len(faults)} 个故障 × 8 个算子）：" + "，".join(f"{k} {v}" for k, v in sorted(lv.items())) +
          "；原因：" + "，".join(f"{k} {v}" for k, v in why.items()) + "。", ""]
    L += [f"### {tag}.8 T̂ 误差与后端噪声（各算子，本组故障上取中位 [最小, 最大]）", "",
          "误差 = T̂ / 后端 10 种子均值 − 1；噪声 = (最大 − 最小) / 均值，10 个种子。A3h 停止档取 5 个种子的中位，FULL 取 14 次的中位。"
          "故障前一行以 B4 后端均值为分母，与故障无关。", "",
          "| 算子 | 起点 | 状态 | 误差 | 噪声 |", "|---|---|---|---|---|"]
    for op in c.OPS:
        for st, lvls in (("a1", ("pre", "R0", "FULL")), ("a3h", ("pre", "R0", "REP", "FULL"))):
            for lvl in lvls:
                e = [f["ops"][op][st][lvl]["err" if lvl != "FULL" else "err_median"] for f in faults]
                n = [f["ops"][op][st][lvl]["noise" if lvl != "FULL" else "noise_median"] for f in faults]
                if lvl == "pre":
                    e, n = e[:1], n[:1]
                L.append(f"| {op} | {st.upper()} | {lvl} | {fmt3(e)} | {fmt3(n)} |")
    L.append("")


def md(faults, missing):
    L = ["# C 汇总：GPT 同一物理故障作用于 8 个算子（W1）", "",
         "由 `e2e\\c_report.py report` 生成，只报数字。口径见 `e2e\\PROGRESS.md`“C 第 0 步”（事先写下，看结果后不改）与 `c_report.py` 开头的说明。", "",
         "- **平台：** GPT，序列 2048，2×2 组，W1（MHA 最终广播删去 QKᵀ、S 的消息）。单位：周期。",
         "- **方案：**",
         "  - A1（BusyBarn 起点）：R0、FULL-BusyBarn。",
         "  - A3h 起点：R0、L2Gsync（ε = 2% 停止规则，逐个算子判档）、FULL-A3h。",
         "  - 每个算子各取 B4 W1 的后端最优作起点。",
         "- **组装：** T_block = 2·LN + 8·MHA + 投影 + FFN；TTFT = prefill T_block，TPOT = decode T_block，请求 = TTFT + 输出长度 × TPOT。",
         "  - LN 原样：decode 计 3 个 LN；修正：计 2 个。",
         "- **† (a′)：** A3h、prefill MHA、MCRandom 这一组合用了 seed_broadcast 修法 (a′)（PROGRESS 2026-11-09）。",
         "  - 配对桶为空的分割改由 Dijkstra 路由，不链式。",
         "  - 它的 l1 与 15 个修复只在本机运行；fallback 全程只触发 1 次（l1，生产者 20）。",
         "  - 含它的故障是 MCRandom，表中标 †。",
         f"- **完整的故障：** {len(faults)} 个；不完整：{', '.join(f'{k}（{len(v)} 个算子缺）' for k, v in missing.items()) or '无'}。",
         "  - (b) 组中 MF2、MF4、MF7 与 (a) 组的 PF2、PF4、PF7 坐标相同，未重跑，数字见 (a) 组。", ""]
    L += ["## 1. 运行与核对", ""]
    L += check_section()
    sa = [f for f in faults if FSET.get(f["fault"]) == "a"]
    sb = [f for f in faults if FSET.get(f["fault"]) == "b"]
    set_sections(L, sa, "2", "(a) 组：按物理位置固定的坐标")
    set_sections(L, sb, "3", "(b) 组：在 decode MHA 的 W1 A3h 起点上按 EIR 原规则选的坐标")
    diag = os.path.join(HERE, "c", "C_DIAG.md")          # hand-written read-only diagnosis (2026-11-10), kept apart so a rerun keeps it
    if os.path.exists(diag):
        L += [open(diag, encoding="utf-8").read()]
    open(SUMMARY_MD, "w", encoding="utf-8").write("\n".join(L))
    print(SUMMARY_MD, flush=True)


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


XOUT = os.path.join(ROOTS[0], "_xcheck")


def xcheck(workers):
    """Rerun on this machine one decode repair and one decode FULL per Colab shard (first such job of the frozen list) into
    e2e/c/out/_xcheck (prep copied in), then compare c_eir.fingerprint with the Colab sink: address-independent parts exact,
    backend by mean only (C_COLAB_GUIDE "比较口径")."""
    import shutil
    if not os.path.isdir(os.path.join(XOUT, "prep")):
        shutil.copytree(os.path.join(ROOTS[0], "prep"), os.path.join(XOUT, "prep"))
    pick = []
    for sh in c.colab_shards():
        jobs = [j for p in c.shard_jobs(sh, XOUT) for j in p]
        for kind in ("rep", "full"):
            k = [j for j in jobs if j["kind"] == kind and j["op"].startswith("dc_")]
            if k:
                pick.append((sh, k[len(k) // 2]))
    real = c.shard_jobs
    c.shard_jobs = lambda shard, out: [[j for _, j in pick]] if shard == "xcheck" else real(shard, out)
    c.redirect()          # as c_eir.main: C inputs, faults, work copy; children run `c_eir.py child`
    try:
        c.run("xcheck", XOUT, workers)
    finally:
        c.shard_jobs = real
    rows = []
    for sh, j in pick:
        col = j["sink"].replace(XOUT, ROOTS[1])
        a, b = c.fingerprint(E.read_sink(j["sink"]), j["kind"]), c.fingerprint(E.read_sink(col), j["kind"])
        same = json.loads(json.dumps(a["exact"])) == json.loads(json.dumps(b["exact"]))
        rows.append({"shard": sh, "job": j["name"], "identical": same, "backend_rel": a["backend_mean"] / b["backend_mean"] - 1,
                     "differs": [] if same else [k for k in a["exact"] if json.loads(json.dumps(a["exact"][k])) != json.loads(json.dumps(b["exact"].get(k)))]})
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(os.path.join(ROOTS[0], "c_xcheck.json"), "w"), indent=1)
    print(f"XCHECK {sum(r['identical'] for r in rows)}/{len(rows)} identical", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check(sys.argv[2:])
    elif cmd == "report":
        report()
    elif cmd == "xcheck":
        xcheck(int(sys.argv[2]) if len(sys.argv) > 2 else 14)
