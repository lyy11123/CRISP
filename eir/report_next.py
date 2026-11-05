"""Phase A report (EIR_NEXT_HANDOFF.md 4.5): numbers only. Reads out/next and, for what phase A reuses, out/main.

  python report_next.py prior      section 4.2's comparison
  python report_next.py a          out/next/report_a.md and report_a.json
  python report_next.py diag       out/next/report_diag.md (targeted branches, FFN F4a, F8a)

The 20-seed statistics and the gate below were written and committed before the 255 runs (see PROGRESS).
"""
import collections
import json
import os
import statistics
import sys

import numpy as np
from scipy import stats as sps

import eir
import eir_next as en

mean, median = statistics.mean, statistics.median
METHODS = [("L2L", "原 L2-Local"), ("L2Lsync", "L2L+sync"), ("L2Impact", "L2Impact"), ("L2G", "原 L2-Global"), ("L2Gsync", "L2G+sync")]
PAIRS = [("L2Impact", "L2Lsync"), ("L2Lsync", "L2L"), ("L2Gsync", "L2G"), ("L2Impact", "L2Gsync")]
PAIRS20 = PAIRS + [("L2L", "L2G"), ("L2Lsync", "L2Gsync")]
SEEDS = en.ALL_SEEDS
SEEDS5 = eir.SA_SEEDS
BOOT_N, BOOT_SEED, CI = 10000, 20261105, (2.5, 97.5)
GATE_PAIR, GATE_REL, GATE_MIN = ("L2Impact", "L2Lsync"), -2.0, 2     # at least 2 of 3 FFN combinations: median <= -2% of fault-free, CI excludes 0
COMBOS = [(o, f) for o in en.OPS for f in en.FAULTS]


def pc(a, b):
    return 100 * (a / b - 1)


def f0(x):
    return "—" if x is None else f"{x:,.0f}"


def bk(b):
    return mean(v[0] for v in b.values())


def boot_stats(ds, pre):
    """Paired differences: median, median / fault-free, wins / losses / ties of the first method, 95% percentile bootstrap of the
    median (BOOT_N resamples of the seeds, BOOT_SEED), Wilcoxon signed-rank p (two-sided; zero differences dropped, scipy's default)."""
    v = np.array(ds, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    meds = np.median(v[rng.integers(0, len(v), size=(BOOT_N, len(v)))], axis=1)
    lo, hi = np.percentile(meds, CI)
    try:
        p = float(sps.wilcoxon(v).pvalue) if np.any(v != 0) else None
    except ValueError:
        p = None
    med = float(np.median(v))
    return {"n": len(ds), "median": med, "rel": 100 * med / pre, "win": int((v < 0).sum()), "loss": int((v > 0).sum()), "tie": int((v == 0).sum()),
            "lo": float(lo), "hi": float(hi), "p": p, "excl0": bool(lo > 0 or hi < 0)}


def s2_full(cfg):
    """Stage 2's FULL-A3h runs of a config (fault-free cfg = 'ffn', 'mha')."""
    out = []
    for g, x in eir.cs.variants("A3h"):
        s = eir.cs.load_run(cfg, "A3h", g, x, eir.S2_OUT)["selected"]
        out.append({"run": f"g{g}_{x}", "mean": bk(s["backend"]), "T": s["T_sel"], "own": s["own_cost"]})
    return out


def next_full(op, fid):
    out = []
    for g, x in eir.cs.variants("A3h"):
        j = eir.job_full(en.OUT, op, fid, "A3h", g, x)
        if os.path.exists(j["sink"]):
            s = eir.read_sink(j["sink"])["selected"]
            out.append({"run": f"g{g}_{x}", "mean": bk(s["backend"]), "T": s["T_sel"], "own": s["own_cost"]})
    return out


def full_three(rows, pre):
    """14 runs: best by backend, median, and the run chosen by its own T (T_sel, then own cost, cs.report15's key): what a deployment can pick."""
    b = min(rows, key=lambda r: r["mean"])
    t = min(rows, key=lambda r: (r["T"], r["own"]))
    m = median(r["mean"] for r in rows)
    return {"n": len(rows), "best": b["mean"], "best_run": b["run"], "median": m, "by_T": t["mean"], "by_T_run": t["run"], "by_T_T": t["T"], "pre": pre,
            "best_rel": pc(b["mean"], pre), "median_rel": pc(m, pre), "by_T_rel": pc(t["mean"], pre)}


def full_block():
    """FULL's three rows on every A3h-start combination with data, and the fault-free 14 runs the start was picked from."""
    L = ["\n## FULL-A3h 的三种读法（A3h 起点）\n",
         "每个组合 14 次（g0、g1 × 7 个种子，1,000 步）。**按后端最优**用了部署前拿不到的后端结果，**按 T̂ 自选**（T_sel 最小，平局看自身代价）才是在线部署时拿得到的选法。相对故障前都对同一个起点的后端。\n",
         "| 任务 | 故障 | 故障前后端 | 按后端最优（变体） | 14 次中位数 | 按 T̂ 自选的后端（变体） | 来源 |\n|---|---|---|---|---|---|---|"]
    out = {}
    for op in en.OPS:
        pre = bk(eir.read_sink(eir.job_pre(en.MAIN, "a3h", op)["sink"])["record"]["backend"])
        for fid, rows, src in [("F4", s2_full(f"{op}_f4"), "第 2 阶段"), ("F5", s2_full(f"{op}_f5"), "第 2 阶段"), ("F4a", next_full(op, "F4a"), "`out/next`"), ("F8a", next_full(op, "F8a"), "`out/next`")]:
            if not rows:
                continue
            x = full_three(rows, pre)
            out[f"{op}_{fid}"] = x
            L.append(f"| {op} | {fid} | {f0(pre)} | {f0(x['best'])}（{x['best_rel']:+.1f}%，{x['best_run']}） | {f0(x['median'])}（{x['median_rel']:+.1f}%） | {f0(x['by_T'])}（{x['by_T_rel']:+.1f}%，{x['by_T_run']}） | {src} |")
    L.append("\n起点映射的来历（对比口径）：起点本身也是无故障 14 次中**按后端**挑的最优。\n")
    L.append("| 任务 | 无故障 14 次按后端最优（起点） | 14 次中位数 | 按 T̂ 自选的后端（变体） |\n|---|---|---|---|")
    for op in en.OPS:
        rows = s2_full(op)
        cfg, g, x_ = eir.A3H_RUNS[op]
        name = f"g{g}_{x_}"
        start = next(r for r in rows if r["run"] == name)
        x = full_three(rows, start["mean"])
        out[f"{op}_faultfree"] = {**x, "start_run": name, "start_mean": start["mean"]}
        L.append(f"| {op} | {f0(start['mean'])}（{name}；14 次最优 {x['best_run']} {f0(x['best'])}） | {f0(x['median'])}（{x['median_rel']:+.1f}%） | {f0(x['by_T'])}（{x['by_T_rel']:+.1f}%，{x['by_T_run']}） |")
    L.append("\n第 2 阶段无故障的后端数来自第 2 阶段的进程，`out/main` 的 pre 是同一映射在另一进程的后端，两者可以有进程间差。\n")
    return L, out


def sink_of(op, fid, m, sd):
    if m in en.OLD_METHODS and fid != "F8a" and sd in SEEDS5:
        return eir.job_rep(en.MAIN, "a3h", op, fid, m, en.STEPS, sd)["sink"]
    return en.job_rep(op, fid, m, sd)["sink"]


def load(op, fid, m):
    return {sd: eir.read_sink(sink_of(op, fid, m, sd)) for sd in SEEDS if os.path.exists(sink_of(op, fid, m, sd))}


def landing(runs, accepted_only=True):
    c = collections.Counter()
    for d in runs:
        for r in d["search"]["rows"]:
            if (r["accepted"] or not accepted_only):
                c.update(map(str, r.get("landing", [])))
    return c


def prior():
    L = ["# 第 4.2 节先验：FFN、MHA F4a，s = 100，种子 123\n",
         "| 任务 | 方法 | K | 行为 | 接受 / 100 | 接受的行为移动 | 撤销 | 无提议 | 不同落点核 | 前 2 核份额 | 前 3 核份额 | 落点（接受，前 6） | T̂ 起点 → 最优 | 后端（10 种子） |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for op in en.OPS:
        for m in ("L2Lsync", "L2Impact"):
            d = eir.read_sink(en.job_rep(op, "F4a", m, 123)["sink"])
            s = d["search"]
            rows = s["rows"]
            acc = sum(r["accepted"] for r in rows)
            acc_b = sum(r["accepted"] and "landing" in r for r in rows)
            ill = sum("illegal" in r for r in rows)
            none = sum(r["action"] is None for r in rows)
            lc = landing([d])
            tot = sum(lc.values()) or 1
            top = lc.most_common()
            s2, s3 = sum(v for _, v in top[:2]) / tot, sum(v for _, v in top[:3]) / tot
            verdict[(op, m)] = {"accepted": acc, "accepted_beha": acc_b, "cores": len(lc), "top2": s2, "top3": s3, "illegal": ill, "no_proposal": none}
            L.append(f"| {op} | {m} | {s['local']['n_cores']} | {s['local']['n_behas']} | {acc} | {acc_b} | {ill} | {none} | {len(lc)} | {100 * s2:.0f}% | {100 * s3:.0f}% | {' '.join(f'{k}:{v}' for k, v in top[:6])} | {f0(s['T_start'])} → {f0(s['best_T'])} | {f0(bk(d['record']['backend']))} |")
        im = eir.read_sink(en.job_rep(op, "F4a", "L2Impact", 123)["sink"])["search"]["impact"]
        L.append(f"\n{op} 集合：{im['nodes']}；候选 {im['n_cand']}，第 1 组 {im['n_g1']}（选中 {im['chosen_g1']}），第 2 组 {im['n_g2']}（选中 {im['chosen_g2']}），与几何重合 {im['overlap_geo']}；tag 未对齐 {im['n_unaligned_W']} / {im['n_events']}\n")
        L.append("| 任务 | 方法 | K | 行为 | 接受 / 100 | 接受的行为移动 | 撤销 | 无提议 | 不同落点核 | 前 2 核份额 | 前 3 核份额 | 落点（接受，前 6） | T̂ 起点 → 最优 | 后端（10 种子） |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    L = L[:-1]
    L.append("\n判据（开跑前定）：L2Impact 的接受数不到 L2L+sync 的一半；或接受的行为落点里前 2 核占 ≥ 80%，或不同落点核 ≤ 2。\n")
    trig = {}
    for op in en.OPS:
        a, b = verdict[(op, "L2Impact")], verdict[(op, "L2Lsync")]
        trig[op] = {"accept_under_half": a["accepted"] < b["accepted"] / 2, "concentrated": a["top2"] >= 0.8 or a["cores"] <= 2}
    L.append("触发情况：" + json.dumps(trig))
    p = os.path.join(en.OUT, "report_prior.md")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def full_table(op, fid):
    rows = []
    bad = 0
    f = eir.fault_of(op, fid)
    dead_n = {tuple(x) for x in f["failed_nodes"]}
    dead_l = {(tuple(a), tuple(b)) for a, b in f["failed_links"]}
    for g, x in eir.cs.variants("A3h"):
        j = eir.job_full(en.OUT, op, fid, "A3h", g, x)
        if not os.path.exists(j["sink"]):
            continue
        s = eir.read_sink(j["sink"])["selected"]
        v = [t[0] for t in s["backend"].values()]
        for routes in s["e3map"]["routes"].values():
            for src, tree, _ in routes:
                if tuple(src) in dead_n:
                    bad += 1
                for r, leaves in tree:
                    for l in leaves:
                        if (tuple(r), tuple(l)) in dead_l or tuple(r) in dead_n or tuple(l) in dead_n:
                            bad += 1
        rows.append({"run": f"g{g}_{x}", "mean": mean(v), "min": min(v), "max": max(v), "T": s["T_bc"], "sa_wall": None})
    if not rows:
        return None
    b = min(rows, key=lambda r: r["mean"])
    return {**b, "n": len(rows), "route_or_source_on_failed": bad, "all_means": sorted(r["mean"] for r in rows)}


def report_a():
    res = []
    L = ["# EIR 下一阶段 阶段 A：L2-Impact 与消融\n",
         "只报数字。A3h 起点，s = 100，后端 10 个种子，广播种子 0。SA 种子：FFN 三个组合的 5 个方法和 MHA F4a 的 L2Impact、L2L+sync 是 20 个（123、1、2、3、4、5–19），其余 5 个，见“种子数”列。原 L2-Local、原 L2-Global 的原 5 个种子在 F4a、F5 上复用 `out\\main`，其余新跑。净代价的对照是 `out\\main` 的无故障继续搜索（s = 100，**旧规则**）。决策耗时含并行争用，不作论文数据。\n"]
    rows_all = []
    for op, fid in COMBOS:
        pre = eir.read_sink(eir.job_pre(en.MAIN, "a3h", op)["sink"])
        pre_b = bk(pre["record"]["backend"])
        ctrl_runs = [eir.read_sink(eir.job_ctrl(en.MAIN, "a3h", op, en.STEPS, sd)["sink"]) for sd in SEEDS5]
        ct = mean(bk(x["record"]["backend"]) for x in ctrl_runs)
        l1 = eir.read_sink(eir.job_l1(en.l1_out(fid), "a3h", op, fid)["sink"])
        full = full_table(op, fid) if fid in ("F4a", "F8a") else None
        if full is None:       # F5: same coordinates as stage 2's F5 config
            rows2 = s2_full(f"{op}_f5")
            b2 = min(rows2, key=lambda r: r["mean"])
            full = {"run": b2["run"], "mean": b2["mean"], "min": None, "max": None, "n": 14, "route_or_source_on_failed": None, "all_means": sorted(r["mean"] for r in rows2)}
        c = {"op": op, "fault": fid, "T_pre": pre["T"], "pre": pre_b, "ctrl": ct, "full": full, "methods": {}, "R0": bk(l1["R0"]["backend"]), "L1": bk(l1["L1"]["backend"]), "T_L1": l1["T_L1"], "T_R0": l1["T_R0"], "faults": l1["faults"]}
        L.append(f"\n## {op} {fid}（故障前后端 {f0(pre_b)}，T̂ {f0(pre['T'])}；R0 {f0(c['R0'])}（{pc(c['R0'], pre_b):+.1f}%），L1 {f0(c['L1'])}（{pc(c['L1'], pre_b):+.1f}%），T̂_L1 {f0(l1['T_L1'])}）\n")
        L.append("| 方法 | 种子数 | 后端 | 范围 | 相对故障前 | 净代价 | 行为搬动 | DDR 读源 MB | 唯一块 MB | t_决策 s | T̂ | T̂ 误差 | 撤销比例 | K / 行为覆盖 / DDR 端口 | 放行 | 回弹（真实副本 / 过时 / 其他） |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m, name in METHODS:
            runs = load(op, fid, m)
            if not runs:
                L.append(f"| {name} | 缺 | | | | | | | | | | | | | | |")
                continue
            per = {sd: bk(d["record"]["backend"]) for sd, d in runs.items()}
            allv = [v[0] for d in runs.values() for v in d["record"]["backend"].values()]
            mg = lambda k: mean(d["record"]["migration_vs_start"][k] for d in runs.values())
            n_prop = sum(eir.CAL_PROPOSALS + d["spec"]["steps"] for d in runs.values())
            ill = sum(d["search"]["n_illegal"] + d["search"]["cal_illegal"] for d in runs.values())
            loc = next(iter(runs.values()))["search"]["local"]
            allowed = sum((d["search"]["local_audit"] or {}).get("ddr_outside_allowed", 0) for d in runs.values())
            w = collections.Counter()
            for d in runs.values():
                w.update(d["search"].get("read_watch", {}))
            T = mean(d["record"]["T"] for d in runs.values())
            terr = mean(pc(d["record"]["T"], per[sd]) for sd, d in runs.items())
            x = {"seed_backend": per, "mean": mean(per.values()), "min": min(allv), "max": max(allv), "T": T, "T_err": terr, "revoked": ill / n_prop, "t_decision": mean(d["timing"]["t_decision"] for d in runs.values()),
                 "moved": mg("moved_behas_share"), "ddr_mb": mg("ddr_read_moved_bytes") / 1e6, "uniq_mb": mg("unique_block_bytes") / 1e6, "local": loc, "allowed": allowed, "watch": dict(w),
                 "viol": sum(len(d["record"]["fault_violations"]) for d in runs.values()), "n": len(runs), "accepted": mean(sum(r["accepted"] for r in d["search"]["rows"]) for d in runs.values()),
                 "best_T": {sd: d["search"]["best_T"] for sd, d in runs.items()}, "best_step": {sd: d["search"]["best_step"] for sd, d in runs.items()},
                 "outside": sum((d["search"]["local_audit"] or {}).get("beha_not_movable", 0) + (d["search"]["local_audit"] or {}).get("beha_dest_outside", 0) for d in runs.values()),
                 "stale_final": sum(d["search"].get("stale_final", 0) for d in runs.values()), "sync_entry": next(iter(runs.values()))["search"].get("sync_entry"), "revert": None}
            if m == "L2Impact":
                x["impact"] = {k: v for k, v in next(iter(runs.values()))["search"]["impact"].items() if k not in ("score", "cb", "geo_cov")}
            c["methods"][m] = x
            kc = "—" if not loc else f"{loc['n_cores']} / {100 * loc['beha_share']:.0f}% / {loc['n_ddr_ports']}"
            wt = "—" if m not in eir.SYNC_METHODS else f"{w.get('rebound', 0)}（{w.get('rebound_real_copy', 0)} / {w.get('rebound_stale', 0)} / {w.get('rebound_other', 0)}）"
            L.append(f"| {name} | {x['n']} | {f0(x['mean'])} | [{f0(x['min'])}, {f0(x['max'])}] | {pc(x['mean'], pre_b):+.1f}% | {x['mean'] - ct:+,.0f}（{pc(x['mean'], ct):+.1f}%） | {100 * x['moved']:.1f}% | {x['ddr_mb']:.1f} | {x['uniq_mb']:.1f} | {x['t_decision']:.1f} | {f0(T)} | {terr:+.2f}% | {'—' if not loc else format(100 * x['revoked'], '.1f') + '%'} | {kc} | {allowed if m in ('L2Lsync', 'L2Impact') else '—'} | {wt} |")
        L.append(f"\n对照（无故障继续搜索，s = 100，旧规则）：{f0(ct)}（{pc(ct, pre_b):+.1f}%）。")
        if full:
            rng_ = "" if full["min"] is None else f" [{f0(full['min'])}, {f0(full['max'])}]"
            chk = "" if full["route_or_source_on_failed"] is None else f"；路由或源落在失效元件上 {full['route_or_source_on_failed']}"
            L.append(f"FULL-A3h（14 次取后端最优 {full['run']}）：{f0(full['mean'])}{rng_}，{pc(full['mean'], pre_b):+.1f}%；14 次均值中位数 {f0(median(full['all_means']))}{chk}。三种读法见“FULL-A3h 的三种读法”。")
        res.append(c)
    # ---- K and coverage
    L.append("\n## L2-Impact 的集合\n")
    L.append("| 任务 | 故障 | K | 行为覆盖 | 权重块覆盖 | DDR 端口 | 候选核 | 第 1 组（候选 / 选中） | 第 2 组（候选 / 选中） | 与几何重合 | 几何行为覆盖 | 改变的通信 | W 未对齐 tag |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in res:
        x = c["methods"].get("L2Impact")
        if not x:
            continue
        i, lc = x["impact"], x["local"]
        g = next(iter(load(c["op"], c["fault"], "L2Impact").values()))["search"]["impact"]["geo_cov"]
        L.append(f"| {c['op']} | {c['fault']} | {i['K']} | {100 * lc['beha_share']:.0f}%（{lc['n_behas']}） | {100 * lc['splits_share']:.0f}% | {lc['n_ddr_ports']} | {i['n_cand']} | {i['n_g1']} / {i['chosen_g1']} | {i['n_g2']} / {i['chosen_g2']} | {i['overlap_geo']} | {100 * g['beha_share']:.0f}%（{g['n_behas']}） | {i['n_changed']} | {i['n_unaligned_W']} / {i['n_events']} |")
    # ---- paired
    stats20 = []
    L.append("\n## 20 个种子的配对统计\n")
    L.append(f"差 = 前者 − 后者（后端 10 个种子均值），负数表示前者更好。胜负平从前者看。区间是中位差的 95% 百分位 bootstrap（{BOOT_N} 次重采样，`numpy.default_rng({BOOT_SEED})`）；p 是 Wilcoxon 符号秩（双侧，只作参照）。原 L2L、原 L2G 的种子 123、1、2、3、4 来自 `out/main`（F8a 来自 `out/next`），其余来自 `out/next`，同一代码路径。\n")
    L.append("| 任务 | 故障 | 比较 | n | 中位差 | 中位差 / 故障前 | 胜 / 负 / 平 | 95% 区间 | 区间不含 0 | Wilcoxon p |\n|---|---|---|---|---|---|---|---|---|---|")
    for c in res:
        full20 = [p for p in PAIRS20 if p[0] in c["methods"] and p[1] in c["methods"]]
        for a, b in full20:
            pa, pb = c["methods"][a]["seed_backend"], c["methods"][b]["seed_backend"]
            common = [sd for sd in SEEDS if sd in pa and sd in pb]
            if len(common) < 20:
                continue
            st = boot_stats([pa[sd] - pb[sd] for sd in common], c["pre"])
            stats20.append({"op": c["op"], "fault": c["fault"], "a": a, "b": b, **st})
            L.append(f"| {c['op']} | {c['fault']} | {a} − {b} | {st['n']} | {st['median']:+,.0f} | {st['rel']:+.1f}% | {st['win']} / {st['loss']} / {st['tie']} | [{st['lo']:+,.0f}, {st['hi']:+,.0f}] | {'是' if st['excl0'] else '否'} | {'—' if st['p'] is None else format(st['p'], '.3f')} |")
    gate = [s for s in stats20 if s["op"] == "ffn" and (s["a"], s["b"]) == GATE_PAIR]
    gate_hit = [s["fault"] for s in gate if s["rel"] <= GATE_REL and s["excl0"]]
    gate_ok = len(gate) == 3 and len(gate_hit) >= GATE_MIN
    L.append(f"\n**关口判据（开跑前写定）**：{GATE_PAIR[0]} − {GATE_PAIR[1]} 在 3 个 FFN 组合中至少 {GATE_MIN} 个满足中位差 ≤ {GATE_REL:.0f}%（相对故障前）且区间不含 0。满足的组合：{gate_hit or '无'}。**结论：{'满足' if gate_ok else '不满足'}。**\n")
    L.append("\n## 同种子配对比较（原 5 个种子）\n")
    L.append("差 = 前者 − 后者（后端 10 种子均值），负数表示前者更好。符号计数：前者更好 / 更差 / 相等。只有原 5 个种子。\n")
    L.append("| 任务 | 故障 | 比较 | 种子 123、1、2、3、4 的差 | 中位差 | 中位差 / 故障前 | 符号 |\n|---|---|---|---|---|---|---|")
    pairs = []
    for c in res:
        for a, b in PAIRS:
            if a not in c["methods"] or b not in c["methods"]:
                continue
            pa, pb = c["methods"][a]["seed_backend"], c["methods"][b]["seed_backend"]
            ds = [pa[sd] - pb[sd] for sd in SEEDS5 if sd in pa and sd in pb]
            sign = (sum(v < 0 for v in ds), sum(v > 0 for v in ds), sum(v == 0 for v in ds))
            pairs.append({"op": c["op"], "fault": c["fault"], "a": a, "b": b, "diffs": ds, "median": median(ds), "sign": sign})
            L.append(f"| {c['op']} | {c['fault']} | {a} − {b} | {' '.join(f'{v:+,.0f}' for v in ds)} | {median(ds):+,.0f} | {100 * median(ds) / c['pre']:+.1f}% | {sign[0]} / {sign[1]} / {sign[2]} |")
    L.append("\n合计（所有组合的全部配对）：\n")
    L.append("| 比较 | 配对数 | 前者更好 / 更差 / 相等 | 全部差的中位数 | 组合中位差为负的组合数 |\n|---|---|---|---|---|")
    for a, b in PAIRS:
        ps = [p for p in pairs if p["a"] == a and p["b"] == b]
        ds = [v for p in ps for v in p["diffs"]]
        if ds:
            L.append(f"| {a} − {b} | {len(ds)} | {sum(v < 0 for v in ds)} / {sum(v > 0 for v in ds)} / {sum(v == 0 for v in ds)} | {median(ds):+,.0f} | {sum(p['median'] < 0 for p in ps)} / {len(ps)} |")
    # ---- process noise: the same search rerun (rep_v1)
    L.append("\n## 同一状态换进程的后端差（噪声参照）\n")
    L.append("`out/next/rep_v1/` 是修正“权重移动后全量同步”之前的 90 次 `+sync` 运行。修正后重跑，每步记录、T̂、行为位置和路由是否全同见下表最后一列；全同时，后端的差只来自后端依赖对象地址的已知特性。差 = |重跑 − 第一版|，都是 10 个后端种子的均值。\n")
    L.append("| 任务 | 运行数 | 差的中位数 | 差的最大值 | 中位数 / 故障前 | 最大值 / 故障前 | 记录、T̂、位置、路由全同 |\n|---|---|---|---|---|---|---|")
    noise = {}
    for op in en.OPS:
        ds, same = [], 0
        for fid in en.FAULTS:
            for m in en.NEW_METHODS:
                for sd in SEEDS5:
                    j = en.job_rep(op, fid, m, sd)
                    v1 = os.path.join(en.OUT, "rep_v1", os.path.basename(j["sink"]))
                    if os.path.exists(v1):
                        d2, d1 = eir.read_sink(j["sink"]), eir.read_sink(v1)
                        ds.append(abs(bk(d2["record"]["backend"]) - bk(d1["record"]["backend"])))
                        same += (d2["search"]["rows"] == d1["search"]["rows"] and d2["record"]["T"] == d1["record"]["T"] and d2["record"]["beha_loc"] == d1["record"]["beha_loc"]
                                 and d2["record"]["routes"] == d1["record"]["routes"])
        pre_b = next(c["pre"] for c in res if c["op"] == op)
        noise[op] = {"n": len(ds), "median": median(ds), "max": max(ds), "share_zero": sum(v == 0 for v in ds) / len(ds)}
        L.append(f"| {op} | {len(ds)} | {median(ds):,.0f} | {max(ds):,.0f} | {100 * median(ds) / pre_b:.2f}% | {100 * max(ds) / pre_b:.2f}% | {same} / {len(ds)} |")
    # ---- vs FULL
    L.append("\n## 与 FULL-A3h 的差距（F4a、F8a）\n")
    L.append("FULL-A3h 取 14 次中后端最优；它在故障平台上从头搜索，故障前参照不同，见 `CLAUDE.md` E3 的说明。\n")
    L.append("| 任务 | 故障 | FULL-A3h | " + " | ".join(n for _, n in METHODS) + " |\n|---|---|---|---|---|---|---|---|")
    for c in res:
        if c["full"]:
            L.append(f"| {c['op']} | {c['fault']} | {f0(c['full']['mean'])} | " + " | ".join(f"{pc(c['methods'][m]['mean'], c['full']['mean']):+.1f}%" if m in c["methods"] else "—" for m, _ in METHODS) + " |")
    fb, fx = full_block()
    L += fb
    # ---- self checks
    tot = {k: sum(x[k] for c in res for x in c["methods"].values()) for k in ("viol", "outside", "stale_final", "n")}
    miss = sum(x["watch"].get("rebound_stale", 0) for c in res for x in c["methods"].values())
    drop = sum(x["watch"].get("stale_after_proposal", 0) for c in res for x in c["methods"].values())
    L.append(f"\n自检：运行 {tot['n']} 次（含复用）；故障违规 {tot['viol']}；行为或落点在集合外 {tot['outside']}；`+sync` 结束时过时位置 {tot['stale_final']}；`+sync` 过时回弹 {miss}；每次提议后过时 {drop}。\n")
    os.makedirs(en.OUT, exist_ok=True)
    with open(os.path.join(en.OUT, "report_a.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    with open(os.path.join(en.OUT, "report_a.json"), "w", encoding="utf-8") as f:
        json.dump({"combos": res, "pairs": pairs, "noise": noise, "stats20": stats20, "gate": {"ok": gate_ok, "hit": gate_hit}, "full_three": fx}, f, indent=0, default=str)
    print("\n".join(L))


def diag():
    """Next handoff point 6, read only: FFN F4a, F8a, L2Impact and L2L+sync. Counts from the stored rows (20 seeds); what the
    rows alone cannot tell comes from the step log of the 15 new seeds (row["diag"], added without changing the search)."""
    import pickle
    L = ["# 第 6 点诊断：定向移动的分支（FFN F4a、F8a；L2Impact、L2L+sync）\n",
         "(a) 来自存下的每步记录（20 个种子）；回退、top-K 组成和候选数来自 15 个新种子的步级日志（只读，`check_log` 证明它不改变搜索）。每行是该组合 × 方法在各种子上的合计。\n"]
    res = {}
    corners = {0, 3, 12, 15}
    all_cores = [(d, c) for d in range(4) for c in range(16)]
    for fid in ("F4a", "F8a"):
        base = eir.read_sink(eir.job_l1(en.l1_out(fid), "a3h", "ffn", fid)["sink"])
        s0 = pickle.loads(base["L1_snap"]["blob"])
        tens, vect = collections.Counter(), collections.Counter()
        for k, v in s0[3].items():
            tens[tuple(k[:-1])] += float(v)
        for k, v in s0[4].items():
            vect[tuple(k[:-1])] += float(v)
        pct = lambda load, c: 100 * sum(load[x] <= load[c] for x in all_cores) / len(all_cores)
        idle_all = [c for c in all_cores if tens[c] == 0 and vect[c] == 0]
        for m in ("L2Impact", "L2Lsync"):
            runs = {sd: eir.read_sink(en.job_rep("ffn", fid, m, sd)["sink"]) for sd in SEEDS}
            nodes = sorted(next(iter(runs.values()))["search"]["local_nodes"])
            prop, acc = collections.Counter(), collections.Counter()
            lands = []
            for d in runs.values():
                for r in d["search"]["rows"]:
                    if r["targeted"]:
                        prop[r["action"]] += 1
                        acc[r["action"]] += r["accepted"]
                    if r["accepted"]:
                        lands += [tuple(c) for c in r.get("landing", [])]
            logged = [d for d in runs.values() if "diag" in d["search"]["rows"][0]]
            rows = [r for d in logged for r in d["search"]["rows"]]
            drawn = sum(r["diag"].get("drawn", False) for r in rows)
            fallback = sum(r["diag"].get("fallback", False) for r in rows)
            top, cp = collections.Counter(), collections.Counter()
            chosen = collections.Counter()
            cands = []
            weight_ret = 0
            for r in rows:
                dg = r["diag"]
                top.update(dg.get("top", {}))
                cp.update(dg.get("cp", {}))
                if "chosen" in dg:
                    chosen[dg["chosen"]] += 1
                weight_ret += dg.get("weight_returned", False)
                cands += dg["cands"]
            n_c = [c["n"] for c in cands]
            idle_c = [c["idle"] for c in cands]
            x = {"K": len(nodes), "n_runs": len(runs), "n_logged": len(logged),
                 "prop": dict(prop), "acc": dict(acc), "drawn": drawn, "fallback": fallback, "logged_steps": len(rows),
                 "tensor_pct_median": median(pct(tens, c) for c in nodes), "vector_pct_median": median(pct(vect, c) for c in nodes),
                 "corners": sum(c[1] in corners for c in nodes), "idle_in_set": sum(c in idle_all for c in nodes), "idle_all": len(idle_all),
                 "land_n": len(lands), "land_tensor_pct_median": median(pct(tens, c) for c in lands) if lands else None, "land_vector_pct_median": median(pct(vect, c) for c in lands) if lands else None,
                 "land_idle_share": sum(c in idle_all for c in lands) / len(lands) if lands else None,
                 "tensor_load_set": median(tens[c] for c in nodes), "tensor_load_all": median(tens[c] for c in all_cores), "vector_load_set": median(vect[c] for c in nodes), "vector_load_all": median(vect[c] for c in all_cores),
                 "chosen_load_median": median(c["load"] for c in cands if c["load"] is not None) if cands else None,
                 "comp_dest_calls": len(cands), "comp_dest_none": sum(v == 0 for v in n_c), "cands_median": median(n_c) if n_c else None, "idle_cands_median": median(idle_c) if idle_c else None,
                 "calls_with_idle_cand": sum(v > 0 for v in idle_c) / len(idle_c) if idle_c else None,
                 "top": dict(top), "cp": dict(cp), "chosen": dict(chosen), "weight_returned": weight_ret, "chosen_n": sum(chosen.values())}
            res[f"{fid}_{m}"] = x
    L.append("\n## (a) 定向移动各分支（20 个种子，提议 / 接受）\n")
    L.append("| 故障 | 方法 | weight | comp | consumer | producer | 定向合计 | 定向抽到次数（15 种子） | 回退到随机（15 种子） |\n|---|---|---|---|---|---|---|---|---|")
    for k, x in res.items():
        fid, m = k.split("_")
        cell = lambda a: f"{x['prop'].get(a, 0)} / {x['acc'].get(a, 0)}"
        L.append(f"| {fid} | {m} | {cell('weight')} | {cell('comp')} | {cell('consumer')} | {cell('producer')} | {sum(x['prop'].values())} | {x['drawn']} | {x['fallback']} |")
    L.append("\n## (b) 集合内的核与落点（负载来自 L1 状态；百分位 = 全图 64 个核中负载不高于它的比例）\n")
    L.append("| 故障 | 方法 | K | 张量负载百分位中位数 | 向量负载百分位中位数 | 角落核 | 完全空闲核（集合内 / 全图） | 接受的落点数 | 落点张量 / 向量百分位中位数 | 落点在全图空闲核的份额 | comp_dest 调用 / 无候选 | 候选数中位数 | 负载为 0 的候选数中位数 | 有空闲候选的调用占比 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    fmt = lambda v, f=".0f": "—" if v is None else format(v, f)
    for k, x in res.items():
        fid, m = k.split("_")
        L.append(f"| {fid} | {m} | {x['K']} | {fmt(x['tensor_pct_median'])}% | {fmt(x['vector_pct_median'])}% | {x['corners']} | {x['idle_in_set']} / {x['idle_all']} | {x['land_n']} | {fmt(x['land_tensor_pct_median'])}% / {fmt(x['land_vector_pct_median'])}% | {fmt(None if x['land_idle_share'] is None else 100 * x['land_idle_share'])}% | {x['comp_dest_calls']} / {x['comp_dest_none']} | {fmt(x['cands_median'], '.1f')} | {fmt(x['idle_cands_median'], '.1f')} | {fmt(None if x['calls_with_idle_cand'] is None else 100 * x['calls_with_idle_cand'])}% |")
    L.append("\n百分位有大量并列（很多核负载相同），所以另列绝对负载：\n")
    L.append("| 故障 | 方法 | 张量负载中位数（集合 / 全图） | 向量负载中位数（集合 / 全图） | comp_dest 选中目标的当时负载中位数 |\n|---|---|---|---|---|")
    for k, x in res.items():
        fid, m = k.split("_")
        L.append(f"| {fid} | {m} | {fmt(x['tensor_load_set'])} / {fmt(x['tensor_load_all'])} | {fmt(x['vector_load_set'])} / {fmt(x['vector_load_all'])} | {fmt(x['chosen_load_median'])} |")
    L.append("\n## (c) 两种解释（15 个新种子的步级日志）\n")
    L.append("关键路径 = 每次定向抽到时、集合过滤之前的关键路径事件；top-K = 过滤之后按 W 取的前 K 个（K = 5）。“抽中”是实际随机选中的那个事件。\n")
    L.append("| 故障 | 方法 | 关键路径事件：计算 / DDR 读 / 核间通信 | top-K 事件数 | top-K 中 DDR 读占比 | DDR 读中 ws 在集合内占比 | top-K 中计算占比 | 抽中事件数 | 抽中 DDR 读且 ws 在集合内 | 其中权重移动直接返回（到 consumer/producer 之前） | 占抽中的比例 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for k, x in res.items():
        fid, m = k.split("_")
        t, cpx = x["top"], x["cp"]
        n = t.get("n", 0) or 1
        dr = t.get("ddr_read", 0)
        ch = x["chosen"]
        L.append(f"| {fid} | {m} | {cpx.get('comp', 0)} / {cpx.get('ddr_read', 0)} / {cpx.get('core_comm', 0)} | {t.get('n', 0)} | {100 * dr / n:.0f}% | {100 * t.get('ddr_read_ws_in_set', 0) / (dr or 1):.0f}% | {100 * t.get('comp', 0) / n:.0f}% | {x['chosen_n']} | {ch.get('ddr_read_ws_in_set', 0)} | {x['weight_returned']} | {100 * x['weight_returned'] / (x['chosen_n'] or 1):.0f}% |")
    with open(os.path.join(en.OUT, "report_diag.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(en.OUT, "report_diag.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=0, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    {"prior": prior, "a": report_a, "diag": diag}[sys.argv[1]]()
