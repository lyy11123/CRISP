"""Phase A report (EIR_NEXT_HANDOFF.md 4.5): numbers only. Reads out/next and, for what phase A reuses, out/main.

  python report_next.py prior      section 4.2's comparison
  python report_next.py a          out/next/report_a.md and report_a.json
"""
import collections
import json
import os
import statistics
import sys

import eir
import eir_next as en

mean, median = statistics.mean, statistics.median
METHODS = [("L2L", "原 L2-Local"), ("L2Lsync", "L2L+sync"), ("L2Impact", "L2Impact"), ("L2G", "原 L2-Global"), ("L2Gsync", "L2G+sync")]
PAIRS = [("L2Impact", "L2Lsync"), ("L2Lsync", "L2L"), ("L2Gsync", "L2G"), ("L2Impact", "L2Gsync")]
SEEDS = eir.SA_SEEDS
COMBOS = [(o, f) for o in en.OPS for f in en.FAULTS]


def pc(a, b):
    return 100 * (a / b - 1)


def f0(x):
    return "—" if x is None else f"{x:,.0f}"


def bk(b):
    return mean(v[0] for v in b.values())


def sink_of(op, fid, m, sd):
    if m in en.OLD_METHODS and fid != "F8a":
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
         "只报数字。A3h 起点，s = 100，5 个 SA 种子（123、1、2、3、4），后端 10 个种子，广播种子 0。原 L2-Local、原 L2-Global 在 F4a、F5 上复用 `out\\main`，F8a 上新跑。净代价的对照是 `out\\main` 的无故障继续搜索（s = 100，**旧规则**）。决策耗时含并行争用，不作论文数据。\n"]
    rows_all = []
    for op, fid in COMBOS:
        pre = eir.read_sink(eir.job_pre(en.MAIN, "a3h", op)["sink"])
        pre_b = bk(pre["record"]["backend"])
        ctrl_runs = [eir.read_sink(eir.job_ctrl(en.MAIN, "a3h", op, en.STEPS, sd)["sink"]) for sd in SEEDS]
        ct = mean(bk(x["record"]["backend"]) for x in ctrl_runs)
        l1 = eir.read_sink(eir.job_l1(en.l1_out(fid), "a3h", op, fid)["sink"])
        full = full_table(op, fid) if fid in ("F4a", "F8a") else None
        c = {"op": op, "fault": fid, "T_pre": pre["T"], "pre": pre_b, "ctrl": ct, "full": full, "methods": {}, "R0": bk(l1["R0"]["backend"]), "L1": bk(l1["L1"]["backend"]), "T_L1": l1["T_L1"], "T_R0": l1["T_R0"], "faults": l1["faults"]}
        L.append(f"\n## {op} {fid}（故障前后端 {f0(pre_b)}，T̂ {f0(pre['T'])}；R0 {f0(c['R0'])}（{pc(c['R0'], pre_b):+.1f}%），L1 {f0(c['L1'])}（{pc(c['L1'], pre_b):+.1f}%），T̂_L1 {f0(l1['T_L1'])}）\n")
        L.append("| 方法 | 后端 | 范围 | 相对故障前 | 净代价 | 行为搬动 | DDR 读源 MB | 唯一块 MB | t_决策 s | T̂ | T̂ 误差 | 撤销比例 | K / 行为覆盖 / DDR 端口 | 放行 | 回弹（真实副本 / 过时 / 其他） |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m, name in METHODS:
            runs = load(op, fid, m)
            if not runs:
                L.append(f"| {name} | 缺 | | | | | | | | | | | | | |")
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
            L.append(f"| {name} | {f0(x['mean'])} | [{f0(x['min'])}, {f0(x['max'])}] | {pc(x['mean'], pre_b):+.1f}% | {x['mean'] - ct:+,.0f}（{pc(x['mean'], ct):+.1f}%） | {100 * x['moved']:.1f}% | {x['ddr_mb']:.1f} | {x['uniq_mb']:.1f} | {x['t_decision']:.1f} | {f0(T)} | {terr:+.2f}% | {'—' if not loc else format(100 * x['revoked'], '.1f') + '%'} | {kc} | {allowed if m in ('L2Lsync', 'L2Impact') else '—'} | {wt} |")
        L.append(f"\n对照（无故障继续搜索，s = 100，旧规则）：{f0(ct)}（{pc(ct, pre_b):+.1f}%）。")
        if full:
            L.append(f"FULL-A3h（14 次取后端最优 {full['run']}）：{f0(full['mean'])} [{f0(full['min'])}, {f0(full['max'])}]，{pc(full['mean'], pre_b):+.1f}%；14 次均值中位数 {f0(median(full['all_means']))}；路由或源落在失效元件上 {full['route_or_source_on_failed']}。")
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
    L.append("\n## 同种子配对比较\n")
    L.append("差 = 前者 − 后者（后端 10 种子均值），负数表示前者更好。符号计数：前者更好 / 更差 / 相等。5 个种子，不作显著性结论。\n")
    L.append("| 任务 | 故障 | 比较 | 种子 123、1、2、3、4 的差 | 中位差 | 中位差 / 故障前 | 符号 |\n|---|---|---|---|---|---|---|")
    pairs = []
    for c in res:
        for a, b in PAIRS:
            if a not in c["methods"] or b not in c["methods"]:
                continue
            pa, pb = c["methods"][a]["seed_backend"], c["methods"][b]["seed_backend"]
            ds = [pa[sd] - pb[sd] for sd in SEEDS if sd in pa and sd in pb]
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
    # ---- vs FULL
    L.append("\n## 与 FULL-A3h 的差距（F4a、F8a）\n")
    L.append("FULL-A3h 取 14 次中后端最优；它在故障平台上从头搜索，故障前参照不同，见 `CLAUDE.md` E3 的说明。\n")
    L.append("| 任务 | 故障 | FULL-A3h | " + " | ".join(n for _, n in METHODS) + " |\n|---|---|---|---|---|---|---|---|")
    for c in res:
        if c["full"]:
            L.append(f"| {c['op']} | {c['fault']} | {f0(c['full']['mean'])} | " + " | ".join(f"{pc(c['methods'][m]['mean'], c['full']['mean']):+.1f}%" if m in c["methods"] else "—" for m, _ in METHODS) + " |")
    # ---- self checks
    tot = {k: sum(x[k] for c in res for x in c["methods"].values()) for k in ("viol", "outside", "stale_final", "n")}
    miss = sum(x["watch"].get("rebound_stale", 0) for c in res for x in c["methods"].values())
    L.append(f"\n自检：运行 {tot['n']} 次（含复用）；故障违规 {tot['viol']}；行为或落点在集合外 {tot['outside']}；`+sync` 结束时过时位置 {tot['stale_final']}；`+sync` 过时回弹 {miss}。\n")
    os.makedirs(en.OUT, exist_ok=True)
    with open(os.path.join(en.OUT, "report_a.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    with open(os.path.join(en.OUT, "report_a.json"), "w", encoding="utf-8") as f:
        json.dump({"combos": res, "pairs": pairs}, f, indent=0, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    {"prior": prior, "a": report_a}[sys.argv[1]]()
