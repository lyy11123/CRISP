"""Batch report of the EIR main run: numbers only (handoff sections 7, 8).

  python report_batch.py ffn|mha|ffn14|a3h|afault|ctrl [--out out/main]
Writes <out>/report_<batch>.md and .json. FULL-BusyBarn = E3's FULL state with 10 seeds (job full_e3); FFN and MHA at
F4, F5 also show stage 2's A0 (backend-best of 14). FULL-A3h = stage 2 (FFN, MHA at F4, F5) or this run (1x4 FFN).
"""
import argparse
import json
import os
import statistics

import eir

cs = eir.cs
mean = statistics.mean
BATCH = {"ffn": [("e3", "ffn")], "mha": [("e3", "mha")], "ffn14": [("e3", "ffn14")], "a3h": [("a3h", "ffn"), ("a3h", "mha")],
         "afault": [("a3h", "ffn"), ("a3h", "mha")]}


def bk_stats(b):
    v = [x[0] for x in b.values()]
    return {"mean": mean(v), "min": min(v), "max": max(v), "triple": [mean(x[i] for x in b.values()) for i in range(3)]}


def s2_best(cfg, arm):
    """Backend-best of stage 2's 14 runs of the config: (10-seed mean, min, max, run)."""
    best = None
    for g, x in cs.variants(arm):
        s = cs.load_run(cfg, arm, g, x, eir.S2_OUT)["selected"]
        st = bk_stats(s["backend"])
        if best is None or st["mean"] < best["mean"]:
            best = {**st, "run": f"g{g}_{x}"}
    return best


def full_refs(out, op, fid):
    fb = eir.read_sink(eir.job_full_e3(out, op, fid)["sink"]) if os.path.exists(eir.job_full_e3(out, op, fid)["sink"]) else None
    r = {"FULL_BB": bk_stats(fb["record"]["backend"]) if fb else None, "FULL_BB_T": fb["record"]["T"] if fb else None,
         "FULL_BB_variant": fb["e3"]["variant"] if fb else None,
         "FULL_BB_mig": fb["record"]["migration_vs_start"] if fb else None,
         "FULL_BB_violations": len(fb["record"]["fault_violations"]) if fb else None}
    if (op, fid.rstrip("a")) in {("ffn", "F4"), ("ffn", "F5"), ("mha", "F4"), ("mha", "F5")}:    # F4a, F5a: another link
        cfg = f"{op}_{fid.rstrip('a').lower()}"
        r["FULL_A0_s2"] = s2_best(cfg, "A0")
        r["FULL_A3h"] = s2_best(cfg, "A3h")
    elif op == "ffn14" and fid in ("F4", "F5"):
        best = None
        for g, x in cs.variants("A3h"):
            j = eir.job_full(out, op, fid, "A3h", g, x)
            if os.path.exists(j["sink"]):
                st = bk_stats(eir.read_sink(j["sink"])["selected"]["backend"])
                if best is None or st["mean"] < best["mean"]:
                    best = {**st, "run": f"g{g}_{x}"}
        r["FULL_A3h"] = best
    return r


def pc(a, b):
    return 100 * (a / b - 1)


def ctrl_table(out):
    """Handoff 12.2: (start, op, s) -> 10-seed mean over the 5 SA seeds, with spread, migration and timing."""
    t = {}
    for start, op in eir.CTRL_STARTS:
        for st in eir.CTRL_STEPS:
            runs = [eir.read_sink(j["sink"]) for j in (eir.job_ctrl(out, start, op, st, sd) for sd in eir.SA_SEEDS) if os.path.exists(j["sink"])]
            if not runs:
                continue
            per = [bk_stats(x["record"]["backend"]) for x in runs]
            allv = [v[0] for x in runs for v in x["record"]["backend"].values()]
            mg = lambda f: mean(x["record"]["migration_vs_start"][f] for x in runs)
            t[(start, op, st)] = {"mean": mean(p["mean"] for p in per), "min": min(allv), "max": max(allv), "T": mean(x["record"]["T"] for x in runs),
                                  "triple": [mean(p["triple"][i] for p in per) for i in range(3)], "n": len(runs),
                                  "mig": {f: mg(f) for f in ("moved_behas_share", "ddr_read_moved_bytes", "unique_block_bytes")},
                                  "t_decision": mean(x["timing"]["t_decision"] for x in runs), "T_err": mean(pc(x["record"]["T"], p["mean"]) for x, p in zip(runs, per)),
                                  "cpcm": mean(x["cpcm_calls"]["L2"] for x in runs), "viol": sum(len(x["record"]["fault_violations"]) for x in runs)}
    return t


def write_ctrl(out):
    t = ctrl_table(out)
    L = ["# EIR 无故障对照（12.2）\n", "健康平台，从故障前映射出发，L2-Global 的规则。延迟为后端 10 个种子均值，再对 5 个 SA 种子取平均。\n",
         "| 起点 | 任务 | s | 故障前后端 | 后端 | 范围 | 三元组 | 相对故障前 | 行为搬动 | DDR 读源字节 | 唯一块字节 | t_决策 s | T̂ 误差 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (start, op, st), x in t.items():
        pre = bk_stats(eir.read_sink(eir.job_pre(out, start, op)["sink"])["record"]["backend"])["mean"]
        L.append(f"| {start} | {op} | {st} | {f0(pre)} | {f0(x['mean'])} | [{f0(x['min'])}, {f0(x['max'])}] | {' / '.join(f0(v) for v in x['triple'])} | {pc(x['mean'], pre):+.1f}% | {100 * x['mig']['moved_behas_share']:.1f}% | "
                 f"{f0(x['mig']['ddr_read_moved_bytes'])} | {f0(x['mig']['unique_block_bytes'])} | {x['t_decision']:.1f} | {x['T_err']:+.2f}% |")
    L.append(f"\n自检：{sum(x['n'] for x in t.values())} 次；故障违规 {sum(x['viol'] for x in t.values())}。\n")
    p = os.path.join(out, "report_ctrl.md")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    with open(os.path.join(out, "report_ctrl.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in t.items()}, f, indent=0)
    return p


def combos_of(batch):
    if batch == "afault":
        return eir.a_combos()
    return [c for c in eir.COMBOS if c[:2] in BATCH[batch]]


def collect(out, batch):
    res = []
    ctrl = ctrl_table(out)
    for start, op in BATCH[batch]:
        pre = eir.read_sink(eir.job_pre(out, start, op)["sink"])
        bk_pre = bk_stats(pre["record"]["backend"])
        for s_, o_, fid in combos_of(batch):
            if (s_, o_) != (start, op):
                continue
            d = eir.read_sink(eir.job_l1(out, start, op, fid)["sink"])
            refs = full_refs(out, op, fid)
            c = {"start": start, "op": op, "fault": fid, "T_pre": pre["T"], "pre": bk_pre, **refs, "link_stats": d["link_stats"],
                 "L1_run": {k: d["L1_run"][k] for k in ("attempts", "accepted", "reason")}, "levels": {}, "n_changed_routes_R0": d["n_changed_routes_R0"],
                 "n_changed_routes_L1": d["n_changed_routes_L1"], "n_runs": 0}
            for lev in ("R0", "L1"):
                r = d[lev]
                b = bk_stats(r["backend"])
                c["levels"][lev] = {"T": r["T"], **b, "T_err": pc(r["T"], b["mean"]), "mig_start": r["migration_vs_start"], "mig_R0": r.get("migration_vs_R0"),
                                    "t_R0": d["t_R0"], "t_L1": d["t_L1"] if lev == "L1" else 0, "t_cal": 0, "t_L2": 0, "t_fin": r["t_finalize"],
                                    "t_decision": d["t_R0"] + (d["t_L1"] if lev == "L1" else 0) + r["t_finalize"], "t_backend": r["t_backend"], "viol": len(r["fault_violations"])}
            for m in eir.METHODS:
                for st in eir.steps_of(fid):
                    runs = []
                    for sd in eir.SA_SEEDS:
                        j = eir.job_rep(out, start, op, fid, m, st, sd)
                        if os.path.exists(j["sink"]):
                            runs.append(eir.read_sink(j["sink"]))
                    if not runs:
                        continue
                    c["n_runs"] += len(runs)
                    per = [bk_stats(x["record"]["backend"]) for x in runs]
                    allv = [v[0] for x in runs for v in x["record"]["backend"].values()]
                    own = min(range(len(runs)), key=lambda i: runs[i]["search"]["best_E"] if m == "W" else runs[i]["search"]["best_T"])
                    mg = lambda key, f: mean(x["record"][key][f] for x in runs)
                    tt = lambda f: mean(x["timing"][f] for x in runs)
                    prop = sum(eir.CAL_PROPOSALS + st for _ in runs)
                    ill = sum(x["search"]["n_illegal"] + x["search"]["cal_illegal"] for x in runs)
                    c["levels"][f"{m}_{st}"] = {
                        "T": mean(x["record"]["T"] for x in runs), "mean": mean(p["mean"] for p in per), "median_run": statistics.median(p["mean"] for p in per),
                        "min": min(allv), "max": max(allv), "own_best": per[own]["mean"], "triple": [mean(p["triple"][i] for p in per) for i in range(3)],
                        "T_err": mean(pc(x["record"]["T"], p["mean"]) for x, p in zip(runs, per)),
                        "T_err_max": max(abs(pc(x["record"]["T"], p["mean"])) for x, p in zip(runs, per)),
                        "mig_start": {f: mg("migration_vs_start", f) for f in ("moved_behas_share", "ddr_read_moved_bytes", "unique_block_bytes")},
                        "mig_R0": {f: mg("migration_vs_R0", f) for f in ("moved_behas_share", "ddr_read_moved_bytes", "unique_block_bytes")},
                        "t_R0": tt("t_R0"), "t_L1": tt("t_L1"), "t_cal": tt("t_calibration"), "t_L2": tt("t_L2_search"), "t_fin": tt("t_finalize"),
                        "t_decision": tt("t_decision"), "t_backend": tt("t_backend_offline"), "cpcm_L2": mean(x["cpcm_calls"]["L2"] for x in runs),
                        "cpcm_L1": runs[0]["cpcm_calls"]["L1"], "Tmax": mean(x["search"]["Tmax"] for x in runs),
                        "revoked_share": ill / prop, "local": runs[0]["search"]["local"], "viol": sum(len(x["record"]["fault_violations"]) for x in runs),
                        "ctrl": ctrl.get((start, op, st), {}).get("mean"), "n": len(runs), "local_outside": (sum((x["search"]["local_audit"] or {}).get("beha_not_movable", 0) + (x["search"]["local_audit"] or {}).get("beha_dest_outside", 0) for x in runs))}
            c["stop_level"] = stop_level(c)
            res.append(c)
    return res


def stop_level(c):
    """Handoff section 6: first level whose T is within 2% of the fault-free T (later levels are still run)."""
    lim = c["T_pre"] * (1 + eir.EPS)
    for lev in ("R0", "L1"):
        if c["levels"][lev]["T"] <= lim:
            return lev
    for lev in c["levels"]:
        if lev.startswith("L2") and c["levels"][lev]["T"] <= lim:
            return lev
    return None


def f0(x):
    return "—" if x is None else f"{x:,.0f}"


def write_md(out, batch, res):
    L = [f"# EIR 批次报告：{batch}\n", "只有数字。延迟为后端 10 个种子均值；修复方法再对 5 个 SA 种子取平均，范围是 5 × 10 个值的最小与最大。\n"]
    L.append("## 每个故障的参照\n")
    L.append("| 起点 | 任务 | 故障 | T̂ 故障前 | 故障前后端 | FULL-BusyBarn（E3 状态）[范围] | 变体 | 第 2 阶段 A0 | FULL-A3h | L1 尝试 / 接受 / 结束原因 | R0 / L1 改路数 | 按规则可停的级别 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in res:
        fb, a0, a3 = c["FULL_BB"], c.get("FULL_A0_s2"), c.get("FULL_A3h")
        l1 = c["L1_run"]
        L.append(f"| {c['start']} | {c['op']} | {c['fault']} | {f0(c['T_pre'])} | {f0(c['pre']['mean'])} | {f0(fb['mean'])+' ['+f0(fb['min'])+', '+f0(fb['max'])+']' if fb else '—'} | {c['FULL_BB_variant'] or '—'} | "
                 f"{f0(a0['mean']) if a0 else '—'} | {f0(a3['mean']) if a3 else '—'} | {l1['attempts']} / {l1['accepted']} / {l1['reason']} | {c['n_changed_routes_R0']} / {c['n_changed_routes_L1']} | {c['stop_level'] or '无'} |")
    L.append("\n## 延迟与改进\n")
    if batch == "afault":
        L.append("FULL-A3h 是第 2 阶段在 E3 的 F4、F5 位置上的结果，这里的故障位置不同，只作参照。净代价 = 修复后延迟 − 同起点同步数的无故障继续搜索延迟。\n")
    L.append("| 任务 | 故障 | 方法 | 后端 | 范围 | 自身代价最优 | 三元组 | 相对故障前 | 相对 FULL-BB | 相对 FULL-A3h | 相对上一级 | 净代价 | T̂ | T̂ 对后端误差 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in res:
        ref = c["FULL_BB"]["mean"] if c["FULL_BB"] else None
        a3 = c.get("FULL_A3h")
        for lev, x in c["levels"].items():
            prev = None if lev == "R0" else c["levels"]["R0"]["mean"] if lev == "L1" else c["levels"]["L1"]["mean"]
            L.append(f"| {c['op']} | {c['fault']} | {lev} | {f0(x['mean'])} | [{f0(x['min'])}, {f0(x['max'])}] | {f0(x.get('own_best'))} | {' / '.join(f0(v) for v in x['triple'])} | {pc(x['mean'], c['pre']['mean']):+.1f}% | "
                     f"{'—' if ref is None else format(pc(x['mean'], ref), '+.1f') + '%'} | {'—' if not a3 else format(pc(x['mean'], a3['mean']), '+.1f') + '%'} | {'—' if prev is None else format(pc(x['mean'], prev), '+.1f') + '%'} | {'—' if not x.get('ctrl') else f0(x['mean'] - x['ctrl']) + ' (' + format(pc(x['mean'], x['ctrl']), '+.1f') + '%)'} | {f0(x['T'])} | {x['T_err']:+.2f}% |")
    L.append("\n## 迁移量（相对起点 / 相对 R0）\n")
    L.append("| 任务 | 故障 | 方法 | 行为位置变化比例 | DDR 读源重定位字节 | 唯一权重块字节 |\n|---|---|---|---|---|---|")
    for c in res:
        for lev, x in c["levels"].items():
            s, r = x["mig_start"], x["mig_R0"] or {"moved_behas_share": 0, "ddr_read_moved_bytes": 0, "unique_block_bytes": 0}
            L.append(f"| {c['op']} | {c['fault']} | {lev} | {100 * s['moved_behas_share']:.1f}% / {100 * r['moved_behas_share']:.1f}% | {f0(s['ddr_read_moved_bytes'])} / {f0(r['ddr_read_moved_bytes'])} | {f0(s['unique_block_bytes'])} / {f0(r['unique_block_bytes'])} |")
    L.append("\n## 决策计时（秒，含 14 个并行的争用）与局部集合\n")
    L.append("| 任务 | 故障 | 方法 | t_R0 | t_L1 | t_标定 | t_L2 | t_收尾 | t_决策 | 后端（离线） | CPCM 调用 L1 / L2 | Tmax | 行为覆盖 | 核覆盖 | 提议被撤销比例 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in res:
        for lev, x in c["levels"].items():
            lc = x.get("local")
            L.append(f"| {c['op']} | {c['fault']} | {lev} | {x['t_R0']:.1f} | {x['t_L1']:.1f} | {x['t_cal']:.1f} | {x['t_L2']:.1f} | {x['t_fin']:.2f} | {x['t_decision']:.1f} | {x['t_backend']:.0f} | {x.get('cpcm_L1', '—')} / {f0(x.get('cpcm_L2'))} | "
                     f"{'—' if 'Tmax' not in x else format(x['Tmax'], '.2e')} | {'—' if not lc else format(100 * lc['beha_share'], '.0f') + '%'} | {'—' if not lc else format(100 * lc['core_share'], '.0f') + '%'} | {'—' if not lc else format(100 * x['revoked_share'], '.1f') + '%'} |")
    if batch == "a3h":
        L.append("\n## 故障链路在故障前 A3h 映射上的状态（2.2）\n")
        L.append("| 任务 | 故障 | 链路 | 忙碌率 | CB | CB 排名 / 有 CB 的链路数 |\n|---|---|---|---|---|---|")
        for c in res:
            for ls in c["link_stats"]:
                L.append(f"| {c['op']} | {c['fault']} | `{ls['link']}` | {100 * ls['busy_share']:.1f}% | {f0(ls['cb'])} | {ls['cb_rank'] or '无 CB'} / {ls['n_links_with_cb']} |")
    n = sum(c["n_runs"] for c in res)
    v = sum(x["viol"] for c in res for x in c["levels"].values())
    o = sum(x.get("local_outside", 0) for c in res for x in c["levels"].values())
    L.append(f"\n自检：修复运行 {n} 次；故障违规 {v}；局部审计中行为或落点在集合外 {o}。\n")
    p = os.path.join(out, f"report_{batch}.md")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    with open(os.path.join(out, f"report_{batch}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=0, default=str)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", choices=sorted(BATCH) + ["ctrl"])
    ap.add_argument("--out", default=os.path.join(eir.HERE, "out", "main"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if a.batch == "ctrl":
        print(write_ctrl(out))
        return
    print(write_md(out, a.batch, collect(out, a.batch)))


if __name__ == "__main__":
    main()
