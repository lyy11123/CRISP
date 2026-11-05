"""Tables of EIR_SUMMARY.md from out/main/report_*.json (handoff 12.3). Prints markdown, the text around it is hand-written:
  python make_summary.py > out/main/summary_tables.md
"""
import ast
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "out", "main")
sys.stdout.reconfigure(encoding="utf-8", newline="\n")


def load(b):
    with open(os.path.join(M, f"report_{b}.json"), encoding="utf-8") as f:
        return json.load(f)


with open(os.path.join(M, "report_ctrl.json"), encoding="utf-8") as f:
    CTRL = {ast.literal_eval(k): v for k, v in json.load(f).items()}
med = st.median
n0 = lambda x: f"{x:,.0f}"
pc = lambda a, b: 100 * (a / b - 1)
LV = ["R0", "L1"] + [f"{m}_{s}" for m in ("W", "L2G", "L2L") for s in (30, 100, 300)]
NAME = lambda k: k.replace("_", " s=")


def mig(x):
    m = x["mig_start"]
    return f"{100 * m['moved_behas_share']:.1f}% | {m['ddr_read_moved_bytes'] / 1e6:.1f} | {m['unique_block_bytes'] / 1e6:.1f}"


def main_table(c, ref_note=""):
    """One start x task x fault, as in handoff 12.3."""
    pre = c["pre"]["mean"]
    print(f"\n**{c['op']} {c['fault']}**（故障前后端 {n0(pre)}）\n")
    print("| 方案 | 后端 | 范围 | 相对故障前 | 净代价 | 行为搬动 | DDR 读源 MB | 唯一块 MB | t_决策 s |\n|---|---|---|---|---|---|---|---|---|")
    print(f"| 故障前部署（基准） | {n0(pre)} | [{n0(c['pre']['min'])}, {n0(c['pre']['max'])}] | — | — | — | — | — | — |")
    for k in LV:
        x = c["levels"].get(k)
        if not x:
            continue
        net = "—"
        if x.get("ctrl"):
            d = x["mean"] - x["ctrl"]
            net = f"{d:+,.0f}（{pc(x['mean'], x['ctrl']):+.1f}%）"
        print(f"| {NAME(k)} | {n0(x['mean'])} | [{n0(x['min'])}, {n0(x['max'])}] | {pc(x['mean'], pre):+.1f}% | {net} | {mig(x)} | {x['t_decision']:.1f} |")
    for s in (30, 100, 300):
        y = CTRL.get((c["start"], c["op"], s))
        if y:
            g = y["mig"]
            print(f"| 无故障继续搜索 s={s} | {n0(y['mean'])} | [{n0(y['min'])}, {n0(y['max'])}] | {pc(y['mean'], pre):+.1f}% | — | {100 * g['moved_behas_share']:.1f}% | {g['ddr_read_moved_bytes'] / 1e6:.1f} | {g['unique_block_bytes'] / 1e6:.1f} | {y['t_decision']:.1f} |")
    f = c.get("FULL_A3h")
    if f:
        print(f"| FULL-A3h（延迟参照{ref_note}） | {n0(f['mean'])} | [{n0(f['min'])}, {n0(f['max'])}] | {pc(f['mean'], pre):+.1f}% | — | — | — | — | — |")
    print(f"\n可停级别（T̂ 回到故障前 2% 以内）：{NAME(c['stop_level']) if c['stop_level'] else '无'}。")


def median_table(r, ref=True):
    print("\n| 级别 | n | 相对故障前 | " + ("相对 FULL-BB 中位（最小，最大） | " if ref else "") + "相对 L1 | 净代价 | 行为搬动 | DDR 读源 MB | 唯一块 MB | 相对 R0 搬动 | t_决策 s | T̂ 误差 | 撤销 | 局部覆盖 行为 / 核 |")
    print("|---|---|---|" + ("---|" if ref else "") + "---|---|---|---|---|---|---|---|---|---|")
    for k in LV:
        cs = [c for c in r if k in c["levels"]]
        if not cs:
            continue
        xs = [c["levels"][k] for c in cs]
        p = med(pc(c["levels"][k]["mean"], c["pre"]["mean"]) for c in cs)
        g = [pc(c["levels"][k]["mean"], c["FULL_BB"]["mean"]) for c in cs] if ref else []
        up = "—" if k == "R0" else f"{med(pc(c['levels'][k]['mean'], c['levels']['R0' if k == 'L1' else 'L1']['mean']) for c in cs):+.1f}%"
        nets = [pc(x["mean"], x["ctrl"]) for x in xs if x.get("ctrl")]
        loc = [x["local"] for x in xs if x.get("local")]
        print(f"| {NAME(k)} | {len(cs)} | {p:+.1f}% | " + (f"{med(g):+.1f}%（{min(g):+.1f}，{max(g):+.1f}） | " if ref else "") + f"{up} | {f'{med(nets):+.1f}%' if nets else '—'} | "
              f"{100 * med(x['mig_start']['moved_behas_share'] for x in xs):.1f}% | {med(x['mig_start']['ddr_read_moved_bytes'] for x in xs) / 1e6:.1f} | {med(x['mig_start']['unique_block_bytes'] for x in xs) / 1e6:.1f} | "
              f"{(f"{100 * med(x['mig_R0']['moved_behas_share'] for x in xs):.1f}%" if xs[0]['mig_R0'] else '—')} | {med(x['t_decision'] for x in xs):.1f} | {med(x['T_err'] for x in xs):+.2f}% | "
              f"{f'{100 * med(x['revoked_share'] for x in xs):.1f}%' if loc else '—'} | {f'{100 * med(l['beha_share'] for l in loc):.0f}% / {100 * med(l['core_share'] for l in loc):.0f}%' if loc else '—'} |")


def fault_table(r, ref=True):
    keys = ["R0", "L1", "L2G_100", "L2L_100"]
    print("\n| 任务 | 故障 | 故障前 | " + " | ".join(NAME(k) for k in keys) + " | L2G s=300 | L2L s=300 | " + ("FULL-BB | " if ref else "") + "FULL-A3h | 可停级别 |")
    print("|---|---|---|" + "---|" * len(keys) + "---|---|" + ("---|" if ref else "") + "---|---|")
    for c in r:
        l = c["levels"]
        v = lambda k: n0(l[k]["mean"]) if k in l else "—"
        print(f"| {c['op']} | {c['fault']} | {n0(c['pre']['mean'])} | " + " | ".join(v(k) for k in keys) + f" | {v('L2G_300')} | {v('L2L_300')} | " + (f"{n0(c['FULL_BB']['mean'])} | " if ref else "") +
              f"{n0(c['FULL_A3h']['mean']) if c.get('FULL_A3h') and c['fault'] in ('F4', 'F5') else '—'} | {NAME(c['stop_level']) if c['stop_level'] else '无'} |")


def wins(r):
    for k in ("L1", "L2G_100", "L2L_100"):
        print(f"- {NAME(k)} 不超过 FULL-BB：{sum(c['levels'][k]['mean'] <= c['FULL_BB']['mean'] for c in r)} / {len(r)}。")


def link_table(r):
    print("\n| 任务 | 故障 | 链路 | 忙碌率（CPCM） | CB 排名 / 有 CB 的链路数 |\n|---|---|---|---|---|")
    for c in r:
        for s in c["link_stats"]:
            print(f"| {c['op']} | {c['fault']} | `{s['link']}` | {100 * s['busy_share']:.1f}% | {s['cb_rank'] or '无 CB'} / {s['n_links_with_cb']} |")


def ctrl_table():
    print("\n| 起点 | 任务 | s | 故障前 | 后端 | 相对故障前 | 行为搬动 | DDR 读源 MB | 唯一块 MB | t_决策 s | T̂ 误差 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    pres = {(c["start"], c["op"]): c["pre"]["mean"] for b in ("a3h", "ffn", "mha", "ffn14") for c in load(b)}
    for (a, o, s), y in CTRL.items():
        p = pres[(a, o)]
        g = y["mig"]
        print(f"| {a} | {o} | {s} | {n0(p)} | {n0(y['mean'])} | {pc(y['mean'], p):+.1f}% | {100 * g['moved_behas_share']:.1f}% | {g['ddr_read_moved_bytes'] / 1e6:.1f} | {g['unique_block_bytes'] / 1e6:.1f} | {y['t_decision']:.1f} | {y['T_err']:+.2f}% |")


a3, af, ff, mh, f14 = load("a3h"), load("afault"), load("ffn"), load("mha"), load("ffn14")
print("<!-- A3H_MAIN -->")
for c in a3:
    main_table(c)
print("\n<!-- A3H_LINK -->")
link_table(a3)
print("\n<!-- AF_MAIN -->")
for c in af:
    main_table(c, "，故障位置不同，是 F" + c["fault"][1] + " 的数")
print("\n<!-- AF_LINK -->")
link_table(af)
print("\n<!-- MED_A3H -->")
median_table(a3)
print("\n<!-- MED_AF -->")
median_table(af, ref=False)
for name, r in (("FFN", ff), ("MHA", mh), ("FFN14", f14)):
    print(f"\n<!-- MED_{name} -->")
    median_table(r)
    fault_table(r)
    print()
    wins(r)
print("\n<!-- CTRL -->")
ctrl_table()
print("\n<!-- CHECK -->")
allr = a3 + af + ff + mh + f14
print("viol", sum(x["viol"] for c in allr for x in c["levels"].values()), "outside", sum(x.get("local_outside") or 0 for c in allr for x in c["levels"].values()), "ctrl viol", sum(y["viol"] for y in CTRL.values()))
print("revoked range", min(x["revoked_share"] for c in allr for k, x in c["levels"].items() if k.startswith("L2L")), max(x["revoked_share"] for c in allr for k, x in c["levels"].items() if k.startswith("L2L")))
cov = [x["local"]["beha_share"] for c in allr for k, x in c["levels"].items() if k.startswith("L2L_30")]
print("coverage", min(cov), med(cov), max(cov), sum(0.2 <= v <= 0.4 for v in cov), len(cov))
print("Terr abs range by method")
for m in ("R0", "L1", "W", "L2G", "L2L"):
    v = [x["T_err"] for c in allr for k, x in c["levels"].items() if k.split("_")[0] == m]
    print(m, f"{min(v):+.2f} {med(v):+.2f} {max(v):+.2f}")
