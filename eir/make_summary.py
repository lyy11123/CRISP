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


# ---------------------------------------------------------------- final stop rule, checked offline on the existing runs
EPS, TIER_GAIN, OLD_GAIN = 0.02, 0.02, 0.02
TIERS = (30, 100, 300)


def final_rule(c, m):
    """Across levels: go on until T is within EPS of the fault-free T. Inside L2, tiers 30 -> 100 -> 300: stop within
    EPS, or when the adjacent tiers' T improves by less than TIER_GAIN. Returns (level, reason). T = mean over SA seeds."""
    lv, lim = c["levels"], c["T_pre"] * (1 + EPS)
    for k in ("R0", "L1"):
        if lv[k]["T"] <= lim:
            return k, "进 ε"
    tiers = [s for s in TIERS if f"{m}_{s}" in lv]
    prev = None
    for s in tiers:
        T = lv[f"{m}_{s}"]["T"]
        if T <= lim:
            return f"{m}_{s}", "进 ε"
        if prev is not None and (prev - T) / prev < TIER_GAIN:
            return f"{m}_{s}", f"档改进 {100 * (prev - T) / prev:+.1f}%"
        prev = T
    return f"{m}_{tiers[-1]}", "预算用尽"


def old_rule(c, m):
    """Abolished rule on the chain R0 -> L1 -> L2 s=30 -> 100 -> 300: stay at a level when the next one improves T by
    less than OLD_GAIN."""
    lv = c["levels"]
    chain = ["R0", "L1"] + [f"{m}_{s}" for s in TIERS if f"{m}_{s}" in lv]
    for a_, b_ in zip(chain, chain[1:]):
        if (lv[a_]["T"] - lv[b_]["T"]) / lv[a_]["T"] < OLD_GAIN:
            return a_
    return chain[-1]


def rule_rows(r):
    rows = []
    for c in r:
        lv, pre = c["levels"], c["pre"]["mean"]
        top = max(s for s in TIERS if f"L2G_{s}" in lv)
        for m in ("L2G", "L2L"):
            k, why = final_rule(c, m)
            o = old_rule(c, m)
            rows.append({"c": c, "m": m, "k": k, "why": why, "o": o, "top": top, "T": lv[k]["T"], "b": lv[k]["mean"], "rel": pc(lv[k]["mean"], pre),
                         "ob": lv[o]["mean"], "orel": pc(lv[o]["mean"], pre)})
    return rows


def rule_table(rows):
    L = ["| 任务 | 故障 | 路径 | T̂ 故障前 | 后端故障前 | 选中 | 停止原因 | T̂ | 后端 | 相对故障前 | 最高档 | 旧规则停在 | 旧规则后端 | 旧规则相对故障前 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        c = x["c"]
        L.append(f"| {c['op']} | {c['fault']} | {x['m']} | {n0(c['T_pre'])} | {n0(c['pre']['mean'])} | {NAME(x['k'])} | {x['why']} | {n0(x['T'])} | {n0(x['b'])} | {x['rel']:+.1f}% | s={x['top']} | {NAME(x['o'])} | {n0(x['ob'])} | {x['orel']:+.1f}% |")
    return "\n".join(L)


def rule_counts(rows):
    n = len(rows)
    diff = [x for x in rows if x["k"] != x["o"]]
    worse = [x for x in rows if x["ob"] > x["b"]]
    outs = [x for x in rows if x["why"] != "进 ε"]
    gaps = [pc(x["ob"], x["b"]) for x in rows]
    inside = [x for x in rows if x["why"] == "进 ε"]
    over = [x for x in inside if x["rel"] > 100 * EPS]
    return (f"T̂ 进 ε 的 {len(inside)} 条里，后端相对故障前仍超过 2% 的 {len(over)} 条（T̂ 对后端有偏差，进 ε 不保证后端也在 2% 以内）。"
f"共 {n} 条路径（{n // 2} 个组合 × L2G、L2L）。两条规则选的级别不同的 {len(diff)} 条；旧规则的后端延迟高于定稿规则的 {len(worse)} 条。"
            f"旧规则相对定稿规则的后端差值：中位 {med(gaps):+.1f}%，最小 {min(gaps):+.1f}%，最大 {max(gaps):+.1f}%。"
            f"定稿规则没有进 ε 就停下（档改进不足或预算用尽）的 {len(outs)} 条：" + ("、".join(f"{x['c']['op']} {x['c']['fault']} {x['m']}" for x in outs) or "无") + "。")


def stop_rule_section(a3, af, ff, mh, f14):
    ra, rb = rule_rows(a3 + af), rule_rows(ff + mh + f14)
    ex = next(x for x in ra if x["c"]["op"] == "ffn" and x["c"]["fault"] == "F5" and x["m"] == "L2G")
    return "\n".join([
        "## 8. 定稿停止规则（用户 2026-11-05 拍板，只用已有数据离线检验）", "",
        "**规则**", "",
        "- **跨级**：T̂ 回到故障前 ε = 2% 以内才停，否则继续升级。L1 改进小或为 0 不是停止理由。",
        "- **L2 内部**：按步数档 30 → 100 → 300 判断，已进 ε，或相邻两档 T̂ 改进 < 2% 即停。",
        "- **废除**“相邻两级改进 < 2% 即停”的跨级规则。", "",
        "**检验方法与局限**", "",
        "- 现有 s = 30、100、300 是三次独立降温的运行（`search()` 里温度含 `step / steps`），这里是把它们当独立预算离线检验，不是在线连续检查点。所以相邻档的 T̂ 可能反而上升（“档改进”为负）；在线检查点里最优值单调，不会出现。",
        "- 判停都用 T̂（5 个 SA 种子的均值，R0、L1 是单次）；后端延迟只用来看结果，判停时看不到。实际部署只有一次搜索，单次的判停结果可能与这里的均值不同。",
        "- “档改进 < 2%”时选中的是刚跑完的这一档，不回退到上一档。",
        "- 旧规则的算法：链 R0 → L1 → L2 s=30 → s=100 → s=300，下一级 T̂ 改进 < 2% 就停在当前级。把步数档也算作“级”，这对旧规则是有利的算法。",
        "- 没有 s = 300 的组合最高只到 s = 100（表中“最高档”列）：这些组合“预算用尽”不代表 s = 300 也停不下来。", "",
        f"**为什么废除**：例如 A3h FFN F5，L1 没有候选路径（改进 0），旧规则停在 {NAME(ex['o'])}，后端 {n0(ex['ob'])}，相对故障前 {ex['orel']:+.1f}%；"
        f"L2G s=100 的后端 {n0(ex['c']['levels']['L2G_100']['mean'])}，{pc(ex['c']['levels']['L2G_100']['mean'], ex['c']['pre']['mean']):+.1f}%；定稿规则选 {NAME(ex['k'])}，{ex['rel']:+.1f}%。", "",
        "### 8.1 A3h 起点（F4、F5、F4a、F5a）", "", rule_table(ra), "", rule_counts(ra), "",
        "### 8.2 BusyBarn 起点（E3 位置）", "", rule_table(rb), "", rule_counts(rb), ""])


a3, af, ff, mh, f14 = load("a3h"), load("afault"), load("ffn"), load("mha"), load("ffn14")
if "--write-rule" in sys.argv:
    # replaces the section between the markers of EIR_SUMMARY.md; the rest of that file is edited by hand
    p = os.path.join(HERE, "EIR_SUMMARY.md")
    with open(p, encoding="utf-8") as f:
        s = f.read()
    b, e = "<!-- STOP_RULE_BEGIN -->\n", "<!-- STOP_RULE_END -->"
    if s.count(b) != 1 or s.count(e) != 1:
        raise SystemExit("markers not found in EIR_SUMMARY.md")
    s = s[:s.index(b) + len(b)] + stop_rule_section(a3, af, ff, mh, f14) + s[s.index(e):]
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    raise SystemExit(0)
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
