"""Added after the phase B rule was fixed (事后追加, not part of PROGRESS step 25's rule). Analytic KV estimate, its sensitivity on the
window decision, T̂-versus-backend counts and FULL's three readings. Replays stored data, no new runs. Appends to out/next/report_b.md after
the marker (replaces what was there), writes out/next/report_kv.json. Run report_b.py first.

  python report_kv.py

Model assumption: the artifact has no KV state (MHA_kconv, MHA_vconv are computed from ifmap every round). B_KV = rho x M_KV, T_KV = B_KV / 256
cycles, rho = moved_behas_share of the MHA task against the fault-free mapping. This is an external estimate, not a simulation result.
"""
import json
import os

import eir
import eir_next as en
import report_b as rb
import report_next as rn

MARK = "<!-- 事后追加：report_kv.py -->"
MiB = 2 ** 20
T_MHA, T_FFN = None, None
# label, batch, sq, hidden, layer bytes (or None), task bytes, note
TIERS = [
    {"name": "用户规格档（单层）", "dims": "batch 1，sq 2048，hidden 4096，2 B（规格示意，与 artifact 配置不同）", "layer": 2 * 2048 * 4096 * 2, "task": 2 * 2048 * 4096 * 2 / 8},
    {"name": "artifact 配置（主值）", "dims": "batch 16，sq 2048，hidden 6144，bf16 2 B", "layer": 2 * 16 * 2048 * 6144 * 2, "task": 2 * 16 * 2048 * 6144 * 2 / 8},
    {"name": "压力 1 GiB 作单层", "dims": "无对应配置；batch 16、hidden 6144 时 sq ≈ 2,731", "layer": 2 ** 30, "task": 2 ** 30 / 8},
    {"name": "压力 2 GiB 作单层", "dims": "无对应配置；batch 16、hidden 6144 时 sq ≈ 5,461", "layer": 2 ** 31, "task": 2 ** 31 / 8},
    {"name": "压力 1 GiB 作单任务", "dims": "无对应配置；单任务 8 头时 sq ≈ 21,845", "layer": 2 ** 33, "task": 2 ** 30},
    {"name": "压力 2 GiB 作单任务", "dims": "无对应配置；单任务 8 头时 sq ≈ 43,691", "layer": 2 ** 34, "task": 2 ** 31},
]
SINGLE = ("F4a", "F5", "F8a")
K4 = en.C_FAULTS
METHODS = en.B_METHODS
LABEL = rb.LABEL


def med(x):
    return rb.med(x)


def share_rows(pre_loc, op, fid):
    """rho of R0, L1 and of the three L2 methods at s = 100."""
    fold = en.l1_out(fid) if fid in ("F4a", "F5", "F8a") else en.OUT
    l = eir.read_sink(en.job_l1(op, fid, out=fold)["sink"])
    out = {"R0": [rb.moved_share(pre_loc, l["R0"]["beha_loc"])], "L1": [rb.moved_share(pre_loc, l["L1"]["beha_loc"])]}
    for m in METHODS:
        v = []
        for sd in en.ALL_SEEDS:
            p = en.job_rep(op, fid, m, sd)["sink"]
            if os.path.exists(p):
                v.append(eir.read_sink(p)["record"]["migration_vs_start"]["moved_behas_share"])
        out[m] = v
    return out


def full_rho(pre_loc, op, fid):
    """rho measured on FULL-A3h's own final mappings (selected e3map); None when the stored run has no mapping."""
    rows = []
    for g, x in eir.cs.variants("A3h"):
        if fid == "F5":
            s = eir.cs.load_run(f"{op}_f5", "A3h", g, x, eir.S2_OUT)["selected"]
        else:
            s = eir.read_sink(eir.job_full(en.OUT, op, fid, "A3h", g, x)["sink"])["selected"]
        if "e3map" not in s:
            return None
        rows.append({"rho": rb.moved_share(pre_loc, s["e3map"]["beha_loc"]), "T": s["T_sel"], "own": s["own_cost"]})
    pick = min(rows, key=lambda r: (r["T"], r["own"]))
    return {"by_T": pick["rho"], "median": med(r["rho"] for r in rows), "min": min(r["rho"] for r in rows), "max": max(r["rho"] for r in rows)}


def cyc(rho, M):
    return rho * M / 256


def rng(v, f="{:.1f}%", k=100):
    if not v:
        return "—"
    return f"{f.format(med(v) * k)}（{f.format(min(v) * k)}–{f.format(max(v) * k)}）"


def main():
    D = rb.load()
    res = {}
    pre = eir.read_sink(eir.job_pre(en.MAIN, "a3h", "mha")["sink"])["record"]
    pre_loc = pre["beha_loc"]
    T_M, T_F = D["pre"]["mha"]["bk"], D["pre"]["ffn"]["bk"]
    faults = eir.load_faults()["mha"]
    L = [MARK, "\n# 事后追加（不属于第 25 步开跑前写定的规则）\n"]
    # ---- rho
    L.append("## KV 状态迁移解析估算\n")
    L.append("**这是外部解析估算，不是仿真结果。** artifact 没有 KV 状态（`MHA_kconv`、`MHA_vconv` 每轮从 ifmap 算），公式是模型假设：`B_KV = ρ_MHA × M_KV`，`T_KV = B_KV / 256` 周期（与 T_copy 同口径，单端口，无竞争）。ρ 是 MHA 任务的 `moved_behas_share`，相对故障前映射，按行为个数平均，隐含 KV 均匀分布在行为上。\n")
    L.append("### ρ_MHA（实测）\n")
    L.append("L2 取种子中位数（范围）。FULL 用 `selected.e3map` 的实际最终映射实测，不是 E3 的 97–99%。“相对 FULL 减少”= 1 − ρ_L2 / ρ_FULL按T̂自选。\n")
    L.append("| 故障 | 档 | R0 | L1 | L2L+sync | L2Impact | L2G+sync | FULL 按 T̂ 自选 | FULL 14 次中位数（范围） | L2L+sync 相对 FULL 减少 | L2Impact 相对 FULL 减少 | L2G+sync 相对 FULL 减少 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    rho = {}
    for fid in SINGLE + K4:
        x = share_rows(pre_loc, "mha", fid)
        fu = full_rho(pre_loc, "mha", fid)
        rho[fid] = {**x, "FULL": fu}
        fs = "—" if not fu else f"{fu['by_T'] * 100:.1f}%"
        fm = "没有保存映射" if not fu else f"{fu['median'] * 100:.1f}%（{fu['min'] * 100:.1f}–{fu['max'] * 100:.1f}）"
        red = [("—" if not fu or not x[m] else f"{(1 - med(x[m]) / fu['by_T']) * 100:.0f}%") for m in METHODS]
        L.append(f"| {fid} | {'k=4' if fid in K4 else '单故障'} s=100 | {x['R0'][0] * 100:.1f}% | {x['L1'][0] * 100:.1f}% | " + " | ".join(rng(x[m]) for m in METHODS) + f" | {fs} | {fm} | " + " | ".join(red) + " |")
    # checkpoints
    ck = {}
    for m in METHODS:
        for i, s in enumerate((30, 100, 300)):
            ck[f"{m}_{s}"] = [D["ck"]["mha", m, sd]["ck"][i]["migration_vs_start"]["moved_behas_share"] for sd in en.B_SEEDS]
    fu = rho["F4a"]["FULL"]
    for s in (30, 100, 300):
        red = [("—" if not fu else f"{(1 - med(ck[f'{m}_{s}']) / fu['by_T']) * 100:.0f}%") for m in METHODS]
        L.append(f"| F4a 检查点 | 单故障 s={s} | — | — | " + " | ".join(rng(ck[f"{m}_{s}"]) for m in METHODS) + " | — | — | " + " | ".join(red) + " |")
    main_rows = {}
    for s in (100, 300):
        v = []
        for fid in ("F4", "F5", "F4a", "F5a"):
            for sd in eir.SA_SEEDS:
                p = eir.job_rep(en.MAIN, "a3h", "mha", fid, "L2G", s, sd)["sink"]
                if os.path.exists(p):
                    v.append(eir.read_sink(p)["record"]["migration_vs_start"]["moved_behas_share"])
        main_rows[s] = v
        if v:
            L.append(f"\n主实验 MHA L2G（不带 sync）s={s}：{min(v) * 100:.1f}–{max(v) * 100:.1f}%，中位数 {med(v) * 100:.1f}%（{len(v)} 次）。")
    res["rho"] = {k: {a: b for a, b in v.items()} for k, v in rho.items()}
    res["rho_ck"] = ck
    res["rho_main_L2G"] = main_rows
    # ---- tiers
    L.append("\n### M_KV 档位\n")
    L.append("单层总量对应 64 头；`mha.py` 的 `head_num = 64`、循环 `heads = 8`，一个 MHA 任务只建 8 个头，`end_to_end.py` 的 block 里是 `mha × 8`。ρ 是单任务上测的，所以公式里用单任务列，主值 96 MiB。\n")
    L.append("| 档 | 配置 | 单层总量（字节） | 单任务（字节） | 单任务 MiB | 对一个 2×2 组 1 GiB 的占比 | 放得下 |\n|---|---|---|---|---|---|---|")
    for t in TIERS:
        share = t["task"] / 2 ** 30
        fit = "超过，KV 只能在 DDR，ρ 公式不适用" if share > 1 else ("正好等于总容量，没有余量放别的数据，实际放不下" if share == 1 else "放得下（容量上）")
        L.append(f"| {t['name']} | {t['dims']} | {t['layer']:,.0f} | {t['task']:,.0f} | {t['task'] / MiB:,.0f} | {share * 100:.1f}% | {fit} |")
    L.append("\n### T_KV\n")
    L.append("轮数按故障前后端折算：T_MHA = " + rb.f0(T_M) + "、T_FFN = " + rb.f0(T_F) + " 周期（与时钟无关）。ρ 取中位数；每列是周期（0.5 ns 的 µs / 1 ns 的 µs；MHA 轮 / FFN 轮）。\n")
    head = "| 故障 | 方案 | ρ | " + " | ".join(f"{t['task'] / MiB:,.0f} MiB" for t in TIERS) + " |\n|---|---|---|" + "---|" * len(TIERS)
    L.append(head)

    def row(fid, name, r):
        cells = []
        for t in TIERS:
            c = cyc(r, t["task"])
            cells.append(f"{c:,.0f}（{c * 0.5e-3:,.1f} / {c * 1e-3:,.1f} µs；{c / T_M:.1f} / {c / T_F:.1f}）")
        L.append(f"| {fid} | {name} | {r * 100:.1f}% | " + " | ".join(cells) + " |")

    kv_rows = {}
    for fid in SINGLE + K4:
        x = rho[fid]
        for name, v in (("R0", x["R0"]), ("L1", x["L1"]), *((f"{LABEL[m]} s=100", x[m]) for m in METHODS)):
            if v:
                row(fid, name, med(v))
                kv_rows[f"{fid}_{name}"] = {"rho": med(v), "cycles": {t["name"]: cyc(med(v), t["task"]) for t in TIERS}}
        if x["FULL"]:
            row(fid, "FULL 按 T̂ 自选", x["FULL"]["by_T"])
            row(fid, "FULL 中位数", x["FULL"]["median"])
            kv_rows[f"{fid}_FULL"] = {"rho": x["FULL"]["by_T"], "cycles": {t["name"]: cyc(x["FULL"]["by_T"], t["task"]) for t in TIERS}}
    for m in METHODS:
        row("F4a 检查点", f"{LABEL[m]} s=300", med(ck[f"{m}_300"]))
    res["kv"] = kv_rows
    # ---- behaviours on failed elements
    L.append("\n### 坏元件上的 KV（公式只算拷贝，这部分拷不出来，含 PE 故障时 T_KV 是下界）\n")
    L.append("坏 PE 或坏节点上的 KV 只能重建或依赖副本，这里只列字节，不估重建时间。份额 = 故障前映射里落在失效核上的 MHA 行为占比。DDR 端口节点上没有行为。\n")
    L.append("| 故障 | 失效核 | 行为数 / 总数 | 份额 | " + " | ".join(f"{t['task'] / MiB:,.0f} MiB" for t in TIERS) + " | 可用 SRAM（核 × 16 MiB） |\n|---|---|---|---|" + "---|" * len(TIERS) + "---|")
    badr = {}
    for fid in SINGLE + K4:
        f = faults[fid]
        bad = {tuple(x) for x in f.get("failed_pes", [])} | {tuple(x) for x in f.get("failed_nodes", []) if x[1] < 16}
        n = sum(tuple(v[:2]) in bad for v in pre_loc.values())
        sh = n / len(pre_loc)
        badr[fid] = {"n": n, "total": len(pre_loc), "share": sh, "cores": sorted(bad)}
        L.append(f"| {fid} | {sorted(bad) or '无'} | {n} / {len(pre_loc)} | {sh * 100:.1f}% | " + " | ".join(f"{sh * t['task'] / MiB:,.1f} MiB" for t in TIERS) + f" | {64 - len(bad)} × 16 MiB = {(64 - len(bad)) * 16:,} MiB |")
    res["bad"] = badr
    L.append("\n### 表下说明\n")
    L.append("- **SRAM 核实**：`wamis_hd_round.cfg` 有 `[sram_cfg]`，`sram_capacity = 16,777,216`，`sram_number = 1`；`planar_2d.py` 约第 148 行在每个节点循环里按 `sram_number` 建 SRAM，所以每核一个。`mem_enable` 由 `platform_cfg` 里有没有 `sram_cfg` 决定（第 82–89 行），这份 cfg 下为真。这只建了容量跟踪的器件，这两个任务图里没看到 SRAM 事件。64 核共 1 GiB。")
    L.append("- 比较口径：单任务 M_KV 对一个 2×2 组的 1 GiB。96、128、256 MiB 在容量上放得下，但 SRAM 还要放别的数据，所以占比只是上限判断。单任务 2 GiB 超过容量；单任务 1 GiB 等于总容量，没有余量。这两种情况 KV 只能在 DDR，“KV 随行为驻留”不成立，ρ 公式不适用，只有读路径变化。")
    L.append("- **8 个头是否对应 8 个相互独立的 Die 组：模板不能确认，是假设。** 模板里 `chiplets_per_layer = 1`，只映射一个组；`end_to_end.py` 只是把 `mha[0] * 8` 加起来，没有说明放在哪里（直接相加更像依次占用，这是推测）。如果 8 个任务依次用同一个组，这个组要存 8 × 96 = 768 MiB，占 1 GiB 的 75%，比单任务的 9.4% 紧得多。")
    L.append("- L1 的 ρ 不总是 0：纯链路故障为 0，含 PE 故障时不为 0，因为坏 PE 上的行为在 R0 就搬走了；以上表为准。")
    # ---- sensitivity
    L.append("\n### 敏感性：把 T_KV 加进窗口决策的 C\n")
    L.append("网格：第 25 步的 H（5 档）、r（2）、时钟（2）、搜索耗时倍率（2），加版本（主用、停机）、方法（3）、种子（10），共 2,400 格（T_switch = 0，T_交接 = 0），只有 F4a 有检查点。ρ 是该种子该档的实测值，只改 C，时间轴和窗口规则不变。")
    tiers = rb.tier_table(D)
    sens = {}
    total = 0
    L.append("\n| 单任务 M_KV | 决策改变的格数 / 总格数 | 窗口决策累计完成量最大相对变化 | 固定 ε 最大相对变化 | 最大变化所在格 |\n|---|---|---|---|---|")
    for t in sorted(TIERS, key=lambda x: x["task"]):
        changed, mw, mf, where, cells, details = 0, 0.0, 0.0, None, 0, []
        for m in METHODS:
            for version in ("main", "down"):
                for r in rb.RS:
                    for clock in rb.CLOCKS:
                        for scale in rb.SCALES:
                            for H in rb.HS:
                                for sd in en.B_SEEDS:
                                    rows = tiers[m, sd]
                                    i0 = rb.window_rule(rows, D, m, H, r, None, clock, scale, 0.0, version)
                                    i1 = rb.window_rule(rows, D, m, H, r, None, clock, scale, 0.0, version, M=t["task"])
                                    cells += 1
                                    if i0 != i1:
                                        changed += 1
                                        details.append((m, version, r, clock, scale, H, sd, i0, i1))
                                    n0 = rb.realised(rows, D, m, {"ffn": i0, "mha": i0}, r, 0.0, clock, scale, 0.0, H, version)
                                    n1 = rb.realised(rows, D, m, {"ffn": i1, "mha": i1}, r, 0.0, clock, scale, 0.0, H, version, M=t["task"])
                                    d = abs(n1 - n0) / n0 if n0 else 0.0
                                    if d > mw:
                                        mw, where = d, (m, version, r, clock, scale, H, sd)
                                    fx = {o: rb.fixed_rule(rows, D, o) for o in en.OPS}
                                    f0 = rb.realised(rows, D, m, fx, r, 0.0, clock, scale, 0.0, H, version)
                                    f1 = rb.realised(rows, D, m, fx, r, 0.0, clock, scale, 0.0, H, version, M=t["task"])
                                    if f0:
                                        mf = max(mf, abs(f1 - f0) / f0)
        sens[t["name"]] = {"task_bytes": t["task"], "changed": changed, "cells": cells, "max_window": mw, "max_fixed": mf, "where": where, "details": details[:50]}
        total = cells
        L.append(f"| {t['task'] / MiB:,.0f} MiB（{t['name']}） | {changed} / {cells} | {mw * 100:.5f}% | {mf * 100:.5f}% | {where if mw else '—'} |")
    res["sensitivity"] = sens
    worst = max(sens.values(), key=lambda x: x["task_bytes"])
    L.append(f"\n最大的 M_KV（单任务 2 GiB，这时 ρ 公式其实已不适用，只作上界）下：决策改变 {worst['changed']} / {worst['cells']} 格，窗口决策累计完成量最大相对变化 {worst['max_window'] * 100:.5f}%，固定 ε {worst['max_fixed'] * 100:.5f}%。")
    # ---- T̂ versus backend
    L.append("\n## T̂ 的系统性低估（FFN）与进 ε 的不一致\n")
    res["that"] = that_block(L, D)
    # ---- FULL
    L.append("\n## FULL 三行与 L2G+sync（阶段 A、C）\n")
    res["full"] = full_block(L)
    path = os.path.join(en.OUT, "report_b.md")
    with open(path, encoding="utf-8") as f:
        old = f.read().split(MARK)[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(old + "\n".join(L) + "\n")
    with open(os.path.join(en.OUT, "report_kv.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print("\n".join(L))


def that_block(L, D):
    """Counts: T̂ within 2% of fault-free T̂ against backend within 2% of fault-free backend, one row per run."""
    rows = []
    for op in en.OPS:
        pre = D["pre"][op]
        thr_t, thr_b = pre["T"] * 1.02, pre["bk"] * 1.02

        def add(stage, fid, name, T, b):
            rows.append({"op": op, "stage": stage, "fault": fid, "name": name, "T": T, "bk": b, "err": T / b - 1, "T_in": T <= thr_t, "bk_in": b <= thr_b})

        for stage, fids in (("A", SINGLE), ("C", K4)):
            for fid in fids:
                fold = en.l1_out(fid) if stage == "A" else en.OUT
                l = eir.read_sink(en.job_l1(op, fid, out=fold)["sink"])
                add(stage, fid, "R0", l["T_R0"], rb.bk(l["R0"]["backend"]))
                add(stage, fid, "L1", l["T_L1"], rb.bk(l["L1"]["backend"]))
                for m in METHODS:
                    for sd in en.ALL_SEEDS:
                        p = en.job_rep(op, fid, m, sd)["sink"]
                        if os.path.exists(p):
                            d = eir.read_sink(p)["record"]
                            add(stage, fid, LABEL[m], d["T"], rb.bk(d["backend"]))
        for (o, m, sd), d in D["ck"].items():
            if o == op:
                for i, s in enumerate((30, 100, 300)):
                    c = d["ck"][i]
                    add("B", "F4a", f"{LABEL[m]} s={s}", c["best_T"], rb.bk(c["backend"]))
    L.append("T̂ 误差 = T̂ / 后端 − 1。故障前 FFN：T̂ " + rb.f0(D["pre"]["ffn"]["T"]) + "，后端 " + rb.f0(D["pre"]["ffn"]["bk"]) + f"，{(D['pre']['ffn']['T'] / D['pre']['ffn']['bk'] - 1) * 100:+.1f}%；MHA：T̂ " + rb.f0(D["pre"]["mha"]["T"]) + "，后端 " + rb.f0(D["pre"]["mha"]["bk"]) + f"，{(D['pre']['mha']['T'] / D['pre']['mha']['bk'] - 1) * 100:+.1f}%。\n")
    L.append("| 任务 | 阶段 | 方案组 | 次数 | T̂ 误差中位数 | 范围 |\n|---|---|---|---|---|---|")
    out = {}
    for op in en.OPS:
        for stage in "ACB":
            for grp, pred in (("R0", lambda n: n == "R0"), ("L1", lambda n: n == "L1"), ("L2", lambda n: n not in ("R0", "L1"))):
                v = [r["err"] for r in rows if r["op"] == op and r["stage"] == stage and pred(r["name"])]
                if v:
                    L.append(f"| {op} | {stage} | {grp} | {len(v)} | {med(v) * 100:+.1f}% | {min(v) * 100:+.1f}% – {max(v) * 100:+.1f}% |")
                    out[f"{op}_{stage}_{grp}"] = {"n": len(v), "median": med(v), "min": min(v), "max": max(v)}
    L.append("\n进 ε 看 T̂ ≤ 故障前 T̂ × 1.02；后端回到 2% 以内看后端 ≤ 故障前后端 × 1.02。每次运行一行，R0、L1 按故障各算一次。\n")
    L.append("| 任务 | 阶段 | 次数 | T̂ 进且后端进 | **T̂ 进但后端没进** | T̂ 没进但后端进 | 都没进 |\n|---|---|---|---|---|---|---|")
    for op in en.OPS:
        for stage in "ACB":
            r0 = [r for r in rows if r["op"] == op and r["stage"] == stage]
            a = sum(r["T_in"] and r["bk_in"] for r in r0)
            b = sum(r["T_in"] and not r["bk_in"] for r in r0)
            c = sum(not r["T_in"] and r["bk_in"] for r in r0)
            d = sum(not r["T_in"] and not r["bk_in"] for r in r0)
            L.append(f"| {op} | {stage} | {len(r0)} | {a} | {b} | {c} | {d} |")
            out[f"count_{op}_{stage}"] = {"n": len(r0), "both": a, "T_only": b, "bk_only": c, "neither": d}
    x = [r for r in rows if r["op"] == "ffn" and r["fault"] == "CRandom" and r["name"] in ("R0", "L1")]
    r0, l1 = x[0], x[1]
    L.append(f"\n停止规则早停的例子（FFN CRandom）：L1 的 T̂ {rb.f0(l1['T'])}，进 ε，规则停在 L1；后端 {rb.f0(l1['bk'])}（{(l1['bk'] / D['pre']['ffn']['bk'] - 1) * 100:+.1f}%），比 R0 的 {rb.f0(r0['bk'])}（{(r0['bk'] / D['pre']['ffn']['bk'] - 1) * 100:+.1f}%）还差；T̂ 误差 {l1['err'] * 100:+.1f}%。")
    return out


def full_block(L):
    L.append("FULL 每个组合 14 次。“相同”是按 T̂ 自选与按后端最优选中同一个变体。“差距”= 按 T̂ 自选 − 按后端最优，相对故障前。L2G+sync 取种子中位数和范围，s = 100。\n")
    L.append("| 阶段 | 任务 | 故障 | FULL 最优 | FULL 按 T̂ 自选 | 相同 | 差距 | FULL 单次中位数 | L2G+sync（范围） | 种子数 | L2G+sync − FULL 单次中位数 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    out = {"A": {"same": 0, "n": 0, "gaps": []}, "C": {"same": 0, "n": 0, "gaps": []}, "rows": []}
    for stage, fids in (("A", ("F4", "F5", "F4a", "F8a")), ("C", K4)):
        for op in en.OPS:
            pre = rb.bk(eir.read_sink(eir.job_pre(en.MAIN, "a3h", op)["sink"])["record"]["backend"])
            for fid in fids:
                rows = rn.s2_full(f"{op}_{fid.lower()}") if fid in ("F4", "F5") else rn.next_full(op, fid)
                if not rows:
                    continue
                x = rn.full_three(rows, pre)
                same = x["best_run"] == x["by_T_run"]
                out[stage]["n"] += 1
                out[stage]["same"] += same
                gap = x["by_T_rel"] - x["best_rel"]
                out[stage]["gaps"].append(gap)
                v = []
                for sd in en.ALL_SEEDS:
                    p = en.job_rep(op, fid, "L2Gsync", sd)["sink"]
                    if os.path.exists(p):
                        v.append(rb.bk(eir.read_sink(p)["record"]["backend"]))
                l2 = (f"{(med(v) / pre - 1) * 100:+.1f}%（{(min(v) / pre - 1) * 100:+.1f}–{(max(v) / pre - 1) * 100:+.1f}）", len(v), f"{(med(v) - x['median']) / pre * 100:+.1f}%") if v else ("—", 0, "—")
                L.append(f"| {stage} | {op} | {fid} | {x['best_rel']:+.1f}% | {x['by_T_rel']:+.1f}% | {'是' if same else '否'} | {gap:+.1f} | {x['median_rel']:+.1f}% | {l2[0]} | {l2[1]} | {l2[2]} |")
                out["rows"].append({"stage": stage, "op": op, "fault": fid, **x, "same": same, "gap": gap, "l2g": [med(v) / pre - 1, min(v) / pre - 1, max(v) / pre - 1] if v else None})
    for k in "AC":
        g = out[k]["gaps"]
        L.append(f"\n阶段 {k}：按 T̂ 自选与按后端最优相同 {out[k]['same']} / {out[k]['n']}；差距 ≤ 2 个百分点的 {sum(x <= 2 for x in g)} / {len(g)}，最大 {max(g):+.1f}。")
    return out


if __name__ == "__main__":
    main()
