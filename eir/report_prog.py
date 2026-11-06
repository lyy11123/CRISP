"""事后追加，看过阶段 B 数据后才提出。渐进部署的配对回放。Replays stored data, no new runs. Run report_b.py and report_kv.py first; appends to report_b.md
after its marker, writes out/next/report_prog.json and caches the pair differences in out/next/prog_pairs.json.

  python report_prog.py

Chains (both tasks share the tier, deployed together, times added, PROGRESS step 25): R0 serves from 0.
  A  L1, 30, 100, 300, switch only when the tier's T̂_a is below the deployed one. Decided on T̂ only, completions from the backend.
  B  L1, 30, 100, 300, switch at every tier.
  C  L1, 100, 300, switch at every tier. Skipping s = 30 was decided after seeing the data.
A switch stops service for T_copy + T_switch; the search keeps running. T_copy: the blocks in the new state's placement that the deployed state's lacks.
"""
import json
import os
import pickle
import statistics as st
import sys

import eir
import eir_next as en
import report_b as rb

A_ = os.path.join(eir.HERE, "work", "artifact")
MARK = "<!-- 事后追加：report_prog.py -->"
CACHE = os.path.join(en.OUT, "prog_pairs.json")
NAMES = ("R0", "L1", "s30", "s100", "s300")
CHAINS = {"A": ("L1", "s30", "s100", "s300"), "B": ("L1", "s30", "s100", "s300"), "C": ("L1", "s100", "s300")}
M_MAIN = 2 ** 30 * 3 / 32   # 96 MiB, one MHA task (report_kv)
CMP = ("固定 ε", "窗口决策", "事后最优单次", "R0", "FULL 14 次依次", "FULL 14 次并行")


class Shim:
    def __init__(self, state, ddr, sizes):
        self.state = state
        self.hardware_platform = type("HP", (), {"ddr_dict": {tuple(d) + (0,): None for d in ddr}})()
        self.offlinedata_list = sorted({d for d, _ in sizes}, key=str)
        self.offlinedata_split_dict = {d: [sp for x, sp in sizes if x == d] for d in self.offlinedata_list}


def view(snap, pre):
    """What a pair difference needs from a saved state: placement and where each behaviour is."""
    a = Shim(pickle.loads(snap["blob"]), pre["ddr_nodes"], list(pre["sizes"]))
    return {"p": eir.placement(a), "b": {t: tuple(eir._plain(x.location))[:2] for t, x in a.state[5].items()}}


def pairs(D):
    """(method, seed, task) -> {'i,j': (bytes of j's blocks missing in i, share of behaviours elsewhere)}. Real states."""
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    sys.path[:0] = [A_, os.path.join(A_, "src")]
    out = {}
    for o in en.OPS:
        pre = eir.read_sink(eir.job_pre(en.MAIN, "a3h", o)["sink"])
        l = eir.read_sink(en.job_l1(o, en.B_FAULT, out=en.l1_out(en.B_FAULT))["sink"])
        v0 = [view(l["R0_snap"], pre), view(l["L1_snap"], pre)]
        for m in rb.METHODS:
            for sd in rb.SEEDS:
                d = D["ck"][o, m, sd]
                vs = v0 + [view(c["snap"], pre) for c in d["search"]["checkpoints"]]
                row = {}
                for i in range(5):
                    for j in range(i + 1, 5):
                        by = 0
                        for k, locs in vs[j]["p"].items():
                            by += pre["sizes"][k] * len(set(locs) - set(vs[i]["p"].get(k, ())))
                        bi, bj = vs[i]["b"], vs[j]["b"]
                        row[f"{i},{j}"] = (by, sum(bi[t] != bj[t] for t in bi) / len(bi))
                out[f"{m}|{sd}|{o}"] = row
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def run_chain(steps, H, T0, sw, clock):
    """steps: (t_ready, T_a seconds, bytes). Returns completions in window H."""
    n, cur, Tcur = 0.0, 0.0, T0
    for t, Ta, by in steps:
        t0 = max(t, cur)
        if t0 >= H:
            break
        n += (t0 - cur) / Tcur
        cur = t0 + rb.t_copy(by, clock) + sw
        Tcur = Ta
        if cur >= H:
            return n
    return n + max(H - cur, 0.0) / Tcur


def chain_plan(name, rows, l, r, m, P, sd):
    """Tier indices (into NAMES) deployed in order, with task-wise pair differences. Decided on T̂ for chain A."""
    seq = [NAMES.index(x) for x in CHAINS[name]]
    that = {0: l["mha"]["T_R0"] * r + l["ffn"]["T_R0"]}
    for i in range(1, 5):
        x = rows[i - 1]["That"]
        that[i] = r * x["mha"] + x["ffn"]
    plan, cur = [], 0
    for i in seq:
        if name == "A" and not that[i] < that[cur]:
            continue
        by = {o: P[f"{m}|{sd}|{o}"][f"{cur},{i}"] for o in en.OPS}
        plan.append((i, by))
        cur = i
    return plan


def main():
    D = rb.load()
    tiers = rb.tier_table(D)
    P = pairs(D)
    # self-check: pair difference from R0 equals the stored migration
    bad = 0
    for m in rb.METHODS:
        for sd in rb.SEEDS:
            for i in range(2, 5):
                for o in en.OPS:
                    bad += P[f"{m}|{sd}|{o}"][f"0,{i}"][0] != tiers[m, sd][i - 1]["bytes"][o]
    l = D["l1"]
    res = {"pairs_mismatch": bad, "cells": {}}
    L = [MARK, "\n## 渐进部署（事后追加，看过阶段 B 数据后才提出）\n"]
    L.append("纯后处理，阶段 B 的 10 对种子。不再只部署一次：L1 算完就部署，之后每到一个检查点再热切换，搜索期间服务当前已部署的方案。两任务共用档位，一起部署，时间相加（第 25 步）。切换期间不服务，停 `T_copy + T_switch`，搜索照常。链 A（主结果，在线可判定）：只有该档 T̂_a 低于当前已部署方案才切，判断只用 T̂；链 B：每档都切；链 C：跳过 s=30，**这是看过数据才定的**。\n")
    L.append("**局限：一个物理故障、10 对种子、Python 量级，T_switch 没有依据。** 比值是链 / 对手的累计完成量，同种子配对；只报胜负。\n")
    L.append(f"自检：相邻档差异是用保存的检查点状态算的实际差异，其中相对 R0 的字节与已存的 `migration_vs_R0` 不一致 {bad} 处。\n")

    def ctx(m, sd, r, clock, scale, sw, H):
        rows = tiers[m, sd]
        T0 = rb.best_T0(D, r, 0.0, clock)
        i = rb.window_rule(rows, D, m, H, r, None, clock, scale, sw, "main")
        w = rb.realised(rows, D, m, {"ffn": i, "mha": i}, r, 0.0, clock, scale, sw, H, "main")
        fx = {o: rb.fixed_rule(rows, D, o) for o in en.OPS}
        f = rb.realised(rows, D, m, fx, r, 0.0, clock, scale, sw, H, "main")
        ora = max(rb.realised(rows, D, m, {"ffn": j, "mha": j}, r, 0.0, clock, scale, sw, H, "main") for j in range(4))
        return T0, {"固定 ε": f, "窗口决策": w, "事后最优单次": ora}

    def chain_n(name, m, sd, r, clock, scale, sw, H, T0, M=0.0):
        rows = tiers[m, sd]
        steps = []
        for i, by in chain_plan(name, rows, l, r, m, P, sd):
            x = rows[i - 1]
            t = sum(rb.t_tier(D, o, m, x["s"], scale) for o in en.OPS)
            Ta = rb.round_time(x["bk"]["mha"], x["bk"]["ffn"], r, 0.0, clock)
            steps.append((t, Ta, by["ffn"][0] + by["mha"][0] + by["mha"][1] * M))
        return run_chain(steps, H, T0, sw, clock)

    for m in rb.METHODS:
        for r in rb.RS:
            for clock in rb.CLOCKS:
                fo = rb.full_options(D, clock)
                Tf = rb.round_time(fo["picks"]["mha"]["bk"], fo["picks"]["ffn"]["bk"], r, 0.0, clock)
                for scale in rb.SCALES:
                    for sw in rb.SWITCHES:
                        for H in rb.HS:
                            T0 = rb.best_T0(D, r, 0.0, clock)
                            cf = rb.t_copy(fo["bytes"], clock) + sw
                            fixed_cmp = {"R0": H / T0, "FULL 14 次依次": rb.n_main(H, 14 * fo["t_single"] * scale, cf, Tf, T0), "FULL 14 次并行": rb.n_main(H, fo["t_single"] * scale, cf, Tf, T0)}
                            per = [ctx(m, sd, r, clock, scale, sw, H)[1] for sd in rb.SEEDS]
                            for ch in CHAINS:
                                ns = [chain_n(ch, m, sd, r, clock, scale, sw, H, T0) for sd in rb.SEEDS]
                                cell = {"N": st.median(ns), "vs": {}}
                                for c in CMP:
                                    other = [p[c] for p in per] if c in per[0] else [fixed_cmp[c]] * len(ns)
                                    rt = [a / b if b else float("inf") for a, b in zip(ns, other)]
                                    cell["vs"][c] = {"ratio": st.median(rt), "worst": min(rt), "w": sum(a > b * (1 + 1e-12) for a, b in zip(ns, other)), "l": sum(a < b * (1 - 1e-12) for a, b in zip(ns, other)), "other": st.median(other)}
                                cell["t"] = None
                                res["cells"][f"{ch}|{m}|r{r}|c{clock}|x{scale}|sw{sw}|H{H}"] = cell
    C = res["cells"]

    def get(ch, m, r, clock, scale, sw, H):
        return C[f"{ch}|{m}|r{r}|c{clock}|x{scale}|sw{sw}|H{H}"]

    L.append("### 表 1：链 / 对手（时钟 0.5 ns，搜索 × 1，T_switch = 0）\n")
    L.append("每格：配对比值中位数（胜/负/平，最差一对的比值）。“事后最优单次”是每对种子事后挑的单次部署，在线拿不到。\n")
    L.append("| 方法 | r | 链 | H | " + " | ".join(CMP) + " |\n|---|---|---|---|" + "---|" * len(CMP))

    def fmt(v):
        t = 10 - v["w"] - v["l"]
        return f"{v['ratio']:.3f}（{v['w']}/{v['l']}/{t}，{v['worst']:.3f}）"

    for m in rb.METHODS:
        for r in rb.RS:
            for ch in CHAINS:
                for H in rb.HS:
                    x = get(ch, m, r, 0.5, 1.0, 0.0, H)
                    L.append(f"| {rb.LABEL[m]} | {r} | {ch} | {H:,} s | " + " | ".join(fmt(x["vs"][c]) for c in CMP) + " |")
    # ---- losses
    L.append("\n### 表 2：哪些格子输了（全网格 3 方法 × 2 r × 2 时钟 × 2 搜索倍率 × 4 T_switch × 5 H = 480 格，看配对比值中位数 < 1）\n")
    L.append("| 链 | 对手 | 输的格数 / 480 | 其中 T_switch = 0 | 1 ms | 0.1 s | 10 s | 最小比值 | 所在格 |\n|---|---|---|---|---|---|---|---|---|")
    loss = {}
    for ch in CHAINS:
        for c in CMP:
            rows = [(k, v["vs"][c]) for k, v in C.items() if k.startswith(ch + "|")]
            lost = [(k, v) for k, v in rows if v["ratio"] < 1 - 1e-9]
            by = {sw: sum(f"|sw{sw}|" in k for k, _ in lost) for sw in rb.SWITCHES}
            k_min, v_min = min(rows, key=lambda x: x[1]["ratio"])
            loss[f"{ch}_{c}"] = {"lost": len(lost), "by_sw": by, "min": v_min["ratio"], "where": k_min}
            L.append(f"| {ch} | {c} | {len(lost)} | {by[0.0]} | {by[1e-3]} | {by[0.1]} | {by[10.0]} | {v_min['ratio']:.3f} | `{k_min}` |")
    res["loss"] = loss
    L.append("\n### 表 3：T_switch 多大时开始输（链 A，L2G+sync，时钟 0.5 ns；格内是配对比值中位数首次 < 1 的 T_switch，“—”= 网格内不输）\n")
    L.append("| r | 搜索倍率 | H | " + " | ".join(CMP) + " |\n|---|---|---|" + "---|" * len(CMP))
    thr = {}
    for r in rb.RS:
        for scale in rb.SCALES:
            for H in rb.HS:
                cells = []
                for c in CMP:
                    first = next((sw for sw in rb.SWITCHES if get("A", "L2Gsync", r, 0.5, scale, sw, H)["vs"][c]["ratio"] < 1 - 1e-9), None)
                    cells.append("—" if first is None else f"{first} s")
                    thr[f"r{r}|x{scale}|H{H}|{c}"] = first
                L.append(f"| {r} | ×{scale} | {H:,} s | " + " | ".join(cells) + " |")
    res["threshold"] = thr
    # ---- migration
    L.append("\n### 表 4：累计迁移量（10 对种子中位数，r = 1；范围见 JSON）\n")
    L.append("逐次切换的字节累加，用保存的检查点状态算相邻两次部署之间的实际差异，不是上界。ρ_MHA 累加 = 相邻两次部署之间搬动的 MHA 行为占比相加，可以超过 100%（同一行为被搬多次）。KV 按单任务 96 MiB 累加，是外部解析估算。一次部署的 ρ 相对 R0。\n")
    L.append("| 方法 | 方案 | 切换次数 | FFN 字节 MB | MHA 字节 MB | 合计 T_copy（0.5 ns） | ρ_MHA 累加 | KV 累加 MiB | KV T_KV 周期 |\n|---|---|---|---|---|---|---|---|---|")
    mig = {}
    for m in rb.METHODS:
        rows_out = []
        for ch in CHAINS:
            a = []
            for sd in rb.SEEDS:
                plan = chain_plan(ch, tiers[m, sd], l, 1, m, P, sd)
                a.append({"n": len(plan), "ffn": sum(b["ffn"][0] for _, b in plan), "mha": sum(b["mha"][0] for _, b in plan), "rho": sum(b["mha"][1] for _, b in plan)})
            rows_out.append((f"链 {ch}", a))
        for label, idx in (("一次部署 s=300", 4), ("一次部署 s=100", 3)):
            a = [{"n": 1, "ffn": P[f"{m}|{sd}|ffn"][f"0,{idx}"][0], "mha": P[f"{m}|{sd}|mha"][f"0,{idx}"][0], "rho": P[f"{m}|{sd}|mha"][f"0,{idx}"][1]} for sd in rb.SEEDS]
            rows_out.append((label, a))
        for label, a in rows_out:
            x = {k: st.median(r[k] for r in a) for k in a[0]}
            mig[f"{m}|{label}"] = {**x, "rho_min": min(r["rho"] for r in a), "rho_max": max(r["rho"] for r in a), "n_min": min(r["n"] for r in a), "n_max": max(r["n"] for r in a)}
            L.append(f"| {rb.LABEL[m]} | {label} | {x['n']:.0f}（{min(r['n'] for r in a)}–{max(r['n'] for r in a)}） | {x['ffn'] / 1e6:.1f} | {x['mha'] / 1e6:.1f} | {rb.sec(rb.t_copy(x['ffn'] + x['mha'], 0.5))} | {x['rho'] * 100:.1f}% | {x['rho'] * M_MAIN / 2 ** 20:.1f} | {x['rho'] * M_MAIN / 256:,.0f} |")
    res["migration"] = mig
    # ---- KV in the stop
    d96, d2g = 0.0, 0.0
    for m in rb.METHODS:
        for r in rb.RS:
            for clock in rb.CLOCKS:
                for scale in rb.SCALES:
                    for H in rb.HS:
                        T0 = rb.best_T0(D, r, 0.0, clock)
                        for ch in CHAINS:
                            for sd in rb.SEEDS:
                                n0 = chain_n(ch, m, sd, r, clock, scale, 0.0, H, T0)
                                d96 = max(d96, abs(chain_n(ch, m, sd, r, clock, scale, 0.0, H, T0, M=M_MAIN) / n0 - 1) if n0 else 0.0)
                                d2g = max(d2g, abs(chain_n(ch, m, sd, r, clock, scale, 0.0, H, T0, M=2 ** 31) / n0 - 1) if n0 else 0.0)
    res["kv_stop"] = {"96MiB": d96, "2GiB": d2g}
    L.append(f"\n把累加的 T_KV 计入切换停机（T_switch = 0，全部 H、r、时钟、搜索倍率、链、方法、种子）：累计完成量最大变化，单任务 96 MiB {d96 * 100:.3f}%，单任务 2 GiB {d2g * 100:.3f}%（2 GiB 时 ρ 公式其实已不适用，只作上界）。")
    path = os.path.join(en.OUT, "report_b.md")
    with open(path, encoding="utf-8") as f:
        old = f.read().split(MARK)[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(old + "\n".join(L) + "\n")
    with open(os.path.join(en.OUT, "report_prog.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=0, ensure_ascii=False, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
