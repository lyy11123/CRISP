"""事后追加，看过阶段 B 数据后才提出。FFN 上 T̂ 低估的只读诊断。不改 CPCM，不重跑 SA，不覆盖已有结果。

  python that_diag.py select            cases.json (rule below), no runs
  python that_diag.py run --workers 14  backend and CPCM reevaluation of each saved state, out/next/that_diag/
  python that_diag.py report            tables, appended to report_b.md after its marker (run report_prog.py first)

Case rule (fixed before the runs, PROGRESS step 28). "T̂ 进 ε": T̂ <= fault-free T̂ x 1.02. Backend values are the stored 10-seed means.
- FFN fault-free A3h state; FFN fault-free BusyBarn-start state (never selected on T̂); MHA fault-free A3h state (reference).
- FFN R0 and L1 of the 7 faults F4a, F5, F8a, CRandom, CCluster, CEdge, CScratch (CRandom's are the named cases).
- FFN L2G+sync. Combinations: A = F4a, F5, F8a s = 100; C = the four k = 4 instances s = 100; B = F4a checkpoints 30, 100, 300.
  Problem: T̂ in ε and backend in [+3%, +6%]. Control: T̂ in ε and backend <= +2%. Per combination, in seed order
  (123, 1, 2, ...; within a seed B goes by checkpoint), the first 3 of each.
"""
import argparse
import json
import math
import os
import statistics as st

import eir
import eir_next as en
import report_b as rb
import report_next as rn

OUT = os.path.join(en.OUT, "that_diag")
CASES = os.path.join(OUT, "cases.json")
FAULTS = ("F4a", "F5", "F8a") + tuple(en.C_FAULTS)
A = ("F4a", "F5", "F8a")
C = tuple(en.C_FAULTS)
CL = ("D2D", "片内链路", "DDR 读", "DDR 写", "计算")
MARK = "<!-- 事后追加：that_diag.py -->"
EPS, LO, HI = 0.02, 0.03, 0.06


def bk(b):
    return rb.bk(b)


def l1_sink(op, fid):
    return en.job_l1(op, fid, out=en.l1_out(fid) if fid in A else en.OUT)["sink"]


def select():
    os.makedirs(OUT, exist_ok=True)
    pre = eir.read_sink(eir.job_pre(en.MAIN, "a3h", "ffn")["sink"])
    Tp, Bp = pre["record"]["T"], bk(pre["record"]["backend"])
    cases = [{"id": "pre_ffn", "group": "故障前 A3h", "start": "a3h", "op": "ffn", "fault": None, "T_stored": pre["T"], "bk_stored": Bp}]
    e = eir.read_sink(eir.job_pre(en.MAIN, "e3", "ffn")["sink"])
    cases.append({"id": "pre_e3_ffn", "group": "故障前 BusyBarn 起点（未按 T̂ 选）", "start": "e3", "op": "ffn", "fault": None, "T_stored": e["T"], "bk_stored": bk(e["record"]["backend"])})
    m = eir.read_sink(eir.job_pre(en.MAIN, "a3h", "mha")["sink"])
    cases.append({"id": "pre_mha", "group": "MHA 故障前（参照）", "start": "a3h", "op": "mha", "fault": None, "T_stored": m["T"], "bk_stored": bk(m["record"]["backend"])})
    for fid in FAULTS:
        l = eir.read_sink(l1_sink("ffn", fid))
        for name, key, T in (("R0", "R0_snap", l["T_R0"]), ("L1", "L1_snap", l["T_L1"])):
            cases.append({"id": f"{name}_{fid}", "group": name, "start": "a3h", "op": "ffn", "fault": fid, "T_stored": T, "bk_stored": bk(l[name]["backend"]), "src": {"sink": l1_sink("ffn", fid), "path": [key]}})
    pools = []
    for phase, fids in (("A", A), ("C", C)):
        for fid in fids:
            rows = []
            for sd in en.ALL_SEEDS:
                p = en.job_rep("ffn", fid, "L2Gsync", sd)["sink"]
                if os.path.exists(p):
                    d = eir.read_sink(p)["record"]
                    rows.append({"T": d["T"], "bk": bk(d["backend"]), "seed": sd, "s": 100, "sink": p, "path": ["final_snap"]})
            pools.append((phase, fid, rows))
    rows = []
    for sd in en.B_SEEDS:
        p = en.b_job("ffn", "L2Gsync", sd)["sink"]
        d = eir.read_sink(p)
        for i, c in enumerate(d["ck"]):
            rows.append({"T": c["best_T"], "bk": bk(c["backend"]), "seed": sd, "s": c["step"], "sink": p, "path": ["search", "checkpoints", i, "snap"]})
    pools.append(("B", "F4a", rows))
    counts = {}
    for phase, fid, rows in pools:
        prob = [r for r in rows if r["T"] <= Tp * (1 + EPS) and Bp * (1 + LO) <= r["bk"] <= Bp * (1 + HI)]
        ctrl = [r for r in rows if r["T"] <= Tp * (1 + EPS) and r["bk"] <= Bp * (1 + EPS)]
        counts[f"{phase}_{fid}"] = {"pool": len(rows), "problem": len(prob), "control": len(ctrl)}
        for group, sel in (("问题例", prob), ("对照例", ctrl)):
            for r in sel[:3]:
                cases.append({"id": f"{group}_{phase}_{fid}_s{r['s']}_r{r['seed']}", "group": group, "phase": phase, "start": "a3h", "op": "ffn", "fault": fid, "T_stored": r["T"], "bk_stored": r["bk"], "seed": r["seed"], "s": r["s"],
                              "src": {"sink": r["sink"], "path": r["path"]}})
    with open(CASES, "w", encoding="utf-8") as f:
        json.dump({"cases": cases, "counts": counts, "Tp": Tp, "Bp": Bp}, f, indent=1, ensure_ascii=False)
    print(json.dumps(counts, ensure_ascii=False, indent=0))
    print(len(cases), "cases")


def jobs():
    with open(CASES, encoding="utf-8") as f:
        cs = json.load(f)["cases"]
    out = []
    for c in cs:
        j = {"kind": "that_diag", "name": "diag_" + c["id"], "start": c["start"], "op": c["op"], "T_stored": c["T_stored"], "src": c.get("src"), "sink": os.path.join(OUT, "runs", c["id"] + ".pkl.gz")}
        if c["fault"]:
            j["fault"] = c["fault"]
        out.append(j)
    return out


# ---------------------------------------------------------------- report
def pct(x):
    return f"{x * 100:+.2f}%"


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    a, b = rank(x), rank(y)
    ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else float("nan")


def slope(x, y):
    mx, my = st.mean(x), st.mean(y)
    den = sum((p - mx) ** 2 for p in x)
    return sum((p - mx) * (q - my) for p, q in zip(x, y)) / den if den else float("nan")


def report():
    with open(CASES, encoding="utf-8") as f:
        cf = json.load(f)
    cs = cf["cases"]
    R = {}
    for c, j in zip(cs, jobs()):
        R[c["id"]] = (c, eir.read_sink(j["sink"]))
    L = [MARK, "\n## FFN 上 T̂ 低估的只读诊断（事后追加，看过阶段 B 数据后才提出）\n"]
    L.append("不改 CPCM，不重跑 SA。对保存的状态重新跑后端 10 个种子，CPCM 确定性一次，CPCM 随机打破平局（V3 的诊断模式，不是 T̂）40 次，其中 10 次做链分解。关键链用 `e2_attribution.Sim.chain()`，后端和 CPCM 都读同一种记录。链占用是对 [0, T] 的划分，所以类别差相加等于总差。\n")
    bad = sum(not d["T_equal"] for _, d in R.values())
    drift = [abs(d["bk_mean"] / c["bk_stored"] - 1) for c, d in R.values()]
    L.append(f"自检：重建的 T̂ 与保存值不同 {bad} / {len(R)}；新后端均值与保存均值之差中位数 {st.median(drift) * 100:.2f}%，最大 {max(drift) * 100:.2f}%（换进程噪声）；链和对 T 的差最大 {max(abs(d['m_backend']['chain_sum'] / d['m_backend']['T'] - 1) for _, d in R.values()) * 100:.2f}%。\n")
    L.append("选例计数（池子大小 / 问题例 / 对照例，每个组合各取前 3 个）：" + "；".join(f"{k} {v['pool']} / {v['problem']} / {v['control']}" for k, v in cf["counts"].items()) + "。\n")

    def row(d):
        Tb, Td, Tr = d["bk_mean"], d["T_det"], st.mean(d["T_rng"])
        q = sorted(d["T_rng"])
        return {"Tb": Tb, "Td": Td, "Tr": Tr, "err": Td / Tb - 1, "luck": (Td - Tr) / Tb, "model": (Tr - Tb) / Tb, "below": sum(t < Td for t in d["T_rng"]) / len(q), "rmin": q[0], "rmax": q[-1], "bmin": min(d["backend"]), "bmax": max(d["backend"])}

    def groups():
        g = {}
        order = ["pre_e3_ffn", "pre_mha", "pre_ffn"]
        for k in order:
            g[R[k][0]["group"]] = [k]
        for name in ("R0", "L1"):
            g[f"{name}（7 个故障）"] = [k for k, (c, _) in R.items() if c["group"] == name]
        g["CRandom R0"] = ["R0_CRandom"]
        g["CRandom L1"] = ["L1_CRandom"]
        for ph in "ACB":
            for name in ("问题例", "对照例"):
                g[f"{name} 阶段 {ph}"] = [k for k, (c, _) in R.items() if c["group"] == name and c.get("phase") == ph]
        return {k: v for k, v in g.items() if v}

    G = groups()
    L.append("### 表 1：总误差 = 平局运气 + 模型差\n")
    L.append("总误差 = T̂ / 后端 − 1。平局运气 = (T̂ − 随机平局 CPCM 均值) / 后端：T̂ 比同一状态随机平局时的 CPCM 低多少。模型差 = (随机平局 CPCM 均值 − 后端) / 后端：去掉平局选择后，CPCM 和后端还差多少。“T̂ 排位”= 40 次随机平局里比确定性 T̂ 还低的比例，越接近 0 说明确定性顺序越幸运。\n")
    L.append("| 组 | 例数 | 后端 | T̂ | 随机平局 CPCM 均值 | 总误差 | 平局运气 | 模型差 | T̂ 排位 |\n|---|---|---|---|---|---|---|---|---|")
    table1 = {}
    for name, keys in G.items():
        rows = [row(R[k][1]) for k in keys]
        x = {a: st.median(r[a] for r in rows) for a in rows[0]}
        table1[name] = {"n": len(rows), **x}
        L.append(f"| {name} | {len(rows)} | {rb.f0(x['Tb'])} | {rb.f0(x['Td'])} | {rb.f0(x['Tr'])} | {pct(x['err'])} | {pct(x['luck'])} | {pct(x['model'])} | {x['below'] * 100:.0f}% |")
    # ---- classes
    base = R["pre_ffn"][1]
    Tb0 = base["bk_mean"]

    def diff(d, a, b, k, c):
        return (d[a][k].get(c, 0) - d[b][k].get(c, 0)) / d["bk_mean"]

    L.append("\n### 表 2：多出来的低估落在哪一类（关键链占用，后端 − T̂，相对后端，再减故障前基准）\n")
    L.append("每格 = 该组该类的 (后端链占用 − CPCM 确定性链占用) / 后端，减去故障前 A3h 的同一量。正数表示这一类在后端关键链上比 T̂ 看到的多占，且多于故障前。一行相加 = 总低估相对故障前的增量（取负号的总误差增量）。各组取中位数，所以相加只是近似。\n")
    L.append("| 组 | 例数 | 多出的低估（总） | " + " | ".join(CL) + " |\n|---|---|---|" + "---|" * len(CL))
    L.append("| 故障前基准（未减） | 1 | " + pct(-(base["T_det"] / Tb0 - 1)) + " | " + " | ".join(pct(diff(base, "m_backend", "m_det", "chain", c)) for c in CL) + " |")
    t2 = {}
    b0 = {c: diff(base, "m_backend", "m_det", "chain", c) for c in CL}
    e0 = -(base["T_det"] / Tb0 - 1)
    for name, keys in G.items():
        if keys == ["pre_ffn"] or keys == ["pre_mha"]:
            continue
        v = {c: st.median(diff(R[k][1], "m_backend", "m_det", "chain", c) - b0[c] for k in keys) for c in CL}
        tot = st.median(-(R[k][1]["T_det"] / R[k][1]["bk_mean"] - 1) - e0 for k in keys)
        t2[name] = {"total": tot, **v}
        L.append(f"| {name} | {len(keys)} | {pct(tot)} | " + " | ".join(pct(v[c]) for c in CL) + " |")
    L.append("\n### 表 3：全部活动的排队等待（后端 − T̂，相对后端，不减基准）\n")
    L.append("等待 = 活动开始 − 它最早能开始的时刻，按资源类别在全部活动上累加（不只关键链）；各活动重叠，所以比值可以大于 100%，只看同类之间的差。\n")
    L.append("| 组 | 例数 | " + " | ".join(CL) + " |\n|---|---|" + "---|" * len(CL))
    t3 = {}
    for name, keys in G.items():
        v = {c: st.median(diff(R[k][1], "m_backend", "m_det", "wait", c) for k in keys) for c in CL}
        t3[name] = v
        L.append(f"| {name} | {len(keys)} | " + " | ".join(pct(v[c]) for c in CL) + " |")
    L.append("\n### 表 4：同一差额分两段（关键链占用，相对后端，不减基准）\n")
    L.append("前一段 = 随机平局 CPCM − 确定性 T̂（平局运气），后一段 = 后端 − 随机平局 CPCM（模型差）。\n")
    L.append("| 组 | 段 | " + " | ".join(CL) + " |\n|---|---|" + "---|" * len(CL))
    t4 = {}
    for name, keys in G.items():
        a = {c: st.median(diff(R[k][1], "m_rng", "m_det", "chain", c) for k in keys) for c in CL}
        b = {c: st.median(diff(R[k][1], "m_backend", "m_rng", "chain", c) for k in keys) for c in CL}
        t4[name] = {"luck": a, "model": b}
        L.append(f"| {name} | 平局运气 | " + " | ".join(pct(a[c]) for c in CL) + " |")
        L.append(f"| {name} | 模型差 | " + " | ".join(pct(b[c]) for c in CL) + " |")
    # ---- existing-data tests
    L.append("\n### 表 5：只用已有数据的对照（没有新运行）\n")
    tests = existing(L, cf)
    res = {"table1": table1, "table2": t2, "table3": t3, "table4": t4, "tests": tests, "counts": cf["counts"], "cases": {k: {"T_det": d["T_det"], "bk_mean": d["bk_mean"], "T_rng": d["T_rng"], "backend": d["backend"], "T_equal": d["T_equal"], "m_backend": d["m_backend"], "m_det": d["m_det"], "m_rng": d["m_rng"]} for k, (_, d) in R.items()}}
    with open(os.path.join(en.OUT, "report_that.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    path = os.path.join(en.OUT, "report_b.md")
    with open(path, encoding="utf-8") as f:
        old = f.read().split(MARK)[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(old + "\n".join(L) + "\n")
    print("\n".join(L))


def existing(L, cf):
    out = {}
    D = rb.load()
    Tp, Bp = cf["Tp"], cf["Bp"]
    # no-fault search: the fault-free state searched further with T̂, nothing broken
    L.append("**(a) 无故障搜索（`out/main/ctrl`，L2G，不带 sync）**：没有故障，只有按 T̂ 的搜索。\n")
    L.append("| 起点 | 步数 | 种子数 | T̂ 中位数 | 后端中位数 | 误差中位数 | 范围 |\n|---|---|---|---|---|---|---|")
    for start in ("a3h", "e3"):
        pre = eir.read_sink(eir.job_pre(en.MAIN, start, "ffn")["sink"])
        e0 = pre["T"] / bk(pre["record"]["backend"]) - 1
        L.append(f"| {start} | 0（起点本身） | 1 | {rb.f0(pre['T'])} | {rb.f0(bk(pre['record']['backend']))} | {pct(e0)} | — |")
        out[f"ctrl_{start}_0"] = e0
        for s in (30, 100, 300):
            v = []
            for sd in eir.SA_SEEDS:
                p = eir.job_ctrl(en.MAIN, start, "ffn", s, sd)["sink"]
                if os.path.exists(p):
                    r = eir.read_sink(p)["record"]
                    v.append((r["T"], bk(r["backend"])))
            if v:
                e = [a / b - 1 for a, b in v]
                L.append(f"| {start} | {s} | {len(v)} | {rb.f0(st.median(a for a, _ in v))} | {rb.f0(st.median(b for _, b in v))} | {pct(st.median(e))} | {pct(min(e))} – {pct(max(e))} |")
                out[f"ctrl_{start}_{s}"] = {"median": st.median(e), "min": min(e), "max": max(e), "n": len(e)}
    # search depth on the same runs
    L.append("\n**(b) 同一批种子随搜索加深（阶段 B 检查点，F4a，FFN，10 个种子）**：\n")
    L.append("| 方法 | L1 误差 | s=30 | s=100 | s=300 |\n|---|---|---|---|---|")
    l = D["l1"]["ffn"]
    for m in en.B_METHODS:
        cells = []
        for i in range(3):
            e = [D["ck"]["ffn", m, sd]["ck"][i]["best_T"] / bk(D["ck"]["ffn", m, sd]["ck"][i]["backend"]) - 1 for sd in en.B_SEEDS]
            cells.append(st.median(e))
        out[f"depth_{m}"] = cells
        L.append(f"| {rb.LABEL[m]} | {pct(l['T_L1'] / l['bk_L1'] - 1)} | " + " | ".join(pct(x) for x in cells) + " |")
    # rank: within one combination, lower T̂ against error
    L.append("\n**(c) 同一组合内 T̂ 排名与误差（FFN，s = 100，种子间）**：Spearman(T̂, 误差) 为正 = T̂ 越低低估越多；斜率 = 后端对 T̂ 的回归斜率，小于 1 说明 T̂ 上的改进只有一部分兑现到后端。\n")
    L.append("| 故障 | 方法 | n | Spearman(T̂, 误差) | 后端对 T̂ 的斜率 |\n|---|---|---|---|---|")
    rs = []
    for fid in FAULTS:
        for m in en.B_METHODS:
            v = []
            for sd in en.ALL_SEEDS:
                p = en.job_rep("ffn", fid, m, sd)["sink"]
                if os.path.exists(p):
                    r = eir.read_sink(p)["record"]
                    v.append((r["T"], bk(r["backend"])))
            if len(v) >= 5:
                x, y = [a for a, _ in v], [b for _, b in v]
                e = [a / b - 1 for a, b in v]
                row = {"fault": fid, "m": m, "n": len(v), "rho": spearman(x, e), "slope": slope(x, y)}
                rs.append(row)
                L.append(f"| {fid} | {rb.LABEL[m]} | {len(v)} | {row['rho']:+.2f} | {row['slope']:.2f} |")
    ok = [r for r in rs if r["rho"] == r["rho"]]
    L.append(f"\nSpearman 中位数 {st.median(r['rho'] for r in ok):+.2f}，为正的 {sum(r['rho'] > 0 for r in ok)} / {len(ok)}；斜率中位数 {st.median(r['slope'] for r in ok):.2f}。")
    out["rank"] = rs
    # FULL
    L.append("\n**(d) FULL-A3h（FFN，14 次）：同一组合内 T̂ 最低一半与最高一半的误差**：\n")
    L.append("| 故障 | T̂ 低一半误差中位数 | T̂ 高一半误差中位数 | 全部中位数 |\n|---|---|---|---|")
    fr = {}
    for fid in ("F4", "F5", "F4a", "F8a") + C:
        rows = rn.s2_full(f"ffn_{fid.lower()}") if fid in ("F4", "F5") else rn.next_full("ffn", fid)
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: (r["T"], r["own"]))
        e = [r["T"] / r["mean"] - 1 for r in rows]
        h = len(e) // 2
        fr[fid] = {"low": st.median(e[:h]), "high": st.median(e[h:]), "all": st.median(e)}
        L.append(f"| {fid} | {pct(fr[fid]['low'])} | {pct(fr[fid]['high'])} | {pct(fr[fid]['all'])} |")
    out["full"] = fr
    # BusyBarn start: repair on a state that was not selected on T̂
    L.append("\n**(e) BusyBarn 起点的修复（`out/main`，FFN，L2G，s = 100，不带 sync，5 个种子）**：起点没有按 T̂ 选过。\n")
    L.append("| 故障 | R0 误差 | L1 误差 | L2G 误差中位数 |\n|---|---|---|---|")
    pre = eir.read_sink(eir.job_pre(en.MAIN, "e3", "ffn")["sink"])
    bb = {}
    for fid in ("F1", "F2", "F4", "F5", "F6", "F7", "F8"):
        p = eir.job_l1(en.MAIN, "e3", "ffn", fid)["sink"]
        if not os.path.exists(p):
            continue
        l1 = eir.read_sink(p)
        v = []
        for sd in eir.SA_SEEDS:
            q = eir.job_rep(en.MAIN, "e3", "ffn", fid, "L2G", 100, sd)["sink"]
            if os.path.exists(q):
                r = eir.read_sink(q)["record"]
                v.append(r["T"] / bk(r["backend"]) - 1)
        if v:
            bb[fid] = {"R0": l1["T_R0"] / bk(l1["R0"]["backend"]) - 1, "L1": l1["T_L1"] / bk(l1["L1"]["backend"]) - 1, "L2G": st.median(v)}
            L.append(f"| {fid} | {pct(bb[fid]['R0'])} | {pct(bb[fid]['L1'])} | {pct(bb[fid]['L2G'])} |")
    out["bb"] = bb
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "run", "report", "smoke"])
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    if a.cmd == "select":
        select()
    elif a.cmd == "smoke":
        js = [j for j in jobs() if j["name"] in ("diag_pre_ffn", "diag_L1_CRandom")]
        print(eir.run_phases([js], OUT, 2))
        for j in js:
            d = eir.read_sink(j["sink"])
            print(j["name"], d["T_det"], d["T_equal"], d["bk_mean"], st.mean(d["T_rng"]), d["m_backend"]["chain"], d["m_det"]["chain"], d["m_backend"]["chain_sum"], d["m_det"]["chain_sum"])
    elif a.cmd == "run":
        if en.run_logged(f"that_diag_w{a.workers}", jobs(), a.workers):
            raise SystemExit(1)
    else:
        report()
