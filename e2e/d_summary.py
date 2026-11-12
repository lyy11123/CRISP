"""D summary (numbers only): python e2e/d_summary.py -> e2e/D_SUMMARY.md and e2e/d/out/prod/summary.json.

Rules: PROGRESS 2026-11-11 "比较规则" (1-11) and "补充规则" (seeds 4-10 as a side column). Every result is read from e2e/d/out/prod/sims,
whatever machine produced it; the source of each job is taken from e2e/d/out/prod/sources.json (written by `sources`).

  python e2e/d_summary.py sources      who produced each sink (local / package 1 / package 2 / package 3), from the run records
  python e2e/d_summary.py              the summary
"""
import collections
import gzip
import json
import os
import pickle
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import d_batch as B  # noqa: E402

d = B.d
OUT = os.path.join(d.OUT, "prod")
SIMS = os.path.join(OUT, "sims")
REMOTE = os.path.join(d.OUT, "remote")
ARMS = ["B-R0", "B-FULL", "-DSTU", "CRISP", "-DSTU-prio", "-EIR", "-A3h", "rev", "noKV-B-R0", "noKV-CRISP"]
ARM_ZH = {"B-R0": "B-R0", "B-FULL": "B-FULL", "-DSTU": "−DSTU", "CRISP": "CRISP", "-DSTU-prio": "−DSTU 优先级", "-EIR": "−EIR",
          "-A3h": "−A3h", "rev": "反序参照", "noKV-B-R0": "无 KV 对照 B-R0", "noKV-CRISP": "无 KV 对照 CRISP"}
COUNT = {"pf_ln": 2, "pf_mha": 8, "pf_proj": 1, "pf_ffn": 1}
OUTS = (128, 512, 2048)
mean, median = statistics.mean, statistics.median


def load(p):
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def all_jobs():
    jobs = []
    for b in ("D1", "D2", "D3", "D4", "D5"):
        _, js = B.build(b, OUT)
        jobs += [dict(j, batch=b, part="base") for j in js]
        jobs += [dict(j, batch=b, part="ext") for j in B.ext_jobs(js)]
    _, sa = B.sa_jobs(OUT)
    jobs += [dict(j, batch="D1", part="SA") for j in sa]
    return jobs


def sources():
    have = {}
    for pk in ("pkg1", "pkg2", "pkg3"):
        p = os.path.join(REMOTE, pk, "top", "runs_0.json")
        for r in json.load(open(p)):
            have[r["name"]] = pk
    res = {}
    for j in all_jobs():
        n = j["name"]
        if not os.path.exists(os.path.join(SIMS, n + ".pkl.gz")):
            res[n] = None
        else:
            res[n] = have.get(n, "local")
    json.dump(res, open(os.path.join(OUT, "sources.json"), "w"), indent=1)
    c = collections.Counter(v for v in res.values())
    print(c)
    return res


def cname_of(cond):
    return cond


def collect():
    """(cond, arm) -> pooled solos / overlaps; non-MHA solos by (cond, arm)."""
    rows = collections.defaultdict(lambda: {"solo": {}, "soloD": {}, "ovs": [], "nonmha": {}, "jobs": []})
    sa, ffncheck, dup = [], [], {}
    audit = {"n": 0, "bad": 0}
    for j in all_jobs():
        p = os.path.join(SIMS, j["name"] + ".pkl.gz")
        if not os.path.exists(p):
            continue
        s = load(p)
        for r in s["runs"]:
            if "audit" in r:
                audit["n"] += r["audit"]["n"]
                audit["bad"] += r["audit"]["bad"]
        if j["part"] == "SA":
            sa.append((j, s))
            continue
        if j["kind"] == "ffncheck":
            ffncheck.append((j, s))
            continue
        for arm in j["arms"]:
            key = (j["cond"], arm)
            if j["batch"] == "D5" and j["cond"] == "N":     # D1 has the same -A3h cell: keep D1's, compare
                dup.setdefault(j["name"], s)
                continue
            R = rows[key]
            R["jobs"].append(j["name"])
            if j["kind"] == "mha":
                for r in s["runs"]:
                    if r["label"].startswith("soloP"):
                        R["solo"][r["seed"]] = r["out"][0]
                    elif r["label"].startswith("soloD"):
                        R["soloD"][r["seed"]] = r["out"][0]
                    elif r["kind"] == "ov":
                        R["ovs"].append(r)
            else:
                R["nonmha"][j["kind"]] = mean(r["out"][0] for r in s["runs"])
    # shared solos (rule 7 / D0 item 5): -DSTU-prio and rev pair with CRISP's P solo of the same state
    for (cond, arm), R in rows.items():
        if arm in ("-DSTU-prio", "rev") and not R["solo"] and (cond, "CRISP") in rows:
            R["solo"], R["soloD"] = rows[(cond, "CRISP")]["solo"], rows[(cond, "CRISP")]["soloD"]
            R["nonmha"] = rows[(cond, "CRISP")]["nonmha"]
        if arm in ("B-FULL",) and not R["nonmha"] and (cond, "B-R0") in rows and cond[0:1] != "P":
            pass
    return rows, sa, ffncheck, dup, audit


def cell(R, seeds):
    ovs = [r for r in R["ovs"] if r["seed"] in seeds]
    if not ovs:
        return None
    solo = {k: v for k, v in R["solo"].items() if k in seeds}
    soloD = [v for k, v in R["soloD"].items() if k <= 3]
    dl = [r["streams"][1]["C"] - r["tau"] for r in ovs]
    dp = [r["streams"][0]["C"] - solo[r["seed"]] for r in ovs if r["seed"] in solo]
    TD = mean(soloD) if soloD else None
    TP = mean(solo.values()) if solo else None
    kvw = [r.get("kv_window") for r in ovs]
    inw = sum(1 for r, w in zip(ovs, kvw) if w and w[0] is not None and w[0] <= r["tau"] + B.T_REF["D"] and r["tau"] <= w[1])
    kv_end = [w[1] for w in kvw if w and w[1] is not None]
    c = {"n": len(ovs), "D_lat": mean(dl), "D_lat_min": min(dl), "D_lat_max": max(dl), "T_D_self": TD, "T_P_mha_self": TP,
         "D_over_self": mean(dl) / TD - 1 if TD else None, "D_over_ref": mean(dl) / B.T_REF["D"] - 1,
         "Lhat_D": max((x - 1.5 * B.T_REF["D"]) / B.T_REF["D"] for x in dl),
         "Lhat_P": max((r["streams"][0]["C"] - 1.5 * B.T_REF["pf_mha"]) / B.T_REF["pf_mha"] for r in ovs),
         "viol_D": sum(1 for x in dl if x > 1.5 * B.T_REF["D"]), "viol_P": sum(1 for r in ovs if r["streams"][0]["C"] > 1.5 * B.T_REF["pf_mha"]),
         "in_window": inw, "kv_end_mean": mean(kv_end) if kv_end else None}
    if dp:
        c.update(dP=mean(dp), dP_min=min(dp), dP_max=max(dp))
    if solo:
        c["P_solo_range"] = max(solo.values()) - min(solo.values())
    return c


def extrapolate(R, c):
    """Second layer (rule 9; D_PLAN v2 section 6), labelled as an analytic extrapolation from single overlaps."""
    nm = R["nonmha"]
    if c is None or c["T_D_self"] is None or c["T_P_mha_self"] is None or not all(k in nm for k in ("pf_ln", "pf_proj", "pf_ffn")):
        return None
    T = {"pf_ln": nm["pf_ln"], "pf_mha": c["T_P_mha_self"], "pf_proj": nm["pf_proj"], "pf_ffn": nm["pf_ffn"]}
    ttft_iso = sum(COUNT[k] * T[k] for k in T)
    w_mha = COUNT["pf_mha"] * T["pf_mha"] / ttft_iso
    tpot_iso = c["T_D_self"]
    tpot_ov = tpot_iso + w_mha * (c["D_lat"] - tpot_iso)
    dP = c.get("dP", 0.0)
    ttft_ov_rule = ttft_iso + 8 * dP * (8 * T["pf_mha"] / tpot_ov)         # the formula as written in rule 9
    ttft_ov_once = ttft_iso + 8 * dP * (T["pf_mha"] / tpot_ov)             # 8 MHA tasks counted once (see the summary's note)
    out = {"TTFT_iso": ttft_iso, "TPOT_iso": tpot_iso, "TPOT_ov": tpot_ov, "w_MHA": w_mha, "TTFT_ov_rule": ttft_ov_rule,
           "TTFT_ov_once": ttft_ov_once, "TTFT_ov_dP0": ttft_iso}
    for n in OUTS:
        out[f"R_iso_{n}"] = ttft_iso + n * tpot_iso
        out[f"R_ov_{n}"] = ttft_ov_once + n * tpot_ov
        out[f"R_ov_dP0_{n}"] = ttft_iso + n * tpot_ov
    return out


def fmt(v, pct=False, dig=0):
    if v is None:
        return "—"
    if pct:
        return f"{100 * v:+.1f}%"
    return f"{v:,.{dig}f}".replace("-", "−")


def group_of(cond):
    return "N" if cond == "N" else cond.split("-")[0]


def main():
    rows, sa, ffncheck, dup, audit = collect()
    src = sources()
    res = {"audit": audit, "cells": {}, "supp": {}, "extra": {}}
    for (cond, arm), R in rows.items():
        k = f"{cond}|{arm}"
        res["cells"][k] = cell(R, {1, 2, 3})
        seeds = set(range(1, 11))
        overlap = {(r["seed"], r["tau"]) for r in R["ovs"]}
        full_grid = {(seed, tau) for seed in seeds for tau in B.TAUS}
        res["supp"][k] = cell(R, seeds) if overlap == full_grid and seeds <= R["solo"].keys() else None
        nonmha = R["nonmha"]
        if cond.startswith(("D-", "X-")) and ("N", arm) in rows:
            nonmha = {**rows[("N", arm)]["nonmha"], **nonmha}
        res["extra"][k] = extrapolate(dict(R, nonmha=nonmha), res["cells"][k])
        res["cells"][k] and res["cells"][k].update(nonmha=R["nonmha"])
    conds = sorted({c for c, _ in rows}, key=lambda c: (["N", "D", "P", "X"].index(group_of(c)), c))
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    L = ["# D 汇总：A3h + EIR + DSTU 联合实验（S-B 分组为主，S-A 缩小版另列）", "",
         "由 `python e2e/d_summary.py` 生成，只报数字。规则：`PROGRESS.md` 2026-11-11 的“比较规则”（1–11）与“补充规则”（种子 4–10 只作补充列）。方案：`D_PLAN.md` 第 2 版。单位：周期。", ""]
    c = collections.Counter(v for v in src.values())
    L += ["## 1. 结果来源与核对", "",
          "| 来源 | 作业数 |", "|---|---|",
          f"| 本机（本会话与用户在本机用同一代码续跑的部分） | {c.get('local', 0)} |",
          f"| 结果包 1（D3、D4、D5 的种子 1–3 主体，128 个并行的机器） | {c.get('pkg1', 0)} |",
          f"| 结果包 2（D2–D4 的种子 4–10 补充，云服务器） | {c.get('pkg2', 0)} |",
          f"| 结果包 3（D5 的种子 4–10 补充，云服务器） | {c.get('pkg3', 0)} |",
          f"| 缺 | {c.get(None, 0)} |", "",
          f"- 按跳发放审计：{audit['n']:,} 次，选中键不是最小键 {audit['bad']} 次。", ""]
    if dup:
        same = []
        for n, s in dup.items():
            d1 = load(os.path.join(SIMS, n.replace("D5_", "D1_", 1) + ".pkl.gz"))
            a = [(r["label"], r["out"]) for r in s["runs"]]
            b = [(r["label"], r["out"]) for r in d1["runs"]]
            same.append((n, a == b))
        L += [f"- 重复格核对：D5 的无故障 −A3h 作业与 D1 的同一格是同一配置，{sum(x[1] for x in same)}/{len(same)} 个作业逐个运行的总延迟相同"
              f"（{', '.join(n for n, x in same if not x) or '全部相同'}）。主表用 D1 的。", ""]
    # ---- main tables
    def table(title, key, pct, cs, arms, supp=False):
        S_ = res["supp"] if supp else res["cells"]
        out = [f"### {title}", "", "| 条件 | " + " | ".join(ARM_ZH[a] for a in arms) + " |", "|---|" + "---|" * len(arms)]
        for cc in cs:
            vals = []
            for a in arms:
                x = S_.get(f"{cc}|{a}")
                vals.append(fmt(x.get(key) if x else None, pct=pct, dig=3 if key.startswith("Lhat") else 0))
            out.append(f"| {cc} | " + " | ".join(vals) + " |")
        # median over the group's conditions
        if len(cs) > 1:
            meds = []
            for a in arms:
                v = [S_[f"{cc}|{a}"][key] for cc in cs if S_.get(f"{cc}|{a}") and S_[f"{cc}|{a}"].get(key) is not None]
                dg = 3 if key.startswith("Lhat") else 0
                meds.append(f"{fmt(median(v), pct=pct, dig=dg)} [{fmt(min(v), pct=pct, dig=dg)}, {fmt(max(v), pct=pct, dig=dg)}]" if v else "—")
            out.append("| 中位 [最小, 最大] | " + " | ".join(meds) + " |")
        return out + [""]
    groups = [("2. 无故障（D1）", [x for x in conds if x == "N"]), ("3. D 组故障（C 的坐标改写进 D 组）", [x for x in conds if x.startswith("D-")]),
              ("4. P 组故障（C 的坐标改写进 P 组）", [x for x in conds if x.startswith("P-")]), ("5. 组间 D2D 故障", [x for x in conds if x.startswith("X-")])]
    for title, cs in groups:
        L += [f"## {title}", ""]
        L += table("D 链延迟（pf_mha 单元，种子 1–3、3 个 g 的均值）", "D_lat", False, cs, ARMS)
        L += table("D 链延迟相对公共参照 T_D,ref = 355,201", "D_over_ref", True, cs, ARMS)
        L += table("D 链延迟相对本臂本条件的 D 链单独", "D_over_self", True, cs, ARMS)
        L += table("L̂_D 最大（公共截止时间 d = r + 1.5·T_D,ref）", "Lhat_D", False, cs, ARMS)
        L += table("P 单独（pf_mha，本臂本条件，种子 1–3 均值）", "T_P_mha_self", False, cs, ARMS)
        L += table("配对 ΔP 均值（种子 1–3）", "dP", False, cs, ARMS)
        L += table("补充列：配对 ΔP 均值（种子 1–10）", "dP", False, cs, ARMS, supp=True)
        L += table("补充列：P 单独的种子极差（种子 1–10）", "P_solo_range", False, cs, ARMS, supp=True)
    # Lhat formatting fix: Lhat values are ratios, shown with 3 decimals
    L = [ln for ln in L]
    # ---- second layer
    L += ["## 6. 第二层：基于单次重叠的解析外推（不是仿真结果）", "",
          "- TTFT_iso = 2·T_LN + 8·T_MHA + T_proj + T_FFN（本臂本条件的单独均值）；TPOT_iso = D 链单独均值。",
          "- TPOT′ = TPOT_iso + w_MHA·(D 链延迟 − TPOT_iso)，w_MHA = 8·T_MHA / TTFT_iso。",
          "- TTFT′：规则第 9 条写的是 TTFT_iso + 8·ΔP·(8·T_MHA / TPOT′)。按 D_PLAN 第 6 节的推导（每个 MHA 任务期间插入 T_MHA/TPOT′ 个 decode 步，共 8 个任务），"
          "8 个任务只应计一次，即 TTFT_iso + 8·ΔP·(T_MHA / TPOT′)；规则中的式子多乘了一个 8（事后发现的笔误，未改规则）。下表两者都列，并列 ΔP = 0 的值；"
          "种子 1–10 下，CRISP 与 B-R0 的 ΔP 均值各有 41/41 落入各自同条件的无 KV 对照经验范围。",
          "D 组与组间故障的非 MHA P 单独时间按 D_PLAN §5 使用同臂无故障结果。", ""]
    L += ["| 条件 | 臂 | TTFT_iso | TPOT_iso | TPOT′ | TTFT′（规则式） | TTFT′（计一次） | 请求 2048：R-iso | R-ov（计一次） | R-ov（ΔP = 0） |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for cc in [x for x in conds if x in ("N",) or x.endswith(("PF4", "PF6", "PCScratch", "MF5", "G1", "G2"))]:
        for a in ("B-R0", "CRISP", "-DSTU", "-A3h"):
            e = res["extra"].get(f"{cc}|{a}")
            if e:
                L.append(f"| {cc} | {ARM_ZH[a]} | {fmt(e['TTFT_iso'])} | {fmt(e['TPOT_iso'])} | {fmt(e['TPOT_ov'])} | {fmt(e['TTFT_ov_rule'])} | "
                         f"{fmt(e['TTFT_ov_once'])} | {fmt(e['R_iso_2048'])} | {fmt(e['R_ov_2048'])} | {fmt(e['R_ov_dP0_2048'])} |")
    L += ["", "全部条件与输出长度 128、512 的值在 `e2e\\d\\out\\prod\\summary.json` 的 `extra`。", ""]
    # ---- S-A reduced
    L += ["## 7. S-A 缩小版（P、D 同在一个 2×2 组；f = 0.4，τ = 0.4 × 该臂 P 单独均值；种子 1–3）", "",
          "| 条件 | 臂 | P | P 单独均值 | D 链单独均值 | D 链延迟（种子 1、2、3） | 相对 D 链单独 |", "|---|---|---|---|---|---|---|"]
    for j, s in sorted(sa, key=lambda x: x[0]["name"]):
        sp = [r["out"][0] for r in s["runs"] if r["label"].startswith("soloP")]
        sd = [r["out"][0] for r in s["runs"] if r["label"].startswith("soloD")]
        dl = [r["streams"][1]["C"] - r["tau"] for r in s["runs"] if r["kind"] == "ov"]
        L.append(f"| {j['cond']} | {ARM_ZH[j['arms'][0]]} | {j['kind'][3:]} | {fmt(mean(sp))} | {fmt(mean(sd))} | {', '.join(fmt(x) for x in dl)} | ×{mean(dl) / mean(sd):.1f} |")
    L += ["", "## 8. 非 MHA 核对单元（S-B，CRISP，pf_ffn，种子 1，τ = 0.5 × P 单独）", "",
          "| 条件 | P 单独 | P 重叠 | D 链单独 | D 链延迟 |", "|---|---|---|---|---|"]
    for j, s in ffncheck:
        R = {r["label"]: r for r in s["runs"]}
        ov = R["ov_fp0.5_s1"]
        L.append(f"| {j['cond']} | {fmt(R['soloP_s1']['out'][0])} | {fmt(ov['streams'][0]['C'])} | {fmt(R['soloD_s1']['out'][0])} | {fmt(ov['streams'][1]['C'] - ov['tau'])} |")
    L += [""]
    txt = "\n".join(L)
    # ratios (L-hat) printed with 3 decimals instead of integers
    with open(os.path.join(HERE, "D_SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("written", len(L), "lines")


if __name__ == "__main__":
    if sys.argv[1:] == ["sources"]:
        sources()
    else:
        main()
