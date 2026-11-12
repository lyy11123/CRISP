"""Rebuild the CRISP D-stage tables from Claude's frozen job plans and results.

Pass the Claude worktree's e2e directory with --e2e. A finished cloud D5
archive can be read directly with --d5-archive.
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import itertools
import json
from pathlib import Path
import shutil
import statistics
import sys
import tarfile
import tempfile


BATCHES = ("D1", "D2", "D3", "D4", "D5")
GROUPS = {"N": "无故障", "D": "D 组故障", "P": "P 组故障", "X": "组间 D2D 故障"}
BASE_SEEDS = set(range(1, 4))
SUPP_SEEDS = set(range(1, 11))
BASELINES = ("B-R0", "B-FULL", "-DSTU", "-DSTU-prio", "-EIR", "-A3h", "rev")
SOURCE_NAMES = {"local": "本机", "pkg1": "D3–D5 云端主体", "pkg2": "D2–D4 云端补充", "pkg3": "D5 云端补充", "D5-cloud": "D5 云端补充"}


def read_jobs(prod):
    jobs = []
    for batch in BATCHES:
        plan = json.loads((prod / "batches" / batch / "plan.json").read_text(encoding="utf-8"))
        for job in plan["jobs"]:
            part = "base"
            if job["kind"].startswith("SA_"):
                part = "SA"
            elif job["name"].endswith(("_x0", "_x1", "_x2")):
                part = "ext"
            jobs.append(dict(job, batch=batch, part=part))
    return jobs


def stage_results(prod, jobs, stage, d5_archive):
    sources_file = prod / "sources.json"
    previous = json.loads(sources_file.read_text(encoding="utf-8")) if sources_file.is_file() else {}
    sources = {}
    for job in jobs:
        name = job["name"]
        source = prod / "sims" / f"{name}.pkl.gz"
        if source.is_file():
            shutil.copyfile(source, stage / source.name)
            sources[name] = previous.get(name) or "local"
    if d5_archive:
        wanted = {f"out/sims/{job['name']}.pkl.gz": job["name"] for job in jobs
                  if job["batch"] == "D5" and job["part"] == "ext"}
        with tarfile.open(d5_archive, "r:gz") as archive:
            for member in archive:
                name = wanted.get(member.name)
                if name is not None:
                    with archive.extractfile(member) as data, (stage / f"{name}.pkl.gz").open("wb") as sink:
                        shutil.copyfileobj(data, sink)
                    sources[name] = "D5-cloud"
    return sources


def grid_complete(row, seeds, taus):
    expected = sorted(itertools.product(seeds, taus))
    observed = sorted((r["seed"], r["tau"]) for r in row["ovs"] if r["seed"] in seeds)
    return expected == observed and seeds <= row["solo"].keys() and BASE_SEEDS <= row["soloD"].keys()


def units(row, settings, base):
    times = dict(zip(settings.TAUS, settings.G))
    result = []
    for run in sorted(row["ovs"], key=lambda r: (r["tau"], r["seed"])):
        seed, tau = run["seed"], run["tau"]
        cp = run["streams"][0]["C"]
        cd = run["streams"][1]["C"]
        solo_p = row["solo"].get(seed)
        ref_p, ref_d = settings.T_REF["pf_mha"], settings.T_REF["D"]
        lp = cp / ref_p - 1.5
        ld = (cd - tau) / ref_d - 1.5
        window = run.get("kv_window")
        kv_start, kv_end = window or (None, None)
        result.append({
            "seed": seed, "g": times[tau], "tau": tau, "C_P": cp, "C_D": cd,
            "D_lat": cd - tau, "P_solo": solo_p, "dP": cp - solo_p if solo_p is not None else None,
            "kv_start": kv_start, "kv_end": kv_end,
            "tau_inside_kv": bool(kv_start <= tau <= kv_end) if kv_start is not None and kv_end is not None else None,
            "Lhat_P": lp, "Lhat_D": ld, "Lhat_max": max(lp, ld),
            "violations": int(lp > 0) + int(ld > 0),
            "Lhat_P_self": cp / base["T_P_mha_self"] - 1.5 if base else None,
            "Lhat_D_self": (cd - tau) / base["T_D_self"] - 1.5 if base else None,
        })
    return result


def comparisons(cells, conditions, metric, baselines=BASELINES):
    output = {}
    for group, members in conditions.items():
        output[group] = {}
        for arm in baselines:
            changes = {}
            for cond in members:
                crisp, baseline = cells.get(f"{cond}|CRISP"), cells.get(f"{cond}|{arm}")
                if crisp and baseline:
                    changes[cond] = crisp[metric] / baseline[metric] - 1
            values = list(changes.values())
            if values:
                output[group][arm] = {
                    "median": statistics.median(values), "min": min(values), "max": max(values),
                    "lower": sum(v < 0 for v in values), "equal": sum(v == 0 for v in values),
                    "higher": sum(v > 0 for v in values), "per_condition": changes,
                }
    return output


def noise_ranges(cells, conditions):
    output = {}
    for arm in ("B-R0", "CRISP"):
        rows = []
        for cond in conditions:
            measured, control = cells.get(f"{cond}|{arm}"), cells.get(f"{cond}|noKV-{arm}")
            if measured and control:
                rows.append({"condition": cond, "mean_dP": measured["dP"], "noKV_min": control["dP_min"],
                             "noKV_max": control["dP_max"], "within": control["dP_min"] <= measured["dP"] <= control["dP_max"]})
        output[arm] = rows
    return output


def summarize(summary, jobs, sources):
    rows, sa, ffncheck, duplicate, audit = summary.collect()
    data = {"audit": audit, "cells": {}, "supplement": {}, "coverage": {}, "units": {}, "extrapolation": {},
            "nonmha_P": {}, "reference_cycles": summary.B.T_REF, "g_to_tau": dict(zip(summary.B.G, summary.B.TAUS))}
    data["jobs"] = {b: {part: dict(planned=sum(j["batch"] == b and j["part"] == part for j in jobs),
                                   completed=sum(j["batch"] == b and j["part"] == part and j["name"] in sources for j in jobs))
                        for part in ("base", "ext", "SA")} for b in BATCHES}
    data["sources"] = sources
    data["missing_jobs"] = [j["name"] for j in jobs if j["name"] not in sources]
    for (cond, arm), row in sorted(rows.items()):
        key = f"{cond}|{arm}"
        main_ok = grid_complete(row, BASE_SEEDS, summary.B.TAUS)
        supp_ok = grid_complete(row, SUPP_SEEDS, summary.B.TAUS)
        base = summary.cell(row, BASE_SEEDS) if main_ok else None
        supp = summary.cell(row, SUPP_SEEDS) if supp_ok else None
        for cell in (base, supp):
            if cell:
                cell["Lhat_max"] = max(cell["Lhat_D"], cell["Lhat_P"])
        data["cells"][key], data["supplement"][key] = base, supp
        data["coverage"][key] = {"overlap_count": len(row["ovs"]), "P_solo_seeds": sorted(row["solo"]),
                                 "main_complete": main_ok, "supplement_complete": supp_ok}
        data["units"][key] = units(row, summary.B, base)
        nonmha = row["nonmha"]
        if cond.startswith(("D-", "X-")) and ("N", arm) in rows:
            nonmha = {**rows[("N", arm)]["nonmha"], **nonmha}
        data["nonmha_P"][key] = nonmha
        data["extrapolation"][key] = summary.extrapolate(dict(row, nonmha=nonmha), base)
    conditions = sorted({cond for cond, _ in rows})
    grouped = {g: [c for c in conditions if summary.group_of(c) == g] for g in GROUPS}
    condition_groups = {"all": conditions, **grouped}
    data["comparisons"] = comparisons(data["cells"], condition_groups, "D_lat")
    data["D5_supplement_comparisons"] = comparisons(data["supplement"], condition_groups, "D_lat", ("-A3h",))
    data["P_solo_comparisons"] = comparisons(data["cells"], condition_groups, "T_P_mha_self", ("B-R0", "-A3h"))
    data["request_comparisons"] = {
        metric: comparisons(data["extrapolation"], {"all": conditions}, metric, ("B-R0",))["all"].get("B-R0")
        for metric in ("TTFT_iso", "TPOT_ov", "R_ov_128", "R_ov_512", "R_ov_2048", "R_ov_dP0_2048")
    }
    data["noKV_ranges"] = {"main": noise_ranges(data["cells"], conditions),
                           "supplement": noise_ranges(data["supplement"], conditions)}
    data["SA"] = [{"job": j["name"], "condition": j["cond"], "arms": j["arms"], "op": j["kind"][3:],
                   "D_solo_mean": statistics.mean(r["out"][0] for r in s["runs"] if r["label"].startswith("soloD")),
                   "D_overlap": [r["streams"][1]["C"] - r["tau"] for r in s["runs"] if r["kind"] == "ov"]}
                  for j, s in sorted(sa, key=lambda pair: pair[0]["name"])]
    data["ffncheck"] = [{"job": j["name"], "condition": j["cond"], "runs": [
        {"label": r["label"], "out": r["out"], "tau": r.get("tau"),
         "completion": {k: v["C"] for k, v in r["streams"].items()}} for r in s["runs"]]} for j, s in ffncheck]
    data["D5_N_repeats"] = {}
    for name, sink in duplicate.items():
        d1 = Path(summary.SIMS) / (name.replace("D5_", "D1_", 1) + ".pkl.gz")
        if d1.is_file():
            get_runs = lambda s: [(r["label"], r["out"], {k: v["C"] for k, v in r["streams"].items()}) for r in s["runs"]]
            data["D5_N_repeats"][name] = get_runs(sink) == get_runs(summary.load(d1))
    return data


def write_report(data, target, summary):
    source_counts = Counter(data["sources"].values())
    date = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    main_n = sum(cell is not None for cell in data["cells"].values())
    supp_n = sum(cell is not None for cell in data["supplement"].values())
    lines = ["# CRISP D 阶段结果汇总", "", "## 数据与口径", "",
             f"{date} 依据 D 阶段作业清单和落盘结果重新汇总。", "",
             "| 来源 | 已落盘作业 |", "|---|---:|"]
    lines.extend(f"| {SOURCE_NAMES.get(source, source)} | {n} |" for source, n in sorted(source_counts.items()))
    lines.extend([f"| 待落盘 | {len(data['missing_jobs'])} |", "",
                  f"S-B 主表有 {main_n} 个完整条件–实验臂单元格，每格用种子 1–3 和 3 个 g。"
                  f"种子 1–10 补充列有 {supp_n} 个完整单元格，每格使用 30 次重叠和 10 个配对 P 单独种子。",
                  f"按跳审计共 {data['audit']['n']:,} 次，发现 {data['audit']['bad']} 次仲裁键差异。"])
    repeats = data["D5_N_repeats"]
    if repeats:
        lines.append(f"无故障 −A3h 重复运行中，{sum(repeats.values())}/{len(repeats)} 个作业的逐次完成时间与 D1 一致。")
    lines.extend(["",
                  "## S-B 主表的同条件比较", "",
                  "每个条件先计算 CRISP 的 D 链延迟相对变化，再对该组条件取中位数和范围。方向按 CRISP 的延迟计。", "",
                  "| 条件组 | 对照臂 | 中位变化 | 最小–最大变化 | 更低 / 相同 / 更高 |",
                  "|---|---|---:|---:|---:|"])
    for group, comparisons_by_arm in data["comparisons"].items():
        for arm, row in comparisons_by_arm.items():
            lines.append(f"| {GROUPS.get(group, '全部条件')} | {summary.ARM_ZH[arm]} | {row['median']:+.2%} | "
                         f"{row['min']:+.2%} 至 {row['max']:+.2%} | {row['lower']} / {row['equal']} / {row['higher']} |")
    healthy = data["cells"].get("N|CRISP")
    healthy_baseline = data["cells"].get("N|B-R0")
    if healthy and healthy_baseline:
        lines.extend(["", f"无故障时，D 链延迟从 {healthy_baseline['D_lat']:,.0f} 降至 {healthy['D_lat']:,.0f} 周期。"
                      f"P 单独 MHA 从 {healthy_baseline['T_P_mha_self']:,.0f} 增至 {healthy['T_P_mha_self']:,.0f} 周期。"])
    p_solo = data["P_solo_comparisons"]["all"].get("B-R0")
    if p_solo:
        lines.extend(["", f"P 单独 MHA 相对 B-R0 的同条件变化中位数为 {p_solo['median']:+.2%}，"
                      f"范围 {p_solo['min']:+.2%} 至 {p_solo['max']:+.2%}，"
                      f"延迟更高 {p_solo['higher']}/{len(p_solo['per_condition'])} 个条件。"])
    lines.extend(["", "| 主表实验臂 | 重叠单元数 | D 超过公共截止时间 | P 超过公共截止时间 |",
                  "|---|---:|---:|---:|"])
    for arm in ("B-R0", "B-FULL", "CRISP", "-DSTU", "-DSTU-prio", "-EIR", "-A3h"):
        selected = [cell for key, cell in data["cells"].items() if key.endswith(f"|{arm}") and cell]
        lines.append(f"| {summary.ARM_ZH[arm]} | {sum(c['n'] for c in selected)} | "
                     f"{sum(c['viol_D'] for c in selected)} | {sum(c['viol_P'] for c in selected)} |")
    lines.extend(["", "## 配对 ΔP 与无 KV 对照", "",
                  "| 种子 | 实验臂 | ΔP 均值落在同条件无 KV 经验范围内 |",
                  "|---|---|---:|"])
    for label, name in (("1–3", "main"), ("1–10", "supplement")):
        for arm, rows in data["noKV_ranges"][name].items():
            lines.append(f"| {label} | {arm} | {sum(r['within'] for r in rows)}/{len(rows)} |")
    d5 = data["D5_supplement_comparisons"]["all"].get("-A3h")
    if d5:
        lines.extend(["", f"种子 1–10 中，CRISP 的 D 链延迟相对同条件 −A3h 的中位变化为 {d5['median']:+.2%}，"
                      f"范围 {d5['min']:+.2%} 至 {d5['max']:+.2%}，更低 {d5['lower']}/{len(d5['per_condition'])} 个条件。"])
    if data["SA"]:
        lines.extend(["", "## S-A 共置", ""])
        for op in ("pf_mha", "pf_ffn"):
            ratios = [statistics.mean(s["D_overlap"]) / s["D_solo_mean"] for s in data["SA"] if s["op"] == op]
            if ratios:
                lines.append(f"- P 为 `{op}` 时，{len(ratios)} 个作业的 D 链平均延迟为同臂单独运行的 "
                             f"{min(ratios):.1f}–{max(ratios):.1f} 倍。")
    extrapolations = [x for x in data["extrapolation"].values() if x]
    nonpositive = sum(x["TTFT_ov_rule"] <= 0 for x in extrapolations)
    lines.extend(["", "## 逐单元与请求延迟", "",
                  "JSON 保存每个条件、实验臂、种子和 g 的 D 链延迟、配对 ΔP、KV 时间及公共与自身迟约。"
                  "自身迟约以种子 1–3 的单独均值为参照。", "",
                  "请求延迟、TTFT 和 TPOT 位于 `extrapolation`，由单次重叠结果解析外推。D 组与组间故障的非 MHA P 单独按 D_PLAN §5 复用同臂无故障值。"
                  "下表按 D_PLAN 的推导将 8 个 MHA 任务计一次，同时列出 ΔP 取零的请求延迟。"
                  f"原规则式重复乘了 8，产生 {nonpositive}/{len(extrapolations)} 个非正 TTFT 值，保留在 `TTFT_ov_rule` 供复算。", "",
                  "| 解析外推量 | CRISP 相对 B-R0 的同条件中位变化 | 条件数 |", "|---|---:|---:|"])
    labels = {"TTFT_iso": "单独 TTFT", "TPOT_ov": "重叠 TPOT′", "R_ov_128": "128 token 请求（MHA 计一次）",
              "R_ov_512": "512 token 请求（MHA 计一次）", "R_ov_2048": "2048 token 请求（MHA 计一次）",
              "R_ov_dP0_2048": "2048 token 请求（ΔP = 0）"}
    for key, row in data["request_comparisons"].items():
        if row:
            lines.append(f"| {labels[key]} | {row['median']:+.2%} | {len(row['per_condition'])} |")
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--d5-archive", type=Path)
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.e2e.resolve()))
    import d_summary as summary

    prod = args.e2e / "d/out/prod"
    jobs = read_jobs(prod)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crisp_d_sims_", dir=args.scratch) as temporary:
        stage = Path(temporary)
        sources = stage_results(prod, jobs, stage, args.d5_archive)
        summary.SIMS = str(stage)
        summary.all_jobs = lambda: jobs
        data = summarize(summary, jobs, sources)
    json_path = args.out_dir / "CRISP_D_STAGE.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, default=float, separators=(",", ":"))
        stream.write("\n")
    report_path = args.out_dir / "CRISP_D_STAGE.md"
    write_report(data, report_path, summary)
    print(f"{len(sources)}/{len(jobs)} results | {sum(v is not None for v in data['cells'].values())} main cells | "
          f"{sum(v is not None for v in data['supplement'].values())} supplemental cells")
    print(report_path)
    print(json_path)


if __name__ == "__main__":
    main()
