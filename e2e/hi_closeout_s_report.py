"""Recompute S's factorial pairs and actual search/backend ledger from receipts."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics

import hi_closeout_s as run


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def stats(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values), mean=statistics.mean(values))


def plot(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"M": "#2563a6", "R": "#dd7f22", "S": "#34875a"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True, constrained_layout=True)
    selected = [r for r in rows if r["hardware"] == "wsc"]
    labels = [f"{r['factor']}: {r['baseline']} to {r['target']}" for r in selected]
    for ax, hw in zip(axes, ("wsc", "face")):
        group = [r for r in rows if r["hardware"] == hw]
        for index, r in enumerate(group):
            for metric, offset, marker in (("P_pct", -.14, "o"), ("D_pct", .14, "s")):
                y = index + offset
                ax.plot([r[metric + "_minimum"], r[metric + "_maximum"]], [y, y], color=colors[r["factor"]], alpha=.5)
                ax.scatter(r[metric + "_median"], y, color=colors[r["factor"]], marker=marker, s=28)
        ax.axvline(0, color="#777777", linewidth=.8)
        ax.set_title("WSC-LLM" if hw == "wsc" else "FACE")
        ax.set_xlabel("Paired change (%)")
        ax.set_yticks(range(len(labels)), labels)
        ax.grid(axis="x", alpha=.15)
        ax.invert_yaxis()
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="black", marker="o", linestyle="", label="P completion"),
                        Line2D([], [], color="black", marker="s", linestyle="", label="D latency")], loc="outside upper center", ncol=2)
    folder = destination / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(folder / ("s_conditional_increments." + extension), dpi=180)
    plt.close(fig)


def report(out, destination):
    manifest = run.READ(run.HERE / "s_manifest.json")
    records = {j["id"]: run.receipt(out, j["id"]) for j in manifest["jobs"]}
    assert len(records) == 408
    for job in manifest["jobs"]:
        record = records[job["id"]]
        assert record["status"] in ("ok", "reused") and record["spec"] == job["spec"]
    joints, repair, ledger = [], [], []
    for name, r in records.items():
        s = r["spec"]
        ledger.append(dict(job_id=name, kind=s["kind"], hardware=s["hardware_profile"], status=r["status"],
                           backend_calls=r.get("backend_calls", 0), search_proposals=r.get("search_proposals", 0),
                           wall_seconds=r["wall_seconds"], peak_rss_gb=r["peak_rss_gb"]))
        if s["kind"] in ("s_joint", "s_joint_reuse"):
            inner = r["resolved_simulation_spec"]
            assert inner["fresh_protocol"] == "fresh_transform_v1"
            p, d = run.receipt(out, s["P_solo"]), run.receipt(out, s["D_solo"])
            assert r["metrics"]["P_solo"] == p["completion"]["0"]
            assert r["metrics"]["D_solo"] == d["completion"]["1"]
            assert abs(r["metrics"]["D_interference"] - (r["metrics"]["D_latency"] / d["completion"]["1"] - 1)) < 1e-12
            expected_tau = int(round(inner["reference"]["KV_start"] + s["phase"] *
                                     (inner["reference"]["KV_end"] - inner["reference"]["KV_start"])))
            assert inner["tau"] == expected_tau
            joints.append(dict(job_id=name, hardware=s["hardware_profile"], MRS=s["arm_bits"],
                                phase=s["phase"], seed=s["seed"], status=r["status"],
                                backend_calls=r["backend_calls"], reference=inner["reference"], tau_D=inner["tau"],
                                KV_backend_start=r["kv_window"][0], KV_backend_end=r["kv_window"][1],
                                C4_deadlines=inner["frozen_C4_deadlines"], C4_rank=inner["rank"],
                                reused_from=r.get("reused_from", ""), **r["metrics"]))
        if s["kind"] == "s_rep" and r["status"] == "ok":
            assert r["search_proposals"] == s["steps"] and r["backend_calls"] == 10
            repair.append(dict(job_id=name, hardware=s["hardware_profile"], op=s["phase_op"], steps=s["steps"], seed=s["seed"],
                                healthy_A1_T=s["healthy_T"], T=r["T"], backend_mean=r["backend_mean"],
                                search_proposals=r["search_proposals"], migration=r["migration"],
                                calibration_seconds=r["timing"]["t_calibration"], search_seconds=r["timing"]["t_L2_search"],
                                finalize_seconds=r["timing"]["t_finalize"],
                                backend_offline_seconds=r["timing"]["t_backend_offline"]))
    assert len(joints) == 144 and sum(r["backend_calls"] for r in joints) == 108
    assert sum(r["status"] == "reused" for r in joints) == 36
    expected = {(hw, bits, phase, seed) for hw in ("wsc", "face")
                for bits in ("000", "001", "010", "011", "100", "101", "110", "111")
                for phase in (.1, .5, .9) for seed in (1, 2, 3)}
    bykey = {(r["hardware"], r["MRS"], r["phase"], r["seed"]): r for r in joints}
    assert set(bykey) == expected
    pairs = []
    for key, base in bykey.items():
        hw, bits, phase, seed = key
        for index, factor in enumerate(("M", "R", "S")):
            if bits[index] != "0":
                continue
            other = bits[:index] + "1" + bits[index + 1:]
            target = bykey[hw, other, phase, seed]
            pairs.append(dict(hardware=hw, factor=factor, baseline=bits, target=other, phase=phase, seed=seed,
                              P_pct=100 * (target["P_completion"] / base["P_completion"] - 1),
                              D_pct=100 * (target["D_latency"] / base["D_latency"] - 1),
                              D_interference_delta=target["D_interference"] - base["D_interference"],
                              Lhat_max_delta=target["Lhat_max"] - base["Lhat_max"]))
    summaries = []
    groups = collections.defaultdict(list)
    for row in pairs:
        groups[row["hardware"], row["factor"], row["baseline"], row["target"]].append(row)
    for (hw, factor, base, target), values in groups.items():
        summaries.append(dict(hardware=hw, factor=factor, baseline=base, target=target, n=len(values),
                              **{metric + "_" + k: v for metric in ("P_pct", "D_pct", "D_interference_delta", "Lhat_max_delta")
                                 for k, v in stats([r[metric] for r in values]).items()}))
    stops = []
    for hw in ("wsc", "face"):
        for op in run.joint.DC:
            prefix = f"S_{hw}_{op}_A1__"
            terminal = records[prefix + "stop300"]
            l1 = records[prefix + "l1"]
            relevant = [r for r in repair if r["hardware"] == hw and r["op"] == op]
            costs = l1["timing"]["t_R0"] + l1["timing"]["t_L1"] + sum(r["calibration_seconds"] + r["search_seconds"] + r["finalize_seconds"] for r in relevant)
            stops.append(dict(hardware=hw, op=op, selected_level=terminal["level"], stop_reason=terminal["reason"],
                              T_pre=l1["T_pre"], T_R0=l1["T_R0"], T_L1=l1["T_L1"], stop_mean_T=terminal["T"],
                              selected_source=terminal["source_job"], export=terminal["export"],
                              actual_L2_trajectories=len(relevant), research_five_seed_decision_seconds=costs,
                              all_L2_backend_seconds=sum(r["backend_offline_seconds"] for r in relevant)))
    destination = Path(destination)
    for name, rows in (("s_144_coverage", joints), ("s_conditional_pairs", pairs), ("s_conditional_summary", summaries),
                       ("s_a1_repair", repair), ("s_a1_stops", stops), ("s_execution_ledger", ledger)):
        table(destination / "tables" / (name + ".csv"), rows)
    summary = dict(status="passed", jobs=408, joint_logical=144, joint_new=108, joint_reused=36,
                   actual_L2=len(repair), actual_search_proposals=sum(r["search_proposals"] for r in repair),
                   actual_backend_calls=sum(r["backend_calls"] for r in ledger),
                   solo_logical=96, solo_actual=sum(r["backend_calls"] for r in ledger if r["kind"] == "s_solo"),
                   solo_reused=sum(r["status"] == "reused" for r in ledger if r["kind"] == "s_solo"),
                   process_hours=sum(r["wall_seconds"] for r in ledger) / 3600,
                   peak_task_rss_gb=max(r["peak_rss_gb"] for r in ledger))
    run.WRITE(destination / "s_completion_check.json", summary)
    plot(summaries, destination)
    lines = ["# GPT S 系统八组合", "", f"144 项联合已齐：108 项新增、36 项由 fresh_transform_v1 同身份复用。共 {summary['actual_L2']} 条实际 A1 修复搜索、{summary['actual_backend_calls']} 次新增后端；96 项逻辑配对单跑中实际运行 {summary['solo_actual']} 次。", "",
             "所有八臂使用健康 A1/original 的共同到达时刻与评价参考。A1-EIR 的停止条件使用自身健康 CPCM，C4 内部期限按各自当前图计算。原 300 步修复规则保持不变。", "",
             "| 硬件 | 增加能力 | 对比 | P 完成时间中位变化 | D 延迟中位变化 | D 延迟范围 |", "|---|---|---|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['hardware']} | {r['factor']} | {r['baseline']}→{r['target']} | {r['P_pct_median']:+.3f}% | {r['D_pct_median']:+.3f}% | [{r['D_pct_minimum']:+.3f}%, {r['D_pct_maximum']:+.3f}%] |")
    lines += ["", "每行含相同硬件、三相位、三种子的九个配对。M 是 A1→A3h，R 是 R0→EIR，S 是 original→All256+C4；S 同时改变切块、按跳服务与优先级，表示传输方案整体增量。", "",
              "成本表逐作业记录真实耗时、搜索提议数和后端调用数。研究五种子累计决策时间包含已执行各档校准/搜索/导出，十种子后端验证另外列示；这些并行时长不替代 J 阶段空闲单进程计时。", "",
              "所有三相位都依据共同 A1 参考释放；g=0.9 不代表每个方法自身 KV 已经完成90%。配对单跑只复用完整图、实际遍历顺序、平台、执行方式、rank、释放和种子完全相同的记录。", ""]
    (destination / "S_FACTORIAL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(run.OUT))
    parser.add_argument("--destination", default=str(run.HERE))
    args = parser.parse_args()
    print(json.dumps(report(run.fault.path(args.out), run.fault.path(args.destination))))
