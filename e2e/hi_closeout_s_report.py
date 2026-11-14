"""Recompute S's factorial pairs and actual search/backend ledger from receipts."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics

import hi_closeout_s as run
import hi_closeout_audit as audit

ARMS = {"000": ("A1", "R0", "original"), "001": ("A1", "R0", "All256+C4"),
        "010": ("A1", "EIR", "original"), "011": ("A1", "EIR", "All256+C4"),
        "100": ("A3h", "R0", "original"), "101": ("A3h", "R0", "All256+C4"),
        "110": ("A3h", "EIR", "original"), "111": ("A3h", "EIR", "All256+C4")}


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


def source_checks(records, out):
    """Check saved sources and the two different uses of CPCM, without a replay."""
    checks, mapping, cost_rows = [], [], []
    for hw in ("wsc", "face"):
        cid = f"F_{hw}_2x2_fixed_d2d1"
        reference = run.READ(run.OLD / "results" / (cid + "__joint_reference.json"))
        common = {k: reference[k] for k in ("T_P", "T_D", "KV_start", "KV_end")}
        for state in run.STATES:
            prepared = records[f"S_{hw}__graph_{state}"]
            assert audit.sha(audit.local(prepared["graph_record"])) == prepared["graph_sha256"]
            graph = run.READ(prepared["graph_record"])
            for stream in (0, 1):
                assert audit.sha(audit.local(graph["files"][f"orig_{stream}"])) == prepared["graph_files_sha256"][str(stream)]
            method = "A1" if state.startswith("A1") else "A3h"
            p_source = run.READ(run.OLD / "results" / (cid + f"__pre_pf_mha_{method}.json"))["exports"]["healthy"]
            assert graph["source_exports"]["P"] == p_source
            for op in run.joint.DC:
                if state == "A1EIR":
                    stop = records[f"S_{hw}_{op}_A1__stop300"]
                    assert graph["source_exports"][op] == stop["export"]
                    selected = records[stop["source_job"]]
                    level = stop["level"]
                    l1 = records[f"S_{hw}_{op}_A1__l1"]
                    healthy = run.READ(run.OLD / "results" / (cid + f"__pre_{op}_A1.json"))
                    assert l1["L1_evaluated"] and l1["backend_calls"] == 20
                    assert l1["T_pre"] == l1["healthy_T"] == l1["spec"]["healthy_T"] == healthy["T"]
                    assert l1["stopping_reference"] == "own healthy A1 deterministic CPCM"
                elif state == "A3hEIR":
                    stop = run.READ(run.OLD / "results" / (cid + f"__stop_{op}_PF4_300.json"))
                    assert graph["source_exports"][op] == stop["export"]
                    selected = run.READ(run.OLD / "results" / (stop["source_job"] + ".json"))
                    level = stop["level"]
                else:
                    selected = run.READ(run.OLD / "results" / (cid + f"__l1_{op}_{method}_PF4.json"))
                    assert graph["source_exports"][op] == selected["exports"]["R0"]
                    level = "R0"
                label = "R0" if level == "R0" else "L1" if level == "L1" else None
                migration = selected["migration_R0" if label == "R0" else "migration"]
                mapping.append(dict(hardware=hw, state=state, op=op, selected_level=level,
                    selected_source=selected["job_id"], selected_export=graph["source_exports"][op],
                    selected_backend_mean_ns=selected[label + "_backend_mean"] if label else selected["backend_mean"],
                    selected_T_hat_ns=selected["T_" + label] if label else selected["T"],
                    source_scope="new A1 EIR" if state == "A1EIR" else "existing F/G state",
                    **migration))
            for r in records.values():
                s = r["spec"]
                if s.get("hardware_profile") != hw or s.get("mapping_state") != state or "resolved_simulation_spec" not in r:
                    continue
                inner = r["resolved_simulation_spec"]
                assert inner["reference"] == common
                phase = s.get("phase", .1)
                tau = int(round(common["KV_start"] + phase * (common["KV_end"] - common["KV_start"])))
                deadlines = {str(i): 1.5 * graph["that"][str(i)] + (tau if i else 0) for i in (0, 1)}
                order = sorted(deadlines, key=lambda i: (deadlines[i], i))
                assert inner["tau"] == tau and inner["frozen_C4_deadlines"] == deadlines
                assert inner["rank"] == {i: order.index(i) for i in order}
                checks.append(dict(job_id=s["job_id"], mapping_state=state, shared_reference=common,
                    current_graph_T_hat=graph["that"], tau=tau, C4_deadlines=deadlines, C4_order=order))
        # A3h costs are recorded historical work, not charged to this closeout.
        for method in ("A1", "A3h"):
            for op in run.joint.DC:
                l1 = (records[f"S_{hw}_{op}_A1__l1"] if method == "A1" else
                      run.READ(run.OLD / "results" / (cid + f"__l1_{op}_A3h_PF4.json")))
                raw = audit.raw_read(l1["sink"])
                reps = ([r for r in records.values() if r["kind"] == "s_rep" and r["status"] == "ok"
                         and r["spec"]["hardware_profile"] == hw and r["spec"]["phase_op"] == op] if method == "A1" else
                        [run.READ(p) for p in (run.OLD / "results").glob(cid + f"__rep_{op}_PF4_s*_r*.json")])
                reps = [r for r in reps if r["status"] == "ok"]
                restore = sum(audit.raw_read(r["sink"])["search"]["t_restore_best"] for r in reps)
                decision = raw["t_R0"] + raw["t_L1"] + restore + sum(
                    r["timing"]["t_calibration"] + r["timing"]["t_L2_search"] + r["timing"]["t_finalize"] for r in reps)
                cost_rows.append(dict(hardware=hw, method=method, op=op,
                    source_scope="new closeout work" if method == "A1" else "historical F/G work",
                    actual_L2=len(reps), search_proposals=sum(r["steps"] for r in reps),
                    restore_best_seconds=restore, research_five_seed_decision_seconds=decision,
                    backend_calls=20 + 10*len(reps),
                    backend_validation_seconds=raw["R0"]["t_backend"] + raw["L1"]["t_backend"] +
                        sum(r["timing"]["t_backend_offline"] for r in reps)))
    assert len(checks) == 240 and len(mapping) == 32
    return checks, mapping, cost_rows


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
    axes[0].invert_yaxis()
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
    source_evidence, mapping, mapping_costs = source_checks(records, out)
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
    summaries.sort(key=lambda r: (("wsc", "face").index(r["hardware"]), "MRS".index(r["factor"]), r["baseline"]))
    phase_rows = []
    for (hw, factor, base, target), values in groups.items():
        for phase in (.1, .5, .9):
            subset = [r for r in values if r["phase"] == phase]
            phase_rows.append(dict(hardware=hw, factor=factor, baseline=base, target=target, phase=phase, n=len(subset),
                **{metric + "_" + k:v for metric in ("P_pct", "D_pct") for k,v in stats([r[metric] for r in subset]).items()}))
    arm_rows, overall_rows = [], []
    for hw in ("wsc", "face"):
        for bits in ARMS:
            subset = [r for r in joints if r["hardware"] == hw and r["MRS"] == bits]
            arm_rows.append(dict(hardware=hw, MRS=bits, n=len(subset),
                new_backend_calls=sum(r["backend_calls"] for r in subset),
                current_joint_task_seconds=sum(records[r["job_id"]]["wall_seconds"] for r in subset),
                **{metric + "_" + k:v for metric in ("P_completion", "D_latency", "Lhat_max", "D_interference")
                   for k,v in stats([r[metric] for r in subset]).items()}))
        for bits in ("110", "111"):
            changes = [{metric:100*(bykey[hw,bits,p,s][field]/bykey[hw,"000",p,s][field]-1)
                        for metric,field in (("P_pct","P_completion"),("D_pct","D_latency"))}
                       for p in (.1,.5,.9) for s in (1,2,3)]
            overall_rows.append(dict(hardware=hw, baseline="000", target=bits,
                P_improved=sum(r["P_pct"] < 0 for r in changes), D_improved=sum(r["D_pct"] < 0 for r in changes),
                **{metric + "_" + k:v for metric in ("P_pct", "D_pct") for k,v in stats([r[metric] for r in changes]).items()}))
    stops = []
    for hw in ("wsc", "face"):
        for op in run.joint.DC:
            prefix = f"S_{hw}_{op}_A1__"
            terminal = records[prefix + "stop300"]
            l1 = records[prefix + "l1"]
            relevant = [r for r in repair if r["hardware"] == hw and r["op"] == op]
            cost = next(r for r in mapping_costs if (r["hardware"],r["method"],r["op"]) == (hw,"A1",op))
            stops.append(dict(hardware=hw, op=op, selected_level=terminal["level"], stop_reason=terminal["reason"],
                              T_pre=l1["T_pre"], T_R0=l1["T_R0"], T_L1=l1["T_L1"], stop_mean_T=terminal["T"],
                              selected_source=terminal["source_job"], export=terminal["export"],
                              actual_L2_trajectories=len(relevant), research_five_seed_decision_seconds=cost["research_five_seed_decision_seconds"],
                              restore_best_seconds=cost["restore_best_seconds"],
                              all_L2_backend_seconds=sum(r["backend_offline_seconds"] for r in relevant)))
    destination = Path(destination)
    for name, rows in (("s_144_coverage", joints), ("s_conditional_pairs", pairs), ("s_conditional_summary", summaries),
                       ("s_a1_repair", repair), ("s_a1_stops", stops), ("s_execution_ledger", ledger),
                       ("s_mapping_states", mapping), ("s_mapping_costs", mapping_costs),
                       ("s_arm_summary", arm_rows), ("s_full_vs_baseline", overall_rows),
                       ("s_phase_conditionals", phase_rows)):
        table(destination / "tables" / (name + ".csv"), rows)
    summary = dict(status="passed", jobs=408, joint_logical=144, joint_new=108, joint_reused=36,
                   actual_L2=len(repair), actual_search_proposals=sum(r["search_proposals"] for r in repair),
                   actual_backend_calls=sum(r["backend_calls"] for r in ledger),
                   solo_logical=96, solo_actual=sum(r["backend_calls"] for r in ledger if r["kind"] == "s_solo"),
                   solo_reused=sum(r["status"] == "reused" for r in ledger if r["kind"] == "s_solo"),
                   process_hours=sum(r["wall_seconds"] for r in ledger) / 3600,
                   peak_task_rss_gb=max(r["peak_rss_gb"] for r in ledger))
    run.WRITE(destination / "s_completion_check.json", summary)
    run.WRITE(destination / "evidence/s_source_reference_audit.json", dict(status="passed", graph_sources=8,
        operator_state_exports=32, reference_and_C4_checked_jobs=len(source_evidence), new_backend_calls=0,
        new_searches=0, checks=source_evidence))
    plot(summaries, destination)
    lines = ["# GPT S 系统八组合", "", f"144 项联合已齐：108 项新增、36 项由 fresh_transform_v1 同身份复用。共 {summary['actual_L2']} 条实际 A1 修复搜索、{summary['actual_backend_calls']} 次新增后端；96 项逻辑配对单跑中实际运行 {summary['solo_actual']} 次，另24项按实际图及执行身份复用。", "",
             "三位数字依次表示映射、修复和传输选择。P 是一个 prefill MHA 流，D 是 LN、8个代表MHA、projection与FFN组成的decode链；PF4故障施加于D侧。M切换时P和D映射一起切换；R只修复D侧；S采用All256+C4。", "",
             "| 组合 | 映射 | 故障后状态 | 传输 |", "|---|---|---|---|"]
    for bits, (method,repair_mode,transport) in ARMS.items():
        lines.append(f"| {bits} | {method} | {repair_mode} | {transport} |")
    lines += ["", "A3h映射和EIR均明显缩短D链；All256+C4的增量取决于硬件和到达相位，并伴随P完成时间增加。FACE上已有A3h+EIR时，110→111的D延迟中位数反而增加3.679%，P增加9.743%；应同时保留110和111的结果。", "",
              "| 硬件 | 相对000 | P中位变化 [最小,最大] | D中位变化 [最小,最大] | P改善数/9 | D改善数/9 |",
              "|---|---|---:|---:|---:|---:|"]
    for r in overall_rows:
        lines.append(f"| {r['hardware']} | {r['target']} | {r['P_pct_median']:+.3f}% [{r['P_pct_minimum']:+.3f}%, {r['P_pct_maximum']:+.3f}%] | {r['D_pct_median']:+.3f}% [{r['D_pct_minimum']:+.3f}%, {r['D_pct_maximum']:+.3f}%] | {r['P_improved']} | {r['D_improved']} |")
    lines += ["", "所有八臂使用健康 A1/original 的共同到达时刻与评价参考。A1-EIR的停止条件使用自身健康CPCM，C4内部期限按各自当前All256图计算，即dP=1.5×T̂P、dD=τ+1.5×T̂D。共同参考决定τ与评价分母，当前图决定C4优先级。已逐项核对8个图来源、32个算子状态和240个联合/单跑配置。", "",
             "| 硬件 | 增加能力 | 对比 | P 完成时间中位变化 | D 延迟中位变化 | D 延迟范围 |", "|---|---|---|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['hardware']} | {r['factor']} | {r['baseline']}→{r['target']} | {r['P_pct_median']:+.3f}% | {r['D_pct_median']:+.3f}% | [{r['D_pct_minimum']:+.3f}%, {r['D_pct_maximum']:+.3f}%] |")
    lines += ["", "每行含相同硬件、三相位、三种子的九个配对。M 是 A1→A3h，R 是 R0→EIR，S 是 original→All256+C4；S 同时改变切块、按跳服务与优先级，表示传输方案整体增量。百分比按每对结果计算后再取中位数，不把各能力的百分比相加。", "",
              "WSC四种条件下M的D延迟中位变化为−64.937%至−44.799%，R为−50.979%至−20.705%；FACE的M为−44.054%至−31.609%，R为−20.446%至−5.128%。FACE的101→111有一个配对D延迟增加0.075%，其余R配对均改善。P的变化独立列示。", "",
              "![各能力在不同已有配置下的配对增量](figures/s_conditional_increments.png)", "",
              "All256+C4在WSC的110→111中位D改善7.224%，但该条件g=0.9的三个种子均回退；FACE同一条件九个配对中八个D回退。下表按到达相位拆开，展示共同参考时间轴上的差异。", "",
              "| 硬件 | 110→111到达相位g | P中位变化 | D中位变化 [最小,最大] |", "|---|---:|---:|---:|"]
    for r in phase_rows:
        if r["baseline"] == "110" and r["target"] == "111":
            lines.append(f"| {r['hardware']} | {r['phase']} | {r['P_pct_median']:+.3f}% | {r['D_pct_median']:+.3f}% [{r['D_pct_minimum']:+.3f}%, {r['D_pct_maximum']:+.3f}%] |")
    lines += ["", "A1的EIR是实际完成的L1与30/100/300步独立修复。WSC的MHA在300步档停止，projection和FFN在100步档停止；FACE三者在30步档停止，两种硬件的LN均保留R0。八个算子最终都按自身健康A1的2%条件停止。停止判断用该档五轨迹平均CPCM，联合图取该档后端中位代表，两项分别保存。", "",
              "| 硬件 | 算子 | 停止档 | 实际L2数 | 五种子累计决策秒 | 后端验证秒 | 最终新增权重副本MiB | 搬迁行为比例 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in stops:
        m = next(v for v in mapping if (v["hardware"],v["state"],v["op"]) == (r["hardware"],"A1EIR",r["op"]))
        c = next(v for v in mapping_costs if (v["hardware"],v["method"],v["op"]) == (r["hardware"],"A1",r["op"]))
        lines.append(f"| {r['hardware']} | {r['op']} | {r['selected_level']} | {r['actual_L2_trajectories']} | {c['research_five_seed_decision_seconds']:.3f} | {c['backend_validation_seconds']:.3f} | {m['unique_block_bytes']/2**20:.3f} | {m['moved_behas_share']:.2%} |")
    lines += ["", f"本轮S作业体累计{summary['process_hours']:.5f}进程小时、最高RSS {summary['peak_task_rss_gb']:.3f} GiB。840次新增后端 = 8×20次A1 R0/L1 + 50×10次实际L2 + 72次配对单跑 + 108次联合。搜索提议3900次，校准另列于质量总账；已有36次联合与24次单跑复用不计新增后端。", "",
              "迁移按每个算子相对自身健康映射计算，包含行为落点、DDR读取源变化和去重权重副本字节；32个完整状态见[s_mapping_states.csv](tables/s_mapping_states.csv)。它们是模板算子的迁移量，未简单乘8当成整个MHA链的物理权重迁移。三相位、三种子和两种传输共享同一映射状态，不重复计算修复成本。", "",
              "[s_mapping_costs.csv](tables/s_mapping_costs.csv)同时保存新增A1与历史A3h修复成本，并标明来源。研究五种子累计决策包含R0/L1、各已执行档校准、搜索、恢复最佳状态和最终CPCM；十种子后端验证另外列示。报告补计原始记录中的恢复时间，没有修改执行过程。这些并行时长与J阶段的空闲单进程计时分开。", "",
              "[s_arm_summary.csv](tables/s_arm_summary.csv)保留每个组合P、D、L̂与干扰的均值、中位数和极值；[s_conditional_pairs.csv](tables/s_conditional_pairs.csv)保留全部216个条件配对；[s_phase_conditionals.csv](tables/s_phase_conditionals.csv)按相位拆分。图中的点是中位数、线段是最小至最大，两个符号分别表示P和D。", "",
              "所有三相位都依据共同A1参考释放；g=0.9表示参考KV窗口的90%位置。配对单跑只复用完整图、实际遍历顺序、平台、执行方式、rank、释放和种子完全相同的记录。当前结果衡量这两种硬件、PF4、一个P流与一条D链的固定工作负载完成时间和干扰。", ""]
    (destination / "S_FACTORIAL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def report_with_input_inventory(out, destination):
    """Record data actually read by one report run, including historical costs."""
    accessed = collections.defaultdict(collections.Counter)
    original_json, original_pickle, original_sha = run.READ, audit.raw_read, audit.sha

    def record(filename, operation):
        path = audit.local(filename).resolve()
        accessed[path][operation] += 1

    def json_read(filename):
        record(filename, "json_reads")
        return original_json(filename)

    def pickle_read(filename):
        record(filename, "pickle_reads")
        return original_pickle(filename)

    def hash_read(filename):
        record(filename, "hash_reads")
        return original_sha(filename)

    run.READ, audit.raw_read, audit.sha = json_read, pickle_read, hash_read
    try:
        summary = report(out, destination)
    finally:
        run.READ, audit.raw_read, audit.sha = original_json, original_pickle, original_sha

    hi = (run.ROOT / "e2e/hi").resolve()
    files = [dict(path=path.relative_to(run.ROOT).as_posix(), sha256=original_sha(path),
                  bytes=path.stat().st_size, outside_hi=not path.is_relative_to(hi),
                  reads=dict(accessed[path])) for path in sorted(accessed)]
    outside = [r for r in files if r["outside_hi"]]
    groups = collections.defaultdict(list)
    for row in outside:
        groups["/".join(Path(row["path"]).parts[:3])].append(row)
    inventory = dict(status="passed", report="S_FACTORIAL_SUMMARY.md", file_count=len(files),
        bytes=sum(r["bytes"] for r in files), outside_hi_files=len(outside),
        outside_hi_bytes=sum(r["bytes"] for r in outside),
        outside_hi_by_folder={k:dict(files=len(v), bytes=sum(r["bytes"] for r in v)) for k,v in sorted(groups.items())},
        source_sha256={audit.relative(Path(p)):original_sha(p) for p in (__file__, audit.__file__, run.__file__)},
        read_scope="JSON inputs, raw pickles and explicitly hashed graph files read during this report run; code imports listed separately",
        new_searches=0, new_backend_calls=0, result_summary=summary, files=files)
    run.WRITE(Path(destination) / "evidence/s_report_inputs.json", inventory)
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(run.OUT))
    parser.add_argument("--destination", default=str(run.HERE))
    parser.add_argument("--record-inputs", action="store_true")
    args = parser.parse_args()
    if args.record_inputs:
        value = report_with_input_inventory(run.fault.path(args.out), run.fault.path(args.destination))
        print(json.dumps({k:value[k] for k in ("status", "file_count", "bytes", "outside_hi_files", "outside_hi_bytes", "outside_hi_by_folder")}))
    else:
        print(json.dumps(report(run.fault.path(args.out), run.fault.path(args.destination))))
