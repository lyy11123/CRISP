"""Report Split backend samples, construction costs, and the three-case main figure."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import statistics

import post_fg_split as split


def timing_section(dest):
    cloud = split.read(dest / "out/split_cloud_timing/timing.json")
    local = {p.stem: split.read(p) for p in (dest / "out/split_prepare/metadata").glob("*.json")}
    rows = []
    for row in cloud["selected"]:
        original = local[f"{row['case']}_m{row['factor']}"]
        rows.append(dict(case=row["case"], factor=row["factor"], axis=row["axis"],
                         local_selected_build_seconds=original["build_seconds"],
                         local_candidates_seconds=original["axis_selection_seconds"],
                         cloud_selected_build_seconds=row["selected_build_seconds"],
                         cloud_candidates_seconds=row["candidate_build_seconds"],
                         cloud_decision_wall_seconds=row["decision_wall_seconds"],
                         graph_sha256=row["graph_sha256"]))
    split.table(dest / "tables/split_construction_timing.csv", rows)
    text = ["## 构图和选轴耗时", "",
            "正式输入的 `build_seconds` 和 `axis_selection_seconds` 来自 Windows 本机准备图时的记录，后端作业会原样保留这两个字段。云端的构图成本在同一服务器上另行测量，使用独立目录 `out/split_cloud_timing`。", "",
            "每个 m1 构造 N 轴候选；每个 m2/m4 分别构造 N/M 两个候选，再按预先确定的规则选轴。下表“全部候选”是候选构图时间之和，包含放置、路由、依赖重接、图检查和工作量统计；选中图这一列仅含最后选中的候选。", "",
            "| 案例 | m | 轴 | 本机选中图s | 本机全部候选s | 云端选中图s | 云端全部候选s |",
            "|---|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f"| {r['case']} | {r['factor']} | {r['axis']} | {r['local_selected_build_seconds']:.4f} | {r['local_candidates_seconds']:.4f} | {r['cloud_selected_build_seconds']:.4f} | {r['cloud_candidates_seconds']:.4f} |")
    text += ["", "| 案例 | 读取s | 两个平台s | identity与健康统计s | 构图并选轴s | 图摘要核对s | 图写盘s | 案例全流程s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in cloud["cases"]:
        text.append(f"| {r['case']} | {r['read_seconds']:.4f} | {r['platform_seconds']:.4f} | {r['identity_seconds']+r['baseline_metrics_seconds']:.4f} | {r['decision_wall_seconds']:.4f} | {r['verify_seconds']:.4f} | {r['write_graph_seconds']:.4f} | {r['complete_case_seconds']:.4f} |")
    text += ["", f"云端 Python {cloud['python']}，{cloud['system']}，{cloud['cpu_count']} 个逻辑CPU；本项用一个进程。完整入口耗时 {cloud['complete_entry_seconds']:.3f} s，覆盖15次候选构图、9张所选图写盘、3次identity与输入读取、平台创建、一致性核对。案例全流程还包含元数据写盘；构图、核对与写盘分别计时。此次没有后端评估或SA。", "",
             "9张图的规范化内容SHA-256逐项等于正式清单和已上传图，原清单与9个正式图文件SHA-256在复测前后保持一致。正式实验继续使用原输入图。", "",
             "首次云端尝试在导入tensorcore时终止，尚未构图。服务器保存的是 `e2e/work/artifact`；新增Split入口随后通过 `f_config.artifact_root()`使用这份配置好的artifact。冻结源码保持原样，重测通过上述全部一致性检查。", "",
             "证据：[云端计时](out/split_cloud_timing/timing.json)、[首次错误](out/split_cloud_timing/first_attempt_error.log)、[修正记录](out/split_cloud_timing/first_attempt_failure.json)、[下载核对](evidence/split_cloud_timing_download.json)。", ""]
    return text


def backend_overview(dest):
    samples = split.read(dest / "tables/split_runs.json")
    summary = split.read(dest / "tables/split_summary.json")
    groups = defaultdict(list)
    for row in samples:
        groups[(row["case"], row["method"])].append(row["backend_ns"])
    methods = ("R0", "EIR-representative", "Local-m1", "Split-m2", "Split-m4")
    bounds = {r["case"]: r["rigid_tensor_lower_bound_ns"] for r in summary}
    rows = []
    for case, _, _ in split.CASES:
        for method in methods:
            values = groups[(case, method)]
            assert len(values) == 3
            rows.append(dict(case=case, method=method, mean_ns=statistics.mean(values),
                             min_ns=min(values), max_ns=max(values), seeds=3,
                             rigid_tensor_lower_bound_ns=bounds[case],
                             all_samples_below_rigid_bound=max(values) < bounds[case]))
    split.table(dest / "tables/split_all_methods.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1))
    titles = ("FACE: projection, 2×2", "BusyBarn: FFN, 2×3", "BusyBarn: FFN, 3×3")
    labels = ("R0", "EIR", "Local\nm1", "Split\nm2", "Split\nm4")
    colors = ("#a6adb5", "#e6a65a", "#5d84af", "#42a6b3", "#3b8965")
    for ax, (case, _, _), title in zip(axes, split.CASES, titles):
        data = [r for r in rows if r["case"] == case]
        means = np.array([r["mean_ns"] for r in data]) / 1e6
        lower = means - np.array([r["min_ns"] for r in data]) / 1e6
        upper = np.array([r["max_ns"] for r in data]) / 1e6 - means
        x = np.arange(5)
        ax.bar(x, means, color=colors, width=.68, zorder=3,
               yerr=np.vstack((lower, upper)), capsize=4,
               error_kw={"elinewidth": 1.1, "capthick": 1.1, "ecolor": "#273442"})
        ax.axhline(bounds[case] / 1e6, color="#b33a3a", linestyle="--", linewidth=1.4, zorder=4)
        ax.set_xticks(x, labels)
        ax.set_title(title, fontweight="bold", fontsize=11, pad=11)
        ax.set_ylabel("Backend latency (million ns)")
        ax.set_ylim(0, max(max(r["max_ns"] for r in data), bounds[case]) / 1e6 * 1.15)
        ax.grid(axis="y", color="#e5e9ed", linewidth=.7, zorder=0)
    fig.legend([Line2D([0], [0], color="#b33a3a", linestyle="--", linewidth=1.4)],
               ["Valid compute lower bound for unchanged block sizes"], loc="upper center",
               bbox_to_anchor=(.5, 1.01), frameon=False)
    fig.text(.5, .01, "Bars: mean of backend seeds 1–3; whiskers: min–max. EIR uses the saved stopping-tier representative.",
             ha="center", fontsize=9, color="#46515e")
    fig.tight_layout(rect=(0, .065, 1, .92), w_pad=2.0)
    folder = dest / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(folder / f"split_backend_main.{extension}", dpi=220, bbox_inches="tight")
    svg = folder / "split_backend_main.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
    plt.close(fig)
    text = ["## 三个案例的五种方案", "", "![三案例后端均值、范围与有效刚性计算下界](figures/split_backend_main.png)", "",
            "每根柱为后端种子1–3的均值，误差线为最小值到最大值。R0与EIR使用已有F/G样本，EIR为原停止档后端中位代表；Local-m1、Split-m2/m4使用本轮27项结果。红虚线是固定原块大小时的有效Tensor计算下界，三图分别使用各自纵轴。", "",
            "| 案例 | 方案 | 均值ns | 最小ns | 最大ns | 有效刚性计算下界ns |",
            "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f"| {r['case']} | {r['method']} | {r['mean_ns']:,.1f} | {r['min_ns']:,} | {r['max_ns']:,} | {r['rigid_tensor_lower_bound_ns']:,.1f} |")
    text += ["", "导出：[PDF](figures/split_backend_main.pdf)、[SVG](figures/split_backend_main.svg)、[全部方案表](tables/split_all_methods.csv)。", ""]
    return text


def costs_and_trace(dest):
    summary = split.read(dest / "tables/split_summary.json")
    m1 = {r["case"]: r for r in summary if r["factor"] == 1}
    costs = []
    for row in summary:
        base = m1[row["case"]]
        assert row["effective_flops"] == base["effective_flops"]
        assert row["rounded_flops"] == base["rounded_flops"]
        costs.append(dict(case=row["case"], factor=row["factor"],
                          effective_flops=row["effective_flops"], rounded_flops=row["rounded_flops"],
                          added_rounded_flops=0,
                          route_bytes_delta=row["route_link_bytes"]-base["route_link_bytes"],
                          route_change_percent=100*(row["route_link_bytes"]/base["route_link_bytes"]-1),
                          ddr_read_bytes_delta=row["ddr_read_bytes"]-base["ddr_read_bytes"],
                          ddr_read_change_percent=100*(row["ddr_read_bytes"]/base["ddr_read_bytes"]-1),
                          extra_weight_input_bytes=row["extra_weight_input_bytes"],
                          activation_input_bytes_delta=row["affected_activation_input_bytes"]-base["affected_activation_input_bytes"],
                          ddr_write_bytes_delta=row["ddr_write_bytes"]-base["ddr_write_bytes"]))
    split.table(dest / "tables/split_cost_changes.csv", costs)
    payload = split.static.read_pickle(dest / "out/split_prepare/graphs/G33_ffn_m4.pkl.gz")
    events = pickle.loads(payload["events_blob"])
    traces, records, hashes = {}, {}, {}
    for seed in (1, 2, 3):
        path = dest / f"out/main/traces/split_G33_ffn_m4_s{seed}.json"
        traces[seed] = {tuple(r["tag"]): r for r in split.read(path)}
        records[seed] = split.read(dest / f"out/main/results/split_G33_ffn_m4_s{seed}.json")
        hashes[seed] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert set(traces[seed]) == set(events)
        assert max(r["end_ns"] for r in traces[seed].values()) == records[seed]["backend_triple_ns"][0]
    slow = max(records, key=lambda s: records[s]["backend_triple_ns"][0])
    fast = min(records, key=lambda s: records[s]["backend_triple_ns"][0])
    children = [tuple(c["tag"]) for parent in payload["info"]["parent_children"] for c in parent["children"]
                if tuple(c["tag"]) != tuple(parent["parent"]) and events[tuple(c["tag"])].comp_device == "tensorcore"]
    delayed = max(children, key=lambda tag: traces[slow][tag]["end_ns"]-traces[fast][tag]["end_ns"])
    device = tuple(events[delayed].comp_location)

    def event_row(seed, tag):
        event, trace = events[tag], traces[seed][tag]
        ready = max((traces[seed][t]["end_ns"] for t in event.dependency_set), default=0)
        return dict(seed=seed, tag=list(tag), name=event.event_name, kind=event.event_type,
                    ready_ns=ready, start_ns=trace["start_ns"], end_ns=trace["end_ns"],
                    waiting_ns=trace["start_ns"]-ready,
                    device=list(event.comp_location) if event.event_type == "computation" else None)

    core = []
    chains = {}
    for seed in (1, 2, 3):
        core.extend(sorted((event_row(seed, tag) for tag, event in events.items()
                            if event.event_type == "computation" and event.comp_device == "tensorcore"
                            and tuple(event.comp_location) == device), key=lambda r: r["start_ns"]))
        chain = []
        tag = max(traces[seed], key=lambda t: traces[seed][t]["end_ns"])
        while True:
            chain.append(event_row(seed, tag))
            if not events[tag].dependency_set:
                break
            tag = max(events[tag].dependency_set, key=lambda t: traces[seed][t]["end_ns"])
        chains[seed] = chain
    evidence = dict(case="G33_ffn", factor=4, slow_seed=slow, fast_seed=fast,
                    graph_sha256=split.graph_digest(events), delayed_child=list(delayed),
                    tensor_device=list(device), source_trace_sha256=hashes,
                    backend_ns={s: r["backend_triple_ns"][0] for s, r in records.items()},
                    core_service_order=core, latest_predecessor_chains=chains,
                    new_simulations=0)
    split.write(dest / "evidence/split_g33_m4_trace_diagnosis.json", evidence)
    split.table(dest / "tables/split_g33_m4_core_order.csv", core)
    a, b = event_row(slow, delayed), event_row(fast, delayed)
    texts = ["## 结果解释与通信成本", "",
             "三个案例的m2全部三个种子都低于原块大小下的有效计算下界，说明局部拆分已达到固定分片放置无法达到的总延迟。FACE与2×3 FFN的m4也都低于该下界；相对R0均值分别降低34.87%和21.74%。Local-m1仍在下界以上，收益主要来自粒度变化。", "",
             "3×3 FFN的m2均值为23.132M ns、范围23.064–23.170M ns，相对R0降低14.87%。m4均值24.131M ns，种子1/2/3分别为29.294/21.465/21.633M ns：更细的块降低了计算峰值，同时服务顺序产生了长尾。本轮m2在三个案例的三种子范围均低于刚性下界；m4在3×3上同时出现两个更低值和一个长尾。", "",
             "各方案有效FLOP和硬件grain取整后的FLOP逐项相同，健康计算移动数均为0。下面通信变化均相对Local-m1，以区分拆分增加的代价；独占有向链字节计多播共享前缀一次。", "",
             "| 案例 | m | 有向链字节变化 | DDR读增加MiB | DDR读变化 | 权重输入增加MiB | 受影响激活输入增加MiB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in costs:
        if row["factor"] > 1:
            texts.append(f"| {row['case']} | {row['factor']} | {row['route_change_percent']:+.3f}% | {row['ddr_read_bytes_delta']/2**20:.3f} | {row['ddr_read_change_percent']:+.3f}% | {row['extra_weight_input_bytes']/2**20:.3f} | {row['activation_input_bytes_delta']/2**20:.3f} |")
    texts += ["", "FACE与2×3采用M切分，需要复制B/bias；3×3采用N切分，B/bias按列划分，增加的是各子块所需完整A的读取与转送。三个案例DDR写入量保持相同。总链路字节增幅较小，但它本身不决定任务何时就绪、哪个计算先占用接收核。", "",
              "### 3×3 m4种子1的长尾", "",
              f"已有trace显示，末端依赖链上的切块为 `{tuple(delayed)}`，落在Tensor设备 `{device}`。种子{slow}中，该块在{a['ready_ns']:,} ns已具备全部输入，在{a['start_ns']:,} ns才开始，排队{a['waiting_ns']:,} ns；种子{fast}中它在{b['start_ns']:,} ns开始。相同图、落点、FLOP和字节量下，接收核的服务次序不同。", "",
              "种子1中，该切块比健康块12晚23 ns就绪，健康块12先占核；该核随后又服务了健康块92。切块结束后，其GELU与第二层块104继续执行，将末端结果推迟到29.294M ns。种子2中该切块首先被服务，种子3中它排在一个健康块之后。这里可直接看到输入就绪时刻、非抢占长计算及下游依赖如何放大一个较小的调度变化。", "",
              "| 种子 | 核上服务顺序 | 输入就绪ns | 开始ns | 结束ns | 排队ns |",
              "|---:|---|---:|---:|---:|---:|"]
    for row in core:
        texts.append(f"| {row['seed']} | `{tuple(row['tag'])}` | {row['ready_ns']:,} | {row['start_ns']:,} | {row['end_ns']:,} | {row['waiting_ns']:,} |")
    texts += ["", "该解释只使用已保存的事件时间、依赖和设备位置；没有增加仿真。证据：[三种子核上时间表](tables/split_g33_m4_core_order.csv)、[终点依赖与trace摘要](evidence/split_g33_m4_trace_diagnosis.json)、[通信与取整变化表](tables/split_cost_changes.csv)。", ""]
    return texts


def report(run_out, out, dest, timing_only):
    if timing_only:
        split.render_report(dest)
        overview = []
    else:
        split.setup()
        split.report(run_out, out, dest)
        overview = backend_overview(dest) + costs_and_trace(dest)
    path = dest / "EIR_SPLIT.md"
    text = path.read_text(encoding="utf-8").replace("python -B e2e/post_fg_split.py prepare\n", "")
    text = text.replace("python -B e2e/post_fg_split.py report --out e2e/post_fg/out/main", "python -B e2e/post_fg_split_report.py")
    path.write_text(text + "\n" + "\n".join(overview + timing_section(dest)), encoding="utf-8")
    print(json.dumps(dict(status="timing_report_ready" if timing_only else "complete", report=str(path))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-out", type=Path, default=split.ROOT / "e2e/fg/out/main")
    parser.add_argument("--out", type=Path, default=split.ROOT / "e2e/post_fg/out/main")
    parser.add_argument("--dest", type=Path, default=split.ROOT / "e2e/post_fg")
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    report(args.run_out.resolve(), args.out.resolve(), args.dest.resolve(), args.timing_only)
