"""Generate the F/G reports from the completed, read-only analysis tables."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"wsc":"WSC-LLM", "watos":"WATOS", "face":"FACE", "busybarn":"BusyBarn"}
METHODS = ("A1", "A3h", "Gemini")
ARMS = ("B-R0", "-DSTU", "-priority", "-EIR", "CRISP")
CONDITIONS = {"healthy":"健康", "D":"D 组 D2D 故障", "cross":"组间链路故障"}
PROTOCOLS = {"backend_best":"后端选优", "candidate_median":"十四候选中位数", "random_cpcm":"随机平局 CPCM 选优", "deterministic_cpcm":"确定性 CPCM 选优（附加）"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value, digits=3):
    return "—" if value is None else f"{value:,.{digits}f}"


def pct(value):
    return "—" if value is None else f"{100*value:+.2f}%"


def share(value):
    return "—" if value is None else f"{100*value:.2f}%"


def config(row):
    if row["stage"] in ("F", "F_bw"):
        suffix = f" D2D×{row['d2d_scale']:g}" if row["stage"] == "F_bw" else ""
        return LABELS[row["hardware_profile"]] + suffix
    if row["stage"] == "E_reuse":
        return "2×2 E 基线"
    return row["shape"].replace("x","×") + (" 固定切分" if row["split_mode"] == "fixed" else " 增长切分")


def table(headers, rows):
    result = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        result.append("| " + " | ".join("—" if v is None else str(v).replace("|", "／").replace("\n", " ") for v in row) + " |")
    return "\n".join(result)


def link(name):
    return f"[{name}.csv](fg/tables/{name}.csv)"


def section(title, *content):
    return "\n\n".join([title, *content])


def select(rows, stage):
    return [r for r in rows if r["stage"] == stage]


def interval(values, scale=1, percent=False):
    if percent:
        return f"{pct(statistics.median(values))} [{pct(min(values))}, {pct(max(values))}]"
    return f"{number(statistics.median(values)/scale)} [{number(min(values)/scale)}, {number(max(values)/scale)}]"


def intro(stage, data):
    layers = [r for r in select(data["layers"], stage) if r["family"] == "healthy" and r["protocol"] == "backend_best" and r["method"] == "A3h"]
    changes = [r["request_2048_relative_to_A1"] for r in layers]
    decode = [r["decode_relative_to_A1"] for r in layers]
    label = "三种硬件参数" if stage=="F" else "两种新增规模、两种切分方式"
    comparison=next(r for r in data["joint_comparison_summary"] if r["scope"]==stage and r["reference_arm"]=="-DSTU")
    fault=[r for r in select(data["fault_summary"],stage) if r["scheme"]=="EIR"]
    fault_changes=[r["backend_change_vs_A3h_R0"] for r in fault]
    lines = [f"# {stage} 阶段结果："+("硬件参数迁移与带宽变化" if stage=="F" else "固定工作量下的规模扩展"),
        f"2026-11-13，Codex。F/G 已完成 {data['workflow_checks']['completed_jobs']:,}/{data['workflow_checks']['planned_jobs']:,} 个调度作业。本文由已保存的候选、故障修复和联合实验结果生成；表中时间采用统一的 1 ns/tick。",
        f"{label}下，A3h 相对 A1 的单层 decode 时间变化为 **{pct(min(decode))} 至 {pct(max(decode))}**，固定 KV、生成 2048 步的请求时间变化为 **{pct(min(changes))} 至 {pct(max(changes))}**。具体配置、故障和到达相位分别列出。",
        f"EIR 的 {len(fault)} 个算子×故障中，{sum(v<0 for v in fault_changes)} 个比 A3h-R0 更低、{sum(v==0 for v in fault_changes)} 个相同、{sum(v>0 for v in fault_changes)} 个更高。完整 CRISP 的 D 延迟降低包含映射与 EIR 的贡献；单独比较是否启用 DSTU，D 在 {comparison['D_latency_ratio_change_lower_count']}/{comparison['cells']} 行更低，P 完成时间在 {comparison['P_completion_ratio_change_higher_count']}/{comparison['cells']} 行更高，最大归一化迟约也在 {comparison['Lhat_max_difference_higher_count']}/{comparison['cells']} 行更高。",
        f"DSTU 的 P 完成时间变化中位数为 **{pct(comparison['P_completion_ratio_change_median'])}**，P 单跑本身的变化中位数为 **{pct(comparison['P_solo_ratio_change_median'])}**。这些结果体现切块和按跳执行的服务时间代价，以及对 D 的优先服务效果；P 的全部变化不能归于与 D 重叠产生的干扰。",
        "本阶段的直接仿真对象是单个算子及相邻 P/D 组上的联合执行；单层和请求时间由算子结果按已确定公式相加。E 的 44 层混合布局见 [E_SUMMARY.md](E_SUMMARY.md)，F/G 的新硬件和 6/9 Die 组采用本阶段实际测量的单组结果。"]
    return "\n\n".join(lines)


def protocol(stage, data, jobs):
    stages = {"F", "F_bw"} if stage=="F" else {"G"}
    counts = Counter(j["kind"] for j in jobs if j["spec"]["stage"] in stages)
    candidates = [r for r in data["candidates"] if r["stage"] in stages and r["kind"] == "search"]
    terminals = [r for r in data["candidates"] if r["stage"] == stage and r["kind"] != "search"]
    rows = [["健康搜索",sum(r["family"]=="healthy" for r in candidates)],
            ["PF4 A3h-FULL 搜索",sum(r["family"]=="full" for r in candidates)],
            ["D2D 带宽重新搜索",sum(r["family"]=="bandwidth" for r in candidates)],
            ["Gemini 结束状态记录",len(terminals)],
            ["联合作业（含带宽子实验）",counts["joint"]],
            ["配对 P/D 单跑作业",counts["joint_solo"]],
            ["FFN 无共享资源核对",counts["joint_ffn_check"]],
            ["全部调度作业",sum(counts.values())]]
    return section("## 1. 完成内容与固定设置", table(["项目", "数量"],rows),
        "F/G 共用队列共 6,813 个调度作业。健康、FULL 与带宽搜索合计 3,024 次；修复档位、配对单跑、停止判断和图构建分别记账。小批结果计入正式清单，Gemini 的 56 个结束状态不另算搜索。",
        table(["项目", "设定"], [
            ["工作负载", "GPT W1；batch 16；hidden 6144；prefill 长度 2048；decode KV 长度固定 2048"],
            ["头数", "MHA 微图 8 头、每头 96；层组装乘 8，总计 64 头"],
            ["层公式", "prefill = 2 LN + 8 MHA + proj + FFN；decode = LN + 8 MHA + proj + FFN"],
            ["LN", "decode FFN 自带一次 LN，因此外部只再加一次 LN"],
            ["请求时间", "T(N) = P + N×D，分别列 N=512/2048，沿用固定 KV 2048 的每步 D"],
            ["A1", "g0/g1 × 7 个既定损失权重；SA 种子 123，14 个候选"],
            ["A3h", "g0/g1 × SA 种子 {123,1,…,6}，14 个候选"],
            ["Gemini", "SA 种子 {123,1,…,13}，14 个候选；保留原生移动、目标、温度与 XY 路由"],
            ["搜索预算", "每候选 1000 步；每保存状态使用后端种子 1–10；随机平局 CPCM 使用种子 1–10"],
            ["EIR", "L1 → L2-Global(+sync)，五个 SA 种子 {123,1,2,3,4}，独立 30/100/300 步"],
            ["DSTU", "全链路 256 KB 切块＋按跳 C4；联合评估种子 1–3；相位 0.1/0.5/0.9"],
        ]),
        "主性能表采用十四候选中的后端均值最小者；候选中位数、随机平局 CPCM 选优和确定性 CPCM 选优另列。后端均值均来自十个种子。一次搜索得到的候选差异与同一候选的后端波动分别统计。")


def hardware_section():
    return section("## 2. 实际采用的硬件参数",
        "F 固定为 2×2 Die、每 Die 4×4 逻辑核，共 64 核。表中的 TensorCore 算力通过原时间公式的计算缩放实现；向量单元保持 1024 lanes/tick，片内网络保持 256 B/ns、1 ns。",
        table(["配置", "TFLOPS/Die", "计算缩放", "D2D 每有向链 B/ns", "DDR 每端口/方向 B/ns"], [
            ["BusyBarn（E 基线）",131.072,1,256,256],
            ["WSC-LLM",261.12,1.9921875,125,250],
            ["WATOS",708,5.401611328,250,250],
            ["FACE HBM/D2D",131.072,1,355.625,410],
        ]),
        "D2D 每边有两条双向物理连接，论文总带宽按每 Die 四边、双向合计分给 16 条有向链；D2D 延迟 20 ns。DDR 每 Die 八端口，读写方向独立、各自均分标称带宽，链路延迟 50 ns。DDR 容量字段为 32 GiB/64 GB/70 GB/128 GB，当前服务模型不以容量字段决定时长。SRAM 均为每核 16 MiB。",
        "四配置的 16 个 GEMM、向量、D2D、读写重叠核对与计算公式一致。完整参数、来源和单位换算见 [hardware_parameters.md](f/tables/hardware_parameters.md) 与 [hardware_checks.json](f/tables/hardware_checks.json)。")


def scale_section():
    return section("## 2. 规模与切分的实际变化",
        "每 Die 沿用 BusyBarn 参数与 4×4 核。2×2、2×3、3×3 分别对应 64、96、144 核；输入 B/S/H 和 64 个模型头保持一致。固定切分只改变可放置的硬件资源，增长切分进一步细化以下阶段。",
        table(["模板字段", "2×2 / 固定切分", "2×3 增长切分", "3×3 增长切分"], [
            ["prefill LN", "(1,64,1)/(1,8,8)","(1,96,1)/(1,12,8)","(1,144,1)/(1,18,8)"],
            ["decode LN", "(1,1,1)/(1,1,1)","相同","相同"],
            ["MHA", "(1,8,1,1)","(1,12,1,1)","(1,18,1,1)"],
            ["prefill proj", "(1,8,1,8)","(1,12,1,8)","(1,18,1,8)"],
            ["decode proj", "(1,1,8,8)","(1,1,12,8)","(1,1,18,8)"],
            ["prefill FFN", "(1,4,1,16)","(1,6,1,16)","(1,9,1,16)"],
            ["decode FFN", "(1,1,4,16)","(1,1,6,16)","(1,1,9,16)"],
        ]),
        "decode MHA 的新增切分主要作用于 QKᵀ 的输出序列维，Q/K/V 投影、softmax 和 PV 的实际并行度逐项保存在工作量表。保存快照表明，四种 G 配置的 prefill MHA 取整乘加量均比有效量高 3.0303%，prefill proj/FFN 的该项增量均为 0。增长切分增加了归约事件：prefill MHA 从128个增至192/288个，FFN从8个增至12/18个。第8节列出所有算子的实际计算量与归约规模。",
        "P/D 联合平台把两个组左右相邻排列：2×3 使用 2×6、192 核；3×3 使用 3×6、288 核。每组 KV 接收目标由 E/F 的 32 个增加到 48/72 个，完整输入与拓扑检查见 [g_structure_checks.json](f/tables/g_structure_checks.json)。")


def healthy_section(stage, data):
    layers = [r for r in select(data["layers"],stage) if r["family"]=="healthy"]
    best = [r for r in layers if r["protocol"]=="backend_best"]
    display=best
    if stage=="F":
        display=[r for r in data["layers"] if r["stage"]=="E_reuse" and r["protocol"]=="backend_best"]+best
    rows = [[config(r),r["method"],number(r["prefill_ns"]/1e6),number(r["decode_ns"]/1000),
             number(r["request_512_ns"]/1e6),number(r["request_2048_ns"]/1e6)] for r in display]
    relative = [[config(r),pct(r["prefill_relative_to_A1"]),pct(r["decode_relative_to_A1"]),
                 pct(r["request_2048_relative_to_A1"]),pct(r["request_2048_relative_to_Gemini"])]
                for r in best if r["method"]=="A3h"]
    other = [[config(r),r["method"],PROTOCOLS[r["protocol"]],number(r["prefill_ns"]/1e6),
              number(r["decode_ns"]/1000),number(r["request_2048_ns"]/1e6)]
             for r in layers if r["protocol"]!="backend_best"]
    operators = [r for r in select(data["operators"],stage) if r["family"]=="healthy"]
    groups = defaultdict(dict)
    for r in operators: groups[r["config_id"],r["phase"],r["op"]][r["method"]]=r
    op_rows = []
    higher = []
    for (_,phase,op), group in sorted(groups.items()):
        a1,a3,gemini = [group[m] for m in METHODS]
        op_rows.append([config(a1),phase,op.upper(),*[number(r["backend_best_ns"]/1000) for r in (a1,a3,gemini)],pct(a3["backend_best_ns"]/a1["backend_best_ns"]-1)])
        if a3["backend_best_ns"] > a1["backend_best_ns"]:
            higher.append([config(a1),phase,op.upper(),pct(a3["backend_best_ns"]/a1["backend_best_ns"]-1),number((a3["backend_best_ns"]-a1["backend_best_ns"])/1000)])
    interpretation=""
    if stage=="F":
        e=next(r for r in data["layers"] if r["stage"]=="E_reuse" and r["protocol"]=="backend_best" and r["method"]=="A3h")
        relative_e=[[config(r),pct(r["prefill_ns"]/e["prefill_ns"]-1),pct(r["decode_ns"]/e["decode_ns"]-1),pct(r["request_2048_ns"]/e["request_2048_ns"]-1)] for r in best if r["method"]=="A3h"]
        interpretation=section("### 资源速度改变后的 A3h 结果",
            table(["配置","相对 E 默认硬件：Prefill","Decode","请求2048"],relative_e),
            "WSC 的 TensorCore 加速约两倍，同时采用较低的 D2D 单链带宽；WATOS 主要增加 TensorCore 算力，向量与片内网络速度保持不变；FACE 主要增加 DDR/D2D 带宽。端到端时间因而取决于各阶段计算、通信和重叠时序，而不是单独跟随峰值算力。第8节保留后端时间分解与保存映射的资源工作量。")
    return section("## 3. 健康映射：单层、请求和逐算子结果",
        table(["配置","方法","Prefill / ms","Decode / μs","请求512 / ms","请求2048 / ms"],rows),
        "Prefill 对应本层 TTFT，decode 对应本层 TPOT。请求时间采用 P+N×D；模型输出头、采样和动态 KV 增长不在这组算子组装项内。",
        table(["配置","A3h 对 A1：Prefill","Decode","请求2048","A3h 对 Gemini：请求2048"],relative),
        interpretation,
        "### 逐算子后端选优",
        "时间单位为 μs；MHA 一行仍是八头微图，上一张单层表已乘八。",
        table(["配置","阶段","算子","A1","A3h","Gemini","A3h 对 A1"],op_rows),
        "单层改善并非每个算子都降低。以下列出 A3h 高于 A1 的健康算子，同时给出绝对差值：",
        table(["配置","阶段","算子","A3h相对A1","增加/μs"],higher),
        "### 候选中位数与随机平局 CPCM 选择",
        table(["配置","方法","选择方式","Prefill / ms","Decode / μs","请求2048 / ms"],other),
        f"完整十四候选、算子选择和层组装分别见 {link('candidates')}、{link('operators')}、{link('layers')}。")


def variability_section(stage,data):
    operators = [r for r in select(data["operators"],stage) if r["family"]=="healthy"]
    rows = []
    for method in METHODS:
        values = [r for r in operators if r["method"]==method]
        rows.append([method,len(values),sum(r["random_cpcm_candidate"]==r["backend_best_candidate"] for r in values),
            interval([r["random_selector_gap"] for r in values],percent=True),
            interval([r["deterministic_selector_gap"] for r in values],percent=True)])
    candidates = [r for r in select(data["candidates"],stage) if r["family"]=="healthy" and r["kind"]=="search"]
    spread = []
    for method in METHODS:
        values = [r for r in candidates if r["method"]==method]
        spreads = [(r["backend_max_ns"]-r["backend_min_ns"])/r["backend_mean_ns"] for r in values]
        spread.append([method,len(values),interval(spreads,percent=True)])
    pairs=select(data["gemini_terminal_pairs"],stage)
    pair_values=[r["terminal_vs_best"] for r in pairs]
    pair_rows=[[config(r),r["phase"],r["op"].upper(),number(r["best_ns"]/1000),number(r["terminal_ns"]/1000),pct(r["terminal_vs_best"]),"复用" if r["backend_reused"] else "独立评估"] for r in pairs]
    result = section("## 4. 选优差异与后端种子波动",
        "选优延迟差定义为所选候选的后端十种子均值除以后端最佳候选均值，再减 1。下表按健康算子汇总；范围覆盖这些算子。",
        table(["方法","算子数","随机 CPCM 选中同一候选","随机选择差：中位 [最小,最大]","确定性选择差：中位 [最小,最大]"],rows),
        "同一候选的种子极差定义为 `(最大值−最小值)/均值`，下面的中位数与范围再跨该方法的全部健康候选统计。它与十四次搜索的候选差异是两类变化。",
        table(["方法","候选数","十种子相对极差：中位 [最小,最大]"],spread),
        f"十四候选均值的四分位数、标准差和最大/最小值见 {link('candidate_distribution')}；逐次后端总时间、纯计算和纯通信时间见 {link('backend_samples')}。BusyBarn 的平局顺序受随机种子及进程内对象顺序共同影响，本报告保留十次评估的实际分布。",
        f"Gemini 种子 123 的 {len(pairs)} 对最好状态与结束状态，结束态相对最好态的后端变化为 **{interval(pair_values,percent=True)}**；{sum(r['backend_reused'] for r in pairs)} 对保存状态相同而直接复用评估。这里的最好态指 Gemini 自身搜索目标的最好态。",
        table(["配置","阶段","算子","最好态/μs","结束态/μs","结束态变化","评估方式"],pair_rows),
        "完整配对见 "+link("gemini_terminal_pairs")+"。这些配对来自同一次退火轨迹，不另算搜索候选。")
    return result


def fault_section(stage,data):
    configs = select(data["fault_configs"],stage)
    fault_rows = [[config(r),r["fault_id"],r["failed_nodes"] or "—",r["failed_pes"] or "—",r["failed_links"] or "—",
                  share(r["failed_compute_group_fraction"]),share(r["failed_D2D_pair_fraction"]),share(r["failed_DDR_port_fraction_die"])] for r in configs]
    layers = select(data["fault_layers"],stage)
    layer_rows = [[config(r),r["fault_id"],r["scheme"],number(r["prefill_ns"]/1e6),number(r["decode_ns"]/1000),
                   number(r["request_2048_ns"]/1e6),pct(r["prefill_change_vs_healthy"]),pct(r["decode_change_vs_healthy"]),pct(r["request_2048_change_vs_healthy"])] for r in layers]
    eir = [r for r in select(data["fault_summary"],stage) if r["scheme"]=="EIR"]
    counts = Counter((r["stop_level"],r["stop_reason"]) for r in eir)
    stop_rows = [[level,reason,count] for (level,reason),count in sorted(counts.items())]
    stop_measured=[]
    for reason in sorted({r["stop_reason"] for r in eir}):
        matching=[r for r in eir if r["stop_reason"]==reason]
        stop_measured.append([reason,len(matching),sum(r["backend_change_vs_healthy"]<=.02 for r in matching),
                              interval([r["backend_change_vs_healthy"] for r in matching],percent=True)])
    better = sum(r["request_2048_change_vs_A3h_R0"] < 0 for r in layers if r["scheme"]=="EIR")
    eir_layers = [r for r in layers if r["scheme"]=="EIR"]
    exceptions=[r for r in eir_layers if r["request_2048_change_vs_A3h_R0"]>=0]
    exception_text="；".join(f"{config(r)} {r['fault_id']} 的变化为 {100*r['request_2048_change_vs_A3h_R0']:+.5f}%" for r in exceptions)
    tiers=select(data["fault_tiers"],stage)
    executed=[r for r in tiers if r["status"]!="reused"]
    worse=[r for r in eir if r["backend_change_vs_A3h_R0"]>0]
    worst_healthy=max(eir,key=lambda r:r["backend_change_vs_healthy"])
    worserows=[[config(r),r["fault_id"],r["phase_op"],pct(r["backend_change_vs_A3h_R0"]),pct(r["backend_change_vs_healthy"]),r["stop_level"],r["stop_reason"]] for r in worse]
    repair_rows = [[config(r),r["fault_id"],r["phase_op"],r["stop_level"],r["n_sa_states"],
                   pct(r["backend_change_vs_A3h_R0"]),number(r["unique_block_bytes_mean"]/2**20),
                   share(r["moved_behas_share_mean"]),r["selected_search_steps"],r["executed_L2_steps"],
                   number(r["selected_t_decision_seconds_mean"]),r["representative_sa_seed"]] for r in eir]
    return section("## 5. 同一物理故障下的恢复",
        "每个配置用健康 A3h decode MHA 作为锚定算子，沿用 C 的选点函数。PF4 是一条 D2D 物理连接的双向失效；MF1 为非角落核及路由器；MF6 只关闭 PE、路由器保留；MF8 为 DDR 端口。选中的同一物理位置作用于 P/D 八个算子。G 的故障部分仅使用增长切分，保留 PF4/MF6。",
        table(["配置","故障","失效节点","只坏 PE","失效有向链","组计算比例","该Die对D2D比例","Die DDR 端口比例"],fault_rows),
        "坐标是单组内的 `(Die,core)`，有向链为端点对；组间联合执行时按左右布局重编号。PF4 删除相邻两个 Die 间两条连接中的一条，所以该 Die 对的连接失效比例为 50%；全组各类型链路的失效带宽另按实际拓扑统计。选点依据、输入候选、各链路类型的失效带宽比例见 "+link("fault_configs")+"。",
        "### 单层恢复与固定 KV 请求",
        "健康基准采用各方案配对的故障前状态重评。EIR 在 L2 停止时先平均五个 SA 种子的后端均值，在 R0/L1 停止时使用该单一状态，然后按 fixed LN 公式相加。联合实验对 L2 的五状态另取后端均值中位代表，均值相同时取较小的 SA 种子；对 R0/L1 直接使用停止状态。",
        table(["配置","故障","方案","Prefill / ms","Decode / μs","请求2048 / ms","Prefill变化","Decode变化","请求变化"],layer_rows),
        f"EIR 在 {len(eir_layers)} 个配置×故障的请求2048结果中，{better} 个低于对应 A3h-R0；变化中位数及范围为 **{interval([r['request_2048_change_vs_A3h_R0'] for r in eir_layers],percent=True)}**。PF4 的 A3h-FULL 是相同 1000 步、十四候选预算下的从头搜索参照。",
        exception_text+"。" if exceptions else "F 的全部十二个配置×故障在请求2048上均有降低。",
        "### 停止档、五种子与迁移",
        "R0/L1 的 CPCM 估值回到故障前 +2% 以内就停止；否则进入 L2。30 步档判断恢复目标，100/300 步档同时判断是否达到目标或相邻档改善小于 2%。三档分别从同一 L1 起点独立搜索，逐档累计最多 430 步/种子。",
        table(["停止档","原因（结果记录）","算子×故障数"],stop_rows),
        table(["停止原因","单元数","后端在健康+2%以内","后端相对健康：中位 [最小,最大]"],stop_measured),
        "`within_epsilon` 表示 CPCM 估值进入 +2%，上表另按后端均值计算实际达到 +2% 的数量。`tier_gain` 表示相邻独立搜索档改善不足 2%；`budget` 表示 300 步档仍未达到恢复目标、相邻档改善也未触发停止，因此按预算结束。",
        f"本阶段的修复档位记录为 {len(tiers)} 条，实际执行 {len(executed)} 条，提前停止后复用 {len(tiers)-len(executed)} 条。R0/L1 停止的配置只有一个终态；L2 停止的配置保留五个独立 SA 种子。",
        "EIR 后高于 R0 的算子列在下面。停止使用 CPCM 的恢复程度与相邻档改善，最终质量仍由保存状态的后端十种子均值报告。",
        table(["配置","故障","算子","EIR对R0","EIR对健康","停止档","原因"],worserows),
        f"停止后相对健康的最大算子差异是 **{config(worst_healthy)} {worst_healthy['phase_op']} {worst_healthy['fault_id']}：{pct(worst_healthy['backend_change_vs_healthy'])}**，停在 {worst_healthy['stop_level']}（{worst_healthy['stop_reason']}）。这行与单层相加后的变化同时保留。",
        "下表保留每个修复配置；迁移量为保存的唯一权重块字节，行为比例统计位置变化。五种子的均值、中位数、范围、CPCM 值和实际步数完整列于附表。",
        "已执行总 L2 步数累加所有到达档位和五个种子，例如运行30与100两档为 `5×(30+100)=650` 步。所选 L2 档决策秒是该档五种子 `t_decision` 的均值，含 R0、L1、标定、该档搜索和终态处理；前面各独立 L2 档时间及离线后端评估另列，它也不是五种子的总用时。R0/L1 的 `t_decision` 分别为 `t_R0` 与 `t_R0+t_L1`，其终态处理和后端评估另列。",
        table(["配置","故障","算子","停止档","状态数","EIR 对 R0","权重块均值/MiB","行为迁移比例","选中步数","已执行总L2步数","所选档决策秒均值","联合代表种子"],repair_rows),
        f"逐算子 {link('fault_summary')}；五种子终态 {link('fault_seed_results')}；实际执行与复用档位 {link('fault_tiers')}；每次停止判断 {link('fault_stops')}；后端逐种子 {link('fault_backend_samples')}；单层组装 {link('fault_layers')}。")


def joint_section(stage,data):
    rows = select(data["joint_summary"],stage)
    byphase = defaultdict(dict)
    for r in rows: byphase[r["config_id"],r["condition"],r["phase"]][r["arm"]]=r
    drows=[]
    for (_,condition,phase),arms in sorted(byphase.items()):
        base=next(iter(arms.values()))
        drows.append([config(base),CONDITIONS[condition],phase,
                      *[number(arms[a]["D_latency_ns_mean"]/1000) if a in arms else "同 CRISP" for a in ARMS]])
    prows = [[config(r),CONDITIONS[r["condition"]],r["arm"],number(r["P_completion_ns_mean"]/1000),
              f"{number(r['paired_delta_P_ns_mean']/1000)} [{number(r['paired_delta_P_ns_min']/1000)}, {number(r['paired_delta_P_ns_max']/1000)}]",
              number(r["Lhat_D_mean"]),number(r["D_self_amplification_mean"]),number(r["D_common_ref_ratio_mean"])]
             for r in rows if r["phase"]==.5]
    refs=select(data["joint_references"],stage)
    refrows=[[config(r),number(r["T_P_ns"]/1000),number(r["T_D_ns"]/1000),number(r["KV_start_ns"]/1000),number(r["KV_end_ns"]/1000)] for r in refs]
    ffn=select(data["joint_ffn_checks"],stage)
    ffnrows=[[config(r),r["shared_P_D_links"],r["shared_KV_D_links"],r["shared_compute"],
              number(r["paired_delta_P_ns"]/1000),pct(r["D_self_amplification"]-1),r["audit_n"],r["audit_bad"]] for r in ffn]
    comparisons=[]
    for other in ("B-R0","-DSTU","-priority","-EIR"):
        differences=[]
        for arms in byphase.values():
            if other in arms:
                differences.append(arms["CRISP"]["D_latency_ns_mean"]/arms[other]["D_latency_ns_mean"]-1)
        comparisons.append([other,len(differences),sum(v<0 for v in differences),interval(differences,percent=True)])
    dstu_changes=[(key,arms["CRISP"]["D_latency_ns_mean"]/arms["-DSTU"]["D_latency_ns_mean"]-1)
                  for key,arms in byphase.items()]
    worse=[(key,value) for key,value in dstu_changes if value>0]
    late=[(key,value) for key,value in worse if key[2]==.9]
    component_stats=[r for r in data["joint_comparison_summary"] if r["scope"]==stage]
    component_rows=[[r["reference_arm"],r["cells"],f"{r['D_latency_ratio_change_lower_count']}/{r['D_latency_ratio_change_equal_count']}/{r['D_latency_ratio_change_higher_count']}",
                     pct(r["D_latency_ratio_change_median"]),pct(r["P_completion_ratio_change_median"]),pct(r["P_solo_ratio_change_median"]),number(r["Lhat_max_difference_median"])] for r in component_stats]
    dstu=next(r for r in component_stats if r["reference_arm"]=="-DSTU")
    samples=select(data["joint_runs"],stage)
    deadline_rows=[]
    for arm in ARMS:
        group=[r for r in samples if r["arm"]==arm]
        deadline_rows.append([arm,len(group),sum(r["Lhat_max"]>0 for r in group),
                              number(min(r["Lhat_max"] for r in group)),number(max(r["Lhat_max"] for r in group))])
    return section("## 6. EIR 与 DSTU 的联合效果",
        "P 为 prefill MHA，包含跨组 KV；D 为完整 decode 链，P/D 各占一个组。B-R0 使用 A1-R0 与未切块原后端；CRISP 使用 A3h+EIR、256 KB 切块与 C4；−DSTU 只关闭切块和按跳执行；−priority 保留切块并使用 P1；−EIR 使用 A3h-R0+C4，仅 D 组故障新增运行。",
        "### 公共参照和到达时刻",
        "每个配置冻结健康 A1、原后端的种子1–3均值。D 释放时刻为 `KV_start + g×(KV_end−KV_start)`；同配置的全部方法与故障使用同一时刻。P 期限为 `1.5×T_P,ref`，D 期限为 `τ+1.5×T_D,ref`。实际 KV 窗口仍取各次仿真保存值。",
        "这套公共期限用于结果评价；C4 的排序键沿用自身图的 CPCM 估值构造的 1.5 倍期限。",
        table(["配置","P参照/μs","D参照/μs","KV开始/μs","KV结束/μs"],refrows),
        "### 三个到达相位的 D 链延迟",
        "单位 μs，单元格为三个后端种子的均值。健康与纯组间故障的 −EIR 复用 CRISP。",
        table(["配置","状态","g",*ARMS],drows),
        "### 各组件带来的 D 延迟变化",
        "下表先对每个配置、状态、相位计算三个种子均值，再将 CRISP 与对应对照相除。−EIR 只统计 D 组内部故障。",
        table(["CRISP 相对对照","配置×状态×相位数","CRISP更低的行数","变化中位 [最小,最大]"],comparisons),
        f"相对 −DSTU，CRISP 在 {len(worse)}/{len(dstu_changes)} 行的 D 延迟更高，其中 {len(late)} 行位于后段相位 g=0.9。切块与 C4 的增量因此随 KV/D 到达关系而变；映射和 EIR 的收益应与这一增量分别看。各相位对应的 P 完成和迟约也一并保留。",
        table(["CRISP相对对照","行数","D降低/相同/升高","D变化中位","P完成变化中位","P单跑变化中位","L̂max差中位"],component_rows),
        f"相对 −DSTU，P 完成时间在 {dstu['P_completion_ratio_change_higher_count']}/{dstu['cells']} 行更高，L̂_max 在 {dstu['Lhat_max_difference_higher_count']}/{dstu['cells']} 行更高。P 单跑中位变化 {pct(dstu['P_solo_ratio_change_median'])}，说明启用切块与按跳执行本身已改变 P 的服务时间；配对 ΔP 才衡量在本臂基础上叠加 D 后的变化。L̂_max 为 P/D 两流归一化迟约的最大值，按每个后端种子先取最大再汇总。完整对照分布见 {link('joint_comparisons')}、{link('joint_comparison_summary')}。",
        "按本实验设定的公共 1.5 倍期限，CRISP 与 −DSTU 的两流均在期限前完成；二者 L̂_max 的差异表示提前完成余量不同。下面逐后端种子统计是否至少一流超出期限。",
        table(["方法","逐种子运行数","L̂_max>0次数","最小L̂_max","最大L̂_max"],deadline_rows),
        "### 中间相位的 P 完成、配对干扰与迟约",
        "ΔP = 同种子联合 P 完成时间−本臂本条件 P 单跑时间；负数表示该次联合执行更早完成。自身放大 = 联合 D 延迟 / 本臂 D 单跑；公共比值 = 联合 D 延迟 / 健康 A1 的 D 参照。L̂_D = (D 完成−D 期限)/T_D,ref，负数表示早于期限。",
        table(["配置","状态","方法","P完成/μs","ΔP均值 [最小,最大]/μs","L̂_D均值","D自身放大","D公共比值"],prows),
        "0.1/0.5/0.9 的 P 完成、ΔP、迟约、自身放大、公共比值及各自种子范围全部保存在 "+link("joint_summary")+"；逐种子的实际 KV/D 起止时刻与重叠见 "+link("joint_runs")+"。可据此逐种子查看 KV 窗口、联合完成时间及配对 ΔP。",
        "### FFN 结构核对与按跳仲裁",
        table(["配置","P-D共享链","KV-D共享链","共享计算","ΔP/μs","D相对单跑","仲裁次数","非最小键次数"],ffnrows),
        "FFN 核对点的物理资源交集为零；联合与分开运行的残余差异保留为后端平局变化。按跳仲裁逐次核对当前选择的键是否为可服务候选中的最小键。F/G 的 609 个联合运行（含带宽扫描）中，KV 总体起止区间与 D 活动区间均有交叠；保存内容包括图、路径和总体起止窗口，逐链路活动时序未保留。完整图事件、依赖、KV payload、目标数、多播链路字节与共享资源见 "+link("joint_graphs")+"；核对点见 "+link("joint_ffn_checks")+"。",
        "![D组故障下的EIR与DSTU增量](fg/figures/fg_joint_increment.png)")


def bandwidth_section(data):
    search=[r for r in data["bandwidth_search"] if r["protocol"]=="backend_best"]
    joint=data["bandwidth_joint"]
    mha2=next(r for r in search if r["d2d_scale"]==2 and r["method"]=="A3h" and r["op"]=="mha")
    ffn_half=next(r for r in search if r["d2d_scale"]==.5 and r["method"]=="A3h" and r["op"]=="ffn")
    crisp={r["d2d_scale"]:r for r in joint if r["arm"]=="CRISP"}
    return section("## 7. D2D 带宽变化",
        "只改变 WATOS 的 D2D 带宽，采用 0.5×/1×/2×。第一组在每种带宽重新搜索 decode MHA/FFN 的 A1/A3h；第二组固定复用 WATOS 1× 的 P/D 映射，重新评估配对单跑与联合执行。",
        table(["D2D倍数","算子","方法","时间/μs","相对1×时间"],[[r["d2d_scale"],r["op"].upper(),r["method"],number(r["cycles_ns"]/1000),number(1+r["relative_to_1x"])] for r in search]),
        f"A3h FFN 在半带宽下相对 1× 仅变化 {pct(ffn_half['relative_to_1x'])}；MHA 对半带宽更敏感。2× 带宽下重新搜索的 A3h MHA 比 1× 高 {pct(mha2['relative_to_1x'])}，各点保存的是各自搜索产生的候选，映射随带宽一起变化。",
        "下面为健康、中间相位 g=0.5，三个种子均值。各带宽均重新生成自己的公共参照与 KV 到达窗口。",
        table(["D2D倍数","方法","D延迟/μs","D/1×","P完成/μs","配对ΔP/μs"],[[r["d2d_scale"],r["arm"],number(r["D_latency_ns_mean"]/1000),number(r["D_latency_ns_ratio_to_1x"]),number(r["P_completion_ns_mean"]/1000),number(r["paired_delta_P_ns_mean"]/1000)] for r in joint]),
        f"固定映射的 CRISP 联合 D 延迟从 0.5× 的 {number(crisp[.5]['D_latency_ns_mean']/1000)} μs 降到 1× 的 {number(crisp[1]['D_latency_ns_mean']/1000)} μs，2× 为 {number(crisp[2]['D_latency_ns_mean']/1000)} μs。此配置在 1× 之后继续增加 D2D 带宽的收益已经很小。",
        "![带宽扫描](fg/figures/fg_bandwidth.png)",
        "完整选择结果与种子范围见 "+link("bandwidth_search")+"、"+link("bandwidth_joint")+"。")


def scaling_section(data):
    layers=[r for r in data["layers"] if r["stage"]=="G" and r["family"]=="healthy" and r["protocol"]=="backend_best"]
    rows=[[config(r),r["method"],number(r["prefill_speedup_vs_64cores"]),number(r["decode_speedup_vs_64cores"]),
           number(r["request_2048_speedup_vs_64cores"]),number(100*r["request_2048_strong_scaling_efficiency"],2)+"%"] for r in layers]
    baseline=[r for r in data["layers"] if r["stage"]=="E_reuse" and r["protocol"]=="backend_best"]
    baserows=[[r["method"],number(r["prefill_ns"]/1e6),number(r["decode_ns"]/1000),number(r["request_2048_ns"]/1e6)] for r in baseline]
    refinements=[]
    for shape in ("2x3","3x3"):
        fixed=next(r for r in layers if r["shape"]==shape and r["split_mode"]=="fixed" and r["method"]=="A3h")
        growth=next(r for r in layers if r["shape"]==shape and r["split_mode"]=="growth" and r["method"]=="A3h")
        refinements.append([shape.replace("x","×"),*[pct(growth[m+"_ns"]/fixed[m+"_ns"]-1) for m in ("prefill","decode","request_2048")]])
    return section("## 7. 相对 64 核的加速与资源利用",
        "加速比采用同方法、同选择方式的 E 2×2 基线：`S=T_64/T_n`；强扩展效率为 `S/(n/64)`。单位工作量保持不变。",
        table(["E基线方法","Prefill/ms","Decode/μs","请求2048/ms"],baserows),
        table(["新增配置","方法","Prefill加速","Decode加速","请求2048加速","请求强扩展效率"],rows),
        "同规模下，A3h 的增长切分相对固定切分变化如下：",
        table(["规模","Prefill变化","Decode变化","请求2048变化"],refinements),
        "固定切分增加了可选位置与路径，但原计算分片数不变；增长切分增加计算分片，同时增加部分归约、取整和通信事件。因此核数增加、工作量细分和映射质量分别影响结果。三种选择方式下的全部指标见 "+link("scaling")+"。",
        "![规模与优化成本](fg/figures/fg_scaling_cost.png)")


def cost_section(stage,data):
    candidates=[r for r in select(data["candidates"],stage) if r["family"]=="healthy" and r["kind"]=="search"]
    receipts={r["job_id"]:r for r in data["job_timing"]}
    groups=defaultdict(list)
    for r in candidates:
        if r["op"] in ("mha","ffn"):
            groups[r["config_id"],r["phase"],r["op"],r["method"]].append(r)
    rows=[]
    for _,rs in sorted(groups.items()):
        r=rs[0]
        rows.append([config(r),r["phase"],r["op"].upper(),r["method"],
                     interval([v["search_seconds"] for v in rs],scale=60),
                     interval([receipts[v["candidate_id"]]["wall_seconds"] for v in rs],scale=60),
                     number(max(receipts[v["candidate_id"]]["peak_rss_gb"] for v in rs))])
    graph=[r for r in select(data["graph_metrics"],stage) if r["family"]=="healthy" and r["protocol"]=="backend_best" and r["method"]=="A3h"]
    graphrows=[[config(r),r["phase"],r["op"].upper(),r["graph_events"],r["dependency_edges"],
                number(r["communication_bytes"]/2**20),number(r["routed_link_bytes"]/2**30),number(r["mean_branch_hops"]),
                number(r["cpcm_compile_seconds"]*1000),number(r["cpcm_simulate_seconds"]*1000)] for r in graph]
    components=[r for r in select(data["backend_components"],stage) if r["family"]=="healthy" and r["protocol"]=="backend_best" and r["op"] in ("mha","ffn")]
    componentrows=[[config(r),r["phase"],r["op"].upper(),r["method"],number(r["total_ns"]/1000),
                    number(100*r["compute_only_fraction"],2)+"%",number(100*r["communication_only_fraction"],2)+"%",number(100*r["overlap_fraction"],2)+"%"] for r in components]
    workload=select(data["workload"],stage)
    workrows=[[config(r),r["phase"],r["op"].upper(),number(r["tensor_effective_multiply_add_flops"]/1e9),
               number(r["tensor_grain_rounded_flops"]/1e9),pct(r["tensor_padding_fraction"]),r["reduction_events"],r["reduction_input_elements"]] for r in workload]
    loads=select(data["resource_loads"],stage)
    loadgroups=defaultdict(dict)
    for r in loads:
        if r["phase"]=="prefill" and r["op"] in ("mha","ffn"):
            loadgroups[r["config_id"],r["op"],r["method"]][r["resource"]]=r
    loadrows=[]
    for _,group in sorted(loadgroups.items()):
        r=next(iter(group.values()))
        loadrows.append([config(r),r["op"].upper(),r["method"],*[number(group[res]["maximum"]/1000)
                         for res in ("tensorcore","ddr2co","co2co","ch2ch")],
                         number(group["tensorcore"]["max_to_mean"]),number(group["ddr2co"]["max_to_mean"])])
    load_interpretation=""
    if loads:
        focus="F_watos_2x2_fixed_d2d1" if stage=="F" else "G_busybarn_3x3_growth_d2d1"
        ffn_a=loadgroups[focus,"ffn","A3h"]
        ffn_g=loadgroups[focus,"ffn","Gemini"]
        mha_a=loadgroups[focus,"mha","A3h"]
        mha_g=loadgroups[focus,"mha","Gemini"]
        name="WATOS" if stage=="F" else "3×3 增长切分"
        load_interpretation=(f"以{name}为例，prefill FFN 的 Gemini 最忙 TensorCore 累计工作为 {number(ffn_g['tensorcore']['maximum']/1e6)} ms，A3h 为 {number(ffn_a['tensorcore']['maximum']/1e6)} ms，最大值/平均值分别为 {number(ffn_g['tensorcore']['max_to_mean'])} 与 {number(ffn_a['tensorcore']['max_to_mean'])}。"
            f"prefill MHA 的 Gemini 最忙 DDR 读链与片内链分别承担 {number(mha_g['ddr2co']['maximum']/1e6)} / {number(mha_g['co2co']['maximum']/1e6)} ms，A3h 为 {number(mha_a['ddr2co']['maximum']/1e6)} / {number(mha_a['co2co']['maximum']/1e6)} ms。保存映射中同时出现计算和通信负载集中，与其较长的 prefill 后端时间相对应。")
    decomposition=""
    if stage=="F":
        watos=next(r for r in components if r["hardware_profile"]=="watos" and r["phase"]=="decode" and r["op"]=="ffn" and r["method"]=="A3h")
        face=next(r for r in components if r["hardware_profile"]=="face" and r["phase"]=="prefill" and r["op"]=="ffn" and r["method"]=="A3h")
        decomposition=f"例如 WATOS 的 A3h decode FFN 纯计算仅占 {share(watos['compute_only_fraction'])}，纯通信占 {share(watos['communication_only_fraction'])}；FACE 的 A3h prefill FFN 则有 {share(face['compute_only_fraction'])} 的纯计算。资源配比与阶段不同，增加算力或带宽所能压缩的时间不同。"
    return section("## 8. 图规模、搜索成本与资源工作量",
        "下面按每类十四个候选统计搜索与完整作业耗时；时间是本次并行服务器运行中测得的墙钟。完整作业用成功 receipt 计时，还含图构建、保存状态、十次后端与十次随机平局 CPCM 评估；Gemini 种子123还包含结束状态的处理与评估。峰值内存同样取完整作业 receipt。",
        table(["配置","阶段","算子","方法","搜索分钟：中位 [最小,最大]","作业分钟：中位 [最小,最大]","最大峰值GiB"],rows),
        "### A3h 所选图的大小与一次 CPCM 计算",
        table(["配置","阶段","算子","事件","依赖边","通信MiB","路由链GiB","平均路径跳数","编译ms","推演ms"],graphrows),
        "事件数、依赖边数、通信字节、路径长度、CPCM 编译与推演时间见 "+link("graph_metrics")+"。有效乘加量为 TensorCore 事件的 `2×M×K×N` 之和，取整量按原 128×64 grain 计算；matmac 的额外偏置加法另于乘加量之外。归约事件数和输入元素见 "+link("workload")+"。",
        table(["配置","阶段","算子","Tensor有效GFLOP","Tensor取整GFLOP","取整增加","归约事件","归约输入元素"],workrows) if workload else "快照工作量数据随完整下载包解析后写入 workload 表。",
        "八类算子的 TensorCore 有效乘加量在七个 F/G 配置中逐类一致；LN 的向量运算与新增归约另列于工作量表，头数与输入形状也已从快照核对。见 "+link("workload_checks")+"。",
        "### 后端时间构成",
        table(["配置","阶段","算子","方法","总时间/μs","纯计算占比","纯通信占比","重叠占比"],componentrows),
        "后端记录的三个数是总延迟、未被掩盖的纯计算、未被掩盖的纯通信；重叠时间由总延迟减后两项得到。按资源累加的计算、DDR、片内网络和 D2D 工作量由保存映射和通信路径提取，表示资源所承担的服务量。逐链路排队时间未保存。分解结果见 "+link("backend_components")+"。",
        decomposition,
        "### Prefill 映射中的资源负载",
        "下表给各资源类型中最忙的单个资源所承担的累计服务时间，单位 μs；最后两列是该最大值除以全部对应资源的平均值。计算采用保存映射、实际路径和本配置硬件服务公式。",
        table(["配置","算子","方法","最忙计算核","最忙DDR读链","最忙片内链","最忙D2D链","计算最大/平均","DDR最大/平均"],loadrows) if loads else "资源负载随快照和 raw 完整下载后从保存路径计算。",
        load_interpretation,
        "这些值是服务工作量，不含排队等待。Gemini 的原生 hops 目标、DDR 分配与 XY 路由一同保留，表格描述的是其最终映射呈现的负载分布；DDR 策略与路由各自的因果贡献没有在本阶段拆开测量。各类型的总量、最大值、资源数量和源快照见 "+link("resource_loads")+"。",
        "并行运行中的耗时包含资源争用，适合说明此次计算开销和排队需求；论文所需的串行优化时延在 J 阶段统一测量。控制器各次接续、逐任务耗时与失败重试分别见 "+link("controller_timing")+"、"+link("job_timing")+"、"+link("retry_attempts")+"。")


def reproduction(stage,data,completion):
    audit=data["summary"]["audit"]
    checks=data["workflow_checks"]
    issue_count=len(checks["missing_jobs"])
    timing=data["summary"]["observed_time"]
    sources=data["summary"]["sources"]
    run_out=Path(sources["run_output"]).as_posix()
    run_root=Path(sources["run_root"]).as_posix()
    workflow=Path(sources["workflow_manifest"]).as_posix()
    metadata_option=" --metadata-only" if audit["mode"]=="metadata_only" else ""
    raw_option=" --skip-raw" if not checks["raw_backend_samples_read"] else ""
    lines=[f"候选记录核对：{audit['completed_records']}/{audit['expected_records']}，问题 {len(audit['issues'])} 项；工作流清单读取 {checks['completed_jobs']}/{checks['planned_jobs']}，问题 {issue_count} 项。逐项记录见 [summary.json](fg/tables/summary.json)、[workflow_checks.json](fg/tables/workflow_checks.json)。",
           "完整输出位置：`E:\\58Institute\\PhD_TERM3\\CRISP_codex\\e2e\\fg\\out\\main`；归档位置：`e2e/fg/dist/`。下载包、解包文件和完整性核对记录见 [completion_audit.json](fg/evidence/completion_audit.json)。COS 路径为 `crisp/FG_20261113`，运行期间每三分钟上传。",
           f"搜索和 Gemini 终态共保留 {audit['backend_samples']:,} 个后端样本，其中 {audit['backend_calls']:,} 次实际调用、{audit['reused_backend_samples']:,} 个复用样本；随机平局 CPCM 为 {audit['random_tie_calls']:,} 次。全部 {timing['successful_job_receipts']:,} 份成功作业记录累计 {timing['successful_process_hours']:.3f} 进程小时。两次失败尝试共 {timing['failed_attempt_process_hours']:.3f} 进程小时，另行记录。",
           "正式队列于北京时间 2026-11-13 00:39:52 开始。早期 2×3 增长切分 MF6 的广播处理报错经修正后续跑成功；14:37 的一次 OOM 使一个联合作业失败、四项待续，其他任务继续完成。最后五项按五并行补齐，参数和种子保持原样。各控制器按任务身份合并统计，同一完成结果只计一次。",
           "实验实现和运行期修正见 [FG_REMOTE_GUIDE.md](FG_REMOTE_GUIDE.md)、[PROGRESS.md](PROGRESS.md) 的相应日期记录。BusyBarn artifact 版本取仓库 `third_party/isca2026-busybarn-artifact.COMMIT`；平台参数和软件环境随完整运行包保存。",
           "用户已手动关闭云服务器，以下后处理和复算均读取本机已保存文件。"]
    if checks["raw_backend_samples_read"]:
        raw_checks=read(ROOT/"e2e/fg/evidence/workflow_report_checks.json")["raw_backend_checks"]
        lines.append(f"故障后处理从 raw 读取 {raw_checks['raw_backend_samples']:,} 个健康/修复后端样本，另引用搜索部分的 {raw_checks['FULL_backend_samples']:,} 个所选 FULL 样本；{raw_checks['state_means_checked']:,} 个状态的十种子均值均与原始 JSON 一致。{raw_checks['R0_L1_decision_times_checked']} 个 R0/L1 停止状态的决策时长已从 raw 补齐；L2 的各时间分项也已独立相加核对。FULL 样本属于前面的搜索样本，不重复计数。")
    if completion:
        elapsed=(datetime.fromisoformat(completion["completed_beijing"])-datetime.fromisoformat(completion["formal_started_beijing"])).total_seconds()
        downloaded="已下载" if completion["complete_archive_downloaded"] else "下载进行中"
        verified="实验输出与运行源码全部匹配清单" if completion["experiment_file_hashes_verified"] else "待核对"
        lines.append(table(["项目","实际记录"],[
            ["正式开始（北京时间）",completion["formal_started_beijing"]],
            ["最后作业结束（北京时间）",completion["completed_beijing"]],
            ["正式开始至最后结束",f"{int(elapsed//3600)}小时{int(elapsed%3600//60)}分{int(elapsed%60)}秒（小批开始更早，已计入总作业）"],
            ["最后五项续跑",f"{completion['finish_retry']['wall_seconds']:.3f}秒，5并行；采样最小可用内存{completion['finish_retry']['min_available_gib']:.3f} GiB"],
            ["小批／正式启动源码", "35711b9 ／ b2a1543"],
            ["后续接续源码提交",completion["runtime_source_commit"]],
            ["执行环境", "Linux，Python 3.13.18；128 vCPU，并行上限128；最后五项使用5并行"],
            ["BusyBarn artifact",(ROOT/"third_party/isca2026-busybarn-artifact.COMMIT").read_text(encoding="utf-8-sig").splitlines()[0]],
            ["完整包",downloaded],["实验输出与源码摘要",verified],
        ]))
        if "complete_archive" in completion:
            archive=completion["complete_archive"]
            lines.append(table(["归档","字节","文件数","SHA-256"],[[archive["name"],archive["bytes"],archive["files"],archive["sha256"]]]))
            extraction=completion["extraction"]
            lines.append(f"完整压缩包的 SHA-256 与服务器记录一致。解包 {extraction['files']:,} 个文件，清单记录 {extraction['hashed_files']:,} 个文件摘要，其中 {extraction['matching_files']:,} 个匹配。唯一差异是 `build_final_archive.log` 在生成清单后追加了 26 字节的文件数量行；该打包日志单列，实验输出与运行源码全部匹配。`FG_20261113_files.json` 清单自身作为归档文件另计。")
        lines.append("最终下载与结束状态由本次 [完成核对记录](fg/evidence/completion_audit.json) 给出。该记录分别保存元数据包与完整包的 SHA-256。")
    lines += ["在仓库顶层复算，以下命令只读取已保存结果并重建表、图和正文：",
        f"```powershell\npython -B e2e/fg_report.py --manifest e2e/fg/manifest/search.json --run-out {run_out} --out e2e/fg/tables --workflow-manifest {workflow} --run-root {run_root}{metadata_option}\npython -B e2e/fg_workflow_report.py --manifest e2e/fg/manifest/all_jobs.json --run-out {run_out} --out e2e/fg/tables{raw_option}\npython -B e2e/fg_figures.py --tables e2e/fg/tables --out e2e/fg/figures\npython -B e2e/fg_summary.py --tables e2e/fg/tables --out e2e --completion e2e/fg/evidence/completion_audit.json\n```",
        "独立算术核对覆盖层组装、选择规则、种子配对、停止判断和完整作业计时；结果见 [healthy_report_checks.json](fg/evidence/healthy_report_checks.json) 与 [workflow_report_checks.json](fg/evidence/workflow_report_checks.json)。复算检查：",
        f"```powershell\npython -B e2e/fg_report_check.py --tables e2e/fg/tables --run-out {run_out} --out e2e/fg/evidence/healthy_report_checks.json\npython -B e2e/fg_workflow_check.py --tables e2e/fg/tables --run-out {run_out} --report e2e/fg/evidence/workflow_report_checks.json\n```",
        "完整数据表同步保存 CSV 和 JSON；四类主图同步保存 PNG、SVG 和逐点数据。GitHub 分支为 `codex/e2e-handoff`，结果原始包保存在本机和 COS，分析代码、正文、表、图及核对证据按仓库约定推送。"]
    return section("## 9. 运行记录、下载与复算",*lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables",type=Path,default=ROOT/"e2e/fg/tables")
    parser.add_argument("--out",type=Path,default=ROOT/"e2e")
    parser.add_argument("--manifest",type=Path,default=ROOT/"e2e/fg/manifest/all_jobs.json")
    parser.add_argument("--completion",type=Path)
    args=parser.parse_args()
    names=("candidates","operators","layers","fault_configs","fault_summary","fault_layers","fault_tiers","joint_summary",
           "joint_references","joint_ffn_checks","bandwidth_search","bandwidth_joint","summary","workflow_checks",
           "gemini_terminal_pairs","graph_metrics","backend_components","workload","resource_loads","joint_comparison_summary","joint_runs","job_timing")
    data={name:read(args.tables/(name+".json")) for name in names}
    jobs=read(args.manifest)
    completion=read(args.completion) if args.completion else None
    args.out.mkdir(parents=True,exist_ok=True)
    for stage in ("F","G"):
        parts=[intro(stage,data),protocol(stage,data,jobs),hardware_section() if stage=="F" else scale_section(),
               healthy_section(stage,data),variability_section(stage,data),fault_section(stage,data),joint_section(stage,data),
               bandwidth_section(data) if stage=="F" else scaling_section(data),cost_section(stage,data),reproduction(stage,data,completion)]
        if stage=="F":
            parts[3]+="\n\n![硬件性能与故障恢复](fg/figures/fg_hardware_recovery.png)"
        path=args.out/(stage+"_SUMMARY.md")
        path.write_text("\n\n".join(parts)+"\n",encoding="utf-8")
        print(str(path))


if __name__=="__main__":
    main()
