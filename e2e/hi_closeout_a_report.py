"""A report and saved-state CPCM explanations; no backend or SA evaluation."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import time

import hi_closeout_a as run
from hi_closeout_p0 import csv_write, json_write

ROOT, BASE, OLD, OUT = run.ROOT, run.BASE, run.OLD, run.OUT
KINDS = {"dev:tensorcore":"TensorCore", "dev:vectorunit":"Vector", "ch2ch":"D2D",
         "co2co":"NoC", "ddr2co":"DDR", "co2ddr":"DDR"}
CATEGORIES = ("TensorCore", "Vector", "D2D", "NoC", "DDR", "release_gap_other")


def paths(out, progress):
    inputs = run.frozen.read_json(run.INPUTS)
    old = run.frozen.read_json(inputs["old_300_result"])
    cid = old["spec"]["config_id"]
    pre_path = OLD/"results"/(cid+"__pre_dc_ffn_A3h.json")
    l1_path = OLD/"results"/(cid+"__l1_dc_ffn_A3h_PF4.json")
    pre, l1 = run.frozen.read_json(pre_path), run.frozen.read_json(l1_path)
    states = [dict(state="healthy", repair_seed=None, budget=0, export=pre["exports"]["healthy"],
                   T_hat_ns=pre["T"], backend_mean_ns=pre["backend_mean"], source_result=run.frozen.relative(pre_path), spec=pre["spec"]),
              *[dict(state=level, repair_seed=None, budget=0, export=l1["exports"][level],
                     T_hat_ns=l1["T_"+level], backend_mean_ns=l1[level+"_backend_mean"],
                     source_result=run.frozen.relative(l1_path), spec=l1["spec"]) for level in ("R0","L1")]]
    results = []
    for budget in (30,100,300,1000):
        for seed in run.SEEDS:
            source = (out/"results"/f"A__qwen3moe_busybarn_dc_ffn_PF4_s1000_r{seed}.json" if budget == 1000 else
                      OLD/"results"/f"{cid}__rep_dc_ffn_PF4_s{budget}_r{seed}.json")
            if budget == 1000 and progress and not source.exists():
                continue
            result = run.frozen.read_json(source)
            assert result["status"] == "ok" and result["steps"] == budget and result["seed"] == seed
            results.append((source,result))
            if budget >= 300:
                states.append(dict(state=f"s{budget}", repair_seed=seed, budget=budget,
                    export=result["exports"]["final"], T_hat_ns=result["T"], backend_mean_ns=result["backend_mean"],
                    source_result=run.frozen.relative(source), spec=result["spec"]))
    return inputs, results, states


def budget_tables(inputs, results, healthy, r0):
    rows, samples, checkpoints = [], [], []
    for path, result in results:
        raw = run.frozen.read_pickle(result["sink"])
        search = raw["search"]
        spec = result["spec"]
        seed, steps = spec["seed"], spec["steps"]
        assert len(search["rows"]) == steps and raw["proposals"]["steps"] == steps
        assert raw["T_L1_check"] == search["T_start"]
        assert raw["spec"]["checks"] is False and search["check_revert"]["n"] == 0
        assert set(raw["record"]["backend"]) == set(range(1,11))
        row = dict(budget=steps, repair_seed=seed, job_id=result["job_id"], T_hat_ns=result["T"],
            backend_mean_ns=result["backend_mean"], backend_min_ns=min(v[0] for v in raw["record"]["backend"].values()),
            backend_max_ns=max(v[0] for v in raw["record"]["backend"].values()),
            relative_to_healthy=result["backend_mean"]/healthy-1, relative_to_R0=result["backend_mean"]/r0-1,
            T_L1_check=raw["T_L1_check"], best_E=search["best_E"], best_step=search["best_step"],
            search_proposals=len(search["rows"]), accepted_proposals=sum(r["accepted"] for r in search["rows"]),
            calibration_proposals=raw["proposals"]["calibration"], calibration_seconds=search["t_calibration"],
            search_seconds=search["t_L2_search"], restore_best_seconds=search["t_restore_best"],
            final_cpcm_seconds=raw["record"]["t_finalize"], backend_validation_seconds=raw["record"]["t_backend"],
            backend_actual_calls_seconds=sum(raw["record"]["backend_wall"]),
            new_decision_seconds=search["t_calibration"]+search["t_L2_search"]+search["t_restore_best"]+raw["record"]["t_finalize"],
            inherited_R0_L1_seconds=raw["timing"]["t_R0"]+raw["timing"]["t_L1"],
            full_job_seconds=result["wall_seconds"], peak_rss_gib=result["peak_rss_gb"],
            migration_unique_block_bytes=result["migration"]["unique_block_bytes"], migration=result["migration"],
            source=run.frozen.relative(path), source_sha256=run.sha(path), raw=result["sink"], raw_sha256=run.sha(run.frozen.path(result["sink"])))
        rows.append(row)
        for backend_seed, triple in sorted(raw["record"]["backend"].items()):
            samples.append(dict(budget=steps,repair_seed=seed,backend_seed=backend_seed,backend_ns=triple[0],
                unhidden_compute_ns=triple[1],unhidden_communication_ns=triple[2],job_id=result["job_id"]))
        if steps == 1000:
            assert result["source_input_sha256"] == inputs["sha256"] and result["actual_search_proposals"] == 1000
            assert result["backend_calls"] == 10 and result["checkpoint_backend_calls"] == 0
            assert result["finish_reason"] == "independent_1000_step_budget_completed" and not result["original_stop300_inherited"]
            assert [c["step"] for c in search["checkpoints"]] == run.CHECKPOINTS
            for ck, meta in zip(search["checkpoints"], result["checkpoints"]):
                assert all(ck[k] == meta[k] for k in ("step","best_E","best_T","best_step","t_search"))
                assert meta["snapshot_sha256"] == run.sha(run.frozen.path(meta["snapshot"]))
                checkpoints.append(dict(job_id=result["job_id"], **meta))
    return rows, samples, checkpoints


def explain(state, out, fault):
    import f_config
    import cpcm
    began = time.perf_counter()
    hp = f_config.build_platform(state["spec"], out/"a_diagnostics", faults=None if state["state"] == "healthy" else fault)
    exported = run.frozen.read_pickle(state["export"])
    assert run.frozen.platform_signature(hp) == exported["platform_sha256"]
    events = pickle.loads(exported["events_blob"])
    graph = cpcm.compile_bb(events, hp)
    prepared = time.perf_counter()
    schedule = cpcm.simulate(graph, mode="bb")
    simulated = time.perf_counter()
    assert schedule.T == state["T_hat_ns"] == exported["T"]
    analysis = cpcm.Analysis(schedule)
    chain = analysis.critical_path()
    blocking = analysis.blocking()
    common = {k:state[k] for k in ("state","repair_seed","budget","T_hat_ns","backend_mean_ns","source_result","export")}
    common["export_sha256"] = run.sha(run.frozen.path(state["export"]))
    # A chain edge carries lag from predecessor start to successor start. Hop edges can carry zero.
    segments = []
    if chain[0]["s"]:
        segments.append(dict(**common, category="release_gap_other", begin_ns=0, end_ns=chain[0]["s"], duration_ns=chain[0]["s"],
                             resource="initial_release", successor_edge="start"))
    for index, activity in enumerate(chain):
        end = chain[index+1]["s"] if index+1 < len(chain) else schedule.T
        start = activity["s"]
        assert end >= start
        service_end = min(activity["f"], end)
        category = KINDS[graph.res_kind[activity["res"]]]
        successor_edge = chain[index+1]["edge"] if index+1 < len(chain) else "finish"
        if service_end > start:
            segments.append(dict(**common, category=category, begin_ns=start, end_ns=service_end,
                duration_ns=service_end-start, resource=repr(graph.res_name[activity["res"]]), successor_edge=successor_edge))
        if end > service_end:
            segments.append(dict(**common, category="release_gap_other", begin_ns=service_end, end_ns=end,
                duration_ns=end-service_end, resource="gap_after_"+repr(graph.res_name[activity["res"]]), successor_edge=successor_edge))
    assert sum(r["duration_ns"] for r in segments) == schedule.T
    assert segments[0]["begin_ns"] == 0 and segments[-1]["end_ns"] == schedule.T
    assert all(a["end_ns"] == b["begin_ns"] for a,b in zip(segments,segments[1:]))
    totals = Counter()
    for row in segments:
        totals[row["category"]] += row["duration_ns"]
    composition = [dict(**common, category=k, chain_time_ns=totals[k], share_of_T_hat=totals[k]/schedule.T) for k in CATEGORIES]
    waits = [dict(**common, category=KINDS[graph.res_kind[q]], resource=repr(graph.res_name[q]),
                  cb_excl_ns=value, share_of_T_hat=value/schedule.T) for q,value in sorted(blocking.items(),key=lambda kv:(-kv[1],kv[0]))]
    loads = []
    for q, acts in enumerate(schedule.res_seq):
        if acts:
            service = sum(schedule.a_f[a]-schedule.a_s[a] for a in acts)
            loads.append(dict(**common, category=KINDS[graph.res_kind[q]], resource=repr(graph.res_name[q]),
                resource_service_ns=service, activities=len(acts), service_over_T_hat=service/schedule.T))
    ended = time.perf_counter()
    timing = dict(**common, prepare_and_compile_seconds=prepared-began, fixed_cpcm_seconds=simulated-prepared,
        analysis_seconds=ended-simulated, diagnosis_seconds=ended-began, backend_calls=0,
        activities=len(schedule.a_s), chain_activities=len(chain), chain_sum_ns=sum(totals.values()),
        cb_excl_total_ns=sum(blocking.values()), chain_segments_nonoverlap_and_sum_to_T_hat=True,
        resource_classification="actual link/device type; DDR message transit credited per actual NoC/D2D link",
        execution="original deterministic CPCM bb; saved-state replay, not backend event trace")
    return composition, waits, loads, segments, timing


def render(output, rows, checkpoints, diagnostics, complete):
    summary = []
    for budget in (30,100,300,1000):
        group = [r for r in rows if r["budget"] == budget]
        if group:
            summary.append(dict(budget=budget,n=len(group),backend_median_ns=statistics.median(r["backend_mean_ns"] for r in group),
                backend_min_ns=min(r["backend_mean_ns"] for r in group), backend_max_ns=max(r["backend_mean_ns"] for r in group),
                T_hat_median_ns=statistics.median(r["T_hat_ns"] for r in group),
                decision_total_seconds=sum(r["new_decision_seconds"] for r in group),
                decision_median_seconds=statistics.median(r["new_decision_seconds"] for r in group),
                backend_validation_total_seconds=sum(r["backend_validation_seconds"] for r in group),
                job_process_seconds=sum(r["full_job_seconds"] for r in group)))
    csv_write(output/"tables/a_budget_summary.csv",summary)
    lines=["# A：Qwen decode FFN / PF4 的独立预算诊断","",
        ("五条完整 1000 步轨迹和 50 次最终后端已齐。" if complete else "当前报告保留已完成的原 30/100/300 步与已收到的 1000 步结果，1000 步未齐五种子时不形成最终比较。"),"",
        "主硬件 BusyBarn 2×2，健康起点为既有 A3h 赢家 I_qwen3moe_busybarn_decode_ffn_A3h_g0_s6，故障为原 PF4。1000 步从同一 L1 快照独立开始，没有继承 stop300；动作、校准、L2G+sync、精确回滚和十后端种子沿用原规则。搜索使用已验证确定性 bb 加速，关闭额外 checks。","",
        "## 独立预算终态","",
        "下表以各修复状态的十种子后端均值为单位，再在五个修复种子间取中位数与范围。30/100/300 为已完成主协议，1000 为单列预算诊断。", "",
        "| 独立预算 | 修复种子数 | 后端均值的中位数 [最小,最大] µs | CPCM 中位 µs | 单轨迹决策中位 s | 本档全部候选决策 s | 本档后端验证 s |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['budget']} | {r['n']} | {r['backend_median_ns']/1000:.3f} [{r['backend_min_ns']/1000:.3f}, {r['backend_max_ns']/1000:.3f}] | {r['T_hat_median_ns']/1000:.3f} | {r['decision_median_seconds']:.3f} | {r['decision_total_seconds']:.3f} | {r['backend_validation_total_seconds']:.3f} |")
    if complete:
        old,new=([r for r in summary if r["budget"]==n][0] for n in (300,1000))
        lines += ["",f"1000 步相对独立 300 步的后端中位变化为 {new['backend_median_ns']/old['backend_median_ns']-1:+.3%}；五候选新增决策成本 {new['decision_total_seconds']:.1f} 秒，后端验证 {new['backend_validation_total_seconds']:.1f} 秒。", "",
            "这项对比检验增加预算的作用；无论有无残差，都不据此自动追加搜索预算。仅凭曲线变平也不能证明达到拓扑容量下界。"]
    if checkpoints:
        lines += ["", "![独立预算终态与1000步内最佳状态](figures/a_budget_convergence.png)"]
    lines += ["", "每条轨迹的新增决策成本为校准、搜索、最佳状态恢复及终态 CPCM；继承的 R0/L1 时间单独保存，未再次计入新执行。五候选研究汇总成本与单轨迹成本分列。升级到原 300 步的流程累计包括原 30、100、300 三档；1000 步诊断本身不是这个在线升级规则的新增正式档位。", "",
        "[独立预算逐状态表](tables/a_independent_budgets.csv)；[全部后端样本](tables/a_backend_samples.csv)。", "",
        "## 1000 步日程内检查点", "",
        f"已保存 {len(checkpoints)} 个检查点。每条完整轨迹记录 30/100/300/500/700/1000 步的当前最小 E 状态及对应 T̂、best_step、接受数、校准和累计搜索时间、快照路径。ck_backend=False，没有中途后端评估。", "",
        "1000 步日程的温度随总预算定义，第 300 步检查点不替代原 300 步独立实验。最佳状态由既定 E（含微小距离项）选出，没有从拒绝提议的 T_new 中另取最小值。", "",
        "[检查点逐行数据](tables/a_checkpoints.csv)。本机检查点开关验证使用同 L1 两条 8 步轨迹与各 3 次校准，逐提议、最终状态和 Python/NumPy 随机状态一致，零后端；它只验证记录行为，不计入正式质量样本。", "",
        "## CPCM 时序解释", "",
        f"已对 {len(diagnostics)} 个保存状态做固定确定性 CPCM 推演，后端新增 0；诊断总耗时 {sum(r['diagnosis_seconds'] for r in diagnostics):.3f} 秒。每次平台身份与保存值匹配，重放 T̂ 与原值完全相等。", "",
        "关键链构成按前驱活动开始到后继开始的 lag 分段，最后计至 T̂；hop 同时发放的零 lag 不重复计时。每段不重叠，总和严格等于该次 T̂。类别按实际资源划分 TensorCore、Vector、D2D、片内 NoC、DDR 接口和 release/gap/other；DDR 发出消息在 NoC/D2D 上的服务归入实际链路。", "",
        "排他等待使用原 Analysis.blocking() 默认 cb_excl，另列主要资源绝对等待量。关键链构成和等待归因是两种解释，不相加。它们来自 CPCM 保存状态，不是逐事件后端实测；后端总延迟在每行旁列。", "",
        "[关键链构成](tables/a_chain_composition.csv)、[连续时间分段](tables/a_chain_segments.csv)、[排他等待](tables/a_exclusive_wait.csv)、[资源负载](tables/a_resource_load.csv)、[诊断耗时](tables/a_diagnostic_timing.csv)。", ""]
    (output/"A_QWEN_DFFN_BUDGET.md").write_text("\n".join(lines),encoding="utf-8")


def plot(output, rows, checkpoints):
    if not checkpoints:return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for seed in run.SEEDS:
        selected=[r for r in rows if r["repair_seed"]==seed]
        axes[0].plot([r["budget"] for r in selected],[r["backend_mean_ns"]/1000 for r in selected],"o-",label=str(seed),alpha=.8)
        selected=[r for r in checkpoints if r["repair_seed"]==seed]
        if selected:axes[1].plot([r["step"] for r in selected],[r["best_T"]/1000 for r in selected],"o-",label=str(seed),alpha=.8)
    axes[0].set(title="独立预算终态（十后端种子均值）",xlabel="独立搜索预算",ylabel="后端延迟（µs）",xticks=[30,100,300,1000])
    axes[1].set(title="1000 步内当前最优 E 状态对应的 T̂",xlabel="检查点步数",ylabel="确定性 CPCM T̂（µs）",xticks=run.CHECKPOINTS)
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(title="修复种子",fontsize=8,ncol=2)
    fig.tight_layout();folder=output/"figures";folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/"a_budget_convergence.png",dpi=180);fig.savefig(folder/"a_budget_convergence.svg");plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--output",type=Path,default=BASE)
    parser.add_argument("--progress",action="store_true")
    args=parser.parse_args()
    import post_fg_split
    post_fg_split.setup()
    inputs,results,states=paths(args.out,args.progress)
    budget,samples,checkpoints=budget_tables(inputs,results,states[0]["backend_mean_ns"],states[1]["backend_mean_ns"])
    fault=run.frozen.read_json(inputs["paths"]["faults"])["faults"]["PF4"]
    composition,waits,loads,segments,diagnostics=[],[],[],[],[]
    for state in states:
        c,w,l,s,d=explain(state,args.out,fault)
        composition+=c;waits+=w;loads+=l;segments+=s;diagnostics.append(d)
    tables=dict(a_independent_budgets=budget,a_backend_samples=samples,a_checkpoints=checkpoints,
        a_chain_composition=composition,a_exclusive_wait=waits,a_resource_load=loads,a_chain_segments=segments,a_diagnostic_timing=diagnostics)
    for name,rows in tables.items():
        if rows: csv_write(args.output/"tables"/(name+".csv"),rows)
    complete=sum(r["budget"]==1000 for r in budget)==5
    if not args.progress: assert complete and len(checkpoints)==30 and len(diagnostics)==13
    plot(args.output,budget,checkpoints)
    render(args.output,budget,checkpoints,diagnostics,complete)
    audit=dict(status="complete" if complete else "partial",independent_budget_rows=len(budget),
        new_1000_step_searches=sum(r["budget"]==1000 for r in budget),
        new_1000_backend_samples=sum(r["budget"]==1000 for r in samples), checkpoint_rows=len(checkpoints),
        diagnostics=len(diagnostics), diagnostic_backend_calls=0,
        all_saved_T_hat_equal=True, all_chain_segment_sums_equal_T_hat=True,
        exclusive_wait_source="Analysis.blocking(raw=False)", source_input_sha256=inputs["sha256"])
    json_write(args.output/"evidence/a_report_audit.json",audit)
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
