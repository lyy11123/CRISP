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
        assert [r["step"] for r in search["rows"]] == list(range(1, steps+1))
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
            Tmax=search["Tmax"], Tmin=search["Tmin"], final_step_temperature=search["rows"][-1]["temp"],
            search_seconds=search["t_L2_search"], restore_best_seconds=search["t_restore_best"],
            final_cpcm_seconds=raw["record"]["t_finalize"], backend_validation_seconds=raw["record"]["t_backend"],
            backend_actual_calls_seconds=sum(raw["record"]["backend_wall"]),
            new_decision_seconds=search["t_calibration"]+search["t_L2_search"]+search["t_restore_best"]+raw["record"]["t_finalize"],
            inherited_R0_L1_seconds=raw["timing"]["t_R0"]+raw["timing"]["t_L1"],
            full_job_seconds=result["wall_seconds"], peak_rss_gib=result["peak_rss_gb"],
            migration_unique_block_bytes=result["migration"]["unique_block_bytes"], migration=result["migration"],
            source=run.frozen.relative(path), source_sha256=run.sha(path), raw=result["sink"], raw_sha256=run.sha(run.frozen.path(result["sink"])))
        rows.append(row)
        row["other_job_seconds"] = row["full_job_seconds"]-row["new_decision_seconds"]-row["backend_validation_seconds"]
        row["backend_copy_setup_seconds"] = row["backend_validation_seconds"]-row["backend_actual_calls_seconds"]
        assert row["other_job_seconds"] >= 0
        for backend_seed, triple in sorted(raw["record"]["backend"].items()):
            samples.append(dict(budget=steps,repair_seed=seed,backend_seed=backend_seed,backend_ns=triple[0],
                unhidden_compute_ns=triple[1],unhidden_communication_ns=triple[2],job_id=result["job_id"]))
        if steps == 1000:
            assert result["source_input_sha256"] == inputs["sha256"] and result["actual_search_proposals"] == 1000
            assert result["backend_calls"] == 10 and result["checkpoint_backend_calls"] == 0
            assert result["finish_reason"] == "independent_1000_step_budget_completed" and not result["original_stop300_inherited"]
            assert [c["step"] for c in search["checkpoints"]] == run.CHECKPOINTS
            assert all(a["best_E"] >= b["best_E"] for a,b in zip(search["checkpoints"],search["checkpoints"][1:]))
            assert search["checkpoints"][-1]["best_T"] == search["best_T"] == result["T"]
            for ck, meta in zip(search["checkpoints"], result["checkpoints"]):
                assert all(ck[k] == meta[k] for k in ("step","best_E","best_T","best_step","t_search"))
                assert meta["snapshot_sha256"] == run.sha(run.frozen.path(meta["snapshot"]))
                checkpoints.append(dict(job_id=result["job_id"], **meta))
    return rows, samples, checkpoints


def comparison_tables(rows, checkpoints, composition, waits, loads, diagnostics):
    paired, checkpoint_change = [], []
    for seed in run.SEEDS:
        old = next(r for r in rows if r["budget"] == 300 and r["repair_seed"] == seed)
        new = next((r for r in rows if r["budget"] == 1000 and r["repair_seed"] == seed), None)
        if new is None: continue
        paired.append(dict(repair_seed=seed, backend_300_ns=old["backend_mean_ns"], backend_1000_ns=new["backend_mean_ns"],
            backend_change=new["backend_mean_ns"]/old["backend_mean_ns"]-1,
            T_hat_300_ns=old["T_hat_ns"], T_hat_1000_ns=new["T_hat_ns"], T_hat_change=new["T_hat_ns"]/old["T_hat_ns"]-1,
            healthy_residual_300=old["relative_to_healthy"], healthy_residual_1000=new["relative_to_healthy"],
            migration_300_bytes=old["migration_unique_block_bytes"], migration_1000_bytes=new["migration_unique_block_bytes"],
            decision_cost_ratio=new["new_decision_seconds"]/old["new_decision_seconds"],
            best_step_300=old["best_step"], best_step_1000=new["best_step"],
            accepted_300=old["accepted_proposals"], accepted_1000=new["accepted_proposals"]))
        ck = {r["step"]:r for r in checkpoints if r["repair_seed"] == seed}
        checkpoint_change.append(dict(repair_seed=seed, **{f"T_at_{n}_ns":ck[n]["best_T"] for n in run.CHECKPOINTS},
            within_300_to_1000=ck[1000]["best_T"]/ck[300]["best_T"]-1,
            within_700_to_1000=ck[1000]["best_T"]/ck[700]["best_T"]-1,
            independent300_vs_checkpoint300=ck[300]["best_T"]/old["T_hat_ns"]-1,
            old300_equals_1000_checkpoint300=ck[300]["best_T"] == old["T_hat_ns"],
            additional_search_300_to_1000_seconds=ck[1000]["t_search"]-ck[300]["t_search"]))
    state_rows = []
    for item in diagnostics:
        key = item["state"], item["repair_seed"]
        select = lambda source:[r for r in source if (r["state"],r["repair_seed"]) == key]
        chain, wait, load = select(composition), select(waits), select(loads)
        cat_wait = {k:sum(r["cb_excl_ns"] for r in wait if r["category"] == k) for k in CATEGORIES}
        lead_wait = max(wait,key=lambda r:r["cb_excl_ns"])
        peak = max(load,key=lambda r:r["resource_service_ns"])
        state_rows.append(dict(state=key[0],repair_seed=key[1],T_hat_ns=item["T_hat_ns"],backend_mean_ns=item["backend_mean_ns"],
            backend_vs_CPCM=item["backend_mean_ns"]/item["T_hat_ns"]-1,
            **{f"chain_{r['category']}_ns":r["chain_time_ns"] for r in chain},
            **{f"wait_{k}_ns":v for k,v in cat_wait.items()},
            cb_excl_total_ns=sum(cat_wait.values()), lead_wait_category=lead_wait["category"],
            lead_wait_resource=lead_wait["resource"], lead_wait_ns=lead_wait["cb_excl_ns"],
            peak_load_category=peak["category"], peak_load_resource=peak["resource"],
            peak_load_ns=peak["resource_service_ns"], peak_load_over_T_hat=peak["service_over_T_hat"]))
    return paired, checkpoint_change, state_rows


def completion_evidence(results, inputs, out):
    jobs={j["id"]:j for j in run.frozen.read_json(run.MANIFEST)["jobs"]}
    receipts=[]
    for path, result in results:
        if result["steps"] != 1000:continue
        job=jobs[result["job_id"]]
        payload={k:job[k] for k in ("id","kind","spec","result")};payload["argv"]=job.get("argv")
        identity=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        receipt_path=out/"receipts"/(job["id"]+".json")
        receipt=run.frozen.read_json(receipt_path)
        assert receipt["rc"] == 0 and receipt["identity"] == identity and not receipt["cached"]
        assert result["spec"] == job["spec"]
        assert result["protocol"]["exact_rollback"] and not result["protocol"]["debug_checks"]
        receipts.append(dict(job_id=job["id"],identity=identity,receipt_seconds=receipt["wall_seconds"],
            result_sha256=run.sha(path),receipt_sha256=run.sha(receipt_path),protocol=result["protocol"]))
    for key,value in inputs["sha256"].items():
        assert run.sha(run.frozen.path(inputs["paths"][key])) == value
    return receipts


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


def render(output, rows, checkpoints, diagnostics, complete, paired, checkpoint_change, state_rows):
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
        healthy=next(r for r in state_rows if r["state"]=="healthy")
        r0=next(r for r in state_rows if r["state"]=="R0")
        l1=next(r for r in state_rows if r["state"]=="L1")
        improved=sum(r["backend_change"]<0 for r in paired)
        lines += ["", f"健康、R0、L1 的十次后端均值分别为 {healthy['backend_mean_ns']/1000:.3f}、{r0['backend_mean_ns']/1000:.3f}、{l1['backend_mean_ns']/1000:.3f} µs。五个修复种子中 {improved}/5 在独立 1000 步下改善；后端中位健康残差由 {old['backend_median_ns']/healthy['backend_mean_ns']-1:+.2%} 降到 {new['backend_median_ns']/healthy['backend_mean_ns']-1:+.2%}。新增预算有实际收益，且尚未回到健康参考附近。原 300 步仍是正式停止规则得到的主结果。"]
        lines += [f"五个1000步终态的确定性 T̂ 仍高于健康 T̂ 的1.02倍；其最小残差为 {min(r['T_hat_ns'] for r in rows if r['budget']==1000)/healthy['T_hat_ns']-1:+.2%}，因此都未进入既定2%恢复范围。"]
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
    if complete:
        lines += detail_sections(rows, paired, checkpoint_change, state_rows)
    (output/"A_QWEN_DFFN_BUDGET.md").write_text("\n".join(lines),encoding="utf-8")


def detail_sections(rows, paired, checkpoint_change, states):
    lines=["## 各种子的变化与迁移", "",
        "同一个修复种子在两个独立预算下分别运行，温度日程也随预算变化。下表是种子标签对应的终态比较。", "",
        "| 修复种子 | 300→1000 后端 µs | 后端变化 | 1000 步健康残差 | 唯一迁移块 MiB，300→1000 | 1000 最佳步 / 接受数 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['repair_seed']} | {r['backend_300_ns']/1000:.3f} → {r['backend_1000_ns']/1000:.3f} | {r['backend_change']:+.3%} | {r['healthy_residual_1000']:+.3%} | {r['migration_300_bytes']/2**20:.3f} → {r['migration_1000_bytes']/2**20:.3f} | {r['best_step_1000']} / {r['accepted_1000']} |")
    bad=next(r for r in paired if r["repair_seed"]==4)
    good=sorted(paired,key=lambda r:r["backend_change"])
    lines += ["", f"seed 2、3 的后端分别改善 {good[0]['backend_change']:.2%}、{good[1]['backend_change']:.2%}。seed 4 的独立 1000 步后端为 {bad['backend_change']:+.3%}，确定性 CPCM 也为 {bad['T_hat_change']:+.3%}；因此增加预算的收益取决于轨迹，不能由步数直接保证。",
        f"五状态唯一迁移块中位数由 {statistics.median(r['migration_300_bytes'] for r in paired)/2**20:.3f} MiB 增到 {statistics.median(r['migration_1000_bytes'] for r in paired)/2**20:.3f} MiB。这里统计去重后的实际迁移块字节；移动的行为数与DDR读取来源变化另保留在原 migration 字段中。", "",
        "## 同一条 1000 步轨迹后半段还有多少变化", "",
        "| 修复种子 | 第300步 T̂ µs | 第700步 T̂ µs | 第1000步 T̂ µs | 300→1000变化 | 700→1000变化 | 原独立300与本轨迹第300步的 T̂ 是否相等 |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for r in checkpoint_change:
        lines.append(f"| {r['repair_seed']} | {r['T_at_300_ns']/1000:.3f} | {r['T_at_700_ns']/1000:.3f} | {r['T_at_1000_ns']/1000:.3f} | {r['within_300_to_1000']:+.3%} | {r['within_700_to_1000']:+.3%} | {'是' if r['old300_equals_1000_checkpoint300'] else '否'} |")
    early=next(r for r in checkpoint_change if r["repair_seed"]==1)
    lines += ["", f"五条轨迹在第 300 步以后都找到更小 E 的状态。seed 1 的主要下降发生在 700→1000 步，T̂ 再降 {early['within_700_to_1000']:.3%}；seed 2、3 后段也继续改善。seed 123、4 在 700 步后只小幅变化。最终 best_step 为 974–999，这些记录说明部分轨迹仍在较晚阶段找到更好的映射。五条轨迹中的三条在第300步已不同于原独立300步终态，正好体现两种日程不能互相替代。", "",
        "## 等待为何降低，以及剩余等待在哪里", "",
        "下表每行是一个实际保存状态。关键链时间列互斥相加得到 T̂；最大排他等待仅是等待归因中的首项，另表保留其余资源。", "",
        "| 状态 | 后端 / T̂ µs | Tensor µs | D2D µs | NoC µs | Vector+DDR+gap µs | 最大排他等待类别 / µs | 最忙资源服务 / T̂ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in states:
        label=r["state"]+(f" r{r['repair_seed']}" if r["repair_seed"] is not None else "")
        other=sum(r[f"chain_{k}_ns"] for k in ("Vector","DDR","release_gap_other"))
        lines.append(f"| {label} | {r['backend_mean_ns']/1000:.3f} / {r['T_hat_ns']/1000:.3f} | {r['chain_TensorCore_ns']/1000:.3f} | {r['chain_D2D_ns']/1000:.3f} | {r['chain_NoC_ns']/1000:.3f} | {other/1000:.3f} | {r['lead_wait_category']} / {r['lead_wait_ns']/1000:.3f} | {r['peak_load_over_T_hat']:.2%} |")
    special=next(r for r in states if r["state"]=="s1000" and r["repair_seed"]==3)
    lines += ["", "PF4 后的 R0 主要等待集中在剩余 D2D ((0,7),(1,4))：排他等待 457.579 µs。L1 把它降到 377.080 µs；原五条 300 步终态仍以 D2D 为最大等待来源，首项为 310.584–358.491 µs。",
        "1000 步的 seed 1、2 将主 D2D 等待进一步降至 286.622 和 296.032 µs，同时最忙资源总服务也下降。seed 123 的 D2D 等待分散到两个方向，首项下降后仍须看两项合计，不能只按首项大小推断总收益。",
        f"seed 3 出现不同的限制因素：其关键链 NoC 占 {special['chain_NoC_ns']/special['T_hat_ns']:.2%}，排他等待主要来自 Die 2 内部 ((2,7),(2,6)) 的 178.651 µs 与 ((2,3),(2,7)) 的 92.212 µs，D2D 排他等待仅 0.495 µs。此时 D2D 仍有接近 T̂ 的资源服务负载，但完成时刻的关键阻塞已转移到片内链路。seed 4 则仍由 ((1,4),(0,7)) 的 344.262 µs D2D 等待主导，对应它的预算扩大未改善终态。",
        "这些现象支持‘更多重映射预算减轻了部分链路排队，但轨迹间仍有不同的热点’。当前诊断解释的是已找到映射的时间线，没有建立对所有可行映射都成立的容量下界。",
        "", "![关键链时间与排他等待](figures/a_chain_wait.png)", "",
        "图中300步r3与1000步r2分别是各预算的后端中位状态；另列1000步r3，展示关键等待转向NoC的实例。两张子图分别展示时间构成与等待归因，不互相相加。", "",
        "## 本次实际耗时与资源", "",
        "以下是质量实验期间的实测进程时间，同机存在其他质量任务；正式空闲单进程计时由 J 单列。校准、搜索、恢复、最终 CPCM 与十次后端验证分别记账，未重新收取已有 R0/L1。", "",
        "| 种子 | 校准 s | 搜索 s | 恢复+最终CPCM s | 十次后端含准备 s | 其他初始化/导出/检查点落盘 s | 完整进程 min | 峰值 GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    new=[r for r in rows if r["budget"]==1000]
    for r in new:
        lines.append(f"| {r['repair_seed']} | {r['calibration_seconds']:.3f} | {r['search_seconds']:.3f} | {r['restore_best_seconds']+r['final_cpcm_seconds']:.3f} | {r['backend_validation_seconds']:.3f} | {r['other_job_seconds']:.3f} | {r['full_job_seconds']/60:.3f} | {r['peak_rss_gib']:.3f} |")
    cost=sum(r["new_decision_seconds"] for r in new)
    oldcost=sum(r["new_decision_seconds"] for r in rows if r["budget"]==300)
    lines += ["", f"本次 5000 个正式提议加 75 个校准提议；五条新轨迹决策累计 {cost:.3f} s，十种子后端验证累计 {sum(r['backend_validation_seconds'] for r in new):.3f} s，其中真实后端函数 {sum(r['backend_actual_calls_seconds'] for r in new):.3f} s，其余为复制与种子准备。完整进程累计 {sum(r['full_job_seconds'] for r in new)/3600:.3f} 进程小时，最长单进程 {max(r['full_job_seconds'] for r in new)/60:.3f} min。",
        f"相对于原五条独立 300 步，新五条 1000 步决策总成本为 {cost/oldcost:.3f} 倍；后端中位改善 11.257%，中位迁移字节增加。这个成本—恢复幅度的取舍作为 A 的预算诊断单列，正式停止规则与原 300 步结果保持原样。", "",
        "可复算入口：`python -B e2e/hi_closeout_a_report.py`。全部20个独立预算终态、200个后端样本、30个检查点、13状态诊断和5个成功回执均在CSV/审计记录中；其中新增只占5条1000步轨迹和50个最终后端样本。本次报告新增13次确定性CPCM解释，后端与搜索新增0。", ""]
    return lines


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
    axes[1].set(title=r"1000 步内当前最优 E 状态对应的 $\hat{T}$",xlabel="检查点步数",ylabel=r"确定性 CPCM $\hat{T}$（µs）",xticks=run.CHECKPOINTS)
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(title="修复种子",fontsize=8,ncol=2)
    fig.tight_layout();folder=output/"figures";folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/"a_budget_convergence.png",dpi=180);fig.savefig(folder/"a_budget_convergence.svg");plt.close(fig)


def plot_wait(output, states):
    if len(states)!=13:return
    import matplotlib.pyplot as plt
    labels=[("healthy",None,"健康"),("R0",None,"R0"),("L1",None,"L1"),
            ("s300",3,"300步 r3（中位）"),("s1000",2,"1000步 r2（中位）"),("s1000",3,"1000步 r3（NoC）")]
    chosen=[next(r for r in states if (r["state"],r["repair_seed"])==(state,seed)) for state,seed,_ in labels]
    colors={"TensorCore":"#8FBC8F","Vector":"#B7A9D8","D2D":"#E69F00","NoC":"#56B4E9","DDR":"#8C6D31","release_gap_other":"#B0B0B0"}
    names={"TensorCore":"TensorCore","Vector":"Vector","D2D":"D2D","NoC":"NoC","DDR":"DDR接口","release_gap_other":"释放/间隔"}
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,sharex=True)
    for ax,prefix,title in zip(axes,("chain","wait"),("关键链时间构成（严格相加为预测延迟）","排他等待归因（单独解释等待来源）")):
        left=[0.]*len(chosen)
        for category in CATEGORIES:
            vals=[r[f"{prefix}_{category}_ns"]/1000 for r in chosen]
            if max(vals)==0:continue
            ax.barh(range(len(chosen)),vals,left=left,color=colors[category],label=names[category],height=.65)
            left=[a+b for a,b in zip(left,vals)]
        ax.set(title=title,xlabel="时间（µs）",yticks=list(range(len(chosen))),yticklabels=[s[2] for s in labels])
        ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True)
    upper=max(sum(r[f"{prefix}_{category}_ns"] for category in CATEGORIES)/1000 for r in chosen for prefix in ("chain","wait"))*1.07
    axes[0].set_xlim(0,upper)
    axes[0].invert_yaxis();fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=5,frameon=False)
    fig.tight_layout(rect=(0,.08,1,1))
    for suffix in ("png","svg"):fig.savefig(output/"figures"/("a_chain_wait."+suffix),dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--output",type=Path,default=BASE)
    parser.add_argument("--progress",action="store_true")
    args=parser.parse_args()
    import post_fg_split
    post_fg_split.setup()
    inputs,results,states=paths(args.out,args.progress)
    receipts=completion_evidence(results,inputs,args.out)
    budget,samples,checkpoints=budget_tables(inputs,results,states[0]["backend_mean_ns"],states[1]["backend_mean_ns"])
    fault=run.frozen.read_json(inputs["paths"]["faults"])["faults"]["PF4"]
    composition,waits,loads,segments,diagnostics=[],[],[],[],[]
    for state in states:
        c,w,l,s,d=explain(state,args.out,fault)
        composition+=c;waits+=w;loads+=l;segments+=s;diagnostics.append(d)
    paired,checkpoint_change,state_rows=comparison_tables(budget,checkpoints,composition,waits,loads,diagnostics)
    tables=dict(a_independent_budgets=budget,a_backend_samples=samples,a_checkpoints=checkpoints,
        a_chain_composition=composition,a_exclusive_wait=waits,a_resource_load=loads,a_chain_segments=segments,a_diagnostic_timing=diagnostics,
        a_pair_300_1000=paired,a_checkpoint_changes=checkpoint_change,a_state_diagnosis=state_rows)
    for name,rows in tables.items():
        if rows: csv_write(args.output/"tables"/(name+".csv"),rows)
    complete=sum(r["budget"]==1000 for r in budget)==5
    if not args.progress: assert complete and len(checkpoints)==30 and len(diagnostics)==13
    plot(args.output,budget,checkpoints)
    plot_wait(args.output,state_rows)
    render(args.output,budget,checkpoints,diagnostics,complete,paired,checkpoint_change,state_rows)
    assert all(run.sha(run.frozen.path(r["source"]))==r["source_sha256"] and
               run.sha(run.frozen.path(r["raw"]))==r["raw_sha256"] for r in budget)
    audit=dict(status="complete" if complete else "partial",independent_budget_rows=len(budget),
        new_1000_step_searches=sum(r["budget"]==1000 for r in budget),
        new_1000_backend_samples=sum(r["budget"]==1000 for r in samples), checkpoint_rows=len(checkpoints),
        diagnostics=len(diagnostics), diagnostic_backend_calls=0,
        new_search_proposals=sum(r["search_proposals"] for r in budget if r["budget"]==1000),
        new_calibration_proposals=sum(r["calibration_proposals"] for r in budget if r["budget"]==1000),
        successful_receipts=receipts, original_300_results_retained=True,
        all20_result_and_raw_hashes_unchanged_by_report=True,
        report_new_searches=0, report_fixed_CPCM_diagnostics=len(diagnostics),
        source_results={r["source"]:r["source_sha256"] for r in budget},
        source_raw={r["raw"]:r["raw_sha256"] for r in budget},
        all_saved_T_hat_equal=True, all_chain_segment_sums_equal_T_hat=True,
        exclusive_wait_source="Analysis.blocking(raw=False)", source_input_sha256=inputs["sha256"])
    json_write(args.output/"evidence/a_report_audit.json",audit)
    print(json.dumps({k:audit[k] for k in ("status","new_1000_step_searches","new_1000_backend_samples",
        "checkpoint_rows","diagnostics","new_search_proposals","new_calibration_proposals",
        "all20_result_and_raw_hashes_unchanged_by_report")},indent=2))


if __name__=="__main__":main()
