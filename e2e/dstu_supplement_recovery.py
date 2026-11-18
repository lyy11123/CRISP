"""Read saved J3/PF4 states for deployment views; no search or backend calls."""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics

import hi_closeout_p0 as p0

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "e2e/dstu_supplement"
OLD = ROOT / "e2e/hi/closeout"
JOUT = ROOT / "e2e/hi/out/closeout_j_20261115"
SOURCES = {}


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def data(path):
    path = ROOT / path
    value = path.read_bytes()
    SOURCES[relative(path)] = hashlib.sha256(value).hexdigest()
    return value


def read(path):
    return json.loads(data(path).decode("utf-8-sig"))


def rows(path):
    return list(csv.DictReader(data(path).decode("utf-8-sig").splitlines()))


def saved(path):
    return pickle.loads(gzip.decompress(data(path)))


def write_rows(name, values):
    p0.csv_write(BASE / name, values)


def selectors():
    values, _, _ = p0.eir_tables()
    SOURCES.update(p0.SOURCES)
    states = rows(OLD / "tables/b_states.csv")
    units = rows(OLD / "tables/b_units.csv")
    stops = rows(OLD / "tables/b_stop_decisions.csv")
    for unit in units:
        budget = int(unit["EIR_level"][1:])
        stop = next(r for r in stops if r["config_id"] == unit["config_id"]
                    and r["decision"] == "evaluated" and r["stop"] == "True")
        candidates = [r for r in states if r["config_id"] == unit["config_id"]
                      and r["state"] == "L2" and int(r["budget"]) == budget]
        assert {int(r["repair_seed"]) for r in candidates} == set(p0.SEEDS)
        fixed = next(r for r in candidates if int(r["repair_seed"]) == 123)
        best = min(candidates, key=lambda r: (float(r["T_hat_ns"]), p0.SEEDS.index(int(r["repair_seed"]))))
        median = statistics.median(float(r["backend_mean_ns"]) for r in candidates)
        main = min((r for r in candidates if float(r["backend_mean_ns"]) == median),
                   key=lambda r: int(r["repair_seed"]))
        assert main["job_id"] == unit["EIR_source_job"] == stop["source_job"]
        for r in candidates:
            result = read(ROOT / "e2e/hi/out/closeout_20261115/results" / (r["job_id"] + ".json"))
            assert result["T"] == float(r["T_hat_ns"])
            assert result["backend_mean"] == float(r["backend_mean_ns"])
            assert result["migration"]["unique_block_bytes"] == int(r["migration_unique_block_bytes"])
        for rule, selected in (("fixed_seed123", fixed), ("minimum_T_hat", best), ("backend_median", main)):
            backend = float(selected["backend_mean_ns"])
            values.append(dict(model=unit["model"], hardware_profile=unit["hardware_profile"],
                config_id=unit["config_id"], phase_op=unit["phase_op"], fault="PF4", start_arm="A3h",
                stop_level=unit["EIR_level"], stop_reason=unit["EIR_reason"], candidates=5,
                rule=rule, selected_job=selected["job_id"], repair_seed=selected["repair_seed"],
                T_hat_ns=float(selected["T_hat_ns"]), backend_mean_ns=backend,
                healthy_backend_mean_ns=float(unit["healthy_backend_mean_ns"]),
                R0_backend_mean_ns=float(unit["R0_backend_mean_ns"]),
                relative_to_healthy=backend/float(unit["healthy_backend_mean_ns"])-1,
                relative_to_R0=backend/float(unit["R0_backend_mean_ns"])-1,
                relative_to_main_representative=backend/float(unit["EIR_backend_mean_ns"])-1,
                same_as_main_representative=selected["job_id"] == unit["EIR_source_job"],
                **{key: float(selected[key]) if key.endswith("share") else int(selected[key])
                   for key in ("migration_unique_block_bytes", "migration_moved_behas", "migration_moved_behas_share",
                               "migration_ddr_read_moved_bytes", "migration_ddr_read_moved_share")},
                source="e2e/hi/out/closeout_20261115/results/"+selected["job_id"]+".json"))
    assert len(values) == 72
    assert len({(r["config_id"], r["phase_op"]) for r in values}) == 24
    old = p0.eir_tables()[0]
    assert values[:66] == old
    write_rows("tables/eir_selector_24points.csv", values)
    return values


def migration(previous, current, sizes, hp, router, eta):
    """Serialize new DDR copies on shortest service-time paths of the PF4 topology."""
    transfers = []
    for block, targets in current["placement"].items():
        existing = set(previous["placement"][block])
        for target in sorted(set(targets)-existing):
            count = int(sizes[block])
            choices = []
            for donor in sorted(existing):
                route = router.path(donor, target, count)
                cost = sum(math.ceil(math.ceil(count/(hp.links_dict[edge].bandwidth*eta)) *
                                     hp.links_dict[edge].timeunit) + hp.links_dict[edge].latency for edge in route)
                choices.append((cost, donor, route))
            cost, donor, route = min(choices)
            transfers.append(dict(block=str(block), bytes=count, source=donor, target=target,
                                  route=route, serialized_hop_service_ns=cost,
                                  links=[dict(edge=edge, kind=hp.links_dict[edge].link_type,
                                    bandwidth_bytes_per_timeunit=hp.links_dict[edge].bandwidth,
                                    timeunit_ns=hp.links_dict[edge].timeunit,
                                    latency_ns=hp.links_dict[edge].latency) for edge in route]))
    return dict(bytes=sum(r["bytes"] for r in transfers), copies=len(transfers),
                byte_hops=sum(r["bytes"]*len(r["route"]) for r in transfers),
                delta_seconds=sum(r["serialized_hop_service_ns"] for r in transfers)/1e9,
                transfers=transfers)


def control(result):
    t = result["J3_timing"]
    return t["new_control_seconds"] + t["final_CPCM_seconds"] + t["state_export_seconds"]


def time_parts(measured):
    ts = [r["J3_timing"] for r in measured]
    allcontrol = sum(control(r) for r in measured)
    r0 = ts[0]["R0_prepare_seconds"]
    validation = sum(t["backend_validation_seconds"] for t in ts)
    warmup = sum(t["numba_warmup_or_cache_seconds"] for t in ts)
    wall = sum(r["wall_seconds"] for r in measured)
    # Loading, platform construction and serialization were not independently timed.
    residual = wall-allcontrol-validation-warmup
    assert residual >= -1e-7
    return dict(algorithm_control_with_finalize_export_seconds=allcontrol,
                decision_after_R0_ready_seconds=allcontrol-r0,
                R0_prepare_seconds=r0, backend_validation_seconds=validation,
                backend_actual_calls_seconds=sum(t["backend_actual_calls_seconds"] for t in ts),
                numba_first_call_seconds=warmup,
                loading_platform_and_other_seconds=residual,
                measured_job_wall_seconds=wall,
                backend_calls=sum(t["backend_calls"] for t in ts))


def fixed_counts(horizon, t0, t1, decision, pause):
    return min(horizon, decision)/t0 + max(horizon-decision-pause, 0)/t1


def single_trajectory():
    import post_fg_split as split
    split.setup()
    inputs = read(OLD / "manifests/j3_inputs.json")["cases"]
    cumulative = rows(OLD / "tables/j3_cumulative.csv")
    online, stages, payback, migration_log, progressive_log = [], [], [], [], []
    for case in ("gpt64", "gpt144", "qwen"):
        ci = inputs[case]
        pre = saved(ci["pre_sink"])
        hp = split.platform(ci["base"])
        failed = {(tuple(a), tuple(b)) for a, b in ci["physical_fault"]["failed_links"]}
        assert all(edge in hp.links_dict for edge in failed)
        for edge in failed:
            del hp.links_dict[edge]
        router = split.Router(hp)
        for seed in (123, 1, 2):
            records = {level: read(JOUT / "results" / f"J3__{case}_r{seed}__{level}.json")
                       for level in ("l1", "s30", "s100", "s300")}
            base = records["l1"]
            rawbase = saved(base["sink"])
            assert base["T_pre"] == pre["T"]
            level = "R0" if base["T_R0"] <= 1.02*base["T_pre"] else "L1"
            stage_states = [(level, base["T_"+level], base[level+"_backend_mean"], rawbase[level], base)]
            stopping = base["T_"+level] <= 1.02*base["T_pre"]
            reason = "within_epsilon" if stopping else "continue"
            stages.append(dict(input_id=case, repair_seed=seed, level=level,
                T_hat_ns=base["T_"+level], healthy_T_hat_ns=base["T_pre"],
                gain_from_previous_budget=None, stop=stopping, reason=reason,
                source=relative(JOUT / "results" / (base["job_id"]+".json"))))
            measured = [base]
            previous = None
            for budget in (30, 100, 300):
                if stopping:
                    break
                level = f"s{budget}"
                r = records[level]
                assert r["status"] == "ok" and r["steps"] == budget and r["seed"] == seed
                raw = saved(r["sink"])
                assert list(raw["search"]["rows"])[-1]["step"] == budget
                assert set(raw["record"]["backend"]) == set(range(1, 11))
                gain = None if previous is None else (previous-r["T"])/previous
                reason = ("within_epsilon" if r["T"] <= 1.02*base["T_pre"] else
                          "tier_gain" if gain is not None and gain < .02 else
                          "budget" if budget == 300 else "continue")
                stopping = reason != "continue"
                stages.append(dict(input_id=case, repair_seed=seed, level=level, T_hat_ns=r["T"],
                    healthy_T_hat_ns=base["T_pre"], gain_from_previous_budget=gain,
                    stop=stopping, reason=reason, source=relative(JOUT / "results" / (r["job_id"]+".json"))))
                stage_states.append((level, r["T"], r["backend_mean"], raw["record"], r))
                measured.append(r)
                previous = r["T"]
            last = stage_states[-1]
            costs = time_parts(measured)
            budget = int(last[0][1:]) if last[0].startswith("s") else 0
            if budget:
                old = next(x for x in cumulative if x["input_id"] == case and int(x["repair_seed"]) == seed
                           and int(x["through_budget"]) == budget)
                assert math.isclose(costs["algorithm_control_with_finalize_export_seconds"],
                                    float(old["measured_control_with_finalize_export_seconds"]), abs_tol=1e-6)
            nsteps = sum(r["steps"] for r in measured[1:])
            row = dict(input_id=case, model=base["spec"]["model"], phase_op=base["spec"]["phase_op"],
                shape=base["spec"]["shape"], fault="PF4", repair_seed=seed,
                selected_level=last[0], stop_reason=reason, evaluated_budgets=[r["steps"] for r in measured[1:]],
                actual_search_proposals=nsteps,
                actual_calibration_proposals=sum(r["J3_timing"]["calibration_proposals"] for r in measured[1:]),
                healthy_T_hat_ns=base["T_pre"], selected_T_hat_ns=last[1],
                healthy_backend_mean_ns=statistics.mean(v[0] for v in pre["record"]["backend"].values()),
                R0_backend_mean_ns=base["R0_backend_mean"], selected_backend_mean_ns=last[2],
                selected_backend_relative_to_R0=last[2]/base["R0_backend_mean"]-1,
                selected_backend_relative_to_healthy=last[2]/statistics.mean(v[0] for v in pre["record"]["backend"].values())-1,
                migration_from_healthy_unique_bytes=last[3]["migration_vs_start"]["unique_block_bytes"],
                **costs, source_jobs=[r["job_id"] for r in measured],
                stop_input="own deterministic CPCM only", timing_source="J3 otherwise-idle serial processes",
                budget_semantics="independent L1 starts; 30+100+300 is 430 proposals")
            online.append(row)
            t0, t1 = base["R0_backend_mean"]/1e9, last[2]/1e9
            for eta in (1., .5):
                direct = migration(rawbase["R0"], last[3], pre["sizes"], hp, router, eta)
                assert direct["bytes"] == last[3]["migration_vs_R0"]["unique_block_bytes"]
                migration_log.append(dict(input_id=case, repair_seed=seed, mode="final_only",
                    from_level="R0", to_level=last[0], effective_bandwidth_fraction=eta, **direct))
                decision, pause = costs["decision_after_R0_ready_seconds"], direct["delta_seconds"]
                improves = t1 < t0
                hbg = decision+pause*t0/(t0-t1) if improves else None
                hstop = (decision+pause)*t0/(t0-t1) if improves else None
                if improves:
                    assert math.isclose(fixed_counts(hbg, t0, t1, decision, pause), hbg/t0, rel_tol=1e-10)
                    assert math.isclose(max(hstop-decision-pause, 0)/t1, hstop/t0, rel_tol=1e-10)
                payback.append(dict(input_id=case, repair_seed=seed, mode="final_only", selected_level=last[0],
                    effective_bandwidth_fraction=eta, T0_seconds=t0, T1_seconds=t1,
                    decision_seconds=decision, deployment_pause_seconds=pause,
                    actual_new_weight_copy_bytes=direct["bytes"], migration_byte_hops=direct["byte_hops"],
                    background_break_even_seconds=hbg, stop_search_break_even_seconds=hstop,
                    service_improves=improves, estimate="fixed operator service; R0 already in service; serial routed copies",
                    **{k:costs[k] for k in ("R0_prepare_seconds", "backend_validation_seconds", "numba_first_call_seconds",
                                          "loading_platform_and_other_seconds")}))
                elapsed, work_done, pauses, migrated, byte_hops = 0., 0., 0., 0, 0
                current, current_level, current_T, current_backend = rawbase["R0"], "R0", base["T_R0"], t0
                for index, (candidate_level, predicted, backend, state, result) in enumerate(stage_states):
                    step_control = control(result) - (base["J3_timing"]["R0_prepare_seconds"] if index == 0 else 0)
                    work_done += step_control/current_backend
                    elapsed += step_control
                    deploy = predicted < current_T
                    transfer = migration(current, state, pre["sizes"], hp, router, eta) if deploy else dict(bytes=0,copies=0,byte_hops=0,delta_seconds=0.,transfers=[])
                    migration_log.append(dict(input_id=case, repair_seed=seed, mode="progressive",
                        from_level=current_level, to_level=candidate_level, effective_bandwidth_fraction=eta,
                        deployed=deploy, **transfer))
                    before = current_level
                    elapsed += transfer["delta_seconds"]
                    pauses += transfer["delta_seconds"]
                    migrated += transfer["bytes"]
                    byte_hops += transfer["byte_hops"]
                    if deploy:
                        current, current_level, current_T, current_backend = state, candidate_level, predicted, backend/1e9
                    progressive_log.append(dict(input_id=case, repair_seed=seed, effective_bandwidth_fraction=eta,
                        evaluated_level=candidate_level, previous_deployed=before, deployed=deploy,
                        active_level=current_level, incremental_control_seconds=step_control,
                        incremental_pause_seconds=transfer["delta_seconds"], adjacent_copy_bytes=transfer["bytes"],
                        elapsed_seconds=elapsed, processed_equivalent_operators=work_done,
                        baseline_equivalent_operators=elapsed/t0))
                gain = work_done-elapsed/t0
                improves = current_backend < t0
                sustainable = (elapsed + max(-gain,0)/(1/current_backend-1/t0)) if improves else None
                assert math.isclose(elapsed, decision+pauses, abs_tol=1e-6)
                payback.append(dict(input_id=case, repair_seed=seed, mode="progressive",
                    selected_level=current_level, effective_bandwidth_fraction=eta,
                    T0_seconds=t0, T1_seconds=current_backend, decision_seconds=decision,
                    deployment_pause_seconds=pauses, actual_new_weight_copy_bytes=migrated,
                    migration_byte_hops=byte_hops, service_improves=improves,
                    gain_at_last_deployment_operator_equivalents=gain,
                    sustained_break_even_after_last_deployment_seconds=sustainable,
                    estimate="piecewise service; deploy only own-CPCM improvement; adjacent-state copies"))
    write_rows("tables/online_single_trajectory.csv", online)
    write_rows("tables/recovery_payback.csv", payback)
    write_rows("evidence/recovery_stop_stages.csv", stages)
    write_rows("evidence/recovery_progressive_stages.csv", progressive_log)
    p0.json_write(BASE / "evidence/recovery_migration_paths.json", migration_log)
    return online, payback, stages


def report(selection, online, payback):
    diagnostics = [r for r in rows(OLD / "tables/a_independent_budgets.csv") if int(r["budget"]) == 1000]
    assert len(diagnostics) == 5
    for r in diagnostics:
        r["timing_source"] = "historical parallel quality batch; not J3 serial timing"
        r["included_in_J3_deployment_view"] = False
    write_rows("evidence/recovery_qwen1000_cost.csv", diagnostics)
    lines = ["# 单轨迹停档、状态选择与恢复回本", "",
        "本报告读取已有 JSON、CSV 与保存态完成后处理。P0 补齐为 24 个 PF4 点、72 条选法记录；J3 的三个场景、各三个修复种子形成九条单轨迹部署视图。新增搜索、CPCM 推演和真实后端调用均为零。", "",
        "## 单轨迹停档", "",
        "每条轨迹独立使用自身确定性 CPCM。先检查 R0、L1 是否回到健康值的 2% 以内；L1 未恢复则进入 30 步。30 步只检查恢复目标，100、300 步再检查相对上一预算的改善是否小于 2%，最后最多到 300。停止判断不使用后端成绩。", "",
        "30、100、300 都从同一 L1 状态独立开始；依次检查三档实际需要 430 个搜索提议和 45 个校准提议。这里复用已执行的 J3 保存态作部署分析，没有把独立预算改写为连续 300 步，也没有替换原研究中的五种子平均停档和后端中位代表。", "",
        "| 场景 | 修复种子 | 停止档 / 原因 | 实际搜索步数 | 相对 R0 后端变化 | R0 就绪后决策秒 | 论文后端评测秒 |",
        "|---|---:|---|---:|---:|---:|---:|"]
    for r in online:
        lines.append(f"| {r['input_id']} | {r['repair_seed']} | {r['selected_level']} / {r['stop_reason']} | {r['actual_search_proposals']} | {r['selected_backend_relative_to_R0']:+.3%} | {r['decision_after_R0_ready_seconds']:.3f} | {r['backend_validation_seconds']:.3f} |")
    lines += ["", "## 24 点状态选择", "",
        "原 22 点的 66 行逐字段保留，新增主硬件与 WATOS 的 Qwen prefill FFN/PF4 两点。每点仍比较固定 seed123、最小自身预测、后端中位代表三种选择；后端中位只作研究汇总参照。新主硬件点按五种子平均在 s300 停止，WATOS 在 s100 停止。", "",
        "| 选择方法 | 与原主代表相同 / 24 | 相对主代表后端中位变化 | 最小变化 | 最大变化 |",
        "|---|---:|---:|---:|---:|"]
    for rule in ("fixed_seed123", "minimum_T_hat", "backend_median"):
        subset = [r for r in selection if r["rule"] == rule]
        diff = [r["relative_to_main_representative"] for r in subset]
        lines.append(f"| {rule} | {sum(r['same_as_main_representative'] for r in subset)}/24 | {statistics.median(diff):+.3%} | {min(diff):+.3%} | {max(diff):+.3%} |")
    lines += ["", "## 固定服务时间下的恢复回本", "",
        "计时从 R0 已可服务开始。td 包含 L1 搜索、实际执行的独立 L2 预算及共用的最终 CPCM 检查与状态导出；R0 准备秒数另列。最终检查与导出包含 R0/L1 两态，统一计入 td。JIT 首次调用与加载/平台构造/其他残差另列；后者由作业总时间扣除已计时项目得到。论文十种子后端验证单列。原始完整算法控制秒数同时保留，便于从故障发生时另行组装启动时间。", "",
        "单次部署使用最后选择态与 R0 的实际权重驻留集合之差。按每个新增 DDR 权重副本的真实字节数，在删去 PF4 两条有向链路后的平台上按标称参数选择最短服务路径；每条复制逐跳串行、各块也串行，链路服务时间使用原平台的带宽、timeunit 与延迟。有效带宽分别取标称值的 100% 和 50%，在候选路径间选择服务时间最小的可用旧副本。这是两种解析部署假设。完整源、目的、路径和各跳参数保存在 recovery_migration_paths.json。", "",
        "后台搜索的处理量为 N(H)=min(H,td)/T0+max(H−td−δ,0)/T1。T1<T0 时，后台回本 H*=td+δT0/(T0−T1)；停服务搜索则 H*=(td+δ)T0/(T0−T1)。表中 T 为该代表算子的十种子后端平均纳秒转秒，未外推为模型请求吞吐。", "",
        "| 场景 | 种子 | 新增复制 MiB | 部署暂停 ms（标称带宽） | 后台回本 s | 停服搜索回本 s |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in payback:
        if r["mode"] == "final_only" and r["effective_bandwidth_fraction"] == 1:
            lines.append(f"| {r['input_id']} | {r['repair_seed']} | {r['actual_new_weight_copy_bytes']/2**20:.6f} | {r['deployment_pause_seconds']*1000:.6f} | {r['background_break_even_seconds']:.3f} | {r['stop_search_break_even_seconds']:.3f} |")
    lines += ["", "渐进部署仅在自身 CPCM 比当前部署态更小时切换，逐段累计真实控制耗时，暂停按相邻已部署状态的新增权重副本计算。该视图使用各段已有后端均值积分处理量，CSV 的 sustained_break_even_after_last_deployment_seconds 指最后一次部署之后持续回本的时刻。若某预算预测没有改善，搜索成本仍计入、迁移为零。", "",
        "Qwen 的五条 1000 步诊断结果保存在 evidence/recovery_qwen1000_cost.csv。它们来自此前并行质量批次，未加入 J3 串行计时或九条部署视图；原 300 步质量表保留。", "",
        "## 文件与复算", "",
        "- tables/online_single_trajectory.csv：九条最终选择、决策成本和质量。",
        "- tables/eir_selector_24points.csv：24 点×3 选法，含新增两点与原始结果路径。",
        "- tables/recovery_payback.csv：单次/渐进部署、100%/50% 有效带宽共 36 行。",
        "- evidence/recovery_stop_stages.csv 与 recovery_progressive_stages.csv：逐档判断和服务积分。",
        "- evidence/recovery_migration_paths.json：每个新增权重副本的来源、目标、字节和解析路径。",
        "- evidence/recovery_audit.json：原始输入摘要及复算断言结果。", "",
        "从仓库根目录运行：`e2e/post_fg/dist/venv/Scripts/python.exe -B e2e/dstu_supplement_recovery.py`。", ""]
    (BASE / "ONLINE_RECOVERY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    for path in (Path(__file__), ROOT/"e2e/hi_closeout_p0.py", ROOT/"e2e/post_fg_split.py",
                 ROOT/"e2e/f_config.py", ROOT/"e2e/hi_closeout_j_eir.py"):
        data(path)
    selection = selectors()
    online, payback, stages = single_trajectory()
    report(selection, online, payback)
    assert len(online) == 9 and len(payback) == 36
    outputs = ["ONLINE_RECOVERY.md", "tables/online_single_trajectory.csv", "tables/eir_selector_24points.csv",
               "tables/recovery_payback.csv", "evidence/recovery_stop_stages.csv",
               "evidence/recovery_progressive_stages.csv", "evidence/recovery_migration_paths.json",
               "evidence/recovery_qwen1000_cost.csv"]
    audit = dict(status="passed", selector_points=24, selector_rows=len(selection),
        unchanged_old_selector_rows=66, single_trajectories=len(online), payback_rows=len(payback),
        new_searches=0, new_CPCM_replays=0, new_backend_calls=0,
        stops=dict(Counter(r["selected_level"] for r in online)), source_sha256=SOURCES,
        output_sha256={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in outputs},
        checks=["old 66 selector rows unchanged", "new PF FFN selected state agrees with final five-seed stop",
                "own-predictor-only stopping", "J3 cumulative measured costs match",
                "background and stop-search break-even formulas numerically verified",
                "progressive clocks equal measured control plus adjacent migration pauses"],
        units=dict(service="seconds from ns / 1e9", decision="host wall seconds", migration="bytes and topology service ns"))
    p0.json_write(BASE / "evidence/recovery_audit.json", audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ("source_sha256","output_sha256","checks")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
