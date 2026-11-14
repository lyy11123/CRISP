"""Read the two final Qwen PF4 points and assemble the complete main platform.

The original ten PF4 points and 300-step outcomes remain the main comparison.
The separate 1000-step diagnostic never enters this report's EIR columns.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

import f_config
import hi_config
import hi_closeout_b as run
import hi_h4_report as shared
import hi_i4_report as old_i4
from post_fg_diag import table

ROOT = run.ROOT
DEST = run.BASE
HERE = ROOT / "e2e/hi"
KINDS = dict(fault_selection=2, fault_select=2, fault_pre=2, fault_l1=2, fault_stop=8, fault_rep=30)


def check_inputs():
    manifest = shared.read(run.MANIFEST)
    assert len(manifest["jobs"]) == 46 and Counter(j["kind"] for j in manifest["jobs"]) == KINDS
    for j in manifest["jobs"]:
        s = j["spec"]
        assert s["model"] == "qwen3moe" and s.get("phase_op") in (None, "pf_ffn")
        assert s.get("fault_id") in (None, "PF4") and s.get("start_arm") in (None, "A3h")
    records = shared.read(run.INPUTS)["candidates"]
    needed = {n for j in manifest["jobs"] if j["kind"] == "fault_selection" for n in j["spec"]["candidate_ids"]}
    assert len(records) == 56 and set(records) == needed
    index = {}
    for name, source in records.items():
        for key in ("record", "raw", "snapshot"):
            assert shared.sha(shared.path(source[key])) == source["sha256"][key]
        r = shared.read(shared.path(source["record"]))
        assert r["status"] == "ok" and r["candidate_id"] == name and r["model"] == "qwen3moe"
        assert r["method"] == "A3h" and [v["seed"] for v in r["backend"]] == list(range(1, 11))
        index[name] = dict(source, healthy_T_hat=r["T_hat"], backend_mean_ns=statistics.mean(v["cycles"] for v in r["backend"]))
    return manifest, index


def read_units(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["EIR_complete"] = row["EIR_complete"] == "True"
        for key in list(row):
            if row[key] == "":
                row[key] = None
            elif key.endswith("_ns") or "relative_to" in key:
                row[key] = float(row[key])
            elif key.endswith("_migration"):
                row[key] = json.loads(row[key])
        row["EIR_repair_seed"] = int(row["EIR_repair_seed"]) if row.get("EIR_repair_seed") is not None else None
    return rows


def old_points():
    paths = [(HERE / "tables/i4_units.csv", HERE / "evidence/i4_completion_audit.json", "i4_light"),
             (HERE / "tables/i4_watos_dffn_units.csv", HERE / "evidence/i4_watos_dffn_completion_audit.json", "i4_watos_dffn")]
    rows, sources = [], []
    for units_path, audit_path, batch in paths:
        audit = shared.read(audit_path)
        assert audit["status"] == "complete"
        units = read_units(units_path)
        assert all(r["model"] == "qwen3moe" and r["fault"] == "PF4" for r in units)
        assert all(r["EIR_complete"] for r in units if r["start_arm"] == "A3h")
        rows.extend(dict(r, batch=batch) for r in units)
        sources.append(dict(batch=batch, units=str(units_path.relative_to(ROOT)), units_sha256=shared.sha(units_path),
            audit=str(audit_path.relative_to(ROOT)), audit_sha256=shared.sha(audit_path),
            completed_jobs=audit["completed_jobs"], actual_L2=audit["actual_repair_trajectories"],
            reused_L2=audit["reused_repair_jobs"], backend_samples=audit["backend_samples"]))
    assert sum(r["start_arm"] == "A3h" for r in rows) == 10
    return rows, sources


def layer_tables(units):
    contract = hi_config.model_contract("qwen3moe")
    assert contract["prefill_chain"] == dict(norm=2, mha=4, proj=1, ffn=1)
    assert contract["decode_chain"] == dict(norm=1, mha=4, proj=1, ffn=1)
    assert contract["decode_ffn_includes_rmsnorm"] and not contract["prefill_ffn_includes_rmsnorm"]
    selected = {r["phase_op"]: r for r in units if r["hardware_profile"] == "busybarn" and r["start_arm"] == "A3h"}
    hardware = f_config.hardware_parameters("busybarn")
    layers = []
    for phase, prefix in (("prefill", "pf"), ("decode", "dc")):
        weights = dict(ln=2 if phase == "prefill" else 1, mha=4, proj=1, ffn=1)
        if not all(prefix + "_" + op in selected for op in weights):
            continue
        cells = {op: selected[prefix + "_" + op] for op in weights}
        amount = contract["batch"] * (2048 if phase == "prefill" else 1) * contract["hidden"] * 2
        handoff = hardware["d2d_latency_ticks"] + amount / hardware["d2d_bytes_per_tick_directed_link"]
        assert handoff == (524308 if phase == "prefill" else 276)
        healthy = sum(weights[o] * cells[o]["healthy_backend_mean_ns"] for o in weights) + handoff
        r0 = sum(weights[o] * cells[o]["R0_backend_mean_ns"] for o in weights) + handoff
        for stage in ("healthy", "R0", "L1", "EIR"):
            if stage == "EIR" and not all(r["EIR_complete"] for r in cells.values()):
                continue
            total = sum(weights[o] * cells[o][stage + "_backend_mean_ns"] for o in weights)
            source = "EIR_source_job" if stage == "EIR" else "healthy_source_job" if stage == "healthy" else "l1_source_job"
            layers.append(dict(model="qwen3moe", hardware_profile="busybarn", start_arm="A3h", phase=phase, state=stage,
                weights=weights, batch=16, sequence_or_KV_length=2048, Q_heads_per_representative=8, Q_head_groups=4,
                external_norm_count=weights["ln"], ffn_embedded_norm_count=int(phase == "decode"),
                operator_sum_ns=total, handoff_bytes=amount, handoff_ns=handoff, layer_ns=total+handoff,
                relative_to_healthy=(total+handoff)/healthy-1, relative_to_R0=(total+handoff)/r0-1,
                source_jobs={o: cells[o][source] for o in weights},
                measurement="actual operator backend means serially added, plus unchanged adjacent-group handoff; one-layer reference"))
    requests = []
    lookup = {(r["phase"], r["state"]): r for r in layers}
    for stage in ("healthy", "R0", "L1", "EIR"):
        if not all((phase, stage) in lookup for phase in ("prefill", "decode")):
            continue
        p, d = lookup["prefill", stage], lookup["decode", stage]
        healthy = lookup["prefill", "healthy"]["layer_ns"] + 2048 * lookup["decode", "healthy"]["layer_ns"]
        r0 = lookup["prefill", "R0"]["layer_ns"] + 2048 * lookup["decode", "R0"]["layer_ns"]
        value = p["layer_ns"] + 2048 * d["layer_ns"]
        requests.append(dict(model="qwen3moe", hardware_profile="busybarn", start_arm="A3h", state=stage,
            additional_decode_steps=2048, fixed_KV_length=2048, prefill_ns=p["layer_ns"], decode_ns=d["layer_ns"],
            request_ns=value, relative_to_healthy=value/healthy-1, relative_to_R0=value/r0-1,
            measurement="P + 2048 D at fixed KV2048, one-layer reference; not a 48-layer or growing-KV simulation"))
    return layers, requests


def coverage(merged, complete_configs):
    by_key = {(r["hardware_profile"], r["phase_op"]): r for r in merged if r["start_arm"] == "A3h"}
    assert len(by_key) == sum(r["start_arm"] == "A3h" for r in merged)
    expected = {(h, p + "_" + op) for h, ops in (("busybarn", ("ln", "mha", "proj", "ffn")), ("watos", ("mha", "ffn")))
                for p in ("pf", "dc") for op in ops}
    rows = []
    for hardware, op in sorted(expected):
        r = by_key.get((hardware, op))
        done = r is not None and r["EIR_complete"] and (r["batch"] != "closeout_B" or r["config_id"] in complete_configs)
        rows.append(dict(hardware_profile=hardware, phase_op=op, batch=r["batch"] if r else "closeout_B",
            complete=done, EIR_source_job=r.get("EIR_source_job") if r else None,
            EIR_backend_mean_ns=r.get("EIR_backend_mean_ns") if r else None))
    return rows


def render(audit, units, groups, costs, layers, requests, coverage_rows):
    text = ["# Qwen 两个 prefill FFN / PF4 点及完整层参考", "",
        f"本批已完成 {audit['completed_jobs']}/46 作业，真实 L2 {audit['actual_repair_trajectories']} 条，停档复用 {audit['reused_repair_jobs']} 条，新增后端 {audit['backend_samples']} 次。与旧十个 PF4 点合并，当前已核对 {sum(r['complete'] for r in coverage_rows)}/12 点。", "",
        "两种硬件各复用已有十四候选中的正式 A3h 赢家；固定故障仍为 `(die0,core11) ↔ (die1,core8)` 双向连接失效。健康重放、R0、L1 和每条实际 L2 的后端种子均为1–10。旧896健康候选和此前完成的故障记录保持不变。", "",
        "| 硬件 | 健康ms | R0 ms | L1 ms | EIR ms | EIR/健康−1 | EIR/R0−1 | 停止档/原因 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in units:
        if r["EIR_complete"]:
            last = f"{r['EIR_backend_mean_ns']/1e6:.6f} | {r['EIR_relative_to_healthy']:+.2%} | {r['EIR_relative_to_R0']:+.2%} | {r['EIR_level']}/{r['EIR_reason']}"
        else:
            last = "待完成 | — | — | 进行中"
        text.append(f"| {r['hardware_profile']} | {r['healthy_backend_mean_ns']/1e6:.6f} | {r['R0_backend_mean_ns']/1e6:.6f} | {r['L1_backend_mean_ns']/1e6:.6f} | {last} |")
    text += ["", "各档都是从同一 L1 状态开始的独立30/100/300步搜索，修复种子为123、1、2、3、4。跨级回到健康估值2%以内停止；100/300档还按相邻两档平均CPCM改善不足2%停止，300档为预算终点。停止使用五轨迹平均估值，最终导出使用该档五个后端均值的中位代表。另行开展的1000步诊断不会替换这里的原300步结果。", "",
        "| 硬件 | 独立步数 | 实际轨迹 | 停档复用 | 待处理 | 五轨迹平均CPCM ns | 五个后端均值中位数ns | 五个后端均值范围ns |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in groups:
        spread = f"[{r['trajectory_backend_mean_min_ns']}, {r['trajectory_backend_mean_max_ns']}]" if "trajectory_backend_mean_min_ns" in r else "—"
        text.append(f"| {r['hardware_profile']} | {r['budget']} | {r['actual_trajectories']} | {r['reused']} | {r['pending']} | {r.get('T_hat_mean_ns','—')} | {r.get('trajectory_backend_mean_median_ns','—')} | {spread} |")
    text += ["", "主硬件层参考", "",
        "Prefill = 2×RMSNorm + 4×代表MHA + projection + FFN + 524308ns handoff。Decode = RMSNorm + 4×代表MHA + projection + FFN + 276ns handoff；decode FFN已经包含第二次RMSNorm。每个MHA代表图含8个Q头和一组K/V，Qwen层乘4。", "",
        "| 阶段 | 状态 | 算子和ms | handoff ns | 层参考ms | 相对健康 | 相对R0 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in layers:
        text.append(f"| {r['phase']} | {r['state']} | {r['operator_sum_ns']/1e6:.6f} | {r['handoff_ns']:.0f} | {r['layer_ns']/1e6:.6f} | {r['relative_to_healthy']:+.2%} | {r['relative_to_R0']:+.2%} |")
    text += ["", "| 状态 | P ms | D ms | P+2048D 秒 | 相对健康 | 相对R0 |", "|---|---:|---:|---:|---:|---:|"]
    for r in requests:
        text.append(f"| {r['state']} | {r['prefill_ns']/1e6:.6f} | {r['decode_ns']/1e6:.6f} | {r['request_ns']/1e9:.6f} | {r['relative_to_healthy']:+.2%} | {r['relative_to_R0']:+.2%} |")
    text += ["", "这是同一模型、同一PF4下算子均值的串行组装。KV固定2048；没有逐token增长，也不是48层或流水线联合仿真。组内故障不改变解析handoff项。WATOS只保留prefill/decode MHA与FFN四类独立算子，不补造缺失的LN/projection或整层指标。", "",
        "迁移与成本", "",
        "每状态的行为搬迁、DDR读取源变化、唯一迁移块字节与块数保存在逐状态表；最终代表和全部修复种子都保留，后端范围不省略。", "",
        "| 类型 | 作业数 | 后端调用秒 | 后端复制/准备秒 | 校准秒 | 搜索秒 | 恢复与最终CPCM秒 | 其他秒 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in ("fault_pre", "fault_l1", "fault_rep"):
        rows = [r for r in costs if r["kind"] == kind]
        if not rows:
            continue
        total = lambda key: sum(r[key] for r in rows)
        text.append(f"| {kind} | {len(rows)} | {total('backend_call_seconds'):.3f} | {total('backend_total_seconds')-total('backend_call_seconds'):.3f} | {total('calibration_seconds'):.3f} | {total('search_seconds')+total('L1_search_seconds'):.3f} | {total('restore_best_seconds')+total('final_cpcm_seconds'):.3f} | {total('other_seconds')+total('R0_prepare_seconds'):.3f} |")
    text += ["", f"已完成作业累计 {audit['receipt_process_hours']:.5f} 进程小时（含子进程启动）；任务体 {audit['task_process_hours']:.5f} 进程小时。L2记录中继承的R0/L1耗时单列，不再计入新的修复决策。实际发生的各档校准、搜索、导出与后端都计入。并行进程小时不作为墙钟或论文单进程计时。", "",
        "复算：`python e2e/hi_closeout_b_report.py`；运行中加 `--progress`。本批逐作业、逐状态、逐后端种子、独立预算、停止原因与迁移明细位于 `tables/b_*.csv`，12点合并表与主硬件层/请求表也保留原来源作业。", ""]
    return "\n".join(text)


def report(out, progress):
    manifest, index = check_inputs()
    # The shared pool also contains A/C/S jobs, so membership is checked here,
    # not with the older report's whole-directory equality check.
    jobs, present, missing, states, samples, costs, checks = shared.collect(Path(out), manifest, True, False, "qwen3moe", KINDS)
    stops, groups, trajectories = shared.stop_tables(jobs, present)
    units = shared.unit_tables(jobs, present, states, stops, index)
    job_rows = shared.job_rows(jobs, present, costs)
    complete = not missing
    if not progress:
        assert complete and len(units) == 2 and all(r["EIR_complete"] for r in units)
    actual = sum(r["status"] == "ok" for r in trajectories)
    reused = sum(r["status"] == "reused" for r in trajectories)
    assert len(samples) == sum(r.get("actual_backend_calls", 0) for r in job_rows)
    if complete:
        assert actual + reused == 30 and len(stops) == 8 and len(samples) == 60 + 10 * actual
    historical, source_records = old_points()
    merged = historical + [dict(r, batch="closeout_B") for r in units]
    layers, requests = layer_tables(merged)
    old_decode = old_i4.decode_table(historical)
    for row in layers:
        if row["phase"] == "decode":
            old = next(r for r in old_decode if r["start_arm"] == "A3h" and r["state"] == row["state"])
            assert row["layer_ns"] == old["decode_ns"]
    point_jobs = {config: [name for name, job in jobs.items() if job["spec"]["config_id"] == config]
                  for config in {job["spec"]["config_id"] for job in jobs.values()}}
    complete_configs = {config for config, names in point_jobs.items() if all(name in present for name in names)}
    coverage_rows = coverage(merged, complete_configs)
    audit = dict(status="complete" if complete else "in_progress", completed_jobs=len(present), expected_jobs=46,
        missing_jobs=len(missing), missing=missing, actual_repair_trajectories=actual, reused_repair_jobs=reused,
        backend_samples=len(samples), stop_records=len(stops), healthy_candidates=56, new_healthy_searches=0,
        receipt_process_hours=sum(r["receipt"]["wall_seconds"] for r in present.values())/3600,
        task_process_hours=sum(r["result"]["wall_seconds"] for r in present.values())/3600,
        maximum_receipt_seconds=max((r["receipt"]["wall_seconds"] for r in present.values()), default=0),
        peak_rss_gb=max((r["receipt"]["peak_rss_gb"] for r in present.values()), default=0),
        I4_PF4_complete=sum(r["complete"] for r in coverage_rows), I4_PF4_expected=12,
        point_job_completion=[dict(config_id=config, completed_jobs=sum(name in present for name in names),
            expected_jobs=len(names), all_jobs_complete=config in complete_configs) for config, names in sorted(point_jobs.items())],
        historical_sources=source_records, input_checks=checks, manifest_sha256=shared.sha(run.MANIFEST),
        inputs_sha256=shared.sha(run.INPUTS), source_sha256=shared.sha(__file__),
        active_EIR_budgets=[30,100,300], larger_diagnostic_not_used=True,
        WATOS_complete_layer_generated=False, newly_run_by_report=dict(searches=0,backend_calls=0))
    values = dict(jobs=job_rows, states=states, backend_samples=samples, timing=costs, stop_decisions=stops,
        repair_trajectories=trajectories, budget_groups=groups, units=units, all_I4_units=merged,
        pf4_coverage=coverage_rows, layers=layers, requests=requests,
        watos_operators=[r for r in merged if r["hardware_profile"] == "watos" and r["start_arm"] == "A3h"])
    for name, rows in values.items():
        table(DEST / "tables" / ("b_" + name + ".csv"), rows)
    shared.write(DEST / "evidence/b_completion.json", audit)
    (DEST / "B_I4_PF4_COMPLETE.md").write_text(render(audit, units, groups, costs, layers, requests, coverage_rows), encoding="utf-8")
    if complete:
        assert audit["I4_PF4_complete"] == 12 and len(layers) == 8 and len(requests) == 4
        table(HERE / "tables/i4_pf4_coverage.csv", coverage_rows)
        text = ["# I4 PF4 完成范围", "", "当前本机正式结果已覆盖12/12个预定A3h PF4点：I4-light九点、WATOS decode FFN一点、本轮两种硬件prefill FFN两点。", "",
            "| 硬件 | 算子 | 批次 | 最终EIR状态已齐 |", "|---|---|---|---|"]
        for r in coverage_rows:
            text.append(f"| {r['hardware_profile']} | {r['phase_op']} | {r['batch']} | 是 |")
        text += ["", f"三个批次共270个作业。实际L2 {actual+sum(r['actual_L2'] for r in source_records)} 条、停档复用 {reused+sum(r['reused_L2'] for r in source_records)} 条；后端共 {len(samples)+sum(r['backend_samples'] for r in source_records)} 次。四个A1-R0对照属于原I4-light，未计入12个A3h点。", "",
            "详细新结果、原十点合并、主硬件P/D与固定KV的P+2048D参考见 [B_I4_PF4_COMPLETE.md](closeout/B_I4_PF4_COMPLETE.md)。WATOS仅有MHA/FFN四类结果，不生成完整层。MF6/Split是C的独立工作包，不混入PF4覆盖。", ""]
        (HERE / "I4_PF4_COVERAGE.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ("status","completed_jobs","actual_repair_trajectories","reused_repair_jobs","backend_samples","I4_PF4_complete")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=run.OUT)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    report(args.out, args.progress)
