"""Read-only C MF6/Split report, including original rigid EIR search costs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import statistics

import fg_batch
import hi_closeout_c as run
import hi_h4_report as fault_report
from post_fg_diag import table

ROOT = run.ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backend(record):
    assert set(record["backend"]) == set(range(1, 11))
    assert not record["fault_violations"]
    return [record["backend"][s][0] for s in range(1, 11)]


def graph_metrics(export, spec, dead):
    data = run.split.static.read_pickle(ROOT / export)
    events = pickle.loads(data["events_blob"])
    hp = run.split.platform(spec, dead)
    metrics = run.split.event_metrics(events, hp)
    bounds = run.split.tensor_bounds(events, hp)
    return metrics, bounds


def offline_migration(state, info):
    """Match the EIR migration definition: new DDR copies, not PE input reads."""
    _, original_flows = run.split.input_flows(state)
    compared = 0
    for transfer in info["transfers"]:
        if not transfer["offline"]:
            continue
        candidates = [f for f in original_flows if f["offline"] and tuple(f["data"]) == tuple(transfer["data"])
                      and run.split.overlap(f["region"], transfer["region"]) == tuple(map(tuple, transfer["region"]))]
        assert candidates
        assert tuple(transfer["source"]) in {tuple(f["source"]) for f in candidates}
        compared += 1
    return dict(unique_ddr_migration_bytes=0, unique_ddr_migration_blocks=0,
                matched_offline_transfers=compared,
                definition="new DDR source copies of unique weight slices, as in EIR; local input traffic reported separately")


def collect(out, progress=False):
    run.split.setup()
    out = Path(out).resolve()
    local = run.private_out(out)
    manifest = run.frozen.read_json(run.MANIFEST)
    jobs = {j["id"]: j for j in manifest["jobs"]}
    present, missing, checks = {}, [], []
    for name, job in jobs.items():
        path = out / job["result"]
        receipt_path = out / "receipts" / (name + ".json")
        if not path.exists() or not receipt_path.exists():
            missing.append(name)
            continue
        result = run.frozen.read_json(path)
        receipt = run.frozen.read_json(receipt_path)
        assert receipt["identity"] == fg_batch.identity(job) and receipt["rc"] == 0
        assert result["job_id"] == name and result["kind"] == job["kind"]
        assert result["status"] in ("ok", "reused")
        if result["status"] == "ok":
            assert result["spec"] == job["spec"]
        else:
            assert job["kind"] == "fault_rep"
        present[name] = dict(spec=job["spec"], result=result, receipt=receipt)
        checks.append(dict(job_id=name, identity=receipt["identity"], result_sha256=sha(path),
                           receipt_sha256=sha(receipt_path), result_status=result["status"]))
    if not progress:
        assert not missing, missing
        for model in run.hi_config.MODELS:
            r = run.frozen.read_json(local / "representatives" / (model + ".json"))
            assert not r["applicable"] or "C_" + model + "__stop300" in jobs

    samples, all_states, costs, inventories = [], [], [], []
    for name, item in present.items():
        s, r = item["spec"], item["result"]
        cost = dict(job_id=name, model=s["model"], kind=s["kind"], status=r["status"],
                    wall_seconds=item["receipt"]["wall_seconds"], task_body_seconds=r["wall_seconds"],
                    peak_rss_gb=item["receipt"]["peak_rss_gb"],
                    actual_search_trajectories=0, actual_backend_calls=0,
                    backend_seconds=0., calibration_seconds=0., search_seconds=0.,
                    restore_seconds=0., inherited_R0_seconds=0., inherited_L1_seconds=0.)
        if r["status"] == "reused":
            costs.append(cost)
            continue
        if s["kind"] == "fault_select":
            assert r["anchor_backend_calls"] == 1
            assert r["faults"]["MF6"]["failed_nodes"] == [] and r["faults"]["MF6"]["failed_links"] == []
            assert len(r["faults"]["MF6"]["failed_pes"]) == 1
            cost["actual_backend_calls"] = 1
        elif s["kind"] in ("fault_pre", "fault_l1", "fault_rep"):
            raw = run.frozen.read_pickle(r["sink"])
            assert raw["backend_seeds"] == list(range(1, 11))
            if s["kind"] == "fault_pre":
                entries = [("healthy", raw["record"])]
                assert raw["T"] == r["T"]
                reused = "reused_healthy_record" in r
                cost["healthy_reused"] = reused
            elif s["kind"] == "fault_l1":
                entries = [("R0", raw["R0"]), ("L1", raw["L1"])]
                reused = False
            else:
                entries = [("L2", raw["record"])]
                reused = False
                assert s["steps"] in run.frozen.STEPS and s["seed"] in run.frozen.SEEDS
                assert raw["proposals"]["steps"] == s["steps"]
                assert [row["step"] for row in raw["search"]["rows"]] == list(range(1, s["steps"] + 1))
                assert raw["spec"]["method"] == "L2Gsync" and raw["spec"]["checks"] is False
                assert raw["T_L1_check"] == raw["search"]["T_start"]
                cost.update(actual_search_trajectories=1, calibration_seconds=raw["search"]["t_calibration"],
                    search_seconds=raw["search"]["t_L2_search"], restore_seconds=raw["search"]["t_restore_best"],
                    inherited_R0_seconds=raw["timing"]["t_R0"], inherited_L1_seconds=raw["timing"]["t_L1"],
                    calibration_proposals=raw["proposals"]["calibration"], search_proposals=raw["proposals"]["steps"],
                    best_step=raw["search"]["best_step"])
            for label, record in entries:
                backend(record)
                state_row = fault_report.state_rows(s, label, record, r, samples)
                state_row["reused_healthy_record"] = r.get("reused_healthy_record")
                all_states.append(state_row)
            if not reused:
                cost["actual_backend_calls"] = 10 * len(entries)
                cost["backend_seconds"] = sum(sum(rec["backend_wall"]) for _, rec in entries)
            inventories.append(dict(job_id=name, raw_sha256=sha(ROOT / r["sink"]),
                exports_sha256={k: sha(ROOT / p) for k, p in r["exports"].items()}))
        elif s["kind"] == "c_split_eval":
            assert r["seed"] == s["seed"] and r["actual_backend_calls"] == 1
            cost.update(actual_backend_calls=1, backend_seconds=r["backend_seconds"],
                        evaluation_CPCM_seconds=r["cpcm_seconds"])
            samples.append(dict(job_id=name, model=s["model"], phase_op=s["phase_op"],
                state=s["method"], backend_seed=s["seed"], backend_ns=r["backend_triple_ns"][0],
                unhidden_compute_ns=r["backend_triple_ns"][1], unhidden_communication_ns=r["backend_triple_ns"][2],
                backend_call_seconds=r["backend_seconds"]))
        costs.append(cost)

    stops, _, trajectories = fault_report.stop_tables(jobs, present) if any(j["kind"] == "fault_stop" for j in jobs.values()) else ([], [], [])
    quality, static_rows, choices, coverage, migrations = [], [], [], [], []
    for model in run.hi_config.MODELS:
        prefix = "C_" + model + "__"
        rep_path = local / "representatives" / (model + ".json")
        if not rep_path.exists():
            coverage.append(dict(model=model, state="pending anchor/structural selection"))
            continue
        representative = run.frozen.read_json(rep_path)
        coverage.append(dict(model=model, phase_op=representative["phase_op"], applicable=representative["applicable"],
                             failed_pe=representative["failed_pe"], reason=representative["reason"],
                             rule=representative["rule"], inspections=representative["inspections"]))
        if not representative["applicable"]:
            continue
        needed = [prefix + k for k in ("pre", "l1", "stop300", "split_prepare")]
        needed += [prefix + f"m{m}_s{s}" for m in (1, 2) for s in range(1, 11)]
        if any(name not in present for name in needed):
            continue
        pre = present[prefix + "pre"]["result"]
        l1 = present[prefix + "l1"]["result"]
        stop = present[prefix + "stop300"]["result"]
        pre_raw = run.frozen.read_pickle(pre["sink"])
        l1_raw = run.frozen.read_pickle(l1["sink"])
        selected_raw = run.frozen.read_pickle(present[stop["source_job"]]["result"]["sink"])
        final_record = selected_raw[stop["level"]] if stop["level"] in ("R0", "L1") else selected_raw["record"]
        healthy = statistics.mean(backend(pre_raw["record"]))
        r0mean = statistics.mean(backend(l1_raw["R0"]))
        s = present[prefix + "pre"]["spec"]
        chosen_state = run.load_split_state(present[prefix + "split_prepare"]["spec"], local)
        prepared = present[prefix + "split_prepare"]["result"]
        choices += [dict(model=model, **r) for r in prepared["choices"]]
        bound = run.frozen.read_json(local / "metadata" / f"C_{model}_m1.json")["rigid_tensor_load_bound"]["lower_bound_ns"]
        final_arms = [("healthy", backend(pre_raw["record"]), pre_raw["record"]["T"], pre["exports"]["healthy"], pre_raw["record"]),
                      ("R0", backend(l1_raw["R0"]), l1_raw["R0"]["T"], l1["exports"]["R0"], l1_raw["R0"]),
                      ("rigid EIR", backend(final_record), final_record["T"], stop["export"], final_record)]
        for method, values, estimate, exported, raw_record in final_arms:
            dead = None if method == "healthy" else tuple(representative["failed_pe"])
            metrics, state_bound = graph_metrics(exported, s, dead)
            row = dict(model=model, phase_op=s["phase_op"], method=method, backend_mean_ns=statistics.mean(values),
                backend_min_ns=min(values), backend_max_ns=max(values), T_hat_ns=estimate,
                relative_to_healthy=statistics.mean(values)/healthy-1, relative_to_R0=statistics.mean(values)/r0mean-1,
                rigid_tensor_lower_bound_ns=bound, all_backend_below_rigid_bound=max(values)<bound,
                tensor_peak_ns=metrics["compute_peak_ns"]["tensorcore"], tensor_load_bound_ns=state_bound["lower_bound_ns"],
                EIR_stop_level=stop["level"] if method == "rigid EIR" else None,
                EIR_stop_reason=stop["reason"] if method == "rigid EIR" else None,
                EIR_repair_seed=stop.get("seed") if method == "rigid EIR" else None)
            quality.append(row)
            static_rows.append(dict(model=model, method=method, **metrics))
            migrations.append(dict(model=model, method=method, **raw_record.get("migration_vs_start", {})))
        for factor, method in ((1, "Local-m1"), (2, "Split-m2")):
            members = [present[prefix + f"m{factor}_s{seed}"]["result"] for seed in range(1, 11)]
            info = run.frozen.read_json(local / "metadata" / f"C_{model}_m{factor}.json")
            values = [r["backend_triple_ns"][0] for r in members]
            assert len({r["T_hat_ns"] for r in members}) == 1
            assert info["coverage_checked"] and info["healthy_computation_moved"] == 0
            assert info["metrics"]["tensor_effective_flops"] == info["baseline_metrics"]["tensor_effective_flops"]
            quality.append(dict(model=model, phase_op=s["phase_op"], method=method, backend_mean_ns=statistics.mean(values),
                backend_min_ns=min(values), backend_max_ns=max(values), T_hat_ns=members[0]["T_hat_ns"],
                relative_to_healthy=statistics.mean(values)/healthy-1, relative_to_R0=statistics.mean(values)/r0mean-1,
                rigid_tensor_lower_bound_ns=bound, all_backend_below_rigid_bound=max(values)<bound,
                tensor_peak_ns=info["metrics"]["compute_peak_ns"]["tensorcore"],
                tensor_load_bound_ns=info["tensor_load_bound"]["lower_bound_ns"], axis=info["axis"],
                decision_seconds=info["axis_selection_seconds"]))
            static_rows.append(dict(model=model, method=method, **info["metrics"],
                block_time_distribution_ns=info["rigid_block_time_distribution_ns"],
                parent_arithmetic=info["parent_arithmetic"],
                affected_weight_network_bytes_before=info["affected_weight_network_bytes_before"],
                affected_weight_network_bytes_after=info["affected_weight_network_bytes_after"],
                affected_activation_network_bytes_after=info["affected_activation_network_bytes_after"]))
            migrations.append(dict(model=model, method=method, **offline_migration(chosen_state, info)))

    summary = dict(status="running" if missing else "complete", planned_jobs=len(jobs), complete_jobs=len(present),
        missing=missing, actual_search_trajectories=sum(r["actual_search_trajectories"] for r in costs),
        actual_backend_calls=sum(r["actual_backend_calls"] for r in costs),
        reused_stop_jobs=sum(r["status"] == "reused" for r in costs),
        reused_healthy_jobs=sum(r.get("healthy_reused", False) for r in costs),
        process_hours=sum(r["wall_seconds"] for r in costs)/3600,
        maximum_job_seconds=max((r["wall_seconds"] for r in costs), default=0),
        peak_rss_gb=max((r["peak_rss_gb"] for r in costs), default=0),
        coverage=coverage, quality=quality, manifest_sha256=sha(run.MANIFEST),
        newly_run_by_report=dict(backend_calls=0, search_trajectories=0))
    return summary, dict(c_quality=quality, c_backend_samples=samples, c_all_EIR_states=all_states,
        c_costs=costs, c_stops=stops, c_trajectories=trajectories, c_static=static_rows,
        c_axis_choices=choices, c_migration=migrations), dict(job_checks=checks, raw_inventory=inventories)


def report(out, progress=False):
    summary, tables, evidence = collect(out, progress)
    for name, rows in tables.items():
        table(run.DEST / "tables" / (name + ".csv"), rows)
    run.frozen.write_json(run.DEST / "evidence/c_completion.json", dict(summary=summary, **evidence))
    text = ["# 两个新模型的 MF6 与局部切分", "", "本报告只读取保存结果；不调用后端或搜索。", "",
        f"已完成 {summary['complete_jobs']}/{summary['planned_jobs']} 个作业；实际新增后端 {summary['actual_backend_calls']} 次、L2 搜索 {summary['actual_search_trajectories']} 条。停档复用 {summary['reused_stop_jobs']} 项、健康状态复用 {summary['reused_healthy_jobs']} 项。", "",
        "故障由健康 A3h decode MHA 的一次 seed1 后端选定：非角落 PE 的 Tensor/Vector 忙碌时间最大者，平局按坐标。PE 的计算单元失效，路由器与链路保留。随后按预先固定的 FFN、projection 顺序选择有孤儿 Tensor 且可合法二分的算子，没有用修复结果筛选位置或算子。", ""]
    for row in summary["coverage"]:
        text.append(f"- {row['model']}：{row.get('phase_op') or '尚无合法代表点'}；{row.get('reason', row.get('state'))}。")
    text += ["", "| 模型/算子 | 状态 | 后端均值ns [最小,最大] | 相对健康 | 相对R0 | 全部十样本低于原固定块下界 |", "|---|---|---:|---:|---:|---|"]
    for r in summary["quality"]:
        text.append(f"| {r['model']}/{r['phase_op']} | {r['method']} | {r['backend_mean_ns']:,.1f} [{r['backend_min_ns']:,}, {r['backend_max_ns']:,}] | {r['relative_to_healthy']:+.2%} | {r['relative_to_R0']:+.2%} | {'是' if r['all_backend_below_rigid_bound'] else '否'} |")
    text += ["", "rigid EIR 按原 2% 及 30/100/300 独立预算停止，内部五条修复轨迹和每条十次后端全部计入成本；上表取原停止档的后端中位代表，完整五种子结果另表保存。Local-m1 与 Split-m2 从同一健康图的故障孤儿块出发，用同一放置和路径规则，健康核原计算落点不变。", "",
        "Split 仅比较合法的 N、M 轴，依次最小化实际取整 FLOP 增量、完整图链路字节、固定轴顺序。表中保存所有候选轴、实际输入复制、输出区域覆盖和取整变化；不假设通信按二分比例减少。", "",
        "固定块下界按真实块时长和故障后可用 Tensor 单元计算：max(max c、sum c/P、ceil(K/P) min c)。同时报告新图下界与最忙核负载。只有后端总延迟实际低于旧块下界，才构成超出旧分块放置可达到范围的证据。", "",
        "唯一权重迁移字节沿用 EIR 的 DDR 来源新增副本定义；局部切分保留原 DDR 源，额外 PE 输入复制和链路传输另列。健康输入和全体原始候选未修改。", "",
        f"累计作业成本 {summary['process_hours']:.4f} 进程小时，最长作业 {summary['maximum_job_seconds']/60:.2f} 分钟，最高 RSS {summary['peak_rss_gb']:.3f} GiB。这是并行质量运行成本；正式单进程计时由 J 报告。", "",
        "复算：`python e2e/hi_closeout_c_report.py --out e2e/hi/out/closeout_20261115`。逐种子样本、全部 EIR 档位、停止决策、通信/下界、轴选择、迁移及逐项成本在 `tables/c_*.csv`；输入/回执/原始状态摘要在 `evidence/c_completion.json`。", ""]
    (run.DEST / "C_NEW_MODEL_SPLIT.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("coverage", "quality", "missing")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=run.OUT)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    report(args.out, args.progress)
