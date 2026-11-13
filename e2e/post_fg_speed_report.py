"""Compare saved post-F/G execution traces and report measured speed changes."""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics

from e_run import write_json


def collect(run_out):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in (run_out / "results").glob("WP2_*.json")]
    return sorted(records, key=lambda r: r["job_id"])


def report(run_out, out):
    records = collect(run_out)
    graphs = [r for r in records if r["kind"] in ("speed_graph", "speed_graph_full")]
    full_graphs = [r for r in graphs if r["kind"] == "speed_graph_full"]
    searches = [r for r in records if r["kind"] == "speed_search"]
    eirs = [r for r in records if r["kind"] == "speed_eir"]
    comparisons = []
    pairs = []
    semantic_fields = ("trajectory", "calibration", "terminal_state", "terminal_schedule", "selected", "best_step")
    for phase in ("decode", "prefill"):
        rows = {r["spec"]["execution"]: r for r in searches if r["spec"]["phase"] == phase}
        for baseline, candidate in (("checks", "normal"), ("normal", "fast"), ("checks", "fast")):
            if baseline in rows and candidate in rows:
                old, new = rows[baseline]["timing"], rows[candidate]["timing"]
                pairs.append(dict(phase=phase, comparison=candidate + "_vs_" + baseline,
                                  semantic_equal=all(rows[baseline][k] == rows[candidate][k] for k in semantic_fields),
                                  calibration_change_percent=100*(new["calibration"]/old["calibration"]-1),
                                  sa_change_percent=100*(new["sa_wall"]/old["sa_wall"]-1),
                                  complete_change_percent=100*(new["complete_short_search"]/old["complete_short_search"]-1),
                                  complete_speedup=old["complete_short_search"]/new["complete_short_search"]))
        if set(rows) != {"checks", "normal", "fast"}:
            continue
        checks = {field: rows["checks"][field] == rows["normal"][field] == rows["fast"][field] for field in semantic_fields}
        checks["accept_reject_targeted_present"] = all(v > 0 for r in rows.values() for v in r["coverage"].values())
        checks["checks_revert_exact"] = all(v == 0 for k, v in rows["checks"]["check_revert"].items() if k != "n")
        checks["same_short_function"] = len({r["patched_function_sha256"] for r in rows.values()}) == 1
        checks["thirty_step_prefix_of_1000"] = all(r["cooling_steps"] == 1000 and r["executed_steps"] == 30
                                                   for r in rows.values())
        comparisons.append(dict(phase=phase, status="passed" if all(checks.values()) else "different",
                                checks=checks, jobs=[r["job_id"] for r in rows.values()],
                                cooling_steps=rows["normal"]["cooling_steps"],
                                executed_steps=rows["normal"]["executed_steps"]))
    eir_modes = {r["spec"]["execution"]: r for r in eirs}
    eir_check = None
    if set(eir_modes) == {"checks", "normal", "fast"}:
        eq = eir_modes["checks"]["quality"] == eir_modes["normal"]["quality"] == eir_modes["fast"]["quality"]
        coverage = all(v > 0 for r in eirs for v in r["coverage"].values())
        exact_revert = all(v == 0 for k, v in eir_modes["checks"]["check_revert"].items() if k != "n")
        eir_check = dict(status="passed" if eq and coverage and exact_revert else "different", quality_equal=eq,
                         checks_revert_exact=exact_revert,
                         coverage_present=coverage, jobs=[r["job_id"] for r in eirs])
    timings = []
    for r in graphs:
        t = r["timing"]
        old = statistics.median(t["original"])
        new = statistics.median(x["total_seconds"] for x in t["accelerated"])
        timings.append(dict(job_id=r["job_id"], kind=r["kind"], phase=r["spec"]["phase"], execution="paired_graph",
                            compile_seconds=t["compile_seconds"], old_forward_seconds=old, new_forward_seconds=new,
                            forward_speedup=old/new, kernel_seconds=statistics.median(x["kernel_seconds"] for x in t["accelerated"]),
                            pack_seconds=statistics.median(x["pack_seconds"] for x in t["accelerated"]),
                            unpack_seconds=statistics.median(x["unpack_seconds"] for x in t["accelerated"]),
                            cold_first_call_seconds=t["cold"]["total_seconds"], peak_rss_gb=r["peak_rss_gb"]))
    for r in searches:
        t = r["timing"]
        timings.append(dict(job_id=r["job_id"], kind=r["kind"], phase=r["spec"]["phase"], execution=r["spec"]["execution"],
                            calibration_seconds=t["calibration"], sa_wall_seconds=t["sa_wall"],
                            complete_short_search_seconds=t["complete_short_search"], cold_first_call_seconds=t["cold_jit"],
                            outside_calibration_and_sa_seconds=t["complete_short_search"]-t["calibration"]-t["sa_wall"],
                            unassigned_sa_seconds=t["sa_wall"]-sum(t["step_fields"].values()),
                            peak_rss_gb=r["peak_rss_gb"], **t["step_fields"]))
    for r in eirs:
        t = r["timing"]
        timings.append(dict(job_id=r["job_id"], kind=r["kind"], phase=r["spec"]["phase"], execution=r["spec"]["execution"],
                            calibration_seconds=t["calibration_seconds"], sa_wall_seconds=t["search_seconds"],
                            restore_best_seconds=t["restore_best_seconds"], complete_short_search_seconds=t["search_call_seconds"],
                            full_job_seconds=r["wall_seconds"], peak_rss_gb=r["peak_rss_gb"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "evidence").mkdir(exist_ok=True)
    fields = list(dict.fromkeys(key for row in timings for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(timings)
    (out / "tables/timing.csv").write_text(buffer.getvalue(), encoding="utf-8-sig", newline="")
    if pairs:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
        (out / "tables/short_search_comparisons.csv").write_text(buffer.getvalue(), encoding="utf-8-sig", newline="")
    complete = len(full_graphs) == 2 and len(comparisons) == 2 and eir_check is not None
    same = all(r["status"] == "passed" for r in comparisons) and (eir_check is None or eir_check["status"] == "passed")
    evidence = dict(status="different" if not same else "passed" if complete else "pending",
                    graph_jobs=[dict(job_id=r["job_id"], **r["equivalence"]) for r in graphs],
                    searches=comparisons, search_pairs=pairs, eir=eir_check, input_root=str(run_out), records=len(records))
    write_json(out / "evidence/equivalence_results.json", evidence)
    text = ["# bb CPCM 与正常执行的提速试点", "", "本轮比较原实现开启检查、原实现正常执行、Numba 正常执行。所有数字来自保存的本次任务记录；独立后端评估次数为零。", "",
            "短搜索保留原 1000 步的温度公式，仅截断主循环；温度校准、精确状态回滚和动作规则保持不变。", "",
            "## 等价检查", "", "| 图 | 保存图逐活动核对 | 短轨迹核对 |", "|---|---|---|"]
    for phase in ("decode", "prefill"):
        g = [r for r in full_graphs if r["spec"]["phase"] == phase]
        c = [r for r in comparisons if r["phase"] == phase]
        text.append(f"| 3×3 增长 {phase} MHA | {'通过' if g else '尚未完成'} | {c[0]['status'] if c else '尚未完成'} |")
    text += ["", "逐活动核对包含资源顺序、依赖开始、最后发出与结束时间、关键链及阻塞归因；短轨迹核对动作、接受、CP/W、最终状态和最好映射。", "",
             "短搜索结果中的 `terminal_schedule` 是SA当前状态的含广播完整图；动作的 `T_new/E_new` 也来自这一完整图。`selected.state` 是选中映射，`selected.T/W/CP` 是该快照在广播前的描述性记录，不能把 `selected.T` 当成SA能量使用的完整时间。主等价结论来自完整轨迹的动作/能量、terminal完整图及选中映射同时一致。", "", "## 保存图推演", "",
             "| 图 | 原推演秒 | 新推演秒（含数组装卸） | 倍数 | 内核秒 | 首次调用秒 |", "|---|---:|---:|---:|---:|---:|"]
    for row in timings:
        if row["kind"] in ("speed_graph", "speed_graph_full"):
            scope = "含广播完整图" if row["kind"] == "speed_graph_full" else "广播前图"
            text.append(f"| {row['phase']} {scope} | {row['old_forward_seconds']:.6f} | {row['new_forward_seconds']:.6f} | {row['forward_speedup']:.3f} | {row['kernel_seconds']:.6f} | {row['cold_first_call_seconds']:.6f} |")
    text += ["", "完整图使用与搜索相同的 BcCache.graph 构造 W1 广播；原、新内核评估同一张图。广播前图仅作实现分项比较，搜索估时采用完整图和短搜索实测。首次调用包含 JIT 编译或机器码缓存加载及数组装卸；稳态取三次中位数。图解析另计，不能把内核倍数直接代入完整搜索耗时。", "", "## 完整短搜索", "",
             "| 图 | 执行方式 | 校准秒 | SA秒 | 两者之外秒 | 完整任务秒 | 峰值GiB | 接受/拒绝/靶向 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in searches:
        t = row["timing"]
        coverage = row["coverage"]
        remainder = t['complete_short_search']-t['calibration']-t['sa_wall']
        text.append(f"| {row['spec']['phase']} | {row['spec']['execution']} | {t['calibration']:.3f} | {t['sa_wall']:.3f} | {remainder:.3f} | {t['complete_short_search']:.3f} | {row['peak_rss_gb']:.3f} | {coverage['accepted']}/{coverage['rejected']}/{coverage['targeted']} |")
    text += ["", "两者之外的时间包含模板初始化、图与对象构造、首次内核调用、选中状态描述及文件保存；本入口没有把这些部分分别计时，不能把这一差值全部称为初始化时间。三组均从模板开始，完整任务列也包含各自的这部分成本。"]
    text += ["", "| 图 | 比较 | 校准变化 | SA变化 | 完整任务变化 | 完整任务加速倍数 | 语义逐项一致 |", "|---|---|---:|---:|---:|---:|---|"]
    for pair in pairs:
        text.append(f"| {pair['phase']} | {pair['comparison']} | {pair['calibration_change_percent']:+.2f}% | {pair['sa_change_percent']:+.2f}% | {pair['complete_change_percent']:+.2f}% | {pair['complete_speedup']:.3f} | {pair['semantic_equal']} |")
    text += ["", "校准、SA循环与完整任务分别比较。原实现的校准不受 checks 开关控制，checks/normal之间的校准耗时变化反映这两次测量的时间波动。每方式只执行这一条确定性短轨迹，表中百分比是配对实测变化。"]
    text += ["", "## 复用 L1 的短 EIR", "", "G3×3 增长 decode MHA/PF4/r123 使用已有 L1 状态，保留原 30 步 L2Gsync 搜索及校准，省去独立后端重复。", "",
             "| 执行方式 | 校准秒 | 搜索秒 | 恢复最好状态秒 | 搜索调用秒 | 整个作业秒 |", "|---|---:|---:|---:|---:|---:|"]
    for r in eirs:
        t = r["timing"]
        text.append(f"| {r['spec']['execution']} | {t['calibration_seconds']:.6f} | {t['search_seconds']:.6f} | {t['restore_best_seconds']:.6f} | {t['search_call_seconds']:.6f} | {r['wall_seconds']:.6f} |")
    text += ["", "EIR 等价结果：" + (eir_check["status"] if eir_check else "尚未完成") + "。", "",
             "这些是短轨迹和既有保存图的测量。后续 H/I 的工作量与配置确定后再估算进程小时、可并行墙钟和评估成本；A–G 的质量数据继续保留。", "",
             "来源：`tables/timing.csv`、`evidence/equivalence_results.json`，每项结果包含原任务 spec 和完整轨迹。", ""]
    (out / "BB_SPEEDUP.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps(dict(status=evidence["status"], records=len(records), output=str(out))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("e2e/post_fg"))
    args = parser.parse_args()
    report(args.run_out, args.out)
