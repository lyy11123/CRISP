"""Summarize E candidate results, fixed layouts, and C fault reuse without running simulations."""

import argparse
from collections import Counter, defaultdict
import csv
import glob
import json
from pathlib import Path
import statistics

from e_layouts import capacity_rows, layouts


HERE = Path(__file__).resolve().parent
PHASES = ("prefill", "decode")
OPS = ("ln", "mha", "proj", "ffn")
SHAPES = ("2x2", "1x4")
METHODS = ("A1", "A3h", "Gemini")
KINDS = ("search", "reeval", "terminal")
LN_MODES = ("asis", "fixed")
OUTPUT_LENGTHS = (128, 512, 2048)
SEEDS = list(range(1, 11))
CANDIDATES = 14
SEARCH_STEPS = 1000
FAULT_GROUPS = ("G2", "G0", "G1")

C_SCHEMES = {
    "A1 R0": ("A1", "A1 故障前"),
    "A1 FULL-BusyBarn": ("A1", "A1 故障前"),
    "A3h R0": ("A3h", "A3h 故障前"),
    "A3h L2Gsync 停止档（均值）": ("A3h", "A3h 故障前"),
    "A3h FULL-A3h": ("A3h", "A3h 故障前"),
}


def candidate_kind(shape, phase, op, method):
    return "reeval" if method != "Gemini" and (shape == "2x2" or (phase, op) == ("decode", "ffn")) else "search"


def all_cells():
    return [(shape, phase, op, method) for shape in SHAPES for phase in PHASES for op in OPS for method in METHODS]


def planned_counts():
    result = {(candidate_kind(*cell), *cell): CANDIDATES for cell in all_cells()}
    result.update({("terminal", shape, phase, op, "Gemini"): 1 for shape in SHAPES for phase in PHASES for op in OPS})
    return result


def read_records(inputs):
    paths = set()
    for pattern in inputs:
        for match in glob.glob(pattern, recursive=True):
            path = Path(match).resolve()
            paths.update(path.rglob("results/*.json") if path.is_dir() else (path,))
            if path.is_dir() and path.name == "results":
                paths.update(path.glob("*.json"))
    if not paths:
        raise ValueError("No E result JSON files matched --inputs")
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths)]


def record_cell(record):
    return tuple(record[key] for key in ("shape", "phase", "op", "method"))


def formal_results(records):
    accepted = {}
    excluded = Counter()
    plan = planned_counts()
    for record in records:
        if record.get("status") != "ok":
            excluded["unsuccessful"] += 1
            continue
        if record.get("search_steps") != SEARCH_STEPS:
            excluded["nonformal_steps"] += 1
            continue
        class_key = record["kind"], *record_cell(record)
        if class_key not in plan:
            raise ValueError(f"Unexpected formal class {class_key}")
        backend = record["backend"]
        if sorted(item["seed"] for item in backend) != SEEDS or any(item["cycles"] <= 0 for item in backend):
            raise ValueError(f"Invalid backend evaluation in {record['candidate_id']}")
        key = *class_key, record["candidate_id"]
        if key in accepted and record != accepted[key]:
            raise ValueError(f"Different results for candidate {key}; choose the intended batch")
        accepted[key] = record
    return list(accepted.values()), dict(excluded)


def operator_rows(formal):
    by_cell = defaultdict(list)
    for record in formal:
        if record["kind"] != "terminal":
            cycles = [item["cycles"] for item in record["backend"]]
            by_cell[record_cell(record)].append({"id": record["candidate_id"], "order": record["candidate_order"],
                                                "mean": statistics.mean(cycles), "min": min(cycles), "max": max(cycles),
                                                "T_hat": record.get("T_hat"), "T_hat_rand_mean": record.get("T_hat_rand_mean")})
    rows = []
    for shape, phase, op, method in all_cells():
        candidates = by_cell[shape, phase, op, method]
        row = {"shape": shape, "phase": phase, "op": op, "method": method, "completed": len(candidates),
               "planned": CANDIDATES}
        orders = {candidate["order"] for candidate in candidates}
        if len(orders) != len(candidates) or not orders <= set(range(CANDIDATES)):
            raise ValueError(f"Invalid or duplicate candidate order in {row}")
        if len(candidates) == CANDIDATES:
            best = min(candidates, key=lambda x: (x["mean"], x["order"]))
            values = [x["mean"] for x in candidates]
            row.update(candidate=best["id"], cycles=best["mean"], backend_min=best["min"], backend_max=best["max"],
                       candidate_median=statistics.median(values), candidate_min=min(values), candidate_max=max(values))
            for field, prefix in (("T_hat", "det_cpcm"), ("T_hat_rand_mean", "random_cpcm")):
                if all(x[field] is not None for x in candidates):
                    selected = min(candidates, key=lambda x: (x[field], x["order"]))
                    row.update({f"{prefix}_candidate": selected["id"], f"{prefix}_backend_cycles": selected["mean"]})
        rows.append(row)
    by_key = {(r["shape"], r["phase"], r["op"], r["method"]): r for r in rows}
    for r in rows:
        if r["method"] == "A3h" and "cycles" in r:
            for other in ("A1", "Gemini"):
                reference = by_key[r["shape"], r["phase"], r["op"], other]
                if "cycles" in reference:
                    r[f"relative_to_{other}"] = r["cycles"] / reference["cycles"] - 1
    return rows


def layer_rows(operators):
    cycles = {(r["shape"], r["phase"], r["method"], r["op"]): r.get("cycles") for r in operators}
    rows = []
    for shape in SHAPES:
        for method in METHODS:
            for ln in LN_MODES:
                row = {"shape": shape, "method": method, "ln": ln}
                for phase in PHASES:
                    lnorm, mha, proj, ffn = (cycles[shape, phase, method, op] for op in OPS)
                    coefficient = 1 if phase == "decode" and ln == "fixed" else 2
                    row[phase] = (coefficient * lnorm + 8 * mha + proj + ffn
                                  if None not in (lnorm, mha, proj, ffn) else None)
                rows.append(row)
    return rows


def model_rows(layers, fixed_layouts):
    rows = []
    table = {(r["shape"], r["method"], r["ln"]): r for r in layers}
    for layout in fixed_layouts:
        for method in METHODS:
            for ln in LN_MODES:
                row = {"layout": layout.name, "method": method, "ln": ln}
                for phase in PHASES:
                    values = [(g.layers, table[g.shape, method, ln][phase]) for g in layout.groups]
                    if all(value is not None for _, value in values):
                        intra = sum(n * value for n, value in values)
                        row[f"{phase}_within_groups"] = intra
                        row[f"{phase}_between_groups"] = layout.transfer_cycles(phase)
                        row[phase] = intra + row[f"{phase}_between_groups"]
                if all(phase in row for phase in PHASES):
                    row.update({f"request_{length}": row["prefill"] + length * row["decode"] for length in OUTPUT_LENGTHS})
                rows.append(row)
    table = {(r["layout"], r["method"], r["ln"]): r for r in rows}
    for r in rows:
        if r["method"] == "A3h":
            for other in ("A1", "Gemini"):
                reference = table[r["layout"], other, r["ln"]]
                for key in (*PHASES, *(f"request_{length}" for length in OUTPUT_LENGTHS)):
                    if key in r and key in reference:
                        r[f"{key}_relative_to_{other}"] = r[key] / reference[key] - 1
    return rows


def stage_bottleneck(stage_cycles):
    cycles = max(stage_cycles.values())
    return cycles, ";".join(group for group, value in stage_cycles.items() if value == cycles)


def pipeline_fields(stages):
    """Ideal stage service demand; requests sum P + N*D within each group first."""
    row = {}
    for when, phases in stages.items():
        demands = dict(phases)
        for length in OUTPUT_LENGTHS:
            demands[f"request_{length}"] = {
                group: value + length * phases["decode"][group]
                for group, value in phases["prefill"].items()
            }
        for key, values in demands.items():
            cycles, groups = stage_bottleneck(values)
            row[f"pipeline_stage_{key}_{when}"] = cycles
            row[f"pipeline_bottleneck_groups_{key}_{when}"] = groups
    for key in (*PHASES, *(f"request_{length}" for length in OUTPUT_LENGTHS)):
        before, after = (row[f"pipeline_stage_{key}_{when}"] for when in ("before", "after"))
        row[f"pipeline_stage_{key}_change"] = after / before - 1
        row[f"pipeline_ideal_throughput_{key}_change"] = before / after - 1
    return row


def fault_rows(layers, mix, c_report):
    table = {(r["shape"], r["method"], r["ln"]): r for r in layers}
    rows = []
    for damage_group in FAULT_GROUPS:
        healthy_groups = [g for g in mix.groups if g.name != damage_group]
        affected = next(g for g in mix.groups if g.name == damage_group)
        for fault in c_report["faults"]:
            by_scheme = {(r["scheme"], r["ln"]): r for r in fault["assembly"]}
            for scheme, (method, before_scheme) in C_SCHEMES.items():
                for ln in LN_MODES:
                    healthy = {phase: [(g.layers, table[g.shape, method, ln][phase]) for g in healthy_groups] for phase in PHASES}
                    if any(value is None for values in healthy.values() for _, value in values):
                        continue
                    before, after = by_scheme[before_scheme, ln], by_scheme[scheme, ln]
                    row = {"fault": fault["fault"], "scheme": scheme, "method": method, "ln": ln,
                           "layout": mix.name, "group": damage_group, "affected_layers": affected.layers}
                    stages = {when: {} for when in ("before", "after")}
                    for phase, key in (("prefill", "TTFT"), ("decode", "TPOT")):
                        fixed = sum(n * value for n, value in healthy[phase]) + mix.transfer_cycles(phase)
                        row[f"local_{phase}_before"] = before[key] * affected.layers
                        row[f"local_{phase}_after"] = after[key] * affected.layers
                        row[f"model_{phase}_before"] = fixed + row[f"local_{phase}_before"]
                        row[f"model_{phase}_after"] = fixed + row[f"local_{phase}_after"]
                        row[f"paired_before_vs_E_healthy_{phase}"] = before[key] / table[affected.shape, method, ln][phase] - 1
                        for when, source in (("before", before), ("after", after)):
                            stages[when][phase] = {
                                g.name: g.layers * (source[key] if g.name == damage_group else table[g.shape, method, ln][phase])
                                for g in mix.groups
                            }
                    for scope in ("local", "model"):
                        for length in OUTPUT_LENGTHS:
                            for when in ("before", "after"):
                                row[f"{scope}_request_{length}_{when}"] = (row[f"{scope}_prefill_{when}"] + length * row[f"{scope}_decode_{when}"])
                        for key in (*PHASES, *(f"request_{length}" for length in OUTPUT_LENGTHS)):
                            row[f"{scope}_{key}_change"] = row[f"{scope}_{key}_after"] / row[f"{scope}_{key}_before"] - 1
                    for length in OUTPUT_LENGTHS:
                        e_layer = table[affected.shape, method, ln]
                        e_request = e_layer["prefill"] + length * e_layer["decode"]
                        row[f"paired_before_vs_E_healthy_request_{length}"] = (before["TTFT"] + length * before["TPOT"]) / e_request - 1
                    row.update(pipeline_fields(stages))
                    rows.append(row)
    return rows


def pipeline_summary_rows(faults):
    rows = []
    for group in FAULT_GROUPS:
        for ln in LN_MODES:
            for scheme in C_SCHEMES:
                selected = [r for r in faults if (r["group"], r["ln"], r["scheme"]) == (group, ln, scheme)]
                if not selected:
                    continue
                row = {"group": group, "affected_layers": selected[0]["affected_layers"], "ln": ln,
                       "scheme": scheme, "fault_count": len(selected)}
                for key in (*PHASES, *(f"request_{length}" for length in OUTPUT_LENGTHS)):
                    for metric in (f"model_{key}_change", f"pipeline_stage_{key}_change", f"pipeline_ideal_throughput_{key}_change"):
                        values = [r[metric] for r in selected]
                        row.update({f"{metric}_{label}": value for label, value in (
                            ("min", min(values)), ("median", statistics.median(values)), ("max", max(values)))})
                rows.append(row)
    return rows


def timing(formal):
    by_class = defaultdict(list)
    for record in formal:
        by_class[record["kind"], *record_cell(record)].append(record)
    rows = []
    for key, planned in planned_counts().items():
        kind, shape, phase, op, method = key
        records = by_class[key]
        wall = [r["wall_seconds"] for r in records if r.get("wall_seconds") is not None]
        peaks = [r["peak_rss_gb"] for r in records if r.get("peak_rss_gb") is not None]
        completed = len(records)
        remaining = planned - completed
        if remaining < 0:
            raise ValueError(f"Too many results in {key}")
        row = {"kind": kind, "shape": shape, "phase": phase, "op": op, "method": method, "planned": planned,
               "completed": completed, "timed": len(wall), "remaining": remaining, "observed_wall_seconds": sum(wall),
               "observed_search_seconds": sum(r.get("search_seconds") or 0 for r in records),
               "observed_evaluation_seconds": sum(r.get("evaluation_seconds") or 0 for r in records),
               "peak_rss_gb": max(peaks, default=None)}
        if wall:
            row.update(min_seconds=min(wall), median_seconds=statistics.median(wall), max_seconds=max(wall))
        if wall or remaining == 0:
            row.update(remaining_low_seconds=remaining * min(wall, default=0), remaining_high_seconds=remaining * max(wall, default=0))
        rows.append(row)
    totals = {}
    for kind in KINDS:
        group = [r for r in rows if r["kind"] == kind]
        missing = [r for r in group if "remaining_low_seconds" not in r]
        totals[kind] = {"planned": sum(r["planned"] for r in group), "completed": sum(r["completed"] for r in group),
                        "observed_process_hours": sum(r["observed_wall_seconds"] for r in group) / 3600,
                        "estimated_remaining_process_hours_low": sum(r.get("remaining_low_seconds", 0) for r in group) / 3600,
                        "estimated_remaining_process_hours_high": sum(r.get("remaining_high_seconds", 0) for r in group) / 3600,
                        "untimed_remaining_jobs": sum(r["remaining"] for r in missing),
                        "untimed_remaining_classes": len(missing)}
    return rows, totals


def terminal_rows(formal):
    rows = []
    for record in formal:
        if record["kind"] == "terminal":
            cycles = [item["cycles"] for item in record["backend"]]
            rows.append({"shape": record["shape"], "phase": record["phase"], "op": record["op"],
                         "candidate": record["candidate_id"], "cycles": statistics.mean(cycles),
                         "backend_min": min(cycles), "backend_max": max(cycles)})
    return rows


def candidate_rows(formal):
    rows = []
    for record in formal:
        values = [item["cycles"] for item in record["backend"]]
        rows.append({key: record[key] for key in (
            "candidate_id", "kind", "shape", "phase", "op", "method", "wall_seconds", "peak_rss_gb")} |
                    {"mean_cycles": statistics.mean(values), "min_cycles": min(values), "max_cycles": max(values),
                     "T_hat": record["T_hat"], "T_hat_rand_mean": record["T_hat_rand_mean"]})
    return rows


def write_csv(path, rows, empty_fields=()):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or empty_fields
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, out_dir, c_report_path):
    formal, excluded = formal_results(records)
    fixed = layouts()
    operators = operator_rows(formal)
    layers = layer_rows(operators)
    models = model_rows(layers, fixed)
    have_c = c_report_path.is_file()
    faults = fault_rows(layers, fixed[0], json.loads(c_report_path.read_text(encoding="utf-8"))) if have_c else []
    times, totals = timing(formal)
    capacity = [row for layout in fixed for row in capacity_rows(layout)]
    terminals = terminal_rows(formal)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"candidate_table": candidate_rows(formal), "operator_table": operators,
              "layer_table": layers, "model_table": models, "fault_model_table": faults,
              "capacity_table": capacity, "timing_table": times, "terminal_table": terminals,
              "pipeline_summary_table": pipeline_summary_rows(faults)}
    empty_fields = {"fault_model_table": ("fault", "scheme", "method", "ln", "layout", "group", "affected_layers"),
                    "terminal_table": ("shape", "phase", "op", "candidate", "cycles", "backend_min", "backend_max"),
                    "pipeline_summary_table": ("group", "affected_layers", "ln", "scheme", "fault_count")}
    for name, rows in tables.items():
        write_csv(out_dir / f"{name}.csv", rows, empty_fields.get(name, ()))
    (out_dir / "layouts.json").write_text(json.dumps({layout.name: layout.describe() for layout in fixed}, indent=2) + "\n", encoding="utf-8")
    metadata = {"completion": totals, "excluded": excluded, "c_report_available": have_c,
                "fault_assembly": {
                    "groups": list(FAULT_GROUPS),
                    "sources": "C paired before/after on damaged 2x2 group; E healthy layer timings on other groups",
                    "serial_model": "sum of group service cycles plus 13 serial inter-group transfers",
                    "pipeline_stage": "max_g n_g*t_g, separately for P, D and P+N*D; request service sums P+N*D in each group before taking max",
                    "pipeline_ideal_throughput_change": "stage_cycles_before / stage_cycles_after - 1",
                    "pipeline_assumptions": "1/B is an ideal service-rate upper bound from per-group work; enough independent requests must keep every group occupied. Excludes inter-group transfer, shared-link contention, pipeline fill/drain and autoregressive feedback scheduling",
                    "baseline_difference": "paired_before_vs_E_healthy_* records the C/E healthy timing difference for the same group shape; C paired baseline is retained",
                    "placement": "Same C local fault reused analytically at G2 (3 layers), G0 and G1 (4 layers each); group-interconnect transfer term remains fixed"
                }}
    (out_dir / "summary.json").write_text(json.dumps(tables | metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = ["# E 阶段汇总", "", "正式候选采用 1,000 步和后端种子 1–10。每个算子单元格有齐 14 个候选时才给选优和组装结果。", "",
            "| 作业类别 | 完成 / 计划 | 已测进程小时 | 已测速率覆盖的剩余进程小时 | 待测速作业 |",
            "|---|---:|---:|---:|---:|"]
    for kind, row in totals.items():
        text.append(f"| {kind} | {row['completed']} / {row['planned']} | {row['observed_process_hours']:.2f} | "
                    f"{row['estimated_remaining_process_hours_low']:.2f}–{row['estimated_remaining_process_hours_high']:.2f} | "
                    f"{row['untimed_remaining_jobs']} |")
    text += ["", "剩余时间仅覆盖同形状、同阶段、同算子、同方法且已有样本的作业，取样本最小至最大用时。单样本的两个端点相同。",
             "", "完整模型按 44 层相加并计串行组间激活传输。请求长度表示 prefill 后追加的 decode 步数，KV 长度固定为 2048。",
             "", "容量表列的是逻辑权重、共享历史 KV 和显式激活缓冲假设。DDR 容量采用每 Die 32 GiB，SRAM 以每核 16 MiB 列出。", ""]
    if excluded:
        text += [f"另有 {excluded.get('nonformal_steps', 0)} 条非正式步数记录和 {excluded.get('unsuccessful', 0)} 条未成功记录。", ""]
    if not have_c:
        text += ["当前未找到 C 故障数据。", ""]
    finished_models = [r for r in models if "request_2048" in r]
    if finished_models:
        text += ["## 44 层组装（周期）", "", "| 布局 | 方法 | LN | Prefill | Decode 一步 | 请求 128 | 请求 512 | 请求 2048 |",
                 "|---|---|---|---:|---:|---:|---:|---:|"]
        for row in finished_models:
            values = [f"{row[k]:,.0f}" for k in (*PHASES, *(f"request_{n}" for n in OUTPUT_LENGTHS))]
            text.append(f"| {row['layout']} | {row['method']} | {row['ln']} | " + " | ".join(values) + " |")
        text.append("")
    if faults:
        text += ["## 故障位置与最慢组的解析计算", "",
                 "同一 C 故障分别放在 G2（三层）、G0 或 G1（四层）。受损组的前后时间沿用 C 成对数据，其余组使用 E 健康数据。", "",
                 "最慢组周期 B 为 max_g(n_g × t_g)，分别计算 prefill、decode，以及每组 P + N × D 的请求服务量。1/B 是按组工作量算出的理想服务率上限，其相对变化为 B_before / B_after − 1；周期增加与服务率下降是倒数关系。该计算假设有足量独立请求使各组持续工作，组间传输、共享链路争用、填充排空及自回归反馈调度另计。", "",
                 "C 与 E 的健康测量不同，可能改变四层组之间的瓶颈归属；逐行 paired_before_vs_E_healthy_* 列出差值。", "",
                 "| 受损组 | 方案 | 44 层请求2048变化中位数 | 最慢组请求服务量变化中位数 | 理想服务率变化中位数 |",
                 "|---|---|---:|---:|---:|"]
        for row in tables["pipeline_summary_table"]:
            if row["ln"] == "asis":
                keys = ("model_request_2048_change_median", "pipeline_stage_request_2048_change_median", "pipeline_ideal_throughput_request_2048_change_median")
                text.append(f"| {row['group']} | {row['scheme']} | " + " | ".join(f"{row[key]:+.2%}" for key in keys) + " |")
        text.append("")
    (out_dir / "E_SUMMARY.md").write_text("\n".join(text), encoding="utf-8")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Result JSON files, globs, or a batch/results directory")
    parser.add_argument("--out-dir", type=Path, default=HERE / "e" / "out" / "report")
    parser.add_argument("--c-report", type=Path, default=HERE / "c" / "out" / "c_report.json")
    args = parser.parse_args()
    print(json.dumps(summarize(read_records(args.inputs), args.out_dir, args.c_report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
