"""Read F/G results and saved graphs; write pilot and healthy-candidate tables.

This performs no search or backend simulation. Latency is reported per layer on
one Die group; the 56-Die assembly is not inferred from an unmeasured tiling.
"""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import pickle
import statistics

from e_run import ROOT, write_json
from fg_check import check, estimate, read

OPS = ("ln", "mha", "proj", "ffn")
PHASES = ("prefill", "decode")
BASELINE_ID = "E_busybarn_2x2_fixed_d2d1"
ID_FIELDS = ("config_id", "stage", "hardware_profile", "shape", "split_mode", "d2d_scale", "family", "fault_id")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def candidate_rows(records):
    rows = []
    for r in records.values():
        values = [v["cycles"] for v in r["backend"]]
        row = {k: r[k] for k in (*ID_FIELDS, "candidate_id", "candidate_order", "phase", "op", "method", "kind",
                                "search_steps", "wall_seconds", "search_seconds", "evaluation_seconds", "peak_rss_gb",
                                "T_hat", "T_hat_rand_mean")}
        row.update(backend_mean_ns=statistics.mean(values), backend_min_ns=min(values), backend_max_ns=max(values),
                   backend_seed_count=len(values), backend_seeds=[v["seed"] for v in r["backend"]],
                   backend_reused_from=r.get("evaluation_reused_from"),
                   workload_sha256=r["workload_signature"]["sha256"], **r["fg_metrics"])
        rows.append(row)
    return rows


def operator_rows(candidates):
    grouped = defaultdict(list)
    for row in candidates:
        if row["kind"] != "terminal":
            grouped[tuple(row[k] for k in (*ID_FIELDS, "phase", "op", "method"))].append(row)
    rows = []
    for key, candidates_ in sorted(grouped.items()):
        row = dict(zip((*ID_FIELDS, "phase", "op", "method"), key))
        row.update(completed=len(candidates_), planned=14,
                   all_candidate_mean_ns=statistics.mean(c["backend_mean_ns"] for c in candidates_),
                   candidate_mean_min_ns=min(c["backend_mean_ns"] for c in candidates_),
                   candidate_mean_max_ns=max(c["backend_mean_ns"] for c in candidates_))
        if len(candidates_) == 14:
            assert {c["candidate_order"] for c in candidates_} == set(range(14)), key
            best = min(candidates_, key=lambda c: (c["backend_mean_ns"], c["candidate_order"]))
            random = min(candidates_, key=lambda c: (c["T_hat_rand_mean"], c["candidate_order"]))
            deterministic = min(candidates_, key=lambda c: (c["T_hat"], c["candidate_order"]))
            row.update(backend_best_candidate=best["candidate_id"], backend_best_ns=best["backend_mean_ns"],
                       candidate_median_ns=statistics.median(c["backend_mean_ns"] for c in candidates_),
                       random_cpcm_candidate=random["candidate_id"], random_cpcm_ns=random["backend_mean_ns"],
                       random_selector_gap=random["backend_mean_ns"] / best["backend_mean_ns"] - 1,
                       deterministic_cpcm_candidate=deterministic["candidate_id"],
                       deterministic_cpcm_ns=deterministic["backend_mean_ns"],
                       deterministic_selector_gap=deterministic["backend_mean_ns"] / best["backend_mean_ns"] - 1,
                       selection_status="fourteen_candidates_complete")
        else:
            row["selection_status"] = "partial_candidates"
            if len(candidates_) == 1:
                row.update(pilot_candidate=candidates_[0]["candidate_id"], pilot_candidate_ns=candidates_[0]["backend_mean_ns"])
        rows.append(row)
    return rows


def baseline_operators(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    result = []
    for row in rows:
        if row["shape"] != "2x2":
            continue
        result.append(dict(config_id=BASELINE_ID, stage="E_reuse", hardware_profile="busybarn", shape="2x2",
            split_mode="fixed", d2d_scale=1.0, family="healthy", fault_id="healthy", phase=row["phase"],
            op=row["op"], method=row["method"], completed=14, planned=14,
            backend_best_candidate=row["candidate"], backend_best_ns=float(row["cycles"]),
            candidate_median_ns=float(row["candidate_median"]),
            random_cpcm_candidate=row["random_cpcm_candidate"], random_cpcm_ns=float(row["random_cpcm_backend_cycles"]),
            random_selector_gap=float(row["random_cpcm_backend_cycles"])/float(row["cycles"])-1,
            deterministic_cpcm_candidate=row["det_cpcm_candidate"],
            deterministic_cpcm_ns=float(row["det_cpcm_backend_cycles"]),
            deterministic_selector_gap=float(row["det_cpcm_backend_cycles"])/float(row["cycles"])-1,
            selection_status="reused_E_fourteen_candidates", source=str(path.relative_to(ROOT))))
    return result


def layer_rows(operators):
    groups = defaultdict(dict)
    for row in operators:
        key = tuple(row[k] for k in (*ID_FIELDS, "method"))
        groups[key][row["phase"], row["op"]] = row
    rows = []
    for key, cells in sorted(groups.items()):
        if not all((phase, op) in cells for phase in PHASES for op in OPS):
            continue
        for protocol in ("backend_best", "candidate_median", "random_cpcm", "pilot_candidate"):
            if not all(protocol + "_ns" in r for r in cells.values()):
                continue
            row = dict(zip((*ID_FIELDS, "method"), key))
            row.update(protocol=protocol, ln_count="fixed", micro_mha_heads=8, mha_multiplier=8,
                       model_heads=64, batch=16, hidden=6144, prefill_length=2048, decode_kv_length=2048,
                       measurement="operator backend results assembled into one layer")
            for phase in PHASES:
                row[phase + "_ns"] = sum((1 if phase == "decode" else 2) * cells[phase, op][protocol + "_ns"]
                    if op == "ln" else (8 if op == "mha" else 1) * cells[phase, op][protocol + "_ns"] for op in OPS)
            row["TTFT_us"] = row["prefill_ns"] / 1000
            row["TPOT_us"] = row["decode_ns"] / 1000
            for length in (512, 2048):
                row[f"request_{length}_ns"] = row["prefill_ns"] + length * row["decode_ns"]
            rows.append(row)
    lookup = {(r["config_id"], r["family"], r["fault_id"], r["method"], r["protocol"]): r for r in rows}
    for row in rows:
        if row["method"] == "A3h":
            for other in ("A1", "Gemini"):
                ref = lookup.get((row["config_id"], row["family"], row["fault_id"], other, row["protocol"]))
                if ref:
                    for metric in ("prefill", "decode", "request_512", "request_2048"):
                        row[f"{metric}_relative_to_{other}"] = row[metric+"_ns"] / ref[metric+"_ns"] - 1
        if row["stage"] == "G" and row["family"] == "healthy":
            ref = lookup.get((BASELINE_ID, "healthy", "healthy", row["method"], row["protocol"]))
            if ref:
                cores = math.prod(map(int, row["shape"].split("x"))) * 16
                for metric in ("prefill", "decode", "request_512", "request_2048"):
                    speedup = ref[metric+"_ns"] / row[metric+"_ns"]
                    row[metric+"_speedup_vs_64cores"] = speedup
                    row[metric+"_strong_scaling_efficiency"] = speedup / (cores / 64)
    return rows


def snapshot_workload(path, spec):
    import e2e_pilot
    c = e2e_pilot.C
    c.WORK = str(ROOT / "e2e/work/artifact")
    c._artifact_path()
    with gzip.open(path, "rb") as file:
        blob, _ = pickle.load(file)
    behaviours, data, events = pickle.loads(blob)
    effective, rounded = 0, 0
    counts, stage_counts, stage_ops, stage_flops = Counter(), Counter(), defaultdict(set), Counter()
    reduction_inputs = 0
    for b in behaviours.values():
        counts[b.beha_type] += 1
        stage = b.beha_name.rsplit("_", 1)[0]
        stage_counts[stage] += 1
        stage_ops[stage].add(b.beha_tag[:-1])
        if b.device == "tensorcore":
            a, weight = b.beha_datashape[:2]
            k = int(a[-1]); outputs = math.prod(int(n) for n in a[:-1]) * int(weight[-1])
            logical = 2 * k * outputs
            executed = 2 * math.ceil(k / 128) * 128 * math.ceil(outputs / 64) * 64
            effective += logical; rounded += executed; stage_flops[stage] += logical
        if b.beha_type in ("matadd", "redsum", "redmax"):
            reduction_inputs += sum(math.prod(int(n) for n in shape) for shape in b.beha_datashape)
    heads = len(set().union(*(v for k, v in stage_ops.items() if k.startswith("MHA_qconv")))) if spec["op"] == "mha" else None
    inputs = [{"tag": list(tag), "name": d.data_name, "shape": list(map(int, d.data_shape)),
               "elements": math.prod(map(int, d.data_shape))} for tag, d in data.items() if d.data_name == "ifmap"]
    primary = inputs[0]
    assert primary["shape"][0] == 16 and primary["shape"][-1] == 6144, primary
    assert primary["shape"][-2] == (2048 if spec["phase"] == "prefill" else 1), primary
    if spec["op"] == "mha":
        assert heads == 8, (spec["candidate_id"], heads)
    return {**{k: spec[k] for k in (*ID_FIELDS, "phase", "op")},
            "source_candidate": spec["candidate_id"], "source_snapshot": str(path),
            "input_tensors": inputs, "model_heads":64, "micro_mha_heads":heads,
            "mha_aggregation_multiplier":8 if spec["op"]=="mha" else None,
            "tensor_effective_multiply_add_flops":effective, "tensor_grain_rounded_flops":rounded,
            "tensor_padding_fraction":rounded/effective-1 if effective else None,
            "tensor_bias_additions_in_flop_count":False,
            "reduction_events":sum(counts[k] for k in ("matadd","redsum","redmax")),
            "reduction_input_elements":reduction_inputs,
            "computation_events_by_type":dict(counts), "computation_events_by_stage":dict(stage_counts),
            "effective_tensor_flops_by_stage":dict(stage_flops),
            "prebroadcast_events":len(events), "tensor_objects":len(data),
            "intermediate_tensor_elements_sum":sum(math.prod(map(int,d.data_shape)) for d in data.values())}


def workload_rows(specs, records, out):
    samples = {}
    for spec in specs:
        if spec["candidate_id"] in records and spec["family"] == "healthy":
            samples.setdefault((spec["config_id"],spec["phase"],spec["op"]),spec)
    rows = [snapshot_workload(out/"snapshots"/(s["candidate_id"]+".pkl.gz"),s) for s in samples.values()]
    by_op = defaultdict(set)
    for row in rows:
        by_op[row["phase"],row["op"]].add(row["tensor_effective_multiply_add_flops"])
    return rows, [{"phase":key[0],"op":key[1],"effective_tensor_work_equal_across_configurations":len(values)==1,
                   "effective_tensor_work_values":sorted(values)} for key,values in sorted(by_op.items())]


def observed_time(out, records, batch_path):
    result = dict(result_process_hours=sum(r["wall_seconds"] for r in records.values())/3600,
                  max_peak_rss_gb=max((r["peak_rss_gb"] for r in records.values()),default=0),
                  backend_seed_counts=sorted({len(r["backend"]) for r in records.values()}),
                  backend_actual_calls=sum(len(r["backend"]) for r in records.values() if "evaluation_reused_from" not in r),
                  backend_reused_samples=sum(len(r["backend"]) for r in records.values() if "evaluation_reused_from" in r))
    if batch_path:
        batch=read(batch_path)
        fresh=[r for r in batch["runs"] if not r.get("cached")]
        seconds=sum(r["wall_seconds"] for r in fresh)
        result.update(batch_file=str(batch_path),batch_wall_seconds=batch["wall_seconds"],
            worker_process_hours=seconds/3600,mean_active_workers=seconds/batch["wall_seconds"],
            configured_worker_limit=batch["workers"],max_observed_active=batch["max_active"],
            min_available_gib=batch["min_available_gib"],failed=batch["failed"],pending=batch["pending"],
            completed=batch["completed"],platform=batch["platform"],python=batch["python"])
    return result


def markdown(result):
    a=result["audit"]; t=result["observed_time"]
    lines=["# F/G 已完成候选汇总","",f"读取 {a['completed_records']}/{a['expected_records']} 条结果；核对问题 {len(a['issues'])} 项。",
        "", f"每条结果的后端种子数为 {t['backend_seed_counts']}。十四候选齐备才计算后端选优、候选中位数与随机平局 CPCM 选优；单候选小批仅列该候选的表现。",
        "", "层公式采用 E fixed：prefill = 2 LN + 8 MHA + proj + FFN；decode = LN + 8 MHA + proj + FFN，其中 decode FFN 已含一次 LN。MHA 微图八头、总层乘八，共六十四头。",
        "", "下表为一个 Die 组上的单层组装时间，单位微秒；请求时间仍固定 KV 长度 2048。", "",
        "| 配置 | 方法 | 候选处理 | TTFT / μs | TPOT / μs |", "|---|---|---|---:|---:|"]
    for row in result["layers"]:
        lines.append(f"| {row['config_id']} | {row['method']} | {row['protocol']} | {row['TTFT_us']:.3f} | {row['TPOT_us']:.3f} |")
    lines += ["",f"保存结果累计耗时 {t['result_process_hours']:.2f} 进程小时，单进程最高峰值 {t['max_peak_rss_gb']:.3f} GiB。"]
    if "batch_wall_seconds" in t:
        lines += [f"实际批次用时 {t['batch_wall_seconds']/60:.2f} 分钟，平均活跃进程 {t['mean_active_workers']:.1f}，最高同时 {t['max_observed_active']}；采样最低可用内存 {t['min_available_gib']:.2f} GiB。"]
    p=result.get("search_projection")
    if p:
        lines += ["",f"按已测类别估计，剩余搜索约 {p['remaining_process_hours']:.2f} 进程小时；按 {p['workers']} 并行、长任务优先排队约 {p['ideal_longest_first_hours']:.2f} 小时，尚未测得的搜索有 {p['remaining_jobs_unmeasured']} 个。修复、配对单跑和联合实验另计；调度还需加入任务依赖与实测内存。"]
    lines += ["", "工作量与取整量由保存快照解析：有效乘加量为各 TensorCore 事件 2×M×K×N 之和；取整量应用原 128×64 grain。matmac 的额外偏置加法未计入该乘加量。归约单列 matadd/redsum/redmax 的事件与输入元素。通信、路径和事件图指标来自运行记录。", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--run-out",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--full-manifest",type=Path)
    parser.add_argument("--batch",type=Path)
    parser.add_argument("--workers",type=int,default=128)
    parser.add_argument("--allow-incomplete",action="store_true")
    parser.add_argument("--baseline-table",type=Path,default=ROOT/"e2e/e/tables/operator_table.csv")
    parser.add_argument("--skip-snapshots",action="store_true")
    args=parser.parse_args()
    specs=read(args.manifest)
    audit,records=check(specs,args.run_out,args.allow_incomplete)
    candidates_=candidate_rows(records)
    operators=operator_rows(candidates_)
    if args.baseline_table.exists(): operators+=baseline_operators(args.baseline_table)
    layers=layer_rows(operators)
    work,workchecks=([],[]) if args.skip_snapshots else workload_rows(specs,records,args.run_out)
    result=dict(audit=audit,observed_time=observed_time(args.run_out,records,args.batch),
                sources=dict(manifest=str(args.manifest),run_output=str(args.run_out),baseline_table=str(args.baseline_table)),
                operators=operators,layers=layers,workloads=work,workload_checks=workchecks,
                formula=dict(prefill="2 LN + 8 MHA + proj + FFN",decode="LN + 8 MHA + proj + FFN",
                             scope="one layer on one Die group",tick_seconds=1e-9))
    if args.full_manifest: result["search_projection"]=estimate(read(args.full_manifest),records,args.workers)
    args.out.mkdir(parents=True,exist_ok=True)
    for name,rows in (("candidates",candidates_),("operators",operators),("layers",layers),("workload",work)):
        write_csv(args.out/(name+".csv"),rows)
    write_json(args.out/"summary.json",result)
    (args.out/"SUMMARY.md").write_text(markdown(result),encoding="utf-8")
    print(json.dumps(dict(completed=audit["completed_records"],issues=audit["issues"],
                         workload_checks=workchecks,layers=len(layers)),ensure_ascii=False))
    return bool(audit["issues"] or any(not r["effective_tensor_work_equal_across_configurations"] for r in workchecks))


if __name__=="__main__":
    raise SystemExit(main())
