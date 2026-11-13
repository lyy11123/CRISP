"""Read saved F/G states and SA rows; no search or backend simulation."""

import argparse
import ast
from collections import Counter, defaultdict
import configparser
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics

import e_gemini_analysis as static
import f_config

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("FACE_proj", "F_face_2x2_fixed_d2d1", "pf_proj"),
    ("G23_ffn", "G_busybarn_2x3_growth_d2d1", "pf_ffn"),
    ("G33_ffn", "G_busybarn_3x3_growth_d2d1", "pf_ffn"),
)
SEEDS = (123, 1, 2, 3, 4)
TIME_KEYS = ("t_pre", "t_move", "t_target", "t_energy", "t_bcast", "t_revert", "t_snapshot")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else v
                         for k, v in row.items()} for row in rows)
    temp.replace(path)
    write(path.with_suffix(".json"), rows)


def service_units(spec):
    classes = static.pickle_types()
    parser = configparser.ConfigParser()
    parser.read_string(f_config.config_text(spec))
    cfg = {section: {k: ast.literal_eval(v) if v[:1].isdigit() or v.startswith("[") else v
                     for k, v in parser[section].items()} for section in parser.sections()}
    units = {k: cls((0, 0, 0), cfg[k + "_cfg"])
             for k, cls in zip(("tensorcore", "vectorunit"), classes)}
    frequency = {k: cfg["time_unit"]["frequency"] *
                 (cfg["tensorcore_cfg"]["fg_compute_scale"] if k == "tensorcore" else 1)
                 for k in units}
    return units, frequency


def duration(event, units, frequency):
    return int(units[event.comp_device].working_time(event.comp_datashape, event.comp_type,
                                                    frequency[event.comp_device]))


def diagnose(run_out, dest):
    static.pickle_types()
    summaries, loads, events, sources = [], [], [], []
    checks = []
    for case, cid, op in CASES:
        pre = read(run_out / "results" / f"{cid}__pre_{op}_A3h.json")
        l1 = read(run_out / "results" / f"{cid}__l1_{op}_A3h_MF6.json")
        stop = read(run_out / "results" / f"{cid}__stop_{op}_MF6_300.json")
        fault = read(run_out / "faults" / f"{cid}.json")["faults"]["MF6"]
        dead = tuple(fault["failed_pes"][0])
        spec = pre["spec"]
        nr, nc = map(int, spec["shape"].split("x"))
        units, frequency = service_units(spec)
        state_inputs = [("healthy", None, pre["exports"]["healthy"], pre["T"], pre["backend_mean"]),
                        ("R0", None, l1["exports"]["R0"], l1["T_R0"], l1["R0_backend_mean"]),
                        ("L1", None, l1["exports"]["L1"], l1["T_L1"], l1["L1_backend_mean"])]
        assert stop["level"].startswith("s"), stop
        for seed in SEEDS:
            row = read(run_out / "results" / f"{cid}__rep_{op}_MF6_{stop['level']}_r{seed}.json")
            assert not row.get("skipped"), row
            state_inputs.append(("EIR", seed, row["exports"]["final"], row["T"], row["backend_mean"]))
        healthy_shapes = None
        for state, seed, stored_path, that, backend in state_inputs:
            source = ROOT / stored_path
            exported = static.read_pickle(source)
            graph = pickle.loads(exported["events_blob"])
            by_unit, blocks = defaultdict(Counter), defaultdict(list)
            identity = dict(case=case, config_id=cid, op=op, state=state, seed=seed,
                            failed_pe=dead, source_graph=stored_path)
            shapes = Counter()
            for tag, event in graph.items():
                if event.event_type != "computation":
                    continue
                service = duration(event, units, frequency)
                location = tuple(event.comp_location)
                by_unit[event.comp_device][location] += service
                blocks[event.comp_device].append(service)
                shapes[event.comp_device, event.comp_type, str(event.comp_datashape)] += 1
                events.append(dict(identity, tag=tag, name=event.event_name, device=event.comp_device,
                                   kind=event.comp_type, shape=event.comp_datashape, location=location,
                                   service_ns=service, original_orphan=(state == "healthy" and location[:2] == dead)))
                if state != "healthy":
                    assert location[:2] != dead, (case, state, seed, tag)
            if state == "healthy":
                healthy_shapes = shapes
            else:
                assert shapes == healthy_shapes, (case, state, seed, "computation changed")
            live = nr * nc * 16 - (state != "healthy")
            for device, times in blocks.items():
                equal = len(set(times)) == 1
                basic = max(max(times), sum(times) / live)
                count_bound = math.ceil(len(times) / live) * min(times)
                lower = max(basic, count_bound)
                peak = max(by_unit[device].values())
                assert peak >= lower and backend >= peak
                summaries.append(dict(identity, device=device, live_units=live, blocks=len(times),
                                      duration_classes=len(set(times)), equal_duration=equal,
                                      min_block_ns=min(times), max_block_ns=max(times), sum_service_ns=sum(times),
                                      basic_lower_bound_ns=basic, count_lower_bound_ns=count_bound,
                                      rigid_lower_bound_ns=lower, peak_ns=peak, peak_over_bound=peak/lower,
                                      T_hat_ns=that, backend_mean_ns=backend,
                                      EIR_selected_level=stop["level"], EIR_representative_seed=stop["seed"],
                                      same_fault_FULL="not run"))
                for die in range(nr*nc):
                    for core in range(16):
                        if state != "healthy" and (die,core) == dead:
                            continue
                        loads.append(dict(identity, device=device, location=(die,core,0),
                                          service_ns=by_unit[device][die,core,0]))
            sources.append(dict(identity, sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                event_count=len(graph), T_hat_ns=that))
        checks.append(dict(case=case, state_graphs=8, EIR_seeds=list(SEEDS),
                           computation_shapes_preserved=True, no_computation_on_failed_PE=True,
                           service_peak_above_lower_bound=True, backend_above_compute_peak=True))
    tables = dest / "tables"
    table(tables / "mf6_summary.csv", summaries)
    table(tables / "mf6_loads.csv", loads)
    table(tables / "mf6_events.csv", events)
    write(dest / "evidence" / "mf6_sources.json", sources)
    write(dest / "evidence" / "mf6_checks.json", dict(status="passed", cases=checks,
          semantics="actual module working_time; no search or backend evaluation"))
    return summaries


def sa_cost(run_out, dest):
    rows = []
    for phase in ("prefill", "decode"):
        paths = sorted((run_out / "raw").glob(f"G_busybarn_3x3_growth_d2d1__healthy_healthy_{phase}_mha_A3h_*.pkl.gz"))
        assert len(paths) == 14
        for path in paths:
            raw = static.read_pickle(path)
            timings = {key: sum(step.get(key, 0) for step in raw["rows"]) for key in TIME_KEYS}
            accounted = sum(timings.values())
            rows.append(dict(candidate_id=path.name.removesuffix(".pkl.gz"), phase=phase,
                             source_raw=path.relative_to(ROOT).as_posix(), steps=len(raw["rows"]),
                             accepted=sum(step["accepted"] for step in raw["rows"]),
                             rejected=sum(not step["accepted"] for step in raw["rows"]),
                             targeted=sum(step["targeted"] for step in raw["rows"]),
                             check_revert_n=raw["check_revert"]["n"], **timings,
                             sa_wall_seconds=raw["sa_wall"], accounted_seconds=accounted,
                             unassigned_seconds=raw["sa_wall"]-accounted,
                             check_revert=raw["check_revert"]))
    table(dest / "tables" / "sa_cost_breakdown.csv", rows)
    return rows


def report(dest, mf6, sa):
    text = ["# MF6 固定计算块诊断", "", "2026-11-13，Codex。只读保存图与原始评估；没有运行新搜索或后端仿真。", "",
            "三例均沿用自身 MF6：一颗 PE 的 TensorCore/VectorUnit 不可用，路由器保留。EIR 在 s100 因相邻档改进不足停止；以下覆盖五个修复种子。", "",
            "| 案例 | 健康Tensor块数 | 失效PE | R0峰值ns | EIR五种子峰值范围ns | 有效刚性下界ns | 峰值/下界 |",
            "|---|---:|---|---:|---|---:|---:|"]
    for case,_,_ in CASES:
        subset = [r for r in mf6 if r["case"] == case and r["device"] == "tensorcore"]
        r0 = next(r for r in subset if r["state"] == "R0")
        repaired = [r["peak_ns"] for r in subset if r["state"] == "EIR"]
        text.append(f"| {case} | {r0['blocks']} | {r0['failed_pe']} | {r0['peak_ns']:,} | {min(repaired):,}–{max(repaired):,} | {r0['rigid_lower_bound_ns']:,.0f} | {r0['peak_over_bound']:.6f} |")
    text += ["", "FACE投影的Tensor块等时，使用 ceil(K/P_live)×c。两个FFN因序列片段取整包含两种接近的块时长，分别报告基本下界 max(max c, sum c/P_live)，以及计数下界 ceil(K/P_live)×min c，取两者较大值。计数下界的理由是：K块分给P_live个核，至少一个核有ceil(K/P_live)块，每块时长不低于min c。这是有效下界，没有假定这些不等时块完全相同。", "",
             "三例Tensor负载已经达到或很接近该刚性下界。该结论约束计算负载；实际后端总延迟仍包含依赖、输入输出和链路争用。Vector负载单独保留在完整表中，未假定向量块也平衡。三个MF6均没有同条件FULL，PF4 FULL不用于此处。", "",
             "原孤儿来自健康图，FACE为1 Tensor+1 Vector；2×3 FFN为2 Tensor+2 Vector（含16项部分和归约）；3×3 FFN为2 Tensor+1 Vector。Split须保留原归约，向量孤儿按原粒度迁移。", "",
             "## 大图SA时间", "", "| 相位 | 候选数 | SA中位秒 | 已记录分项中位秒 | 未归属中位秒 | 回滚检查中位次数 |", "|---|---:|---:|---:|---:|---:|"]
    for phase in ("prefill", "decode"):
        selected = [r for r in sa if r["phase"] == phase]
        med = lambda key: statistics.median(r[key] for r in selected)
        text.append(f"| {phase} MHA | {len(selected)} | {med('sa_wall_seconds'):.3f} | {med('accounted_seconds'):.3f} | {med('unassigned_seconds'):.3f} | {med('check_revert_n'):.1f} |")
    text += ["", "普通t_energy不包含拒绝后为验证精确回滚再次执行的energy；剩余时间还含其它循环工作。这里仅列sa_wall减七类记录时间，不把差额全归到某个热点。WP2用短轨迹单独测量。上述均是先前并行环境的主机墙钟，不是芯片时间或J阶段串行计时。", "",
             "## 文件", "", "- [逐状态摘要](tables/mf6_summary.csv)、[逐核负载](tables/mf6_loads.csv)、[计算事件](tables/mf6_events.csv)",
             "- [原始状态与SHA-256](evidence/mf6_sources.json)、[检查](evidence/mf6_checks.json)",
             "- [SA逐候选分项](tables/sa_cost_breakdown.csv)", "",
             "复算：`python -B e2e/post_fg_diag.py --run-out e2e/fg/out/main --out e2e/post_fg`。"]
    (dest / "MF6_DIAG.md").write_text("\n".join(text)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-out", type=Path, default=ROOT/"e2e/fg/out/main")
    parser.add_argument("--out", type=Path, default=ROOT/"e2e/post_fg")
    args = parser.parse_args()
    args.run_out = args.run_out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    mf6 = diagnose(args.run_out, args.out)
    sa = sa_cost(args.run_out, args.out)
    report(args.out, mf6, sa)
    print(json.dumps(dict(MF6_state_resource_rows=len(mf6), SA_candidates=len(sa), status="passed")))


if __name__ == "__main__":
    main()
