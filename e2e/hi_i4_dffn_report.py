"""Read the separate I4 WATOS decode-FFN PF4 batch and summarize coverage."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

import hi_h4_report as shared
from hi_split_supplement import read,write

ROOT,HERE=shared.ROOT,shared.HERE
KINDS=dict(fault_selection=1,fault_select=1,fault_pre=1,fault_l1=1,fault_stop=4,fault_rep=15)
MANIFEST=HERE/"manifest/i4_watos_dffn.json"
INPUTS=HERE/"manifest/i4_watos_dffn_healthy_inputs.json"


def check_plan(manifest):
    jobs=manifest["jobs"]
    assert len(jobs)==23 and Counter(j["kind"] for j in jobs)==KINDS
    for job in jobs:
        spec=job["spec"]
        assert spec["model"]=="qwen3moe" and spec["hardware_profile"]=="watos"
        assert spec.get("phase_op") in (None,"dc_ffn") and spec.get("fault_id") in (None,"PF4")
        assert spec.get("start_arm") in (None,"A3h")
    inputs=read(INPUTS)["candidates"]
    wanted={name for j in jobs if j["kind"]=="fault_selection" for name in j["spec"]["candidate_ids"]}
    assert len(inputs)==28 and set(inputs)==wanted
    records,index={},{}
    for name,source in inputs.items():
        for field in ("record","raw","snapshot"):
            assert shared.sha(shared.path(source[field]))==source["sha256"][field]
        record=read(shared.path(source["record"]))
        assert record["status"]=="ok" and record["candidate_id"]==name and record["model"]=="qwen3moe"
        assert record["hardware_profile"]=="watos" and record["phase"]=="decode" and record["method"]=="A3h"
        records[name]=record
        index[name]=dict(source,healthy_T_hat=record["T_hat"],backend_mean_ns=statistics.mean(r["cycles"] for r in record["backend"]))
    assert Counter(r["op"] for r in records.values())==dict(mha=14,ffn=14)
    return index


def render(audit,units,groups,rows,timing):
    text=["# I4：WATOS decode FFN 补充结果","",
        f"独立批次已完成 {audit['completed_jobs']}/23 作业；实际 L2 轨迹 {audit['actual_repair_trajectories']} 条，停止后复用 {audit['reused_repair_jobs']} 条，后端样本 {audit['backend_samples']} 个。",
        "本批只补充 Qwen WATOS decode FFN 的一个 A3h PF4 点，原 I4-light 的 201 作业、九个 PF4 点及结果保持独立。28 个既有候选包括 FFN 14 个候选和 MHA 14 个平台参照；MHA 不重新搜索或运行后端。固定故障为 `(die0,core11) ↔ (die1,core8)` 双向连接失效。","",
        "## 结果","",
        "健康、R0、L1 与实际修复状态分别评估后端种子1–10。停止判断使用 CPCM；进入 L2 时用该档五个修复种子的平均 CPCM 决定是否停止，再按后端均值取五条轨迹的中位代表。没有用后端最好种子替代既定代表。","",
        "| 健康 ms | R0 ms | L1 ms | EIR ms | R0/健康−1 | EIR/健康−1 | EIR/R0−1 | 停止层级/原因 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in units:
        final=f"{r['EIR_backend_mean_ns']/1e6:.6f}" if r["EIR_complete"] else "待完成"
        rel=[f"{r[k]:+.3%}" if r["EIR_complete"] else "待完成" for k in ("EIR_relative_to_healthy","EIR_relative_to_R0")]
        stop=r["EIR_level"]+"/"+r["EIR_reason"] if r["EIR_complete"] else "进行中"
        text.append(f"| {r['healthy_backend_mean_ns']/1e6:.6f} | {r['R0_backend_mean_ns']/1e6:.6f} | {r['L1_backend_mean_ns']/1e6:.6f} | {final} | {r['R0_relative_to_healthy']:+.3%} | {' | '.join(rel)} | {stop} |")
        if r["EIR_complete"]:
            text += ["",f"最终代表来自 `{r['EIR_source_job']}`；代表自身 CPCM 为 {r['EIR_selected_T_hat_ns']:,.3f} ns，停止决策的 CPCM 均值为 {r['EIR_stop_mean_T_hat_ns']:,.3f} ns。"]
            migration=r["EIR_migration"]
            text += [f"L1 将后端均值从 R0 的 {r['R0_backend_mean_ns']/1e6:.6f} ms 降至 {r['L1_backend_mean_ns']/1e6:.6f} ms（{r['L1_relative_to_R0']:+.2%}）；继续局部重映射后为 {r['EIR_backend_mean_ns']/1e6:.6f} ms，相对 R0 变化 {r['EIR_relative_to_R0']:+.2%}。最终仍比健康高 {r['EIR_relative_to_healthy']:.2%}，平均 CPCM 也高 {r['EIR_stop_T_hat_relative_to_healthy']:.2%}，因此是用完300步预算后停止，未恢复到健康2%以内。",
                f"最终代表为修复种子 {r['EIR_repair_seed']}，搬动 {migration['moved_behas']}/{migration['n_behas']} 个行为（{migration['moved_behas_share']:.2%}），唯一迁移块 {migration['unique_blocks']} 个、{migration['unique_block_bytes']/2**20:.3f} MiB。"]
    text += ["", "## 各预算与实际评估量","",
        "修复种子固定为 {123,1,2,3,4}，30/100/300 步独立从 L1 启动。回到健康估值2%以内停止；100/300档同时检查相邻两档平均 CPCM 改善是否小于2%，300档为预算终点。","",
        "| 预算 | 实际轨迹 | 复用 | 待处理 | 五轨迹平均 CPCM ns | 五个后端均值的中位数 ns |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in groups:
        text.append(f"| {r['budget']} | {r['actual_trajectories']} | {r['reused']} | {r['pending']} | {r.get('T_hat_mean_ns','—')} | {r.get('trajectory_backend_mean_median_ns','—')} |")
    text += ["", f"最多15条修复轨迹、180次后端：健康10次 + R0/L1各10次 + 15×10修复评估。当前实际评估为 {audit['backend_samples']} 次；停止后复用不增加后端。六条控制作业为一次选优、一次固定故障设定、四次停止/状态传递。", "",
        "## 分项时间与迁移","",
        "| 类别 | 数量 | 后端调用秒 | 后端准备秒 | 校准秒 | 搜索秒 | 恢复与最终 CPCM 秒 | 其他秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in ("fault_pre","fault_l1","fault_rep"):
        selected=[r for r in timing if r["kind"]==kind]
        if not selected:continue
        total=lambda k:sum(r[k] for r in selected)
        text.append(f"| {kind} | {len(selected)} | {total('backend_call_seconds'):.3f} | {total('backend_total_seconds')-total('backend_call_seconds'):.3f} | {total('calibration_seconds'):.3f} | {total('search_seconds')+total('L1_search_seconds'):.3f} | {total('restore_best_seconds')+total('final_cpcm_seconds'):.3f} | {total('other_seconds')+total('R0_prepare_seconds'):.3f} |")
    runtime=audit["execution"]
    text += ["",f"累计作业耗时 {runtime['completed_job_process_hours']:.6f} 进程小时。L2 中继承的 t_R0/t_L1 不重复加到当前作业。迁移块、行为变化、DDR源变化与每个状态10样本全部保存在明细表中。"]
    if "wall_seconds" in runtime:text.append(f"从 {runtime['first_start_utc']} 到 {runtime['completed_utc']}，经过 {runtime['wall_seconds']/60:.3f} 分钟。该批与其他实验共享机器，进程小时与经过时间分列。")
    if "initial_estimate" in audit and audit["status"]=="complete":
        estimate=audit["initial_estimate"]
        text.append(f"初始安排 {estimate['planning_minutes'][0]:.2f}–{estimate['planning_minutes'][1]:.2f} 分钟，实际{runtime['wall_seconds']/60:.3f}分钟落在该范围内。初始累计作业估计为 {estimate['estimated_upper_process_hours']:.6f} 进程小时，实际 {runtime['completed_job_process_hours']:.6f} 进程小时；各预算均执行五条轨迹，没有停档节省。")
    text += ["", "## 与 I4-light 合并后的覆盖","",
        "本批独立完成后，与九个 I4-light PF4 点合计覆盖10/12个预定 A3h PF4 点；两种硬件的 prefill FFN 仍待后续。只有已有完整主硬件 decode 四算子可解析组装，WATOS 缺 LN/projection 故障结果，不组装完整 WATOS decode。没有完整 Qwen prefill、请求时延或 MF6/Split 结论。", "",
        "当前实际合并覆盖由 `I4_PF4_COVERAGE.md` 和 `tables/i4_pf4_coverage.csv` 单独列出。独立批次明细均使用 `tables/i4_watos_dffn_*.csv` 前缀，核对记录为 `evidence/i4_watos_dffn_completion_audit.json`。"]
    return text


def coverage(destination,audit,units):
    light_audit_path=destination/"evidence/i4_completion_audit.json"
    light_unit_path=destination/"tables/i4_units.csv"
    light,records=None,[]
    if light_audit_path.exists() and light_unit_path.exists():
        light=read(light_audit_path)
        assert not light["smoke_only"]
        with light_unit_path.open(encoding="utf-8-sig",newline="") as stream:
            for row in csv.DictReader(stream):
                if row["start_arm"]!="A3h":continue
                records.append(dict(batch="i4_light",hardware_profile=row["hardware_profile"],phase_op=row["phase_op"],
                    complete=row["EIR_complete"]=="True",EIR_source_job=row.get("EIR_source_job"),
                    EIR_backend_mean_ns=float(row["EIR_backend_mean_ns"]) if row.get("EIR_backend_mean_ns") else None))
    for r in units:
        records.append(dict(batch="i4_watos_dffn",hardware_profile=r["hardware_profile"],phase_op=r["phase_op"],
            complete=r["EIR_complete"],EIR_source_job=r.get("EIR_source_job"),EIR_backend_mean_ns=r.get("EIR_backend_mean_ns")))
    by_key={(r["hardware_profile"],r["phase_op"]):r for r in records}
    assert len(by_key)==len(records)
    expected={("busybarn",p+"_"+o) for p in ("pf","dc") for o in ("ln","mha","proj","ffn")}|{
        ("watos",p+"_"+o) for p in ("pf","dc") for o in ("mha","ffn")}
    rows=[]
    for hardware,op in sorted(expected):
        r=by_key.get((hardware,op))
        rows.append(dict(hardware_profile=hardware,phase_op=op,**({k:v for k,v in r.items() if k not in ("hardware_profile","phase_op")} if r else dict(batch="planned_later",complete=False))))
    complete=sum(r["complete"] for r in rows)
    shared.common.table(destination/"tables/i4_pf4_coverage.csv",rows)
    note=["# I4 PF4 完成范围","",f"当前本机正式结果覆盖 {complete}/12 个预定 A3h PF4 点。两个独立批次的完成数和计时分别列出。","",
        "| 硬件 | 算子 | 来源 | 最终 EIR 状态已齐 |","|---|---|---|---|"]
    for r in rows:note.append(f"| {r['hardware_profile']} | {r['phase_op']} | {r['batch']} | {'是' if r['complete'] else '否'} |")
    note += ["",f"I4-light 本机作业数：{light['completed_jobs'] if light else 0}/201；WATOS decode FFN：{audit['completed_jobs']}/23。A1-R0 四点属于 I4-light 对照，未计入12个 A3h PF4点。两个prefill FFN、MF6与Split仍未补齐；本页不生成完整prefill、请求或WATOS整层结果。"]
    if light:
        note += [f"两批合计 {light['completed_jobs']+audit['completed_jobs']}/224 个作业，实际 L2 {light['actual_repair_trajectories']+audit['actual_repair_trajectories']} 条，停止后复用 {light['reused_repair_jobs']+audit['reused_repair_jobs']} 条；后端样本共 {light['backend_samples']+audit['backend_samples']} 个。两个独立队列有时间重叠，不能相加其经过时间。"]
    (destination/"I4_PF4_COVERAGE.md").write_text("\n".join(note)+"\n",encoding="utf-8")


def run(args):
    manifest=read(args.manifest);index=check_plan(manifest)
    if args.check:
        print(json.dumps(dict(status="plan_and_inputs_passed",jobs=23,healthy_inputs=28,A3h_PF4_points=1,
            maximum_repair_trajectories=15,maximum_backend_samples=180)));return
    jobs,present,missing,states,samples,timing,checks=shared.collect(args.out,manifest,args.progress,False,"qwen3moe",KINDS)
    stops,groups,trajectories=shared.stop_tables(jobs,present)
    units=shared.unit_tables(jobs,present,states,stops,index)
    rows=shared.job_rows(jobs,present,timing);runtime=shared.execution(args.out,present,False)
    actual=sum(r["status"]=="ok" for r in trajectories);reused=sum(r["status"]=="reused" for r in trajectories)
    finish=runtime["controller_finish"]
    complete=not missing and finish is not None and finish["rc"]==0 and finish["completed"]==23 and finish["expected"]==23
    audit=dict(status="complete" if complete else "in_progress",completed_jobs=len(present),expected_jobs=23,
        missing_jobs=len(missing),missing=missing,actual_repair_trajectories=actual,reused_repair_jobs=reused,
        backend_samples=len(samples),A3h_PF4_units=sum(r["EIR_complete"] for r in units),healthy_input_candidates=28,
        new_healthy_searches=0,additional_MHA_backend_calls=0,maximum_backend_samples=180,
        kinds=dict(Counter(v["spec"]["kind"] for v in present.values())),stop_records=len(stops),
        faults=["PF4"],hardware="watos",phase_op="dc_ffn",execution=runtime,input_checks=checks,
        manifest_sha256=shared.sha(args.manifest),healthy_input_index_sha256=shared.sha(INPUTS),
        report_source_sha256=shared.sha(__file__),shared_report_source_sha256=shared.sha(Path(shared.__file__)))
    plan_path=HERE/"evidence/i4_watos_dffn_plan.json"
    estimate=read(plan_path)
    audit["initial_estimate"]={k:estimate[k] for k in ("planning_minutes","estimated_upper_process_hours","estimate_basis")}
    audit["initial_estimate"].update(source=plan_path.relative_to(ROOT).as_posix(),source_sha256=shared.sha(plan_path))
    assert len(samples)==sum(r["actual_backend_calls"] for r in rows if r["status"]!="pending")
    if not args.progress:
        assert complete and actual+reused==15 and len(stops)==4 and len(units)==1 and audit["A3h_PF4_units"]==1
        assert len(samples)==30+actual*10
    tables=dict(jobs=rows,states=states,backend_samples=samples,timing=timing,repair_trajectories=trajectories,
        budget_groups=groups,stop_decisions=stops,units=units)
    for name,values in tables.items():shared.common.table(args.destination/"tables"/("i4_watos_dffn_"+name+".csv"),values)
    write(args.destination/"evidence/i4_watos_dffn_completion_audit.json",audit)
    (args.destination/"I4_WATOS_DFFN_REPORT.md").write_text("\n".join(render(audit,units,groups,rows,timing))+"\n",encoding="utf-8")
    coverage(args.destination,audit,units)
    print(json.dumps({k:audit[k] for k in ("status","completed_jobs","actual_repair_trajectories","reused_repair_jobs","backend_samples")}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"out/i4_watos_dffn")
    p.add_argument("--manifest",type=Path,default=MANIFEST)
    p.add_argument("--destination",type=Path,default=HERE)
    p.add_argument("--progress",action="store_true")
    p.add_argument("--check",action="store_true")
    run(p.parse_args())


if __name__=="__main__":main()
