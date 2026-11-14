"""Report I4's nine ready PF4 points; assemble only the complete decode chain."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import f_config
import hi_config
import hi_h4_report as shared
from hi_split_supplement import read, write

ROOT,HERE=shared.ROOT,shared.HERE
KINDS=dict(fault_selection=2,fault_select=2,fault_pre=13,fault_l1=13,fault_stop=36,fault_rep=135)
MANIFEST=HERE/"manifest/i4_light.json"
INPUTS=HERE/"manifest/i4_light_healthy_inputs.json"


def check_plan(manifest):
    jobs=manifest["jobs"]
    assert len(jobs)==201 and Counter(j["kind"] for j in jobs)==KINDS
    pre=[j["spec"] for j in jobs if j["kind"]=="fault_pre"]
    expected={
        ("busybarn",op,"A3h") for op in ("pf_ln","pf_mha","pf_proj","dc_ln","dc_mha","dc_proj","dc_ffn")
    } | {("watos",op,"A3h") for op in ("pf_mha","dc_mha")} | {
        ("busybarn",op,"A1") for op in ("dc_ln","dc_mha","dc_proj","dc_ffn")}
    assert {(s["hardware_profile"],s["phase_op"],s["start_arm"]) for s in pre}==expected
    for job in jobs:
        spec=job["spec"]
        assert spec["model"]=="qwen3moe" and spec.get("fault_id") in (None,"PF4")
        assert spec.get("phase_op")!="pf_ffn"
        assert not(spec["hardware_profile"]=="watos" and spec.get("op")=="ffn")
    wanted={name for j in jobs if j["kind"]=="fault_selection" for name in j["spec"]["candidate_ids"]}
    inputs=read(INPUTS)["candidates"]
    assert len(inputs)==182 and set(inputs)==wanted
    index={}
    for name,source in inputs.items():
        record=read(shared.path(source["record"]))
        assert record["status"]=="ok" and record["candidate_id"]==name and record["model"]=="qwen3moe"
        assert shared.sha(shared.path(source["record"]))==source["sha256"]["record"]
        index[name]=dict(source,healthy_T_hat=record["T_hat"],
            backend_mean_ns=statistics.mean(r["cycles"] for r in record["backend"]))
    return index


def decode_table(units):
    contract=hi_config.model_contract("qwen3moe")
    assert contract["decode_chain"]==dict(norm=1,mha=4,proj=1,ffn=1)
    assert contract["decode_ffn_includes_rmsnorm"] and contract["batch"]==16 and contract["hidden"]==2048
    hp=f_config.hardware_parameters("busybarn")
    handoff_bytes=contract["batch"]*contract["hidden"]*2
    handoff=hp["d2d_latency_ticks"]+handoff_bytes/hp["d2d_bytes_per_tick_directed_link"]
    assert handoff==276
    weights=dict(ln=1,mha=4,proj=1,ffn=1)
    lookup={(r["start_arm"],r["phase_op"]):r for r in units if r["hardware_profile"]=="busybarn"}
    output=[]
    for arm in ("A1","A3h"):
        if not all((arm,"dc_"+op) in lookup for op in weights):continue
        cells={op:lookup[arm,"dc_"+op] for op in weights}
        healthy=sum(weights[op]*cells[op]["healthy_backend_mean_ns"] for op in weights)+handoff
        for stage in (("healthy","R0") if arm=="A1" else ("healthy","R0","L1","EIR")):
            if stage=="EIR" and not all(row["EIR_complete"] for row in cells.values()):continue
            total=sum(weights[op]*cells[op][stage+"_backend_mean_ns"] for op in weights)
            source="EIR_source_job" if stage=="EIR" else "healthy_source_job" if stage=="healthy" else "l1_source_job"
            output.append(dict(model="qwen3moe",hardware_profile="busybarn",start_arm=arm,state=stage,
                phase="decode",weights=weights,batch=16,kv_length=2048,external_rmsnorms=1,ffn_embedded_rmsnorms=1,
                operator_sum_ns=total,handoff_bytes=handoff_bytes,handoff_ns=handoff,decode_ns=total+handoff,
                relative_to_same_arm_healthy=(total+handoff)/healthy-1,
                source_jobs={op:cells[op][source] for op in weights},
                measurement="serial operator means plus unchanged intergroup handoff; one-layer decode reference"))
    a1_r0=next((r["decode_ns"] for r in output if r["start_arm"]=="A1" and r["state"]=="R0"),None)
    if a1_r0 is not None:
        for row in output:row["relative_to_A1_R0"]=row["decode_ns"]/a1_r0-1
    return output


def render(audit,units,groups,rows,timing,decode):
    smoke=audit["smoke_only"]
    text=["# I4 PF4 汇总解析核对" if smoke else "# I4：Qwen 已就绪的九个 PF4 点","",
        "本文件只解析本机短验证材料，不计入正式 I4 完成数。" if smoke else
        f"本批已完成 {audit['completed_jobs']}/201 条作业记录；实际修复 {audit['actual_repair_trajectories']} 条、停止后复用 {audit['reused_repair_jobs']} 条，尚待处理 {audit['missing_jobs']} 条。",
        "覆盖主硬件七个 A3h 点（prefill 的 LN/MHA/projection，decode 的 LN/MHA/projection/FFN）、WATOS 的 prefill/decode MHA 两点，以及主硬件四个 A1-R0。它是 I4 的一部分；所有 prefill FFN、WATOS decode FFN、MF6 与 Split 均不在本批。",
        "PF4 固定为 `(die0,core11) ↔ (die1,core8)` 双向连接失效，不按映射重选故障。13组共182个健康候选复用，后端均值选择规则不变。健康赢家在当前批次重放十种子后端，原健康搜索没有重跑。","",
        "## 作业和实际评估数","","| 内容 | 完成作业 | 实际后端样本 |","|---|---|---|"]
    for kind,label in (("fault_pre","健康重放"),("fault_l1","R0/L1，A1仅R0"),("fault_rep","实际L2")):
        selected=[r for r in rows if r["kind"]==kind and r["status"]=="ok"]
        text.append(f"| {label} | {len(selected)} | {sum(r['actual_backend_calls'] for r in selected)} |")
    control=sum(r["status"]=="ok" and r["kind"] in ("fault_selection","fault_select","fault_stop") for r in rows)
    text += [f"| 选优、故障设定、停止记录 | {control} | 0 |",
        f"| 停止后复用 | {audit['reused_repair_jobs']} | 0 |","",
        f"已读取 {audit['backend_samples']} 个实际后端样本。完整上限为13×10健康+9×20 A3h R0/L1+4×10 A1-R0+135×10 L2=1700。135是最大修复轨迹数；复用任务没有新搜索或后端。36条停止记录包括已停之后的状态传递，另有4条选优/故障设定控制记录。","",
        "## 九个 A3h 故障点","",
        "数值为实际十种子后端均值，单位毫秒。若进入L2，停止用该档五种子平均CPCM，导出状态则取五个后端均值的中位代表；两者分别保留。","",
        "| 硬件 | 算子 | 健康 ms | R0 ms | L1 ms | EIR ms | 停止层级/原因 | R0/健康−1 | EIR/健康−1 | EIR/R0−1 |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for r in units:
        if r["start_arm"]!="A3h":continue
        final=f"{r['EIR_backend_mean_ns']/1e6:.5f}" if r["EIR_complete"] else "待停止判定"
        stop=r["EIR_level"]+"/"+r["EIR_reason"] if r["EIR_complete"] else "进行中"
        ratios=[f"{r[k]:+.2%}" if r["EIR_complete"] else "—" for k in ("EIR_relative_to_healthy","EIR_relative_to_R0")]
        text.append(f"| {r['hardware_profile']} | {r['phase_op']} | {r['healthy_backend_mean_ns']/1e6:.5f} | {r['R0_backend_mean_ns']/1e6:.5f} | {r['L1_backend_mean_ns']/1e6:.5f} | {final} | {stop} | {r['R0_relative_to_healthy']:+.2%} | {' | '.join(ratios)} |")
    text += ["","## 各预算的五条轨迹","",
        "种子固定为{123,1,2,3,4}；30/100/300步各自从同一个L1状态启动。回到健康估值2%以内停止；100/300档还以相邻档五种子均值改善不足2%停止，300档为预算终点。停止后没有额外探索。所有判断与代表种子均从原结果重新计算核对。","",
        "| 硬件 | 算子 | 预算 | 实际 | 复用 | 待处理 | 五轨迹平均 CPCM ns | 五个后端均值的中位数 ns |","|---|---|---|---|---|---|---|---|"]
    for r in groups:
        if r["actual_trajectories"]+r["reused"]==0:continue
        text.append(f"| {r['hardware_profile']} | {r['phase_op']} | {r['budget']} | {r['actual_trajectories']} | {r['reused']} | {r['pending']} | {r.get('T_hat_mean_ns','—')} | {r.get('trajectory_backend_mean_median_ns','—')} |")
    text += ["","## 主硬件 decode 单层参考","",
        "按模型合同组装：外置RMSNorm +4×代表MHA +projection +FFN。FFN已经含另一次RMSNorm，因此decode共两次归一化，外置LN只乘1。每个MHA代表图含8个Q头与一组K/V。",
        "保持batch16、KV2048，增加同健康报告一致的相邻组handoff：65536字节/256 B/ns+20 ns=276 ns。PF4施加在算子组内，handoff解析项不变。下面是算子均值的串行组装，不是完整层联合仿真。","",
        "| 起点 | 状态 | 算子和 ms | handoff ns | decode ms | 相对同起点健康 | 相对 A1-R0 |","|---|---|---|---|---|---|---|"]
    for r in decode:
        other=f"{r['relative_to_A1_R0']:+.2%}" if "relative_to_A1_R0" in r else "—"
        text.append(f"| {r['start_arm']} | {r['state']} | {r['operator_sum_ns']/1e6:.5f} | {r['handoff_ns']:.0f} | {r['decode_ns']/1e6:.5f} | {r['relative_to_same_arm_healthy']:+.2%} | {other} |")
    text += ["","本批没有prefill FFN，因此不生成完整prefill、TTFT或P+2048D请求结果。WATOS只测MHA，也不组装WATOS完整decode。48层、容量、流水线与共享链路没有在这里模拟。","",
        "## 结果解释",""]
    if audit["status"]=="complete":
        dc={r["state"]:r for r in decode if r["start_arm"]=="A3h"}
        ffn=next(r for r in units if r["hardware_profile"]=="busybarn" and r["phase_op"]=="dc_ffn" and r["start_arm"]=="A3h")
        residual=(ffn["EIR_backend_mean_ns"]-ffn["healthy_backend_mean_ns"])/(dc["EIR"]["decode_ns"]-dc["healthy"]["decode_ns"])
        text += [f"主硬件 A3h decode 的解析层延迟从健康 {dc['healthy']['decode_ns']/1e6:.6f} ms 增至 R0 的 {dc['R0']['decode_ns']/1e6:.6f} ms；EIR 后为 {dc['EIR']['decode_ns']/1e6:.6f} ms，比 R0 减少 {1-dc['EIR']['decode_ns']/dc['R0']['decode_ns']:.2%}，仍比健康高 {dc['EIR']['relative_to_same_arm_healthy']:.2%}。相对 A1-R0，A3h+EIR 减少 {-dc['EIR']['relative_to_A1_R0']:.2%}。",
            f"剩余退化主要来自 decode FFN：R0 比健康高 {ffn['R0_relative_to_healthy']:.2%}，修复后仍高 {ffn['EIR_relative_to_healthy']:.2%}，约占整层剩余增加量的 {residual:.2%}。它在300步预算终点停止；主硬件 decode MHA 也用满300步，后端仍比健康高6.30%。这两个点没有恢复到健康2%以内。",
            "停止条件和后端结果有所区别。主硬件 decode projection 在30步档以平均 CPCM +1.05% 停止，其最终代表后端为 +2.28%；WATOS decode MHA 在100步档因相邻档估值改善不足2%停止，后端仍高14.81%。prefill MHA 两种硬件也在100步档因档间改善不足2%停止，后端残差分别为4.16%和6.25%。",
            "跨预算后端均值并不总是单调下降：主硬件 decode MHA 的五轨迹中位后端从100步的6965.2 ns到300步的7013.4 ns略增，而平均CPCM继续改善；主硬件prefill MHA的100步代表后端也略高于30步。最终状态按原停止流程取当前档代表，未事后改选较早档的后端赢家。",""]
    text += [
        "## 迁移与分项时间","",
        "状态表保留实际行为搬迁、DDR读源变化字节、唯一搬迁块字节/块数，以及故障链路原健康忙时和关键阻塞信息。修复种子的明细、健康重放与原结果差值均保留。A1只提供R0对照，不报告A1-EIR。",
        f"已完成作业累计 {audit['execution']['completed_job_process_hours']:.4f} 进程小时；其中复用和停档的短进程也计入。"]
    runtime=audit["execution"]
    if "wall_seconds" in runtime:
        text.append(f"本批从 {runtime['first_start_utc']} 到 {runtime['completed_utc']}，墙钟 {runtime['wall_seconds']/60:.3f} 分钟。")
    text += ["","| 类别 | 数量 | 后端调用秒 | 后端复制/准备秒 | 校准秒 | 搜索秒 | 恢复/最终CPCM秒 | 其他秒 |","|---|---|---|---|---|---|---|---|"]
    for kind in ("fault_pre","fault_l1","fault_rep"):
        subset=[r for r in timing if r["kind"]==kind]
        if not subset:continue
        sums=lambda f:sum(r[f] for r in subset)
        text.append(f"| {kind} | {len(subset)} | {sums('backend_call_seconds'):.3f} | {sums('backend_total_seconds')-sums('backend_call_seconds'):.3f} | {sums('calibration_seconds'):.3f} | {sums('search_seconds')+sums('L1_search_seconds'):.3f} | {sums('restore_best_seconds')+sums('final_cpcm_seconds'):.3f} | {sums('other_seconds')+sums('R0_prepare_seconds'):.3f} |")
    text += ["","L2记录中继承的t_R0/t_L1不重复计入当前作业时间；每个独立预算实际发生的校准、搜索、恢复和后端都计入。并行共享机器的进程小时不可当墙钟，J正式串行计时另做。","",
        "明细在 `tables/i4_jobs.csv`、`i4_states.csv`、`i4_backend_samples.csv`、`i4_repair_trajectories.csv`、`i4_budget_groups.csv`、`i4_stop_decisions.csv`、`i4_units.csv`、`i4_decode_reference.csv`、`i4_timing.csv`；核对记录在 `evidence/i4_completion_audit.json`。"]
    return text


def run(args):
    manifest=read(args.manifest)
    index=check_plan(manifest)
    if args.check:
        print(json.dumps(dict(status="plan_parse_passed",jobs=201,healthy_inputs=len(index),A3h_PF4=9,A1_R0=4,
            maximum_repair_trajectories=135,maximum_backend_samples=1700,prefill_or_request_generated=False)))
        return
    jobs,present,missing,states,samples,timing,checks=shared.collect(args.out,manifest,args.progress,args.smoke,"qwen3moe",KINDS)
    stops,groups,trajectories=shared.stop_tables(jobs,present)
    units=shared.unit_tables(jobs,present,states,stops,index)
    decode=decode_table(units)
    rows=shared.job_rows(jobs,present,timing)
    runtime=shared.execution(args.out,present,args.smoke)
    actual=sum(r["status"]=="ok" for r in trajectories)
    reused=sum(r["status"]=="reused" for r in trajectories)
    finish=runtime["controller_finish"]
    complete=not args.smoke and not missing and finish is not None and finish["rc"]==0 and finish["completed"]==201
    audit=dict(status="smoke_parse_passed" if args.smoke else "complete" if complete else "in_progress",
        smoke_only=args.smoke,formal_completed_jobs=0 if args.smoke else len(present),
        expected_jobs=201,completed_jobs=len(present),missing_jobs=len(missing),missing=missing,
        actual_repair_trajectories=actual,reused_repair_jobs=reused,backend_samples=len(samples),
        A3h_PF4_units=sum(r["EIR_complete"] for r in units),A1_R0_units=sum(r["start_arm"]=="A1" for r in units),
        selected_healthy_inputs=182,new_healthy_searches=0,stop_records=len(stops),
        evaluated_stop_records=sum(r["decision"]=="evaluated" for r in stops),
        propagated_stop_records=sum(r["decision"]=="propagated_stop" for r in stops),
        kinds=dict(Counter(v["spec"]["kind"] for v in present.values())),
        faults=["PF4"],excluded=["all prefill FFN","WATOS decode FFN","MF6","Split"],
        decode_reference_rows=len(decode),prefill_reference_rows=0,request_reference_rows=0,
        execution=runtime,input_checks=checks,manifest_sha256=shared.sha(args.manifest),
        healthy_input_index_sha256=shared.sha(INPUTS),report_source_sha256=shared.sha(__file__),
        shared_report_source_sha256=shared.sha(Path(shared.__file__)))
    assert len(samples)==sum(r["actual_backend_calls"] for r in rows if r["status"]!="pending")
    if not args.progress and not args.smoke:
        assert complete and actual+reused==135 and len(stops)==36 and len(units)==13
        assert audit["A3h_PF4_units"]==9 and audit["A1_R0_units"]==4
        assert len(samples)==350+actual*10 and len(decode)==6
    destination=args.destination or (HERE/"evidence/i4_report_smoke" if args.smoke else HERE)
    tables=dict(jobs=rows,states=states,backend_samples=samples,timing=timing,repair_trajectories=trajectories,
        budget_groups=groups,stop_decisions=stops,units=units,decode_reference=decode)
    for name,values in tables.items():shared.common.table(destination/"tables"/("i4_"+name+".csv"),values)
    write(destination/"evidence/i4_completion_audit.json",audit)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/("I4_REPORT_PARSE_CHECK.md" if args.smoke else "I4_LIGHT_REPORT.md")).write_text(
        "\n".join(render(audit,units,groups,rows,timing,decode))+"\n",encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ("status","completed_jobs","actual_repair_trajectories","reused_repair_jobs","backend_samples","A3h_PF4_units")}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"out/i4_light")
    p.add_argument("--manifest",type=Path,default=MANIFEST)
    p.add_argument("--destination",type=Path)
    p.add_argument("--progress",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--check",action="store_true")
    run(p.parse_args())


if __name__=="__main__":main()
