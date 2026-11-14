"""Read Qwen I5 joint results; share metric formulas, not Llama result text."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics

import fg_batch
import hi_config
import hi_h5_report as shared
from hi_split_supplement import read,write

ROOT,HERE=shared.ROOT,shared.HERE
COUNTS=shared.COUNTS


def check_plan(manifest,inputs):
    jobs=manifest["jobs"]
    assert len(jobs)==110 and Counter(j["kind"] for j in jobs)==COUNTS
    assert manifest["logical_paired_solos"]==42 and manifest["actual_paired_solos"]==39 and len(manifest["solo_aliases"])==3
    expected={(condition,arm,g,s) for condition,arms in (
        ("healthy",("A1_no_DSTU","A3h_no_DSTU","A3h_A256")),
        ("D",("A1_R0_no_DSTU","A3h_R0_no_DSTU","A3h_EIR_no_DSTU","A3h_EIR_A256")))
        for arm in arms for g in (.1,.5,.9) for s in (1,2,3)}
    actual={(j["spec"]["condition"],j["spec"]["arm"],j["spec"]["phase"],j["spec"]["seed"]) for j in jobs if j["kind"]=="joint"}
    assert actual==expected
    assert inputs["model"]=="qwen3moe"
    assert inputs["P"]["A1"]["candidate_id"]=="I_qwen3moe_busybarn_prefill_mha_A1_g0_l1"
    assert inputs["P"]["A3h"]["candidate_id"]=="I_qwen3moe_busybarn_prefill_mha_A3h_g0_s3"
    source_hashes={}
    for source in inputs["P"].values():
        for key in ("record","raw","snapshot"):
            p=shared.path(source[key]);fingerprint=shared.sha(p)
            assert fingerprint==source["sha256"][key]
            source_hashes[p.relative_to(ROOT).as_posix()]=fingerprint
    for state in inputs["D"].values():
        for source in state.values():
            p=shared.path(source["file"]);fingerprint=shared.sha(p)
            assert fingerprint==source["sha256"]
            source_hashes[p.relative_to(ROOT).as_posix()]=fingerprint
    contract=hi_config.model_contract("qwen3moe")
    assert contract["decode_chain"]==dict(norm=1,mha=4,proj=1,ffn=1) and contract["decode_ffn_includes_rmsnorm"]
    assert contract["prefill_kv_output_bytes_per_group"]==contract["batch"]*contract["sequence_length"]*contract["head_dim"]*2*2==8*1024**2
    return source_hashes


def collect(out,manifest,progress):
    results,rows,checks,missing={},[],[],[]
    jobs={j["id"]:j for j in manifest["jobs"]}
    for name,job in jobs.items():
        rp=out/job["result"];receipt_path=out/"receipts"/(name+".json")
        if not rp.exists() or not receipt_path.exists():missing.append(name);continue
        r,receipt=read(rp),read(receipt_path)
        assert receipt["rc"]==0 and receipt["identity"]==fg_batch.identity(job)
        assert r["job_id"]==name and r["kind"]==job["kind"] and r["spec"]==job["spec"]
        assert r["status"] in ("ok","reused") and (r["status"]=="ok" or r["kind"]=="joint_graph")
        spec=r["spec"]
        assert spec["model"]=="qwen3moe" and spec["model_contract_sha256"]==hi_config.digest(hi_config.model_contract("qwen3moe"))
        call=r["kind"] in ("joint_solo","joint")
        if call:
            sim=r["simulation"]
            assert sim["seed"]==spec["seed"] and sim["backend"]==spec["backend"]
            assert all(s["n"]==s["n_done"] for s in sim["streams"].values())
            if spec["backend"]=="hop":assert sim["audit"]["bad"]==0
        results[name]=r
        rows.append(dict(job_id=name,kind=r["kind"],status=r["status"],wall_seconds=r["wall_seconds"],
            receipt_seconds=receipt["wall_seconds"],peak_rss_gib=r["peak_rss_gb"],backend_calls=int(call),
            backend_execution_seconds=r["simulation"]["wall"] if call else 0,
            condition=spec.get("condition"),arm=spec.get("arm"),phase=spec.get("phase"),seed=spec.get("seed")))
        checks.append(dict(job_id=name,result_sha256=shared.sha(rp),receipt_sha256=shared.sha(receipt_path),receipt_identity_matches=True))
    if not progress:
        assert not missing and {p.stem for p in (out/"results").glob("*.json")}==set(jobs)
        assert {p.stem for p in (out/"receipts").glob("*.json")}==set(jobs)
    return results,rows,checks,missing


def graph_rows(results):
    rows=[]
    contract=hi_config.model_contract("qwen3moe")
    for r in results.values():
        if r["kind"]!="joint_graph":continue
        assert r["chain"]==["dc_ln"]+["dc_mha"]*4+["dc_proj","dc_ffn"]
        assert r["decode_chain_contract"]==dict(mha_groups=4,external_norms=1,ffn_embedded_norms=1,stages=7)
        assert r["kv"]["unique_payload_bytes"]==contract["prefill_kv_output_bytes_per_group"] and r["kv"]["messages"]>0
        assert r["shared"]["compute"]==0 and r["shared"]["KV_D_links"]>0
        if r["spec"]["condition"]=="D":assert r["fault"]["failed_links"]==[[[2,11],[3,8]],[[3,8],[2,11]]]
        else:assert not r["fault"]["failed_links"]
        rows.append(dict(job_id=r["job_id"],mapping_state=r["spec"]["mapping_state"],condition=r["spec"]["condition"],
            P_events=r["sizes"]["0"]["events"],D_events=r["sizes"]["1"]["events"],chain=r["chain"],chain_edges=r["chain_edges"],
            KV=r["kv"],shared=r["shared"],chunked_sizes=r["sizes"],CPCM_hop_that=r["that"],platform_sha256=r["platform_sha256"],
            stream_identity=r["stream_identity"],actual_iteration_sha256=r["actual_iteration_sha256"],
            source_exports_sha256={k:shared.sha(shared.path(v)) for k,v in r["source_exports"].items()},
            graph_files_sha256={k:shared.sha(shared.path(v)) for k,v in r["files"].items()},
            graph_prepare_seconds=r["wall_seconds"],peak_rss_gib=r["peak_rss_gb"]))
    return rows


def execution(out,jobs):
    runtime=shared.execution(out,jobs)
    runtime["batches"]=[]
    for prefix,label in (("pilot_run/","pilot"),("","remaining")):
        start=runtime["records"].get(prefix+"i5.start.json")
        end=runtime["records"].get(prefix+"controller_finish.json")
        if start and end:
            runtime["batches"].append(dict(batch=label,start_utc=start["created_utc"],finish_utc=end["utc"],
                completed_including_cached=end["completed"],
                elapsed_seconds=(datetime.fromisoformat(end["utc"])-datetime.fromisoformat(start["created_utc"])).total_seconds()))
    estimate_path=HERE/"evidence/i5_pilot_estimate.json"
    if estimate_path.exists():
        estimate=read(estimate_path)
        pending={r["id"] for r in estimate["jobs"] if not r["completed"]}
        actual=[r for r in jobs if r["job_id"] in pending]
        runtime["pilot_updated_estimate"]=dict(source=estimate_path.relative_to(ROOT).as_posix(),source_sha256=shared.sha(estimate_path),
            forecast_completed=estimate["completed"],forecast_remaining=estimate["remaining"],
            forecast_remaining_process_minutes=estimate["remaining_process_minutes"],queue_forecasts=estimate["queues"],
            actual_remaining_completed=len(actual),actual_remaining_job_process_minutes=sum(r["wall_seconds"] for r in actual)/60,
            actual_remaining_receipt_process_minutes=sum(r["receipt_seconds"] for r in actual)/60,
            basis=estimate["assumptions"])
    return runtime


def render(audit,inputs,graphs,refs,arms,phases,paired,differences,costs):
    text=["# I5：Qwen P/D 联合结果","",
        f"正式作业完成 {audit['completed_jobs']}/110；其中联合 {audit['completed_joint']}/63、配对单跑 {audit['completed_solos']}/39。"
        +("本批已全部完成。" if audit["status"]=="complete" else "下表标明当前已有样本数。"),"",
        "主硬件2×4晶圆包含两个2×2组，batch16。P是8个Q头+1组K/V的prefill MHA代表图及真实跨组KV写回，Qwen head_dim=64，逻辑KV为16×2048×64×2×2=8MiB。D是外置LN+4个MHA头组+projection+FFN，共7段；FFN内含另一个RMSNorm。P/D为独立请求，只共享资源。","",
        f"P从各14候选选取后端均值赢家：A1 `{inputs['P']['A1']['candidate_id']}`，A3h `{inputs['P']['A3h']['candidate_id']}`。D使用I4-light健康、R0和stop300最终代表状态。故障为右组 `(die2,core11) ↔ (die3,core8)` 双向连接失效。",
        "健康3臂、D故障4臂，各3相位×3后端种子。42个逻辑单跑合为39次实际单跑，只复用同PF4平台上A3h-R0/EIR完全相同P流的三个original单跑；逐次核对平台、完整有序事件和实际遍历指纹。","",
        "## 公共参考与指标","",
        "健康A1/original的3个P单跑和3个D单跑给出T_P、T_D及KV窗口均值，所有实验臂共用τ=round(KV_start+g×(KV_end−KV_start))。P评价期限为1.5T_P，D为τ+1.5T_D；C4内部排序期限按各自图的CPCM估值计算，和公共评价期限分别记录。",
        "P完成时间=C_P，D延迟=C_D−τ，配对ΔP=C_P−P_solo；自身放大率除以同种子、同后端单跑。L̂_P=(C_P−1.5T_P)/T_P，L̂_D=(C_D−τ−1.5T_D)/T_D，L̂_max取较大者。"]
    for r in refs:text.append(f"公共参考T_P={r['T_P']:,.3f}ns，T_D={r['T_D']:,.3f}ns；KV窗口[{r['KV_start']:,.3f}, {r['KV_end']:,.3f}]ns。")
    if audit["status"]=="complete":
        text += ["","![I5：Qwen七个实验臂的P完成时间与D延迟](figures/i5_joint.png)","",
            "柱与圆点为每臂9样本均值，横线为实测最小至最大值；健康和D故障分别分组。矢量图：[SVG](figures/i5_joint.svg)；逐样本绘图输入和核对值见 `figures/i5_joint_plot_inputs.json`。"]
    text += ["","| 条件 | 实验臂 | 样本数 | P均值ms | D均值ms | 配对ΔP均值ms | L̂_max中位数[最小,最大] | 两期限均满足 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in arms:text.append(f"| {r['condition']} | {r['arm']} | {r['n']}/9 | {r['P_completion_mean']/1e6:.6f} | {r['D_latency_mean']/1e6:.6f} | {r['paired_delta_P_mean']/1e6:.6f} | {r['Lhat_max_median']:.5f} [{r['Lhat_max_min']:.5f}, {r['Lhat_max_max']:.5f}] | {r['deadline_successes']}/{r['n']} |")
    text += ["","## 同相位、同种子配对比较","",
        "original使用整包后端，A256使用全链路256KiB+C4按跳后端，两者的变化是完整执行方式的变化。映射和EIR则在相同original后端内比较。每对先计算比例，再对9对取中位数和范围。","",
        "| 条件 | 前者/后者 | 配对数 | P变化中位数[范围] | D变化中位数[范围] | L̂_max差中位数[范围] | L̂改善/相同/恶化 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in paired:
        def show(k,percent):
            scale=100 if percent else 1;suffix="%" if percent else ""
            return f"{r[k+'_median']*scale:+.3f}{suffix} [{r[k+'_min']*scale:+.3f}{suffix}, {r[k+'_max']*scale:+.3f}{suffix}]"
        text.append(f"| {r['condition']} | {r['target']}/{r['baseline']} | {r['n']}/9 | {show('P_relative_change',True)} | {show('D_relative_change',True)} | {show('Lhat_max_change',False)} | {r['Lhat_max_change_better']}/{r['Lhat_max_change_same']}/{r['Lhat_max_change_worse']} |")
    text += ["","## 释放相位","","| 条件 | 实验臂 | g=0.1 D/ΔP μs | g=0.5 D/ΔP μs | g=0.9 D/ΔP μs |","|---|---|---:|---:|---:|"]
    phase_lookup={(r["condition"],r["arm"],r["phase"]):r for r in phases}
    for r in arms:
        cells=[]
        for g in (.1,.5,.9):
            p=phase_lookup.get((r["condition"],r["arm"],g))
            cells.append(f"{p['D_latency_mean']/1000:.3f}/{p['paired_delta_P_mean']/1000:+.3f} ({p['n']}/3)" if p else "待完成")
        text.append(f"| {r['condition']} | {r['arm']} | {' | '.join(cells)} |")
    if audit["status"]=="complete":
        text += ["", "## 结果解释", ""]
        pairs={(r["condition"],r["target"],r["baseline"]):r for r in paired}
        mapping=pairs["healthy","A3h_no_DSTU","A1_no_DSTU"]
        repair=pairs["D","A3h_EIR_no_DSTU","A3h_R0_no_DSTU"]
        system=pairs["D","A3h_EIR_A256","A1_R0_no_DSTU"]
        dstu=[pairs["healthy","A3h_A256","A3h_no_DSTU"],pairs["D","A3h_EIR_A256","A3h_EIR_no_DSTU"]]
        text += [f"健康映射从A1改为A3h后，P与D配对变化中位数分别为{mapping['P_relative_change_median']:+.2%}、{mapping['D_relative_change_median']:+.2%}。同一original后端内，故障EIR相对A3h-R0的D变化为{repair['D_relative_change_median']:+.2%}，P变化为{repair['P_relative_change_median']:+.3%}。",
            f"A256呈现P/D之间的取舍：健康/修复后P中位变化为{dstu[0]['P_relative_change_median']:+.2%}/{dstu[1]['P_relative_change_median']:+.2%}，D为{dstu[0]['D_relative_change_median']:+.2%}/{dstu[1]['D_relative_change_median']:+.2%}。两个条件分别有{dstu[0]['Lhat_max_change_better']}和{dstu[1]['Lhat_max_change_better']}对L̂_max改善，均有3对增大。"]
        for c,target,baseline in (("healthy","A3h_A256","A3h_no_DSTU"),("D","A3h_EIR_A256","A3h_EIR_no_DSTU")):
            medians=[]
            for g in (.1,.5,.9):
                values=[r["D_relative_change"] for r in differences if r["condition"]==c and r["target"]==target and r["baseline"]==baseline and r["phase"]==g]
                assert len(values)==3
                medians.append(statistics.median(values))
            text.append(f"{c}条件中，A256在g=0.1/0.5/0.9的D配对中位变化依次为{medians[0]:+.2%}/{medians[1]:+.2%}/{medians[2]:+.2%}。较晚释放时original本身的D延迟已降低，A256不再更快；g=0.9的三个种子均出现这一回退。")
        text += [f"完整A3h+EIR+A256相对A1-R0，P变化中位数{system['P_relative_change_median']:+.2%}，D为{system['D_relative_change_median']:+.2%}。这包含映射、修复和服务方式三部分。",
            f"63个联合样本中，{sum(r['deadline_successes'] for r in arms)}个满足共同的两条期限。7个超期样本全部来自A1-R0的D，A3h-R0已在9个样本中全部满足期限；不能把这7个样本的恢复单独归为EIR或A256。ΔP是与配对单跑的时间差，自身放大率以配对单跑为分母，两者与公共期限指标分列。"]
    text += ["","## 图规模和分项时间","","| 组合图 | P事件 | D事件 | 逻辑KV MiB | KV/D共享链路 | 构图秒 | RSS GiB |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in graphs:text.append(f"| {r['mapping_state']} | {r['P_events']} | {r['D_events']} | {r['KV']['unique_payload_bytes']/2**20:.1f} | {r['shared']['KV_D_links']} | {r['graph_prepare_seconds']:.3f} | {r['peak_rss_gib']:.3f} |")
    text += ["","| 类别 | 完成/计划 | 作业秒之和 | 后端执行秒 | 其余准备与收尾秒 | 最长作业秒 | 峰值RSS GiB |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in costs:text.append(f"| {r['kind']} | {r['completed']}/{r['expected']} | {r['total_job_seconds']:.3f} | {r['backend_execution_seconds']:.3f} | {r['non_backend_job_seconds']:.3f} | {r['max_job_seconds']:.3f} | {r['peak_rss_gib']:.3f} |")
    runtime=audit["execution"]
    text += ["",f"已完成作业累计{runtime['completed_job_process_hours']:.6f}进程小时，后端纯执行{runtime['backend_execution_process_hours']:.6f}进程小时。二者为包含关系，不相加。进程小时也不等于并行经过时间。"]
    for b in runtime["batches"]:text.append(f"{b['batch']}阶段从{b['start_utc']}到{b['finish_utc']}，经过{b['elapsed_seconds']:.3f}秒，结束记录完成数含先前复用为{b['completed_including_cached']}。")
    if "elapsed_seconds_including_between_batches" in runtime:text.append(f"首次试点到最近批次结束共{runtime['elapsed_seconds_including_between_batches']/60:.3f}分钟，包含两批之间的等待。")
    if "max_active" in runtime:text.append(f"I5本池峰值活跃{runtime['max_active']}，最低可用内存{runtime['min_available_gib']:.3f}GiB，保存失败尝试{len(runtime['failed_attempts'])}条。")
    if "pilot_updated_estimate" in runtime:
        e=runtime["pilot_updated_estimate"]
        text += ["","试点后更新的剩余96项估计如下，等待其他批次释放槽位的时间另计：","",
            "| 可用并行槽 | 预测经过时间（分钟） |","|---:|---:|"]
        for q in e["queue_forecasts"]:text.append(f"| {q['available_slots']} | {q['minutes']:.3f} |")
        text += ["",f"预测剩余作业累计{e['forecast_remaining_process_minutes']:.3f}进程分钟；实际完成{e['actual_remaining_completed']}项，作业时间合计{e['actual_remaining_job_process_minutes']:.3f}进程分钟，含启动与轮询的receipt合计{e['actual_remaining_receipt_process_minutes']:.3f}进程分钟。"]
        remaining=next((r for r in runtime["batches"] if r["batch"]=="remaining"),None)
        if remaining:text.append(f"实际剩余批次经过{remaining['elapsed_seconds']/60:.3f}分钟，本池峰值44；对应试点模型32–64槽约2.40–3.14分钟的预测范围。此前安排约3–6分钟，实际略早完成。估时文件原样保存在`evidence/i5_pilot_estimate.json`。")
    text += ["","逐次样本、三种子分相位、配对差值、图文件指纹和分项耗时见`tables/i5_*.csv`；原始结果复算记录为`evidence/i5_completion_audit.json`。本批是代表prefill MHA与完整decode算子链的联合仿真，不含完整prefill FFN、48层同时执行或MF6/Split。"]
    return text


def run(args):
    manifest,inputs=read(args.manifest),read(args.inputs)
    sources=check_plan(manifest,inputs)
    if args.check:
        print(json.dumps(dict(status="plan_and_inputs_passed",jobs=110,joint=63,paired_solos=39,source_files=len(sources),decode_stages=7,KV_bytes=8388608)));return
    results,jobs,checks,missing=collect(args.out,manifest,args.progress)
    graphs=graph_rows(results);solos,refs,joints=shared.metric_rows(results)
    arms,phases,differences,paired=shared.summaries(joints)
    runtime=execution(args.out,jobs);costs=shared.cost_rows(jobs);finish=runtime["controller_finish"]
    complete=not missing and finish is not None and finish["rc"]==0 and finish["completed"]==110 and finish["expected"]==110
    audit=dict(status="complete" if complete else "in_progress",completed_jobs=len(results),expected_jobs=110,
        completed_joint=len(joints),expected_joint=63,completed_solos=len(solos),expected_solos=39,
        backend_calls=sum(r["backend_calls"] for r in jobs),expected_backend_calls=102,
        kinds=dict(Counter(r["kind"] for r in results.values())),missing_jobs=len(missing),missing=missing,
        graph_count=len(graphs),reference_count=len(refs),new_healthy_searches=0,logical_solos=42,actual_solos_planned=39,
        solo_aliases=manifest["solo_aliases"],execution=runtime,input_checks=checks,source_input_sha256=sources,
        manifest_sha256=shared.sha(args.manifest),input_index_sha256=shared.sha(args.inputs),report_source_sha256=shared.sha(__file__),
        shared_report_source_sha256=shared.sha(Path(shared.__file__)),model_contract=hi_config.model_contract("qwen3moe"),
        scope="Qwen representative prefill MHA plus seven-stage full decode operator chain; health and D PF4")
    if not args.progress:
        assert complete and Counter(r["kind"] for r in results.values())==COUNTS and audit["backend_calls"]==102
        assert len(joints)==63 and len(solos)==39 and len(graphs)==5 and len(refs)==1
        assert len(arms)==7 and len(phases)==21 and len(differences)==54 and len(paired)==6
    tables=dict(jobs=jobs,graphs=graphs,solos=solos,reference=refs,joints=joints,by_arm=arms,by_arm_phase=phases,
        paired_differences=differences,paired_summary=paired,costs=costs)
    for name,values in tables.items():shared.common.table(args.destination/"tables"/("i5_"+name+".csv"),values)
    write(args.destination/"evidence/i5_completion_audit.json",audit)
    (args.destination/"I5_REPORT.md").write_text("\n".join(render(audit,inputs,graphs,refs,arms,phases,paired,differences,costs))+"\n",encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ("status","completed_jobs","completed_joint","completed_solos","backend_calls")}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE/"out/i5")
    parser.add_argument("--manifest",type=Path,default=HERE/"manifest/i5.json")
    parser.add_argument("--inputs",type=Path,default=HERE/"manifest/i5_inputs.json")
    parser.add_argument("--destination",type=Path,default=HERE)
    parser.add_argument("--progress",action="store_true")
    parser.add_argument("--check",action="store_true")
    run(parser.parse_args())


if __name__=="__main__":main()
