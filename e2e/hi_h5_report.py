"""Read H5 results and recompute common-reference and paired P/D metrics."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

import fg_batch
import hi_config
import hi_next_report as common
from hi_split_supplement import read,write

ROOT,HERE=common.ROOT,common.HERE
COUNTS=dict(hi_joint_export=2,joint_graph=5,joint_solo=39,joint_reference=1,joint=63)
METRICS=("P_completion","D_latency","paired_delta_P","P_self_amplification","D_self_amplification","Lhat_P","Lhat_D","Lhat_max")
COMPARISONS=(
    ("healthy","A3h_no_DSTU","A1_no_DSTU","mapping"),
    ("healthy","A3h_A256","A3h_no_DSTU","service_with_A256"),
    ("D","A3h_R0_no_DSTU","A1_R0_no_DSTU","mapping"),
    ("D","A3h_EIR_no_DSTU","A3h_R0_no_DSTU","EIR"),
    ("D","A3h_EIR_A256","A3h_EIR_no_DSTU","service_with_A256"),
    ("D","A3h_EIR_A256","A1_R0_no_DSTU","complete_system"))


def path(value):
    p=Path(value.replace("\\","/"))
    return p if p.is_absolute() else ROOT/p


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a,b):
    assert abs(a-b)<=max(1e-9,abs(b)*1e-12),(a,b)


def stats(values,prefix):
    return {prefix+"_"+name:fn(values) for name,fn in (("mean",statistics.mean),("median",statistics.median),("min",min),("max",max))}


def check_plan(manifest,inputs):
    jobs=manifest["jobs"]
    assert len(jobs)==110 and Counter(j["kind"] for j in jobs)==COUNTS
    assert manifest["logical_paired_solos"]==42 and manifest["actual_paired_solos"]==39 and len(manifest["solo_aliases"])==3
    expected={(condition,arm,g,seed) for condition,arms in (
        ("healthy",("A1_no_DSTU","A3h_no_DSTU","A3h_A256")),
        ("D",("A1_R0_no_DSTU","A3h_R0_no_DSTU","A3h_EIR_no_DSTU","A3h_EIR_A256")))
        for arm in arms for g in (.1,.5,.9) for seed in (1,2,3)}
    observed={(j["spec"]["condition"],j["spec"]["arm"],j["spec"]["phase"],j["spec"]["seed"]) for j in jobs if j["kind"]=="joint"}
    assert observed==expected
    assert inputs["P"]["A1"]["candidate_id"]=="H_llama_busybarn_prefill_mha_A1_g0_l4"
    assert inputs["P"]["A3h"]["candidate_id"]=="H_llama_busybarn_prefill_mha_A3h_g0_s123"
    source_hashes={}
    for source in inputs["P"].values():
        for key in ("record","raw","snapshot"):
            file=path(source[key]);fingerprint=sha(file)
            assert fingerprint==source["sha256"][key]
            source_hashes[str(file.relative_to(ROOT))]=fingerprint
    for state in inputs["D"].values():
        for source in state.values():
            file=path(source["file"]);fingerprint=sha(file)
            assert fingerprint==source["sha256"]
            source_hashes[str(file.relative_to(ROOT))]=fingerprint
    assert hi_config.model_contract("llama")["decode_chain"]==dict(norm=1,mha=8,proj=1,ffn=1)
    return source_hashes


def collect(out,manifest,progress):
    jobs={j["id"]:j for j in manifest["jobs"]}
    results,rows,checks,missing={ },[],[],[]
    for name,job in jobs.items():
        rp=out/job["result"];receipt_path=out/"receipts"/(name+".json")
        if not rp.exists() or not receipt_path.exists():
            missing.append(name);continue
        r,receipt=read(rp),read(receipt_path)
        assert receipt["rc"]==0 and receipt["identity"]==fg_batch.identity(job)
        assert r["job_id"]==name and r["kind"]==job["kind"] and r["spec"]==job["spec"]
        assert r["status"] in ("ok","reused") and (r["status"]=="ok" or r["kind"]=="joint_graph")
        spec=r["spec"]
        assert spec["model"]=="llama" and spec["model_contract_sha256"]==hi_config.digest(hi_config.model_contract("llama"))
        backend_call=r["kind"] in ("joint_solo","joint")
        if backend_call:
            simulation=r["simulation"]
            assert simulation["seed"]==spec["seed"] and simulation["backend"]==spec["backend"]
            assert all(s["n"]==s["n_done"] for s in simulation["streams"].values())
            if spec["backend"]=="hop":assert simulation["audit"]["bad"]==0
        row=dict(job_id=name,kind=r["kind"],status=r["status"],wall_seconds=r["wall_seconds"],
            receipt_seconds=receipt["wall_seconds"],peak_rss_gib=r["peak_rss_gb"],backend_calls=int(backend_call),
            backend_execution_seconds=r["simulation"]["wall"] if backend_call else 0,
            condition=spec.get("condition"),arm=spec.get("arm"),phase=spec.get("phase"),seed=spec.get("seed"))
        results[name]=r;rows.append(row)
        checks.append(dict(job_id=name,result_sha256=sha(rp),receipt_sha256=sha(receipt_path),receipt_identity_matches=True))
    if not progress:
        assert not missing and {p.stem for p in (out/"results").glob("*.json")}==set(jobs)
        assert {p.stem for p in (out/"receipts").glob("*.json")}==set(jobs)
    return results,rows,checks,missing


def graph_rows(results):
    output=[]
    for r in results.values():
        if r["kind"]!="joint_graph":continue
        assert r["chain"]==["dc_ln"]+["dc_mha"]*8+["dc_proj","dc_ffn"]
        assert r["decode_chain_contract"]==dict(mha_groups=8,external_norms=1,ffn_embedded_norms=1,stages=11)
        assert r["kv"]["unique_payload_bytes"]==8*1024**2 and r["kv"]["messages"]>0
        assert r["shared"]["compute"]==0 and r["shared"]["KV_D_links"]>0
        source_hashes={k:sha(path(v)) for k,v in r["source_exports"].items()}
        files={k:sha(path(v)) for k,v in r["files"].items()}
        if r["spec"]["condition"]=="D":assert r["fault"]["failed_links"]==[[[2,11],[3,8]],[[3,8],[2,11]]]
        else:assert not r["fault"]["failed_links"]
        output.append(dict(job_id=r["job_id"],mapping_state=r["spec"]["mapping_state"],condition=r["spec"]["condition"],
            P_events=r["sizes"]["0"]["events"],D_events=r["sizes"]["1"]["events"],
            chain=r["chain"],chain_edges=r["chain_edges"],KV=r["kv"],shared=r["shared"],
            chunked_sizes=r["sizes"],CPCM_hop_that=r["that"],platform_sha256=r["platform_sha256"],
            stream_identity=r["stream_identity"],actual_iteration_sha256=r["actual_iteration_sha256"],
            source_exports_sha256=source_hashes,graph_files_sha256=files,
            graph_prepare_seconds=r["wall_seconds"],peak_rss_gib=r["peak_rss_gb"]))
    return output


def metric_rows(results):
    solos,references,joints=[],[],[]
    for r in results.values():
        spec=r["spec"]
        if r["kind"]=="joint_solo":
            sim=r["simulation"];stream=str(spec["stream"])
            assert set(sim["streams"])=={stream} and sim["release"]=={stream:0}
            solos.append(dict(job_id=r["job_id"],graph_job=spec["graph_job"],condition=spec["condition"],
                backend=spec["backend"],stream=spec["stream"],seed=spec["seed"],completion_ns=sim["streams"][stream]["C"],
                events=sim["streams"][stream]["n"],KV_window=sim.get("kv_window"),simulation_seconds=sim["wall"]))
        elif r["kind"]=="joint_reference":
            ps=[results[name]["simulation"] for name in spec["P_solos"]]
            ds=[results[name]["simulation"] for name in spec["D_solos"]]
            assert [s["seed"] for s in ps]==[1,2,3] and [s["seed"] for s in ds]==[1,2,3]
            assert all(s["backend"]=="orig" for s in ps+ds)
            values=dict(T_P=statistics.mean(s["streams"]["0"]["C"] for s in ps),
                T_D=statistics.mean(s["streams"]["1"]["C"] for s in ds),
                KV_start=statistics.mean(s["kv_window"][0] for s in ps),KV_end=statistics.mean(s["kv_window"][1] for s in ps))
            for k,v in values.items():close(r[k],v)
            references.append(dict(job_id=r["job_id"],**values,P_solos=spec["P_solos"],D_solos=spec["D_solos"],
                public_P_relative_deadline=1.5*values["T_P"],public_D_relative_deadline=1.5*values["T_D"]))
        elif r["kind"]=="joint":
            graph,reference=results[spec["graph_job"]],results[spec["reference_job"]]
            sim=r["simulation"]
            tau=int(round(reference["KV_start"]+spec["phase"]*(reference["KV_end"]-reference["KV_start"])))
            assert r["tau"]==tau and sim["release"]=={"0":0,"1":tau}
            paired={stream:results[spec[label]] for stream,label in ((0,"P_solo"),(1,"D_solo"))}
            for stream,solo in paired.items():
                source=results[solo["graph_job"]]
                assert source["platform_sha256"]==graph["platform_sha256"]
                assert source["stream_identity"][f"{spec['backend']}_{stream}"]==graph["stream_identity"][f"{spec['backend']}_{stream}"]
                assert source["actual_iteration_sha256"][f"{spec['backend']}_{stream}"]==graph["actual_iteration_sha256"][f"{spec['backend']}_{stream}"]
                assert solo["spec"]["seed"]==spec["seed"] and solo["spec"]["backend"]==spec["backend"]
            psolo,dsolo=(paired[stream]["simulation"]["streams"][str(stream)]["C"] for stream in (0,1))
            cp,cd=(sim["streams"][str(stream)]["C"] for stream in (0,1))
            metrics=dict(P_completion=cp,D_latency=cd-tau,paired_delta_P=cp-psolo,P_solo=psolo,D_solo=dsolo,
                P_self_amplification=cp/psolo,D_self_amplification=(cd-tau)/dsolo,
                Lhat_P=(cp-1.5*reference["T_P"])/reference["T_P"],
                Lhat_D=(cd-tau-1.5*reference["T_D"])/reference["T_D"])
            metrics["Lhat_max"]=max(metrics["Lhat_P"],metrics["Lhat_D"])
            for k,v in metrics.items():close(r["metrics"][k],v)
            internal={}
            if spec["policy"]=="C4":
                deadlines={0:1.5*graph["that"]["0"],1:tau+1.5*graph["that"]["1"]}
                order=sorted(deadlines,key=lambda k:(deadlines[k],k))
                internal=dict(C4_internal_deadline_estimates=deadlines,C4_derived_rank={k:order.index(k) for k in order},
                    C4_rank_source="recomputed frozen formula on this graph; distinct from public evaluation deadlines")
            row=dict(job_id=r["job_id"],condition=spec["condition"],arm=spec["arm"],backend=spec["backend"],policy=spec["policy"],
                phase=spec["phase"],seed=spec["seed"],tau_ns=tau,**metrics,
                P_deadline_ns=1.5*reference["T_P"],D_absolute_deadline_ns=tau+1.5*reference["T_D"],
                both_deadlines_met=metrics["Lhat_max"]<=0,P_solo_job=spec["P_solo"],D_solo_job=spec["D_solo"],
                graph_job=spec["graph_job"],reference_job=spec["reference_job"],KV_window=sim.get("kv_window"),
                shared=r["shared"],simulation_seconds=sim["wall"],full_job_seconds=r["wall_seconds"],**internal)
            joints.append(row)
    return solos,references,joints


def summaries(joints):
    arms,phases=[],[]
    for phase_specific,target in ((False,arms),(True,phases)):
        groups=defaultdict(list)
        for r in joints:groups[(r["condition"],r["arm"],r["phase"] if phase_specific else "all")].append(r)
        for (condition,arm,phase),rows in groups.items():
            row=dict(condition=condition,arm=arm,phase=phase,n=len(rows),expected=3 if phase_specific else 9,
                completed=len(rows)==(3 if phase_specific else 9),deadline_successes=sum(r["both_deadlines_met"] for r in rows))
            for metric in METRICS:row.update(stats([r[metric] for r in rows],metric))
            target.append(row)
    lookup={(r["condition"],r["arm"],r["phase"],r["seed"]):r for r in joints}
    differences=[]
    for condition,target,baseline,meaning in COMPARISONS:
        for phase in (.1,.5,.9):
            for seed in (1,2,3):
                a,b=(lookup.get((condition,arm,phase,seed)) for arm in (target,baseline))
                if a is None or b is None:continue
                assert a["tau_ns"]==b["tau_ns"] and a["reference_job"]==b["reference_job"]
                differences.append(dict(condition=condition,target=target,baseline=baseline,meaning=meaning,phase=phase,seed=seed,
                    P_relative_change=a["P_completion"]/b["P_completion"]-1,D_relative_change=a["D_latency"]/b["D_latency"]-1,
                    Lhat_max_change=a["Lhat_max"]-b["Lhat_max"],paired_delta_P_change=a["paired_delta_P"]-b["paired_delta_P"],
                    target_job=a["job_id"],baseline_job=b["job_id"]))
    grouped=defaultdict(list)
    for r in differences:grouped[r["condition"],r["target"],r["baseline"],r["meaning"]].append(r)
    paired=[]
    for (condition,target,baseline,meaning),rows in grouped.items():
        row=dict(condition=condition,target=target,baseline=baseline,meaning=meaning,n=len(rows),expected=9,completed=len(rows)==9)
        for metric in ("P_relative_change","D_relative_change","Lhat_max_change","paired_delta_P_change"):
            vals=[r[metric] for r in rows];row.update(stats(vals,metric))
            row.update({metric+"_better":sum(v<0 for v in vals),metric+"_same":sum(v==0 for v in vals),metric+"_worse":sum(v>0 for v in vals)})
        paired.append(row)
    return arms,phases,differences,paired


def execution(out,jobs):
    records={p.relative_to(out).as_posix():read(p) for folder in (out,out/"pilot_run")
        for pattern in ("*.start.json","*.status.json","*.batch.json","controller_finish.json")
        for p in sorted(folder.glob(pattern))}
    starts=[r["created_utc"] for name,r in records.items() if name.endswith(".start.json")]
    finish=records.get("controller_finish.json")
    runtime=dict(records=records,controller_finish=finish,
        completed_job_process_hours=sum(r["wall_seconds"] for r in jobs)/3600,
        completed_receipt_process_hours=sum(r["receipt_seconds"] for r in jobs)/3600,
        backend_execution_process_hours=sum(r["backend_execution_seconds"] for r in jobs)/3600,
        failed_attempts=[dict(source=p.relative_to(ROOT).as_posix(),**read(p)) for p in sorted(out.glob("failed_attempts/*/*/receipt.json"))])
    if starts:
        start=min(starts);runtime["first_start_utc"]=start
        if finish:
            runtime.update(last_finish_utc=finish["utc"],
                elapsed_seconds_including_between_batches=(datetime.fromisoformat(finish["utc"])-datetime.fromisoformat(start)).total_seconds())
    runtime["batches"]=[]
    for prefix,label in (("pilot_run/","pilot"),("","remaining")):
        start_record=records.get(prefix+"h5.start.json")
        end_record=records.get(prefix+"controller_finish.json")
        if start_record and end_record:
            runtime["batches"].append(dict(batch=label,start_utc=start_record["created_utc"],finish_utc=end_record["utc"],
                completed_including_cached=end_record["completed"],
                elapsed_seconds=(datetime.fromisoformat(end_record["utc"])-datetime.fromisoformat(start_record["created_utc"])).total_seconds()))
    resources=[r for r in records.values() if "min_available_gib" in r]
    if resources:
        runtime.update(max_active=max(r["max_active"] for r in resources),
            min_available_gib=min(r["min_available_gib"] for r in resources))
        combined=[r["max_combined_active"] for r in resources if "max_combined_active" in r]
        if combined:runtime["max_combined_active"]=max(combined)
    return runtime


def cost_rows(jobs):
    output=[]
    for kind in COUNTS:
        rows=[r for r in jobs if r["kind"]==kind]
        if not rows:continue
        output.append(dict(kind=kind,completed=len(rows),expected=COUNTS[kind],
            total_job_seconds=sum(r["wall_seconds"] for r in rows),
            backend_execution_seconds=sum(r["backend_execution_seconds"] for r in rows),
            non_backend_job_seconds=sum(r["wall_seconds"]-r["backend_execution_seconds"] for r in rows),
            max_job_seconds=max(r["wall_seconds"] for r in rows),
            median_job_seconds=statistics.median(r["wall_seconds"] for r in rows),
            peak_rss_gib=max(r["peak_rss_gib"] for r in rows),
            backend_calls=sum(r["backend_calls"] for r in rows)))
    return output


def render(audit,inputs,graphs,refs,joints,arms,phases,paired,costs):
    text=["# H5：Llama P/D 联合结果","",
        f"正式作业已完成 {audit['completed_jobs']}/110，其中联合 {len(joints)}/63、配对单跑 {audit['completed_solos']}/39。"
        +( "本批已全部结束。" if audit["status"]=="complete" else "以下是已完成部分；未齐三种子、三相位的行同时列出实际数量。"),"",
        "主硬件 2×4 晶圆包含两个不相交 2×2 组。P 是 8 个 Q 头、1 组 K/V 的 prefill MHA 代表子图，带真实跨组 KV 写回；D 是一个外置 LN、8 个 MHA 头组、projection、含另一个 RMSNorm 的 FFN，共 11 段。P/D 是独立请求，只共享资源。这里没有模拟完整 prefill FFN 或 80 层同时执行。","",
        "![H5 联合结果：七个实验臂的 P 完成时间与 D 延迟](figures/h5_joint.png)","",
        "图中柱与圆点为每臂9样本均值，横线为实测最小至最大值；健康和D故障分别分组，两个面板使用各自横轴。可编辑矢量图：[SVG](figures/h5_joint.svg)。绘图输入与63个源样本在 `figures/h5_joint_plot_inputs.json`。","",
        "## 输入与运行方式","",
        f"P 侧均从正式 14 候选按后端均值选优：A1 `{inputs['P']['A1']['candidate_id']}`；A3h `{inputs['P']['A3h']['candidate_id']}`。D 直接使用 H4 的健康、R0 或 stop300 最终代表导出，保留既定停止规则及代表修复种子。未重新搜索。",
        "D 故障为右侧组 `(die2,core11) ↔ (die3,core8)` 双向连接失效。健康三臂、D 故障四臂，各 3 个相位 × 3 个后端种子。42 个逻辑配对单跑合为 39 次实际单跑：只复用故障平台上 A3h-R0 与 A3h-EIR 相同 P 流的三个 original 单跑。每个联合结果的配对流核对平台、全事件字段及实际遍历顺序。","",
        "无 DSTU 使用 original 整包后端；A256 使用全链路 256 KiB + C4 按跳后端。两者差值反映这套执行方式的整体变化，不能单独归因为优先级排序。相同后端内的 A1/A3h、R0/EIR 比较另列。","",
        "## 公共参考与逐次结果","",
        "统一参考来自健康 A1/original 的三个 P 单跑、三个 D 单跑：T_P、T_D 为均值，KV 起止时刻也取均值；释放时刻 τ=round(KV_start+g×(KV_end−KV_start))。所有实验臂使用相同参考。评价期限为 P 的 1.5T_P 和 D 的 τ+1.5T_D。C4 的内部排序仍按当前图自身 CPCM 估值生成期限，已在逐次表单独记录，没有替换成公共评价期限。"]
    if refs:
        r=refs[0]
        text += ["",f"参考 T_P={r['T_P']:,.3f} ns，T_D={r['T_D']:,.3f} ns；KV 窗口 [{r['KV_start']:,.3f}, {r['KV_end']:,.3f}] ns。"]
    text += ["", "P 完成时间 C_P；D 延迟 C_D−τ；配对 ΔP=C_P−P_solo；自身放大率分别除以同种子、同后端配对单跑。L̂_P=(C_P−1.5T_P)/T_P，L̂_D=(C_D−τ−1.5T_D)/T_D，L̂_max 取两者最大值，≤0 表示两条期限均满足。", "",
        "| 条件 | 实验臂 | 已完成/9 | P 完成时间均值（ms） | D 延迟均值（ms） | 配对 ΔP 均值（ms） | L̂_max 中位数 [最小,最大] | 两条期限均满足 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in arms:
        text.append(f"| {r['condition']} | {r['arm']} | {r['n']}/9 | {r['P_completion_mean']/1e6:.6f} | {r['D_latency_mean']/1e6:.6f} | {r['paired_delta_P_mean']/1e6:.6f} | {r['Lhat_max_median']:.5f} [{r['Lhat_max_min']:.5f}, {r['Lhat_max_max']:.5f}] | {r['deadline_successes']}/{r['n']} |")
    text += ["", "相位与种子逐项值在 `tables/h5_joints.csv`；三种子分相位汇总在 `h5_by_arm_phase.csv`；单跑全部样本在 `h5_solos.csv`。各臂行的均值取当前完成样本，只有 n=9 才覆盖整套相位与种子。", "",
        "## 同相位、同种子的比较","",
        "| 条件 | 比较（前者/后者） | 已配对/9 | P 变化中位数 [最小,最大] | D 变化中位数 [最小,最大] | L̂_max 差中位数 [最小,最大] | L̂_max 改善/相同/恶化 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in paired:
        def show(k,percent):
            scale=100 if percent else 1;suffix="%" if percent else ""
            return f"{r[k+'_median']*scale:+.3f}{suffix} [{r[k+'_min']*scale:+.3f}{suffix}, {r[k+'_max']*scale:+.3f}{suffix}]"
        text.append(f"| {r['condition']} | {r['target']} / {r['baseline']} | {r['n']}/9 | {show('P_relative_change',True)} | {show('D_relative_change',True)} | {show('Lhat_max_change',False)} | {r['Lhat_max_change_better']}/{r['Lhat_max_change_same']}/{r['Lhat_max_change_worse']} |")
    text += ["", "变化比例按每对样本分别计算后取中位数与范围；没有用两组中位数相除。配对 ΔP 的差、每对样本及源 ID 另见 `h5_paired_differences.csv`。负值表示减少，回退值全部保留。", "",
        "## 三个释放相位与结果解释","",
        "| 条件 | 实验臂 | g=0.1 的 D / ΔP（μs） | g=0.5 的 D / ΔP（μs） | g=0.9 的 D / ΔP（μs） |",
        "|---|---|---:|---:|---:|"]
    phase_lookup={(r["condition"],r["arm"],r["phase"]):r for r in phases}
    for r in arms:
        cells=[]
        for g in (.1,.5,.9):
            p=phase_lookup.get((r["condition"],r["arm"],g))
            cells.append(f"{p['D_latency_mean']/1000:.3f} / {p['paired_delta_P_mean']/1000:+.3f} ({p['n']}/3)" if p else "待完成")
        text.append(f"| {r['condition']} | {r['arm']} | {' | '.join(cells)} |")
    if audit["status"]=="complete":
        compare={(r["condition"],r["target"],r["baseline"]):r for r in paired}
        health=compare["healthy","A3h_no_DSTU","A1_no_DSTU"]
        repair=compare["D","A3h_EIR_no_DSTU","A3h_R0_no_DSTU"]
        system=compare["D","A3h_EIR_A256","A1_R0_no_DSTU"]
        healthy_a=compare["healthy","A3h_A256","A3h_no_DSTU"]
        repaired_a=compare["D","A3h_EIR_A256","A3h_EIR_no_DSTU"]
        text += ["",f"主要收益来自映射与故障修复：健康 A3h/original 相对 A1/original 的 D 延迟配对中位变化为 {health['D_relative_change_median']:+.2%}；故障后 EIR/original 相对 A3h-R0/original 为 {repair['D_relative_change_median']:+.2%}。完整 A3h+EIR+A256 相对 A1-R0 的 D 变化为 {system['D_relative_change_median']:+.2%}，这是整个系统的比较。",
            f"本批 A256 没有形成额外的延迟改善。相对同一映射的 original 后端，健康/修复后 P 变化中位数分别为 {healthy_a['P_relative_change_median']:+.2%}/{repaired_a['P_relative_change_median']:+.2%}，D 分别为 {healthy_a['D_relative_change_median']:+.2%}/{repaired_a['D_relative_change_median']:+.2%}；18 对样本的 L̂_max 全部增大。其配对 ΔP 接近零或为负，但 A256 自身 P 单跑就较慢，因此不能把较小的 ΔP 等同于更短的 P 完成时间。",
            "三个相位下 D 的均值差异较小，P 的重叠影响更依赖释放位置：故障 original 的配对 ΔP 从 g=0.1 到 g=0.9 均下降，A1-R0 为 +200.550→+11.858 μs，A3h-R0 为 +98.950→+17.498 μs，EIR 为 +73.158→+7.535 μs。健康 A3h 的 ΔP 在 g=0.5 反而较高，因此没有统一的单调相位规律。每个相位仅三种子，全部样本保留供后续联合分析。",
            "63 个联合样本的 L̂_max 全部小于零，当前 1.5 倍健康 A1 期限未造成超期；本批能够比较延迟与重叠代价，不能据此声称 A256 在这些点上消除了期限违约。负的 ΔP 表示该次联合运行比配对单跑更早完成，按原后端样本保留，没有截断为零。"]
    text += ["",
        "## 图规模与计算开销","",
        "| 组合图 | P 原事件数 | D 原事件数 | KV 逻辑载荷（MiB） | KV/D 共享链路 | P/D 共享计算设备 | 构图秒 | 峰值 RSS（GiB） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in graphs:text.append(f"| {r['mapping_state']} | {r['P_events']} | {r['D_events']} | {r['KV']['unique_payload_bytes']/2**20:.1f} | {r['shared']['KV_D_links']} | {r['shared']['compute']} | {r['graph_prepare_seconds']:.3f} | {r['peak_rss_gib']:.3f} |")
    text += ["", "切块后事件数、跳活动数、KV multicast 链路字节、11 段连接数与实际图文件指纹均在 `h5_graphs.csv`。", "",
        "| 作业类别 | 已完成/计划 | 作业耗时之和（秒） | 其中后端执行（秒） | 其余准备与收尾（秒） | 最长作业（秒） | 峰值 RSS（GiB） |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in costs:text.append(f"| {r['kind']} | {r['completed']}/{r['expected']} | {r['total_job_seconds']:.3f} | {r['backend_execution_seconds']:.3f} | {r['non_backend_job_seconds']:.3f} | {r['max_job_seconds']:.3f} | {r['peak_rss_gib']:.3f} |")
    runtime=audit["execution"]
    text += ["",f"已完成作业累计 {runtime['completed_job_process_hours']:.6f} 进程小时，其中后端执行 {runtime['backend_execution_process_hours']:.6f} 进程小时。进程小时是各进程耗时相加，不等于并行批次的经过时间。"]
    if "elapsed_seconds_including_between_batches" in runtime:
        text += [f"从首次试点启动 {runtime['first_start_utc']} 到最近批次记录 {runtime['last_finish_utc']}，经过 {runtime['elapsed_seconds_including_between_batches']/60:.3f} 分钟，包含试点与全量之间的等待；最近结束标记为 {runtime['controller_finish']['completed']}/{runtime['controller_finish']['expected']} 项。"]
    for batch in runtime["batches"]:
        text.append(f"{'14项试点' if batch['batch']=='pilot' else '其余96项（复用已完成14项）'}调度阶段经过 {batch['elapsed_seconds']:.3f} 秒。")
    if "max_active" in runtime:
        text += [f"H5 本池观测峰值活跃 {runtime['max_active']}；最低可用内存 {runtime['min_available_gib']:.3f} GiB。当前保存失败尝试 {len(runtime['failed_attempts'])} 条。本批记录未提供其他并行批次合计的活跃峰值。"]
    text += ["", "作业时间来自该作业结果与 receipt；后端时间使用 simulation.wall，构图、平台准备、拷贝、记录整理等归入剩余时间，不重复相加。最初清单的每类秒数仅为启动占位，不作为论文串行计时。完整批次结果、试点批次记录、资源记录和源文件哈希在 `evidence/h5_completion_audit.json`。"]
    return text


def run(args):
    manifest,inputs=read(args.manifest),read(args.inputs)
    sources=check_plan(manifest,inputs)
    if args.check:
        print(json.dumps(dict(status="plan_and_inputs_passed",jobs=110,joints=63,paired_solos=39,
            new_searches=0,source_files=len(sources),decode_stages=11)));return
    results,jobs,checks,missing=collect(args.out,manifest,args.progress)
    graphs=graph_rows(results);solos,refs,joints=metric_rows(results)
    arms,phases,differences,paired=summaries(joints)
    runtime=execution(args.out,jobs);costs=cost_rows(jobs)
    finish=runtime["controller_finish"]
    complete=not missing and finish is not None and finish["rc"]==0 and finish["completed"]==110 and finish["expected"]==110
    audit=dict(status="complete" if complete else "in_progress",completed_jobs=len(results),expected_jobs=110,
        completed_joint=len(joints),expected_joint=63,completed_solos=len(solos),expected_solos=39,
        backend_calls=sum(r["backend_calls"] for r in jobs),expected_backend_calls=102,
        kinds=dict(Counter(r["kind"] for r in results.values())),missing_jobs=len(missing),missing=missing,
        graph_count=len(graphs),reference_count=len(refs),new_healthy_searches=0,logical_solos=42,
        actual_solos_planned=39,solo_aliases=manifest["solo_aliases"],execution=runtime,
        input_checks=checks,source_input_sha256=sources,manifest_sha256=sha(args.manifest),
        input_index_sha256=sha(args.inputs),report_source_sha256=sha(__file__),
        scope="Llama representative prefill MHA plus complete decode operator chain; health and D PF4")
    if not args.progress:
        assert complete and Counter(r["kind"] for r in results.values())==COUNTS
        assert len(joints)==63 and len(solos)==39 and len(graphs)==5 and len(refs)==1
        assert audit["backend_calls"]==102 and len(arms)==7 and len(phases)==21 and len(differences)==54 and len(paired)==6
    tables=dict(jobs=jobs,graphs=graphs,solos=solos,reference=refs,joints=joints,by_arm=arms,by_arm_phase=phases,
        paired_differences=differences,paired_summary=paired,costs=costs)
    for name,values in tables.items():common.table(args.destination/"tables"/("h5_"+name+".csv"),values)
    write(args.destination/"evidence/h5_completion_audit.json",audit)
    (args.destination/"H5_REPORT.md").write_text("\n".join(render(audit,inputs,graphs,refs,joints,arms,phases,paired,costs))+"\n",encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ("status","completed_jobs","completed_joint","completed_solos","backend_calls")}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE/"out/h5")
    parser.add_argument("--manifest",type=Path,default=HERE/"manifest/h5.json")
    parser.add_argument("--inputs",type=Path,default=HERE/"manifest/h5_inputs.json")
    parser.add_argument("--destination",type=Path,default=HERE)
    parser.add_argument("--progress",action="store_true")
    parser.add_argument("--check",action="store_true")
    run(parser.parse_args())


if __name__=="__main__":main()
