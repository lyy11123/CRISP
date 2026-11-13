"""Read complete post-F/G DSTU runs and representative traces; no simulation."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from e_run import ROOT, write_json

ARMS=("BB0","H0","A256","A256-M","S256","S256-M")
NAMES={"wsc":"WSC-LLM","face":"FACE"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compact(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"))


def table(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path.with_suffix(".json"),rows)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix(".csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:compact(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def source_rows(manifest, out, scope):
    rows=[]
    for spec in read(manifest)["jobs"]:
        filename=Path(out)/"results"/(spec["job_id"]+".json")
        result=read(filename)
        assert result["status"]=="ok" and result["spec"]==spec
        if spec.get("alias_job"):
            assert result["alias_identity_checked"] and result["backend_calls"]==0
            original=read(Path(out)/"results"/(spec["alias_job"]+".json"))
            assert result["evaluation_identity"]==original["evaluation_identity"]
        sample=dict(scope=scope,job_id=spec["job_id"],hardware=spec["hardware_profile"],
                    condition=spec["condition"],arm=spec["arm"],run_type=spec["run_type"],
                    seed=spec["seed"],phase=spec.get("phase"),stream=spec.get("stream"),
                    source_result=filename.as_posix(),source_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                    reuse="historical" if spec.get("reuse_job") else "batch_alias" if spec.get("alias_job") else "new",
                    source_job=spec.get("reuse_job") or spec.get("alias_job"),
                    backend_calls=result["backend_calls"],wall_seconds=result["wall_seconds"],
                    simulation_seconds=result["simulation_seconds"],peak_rss_gb=result["peak_rss_gb"],
                    timing_includes_trace=result.get("timing_includes_resource_trace",False),
                    frozen_rank=spec["rank"],platform_graph_sha256=spec["graph_sha256"],
                    chunk_stats=result.get("chunk_stats"),transform_seconds=result.get("transform_seconds"),
                    candidate_counters=result.get("policy_counters"),
                    candidate_counters_available=result.get("candidate_counters_available",False),
                    alias_identity_checked=result.get("alias_identity_checked"),
                    evaluation_identity=result.get("evaluation_identity"),
                    issue_trace=result.get("issue_trace"),message_trace=result.get("message_trace"))
        if spec["run_type"]=="solo":
            sample["completion_ns"]=result["completion"][str(spec["stream"])]
            sample["first_start_ns"]=result["first_start"][str(spec["stream"])]
        else:
            tau=spec["tau"];reference=spec["reference"]
            cp=result["completion"]["0"];cd=result["completion"]["1"]-tau
            lp=(cp-1.5*reference["T_P"])/reference["T_P"]
            ld=(cd-1.5*reference["T_D"])/reference["T_D"]
            assert cp==result["metrics"]["P_completion"] and cd==result["metrics"]["D_latency"]
            assert abs(lp-result["metrics"]["Lhat_P"])<1e-12 and abs(ld-result["metrics"]["Lhat_D"])<1e-12
            sample.update(P_completion_ns=cp,D_latency_ns=cd,tau_ns=tau,
                          actual_D_start_ns=result["first_start"]["1"],actual_D_end_ns=result["completion"]["1"],
                          reference_P_ns=reference["T_P"],reference_D_ns=reference["T_D"],
                          P_deadline_ns=1.5*reference["T_P"],D_deadline_ns=tau+1.5*reference["T_D"],
                          Lhat_P=lp,Lhat_D=ld,Lhat_max=max(lp,ld),deadline_violation=max(lp,ld)>0,
                          old_trace_comparison=result.get("old_result_comparison"),
                          trace_comparison_rule=result.get("trace_comparison_rule"))
        window=result.get("kv_window")
        sample.update(KV_start_ns=window[0] if window else None,KV_end_ns=window[1] if window else None)
        if spec["run_type"]=="joint" and window:
            sample["KV_D_envelope_overlap_ns"]=max(0,min(window[1],sample["actual_D_end_ns"])-max(window[0],sample["actual_D_start_ns"]))
        rows.append(sample)
    return rows


def paired(rows):
    solos={(r["condition"],r["hardware"],r["arm"],r["stream"],r["seed"]):r for r in rows if r["run_type"]=="solo"}
    joints={(r["condition"],r["hardware"],r["arm"],r["phase"],r["seed"]):r for r in rows if r["run_type"]=="joint"}
    checks=[]
    for row in rows:
        if row["run_type"]!="joint":continue
        cond,hw,arm,phase,seed=(row[k] for k in ("condition","hardware","arm","phase","seed"))
        bb=joints[(cond,hw,"BB0",phase,seed)];old=joints[(cond,hw,"A256",phase,seed)]
        for stream,label,metric in ((0,"P","P_completion_ns"),(1,"D","D_latency_ns")):
            solo=solos[(cond,hw,arm,stream,seed)]["completion_ns"]
            bsolo=solos[(cond,hw,"BB0",stream,seed)]["completion_ns"]
            oldsolo=solos[(cond,hw,"A256",stream,seed)]["completion_ns"]
            own=solo-bsolo;interference=row[metric]-solo;binterference=bb[metric]-bsolo
            lhs=row[metric]-bb[metric];rhs=own+interference-binterference
            assert abs(lhs-rhs)<1e-7
            checks.append(dict(job=row["job_id"],stream=label,joint_difference_ns=lhs,
                               own_plus_interference_difference_ns=rhs,error_ns=lhs-rhs))
            row.update({f"{label}_solo_ns":solo,f"{label}_BB0_solo_ns":bsolo,f"{label}_A256_solo_ns":oldsolo,
                        f"{label}_O_ns":own,f"{label}_J_ns":interference,f"{label}_BB0_J_ns":binterference,
                        f"{label}_change_vs_BB0":row[metric]/bb[metric]-1,
                        f"{label}_change_vs_A256":row[metric]/old[metric]-1})
        row["Lhat_change_vs_A256"]=row["Lhat_max"]-old["Lhat_max"]
    return checks


def group_stats(rows, keys, metrics):
    groups=collections.defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for identity,group in sorted(groups.items(),key=lambda x:str(x[0])):
        row=dict(zip(keys,identity));row["n"]=len(group)
        for metric in metrics:
            values=[r[metric] for r in group]
            row[metric+"_mean"]=statistics.mean(values)
            row[metric+"_median"]=statistics.median(values)
            row[metric+"_min"]=min(values);row[metric+"_max"]=max(values)
            row[metric+"_lower"]=sum(v<0 for v in values)
            row[metric+"_equal"]=sum(v==0 for v in values)
            row[metric+"_higher"]=sum(v>0 for v in values)
        row["deadline_violations"]=sum(r.get("deadline_violation",False) for r in group)
        result.append(row)
    return result


def trace_tables(runs,out):
    """Use recorded issue times; do not infer activity from a KV envelope."""
    summaries=[];resources=[];parents=[];consumers=[];messages={};starts={}
    for run in runs:
        if not run["issue_trace"]:continue
        prefix={k:run[k] for k in ("job_id","hardware","condition","arm","phase","seed")}
        resource_rows={};counts=collections.Counter();previous_stream={}
        filename=Path(out)/"traces"/Path(run["issue_trace"]).name
        with gzip.open(filename,"rt",encoding="utf-8") as stream:
            for line in stream:
                r=json.loads(line);link=compact(r["resource"])
                row=resource_rows.setdefault(link,dict(prefix,resource=link,issues=0,
                                                       queue_wait_sum_ns=0,queue_wait_max_ns=0,
                                                       queue_wait_samples=0,same_stream_insertions=0,
                                                       original_key_ties=0,original_nonminimum_issues=0,
                                                       higher_priority_takeovers=0))
                row["issues"]+=1;counts["issues"]+=1
                if "head_ready" in r:
                    wait=r["t"]-r["head_ready"];assert wait>=0
                    row["queue_wait_sum_ns"]+=wait;row["queue_wait_max_ns"]=max(row["queue_wait_max_ns"],wait)
                    row["queue_wait_samples"]+=1;counts["queue_wait_ns"]+=wait
                    row["same_stream_insertions"]+=int(r["same_stream_insertion"])
                    row["original_key_ties"]+=int(r["original_true_tie"])
                    row["original_nonminimum_issues"]+=int(r["original_key"]!=r["original_min"])
                    before=previous_stream.get(link)
                    takeover=before is not None and run["frozen_rank"][str(before)]>run["frozen_rank"][str(r["stream"])] and r["lower_rank_ready"]
                    row["higher_priority_takeovers"]+=int(takeover)
                previous_stream[link]=r["stream"]
        records=list(resource_rows.values());resources.extend(records)
        tracked=run["arm"]!="BB0"
        aggregate=dict(prefix,issues=counts["issues"],resources=len(records),candidate_trace_available=tracked,
                       queue_wait_sum_ns=sum(r["queue_wait_sum_ns"] for r in records) if tracked else None,
                       queue_wait_max_ns=max(r["queue_wait_max_ns"] for r in records) if tracked else None,
                       same_stream_insertions=sum(r["same_stream_insertions"] for r in records) if tracked else None,
                       original_key_ties=sum(r["original_key_ties"] for r in records) if tracked else None,
                       original_nonminimum_issues=sum(r["original_nonminimum_issues"] for r in records) if tracked else None,
                       higher_priority_takeovers=sum(r["higher_priority_takeovers"] for r in records) if tracked else None,
                       original_BB0_difference=run.get("old_trace_comparison"))
        with gzip.open(Path(out)/"traces"/Path(run["message_trace"]).name,"rt",encoding="utf-8") as stream:
            detail=json.load(stream)
        key=tuple(prefix[k] for k in ("hardware","condition","phase","seed","arm"))
        messages[key]={compact(r["parent"]):r for r in detail["parents"]}
        starts[key]={compact(r["tag"]):r for r in detail["consumers"]}
        for row in detail["parents"]:parents.append(dict(prefix,**row))
        for row in detail["consumers"]:consumers.append(dict(prefix,**row))
        aggregate["service_occupancy_by_link_kind_ns"]=detail["occupancy"]
        summaries.append(aggregate)
    comparisons=[]
    for key,current in messages.items():
        if key[-1]=="A256":continue
        base_key=key[:-1]+("A256",)
        base=messages[base_key];base_consumers=starts[base_key];new_consumers=starts[key]
        assert set(current)==set(base)
        for parent,now in current.items():
            old=base[parent]
            for tag in old["consumers"]:
                consumer=compact(tag)
                if consumer not in base_consumers:continue
                before=base_consumers[consumer];after=new_consumers[consumer]
                comparisons.append(dict(hardware=key[0],condition=key[1],phase=key[2],seed=key[3],arm=key[4],
                    parent=old["parent"],consumer=tag,stream=old["stream"],
                    old_last_finish_ns=old["last_finish"],new_last_finish_ns=now["last_finish"],
                    last_finish_delta_ns=now["last_finish"]-old["last_finish"],
                    old_consumer_ready_ns=before["ready"],new_consumer_ready_ns=after["ready"],
                    old_consumer_start_ns=before["start"],new_consumer_start_ns=after["start"],
                    consumer_ready_delta_ns=after["ready"]-before["ready"],
                    consumer_start_delta_ns=after["start"]-before["start"],
                    baseline_last_dependency=old["last_finish"]==before["ready"],
                    earlier_message_and_consumer=now["last_finish"]<old["last_finish"] and after["ready"]<before["ready"] and after["start"]<before["start"],
                    old_chunks=old["chunks"],new_chunks=now["chunks"],bytes=old["bytes"]))
    return summaries,resources,parents,consumers,comparisons


def figures(phase_rows,trace_rows,chains,folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"none"})
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);outputs=[]
    colors={"H0":"#7895B2","A256-M":"#D99652","S256":"#866EB0","S256-M":"#167C80"}
    markers={.1:"o",.5:"^",.9:"s"}
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.0),constrained_layout=True)
    plotdata=[]
    for ax,hw in zip(axes,("wsc","face")):
        selected=[r for r in phase_rows if r["condition"]=="healthy" and r["hardware"]==hw and r["arm"] in colors]
        for arm,color in colors.items():
            group=sorted((r for r in selected if r["arm"]==arm),key=lambda r:r["phase"])
            ax.plot([r["P_change_vs_A256_mean"]*100 for r in group],[r["D_change_vs_A256_mean"]*100 for r in group],color=color,alpha=.5,lw=1)
            for r in group:
                ax.scatter(r["P_change_vs_A256_mean"]*100,r["D_change_vs_A256_mean"]*100,marker=markers[r["phase"]],color=color,s=55,zorder=3)
                plotdata.append(r)
        ax.axhline(0,color="grey",lw=.8);ax.axvline(0,color="grey",lw=.8)
        ax.set(title=NAMES[hw],xlabel="P completion change vs A256 (%)",ylabel="D latency change vs A256 (%)")
        ax.grid(alpha=.2);ax.spines[["top","right"]].set_visible(False)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=c,marker="o",label=a) for a,c in colors.items()]+[Line2D([],[],color="black",linestyle="",marker=m,label=f"g={g}") for g,m in markers.items()]
    fig.legend(handles=handles,loc="outside upper center",ncol=7,frameon=False,fontsize=9)
    save_figure(fig,folder,"dstu_quality_tradeoff",plotdata);outputs.append("dstu_quality_tradeoff")
    if trace_rows:
        points=sorted({(r["hardware"],r["phase"]) for r in trace_rows},key=lambda v:(v[0],v[1]))
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.0),constrained_layout=True)
        methods=("A256","A256-M","S256-M");colors2={"A256":"#7895B2",**colors};plotdata=[]
        for method_index,arm in enumerate(methods):
            x=[i+(method_index-1)*.24 for i in range(len(points))]
            values=[];starts=[]
            for hw,g in points:
                r=next(r for r in trace_rows if r["hardware"]==hw and r["phase"]==g and r["arm"]==arm)
                values.append(r["same_stream_insertions"]);plotdata.append(r)
                unique={compact(c["consumer"]):c for c in chains if c["hardware"]==hw and c["phase"]==g and c["arm"]==arm and c["baseline_last_dependency"]}
                if arm=="A256":starts.append(0)
                else:
                    assert unique,(hw,g,arm,"no baseline last-dependency consumers")
                    starts.append(statistics.median(c["consumer_start_delta_ns"] for c in unique.values())/1000)
            axes[0].bar(x,values,.23,color=colors2[arm],label=arm)
            axes[1].bar(x,starts,.23,color=colors2[arm],label=arm)
        labels=[f"{NAMES[hw]}\ng={g}" for hw,g in points]
        for ax in axes:ax.set_xticks(range(len(points)),labels);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True);ax.spines[["top","right"]].set_visible(False)
        axes[0].set_ylabel("Same-stream parent insertions");axes[0].set_yscale("symlog",linthresh=1)
        axes[1].set_ylabel("Consumer start change vs A256 (us)");axes[1].axhline(0,color="grey",lw=.8)
        axes[0].set_title("(a) Recorded resource decisions");axes[1].set_title("(b) Median over eligible consumers")
        fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncol=3,frameon=False)
        save_figure(fig,folder,"dstu_message_mechanism",plotdata);outputs.append("dstu_message_mechanism")
    return outputs


def save_figure(fig,folder,name,data):
    import matplotlib.pyplot as plt
    fig.savefig(folder/(name+".png"),dpi=220,bbox_inches="tight",facecolor="white")
    svg=folder/(name+".svg");fig.savefig(svg,bbox_inches="tight",facecolor="white")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
    plt.close(fig);table(folder/name,data)


def markdown(rows,summary,trace_summary,chains,counts,rank_checks,path):
    lines=["# DSTU 机制与策略试点", "", "使用既有 F/G 的 A3h 映射、路径和硬件；全部新结果独立保存。",
           "C4 排名、公共释放时刻与公共期限取现有 All256/健康 A1 参考。Shared-only 单跑继续使用联合输入的掩码。", "",
           "| 范围 | 逻辑项 | 历史复用 | 本批等价复用 | 新后端调用 |",
           "|---|---:|---:|---:|---:|"]
    for scope,c in counts.items():lines.append(f"| {scope} | {c['logical_units']} | {c['historical']} | {c['batch_alias']} | {c['backend_calls']} |")
    lines += ["", "S256 的 D 单跑复用同配置 H0；执行前核对过滤后的完整事件图、平台、C4、释放和种子。S256 的共享掩码与 P 变换摘要单独保存。", "",
              "## 全部相位与种子的结果", "", "负变化表示低于原 All256+C4。每硬件、条件、方案按三个相位各三个种子统计；故障确认沿用一个统一选定方案。", "",
              "| 条件 | 硬件 | 方案 | P变化：中位 [范围] | D变化：中位 [范围] | D改善/持平/变差 | 超期数 |", "|---|---|---|---|---|---|---:|"]
    for r in summary:
        def val(k):return f"{r[k+'_median']:+.2%} [{r[k+'_min']:+.2%}, {r[k+'_max']:+.2%}]"
        lines.append(f"| {r['condition']} | {NAMES[r['hardware']]} | {r['arm']} | {val('P_change_vs_A256')} | {val('D_change_vs_A256')} | {r['D_change_vs_A256_lower']}/{r['D_change_vs_A256_equal']}/{r['D_change_vs_A256_higher']} | {r['deadline_violations']} |")
    lines += ["", "![健康延迟变化](figures/dstu_quality_tradeoff.png)", "", "每点平均同相位的三个后端种子，误差范围和逐种子值保存在完整表。图中零点为当前 A256；同时保留 H0、A256-M、S256、S256-M，拆开切块范围与消息服务顺序。", "",
              "## 单跑自身变化与联合额外变化", "", "`O_X = 新方案单跑 − BB0单跑`；`J_X = 同方案联合 − 同方案单跑`。逐项复算 `联合差 = O_X + J_X − J_X(BB0)`。D 的联合时间均为完成减释放。",
              "P 期限为公共 `1.5 T_P`，D 的绝对期限为 `tau + 1.5 T_D`；Lhat_D 分子使用D延迟减`1.5 T_D`。公共分母不随策略改变。负迟约为提前完成余量，余量减少与真正超期分开报告。",
              "完整 [逐种子表](tables/dstu_runs.csv)、[相位汇总](tables/dstu_phase_summary.csv)、[O/J检查](evidence/dstu_report_checks.json) 同时保留不利结果。", "",
              "## 代表资源日志", "", "健康代表点固定为 WSC 的 g=0.1/0.9 与 FACE 的 g=0.9，均 seed1。新的四方案直接保留日志；旧 BB0/A256 另行重放。BB0 的重放与历史单次偏差完整列出，机制解释使用这次日志自身。Hop 的同图、同策略、同种子重放严格核对。"]
    if trace_summary:
        lines += ["", "| 硬件/相位 | 方案 | 同流插入 | 原键真平局 | 原键非最小发放 | 资源等待总和(μs) |", "|---|---|---:|---:|---:|---:|"]
        for r in trace_summary:
            wait="未记录候选" if r["queue_wait_sum_ns"] is None else f"{r['queue_wait_sum_ns']/1000:.3f}"
            def show(k):return "—" if r[k] is None else str(r[k])
            lines.append(f"| {NAMES[r['hardware']]}/{r['phase']} | {r['arm']} | {show('same_stream_insertions')} | {show('original_key_ties')} | {show('original_nonminimum_issues')} | {wait} |")
        lines += ["", "BB0重放与历史结果的单次差异：", "", "| 硬件/相位 | P完成变化(ns) | D延迟变化(ns) | Lhat_P变化 | Lhat_D变化 | Lhat_max变化 |", "|---|---:|---:|---:|---:|---:|"]
        for r in trace_summary:
            if r["arm"]!="BB0":continue
            delta=r["original_BB0_difference"]
            lines.append(f"| {NAMES[r['hardware']]}/{r['phase']} | {delta['P_completion']['delta']} | {delta['D_latency']['delta']} | {delta['Lhat_P']['delta']:+.8f} | {delta['Lhat_D']['delta']:+.8f} | {delta['Lhat_max']['delta']:+.8f} |")
        eligible=[r for r in chains if r["baseline_last_dependency"]]
        lines += ["", "![消息服务机制](figures/dstu_message_mechanism.png)", "",
                  f"逐消息与消费者表共 {len(chains)} 个比较，其中 {len(eligible)} 项在A256中是消费者的最后到达依赖。图(b)仅统计这些依赖所连的消费者，每个消费者计一次；表中仍保存全部消息及消费者。",
                  "[消息与消费者比较](tables/dstu_message_consumer.csv) 给出原消息末块、消费者就绪与实际启动的变化。资源等待是某块首次成为该链路候选至实际发放的时间；不同链路的等待总和不是端到端延迟。",
                  "原键平局和同流插入分别计数；现有记录没有保存旧parent的完整候选key，不能把每次插入都归于随机平局。连续优先允许同流owner优先于其他更长路径消息，因此也包含非平局顺序调整。",
                  "插入次数是绝对次数，也随块数变化。A256与A256-M使用同一切块图，可以直接隔离顺序调整；S256-M同时改变切块范围，需结合S256对照解释。",
                  "[资源等待表](tables/dstu_resource_wait.csv)、[所有消息](tables/dstu_messages.csv)、[所有计算消费者](tables/dstu_consumers.csv) 保留逐项数据。KV/D总体包络仅用于描述释放关系，实际资源竞争由资源日志分析。"]
    lines += ["", "## 运行成本与后续取舍", "", "[运行清单](tables/dstu_runs.csv) 分开构图、后端和完整进程墙钟，记录峰值内存。带资源日志的计时单列，不能作为无日志策略的运行开销。历史复用和批内等价复用不计新的后端调用。",
              "同 seed 便于配对；不同图/策略的shuffle消耗不同，不表示相同随机样本路径。H0与BB0同时包含后端及仲裁差异，不把二者全差异归为切块。",
              "健康全部结果用于统一选取一个故障确认方案，保留不同相位的回退；若无合适改善方案则保留当前策略。性能比较和机制链条分别报告，不因总时间下降就认定所有收益来自同流插入减少。"]
    if rank_checks:
        lines += ["", "## 定版候选按正常规则重新生成排名", "", "使用候选的实际切块图，按原规则分别求两流CPCM hop时间，再比较`1.5×P估值`与`tau+1.5×D估值`。这一步单独计CPCM调用，没有新增后端评估。", "",
                  "| 硬件 | 条件 | 方案 | 相位 | 冻结排名 | 重算排名 | 一致 |", "|---|---|---|---:|---|---|---|"]
        for report in rank_checks:
            for r in report["rows"]:
                lines.append(f"| {NAMES[r['hardware']]} | {r['condition']} | {r['arm']} | {r['phase']} | {compact(r['frozen_rank'])} | {compact(r['recomputed_rank'])} | {r['same_rank']} |")
        if any(not r["same_rank"] for result in rank_checks for r in result["rows"]):
            lines.append("\n存在排名变化，当前质量表仍是冻结排名结果；采用正常排名后的增量确认需要单列任务。")
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(args):
    out=Path(args.out);tables=Path(args.tables);evidence=tables.parent/"evidence"
    rows=source_rows(args.health_manifest,out,"healthy")
    if args.fault_manifest:rows+=source_rows(args.fault_manifest,out,"PF4")
    checks=paired(rows)
    trace_runs=source_rows(args.trace_manifest,out,"baseline_trace") if args.trace_manifest else []
    allruns=rows+trace_runs
    metrics=("P_change_vs_A256","D_change_vs_A256","Lhat_change_vs_A256","P_O_ns","D_O_ns","P_J_ns","D_J_ns")
    joints=[r for r in rows if r["run_type"]=="joint"]
    summary=group_stats(joints,("condition","hardware","arm"),metrics)
    phases=group_stats(joints,("condition","hardware","arm","phase"),metrics)
    table(tables/"dstu_runs",rows);table(tables/"dstu_summary",summary);table(tables/"dstu_phase_summary",phases)
    trace_summaries,resources,parents,consumers,chains=trace_tables(allruns,out) if args.trace_manifest else ([],[],[],[],[])
    for name,value in (("dstu_trace_runs",trace_runs),("dstu_trace_summary",trace_summaries),("dstu_resource_wait",resources),
                       ("dstu_messages",parents),("dstu_consumers",consumers),("dstu_message_consumer",chains)):
        if value:table(tables/name,value)
    counts={}
    for scope in sorted({r["scope"] for r in allruns}):
        selected=[r for r in allruns if r["scope"]==scope]
        counts[scope]=dict(logical_units=len(selected),historical=sum(r["reuse"]=="historical" for r in selected),
                           batch_alias=sum(r["reuse"]=="batch_alias" for r in selected),backend_calls=sum(r["backend_calls"] for r in selected))
    plots=figures(phases,trace_summaries,chains,args.figures)
    rank_checks=[read(p) for p in args.rank_check]
    validation=dict(status="passed",run_counts=counts,paired_identity_checks=len(checks),
                    max_identity_error_ns=max(abs(r["error_ns"]) for r in checks),
                    public_deadline_and_Lhat_checks=len(joints)+len(trace_runs),
                    traced_runs=len(trace_summaries),message_consumer_comparisons=len(chains),figures=plots,
                    normal_rank_checks=sum(len(r["rows"]) for r in rank_checks),
                    normal_rank_changes=sum(not r["same_rank"] for result in rank_checks for r in result["rows"]),
                    sources=dict(health=str(args.health_manifest),fault=str(args.fault_manifest),trace=str(args.trace_manifest),out=str(out)))
    write_json(evidence/"dstu_report_checks.json",validation)
    markdown(rows,summary,trace_summaries,chains,counts,rank_checks,tables.parent/"DSTU_REFINEMENT.md")
    print(json.dumps(validation))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--health-manifest",default="e2e/post_fg/manifest/dstu_health.json")
    p.add_argument("--fault-manifest")
    p.add_argument("--trace-manifest")
    p.add_argument("--rank-check",action="append",default=[])
    p.add_argument("--out",default="e2e/post_fg/out/main")
    p.add_argument("--tables",default="e2e/post_fg/tables")
    p.add_argument("--figures",default="e2e/post_fg/figures")
    run(p.parse_args())
