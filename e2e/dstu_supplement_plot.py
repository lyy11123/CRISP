"""Plot the complete 108 matched S/H/I groups; read saved report tables only."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "e2e/dstu_supplement"
CASE = ("scope", "model", "hardware", "mapping_state", "phase", "seed")
GROUPS = (("gpt", "wsc", "GPT / WSC-LLM", 36), ("gpt", "face", "GPT / FACE", 36),
          ("llama", "busybarn", "Llama / main", 18), ("qwen3moe", "busybarn", "Qwen3-MoE / main", 18))
CHOICES = ("Fixed-A256", "Fixed-S256", "Oracle-min-Lhatmax", "Oracle-min-D", "Oracle-min-D-Ple3pct", "Online-CPCM-Gated")
LABELS = ("A256", "S256", "Min Lhat", "Min D", "Min D\nP ≤ +3%", "CPCM\ngated")
COLORS = {"Fixed-A256":"#D47B2C", "Fixed-S256":"#168477", "Online-CPCM-Gated":"#7150A1"}
GROUP_COLORS = ("#2D6DA8", "#C07831", "#3E9372", "#8A65A4")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, values):
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for row in values for k in row))
    temporary=path.with_suffix(path.suffix+".part")
    with temporary.open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=columns)
        writer.writeheader(); writer.writerows(values)
    temporary.replace(path)


def key(row):
    return tuple(row[x] for x in CASE)


def mean(rows, metric):
    return statistics.mean(float(row[metric]) for row in rows)


def save(fig, folder, name):
    folder.mkdir(parents=True,exist_ok=True)
    for extension in ("png","svg"):
        path=folder/f"{name}.{extension}"
        fig.savefig(path,dpi=240,bbox_inches="tight",facecolor="white",
                    metadata={"Date":None} if extension=="svg" else None)
    plt.close(fig)


def overall(choices, folder):
    points=[]
    grouping=[(None,None,"All samples",108),*GROUPS]
    for model,hardware,label,n in grouping:
        for selection in CHOICES:
            selected=[r for r in choices if r["selection"]==selection and
                      (model is None or (r["model"],r["hardware"])==(model,hardware))]
            assert len(selected)==n
            points.append(dict(group=label,model=model or "all",hardware=hardware or "all",
                n=n,selection=selection,P_change_pct=mean(selected,"P_change_vs_BB0_pct"),
                D_change_pct=mean(selected,"D_change_vs_BB0_pct")))
    matrices=[np.array([[next(r for r in points if r["group"]==g[2] and r["selection"]==choice)[f"{stream}_change_pct"]
                         for choice in CHOICES] for g in grouping]) for stream in ("P","D")]
    limit=math.ceil(max(abs(v) for array in matrices for v in array.flat)/5)*5
    norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    fig,axes=plt.subplots(1,2,figsize=(13.2,5.0),layout="constrained")
    for ax,matrix,title in zip(axes,matrices,("(a) Prefill completion", "(b) Decode latency")):
        im=ax.imshow(matrix,cmap="RdBu_r",norm=norm,aspect="auto")
        ax.set_xticks(range(6),LABELS)
        ax.set_yticks(range(5),[f"{g[2]} (n={g[3]})" for g in grouping])
        if ax is axes[1]:
            ax.set_yticklabels([])
        ax.set_title(title,fontweight="bold",pad=13)
        ax.tick_params(axis="both",length=0,labelsize=9)
        for i in range(5):
            for j in range(6):
                ax.text(j,i,f"{matrix[i,j]:+.2f}%",ha="center",va="center",
                        fontsize=9,fontweight="bold" if i==0 else "normal",
                        color="white" if abs(matrix[i,j])>.67*limit else "#202020")
        ax.axhline(.5,color="#273547",lw=1.4)
        for x in (1.5,4.5): ax.axvline(x,color="white",lw=2.7)
        ax.set_xticks(np.arange(-.5,6,1),minor=True)
        ax.set_yticks(np.arange(-.5,5,1),minor=True)
        ax.grid(which="minor",color="white",linewidth=.5)
        ax.tick_params(which="minor",length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.text(1/6,-.22,"Fixed",ha="center",va="top",transform=ax.transAxes,fontsize=9,fontweight="bold")
        ax.text(7/12,-.22,"Hindsight oracle",ha="center",va="top",transform=ax.transAxes,fontsize=9,fontweight="bold")
        ax.text(11/12,-.22,"Online",ha="center",va="top",transform=ax.transAxes,fontsize=9,fontweight="bold")
    colorbar=fig.colorbar(im,ax=list(axes),orientation="horizontal",shrink=.72,pad=.11,aspect=45)
    colorbar.set_label("Paired change vs original (BB0), %  ·  lower is better",fontsize=10)
    save(fig,folder,"strategy_overall_means")
    return points


def phases(choices, folder):
    points=[]
    for index,(model,hardware,label,n) in enumerate(GROUPS):
        for phase in (.1,.5,.9):
            for selection in ("Fixed-A256","Fixed-S256","Online-CPCM-Gated"):
                selected=[r for r in choices if (r["model"],r["hardware"],r["selection"])==(model,hardware,selection)
                          and float(r["phase"])==phase]
                assert len(selected)==n//3
                points.append(dict(group=label,model=model,hardware=hardware,n=len(selected),phase=phase,
                    selection=selection,P_change_pct=mean(selected,"P_change_vs_BB0_pct"),
                    D_change_pct=mean(selected,"D_change_vs_BB0_pct")))
    fig,axes=plt.subplots(2,1,figsize=(12.8,7.0),sharex=True,layout="constrained")
    for ax,stream,title in zip(axes,("P","D"),("(a) Prefill completion", "(b) Decode latency")):
        ax.axhline(0,color="#576272",lw=.9)
        for group_index,(_,_,group,n) in enumerate(GROUPS):
            xs=[group_index*4+x for x in range(3)]
            if group_index%2==1: ax.axvspan(xs[0]-.5,xs[-1]+.5,color="#EFF3F6",zorder=0)
            for selection in ("Fixed-A256","Fixed-S256","Online-CPCM-Gated"):
                selected=[r for r in points if r["group"]==group and r["selection"]==selection]
                selected.sort(key=lambda r:r["phase"])
                online=selection=="Online-CPCM-Gated"
                ax.plot(xs,[r[f"{stream}_change_pct"] for r in selected],
                    color=COLORS[selection],lw=1.9,linestyle="--" if online else "-",
                    marker="s" if online else "o",markersize=7 if online else 4.8,
                    markerfacecolor="none" if online else COLORS[selection],markeredgewidth=1.4,
                    label=selection if group_index==0 else None,zorder=3 if online else 2)
            ax.text(xs[1],1.025,group,transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=10)
        ax.set_title(title,loc="left",fontweight="bold",pad=31)
        ax.set_ylabel("Mean change vs original (%)")
        ax.grid(axis="y",alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlim(-.5,14.5)
    axes[-1].set_xticks([group*4+x for group in range(4) for x in range(3)],
                       ["0.1","0.5","0.9"]*4)
    axes[-1].set_xlabel("Arrival phase g in the common reference KV window")
    handles=[Line2D([],[],color=COLORS[s],marker="s" if s=="Online-CPCM-Gated" else "o",
                    markerfacecolor="none" if s=="Online-CPCM-Gated" else COLORS[s],
                    linestyle="--" if s=="Online-CPCM-Gated" else "-",label=label)
             for s,label in (("Fixed-A256","Fixed A256"),("Fixed-S256","Fixed S256"),("Online-CPCM-Gated","Online CPCM-gated"))]
    fig.legend(handles=handles,loc="outside lower center",ncol=3,frameon=False)
    save(fig,folder,"strategy_phase_means")
    return points


def windows(choices, samples, folder):
    fixed={key(r):r for r in choices if r["selection"]=="Fixed-S256"}
    points=[]
    for sample in samples:
        matched=fixed[key(sample)]
        assert int(sample["shared_resource_count"])>0
        ratio=float(sample["tau_ns"])/float(sample["predicted_KV_end_ns"])
        assert (ratio<1)==(sample["chosen_transport"]=="S256")
        points.append({**{k:sample[k] for k in CASE},"tau_over_predicted_KV_end":ratio,
            "chosen_transport":sample["chosen_transport"],"shared_resource_count":int(sample["shared_resource_count"]),
            "predicted_KV_end_ns":float(sample["predicted_KV_end_ns"]),"tau_ns":float(sample["tau_ns"]),
            "P_change_pct":float(matched["P_change_vs_BB0_pct"]),"D_change_pct":float(matched["D_change_vs_BB0_pct"]),
            "source_gate":sample["source_gate"],"S256_job":matched["chosen_job"]})
    assert len(points)==108
    fig,axes=plt.subplots(1,2,figsize=(12.0,5.2),layout="constrained")
    right=max(r["tau_over_predicted_KV_end"] for r in points)*1.09
    for ax,stream,title in zip(axes,("P","D"),("(a) S256 prefill outcome", "(b) S256 decode outcome")):
        ax.axvspan(1,right,color="#EFF0F2",zorder=0)
        ax.axvline(1,color="#313946",linestyle="--",lw=1.3)
        ax.axhline(0,color="#616B76",linewidth=.85)
        for (model,hardware,label,_),color in zip(GROUPS,GROUP_COLORS):
            selected=[r for r in points if (r["model"],r["hardware"])==(model,hardware)]
            ax.scatter([r["tau_over_predicted_KV_end"] for r in selected],
                       [r[f"{stream}_change_pct"] for r in selected],s=29,c=color,alpha=.76,
                       edgecolors="white",linewidths=.35,label=label)
        ax.set_xlim(0,right)
        ax.set_xlabel(r"Known release $\tau$ / predicted KV end")
        ax.set_ylabel("S256 change vs original (%)")
        ax.set_title(title,loc="left",fontweight="bold",pad=29)
        ax.text(.5,1.015,"Select S256",transform=ax.get_xaxis_transform(),ha="center",fontsize=10)
        ax.text((right+1)/2,1.015,"Select original",transform=ax.get_xaxis_transform(),ha="center",fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.18)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4,frameon=False,fontsize=9)
    save(fig,folder,"gate_prediction_and_outcomes")
    return points


def render(base):
    paths=[base/"tables"/name for name in ("s256_strategy_choices.csv","s256_strategy_summary.csv","online_window_samples.csv")]
    sources={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in paths}
    report=json.loads((base/"evidence/report_audit.json").read_text(encoding="utf-8"))
    assert report["status"]=="complete" and report["complete_groups"]==108
    choices,summary,samples=map(read_csv,paths)
    assert len(choices)==756 and len(samples)==108
    assert len({key(r) for r in choices})==108
    for row in summary:
        selected=[r for r in choices if r["selection"]==row["selection"]]
        for stream in ("P","D"):
            assert math.isclose(mean(selected,f"{stream}_change_vs_BB0_pct"),
                                float(row[f"{stream}_change_vs_BB0_pct_mean"]),abs_tol=1e-10)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":11,
                        "axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,
                        "svg.fonttype":"none","svg.hashsalt":"CRISP_DSTU_20261118","savefig.dpi":240})
    outputs=[("figure_overall_points",overall(choices,base/"figures")),
             ("figure_phase_points",phases(choices,base/"figures")),
             ("figure_gate_points",windows(choices,samples,base/"figures"))]
    for name,values in outputs: write_csv(base/"tables"/(name+".csv"),values)
    current={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in paths}
    assert sources==current
    captions=["# DSTU 最终图表说明", "", "三张图均使用本轮完整的 108 个配对组：GPT/WSC-LLM 36、GPT/FACE 36、Llama 主硬件 18、Qwen3-MoE 主硬件 18。每个样本先除以相同输入的 original（BB0），再等权平均；历史 36 组独立保留在既有报告。", "",
        "## 图 1：完整集合与各模型、硬件的策略选择", "",
        "![完整均值](figures/strategy_overall_means.png)", "",
        "左图是 P 完成时间、右图是 D 延迟相对 original 的变化，负值表示改善。第一行等权平均全部 108 个样本，下面四行分别使用各组全部样本，因样本数不同，第一行不等于四个组均值的简单平均。六列分为固定策略、事后选择、前验在线选择。Min Lhat 最小化最大归一化迟约，Min D 最小化 D 延迟，Min D P≤+3% 在 P 不超过同输入 original 的 103% 时选 D 最快策略。每个事后选择使用完整执行策略的一行成绩，在线列使用评估前冻结的 CPCM 决定。图中直接标注均值，没有将场景差异画成置信区间。", "",
        "数据：tables/figure_overall_points.csv；来源：s256_strategy_choices.csv，已与 s256_strategy_summary.csv 逐项核对。", "",
        "## 图 2：相位与模型、硬件条件", "",
        "![分相位均值](figures/strategy_phase_means.png)", "",
        "每个模型、硬件组内保留三相位，实心圆分别表示固定 A256 与 S256，空心方形表示 Online-CPCM-Gated。每个点平均该组在该相位的全部映射、修复状态与后端种子：两种 GPT 硬件各 12 个样本，Llama/Qwen 各 6 个。不同模型组之间不连线；original 为零线。这里 g 按共同 A1 健康参考窗口定义，不代表各映射相同的真实重叠。", "",
        "数据：tables/figure_phase_points.csv。", "",
        "## 图 3：前验窗口判断与已有后端现象", "",
        "![预测与后端现象](figures/gate_prediction_and_outcomes.png)", "",
        "每点是一条后端种子结果，共 108 点；横轴为已知 D 释放时刻除以 CPCM 在未切块 P 图上的 KV 结束预测。所有本轮状态都有共享有向链路，因此横轴小于 1 的冻结动作是 S256，大于等于 1 的冻结动作是 original。纵轴分别展示固定 S256 相对同输入 original 的 P/D 实际变化，作用是解释该前验判断对应的执行结果；这些实际结果没有参与动作选择。三个种子共享同一前验横坐标，竖向分布是保存样本本身。图中没有按 g=0.9 强制旁路，也没有依据收益重新调整阈值。", "",
        "数据：tables/figure_gate_points.csv；预测与实际窗口源数据分别保留 source_gate 和 S256_job。", "",
        "## 复算", "",
        "`e2e/post_fg/dist/venv/Scripts/python.exe -B e2e/dstu_supplement_plot.py`", "",
        "输出 PNG 和可编辑文字的 SVG。输入、图和绘图数据 SHA-256 在 evidence/figure_audit.json。", ""]
    (base/"FIGURE_GUIDE.md").write_text("\n".join(captions),encoding="utf-8")
    generated=[*sorted((base/"figures").glob("*.png")),*sorted((base/"figures").glob("*.svg")),
               *(base/"tables"/(name+".csv") for name,_ in outputs),base/"FIGURE_GUIDE.md"]
    audit=dict(status="generated",complete_groups=108,selection_rows=756,
               source_sha256=sources,plot_script_sha256=sha(__file__),
               output_sha256={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in generated},
               figure_count=3,panels_per_figure=2,aggregation="mean of per-sample percentages; no confidence intervals",
               visual_inspection="pending",new_backend_calls=0)
    (base/"evidence/figure_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="generated",figures=3,complete_groups=108,source_files=len(sources))))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=BASE)
    render(parser.parse_args().base.resolve())
