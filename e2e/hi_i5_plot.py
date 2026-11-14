"""Two-panel I5 figure from the 63 joint samples and checked arm summaries."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"e2e/hi"
ARMS=(
    ("healthy","A1_no_DSTU","A1 · original",0,"#8B959E"),
    ("healthy","A3h_no_DSTU","A3h · original",1,"#277DA8"),
    ("healthy","A3h_A256","A3h · A256",2,"#D49A37"),
    ("D","A1_R0_no_DSTU","A1-R0 · original",4,"#8B959E"),
    ("D","A3h_R0_no_DSTU","A3h-R0 · original",5,"#277DA8"),
    ("D","A3h_EIR_no_DSTU","A3h-EIR · original",6,"#328573"),
    ("D","A3h_EIR_A256","A3h-EIR · A256",7,"#D49A37"))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def inputs(tables):
    paths=[tables/"i5_joints.csv",tables/"i5_by_arm.csv"]
    joints,arms=(read_csv(p) for p in paths)
    assert len(joints)==63 and len({r["job_id"] for r in joints})==63 and len(arms)==7
    groups=defaultdict(list)
    for r in joints:groups[r["condition"],r["arm"]].append(r)
    summaries={(r["condition"],r["arm"]):r for r in arms}
    values=[]
    for condition,arm,label,y,color in ARMS:
        rows=groups[condition,arm]
        assert len(rows)==9 and {(float(r["phase"]),int(r["seed"])) for r in rows}=={(g,s) for g in (.1,.5,.9) for s in (1,2,3)}
        reference=summaries[condition,arm]
        assert reference["completed"]=="True" and int(reference["n"])==9
        for metric in ("P_completion","D_latency"):
            samples=[float(r[metric]) for r in rows]
            measured={name:fn(samples) for name,fn in (("mean",statistics.mean),("min",min),("max",max))}
            for name,v in measured.items():assert math.isclose(v,float(reference[metric+"_"+name]),rel_tol=1e-12)
            values.append(dict(condition=condition,arm=arm,label=label,y=y,color=color,metric=metric,n=9,
                **{name+"_ns":v for name,v in measured.items()},**{name+"_ms":v/1e6 for name,v in measured.items()},
                sample_job_ids=[r["job_id"] for r in rows],samples_ns=samples))
    return values,{p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def render(values,out):
    style={"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":12,"axes.labelsize":10,
        "svg.fonttype":"none","svg.hashsalt":"crisp-i5-v1","figure.facecolor":"white","axes.facecolor":"white",
        "axes.axisbelow":True}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(10.2,5.8),sharey=True)
        fig.subplots_adjust(left=.215,right=.975,top=.76,bottom=.23,wspace=.16)
        fig.suptitle("Qwen3-MoE P/D overlap: mapping, repair and A256",x=.035,y=.97,ha="left",fontsize=15,fontweight="bold")
        fig.text(.035,.918,"BusyBarn 2 × 4 dies · P: representative prefill MHA + cross-group KV · D: full 7-stage decode chain",
            fontsize=9.2,color="#45515C")
        fig.text(.035,.871,"Nine samples per arm: 3 release phases × 3 backend seeds. Lower is faster.",fontsize=9.2,color="#45515C")
        for axis,metric,title in zip(axes,("P_completion","D_latency"),("(a) P completion time","(b) D latency after release")):
            rows=[r for r in values if r["metric"]==metric]
            max_value=max(r["max_ms"] for r in rows)
            axis.axhspan(3.45,7.5,color="#F3F5F7",zorder=0)
            for r in rows:
                axis.barh(r["y"],r["mean_ms"],height=.55,color=r["color"],edgecolor="white",linewidth=.6,zorder=2)
                axis.errorbar(r["mean_ms"],r["y"],xerr=[[r["mean_ms"]-r["min_ms"]],[r["max_ms"]-r["mean_ms"]]],
                    fmt="o",markersize=3.6,markerfacecolor="white",markeredgecolor="#202B33",color="#202B33",
                    elinewidth=1,capsize=3,capthick=1,zorder=3)
                axis.text(r["max_ms"]+max_value*.025,r["y"],f"{r['mean_ms']:.3f}",va="center",fontsize=9,color="#202B33")
            axis.set_xlim(0,max_value*1.18)
            axis.set_ylim(7.65,-.75)
            axis.set_title(title,loc="left",pad=14)
            axis.set_xlabel("Time (ms)",labelpad=8)
            axis.xaxis.set_major_locator(MaxNLocator(5))
            axis.grid(axis="x",color="#DFE4E8",linewidth=.7)
            axis.tick_params(axis="both",length=0,pad=6)
            for side in ("top","right","left"):axis.spines[side].set_visible(False)
            axis.spines["bottom"].set_color("#BDC6CD")
        axes[0].set_yticks([r[3] for r in ARMS],[r[2] for r in ARMS],fontsize=9.3)
        axes[0].text(-.50,-.65,"Healthy",transform=axes[0].get_yaxis_transform(),ha="left",va="bottom",fontsize=10.5,fontweight="bold",color="#34414C",clip_on=False)
        axes[0].text(-.50,3.45,"D-group PF4 fault",transform=axes[0].get_yaxis_transform(),ha="left",va="bottom",fontsize=10.5,fontweight="bold",color="#34414C",clip_on=False)
        handles=[Line2D([0],[0],color="#202B33",marker="o",markerfacecolor="white",markersize=4,linewidth=1,
            label="Bar / dot: mean     Whiskers: observed min–max across nine samples")]
        fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.025,.15),frameon=False,fontsize=9,handlelength=2.5)
        fig.text(.035,.09,"Original uses whole-message service; A256 uses 256 KiB chunks with hop-level C4. Their difference includes both changes.",
            fontsize=8.7,color="#45515C")
        fig.text(.035,.043,"Mapping and EIR reduce D latency. A256 slightly reduces mean D latency while increasing mean P completion time.",
            fontsize=8.7,color="#45515C")
        out.mkdir(parents=True,exist_ok=True)
        title="Qwen3-MoE I5 joint P/D: all seven arms, mean and observed range over nine samples"
        png,svg=out/"i5_joint.png",out/"i5_joint.svg"
        fig.savefig(png,dpi=260,bbox_inches="tight",pad_inches=.10,metadata={"Title":title})
        fig.savefig(svg,bbox_inches="tight",pad_inches=.10,metadata={"Title":title,"Date":None})
        plt.close(fig)
    return {p.relative_to(ROOT).as_posix():sha(p) for p in (png,svg)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables",type=Path,default=HERE/"tables")
    parser.add_argument("--out",type=Path,default=HERE/"figures")
    args=parser.parse_args()
    values,sources=inputs(args.tables)
    files=render(values,args.out)
    record=dict(status="passed",joint_samples=63,arms=7,panels=2,plotted_values=values,
        sample_rule="3 release phases x 3 backend seeds per arm; mean and observed min/max, not confidence intervals",
        source_sha256=sources,plot_source_sha256=sha(Path(__file__)),output_sha256=files)
    (args.out/"i5_joint_plot_inputs.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="passed",panels=2,arms=7,joint_samples=63,outputs=list(files))))


if __name__=="__main__":main()
