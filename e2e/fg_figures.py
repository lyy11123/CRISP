"""Plot F/G summary tables; this script does not run simulations."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("A1", "A3h", "Gemini")
ARMS = ("B-R0", "-DSTU", "-priority", "-EIR", "CRISP")
COLORS = {"A1": "#7895B2", "A3h": "#167C80", "Gemini": "#D99652",
          "B-R0": "#8C9BAA", "-DSTU": "#D39B56", "-priority": "#866EB0",
          "-EIR": "#CB6A66", "CRISP": "#167C80", "A3h-R0": "#D39B56", "EIR": "#167C80"}
LABELS = {"wsc": "WSC-LLM", "watos": "WATOS", "face": "FACE", "busybarn": "BusyBarn"}


def read_rows(folder, name):
    return json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))


def short(row):
    if row["stage"] in ("F", "F_bw"):
        return LABELS[row["hardware_profile"]]
    return row["shape"].replace("x", "×") + (" fixed" if row["split_mode"] == "fixed" else " finer")


def configuration_order(rows):
    rows = {r["config_id"]: r for r in rows}
    def key(item):
        row = item[1]
        return (0 if row["stage"] == "F" else 1,
                {"wsc": 0, "watos": 1, "face": 2, "busybarn": 3}[row["hardware_profile"]],
                row["shape"], row["split_mode"])
    return dict(sorted(rows.items(), key=key))


def save(fig, out, name, source_rows, description):
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (out / f"{name}.json").write_text(json.dumps(dict(description=description, rows=source_rows),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(dict.fromkeys(k for row in source_rows for k in row))
    with (out / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in source_rows)
    return dict(figure=name, plotted_rows=len(source_rows), description=description,
                outputs=[str(out / f"{name}.{ext}") for ext in ("png", "svg", "csv", "json")])


def style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)


def hardware(tables, out):
    layers = [r for r in read_rows(tables, "layers") if r["stage"] == "F"
              and r["family"] == "healthy" and r["protocol"] == "backend_best"]
    faults = [r for r in read_rows(tables, "fault_layers") if r["stage"] == "F"]
    configs = configuration_order(layers)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    x = np.arange(len(configs))
    plotted = []
    for index, method in enumerate(METHODS):
        rows = [next(r for r in layers if r["config_id"] == c and r["method"] == method) for c in configs]
        axes[0].bar(x+(index-1)*.24, [r["request_2048_ns"]/1e6 for r in rows], .23,
                    label=method, color=COLORS[method])
        plotted.extend(dict(panel="healthy_request", **r) for r in rows)
    axes[0].set_xticks(x, [short(r) for r in configs.values()])
    axes[0].set_title("(a) Healthy layer: prefill + 2048 decode steps")
    style(axes[0], "Fixed-KV request time (ms)")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    for index, scheme in enumerate(("A3h-R0", "EIR")):
        stats = []
        for config in configs:
            rows = [r for r in faults if r["config_id"] == config and r["scheme"] == scheme]
            values = [100*r["request_2048_change_vs_healthy"] for r in rows]
            center = statistics.median(values)
            stats.append((center, center-min(values), max(values)-center))
            plotted.extend(dict(panel="fault_recovery", **r) for r in rows)
        axes[1].bar(x+(index-.5)*.29, [v[0] for v in stats], .28, label=scheme,
                    color=COLORS[scheme], yerr=np.array([[v[1] for v in stats], [v[2] for v in stats]]),
                    capsize=3, error_kw=dict(lw=1))
    axes[1].set_xticks(x, [short(r) for r in configs.values()])
    axes[1].set_title("(b) Four physical faults: median and range")
    style(axes[1], "Request change from paired healthy (%)")
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].legend(frameon=False, ncol=2, fontsize=9)
    return save(fig, out, "fg_hardware_recovery", plotted,
        "Left: backend-best single-layer fixed-KV request. Right: median and min/max across four physical faults; EIR uses the mean of five SA states for L2, or the single stopped R0/L1 state.")


def bandwidth(tables, out):
    searches = [r for r in read_rows(tables, "bandwidth_search") if r["protocol"] == "backend_best"]
    joint = read_rows(tables, "bandwidth_joint")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    plotted = []
    for method in ("A1", "A3h"):
        for op, linestyle in (("mha", "-"), ("ffn", "--")):
            rows = sorted([r for r in searches if r["method"] == method and r["op"] == op], key=lambda r:r["d2d_scale"])
            axes[0].plot([r["d2d_scale"] for r in rows], [1+r["relative_to_1x"] for r in rows],
                         marker="o", color=COLORS[method], ls=linestyle, label=f"{method} {op.upper()}")
            plotted.extend(dict(panel="new_search", **r) for r in rows)
    axes[0].set_title("(a) Decode operators, searched at each bandwidth")
    style(axes[0], "Operator time / its 1× value")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    for arm in ("B-R0", "-DSTU", "-priority", "CRISP"):
        rows = sorted([r for r in joint if r["arm"] == arm], key=lambda r:r["d2d_scale"])
        axes[1].plot([r["d2d_scale"] for r in rows], [r["D_latency_ns_mean"]/1000 for r in rows],
                     marker="o", color=COLORS[arm], label=arm)
        plotted.extend(dict(panel="frozen_mapping_joint", **r) for r in rows)
    axes[1].set_title("(b) Joint run, frozen 1× mapping, g = 0.5")
    style(axes[1], "Decode-chain latency (μs)")
    axes[1].legend(frameon=False, ncol=2, fontsize=9)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([.5,1,2], ["0.5×", "1×", "2×"])
        ax.set_xlabel("WATOS D2D bandwidth")
    return save(fig, out, "fg_bandwidth", plotted,
        "Separate experiments: re-searched decode operators and frozen-WATOS-1x-mapping joint execution. Joint points average three backend seeds.")


def scaling(tables, out):
    layers = [r for r in read_rows(tables, "layers") if r["stage"] == "G"
              and r["family"] == "healthy" and r["protocol"] == "backend_best"]
    candidates = [r for r in read_rows(tables, "candidates") if r["stage"] == "G"
              and r["family"] == "healthy" and r["method"] == "A3h"
              and r["phase"] == "prefill" and r["op"] == "mha" and r["kind"] == "search"]
    receipts={r["job_id"]:r for r in read_rows(tables,"job_timing")}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    plotted = []
    for method in METHODS:
        for split, linestyle in (("fixed", "-"), ("growth", "--")):
            rows = sorted([r for r in layers if r["method"] == method and r["split_mode"] == split],key=lambda r:r["shape"])
            cores = [np.prod([int(n) for n in r["shape"].split("x")])*16 for r in rows]
            axes[0].plot([64,*cores], [1,*[r["request_2048_speedup_vs_64cores"] for r in rows]],
                         marker="o", color=COLORS[method], ls=linestyle,
                         label=method+(" fixed" if split=="fixed" else " finer"))
            plotted.extend(dict(panel="speedup", **r) for r in rows)
    axes[0].plot([64,144], [1,2.25], color="#b0b0b0", ls=":", label="linear")
    axes[0].set_xticks([64,96,144])
    axes[0].set_xlabel("Cores per Die group")
    axes[0].set_title("(a) Fixed-work request speedup")
    style(axes[0], "Speedup over matching E 2×2 method")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    configs = configuration_order(candidates)
    values, ranges, memory = [], [], []
    for config in configs:
        rows = [r for r in candidates if r["config_id"] == config]
        times = [receipts[r["candidate_id"]]["wall_seconds"]/60 for r in rows]
        center = statistics.median(times)
        values.append(center)
        ranges.append((center-min(times), max(times)-center))
        memory.append(max(receipts[r["candidate_id"]]["peak_rss_gb"] for r in rows))
        plotted.extend(dict(panel="measured_job_cost", **r,
            job_wall_seconds=receipts[r["candidate_id"]]["wall_seconds"],
            job_peak_rss_gb=receipts[r["candidate_id"]]["peak_rss_gb"],
            source_receipt=receipts[r["candidate_id"]]["source_receipt"]) for r in rows)
    x = np.arange(len(configs))
    axes[1].bar(x, values, .55, color=COLORS["A3h"], yerr=np.array(ranges).T, capsize=3, label="job time")
    axes[1].set_xticks(x, [short(r).replace(" ","\n") for r in configs.values()])
    axes[1].set_title("(b) A3h prefill MHA: 14 concurrent-run jobs")
    style(axes[1], "Median job time, min–max (minutes)")
    memax = axes[1].twinx()
    memax.plot(x, memory, "o--", color=COLORS["Gemini"], label="peak RSS")
    memax.set_ylabel("Largest per-job peak RSS (GiB)")
    memax.spines["top"].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    mh, ml = memax.get_legend_handles_labels()
    axes[1].legend(handles+mh, labels+ml, frameon=False, fontsize=9, loc="upper left")
    return save(fig, out, "fg_scaling_cost", plotted,
        "Strong scaling with unchanged effective workload. Runtime panel uses full job wall time from the parallel run, including search and evaluation; not serial algorithm timing.")


def joint_increment(tables, out):
    rows = [r for r in read_rows(tables, "joint_summary") if r["stage"] in ("F", "G")
            and r["condition"] == "D" and r["phase"] == .5]
    configs = configuration_order(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(len(configs)); width=.15
    plotted = []
    for i, arm in enumerate(ARMS):
        values = [next(r for r in rows if r["config_id"] == c and r["arm"] == arm) for c in configs]
        refs = [next(r for r in rows if r["config_id"] == c and r["arm"] == "B-R0")["D_latency_ns_mean"] for c in configs]
        prefs = [next(r for r in rows if r["config_id"] == c and r["arm"] == "-DSTU")["P_completion_ns_mean"] for c in configs]
        axes[0].bar(x+(i-2)*width,[r["D_latency_ns_mean"]/ref for r,ref in zip(values,refs)],
                    width*.95, label=arm, color=COLORS[arm])
        axes[1].bar(x+(i-2)*width,[r["P_completion_ns_mean"]/ref for r,ref in zip(values,prefs)],
                    width*.95, label=arm, color=COLORS[arm])
        plotted.extend(dict(panel="D_fault_g0.5", **r) for r in values)
    for ax in axes:
        ax.set_xticks(x, [short(r).replace(" finer", "\nfiner") for r in configs.values()], fontsize=9)
    style(axes[0], "Decode-chain latency / B-R0")
    style(axes[1], "Prefill completion time / −DSTU")
    axes[0].axhline(1, color="black", lw=.7, ls=":")
    axes[1].axhline(1, color="black", lw=.7, ls=":")
    axes[0].set_title("(a) Internal D2D fault, g = 0.5")
    axes[1].set_title("(b) Prefill completion with the same arrival")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=10)
    return save(fig, out, "fg_joint_increment", plotted,
        "D-group D2D fault at the common g=0.5 arrival. Left: decode latency relative to B-R0. Right: prefill completion relative to -DSTU; CRISP pays a prefill cost. Bars average three backend seeds. Paired solo changes and all arrival phases remain in joint_summary.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=ROOT/"e2e/fg/tables")
    parser.add_argument("--out", type=Path, default=ROOT/"e2e/fg/figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.titlesize":11,
                         "svg.fonttype":"none", "figure.dpi":110})
    result = [f(args.tables, args.out) for f in (hardware, bandwidth, scaling, joint_increment)]
    (args.out/"figure_manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
