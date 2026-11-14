"""Plot completed, assembled Llama health results from next_layers.csv only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "e2e/hi"
METHODS = ("A1", "A3h", "Gemini")
COLORS = ("#8B959E", "#277DA8", "#D4A34B")
PHASES = ("prefill", "decode")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def inputs(layers_path, operators_path):
    layers = read_csv(layers_path)
    rows = [row for row in layers if row["model"] == "llama" and row["hardware_profile"] == "busybarn"]
    assert len(rows) == 12
    lookup = {(row["protocol"], row["method"]): row for row in rows}
    assert len(lookup) == 12
    operators = [row for row in read_csv(operators_path)
                 if row["model"] == "llama" and row["hardware_profile"] == "busybarn"]
    assert len(operators) == 24 and all(int(row["completed"]) == 14 for row in operators)
    values = []
    for method in METHODS:
        for protocol in ("random_cpcm", "backend"):
            row = lookup[protocol, method]
            assert row["shape"] == "2x2" and int(row["batch"]) == 8
            assert int(row["mha_multiplier"]) == 8 and int(row["kv_length"]) == 2048
            assert int(row["prefill_norm_multiplier"]) == 2 and int(row["decode_norm_multiplier"]) == 1
            for phase in PHASES:
                total = float(row[phase + "_ns"])
                summed = float(row[phase + "_operator_sum_ns"]) + float(row[phase + "_handoff_ns"])
                assert abs(total - summed) <= 1e-7 * max(1, total)
                values.append(dict(method=method, protocol=protocol, phase=phase, latency_ns=total,
                    latency_ms=total / 1e6, handoff_ns=float(row[phase + "_handoff_ns"]),
                    chosen_operators=json.loads(row[phase + "_sources"])))
        for phase in PHASES:
            assert float(lookup["backend", method][phase + "_ns"]) <= float(lookup["random_cpcm", method][phase + "_ns"]) + 1e-6
    return values


def render(layers_path, operators_path, out):
    values = inputs(layers_path, operators_path)
    lookup = {(row["phase"], row["method"], row["protocol"]): row for row in values}
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
             "axes.labelsize": 10, "legend.fontsize": 9, "svg.fonttype": "none",
             "figure.facecolor": "white", "axes.facecolor": "white", "axes.axisbelow": True}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
        fig.subplots_adjust(left=.095, right=.975, bottom=.275, top=.74, wspace=.32)
        fig.suptitle("Llama: healthy one-layer latency", x=.095, y=.97, ha="left", fontsize=14, fontweight="bold")
        fig.text(.095, .905, "BusyBarn 2 × 2 die group · batch 8 · 14 candidates per operator and method",
                 fontsize=9.5, color="#45515C")
        legend = [Patch(facecolor="#8B959E", label="Random-tie CPCM selection"),
                  Line2D([0], [0], marker="D", markersize=5.5, markerfacecolor="white",
                         markeredgecolor="#202B33", color="none", label="Backend-mean selection (reference)")]
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.086, .875), frameon=False,
                   ncol=2, columnspacing=1.7, handletextpad=.7)
        for axis, phase, title in zip(axes, PHASES, ("Prefill · sequence 2048", "Decode · KV length 2048")):
            primary = [lookup[phase, method, "random_cpcm"]["latency_ms"] for method in METHODS]
            reference = [lookup[phase, method, "backend"]["latency_ms"] for method in METHODS]
            bars = axis.bar(range(3), primary, width=.58, color=COLORS, edgecolor="white", linewidth=.8)
            axis.scatter(range(3), reference, marker="D", s=32, facecolors="white", edgecolors="#202B33",
                         linewidths=1.15, zorder=4)
            for bar, value in zip(bars, primary):
                axis.annotate(f"{value:.3g}", (bar.get_x() + bar.get_width()/2, value),
                              xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", fontsize=10)
            axis.set_xticks(range(3), METHODS)
            axis.set_title(title, loc="left", pad=12)
            axis.set_ylabel("One-layer latency (ms)")
            axis.set_ylim(0, max(primary)*1.23)
            axis.set_xlim(-.65, 2.65)
            axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.grid(axis="y", color="#E3E7EB", linewidth=.7)
            axis.tick_params(axis="both", length=0, pad=6)
            for side in ("top", "right", "left"):
                axis.spines[side].set_visible(False)
            axis.spines["bottom"].set_color("#B8C1C9")
        fig.text(.095, .165, "Layer assembly: prefill = 2 LN + 8 MHA + projection + FFN; decode = 1 LN + 8 MHA + projection + FFN.",
                 fontsize=8.3, color="#45515C")
        fig.text(.095, .119, "Each MHA subgraph represents 8 Q heads + 1 KV group. Selected operators use ten-seed backend means.",
                 fontsize=8.3, color="#45515C")
        fig.text(.095, .073, "Includes one analytical adjacent-group activation transfer. Both panels are fault-free; y-axis scales differ.",
                 fontsize=8.3, color="#45515C")
        out.mkdir(parents=True, exist_ok=True)
        metadata = {"Title": "Llama healthy one-layer latency: random-tie CPCM and backend selection"}
        fig.savefig(out / "next_health.png", dpi=240, metadata=metadata, bbox_inches="tight", pad_inches=.08)
        fig.savefig(out / "next_health.svg", metadata=dict(metadata, Date=None), bbox_inches="tight", pad_inches=.08)
        plt.close(fig)
    evidence = dict(input_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (layers_path, operators_path)}, values=values,
                    model="llama", workload="W1", fault="healthy", mha_layer_multiplier=8,
                    represented_q_heads=8, represented_kv_groups=1,
                    plotted_metric="sum of selected per-operator ten-seed backend means plus analytical handoff",
                    primary_selector="random_cpcm", reference_selector="backend", source_rows=12)
    (out / "next_health_plot.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=Path, default=HERE / "tables/next_layers.csv")
    parser.add_argument("--operators", type=Path, default=HERE / "tables/next_operators.csv")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    evidence = render(args.layers.resolve(), args.operators.resolve(), args.out.resolve())
    print(json.dumps(dict(status="passed", plotted_rows=len(evidence["values"]), output=str(args.out)), ensure_ascii=False))


if __name__ == "__main__":
    main()
