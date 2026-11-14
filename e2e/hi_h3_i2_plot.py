"""Draw completed Qwen layer and Llama hardware comparisons from H3/I2 CSVs."""
import argparse
import csv
import hashlib
import json
import math
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
PHASES = ("prefill", "decode")
PROTOCOLS = ("random_cpcm", "backend")
COLORS = {"A1": "#8B959E", "A3h": "#277DA8", "Gemini": "#D4A34B"}
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
         "axes.labelsize": 10, "legend.fontsize": 9, "svg.fonttype": "none",
         "svg.hashsalt": "crisp-h3-i2-v1", "figure.facecolor": "white",
         "axes.facecolor": "white", "axes.axisbelow": True}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axes_style(axis):
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.grid(axis="y", color="#E3E7EB", linewidth=.7)
    axis.tick_params(axis="both", length=0, pad=6)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#B8C1C9")


def reference_marker():
    return Line2D([0], [0], marker="D", markersize=5.5, markerfacecolor="white",
                  markeredgecolor="#202B33", color="none", label="Backend-mean selection (reference)")


def save(fig, out, stem, title):
    metadata = {"Title": title}
    fig.savefig(out / (stem + ".png"), dpi=240, metadata=metadata, bbox_inches="tight", pad_inches=.08)
    fig.savefig(out / (stem + ".svg"), metadata=dict(metadata, Date=None), bbox_inches="tight", pad_inches=.08)
    plt.close(fig)
    return {name: sha(out / name) for name in (stem + ".png", stem + ".svg")}


def qwen_values(layers_path, operators_path):
    layers = [r for r in read_csv(layers_path) if r["model"] == "qwen3moe" and r["hardware_profile"] == "busybarn"]
    operators = [r for r in read_csv(operators_path) if r["model"] == "qwen3moe" and r["hardware_profile"] == "busybarn"]
    assert len(layers) == 12 and len(operators) == 24
    assert all(int(r["completed"]) == 14 for r in operators)
    lookup = {(r["protocol"], r["method"]): r for r in layers}
    assert len(lookup) == 12
    values = []
    for method in METHODS:
        for protocol in PROTOCOLS:
            row = lookup[protocol, method]
            assert row["shape"] == "2x2" and int(row["batch"]) == 16
            assert int(row["mha_multiplier"]) == 4 and int(row["kv_length"]) == 2048
            assert int(row["prefill_norm_multiplier"]) == 2 and int(row["decode_norm_multiplier"]) == 1
            for phase in PHASES:
                total = float(row[phase + "_ns"])
                assembled = float(row[phase + "_operator_sum_ns"]) + float(row[phase + "_handoff_ns"])
                assert math.isclose(total, assembled, rel_tol=1e-12)
                values.append(dict(method=method, protocol=protocol, phase=phase, latency_ns=total,
                    latency_ms=total/1e6, handoff_ns=float(row[phase + "_handoff_ns"]),
                    chosen_operators=json.loads(row[phase + "_sources"])))
        for phase in PHASES:
            assert float(lookup["backend", method][phase + "_ns"]) <= float(lookup["random_cpcm", method][phase + "_ns"]) + 1e-6
    return values


def draw_qwen(values, out):
    lookup = {(r["phase"], r["method"], r["protocol"]): r for r in values}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.5))
    fig.subplots_adjust(left=.095, right=.975, bottom=.275, top=.74, wspace=.32)
    fig.suptitle("Qwen3-MoE: healthy one-layer latency", x=.095, y=.97, ha="left", fontsize=14, fontweight="bold")
    fig.text(.095, .905, "BusyBarn 2 × 2 die group · batch 16 · 14 candidates per operator and method",
             fontsize=9.5, color="#45515C")
    fig.legend(handles=[Patch(facecolor=COLORS["A1"], label="Random-tie CPCM selection"), reference_marker()],
               loc="upper left", bbox_to_anchor=(.086, .875), frameon=False, ncol=2, columnspacing=1.7)
    for axis, phase, title in zip(axes, PHASES, ("Prefill · sequence 2048", "Decode · KV length 2048")):
        primary = [lookup[phase, method, "random_cpcm"]["latency_ms"] for method in METHODS]
        reference = [lookup[phase, method, "backend"]["latency_ms"] for method in METHODS]
        bars = axis.bar(range(3), primary, width=.58, color=[COLORS[m] for m in METHODS], edgecolor="white", linewidth=.8)
        axis.scatter(range(3), reference, marker="D", s=32, facecolors="white", edgecolors="#202B33", linewidths=1.15, zorder=4)
        for bar, value in zip(bars, primary):
            axis.annotate(f"{value:.3g}", (bar.get_x()+bar.get_width()/2, value), xytext=(0, 7),
                          textcoords="offset points", ha="center", va="bottom", fontsize=10)
        axis.set_xticks(range(3), METHODS)
        axis.set_title(title, loc="left", pad=12)
        axis.set_ylabel("One-layer latency (ms)")
        axis.set_ylim(0, max(primary)*1.23)
        axis.set_xlim(-.65, 2.65)
        axes_style(axis)
    fig.text(.095, .165, "Layer assembly: prefill = 2 LN + 4 MHA + projection + FFN; decode = 1 LN + 4 MHA + projection + FFN.",
             fontsize=8.3, color="#45515C")
    fig.text(.095, .119, "Each MHA subgraph represents 8 Q heads + 1 KV group. Selected operators use ten-seed backend means.",
             fontsize=8.3, color="#45515C")
    fig.text(.095, .073, "Includes one analytical adjacent-group activation transfer. Both panels are fault-free; y-axis scales differ.",
             fontsize=8.3, color="#45515C")
    return save(fig, out, "h3_i2_qwen_layer", "Qwen3-MoE healthy assembled layer: random-tie CPCM selection and backend reference")


def hardware_values(path):
    rows = [r for r in read_csv(path) if r["model"] == "llama"]
    assert len(rows) == 32 and all(int(r["candidates_per_hardware"]) == 14 for r in rows)
    lookup = {(r["phase"], r["op"], r["method"], r["protocol"]): r for r in rows}
    assert len(lookup) == 32
    values = []
    for phase in PHASES:
        for op in ("mha", "ffn"):
            baseline = lookup[phase, op, "A1", "random_cpcm"]
            denominator = float(baseline["busybarn_ns"])
            assert denominator > 0
            for method in ("A1", "A3h"):
                for protocol in PROTOCOLS:
                    row = lookup[phase, op, method, protocol]
                    assert math.isclose(float(row["watos_ns"])/float(row["busybarn_ns"])-1,
                                        float(row["watos_relative_to_busybarn"]), abs_tol=1e-12)
                    for hardware in ("busybarn", "watos"):
                        value = float(row[hardware + "_ns"])
                        values.append(dict(phase=phase, op=op, method=method, protocol=protocol, hardware=hardware,
                            latency_ns=value, normalized=value/denominator, candidate=row[hardware + "_candidate"],
                            denominator_ns=denominator, denominator_candidate=baseline["busybarn_candidate"],
                            denominator_rule="same phase and operator: BusyBarn A1 selected by random-tie CPCM"))
                        if protocol == "backend":
                            assert value <= float(lookup[phase, op, method, "random_cpcm"][hardware + "_ns"]) + 1e-6
    assert len(values) == 32
    return values


def draw_hardware(values, out):
    lookup = {(r["phase"], r["op"], r["hardware"], r["method"], r["protocol"]): r for r in values}
    groups = (("mha", "busybarn"), ("mha", "watos"), ("ffn", "busybarn"), ("ffn", "watos"))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.285, top=.73, wspace=.17)
    fig.suptitle("Llama: MHA and FFN across hardware configurations", x=.085, y=.97, ha="left", fontsize=14, fontweight="bold")
    fig.text(.085, .905, "Healthy operator results · batch 8 · bars use random-tie CPCM selection · 14 candidates per method",
             fontsize=9.2, color="#45515C")
    fig.legend(handles=[Patch(facecolor=COLORS[m], label=m) for m in ("A1", "A3h")]+[reference_marker()],
               loc="upper left", bbox_to_anchor=(.076, .875), frameon=False, ncol=3, columnspacing=1.8)
    ymax = max(r["normalized"] for r in values)*1.22
    width = .32
    for axis, phase, title in zip(axes, PHASES, ("Prefill · sequence 2048", "Decode · KV length 2048")):
        for mi, method in enumerate(("A1", "A3h")):
            positions = [index + (mi-.5)*width for index in range(4)]
            primary = [lookup[phase, op, hardware, method, "random_cpcm"]["normalized"] for op, hardware in groups]
            reference = [lookup[phase, op, hardware, method, "backend"]["normalized"] for op, hardware in groups]
            bars = axis.bar(positions, primary, width=width*.94, color=COLORS[method], edgecolor="white", linewidth=.7)
            axis.scatter(positions, reference, marker="D", s=20, facecolors="white", edgecolors="#202B33", linewidths=.95, zorder=4)
            for bar, value in zip(bars, primary):
                axis.annotate(f"{value:.2f}", (bar.get_x()+bar.get_width()/2, value), xytext=(0, 5),
                              textcoords="offset points", ha="center", va="bottom", fontsize=8)
        axis.axhline(1, color="#737F89", linewidth=.7, linestyle=(0, (3, 3)), zorder=0)
        axis.axvline(1.5, color="#D8DEE3", linewidth=.7, zorder=0)
        axis.set_xticks(range(4), ["MHA\nBusyBarn", "MHA\nWATOS", "FFN\nBusyBarn", "FFN\nWATOS"], fontsize=8.8)
        axis.set_title(title, loc="left", pad=12)
        axis.set_xlim(-.65, 3.65)
        axis.set_ylim(0, ymax)
        axes_style(axis)
    axes[0].set_ylabel("Latency / BusyBarn A1")
    fig.text(.085, .17, "Each operator/phase uses its own BusyBarn A1 bar as 1.0. Bars and diamonds share that fixed denominator.",
             fontsize=8.3, color="#45515C")
    fig.text(.085, .122, "MHA denotes one 8Q + 1KV representative subgraph; FFN denotes one native operator. Values are ten-seed backend means.",
             fontsize=8.3, color="#45515C")
    fig.text(.085, .074, "WATOS covers MHA/FFN operators only. This figure does not assemble a complete WATOS layer or add activation handoff.",
             fontsize=8.3, color="#45515C")
    return save(fig, out, "h3_i2_llama_hardware", "Llama MHA and FFN operator comparison: BusyBarn and WATOS, healthy")


def render(tables, out, which="all"):
    paths = dict(layers=tables/"h3_i2_qwen_layers.csv", operators=tables/"h3_i2_operators.csv",
                 hardware=tables/"h3_i2_hardware_comparison.csv")
    include_qwen = which in ("all", "qwen")
    include_hardware = which in ("all", "llama_hardware")
    qwen = qwen_values(paths["layers"], paths["operators"]) if include_qwen else []
    hardware = hardware_values(paths["hardware"]) if include_hardware else []
    used_paths = [paths[key] for key in ("layers", "operators") if include_qwen]
    if include_hardware:
        used_paths.append(paths["hardware"])
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        outputs = {}
        if include_qwen:
            outputs.update(draw_qwen(qwen, out))
        if include_hardware:
            outputs.update(draw_hardware(hardware, out))
    evidence = dict(which=which, input_sha256={str(path.relative_to(ROOT)): sha(path) for path in used_paths},
        output_sha256=outputs, qwen_plotted_values=qwen, llama_hardware_plotted_values=hardware,
        primary_selector="random_cpcm", reference_selector="backend", backend_samples_per_selected_operator=10)
    if include_qwen:
        evidence["qwen_layer"] = dict(model="qwen3moe", batch=16, mha_layer_multiplier=4, represented_q_heads=8,
            represented_kv_groups=1, fault="healthy", aggregation="selected operator means plus analytical adjacent-group handoff")
    if include_hardware:
        evidence["llama_hardware"] = dict(model="llama", batch=8, scope="MHA/FFN operator only", complete_layer=False,
            handoff_added=False, fault="healthy", reference_denominator="BusyBarn A1 random-CPCM choice, fixed for both selectors")
    evidence_name = "h3_i2_plot_inputs.json" if which == "all" else f"h3_i2_{which}_plot_inputs.json"
    (out/evidence_name).write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=HERE/"tables")
    parser.add_argument("--out", type=Path, default=HERE/"figures")
    parser.add_argument("--which", choices=("all", "qwen", "llama_hardware"), default="all")
    args = parser.parse_args()
    evidence = render(args.tables.resolve(), args.out.resolve(), args.which)
    print(json.dumps(dict(status="passed", qwen_values=len(evidence["qwen_plotted_values"]),
                          hardware_values=len(evidence["llama_hardware_plotted_values"]), output=str(args.out)), ensure_ascii=False))


if __name__ == "__main__":
    main()
