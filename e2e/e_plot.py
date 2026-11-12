"""Plot completed E-stage tables from e_report.py without running simulations."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


METHODS = ("A1", "A3h", "Gemini")
LABELS = {"A1": "BusyBarn A1", "A3h": "CRISP mapping A3h", "Gemini": "Gemini"}
COLORS = {"A1": "#647588", "A3h": "#16679D", "Gemini": "#C5742B"}
SHAPES = ("2x2", "1x4")
LAYOUTS = ("L_mix", "L_strip")
LAYOUT_LABELS = {"L_mix": "Mixed groups", "L_strip": "Strip groups"}
PHASES = ("prefill", "decode")
PHASE_LABELS = ("(a) Prefill", "(b) One decode step")
STEPS = (128, 512, 2048)
FAULTS = {"PF4": "D2D link\n(PF4)", "PF1": "Core and router\n(PF1)", "PF8": "DDR port\n(PF8)"}
FAULT_STYLES = {
    "A1 R0": ("A1 R0", "#96A4B4"),
    "A1 FULL-BusyBarn": ("A1 FULL", "#495C71"),
    "A3h R0": ("A3h R0", "#75A9CE"),
    "A3h L2Gsync 停止档（均值）": ("A3h + EIR", "#16679D"),
    "A3h FULL-A3h": ("A3h FULL", "#775D95"),
}


def select_rows(rows, ln, fields, expected):
    selected = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if row["ln"] == ln and key in expected:
            if key in selected:
                raise ValueError(f"Duplicate {ln} result for {key}")
            selected[key] = row
    missing = sorted(expected - selected.keys())
    if missing:
        raise ValueError(f"Missing {ln} results for {missing}")
    return selected


def positive_latencies(rows):
    for key, row in rows.items():
        for phase in PHASES:
            value = row.get(phase)
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Incomplete {phase} latency for {key}")


def load_tables(data, ln):
    layers = select_rows(data["layer_table"], ln, ("shape", "method"),
                         {(shape, method) for shape in SHAPES for method in METHODS})
    models = select_rows(data["model_table"], ln, ("layout", "method"),
                         {(layout, method) for layout in LAYOUTS for method in METHODS})
    faults = select_rows([row for row in data["fault_model_table"] if row["group"] == "G2"], ln, ("layout", "fault", "scheme"),
                         {("L_mix", fault, scheme) for fault in FAULTS for scheme in FAULT_STYLES})
    positive_latencies(layers)
    positive_latencies(models)
    for key, row in models.items():
        for n in STEPS:
            if not math.isclose(row[f"request_{n}"], row["prefill"] + n * row["decode"], rel_tol=1e-10):
                raise ValueError(f"Inconsistent request latency for {key}, N={n}")
    for key, row in faults.items():
        if row["group"] != "G2" or row["affected_layers"] != 3:
            raise ValueError(f"Unexpected damaged group for {key}")
        for scope in ("local", "model"):
            if not math.isfinite(row[f"{scope}_request_2048_change"]):
                raise ValueError(f"Missing fault latency change for {key}, {scope}")
    return layers, models, faults


def axes_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9E0E6", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#657382")
    ax.tick_params(color="#657382", labelcolor="#17232E", length=3)


def legend(fig, axes, columns):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=columns, frameon=False)


def phase_bars(data, categories, category_labels, scales, units):
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.9))
    width = 0.23
    for phase, title, scale, unit, ax in zip(PHASES, PHASE_LABELS, scales, units, axes):
        for index, method in enumerate(METHODS):
            x = [i + (index - 1) * width for i in range(len(categories))]
            y = [data[category, method][phase] / scale for category in categories]
            ax.bar(x, y, width=width * 0.88, label=LABELS[method], color=COLORS[method])
        ax.set_xticks(range(len(categories)), category_labels)
        ax.set_title(title, loc="left", pad=6)
        ax.set_ylabel(f"Latency ({unit} cycles)")
        ax.set_ylim(bottom=0)
        axes_style(ax)
    legend(fig, axes, 3)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=2.2)
    return fig


def request_lines(models):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharey=True)
    ymax = 1.05 * max(row[f"request_{STEPS[-1]}"] for row in models.values()) / 1e9
    for i, (layout, ax) in enumerate(zip(LAYOUTS, axes)):
        for method, line, marker in zip(METHODS, ("--", "-", ":"), ("s", "o", "^")):
            row = models[layout, method]
            ax.plot((0, STEPS[-1]), (row["prefill"] / 1e9, row[f"request_{STEPS[-1]}"] / 1e9),
                    label=LABELS[method], color=COLORS[method], linestyle=line, linewidth=1.55)
            ax.scatter(STEPS, [row[f"request_{n}"] / 1e9 for n in STEPS], color=COLORS[method],
                       marker=marker, s=19, zorder=3, clip_on=False)
        ax.set_title(f"({chr(97 + i)}) {LAYOUT_LABELS[layout]}", loc="left", pad=6)
        ax.set_xlabel("Decode steps after prefill")
        ax.set_xlim(0, STEPS[-1])
        ax.set_ylim(0, ymax)
        ax.set_xticks((0, 128, 512, 1024, 2048))
        axes_style(ax)
    axes[0].set_ylabel("Request latency (G cycles)")
    legend(fig, axes, 3)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=2.1)
    return fig


def fault_bars(faults):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.15))
    width = 0.15
    for scope, title, ax in zip(("local", "model"), ("(a) Damaged group (3 layers)", "(b) Full model (44 layers)"), axes):
        for i, (scheme, (label, color)) in enumerate(FAULT_STYLES.items()):
            x = [j + (i - 2) * width for j in range(len(FAULTS))]
            y = [faults["L_mix", fault, scheme][f"{scope}_request_2048_change"] for fault in FAULTS]
            ax.bar(x, y, width=width * 0.9, label=label, color=color)
        ax.set_title(title, loc="left", pad=6)
        ax.set_xticks(range(len(FAULTS)), list(FAULTS.values()))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylabel("Request latency change")
        ax.axhline(0, color="#657382", linewidth=0.7)
        axes_style(ax)
    legend(fig, axes, 5)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=2.0)
    return fig


def save_figure(fig, out, stem, title):
    try:
        svg = out / f"{stem}.svg"
        fig.savefig(svg, bbox_inches="tight", pad_inches=0.07, metadata={"Title": title, "Date": None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        fig.savefig(out / f"{stem}.png", bbox_inches="tight", pad_inches=0.07, dpi=300, metadata={"Title": title})
    finally:
        plt.close(fig)


def render(summary_path, out, ln):
    raw = summary_path.read_bytes()
    layers, models, faults = load_tables(json.loads(raw), ln)
    out.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
             "legend.fontsize": 8.5, "svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white"}
    with plt.rc_context(style):
        save_figure(phase_bars(layers, SHAPES, ("2×2 group", "1×4 group"), (1e6, 1e3), ("M", "k")),
                    out, "e_layer_latency", "GPT W1 one-layer latency by group shape")
        save_figure(phase_bars(models, LAYOUTS, ("Mixed groups", "Strip groups"), (1e9, 1e6), ("G", "M")),
                    out, "e_model_latency", "GPT W1 44-layer latency by wafer layout")
        save_figure(request_lines(models), out, "e_request_latency", "GPT W1 request latency at fixed KV length")
        save_figure(fault_bars(faults), out, "e_fault_latency", "GPT W1 2048-step request latency change under representative faults")

    mode = ("`asis` 沿用 B 原式，decode FFN 内的一次 LN 与外加两次 LN 均计入。" if ln == "asis" else
            "`fixed` 外加一次 decode LN，连同 decode FFN 内的一次共两次。")
    notes = ["# E 阶段图说明", "", mode, "两种 LN 计数的数值均见 `layer_table.csv` 和 `model_table.csv`。",
             "A3h 是在原后端评估的 CRISP 映射。健康图使用各单元 14 个候选中的后端均值最优结果。", "",
             "- `e_layer_latency` 比较 2×2、1×4 组上一个 GPT W1 block 的 prefill 与 decode 一步。各 phase 的纵轴单位独立。",
             "- `e_model_latency` 比较 56 Die 混合和长条布局的 44 层组装时间，包含 13 次串行组间激活传输。组内和组间时间可在 `model_table.csv` 中查看。",
             "- `e_request_latency` 按固定 KV 长度 2048 计算 `R(N) = P_model + N × D_model`。N 为 prefill 后追加的 decode 步数，标点对应 128、512、2048。",
             "- `e_fault_latency` 采用 2048 个 decode 步，以同方法、同起点故障前结果为分母。PF4 为 G2 内双向 D2D 链路故障，PF1 为核和路由器故障，PF8 为 DDR 访问端口故障。左图只计受损的 3 层，右图保留另外 41 层及组间传输，两图的纵轴范围独立。", "",
             f"输入 `summary.json` 的 SHA-256 为 `{hashlib.sha256(raw).hexdigest()}`。SVG 保留文本，字体为 DejaVu Sans。PNG 为 300 dpi。", ""]
    (out / "FIGURE_NOTES.md").write_text("\n".join(notes), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Completed e_report summary.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ln", choices=("asis", "fixed"), default="asis")
    args = parser.parse_args()
    render(args.summary, args.out_dir, args.ln)
    print(f"Wrote four SVG/PNG figure pairs to {args.out_dir} ({args.ln})")


if __name__ == "__main__":
    main()
