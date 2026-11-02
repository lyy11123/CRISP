"""
Pie chart of transformer block wall-clock time breakdown from sbatch log files.

Reads profile sections from existing log files, sums with weights
(2*LN + MHA + Proj + FFN), and draws a single pie chart with 6 slices.

Usage:
    python plot_pie_time.py
    python plot_pie_time.py --topo 4x4 --chiplet ch1x4 --bw bw1024 --sp sp8
"""
import os
import re
import glob
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

file_path = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(file_path, "sbatch_log")
RESULTS_DIR = os.path.join(file_path, "results")

OPERATORS = ["ln", "mha", "proj", "ffn"]
WEIGHTS = {"ln": 2, "mha": 1, "proj": 1, "ffn": 1}

SLICES = [
    ("Path Profiling\nSection V-B1",            ["init_hardware"]),
    ("Building Events\nSection III-A", ["build_operator", "build_event"]),
    ("Mapping\nSection IV",           ["stream_mapping"]),
    ("BALD Routing\nSection V-B2",         ["reroute_dijkstra"]),
    ("Backend\nSection VI-A",             ["event_driver"]),
    ("Others\nFile operations",                   ["deepcopy_event", "add_broadcast",
                                  "barrier_sync", "save_records"]),
]

SLICE_COLORS = ["#118ab2", "#f77f00", "#06d6a0", "#ef476f", "#ffd166", "#adb5bd"]


def find_log_file(op, variant, chiplet, bw, topo, sp, barrier, reroute):
    b_tag = "_barrier" if barrier else ""
    r_tag = "_reroute" if reroute else ""
    pattern = f"{op}_{variant}_{chiplet}_{bw}_{topo}_{sp}{b_tag}{r_tag}_[0-9]*.txt"
    matches = glob.glob(os.path.join(LOG_DIR, pattern))
    if not matches:
        return None
    matches.sort()
    return matches[-1]


def parse_profile(fpath):
    """Parse === Profile (wall-clock) === section from a log file."""
    with open(fpath) as f:
        text = f.read()

    timings = {}
    in_profile = False
    for line in text.splitlines():
        if "=== Profile (wall-clock) ===" in line:
            in_profile = True
            continue
        if in_profile:
            if line.startswith("TOTAL") or line.startswith("─"):
                continue
            m = re.match(r"(\S+)\s+([\d.]+)\s+[\d.]+%", line)
            if m:
                timings[m.group(1)] = float(m.group(2))
    return timings if timings else None


def main():
    parser = argparse.ArgumentParser(description="Pie chart of time breakdown from logs")
    parser.add_argument("--topo", default="16x16")
    parser.add_argument("--chiplet", default="ch1x1")
    parser.add_argument("--bw", default="bw96")
    parser.add_argument("--sp", default="sp8")
    parser.add_argument("--variant", default="busybarn")
    parser.add_argument("--barrier", action="store_true")
    parser.add_argument("--reroute", action="store_true", default=True)
    parser.add_argument("--no-reroute", dest="reroute", action="store_false")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams['font.family'] = 'sans-serif'

    b_str = "+barrier" if args.barrier else ""
    r_str = "+reroute" if args.reroute else ""
    print(f"Config: {args.topo} / {args.chiplet} / {args.bw} / {args.sp} / "
          f"{args.variant}{b_str}{r_str}")

    all_timings = {}
    for op in OPERATORS:
        fpath = find_log_file(op, args.variant, args.chiplet, args.bw,
                              args.topo, args.sp, args.barrier, args.reroute)
        if fpath is None:
            print(f"  MISSING log for {op}")
            return
        timings = parse_profile(fpath)
        if timings is None:
            print(f"  NO PROFILE in {os.path.basename(fpath)}")
            return
        total = sum(timings.values())
        print(f"  {op}: {total:.1f}s  ({os.path.basename(fpath)})")
        all_timings[op] = timings

    slice_values = []
    for label, phases in SLICES:
        total = 0.0
        for op in OPERATORS:
            w = WEIGHTS[op]
            for phase in phases:
                total += all_timings[op].get(phase, 0) * w
        slice_values.append(total)

    grand_total = sum(slice_values)
    print(f"\nWeighted totals (2*LN + MHA + Proj + FFN):")
    for (label, _), val in zip(SLICES, slice_values):
        pct = val / grand_total * 100 if grand_total > 0 else 0
        print(f"  {label.replace(chr(10), ' '):<30s} {val:>8.1f}s  ({pct:.1f}%)")
    print(f"  {'TOTAL':<30s} {grand_total:>8.1f}s")

    labels = [s[0] for s in SLICES]
    fig, ax = plt.subplots(figsize=(12, 6))
    explode = [0] * len(slice_values)
    explode[1] = 0.15
    explode[3] = 0.15
    explode[-1] = 0.15  
    wedges, _, autotexts = ax.pie(
        slice_values, colors=SLICE_COLORS,
        explode=explode,
        autopct='',
        startangle=180, pctdistance=0.75,
        wedgeprops=dict(edgecolor='white', linewidth=1.5))

    SMALL_LABEL_POS = {
        0: (-1.1, -0.15),
        1: (-1.1, -0.7),
        3: (-0.9, 0.9),
        5: (-1.1, 0.6),
    }
    total = sum(slice_values)
    for i, (wedge, val) in enumerate(zip(wedges, slice_values)):
        pct = val / total * 100 if total > 0 else 0
        angle = (wedge.theta2 + wedge.theta1) / 2

        if pct >= 5:
            x = 0.75 * wedge.r * plt.np.cos(plt.np.deg2rad(angle))
            y = 0.74 * wedge.r * plt.np.sin(plt.np.deg2rad(angle))
            ax.text(x, y, f'{pct:.1f}%', ha='center', va='center',
                   fontsize=24, fontweight='bold')
        else:
            x1 = 1.0 * wedge.r * plt.np.cos(plt.np.deg2rad(angle))
            y1 = 1.0 * wedge.r * plt.np.sin(plt.np.deg2rad(angle))
            if i in SMALL_LABEL_POS:
                x2, y2 = SMALL_LABEL_POS[i]
            else:
                x2 = 1.1 * wedge.r * plt.np.cos(plt.np.deg2rad(angle))
                y2 = 1.6 * wedge.r * plt.np.sin(plt.np.deg2rad(angle))

            ax.plot([x1, x2], [y1, y2], color='gray', linewidth=2)
            ha = 'left' if x2 > 0 else 'right'
            ax.text(x2, y2, f'{pct:.1f}%', ha=ha, va='center',
                   fontsize=24, fontweight='bold')

    first_angle = (wedges[0].theta1 + wedges[0].theta2 * 2) / 3
    last_angle = (wedges[-2].theta1 + wedges[-2].theta2 * 1.2 ) / 2.2
    radius = 0.36

    angles = plt.np.linspace(plt.np.deg2rad(first_angle),
                            plt.np.deg2rad(last_angle), 200)
    x_arc = radius * plt.np.cos(angles)
    y_arc = radius * plt.np.sin(angles)

    ax.plot(x_arc, y_arc, color='grey', linewidth=6, linestyle='--')

    arrow_angle = plt.np.deg2rad(last_angle)
    arrow_x = radius * plt.np.cos(arrow_angle)
    arrow_y = radius * plt.np.sin(arrow_angle)
    dx = -plt.np.sin(arrow_angle) * 0.08
    dy = plt.np.cos(arrow_angle) * 0.08
    ax.arrow(arrow_x - dx, arrow_y - dy, dx, dy,
            head_width=0.06, head_length=0.1, fc='grey', ec='grey', linewidth=6)

    ax.set_position([0.0, 0.05, 0.55, 0.9])

    leg1 = fig.legend(wedges, labels, loc='upper left', bbox_to_anchor=(0.445, 0.87),
                      ncol=1, fontsize=20, frameon=False)

    arrow_handle = Line2D([0], [0], color='grey', linewidth=4, linestyle='--',
                          marker='>', markersize=14, markeredgecolor='grey')
    fig.legend([arrow_handle], ["Execution Order"],
               loc='lower left', bbox_to_anchor=(0.12, 0.05),
               fontsize=20, frameon=False)

    b_tag = "_barrier" if args.barrier else ""
    r_tag = "_reroute" if args.reroute else ""
    base = f"pie_time_{args.variant}_{args.chiplet}_{args.bw}_{args.topo}_{args.sp}{b_tag}{r_tag}"
    out_pdf = os.path.join(RESULTS_DIR, base + ".pdf")
    out_png = os.path.join(RESULTS_DIR, base + ".png")
    plt.savefig(out_pdf, dpi=300)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {out_png}")
    print(f"Saved → {out_pdf}")


if __name__ == "__main__":
    main()
