"""
Plot transformer block latency for wsc 16x16 bw96 ch1x1 experiments.

Single figure: Barrier (No/Yes) -> 4 bars.
Stacked segments: Compute (blue), Overlap (orange), Comm (green).
Bar heights normalised by Gemini-no-barrier-no-reroute baseline.
Ratio labels relative to Gemini baseline.

Per-op sp: all sp8.
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


file_path = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(file_path, "sbatch_log")
pic_dir = os.path.join(file_path, "results")

OPERATORS = ["ln", "mha", "proj", "ffn"]
WEIGHTS = {"ln": 2, "mha": 1, "proj": 1, "ffn": 1}

CHIPLET = "ch1x1_bw96"
TOPO = "16x16"
PER_OP_SP = {"ffn": "sp8", "proj": "sp8", "mha": "sp8", "ln": "sp8"}

COLORS = {
    'Compute': "#118ab2",
    'Overlap': "#f77f00",
    'Comm':    "#06d6a0",
}
BAR_COLORS = {
    "G+XY":   "#7fc97f",
    "B+XY":   "#beaed4",
    "G+BALD": "#386cb0",
    "B+BALD": "#f0027f",
}

BAR_CONFIGS = [
    ("gemini",   False, "G+XY"),
    ("busybarn", False, "B+XY"),
    ("gemini",   True,  "G+BALD"),
    ("busybarn", True,  "B+BALD"),
]


def parse_sa_time_cost(text):
    m = re.search(
        r"(?:SA \+ reroute_dijkstra|SA) time cost: "
        r"\(([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\)",
        text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def find_log_file(op, variant, sp, barrier, reroute):
    b_tag = "_barrier" if barrier else ""
    r_tag = "_reroute" if reroute else ""
    pattern = (f"{op}_{variant}_{CHIPLET}_{TOPO}_{sp}"
               f"{b_tag}{r_tag}_[0-9]*.txt")
    matches = glob.glob(os.path.join(log_dir, pattern))
    if not matches:
        return None
    matches.sort()
    return matches[-1]


def read_operator(op, variant, sp, barrier, reroute):
    fpath = find_log_file(op, variant, sp, barrier, reroute)
    if fpath is None:
        return None
    with open(fpath) as f:
        text = f.read()
    return parse_sa_time_cost(text)


def get_transformer_block(variant, barrier, reroute, verbose=False):
    """Compute transformer block = 2*LN + MHA + Proj + FFN.
    Returns (total_t, total_c, total_m, op_details) or None."""
    total_t, total_c, total_m = 0.0, 0.0, 0.0
    op_details = {}
    for op in OPERATORS:
        sp = PER_OP_SP[op]
        val = read_operator(op, variant, sp, barrier, reroute)
        if val is None:
            b_str = "+barrier" if barrier else ""
            r_str = "+reroute" if reroute else ""
            if verbose:
                print(f"  MISSING: {op}_{variant}_{CHIPLET}_{TOPO}_{sp}"
                      f"{b_str}{r_str}")
            return None
        w = WEIGHTS[op]
        op_details[op] = val
        total_t += val[0] * w
        total_c += val[1] * w
        total_m += val[2] * w
    return total_t, total_c, total_m, op_details


def decompose(tc):
    """Split (total, pure_comp, pure_comm) into (comp, overlap, comm)."""
    total, compute, comm = tc[0], tc[1], tc[2]
    overlap = total - compute - comm
    return compute, max(overlap, 0), comm


def print_operator_table():
    header = (f"\n{'='*100}\n"
              f"  Topology: {TOPO}  BW: bw96  Chiplet: {CHIPLET}  "
              f"(per-op sp: {PER_OP_SP})\n"
              f"{'='*100}")
    print(header)

    baselines = {}
    for op in OPERATORS:
        sp = PER_OP_SP[op]
        val = read_operator(op, "gemini", sp, False, False)
        baselines[op] = val

    block_baseline = get_transformer_block("gemini", False, False)

    print(f"\n  --- {CHIPLET} / {TOPO} ---")
    print(f"  {'Config':<28} {'Op':<6} {'SP':<6} "
          f"{'Total':>12} {'Comp':>12} {'Comm':>12} "
          f"{'Speedup':>8}")
    print(f"  {'-'*92}")

    for barrier in [False, True]:
        for variant, reroute, label in BAR_CONFIGS:
            result = get_transformer_block(variant, barrier, reroute)
            b_str = "+bar" if barrier else ""
            tag = f"{label}{b_str}"

            if result is None:
                print(f"  {tag:<28} {'BLOCK':<6} {'':<6} "
                      f"{'N/A':>12} {'N/A':>12} {'N/A':>12} "
                      f"{'N/A':>8}")
                continue

            _, _, _, op_details = result

            for op in OPERATORS:
                sp = PER_OP_SP[op]
                val = op_details[op]
                base = baselines.get(op)
                if base and base[0] > 0:
                    spdup = f"{base[0]/val[0]:.2f}x"
                else:
                    spdup = "N/A"
                print(f"  {tag:<28} {op:<6} {sp:<6} "
                      f"{val[0]:>12.0f} {val[1]:>12.0f} "
                      f"{val[2]:>12.0f} {spdup:>8}")

            block_total = result[0]
            if block_baseline and block_baseline[0] > 0:
                bspdup = f"{block_baseline[0]/block_total:.2f}x"
            else:
                bspdup = "N/A"
            comp, ovl, comm = decompose(result)
            print(f"  {tag:<28} {'BLOCK':<6} {'':<6} "
                  f"{block_total:>12.0f} {comp:>12.0f} "
                  f"{comm:>12.0f} {bspdup:>8}")
            print(f"  {'':<28} {'':>6} {'':>6} "
                  f"{'(overlap':>12} {ovl:>12.0f}{')':<12}")


def plot_figure():
    n_bars = len(BAR_CONFIGS)

    global_baseline = get_transformer_block(
        "gemini", barrier=False, reroute=False, verbose=False)
    if global_baseline is None or global_baseline[0] == 0:
        global_norm = 1.0
    else:
        global_norm = global_baseline[0]

    all_data = {}
    for barrier in [False, True]:
        all_data[barrier] = []
        for variant, reroute, label in BAR_CONFIGS:
            tc = get_transformer_block(
                variant, barrier=barrier, reroute=reroute)
            if tc is None:
                all_data[barrier].append((0, 0, 0))
            else:
                comp, ovl, comm = decompose(tc)
                all_data[barrier].append(
                    (comp / global_norm,
                     ovl / global_norm,
                     comm / global_norm))


    bar_w = 0.2
    gap_within_barrier = 0.06
    gap_between_barriers = 0.14

    x_positions = []
    bar_decomps = []
    bar_label_list = []

    barrier_center_positions = []
    barrier_center_labels = []

    x = 0.0
    for barrier in [False, True]:
        b_start = x
        for bar_idx in range(n_bars):
            x_positions.append(x)
            bar_decomps.append(all_data[barrier][bar_idx])
            bar_label_list.append(BAR_CONFIGS[bar_idx][2])
            x += bar_w + gap_within_barrier
        b_end = x - gap_within_barrier
        barrier_center_positions.append((b_start + b_end) / 2)
        barrier_center_labels.append(
            "Dataflow" if not barrier else "Bulk-Synchronous")
        x += gap_between_barriers

    x_positions = np.array(x_positions)
    comps = np.array([d[0] for d in bar_decomps])
    ovls = np.array([d[1] for d in bar_decomps])
    comms = np.array([d[2] for d in bar_decomps])


    fig, ax = plt.subplots(figsize=(12, 6))
    plt.rcParams['font.family'] = 'sans-serif'

    ax.bar(x_positions, comps, bar_w, color=COLORS['Compute'],
           edgecolor='none')
    ax.bar(x_positions, ovls, bar_w, bottom=comps,
           color=COLORS['Overlap'], edgecolor='none')
    ax.bar(x_positions, comms, bar_w, bottom=comps + ovls,
           color=COLORS['Comm'], edgecolor='none')

    MIN_INSIDE = 0.06
    for i in range(len(x_positions)):
        total_norm = comps[i] + ovls[i] + comms[i]
        if total_norm == 0:
            continue
        segs = [
            (comps[i], 0, comps[i] / total_norm, 'Compute'),
            (ovls[i], comps[i], ovls[i] / total_norm, 'Overlap'),
            (comms[i], comps[i] + ovls[i], comms[i] / total_norm, 'Comm'),
        ]
        for height, bottom, pct, _ in segs:
            if pct < 0.005:
                continue
            if height >= MIN_INSIDE:
                ax.text(x_positions[i], bottom + height / 3, f"{pct:.0%}",
                        ha='center', va='center', fontsize=20,
                        color='white', fontweight='bold')
            else:
                ax.text(x_positions[i], bottom,
                        f"{pct:.0%}",
                        ha='center', va='top', fontsize=20,
                        color='white', fontweight='bold')

    n_bars = len(BAR_CONFIGS)
    for grp_idx, barrier in enumerate([False, True]):
        base_i = grp_idx * n_bars
        base_top = comps[base_i] + ovls[base_i] + comms[base_i]
        if base_top == 0:
            continue
        for bar_offset in range(1, n_bars):
            other_i = base_i + bar_offset
            other_top = comps[other_i] + ovls[other_i] + comms[other_i]
            if other_top == 0:
                continue
            spd = base_top / other_top
            ax.annotate(
                '',
                xy=(x_positions[other_i], other_top),
                xytext=(x_positions[base_i], base_top),
                arrowprops=dict(arrowstyle='->', color='black',
                                lw=1.5, connectionstyle='arc3,rad=-0.2'),
            )
            speedup_label_pos = {
                (0, 1): (x_positions[base_i + 1], base_top - 0.12),
                (0, 2): (x_positions[base_i + 2], base_top),
                (0, 3): (x_positions[base_i + 3] + 0.02, base_top - 0.25),
                (1, 1): (x_positions[base_i + 1] + 0.02, base_top - 0.1),
                (1, 2): (x_positions[base_i + 2], base_top - 0.01),
                (1, 3): (x_positions[base_i + 3] + 0.02, base_top - 0.18),
            }
            lbl_x, lbl_y = speedup_label_pos.get(
                (grp_idx, bar_offset),
                ((x_positions[base_i] + x_positions[other_i]) / 2,
                 (base_top + other_top) / 2))
            ax.text(lbl_x, lbl_y, f"{spd:.2f}x",
                    fontsize=20, fontweight='bold', color='black',
                    ha='center', va='bottom')


    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_label_list, fontsize=14, rotation=0, ha='center')

    GROUP_LABEL_POS = {
        "Dataflow":          (0.4, -0.1),
        "Bulk-Synchronous":  (1.6, -0.1) if len(barrier_center_positions) > 1 else None,
    }
    for label, pos in GROUP_LABEL_POS.items():
        if pos is not None:
            ax.text(pos[0], pos[1], label,
                    ha='center', va='top', fontsize=24, fontweight='bold')


    seg_handles = [
        mpatches.Patch(facecolor=COLORS['Compute'], edgecolor='black',
                       label='Compute'),
        mpatches.Patch(facecolor=COLORS['Overlap'], edgecolor='black',
                       label='Overlap'),
        mpatches.Patch(facecolor=COLORS['Comm'], edgecolor='black',
                       label='Comm'),
    ]
    ax.legend(handles=seg_handles, fontsize=20,
              loc='upper center', bbox_to_anchor=(0.28, 1.),
              frameon=False, ncol=3, columnspacing=0.2,
              handletextpad=0.1)

    sp_detail = ", ".join(f"{op}={PER_OP_SP[op]}" for op in OPERATORS)
    ax.set_ylabel('Normalized Latency', fontsize=20)

    max_val = max((comps[i] + ovls[i] + comms[i])
                  for i in range(len(x_positions)))
    ax.set_ylim(0, max(max_val * 1.15, 1.2))
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(False)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_base = f"transformer_block_{TOPO}_bw96_1x1"
    out_pdf = os.path.join(pic_dir, out_base + ".pdf")
    out_png = os.path.join(pic_dir, out_base + ".png")
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(pic_dir, exist_ok=True)
    plot_figure()
