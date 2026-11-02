"""
Heatmap of transformer block per-core and per-link working time.

Adapted to sa_convergence.py topology: runs SA for each operator, evaluates
at --steps, extracts per-core (TC+VU) and per-link working time from event_driver,
then draws a flat grid where each core is a colored rectangle and each
directional link is a colored line segment.

Distinguishes co2co (intra-die) vs ch2ch (inter-die) links visually.
Draws chiplet boundary boxes.

Usage:
    python plot_hot_transformer_block.py                          # both variants
    python plot_hot_transformer_block.py --variant busybarn        # single variant
    python plot_hot_transformer_block.py --operator ffn            # single operator
    python plot_hot_transformer_block.py --force                   # recompute
"""
import os
import sys
import argparse
import random
import time
import numpy as np
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle

file_path = os.path.dirname(os.path.realpath(__file__))

from operator_setup import (
    init_hardware, finalize_graph, build_operator,
    RESULTS_DIR, TOPOLOGY_CLASSES,
)
from add_communication import (
    build_event_v2 as build_event, add_broadcast_v2 as add_broadcast,
    reroute_dijkstra, apply_barrier_sync_all,
)
from Stream_Mapping import stream_mapping_v2 as stream_mapping
from event_driver import event_driver_v2 as event_driver

OPERATORS = ["ln", "mha", "proj", "ffn"]
WEIGHTS = {"ln": 2, "mha": 1, "proj": 1, "ffn": 1}

SPLIT_DEGREES = {
    "ffn":  (1, 4, 1, 32),
    "ln":   (1, 4, 1, 1, 1, 1),
    "mha":  (1, 2, 1, 1),
    "proj": (1, 4, 1, 4),
}

DEFAULT_HW_CFG = os.path.join(
    file_path, "../src/platform/cfgs/wamis_hd_round.cfg")


DEFAULTS = {
    "busybarn": {
        "ffn":  dict(alpha=1,   beta=100, gamma=1,   loss_ratio=[2, 1, 1, 0.1],
                     random_threshold=0.3, lp=False, region_restricted=False,
                     hops_only=False, reroute=True),
        "ln":   dict(alpha=100, beta=1,   gamma=100, loss_ratio=[1, 1, 0, 1],
                     random_threshold=0.5, lp=False, region_restricted=False,
                     hops_only=False, reroute=True),
        "mha":  dict(alpha=1,   beta=100, gamma=1,   loss_ratio=[1, 0.1, 1, 1],
                     random_threshold=0.3, lp=False, region_restricted=False,
                     hops_only=False, reroute=True),
        "proj": dict(alpha=1,   beta=100, gamma=1,   loss_ratio=[1, 1, 1, 0.1],
                     random_threshold=0.3, lp=False, region_restricted=False,
                     hops_only=False, reroute=True),
    },
    "gemini": {
        "ffn":  dict(alpha=100, beta=1, gamma=100, loss_ratio=[1, 1, 1, 1],
                     random_threshold=0.3, lp=True, region_restricted=True,
                     hops_only=True, reroute=False),
        "ln":   dict(alpha=100, beta=1, gamma=100, loss_ratio=[1, 1, 0, 1],
                     random_threshold=0.5, lp=True, region_restricted=True,
                     hops_only=True, reroute=False),
        "mha":  dict(alpha=100, beta=1, gamma=100, loss_ratio=[1, 1, 1, 1],
                     random_threshold=0.3, lp=True, region_restricted=True,
                     hops_only=True, reroute=False),
        "proj": dict(alpha=100, beta=1, gamma=100, loss_ratio=[1, 1, 1, 1],
                     random_threshold=0.5, lp=True, region_restricted=True,
                     hops_only=True, reroute=False),
    },
}


def _reset_hw_runtime(hw):
    """Reset mutable runtime state on all modules and links."""
    for dev_dict in hw.modules_dict.values():
        for dev in dev_dict.values():
            dev.work_flag = False
            dev.work_endtime = None
            dev.work_record = []
    for lnk in hw.links_dict.values():
        if hasattr(lnk, 'work_flag'):
            lnk.work_flag = False
            lnk.work_endtime = None
            lnk.work_record = []


def extract_utilization(hw):
    """Extract per-node and per-link utilization from hw after event_driver.

    Returns (total_time, tc_dict, vu_dict, link_dict) where:
      tc_dict[(chip_idx, core_idx)] = util_fraction
      vu_dict[(chip_idx, core_idx)] = util_fraction
      link_dict[((src_chip, src_core), (dst_chip, dst_core))] = (util_fraction, link_type)
    """
    total_time = 0
    for dev_dict in hw.modules_dict.values():
        for dev in dev_dict.values():
            for rec in dev.work_record:
                total_time = max(total_time, rec[1])
    for lnk in hw.links_dict.values():
        if hasattr(lnk, 'work_record'):
            for rec in lnk.work_record:
                total_time = max(total_time, rec[1])

    if total_time == 0:
        return 0, {}, {}, {}

    tc_dict = {}
    for location, dev in hw.modules_dict.get("tensorcore", {}).items():
        node = location[:2]
        busy = sum(end - start for start, end, _ in dev.work_record)
        tc_dict[node] = tc_dict.get(node, 0) + busy / total_time

    vu_dict = {}
    for location, dev in hw.modules_dict.get("vectorunit", {}).items():
        node = location[:2]
        busy = sum(end - start for start, end, _ in dev.work_record)
        vu_dict[node] = vu_dict.get(node, 0) + busy / total_time

    link_dict = {}
    for link_key, lnk in hw.links_dict.items():
        if hasattr(lnk, 'work_record') and lnk.work_record:
            busy = sum(end - start for start, end, _ in lnk.work_record)
            link_dict[link_key] = (busy / total_time, lnk.link_type)
        elif hasattr(lnk, 'link_type'):
            link_dict[link_key] = (0.0, lnk.link_type)

    return total_time, tc_dict, vu_dict, link_dict


def run_operator(op, variant, hw_cfg, hw_topology, steps, barrier,
                 split_degree=None, seed=123):
    """Run SA for one operator+variant, evaluate, return utilization data."""
    random.seed(seed)
    np.random.seed(seed)

    cfg = DEFAULTS[variant][op]
    hw = init_hardware(cfg_path=hw_cfg, topology=hw_topology)
    data_dict, beha_dict, broadcast_datatags = build_operator(
        op, split_degree=split_degree)
    layers_regions = finalize_graph(beha_dict, data_dict, hw, greedy_flag=False)

    event_dict = {}
    hops, cd, cl, tc_loads, vu_loads = build_event(
        beha_dict=beha_dict, data_dict=data_dict, hardware_platform=hw,
        event_dict=event_dict, dijkstra_routing=False,
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])

    saved_ll = dict(hw.link_loads_dict)
    saved_lc = dict(hw.link_loads_count)
    init_ed = deepcopy(event_dict)
    add_broadcast(
        data_tags=broadcast_datatags, ddr_chiplets=layers_regions[0],
        beha_dict=beha_dict, data_dict=data_dict, hardware_platform=hw,
        event_dict=init_ed, dijkstra_routing=False,
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])
    init_hw = deepcopy(hw)
    init_result = event_driver(events_dict=init_ed, hardware_platform=init_hw)
    _reset_hw_runtime(hw)
    for k in hw.link_loads_dict:
        hw.link_loads_dict[k] = saved_ll[k]
        hw.link_loads_count[k] = saved_lc[k]

    stream_mapping(
        hops=hops, communication_distances=cd,
        communication_loads_dict=cl,
        tensorcore_loads_dict=tc_loads,
        vectorunit_loads_dict=vu_loads,
        layers_regions=layers_regions,
        beha_dict=beha_dict, data_dict=data_dict,
        hardware_platform=hw, event_dict=event_dict,
        random_threshold=[cfg["random_threshold"]],
        LP_flag=cfg["lp"],
        region_restriced=cfg["region_restricted"],
        related=False,
        hops_only=cfg["hops_only"],
        loss_ratio=cfg["loss_ratio"],
        mem_enable=True,
        mem_threshold=[0.3],
        mem_datatags=[],
        mem_random=True,
        dijkstra_routing=False,
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"],
        t_max=10, t_min=1e-6, steps=steps,
    )

    if cfg["reroute"]:
        reroute_dijkstra(
            beha_dict=beha_dict, data_dict=data_dict,
            hardware_platform=hw, event_dict=event_dict,
            alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])

    add_broadcast(
        data_tags=broadcast_datatags, ddr_chiplets=layers_regions[0],
        beha_dict=beha_dict, data_dict=data_dict, hardware_platform=hw,
        event_dict=event_dict, dijkstra_routing=cfg["reroute"],
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])

    if barrier:
        apply_barrier_sync_all(event_dict, data_dict)

    result = event_driver(events_dict=event_dict, hardware_platform=hw)

    total_time, tc_dict, vu_dict, link_dict = extract_utilization(hw)
    return total_time, tc_dict, vu_dict, link_dict


def _cache_base(op, variant, steps, barrier):
    b_tag = "_barrier" if barrier else ""
    return os.path.join(RESULTS_DIR,
                        f"hot_{op}_{variant}{b_tag}_step{int(steps)}")


def save_utilization(op, variant, steps, barrier,
                     total_time, tc_dict, vu_dict, link_dict):
    base = _cache_base(op, variant, steps, barrier)
    lk_keys = list(link_dict.keys())
    lk_utils = np.array([link_dict[k][0] for k in lk_keys])
    lk_types = [link_dict[k][1] for k in lk_keys]
    np.savez(base + "_util.npz",
             total_time=np.array([total_time]),
             tc_keys=np.array(list(tc_dict.keys()), dtype=object),
             tc_vals=np.array(list(tc_dict.values())),
             vu_keys=np.array(list(vu_dict.keys()), dtype=object),
             vu_vals=np.array(list(vu_dict.values())),
             link_keys=np.array(lk_keys, dtype=object),
             link_vals=lk_utils,
             link_types=np.array(lk_types, dtype=object))


def load_utilization(op, variant, steps, barrier):
    base = _cache_base(op, variant, steps, barrier)
    fpath = base + "_util.npz"
    if not os.path.exists(fpath):
        return None
    data = np.load(fpath, allow_pickle=True)
    total_time = int(data["total_time"][0])
    tc_dict = {tuple(k): v for k, v in zip(data["tc_keys"], data["tc_vals"])}
    vu_dict = {tuple(k): v for k, v in zip(data["vu_keys"], data["vu_vals"])}
    link_dict = {}
    for k, v, t in zip(data["link_keys"], data["link_vals"], data["link_types"]):
        link_dict[tuple(tuple(x) for x in k)] = (float(v), str(t))
    return total_time, tc_dict, vu_dict, link_dict


def run_or_load(op, variant, hw_cfg, hw_topology, steps, barrier,
                force=False, split_degree=None, seed=123):
    """Run SA or load cached utilization."""
    if not force:
        cached = load_utilization(op, variant, steps, barrier)
        if cached is not None:
            return cached

    total_time, tc_dict, vu_dict, link_dict = run_operator(
        op, variant, hw_cfg, hw_topology, steps, barrier,
        split_degree=split_degree, seed=seed)

    save_utilization(op, variant, steps, barrier,
                     total_time, tc_dict, vu_dict, link_dict)
    return total_time, tc_dict, vu_dict, link_dict


def chip_core_to_flat(chip_idx, core_idx, l1_h, l1_w, l2_w):
    """Convert (chiplet_idx, core_idx) to flat (row, col) on the global grid."""
    chip_y, chip_x = divmod(chip_idx, l2_w)
    core_y, core_x = divmod(core_idx, l1_w)
    return chip_y * l1_h + core_y, chip_x * l1_w + core_x


def _draw_on_ax(ax, node_grid, co2co_links, ch2ch_links,
                total_rows, total_cols, l1_h, l1_w, l2_h, l2_w,
                node_norm, link_norm, title,
                title_x=0.5, title_y=1.0):
    """Draw one heatmap panel on a given axes with shared norms."""
    node_cmap = cm.YlOrRd
    _greens = mcolors.LinearSegmentedColormap.from_list(
        'Greens4', ['#e5f5e0', '#a1d99b', '#31a354', '#00441b'])
    co2co_cmap = _greens
    ch2ch_cmap = _greens

    cell_size = 1.2
    chiplet_gap = 0.3
    co2co_offset = cell_size * 0.20
    ch2ch_offset = cell_size * 0.20

    def rc_to_xy(r, c):
        chip_r, core_r = divmod(r, l1_h)
        chip_c, core_c = divmod(c, l1_w)
        x = chip_c * (l1_w * cell_size + chiplet_gap) + core_c * cell_size
        y_total = l2_h * (l1_h * cell_size + chiplet_gap) - chiplet_gap
        y = y_total - (chip_r * (l1_h * cell_size + chiplet_gap) + core_r * cell_size)
        return x, y

    segments_co2co, colors_co2co = [], []
    for (sr, sc, dr, dc), val in co2co_links.items():
        x1, y1 = rc_to_xy(sr, sc)
        x2, y2 = rc_to_xy(dr, dc)
        dx, dy = x2 - x1, y2 - y1
        length = max(np.sqrt(dx * dx + dy * dy), 1e-9)
        nx, ny = -dy / length, dx / length
        segments_co2co.append([(x1 + nx * co2co_offset, y1 + ny * co2co_offset),
                               (x2 + nx * co2co_offset, y2 + ny * co2co_offset)])
        colors_co2co.append(val)

    ch2ch_shrink = 0.15
    segments_ch2ch, colors_ch2ch = [], []
    for (sr, sc, dr, dc), val in ch2ch_links.items():
        x1, y1 = rc_to_xy(sr, sc)
        x2, y2 = rc_to_xy(dr, dc)
        dx, dy = x2 - x1, y2 - y1
        length = max(np.sqrt(dx * dx + dy * dy), 1e-9)
        nx, ny = -dy / length, dx / length
        ux, uy = dx / length, dy / length
        sx1 = x1 + ux * length * ch2ch_shrink + nx * ch2ch_offset
        sy1 = y1 + uy * length * ch2ch_shrink + ny * ch2ch_offset
        sx2 = x2 - ux * length * ch2ch_shrink + nx * ch2ch_offset
        sy2 = y2 - uy * length * ch2ch_shrink + ny * ch2ch_offset
        segments_ch2ch.append([(sx1, sy1), (sx2, sy2)])
        colors_ch2ch.append(val)

    def _darken_cmap(cmap, factor=0.5):
        """Return a colormap that produces darker versions of the input cmap."""
        colors_list = [cmap(i) for i in np.linspace(0, 1, 256)]
        dark = [(r * factor, g * factor, b * factor, a) for r, g, b, a in colors_list]
        return mcolors.ListedColormap(dark)

    if segments_co2co:
        lc_edge = LineCollection(segments_co2co, cmap=_darken_cmap(co2co_cmap),
                                 norm=link_norm, linewidths=26.0, zorder=1, alpha=0.9)
        lc_edge.set_array(np.array(colors_co2co))
        ax.add_collection(lc_edge)
        lc = LineCollection(segments_co2co, cmap=co2co_cmap, norm=link_norm,
                            linewidths=18.0, zorder=1, alpha=0.8)
        lc.set_array(np.array(colors_co2co))
        ax.add_collection(lc)

    if segments_ch2ch:
        lc_edge = LineCollection(segments_ch2ch, cmap=_darken_cmap(ch2ch_cmap),
                                 norm=link_norm, linewidths=26.0, zorder=1, alpha=0.9)
        lc_edge.set_array(np.array(colors_ch2ch))
        ax.add_collection(lc_edge)
        lc = LineCollection(segments_ch2ch, cmap=ch2ch_cmap, norm=link_norm,
                            linewidths=18.0, zorder=1, alpha=0.9)
        lc.set_array(np.array(colors_ch2ch))
        ax.add_collection(lc)

    for ch_r in range(l2_h):
        for ch_c in range(l2_w):
            x0, y0 = rc_to_xy(ch_r * l1_h + l1_h - 1, ch_c * l1_w)
            x1, y1 = rc_to_xy(ch_r * l1_h, ch_c * l1_w + l1_w - 1)
            pad = cell_size * 0.5
            rect = Rectangle(
                (x0 - pad, y0 - pad),
                (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad,
                linewidth=2.5, edgecolor='#555555', facecolor='none',
                linestyle='--', zorder=0, alpha=0.6)
            ax.add_patch(rect)
            cx = (x0 + x1) / 2
            if ch_r == l2_h - 1:
                cy = y0 - pad - 0.04
                ax.text(cx, cy, f'Die {ch_r * l2_w + ch_c}',
                        ha='center', va='top', fontsize=44,
                        color='#555555', fontweight='bold')
            else:
                cy = y1 + pad - 0.02
                ax.text(cx, cy, f'Die {ch_r * l2_w + ch_c}',
                        ha='center', va='bottom', fontsize=44,
                        color='#555555', fontweight='bold')

    pad = cell_size * 0.75
    for r in range(total_rows):
        for c in range(total_cols):
            x, y = rc_to_xy(r, c)
            color = node_cmap(node_norm(node_grid[r, c]))
            edge_color = (color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, 1.0)
            rect = plt.Rectangle((x - pad / 2, y - pad / 2), pad, pad,
                                 facecolor=color, edgecolor=edge_color,
                                 linewidth=2.5, zorder=2)
            ax.add_patch(rect)
            chip_r, core_r = divmod(r, l1_h)
            chip_c, core_c = divmod(c, l1_w)
            core_idx = core_r * l1_w + core_c
            txt_color = 'white' if node_norm(node_grid[r, c]) > 0.90 else 'black'
            ax.text(x, y, f'{core_idx}', ha='center', va='center',
                    fontsize=52, color=txt_color, fontweight='bold', zorder=3)

    x_min, y_min = rc_to_xy(total_rows - 1, 0)
    x_max, y_max = rc_to_xy(0, total_cols - 1)
    margin = cell_size * 0.6
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title(title, fontsize=44, fontweight='bold', pad=15,
                 x=title_x, y=title_y)

    return segments_co2co, segments_ch2ch


def draw_combined(variant_data, total_rows, total_cols,
                  l1_h, l1_w, l2_h, l2_w, out_base,
                  title_x=0.5, title_y=-0.2):
    """Draw side-by-side heatmaps for multiple variants with shared colorbars.

    variant_data: list of (variant_name, node_grid, co2co_work, ch2ch_work, title, stats)
    title_x, title_y: position of each panel title in axes coordinates (default: centered top)
    """
    plt.rcParams['font.family'] = 'sans-serif'

    all_node_vals = [d[1].max() for d in variant_data]
    node_vmax = max(max(all_node_vals), 0.01)
    node_norm = mcolors.Normalize(vmin=0, vmax=node_vmax)

    all_link_vals = []
    for _, _, co2co, ch2ch, _, _ in variant_data:
        all_link_vals.extend(co2co.values())
        all_link_vals.extend(ch2ch.values())
    link_vmax = max(max(all_link_vals), 0.01) if all_link_vals else 0.01
    link_norm = mcolors.Normalize(vmin=0, vmax=link_vmax)

    n = len(variant_data)
    cell_size = 1.2
    chiplet_gap = 0.3
    panel_w = l2_w * (l1_w * cell_size + chiplet_gap) + 0.5
    panel_h = l2_h * (l1_h * cell_size + chiplet_gap) + 1.5
    fig_w = panel_w * n + 3.5
    fig_h = panel_h + 2.0

    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    has_co2co = False
    has_ch2ch = False
    for i, (_, node_grid, co2co, ch2ch, title, _) in enumerate(variant_data):
        seg_co, seg_ch = _draw_on_ax(
            axes[i], node_grid, co2co, ch2ch,
            total_rows, total_cols, l1_h, l1_w, l2_h, l2_w,
            node_norm, link_norm, title,
            title_x=title_x, title_y=title_y)
        has_co2co = has_co2co or bool(seg_co)
        has_ch2ch = has_ch2ch or bool(seg_ch)

    def _fmt_cycles(x, _):
        if x == 0:
            return '0'
        exp = int(np.floor(np.log10(abs(x))))
        coeff = x / 10**exp
        if abs(coeff - round(coeff)) < 0.01:
            return f'{round(coeff):.0f}e{exp}'
        return f'{coeff:.1f}e{exp}'

    node_cmap = cm.YlOrRd
    _greens = mcolors.LinearSegmentedColormap.from_list(
        'Greens4', ['#e5f5e0', '#a1d99b', '#31a354', '#00441b'])
    co2co_cmap = _greens
    ch2ch_cmap = _greens

    cbar_node_pos = [1.16, 0.03, 0.025, 0.9]
    cbar_link_pos = [1.02, 0.03, 0.025, 0.9]

    sm_node = cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    sm_node.set_array([])
    cax_node = fig.add_axes(cbar_node_pos)
    cbar_node = fig.colorbar(sm_node, cax=cax_node)
    cbar_node.set_label('Core Working Time (cycles)', fontsize=36)
    cbar_node.ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_cycles))
    cbar_node.ax.tick_params(labelsize=32)

    if has_co2co or has_ch2ch:
        sm_link = cm.ScalarMappable(
            cmap=co2co_cmap if has_co2co else ch2ch_cmap, norm=link_norm)
        sm_link.set_array([])
        cax_link = fig.add_axes(cbar_link_pos)
        cbar_link = fig.colorbar(sm_link, cax=cax_link)
        cbar_link.set_label('Link Working Time (cycles)', fontsize=36)
        cbar_link.ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_cycles))
        cbar_link.ax.tick_params(labelsize=32)


    out_pdf = os.path.join(RESULTS_DIR, out_base + ".pdf")
    out_png = os.path.join(RESULTS_DIR, out_base + ".png")
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Heatmap of transformer block utilization "
                    "(wamis_hd_around topology)")

    p.add_argument("--variant", type=str, default="all",
                   choices=["busybarn", "gemini", "all"])
    p.add_argument("--operator", type=str, default="all",
                   choices=["ln", "mha", "proj", "ffn", "all"])
    p.add_argument("--barrier", action="store_true")
    p.add_argument("--steps", type=float, default=1e3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cached .npz exists")

    p.add_argument("--hw-cfg", type=str, default=DEFAULT_HW_CFG)
    p.add_argument("--hw-topology", type=str, default="wamis_hd_around",
                   choices=list(TOPOLOGY_CLASSES.keys()))

    p.add_argument("--title-x", type=float, default=0.5,
                   help="Panel title x position in axes coords (default: 0.5)")
    p.add_argument("--title-y", type=float, default=-0.12,
                   help="Panel title y position in axes coords (default: 1.0)")

    for op_name, default_sd in SPLIT_DEGREES.items():
        p.add_argument(f"--split-{op_name}", type=int, nargs='+',
                       default=list(default_sd),
                       metavar='N',
                       help=f"{op_name} split degree (default: {list(default_sd)})")

    args = p.parse_args()
    args.split_degrees = {}
    for op_name in SPLIT_DEGREES:
        args.split_degrees[op_name] = tuple(getattr(args, f"split_{op_name}"))
    return args


def collect_variant_data(variant, ops, args, hw_ref):
    """Run all operators for one variant, return aggregated working time data."""
    steps = int(args.steps)
    barrier = args.barrier

    l1_h = hw_ref.l1_height
    l1_w = hw_ref.l1_width
    l2_h = hw_ref.l2_height
    l2_w = hw_ref.l2_width
    n_cores_per_chiplet = l1_h * l1_w
    total_rows = l2_h * l1_h
    total_cols = l2_w * l1_w

    node_busy = {}
    co2co_busy = {}
    ch2ch_busy = {}
    op_times = {}

    for op in ops:
        sd = args.split_degrees.get(op)
        total_time, tc_dict, vu_dict, lk_dict = run_or_load(
            op, variant, args.hw_cfg, args.hw_topology,
            steps, barrier, force=args.force,
            split_degree=sd, seed=args.seed)

        if total_time == 0:
            continue

        op_times[op] = total_time
        w = WEIGHTS[op] if args.operator == "all" else 1

        all_keys = set(tc_dict.keys()) | set(vu_dict.keys())
        for key in all_keys:
            if key[1] >= n_cores_per_chiplet:
                continue
            r, c = chip_core_to_flat(key[0], key[1], l1_h, l1_w, l2_w)
            tc_busy = tc_dict.get(key, 0) * total_time
            vu_busy = vu_dict.get(key, 0) * total_time
            node_busy[(r, c)] = node_busy.get((r, c), 0) + (tc_busy + vu_busy) * w

        for (src, dst), (util, ltype) in lk_dict.items():
            if src[1] >= n_cores_per_chiplet or dst[1] >= n_cores_per_chiplet:
                continue
            sr, sc = chip_core_to_flat(src[0], src[1], l1_h, l1_w, l2_w)
            dr, dc = chip_core_to_flat(dst[0], dst[1], l1_h, l1_w, l2_w)
            lk_key = (sr, sc, dr, dc)
            busy = util * total_time

            if ltype == "ch2ch":
                ch2ch_busy[lk_key] = ch2ch_busy.get(lk_key, 0) + busy * w
            else:
                co2co_busy[lk_key] = co2co_busy.get(lk_key, 0) + busy * w

    node_grid = np.zeros((total_rows, total_cols))
    for (r, c), busy in node_busy.items():
        node_grid[r, c] = busy

    co2co_work = dict(co2co_busy)
    ch2ch_work = dict(ch2ch_busy)

    block_time = sum(op_times.get(op, 0) * (WEIGHTS[op] if args.operator == "all" else 1)
                     for op in ops)

    stats = {
        "block_time": block_time,
        "op_times": op_times,
        "split_degrees": {op: args.split_degrees.get(op) for op in ops},
        "node_max": node_grid.max(),
        "node_min": node_grid.min(),
        "node_avg": node_grid.mean(),
        "node_std": node_grid.std(),
        "co2co_avg": np.mean(list(co2co_work.values())) if co2co_work else 0,
        "ch2ch_avg": np.mean(list(ch2ch_work.values())) if ch2ch_work else 0,
    }

    return node_grid, co2co_work, ch2ch_work, stats


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    variants = ["gemini", "busybarn"] if args.variant == "all" else [args.variant]
    ops = OPERATORS if args.operator == "all" else [args.operator]

    op_label = "transformer_block" if args.operator == "all" else args.operator
    steps = int(args.steps)
    b_str = "+barrier" if args.barrier else ""
    hw_ref = init_hardware(cfg_path=args.hw_cfg, topology=args.hw_topology)
    l1_h = hw_ref.l1_height
    l1_w = hw_ref.l1_width
    l2_h = hw_ref.l2_height
    l2_w = hw_ref.l2_width
    total_rows = l2_h * l1_h
    total_cols = l2_w * l1_w

    variant_data = []
    for variant in variants:
        node_grid, co2co_work, ch2ch_work, stats = collect_variant_data(
            variant, ops, args, hw_ref)

        title = variant.capitalize()
        variant_data.append((variant, node_grid, co2co_work, ch2ch_work, title, stats))

    b_tag = "_barrier" if args.barrier else ""
    out_base = f"hot_{op_label}_combined{b_tag}_step{steps}"
    draw_combined(variant_data, total_rows, total_cols,
                  l1_h, l1_w, l2_h, l2_w, out_base,
                  title_x=args.title_x, title_y=args.title_y)

    if len(variant_data) == 2:
        s0 = variant_data[0][5]
        s1 = variant_data[1][5]
        if s0['block_time'] > 0 and s1['block_time'] > 0:
            speedup = s0['block_time'] / s1['block_time']
            reduction = (1 - s1['block_time'] / s0['block_time']) * 100
            v0_name = variant_data[0][0].capitalize()
            v1_name = variant_data[1][0].capitalize()
            print(f"{v1_name} vs {v0_name}: "
                  f"{speedup:.2f}x speedup ({reduction:+.1f}% latency)")


if __name__ == "__main__":
    main()
