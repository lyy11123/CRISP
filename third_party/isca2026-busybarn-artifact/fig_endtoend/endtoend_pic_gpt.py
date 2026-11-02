
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import font_manager

file_path = os.path.dirname(os.path.realpath(__file__))

plt.rcParams['font.family'] = 'sans-serif'


json_path = os.path.join(file_path, "results", "endtoend_speedup.json")
if os.path.exists(json_path):
    with open(json_path) as f:
        raw = json.load(f)
    data = {int(k): v for k, v in raw.items()}
    pass
else:
    raise FileNotFoundError(f"{json_path} not found. Run end_to_end.py first.")

sizes    = [512, 2048, 8192]
backends = ['dojo', 'cerebras', 'manual']
backend_list = ['HW1', 'HW2', 'HW3']
models   = ['gpt']
models_dict = {'gpt': 'GPT-NeoX-20B'}

colors = {'gpt': '#80ed99'}
edgecolors = {'gpt': '#06d6a0'}

bar_width       = 0.35
metric_offset   = bar_width / 2
gap_between     = 1.0
x_groups        = np.arange(len(backends)) * (2 * bar_width + gap_between)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharey=False)

y_limits = {
    512: (0.0, 2.5),
    2048: (0.0, 2.0),
    8192: (0.0, 1.8)
}

for ax, size in zip(axes, sizes):
    subset = data[size]
    ax.set_ylim(y_limits[size])
    for i, backend in enumerate(backends):
        base_x = x_groups[i]
        x_prefill = base_x - metric_offset
        x_decode  = base_x + metric_offset

        y_prefill = subset['gpt'][backend]['prefill']
        y_decode  = subset['gpt'][backend]['decode']

        ax.bar(x_prefill, y_prefill, bar_width,
               color=colors['gpt'],
               edgecolor=edgecolors['gpt'],
               linewidth=4)
        ax.bar(x_decode, y_decode, bar_width,
               color=colors['gpt'],
               edgecolor='black',
               linewidth=3,
               hatch='/')

    text_x_pos      = 0.02
    text_y_pos      = 0.95
    text_fontsize   = 24
    text_fontweight = 'bold'

    ax.text(
        text_x_pos, text_y_pos, "seq="+str(size),
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=text_fontsize,
        fontweight=text_fontweight,
        zorder=30
    )

    ax.set_xticks(x_groups)
    ax.set_xticklabels(backend_list, fontsize=28)

    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylabel("Speedup", fontsize=28)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))

    ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', color='gray', linewidth=1.2, zorder=0)

ax0 = axes[0]

model_handles = [
    Patch(facecolor=colors['gpt'], edgecolor=edgecolors['gpt'],
          label=models_dict['gpt'], linewidth=3)
]
metric_handles = [
    Patch(facecolor='white', edgecolor='black', linewidth=3, label='prefill'),
    Patch(facecolor='white', edgecolor='black', linewidth=3, hatch='/', label='decode'),
]

ax0.legend(
    handles=model_handles + metric_handles,
    loc='upper right',
    bbox_to_anchor=(1.01, 1.5),
    ncol=3,
    frameon=False,
    fontsize=20,
    columnspacing=0.5,
    handletextpad=0.2,
    borderpad=0.1,
)

plt.tight_layout(pad=1.0)
plt.subplots_adjust(hspace=0.25)
plt.savefig(os.path.join(file_path, "endtoend_gpt.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(file_path, "endtoend_gpt.png"), dpi=300, bbox_inches='tight')

import statistics

for hw in backends:
    values = []
    for size in data:
        if 'gpt' in data[size]:
            values.append(data[size]['gpt'][hw]['prefill'])
            values.append(data[size]['gpt'][hw]['decode'])
    if values:
        print(f"{hw}: min={min(values):.4f}, max={max(values):.4f}, geomean={statistics.geometric_mean(values):.4f}")

all_values = [
    data[size]['gpt'][hw][metric]
    for size in data
    if 'gpt' in data[size]
    for hw in backends
    for metric in ['prefill', 'decode']
]
if all_values:
    print(f"GPT overall geomean: {statistics.geometric_mean(all_values):.4f}")
