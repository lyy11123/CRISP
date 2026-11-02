
import os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['font.family'] = 'sans-serif'


import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
from cfg import co_shapes as cfg_co_shapes, failures as cfg_failures

sql_list = [512]
ch_shapes = [(1,1)]
co_shapes = cfg_co_shapes
tensorcore_grain_list = [(64,64)]
failures = cfg_failures

pattern = re.compile(r"SA time cost:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")

def find_min_abc(prefixes):
    best = None
    for pre in prefixes:
        path = f"./results/{pre}.txt"
        if not os.path.isfile(path):
            continue
        with open(path, "r") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    a,b,c = map(int, m.groups())
        if best is None or a < best[0]:
            best = (a,b,c)
    return best or (np.inf, np.inf, np.inf)

records = []
for sq in sql_list:
    for ch in ch_shapes:
        ch_tag = f"{ch[0]}x{ch[1]}"
        for tg in tensorcore_grain_list:
            tg_tag = f"{tg[0]}x{tg[1]}"
            for co_idx, co in enumerate(co_shapes):
                busy_a_list, busy_b_list, busy_c_list = [], [], []
                gem_a_list, gem_b_list, gem_c_list = [], [], []

                for fail_idx, _ in enumerate(failures):
                    ln   = find_min_abc([f"ln_sq{sq}_greedy{gi}_lossratio{li}_ch{ch_tag}_bw256_co{co[0]}x{co[1]}_bw256_t{tg_tag}_failpattern{fail_idx}"
                                        for gi in (0,1) for li in range(7)])
                    mha  = find_min_abc([f"mha_sq{sq}_greedy{gi}_lossratio{li}_ch{ch_tag}_bw256_co{co[0]}x{co[1]}_bw256_t{tg_tag}_failpattern{fail_idx}"
                                        for gi in (0,1) for li in range(7)])
                    proj = find_min_abc([f"proj_sq{sq}_greedy{gi}_lossratio{li}_ch{ch_tag}_bw256_co{co[0]}x{co[1]}_bw256_t{tg_tag}_failpattern{fail_idx}"
                                        for gi in (0,1) for li in range(7)])
                    ffn  = find_min_abc([f"ffn_sq{sq}_greedy{gi}_lossratio{li}_ch{ch_tag}_bw256_co{co[0]}x{co[1]}_bw256_t{tg_tag}_failpattern{fail_idx}"
                                        for gi in (0,1) for li in range(7)])
                    busy_a = ln[0]*2 + mha[0]*8 + proj[0] + ffn[0]
                    busy_b = ln[1]*2 + mha[1]*8 + proj[1] + ffn[1]
                    busy_c = ln[2]*2 + mha[2]*8 + proj[2]

                    busy_a_list.append(busy_a)
                    busy_b_list.append(busy_b)
                    busy_c_list.append(busy_c)

                    ga = gb = gc = 0
                    for tag,m in [("ln",2),("mha",8),("proj",1),("ffn",1)]:
                        pre = f"{tag}_sq{sq}_gemini_ch{ch_tag}_bw256_co{co[0]}x{co[1]}_bw256_t{tg_tag}_failpattern{fail_idx}"
                        a,b,c = find_min_abc([pre])
                        ga += a*m; gb += b*m; gc += c*m

                    gem_a_list.append(ga)
                    gem_b_list.append(gb)
                    gem_c_list.append(gc)

                records.append({
                    "sq":   sq,
                    "ch":   ch_tag,
                    "co":   f"{co[0]}x{co[1]}",
                    "co_idx": co_idx,
                    "tg":   tg_tag,
                    "busy": (np.mean(busy_a_list), np.mean(busy_b_list), np.mean(busy_c_list)),
                    "gem":  (np.mean(gem_a_list), np.mean(gem_b_list), np.mean(gem_c_list)),
                })

colors = {
    'Compute': "#118ab2",
    'Overlap': "#f77f00",
    'Comm':    "#06d6a0",
}
colors_G = {
    'Compute': "#4cc9f0",
    'Overlap': "#fcbf49",
    'Comm':    "#80ed99"
}

busy_a   = np.array([r['busy'][0] for r in records])
busy_b   = np.array([r['busy'][1] for r in records])
busy_c   = np.array([r['busy'][2] for r in records])
busy_ovl = busy_a - busy_b - busy_c

gem_a    = np.array([r['gem'][0] for r in records])
gem_b    = np.array([r['gem'][1] for r in records])
gem_c    = np.array([r['gem'][2] for r in records])
gem_ovl  = gem_a - gem_b - gem_c

min_total = min(busy_a.min(), gem_a.min())
busy_b_norm   = busy_b   / min_total
busy_ovl_norm = busy_ovl / min_total
busy_c_norm   = busy_c   / min_total
gem_b_norm    = gem_b    / min_total
gem_ovl_norm  = gem_ovl  / min_total
gem_c_norm    = gem_c    / min_total

n = len(records)
ind = np.arange(n)
bar_w = 0.35
x_busy = ind - bar_w/2
x_gem  = ind + bar_w/2

fig, ax = plt.subplots(figsize=(16, 6))
plt.rcParams['font.family'] = 'sans-serif'

for i, r in enumerate(records):
    ax.bar(x_busy[i], busy_b_norm[i],   bar_w, edgecolor='black', linewidth=0,
           color=colors['Compute'], label='_nolegend_')
    ax.bar(x_busy[i], busy_ovl_norm[i], bar_w, edgecolor='black', linewidth=0,
           bottom=busy_b_norm[i], color=colors['Overlap'], label='_nolegend_')
    ax.bar(x_busy[i], busy_c_norm[i],   bar_w, edgecolor='black', linewidth=0,
           bottom=busy_b_norm[i]+busy_ovl_norm[i], color=colors['Comm'], label='_nolegend_')

for i, r in enumerate(records):
    ax.bar(x_gem[i], gem_b_norm[i],   bar_w, edgecolor='black', linewidth=0,
           color=colors_G['Compute'], label='_nolegend_')
    ax.bar(x_gem[i], gem_ovl_norm[i], bar_w, edgecolor='black', linewidth=0,
           bottom=gem_b_norm[i], color=colors_G['Overlap'], label='_nolegend_')
    ax.bar(x_gem[i], gem_c_norm[i],   bar_w, edgecolor='black', linewidth=0,
           bottom=gem_b_norm[i]+gem_ovl_norm[i], color=colors_G['Comm'], label='_nolegend_')

legend_handles = [
    mpatches.Patch(facecolor=colors['Compute'], edgecolor='black', label='Compute (B)'),
    mpatches.Patch(facecolor=colors_G['Compute'], edgecolor='black', label='Compute (G)'),
    mpatches.Patch(facecolor=colors['Overlap'], edgecolor='black', label='Overlap (B)'),
    mpatches.Patch(facecolor=colors_G['Overlap'], edgecolor='black', label='Overlap (G)'),
    mpatches.Patch(facecolor=colors['Comm'],    edgecolor='black', label='Comm (B)'),
    mpatches.Patch(facecolor=colors_G['Comm'],    edgecolor='black', label='Comm (G)'),
]
ax.legend(handles=legend_handles,
          ncol=3, fontsize=27, loc='upper center', frameon=False,
          bbox_to_anchor=(0.41, 1.42),
          handletextpad=0.2, 
          labelspacing=0.15, 
          columnspacing=0.8,
          handleheight=0.8,
          handlelength=2.0)

plt.grid(axis='y', ls='--')
co_labels = [r['co'] for r in records]

ax.set_xticks(ind)
ax.set_xticklabels(co_labels, fontsize=20, rotation=0)
ax.tick_params(axis='both', which='major', labelsize=29)
ax.set_xlabel('Core Shape', fontsize=35)
ax.set_yticks([0,0.5,1.0,1.5,2.0,2.5])
ax.set_ylabel('Normalized Latency', fontsize=30)

ax2 = ax.twinx()
speedup = gem_a / busy_a
x_speed  = (x_busy + x_gem) / 2
ax2.plot(
    x_speed,
    speedup,
    marker='*',
    linestyle='',
    color='#a24ccd',
    markeredgecolor='#7400b8',
    markersize=30,
    linewidth=12,
    label='Speedup'
)
ax2.set_ylim(1, 1.63)
ax2.tick_params(axis='both', which='major', labelsize=29)
ax2.set_yticks([1.0, 1.2, 1.4, 1.6])
ax2.set_ylabel('Speedup', fontsize=30)
ax2.legend(ncol=1, 
          fontsize=28, 
          loc='upper center',
          frameon=False,
          bbox_to_anchor=(0.86, 1.34),
          handletextpad=0.0, 
          labelspacing=0.1, 
          columnspacing=0.1,
          markerscale=1.4)

print("\n=== Speedup (BusyBarn over Gemini) ===")
for label, spd in zip(co_labels, speedup):
    print(f"  {label}: {spd:.3f}x")
print(f"  min={min(speedup):.3f}x  max={max(speedup):.3f}x  mean={np.mean(speedup):.3f}x")


plt.tight_layout()
plt.savefig('./pic/intra_mapping_coreshape_pic.pdf', dpi=300)
plt.savefig('./pic/intra_mapping_coreshape_pic.png')
