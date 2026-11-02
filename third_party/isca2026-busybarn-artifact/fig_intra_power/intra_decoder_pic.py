
import os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']

sql_list = [512]
ch_shapes = [(1,1)]
co_shapes = [(4,4)]
tensorcore_grain_list = [(64,64),(128,64),(128,128)]
failures = [([],[]),([(0, 5)], []),]

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
        for co in co_shapes:
            for tg in tensorcore_grain_list:
                tg_tag = f"{tg[0]}x{tg[1]}"
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

                    ga = gb = gc = 0
                    for tag,m in [("ln",2),("mha",8),("proj",1),("ffn",1)]:
                        pre = f"{tag}_sq{sq}_gemini_ch{ch_tag}_bw256_co{co[0]}x{co[1]}_bw256_t{tg_tag}_failpattern{fail_idx}"
                        a,b,c = find_min_abc([pre])
                        ga += a*m; gb += b*m; gc += c*m

                    records.append({
                        "sq":   sq,
                        "ch":   ch_tag,
                        "tg":   tg_tag,
                        "fail": fail_idx,
                        "busy": (busy_a, busy_b, busy_c),
                        "gem":  (ga,     gb,     gc),
                    })


alpha_busy = 1
alpha_gem = 0.5
hatch_busy = ''
hatch_gem  = '///'

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

fig, ax = plt.subplots(figsize=(16, 6))
plt.rcParams['font.family'] = 'sans-serif'


for i, r in enumerate(records):
    hatch_busy = '//' if r['fail'] == 1 else ''
    edge_color = '#d81159' if r['fail'] == 1 else 'black'
    ax.bar(x_busy[i], busy_b_norm[i],   bar_w, edgecolor=edge_color, linewidth=0, color=colors['Compute'], alpha=1.0, hatch=hatch_busy, label='Compute (B)' if i==0 else "")
    ax.bar(x_busy[i], busy_ovl_norm[i], bar_w, edgecolor=edge_color, linewidth=0, bottom=busy_b_norm[i], color=colors['Overlap'], alpha=1.0, hatch=hatch_busy, label='Overlap (B)' if i==0 else "")
    ax.bar(x_busy[i], busy_c_norm[i],   bar_w, edgecolor=edge_color, linewidth=0, bottom=busy_b_norm[i]+busy_ovl_norm[i], color=colors['Comm'], alpha=1.0, hatch=hatch_busy, label='Comm (B)' if i==0 else "")

for i, r in enumerate(records):
    hatch_gem = '//' if r['fail'] == 1 else ''
    edge_color = '#d81159' if r['fail'] == 1 else 'black'
    ax.bar(x_gem[i], gem_b_norm[i],   bar_w, edgecolor=edge_color, linewidth=0, color=colors_G['Compute'], alpha=1.0, hatch=hatch_gem, label='Compute (G)' if i==0 else "")
    ax.bar(x_gem[i], gem_ovl_norm[i], bar_w, edgecolor=edge_color, linewidth=0, bottom=gem_b_norm[i], color=colors_G['Overlap'], alpha=1.0, hatch=hatch_gem, label='Overlap (G)' if i==0 else "")
    ax.bar(x_gem[i], gem_c_norm[i],   bar_w, edgecolor=edge_color, linewidth=0, bottom=gem_b_norm[i]+gem_ovl_norm[i], color=colors_G['Comm'], alpha=1.0, hatch=hatch_gem, label='Comm (G)' if i==0 else "")
plt.grid(axis='y', ls='--')

handles, labels = ax.get_legend_handles_labels()
label2handle = dict(zip(labels, handles))

order = ['Compute (B)', 'Compute (G)',
         'Overlap (B)', 'Overlap (G)',
         'Comm (B)',    'Comm (G)']

ordered_handles = [label2handle[l] for l in order if l in label2handle]
ordered_labels  = [l for l in order if l in label2handle]

plt.legend(
          ordered_handles, ordered_labels,
          ncol=3,
          fontsize=27,
          loc='upper center',
          frameon=False,
          bbox_to_anchor=(0.31, 1.42),
          handletextpad=0.2,
          labelspacing=0.15,
          columnspacing=0.6,
          handleheight=0.8,
          handlelength=1.7)

group_centers = [
    (0 + 1) / 2,
    (2 + 3) / 2,
    (4 + 5) / 2,
]
group_labels = ['8 TFLOPs', '16 TFLOPs', '32 TFLOPs']

ax.set_xticks(group_centers)
ax.set_xticklabels(group_labels, fontsize=26)
ax.tick_params(axis='both',
               which='major',
               labelsize=24)
ax.set_xlabel('Core Computation Power and Failure Pattern', fontsize=28)
ax.set_ylabel('Normalized Latency', fontsize=26)
ax.set_yticks([0,1,2,3,4])

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
    markersize=25,
    linewidth=12,
    label='Speedup'
)
ax2.set_ylim(1, 1.5)
ax2.tick_params(axis='y', which='major', labelsize=24)
ax2.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
ax2.set_ylabel('Speedup', fontsize=30)
legend1 = ax2.legend(ncol=1,
          fontsize=28,
          loc='upper center',
          frameon=False,
          bbox_to_anchor=(0.93, 1.42),
          handletextpad=-0.2,
          labelspacing=0.1,
          columnspacing=0.1,
          markerscale=1.3)

power_labels = []
for gl in group_labels:
    power_labels.append(f"{gl} (no fail)")
    power_labels.append(f"{gl} (w/ fail)")
print("\n=== Speedup (BusyBarn over Gemini) ===")
for label, spd in zip(power_labels, speedup):
    print(f"  {label}: {spd:.3f}x")
print(f"  min={min(speedup):.3f}x  max={max(speedup):.3f}x  mean={np.mean(speedup):.3f}x")


patch_no    = mpatches.Patch(
    facecolor='white', edgecolor='black', hatch='', linewidth=1.5,
    label='no failure'
)
patch_fail  = mpatches.Patch(
    facecolor='white', edgecolor='#d81159', hatch='//', linewidth=1.5,
    label='a failed node'
)
ax3 = ax.twinx()
ax3.get_yaxis().set_ticks([])
ax3.legend(
    handles=[patch_no, patch_fail],
    ncol=1,
    fontsize=26,
    loc='upper center',
    frameon=False,
    bbox_to_anchor=(0.76, 1.42),
    handletextpad=0.2,
    labelspacing=0.15,
    columnspacing=0.6,
    handleheight=0.8,
    handlelength=1.6)

plt.tight_layout()
plt.savefig('./pic/intra_mapping_power_pic.pdf', dpi=300)
plt.savefig('./pic/intra_mapping_power_pic.png')
