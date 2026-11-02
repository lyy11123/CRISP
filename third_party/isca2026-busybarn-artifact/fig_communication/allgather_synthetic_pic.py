
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']

file_path = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(file_path, 'results')


message_sizes = [
    1024, 4096, 16384, 65536, 262144,
    1048576, 4194304, 16777216, 67108864, 268435456,
    1073741824, 4294967296, 17179869184
]
message_str = [
    "1KB", "4KB", "16KB", "64KB", "256KB",
    "1MB", "4MB", "16MB", "64MB", "256MB",
    "1GB", "4GB", "16GB"
]
shapes = [(5, 5)]
bw = 256
failures = [
    ([], []),
]
algorithms = ['dij', 'xy', 'ring']


bald_bandwidth = []
xy_bandwidth = []
ring_bandwidth = []

for (r, c) in shapes:
    for fail_idx, failure in enumerate(failures):
        for alg in algorithms:
            for i in range(len(message_str)):
                fname = (f"allgather_{r}x{c}_bw{bw}_failpatterm{fail_idx}_"
                         f"{message_str[i]}_{alg}.txt")
                fpath = os.path.join(output_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, 'r') as f:
                        val = int(f.read().strip())
                    new_val = message_sizes[i] / val 
                    if alg == 'dij':
                        bald_bandwidth.append(new_val)
                    elif alg == 'xy':
                        xy_bandwidth.append(new_val)  
                    elif alg == 'ring':
                        ring_bandwidth.append(new_val)


tacos_time = [24, 24, 48, 144, 504, 1980, 7884, 31476, 125844, 503340, 2013288, 8053080, 32212272]


tacos_bandwidth = []
for i in range(len(message_str)):
    new_val = message_sizes[i] / tacos_time[i]
    tacos_bandwidth.append(new_val)

multitree_time = [30, 30, 60, 180, 630, 2475, 9855, 39345, 157305, 629175, 2516610, 10066350, 40265340]


multitree_bandwidth = []
for i in range(len(message_str)):
    new_val = message_sizes[i] / multitree_time[i]
    multitree_bandwidth.append(new_val)


x = list(range(len(message_str)))

plt.rcParams['font.family'] = 'sans-serif'


plt.figure(figsize=(10, 9.6))
ax = plt.gca()

bald_color      = "#34E185"
tacos_color     = "#7C9517"
multitree_color = "#6C99ED"
xy_color        = '#7090C1'
ring_color      = '#B170C1'
fill_colors = ['#ff87ab', '#fcbf49', '#4cc9f0', '#80ed99', '#c77dff']
line_colors = ['#ff5d8f', '#f77f00', '#118ab2', '#06d6a0', '#7b2cbf']

markers = ['^', 'o', 'v', 'D', 's']


plt.plot(x, bald_bandwidth, label='BALD' , marker=markers[0], markersize=26,
                color=line_colors[0], markerfacecolor=fill_colors[0], markeredgecolor=line_colors[0],
                linestyle='-', markeredgewidth=3, linewidth=8)
plt.plot(x, ring_bandwidth, label='2D Ring' , marker=markers[1], markersize=19,
                color=line_colors[1], markerfacecolor=fill_colors[1], markeredgecolor=line_colors[1],
                linestyle='-', markeredgewidth=3, linewidth=6)
plt.plot(x, tacos_bandwidth, label='TACOS' , marker=markers[2], markersize=17,
                color=line_colors[2], markerfacecolor=fill_colors[2], markeredgecolor=line_colors[2],
                linestyle='-', markeredgewidth=3, linewidth=5)
plt.plot(x, multitree_bandwidth, label='MultiTree' , marker=markers[3], markersize=17,
                color=line_colors[3], markerfacecolor=fill_colors[3], markeredgecolor=line_colors[3],
                linestyle='-', markeredgewidth=3, linewidth=6)
plt.plot(x, xy_bandwidth, label='XY' , marker=markers[4], markersize=17,
                color=line_colors[4], markerfacecolor=fill_colors[4], markeredgecolor=line_colors[4],
                linestyle='-', markeredgewidth=3, linewidth=6)

plt.xticks(x, message_str, rotation=50, fontsize=32)
plt.yticks(fontsize=33)
ax.set_ylim(0.0, 580)
plt.xlabel('Message Size', fontsize=42)
ax.xaxis.set_label_coords(0.5, -0.3)
plt.ylabel('Bandwidth (GB/s)', fontsize=36)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=30, ncol=3, loc='upper center', frameon=False, handletextpad=0.3, columnspacing=1.2, labelspacing=0.3, bbox_to_anchor=(0.5, 1.3))

plt.tight_layout()
plt.savefig(os.path.join(file_path, './pic/allgather_synthetic.pdf'), dpi=300)


speedup = []
for i in range(len(message_str)):
    speedup.append(bald_bandwidth[i] / xy_bandwidth[i])
print("Speedup of BALD over XY:", min(speedup), max(speedup), np.mean(speedup))
