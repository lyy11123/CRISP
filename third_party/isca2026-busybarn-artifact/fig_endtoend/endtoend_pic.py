
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
    data={512: {'gpt': {'dojo': {'prefill': 1.589241253995993, 'decode': 1.3773873660251443}, 'cerebras': {'prefill': 2.1485596921488965, 'decode': 1.1827535421609907}, 'manual': {'prefill': 1.4037210767489965, 'decode': 1.4785892097573363}}, 'opt': {'dojo': {'prefill': 1.760828079895238, 'decode': 1.3380573602832646}, 'cerebras': {'prefill': 1.799968962254866, 'decode': 1.2912505185612726}, 'manual': {'prefill': 1.8175033604961235, 'decode': 1.3825087808291647}}, 'qwen3moe': {'dojo': {'prefill': 1.7879896550799594, 'decode': 1.360553681246379}, 'cerebras': {'prefill': 1.744739281895874, 'decode': 1.1044946011484906}, 'manual': {'prefill': 1.8837017656666661, 'decode': 1.5821500937856157}}, 'qwen': {'dojo': {'prefill': 1.8441760146348911, 'decode': 1.4771658856311842}, 'cerebras': {'prefill': 2.08629156348886, 'decode': 1.509959421652249}, 'manual': {'prefill': 1.8244489243760922, 'decode': 1.4683655707248875}}, 'qwen2moe': {'dojo': {'prefill': 1.4568701529510426, 'decode': 1.3230277667760155}, 'cerebras': {'prefill': 1.4806503735694654, 'decode': 1.0827010744732637}, 'manual': {'prefill': 1.4738168407704058, 'decode': 1.516620483588707}}, 'llama': {'dojo': {'prefill': 1.6625787888278316, 'decode': 1.428824474345616}, 'cerebras': {'prefill': 1.802526691236155, 'decode': 1.4050772290385798}, 'manual': {'prefill': 1.6365049956463615, 'decode': 1.4698699001500175}}}, 2048: {'gpt': {'dojo': {'prefill': 1.3892077107124627, 'decode': 1.354593736098406}, 'cerebras': {'prefill': 1.6365231043279183, 'decode': 1.0976139197538002}, 'manual': {'prefill': 1.3005368692003367, 'decode': 1.4896363630978462}}, 'opt': {'dojo': {'prefill': 1.4318584946029507, 'decode': 1.348852430541597}, 'cerebras': {'prefill': 1.5912990088167263, 'decode': 1.2912660428124318}, 'manual': {'prefill': 1.3787785712975873, 'decode': 1.399587255037218}}, 'qwen3moe': {'dojo': {'prefill': 1.451274990092234, 'decode': 1.3274759164975363}, 'cerebras': {'prefill': 1.270436782005064, 'decode': 1.1070229394553253}, 'manual': {'prefill': 1.6007591575337383, 'decode': 1.5134826119553342}}, 'qwen': {'dojo': {'prefill': 1.611693370136217, 'decode': 1.3734017400368126}, 'cerebras': {'prefill': 1.5170205272421677, 'decode': 1.427786034992658}, 'manual': {'prefill': 1.6947776205164584, 'decode': 1.3384071205356656}}, 'qwen2moe': {'dojo': {'prefill': 1.497509956827733, 'decode': 1.3135691709356168}, 'cerebras': {'prefill': 1.6094188122145516, 'decode': 1.0964350498695428}, 'manual': {'prefill': 1.4723236444078867, 'decode': 1.4861267845848263}}, 'llama': {'dojo': {'prefill': 1.5395595526982313, 'decode': 1.359569747782982}, 'cerebras': {'prefill': 1.6248493870996075, 'decode': 1.2218465063415085}, 'manual': {'prefill': 1.5119613654796886, 'decode': 1.4736743347360854}}}, 8192: {'gpt': {'dojo': {'prefill': 1.2542239514772042, 'decode': 1.3070969347431183}, 'cerebras': {'prefill': 1.2610708099193668, 'decode': 1.2649646243630783}, 'manual': {'prefill': 1.2463516107922674, 'decode': 1.3435620213537574}}, 'opt': {'dojo': {'prefill': 1.3464599888234143, 'decode': 1.2897125606300743}, 'cerebras': {'prefill': 1.4736068959929522, 'decode': 1.2504378154492386}, 'manual': {'prefill': 1.26717469432824, 'decode': 1.321141125010046}}, 'qwen3moe': {'dojo': {'prefill': 1.1739977761361031, 'decode': 1.280849767893188}, 'cerebras': {'prefill': 1.1564228075041105, 'decode': 1.0936844891359199}, 'manual': {'prefill': 1.1716087573889917, 'decode': 1.4350366662550718}}, 'qwen': {'dojo': {'prefill': 1.293657434078635, 'decode': 1.279016526490388}, 'cerebras': {'prefill': 1.1182647316382237, 'decode': 1.2727830032138532}, 'manual': {'prefill': 1.3921059876819941, 'decode': 1.2813670830281378}}, 'qwen2moe': {'dojo': {'prefill': 1.366165771087266, 'decode': 1.3156107403771364}, 'cerebras': {'prefill': 1.2158565341122483, 'decode': 1.1480532907572933}, 'manual': {'prefill': 1.4591982033697906, 'decode': 1.444688747302978}}, 'llama': {'dojo': {'prefill': 1.3358136314575724, 'decode': 1.2249071886551237}, 'cerebras': {'prefill': 1.1835977147971937, 'decode': 1.096777491040152}, 'manual': {'prefill': 1.4413207853331085, 'decode': 1.3077471949203476}}}}

sizes    = [512, 2048, 8192]
backends = ['dojo', 'cerebras', 'manual']
backend_list = ['HW1', 'HW2', 'HW3']
backends_dict = {'dojo': 'HW1', 'cerebras': 'HW2', 'manual': 'HW3'}
models   = ['gpt', 'opt', 'qwen3moe', 'qwen', 'qwen2moe', 'llama']
models_dict = {'gpt': 'GPT-NeoX-20B', 'opt': 'OPT-30B', 'qwen3moe': 'Qwen3-MoE-30B', 'qwen': 'Qwen3-32B', 'llama': 'Llama-3-70B', 'qwen2moe': 'Qwen2-MoE-57B'}
metrics  = ['prefill', 'decode']

colors = {
    'gpt':   '#80ed99',
    'opt':   '#4cc9f0',
    'qwen3moe':  "#ffc971",
    'qwen':  '#db7c26',
    'llama': "#c77dff",
    'qwen2moe': '#ff87ab',
}
edgecolors = {
    'gpt':   '#06d6a0',
    'opt':   '#118ab2',
    'qwen3moe':  "#f2a65a",
    'qwen':  '#d8572a',
    'llama': "#7b2cbf",
    'qwen2moe': '#ff5d8f',
}


group_width     = 3
model_width     = group_width / len(models)
bar_width       = model_width * 0.45
metric_offset   = bar_width / 2
gap_between     = 0.3
x_groups        = np.arange(len(backends)) * (group_width + gap_between)

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharey=False)

y_limits = {
    512: (0.0, 2.33),
    2048: (0.0, 2.0),
    8192: (0.0, 1.86)
}

for ax, size in zip(axes, sizes):
    subset = data[size]
    ax.set_ylim(y_limits[size])
    for i, backend in enumerate(backends):
        base_x = x_groups[i]
        for j, model in enumerate(models):
            center = base_x + (j - (len(models)-1)/2) * model_width
            x_prefill = center - metric_offset
            x_decode  = center + metric_offset

            y_prefill = subset[model][backend]['prefill']
            y_decode  = subset[model][backend]['decode']

            ax.bar(x_prefill, y_prefill, bar_width,
                   color=colors[model],
                   edgecolor=edgecolors[model],
                   linewidth=4)
            ax.bar(x_decode, y_decode, bar_width,
                   color=colors[model],
                   edgecolor='black',
                   linewidth=3,
                   hatch='/')


    text_x_pos      = 0.02
    text_y_pos      = 0.95
    text_fontsize   = 28
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
    ax.set_xticklabels(backend_list, fontsize=36)

    ax.tick_params(axis='y', labelsize=26)
    ax.set_ylabel("Speedup", fontsize=36)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))

    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', color='gray', linewidth=1.2, zorder=0)

ax0 = axes[0]

model_handles = [
    Patch(facecolor=colors[m], edgecolor=edgecolors[m], label=models_dict[m], linewidth=3)
    for m in models
]
metric_handles = [
    Patch(facecolor='white', edgecolor='black', linewidth=3, label='prefill'),
    Patch(facecolor='white', edgecolor='black', linewidth=3, hatch='/', label='decode'),
]

leg1 = ax0.legend(
    handles=model_handles,
    loc='upper left',
    bbox_to_anchor=(-0.01, 1.6),
    ncol=len(models)//2,
    frameon=False,
    fontsize=26,
    columnspacing=0.5,
    handletextpad=0.2,
    borderpad=0.1,
)
ax0.add_artist(leg1)

leg2 = ax0.legend(
    handles=metric_handles,
    loc='upper right',
    bbox_to_anchor=(1.01, 1.6),
    ncol=1,
    frameon=False,
    fontsize=26,
    columnspacing=0.2,
    handletextpad=0.2,
    borderpad=0.1,
)

plt.tight_layout(pad=1.0)
plt.subplots_adjust(hspace=0.25)
plt.savefig("endtoend.pdf", dpi=300, bbox_inches='tight')
plt.savefig("endtoend.png", dpi=300, bbox_inches='tight')

import statistics


backends = ['dojo', 'cerebras', 'manual']

for hw in backends:
    values = []
    for size in data:
        for model in data[size]:
            values.append(data[size][model][hw]['prefill'])
            values.append(data[size][model][hw]['decode'])
    print(f"{hw}: min={min(values):.6f}, max={max(values):.6f}, geomean={statistics.geometric_mean(values):.6f}")

all_values = [
    data[size][model][hw][metric]
    for size in data
    for model in data[size]
    for hw in ['dojo', 'cerebras', 'manual']
    for metric in ['prefill', 'decode']
]

overall_geomean = statistics.geometric_mean(all_values)

print(f"Overall geomean: {overall_geomean:.6f}")
