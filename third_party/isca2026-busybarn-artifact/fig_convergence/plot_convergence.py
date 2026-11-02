"""
Plot SA convergence for transformer block performance (2*LN + MHA + Proj + FFN).

Reads *_convergence.txt files from results/ and produces a single figure
with BusyBarn vs Gemini lines.

Usage:
    python plot_convergence.py
"""
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

file_path = os.path.dirname(os.path.realpath(__file__))
RESULTS_DIR = os.path.join(file_path, "results")


plt.rcParams['font.family'] = 'sans-serif'

COLORS = {
    'busybarn': "#a1dff4",
    'gemini':   "#e9b174",
}
MARKERS = {
    'busybarn': 'o',
    'gemini':   's',
}
LABELS = {
    'busybarn': 'BusyBarn',
    'gemini':   'Gemini',
}


def read_brute_best(op):
    """Read best found time from brute-force result file."""
    fname = f"{op}_distributed_brute.txt"
    fpath = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  Warning: {fname} not found")
        return None
    with open(fpath) as f:
        for line in f:
            if line.strip().startswith("Best found"):
                return int(line.strip().split(":")[-1].strip())
    return None


def read_convergence(op, variant):
    """Read convergence file, return dict {step: time_cost}."""
    fname = f"{op}_distributed_{variant}_convergence.txt"
    fpath = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  Warning: {fname} not found")
        return None

    data = {}
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Step'):
                continue
            parts = line.split()
            data[int(parts[0])] = int(parts[1])
    return data


def compute_transformer(variant):
    """Return (steps, transformer_latency) where latency = 2*LN + MHA + Proj + FFN."""
    ffn = read_convergence('ffn', variant)
    ln  = read_convergence('ln',  variant)
    mha = read_convergence('mha', variant)
    proj = read_convergence('proj', variant)

    if any(d is None for d in [ffn, ln, mha, proj]):
        return None, None

    common_steps = sorted(set(ffn) & set(ln) & set(mha) & set(proj))
    steps = np.array(common_steps)
    latency = np.array([2 * ln[s] + mha[s] + proj[s] + ffn[s] for s in common_steps])
    return steps, latency


fig, ax = plt.subplots(figsize=(12, 6))

mark_every = 10

busybarn_data = None
gemini_data = None

for variant in ['busybarn', 'gemini']:
    steps, latency = compute_transformer(variant)
    if steps is None:
        continue

    if variant == 'busybarn':
        busybarn_data = (steps, latency)
    elif variant == 'gemini':
        gemini_data = (steps, latency)

    ax.plot(steps, latency,
            color=COLORS[variant],
            linewidth=10.0,
            label=LABELS[variant])


brute_vals = {op: read_brute_best(op) for op in ['ffn', 'ln', 'mha', 'proj']}
if all(v is not None for v in brute_vals.values()):
    brute_bound = 2 * brute_vals['ln'] + brute_vals['mha'] + brute_vals['proj'] + brute_vals['ffn']
    ax.axhline(brute_bound, color='#e63946', linestyle='--', linewidth=1.5)
    ax.text(0.225, 0.10, f'After 1M brute-force searches: {brute_bound:.2e} cycles',
            transform=ax.transAxes,
            ha='left', va='center', fontsize=24, color='#e63946')

    if busybarn_data is not None:
        steps, latency = busybarn_data
        idx_1000 = np.where(steps == 1000)[0]
        if len(idx_1000) > 0:
            busybarn_1000 = latency[idx_1000[0]]
            gap = busybarn_1000 - brute_bound
            ratio = (gap / brute_bound) * 100

            ax.annotate('', xy=(1000, brute_bound), xytext=(1000, busybarn_1000),
                       arrowprops=dict(arrowstyle='<->', color='black', lw=2))

            ax.text(0.19, 0.14, f'{ratio:.1f}%',
                   transform=ax.transAxes,
                   fontsize=20, va='center', ha='right')

ax.set_xlabel('SA Iteration', fontsize=26)
ax.set_ylabel('Latency (cycles)', fontsize=26)
ax.tick_params(axis='both', labelsize=24)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(24)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
ax.legend(fontsize=24, frameon=False, loc='upper right')

plt.tight_layout()

out_pdf = os.path.join(RESULTS_DIR, "sa_convergence.pdf")
out_png = os.path.join(RESULTS_DIR, "sa_convergence.png")
plt.savefig(out_pdf, dpi=300)
plt.savefig(out_png, dpi=150)

print("\n=== Convergence Summary ===")
for name, vdata in [("BusyBarn", busybarn_data), ("Gemini", gemini_data)]:
    if vdata is not None:
        steps, latency = vdata
        print(f"  {name}: initial={latency[0]:.2e}  final={latency[-1]:.2e}  "
              f"reduction={((latency[0]-latency[-1])/latency[0]*100):.1f}%")
if all(v is not None for v in brute_vals.values()):
    print(f"  Brute-force bound: {brute_bound:.2e}")
    if busybarn_data is not None:
        _, blat = busybarn_data
        print(f"  BusyBarn final vs brute: gap={((blat[-1]-brute_bound)/brute_bound*100):.1f}%")
    if gemini_data is not None:
        _, glat = gemini_data
        print(f"  Gemini final vs brute:   gap={((glat[-1]-brute_bound)/brute_bound*100):.1f}%")
if busybarn_data is not None and gemini_data is not None:
    _, blat = busybarn_data
    _, glat = gemini_data
    print(f"  BusyBarn over Gemini (final): {glat[-1]/blat[-1]:.3f}x")
