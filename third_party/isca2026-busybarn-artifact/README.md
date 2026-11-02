# BusyBarn — Artifact Evaluation

Reproduce all experimental figures for the ISCA 2026 paper:
*"Mapping and Communication Optimizations with Failure Tolerance for Wafer-Scale LLM Inference"*

---

## Full reproduction 

```bash
# 1. Setup
conda create --name busybarn python=3.9
conda activate busybarn
pip install -r requirements.txt

# 2. Quick test (~10 min):
cd fig_heatmap/
bash ./run.sh 1000

# 3. Run everything (less than 12000 cores x hours)
bash ./run_all.sh 16    # 16 = max concurrent jobs (can be set as real usable cores)

# 4. The script automatically collect all figures and speedup summaries into output/
# To re-collect manually:
bash ./collect_results.sh
```

### Subset reproduction

```bash
# Quick validation (less than 3 hours on 13900K)
bash ./run_quick_test.sh 16    # Fig. 8a, 10c, 11, 12(GPT)
```

### SLURM cluster

See [SLURM.md](SLURM.md) for cluster setup, per-experiment submission, and
the full SLURM run workflow.

---

## Figure Index

| Fig. | Output PDF |
|------|-----------|
| Fig. 8a | `output/fig08a_allgather.pdf` |
| Fig. 8b | `output/fig08b_alltoall.pdf` |
| Fig. 9 | `output/fig09_failures.pdf` |
| Fig. 10a | `output/fig10a_diegroup.pdf` |
| Fig. 10b | `output/fig10b_coreshape.pdf` |
| Fig. 10c | `output/fig10c_power.pdf` |
| Fig. 10d | `output/fig10d_multifaults.pdf` |
| Fig. 11 | `output/fig11_heatmap.pdf` |
| Fig. 12 | `output/fig12_endtoend.pdf` |
| Fig. 13 | `output/fig13_convergence.pdf` |
| Fig. 14 | `output/fig14_ablation.pdf` |
| Fig. 15 | `output/fig15_breakdown.pdf` |

---

## Estimated Time

All times measured on Intel Xeon Platinum 8358P (32C) and Intel Xeon Gold 6348H (24C).

| Experiment | Tasks | Time per Task |
|------------|------:|---------------|
| **Fig. 8-9: Communication** | 1265 | < 10 min |
| | 238 | < 30 min |
| **Fig. 10a: Die group shape** | 1454 | < 10 min |
| | 173 | < 30 min |
| **Fig. 10b: Core shape** | 1081 | < 10 min |
| **Fig. 10c: Power/fault** | 812 | < 5 min |
| **Fig. 10d: Multi-fault** | 728 | < 30 min |
| | 88 | < 60 min |
| **Fig. 11: Heatmap** | 1 | ~5 min |
| **Fig. 13: Convergence** | 6 | < 5 min |
| | 6 | ~6 hours |
| **Fig. 14+15: Ablation + Breakdown** (32 GB/task) | 28 | < 10 min |
| | 4 | < 30 min |

**Fig. 12: End-to-End** (12 model directories, the most time-consuming experiment):

| Model | Phase | Fast Tasks (< 10 min) | Slow Tasks |
|-------|-------|-----------------------|------------|
| GPT-NeoX-20B | prefill | 1143 | 35 tasks < 30 min |
| | decode | 1182 (< 5 min) | -- |
| OPT-30B | prefill | 2852 | 361 tasks < 30 min |
| | decode | 1182 (< 5 min) | -- |
| Llama-3-70B | prefill | 949 | 230 tasks < 60 min |
| | decode | 1186 (< 5 min) | -- |
| Qwen3-32B | prefill | 947 | 240 tasks ~60 min |
| | decode | 1050 | 129 tasks ~30 min |
| Qwen2-MoE-57B | prefill | 822 | 361 tasks ~60 min |
| | decode | 1172 | 6 tasks < 30 min |
| Qwen3-MoE-30B | prefill | 955 | 225 tasks ~60 min |
| | decode | 1174 | 6 tasks < 4 min |

## Quick Validation (Limited Resources)

`run_quick_test.sh` validates
core functionality with 5 representative experiments:

```bash
bash ./run_quick_test.sh 16
```

| Step | Figure | What it validates | Approx. Time |
|------|--------|-------------------|-------------|
| 1 | Fig. 8a | BALD (AllGather communication) | ~30 min |
| 2 | Fig. 10c | Intra-die mapping + fault tolerance | ~15 min |
| 3 | Fig. 11 | SA optimization + workload visualization | ~5 min |
| 4 | Fig. 12 (GPT only) | Full end-to-end pipeline (GPT-NeoX-20B) | ~30 min |

**Coverage:** Communication/routing, intra-die mapping with faults,
workload heatmap visualization, and end-to-end integration.
Omits AllToAll (Fig. 8b), failure patterns (Fig. 9), shape sensitivity (Fig. 10a-b),
multi-fault scaling (Fig. 10d), SA convergence (Fig. 13),
ablation + breakdown (Fig. 14-15), and 5 of 6 end-to-end models.

**Output figures:**
- `fig_communication/pic/allgather_synthetic.pdf`
- `fig_intra_power/pic/intra_mapping_power_pic.pdf`
- `fig_heatmap/results/hot_transformer_block_combined_step1000.pdf`
- `fig_endtoend/endtoend_gpt.pdf`

---

## Per-Figure Reproduction Instructions

See [FIGURES.md](FIGURES.md) for per-figure commands (Local / SLURM), captions,
descriptions, parameters, and speedup summaries.

---

## Directory Structure

```
evaluation/
├── run_all.sh                  # Master script (local)
├── run_all_slurm.sh            # Master script (SLURM)
├── run_quick_test.sh           # Quick validation (6 figures)
├── collect_results.sh          # Collect figures + summaries to output/
├── requirements.txt            # Python dependencies
├── src/                        # Framework source code
├── utils/                      # Shared utilities
├── tool/                       # Visualization (timeline.py)
├── fig_communication/          # Fig. 8-9: AllGather, AllToAll, Failures
├── fig_intra_ch/               # Fig. 10a: Die group shape sensitivity
├── fig_intra_coreshape/        # Fig. 10b: Core shape sensitivity
├── fig_intra_power/            # Fig. 10c: Power/fault sensitivity
├── fig_intra_multifaults/      # Fig. 10d: Multi-fault scaling
├── fig_heatmap/                # Fig. 11: Workload distribution heatmap
├── fig_endtoend/               # Fig. 12: End-to-end model comparison
├── models/                     # 12 model experiment dirs for Fig. 12
├── fig_convergence/            # Fig. 13: SA convergence analysis
├── fig_ablation/               # Fig. 14+15: Ablation + runtime breakdown
└── output/                     # Collected figures + summary.txt (generated)
```

## Collected Output

After experiments complete, `collect_results.sh` (called automatically by `run_all.sh`
and `run_quick_test.sh`) gathers all figures and speedup summaries into `output/`:

```
output/
├── summary.txt                 # Speedup numbers for all figures
├── fig08a_allgather.pdf
├── fig08b_alltoall.pdf
├── fig09_failures.pdf
├── fig10a_diegroup.pdf
├── fig10b_coreshape.pdf
├── fig10c_power.pdf
├── fig10d_multifaults.pdf
├── fig11_heatmap.pdf
├── fig12_endtoend.pdf          # Full (run_all.sh)
├── fig12_endtoend_gpt.pdf      # GPT only (run_quick_test.sh)
├── fig13_convergence.pdf
├── fig14_ablation.pdf
└── fig15_breakdown.pdf
```

Missing figures (e.g., when using `run_quick_test.sh`) are skipped and reported.
To re-collect after manual runs: `bash ./collect_results.sh`

---

## Experiment Architecture

All template-based experiments (Fig. 8-10, 12) follow the same pipeline:

```
cfg.py          →  ./cfg/*.cfg          (hardware configurations)
{ln,mha,proj,ffn}.py  →  ./py/*.py     (parameterized experiment scripts)
run.sh          →  ./results/*.txt      (timing output: "SA time cost: (a, b, c)")
*_pic.py        →  ./pic/*.pdf          (publication figures + speedup printout)
```

SA-based experiments (Fig. 13-15) use `sa_experiment.py` as the unified runner (Fig. 14+15 share runs):
```
cfg.py          →  ./cfg/*.cfg
sa_experiment.py --operator X --variant Y [--barrier] [--profile]
plot_*.py       →  ./results/*.pdf
```

## Result Format

Template-based results are parsed with regex:
```
SA time cost: (compute_cycles, overlap_cycles, comm_cycles)
```

Convergence results use tab-separated format:
```
Step    Time_cost    Energy
0       3485953      ...
100     3200100      ...
```

## Hardware Requirements

- RAM: 32 GB minimum 
- CPU: Multi-core recommended (experiments run in parallel)
- Disk: ~5 GB for generated scripts and results

## Stochastic Reproducibility

SA optimization is seeded (`--seed 123` by default) but results may vary slightly across
platforms due to floating-point ordering. Figures should be qualitatively identical —
same trends, same relative ordering — but absolute numbers may differ by a few percent.

