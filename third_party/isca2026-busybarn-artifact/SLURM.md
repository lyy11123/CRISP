# SLURM Usage

All experiments support both local and SLURM execution. See `README.md` for
local execution instructions.

---

## Quickstart

```bash
# 1. Edit PARTITION in each run_slurm.sh to match your cluster
#    (see "Configuration" section below)

# 2. Submit all experiments
bash ./run_all_slurm.sh

# 3. Monitor
squeue -u $USER

# 4. After all jobs complete, run plotting locally
#    (instructions printed by run_all_slurm.sh)

# 5. Collect all figures and summaries
bash ./collect_results.sh
```

---

## Configuration

Each `run_slurm.sh` has SLURM parameters at the top of the file. Before first use,
edit the `PARTITION` and `MEMORY` variables in each script to match your cluster:

```bash
# In each run_slurm.sh (near the top):
PARTITION="your_partition"    # -p : your cluster partition or job queue
MEMORY="4G"                   # --mem : memory per job
```

Current defaults by experiment type:

| Experiments | Memory |
|-------------|--------|
| fig_communication, fig_intra_ch, fig_intra_coreshape, fig_intra_power, models/* | 4G |
| fig_intra_multifaults, fig_convergence, fig_heatmap, fig_ablation | 32G |


## Per-experiment SLURM

Each experiment has a `run_slurm.sh` that submits jobs via `sbatch --wrap`:
```bash
cd fig_intra_ch/
make generate_exp     # always run locally first
make run_slurm        # or: bash ./run_slurm.sh
# wait for jobs to finish, then:
make draw_pic
```

## Full SLURM run

```bash
bash ./run_all_slurm.sh    # generates locally, submits all compute via sbatch
# after all jobs complete, run plotting commands printed at the end
```

## Makefile targets

| Target | What it does |
|--------|-------------|
| `generate_exp` | Generate configs + experiment scripts (local) |
| `run_scripts` | Run experiments locally with bash job pool |
| `run_slurm` | Submit experiments to SLURM via sbatch |
| `draw_pic` | Generate figures from results (local) |
| `clean` | Remove all generated artifacts |
