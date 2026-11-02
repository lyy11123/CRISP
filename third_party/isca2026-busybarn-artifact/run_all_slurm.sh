#!/usr/bin/env bash

set -u
set -o pipefail

echo "============================================"
echo "  BusyBarn Artifact Evaluation (SLURM)"
echo "============================================"


echo ""
echo "=== Phase 1: Template-based experiments ==="

for d in fig_communication fig_intra_ch fig_intra_coreshape fig_intra_power fig_intra_multifaults; do
    echo "--- $d ---"
    (cd "$d" && make generate_exp && bash ./run_slurm.sh)
done


echo ""
echo "=== Phase 2: End-to-end model experiments ==="

(cd fig_endtoend && bash ./run_slurm.sh)


echo ""
echo "=== Phase 3: SA-based experiments ==="

(cd fig_convergence && bash ./run_slurm.sh)
(cd fig_heatmap && bash ./run_slurm.sh 1000)
(cd fig_ablation && bash ./run_slurm.sh)

echo ""
echo "============================================"
echo "  All jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "  After all jobs complete, run plotting locally:"
echo "    cd fig_communication && make draw_pic"
echo "    cd fig_intra_ch && make draw_pic"
echo "    cd fig_intra_coreshape && make draw_pic"
echo "    cd fig_intra_power && make draw_pic"
echo "    cd fig_intra_multifaults && make draw_pic"
echo "    cd fig_endtoend && python end_to_end.py && python endtoend_pic.py"
echo "    cd fig_convergence && python plot_convergence.py"
echo "    cd fig_ablation && make draw_pic"
echo "============================================"
