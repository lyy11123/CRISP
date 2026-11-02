#!/usr/bin/env bash

set -euo pipefail

JOBS="${1:-16}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils/job_pool.sh"

echo "============================================"
echo "  BusyBarn Artifact Evaluation"
echo "  Max concurrent jobs: $JOBS"
echo "============================================"


echo ""
echo "=== Phase 1: Old backend experiments ==="

reset_tracked_jobs
(echo "[fig_communication] generate + run + plot"
 cd fig_communication && make generate_exp && make run_scripts && make draw_pic) &
track_job "$!" "fig_communication"

(echo "[fig_intra_ch] generate + run + plot"
 cd fig_intra_ch && make generate_exp && bash ./run.sh "$JOBS" && make draw_pic) &
track_job "$!" "fig_intra_ch"

(echo "[fig_intra_coreshape] generate + run + plot"
 cd fig_intra_coreshape && make generate_exp && bash ./run.sh "$JOBS" && make draw_pic) &
track_job "$!" "fig_intra_coreshape"

(echo "[fig_intra_power] generate + run + plot"
 cd fig_intra_power && make generate_exp && bash ./run.sh "$JOBS" && make draw_pic) &
track_job "$!" "fig_intra_power"

(echo "[fig_intra_multifaults] generate + run + plot"
 cd fig_intra_multifaults && make generate_exp && bash ./run.sh "$JOBS" && make draw_pic) &
track_job "$!" "fig_intra_multifaults"

wait_for_tracked_jobs
echo "Phase 1 complete."


echo ""
echo "=== Phase 2: End-to-end model experiments ==="

for d in models/*/; do
  (cd "$d" && make generate_exp)
done

reset_tracked_jobs
for d in models/*/; do
  echo "Running: $d"
  (cd "$d" && bash ./run.sh "$JOBS") &
  track_job "$!" "$(basename "$d")"
done
wait_for_tracked_jobs

echo "[fig_endtoend] aggregate + plot"
(cd fig_endtoend && python end_to_end.py && python endtoend_pic.py)

echo "Phase 2 complete."


echo ""
echo "=== Phase 3: New backend experiments ==="

reset_tracked_jobs
(echo "[fig_convergence] SA convergence + brute baselines + plot"
 cd fig_convergence && bash ./run.sh) &
track_job "$!" "fig_convergence"

(echo "[fig_heatmap] heatmap (runs SA internally)"
 cd fig_heatmap && bash ./run.sh 1000) &
track_job "$!" "fig_heatmap"

wait_for_tracked_jobs

echo "[fig_ablation] ablation study + runtime breakdown (32 runs, memory intensive)"
(cd fig_ablation && bash ./run.sh "$JOBS" && make draw_pic)

echo ""
echo "============================================"
echo "  All figures generated successfully."
echo "============================================"

echo ""
echo "=== Collecting results ==="
bash ./collect_results.sh
