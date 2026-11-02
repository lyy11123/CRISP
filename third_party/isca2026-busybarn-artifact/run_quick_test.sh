#!/usr/bin/env bash

set -euo pipefail

JOBS="${1:-16}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils/job_pool.sh"

echo "============================================"
echo "  BusyBarn Quick Validation Test"
echo "  Figures: 8a, 10c, 11, 12(GPT)"
echo "  Max concurrent jobs: $JOBS"
echo "============================================"


echo ""
echo "=== [1/4] Fig. 8a: AllGather Communication ==="
cd "$SCRIPT_DIR/fig_communication"
make generate_exp
mkdir -p ./results ./logs
jobcount=0
reset_tracked_jobs
for script in ./py/allgather_*.py; do
  [ -e "$script" ] || continue
  base=$(basename "$script" .py)
  python "$script" > "./logs/${base}.log" 2>&1 &
  track_job "$!" "$base"
  jobcount=$((jobcount + 1))
  if (( jobcount >= JOBS )); then
    wait_for_tracked_jobs
    jobcount=0
  fi
done
wait_for_tracked_jobs
python allgather_synthetic_pic.py
echo "[1/4] Done: pic/allgather_synthetic.pdf"


echo ""
echo "=== [2/4] Fig. 10c: Power/Fault Sensitivity ==="
cd "$SCRIPT_DIR/fig_intra_power"
make generate_exp
bash ./run.sh "$JOBS"
make draw_pic
echo "[2/4] Done: pic/intra_mapping_power_pic.pdf"


echo ""
echo "=== [3/4] Fig. 11: Workload Distribution Heatmap ==="
cd "$SCRIPT_DIR/fig_heatmap"
bash ./run.sh 1000
echo "[3/4] Done: results/hot_transformer_block_combined_step1000.pdf"


echo ""
echo "=== [4/4] Fig. 12 (GPT only): End-to-End ==="
for d in intra_mapping_gpt_prefill intra_mapping_gpt_decode; do
  echo "  Generating: models/$d"
  cd "$SCRIPT_DIR/models/$d"
  make generate_exp
done
for d in intra_mapping_gpt_prefill intra_mapping_gpt_decode; do
  echo "  Running: models/$d"
  cd "$SCRIPT_DIR/models/$d"
  bash ./run.sh "$JOBS"
done
cd "$SCRIPT_DIR/fig_endtoend"
python end_to_end.py
python endtoend_pic_gpt.py
echo "[4/4] Done: fig_endtoend/endtoend_gpt.pdf"


echo ""
echo "============================================"
echo "  Quick validation complete."
echo "============================================"

echo ""
echo "=== Collecting results ==="
bash "$SCRIPT_DIR/collect_results.sh"
