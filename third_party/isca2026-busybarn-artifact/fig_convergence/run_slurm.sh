#!/usr/bin/env bash

set -u
set -o pipefail

SUBMIT_DIR="${PWD}"

LOG_DIR="logs"
ERR_DIR="errs"

MAX_INFLIGHT=192
POLL_INTERVAL=30

PARTITION="i96m3tue"
CPUS_PER_TASK=1
MEMORY="32G"
PYTHON_BIN="python"

mkdir -p "$LOG_DIR" "$ERR_DIR" ./results

HW_CFG="../src/platform/cfgs/wamis_hd_distributed.cfg"

submitted=0
failed=0

count_inflight_jobs() {
    squeue -h -u "$USER" -t PD,R | wc -l
}

wait_for_slot() {
    while [ "$(count_inflight_jobs)" -ge "$MAX_INFLIGHT" ]; do
        echo "Job limit reached (${MAX_INFLIGHT}). Waiting ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
    done
}

echo "=== Phase 1: SA convergence traces (8 jobs) ==="
for OP in ffn ln mha proj; do
  for VARIANT in busybarn gemini; do
    SUFFIX="${OP}_distributed_${VARIANT}"
    outname="${SUFFIX}_convergence"

    wait_for_slot

    if sbatch \
        -p "$PARTITION" \
        -n "$CPUS_PER_TASK" \
        --mem="$MEMORY" \
        -o "${LOG_DIR}/${outname}_%j.log" \
        -e "${ERR_DIR}/err_${outname}_%j.log" \
        -J "${outname}" \
        --wrap "cd \"$SUBMIT_DIR\" && \"$PYTHON_BIN\" sa_convergence.py \
            --operator $OP --variant $VARIANT \
            --hw-topology wamis_hdc --hw-cfg $HW_CFG \
            --steps 6000 --eval-interval 100"; then
        submitted=$((submitted + 1))
        echo "Submitted: $outname"
    else
        failed=$((failed + 1))
        echo "Failed: $outname" >&2
    fi
  done
done

echo "=== Phase 2: Brute-force baselines (4 jobs) ==="
for OP in ffn ln mha proj; do
  SUFFIX="${OP}_distributed_brute"

  wait_for_slot

  if sbatch \
      -p "$PARTITION" \
      -n "$CPUS_PER_TASK" \
      --mem="$MEMORY" \
      -o "${LOG_DIR}/${SUFFIX}_%j.log" \
      -e "${ERR_DIR}/err_${SUFFIX}_%j.log" \
      -J "${SUFFIX}" \
      --wrap "cd \"$SUBMIT_DIR\" && \"$PYTHON_BIN\" ${OP}_distributed_brute.py \
          --rounds 1000000 --threads 4"; then
      submitted=$((submitted + 1))
      echo "Submitted: $SUFFIX"
  else
      failed=$((failed + 1))
      echo "Failed: $SUFFIX" >&2
  fi
done

echo "Done."
echo "Submitted: $submitted"
echo "Failed: $failed"
echo "Run 'python plot_convergence.py' after all jobs complete."
