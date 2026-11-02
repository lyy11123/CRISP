#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p ./results ./logs

HW_CFG="../src/platform/cfgs/wamis_hd_distributed.cfg"
HW_ARGS=(--hw-topology wamis_hdc --hw-cfg "$HW_CFG")
CONV_ARGS=(--steps 6000 --eval-interval 100)

echo "=== Phase 1: SA convergence traces (8 runs) ==="
for OP in ffn ln mha proj; do
  for VARIANT in busybarn gemini; do
    SUFFIX="${OP}_distributed_${VARIANT}"
    echo "Running: $SUFFIX"
    python sa_convergence.py --operator $OP --variant $VARIANT \
      "${HW_ARGS[@]}" "${CONV_ARGS[@]}" \
      > "./logs/${SUFFIX}.log" 2>&1
  done
done

echo "=== Phase 2: Brute-force baselines (4 runs) ==="
for OP in ffn ln mha proj; do
  SUFFIX="${OP}_distributed_brute"
  echo "Running: $SUFFIX"
  python "${OP}_distributed_brute.py" --rounds 1000000 --threads 4 \
    > "./logs/${SUFFIX}.log" 2>&1
done

echo "=== Phase 3: Plot ==="
python plot_convergence.py

echo "Done. Figure: results/sa_convergence.pdf"
