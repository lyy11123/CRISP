#!/usr/bin/env bash

set -u
set -o pipefail

SUBMIT_DIR="${PWD}"

STEPS="${1:-1000}"

PARTITION="i96m3tue"
CPUS_PER_TASK=1
MEMORY="32G"
PYTHON_BIN="python"

mkdir -p ./results

echo "Submitting: heatmap (steps=$STEPS)"
if sbatch \
    -p "$PARTITION" \
    -n "$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    -o "results/heatmap_%j.log" \
    -e "results/err_heatmap_%j.log" \
    -J "bb_heatmap" \
    --wrap "cd \"$SUBMIT_DIR\" && \"$PYTHON_BIN\" plot_hot_transformer_block.py --steps ${STEPS}"; then
    echo "Submitted 1 job."
else
    echo "Failed to submit." >&2
fi
