#!/usr/bin/env bash

set -u
set -o pipefail

SUBMIT_DIR="${PWD}"
EVAL_DIR="${SUBMIT_DIR}/.."

echo "Generating model experiments..."
for d in "${EVAL_DIR}"/models/*/; do
    [ -d "$d" ] || continue
    (cd "$d" && make generate_exp)
done

echo "Submitting model experiments..."
for d in "${EVAL_DIR}"/models/*/; do
    [ -d "$d" ] || continue
    [ -f "$d/run_slurm.sh" ] || continue
    echo "  $(basename "$d")"
    (cd "$d" && bash ./run_slurm.sh)
done

echo "All model jobs submitted."
echo "After all jobs complete, run:"
echo "  cd fig_endtoend && python end_to_end.py && python endtoend_pic.py"
