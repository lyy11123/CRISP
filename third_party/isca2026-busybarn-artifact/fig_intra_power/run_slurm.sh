#!/usr/bin/env bash

set -u
set -o pipefail

SUBMIT_DIR="${PWD}"
SCRIPT_DIR="./py"
LOG_DIR="logs"
ERR_DIR="errs"

MAX_INFLIGHT=1024
POLL_INTERVAL=30

PARTITION="i64m512ue"
CPUS_PER_TASK=1
MEMORY="4G"
PYTHON_BIN="python"

mkdir -p "$LOG_DIR" "$ERR_DIR"

submitted=0
failed=0
skipped=0

count_inflight_jobs() {
    squeue -h -u "$USER" -t PD,R | wc -l
}

wait_for_slot() {
    while [ "$(count_inflight_jobs)" -ge "$MAX_INFLIGHT" ]; do
        echo "Job limit reached (${MAX_INFLIGHT}). Waiting ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
    done
}

submit_one() {
    local script="$1"
    local base
    base="$(basename "$script" .py)"

    sbatch \
        -p "$PARTITION" \
        -n "$CPUS_PER_TASK" \
        --mem="$MEMORY" \
        -o "${LOG_DIR}/${base}_%j.txt" \
        -e "${ERR_DIR}/err_${base}_%j.txt" \
        -J "$base" \
        --wrap "cd \"$SUBMIT_DIR\" && \"$PYTHON_BIN\" \"$script\""
}

for script in "$SCRIPT_DIR"/*.py; do
    if [ ! -e "$script" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    wait_for_slot

    if submit_one "$script"; then
        submitted=$((submitted + 1))
        echo "Submitted: $script"
    else
        failed=$((failed + 1))
        echo "Failed: $script" >&2
    fi
done

echo "Done."
echo "Submitted: $submitted"
echo "Failed: $failed"