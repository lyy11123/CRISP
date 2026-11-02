#!/usr/bin/env bash

set -u
set -o pipefail

SUBMIT_DIR="${PWD}"

LOG_DIR="sbatch_log"
ERR_DIR="sbatch_err"

MAX_INFLIGHT=192
POLL_INTERVAL=30

PARTITION="i96m3tue"
CPUS_PER_TASK=1
MEMORY="32G"
PYTHON_BIN="python"

mkdir -p "$LOG_DIR" "$ERR_DIR" ./results

"$PYTHON_BIN" cfg.py

MODEL="qwen2.5-32b"
BATCH_SIZE=1
SEQ_LENGTH=4096
TOPOLOGY="wamis_hdc"
STEPS=1000
CHIPLETS_PER_LAYER=1
CFG_PATH="./cfg/config_ch1x1_bw96_co16x16_bw96_t16x16.cfg"

declare -A SPLITS
SPLITS[ffn]="1 8 1 32"
SPLITS[proj]="1 8 1 8"
SPLITS[mha]="1 4 1 1"
SPLITS[ln]="1 8 1 1 1 1"

declare -A LOSS
LOSS[ffn,busybarn]="2 0.1 1 0.1"
LOSS[ffn,gemini]="1 1 1 1"
LOSS[proj,busybarn]="1 0.1 1 0.1"
LOSS[proj,gemini]="1 1 1 1"
LOSS[mha,busybarn]="1 0.1 1 1"
LOSS[mha,gemini]="1 1 1 1"
LOSS[ln,busybarn]="1 0.1 0 1"
LOSS[ln,gemini]="1 1 0 1"

declare -A RT
RT[ffn,busybarn]=0.3;  RT[ffn,gemini]=0.3
RT[proj,busybarn]=0.5; RT[proj,gemini]=0.5
RT[mha,busybarn]=0.3;  RT[mha,gemini]=0.3
RT[ln,busybarn]=0.5;   RT[ln,gemini]=0.5

OPERATORS=(ffn proj mha ln)
VARIANTS=(busybarn gemini)

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

run_id=1

for op in "${OPERATORS[@]}"; do
  split_degree="${SPLITS[$op]}"
  for var in "${VARIANTS[@]}"; do
    loss_ratio="${LOSS[$op,$var]}"
    rt="${RT[$op,$var]}"
    for barrier in "" "--barrier"; do
      b_tag=""
      [ -n "$barrier" ] && b_tag="_barrier"

      if [ "$var" = "busybarn" ]; then
        reroute_opts=("--no-reroute" "")
        reroute_tags=("" "_reroute")
      else
        reroute_opts=("" "--reroute")
        reroute_tags=("" "_reroute")
      fi

      for i in 0 1; do
        reroute_flag="${reroute_opts[$i]}"
        r_tag="${reroute_tags[$i]}"
        outname="${op}_${var}_ch1x1_bw96_16x16_sp8${b_tag}${r_tag}"

        wait_for_slot

        if sbatch \
            -p "$PARTITION" \
            -n "$CPUS_PER_TASK" \
            --mem="$MEMORY" \
            -o "${LOG_DIR}/${outname}_%j.txt" \
            -e "${ERR_DIR}/err_${outname}_%j.txt" \
            -J "${outname}" \
            --wrap "cd \"$SUBMIT_DIR\" && \"$PYTHON_BIN\" sa_experiment.py \
                --operator ${op} \
                --variant ${var} \
                --model ${MODEL} \
                --batch-size ${BATCH_SIZE} \
                --seq-length ${SEQ_LENGTH} \
                --split-degree ${split_degree} \
                --hw-topology ${TOPOLOGY} \
                --hw-cfg ${CFG_PATH} \
                --steps ${STEPS} \
                --loss-ratio ${loss_ratio} \
                --random-threshold ${rt} \
                --chiplets-per-layer ${CHIPLETS_PER_LAYER} \
                --profile \
                ${barrier} \
                ${reroute_flag}"; then
            submitted=$((submitted + 1))
            echo "Submitted: ${outname}"
        else
            failed=$((failed + 1))
            echo "Failed: ${outname}" >&2
        fi

        run_id=$((run_id + 1))
      done
    done
  done
done

echo "Done."
echo "Submitted: $submitted"
echo "Failed: $failed"
