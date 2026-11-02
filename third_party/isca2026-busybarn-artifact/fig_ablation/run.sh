#!/usr/bin/env bash

set -euo pipefail

DEFAULT_JOBS=4
MAX_JOBS="${1:-$DEFAULT_JOBS}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../utils/job_pool.sh"

python cfg.py

MODEL="qwen2.5-32b"
BATCH_SIZE=1
SEQ_LENGTH=4096
TOPOLOGY="wamis_hdc"
STEPS=1000
CHIPLETS_PER_LAYER=1
CFG_PATH="./cfg/config_ch1x1_bw96_co16x16_bw96_t16x16.cfg"

LOG_DIR="sbatch_log"
mkdir -p "$LOG_DIR" ./results

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

jobcount=0
run_id=1
reset_tracked_jobs

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
        logfile="${LOG_DIR}/${outname}_${run_id}.txt"

        echo "Running: ${outname}"
        python sa_experiment.py \
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
          ${reroute_flag} \
          > "$logfile" 2>&1 &
        track_job "$!" "${outname}_${run_id}"

        run_id=$((run_id + 1))
        jobcount=$((jobcount + 1))
        if (( jobcount >= MAX_JOBS )); then
          wait_for_tracked_jobs
          jobcount=0
        fi
      done
    done
  done
done

wait_for_tracked_jobs
echo "All done. Logs in $LOG_DIR/"
echo "Run 'python plot_16x16_1x1.py' to generate figure."
