#!/usr/bin/env bash

JOB_PIDS=()
JOB_LABELS=()

reset_tracked_jobs() {
  JOB_PIDS=()
  JOB_LABELS=()
}

track_job() {
  JOB_PIDS+=("$1")
  JOB_LABELS+=("$2")
}

wait_for_tracked_jobs() {
  local i
  local pid
  local label
  local failed=0

  for i in "${!JOB_PIDS[@]}"; do
    pid="${JOB_PIDS[$i]}"
    label="${JOB_LABELS[$i]}"
    if ! wait "$pid"; then
      echo "Error: ${label} failed." >&2
      failed=1
    fi
  done

  reset_tracked_jobs
  return "$failed"
}
