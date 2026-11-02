#!/usr/bin/env bash
set -euo pipefail

DEFAULT_JOBS=16
MAX_JOBS="${1:-$DEFAULT_JOBS}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../utils/job_pool.sh"

mkdir -p ./results
jobcount=0
reset_tracked_jobs
for script in ./py/*.py; do
  [ -e "$script" ] || continue
  base=$(basename "$script" .py)
  python "$script" > "./results/${base}.txt" 2>&1 &
  track_job "$!" "$base"
  jobcount=$((jobcount + 1))
  if (( jobcount >= MAX_JOBS )); then
    wait_for_tracked_jobs
    jobcount=0
  fi
done
wait_for_tracked_jobs
echo "All done. Results in ./results/"
