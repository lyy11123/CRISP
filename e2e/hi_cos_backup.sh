#!/usr/bin/env bash
set -Eeuo pipefail
export CRISP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CRISP_COS_DEST="cos://crisp/crisp/HIJ_20261114"
export CRISP_BACKUP_STATE="$HOME/.crisp_cos_backup_state_HIJ_20261114"
source <(sed '/^# The simulator/,$d' "$HOME/crisp_cos_backup.sh")
for file in "$ROOT"/e2e/hi_*.py "$ROOT"/e2e/hi_*.sh "$ROOT"/e2e/j_timing.py "$ROOT"/e2e/HIJ_STAGED_PLAN.md; do
  upload_if_ready "$file" input 10
done
while IFS= read -r -d '' file; do
  relative="${file#"$ROOT/e2e/hi/"}"
  upload_if_ready "$file" "${relative%/*}" "$MIN_AGE"
done < <(find "$ROOT/e2e/hi" -type f ! -name '*.part' ! -path '*/.inflight/*' ! -path '*/__pycache__/*' -print0)
echo "[$(date -Is)] H/I/J backup cycle finished"
