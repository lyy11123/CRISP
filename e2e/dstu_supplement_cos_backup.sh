#!/usr/bin/env bash
set -Eeuo pipefail
export CRISP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CRISP_COS_DEST="cos://crisp/crisp/DSTU_SUPPLEMENT_20261118"
export CRISP_BACKUP_STATE="$HOME/.crisp_cos_backup_state_DSTU_SUPPLEMENT_20261118"
source <(sed '/^# The simulator/,$d' "$HOME/crisp_cos_backup.sh")
for file in "$ROOT"/e2e/dstu_supplement_*.py "$ROOT"/e2e/dstu_supplement_*.sh; do
  upload_if_ready "$file" input 10
done
while IFS= read -r -d '' file; do
  relative="${file#"$ROOT/e2e/dstu_supplement/"}"
  upload_if_ready "$file" "${relative%/*}" "$MIN_AGE"
done < <(find "$ROOT/e2e/dstu_supplement" -type f ! -name '*.part' ! -path '*/.inflight/*' ! -path '*/__pycache__/*' -print0)
echo "[$(date -Is)] S256 supplement backup cycle finished"
