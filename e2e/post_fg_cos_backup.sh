#!/usr/bin/env bash
# Back up this experiment using the existing private COS configuration.
set -Eeuo pipefail
export CRISP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CRISP_COS_DEST="cos://crisp/crisp/POST_FG_20261113"
export CRISP_BACKUP_STATE="$HOME/.crisp_cos_backup_state_POST_FG_20261113"
source <(sed '/^# The simulator/,$d' "$HOME/crisp_cos_backup.sh")

for file in "$ROOT"/BUNDLE*.json "$ROOT"/*.start.json "$ROOT"/e2e/post_fg_*.py "$ROOT"/e2e/post_fg_*.sh "$ROOT"/e2e/POST_FG_PLAN.md; do
  upload_if_ready "$file" input 10
done
upload_if_ready "$HOME/POST_FG_initial.tar.gz" input 0
for group in manifest evidence tables figures out; do
  folder="$ROOT/e2e/post_fg/$group"
  [[ -d "$folder" ]] || continue
  while IFS= read -r -d '' file; do
    relative="${file#"$ROOT/e2e/post_fg/"}"
    upload_if_ready "$file" "${relative%/*}" "$MIN_AGE"
  done < <(find "$folder" -type f ! -name '*.part' ! -path '*/.inflight/*' ! -path '*/__pycache__/*' -print0)
done
echo "[$(date -Is)] Post-F/G backup cycle finished"
