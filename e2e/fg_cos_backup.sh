#!/usr/bin/env bash
# Reuse the server's existing COS uploader for this F/G output tree.
set -Eeuo pipefail
export CRISP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CRISP_COS_DEST="cos://crisp/crisp/FG_20261113"
export CRISP_BACKUP_STATE="$HOME/.crisp_cos_backup_state_FG_20261113"
source <(sed '/^# The simulator/,$d' "$HOME/crisp_cos_backup.sh")

OUT="$ROOT/e2e/fg/out/main"
for file in "$ROOT"/BUNDLE*.json "$ROOT"/*.start.json "$OUT"/*.batch.json "$OUT"/*.status.json "$OUT"/release*.json "$OUT/scheduler_policy.json"; do
  upload_if_ready "$file" meta 10
done
for file in "$ROOT/e2e/fg/manifest"/*.json "$ROOT/e2e/fg/out/"*jobs.json; do
  upload_if_ready "$file" manifests 10
done
for file in "$HOME"/FG_pilot_35711b9.tar.gz "$HOME"/FG_full_20261113.tar.gz "$ROOT"/e2e/fg_cos_backup.sh "$ROOT"/e2e/fg_*.py; do
  upload_if_ready "$file" input 0
done
for group in results snapshots raw exports selection faults joint_cache specs receipts logs; do
  [[ -d "$OUT/$group" ]] || continue
  while IFS= read -r -d '' file; do
    relative="${file#"$OUT/"}"
    upload_if_ready "$file" "${relative%/*}" "$MIN_AGE"
  done < <(find "$OUT/$group" -type f ! -name '*.part' -print0)
done
echo "[$(date -Is)] F/G backup cycle finished"
