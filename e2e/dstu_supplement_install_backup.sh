#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/ubuntu/crisp_hi_20261114
bash -n e2e/dstu_supplement_cos_backup.sh
entry='*/3 * * * * bash /home/ubuntu/crisp_hi_20261114/e2e/dstu_supplement_cos_backup.sh >> /home/ubuntu/crisp_dstu_supplement_cos_backup.log 2>&1'
schedule=$(mktemp)
crontab -l > "$schedule"
if ! grep -qxF "$entry" "$schedule"; then
  printf '%s\n' "$entry" >> "$schedule"
  crontab "$schedule"
fi
rm "$schedule"
nohup bash e2e/dstu_supplement_cos_backup.sh >> /home/ubuntu/crisp_dstu_supplement_cos_backup.log 2>&1 < /dev/null &
printf 'backup_pid=%s\n' "$!"
crontab -l | grep -F dstu_supplement
