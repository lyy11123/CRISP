#!/usr/bin/env bash
# After a stage 2 part: self-check 3 where it applies, markdown summary appended to PROGRESS, commit and push.
# Usage: bash finish_part.sh <part> "<configs>" "<arms for check 3, may be empty>" "<title>"
R=/e/58Institute/PhD_TERM3/CRISP
cd "$R/cpcm_sa" || exit 1
export CPCM_SA_STAGE=s2 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
if [ -n "$3" ]; then
  python cpcm_sa.py check3 --configs $2 --arms $3 --jobs 4 > "out/s2/check3_$1.out" 2>&1
  cp out/s2/check3.json "out/s2/check3_$1.json"
fi
python part_md.py "$1" > "out/s2/md_$1.md"
{ printf '\n## 2026-11-05 第 2 阶段部分结果：%s（Claude Code）\n\n' "$4"
  printf '后端 10 种子均值；三元组为（总延迟 / 未掩盖纯计算 / 未掩盖纯通信）。命令：`bash run_parts.sh %s`。\n\n' "$1"
  tail -n 1 "out/s2/run_$1.out"; echo; cat "out/s2/md_$1.md"; } >> PROGRESS.md
cd "$R" && git add cpcm_sa/PROGRESS.md cpcm_sa/part_md.py cpcm_sa/finish_part.sh cpcm_sa/out/s2/*.out cpcm_sa/out/s2/*.json cpcm_sa/out/s2/*.csv cpcm_sa/out/s2/*.md 2>/dev/null
git commit -q -m "[claude] cpcm_sa: stage 2 part $1 done

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git pull -q --rebase --autostash origin main && git push -q origin main
git log --oneline -1
cat "cpcm_sa/out/s2/md_$1.md"
