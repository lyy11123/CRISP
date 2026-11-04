#!/usr/bin/env bash
# Stage 2 by part (handoff 10.3; user 2026-11-05: report after each part). A0 of the non-MoE configs ran first
# (run_s2.sh a0, self-check 1). Finished runs are skipped (a run's sink exists).
# Usage: bash /e/58Institute/PhD_TERM3/CRISP/cpcm_sa/run_parts.sh <ffn|ffn14|opt|mha|pscan|wall|moe>
HERE=/e/58Institute/PhD_TERM3/CRISP/cpcm_sa
cd "$HERE" || exit 1
export CPCM_SA_STAGE=s2
mkdir -p out/s2
ALL="A0 A1 A2 A3 A3CB A3h"
case "$1" in
  ffn)   C="ffn ffn_f4 ffn_f5"; A=$ALL; R=$ALL ;;
  ffn14) C="ffn14"; A=$ALL; R=$ALL ;;
  opt)   C="opt"; A=$ALL; R=$ALL ;;
  mha)   C="mha mha_f4 mha_f5"; A=$ALL; R=$ALL ;;
  pscan) C="ffn mha"; A="A3hp25 A3hp75"; R="A1 A3hp25 A3h A3hp75" ;;
  wall)  C="ffn mha"; A="A0w A1w"; R="A0 A1 A0w A1w A3h" ;;
  moe)   C="moe"; A="A0 A1 A2 A3h"; R=$A ;;
  *)     echo "unknown part $1"; exit 2 ;;
esac
python cpcm_sa.py run --configs $C --arms $A --jobs 14 > "out/s2/run_$1.out" 2>&1
tail -n 2 "out/s2/run_$1.out"
python cpcm_sa.py report2 --configs $C --arms $R > "out/s2/report_$1.out" 2>&1
cp out/s2/report2.json "out/s2/report_$1.json"
cp out/s2/report2.csv "out/s2/report_$1.csv"
