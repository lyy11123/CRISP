#!/usr/bin/env bash
# Stage 2 driver (handoff 10.3). Usage: bash /e/58Institute/PhD_TERM3/CRISP/cpcm_sa/run_s2.sh <part>
# Parts in order: a0 | arms | scans | moe. Finished runs are skipped (a run's sink exists).
HERE=/e/58Institute/PhD_TERM3/CRISP/cpcm_sa
cd "$HERE" || exit 1
export CPCM_SA_STAGE=s2
mkdir -p out/s2
NONMOE="ffn mha ffn14 opt ffn_f4 ffn_f5 mha_f4 mha_f5"
case "$1" in
  a0)    python cpcm_sa.py run --configs $NONMOE --arms A0 --jobs 14 > out/s2/run_a0.out 2>&1 ;;
  arms)  python cpcm_sa.py run --configs $NONMOE --arms A1 A2 A3 A3CB A3h --jobs 14 > out/s2/run_arms.out 2>&1 ;;
  scans) python cpcm_sa.py run --configs ffn mha --arms A3hp25 A3hp75 A0w A1w --jobs 14 > out/s2/run_scans.out 2>&1 ;;
  moe)   python cpcm_sa.py run --configs moe --arms A0 A1 A2 A3h --jobs 14 > out/s2/run_moe.out 2>&1 ;;
  *)     echo "unknown part $1"; exit 2 ;;
esac
tail -n 2 "out/s2/run_$1.out"
