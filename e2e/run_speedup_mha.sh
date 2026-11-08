#!/bin/bash
# Prefill MHA A1/A3h big mappings with the new code only (C4 + hop seed 1), compared afterwards with out/readonly (E2E_HANDOFF.md section 3).
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
F=out/speedup/final
for a in A1 A3h; do python speedup_check.py pilot out/readonly/prefill_mha_${a}_spec.json $F/prefill_mha_$a.json --seeds 1 --parts c4 hop --new-only > $F/prefill_mha_$a.log 2>&1 & done
wait; cat $F/prefill_mha_*.log
