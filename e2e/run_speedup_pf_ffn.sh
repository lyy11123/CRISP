#!/bin/bash
# Speed-up regression on the two prefill FFN read-only mappings: C4 T-hat and hop backend seed 1, original vs new (bit for bit).
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
for m in prefill_ffn_A1 prefill_ffn_A3h; do
  python speedup_check.py pilot out/readonly/${m}_spec.json out/speedup/$m.json --seeds 1 --parts c4 hop > out/speedup/$m.log 2>&1 &
done
wait
tail -3 out/speedup/prefill_ffn_*.log
