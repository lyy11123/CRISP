#!/bin/bash
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 E2E_RO_SKIP_BACKEND=1
for n in prefill_mha_A1 prefill_mha_A3h; do
  python e2e_pilot.py readonly_child out/readonly_nobk/$n.spec out/readonly_nobk/$n.json > out/readonly_nobk/$n.log 2>&1 &
done
wait
unset E2E_RO_SKIP_BACKEND
for n in decode_mha_A1 decode_mha_A3h; do
  python e2e_pilot.py readonly_child out/readonly_nobk/$n.spec out/readonly_nobk/$n.json > out/readonly_nobk/$n.log 2>&1 &
done
wait
ls out/readonly_nobk
