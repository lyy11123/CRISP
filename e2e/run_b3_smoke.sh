#!/bin/bash
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 E2E_B3=1 E2E_SUB=_b3 CPCM_SA_STAGE=s2 CPCM_SA_STEPS=30
python e2e_pilot.py run --ops ln proj dln dproj --first --arms A1 A3h --jobs 4 --tag _b3 > out/run_b3_smoke.out 2>&1
tail -20 out/run_b3_smoke.out
