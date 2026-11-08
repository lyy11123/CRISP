#!/bin/bash
# B4 batch (PROGRESS.md "B4 第 0 步"): prefill LN A2/A3h first at 5 in parallel, then the rest at 12; cached runs are skipped.
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 E2E_B4=1 E2E_SUB=b4 CPCM_SA_STAGE=s2 E2E_ALERT_S=3600
python e2e_pilot.py run --ops ln --arms A2 A3h --jobs 5 --tag _pf_ln_cpcm > out/run_b4_1.out 2>&1
python e2e_pilot.py run --ops ln proj dln dproj mha_w1 dmha_w1 --jobs 12 --tag _rest > out/run_b4_2.out 2>&1
tail -3 out/run_b4_1.out out/run_b4_2.out
