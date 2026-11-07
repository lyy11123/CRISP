#!/bin/bash
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
python e2e_diag.py ties84 8 > out/run_ties.out 2>&1
tail -50 out/run_ties.out
