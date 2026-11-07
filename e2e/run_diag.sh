#!/bin/bash
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
python e2e_diag.py run --jobs 6 > out/run_diag.out 2>&1
echo "exit $?" >> out/run_diag.out
tail -3 out/run_diag.out
