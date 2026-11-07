#!/bin/bash
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
python e2e_pilot.py readonly --jobs 4 > out/run_readonly.out 2>&1
echo "exit $?" >> out/run_readonly.out
tail -3 out/run_readonly.out
