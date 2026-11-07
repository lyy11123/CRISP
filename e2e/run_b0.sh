#!/bin/bash
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
python e2e_b.py b0 --jobs 8 > out/run_b0.out 2>&1
tail -3 out/run_b0.out
