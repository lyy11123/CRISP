#!/bin/bash
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
python e2e_ro_watch.py 65184 1794050431 > out/run_ro_watch.out 2>&1
tail -2 out/run_ro_watch.out
