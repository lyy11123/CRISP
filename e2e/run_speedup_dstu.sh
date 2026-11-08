#!/bin/bash
# Speed-up regression on the 18 DSTU objects (speedup_dstu.py), 4 at a time.
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
mkdir -p out/speedup/dstu
seq 0 17 | xargs -P 4 -I{} sh -c 'python speedup_dstu.py one {} out/speedup/dstu/item{}.json > out/speedup/dstu/item{}.log 2>&1'
python - <<'PY'
import json, glob
for p in sorted(glob.glob("out/speedup/dstu/item*.json"), key=lambda s: int(s.split("item")[1].split(".")[0])):
    r = json.load(open(p)); print(r["index"], r["name"], r["identical"], r["equals_cached"], round(r["old_wall_s"], 1), round(r["new_wall_s"], 1), r["diff"])
PY
