#!/bin/bash
# B4 probe: one W1 prefill MHA run for its single-run wall clock, plus the A1 trajectory check (W1 vs W0) on prefill and decode.
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 E2E_B4=1 E2E_SUB=b4 CPCM_SA_STAGE=s2
python - <<'PY' > out/run_b4_probe.out 2>&1
import time, threading
from concurrent.futures import ThreadPoolExecutor
import e2e_pilot as P
todo = [("pf_mha_w1", "A3h", 0, 123, True), ("pf_mha_w1", "A1", 0, 0, True), ("dc_mha_w1", "A1", 0, 0, True), ("dc_mha_w1", "A3h", 0, 123, True)]
mins = [P.free_gb()]
def watch():
    while True:
        mins.append(P.free_gb()); time.sleep(5)
threading.Thread(target=watch, daemon=True).start()
with ThreadPoolExecutor(4) as ex:
    print(list(ex.map(lambda t: P.run_one(*t), todo)), flush=True)
print("min free %.2f GB" % min(mins), flush=True)
PY
tail -5 out/run_b4_probe.out
