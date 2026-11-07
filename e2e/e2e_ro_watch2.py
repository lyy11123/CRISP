"""Second watcher: e2e_ro_watch.py died on a tasklist decoding error (stdout None) after prefill_mha_A1 ended. Adopts the running
prefill_mha_A3h: free memory during the run (kill below 3.3 GB), 12 h limit, done or failed.

  python e2e_ro_watch2.py <pid of prefill_mha_A3h> <its start time, seconds since the epoch>
"""
import os
import subprocess
import sys
import time

import e2e_pilot as P

d = os.path.join(P.C.OUT, "readonly")
LIMIT, KILL = 12 * 3600, 3.3
log = os.path.join(d, "driver.log")
pid, t0 = int(sys.argv[1]), float(sys.argv[2])


def say(m):
    line = f"[watch2 {time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(line + chr(10))


def alive():
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, encoding="utf-8", errors="replace").stdout or ""
    return str(pid) in out


mn, alerted = P.free_gb(), False
say(f"adopted prefill_mha_A3h pid {pid}, free {mn:.1f} GB")
while alive():
    fr = P.free_gb()
    mn = min(mn, fr)
    if fr < KILL:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        say(f"KILL prefill_mha_A3h: free {fr:.1f} GB below {KILL} GB")
        break
    if not alerted and time.time() - t0 > LIMIT:
        alerted = True
        say("ALERT prefill_mha_A3h: over 12 hours, still running")
    time.sleep(10)
ok = os.path.exists(os.path.join(d, "prefill_mha_A3h.json"))
say(f"prefill_mha_A3h: {'done' if ok else 'FAILED'} {time.time() - t0:.0f} s, min free {mn:.2f} GB")
