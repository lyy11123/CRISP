"""Watch the read-only processes that outlive the first driver: free memory during the run, per-process limit (12 h).

  python e2e_ro_watch.py <pid of prefill_mha_A1> <its start time, seconds since the epoch>
"""
import os
import subprocess
import sys
import time

import e2e_pilot as P

d = os.path.join(P.C.OUT, "readonly")
LIMIT, KILL = 12 * 3600, 3.3
log = os.path.join(d, "driver.log")


def say(m):
    line = f"[watch {time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(line + chr(10))


def alive(pid):
    return str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout


a1, t_a1 = int(sys.argv[1]), float(sys.argv[2])
p = os.path.join(d, "prefill_mha_A3h")
lf = open(p + ".log", "w", encoding="utf-8")
pr = subprocess.Popen([sys.executable, os.path.join(P.HERE, "e2e_pilot.py"), "readonly_child", p + "_spec.json", p + ".json"], cwd=P.HERE, stdout=lf, stderr=subprocess.STDOUT)
t0, mn = time.time(), P.free_gb()
run_a1, run_a3, al_a1, al_a3 = True, True, False, False
say(f"prefill_mha_A3h: start, free {mn:.1f} GB; adopted prefill_mha_A1 pid {a1}")
while run_a1 or run_a3:
    fr = P.free_gb()
    mn = min(mn, fr)
    if run_a3 and fr < KILL:
        pr.kill()
        run_a3 = False
        say(f"KILL prefill_mha_A3h: free {fr:.1f} GB below {KILL} GB")
    if run_a3 and pr.poll() is not None:
        run_a3 = False
        say(f"prefill_mha_A3h: {'done' if os.path.exists(p + '.json') else 'FAILED'} {time.time() - t0:.0f} s, min free {mn:.1f} GB")
    if run_a1 and not alive(a1):
        run_a1 = False
        say(f"prefill_mha_A1: {'done' if os.path.exists(os.path.join(d, 'prefill_mha_A1.json')) else 'FAILED'} {time.time() - t_a1:.0f} s, min free {mn:.1f} GB")
    if run_a1 and not al_a1 and time.time() - t_a1 > LIMIT:
        al_a1 = True
        say("ALERT prefill_mha_A1: over 12 hours, still running")
    if run_a3 and not al_a3 and time.time() - t0 > LIMIT:
        al_a3 = True
        say("ALERT prefill_mha_A3h: over 12 hours, still running")
    if fr < KILL and run_a1 and not run_a3:
        say(f"ALERT free {fr:.1f} GB with only prefill_mha_A1 left")
        time.sleep(600)
    time.sleep(10)
say(f"finished, min free {mn:.2f} GB")
