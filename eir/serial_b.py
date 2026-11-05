"""Phase B's serial timing (EIR_NEXT_HANDOFF.md sections 6, 8). Machine idle, one child process at a time. Magnitude only, not
paper data. FFN, MHA at F4a: l1 twice (R0, L1), L2 s = 100 of each method once, one 1,000-step FULL-A3h (first variant).
Isolated folder out/next/b_serial; stops at once if any other python process is running.

  python serial_b.py
"""
import json
import os
import subprocess
import time

import eir
import eir_next as en

OUT = os.path.join(en.OUT, "b_serial")
SEED = 123


def others():
    """Other python processes (image name, pid), this one and its launcher excluded."""
    txt = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
    me = {os.getpid(), os.getppid()}
    rows = []
    for line in txt.splitlines():
        p = [x.strip('"') for x in line.split('","')]
        if len(p) > 1 and p[0].lower().startswith("python") and int(p[1]) not in me:
            rows.append((p[0], int(p[1])))
    return rows


def plan():
    seq = []
    for o in en.OPS:
        for i in (1, 2):
            j = en.job_l1(o, en.B_FAULT, out=OUT)
            seq.append(("l1", o, dict(j, name=f"{j['name']}_r{i}", sink=j["sink"].replace(".pkl.gz", f"_r{i}.pkl.gz"))))
        for m in en.B_METHODS:
            seq.append((m, o, en.job_rep(o, en.B_FAULT, m, SEED, out=OUT, steps=100)))
        g, x = eir.cs.variants("A3h")[0]
        seq.append(("full", o, eir.job_full(OUT, o, en.B_FAULT, "A3h", g, x)))
    return seq


def main():
    os.makedirs(OUT, exist_ok=True)
    busy = others()
    if busy:
        raise SystemExit(f"not idle, other python processes: {busy}")
    log = os.path.join(OUT, "serial.jsonl")
    for what, o, j in plan():
        busy = others()
        if busy:     # a process appeared between two measurements: this measurement would not be serial
            raise SystemExit(f"not idle before {j['name']}: {busy}")
        t0 = time.time()
        bad = eir.run_phases([[j]], OUT, 1)
        row = {"what": what, "op": o, "name": j["name"], "sink": j["sink"], "wall_s": time.time() - t0, "bad": list(bad), "free_gb": en.free_gb(),
               "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))}
        print(row, flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + chr(10))
        if bad:
            raise SystemExit(1)
    print("SERIAL DONE", flush=True)


if __name__ == "__main__":
    main()
