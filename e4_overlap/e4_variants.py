"""E4 step 1: check that the winners used by E4 are the lowest-latency of the 14 variants g{0,1} x l{0..6}
(fig_endtoend's find_min_abc rule), on the E4 work copy. A variant whose results/<name>.txt already holds an
"SA time cost" line is read, not rerun (those files came with the copy of third_party, written by E2's sweeps).

  python e4_variants.py --jobs 12
Writes e4_out/variants.json.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "work", "artifact", "models", "intra_mapping_gpt_decode")
PAT = re.compile(r"SA time cost:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
TAIL = "ch2x2_bw256_co4x4_bw256_t128x64_failpattern0"
CLAIMED = {("mha", 512): "greedy0_lossratio0", ("mha", 2048): "greedy0_lossratio4",
           ("mha", 8192): "greedy0_lossratio5", ("ffn", 2048): "greedy0_lossratio6"}


def read_result(name):
    p = os.path.join(MODEL, "results", f"{name}.txt")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        m = PAT.search(f.read())
    return tuple(map(int, m.groups())) if m else None


def run_one(name, log_dir):
    if read_result(name):
        return name, read_result(name), None, "cached"
    t0 = time.time()
    with open(os.path.join(log_dir, f"{name}.log"), "w") as lf:
        rc = subprocess.call([sys.executable, os.path.join(MODEL, "py", f"{name}.py")], stdout=lf,
                             stderr=subprocess.STDOUT)
    return name, read_result(name), time.time() - t0, f"exit={rc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--ch", default="2x2", help="die group; the 1x4 spot check (step 4) uses 1x4")
    ap.add_argument("--keys", nargs="*", help="op:sq pairs to sweep (default: the 2x2 winners claimed by the handoff)")
    ap.add_argument("--out-name", default="variants.json")
    args = ap.parse_args()
    tail = TAIL.replace("ch2x2", f"ch{args.ch}")
    keys = [(k.split(":")[0], int(k.split(":")[1])) for k in args.keys] if args.keys else list(CLAIMED)
    claimed = CLAIMED if not args.keys else {}
    out = os.path.join(HERE, "e4_out")
    log_dir = os.path.join(out, "variant_logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(MODEL, "results"), exist_ok=True)
    jobs = {key: [f"{key[0]}_sq{key[1]}_greedy{g}_lossratio{l}_{tail}" for g in (0, 1) for l in range(7)]
            for key in keys}
    allnames = [n for v in jobs.values() for n in v]
    with ThreadPoolExecutor(args.jobs) as ex:
        res = {n: (r, w, st) for n, r, w, st in ex.map(lambda n: run_one(n, log_dir), allnames)}
    summary = {}
    for (op, sq), names in jobs.items():
        got = {n: res[n][0] for n in names}
        ok = {n: v for n, v in got.items() if v}
        best = min(ok, key=lambda n: ok[n][0]) if ok else None
        cl = f"{op}_sq{sq}_{claimed[(op, sq)]}_{tail}" if (op, sq) in claimed else None
        summary[f"{op}_sq{sq}"] = {
            "n_with_result": len(ok), "best": best, "best_abc": ok.get(best), "claimed": cl,
            "claimed_abc": ok.get(cl), "claimed_is_best": best == cl,
            "variants": {n: {"abc": got[n], "wall": res[n][1], "status": res[n][2]} for n in names}}
        print(f"[e4] {op} sq{sq}: {len(ok)}/14, best {best} {ok.get(best)}, claimed ok: {best == cl}",
              flush=True)
    summary["python"] = sys.executable
    summary["command"] = sys.argv
    with open(os.path.join(out, args.out_name), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
