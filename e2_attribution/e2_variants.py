"""Run the original BusyBarn task scripts (no analysis) for the 14 variants g{0,1} x l{0..6}, as
fig_endtoend/end_to_end.py's find_min_abc does, plus Gemini, and pick the lowest total latency a.

  python e2_variants.py --model intra_mapping_gpt_decode --op ffn --sq 512 --ch 2x2 [--jobs 6]
Writes e2_out/ext/<model>_<op>_sq<sq>_ch<ch>_variants.json. A variant whose results/<name>.txt already
holds an "SA time cost" line is not rerun (pass --force to rerun).
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
ART = os.path.join(HERE, "..", "third_party", "isca2026-busybarn-artifact", "models")
PAT = re.compile(r"SA time cost:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")


def names(op, sq, ch):
    tail = f"ch{ch}_bw256_co4x4_bw256_t128x64_failpattern0"
    v = [f"{op}_sq{sq}_greedy{g}_lossratio{l}_{tail}" for g in (0, 1) for l in range(7)]
    return v, f"{op}_sq{sq}_gemini_{tail}"


def read_result(model, name):
    p = os.path.join(ART, model, "results", f"{name}.txt")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        m = PAT.search(f.read())
    return tuple(map(int, m.groups())) if m else None


def run_one(model, name, force, log_dir):
    if not force and read_result(model, name):
        return name, read_result(model, name), None, "cached"
    script = os.path.join(ART, model, "py", f"{name}.py")
    t0 = time.time()
    with open(os.path.join(log_dir, f"{name}.log"), "w") as lf:
        rc = subprocess.call([sys.executable, script], stdout=lf, stderr=subprocess.STDOUT)
    return name, read_result(model, name), time.time() - t0, f"exit={rc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--sq", type=int, required=True)
    ap.add_argument("--ch", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these task names (timing probes)")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "e2_out", "ext")
    log_dir = os.path.join(out_dir, "variant_logs")
    os.makedirs(log_dir, exist_ok=True)
    variants, gem = names(args.op, args.sq, args.ch)
    todo = args.only or variants + [gem]
    with ThreadPoolExecutor(args.jobs) as ex:
        res = list(ex.map(lambda n: run_one(args.model, n, args.force, log_dir), todo))
    rows = {n: {"a_b_c": abc, "wall_seconds": w, "status": st} for n, abc, w, st in res}
    for n, r in rows.items():
        print(n, r, flush=True)
    if args.only:
        return
    done = {n: rows[n]["a_b_c"] for n in variants if rows[n]["a_b_c"]}
    best = min(done, key=lambda n: done[n][0]) if done else None
    out = {"model": args.model, "op": args.op, "sq": args.sq, "ch": args.ch, "python": sys.executable,
           "variants": rows, "n_variants_with_result": len(done), "best_variant": best,
           "best_a_b_c": done.get(best), "gemini": gem, "gemini_a_b_c": rows[gem]["a_b_c"]}
    path = os.path.join(out_dir, f"{args.model}_{args.op}_sq{args.sq}_ch{args.ch}_variants.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[variants] " + json.dumps({k: out[k] for k in ["model", "op", "sq", "ch", "best_variant", "best_a_b_c",
                                                          "gemini_a_b_c", "n_variants_with_result"]}), flush=True)


if __name__ == "__main__":
    main()
