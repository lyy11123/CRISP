"""Diagnostic for a failed self-check 1: cross-process spread of the same evaluation (same state, same backend
seed, another process), for the original graph (template re-run, capture_seeds) and for the merge tool's
1-stream graph (simulate). No change to either pipeline.

  python e4_check1_noise.py --streams mha2048 mha8192 --repeats 4
Writes e4_out/check1_noise.json.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import e4_overlap as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=4)
    args = ap.parse_args()
    out = {}
    jobs = [(s, "orig", r) for s in args.streams for r in range(args.repeats)] + \
           [(s, "merged", r) for s in args.streams for r in range(args.repeats)]

    def run(job):
        s, kind, r = job
        if kind == "orig":
            o = E._run_capture_child(E.STREAMS[s], E.SEEDS, f"noise{r}")
            return job, o["pre_digest"], {sd: c["result"][0] for sd, c, _ in o["reseeded"]}
        o = E.simulate(f"noise_{s}_{r}", [s], [{"name": "default", "release": {0: 0}}])
        return job, None, {x["seed"]: x["cap"]["result"][0] for x in o["runs"]}

    with ThreadPoolExecutor(min(12, len(jobs))) as ex:
        res = list(ex.map(run, jobs))
    for s in args.streams:
        rec = E.load_stream(s)
        first_orig = {sd: c["result"][0] for sd, c in rec["reseeded"]}
        orig = [first_orig] + [v for (ss, k, _), d, v in res if ss == s and k == "orig"]
        digests = {d for (ss, k, _), d, v in res if ss == s and k == "orig"}
        merged = [v for (ss, k, _), d, v in res if ss == s and k == "merged"]
        om = [float(np.mean([o[sd] for sd in E.SEEDS])) for o in orig]
        mm = [float(np.mean([m[sd] for sd in E.SEEDS])) for m in merged]
        out[s] = {"orig_per_process": orig, "merged_per_process": merged,
                  "orig_3seed_mean_per_process": om, "merged_3seed_mean_per_process": mm,
                  "orig_pre_digest_same_as_capture": digests == {rec["pre_digest"]},
                  "orig_mean_spread_rel": (max(om) - min(om)) / float(np.mean(om)),
                  "merged_mean_spread_rel": (max(mm) - min(mm)) / float(np.mean(mm)),
                  "pooled_orig_mean": float(np.mean(om)), "pooled_merged_mean": float(np.mean(mm)),
                  "pooled_rel_diff": abs(float(np.mean(mm)) - float(np.mean(om))) / float(np.mean(om))}
        print(f"[e4] noise {s}: " + json.dumps({k: v for k, v in out[s].items() if "per_process" not in k or "mean" in k}),
              flush=True)
    out["command"] = sys.argv
    with open(os.path.join(E.OUT, "check1_noise.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
