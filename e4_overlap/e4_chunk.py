"""E4 step 3: chunking probe on one FFN configuration (a2, 2 streams: MHA 2048 + FFN, Delta = 0, 2x2, no fault),
backend seed 1, one process per chunking variant (in parallel), and self-check 4.

  python e4_chunk.py --timeout 1800
Writes e4_out/step3_chunk.json.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import e4_overlap as E

KB = 1024
VARIANTS = [("none", None), ("huge", {"size": 10 ** 12}), ("256K_d2d", {"size": 256 * KB, "only_d2d": True}),
            ("256K", {"size": 256 * KB}), ("64K_d2d", {"size": 64 * KB, "only_d2d": True}), ("64K", {"size": 64 * KB})]
STREAMS = ["mha2048", "ffn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(n, ch) for n, ch in VARIANTS]

    def run(job):
        n, ch = job
        var = {"name": n, "release": {0: 0, 1: 0}}
        if ch:
            var["chunk"] = ch
        return n, E.simulate(f"chunk_a2_mha2048-ffn_d0_{n}", STREAMS, [var], [args.seed], timeout=args.timeout)

    def single(s):
        return s, E.simulate(f"chunkprobe_single_{s}", [s], [{"name": "default", "release": {0: 0}}])

    with ThreadPoolExecutor(len(jobs) + 1) as ex:
        fs = [ex.submit(run, j) for j in jobs]
        fsingle = ex.submit(single, "ffn")
        res = dict(f.result() for f in fs)
        s_ffn = fsingle.result()[1]
    import e4_attr
    T_ffn = E._mean([r["cap"]["result"][0] for r in s_ffn["runs"]])
    with open(os.path.join(E.OUT, "step2_checks.json"), encoding="utf-8") as f:
        T = {0: json.load(f)["T"]["mha2048"], 1: T_ffn}
    out = {"streams": STREAMS, "seed": args.seed, "T": T, "T_ffn_seeds": [r["cap"]["result"][0] for r in s_ffn["runs"]],
           "variants": {}, "timeout": args.timeout}
    base = res["none"]["runs"][0]["cap"] if res["none"]["runs"] else None
    for n, r in res.items():
        if not r["runs"]:
            out["variants"][n] = {"timeout": True, "process_wall": r["process_wall"]}
            continue
        run = r["runs"][0]
        cap, st = run["cap"], run["chunk_stats"]
        m = e4_attr.stream_measures(cap, T, E.KS)
        n_comm = sum(1 for e in cap["events"].values() if e["type"] == "communication")
        out["variants"][n] = {"sim_wall": run["wall"], "process_wall": r["process_wall"], "n_events": len(cap["events"]),
                              "n_comm": n_comm, "latency_a": cap["result"][0], "S_max": m["S_max"],
                              "stretch": m["stretch"], "makespan": m["makespan"], "chunk_stats": st}
        print(f"[e4] chunk {n}: events {len(cap['events'])}, sim {run['wall']:.1f} s, a {cap['result'][0]}, "
              f"S_max {m['S_max']:.4f}", flush=True)
    # self-check 4
    c4 = {}
    for n, v in out["variants"].items():
        st = v.get("chunk_stats")
        if st:
            c4[n] = {"bytes_conserved": st["bytes_conserved"], "tree_conserved": st["tree_conserved"]}
    huge = res["huge"]["runs"][0]["cap"] if res["huge"]["runs"] else None
    gd = E._e3().graph_diff(huge, base) if (huge and base) else None
    c4["huge_graph_diff_vs_none"] = gd
    c4["huge_messages_chunked"] = out["variants"].get("huge", {}).get("chunk_stats", {}).get("messages_chunked")
    c4["huge_same_latency_seed"] = (huge["result"] == base["result"]) if (huge and base) else None
    c4["pass"] = all(x["bytes_conserved"] and x["tree_conserved"] for k, x in c4.items() if isinstance(x, dict)) \
        and gd == 0 and c4["huge_messages_chunked"] == 0
    out["check4"] = c4
    out["wall"] = time.time() - t0
    out["command"] = sys.argv
    out["artifact_commit"] = E._artifact_commit()
    E._write_json("step3_chunk.json", out)
    print("[e4] " + json.dumps({"check4": c4, "walls": {n: v.get("sim_wall") for n, v in out["variants"].items()},
                                "events": {n: v.get("n_events") for n, v in out["variants"].items()}}), flush=True)


if __name__ == "__main__":
    main()
