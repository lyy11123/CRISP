"""E4 step 3: single-evaluation timing probes for the step-4 time estimate (seed 1, Delta = 0, no fault).

  python e4_timing_probe.py
Writes e4_out/step3_timing.json.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import e4_overlap as E

PROBES = [
    ("a1_2s_2048-8192_d0_256K", ["mha2048", "mha8192"], {"size": 256 * 1024}),
    ("a1_2s_2048-8192_d0_256K_d2d", ["mha2048", "mha8192"], {"size": 256 * 1024, "only_d2d": True}),
    ("a1_3s_512-2048-8192_d0", ["mha512", "mha2048", "mha8192"], None),
    ("a2_3s_mha-ffn-mha_d0", ["mha2048", "ffn", "mha2048"], None),
]


def run(p):
    name, streams, ch = p
    var = {"name": "default", "release": {i: 0 for i in range(len(streams))}}
    if ch:
        var["chunk"] = ch
    r = E.simulate(f"timing_{name}", streams, [var], [1], timeout=3000)
    if not r["runs"]:
        return name, {"timeout": True}
    x = r["runs"][0]
    return name, {"sim_wall": x["wall"], "n_events": len(x["cap"]["events"]), "a": x["cap"]["result"][0],
                  "chunk_stats": x["chunk_stats"]}


def main():
    t0 = time.time()
    with ThreadPoolExecutor(len(PROBES)) as ex:
        out = dict(ex.map(run, PROBES))
    for k, v in out.items():
        print(f"[e4] timing {k}: {v.get('sim_wall')} s, events {v.get('n_events')}", flush=True)
    E._write_json("step3_timing.json", {"probes": out, "parallel_processes": len(PROBES), "wall": time.time() - t0,
                                        "command": sys.argv})


if __name__ == "__main__":
    main()
