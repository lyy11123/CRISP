"""E4 diagnosis item 3: a1 512+8192, F4, Delta 0, 64 KB chunking of D2D messages only, backend seed 1, default
arbitration and priority order 01; one process each, killed after 20 min (user limit). Writes
e4_out/diag/c64.json (stream measures with the 6-seed unchunked same-platform T_i, chunk stats, wall).

  python e4_diag_c64.py
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import e4_attr
import e4_grid as G
import e4_overlap as E

LIMIT = 1200
with open(os.path.join(E.OUT, "grid1_state.json"), encoding="utf-8") as f:
    fl = json.load(f)["faults"]["a1_2s_512-8192"]["F4"]
fl["failed_links"] = [tuple(map(tuple, l)) for l in fl["failed_links"]]
keys = [G.fault_key(k, fl) for k in ["mha512", "mha8192"]]
faults = {"failed_nodes": [], "failed_links": fl["failed_links"]}
with open(os.path.join(E.OUT, "T6.json"), encoding="utf-8") as f:
    T6 = json.load(f)["T"]
T = {i: T6[k]["T6"] for i, k in enumerate(keys)}
chunk = {"size": 64 * 1024, "only_d2d": True}
vars_ = [{"name": "default", "release": {0: 0, 1: 0}, "chunk": chunk},
         {"name": "prio_01", "release": {0: 0, 1: 0}, "prio": {0: 0, 1: 1}, "chunk": chunk}]


def one(v):
    r = E.simulate(f"diag_c64_a1_512-8192_F4_d0_{v['name']}", keys, [v], [1], timeout=LIMIT, cfg=E.CFG, faults=faults)
    if r.get("timeout"):
        return {"variant": v["name"], "timeout": LIMIT, "process_wall": r["process_wall"]}
    run = r["runs"][0]
    m = e4_attr.stream_measures(run["cap"], T, E.KS)
    return {"variant": v["name"], "eval_wall": run["wall"], "process_wall": r["process_wall"],
            "n_events": len(run["cap"]["events"]), "chunk_stats": run["chunk_stats"], "result": run["cap"]["result"],
            "latency": m["latency"], "stretch": m["stretch"], "S_max": m["S_max"], "makespan": m["makespan"]}


with ThreadPoolExecutor(2) as ex:
    out = list(ex.map(one, vars_))
os.makedirs(os.path.join(E.OUT, "diag"), exist_ok=True)
E._write_json(os.path.join("diag", "c64.json"), {"keys": keys, "T6": T, "chunk": chunk, "seed": 1, "runs": out,
                                                 "artifact_commit": E._artifact_commit()})
print("[e4] " + json.dumps(out, default=str)[:2000], flush=True)
