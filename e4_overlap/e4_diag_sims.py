"""E4 diagnosis (2026-11-02): simulate the diagnosis configurations that have no cached event graph in
e4_out/sims/ (a1 512+8192 no fault: the step-2 runs kept only runs.csv). Re-runs with the step-2 releases
(release_0 = 0, release_1 = round(Delta * T_min) as recorded in the step-2 summary), seeds 1-3.

  python e4_diag_sims.py
"""
import json
import os

import e4_grid as G
import e4_overlap as E

with open(os.path.join(E.OUT, "a1_2s_512-8192_d0.5_summary.json"), encoding="utf-8") as f:
    rel05 = {int(i): int(round(v)) for i, v in json.load(f)["release_mean"].items()}
keys = ["mha512", "mha8192"]
units = [
    ("a1_2s_512-8192_d0_prio_01", keys, {"name": "prio_01", "release": {0: 0, 1: 0}, "prio": {0: 0, 1: 1}}),
    ("a1_2s_512-8192_d0.5_default", keys, {"name": "default", "release": rel05}),
    ("a1_2s_512-8192_d0.5_prio_01", keys, {"name": "prio_01", "release": rel05, "prio": {0: 0, 1: 1}}),
]
G.run_units([(n, k, v, list(E.SEEDS), E.CFG, None) for n, k, v in units], 3)
G.log(f"releases d0.5: {rel05}")
