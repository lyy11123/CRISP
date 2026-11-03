"""E4b step 1, self-check 3 follow-up: the fixed short-first order on a1 512+8192, no fault, Delta 0, re-run in
separate processes (identical spec) to measure the evaluator's cross-process spread (object-address dependence,
CLAUDE.md). Writes e4b_out/repeat.json.

  python e4b_repeat.py
"""
import json

import e4_overlap as E
import e4b as B

base = {"cfg": E.CFG, "faults": None, "seeds": B.SEEDS, "streams": ["mha512", "mha8192"], "chunk": None,
        "release": {0: 0, 1: 0}}
items = [(f"a1_d0_fixed_01_rep{j}", dict(base, policy={"name": "fixed", "rank": {0: 0, 1: 1}})) for j in (1, 2, 3)]
res = B.run_all(3, items)
out = {n: {i: [r["C"][i] for r in res[n]["runs"]] for i in (0, 1)} for n, _ in items}
B.write_json("repeat.json", out)
print("[e4b] " + json.dumps(out), flush=True)
