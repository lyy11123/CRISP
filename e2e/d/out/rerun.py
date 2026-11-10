import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import d_joint as d
plan = {s["name"]: s for s in json.load(open(os.path.join(d.OUT, "d0", "sim_plan.json"), encoding="utf-8"))}
names = [n for n in plan if n.startswith(("chain_", "op_", "pf_proj_")) or n == "pf_ln_a3h_none"]
items = [{"name": n, "cmd": "sim", "spec": plan[n], "folder": os.path.join(d.OUT, "d0", "sims")} for n in names]
print(d.run_all(items, 8))
