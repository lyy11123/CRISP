import json, os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import d_joint as d
base = os.path.join(d.OUT, "sb_smoke")
plan = {s["name"]: s for s in json.load(open(os.path.join(base, "plan.json"), encoding="utf-8"))}
src = plan["sb_mha_PF4_P"]
items = []
for kv in (True, False):
    sp = copy.deepcopy(src)
    nm = "diag_PF4_P_" + ("kv" if kv else "nokv")
    sp["name"], sp["sink"] = nm, os.path.join(base, "sims", nm + ".pkl.gz")
    sp["groups"]["P"]["kv"] = kv
    runs = [{"kind": "solo", "stream": 0, "backend": "hop", "seed": s, "policy": {"name": "P1"}, "record": True, "label": f"soloP_s{s}"} for s in (1, 2, 3)]
    runs += [{"kind": "solo", "stream": 1, "backend": "hop", "seed": 1, "policy": {"name": "P1"}, "label": "soloD_s1"}]
    # same tau in both variants: the KV window of the kv variant's seed-1 solo run (1,029,753 - 5,383,759), g = 0.4
    tau = int(round(1029753 + 0.4 * (5383759 - 1029753)))
    runs += [{"kind": "ov", "backend": "hop", "seed": s, "policy": {"name": "C4"}, "tau": tau, "record": True, "by_op": True, "label": f"ov_s{s}"} for s in (1, 2, 3)]
    sp["runs"] = runs
    items.append({"name": nm, "cmd": "simsb", "spec": sp, "folder": os.path.join(base, "sims")})
print(d.run_all(items, 2))
