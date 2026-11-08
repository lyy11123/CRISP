"""cProfile of the bb-mode T-hat used in the prefill MHA search (E2E_HANDOFF.md section 3; hot spots only, no optimisation).
One saved prefill MHA state with the template's final broadcast (seed 0), as B0 evaluates it: compile_bb, then simulate(bb).
  python prof_bb.py <cfg> <run name> <out json>
"""
import cProfile
import json
import pstats
import sys
import time

import e2e_b as EB
import e2e_pilot as P

C = P.C
cfg, name, out = sys.argv[1:4]
run, snap = EB.paths(cfg, name)
ctx, beha, data, events, hp = EB.load_state(dict(run=run, snap=snap, cfg=cfg))
EB.broadcast(ctx, beha, data, events, hp)
res = dict(cfg=cfg, name=name)
t0 = time.perf_counter()
g = EB.compiled(events, hp)
res["compile_s"] = time.perf_counter() - t0
t0 = time.perf_counter()
T = C.cpcm.simulate(g, "bb").T
res["simulate_s"] = time.perf_counter() - t0
res.update(T=T, n_events=g.n)
pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
g2 = EB.compiled(events, hp)
T2 = C.cpcm.simulate(g2, "bb").T
pr.disable()
res["profiled_s"] = time.perf_counter() - t0
assert T2 == T
st = pstats.Stats(pr)
tot = sum(v[2] for v in st.stats.values())
rows = sorted(st.stats.items(), key=lambda kv: -kv[1][2])[:15]
res["total_s"] = tot
res["by_tottime"] = [dict(func=f"{k[0].split(chr(92))[-1]}:{k[1]}({k[2]})", calls=v[1], tottime_s=v[2], share=v[2] / tot, cum_s=v[3]) for k, v in rows]
json.dump(res, open(out, "w"), indent=1)
for r in res["by_tottime"]:
    print(f"{r['share']*100:5.1f}%  {r['tottime_s']:6.2f}s  {r['calls']:>10}  {r['func']}")
print({k: v for k, v in res.items() if k != "by_tottime"})
