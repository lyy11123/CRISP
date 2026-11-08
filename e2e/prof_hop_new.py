"""cProfile of fast_hop.event_driver_fast on one read-only mapping (seed 1, tag hash; hot spots only).
  python prof_hop_new.py <readonly spec json> <out txt>
"""
import cProfile
import json
import pstats
import random
import sys
import time

import speedup_check as K

spec = json.load(open(sys.argv[1], encoding="utf-8"))
D, ev, hp = K.prepare(spec)
K.tag_hash()
from e4b_backend import P1  # noqa: E402
from fast_hop import event_driver_fast  # noqa: E402

random.seed(1)
pr = cProfile.Profile()
t0 = time.time()
pr.enable()
tri = event_driver_fast(ev, hp, release={0: 0}, policy=P1(), released={}, audit=[])
pr.disable()
with open(sys.argv[2], "w", encoding="utf-8") as f:
    f.write(f"profiled wall {time.time() - t0:.1f} s, triple {tri}, latency {max(e.end_time for e in ev.values())}\n")
    pstats.Stats(pr, stream=f).sort_stats("tottime").print_stats(15)
