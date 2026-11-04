"""Stage 2 self-check 1: A0's 10-seed mean vs the template's own result. Points over 4% are judged by V2's rule
(PROGRESS step 9): 10 values split at the largest gap, gap >= 2% of the mean and >= 3 x the larger cluster spread, and
the template value inside [min, max] or within 3% of a boundary.
Usage: CPCM_SA_STAGE=s2 python check1_s2.py <configs>"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpcm_sa as C  # noqa: E402

rows = []
for cfg in sys.argv[1:]:
    gaps = []
    for g, x in C.variants("A0"):
        d = C.load_run(cfg, "A0", g, x)
        v = sorted(b[0] for b in d["selected"]["backend"].values())
        m, t = statistics.mean(v), d["template_out"][0]
        rel = (m - t) / t
        gaps.append(abs(rel))
        if abs(rel) <= 0.04:
            continue
        i = max(range(len(v) - 1), key=lambda k: v[k + 1] - v[k])
        gap, spread = v[i + 1] - v[i], max(v[i] - v[0], v[-1] - v[i + 1])
        bimodal = gap >= 0.02 * m and gap >= 3 * spread
        near = v[0] * 0.97 <= t <= v[-1] * 1.03
        rows.append(dict(cfg=cfg, g=g, l=x, mean=round(m, 1), min=v[0], max=v[-1], template=t, rel=round(rel, 4),
                         gap=gap, spread=spread, bimodal=bimodal, template_in_range=v[0] <= t <= v[-1],
                         template_near=near, exempt=bimodal and near))
        print(rows[-1])
    print(cfg, "median", round(statistics.median(gaps), 4), "max", round(max(gaps), 4), "over 4%", sum(e > 0.04 for e in gaps))
print("not exempt:", [r for r in rows if not r["exempt"]])
with open(os.path.join(C.OUT, "check1_" + "_".join(sys.argv[1:3]) + ".json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=1)
