"""E3 step 4: post-fault single-resource analysis (E2's pipeline on E3's saved event graphs).

For every task x fault, the R0 mapping and the best warm-start mapping (smallest 3-seed mean among the W
runs) are analysed with e2_attribution.analyse on each backend seed's final event graph (no new
simulation). The pre-fault reference is the same analysis on the pre-fault graphs captured in E3 step 1
(work copy), so the numbers are directly comparable with E2's.

Per graph:
  reorder  = best single-resource exact reorder gain / a       (E2 reorder_gain_best_resource_over_a)
  delete   = max over the 5 resources with the most exclusive critical wait of
             (a - longest path that avoids the resource) / a    (E2's "delete the resource" bound; over
             E2's 26 tasks this reproduces median 1.85%, max 13.4%)
  top      = resource with the most exclusive critical wait, its busy time / a, its share of the wait

Usage:  python e3_resource.py [--jobs 12]
"""
import argparse
import collections
import csv
import gzip
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e3_warmstart as e3  # noqa: E402

CASES = [("ffn", f) for f in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]] + \
        [("mha", f) for f in ["F1", "F2", "F3", "F4", "F6", "F7", "F8"]] + \
        [(t, f) for t in ["ffn14", "mha14", "ffng", "mhag"] for f in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]]


def metrics(cap):
    """E2 single-resource numbers of one final event graph."""
    e2 = e3._e2()
    try:
        s = e2.analyse(cap)[0]
    except e2.E2Error as exc:
        return {"error": str(exc)}
    a = s["latency_a"]
    top5 = s["reorder_top5_by_wait"]
    dele = [(a - r["longest_path_avoiding_resource"]) / a for r in top5]
    i = int(np.argmax(dele)) if dele else None
    return {"a": a, "reorder": s["reorder_gain_best_resource_over_a"],
            "reorder_resource": s["reorder_gain_best_resource"],
            "reorder_ub": s["reorder_gain_ub_best_resource_over_a"],
            "delete": max(dele) if dele else 0.0, "delete_resource": top5[i]["resource"] if dele else None,
            "top_resource": top5[0]["resource"] if top5 else None,
            "top_busy": top5[0]["busy_over_a"] if top5 else None,
            "top_wait_share": s["wait_top1_resource_share"],
            "critical_wait_over_a": s["critical_wait_over_a"],
            "dstu_ub_local": s["dstu_gain_upper_bound_local_over_a"],
            "top5": [{k: r[k] for k in ["resource", "wait_excl", "busy_over_a", "reorder_gain_over_a",
                                        "longest_path_avoiding_resource"]} for r in top5]}


def _job(item):
    key, cap = item
    return key, metrics(cap)


def _agg(rows):
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"errors": [r["error"] for r in rows]}
    tops = collections.Counter(r["top_resource"] for r in ok)
    return {"n": len(ok), "errors": [r["error"] for r in rows if "error" in r],
            "reorder_mean": float(np.mean([r["reorder"] for r in ok])), "reorder_max": max(r["reorder"] for r in ok),
            "delete_mean": float(np.mean([r["delete"] for r in ok])), "delete_max": max(r["delete"] for r in ok),
            "top_resource": tops.most_common(1)[0][0], "top_resource_votes": dict(tops),
            "top_busy_mean": float(np.mean([r["top_busy"] for r in ok])),
            "top_wait_share_mean": float(np.mean([r["top_wait_share"] for r in ok])),
            "dstu_ub_local_mean": float(np.mean([r["dstu_ub_local"] for r in ok])),
            "per_seed": ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()

    items, meta = [], {}
    # only cases whose summary exists (a fault the mapper cannot run on leaves none)
    cases = [(s, f) for s, f in CASES
             if os.path.exists(os.path.join(e3.OUT, f"{e3.TASKS[s]}_{f}_summary.json"))]
    for short in dict.fromkeys(s for s, _ in cases):
        task = e3.TASKS[short]
        with gzip.open(os.path.join(e3.OUT, f"{task}_prefault.pkl.gz"), "rb") as f:
            pre = pickle.load(f)
        for s, cap in pre["reseeded"]:
            items.append(((short, "pre", "pre", s), cap))
    for short, fid in cases:
        task = e3.TASKS[short]
        with open(os.path.join(e3.OUT, f"{task}_{fid}_summary.json"), encoding="utf-8") as f:
            summ = json.load(f)
        best_w = summ["best_W"]
        with gzip.open(os.path.join(e3.OUT, f"{task}_{fid}_caps.pkl.gz"), "rb") as f:
            caps = pickle.load(f)["runs"]
        meta[(short, fid)] = {"best_W": best_w, "summary": summ}
        for which, run in [("R0", f"{fid}_R0"), ("bestW", best_w)]:
            for s, cap in caps[run]["caps"]:
                if s != "template":
                    items.append(((short, fid, which, s), cap))
    print(f"[e3] step 4: {len(items)} event graphs", flush=True)
    with ProcessPoolExecutor(args.jobs) as ex:
        res = dict(ex.map(_job, items))

    by = collections.defaultdict(list)
    for (short, fid, which, s), m in sorted(res.items(), key=lambda x: str(x[0])):
        by[(short, fid, which)].append({"seed": s, **m})
    agg = {k: _agg(v) for k, v in by.items()}

    rows = []
    for short, fid in cases:
        pre, r0, bw = agg[(short, "pre", "pre")], agg[(short, fid, "R0")], agg[(short, fid, "bestW")]
        summ = meta[(short, fid)]["summary"]
        row = {"task": short, "fault": fid, "failed": summ["table_row"]["failed"],
               "fault_cost": summ["table_row"]["fault_cost"], "best_W": meta[(short, fid)]["best_W"].split("_W_")[1],
               "pre_reorder": pre["reorder_mean"], "pre_delete": pre["delete_mean"],
               "pre_top": pre["top_resource"], "pre_top_busy": pre["top_busy_mean"]}
        for name, g in [("R0", r0), ("W", bw)]:
            row.update({f"{name}_reorder": g.get("reorder_mean"), f"{name}_reorder_max": g.get("reorder_max"),
                        f"{name}_delete": g.get("delete_mean"), f"{name}_delete_max": g.get("delete_max"),
                        f"{name}_top": g.get("top_resource"), f"{name}_top_busy": g.get("top_busy_mean"),
                        f"{name}_top_wait_share": g.get("top_wait_share_mean"),
                        f"{name}_errors": len(g.get("errors", []))})
        rows.append(row)
        task = e3.TASKS[short]
        out = {"task": task, "fault": summ["fault"], "best_W": meta[(short, fid)]["best_W"],
               "prefault": pre, "R0": r0, "best_W_mapping": bw,
               "definitions": __doc__.split("Usage")[0].strip(),
               "e2_reference_same_task_third_party": {"ffn": {"reorder": 0.0327, "delete": 0.0359},
                                                      "mha": {"reorder": 0.0, "delete": 0.032}}.get(short),
               "e2_reference_all_26_tasks": {"delete_median": 0.0185, "delete_max": 0.134, "reorder_max": 0.033}}
        with open(os.path.join(e3.OUT, f"{task}_{fid}_resource.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        # also attach to the case summary
        summ["single_resource"] = {k: out[k] for k in ["prefault", "R0", "best_W_mapping"]}
        with open(os.path.join(e3.OUT, f"{task}_{fid}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2, default=str)

    with open(os.path.join(e3.OUT, "e3_resource_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # extend e3_table.csv with the step-4 columns
    tpath = os.path.join(e3.OUT, "e3_table.csv")
    with open(tpath, encoding="utf-8") as f:
        trows = list(csv.DictReader(f))
    extra = {(r["task"], r["fault"]): r for r in rows}
    cols = ["R0_reorder", "R0_delete", "R0_top", "R0_top_busy", "W_reorder", "W_delete", "W_top", "W_top_busy",
            "pre_reorder", "pre_delete"]
    for t in trows:
        x = extra.get((t["task"], t["fault"]), {})
        for c in cols:
            t[c] = x.get(c)
    with open(tpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in trows for k in r)))
        w.writeheader()
        w.writerows(trows)

    allv = lambda k: [r[k] for r in rows if r[k] is not None]
    line = {f"{r['task']}:{r['fault']}": {k: (round(r[k], 4) if isinstance(r[k], float) else r[k])
                                          for k in ["fault_cost", "R0_reorder", "R0_delete", "W_reorder", "W_delete",
                                                    "R0_top", "R0_top_busy"]} for r in rows}
    line["_prefault"] = {s: {"reorder": round(agg[(s, "pre", "pre")]["reorder_mean"], 4),
                             "delete": round(agg[(s, "pre", "pre")]["delete_mean"], 4)}
                         for s in dict.fromkeys(s for s, _ in cases)}
    line["_summary"] = {"R0_delete_median": round(float(np.median(allv("R0_delete"))), 4),
                        "R0_delete_max": round(max(allv("R0_delete")), 4),
                        "R0_delete_le_5pct": f"{sum(v <= 0.05 for v in allv('R0_delete'))}/{len(allv('R0_delete'))}",
                        "W_delete_median": round(float(np.median(allv("W_delete"))), 4),
                        "W_delete_le_5pct": f"{sum(v <= 0.05 for v in allv('W_delete'))}/{len(allv('W_delete'))}",
                        "R0_reorder_max": round(max(allv("R0_reorder")), 4),
                        "errors": sum(r["R0_errors"] + r["W_errors"] for r in rows)}
    print("[e3] " + json.dumps(line, default=str), flush=True)


if __name__ == "__main__":
    main()
