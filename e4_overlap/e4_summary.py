"""E4a wrap-up (user decision 2026-11-02, "收尾写汇总"): core-plot data under per-hop arbitration and the final
[e4] summary line. No new simulation.

  python e4_summary.py
Writes e4_out/e4_core_plot_hop.csv: one row per hop-grid configuration (aliases included), x = T_max/T_min of the
per-hop T_i (symmetric controls at 1), y = per-hop stream-priority gain (1 - best/default S_max, hop T_i), with
the original backend's gain, its single-resource S_max bound and makespan bounds (from the original
<config>_summary.json) alongside.
"""
import csv
import json
import os

import e4_overlap as E

OUT = E.OUT


def main():
    with open(os.path.join(OUT, "hop", "hop_grid.csv"), encoding="utf-8") as f:
        hop = list(csv.DictReader(f))
    rows = []
    for r in hop:
        Th = json.loads(r["T_hop"])
        To = json.loads(r["T_orig"])
        src = r.get("alias_of") or r["config"]
        with open(os.path.join(OUT, f"{src}_summary.json"), encoding="utf-8") as f:
            s = json.load(f)
        rows.append({"config": r["config"], "alias_of": r.get("alias_of", ""), "composition": r["composition"],
                     "group": r["group"], "n_streams": len(Th), "hardware": r["hw"], "stagger": float(r["stagger"]),
                     "T_ratio_hop": max(Th.values()) / min(Th.values()),
                     "T_ratio_orig": max(To.values()) / min(To.values()),
                     "hop_default_S_max": float(r["hop_default_S_max"]), "hop_best_order": r["hop_best_order"],
                     "hop_best_S_max": float(r["hop_best_S_max"]), "hop_prio_gain": float(r["hop_gain"]),
                     "hop_prio_gain_origT": float(r["hop_gain_origT"]),
                     "orig_default_S_max": float(r["orig_default_S_max"]), "orig_best_order": r["orig_best_order"],
                     "orig_prio_gain": float(r["orig_gain"]),
                     "orig_smax_bound_reduction": s["smax_bound_reduction_mean"],
                     "orig_smax_bound_reduction_ub": s["smax_bound_reduction_ub_mean"],
                     "orig_makespan_reorder_gain": s["makespan_reorder_gain_mean"],
                     "orig_makespan_delete_gain": s["makespan_delete_gain_mean"],
                     "hop_default_makespan": float(r["hop_default_makespan"]),
                     "hop_best_makespan": float(r["hop_best_makespan"]),
                     "orig_default_makespan": float(r["orig_default_makespan"])})
    E._write_csv("e4_core_plot_hop.csv", rows)
    with open(os.path.join(OUT, "fcfs", "fcfs_table.csv"), encoding="utf-8") as f:
        fc = list(csv.DictReader(f))
    with open(os.path.join(OUT, "diag", "hop_chunk.json"), encoding="utf-8") as f:
        ch = json.load(f)["results"]
    sym = [x for x in rows if x["T_ratio_orig"] == 1.0 and not x["alias_of"]]
    asym = [x for x in rows if x["T_ratio_orig"] > 1.0]
    lf = [x for x in fc if x["order"] == "long_first"]
    line = {
        "configs_hop": len(rows),
        "hop_gain_asym": [round(min(x["hop_prio_gain"] for x in asym), 3), round(max(x["hop_prio_gain"] for x in asym), 3)],
        "orig_gain_asym": [round(min(x["orig_prio_gain"] for x in asym), 3), round(max(x["orig_prio_gain"] for x in asym), 3)],
        "hop_gain_sym": [round(min(x["hop_prio_gain"] for x in sym), 3), round(max(x["hop_prio_gain"] for x in sym), 3)],
        "orig_smax_bound_max": round(max(x["orig_smax_bound_reduction"] for x in rows), 4),
        "a2_chunk_prio_over_default": {k: round(v["prio_over_default"], 3) for k, v in ch.items()},
        "slack_order_all_short_first": all(x[f"slack_k{k}_order"] == "prio_01" for x in fc for k in (1.1, 1.5, 2.0)),
        "slack_extra_over_fcfs_long_first": [round(min(float(x["slack_extra_gain_k1.5"]) for x in lf), 3),
                                             round(max(float(x["slack_extra_gain_k1.5"]) for x in lf), 3)],
        "artifact_commit": E._artifact_commit()[:8]}
    print("[e4] " + json.dumps(line, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
