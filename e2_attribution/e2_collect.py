"""Collect e2_out/*_summary.json into one table (e2_out/e2_table.csv) and print it."""
import csv
import glob
import json
import os
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_out")
COLS = [
    ("task", lambda s: s["task"].replace("_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0", "")),
    ("a", lambda s: s["latency_a"]),
    ("c/a", lambda s: s["pure_comm_share_c_over_a"]),
    ("b/a", lambda s: s["pure_comp_share_b_over_a"]),
    ("wait/a", lambda s: s["critical_wait_over_a"]),
    ("raw_wait/a", lambda s: s["raw"]["critical_wait_over_a"]),
    ("top1", lambda s: s["wait_top1_resource_share"]),
    ("top5", lambda s: s["wait_top5_resource_share"]),
    ("blk_comm", lambda s: s["wait_share_by_blocker_type"].get("communication", 0)),
    ("blk_ddr_read", lambda s: s["wait_share_by_blocker_data"].get("ddr_read", 0)),
    ("defer_only", lambda s: s["wait_share_by_category"]["defer_only"]),
    ("reroute_only", lambda s: s["wait_share_by_category"]["reroute_only"]),
    ("both", lambda s: s["wait_share_by_category"]["both"]),
    ("neither", lambda s: s["wait_share_by_category"]["neither"]),
    ("neither_raw", lambda s: s["raw"]["wait_share_by_category"]["neither"]),
    ("alt_path", lambda s: s["wait_share_alt_path_exists"]),
    ("idle_local_alt", lambda s: s["wait_share_idle_local_alt_path_exists"]),
    ("dstu_ub/a", lambda s: s["dstu_gain_upper_bound_over_a"]),
    ("ddr_on_chain/a", lambda s: s["ddr_link_time_on_chain_over_a"]),
    # round 2 (primary: local reroute; reorder bound; DDR / corner load)
    ("neither_local", lambda s: s["wait_share_by_category_local"]["neither"]),
    ("neither_local_no_reorder", lambda s: s["wait_share_by_category_local_or_reorder"]["neither"]),
    ("dstu_ub_local/a", lambda s: s["dstu_gain_upper_bound_local_over_a"]),
    ("reorder_best/a", lambda s: s["reorder_gain_best_resource_over_a"]),
    ("reorder_best_res", lambda s: s["reorder_gain_best_resource"]),
    ("reorder_ub_best/a", lambda s: s["reorder_gain_ub_best_resource_over_a"]),
    ("reorder_top5_sum/a_nonadd", lambda s: s["reorder_gain_top5_sum_over_a_nonadditive"]),
    ("ddr_port_busy_max", lambda s: s["ddr_corner"]["ddr_port_read_busy_over_a"]["max"]),
    ("ddr_port_busy_mean", lambda s: s["ddr_corner"]["ddr_port_read_busy_over_a"]["mean"]),
    ("ddr_bytes_max/mean", lambda s: s["ddr_corner"]["ddr_port_read_bytes_max_over_mean"]),
    ("ddr_ports_busy>50%", lambda s: s["ddr_corner"]["ddr_ports_read_busy_over_50pct"]),
    ("corner_out_busy_max", lambda s: s["ddr_corner"]["corner_max_out_link_busy_over_a"]["max"]),
    ("corner_out_busy_mean", lambda s: s["ddr_corner"]["corner_max_out_link_busy_over_a"]["mean"]),
    ("corners_out>50%", lambda s: s["ddr_corner"]["corners_with_out_link_busy_over_50pct"]),
    ("hotspot_corner", lambda s: s["ddr_corner"]["hotspot_corner"]),
    ("wait_ddr_ports", lambda s: s["ddr_corner"]["wait_share_all_ddr_ports"]),
    ("wait_corner_out", lambda s: s["ddr_corner"]["wait_share_all_corner_out_links"]),
    ("d2d_on_chain/a", lambda s: s["d2d"]["d2d_link_time_on_chain_over_a"]),
    ("d2d_crossing_ev_on_chain/a", lambda s: s["d2d"]["d2d_crossing_event_time_on_chain_over_a"]),
    ("wait_on_d2d", lambda s: s["d2d"]["wait_share_on_d2d_links"]),
    ("wait_blocker_crosses_d2d", lambda s: s["d2d"]["wait_share_blocker_crosses_d2d"]),
    ("d2d_busy_max", lambda s: (s["d2d"]["d2d_link_busy_over_a"] or {}).get("max")),
    ("d2d_busy_mean", lambda s: (s["d2d"]["d2d_link_busy_over_a"] or {}).get("mean")),
    ("d2d_bytes_share", lambda s: s["d2d"]["d2d_bytes_share_of_all_comm_bytes"]),
    ("seed_wait_on_d2d", lambda s: [round(r["d2d"]["wait_share_on_d2d_links"], 3) for r in s["reseeded_runs"] if "d2d" in r]),
    ("checks", lambda s: all([s["check1_chain_end_equals_a"], s["check2_chain_intervals_sum_to_a"], s["check3_all_s_ge_r"],
                              s.get("check4_activity_graph_longest_path_equals_a", True)])),
    ("seed_neither_local", lambda s: [round((r.get("wait_share_by_category_local") or {}).get("neither") or 0, 3)
                                      for r in s["reseeded_runs"]]),
    ("seed_reorder_best", lambda s: [round(r.get("reorder_gain_best_resource_over_a") or 0, 3) for r in s["reseeded_runs"]]),
    ("seed_local_reroute", lambda s: [round(r.get("wait_share_idle_local_alt_path_exists") or 0, 3) for r in s["reseeded_runs"]]),
    ("seed_ddr_port_busy_max", lambda s: [round(r["ddr_corner"]["ddr_port_read_busy_over_a"]["max"], 3)
                                          for r in s["reseeded_runs"] if "ddr_corner" in r]),
    ("seed_corner_out_busy_max", lambda s: [round(r["ddr_corner"]["corner_max_out_link_busy_over_a"]["max"], 3)
                                            for r in s["reseeded_runs"] if "ddr_corner" in r]),
    ("seed_hotspot", lambda s: [r["ddr_corner"]["hotspot_corner"] for r in s["reseeded_runs"] if "ddr_corner" in r]),
    ("seed_a", lambda s: [r.get("latency_a") for r in s["reseeded_runs"]]),
    ("seed_neither", lambda s: [round((r.get("wait_share_by_category") or {}).get("neither") or 0, 3) for r in s["reseeded_runs"]]),
    ("seed_dstu_ub", lambda s: [round(r.get("dstu_gain_upper_bound_over_a") or 0, 3) for r in s["reseeded_runs"]]),
]
rows = []
for p in sorted(glob.glob(os.path.join(out_dir, "*_summary.json"))):
    with open(p, encoding="utf-8") as f:
        s = json.load(f)
    rows.append({k: fn(s) for k, fn in COLS})
with open(os.path.join(out_dir, "e2_table.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=[k for k, _ in COLS])
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
