"""E4b setting 1: compact report tables from e4b_out/setting1_table.csv (no simulation).

  python e4b_s1_report.py
Writes e4b_out/setting1_report.csv (one row per configuration x k: L_hat_max and S_max of P1-P7, DSTU differences
against short first (P3), FCFS (P2) and BestFixed (P5), makespans, stretches) and prints summary lines.
"""
import csv
import json
import os

import e4b as B

POL = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]


def main():
    with open(os.path.join(B.OUT, "setting1_table.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault((r["config"], r["k"]), {})[r["policy"]] = r
    out = []
    for (cfg, k), d in by.items():
        f = lambda p, x: float(d[p][x])
        o = {"config": cfg, "group": d["P1"]["group"], "hw": d["P1"]["hw"], "order": d["P1"]["order"], "k": float(k),
             "alias_of_hw": d["P1"]["alias_of_hw"], "P5_member": d["P5"]["member"]}
        for p in POL:
            o[f"L_{p}"] = f(p, "Lhat_max")
            o[f"S_{p}"] = f(p, "S_max")
            o[f"M_{p}"] = f(p, "makespan")
            o[f"V_{p}"] = f(p, "violations")
            o[f"Lr_{p}"] = f(p, "Lhat_range")
            o[f"st_{p}"] = json.dumps({kk[8:]: round(float(v), 3) for kk, v in d[p].items() if kk.startswith("stretch_") and v})
        for p in ("P6", "P7"):
            for ref in ("P3", "P2", "P5"):
                o[f"dL_{p}_{ref}"] = o[f"L_{p}"] - o[f"L_{ref}"]
                o[f"rS_{p}_{ref}"] = o[f"S_{p}"] / o[f"S_{ref}"] - 1
        out.append(o)
    B.write_csv("setting1_report.csv", out)
    worse7 = [o for o in out if o["S_P7"] > o["S_P3"] * 1.04]
    better6 = [o for o in out if o["S_P6"] < o["S_P5"] * 0.96]
    line = {"rows": len(out), "P7_worse_than_SJF_gt4pct": len(worse7),
            "P6_better_than_BestFixed_gt4pct": len(better6),
            "rS_P6_P3": [round(min(o["rS_P6_P3"] for o in out), 3), round(max(o["rS_P6_P3"] for o in out), 3)],
            "rS_P7_P3": [round(min(o["rS_P7_P3"] for o in out), 3), round(max(o["rS_P7_P3"] for o in out), 3)],
            "rS_P6_P5": [round(min(o["rS_P6_P5"] for o in out), 3), round(max(o["rS_P6_P5"] for o in out), 3)]}
    B.log(json.dumps(line))


if __name__ == "__main__":
    main()
