"""Post-processing only (no simulation): C4 vs C-S paired by row on settings 1-3, rules fixed in PROGRESS before setting 3 ran.
  python dstu_pair.py
Reads out/setting{1,2,3}_report.csv and E4b's tables (P5's seed range); writes out/pair_c4_cs.csv, out/pair_c4_cs.json.
Delta = C4 - C-S, so negative favours C4. |Delta L_hat| > 0.04 counts as a difference.
Strictly better than BestFixed: mean L_hat below P5 by more than the larger of the two seed ranges (E4b's definition).
"""
import csv
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
E4B = os.path.join(HERE, "..", "e4_overlap", "e4b_out")
TH = 0.04


def read(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def p5_ranges():
    out = {}
    for n, fn in ((1, "setting1c_table.csv"), (2, "setting2_table.csv"), (3, "setting3_table.csv")):
        for r in read(os.path.join(E4B, fn)):
            if r["policy"] != "P5":
                continue
            if n == 1:
                k = (r["config"], f"k{float(r['k'])}")
            elif n == 2:
                k = (r["config"], f"kl{float(r['k_long']):g}_ks{float(r['k_short']):g}")
            else:
                k = (r["config"], "s3")
            out[(n, *k)] = float(r["Lhat_range"])
    return out


def nstreams(r):
    return 3 if "3s" in r["group"] else 2


def block(rows, rng):
    if not rows:
        return None
    F = lambda r, k: float(r[k])
    d = [F(r, "L_C4") - F(r, "L_C-S") for r in rows]
    dr = [F(r, "Lraw_C4") - F(r, "Lraw_C-S") for r in rows]
    w4 = max(rows, key=lambda r: F(r, "L_C4") - F(r, "L_C-S"))
    ws = min(rows, key=lambda r: F(r, "L_C4") - F(r, "L_C-S"))

    def strict(p):
        return sum(1 for r in rows if F(r, f"L_{p}") < F(r, "L_P5") - max(rng[r["_k"]], F(r, f"Lrange_{p}")))
    c5 = [F(r, "L_C4") - F(r, "L_P5") for r in rows]
    p45 = [F(r, "L_P4") - F(r, "L_P5") for r in rows]
    s5 = [F(r, "L_C-S") - F(r, "L_P5") for r in rows]
    streams = sum(nstreams(r) for r in rows)
    return {
        "n": len(rows),
        "dL_median": statistics.median(d), "dL_min": min(d), "dL_max": max(d),
        "C4_better": sum(x < -TH for x in d), "CS_better": sum(x > TH for x in d), "tie": sum(abs(x) <= TH for x in d),
        "dLraw_median": statistics.median(dr),
        "worst_L_C4": max(F(r, "L_C4") for r in rows), "worst_L_CS": max(F(r, "L_C-S") for r in rows),
        "L_median_C4": statistics.median(F(r, "L_C4") for r in rows), "L_median_CS": statistics.median(F(r, "L_C-S") for r in rows),
        "max_C4_worse": {"row": f"{w4['config']} {w4['dl']}", "L_C4": F(w4, "L_C4"), "L_CS": F(w4, "L_C-S")},
        "max_CS_worse": {"row": f"{ws['config']} {ws['dl']}", "L_C4": F(ws, "L_C4"), "L_CS": F(ws, "L_C-S")},
        "V_mean_C4": statistics.mean(F(r, "V_C4") for r in rows), "V_mean_CS": statistics.mean(F(r, "V_C-S") for r in rows),
        "viol_rate_C4": sum(F(r, "V_C4") for r in rows) / streams, "viol_rate_CS": sum(F(r, "V_C-S") for r in rows) / streams,
        "strict_P5_C4": strict("C4"), "strict_P5_CS": strict("C-S"),
        "C4_P5_median": statistics.median(c5), "C4_P5_better": sum(x < -TH for x in c5), "C4_P5_worse": sum(x > TH for x in c5), "C4_P5_max": max(c5), "C4_P5_min": min(c5),
        "C4_order_is_P5": sum(1 for r in rows if r["P5_member"] == "fixed_" + r["C4_order"]),
        "P4_P5_median": statistics.median(p45), "P4_P5_worse": sum(x > TH for x in p45), "P4_P5_max": max(p45),
        "CS_P5_median": statistics.median(s5), "CS_P5_better": sum(x < -TH for x in s5), "CS_P5_worse": sum(x > TH for x in s5),
    }


def main():
    rng = p5_ranges()
    res = {}
    table = []
    for n in (1, 2, 3):
        rows = read(os.path.join(OUT, f"setting{n}_report.csv"))
        for r in rows:
            r["_k"] = (n, r["config"], r["dl"])
        parts = {"all": rows}
        if n == 2:
            parts = {"forward": [r for r in rows if r["kind"] == "forward"], "reverse": [r for r in rows if r["kind"] == "reverse"]}
        for pn, sub in parts.items():
            cuts = {"all": sub, "a1": [r for r in sub if r["group"].startswith("a1")], "a2": [r for r in sub if r["group"].startswith("a2")]}
            for hw in ("nofault", "F4", "F5"):
                cuts[hw] = [r for r in sub if r["hw"] == hw]
            for g in ("a1", "a2"):
                for hw in ("nofault", "F4", "F5"):
                    cuts[f"{g}_{hw}"] = [r for r in sub if r["group"].startswith(g) and r["hw"] == hw]
            for c, v in cuts.items():
                b = block(v, rng)
                if b:
                    key = f"s{n}_{pn}_{c}"
                    res[key] = b
                    table.append({"setting": n, "part": pn, "cut": c, **{k: x for k, x in b.items() if not isinstance(x, dict)},
                                  "max_C4_worse_row": b["max_C4_worse"]["row"], "max_CS_worse_row": b["max_CS_worse"]["row"]})
    with open(os.path.join(OUT, "pair_c4_cs.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    fields = []
    for r in table:
        fields += [k for k in r if k not in fields]
    with open(os.path.join(OUT, "pair_c4_cs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    for k, b in res.items():
        if any(k.endswith(x) for x in ("_all", "_a1", "_a2", "_nofault", "_F4", "_F5")) and not any(k.endswith(f"_a{i}_{h}") for i in (1, 2) for h in ("nofault", "F4", "F5")):
            print(k, {a: (round(v, 3) if isinstance(v, float) else v) for a, v in b.items() if a not in ("max_C4_worse", "max_CS_worse", "dLraw_median")}, "dLraw", round(b["dLraw_median"]))
            print("   ", b["max_C4_worse"]["row"], round(b["max_C4_worse"]["L_C4"], 3), round(b["max_C4_worse"]["L_CS"], 3), "|", b["max_CS_worse"]["row"], round(b["max_CS_worse"]["L_C4"], 3), round(b["max_CS_worse"]["L_CS"], 3))


if __name__ == "__main__":
    main()
