"""Compare the DSTU + E4b upstream summaries of the original W0 (dstu/out, e4_overlap/e4b_out), the W0-tag tree and the W1 tree.
Every numeric leaf of each summary JSON is matched by key path; per file: leaves compared, identical, median and max |relative
difference|, for W0-tag vs W0 (hash only) and W1 vs W0-tag (workload only). Writes out/compare_dstu.json.

  python e2e/w1_rerun/compare_dstu.py
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TREES = {"W0": ROOT, "W0tag": os.path.join(HERE, "dstu_w0tag"), "W1": os.path.join(HERE, "dstu_w1")}
SKIP = ("wall", "time", "built", "elapsed", "sec", "mem")


def files():
    t = TREES["W1"]
    out = []
    for pat in ("dstu/out/*.json", "dstu/out/*/*.json", "e4_overlap/e4b_out/*.json"):
        out += [os.path.relpath(p, t).replace("\\", "/") for p in glob.glob(os.path.join(t, pat))]
    return sorted(p for p in out if "/sims" not in p and "/_tmp" not in p)


def flat(o, p="", acc=None):
    acc = {} if acc is None else acc
    if isinstance(o, dict):
        for k, v in o.items():
            if not any(s in str(k).lower() for s in SKIP):
                flat(v, f"{p}/{k}", acc)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            flat(v, f"{p}[{i}]", acc)
    elif isinstance(o, bool):
        acc[p] = float(o)
    elif isinstance(o, (int, float)):
        acc[p] = float(o)
    return acc


def diff(a, b):
    ks = sorted(set(a) & set(b))
    rel = []
    same = 0
    for k in ks:
        x, y = a[k], b[k]
        if x == y:
            same += 1
            rel.append(0.0)
        else:
            rel.append(abs(y - x) / max(abs(x), abs(y), 1e-12))
    return {"n": len(ks), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)), "identical": same,
            "rel_median": round(st.median(rel), 4) if rel else None, "rel_max": round(max(rel), 4) if rel else None}


def main():
    res = {}
    for f in files():
        d = {}
        for w, t in TREES.items():
            p = os.path.join(t, f)
            d[w] = flat(json.load(open(p, encoding="utf-8"))) if os.path.isfile(p) else None
        row = {}
        if d["W0"] is not None and d["W0tag"] is not None:
            row["W0tag_vs_W0"] = diff(d["W0"], d["W0tag"])
        if d["W0tag"] is not None and d["W1"] is not None:
            row["W1_vs_W0tag"] = diff(d["W0tag"], d["W1"])
        res[f] = row
        a, b = row.get("W0tag_vs_W0"), row.get("W1_vs_W0tag")
        fm = lambda r: f"{r['identical']}/{r['n']} same, med {r['rel_median']}, max {r['rel_max']}" if r else "-"
        print(f"{f:45s} | hash: {fm(a):38s} | workload: {fm(b)}")
    json.dump(res, open(os.path.join(HERE, "out", "compare_dstu.json"), "w"), indent=1)


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


# ---- headline numbers (python e2e/w1_rerun/compare_dstu.py headline) ----
def headline():
    import csv
    def rc(t, f):
        return list(csv.DictReader(open(os.path.join(t, f), encoding="utf-8")))
    def js(t, f):
        return json.load(open(os.path.join(t, f), encoding="utf-8"))
    def med(x):
        return round(st.median(x), 3) if x else None
    out = {}
    for w, t in TREES.items():
        h = {}
        for s in ("1", "2", "3"):
            R = rc(t, f"e4_overlap/e4b_out/setting{s}_table.csv")
            L = {}
            for r in R:
                L.setdefault((r["config"], r.get("k", "")), {})[r["policy"]] = float(r["Lhat_max"])
            pol = {}
            for p in ("P1", "P2", "P3", "P4", "P6", "P7", "P7m"):
                d = [v[p] - v["P5"] for v in L.values() if p in v and "P5" in v]
                if d:
                    pol[p] = {"n": len(d), "better": sum(x < -0.04 for x in d), "worse": sum(x > 0.04 for x in d), "dmed": med(d), "dmax": round(max(d), 3)}
            h[f"e4b_s{s}_vs_P5"] = pol
            h[f"e4b_s{s}_P5_minus_P2_median"] = med([v["P5"] - v["P2"] for v in L.values() if "P2" in v])
        for s in (1, 2, 3):
            j = js(t, f"dstu/out/setting{s}.json")
            h[f"dstu_s{s}_compare"] = {k: {x: (round(v[x], 3) if isinstance(v[x], float) else v[x]) for x in ("better", "worse", "dL_median", "dL_max")} for k, v in j["compare"].items()}
            h[f"dstu_s{s}_retention_median"] = {k: round(v["median"], 3) for k, v in j["retention"].items() if v.get("median") is not None}
        p = js(t, "dstu/out/pair_c4_cs.json")
        h["pair"] = {k: {x: (round(v[x], 3) if isinstance(v[x], float) else v[x]) for x in ("n", "C4_better", "CS_better", "tie", "dL_median")} for k, v in p.items() if k.endswith("_all")}
        a = js(t, "dstu/out/setting3_all/setting3_all.json")
        h["s3all_regret_all"] = {k: {x: (round(v[x], 3) if isinstance(v[x], float) else v[x]) for x in v if x in ("median", "max", "over", "under", "same_as_best")} for k, v in a["regret"]["all"].items()}
        h["s3all_hard_b_n"] = a["checks"]["hard_b_n"]
        f2 = js(t, "dstu/out/final/s2.json")
        h["final_s2_regret"] = {g: {k: {x: (round(v[x], 3) if isinstance(v[x], float) else v[x]) for x in v if x in ("median", "max", "over", "under")} for k, v in f2["regret"][g].items()} for g in ("fwd", "rev")}
        h["final_s2_hard_stage1"] = len(f2["hard_stage1"])
        sel = {}
        for r in rc(t, "dstu/out/final/sel_pairs.csv"):
            k = r["pair"].split(" ")[0] + " - " + r["pair"].split(" - ")[1].rstrip("0123456789").strip()
            c = sel.setdefault(k, {"better": 0, "worse": 0, "tie": 0, "dL": []})
            c[{"-1": "better", "1": "worse", "0": "tie"}[r["dL_sign"]]] += 1
            c["dL"].append(float(r["dL"]))
        h["sel_pairs"] = {k: {"better": v["better"], "worse": v["worse"], "tie": v["tie"], "dL_min": round(min(v["dL"]), 3), "dL_max": round(max(v["dL"]), 3)} for k, v in sel.items()}
        oh = {(r["hw"], r["o"], r["scheme"], r["policy"]): float(r["L"]) for r in rc(t, "dstu/out/overhead/rows.csv")}
        cnt = {"all256C4_vs_d2d256C4": [0, 0], "all256C4_vs_all256P1": [0, 0]}
        for (hw, o, sc, pol), v in oh.items():
            if sc == "all256" and pol == "C4":
                cnt["all256C4_vs_d2d256C4"][0 if v < oh[(hw, o, "d2d256", "C4")] else 1] += 1
                cnt["all256C4_vs_all256P1"][0 if v < oh[(hw, o, "all256", "P1")] else 1] += 1
        h["overhead_better_notbetter"] = cnt
        out[w] = h
    json.dump(out, open(os.path.join(HERE, "out", "compare_dstu_headline.json"), "w"), indent=1)
    return out


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "headline":
    headline()
