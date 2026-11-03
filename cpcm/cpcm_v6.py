"""V6: attribution consistency (cpcm/CPCM_HANDOFF.md section 3). Record, do not stop.

On V2's 30 points (5 mappings x no fault + 5 faults; cpcm_out/v2/<task>_<F>.pkl.gz: compiled graph + E2-format
backend records of seeds 1-5): CPCM bb mode -> critical path and CB(rho) (exclusive crediting, main measure, user
2026-11-03) vs E2's critical chain and exclusive wait per resource (e2_attribution.Sim(cap).chain(), the same code
E2 ran) on backend seed 1. Noise reference: E2 on seed 1 vs E2 on seed 2.
Reported per point:
  critical resources: the resources with exclusive critical wait > 0; Jaccard of the two sets, and the share of
    E2's exclusive wait that falls on resources also critical in CPCM;
  top 3 resources by CB: same set? same order? same top 1?
  busiest link != link with the largest CB: in CPCM's schedule (CPCM busy time, CPCM CB) and in the backend
    record (busy time, E2 CB).
Usage: python cpcm_v6.py
"""
import collections
import csv
import gzip
import json
import os
import pickle
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "e2_attribution"))
import cpcm  # noqa: E402
import cpcm_v2  # noqa: E402
import e2_attribution as e2  # noqa: E402

V2 = os.path.join(HERE, "cpcm_out", "v2")


def e2_cb(cap):
    sim = e2.Sim(cap)
    _, rows, segs = sim.chain()
    cb = collections.Counter()
    for s in segs:
        cb[s["res_key"]] += s["wait_excl"]
    return {k: v for k, v in cb.items() if v > 0}


def busiest_link_backend(cap):
    busy = {("link", k): sum(f - s for s, f, _ in rec) for k, rec in cap["links"].items()}
    return max(busy, key=lambda k: (busy[k], str(k))), busy


def compare(a, b):
    """a, b: {resource: CB}. Jaccard, share of a's mass on b's set, top-3 agreement."""
    sa, sb = set(a), set(b)
    jac = len(sa & sb) / len(sa | sb) if sa | sb else 1.0
    mass = sum(a.values())
    cov = sum(v for k, v in a.items() if k in sb) / mass if mass else 1.0
    ta = [k for k, _ in sorted(a.items(), key=lambda x: (-x[1], str(x[0])))[:3]]
    tb = [k for k, _ in sorted(b.items(), key=lambda x: (-x[1], str(x[0])))[:3]]
    return {"jaccard": round(jac, 4), "mass_share_on_other_set": round(cov, 4), "top3_same_set": set(ta) == set(tb),
            "top3_same_order": ta == tb, "top1_same": bool(ta and tb and ta[0] == tb[0]),
            "top3_overlap": len(set(ta) & set(tb)), "n_a": len(sa), "n_b": len(sb)}


def main():
    rows = []
    for name in cpcm_v2.TASKS:
        for fid in ["F0"] + cpcm_v2.FAULTS:
            p = os.path.join(V2, f"{name}_{fid}.pkl.gz")
            if not os.path.exists(p):
                continue
            with gzip.open(p, "rb") as f:
                sink = pickle.load(f)
            g = sink["graph"]
            caps = {r["seed"]: r["cap"] for r in sink["runs"]}
            sc = cpcm.simulate(g, "bb")
            an = cpcm.Analysis(sc)
            cb_c = {g.res_name[q]: v for q, v in an.blocking().items() if v > 0}
            cb_e1, cb_e2 = e2_cb(caps[1]), e2_cb(caps[2])
            busy_c = collections.Counter()
            for a in range(len(sc.a_s)):
                if g.res_kind[sc.a_res[a]] != "dev" and not str(g.res_kind[sc.a_res[a]]).startswith("dev"):
                    busy_c[g.res_name[sc.a_res[a]]] += sc.a_f[a] - sc.a_s[a]
            bl_c = max(busy_c, key=lambda k: (busy_c[k], str(k)))
            links_c = {k: v for k, v in cb_c.items() if k[0] == "link"}
            cbl_c = max(links_c, key=lambda k: (links_c[k], str(k))) if links_c else None
            bl_b, _ = busiest_link_backend(caps[1])
            links_e = {k: v for k, v in cb_e1.items() if k[0] == "link"}
            cbl_e = max(links_e, key=lambda k: (links_e[k], str(k))) if links_e else None
            ce = compare(cb_e1, cb_c)
            ee = compare(cb_e1, cb_e2)
            top = lambda d: [(str(k), v) for k, v in sorted(d.items(), key=lambda x: (-x[1], str(x[0])))[:3]]
            rows.append({"task": name, "fault": fid,
                         **{f"cpcm_vs_e2_{k}": v for k, v in ce.items()},
                         **{f"e2s1_vs_e2s2_{k}": v for k, v in ee.items()},
                         "cpcm_cb_total": sum(cb_c.values()), "e2_cb_total": sum(cb_e1.values()),
                         "cpcm_top3": top(cb_c), "e2_top3": top(cb_e1),
                         "cpcm_busiest_link": str(bl_c), "cpcm_maxcb_link": str(cbl_c),
                         "cpcm_busiest_ne_maxcb": cbl_c is not None and bl_c != cbl_c,
                         "backend_busiest_link": str(bl_b), "e2_maxcb_link": str(cbl_e),
                         "backend_busiest_ne_maxcb": cbl_e is not None and bl_b != cbl_e,
                         "cpcm_cp_rows": len(an.critical_path()), "cpcm_cp_gaps": an.gaps,
                         "cpcm_idle_other": sum(an.idle.values())})
    n = len(rows)

    def frac(key):
        return f"{sum(bool(r[key]) for r in rows)}/{n}"

    def med(key):
        v = [r[key] for r in rows]
        return {"median": round(statistics.median(v), 4), "min": round(min(v), 4), "mean": round(statistics.mean(v), 4)}

    summ = {"points": n,
            "cpcm_vs_e2": {"jaccard": med("cpcm_vs_e2_jaccard"), "mass_share": med("cpcm_vs_e2_mass_share_on_other_set"),
                           "top3_same_set": frac("cpcm_vs_e2_top3_same_set"), "top3_same_order": frac("cpcm_vs_e2_top3_same_order"),
                           "top1_same": frac("cpcm_vs_e2_top1_same"), "top3_overlap": med("cpcm_vs_e2_top3_overlap")},
            "e2_seed1_vs_seed2": {"jaccard": med("e2s1_vs_e2s2_jaccard"), "mass_share": med("e2s1_vs_e2s2_mass_share_on_other_set"),
                                  "top3_same_set": frac("e2s1_vs_e2s2_top3_same_set"),
                                  "top3_same_order": frac("e2s1_vs_e2s2_top3_same_order"),
                                  "top1_same": frac("e2s1_vs_e2s2_top1_same"), "top3_overlap": med("e2s1_vs_e2s2_top3_overlap")},
            "busiest_ne_maxcb": {"cpcm": frac("cpcm_busiest_ne_maxcb"), "backend_e2": frac("backend_busiest_ne_maxcb")},
            "cpcm_cp_gaps_total": sum(r["cpcm_cp_gaps"] for r in rows),
            "cpcm_idle_other_total": sum(r["cpcm_idle_other"] for r in rows)}
    summ["by_task"] = {t: {"top1_same": f"{sum(r['cpcm_vs_e2_top1_same'] for r in rows if r['task'] == t)}/"
                                        f"{sum(1 for r in rows if r['task'] == t)}",
                           "top1_same_e2_noise": f"{sum(r['e2s1_vs_e2s2_top1_same'] for r in rows if r['task'] == t)}/"
                                                 f"{sum(1 for r in rows if r['task'] == t)}",
                           "jaccard_median": round(statistics.median(r["cpcm_vs_e2_jaccard"] for r in rows if r["task"] == t), 4)}
                       for t in dict.fromkeys(r["task"] for r in rows)}
    with open(os.path.join(HERE, "cpcm_out", "v6_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(HERE, "cpcm_out", "v6_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
