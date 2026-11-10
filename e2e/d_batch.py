"""D batches D1-D5 on S-B (D_PLAN.md v2 section 7; rules PROGRESS 2026-11-11 "D：S-B 冒烟与比较规则"). Children are d_joint.py's.

  python e2e/d_batch.py run D1 [--workers 9] [--out DIR] [--peak-gb 3] [--free-gb 3]
  python e2e/d_batch.py plan D2           list the jobs only
  python e2e/d_batch.py report D1         e2e/d/out/batches/D1/report.json (numbers per arm x condition)

Conditions: ("N",) no fault; ("P", fid) / ("D", fid) C's coordinate in the P / D group; ("X", "G1"/"G2") links between the groups.
"""
import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d_joint as d  # noqa: E402

SET_A = ["PF1", "PF2", "PF4", "PF5", "PF6", "PF7", "PF8", "PCRandom", "PCCluster", "PCEdge", "PCScratch"]
SET_B = ["MF1", "MF5", "MF6", "MF8", "MCRandom", "MCCluster", "MCEdge", "MCScratch"]
G = (0.1, 0.5, 0.9)
KV_REF = (833096.3333333334, 4546804.333333333)          # PROGRESS rule 5
TAUS = [int(round(KV_REF[0] + g * (KV_REF[1] - KV_REF[0]))) for g in G]
T_REF = {"pf_ln": 2186864, "pf_mha": 6388643, "pf_proj": 6560063, "pf_ffn": 44532443, "D": 355200.6666666667}
# arm -> (P/D state for the faulty group, fault-free state, backend, policy, kv)
ARMS = {"B-R0": ("a1_R0", "a1_pre", "orig", "P1", True), "B-FULL": ("a1_fullbb", "a1_pre", "orig", "P1", True),
        "CRISP": ("a3h_rep", "a3h_pre", "hop", "C4", True), "-DSTU": ("a3h_rep", "a3h_pre", "orig", "P1", True),
        "-DSTU-prio": ("a3h_rep", "a3h_pre", "hop", "P1", True), "-EIR": ("a3h_R0", "a3h_pre", "hop", "C4", True),
        "-A3h": ("a1_rep", "a1_pre", "hop", "C4", True), "rev": ("a3h_rep", "a3h_pre", "hop", "PD", True),
        "noKV-CRISP": ("a3h_rep", "a3h_pre", "hop", "C4", False), "noKV-B-R0": ("a1_R0", "a1_pre", "orig", "P1", False)}
MAIN_ARMS = ["B-R0", "B-FULL", "CRISP", "-DSTU", "-DSTU-prio", "-EIR", "noKV-CRISP", "noKV-B-R0"]


def batch_conds(name):
    if name == "D1":
        return [("N",)], MAIN_ARMS + ["-A3h", "rev"]
    if name == "D2":
        return [("D", f) for f in SET_A] + [("X", "G1"), ("X", "G2")], MAIN_ARMS + ["rev"]
    if name == "D3":
        return [("P", f) for f in SET_A], MAIN_ARMS + ["rev"]
    if name == "D4":
        return [("D", f) for f in SET_B] + [("P", f) for f in SET_B], MAIN_ARMS
    if name == "D5":
        return [("N",)] + [(g, f) for g in ("P", "D") for f in SET_A + SET_B] + [("X", "G1"), ("X", "G2")], ["-A3h"]
    raise SystemExit(f"unknown batch {name}")


def rev_ok(cond):
    return cond == ("N",) or (cond[0] in ("P", "D") and cond[1] in SET_A)


def states(arm, cond):
    """(P state, P fault id, D state, D fault id) of an arm under a condition."""
    bad, good = ARMS[arm][0], ARMS[arm][1]
    if cond[0] == "P":
        return bad, cond[1], good, None
    if cond[0] == "D":
        return good, None, bad, cond[1]
    return good, None, good, None


def fault24(c, cond):
    if cond[0] == "N":
        return None
    if cond[0] == "X":
        return {"failed_nodes": [], "failed_links": d.INTER[cond[1]], "failed_pes": []}
    f = c.eir.load_faults()["pf_ln"][cond[1]]
    return d.remap_fault({k: f[k] for k in ("failed_nodes", "failed_links", "failed_pes")}, d.SB_MAP["L" if cond[0] == "P" else "R"])


def f22(c, fid):
    if not fid:
        return None
    f = c.eir.load_faults()["pf_ln"][fid]
    return {k: f[k] for k in ("failed_nodes", "failed_links", "failed_pes")}


def cname(cond):
    return "N" if cond == ("N",) else f"{cond[0]}-{cond[1]}"


def build(name, out):
    import c_eir as c
    c.redirect()
    rep = d.creport()
    ctx = d.kv_ctx(c)
    conds, arms = batch_conds(name)
    xf = os.path.join(out, "export")
    os.makedirs(xf, exist_ok=True)
    exports, jobs = {}, {}

    def xp(op, st, fid):
        key = (op, st, fid)
        if key not in exports:
            exports[key] = d.export_item(c, op, st, fid, rep, xf)
        return exports[key]["spec"]["sink"]

    def src_key(op, st, fid):
        """What state a (op, state, fault) really is: a3h_rep at R0 / L1 is the same snapshot as a3h_R0 / a3h_L1."""
        if st == "a3h_rep" and fid:
            lvl = rep[fid]["steps"]["per_op"][op]["level"]
            if lvl in ("R0", "L1"):
                return (op, f"a3h_{lvl}", fid)
        if st == "a1_rep" and fid:
            lvl = json.load(open(os.path.join(d.OUT, "a1rep", "levels.json"), encoding="utf-8"))[fid][op]["level"]
            if lvl in ("R0", "L1"):
                return (op, f"a1_{lvl}", fid)
        return (op, st, fid)
    sims = os.path.join(out, "sims")
    for cond in conds:
        f24 = fault24(c, cond)
        for arm in arms:
            if arm == "rev" and not rev_ok(cond):
                continue
            pst, pfid, dst, dfid = states(arm, cond)
            backend, pol, kv = ARMS[arm][2], ARMS[arm][3], ARMS[arm][4]
            P_changed = cond[0] in ("N", "P")
            # --- pf_mha unit: solos (unless shared) + 9 overlaps
            pk = src_key("pf_mha", pst, pfid)
            dk = tuple(src_key(o, dst, dfid) for o in d.DC)
            ex = {"pf_mha:P": xp("pf_mha", pst, pfid)}
            for o in d.DC:
                ex[f"{o}:D"] = xp(o, dst, dfid)
            solo_needed = pol in ("C4",) or backend == "orig" or not kv
            runs = []
            if solo_needed:
                runs += [{"kind": "solo", "stream": 0, "backend": backend, "seed": s, "policy": {"name": "P1"}, "label": f"soloP_s{s}"} for s in d.SEEDS]
                runs += [{"kind": "solo", "stream": 1, "backend": backend, "seed": s, "policy": {"name": "P1"}, "label": f"soloD_s{s}"} for s in d.SEEDS]
            for gi, tau in enumerate(TAUS):
                for s in d.SEEDS:
                    runs.append({"kind": "ov", "backend": backend, "seed": s, "policy": {"name": pol}, "tau": tau, "by_op": False,
                                 "label": f"ov_g{G[gi]}_s{s}"})
            jk = ("mha", pk, dk, cname(cond), backend, pol, kv)
            nm = f"{name}_{cname(cond)}_{arm}_mha"
            if jk in jobs:
                jobs[jk]["arms"].append(arm)
            else:
                sp = {"name": nm, "sink": os.path.join(sims, nm + ".pkl.gz"), "exports": ex, "fault24": f24, "reps": 1, "kv_ctx": ctx,
                      "groups": {"P": {"map": "L", "fault22": f22(c, pfid), "exports": ["pf_mha:P"], "P": "pf_mha:P", "kv": kv},
                                 "D": {"map": "R", "fault22": f22(c, dfid), "exports": [f"{o}:D" for o in d.DC], "D": {o: f"{o}:D" for o in d.DC}}},
                      "that": [0, 1] if pol == "C4" else [], "runs": runs}
                jobs[jk] = {"name": nm, "cmd": "simsb", "spec": sp, "folder": sims, "arms": [arm], "cond": cname(cond), "kind": "mha"}
            # --- non-MHA prefill operators: solos only (P state changed), backend of the arm; one pf_ffn check unit per group
            if P_changed and kv and pol in ("C4", "P1") and arm not in ("-DSTU-prio",):
                for op in ("pf_ln", "pf_proj", "pf_ffn"):
                    k2 = ("solo", src_key(op, pst, pfid), cname(cond) if cond[0] == "P" else "N", backend)
                    if k2 in jobs:
                        jobs[k2]["arms"].append(arm)
                        continue
                    nm2 = f"{name}_{cname(cond)}_{arm}_{op}"
                    sp = {"name": nm2, "sink": os.path.join(sims, nm2 + ".pkl.gz"), "exports": {f"{op}:P": xp(op, pst, pfid)},
                          "fault24": f24, "reps": 1, "kv_ctx": ctx,
                          "groups": {"P": {"map": "L", "fault22": f22(c, pfid), "exports": [f"{op}:P"], "P": f"{op}:P", "kv": False}},
                          "that": [], "runs": [{"kind": "solo", "stream": 0, "backend": backend, "seed": s, "policy": {"name": "P1"},
                                                "label": f"soloP_s{s}"} for s in d.SEEDS]}
                    jobs[k2] = {"name": nm2, "cmd": "simsb", "spec": sp, "folder": sims, "arms": [arm], "cond": cname(cond), "kind": op}
            first = {"N": ("N",), "D": ("D", conds[0][1]) if conds[0][0] == "D" else None, "P": ("P", conds[0][1]) if conds[0][0] == "P" else None}
            if arm == "CRISP" and cond in (first["N"], first["D"], first["P"]) and name in ("D1", "D2", "D3"):
                nm3 = f"{name}_{cname(cond)}_{arm}_ffncheck"
                ex3 = {"pf_ffn:P": xp("pf_ffn", pst, pfid)}
                for o in d.DC:
                    ex3[f"{o}:D"] = xp(o, dst, dfid)
                sp = {"name": nm3, "sink": os.path.join(sims, nm3 + ".pkl.gz"), "exports": ex3, "fault24": f24, "reps": 1, "kv_ctx": ctx,
                      "groups": {"P": {"map": "L", "fault22": f22(c, pfid), "exports": ["pf_ffn:P"], "P": "pf_ffn:P", "kv": False},
                                 "D": {"map": "R", "fault22": f22(c, dfid), "exports": [f"{o}:D" for o in d.DC], "D": {o: f"{o}:D" for o in d.DC}}},
                      "that": [0, 1], "runs": [{"kind": "solo", "stream": 0, "backend": "hop", "seed": 1, "policy": {"name": "P1"}, "label": "soloP_s1"},
                                               {"kind": "solo", "stream": 1, "backend": "hop", "seed": 1, "policy": {"name": "P1"}, "label": "soloD_s1"},
                                               {"kind": "ov", "backend": "hop", "seed": 1, "policy": {"name": "C4"}, "f_p": 0.5, "label": "ov_fp0.5_s1"}]}
                jobs[("ffncheck", cname(cond))] = {"name": nm3, "cmd": "simsb", "spec": sp, "folder": sims, "arms": [arm], "cond": cname(cond), "kind": "ffncheck"}
    return list(exports.values()), list(jobs.values())


SEEDS_X = (4, 5, 6, 7, 8, 9, 10)           # rule 12: pf_mha cells get seeds 4-10 as supplement jobs


def ext_jobs(jobs):
    """For every pf_mha job: 3 supplement jobs (one per g); the first also runs P solo with seeds 4-10 if the base job has P solos."""
    out = []
    for j in jobs:
        if j["kind"] != "mha":
            continue
        base = j["spec"]
        has_solo = any(r["label"].startswith("soloP") for r in base["runs"])
        b0 = next(r for r in base["runs"] if r["kind"] == "ov")
        for gi, tau in enumerate(TAUS):
            runs = []
            if gi == 0 and has_solo:
                runs += [{"kind": "solo", "stream": 0, "backend": b0["backend"], "seed": s, "policy": {"name": "P1"}, "label": f"soloP_s{s}"} for s in SEEDS_X]
            runs += [{"kind": "ov", "backend": b0["backend"], "seed": s, "policy": dict(b0["policy"]), "tau": tau, "by_op": False,
                      "label": f"ov_g{G[gi]}_s{s}"} for s in SEEDS_X]
            nm = f"{base['name']}_x{gi}"
            sp = dict(base, name=nm, sink=os.path.join(os.path.dirname(base["sink"]), nm + ".pkl.gz"), runs=runs)
            out.append({"name": nm, "cmd": "simsb", "spec": sp, "folder": j["folder"], "arms": list(j["arms"]), "cond": j["cond"], "kind": "mha"})
    return out


def sa_jobs(out):
    """S-A reduced (D_PLAN v2 section 1): the 2x2 co-location of D0, conditions N, PF4, PF6; B-R0, CRISP, -DSTU-prio; f = 0.4."""
    import c_eir as c
    c.redirect()
    rep = d.creport()
    faults = c.eir.load_faults()
    xf = os.path.join(out, "export")
    sims = os.path.join(out, "sims")
    exports, jobs = {}, []
    for fid in (None, "PF4", "PF6"):
        fdef = f22(c, fid)
        for arm in ("B-R0", "CRISP", "-DSTU-prio"):
            bad, good, backend, pol, _ = ARMS[arm]
            st = bad if fid else good
            for op in ("pf_mha", "pf_ffn"):
                ex = {}
                for o in (op,) + d.DC:
                    k = (o, st, fid)
                    if k not in exports:
                        exports[k] = d.export_item(c, o, st, fid, rep, xf)
                    ex[f"{o}:{st}"] = exports[k]["spec"]["sink"]
                runs = [{"kind": "solo", "stream": 0, "backend": backend, "seed": s, "policy": {"name": "P1"}, "label": f"soloP_s{s}"} for s in d.SEEDS]
                runs += [{"kind": "solo", "stream": 1, "backend": backend, "seed": s, "policy": {"name": "P1"}, "label": f"soloD_s{s}"} for s in d.SEEDS]
                runs += [{"kind": "ov", "f": 0.4, "backend": backend, "seed": s, "policy": {"name": pol}, "stop": False, "record": False,
                          "label": f"ov_f0.4_s{s}"} for s in d.SEEDS]
                nm = f"SA_{fid or 'N'}_{arm}_{op}"
                sp = {"name": nm, "mode": "probe", "fault_def": fdef, "sink": os.path.join(sims, nm + ".pkl.gz"), "exports": ex,
                      "P": f"{op}:{st}", "D": {o: f"{o}:{st}" for o in d.DC}, "chunk": True, "that": [0, 1] if pol == "C4" else [], "runs": runs}
                jobs.append({"name": nm, "cmd": "sim", "spec": sp, "folder": sims, "arms": [arm], "cond": fid or "N", "kind": f"SA_{op}"})
    return list(exports.values()), jobs


def run(name, workers, out, part="all"):
    if part == "all" and os.path.exists(os.path.join(out, "EXT_REMOTE")):
        part = "base"            # the seed 4-10 supplements are kept for the remote bundle (d_remote.py) or a later local pass
        print(f"[{name}] EXT_REMOTE present: running the base jobs (seeds 1-3) only", flush=True)
    exports, jobs = build(name, out)
    if part in ("all", "ext", "exports"):
        xj = ext_jobs(jobs)
        jobs = xj if part == "ext" else jobs + xj
    if name == "D1" and part != "ext":
        e2, j2 = sa_jobs(out)
        exports += e2
        jobs += j2
    bdir = os.path.join(out, "batches", name)
    os.makedirs(bdir, exist_ok=True)
    old = os.path.join(bdir, "plan.json")
    if os.path.exists(old):                     # keep the jobs of an earlier run of this batch in the plan (the report reads it)
        prev = json.load(open(old))
        have = {j["name"] for j in jobs}
        jobs = jobs + [j for j in prev["jobs"] if j["name"] not in have and j["kind"] != "export"]
    json.dump({"exports": [e["spec"] for e in exports], "jobs": [{k: v for k, v in j.items() if k != "spec"} | {"spec": j["spec"]} for j in jobs]},
              open(os.path.join(bdir, "plan.json"), "w"), indent=1, default=str)
    print(f"[{name}] exports {len(exports)}, jobs {len(jobs)}", flush=True)
    r1 = d.run_all(exports, workers)
    bad = [x for x in r1 if x.get("rc")]
    if bad:
        raise SystemExit(f"export failures: {bad}")
    # T-hat check of every export before any simulation (rule 1)
    for e in exports:
        x = d._load(e["spec"]["sink"])["d_export"]
        if not x["T_equal"]:
            raise SystemExit(f"{e['name']}: T-hat {x['T']} differs from C's {x['T_ref']}")
    if part == "exports":
        print(f"[{name}] exports done", flush=True)
        return
    jobs.sort(key=lambda j: 0 if j["kind"] in ("mha", "SA_pf_mha", "SA_pf_ffn", "ffncheck") else 1)
    r2 = d.run_all([j for j in jobs if part != "ext" or j["name"].endswith(("_x0", "_x1", "_x2"))], workers)
    json.dump(r2, open(os.path.join(bdir, f"runs_{part}.json"), "w"), indent=1)
    bad = [x for x in r2 if x.get("rc") or x.get("killed")]
    print(f"[{name}] done; failed {len(bad)}: {bad}", flush=True)
    if bad:
        raise SystemExit(1)     # stop the chain: a failed or killed job is reported before anything else runs


# ---------------------------------------------------------------- report
def report(name, out):
    bdir = os.path.join(out, "batches", name)
    plan = json.load(open(os.path.join(bdir, "plan.json")))
    rows = collections.defaultdict(dict)
    audit = {"n": 0, "bad": 0}
    for j in plan["jobs"]:
        p = j["spec"]["sink"]
        if not os.path.exists(p):
            continue
        s = d._load(p)
        for r in s["runs"]:
            if "audit" in r:
                audit["n"] += r["audit"]["n"]
                audit["bad"] += r["audit"]["bad"]
        for arm in j["arms"]:
            key = (j["cond"], arm)
            R = rows[key]
            if j["kind"] == "mha":
                R.setdefault("_solo", {}).update({r["seed"]: r["out"][0] for r in s["runs"] if r["label"].startswith("soloP")})
                R.setdefault("_soloD", []).extend(r["out"][0] for r in s["runs"] if r["label"].startswith("soloD"))
                R.setdefault("_ovs", []).extend(r for r in s["runs"] if r["kind"] == "ov")
                R.setdefault("mha_jobs", []).append(j["name"])
            elif j["kind"] in ("pf_ln", "pf_proj", "pf_ffn"):
                R[f"T_{j['kind']}_self"] = statistics.mean(r["out"][0] for r in s["runs"])
            else:
                R[j["kind"]] = [{"label": r["label"], "out": r["out"], "C": {k: v["C"] for k, v in r["streams"].items()}, "tau": r.get("tau")} for r in s["runs"]]
    # shared solos: an arm whose pf_mha job has no P solo (-DSTU-prio, rev) pairs with the C4 arm of the same state (CRISP): same graph,
    # and C4 = P1 on one stream bit for bit (D0 item 5)
    for (cond, arm), R in rows.items():
        if not R.get("_solo") and arm in ("-DSTU-prio", "rev") and (cond, "CRISP") in rows:
            R["_solo"] = rows[(cond, "CRISP")].get("_solo", {})
            R["_soloD"] = rows[(cond, "CRISP")].get("_soloD", [])
            R["solo_from"] = "CRISP"
    for (cond, arm), R in rows.items():
        solo, ovs = R.pop("_solo", {}), R.pop("_ovs", [])
        soloD = R.pop("_soloD", [])
        if solo:
            s3 = [v for k, v in solo.items() if k <= 3]
            R["T_P_mha_self"] = statistics.mean(s3) if s3 else None
            R["P_mha_solo_range"] = (max(s3) - min(s3)) if s3 else None
            R["P_mha_solo_range_1_10"] = max(solo.values()) - min(solo.values())
            R["n_solo_seeds"] = len(solo)
        if soloD:
            R["T_D_self"] = statistics.mean(soloD)
        if ovs:
            R["units"] = [{"tau": r["tau"], "seed": r["seed"], "D_lat": r["streams"][1]["C"] - r["tau"], "C_P": r["streams"][0]["C"],
                           "kv_window": r.get("kv_window"),
                           "in_window": bool(r.get("kv_window") and r["kv_window"][0] is not None and r["kv_window"][0] <= r["tau"] + 355201 and r["tau"] <= r["kv_window"][1]),
                           "dP": (r["streams"][0]["C"] - solo[r["seed"]]) if r["seed"] in solo else None} for r in ovs]
    res = {"audit": audit, "taus": TAUS, "T_ref": T_REF, "rows": []}
    for (cond, arm), R in sorted(rows.items()):
        u_all = R.get("units") or []
        u = [x for x in u_all if x["seed"] <= 3]          # main table: seeds 1-3 (supplementary rule 2026-11-11)
        row = {"cond": cond, "arm": arm, **{k: v for k, v in R.items() if k != "units"}}
        dp_all = [x["dP"] for x in u_all if x["dP"] is not None]
        if len(u_all) > len(u) and dp_all:
            row["supp_seeds_1_10"] = {"n_units": len(u_all), "dP_mean": statistics.mean(dp_all), "dP_min": min(dp_all), "dP_max": max(dp_all),
                                      "D_lat_mean": statistics.mean(x["D_lat"] for x in u_all)}
        if u:
            dl = [x["D_lat"] for x in u]
            row.update(D_lat_mean=statistics.mean(dl), D_lat_min=min(dl), D_lat_max=max(dl), D_over_ref=statistics.mean(dl) / T_REF["D"] - 1,
                       Lhat_D=max((x["D_lat"] - 1.5 * T_REF["D"]) / T_REF["D"] for x in u),
                       in_window=sum(x["in_window"] for x in u), n_units=len(u))
            dp = [x["dP"] for x in u if x["dP"] is not None]
            if dp:
                row.update(dP_mean=statistics.mean(dp), dP_min=min(dp), dP_max=max(dp))
            row["Lhat_P_mha"] = max((x["C_P"] - 1.5 * T_REF["pf_mha"]) / T_REF["pf_mha"] for x in u)
            if "T_D_self" in R:
                row["D_over_self"] = statistics.mean(dl) / R["T_D_self"] - 1
        res["rows"].append(row)
    json.dump(res, open(os.path.join(bdir, "report.json"), "w"), indent=1, default=str)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("batch")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--out", default=os.path.join(d.OUT, "prod"))
    ap.add_argument("--peak-gb", type=float)
    ap.add_argument("--free-gb", type=float)
    ap.add_argument("--part", default="all", choices=("all", "base", "ext", "exports"))
    a = ap.parse_args()
    if a.peak_gb:
        d.LOCAL_PEAK_GB = a.peak_gb
    if a.free_gb:
        d.LOCAL_FREE_GB = a.free_gb
    d.OUT = os.path.abspath(os.path.join(a.out, ".."))   # logs next to the run folder
    if a.cmd == "plan":
        ex, jobs = build(a.batch, a.out)
        print(len(ex), "exports;", len(jobs), "jobs")
        for j in jobs:
            print(j["name"], j["arms"], len(j["spec"]["runs"]))
    elif a.cmd == "run":
        run(a.batch, a.workers, a.out, a.part)
    elif a.cmd == "report":
        r = report(a.batch, a.out)
        for row in r["rows"]:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("ffncheck",)})
        print(r["audit"])


if __name__ == "__main__":
    main()
