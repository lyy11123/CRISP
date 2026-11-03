"""E4b step 2: setting 1 (same multiplier k), handoff 2.3. Per-hop backend e4b_backend; policies of e4b.py.

Groups: a1 512+8192, a1 2048+8192, a1 512/2048/8192, a1 2048x2 (unchunked); a2 MHA+FFN (256 KB D2D-only).
Hardware: no fault, F4, F5 (E4a batch-1 faults and R0 graphs; a1 2048+8192 has F5 = F4, listed from F4).
Releases (T = the shortest stream alone, per hop, same hardware and chunking): d0 = all 0; short_first = streams in
  increasing T_i released at 0, D, 2D (D = round(0.5 T)); long_first = the reverse order.
k in {1.1, 1.5, 2.0} (d_i = r_i + k T_i).
Policies: P1; P5 members (every fixed stream order); P6 and P7 per k. P2, P3, P4 are the fixed order their key
  induces (same keys order => same choices), so they are read from the corresponding P5 member's runs, not run
  again: P2 = release order (d0: ties everywhere -> P1); P3 = increasing T_i; P4 = increasing d_i (per k).
  P0 (original backend, length layering) once per group at d0, no fault (e4_overlap.simulate).
Backend seeds 1-3. Self-checks: 1 (P1 vs E4a hop grid / hop chunk at d0, stop > 4%; short_first d0.5 recorded),
  2 (fixed orders vs hop grid d0 and fcfs_table long_first 0.5T), 3 (P6 with constant slack order: shadow = short
  first, differ = 0 and S_max within 4% of the short-first order), 4 (audit), 5 (R_hat along the solo runs of
  every reference stream), 7 (walls).

  python e4b_s1.py --probe      # one 3-stream configuration: walls -> estimate
  python e4b_s1.py --jobs 12
Writes e4b_out/setting1_runs.csv, setting1_table.csv, setting1.json.
"""
import argparse
import collections
import csv
import gzip
import itertools
import json
import os
import pickle
import statistics
import sys
import time

import e4_grid as G
import e4_hop as H
import e4_overlap as E
import e4b as B

KS = [1.1, 1.5, 2.0]
# user decision 2026-11-02 ("按 (b) 处理"): configurations where only single seeds fall on another discrete outcome
# (shadow differ = 0) are extended to seeds 1-6 for every policy, and so is the matching E4a per-hop run
EXT6 = {"a2_2s_mha-ffn_F4_short_first"}
EXT_SEEDS = [4, 5, 6]
GRP = {"a1_2s_512-8192": None, "a1_2s_2048-8192": None, "a1_3s_512-2048-8192": None, "a1_2s_2048-2048": None,
       "a2_2s_mha-ffn": "256d2d"}


def load():
    with open(os.path.join(E.OUT, "grid1_state.json"), encoding="utf-8") as f:
        st = json.load(f)
    for g, fs in st["faults"].items():
        for fl in fs.values():
            fl["failed_links"] = [tuple(map(tuple, l)) for l in fl["failed_links"]]
    with open(os.path.join(E.OUT, "hop", "hop_state.json"), encoding="utf-8") as f:
        Thop = json.load(f)["T_hop"]
    return st, Thop


def platforms(st):
    out = []
    same = set(st["F5_same_as_F4"])
    for g, cz in GRP.items():
        keys = G.GROUPS[g][1]
        for hw in ["nofault", "F4", "F5"]:
            if hw == "nofault":
                out.append((g, hw, cz, keys, None, None))
                continue
            src = "F4" if (hw == "F5" and g in same) else hw
            fl = st["faults"][g][src]
            out.append((g, hw, cz, [G.fault_key(k, fl) for k in keys],
                        {"failed_nodes": [], "failed_links": fl["failed_links"]}, "F4" if src != hw else None))
    return out


def releases(T, order, D):
    n = len(T)
    if order == "d0":
        return {i: 0 for i in range(n)}
    by = sorted(range(n), key=lambda i: (T[i], i))
    if order == "long_first":
        by = by[::-1]
    return {i: r * D for r, i in enumerate(by)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    st, Thop = load()
    plats = platforms(st)
    if args.probe:
        plats = [p for p in plats if p[0] == "a1_3s_512-2048-8192" and p[1] == "nofault"]
    # ---- T: unchunked from E4a hop singles; chunked from E4a diag chunk singles, missing ones run here (6 seeds)
    T, need = {}, []
    for g, hw, cz, keys, fa, alias in plats:
        for k in keys:
            if cz is None:
                T[(k, cz)] = Thop[k]
            else:
                p = os.path.join(E.OUT, "diag", "sims", f"chunk_single_{k}_{cz}.pkl.gz")
                if os.path.exists(p):
                    with gzip.open(p, "rb") as f:
                        T[(k, cz)] = E._mean([r["C"][0] for r in pickle.load(f)["runs"]])
                else:
                    need.append((f"Tsingle_{k}_{cz}", {"cfg": E.CFG, "faults": fa, "streams": [k], "release": {0: 0},
                                                       "seeds": G.D1_SEEDS, "policy": {"name": "P1"},
                                                       "chunk": H.CHUNKS[cz]}))
    if need:
        r = B.run_all(args.jobs, need)
        for n, _ in need:
            k, cz = n[len("Tsingle_"):].rsplit("_", 1)
            T[(k, cz)] = E._mean([x["C"][0] for x in r[n]["runs"]])
    # ---- solo reference runs (caps, seeds 1-3) for P6/P7 and self-check 5
    solo_items = {}
    for g, hw, cz, keys, fa, alias in plats:
        for k in keys:
            solo_items[(k, cz)] = (f"solo_{k}_{cz or 'none'}", {"cfg": E.CFG, "faults": fa, "streams": [k], "release": {0: 0},
                                                              "seeds": B.SEEDS, "policy": {"name": "P1"},
                                                              "chunk": H.CHUNKS[cz] if cz else None, "keep_cap": True})
    sres = B.run_all(args.jobs, list(solo_items.values()))
    solo, chk5 = {}, {}
    for (k, cz), (n, _) in solo_items.items():
        caps = {r["seed"]: r["cap"] for r in sres[n]["runs"]}
        solo[(k, cz)] = B.solo_data(caps[1])
        chk5[f"{k}|{cz or 'none'}"] = B.rhat_check(caps, solo[(k, cz)], T[(k, cz)])
    del sres
    bad5 = [k for k, x in chk5.items() if statistics.median([v["median"] for v in x.values()]) > 0.05]
    if bad5:
        raise B.E4bError(f"self-check 5: median R_hat deviation > 5%: {bad5}")
    t1 = time.time()
    # ---- configurations
    cfgs, items = [], []
    for g, hw, cz, keys, fa, alias in plats:
        n = len(keys)
        Tc = {i: T[(k, cz)] for i, k in enumerate(keys)}
        D = int(round(0.5 * min(Tc.values())))
        for order in ["d0", "short_first", "long_first"]:
            rel = releases(Tc, order, D)
            name = f"{g}_{hw}_{order}"
            c = {"name": name, "group": g, "hw": hw, "chunk": cz, "order": order, "streams": keys, "T": Tc,
                 "release": rel, "D": D if order != "d0" else 0, "alias_of_hw": alias, "n": n}
            cfgs.append(c)
            if alias:
                continue
            base = {"cfg": E.CFG, "faults": fa, "streams": keys, "release": rel, "seeds": B.SEEDS,
                    "chunk": H.CHUNKS[cz] if cz else None, "audit": True}
            sd = {i: solo[(k, cz)] for i, k in enumerate(keys)}
            ref_rank = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: (Tc[i], i)))}
            items.append((f"{name}_P1", dict(base, policy={"name": "P1"})))
            for perm in itertools.permutations(range(n)):
                items.append((f"{name}_fixed_{''.join(map(str, perm))}",
                              dict(base, policy={"name": "fixed", "rank": {s: r for r, s in enumerate(perm)}})))
            for k in KS:
                d = {i: rel[i] + k * Tc[i] for i in Tc}
                pp = {"solo": sd, "d": d, "T": Tc, "release": rel, "ref_rank": ref_rank}
                items.append((f"{name}_P6_k{k}", dict(base, policy=dict(pp, name="P6"), shadow_rank=ref_rank)))
                items.append((f"{name}_P7_k{k}", dict(base, policy=dict(pp, name="P7"), shadow_rank=ref_rank)))
    p0 = []
    for g, hw, cz, keys, fa, alias in plats:
        if hw == "nofault":
            var = {"name": "default", "release": {i: 0 for i in range(len(keys))}}
            if cz:
                var["chunk"] = H.CHUNKS[cz]
            p0.append((g, keys, var))
    ext = [(f"{n}_s456", dict(s, seeds=EXT_SEEDS)) for n, s in items if any(n.startswith(c + "_") for c in EXT6)]
    res = B.run_all(args.jobs, items + ext)
    for n, _ in ext:
        res[n[:-len("_s456")]] = {"runs": res[n[:-len("_s456")]]["runs"] + res[n]["runs"]}
    # E4a per-hop (e4_hop child) counterparts of the extended configurations, seeds 4-6, kept in e4b_out/e4a_ext
    e4a_ext = {}
    for cn in EXT6:
        c = next(x for x in cfgs if x["name"] == cn)
        g, hw, order = c["group"], c["hw"], c["order"]
        fa = next(p[4] for p in plats if p[0] == g and p[1] == hw)
        fname = f"{g}_{hw}_{c['chunk'] or 'none'}_{order}_d0.5"
        for vn, pr in (("prio_01", {0: 0, 1: 1}), ("prio_10", {0: 1, 1: 0})):
            r = H.run(f"{fname}_{vn}_s456", os.path.join(B.OUT, "e4a_ext"),
                      {"streams": c["streams"], "cfg": E.CFG, "faults": fa, "release": c["release"], "prio": pr,
                       "seeds": EXT_SEEDS, "mode": "perhop", "chunk": H.CHUNKS[c["chunk"]] if c["chunk"] else None})
            with gzip.open(os.path.join(E.OUT, "fcfs", "sims", f"{fname}_{vn}.pkl.gz"), "rb") as f:
                base3 = pickle.load(f)["runs"]
            e4a_ext[(cn, vn)] = base3 + r["runs"]
    t2 = time.time()
    if args.probe:
        walls = {n: E._mean([r["wall"] for r in v["runs"]]) for n, v in res.items()}
        procs = {n: v["process_wall"] for n, v in res.items()}
        allp = platforms(st)
        n2 = sum(1 for p in allp if not p[5] and len(G.GROUPS[p[0]][1]) == 2) * 3
        n3 = sum(1 for p in allp if not p[5] and len(G.GROUPS[p[0]][1]) == 3) * 3
        per3 = sum(procs.values())  # one 3-stream configuration (13 processes)
        est = (per3 * n3 + per3 * 0.35 * n2) / args.jobs / 3600
        B.write_json("setting1_probe.json", {"walls": walls, "process_walls": procs, "n_configs_3s": n3, "n_configs_2s": n2,
                                             "est_wall_h": est, "solo_wall": t1 - t0})
        B.log(json.dumps({"max_eval": round(max(walls.values()), 1), "sum_process_s": round(per3), "est_h": round(est, 2)}))
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(p0)) as ex:
        p0res = dict(zip([x[0] for x in p0], ex.map(lambda x: E.simulate(f"e4b_P0_{x[0]}", x[1], [x[2]], B.SEEDS,
                                                                         timeout=3600), p0)))
    # ---- summaries
    runs_rows, table, chk = [], [], {"check1": {}, "check2": {}, "check3": {}, "check4": collections.Counter(),
                                     "check7": {}}
    by_name = {c["name"]: c for c in cfgs}
    with open(os.path.join(E.OUT, "fcfs", "fcfs_table.csv"), encoding="utf-8") as f:
        fc = {(r["group"], r["hw"], r["chunk"], r["order"], r["stagger"]): r for r in csv.DictReader(f)}
    with open(os.path.join(E.OUT, "diag", "hop_chunk.json"), encoding="utf-8") as f:
        hcj = json.load(f)["results"]
    for c in cfgs:
        src = c["name"] if not c["alias_of_hw"] else c["name"].replace(f"_{c['hw']}_", f"_{c['alias_of_hw']}_")
        Tc, rel, n = c["T"], c["release"], c["n"]
        fixed = {"".join(map(str, p)): f"{src}_fixed_{''.join(map(str, p))}" for p in itertools.permutations(range(n))}
        sjf = "".join(map(str, sorted(range(n), key=lambda i: (Tc[i], i))))
        fcfs = "".join(map(str, sorted(range(n), key=lambda i: (rel[i], i))))
        for k in KS:
            d = {i: rel[i] + k * Tc[i] for i in Tc}
            edf = "".join(map(str, sorted(range(n), key=lambda i: (d[i], i))))
            pol = {"P1": f"{src}_P1", "P2": f"{src}_P1" if c["order"] == "d0" else fixed[fcfs], "P3": fixed[sjf],
                   "P4": fixed[edf], "P6": f"{src}_P6_k{k}", "P7": f"{src}_P7_k{k}"}
            pol.update({f"fixed_{o}": v for o, v in fixed.items()})
            stat = {}
            for pn, rn in pol.items():
                rs = res[rn]["runs"]
                L = [B.lhat(r, Tc, d) for r in rs]
                S = [B.smax(r, Tc) for r in rs]
                M = [max(r["C"].values()) for r in rs]
                V = [sum(1 for i in Tc if r["C"][i] > d[i]) for r in rs]
                stat[pn] = {"Lhat": E._mean(L), "Lhat_range": max(L) - min(L), "S_max": E._mean(S),
                            "S_range": max(S) - min(S), "makespan": E._mean(M), "violations": E._mean(V),
                            "stretch": {i: E._mean([(r["C"][i] - r["release"][i]) / Tc[i] for r in rs]) for i in Tc},
                            "run": rn}
                for r, l, s_, m, v in zip(rs, L, S, M, V):
                    runs_rows.append({"group": c["group"], "hw": c["hw"], "chunk": c["chunk"] or "none", "order": c["order"],
                                      "k": k, "policy": pn, "run": rn, "seed": r["seed"],
                                      **{f"r_{i}": rel[i] for i in Tc}, **{f"C_{i}": r["C"][i] for i in Tc},
                                      "Lhat_max": l, "S_max": s_, "makespan": m, "violations": v, "wall": r["wall"]})
            best = min((p for p in stat if p.startswith("fixed_")), key=lambda p: stat[p]["Lhat"])
            stat["P5"] = dict(stat[best], member=best)
            for pn, s in stat.items():
                table.append({"config": c["name"], "group": c["group"], "hw": c["hw"], "chunk": c["chunk"] or "none",
                              "order": c["order"], "k": k, "policy": pn, "member": s.get("member", ""),
                              "Lhat_max": s["Lhat"], "Lhat_range": s["Lhat_range"], "S_max": s["S_max"],
                              "S_range": s["S_range"], "makespan": s["makespan"], "violations": s["violations"],
                              **{f"stretch_{i}": v for i, v in s["stretch"].items()}, "alias_of_hw": c["alias_of_hw"] or "",
                              "T": json.dumps(Tc), "release": json.dumps(rel)})
            # self-check 3 (criterion (a))
            r6 = res[pol["P6"]]["runs"]
            sl = {x: sum(r["slack_log"][x] for r in r6) for x in ["ticks", "rank_changes", "ticks_not_ref"]}
            sh = {x: sum(r["shadow"].get(x, 0) for r in r6) for x in ["n", "differ", "differ_stream"]}
            ent = {"slack_log": sl, "shadow": sh, "S_P6": stat["P6"]["S_max"], "S_sjf": stat["P3"]["S_max"]}
            if sl["ticks_not_ref"] == 0:
                ent["pass"] = sh["differ"] == 0 and abs(stat["P6"]["S_max"] / stat["P3"]["S_max"] - 1) <= 0.04
            chk["check3"][f"{c['name']}_k{k}"] = ent
        if not c["alias_of_hw"]:
            for rn in [f"{src}_P1"] + list(fixed.values()) + [f"{src}_P{p}_k{k}" for p in (6, 7) for k in KS]:
                for r in res[rn]["runs"]:
                    for x in ("n_issues", "n_multi", "n_mismatch"):
                        chk["check4"][x] += r["audit"][x]
            w1 = E._mean([r["wall"] for r in res[f"{src}_P1"]["runs"]])
            chk["check7"][c["name"]] = {f"P{p}_k{k}": E._mean([r["wall"] for r in res[f"{src}_P{p}_k{k}"]["runs"]]) / w1
                                        for p in (6, 7) for k in KS}
        # self-check 1 and 2
        s1 = E._mean([B.smax(r, Tc) for r in res[f"{src}_P1"]["runs"]])
        hopn = {"d0": f"{c['group']}{'' if c['hw'] == 'nofault' else '_' + c['hw']}_d0",
                "short_first": f"{c['group']}{'' if c['hw'] == 'nofault' else '_' + c['hw']}_d0.5"}.get(c["order"])
        if c["chunk"] is None and hopn and os.path.exists(os.path.join(E.OUT, "hop", f"{hopn}_hop.json")):
            with open(os.path.join(E.OUT, "hop", f"{hopn}_hop.json"), encoding="utf-8") as f:
                hj = json.load(f)
            chk["check1"][c["name"]] = {"new": s1, "e4a": hj["hop_default_S_max"], "rel": s1 / hj["hop_default_S_max"] - 1,
                                        "e4a_release": hj["release"], "release": rel, "strict": c["order"] == "d0"}
            if c["order"] == "d0":
                for o in fixed:
                    v = hj["per_variant"][f"prio_{o}"]["S_max_hopT"]
                    s = E._mean([B.smax(r, Tc) for r in res[fixed[o]]["runs"]])
                    chk["check2"][f"{c['name']}_{o}"] = {"new": s, "e4a": v, "rel": s / v - 1, "src": "hop_grid"}
        if c["chunk"] and c["order"] == "d0" and f"{c['hw']}_256d2d" in hcj:
            h = hcj[f"{c['hw']}_256d2d"]
            chk["check1"][c["name"]] = {"new": s1, "e4a": h["default"]["S_max"], "rel": s1 / h["default"]["S_max"] - 1,
                                        "strict": True}
            s = E._mean([B.smax(r, Tc) for r in res[fixed["01"]]["runs"]])
            chk["check2"][f"{c['name']}_01"] = {"new": s, "e4a": h["prio_01"]["S_max"], "rel": s / h["prio_01"]["S_max"] - 1,
                                               "src": "hop_chunk"}
        key = (c["group"], c["hw"], c["chunk"] or "none", c["order"], "0.5")
        if c["order"] != "d0" and n == 2 and key in fc:
            r = fc[key]
            for o, col in (("01", "S_short_first"), ("10", "S_long_first")):
                s = E._mean([B.smax(x, Tc) for x in res[fixed[o]]["runs"]])
                ev = float(r[col])
                src_ = "fcfs_table"
                if (c["name"], f"prio_{o}") in e4a_ext:
                    ev = E._mean([B.smax(x, Tc) for x in e4a_ext[(c["name"], f"prio_{o}")]])
                    src_ = "fcfs sims seeds 1-3 + e4b_out/e4a_ext seeds 4-6"
                chk["check2"][f"{c['name']}_{o}"] = {"new": s, "e4a": ev, "rel": s / ev - 1, "n_seeds": len(res[fixed[o]]["runs"]),
                                                    "src": src_, "e4a_Delta": int(r["Delta"]), "Delta": c["D"]}
    # P0
    p0s = {}
    for g, keys, var in p0:
        c = by_name[f"{g}_nofault_d0"]
        rs = p0res[g]["runs"]
        p0s[g] = {"S_max": E._mean([max((max(e["f"] for t, e in r["cap"]["events"].items() if t[1] - 10 == i)) / c["T"][i]
                                        for i in c["T"]) for r in rs]),
                  "makespan": E._mean([r["cap"]["result"][0] for r in rs]), "eval_wall": [r["wall"] for r in rs]}
    B.write_csv("setting1_runs.csv", runs_rows)
    B.write_csv("setting1_table.csv", table)
    stop = []
    if any(abs(v["rel"]) > 0.04 for v in chk["check1"].values() if v["strict"]):
        stop.append("check 1 > 4%")
    if any(abs(v["rel"]) > 0.04 for v in chk["check2"].values()):
        stop.append("check 2 > 4%")
    if any(v.get("pass") is False for v in chk["check3"].values()):
        stop.append("check 3")
    if chk["check4"]["n_mismatch"] or chk["check4"]["n_issues"] < 1000:
        stop.append("check 4")
    if any(x > 10 for v in chk["check7"].values() for x in v.values()):
        stop.append("handoff 6: DSTU > 10x P1")
    out = {"T": {f"{k}|{cz}": v for (k, cz), v in T.items()}, "checks": {**chk, "check4": dict(chk["check4"]), "check5": chk5},
           "P0": p0s, "configs": [{kk: v for kk, v in c.items()} for c in cfgs], "stop": stop,
           "walls": {"solo": t1 - t0, "runs": t2 - t1, "total": time.time() - t0}, "seeds": B.SEEDS, "command": sys.argv,
           "artifact_commit": E._artifact_commit(), "aliases": "P2/P3/P4 = runs of the fixed order their key induces"}
    B.write_json("setting1.json", out)
    B.log(json.dumps({"configs": len(cfgs), "processes": len(items), "wall_s": round(time.time() - t0),
                      "check1_max": round(max(abs(v["rel"]) for v in chk["check1"].values() if v["strict"]), 4),
                      "check2_max": round(max(abs(v["rel"]) for v in chk["check2"].values()), 4),
                      "check3_applicable": sum(1 for v in chk["check3"].values() if "pass" in v),
                      "check4": dict(chk["check4"]), "stop": stop}))
    if stop:
        raise B.E4bError("; ".join(stop))


if __name__ == "__main__":
    try:
        main()
    except (B.E4bError, E.E4Error) as exc:
        print(f"[e4b] STOP: {exc}", flush=True)
        sys.exit(2)
