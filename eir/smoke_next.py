"""Regression and cold smoke of the next stage's switches (EIR_NEXT_HANDOFF.md sections 1, 3, 4.1). Isolated folder; reads the
pre / l1 files of out/main, writes nothing there.

  python smoke_next.py [folder]     default out/_smoke_next

Regression (no new switch): reruns of runs already in out/main with the current code. T, every step of the search and the
local-set audit must equal the stored run; the sink gets only the keys it had (plus final_snap, added before).
Cold smoke (s = 20, checks on, 1 backend seed): L2Lsync, L2Impact, L2Gsync at FFN, MHA F4a, plus a reload of each final
state. Checks: entry sync keeps T (search raises otherwise) and leaves no stale list; exact revert restores
generated_split_location too (it is in the fingerprint with sync on); no stale list at the end; no stale rebound; behaviours
only move inside the set; K cores; set equal across seeds; fault violations 0.
"""
import json
import os
import shutil
import sys

os.environ["EIR_BACKEND_SEEDS"] = "1"
import eir  # noqa: E402

MAIN = os.path.join(eir.HERE, "out", "main")
SEED = 123


def clean(rows):
    return [dict(r) for r in rows]


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(eir.HERE, "out", "_smoke_next"))
    if os.path.isdir(out):
        shutil.rmtree(out)
    res = {"regression": {}, "new": {}}
    # ---- regression
    base = [eir.job_rep(MAIN, "a3h", o, "F4a", m, st, SEED, checks=True) for o, m, st in
            (("ffn", "L2L", 100), ("ffn", "L2G", 100), ("mha", "L2L", 30), ("mha", "L2G", 30), ("ffn", "W", 30))]
    base.append(eir.job_ctrl(MAIN, "a3h", "ffn", 30, SEED, checks=True))
    jobs = [dict(j, name="reg_" + j["name"], sink=os.path.join(out, "reg", j["name"] + ".pkl.gz")) for j in base]
    bad = eir.run_phases([jobs], out, 6)
    ok = not bad
    for j, r in zip(base, jobs):
        new, old = eir.read_sink(r["sink"]), eir.read_sink(j["sink"])
        sn, so = new["search"], old["search"]
        x = {"T": new["record"]["T"], "T_main": old["record"]["T"], "T_equal": new["record"]["T"] == old["record"]["T"], "rows_equal": clean(sn["rows"]) == clean(so["rows"]),
             "best_equal": (sn["best_E"], sn["best_T"], sn["best_step"]) == (so["best_E"], so["best_T"], so["best_step"]), "n_illegal_equal": sn["n_illegal"] == so["n_illegal"],
             "audit_equal": sn["local_audit"] == so["local_audit"], "beha_equal": new["record"]["beha_loc"] == old["record"]["beha_loc"],
             "routes_equal": new["record"]["routes"] == old["record"]["routes"], "no_new_keys": set(sn) == set(so), "revert_mismatch": sum(sn["check_revert"][k] for k in ("E_mismatch", "T_mismatch", "state_mismatch"))}
        x["PASS"] = all(x[k] for k in ("T_equal", "rows_equal", "best_equal", "n_illegal_equal", "audit_equal", "beha_equal", "routes_equal", "no_new_keys")) and x["revert_mismatch"] == 0
        ok = ok and x["PASS"]
        res["regression"][j["name"]] = x
    # ---- cold smoke of the new methods
    pre_w = {o: dict(eir.job_pre(out, "a3h", o), store_w=True) for o in ("ffn", "mha")}
    bad = eir.run_phases([list(pre_w.values())], out, 2)
    ok = ok and not bad
    new_jobs = []
    for o in ("ffn", "mha"):
        for m in ("L2Lsync", "L2Impact", "L2Gsync"):
            for sd in (SEED, 1) if m == "L2Impact" else (SEED,):
                j = eir.job_rep(MAIN, "a3h", o, "F4a", m, 20, sd, checks=True)
                new_jobs.append(dict(j, name="new_" + j["name"], sink=os.path.join(out, "new", j["name"] + ".pkl.gz"), pre_w_sink=pre_w[o]["sink"]))
    bad = eir.run_phases([new_jobs], out, 8)
    ok = ok and not bad
    reloads = [{"kind": "reload", "name": "reload_" + j["name"], "start": j["start"], "op": j["op"], "fault": j["fault"], "from_sink": j["sink"],
                "sink": os.path.join(out, "reload", j["name"] + ".pkl.gz")} for j in new_jobs]
    bad = eir.run_phases([reloads], out, 8)
    ok = ok and not bad
    sets = {}
    for j, rl in zip(new_jobs, reloads):
        d, dr = eir.read_sink(j["sink"]), eir.read_sink(rl["sink"])
        s = d["search"]
        la = s["local_audit"] or {}
        w = s["read_watch"]
        x = {"T_L1": d["T_L1_check"], "T_entry_sync": s["sync_entry"]["T"], "sync_entry": s["sync_entry"], "stale_final": s["stale_final"], "read_watch": w,
             "revert_mismatch": sum(s["check_revert"][k] for k in ("E_mismatch", "T_mismatch", "state_mismatch")), "revert_n": s["check_revert"]["n"],
             "n_illegal": s["n_illegal"], "cal_illegal": s["cal_illegal"], "audit": la, "violations": len(d["record"]["fault_violations"]),
             "local": s["local"], "accepted": sum(r["accepted"] for r in s["rows"]), "T_final": d["record"]["T"], "reload_T_equal": dr["T_equal"], "reload_fp_equal": dr.get("fp_equal")}
        x["PASS"] = (s["sync_entry"]["T"] == d["T_L1_check"] and s["sync_entry"]["stale_after"] == 0 and s["stale_final"] == 0 and w.get("rebound_stale", 0) == 0
                     and x["revert_mismatch"] == 0 and la.get("beha_not_movable", 0) == 0 and la.get("beha_dest_outside", 0) == 0 and x["violations"] == 0 and dr["T_equal"] and dr.get("fp_equal") is True)
        if j["method"] == "L2Impact":
            im = s["impact"]
            x["impact"] = {k: im[k] for k in ("K", "nodes", "n_cand", "n_g1", "n_g2", "chosen_g1", "chosen_g2", "overlap_geo", "tags_changed", "tags_missing_in_graph", "n_events", "n_unaligned_W", "n_changed", "n_behas_around")}
            x["PASS"] = x["PASS"] and len(im["nodes"]) == im["K"] and s["local"]["n_cores"] == im["K"]
            prev = sets.setdefault(j["op"], im["nodes"])
            x["same_set_as_other_seed"] = prev == im["nodes"]
            x["PASS"] = x["PASS"] and x["same_set_as_other_seed"]
        ok = ok and x["PASS"]
        res["new"][j["name"]] = x
    res["PASS"] = ok
    with open(os.path.join(out, "smoke_next.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    print("SMOKE_NEXT PASSED" if ok else "SMOKE_NEXT FAILED")


if __name__ == "__main__":
    main()
