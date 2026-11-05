"""Cold smoke test of EIR (handoff section 9), in its own folder, few steps and seeds.

  python smoke.py [folder]        default out/_smoke
Covers: node fault (FFN F1), D2D link (FFN F4), PE only (MHA F6), DDR port (FFN F8), plus the A3h start (FFN F4).
"""
import collections
import json
import os
import shutil
import subprocess
import sys
import time

os.environ["EIR_BACKEND_SEEDS"] = os.environ.get("EIR_BACKEND_SEEDS", "2")
os.environ["CPCM_SA_STEPS"] = "5"       # only the FULL-A3h job reads it
import eir  # noqa: E402

CASES = [("e3", "ffn", "F1"), ("e3", "ffn", "F4"), ("e3", "mha", "F6"), ("e3", "ffn", "F8"), ("a3h", "ffn", "F4")]
STEPS, SEEDS = 8, [123]
LONG = (("e3", "ffn", "F4"), 60)       # one longer L2-Local run so that every branch of the proposal is drawn


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(eir.HERE, "out", "_smoke"))
    if os.path.isdir(out):
        shutil.rmtree(out)
    res, t0 = {}, time.time()
    run = lambda jobs, workers=8: eir.run_phases([jobs], out, workers)
    bad = run([eir.job_pre(out, s, o) for s, o in sorted({(c[0], c[1]) for c in CASES})])
    bad += run([eir.job_l1(out, *c, checks=True) for c in CASES] + [eir.job_full_e3(out, "ffn", "F6")])
    reps = [eir.job_rep(out, *c, m, st, sd, checks=True) for c in CASES for m in eir.METHODS for st in (0, STEPS) for sd in SEEDS]
    reps += [eir.job_rep(out, *LONG[0], "L2L", LONG[1], 123, checks=True),
             eir.job_rep(out, *LONG[0], "L2G", LONG[1], 123, checks=True)]
    full = eir.job_full(out, "ffn14", "F4", "A3h", 0, 123)
    bad += run(reps + [full])
    print("failed:", bad)
    L1 = {c: eir.read_sink(eir.job_l1(out, *c)["sink"]) for c in CASES}
    R = {j["name"]: (j, eir.read_sink(j["sink"])) for j in reps if os.path.exists(j["sink"])}

    # 1 exact revert
    rev = collections.Counter()
    for d in L1.values():
        for k, v in d["L1_run"]["check_revert"].items():
            rev["L1_" + k] += v
    for _, d in R.values():
        for k, v in d["search"]["check_revert"].items():
            rev["L2_" + k] += v
    res["1_exact_revert"] = {"counts": dict(rev), "pass": rev["L1_n"] > 0 and rev["L2_n"] > 0 and not any(
        v for k, v in rev.items() if "mismatch" in k)}
    # 2 nothing on a failed part
    v2 = sum(len(d[x]["fault_violations"]) for d in L1.values() for x in ("R0", "L1")) + \
        sum(len(d["record"]["fault_violations"]) for _, d in R.values())
    res["2_no_fault_use"] = {"violations": v2, "states_checked": 2 * len(L1) + len(R), "pass": v2 == 0}
    # 3 local set: every proposal audited in the child; the final state re-checked here against the L1 output
    aud, acts, final_bad = collections.Counter(), collections.Counter(), 0
    illegal = 0
    for j, d in R.values():
        if j["method"] != "L2L":
            continue
        aud.update(d["search"]["local_audit"])
        illegal += d["search"]["n_illegal"]
        for r in d["search"]["rows"]:
            acts[(r["action"], r["targeted"])] += 1
        base = eir.read_sink(j["l1_sink"])
        ref, now = base["L1"]["beha_loc"], d["record"]["beha_loc"]
        moved = {t for t in ref if ref[t] != now[t]}
        final_bad += not (moved <= d["search"]["local_behas"] and all(tuple(now[t][:-1]) in d["search"]["local_nodes"] for t in moved))
    res["3_local"] = {"audit": dict(aud), "illegal_rebuild_undone": illegal, "final_states_outside": final_bad,
                      "actions": {str(k): v for k, v in acts.items()},
                      "pass": aud["n"] > 0 and aud["beha_not_movable"] == 0 and aud["beha_dest_outside"] == 0 and final_bad == 0}
    res["3_local"]["coverage"] = {j["name"]: d["search"]["local"] for j, d in R.values() if j["method"] == "L2L" and j["steps"] == 0}
    # 4 s = 0
    s0 = {}
    for j, d in R.values():
        if j["steps"] == 0:
            base = eir.read_sink(j["l1_sink"])
            s0[j["name"]] = {"T_equal": d["record"]["T"] == base["L1"]["T"], "routes_equal": d["record"]["routes"] == base["L1"]["routes"],
                             "state_equal": d["s0_state_equal"], "beha_equal": d["record"]["beha_loc"] == base["L1"]["beha_loc"]}
    res["4_s0"] = {"n": len(s0), "not_equal": {k: v for k, v in s0.items() if not all(v.values())},
                   "pass": len(s0) > 0 and all(all(v.values()) for v in s0.values())}
    # 5 metrics and timing
    d = R[eir.rep_name(*CASES[1], "L2G", STEPS, 123)][1]
    keys = {"moved_behas_share", "ddr_read_moved_bytes", "unique_block_bytes"}
    ok = all(keys <= set(x["record"]["migration_vs_start"]) and keys <= set(x["record"]["migration_vs_R0"]) for _, x in R.values())
    ok = ok and all(set(x["timing"]) >= {"t_R0", "t_L1", "t_calibration", "t_L2_search", "t_finalize", "t_decision", "t_backend_offline"}
                    for _, x in R.values())
    res["5_outputs"] = {"example": {"timing": d["timing"], "migration_vs_start": d["record"]["migration_vs_start"],
                                    "migration_vs_R0": d["record"]["migration_vs_R0"], "cpcm_calls": d["cpcm_calls"], "T": d["record"]["T"],
                                    "backend": d["record"]["backend"], "Tmax": d["search"]["Tmax"]}, "pass": ok}
    # 6 resume
    j = eir.job_rep(out, *CASES[1], "W", 30, 1)
    spec_path = os.path.join(out, "_tmp", "resume.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(j, f)
    os.makedirs(os.path.dirname(j["sink"]), exist_ok=True)
    p = subprocess.Popen([sys.executable, os.path.join(eir.HERE, "eir.py"), "child", spec_path], cwd=eir.HERE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)
    p.kill()
    p.wait()
    no_file = not os.path.exists(j["sink"])
    with open(j["sink"], "wb") as f:      # as a broken disconnect leaves it
        f.write(b"\x1f\x8b broken")
    r1 = eir.run_job(j, out)
    with open(j["sink"], "rb") as f:
        blob = f.read()
    with open(j["sink"], "wb") as f:
        f.write(blob[:len(blob) // 2])    # truncated gzip
    half_rejected = not eir.sink_ok(j["sink"], j)
    r2 = eir.run_job(j, out)
    r3 = eir.run_job(j, out)
    res["6_resume"] = {"killed_child_left_no_sink": no_file, "broken_rerun": r1[1], "half_rejected": half_rejected,
                       "truncated_rerun": r2[1], "then": r3[1],
                       "pass": no_file and r1[1] == "ok" and half_rejected and r2[1] == "ok" and r3[1] == "cached"}
    res["failed"] = bad
    res["wall_s"] = time.time() - t0
    res["PASS"] = not bad and all(res[k]["pass"] for k in res if k[0].isdigit())
    with open(os.path.join(out, "smoke.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: (v if k[0] != "3" else {x: y for x, y in v.items() if x != "coverage"}) for k, v in res.items()},
                     indent=1, default=str))
    print("SMOKE PASSED" if res["PASS"] else "SMOKE FAILED")


if __name__ == "__main__":
    main()
