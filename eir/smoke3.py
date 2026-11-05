"""Small smoke test of search()'s online checkpoints and of the final state saved in the sinks. Isolated folder; reads the
pre / l1 files of out/main, writes nothing there.

  python smoke3.py [folder]     default out/_smoke3

a: no checkpoints; b: same seed with checkpoints 10, 20, 30 in the s = 30 schedule. Checks: a == b (T, state fingerprint,
every step of the search), best_T and best_E never rise over the checkpoints and end at the final best, a == the run
already in out/main (the default path did not change), and the saved final state reloaded in a new process gives T.
"""
import gzip
import json
import os
import pickle
import shutil
import sys

os.environ["EIR_BACKEND_SEEDS"] = "1"
import eir  # noqa: E402

MAIN = os.path.join(eir.HERE, "out", "main")
CK = [10, 20, 30]
STEPS, SEED = 30, 123


def clean_rows(res):
    return [{k: v for k, v in r.items()} for r in res["rows"]]


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(eir.HERE, "out", "_smoke3"))
    if os.path.isdir(out):
        shutil.rmtree(out)
    base = []
    for op, fid, m in (("ffn", "F4", "L2G"), ("ffn", "F4", "L2L"), ("ffn", "F4", "W"), ("mha", "F4", "L2G"), ("mha", "F4", "L2L")):
        base.append(eir.job_rep(MAIN, "a3h", op, fid, m, STEPS, SEED, checks=True))
    base.append(eir.job_ctrl(MAIN, "a3h", "ffn", STEPS, SEED, checks=True))
    jobs_a, jobs_b = [], []
    for j in base:
        a = dict(j, name="a_" + j["name"], sink=os.path.join(out, "a", j["name"] + ".pkl.gz"))
        b = dict(j, name="b_" + j["name"], sink=os.path.join(out, "b", j["name"] + ".pkl.gz"), checkpoints=CK)
        jobs_a.append(a)
        jobs_b.append(b)
    bad = eir.run_phases([jobs_a + jobs_b], out, 6)
    reload_jobs = [{"kind": "reload", "name": "reload_" + j["name"], "start": j["start"], "op": j["op"], "fault": j.get("fault"),
                    "from_sink": a["sink"], "sink": os.path.join(out, "reload", j["name"] + ".pkl.gz")} for j, a in zip(base, jobs_a)]
    bad += eir.run_phases([reload_jobs], out, 6)
    res = {"failed": bad, "runs": {}}
    ok = not bad
    for j, a, b, r in zip(base, jobs_a, jobs_b, reload_jobs):
        da, db, dr, old = eir.read_sink(a["sink"]), eir.read_sink(b["sink"]), eir.read_sink(r["sink"]), eir.read_sink(j["sink"])
        sa, sb, so = da["search"], db["search"], old["search"]
        cks = sb["checkpoints"]
        Ts = [c["best_T"] for c in cks]
        Es = [c["best_E"] for c in cks]
        snap = pickle.dumps(da["final_snap"], protocol=pickle.HIGHEST_PROTOCOL)
        x = {"T_a": da["record"]["T"], "T_b": db["record"]["T"], "T_equal": da["record"]["T"] == db["record"]["T"],
             "fp_equal": da["final_fp"] == db["final_fp"], "rows_equal": clean_rows(sa) == clean_rows(sb), "best_step_equal": sa["best_step"] == sb["best_step"],
             "no_ck_key_when_not_asked": "checkpoints" not in sa,
             "ck": [{k: c[k] for k in ("step", "best_E", "best_T", "best_step", "t_search")} for c in cks],
             "E_monotone": all(p >= q for p, q in zip(Es, Es[1:])), "T_monotone": None if Ts[0] is None else all(p >= q for p, q in zip(Ts, Ts[1:])),
             "ck_last_is_final": cks[-1]["best_E"] == sb["best_E"] and cks[-1]["best_step"] == sb["best_step"],
             "same_as_main_T": da["record"]["T"] == old["record"]["T"], "same_as_main_rows": clean_rows(sa) == clean_rows(so), "T_main": old["record"]["T"],
             "reload_T_equal": dr["T_equal"], "reload_fp_equal": dr.get("fp_equal"),
             "revert_mismatch": sum(sa["check_revert"][k] for k in ("E_mismatch", "T_mismatch", "state_mismatch")),
             "bytes_main_sink": os.path.getsize(j["sink"]), "bytes_with_final_snap": os.path.getsize(a["sink"]), "bytes_with_final_snap_and_3_checkpoints": os.path.getsize(b["sink"]),
             "bytes_final_snap_raw": len(snap), "bytes_final_snap_gz": len(gzip.compress(snap))}
        x["PASS"] = bool(x["T_equal"] and x["fp_equal"] and x["rows_equal"] and x["best_step_equal"] and x["no_ck_key_when_not_asked"] and x["E_monotone"] and x["T_monotone"] is not False
                         and x["ck_last_is_final"] and x["same_as_main_T"] and x["same_as_main_rows"] and x["reload_T_equal"] and x["reload_fp_equal"] and x["revert_mismatch"] == 0)
        ok = ok and x["PASS"]
        res["runs"][j["name"]] = x
    res["PASS"] = ok
    with open(os.path.join(out, "smoke3.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    print("SMOKE3 PASSED" if ok else "SMOKE3 FAILED")


if __name__ == "__main__":
    main()
