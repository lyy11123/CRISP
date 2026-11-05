"""Small smoke test of the two supplementary configurations (handoff section 12), in its own folder.

  python smoke2.py [folder]     default out/_smoke2
Uses the real F4a, F5a of faults.json (run `eir.py select` first). F4a / F5a: pre + l1 + W, L2G, L2L at s = 0 and 8.
Control: the five starts' `ctrl` jobs at s = 0 and 8, also `report_batch.py ctrl`.
"""
import json
import os
import shutil
import subprocess
import sys

os.environ["EIR_BACKEND_SEEDS"] = "2"
import eir  # noqa: E402


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(eir.HERE, "out", "_smoke2"))
    if os.path.isdir(out):
        shutil.rmtree(out)
    combos = eir.a_combos()
    bad = eir.run_phases([[eir.job_pre(out, s, o) for s, o in eir.CTRL_STARTS]], out, 4)
    bad += eir.run_phases([[eir.job_l1(out, *c, checks=True) for c in combos]], out, 4)
    reps = [eir.job_rep(out, *c, m, st, 123, checks=True) for c in combos for m in eir.METHODS for st in (0, 8)]
    ctrl = [eir.job_ctrl(out, s, o, st, 123, checks=True) for s, o in eir.CTRL_STARTS for st in (0, 8)]
    bad += eir.run_phases([reps + ctrl], out, 6)
    res = {"failed": bad, "combos": combos}
    rev = viol = 0
    n_rev = 0
    s0 = []
    for j in reps + ctrl:
        d = eir.read_sink(j["sink"])
        c = d["search"]["check_revert"]
        n_rev += c["n"]
        rev += int(c["E_mismatch"]) + c["T_mismatch"] + c["state_mismatch"]
        viol += len(d["record"]["fault_violations"])
        if j["steps"] == 0:
            ref = eir.read_sink(j["l1_sink"])["L1"] if j["kind"] == "rep" else eir.read_sink(j["pre_sink"])["record"]
            s0.append(d["record"]["T"] == ref["T"] and d["record"]["routes"] == ref["routes"] and d["s0_state_equal"])
    res.update(revert_checks=n_rev, revert_mismatch=rev, fault_violations=viol, s0_runs=len(s0), s0_equal=sum(s0))
    aud = [eir.read_sink(j["sink"])["search"]["local_audit"] for j in reps if j["method"] == "L2L"]
    res["local_outside"] = sum(a.get("beha_not_movable", 0) + a.get("beha_dest_outside", 0) for a in aud)
    res["local_coverage"] = {j["name"]: eir.read_sink(j["sink"])["search"]["local"]["beha_share"] for j in reps if j["method"] == "L2L" and j["steps"] == 0}
    res["link_stats"] = {eir.combo_name(*c): eir.read_sink(eir.job_l1(out, *c)["sink"])["link_stats"] for c in combos}
    rc = subprocess.call([sys.executable, os.path.join(eir.HERE, "report_batch.py"), "ctrl", "--out", out], cwd=eir.HERE)
    res["report_ctrl_exit"] = rc
    res["PASS"] = not bad and n_rev > 0 and rev == 0 and viol == 0 and all(s0) and res["local_outside"] == 0 and rc == 0
    with open(os.path.join(out, "smoke2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    print("SMOKE2 PASSED" if res["PASS"] else "SMOKE2 FAILED")


if __name__ == "__main__":
    main()
