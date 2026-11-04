"""Stage 1.5 self-checks (handoff 9.3) on the runs with checks on (first 2 variants of each config x arm): rejected
moves (E, T, state fingerprint recomputed vs recorded), check 2 (offline T of saved accepted states), check 4.
Usage: CPCM_SA_STAGE=b1 python check15.py mha ffn"""
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpcm_sa as C  # noqa: E402

out = []
tot = dict(rej=0, E=0, T=0, state=0, c2=0, c2_bad=0, c4=0, c4_off=0, c4_T=0)
for cfg in sys.argv[1:]:
    for arm in ("A1", "A2", "A3", "A3h"):
        for g, x in C.variants(arm)[:2]:
            d = C.load_run(cfg, arm, g, x)
            hp = C.platform(cfg)
            r = d["check_revert"]
            bad2 = sum(1 for c in d["check2"] if C.cpcm_sim(pickle.loads(c["state"]), hp)[1].T != c["T_rec"])
            row = dict(cfg=cfg, arm=arm, g=g, x=x, rejected=int(r["n"]), E_mismatch=int(r["E_mismatch"]),
                       T_mismatch=int(r["T_mismatch"]), state_mismatch=int(r["state_mismatch"]),
                       check2_n=len(d["check2"]), check2_mismatch=bad2, check4_n=len(d["check4"]),
                       check4_off_cp=sum(1 for c in d["check4"] if not c["on_fresh_cp"]),
                       check4_T_diff=sum(1 for c in d["check4"] if c["T_fresh"] != c["T_cached"]))
            out.append(row)
            print(row)
            for a, b in (("rej", "rejected"), ("E", "E_mismatch"), ("T", "T_mismatch"), ("state", "state_mismatch"),
                         ("c2", "check2_n"), ("c2_bad", "check2_mismatch"), ("c4", "check4_n"),
                         ("c4_off", "check4_off_cp"), ("c4_T", "check4_T_diff")):
                tot[a] += row[b]
print("TOTAL", tot)
with open(os.path.join(C.OUT, "check15.json"), "w", encoding="utf-8") as f:
    json.dump({"total": tot, "rows": out}, f, indent=1)
