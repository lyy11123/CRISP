"""EIR's E3 (BusyBarn) start under W1: the template's own winner under W1 is A1 g0 l1 (w1_variants.py, 18,458), user 2026-11-09
(E2E_HANDOFF.md section 4 item 8). Captured exactly as E3 captured its start (e3_warmstart.capture: the template runs unchanged in
a child process, the final mapping is read at its 2nd backend call), but in this folder's artifact work copy and into
out/e3cap/ (e3_warmstart/e3_out is never written).

Self-check first: the same capture of l4 must give E3's start (eir/inputs/e3_mha.pkl.gz) bit for bit; only then l1 is captured and
written to eir_inputs/e3_mha.pkl.gz (the l4 one is kept as eir_inputs/e3l4_mha.pkl.gz for the record). The SA trajectory and the
captured mapping do not depend on the final broadcast (the W1 filter acts on it only), so no filter is installed here.

  python e2e/w1_rerun/w1_e3start.py
"""
import gzip
import json
import os
import pickle
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "e3_warmstart"))
import e3_warmstart as E  # noqa: E402

E.WORK = os.path.join(HERE, "work", "artifact")
E.PY_DIR = os.path.join(E.WORK, "models", "intra_mapping_gpt_decode", "py")
E.OUT = os.path.join(HERE, "out", "e3cap")
TASK = "mha_sq2048_greedy0_lossratio{}_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0"
INPUTS = os.path.join(HERE, "eir_inputs")


def cap(l):
    t0 = time.time()
    p = E.capture(TASK.format(l), [1, 2, 3], 1)
    res = {"task": TASK.format(l), "template": p["last"]["result"], "seeds": {str(s): c["result"] for s, c, _ in p["reseeded"]},
           "wall_s": round(time.time() - t0, 1)}
    print(f"[e3start] l{l}: {json.dumps(res)}", flush=True)
    return p["mapping"], res


def main():
    os.makedirs(E.OUT, exist_ok=True)
    with gzip.open(os.path.join(ROOT, "eir", "inputs", "e3_mha.pkl.gz"), "rb") as f:
        e3 = pickle.load(f)
    m4, r4 = cap(4)
    same = {k: m4[k] == e3[k] for k in e3}
    print(f"[e3start] l4 capture vs E3's start: {same}", flush=True)
    if not all(same.values()):
        raise SystemExit("STOP: the l4 capture does not reproduce E3's start; l1 not captured")
    m1, r1 = cap(1)
    dst = os.path.join(INPUTS, "e3_mha.pkl.gz")
    if not os.path.exists(os.path.join(INPUTS, "e3l4_mha.pkl.gz")):
        shutil.copy(dst, os.path.join(INPUTS, "e3l4_mha.pkl.gz"))
    with gzip.open(dst, "wb") as f:
        pickle.dump(m1, f, protocol=pickle.HIGHEST_PROTOCOL)
    moved = sum(tuple(m1["beha_loc"][t]) != tuple(m4["beha_loc"][t]) for t in m4["beha_loc"])
    rep = {"l4": r4, "l4_same_as_e3_start": same, "l1": r1, "l1_vs_l4_moved_behas": moved, "n_behas": len(m4["beha_loc"]),
           "written": dst, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(rep, open(os.path.join(HERE, "e3start_w1.json"), "w"), indent=1)
    print(f"[e3start] l1 written to {dst}; {moved}/{len(m4['beha_loc'])} behaviours on another core than l4", flush=True)


if __name__ == "__main__":
    main()
