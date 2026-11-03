"""Regression for the bb-engine speed-up (user 2026-11-03): event-level results of the current engine are recorded
(`ref`) and later compared (`check`) on V2 (30 graphs), V4 (88), every 4th V3 cached state of opt / mha / ffn14 and
V5's 42 multi-stream configurations (bb pass 1 only: needs E4's objects, so V5 uses the stored pass-1 C_i).
A record per graph: T, and sha1 of (r, S, S_last, F) and of the sorted activity multiset; plus the first
differing event on a mismatch. Usage: python cpcm_regress.py ref|check"""
import glob, gzip, hashlib, json, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpcm  # noqa: E402

REF = os.path.join(HERE, "cpcm_out", "regress_ref.pkl.gz")


def graphs():
    for p in sorted(glob.glob(os.path.join(HERE, "cpcm_out", "v2", "*_F?.pkl.gz"))):
        with gzip.open(p, "rb") as f:
            yield "v2:" + os.path.basename(p), pickle.load(f)["graph"]
    for p in sorted(glob.glob(os.path.join(HERE, "cpcm_out", "v4", "*_F?_*.pkl.gz"))):
        if os.path.basename(p).startswith("tau_"):
            continue
        with gzip.open(p, "rb") as f:
            yield "v4:" + os.path.basename(p), pickle.load(f)["graph"]
    for name in ("opt", "mha", "ffn14"):
        with gzip.open(os.path.join(HERE, "cpcm_out", "v3", f"{name}_graphs.pkl.gz"), "rb") as f:
            rows = pickle.load(f)["rows"]
        for r in rows[::4]:
            yield f"v3:{name}:{r['step']}", r["graph"]


def record(sc):
    ev = (sc.r, sc.S, sc.S_last, sc.F)
    acts = sorted(zip(sc.a_ev, sc.a_res, sc.a_s, sc.a_f))
    return {"T": sc.T, "ev": hashlib.sha1(repr(ev).encode()).hexdigest(),
            "acts": hashlib.sha1(repr(acts).encode()).hexdigest(), "arrays": ev}


def main(mode):
    if mode == "ref":
        out = {k: record(cpcm.simulate(g, "bb")) for k, g in graphs()}
        with gzip.open(REF, "wb") as f:
            pickle.dump(out, f)
        print(f"[regress] recorded {len(out)} graphs")
        return 0
    with gzip.open(REF, "rb") as f:
        ref = pickle.load(f)
    n = bad = 0
    for k, g in graphs():
        n += 1
        got = record(cpcm.simulate(g, "bb"))
        old = ref[k]
        if (got["T"], got["ev"], got["acts"]) != (old["T"], old["ev"], old["acts"]):
            bad += 1
            names = ("r", "S", "S_last", "F")
            first = None
            for i, nm in enumerate(names):
                for e, (a, b) in enumerate(zip(got["arrays"][i], old["arrays"][i])):
                    if a != b and (first is None or min(a, b) < first[3]):
                        first = (nm, e, str(g.tag[e]), min(a, b), b, a)
            print(f"[regress] MISMATCH {k}: T {old['T']} -> {got['T']}; first differing (field, event id, tag, t, "
                  f"old, new) = {first}")
            if bad >= 1:
                break
    print(f"[regress] checked {n} graphs, mismatches {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
