"""Speed-up regression on DSTU rows (E2E_HANDOFF.md section 3): settings 1-3 (dstu.py children: P1 / fixed / C-S / C4 with shadow),
the block-overhead study (dstu_oh.py: o != 0, phi_res selective chunking) and CPCM solo (hop, with the 40 random-tie passes).
Specs are rebuilt in memory from the existing cached results (no dstu run is launched, no dstu file is written); each object runs
once with the original and once with the new implementation in this process, under the tag hash (see speedup_check.py), and the
complete outputs are compared bit for bit (wall-clock fields removed). Also reported: whether the new result equals the cached one
(that one was made under the address hash, so it need not).

  python speedup_dstu.py list                      out/speedup/dstu_items.json
  python speedup_dstu.py one <index> <out json>     one object
"""
import copy
import gzip
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "dstu"))
os.chdir(os.path.join(ROOT, "dstu"))

import dstu as D  # noqa: E402
import dstu2 as S  # noqa: E402
import dstu_oh as OH  # noqa: E402
from dstu import B, E, cpcm  # noqa: E402

OUT = os.path.join(HERE, "out", "speedup")
TMP = os.path.join(OUT, "dstu_tmp")
WALL_KEYS = {"wall", "process_wall", "times", "wall_s"}


def no_launch(*a, **k):
    raise RuntimeError("speedup_dstu: a dstu result is not cached; nothing is launched")


def quiet():
    """Spec building only: cached results are read, nothing runs, nothing is written into dstu/out."""
    D.subprocess = S.subprocess = type("X", (), {"call": staticmethod(no_launch), "STDOUT": subprocess.STDOUT, "TimeoutExpired": subprocess.TimeoutExpired})
    D.write_csv = D.write_json = lambda *a, **k: None
    OH.wj = OH.wc = lambda *a, **k: None
    S.wj = getattr(S, "wj", None) and (lambda *a, **k: None)


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in WALL_KEYS}
    if isinstance(x, (list, tuple)):
        return type(x)(strip(v) for v in x)
    return x


def digest(x):
    return hashlib.sha256(pickle.dumps(x, protocol=4)).hexdigest()


def build_items():
    quiet()
    items = []
    for n in (1, 2, 3):
        S1, plats, T, solo = D.solo_step(1, s3=n == 3)
        ideal = D.load_ideal(plats)
        cfgs = D.build_cfgs(n, plats, T, solo, ideal)
        cfgs = [c for c in cfgs if c["label"].get("draw") in (None, 0)]
        rep = dict(D.rep_items(cfgs))
        new = dict(D.new_items(cfgs, solo, ideal, which=["C-S"]))
        chunked = [c for c in cfgs if c["chunk"]]
        pick = []
        c0 = cfgs[0]
        pick.append(f"rep_{c0['name']}_P1")
        cf = next((c for c in chunked if c["hw"] != "nofault"), chunked[0] if chunked else cfgs[-1])
        pick.append(next(k for k in new if k.startswith(cf["name"] + "_CS")))
        c4 = [k for k in new if "_C4_" in k]
        pick.append(c4[0] if c4 else f"rep_{cfgs[-1]['name']}_fixed_" + D.order_of(cfgs[-1]["d"], cfgs[-1]["n"]))
        for k in pick:
            spec = rep.get(k) or new[k]
            items.append(dict(kind="dstu_child", setting=n, name=k, spec=dict(spec, seeds=spec["seeds"][:1]), cached=os.path.join(D.SIMS, k + ".pkl.gz")))
        s = cf["streams"][0]          # a stream of the same configuration as its faults
        cz = cf["chunk"]
        items.append(dict(kind="dstu_solo", setting=n, name=f"solo_{s}_{cz}_{cf['hw']}", spec={"stream": s, "chunk": D.H.CHUNKS[cz] if cz else None, "faults": cf["faults"]},
                          cached=os.path.join(D.SOLO, f"{s}_{cz or 'none'}.pkl.gz")))
    cf, solo, sc, plans = OH.prep(1)
    its, meta = OH.all_items(cf, solo, plans)
    its = dict(its)
    for nm in ("ohrun_F4_1000_all256_C-S_s1", "ohrun_F4_200_res1620_all_C4_s1", "ohrun_nofault_50_all64_P1_s1", "ohrun_F4_1000_res540_d2d_C4_s1"):
        if nm in its:
            items.append(dict(kind="oh-run", setting="overhead", name=nm, spec=its[nm], cached=os.path.join(S.SIMS if hasattr(S, "SIMS") else "", nm + ".pkl.gz")))
    hw = "F4" if "F4" in cf else next(iter(cf))
    c = cf[hw]
    items.append(dict(kind="oh-solo", setting="overhead", name=f"ohsolo_{hw}_1000_all256_0", spec={"faults": c["faults"], "stream": c["streams"][0], "plan": OH.SCHEMES["all256"], "o": 1000}, cached=None))
    rank = OH.rank_for(c, solo, hw, 200, "all256")
    items.append(dict(kind="oh-score", setting="overhead", name=f"ohscore_{hw}_200_all256", spec={"faults": c["faults"], "streams": c["streams"], "plan": OH.SCHEMES["all256"], "o": 200, "rank": rank,
                                                                                                   "release": c["release"], "T": c["T"], "d": c["d"]}, cached=None))
    return items


def run_item(it):
    k = it["kind"]
    spec = copy.deepcopy(it["spec"])
    if k in ("dstu_child", "dstu_solo"):
        os.makedirs(TMP, exist_ok=True)
        sp = os.path.join(TMP, f"{it['setting']}_{it['name']}.spec")      # setting in the name: settings 1 and 2 share some names
        sink = os.path.join(TMP, f"{it['setting']}_{it['name']}.sink")
        with open(sp, "wb") as f:
            pickle.dump(dict(spec, sink=sink), f, protocol=pickle.HIGHEST_PROTOCOL)
        (D.child if k == "dstu_child" else D.solo_child)(sp)
        with open(sink, "rb") as f:
            r = pickle.load(f)
        os.remove(sink)
        os.remove(sp)
        return r
    return OH.CHILDREN[k](spec)


def install(new):
    import e4b_backend
    import fast_cpcm
    import fast_hop
    if not hasattr(install, "orig"):
        install.orig = (e4b_backend.event_driver_e4b, cpcm.simulate)
    e4b_backend.event_driver_e4b = fast_hop.event_driver_fast if new else install.orig[0]
    cpcm.simulate = fast_cpcm.simulate if new else install.orig[1]


def one(idx, out):
    items = json.load(open(os.path.join(OUT, "dstu_items.json"), encoding="utf-8")) if False else pickle.load(open(os.path.join(OUT, "dstu_items.pkl"), "rb"))
    it = items[idx]
    E._artifact_path()
    from event_notation import event_notation
    event_notation.__hash__ = lambda self: hash(self.event_tag)
    res = {}
    for which in ("old", "new"):
        install(which == "new")
        t0 = time.perf_counter()
        r = run_item(it)
        res[which] = dict(wall=time.perf_counter() - t0, digest=digest(strip(r)), r=strip(r))
        print(f"[{it['name']}] {which} {res[which]['wall']:.1f} s", flush=True)
    same = res["old"]["digest"] == res["new"]["digest"]
    cached = None
    if it.get("cached") and os.path.exists(it["cached"]):
        with gzip.open(it["cached"], "rb") as f:
            c = pickle.load(f)
        cached = digest(strip(c)) == res["new"]["digest"]
    diff = None
    if not same:
        a, b = res["old"]["r"], res["new"]["r"]
        diff = [k for k in (a if isinstance(a, dict) else {}) if digest(a.get(k)) != digest(b.get(k))]
    rec = dict(index=idx, kind=it["kind"], setting=it["setting"], name=it["name"], identical=same, equals_cached=cached, diff=diff,
               old_wall_s=res["old"]["wall"], new_wall_s=res["new"]["wall"], digest=res["new"]["digest"])
    json.dump(rec, open(out, "w"), indent=1, default=str)
    print(json.dumps(rec, default=str), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "list":
        its = build_items()
        os.makedirs(OUT, exist_ok=True)
        pickle.dump(its, open(os.path.join(OUT, "dstu_items.pkl"), "wb"))
        json.dump([{k: v for k, v in i.items() if k != "spec"} for i in its], open(os.path.join(OUT, "dstu_items.json"), "w"), indent=1, default=str)
        for i, x in enumerate(its):
            print(i, x["kind"], x["setting"], x["name"])
    else:
        one(int(sys.argv[2]), os.path.join(HERE, sys.argv[3]) if not os.path.isabs(sys.argv[3]) else sys.argv[3])
