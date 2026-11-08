"""B4 batch (E2E_HANDOFF.md section 2, rules in PROGRESS.md "B4 第 0 步"). SA runs go through e2e_pilot (E2E_B4=1, E2E_SUB=b4); W1 is
the in-memory filter of e2e_pilot.install_w1 / w1_drop (B1's filter). This file adds the random-tie T-hat of every new mapping (B0's
evaluation, with the workload's broadcast) and the tables.

  E2E_B4=1 E2E_SUB=b4 python e2e_b4.py ties --jobs 8      out/b4/ties/*.json
  E2E_B4=1 E2E_SUB=b4 python e2e_b4.py report             out/b4/b4.json
"""
import gzip
import json
import os
import pickle
import random
import statistics
import sys
import time

os.environ.setdefault("E2E_B4", "1")
os.environ.setdefault("E2E_SUB", "b4")
import e2e_b as EB  # noqa: E402
import e2e_pilot as P  # noqa: E402

C = P.C
NEW_CFGS = ["pf_ln", "pf_proj", "dc_ln", "dc_proj", "pf_mha_w1", "dc_mha_w1"]
ARMS = ["A1", "A2", "A3h"]
SEEDS = list(range(1, 11))


def run_path(cfg, name):
    return os.path.join(C.OUT, "runs", name + ".pkl.gz"), os.path.join(C.OUT, "snaps", name + ".pkl.gz")


def ties_child(spec_path, out):
    spec = json.load(open(spec_path, encoding="utf-8"))
    ctx, beha, data, events, hp = EB.load_state(spec)
    added = EB.broadcast(ctx, beha, data, events, hp)
    dropped = len(P.w1_drop(events, added)) if spec["cfg"].endswith("_w1") else 0
    t0 = time.perf_counter()
    g = EB.compiled(events, hp)
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    det = C.cpcm.simulate(g, "bb").T
    t_det = time.perf_counter() - t0
    ties, wall = [], []
    for s in SEEDS:
        t0 = time.perf_counter()
        ties.append(C.cpcm.simulate(g, "bb", rng=random.Random(s)).T)
        wall.append(time.perf_counter() - t0)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(spec, det=det, ties=ties, dropped=dropped, t_compile=t_compile, t_det=t_det, t_tie_each=wall, n_events=g.n, peak_wset_gb=P.peak() / 2 ** 30), f)


def specs():
    out = []
    for cfg in NEW_CFGS:
        for arm in ARMS:
            for i, (g, x) in enumerate(C.variants(arm)):
                n = C.run_name(cfg, arm, g, x)
                run, snap = run_path(cfg, n)
                if not os.path.exists(run):
                    continue
                with gzip.open(run, "rb") as f:
                    sel = pickle.load(f)["selected"]
                out.append(dict(name=n, cfg=cfg, arm=arm, g=g, x=x, idx=i, run=run, snap=snap, mean=C.bmean(sel["backend"]), T_bc=sel["T_bc"]))
    return out


def ties(jobs):
    EB.__file__  # the driver launches this file's children
    sp = specs()
    meta = drive("ties", sp, jobs)
    print(json.dumps(meta))


def drive(tag, sp, jobs):
    # e2e_b.drive starts children of e2e_b.py; point it at this file for the ties child
    import subprocess
    orig = EB.subprocess.Popen

    def popen(args, **kw):
        if len(args) >= 3 and args[2] == "b4_ties_child":
            args = [args[0], os.path.abspath(__file__)] + args[2:]
        return orig(args, **kw)
    EB.subprocess.Popen = popen
    try:
        return EB.drive(tag, sp, jobs, "b4_ties_child")
    finally:
        EB.subprocess.Popen = orig


# ------------------------------------------------------------------ report
PILOT_OUT = os.path.join(P.HERE, "out")
B0_DIR = os.path.join(PILOT_OUT, "b0")
# (phase, op) -> cfg per workload; W0's MHA and FFN are the existing results (pilot / cpcm_sa s2), with B0's random-tie T-hat
CFG = {("prefill", "ln"): ("pf_ln", "pf_ln"), ("prefill", "mha"): ("pf_mha", "pf_mha_w1"), ("prefill", "proj"): ("pf_proj", "pf_proj"), ("prefill", "ffn"): ("pf_ffn", "pf_ffn"),
       ("decode", "ln"): ("dc_ln", "dc_ln"), ("decode", "mha"): ("mha", "dc_mha_w1"), ("decode", "proj"): ("dc_proj", "dc_proj"), ("decode", "ffn"): ("ffn", "ffn")}
COUNT = {"ln": 2, "mha": 8, "proj": 1, "ffn": 1}
OUTS = (128, 512, 2048)
LAYERS = 44
# artifact fig_endtoend/inter_layer.py, dojo (5x5, 2x2 blocks), zigzag placement: 7 groups, 6 boundaries, 1 hop each (computed with compute_inter_time)
INTER = {"prefill": {"artifact": 589944.0, "batch16": 9437304.0}, "decode": {"artifact": 408.0, "batch16": 4728.0}}


def cand_files(cfg, arm, g, x):
    n = C.run_name(cfg, arm, g, x)
    if cfg in NEW_CFGS:
        return n, os.path.join(C.OUT, "runs", n + ".pkl.gz"), os.path.join(C.OUT, "ties", n + ".json")
    base = PILOT_OUT if cfg.startswith("pf_") else P.DECODE_OUT
    return n, os.path.join(base, "runs", n + ".pkl.gz"), os.path.join(B0_DIR, n + ".json")


def pool(cfg, arm):
    rows, missing = [], []
    for i, (g, x) in enumerate(C.variants(arm)):
        n, rp, tp = cand_files(cfg, arm, g, x)
        if not (os.path.exists(rp) and os.path.exists(tp)):
            missing.append(n)
            continue
        with gzip.open(rp, "rb") as f:
            d = pickle.load(f)
        s = d["selected"]
        t = json.load(open(tp, encoding="utf-8"))
        bk = s["backend"]
        m = C.bmean(bk)
        lo, hi = C.spread(bk)
        tri = C.triple(bk)
        if t["det"] != s["T_bc"]:
            raise RuntimeError(f"{n}: det T-hat {t['det']} differs from the run's T_bc {s['T_bc']}")
        rows.append(dict(name=n, idx=i, mean=m, spread=(hi - lo) / m, triple=tri, comm_share=tri[2] / m, det=s["T_bc"], tie=statistics.mean(t["ties"]),
                         err_det=s["T_bc"] / m - 1, err_tie=statistics.mean(t["ties"]) / m - 1, sa_wall=d.get("sa_wall"), eval_wall=s.get("eval_wall")))
    return rows, missing


def pick(rows, sel):
    f = {"backend": lambda r: r["mean"], "det": lambda r: r["det"], "tie": lambda r: r["tie"]}[sel]
    return min(rows, key=lambda r: (f(r), r["idx"]))


def report():
    SEL = ("backend", "det", "tie")
    ops, missing = {}, {}
    for (ph, op), cfgs in CFG.items():
        for w, cfg in zip(("W0", "W1"), cfgs):
            for arm in ARMS:
                rows, miss = pool(cfg, arm)
                if miss:
                    missing[f"{cfg} {arm}"] = len(miss)
                if not rows:
                    continue
                e = dict(phase=ph, op=op, workload=w, cfg=cfg, arm=arm, n=len(rows),
                         err_det_median=statistics.median(r["err_det"] for r in rows), err_det_min=min(r["err_det"] for r in rows), err_det_max=max(r["err_det"] for r in rows),
                         err_tie_median=statistics.median(r["err_tie"] for r in rows), err_tie_min=min(r["err_tie"] for r in rows), err_tie_max=max(r["err_tie"] for r in rows),
                         spread_median=statistics.median(r["spread"] for r in rows), spread_max=max(r["spread"] for r in rows))
                for sel in SEL:
                    p = pick(rows, sel)
                    e[sel] = dict(run=p["name"], mean=p["mean"], triple=p["triple"], comm_share=p["comm_share"], gap_to_best=None)
                b = e["backend"]["mean"]
                for sel in SEL:
                    e[sel]["gap_to_best"] = e[sel]["mean"] / b - 1
                ops[(ph, op, w, arm)] = e
    for (ph, op, w, arm), e in ops.items():
        a1 = ops.get((ph, op, w, "A1"))
        for sel in SEL:
            e[sel]["vs_A1"] = e[sel]["mean"] / a1[sel]["mean"] - 1 if a1 else None
    # assembly: one layer (T_block), then the 44-layer secondary figure
    asm = []
    for w in ("W0", "W1"):
        for sel in SEL:
            for arm in ARMS:
                blk = {}
                for ph in ("prefill", "decode"):
                    parts = {op: ops.get((ph, op, w, arm)) for op in COUNT}
                    if any(v is None for v in parts.values()):
                        blk = None
                        break
                    t = sum(COUNT[op] * parts[op][sel]["mean"] for op in COUNT)
                    blk[ph] = dict(asis=t, fixed=t - (parts["ln"][sel]["mean"] if ph == "decode" else 0))
                if blk is None:
                    continue
                for ln in ("asis", "fixed"):
                    ttft, tpot = blk["prefill"][ln], blk["decode"][ln]
                    r = dict(workload=w, selector=sel, arm=arm, ln=ln, TTFT=ttft, TPOT=tpot, **{f"req{o}": ttft + o * tpot for o in OUTS})
                    for inter in ("artifact", "batch16"):
                        mt, md = LAYERS * ttft + INTER["prefill"][inter], LAYERS * tpot + INTER["decode"][inter]
                        r[f"model_{inter}"] = dict(TTFT=mt, TPOT=md, **{f"req{o}": mt + o * md for o in OUTS})
                    asm.append(r)
    for r in asm:
        a1 = next(x for x in asm if (x["workload"], x["selector"], x["ln"], x["arm"]) == (r["workload"], r["selector"], r["ln"], "A1"))
        r["vs_A1"] = {k: r[k] / a1[k] - 1 for k in ("TTFT", "TPOT", *[f"req{o}" for o in OUTS])}
    out = dict(ops=[dict(v) for v in ops.values()], assembly=asm, missing=missing, inter=INTER, layers=LAYERS)
    with open(os.path.join(C.OUT, "b4.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("missing", missing)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "b4_ties_child":
        ties_child(*sys.argv[2:4])
    elif cmd == "ties":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--jobs", type=int, default=8)
        ties(ap.parse_args().jobs)
