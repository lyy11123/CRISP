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


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "b4_ties_child":
        ties_child(*sys.argv[2:4])
    elif cmd == "ties":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--jobs", type=int, default=8)
        ties(ap.parse_args().jobs)
