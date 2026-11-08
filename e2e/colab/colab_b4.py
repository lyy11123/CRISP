"""B4 validation on Colab (e2e/colab/B4_COLAB_GUIDE.md). Runs from the unpacked bundle; local results come only as e2e/colab/local_ref.json.

  python e2e/colab/colab_b4.py probe  --out OUT                  environment, one T-hat timed and compared with this machine
  python e2e/colab/colab_b4.py smoke                             one 15-step run, 2 backend seeds, in /content/smoke
  python e2e/colab/colab_b4.py run    --out OUT --workers 4      the B4 subset (SUBSET), then compare with local_ref.json
  python e2e/colab/colab_b4.py scan   --out OUT                  workers 4, 7, 8: copies of one run in parallel; wall clock and memory
  python e2e/colab/colab_b4.py report --out OUT                  OUT/b4_colab_report.json
  python e2e/colab/colab_b4.py pack   --out OUT                  OUT/results_b4.tar.gz

Comparisons: T-hat (T_pre, T_bc, T_sel), the SA trajectory (energy of every step, accepted or not) and the selected step must be bit
for bit those of this machine; the backend is compared by its 10-seed mean (it depends on object addresses, known since E0).
"""
import argparse
import gzip
import hashlib
import json
import os
import pickle
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.dirname(HERE)
ROOT = os.path.dirname(E2E)
# (cfg, arm, g, x): decode LN and projection, A1 / A3h, the first 2 trial configs; W1 prefill MHA A1 g0 l0
SUBSET = [("dc_ln", "A1", 0, 0), ("dc_ln", "A1", 0, 1), ("dc_ln", "A3h", 0, 123), ("dc_ln", "A3h", 0, 1),
          ("dc_proj", "A1", 0, 0), ("dc_proj", "A1", 0, 1), ("dc_proj", "A3h", 0, 123), ("dc_proj", "A3h", 0, 1),
          ("pf_mha_w1", "A1", 0, 0)]
SCAN_RUN = ("dc_proj", "A3h", 0, 123)
SCAN_WORKERS = (4, 7, 8)
PROBE_RUN = ("pf_mha_w1", "A1", 0, 0)       # its snapshot is in the bundle (e2e/colab/probe/)


def env(sub, **extra):
    return dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", E2E_B4="1", E2E_SUB=sub, CPCM_SA_STAGE="s2", **extra)


def name_of(t):
    return f"{t[0]}_{t[1]}_g{t[2]}_{'l' if t[1] == 'A1' else 's'}{t[3]}"


def link_out(out):
    """e2e/out -> the Drive folder, so finished runs survive a disconnect."""
    os.makedirs(out, exist_ok=True)
    link = os.path.join(E2E, "out")
    if os.path.islink(link):
        os.unlink(link)
    elif os.path.isdir(link):
        if os.listdir(link):
            raise SystemExit(f"{link} is a real, non-empty folder; remove it first")
        os.rmdir(link)
    os.symlink(out, link, target_is_directory=True)


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 2 ** 20


def run_many(sub, todo, workers, extra=None):
    """Each (cfg, arm, g, x) in its own process through e2e_pilot.run_one; returns walls and the lowest available memory."""
    code = ("import sys,json;sys.path.insert(0,r'%s');import e2e_pilot as P;t=json.loads(sys.argv[1]);"
            "print(json.dumps(P.run_one(t[0],t[1],t[2],t[3],False)))" % E2E)
    mins, stop = [mem_available_gb()], threading.Event()

    def watch():
        while not stop.wait(2):
            mins.append(mem_available_gb())
    threading.Thread(target=watch, daemon=True).start()

    def one(t):
        t0 = time.time()
        p = subprocess.run([sys.executable, "-c", code, json.dumps(list(t))], cwd=E2E, env=env(sub, **(extra or {})), capture_output=True, text=True)
        ok = p.returncode == 0 and '"failed"' not in p.stdout
        print(f"[colab] {sub} {name_of(t)}: {'ok' if ok else 'FAILED'} {time.time() - t0:.0f} s", flush=True)
        if not ok:
            print(p.stdout[-2000:], p.stderr[-2000:])
        return dict(run=name_of(t), ok=ok, wall_s=time.time() - t0)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, todo))
    stop.set()
    return dict(runs=res, total_wall_s=time.time() - t0, min_available_gb=min(mins), start_available_gb=mins[0], workers=workers)


def load_run(sub, name):
    with gzip.open(os.path.join(E2E, "out", sub, "runs", name + ".pkl.gz"), "rb") as f:
        return pickle.load(f)


def fingerprint(d):
    s = d["selected"]
    py = lambda v: None if v is None else (float(v) if not isinstance(v, bool) else bool(v))   # numpy scalars print differently across versions
    rows = [(py(r.get("E")), py(r.get("accepted")), py(r.get("T"))) for r in d["rows"]]
    iv = lambda v: None if v is None else int(v)
    return dict(T_pre=iv(s["T_pre"]), T_bc=iv(s["T_bc"]), T_sel=iv(s["T_sel"]), step=iv(s["step"]), own_cost=float(s["own_cost"]),
                rows=hashlib.sha256(repr(rows).encode()).hexdigest(), backend_mean=statistics.mean(v[0] for v in s["backend"].values()),
                backend=sorted(int(v[0]) for v in s["backend"].values()))


def compare(sub, names):
    ref = json.load(open(os.path.join(HERE, "local_ref.json"), encoding="utf-8"))["runs"]
    out = []
    for n in names:
        p = os.path.join(E2E, "out", sub, "runs", n + ".pkl.gz")
        if not os.path.exists(p):
            out.append(dict(run=n, missing=True))
            continue
        f, r = fingerprint(load_run(sub, n)), ref[n]
        exact = {k: f[k] == r[k] for k in ("T_pre", "T_bc", "T_sel", "step", "own_cost", "rows")}
        out.append(dict(run=n, exact=exact, all_exact=all(exact.values()), backend_mean=f["backend_mean"], backend_mean_local=r["backend_mean"],
                        backend_rel=f["backend_mean"] / r["backend_mean"] - 1, backend_same_values=f["backend"] == r["backend"]))
        print(f"[colab] {n}: T-hat and trajectory {'IDENTICAL' if all(exact.values()) else 'DIFFER ' + str(exact)}; backend mean {f['backend_mean']:.0f} vs local {r['backend_mean']:.0f} ({100 * out[-1]['backend_rel']:+.2f}%)", flush=True)
    return out


def peaks(sub):
    d = os.path.join(E2E, "out", sub, "mem")
    return {f[:-5]: json.load(open(os.path.join(d, f)))["peak_wset"] / 2 ** 30 for f in os.listdir(d)} if os.path.isdir(d) else {}


def probe(out):
    import numpy
    import networkx
    info = dict(nproc=os.cpu_count(), python=sys.version, platform=platform.platform(), numpy=numpy.__version__, networkx=networkx.__version__,
                mem_available_gb=mem_available_gb(), bundle=json.load(open(os.path.join(ROOT, "BUNDLE_INFO.json"), encoding="utf-8")))
    try:
        import simanneal
        info["simanneal"] = getattr(simanneal, "__version__", "?")
    except ImportError:
        info["simanneal"] = None
    with open("/proc/meminfo") as f:
        info["mem_total_gb"] = int(f.readline().split()[1]) / 2 ** 20
    # one T-hat of the W1 prefill MHA probe snapshot, as e2e_b4.ties_child (deterministic and seed-1 random tie)
    code = r'''
import sys, json, time, random
sys.path.insert(0, r"%s")
import e2e_b as EB, e2e_pilot as P
C = P.C
spec = json.load(open(sys.argv[1]))
ctx, beha, data, events, hp = EB.load_state(spec)
added = EB.broadcast(ctx, beha, data, events, hp)
dropped = len(P.w1_drop(events, added))
t0 = time.perf_counter(); g = EB.compiled(events, hp); t1 = time.perf_counter()
det = C.cpcm.simulate(g, "bb").T; t2 = time.perf_counter()
tie = C.cpcm.simulate(g, "bb", rng=random.Random(1)).T; t3 = time.perf_counter()
print(json.dumps(dict(det=det, tie1=tie, dropped=dropped, compile_s=t1 - t0, det_s=t2 - t1, tie_s=t3 - t2)))
''' % E2E
    pd = os.path.join(HERE, "probe")
    spec = dict(cfg=PROBE_RUN[0], run=os.path.join(pd, "run.pkl.gz"), snap=os.path.join(pd, "snap.pkl.gz"))
    sp = "/content/probe_spec.json"
    json.dump(spec, open(sp, "w"))
    p = subprocess.run([sys.executable, "-c", code, sp], cwd=E2E, env=env("probe"), capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit("[colab] PROBE FAILED")
    t = json.loads(p.stdout.strip().splitlines()[-1])
    ref = json.load(open(os.path.join(HERE, "local_ref.json"), encoding="utf-8"))["probe"]
    t.update(det_local=ref["det"], tie1_local=ref["tie1"], det_identical=t["det"] == ref["det"], tie1_identical=t["tie1"] == ref["tie1"],
             local_compile_s=ref["compile_s"], local_det_s=ref["det_s"])
    info["T_hat"] = t
    os.makedirs(out, exist_ok=True)
    json.dump(info, open(os.path.join(out, "probe.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "bundle"}, indent=1))
    print("[colab] PROBE PASSED" if t["det_identical"] and t["tie1_identical"] else "[colab] PROBE: T-hat DIFFERS from this machine's")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "smoke", "run", "scan", "report", "pack"])
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "smoke":
        if os.path.islink(os.path.join(E2E, "out")):
            os.unlink(os.path.join(E2E, "out"))
        os.makedirs(os.path.join(E2E, "out"), exist_ok=True)
        r = run_many("smoke", [SUBSET[2]], 1, dict(CPCM_SA_STEPS="15", E2E_BACKEND_SEEDS="2"))
        ok = r["runs"][0]["ok"]
        shutil.rmtree(os.path.join(E2E, "out"))      # smoke results are not kept; run / scan link e2e/out to Drive
        print(f"[colab] smoke {r['total_wall_s']:.0f} s, min available {r['min_available_gb']:.1f} GB")
        print("[colab] SMOKE PASSED" if ok else "[colab] SMOKE FAILED")
        return
    if not a.out:
        raise SystemExit("--out is required")
    if a.cmd == "probe":
        return probe(a.out)
    if a.cmd == "pack":
        path = os.path.join(a.out, "results_b4.tar.gz")
        with tarfile.open(path, "w:gz") as tar:
            for f in os.listdir(a.out):
                if f != "results_b4.tar.gz":
                    tar.add(os.path.join(a.out, f), f)
        print(f"[colab] {path}: {os.path.getsize(path) / 1e6:.1f} MB")
        return
    link_out(a.out)
    if a.cmd == "run":
        todo = [t for t in SUBSET if not os.path.exists(os.path.join(E2E, "out", "b4", "runs", name_of(t) + ".pkl.gz"))]
        r = run_many("b4", todo, a.workers)
        r["compare"] = compare("b4", [name_of(t) for t in SUBSET])
        r["peak_gb"] = peaks("b4")
        json.dump(r, open(os.path.join(a.out, "run_subset.json"), "w"), indent=1)
        bad = [c["run"] for c in r["compare"] if not c.get("all_exact")]
        print("[colab] RUN: all T-hat identical" if not bad else f"[colab] RUN: DIFFER or missing: {bad}")
    elif a.cmd == "scan":
        res = []
        for w in SCAN_WORKERS:
            sub = f"scan_w{w}"
            shutil.rmtree(os.path.join(E2E, "out", sub), ignore_errors=True)
            # w copies of one run, each in its own folder (scan_w<w>_<i>), so nothing is cached between them
            code_runs = []
            for i in range(w):
                code_runs.append((f"{sub}_{i}", SCAN_RUN))
            mins, stop = [mem_available_gb()], threading.Event()

            def watch():
                while not stop.wait(2):
                    mins.append(mem_available_gb())
            threading.Thread(target=watch, daemon=True).start()
            t0 = time.time()
            with ThreadPoolExecutor(w) as ex:
                rr = list(ex.map(lambda st: run_many(st[0], [st[1]], 1), code_runs))
            stop.set()
            walls = [x["runs"][0]["wall_s"] for x in rr]
            fps = [fingerprint(load_run(s, name_of(SCAN_RUN))) for s, _ in code_runs]
            pk = [v for s, _ in code_runs for v in peaks(s).values()]
            ref = json.load(open(os.path.join(HERE, "local_ref.json"), encoding="utf-8"))["runs"][name_of(SCAN_RUN)]
            rec = dict(workers=w, total_wall_s=time.time() - t0, run_wall_median_s=statistics.median(walls), run_wall_max_s=max(walls),
                       min_available_gb=min(mins), start_available_gb=mins[0], peak_gb_max=max(pk) if pk else None,
                       all_ok=all(x["runs"][0]["ok"] for x in rr), T_bc_identical=all(f["T_bc"] == ref["T_bc"] for f in fps), rows_identical=all(f["rows"] == ref["rows"] for f in fps))
            res.append(rec)
            print(f"[colab] scan {json.dumps(rec)}", flush=True)
            json.dump(res, open(os.path.join(a.out, "scan.json"), "w"), indent=1)
    elif a.cmd == "report":
        rep = {}
        for f in ("probe.json", "run_subset.json", "scan.json"):
            p = os.path.join(a.out, f)
            rep[f[:-5]] = json.load(open(p)) if os.path.exists(p) else None
        json.dump(rep, open(os.path.join(a.out, "b4_colab_report.json"), "w"), indent=1)
        t = (rep["probe"] or {}).get("T_hat", {})
        cmp_ = (rep["run_subset"] or {}).get("compare", [])
        print(f"[colab] probe: nproc {(rep['probe'] or {}).get('nproc')}, T-hat identical {t.get('det_identical')}/{t.get('tie1_identical')}, "
              f"compile {t.get('compile_s', 0):.2f} s, det {t.get('det_s', 0):.2f} s (local {t.get('local_det_s', 0):.2f} s)")
        print(f"[colab] subset: {sum(1 for c in cmp_ if c.get('all_exact'))}/{len(cmp_)} identical; backend mean rel. "
              f"{[round(100 * c['backend_rel'], 2) for c in cmp_ if 'backend_rel' in c]} %")
        for s in rep["scan"] or []:
            print(f"[colab] workers {s['workers']}: run median {s['run_wall_median_s']:.0f} s, max {s['run_wall_max_s']:.0f} s, "
                  f"min available {s['min_available_gb']:.1f} GB, peak {s['peak_gb_max']:.2f} GB, identical {s['T_bc_identical'] and s['rows_identical']}")


if __name__ == "__main__":
    main()
