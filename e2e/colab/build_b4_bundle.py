"""Build e2e/colab/dist/b4_bundle.tar.gz for the B4 validation notebook (B4_COLAB_GUIDE.md). Run it only when the code is frozen and
the local runs of colab_b4.SUBSET exist (out/b4/runs). Reads the working tree; writes only e2e/colab/dist, e2e/colab/local_ref.json and
e2e/colab/probe/.

  python e2e/colab/build_b4_bundle.py

Contents: cpcm, cpcm_sa, e1_fidelity, e3_warmstart (code and _inject), e2e/*.py, e2e/colab/colab_b4.py, the e2e artifact work copy
(src, utils, tool; GPT prefill and decode py + cfg), the artifact commit, local_ref.json (this machine's T-hat, trajectory digest and
backend of every SUBSET run and of the probe) and the probe snapshot. No other results.
"""
import gzip
import hashlib
import io
import json
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.dirname(HERE)
ROOT = os.path.dirname(E2E)
DIST = os.path.join(HERE, "dist")
WORK = os.path.join(E2E, "work", "artifact")
MODELS = ["intra_mapping_gpt_decode", "intra_mapping_gpt_prefill"]
CODE_DIRS = ["cpcm", "cpcm_sa", "e1_fidelity", "e2_attribution", "e3_warmstart"]   # e3_warmstart imports e2_attribution in the final evaluation
sys.path.insert(0, HERE)
import colab_b4 as CB  # noqa: E402


def skip(path):
    parts = path.replace("\\", "/").split("/")
    return "__pycache__" in parts or path.endswith((".pyc", ".pkl", ".gz"))


def fingerprint(d):
    return CB.fingerprint(d)


def local_ref():
    runs = {}
    for t in CB.SUBSET + [CB.SCAN_RUN]:
        n = CB.name_of(t)
        p = os.path.join(E2E, "out", "b4", "runs", n + ".pkl.gz")
        if not os.path.exists(p):
            raise SystemExit(f"local run missing: {p} (build after the local B4 runs)")
        with gzip.open(p, "rb") as f:
            runs[n] = fingerprint(pickle.load(f))
    # the probe: copy the snapshot, compute its T-hat here with the same code as on Colab
    pd = os.path.join(HERE, "probe")
    os.makedirs(pd, exist_ok=True)
    n = CB.name_of(CB.PROBE_RUN)
    shutil.copy(os.path.join(E2E, "out", "b4", "runs", n + ".pkl.gz"), os.path.join(pd, "run.pkl.gz"))
    shutil.copy(os.path.join(E2E, "out", "b4", "snaps", n + ".pkl.gz"), os.path.join(pd, "snap.pkl.gz"))
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
    sp = os.path.join(pd, "spec_local.json")
    json.dump(dict(cfg=CB.PROBE_RUN[0], run=os.path.join(pd, "run.pkl.gz"), snap=os.path.join(pd, "snap.pkl.gz")), open(sp, "w"))
    p = subprocess.run([sys.executable, "-c", code, sp], cwd=E2E, env=dict(os.environ, E2E_B4="1", E2E_SUB="probe", CPCM_SA_STAGE="s2", PYTHONIOENCODING="utf-8"),
                       capture_output=True, text=True)
    os.remove(sp)
    if p.returncode:
        raise SystemExit(p.stdout[-2000:] + p.stderr[-2000:])
    probe = json.loads(p.stdout.strip().splitlines()[-1])
    if probe["det"] != runs[n]["T_bc"]:
        raise SystemExit(f"probe T-hat {probe['det']} differs from the run's T_bc {runs[n]['T_bc']}")
    ref = dict(runs=runs, probe=probe, built=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(ref, open(os.path.join(HERE, "local_ref.json"), "w"), indent=1)
    return ref


def add_text(tar, name, text):
    data = text.encode("utf-8")
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
    tar.addfile(ti, io.BytesIO(data))


def main():
    ref = local_ref()
    os.makedirs(DIST, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "cpcm", "cpcm_sa", "e1_fidelity", "e2_attribution", "e3_warmstart", "e2e/*.py", "e2e/colab/*.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = os.path.join(DIST, "b4_bundle.tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        for d in CODE_DIRS:
            for name in sorted(os.listdir(os.path.join(ROOT, d))):
                p = os.path.join(ROOT, d, name)
                if name.endswith(".py") and os.path.isfile(p):
                    tar.add(p, f"{d}/{name}")
            inj = os.path.join(ROOT, d, "_inject")
            if os.path.isdir(inj):
                for name in sorted(os.listdir(inj)):
                    if not skip(name):
                        tar.add(os.path.join(inj, name), f"{d}/_inject/{name}")
        for name in sorted(os.listdir(E2E)):
            if name.endswith(".py"):
                tar.add(os.path.join(E2E, name), f"e2e/{name}")
        for name in ("colab_b4.py", "local_ref.json"):
            tar.add(os.path.join(HERE, name), f"e2e/colab/{name}")
        for name in ("run.pkl.gz", "snap.pkl.gz"):
            tar.add(os.path.join(HERE, "probe", name), f"e2e/colab/probe/{name}")
        tar.add(os.path.join(ROOT, "third_party", "isca2026-busybarn-artifact.COMMIT"), "third_party/isca2026-busybarn-artifact.COMMIT")
        for sub in ["src", "utils", "tool"]:
            for dp, _, fs in os.walk(os.path.join(WORK, sub)):
                for f in fs:
                    p = os.path.join(dp, f)
                    if not skip(p):
                        tar.add(p, ("e2e/work/artifact/" + os.path.relpath(p, WORK)).replace("\\", "/"))
        for model in MODELS:
            for sub in ["py", "cfg"]:
                for dp, _, fs in os.walk(os.path.join(WORK, "models", model, sub)):
                    for f in fs:
                        p = os.path.join(dp, f)
                        if not skip(p):
                            tar.add(p, ("e2e/work/artifact/" + os.path.relpath(p, WORK)).replace("\\", "/"))
        sha = {n: hashlib.sha256(open(os.path.join(E2E, n), "rb").read()).hexdigest()[:16] for n in ("e2e_pilot.py", "e2e_b4.py", "e2e_b.py")}
        add_text(tar, "BUNDLE_INFO.json", json.dumps({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": head, "uncommitted_code": dirty,
                                                       "sha256_16": sha, "subset": [CB.name_of(t) for t in CB.SUBSET], "local_ref_built": ref["built"]}, indent=1))
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB, git {head[:7]}{' (UNCOMMITTED CODE: ' + dirty + ')' if dirty else ''}")


if __name__ == "__main__":
    main()
