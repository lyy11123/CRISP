"""Run one supplementary experiment from the unpacked Colab bundle, with resume.

  python colab/colab_run.py run   --nb 1a --out /content/drive/MyDrive/CRISP_colab/1a --workers 7
  python colab/colab_run.py smoke --nb 1a
  python colab/colab_run.py report --nb 1a --out /content/drive/MyDrive/CRISP_colab/1a
  python colab/colab_run.py pack  --nb 1a --out /content/drive/MyDrive/CRISP_colab/1a

Rules are stage 2's (CPCM_SA_STAGE=s2): exact revert + T with the cached broadcast; A0 unchanged; 14 runs per arm;
10 backend seeds. A run counts as finished only if its file loads completely; anything else is removed and rerun.
"""
import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SA = os.path.join(ROOT, "cpcm_sa")
ARMS = ["A0", "A1", "A2", "A3h"]
NOTEBOOKS = {"1a": ["moe_f4"], "1b": ["moe_f5"], "2": ["mha_sq512", "mha_sq8192"]}
SMOKE_ARMS = {"1a": ["A1", "A3h"], "1b": ["A1", "A3h"], "2": ["A0", "A3h"]}


def env_for(out_root=None, **extra):
    e = dict(os.environ, CPCM_SA_STAGE="s2", PYTHONUNBUFFERED="1", **extra)
    if out_root:
        e["CPCM_SA_OUT_ROOT"] = out_root
    return e


def sa(cmd, configs, arms, workers, env, first=False):
    args = [sys.executable, "cpcm_sa.py", cmd, "--configs", *configs, "--arms", *arms, "--jobs", str(workers)]
    return subprocess.call(args + (["--first"] if first else []), cwd=SA, env=env)


def link_out(out):
    """cpcm_sa/out -> the Drive folder, so finished runs survive a disconnect."""
    os.makedirs(out, exist_ok=True)
    link = os.path.join(SA, "out")
    if os.path.islink(link):
        os.unlink(link)
    elif os.path.isdir(link):
        raise SystemExit(f"{link} is a real folder; remove it first")
    os.symlink(out, link, target_is_directory=True)


def validate(out):
    """Remove runs left half-written by a disconnect: gzip must be complete, the run finished, A0's template present."""
    sys.path.insert(0, SA)
    os.environ["CPCM_SA_STAGE"] = "s2"
    os.environ["CPCM_SA_OUT_ROOT"] = out
    import pickle
    import cpcm_sa as C
    C._artifact_path()
    runs = os.path.join(C.OUT, "runs")
    if not os.path.isdir(runs):
        print("[colab] no earlier runs")
        return
    good = bad = 0
    for name in sorted(os.listdir(runs)):
        p = os.path.join(runs, name)
        try:
            with gzip.open(p, "rb") as f:
                d = pickle.load(f)
            ok = "selected" in d and ("_A0_" not in name or "template_out" in d)
        except Exception as ex:
            ok, d = False, repr(ex)
        if ok:
            good += 1
        else:
            bad += 1
            os.remove(p)
            snap = os.path.join(C.OUT, "snaps", name)
            if os.path.exists(snap):
                os.remove(snap)
            print(f"[colab] removed unfinished run {name}")
    print(f"[colab] earlier runs: {good} kept, {bad} removed")


def report(nb, env):
    cfgs = NOTEBOOKS[nb]
    rc1 = subprocess.call([sys.executable, "check1_s2.py", *cfgs], cwd=SA, env=env)
    rc2 = sa("report2", cfgs, ARMS, 1, env)
    print(f"[colab] check1 exit {rc1}, report2 exit {rc2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "smoke", "report", "pack"])
    ap.add_argument("--nb", required=True, choices=sorted(NOTEBOOKS))
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--smoke-root", default="/content/smoke")
    a = ap.parse_args()
    cfgs = NOTEBOOKS[a.nb]
    t0 = time.time()
    if a.cmd == "smoke":
        # short and separate from the real results: 15 SA steps, 2 backend seeds, its own folder
        os.makedirs(a.smoke_root, exist_ok=True)
        env = env_for(a.smoke_root, CPCM_SA_STEPS="15", CPCM_SA_BACKEND_SEEDS="2")
        rc = sa("run", cfgs[:1], SMOKE_ARMS[a.nb], min(a.workers, 4), env, first=True)
        files = os.listdir(os.path.join(a.smoke_root, "s2", "runs"))
        print(f"[colab] smoke exit {rc}; {len(files)} run files; {time.time() - t0:.0f} s")
        print("[colab] SMOKE PASSED" if rc == 0 and len(files) == 2 * len(SMOKE_ARMS[a.nb]) else "[colab] SMOKE FAILED")
        return
    if not a.out:
        raise SystemExit("--out is required")
    if a.cmd == "pack":
        path = os.path.join(a.out, f"results_{a.nb}.tar.gz")
        with tarfile.open(path, "w:gz") as tar:
            tar.add(os.path.join(a.out, "s2"), "s2")
        print(f"[colab] {path}: {os.path.getsize(path) / 1e6:.1f} MB")
        return
    link_out(a.out)
    env = env_for()
    if a.cmd == "report":
        report(a.nb, env)
        return
    rc = subprocess.call([sys.executable, "-c",  # its own process: importing cpcm_sa fixes OUT at import time
                          "import sys;sys.path.insert(0,r'%s');import colab_run;colab_run.validate(r'%s')"
                          % (os.path.dirname(os.path.abspath(__file__)), a.out)], cwd=SA, env=env)
    if rc:
        raise SystemExit(f"validation of earlier runs failed: exit {rc}")
    rc = sa("run", cfgs, ARMS, a.workers, env)
    print(f"[colab] run exit {rc}, {(time.time() - t0) / 3600:.2f} h")
    report(a.nb, env)
    shutil.copy(os.path.join(ROOT, "BUNDLE_INFO.json"), a.out)


if __name__ == "__main__":
    main()
