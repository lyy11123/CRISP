"""Build e2e/w1_rerun/dist/w1_eir_bundle.tar.gz for the W1 EIR notebooks (W1_COLAB_GUIDE.md), then check it in a clean room.
Run only when the code is committed and local_ref.json exists (`w1_eir.py run prep` + `localref` on this machine).

  python e2e/w1_rerun/build_w1_bundle.py

Contents: cpcm, cpcm_sa, e1_fidelity, e2_attribution, e3_warmstart, eir (py and _inject), e2e/w1_rerun/w1_eir.py, local_ref.json,
eir_inputs/ (the W1 A3h start, E3's MHA start), eir_faults.json, the W1 artifact work copy (src, utils, tool, GPT decode py + cfg),
the artifact commit. No results.
Clean-room check (as e2e/colab/check_bundle.py): unpack into an empty temporary folder, run with `python -I` and without PYTHONPATH
`w1_eir.py smoke` (every job kind, short) and `w1_eir.py probe` (the W1 A3h start's pre vs local_ref.json); a failure renames the
bundle to .FAILED.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.dirname(HERE)
ROOT = os.path.dirname(E2E)
DIST = os.path.join(HERE, "dist")
WORK = os.path.join(HERE, "work", "artifact")
CODE_DIRS = ["cpcm", "cpcm_sa", "e1_fidelity", "e2_attribution", "e3_warmstart", "eir"]
MODELS = ["intra_mapping_gpt_decode"]


def skip(path):
    parts = path.replace("\\", "/").split("/")
    return "__pycache__" in parts or path.endswith(".pyc")


def add_text(tar, name, text):
    data = text.encode("utf-8")
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
    tar.addfile(ti, io.BytesIO(data))


def build(out):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--"] + CODE_DIRS + ["e2e/w1_rerun/*.py"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
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
        for name in ("w1_eir.py", "local_ref.json", "eir_faults.json"):
            tar.add(os.path.join(HERE, name), f"e2e/w1_rerun/{name}")
        for name in sorted(os.listdir(os.path.join(HERE, "eir_inputs"))):
            if name.startswith(("a3h_mha", "e3_mha")):
                tar.add(os.path.join(HERE, "eir_inputs", name), f"e2e/w1_rerun/eir_inputs/{name}")
        tar.add(os.path.join(ROOT, "third_party", "isca2026-busybarn-artifact.COMMIT"), "third_party/isca2026-busybarn-artifact.COMMIT")
        for sub in ["src", "utils", "tool"]:
            for dp, _, fs in os.walk(os.path.join(WORK, sub)):
                for f in fs:
                    p = os.path.join(dp, f)
                    if not skip(p):
                        tar.add(p, ("e2e/w1_rerun/work/artifact/" + os.path.relpath(p, WORK)).replace("\\", "/"))
        for model in MODELS:
            for sub in ["py", "cfg"]:
                for dp, _, fs in os.walk(os.path.join(WORK, "models", model, sub)):
                    for f in fs:
                        p = os.path.join(dp, f)
                        if not skip(p):
                            tar.add(p, ("e2e/w1_rerun/work/artifact/" + os.path.relpath(p, WORK)).replace("\\", "/"))
        ref = json.load(open(os.path.join(HERE, "local_ref.json"), encoding="utf-8"))
        add_text(tar, "BUNDLE_INFO.json", json.dumps({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": head, "uncommitted_code": dirty,
                                                       "local_ref_built": ref["built"], "local_ref_jobs": len(ref["jobs"])}, indent=1))
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB, git {head[:7]}{' (UNCOMMITTED CODE: ' + dirty + ')' if dirty else ''}", flush=True)


def clean_env():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "EIR_BACKEND_SEEDS", "CPCM_SA_STEPS")}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    return env


def check(bundle, keep=False):
    tmp = tempfile.mkdtemp(prefix="w1_bundle_check_")
    ok = True
    try:
        with tarfile.open(bundle, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        script = os.path.join(tmp, "e2e", "w1_rerun", "w1_eir.py")
        out = os.path.join(tmp, "check_out")
        for cmd in ("smoke", "probe"):
            p = subprocess.run([sys.executable, "-I", script, cmd, "--out", out], cwd=tmp, capture_output=True, text=True, env=clean_env())
            want = "SMOKE PASSED" if cmd == "smoke" else "PROBE PASSED"
            good = p.returncode == 0 and want in p.stdout
            print(f"[check] {cmd}: {'ok' if good else 'FAILED'}", flush=True)
            print("\n".join(x for x in p.stdout.splitlines() if x.startswith("[w1]") or "PASSED" in x or "FAILED" in x), flush=True)
            if not good:
                ok = False
                print(p.stdout[-3000:], p.stderr[-3000:], flush=True)
    finally:
        if keep:
            print(f"[check] kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("[check] BUNDLE CHECK PASSED" if ok else "[check] BUNDLE CHECK FAILED", flush=True)
    return ok


def main():
    os.makedirs(DIST, exist_ok=True)
    out = os.path.join(DIST, "w1_eir_bundle.tar.gz")
    if len(sys.argv) > 1 and sys.argv[1] == "--check-only":
        sys.exit(0 if check(out, "--keep" in sys.argv) else 1)
    build(out)
    if not check(out, "--keep" in sys.argv):
        os.replace(out, out + ".FAILED")
        raise SystemExit(f"bundle check failed; renamed to {out}.FAILED")


if __name__ == "__main__":
    main()
