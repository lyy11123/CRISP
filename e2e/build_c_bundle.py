"""Build e2e/c/dist/c_bundle.tar.gz for the C notebooks (C_COLAB_GUIDE.md), then check it in a clean room.
Run only when the code is committed and the local prep is finished (`c_eir.py run prep`, `localref`).

  python e2e/build_c_bundle.py

Contents: cpcm, cpcm_sa, e1_fidelity, e2_attribution, e3_warmstart, eir (py and _inject), eir/inputs and eir/faults.json (the regression's
EIR starts and faults), the EIR sinks the regression compares with (c_eir.REG), e2e/c_eir.py, e2e/e2e_pilot_peak.py, e2e/c/{faults,
starts,local_ref}.json, e2e/c/inputs, the C artifact work copy (src, utils, tool, GPT prefill and decode py + cfg), the artifact commit,
and this machine's prep sinks (c_prep/prep/..., unpacked into OUT by the notebook: every repair reads its start's pre and l1).
Clean-room check: unpack into an empty temporary folder, `python -I` without PYTHONPATH: `c_eir.py probe` and `c_eir.py check`
(the regression and 32 prep jobs against local_ref.json); a failure renames the bundle to .FAILED.
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
ROOT = os.path.dirname(HERE)
CDIR = os.path.join(HERE, "c")
DIST = os.path.join(CDIR, "dist")
WORK = os.path.join(CDIR, "work", "artifact")
CODE_DIRS = ["cpcm", "cpcm_sa", "e1_fidelity", "e2_attribution", "e3_warmstart", "eir"]
MODELS = ["intra_mapping_gpt_prefill", "intra_mapping_gpt_decode"]
REG_REFS = ["main/pre/pre_a3h_mha", "main/pre/pre_a3h_ffn", "main/pre/pre_e3_mha", "main/pre/pre_e3_ffn", "main/l1/l1_a3h_mha_F4",
            "main/l1/l1_e3_mha_F6", "main/l1/l1_e3_ffn_F1", "next/l1/l1_a3h_mha_CEdge", "next/l1/l1_a3h_ffn_F8a",
            "next/rep/a3h_mha_CEdge_L2Gsync_s100_r123", "next/rep/a3h_ffn_F8a_L2Gsync_s100_r123", "next/full/runs/full_mha_F4a_A3h_g0_123"]


def skip(path):
    parts = path.replace("\\", "/").split("/")
    return "__pycache__" in parts or path.endswith(".pyc")


def add_text(tar, name, text):
    data = text.encode("utf-8")
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
    tar.addfile(ti, io.BytesIO(data))


def add_tree(tar, src, dst):
    for dp, _, fs in os.walk(src):
        for f in fs:
            p = os.path.join(dp, f)
            if not skip(p):
                tar.add(p, (dst + "/" + os.path.relpath(p, src)).replace("\\", "/"))


def build(out, prep_out):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--"] + [f"{d}/*.py" for d in CODE_DIRS] + ["e2e/c_eir.py", "e2e/e2e_pilot_peak.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with tarfile.open(out, "w:gz") as tar:
        for d in CODE_DIRS:
            for name in sorted(os.listdir(os.path.join(ROOT, d))):
                p = os.path.join(ROOT, d, name)
                if name.endswith(".py") and os.path.isfile(p):
                    tar.add(p, f"{d}/{name}")
            inj = os.path.join(ROOT, d, "_inject")
            if os.path.isdir(inj):
                add_tree(tar, inj, f"{d}/_inject")
        add_tree(tar, os.path.join(ROOT, "eir", "inputs"), "eir/inputs")
        tar.add(os.path.join(ROOT, "eir", "faults.json"), "eir/faults.json")
        for r in REG_REFS:
            tar.add(os.path.join(ROOT, "eir", "out", r + ".pkl.gz"), f"eir/out/{r}.pkl.gz")
        for name in ("c_eir.py", "e2e_pilot_peak.py"):
            tar.add(os.path.join(HERE, name), f"e2e/{name}")
        for name in ("faults.json", "starts.json", "local_ref.json"):
            tar.add(os.path.join(CDIR, name), f"e2e/c/{name}")
        add_tree(tar, os.path.join(CDIR, "inputs"), "e2e/c/inputs")
        tar.add(os.path.join(ROOT, "third_party", "isca2026-busybarn-artifact.COMMIT"), "third_party/isca2026-busybarn-artifact.COMMIT")
        for sub in ["src", "utils", "tool"]:
            add_tree(tar, os.path.join(WORK, sub), f"e2e/c/work/artifact/{sub}")
        for model in MODELS:
            for sub in ["py", "cfg"]:
                add_tree(tar, os.path.join(WORK, "models", model, sub), f"e2e/c/work/artifact/models/{model}/{sub}")
        add_tree(tar, os.path.join(prep_out, "prep"), "c_prep/prep")
        ref = json.load(open(os.path.join(CDIR, "local_ref.json"), encoding="utf-8"))
        add_text(tar, "BUNDLE_INFO.json", json.dumps({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": head, "uncommitted_code": dirty,
                                                       "local_ref_built": ref["built"], "local_ref_jobs": len(ref["jobs"]),
                                                       "prep_sinks": sum(1 for _ in os.scandir(os.path.join(prep_out, "prep", "l1")))}, indent=1))
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB, git {head[:7]}{' (UNCOMMITTED CODE: ' + dirty + ')' if dirty else ''}", flush=True)


def clean_env():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "EIR_BACKEND_SEEDS", "CPCM_SA_STEPS", "C_BACKEND_SEEDS")}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    return env


def check(bundle, keep=False):
    tmp = tempfile.mkdtemp(prefix="c_bundle_check_")
    ok = True
    try:
        with tarfile.open(bundle, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        script = os.path.join(tmp, "e2e", "c_eir.py")
        out = os.path.join(tmp, "check_out")
        for cmd, want in (("probe", "PROBE PASSED"), ("check", "CHECK PASSED")):
            p = subprocess.run([sys.executable, "-I", script, cmd, "--out", out, "--workers", "8"], cwd=tmp, capture_output=True, text=True, env=clean_env())
            good = p.returncode == 0 and want in p.stdout
            print(f"[check] {cmd}: {'ok' if good else 'FAILED'}", flush=True)
            print("\n".join(x for x in p.stdout.splitlines() if x.startswith("[c]") or "PASSED" in x or "FAILED" in x), flush=True)
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
    out = os.path.join(DIST, "c_bundle.tar.gz")
    if len(sys.argv) > 1 and sys.argv[1] == "--check-only":
        sys.exit(0 if check(out, "--keep" in sys.argv) else 1)
    build(out, os.path.join(CDIR, "out"))
    if not check(out, "--keep" in sys.argv):
        os.replace(out, out + ".FAILED")
        raise SystemExit(f"bundle check failed; renamed to {out}.FAILED")


if __name__ == "__main__":
    main()
