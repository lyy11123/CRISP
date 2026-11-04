"""Build colab/dist/colab_bundle.tar.gz: CRISP code + the BusyBarn source and task scripts needed on Colab.

Only reads the working tree; never writes outside colab/dist. cpcm_sa/cpcm_sa.py is NOT edited (the local stage 2
children re-read that file): the bundle holds a patched copy, and every patch must match exactly once or the build
stops. Historical outputs (e2_out, e3_out, cpcm_out, cpcm_sa/out, work copies, results) are left out.

  python colab/build_bundle.py
"""
import hashlib
import io
import json
import os
import subprocess
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIST = os.path.join(HERE, "dist")
ART = os.path.join(ROOT, "third_party", "isca2026-busybarn-artifact")
ART_IN_BUNDLE = "cpcm_sa/work/artifact"
MODELS = ["intra_mapping_gpt_decode", "intra_mapping_qwen3moe_decode"]
CODE_DIRS = ["cpcm", "cpcm_sa", "e1_fidelity", "e2_attribution", "e3_warmstart"]
OPS_CHECK = [("intra_mapping_qwen3moe_decode", "ffn", 2048), ("intra_mapping_gpt_decode", "mha", 512),
             ("intra_mapping_gpt_decode", "mha", 8192)]
SUF = "_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0"

PATCHES = [
    # outputs and seed count can be redirected (smoke test only); defaults are unchanged
    ('OUT_ROOT = os.path.join(HERE, "out")',
     'OUT_ROOT = os.environ.get("CPCM_SA_OUT_ROOT") or os.path.join(HERE, "out")'),
    ('BACKEND_SEEDS = list(range(1, 11))',
     'BACKEND_SEEDS = list(range(1, int(os.environ.get("CPCM_SA_BACKEND_SEEDS", "10")) + 1))'),
    # new configs; MoE faults are the links of cpcm/cpcm_out/v2/moe_F4.pkl.gz and moe_F5.pkl.gz
    ('"moe": ("intra_mapping_qwen3moe_decode", "ffn", None, "ch2x2")}',
     '"moe": ("intra_mapping_qwen3moe_decode", "ffn", None, "ch2x2"),\n'
     '           "moe_f4": ("intra_mapping_qwen3moe_decode", "ffn",\n'
     '                      {"failed_nodes": [], "failed_links": [[[3, 2], [1, 14]], [[1, 14], [3, 2]]]}, "ch2x2"),\n'
     '           "moe_f5": ("intra_mapping_qwen3moe_decode", "ffn",\n'
     '                      {"failed_nodes": [], "failed_links": [[[1, 13], [3, 1]], [[3, 1], [1, 13]]]}, "ch2x2"),\n'
     '           "mha_sq512": (GPT, "mha", None, "ch2x2"), "mha_sq8192": (GPT, "mha", None, "ch2x2")}\n'
     'SQ = {"mha_sq512": 512, "mha_sq8192": 8192}'),
    ('"mha_f5": "balanced+fault"}',
     '"mha_f5": "balanced+fault",\n'
     '          "moe_f4": "comm+fault", "moe_f5": "comm+fault", "mha_sq512": "balanced", "mha_sq8192": "balanced"}'),
    ('f"{op}_sq2048_greedy{g}_lossratio{l}_{ch}{SUF}"', 'f"{op}_sq{SQ.get(cfg, 2048)}_greedy{g}_lossratio{l}_{ch}{SUF}"'),
]


def patched_cpcm_sa():
    with open(os.path.join(ROOT, "cpcm_sa", "cpcm_sa.py"), encoding="utf-8", newline="") as f:
        src = f.read()
    orig = hashlib.sha256(src.encode("utf-8")).hexdigest()
    for old, new in PATCHES:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"patch target occurs {n} times, expected 1: {old}")
        src = src.replace(old, new)
    return src, orig


def add_text(tar, name, text):
    data = text.encode("utf-8")
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
    tar.addfile(ti, io.BytesIO(data))


def skip(path):
    parts = path.replace("\\", "/").split("/")
    return "__pycache__" in parts or path.endswith((".pyc", ".pkl", ".gz"))


def main():
    for model, op, sq in OPS_CHECK:
        for g in (0, 1):
            for l in range(7):
                p = os.path.join(ART, "models", model, "py", f"{op}_sq{sq}_greedy{g}_lossratio{l}{SUF}.py")
                if not os.path.isfile(p):
                    raise SystemExit(f"missing task script: {p}")
    os.makedirs(DIST, exist_ok=True)
    src, orig = patched_cpcm_sa()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = os.path.join(DIST, "colab_bundle.tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        for d in CODE_DIRS:
            for name in sorted(os.listdir(os.path.join(ROOT, d))):
                p = os.path.join(ROOT, d, name)
                if name.endswith(".py") and os.path.isfile(p) and not (d == "cpcm_sa" and name == "cpcm_sa.py"):
                    tar.add(p, f"{d}/{name}")
        add_text(tar, "cpcm_sa/cpcm_sa.py", src)
        for d in ("cpcm_sa", "e3_warmstart"):
            inj = os.path.join(ROOT, d, "_inject")
            for name in sorted(os.listdir(inj)):
                if not skip(name):
                    tar.add(os.path.join(inj, name), f"{d}/_inject/{name}")
        tar.add(os.path.join(HERE, "colab_run.py"), "colab/colab_run.py")
        tar.add(ART + ".COMMIT", "third_party/isca2026-busybarn-artifact.COMMIT")
        for sub in ["src", "utils", "tool"]:
            base = os.path.join(ART, sub)
            if os.path.isdir(base):
                for dp, _, fs in os.walk(base):
                    for f in fs:
                        p = os.path.join(dp, f)
                        if not skip(p):
                            tar.add(p, f"{ART_IN_BUNDLE}/{os.path.relpath(p, ART)}".replace("\\", "/"))
        for model in MODELS:
            for sub in ["py", "cfg"]:
                base = os.path.join(ART, "models", model, sub)
                for dp, _, fs in os.walk(base):
                    for f in fs:
                        p = os.path.join(dp, f)
                        if not skip(p):
                            tar.add(p, f"{ART_IN_BUNDLE}/{os.path.relpath(p, ART)}".replace("\\", "/"))
        add_text(tar, "BUNDLE_INFO.json", json.dumps({
            "built": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": head, "cpcm_sa_sha256_before_patch": orig,
            "patches": [p[0] for p in PATCHES]}, indent=1, ensure_ascii=False))
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB, git {head[:7]}, cpcm_sa.py {orig[:12]}")


if __name__ == "__main__":
    main()
