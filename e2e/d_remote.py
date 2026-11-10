"""D remote package: build on this machine, run on any Linux or Windows machine (Colab or a rented server).

  python e2e/d_remote.py build D2 D3 ... [--part ext] [--name NAME]   -> e2e/d/dist/<NAME>.tar.gz, checked in a clean room
  (in the unpacked bundle)  python -I e2e/d_remote.py run --out OUT --workers 7 [--shard i --nshards M] [--peak-gb 5 --free-gb 3]
  (in the unpacked bundle)  python -I e2e/d_remote.py check --out OUT        2 jobs, compared with this machine's results (ref.json)

The bundle holds the code (cpcm, cpcm_sa, e2_attribution, e3_warmstart, e1_fidelity, eir, e4_overlap, dstu, and e2e's c_eir.py,
d_joint.py, d_hop.py, d_batch.py, d_remote.py, fast_hop.py, fast_cpcm.py, e2e_pilot_peak.py), the C artifact work copy (src, utils,
tool, GPT prefill and decode py + cfg), the artifact commit, e2e/c/faults.json, the exported graphs the jobs read, and jobs.json
(specs with paths relative to the bundle root; the runner makes them absolute). Results: OUT/sims/<job>.pkl.gz, logs in OUT/logs.
Pack OUT/sims (and OUT/logs) and bring them back to e2e/d/out/prod/sims on this machine.
"""
import argparse
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


def _root():
    return os.path.dirname(HERE)


# ---------------------------------------------------------------- remote side
def load_jobs(root):
    with open(os.path.join(root, "e2e", "d_jobs.json"), encoding="utf-8") as f:
        return json.load(f)


def absolute(j, root, out):
    sp = json.loads(json.dumps(j["spec"]))
    sp["exports"] = {k: os.path.join(root, v) for k, v in sp["exports"].items()}
    sp["sink"] = os.path.join(out, "sims", os.path.basename(sp["sink"]))
    return {"name": j["name"], "cmd": j["cmd"], "spec": sp, "folder": os.path.join(out, "sims"), "limit_gb": None}


def run_remote(out, workers, shard, nshards, names=None):
    root = _root()
    sys.path.insert(0, HERE)
    import d_joint as d
    d.OUT = out
    jobs = load_jobs(root)
    if names:
        jobs = [j for j in jobs if j["name"] in names]
    jobs = jobs[shard::nshards]
    items = [absolute(j, root, out) for j in jobs]
    os.makedirs(os.path.join(out, "sims"), exist_ok=True)
    print(f"[remote] {len(items)} jobs, {workers} parallel, peak {d.LOCAL_PEAK_GB} GB, free {d.LOCAL_FREE_GB} GB", flush=True)
    t0 = time.time()
    r = d.run_all(items, workers)
    bad = [x for x in r if x.get("rc") or x.get("killed")]
    with open(os.path.join(out, f"runs_{shard}.json"), "w") as f:
        json.dump(r, f, indent=1)
    print(f"[remote] done in {(time.time() - t0) / 3600:.2f} h; failed {len(bad)}: {bad}", flush=True)
    return bad


def check(out):
    """Run the reference jobs (ref.json) and compare: per-hop runs bit for bit (T-hat, every total and per-stream completion)."""
    root = _root()
    with open(os.path.join(root, "e2e", "d_ref.json"), encoding="utf-8") as f:
        ref = json.load(f)
    bad = run_remote(out, 2, 0, 1, names=set(ref))
    sys.path.insert(0, HERE)
    import d_joint as d
    diff = []
    for name, want in ref.items():
        s = d._load(os.path.join(out, "sims", name + ".pkl.gz"))
        got = summary(s)
        if got != want:
            diff.append({"job": name, "want": want, "got": got})
    print(json.dumps({"failed": bad, "differences": diff}, indent=1, default=str)[:4000])
    if bad or diff:
        raise SystemExit(1)
    print("[remote] check: identical")


def summary(s):
    return {"that": {k: v["T_hat"] for k, v in s["checks"].items() if k.startswith("that")},
            "runs": [[r["label"], r["out"], {str(k): v["C"] for k, v in sorted(r["streams"].items())}] for r in s["runs"]]}


# ---------------------------------------------------------------- build side
CODE = ["cpcm", "cpcm_sa", "e1_fidelity", "e2_attribution", "e3_warmstart", "eir", "e4_overlap", "dstu"]
E2E_FILES = ["c_eir.py", "d_joint.py", "d_hop.py", "d_batch.py", "d_remote.py", "fast_hop.py", "fast_cpcm.py", "e2e_pilot_peak.py"]
MODELS = ["intra_mapping_gpt_prefill", "intra_mapping_gpt_decode"]


def _skip(p):
    parts = p.replace("\\", "/").split("/")
    return "__pycache__" in parts or p.endswith(".pyc")


def _add_tree(tar, src, dst, only_py=False):
    for dp, dirs, fs in os.walk(src):
        rel = os.path.relpath(dp, src).replace("\\", "/")
        if only_py and any(x in rel.split("/") for x in ("out", "work", "e4_out", "e4b_out", "inputs", "_tmp", "logs", "sims", "dist")):
            continue
        for f in fs:
            p = os.path.join(dp, f)
            if _skip(p) or (only_py and not (f.endswith(".py") or rel.startswith("_inject"))):
                continue
            tar.add(p, (dst + "/" + os.path.relpath(p, src)).replace("\\", "/"))


def _add_text(tar, name, text):
    data = text.encode("utf-8")
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
    tar.addfile(ti, io.BytesIO(data))


def build(batches, part, name, ref_jobs):
    sys.path.insert(0, HERE)
    import d_joint as d
    import d_batch as B
    out = os.path.join(d.OUT, "prod")
    jobs = []
    for b in batches:
        _, js = B.build(b, out)
        if part in ("ext", "all"):
            xs = B.ext_jobs(js)
            js = xs if part == "ext" else js + xs
        jobs += js
    # every export the jobs read must exist (export them here first: d_batch run --part ext does that, or this call)
    exports = {}
    for j in jobs:
        for k, p in j["spec"]["exports"].items():
            if not os.path.exists(p):
                raise SystemExit(f"{j['name']}: export {p} missing; run the batch's exports first")
            exports[p] = "e2e/d/exports/" + os.path.basename(p)
    rel = []
    for j in jobs:
        sp = json.loads(json.dumps(j["spec"]))
        sp["exports"] = {k: exports[v] for k, v in sp["exports"].items()}
        rel.append({"name": j["name"], "cmd": j["cmd"], "spec": sp, "arms": j["arms"], "cond": j["cond"], "kind": j["kind"]})
    root = d.MAIN
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_root(), capture_output=True, text=True).stdout.strip()
    dist = os.path.join(d.OUT, "..", "dist")
    os.makedirs(dist, exist_ok=True)
    path = os.path.join(dist, name + ".tar.gz")
    work = os.path.join(d.ME2E, "c", "work", "artifact")
    ref = {}
    for nm in ref_jobs:                          # reference results of this machine for `check`
        p = os.path.join(out, "sims", nm + ".pkl.gz")
        if not os.path.exists(p):
            raise SystemExit(f"reference job {nm} has no result on this machine")
        ref[nm] = json.loads(json.dumps(summary(d._load(p)), default=float))
    with tarfile.open(path, "w:gz") as tar:
        for c_ in CODE:
            _add_tree(tar, os.path.join(root, c_), c_, only_py=True)
        for f in E2E_FILES:
            src = os.path.join(HERE, f) if os.path.exists(os.path.join(HERE, f)) else os.path.join(d.ME2E, f)
            tar.add(src, "e2e/" + f)
        tar.add(os.path.join(d.ME2E, "c", "faults.json"), "e2e/c/faults.json")
        for sub in os.listdir(work):
            if sub == "models":
                continue
            p = os.path.join(work, sub)
            if os.path.isdir(p):
                _add_tree(tar, p, "e2e/c/work/artifact/" + sub)
            else:
                tar.add(p, "e2e/c/work/artifact/" + sub)
        for m in MODELS:
            for sub in ("py", "cfg"):
                _add_tree(tar, os.path.join(work, "models", m, sub), f"e2e/c/work/artifact/models/{m}/{sub}")
        tar.add(os.path.join(root, "third_party", "isca2026-busybarn-artifact.COMMIT"), "third_party/isca2026-busybarn-artifact.COMMIT")
        for src, dst in exports.items():
            tar.add(src, dst)
        _add_text(tar, "e2e/d_jobs.json", json.dumps(rel))
        _add_text(tar, "e2e/d_ref.json", json.dumps(ref))
        _add_text(tar, "BUNDLE.json", json.dumps({"git": head, "batches": batches, "part": part, "jobs": len(rel), "exports": len(exports),
                                                  "built": time.strftime("%Y-%m-%d %H:%M:%S")}))
    print(f"bundle {path}: {len(rel)} jobs, {len(exports)} exports, {os.path.getsize(path) / 2 ** 20:.0f} MB", flush=True)
    # clean room: unpack, python -I, check (the reference jobs) -- also proves the bundle runs without this repository
    tmp = tempfile.mkdtemp(prefix="dbundle_")
    try:
        with tarfile.open(path) as tar:
            tar.extractall(tmp)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["D_MAIN"] = tmp
        rc = subprocess.call([sys.executable, "-I", os.path.join(tmp, "e2e", "d_remote.py"), "check", "--out", os.path.join(tmp, "out")], cwd=tmp, env=env)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if rc != 0:
        os.replace(path, path + ".FAILED")
        raise SystemExit(f"clean-room check failed: {path}.FAILED")
    print("clean-room check passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("batches", nargs="*")
    ap.add_argument("--part", default="ext")
    ap.add_argument("--name")
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--peak-gb", type=float)
    ap.add_argument("--free-gb", type=float)
    a = ap.parse_args()
    if a.cmd in ("run", "check"):
        sys.path.insert(0, HERE)
        import d_joint as d
        if a.peak_gb:
            d.LOCAL_PEAK_GB = a.peak_gb
        if a.free_gb:
            d.LOCAL_FREE_GB = a.free_gb
        out = os.path.abspath(a.out)
        if a.cmd == "run":
            sys.exit(1 if run_remote(out, a.workers, a.shard, a.nshards) else 0)
        return check(out)
    if a.cmd == "build":
        return build(a.batches, a.part, a.name or ("d_" + "_".join(a.batches) + "_" + a.part), a.ref)
    raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
