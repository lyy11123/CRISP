"""DSTU under W1, step 0: reselect the BusyBarn variant of every MHA stream as E4 did (e4_overlap/e4_variants.py: the 14 variants
g{0,1} x l{0..6}, the lowest total latency of the template's own final evaluation), with the template's final broadcast filtered to
W1 (B1's filter, every call must drop the same positive number of QK^T / S messages). Release item 3 (user 2026-11-08).

Runs in this folder's own artifact work copy (work_e4/, copied from e4_overlap/work without results), so e4_overlap's results/ are
never written. Hash: --hash tag (event_notation hashed by tag, reproducible; default) or addr (the artifact as is, as E4 ran).

  python e2e/w1_rerun/w1_variants.py setup
  python e2e/w1_rerun/w1_variants.py run --jobs 14 [--hash tag|addr]     writes out/variants/variants_w1_<hash>.json
  python e2e/w1_rerun/w1_variants.py child <task> <hash>                  one template run (internal)
"""
import argparse
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "e4_overlap", "work", "artifact")
WORK = os.path.join(HERE, "work_e4", "artifact")
MODEL = os.path.join(WORK, "models", "intra_mapping_gpt_decode")
OUT = os.path.join(HERE, "out", "variants")
PAT = re.compile(r"SA time cost:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
TAIL = "{ch}_bw256_co4x4_bw256_t128x64_failpattern0"
SQS = (512, 2048, 8192)
CHS = ("ch2x2", "ch1x4")
W0 = {}     # E4's winners, read from e4_out/variants*.json
W1_DROP = ("qkt", "s_softmax")


def names(sq, ch):
    return [f"mha_sq{sq}_greedy{g}_lossratio{l}_{TAIL.format(ch=ch)}" for g in (0, 1) for l in range(7)]


def setup():
    if os.path.isdir(WORK):
        print(f"{WORK} exists")
        return
    shutil.copytree(SRC, WORK, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "results"))
    os.makedirs(os.path.join(MODEL, "results"), exist_ok=True)
    print(f"work copy {WORK} (from {SRC}, without results)")


def child(task, hsh):
    for sub in ("utils", "src/partition/oper", "src/partition/func", "src/partition", "src/mapping", "src/scheduling",
                "src/backend/analytical", "endtoend", "src/scheduling/communication/topology", "tool"):    # the template's own list
        sys.path.append(os.path.join(WORK, sub))
    import add_communication
    if hsh == "tag":
        from event_notation import event_notation
        event_notation.__hash__ = lambda self: hash(self.event_tag)
    orig = add_communication.add_broadcast
    seen = []

    def add_broadcast(*a, **kw):
        ev = kw["event_dict"]
        before = set(ev)
        out = orig(*a, **kw)
        added = set(ev) - before
        drop = {t for t in added if any(k in ev[t].event_name for k in W1_DROP)}
        for t in drop:
            for dep in ev[t].dependency_set:
                if ev.get(dep) is not None:
                    ev[dep].issue_set.discard(t)
            del ev[t]
        if any(x in drop for e in ev.values() for x in e.issue_set):
            raise RuntimeError("W1 filter left dangling issue references")
        seen.append(len(drop))
        if not drop or len(set(seen)) > 1:
            raise RuntimeError(f"W1 drops per broadcast call {seen}")
        print(f"[w1v] dropped {len(drop)}", flush=True)
        return out
    add_communication.add_broadcast = add_broadcast
    os.chdir(os.path.join(MODEL, "py"))
    runpy.run_path(os.path.join(MODEL, "py", task + ".py"), run_name="__main__")


def read(task):
    p = os.path.join(MODEL, "results", f"{task}.txt")
    if not os.path.isfile(p):
        return None
    m = PAT.search(open(p).read())
    return tuple(map(int, m.groups())) if m else None


def run(jobs, hsh):
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    tasks = [n for sq in SQS for ch in CHS for n in names(sq, ch)]
    for n in tasks:      # a result from another hash mode must not be read
        p = os.path.join(MODEL, "results", f"{n}.txt")
        if os.path.exists(p) and not os.path.exists(os.path.join(OUT, "logs", f"{n}_{hsh}.done")):
            os.remove(p)

    def one(n):
        if read(n) and os.path.exists(os.path.join(OUT, "logs", f"{n}_{hsh}.done")):
            return n, read(n), 0.0, "cached"
        t0 = time.time()
        with open(os.path.join(OUT, "logs", f"{n}_{hsh}.log"), "w", encoding="utf-8") as lf:
            rc = subprocess.call([sys.executable, os.path.abspath(__file__), "child", n, hsh], stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
        r = read(n) if rc == 0 else None
        if r:
            open(os.path.join(OUT, "logs", f"{n}_{hsh}.done"), "w").close()
        print(f"[w1v] {n}: exit {rc}, {time.time() - t0:.0f} s, {r}", flush=True)
        return n, r, time.time() - t0, f"exit={rc}"
    with ThreadPoolExecutor(jobs) as ex:
        res = {n: (r, w, s) for n, r, w, s in ex.map(one, tasks)}
    w0 = {}
    for f, ch in (("variants.json", "ch2x2"), ("variants_1x4.json", "ch1x4")):
        d = json.load(open(os.path.join(ROOT, "e4_overlap", "e4_out", f), encoding="utf-8"))
        for k, v in d.items():
            if isinstance(v, dict) and "best" in v:
                w0[(k, ch)] = v
    out = {}
    for sq in SQS:
        for ch in CHS:
            ns = names(sq, ch)
            got = {n: res[n][0] for n in ns}
            if any(v is None for v in got.values()):
                raise SystemExit(f"missing results for mha_sq{sq} {ch}: {[n for n, v in got.items() if v is None]}")
            best = min(ns, key=lambda n: (got[n][0], n))
            ref = w0.get((f"mha_sq{sq}", ch))
            out[f"mha_sq{sq}_{ch}"] = {"best_w1": best, "best_w1_abc": got[best], "best_w0": ref and ref["best"], "best_w0_abc": ref and ref["best_abc"],
                                       "same_variant": bool(ref) and ref["best"] == best,
                                       "variants": {n: {"abc_w1": got[n], "abc_w0": ref and ref["variants"].get(n, {}).get("abc"), "wall": res[n][1]} for n in ns}}
            print(f"[w1v] mha_sq{sq} {ch}: W1 best {best[:36]} {got[best]}; W0 best {ref and ref['best'][:36]}; same {out[f'mha_sq{sq}_{ch}']['same_variant']}", flush=True)
    json.dump({"hash": hsh, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": out}, open(os.path.join(OUT, f"variants_w1_{hsh}.json"), "w"), indent=1)


def main():
    if sys.argv[1] == "child":
        return child(sys.argv[2], sys.argv[3])
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "run"])
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--hash", default="tag", choices=["tag", "addr"])
    a = ap.parse_args()
    setup() if a.cmd == "setup" else run(a.jobs, a.hash)


if __name__ == "__main__":
    main()
