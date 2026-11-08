"""DSTU and its upstream under W1 and W0-tag (E2E_HANDOFF.md section 4, release items 1, 3, 5, 7; user 2026-11-08).

Each workload gets its own tree e2e/w1_rerun/dstu_<wl>/ with verbatim copies of cpcm, e2_attribution, e3_warmstart, e4_overlap and
dstu (code only), the e4_overlap artifact work copy (src, utils, tool, endtoend, GPT decode py + cfg) and the artifact commit file.
The original scripts then run unchanged inside the tree, in the original order and with the original command lines, so every
result lands in the tree (dstu_<wl>/e4_overlap/e4_out, e4b_out, dstu_<wl>/dstu/out); e4_overlap/ and dstu/ are only read.

In every process of the tree (sitecustomize in dstu_<wl>/_w1inject, on PYTHONPATH), as soon as the module is imported:
  event_notation.__hash__  -> hash of the event tag (tag hash, release item 1: both workloads)
  e4b_backend.event_driver_e4b -> e2e/fast_hop.event_driver_fast (bit-identical, SPEEDUP_SUMMARY.md)
  cpcm.simulate            -> e2e/fast_cpcm.simulate (bit-identical)
The E4 capture children (template runs) keep e4_overlap's own sitecustomize, as in E4.

Inputs copied into the tree's e4_out: grid1_state.json (fault coordinates, unchanged), T6.json, variants*.json, and E4a's W0 files
that E4b's self-checks 1 and 2 compare with (hop/*_hop.json, fcfs/fcfs_table.csv, the two fcfs sims of the 6-seed configuration,
diag/hop_chunk.json); E4a itself is not rerun except its single-stream runs, so these checks compare with W0 / address hash.
Streams: w0tag = E4's captured streams as they are. w1 = the same with the final broadcast's QK^T and S messages removed (B1's
filter; 72 per MHA stream, 0 for FFN, checked); a stream whose BusyBarn winner changes under W1 (w1_variants.py) is captured again
from the new winner as E4 did (E.capture, then E4's capture_r0 for each fault platform), then filtered.
Upstream rerun in the tree: E4a's single-stream runs (hop: single_<k>_hop for T_hop; chunked: chunk_single_<k>_<cz>), the same set
as E4 had; E4b settings 1, 1b, 1c, 2, 3 (solo included); then DSTU in the original order.

  python e2e/w1_rerun/w1_dstu.py build <wl>            tree, inputs, streams (w1: filter / recapture)
  python e2e/w1_rerun/w1_dstu.py check <wl>            the patches are active in a child; stream drop counts
  python e2e/w1_rerun/w1_dstu.py regress <wl>          fast vs original backend on E4b policies (P1, fixed, P6, P7, P7m, chunked), tag hash
  python e2e/w1_rerun/w1_dstu.py e4a <wl> --jobs 14    E4a single-stream runs, hop_state.json
  python e2e/w1_rerun/w1_dstu.py pipe <wl> [--from N]  E4b and DSTU steps in order (log out/pipe_<wl>.log)
"""
import argparse
import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.dirname(HERE)
ROOT = os.path.dirname(E2E)
WLS = ("w1", "w0tag")
CODE = ["cpcm", "e2_attribution", "e3_warmstart", "e4_overlap", "dstu"]
W1_DROP = ("qkt", "s_softmax")
VARIANTS_W1 = os.path.join(HERE, "out", "variants", "variants_w1_tag.json")
TAIL = "ch2x2_bw256_co4x4_bw256_t128x64_failpattern0"

# the original command lines, in order (cwd = tree's e4_overlap or dstu; e4_overlap/PROGRESS.md, dstu/PROGRESS.md)
PIPE = [("e4_overlap", "e4b_s1.py --jobs 14"), ("e4_overlap", "e4b_s1b.py --jobs 14"), ("e4_overlap", "e4b_s1c.py --jobs 14"),
        ("e4_overlap", "e4b_s2.py --jobs 14"), ("e4_overlap", "e4b_s3.py --jobs 14"),
        ("dstu", "dstu.py solo --jobs 14"), ("dstu", "dstu.py s1 --jobs 14"), ("dstu", "dstu.py s2 --jobs 14"), ("dstu", "dstu.py s3 --jobs 14"),
        ("dstu", "dstu2.py check"), ("dstu", "dstu2.py 2a --jobs 14"), ("dstu", "dstu2.py pilot --jobs 14"),
        ("dstu", "dstu2.py 2b --jobs 14 POOL_B"), ("dstu", "dstu2.py greedy --jobs 14 POOL_B"), ("dstu", "dstu2.py cm --jobs 14"),
        ("dstu", "dstu2c.py solo --jobs 14"), ("dstu", "dstu2c.py a --jobs 14"), ("dstu", "dstu2c.py b --jobs 14"), ("dstu", "dstu2c.py s1 --jobs 14"),
        ("dstu", "dstu_oh.py run --jobs 14"), ("dstu", "dstu_oh.py key --jobs 14"),
        ("dstu", "dstu_s3all.py solo --jobs 14"), ("dstu", "dstu_s3all.py run --jobs 14"), ("dstu", "dstu_s3all.py ns --jobs 9"),
        ("dstu", "dstu_final.py check --jobs 14"), ("dstu", "dstu_final.py s2 --jobs 14"), ("dstu", "dstu_final.py sel --jobs 14"),
        ("dstu", "dstu_pair.py")]
# E4b setting 1 ends with E4bError when a self-check flags (it did under W0 too: check 2 > 4%, check 3); its files are written first
SOFT_STOP = {"e4b_s1.py --jobs 14"}

SITE = r'''"""W1 rerun tree (e2e/w1_rerun/w1_dstu.py): tag hash, fast hop backend, fast CPCM, patched as each module is imported."""
import importlib.abc
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _event_notation(m):
    m.event_notation.__hash__ = lambda self: hash(self.event_tag)


def _e4b_backend(m):
    sys.path.insert(0, _HERE)
    import fast_hop
    m._orig_event_driver_e4b = m.event_driver_e4b
    m.event_driver_e4b = fast_hop.event_driver_fast


def _cpcm(m):
    sys.path.insert(0, _HERE)
    import fast_cpcm
    m._orig_simulate = m.simulate
    m.simulate = fast_cpcm.simulate


_PATCH = {"event_notation": _event_notation, "e4b_backend": _e4b_backend, "cpcm": _cpcm}
if os.environ.get("W1RR_NOFAST"):
    _PATCH = {"event_notation": _event_notation}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, fn):
        self.inner, self.fn = inner, fn

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.fn(module)
        module.__w1rr_patched__ = True


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _PATCH:
            return None
        for f in sys.meta_path:
            if f is self:
                continue
            spec = getattr(f, "find_spec", lambda *a: None)(name, path, target)
            if spec is not None:
                spec.loader = _Loader(spec.loader, _PATCH[name])
                return spec
        return None


sys.meta_path.insert(0, _Finder())
'''


def tree(wl):
    return os.path.join(HERE, f"dstu_{wl}")


def env_of(wl, **extra):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env.update(PYTHONPATH=os.path.join(tree(wl), "_w1inject"), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", **extra)
    return env


def log(msg, wl=None):
    line = f"[w1dstu {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if wl:
        os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
        with open(os.path.join(HERE, "out", f"pipe_{wl}.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------- build
def copy_code(R):
    for d in CODE:
        src, dst = os.path.join(ROOT, d), os.path.join(R, d)
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(src):
            p = os.path.join(src, name)
            if name.endswith(".py") and os.path.isfile(p):
                shutil.copy2(p, os.path.join(dst, name))
        if os.path.isdir(os.path.join(src, "_inject")):
            shutil.copytree(os.path.join(src, "_inject"), os.path.join(dst, "_inject"), dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.makedirs(os.path.join(R, "third_party"), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, "third_party", "isca2026-busybarn-artifact.COMMIT"), os.path.join(R, "third_party"))
    inj = os.path.join(R, "_w1inject")
    os.makedirs(inj, exist_ok=True)
    open(os.path.join(inj, "sitecustomize.py"), "w", encoding="utf-8").write(SITE)
    for f in ("fast_hop.py", "fast_cpcm.py"):          # they import ../e4_overlap and ../cpcm relative to themselves: the tree's
        shutil.copy2(os.path.join(E2E, f), os.path.join(inj, f))


def copy_work(R):
    src, dst = os.path.join(ROOT, "e4_overlap", "work", "artifact"), os.path.join(R, "e4_overlap", "work", "artifact")
    if os.path.isdir(dst):
        return
    for sub in ("src", "utils", "tool", "endtoend"):
        if os.path.isdir(os.path.join(src, sub)):
            shutil.copytree(os.path.join(src, sub), os.path.join(dst, sub), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for sub in ("py", "cfg"):
        shutil.copytree(os.path.join(src, "models", "intra_mapping_gpt_decode", sub), os.path.join(dst, "models", "intra_mapping_gpt_decode", sub),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.makedirs(os.path.join(dst, "models", "intra_mapping_gpt_decode", "results"), exist_ok=True)


def copy_inputs(R):
    s, d = os.path.join(ROOT, "e4_overlap", "e4_out"), os.path.join(R, "e4_overlap", "e4_out")
    os.makedirs(d, exist_ok=True)
    for f in ("grid1_state.json", "T6.json", "variants.json", "variants_1x4.json", "step2_checks.json", "step3_chunk.json"):
        if os.path.exists(os.path.join(s, f)):
            shutil.copy2(os.path.join(s, f), os.path.join(d, f))
    os.makedirs(os.path.join(d, "hop"), exist_ok=True)
    for f in os.listdir(os.path.join(s, "hop")):
        if f.endswith("_hop.json"):               # E4a grid summaries: E4b check 1 / 2 references (W0)
            shutil.copy2(os.path.join(s, "hop", f), os.path.join(d, "hop", f))
    os.makedirs(os.path.join(d, "fcfs", "sims"), exist_ok=True)
    shutil.copy2(os.path.join(s, "fcfs", "fcfs_table.csv"), os.path.join(d, "fcfs"))
    for f in os.listdir(os.path.join(s, "fcfs", "sims")):
        if f.startswith("a2_2s_mha-ffn_F4_") and "_short_first_d0.5_prio_" in f:     # E4b's 6-seed configuration, seeds 1-3 (check 2 only)
            shutil.copy2(os.path.join(s, "fcfs", "sims", f), os.path.join(d, "fcfs", "sims", f))
    os.makedirs(os.path.join(d, "diag"), exist_ok=True)
    shutil.copy2(os.path.join(s, "diag", "hop_chunk.json"), os.path.join(d, "diag"))
    json.dump({"note": "inputs copied from e4_overlap/e4_out (W0, address hash): grid1_state, T6, variants, E4a's hop/*_hop.json, "
                       "fcfs table and two fcfs sims, diag/hop_chunk.json (E4b self-check references only)"},
              open(os.path.join(d, "W1RR_INPUTS.json"), "w"), indent=1)


def fault_map(R):
    sys.path.insert(0, os.path.join(R, "e4_overlap"))
    st = json.load(open(os.path.join(R, "e4_overlap", "e4_out", "grid1_state.json"), encoding="utf-8"))
    import e4_grid as G
    out = {}
    for g, fs in st["faults"].items():
        keys = (G.GROUPS.get(g) or G.GROUPS_1X4.get(g) or (None, []))[1]
        for hw, fl in fs.items():
            fl = dict(fl, failed_links=[tuple(map(tuple, l)) for l in fl["failed_links"]])    # as e4b_s1.load (the key hashes the tuples)
            for k in keys:
                out[G.fault_key(k, fl)] = (k, {"failed_nodes": [], "failed_links": fl["failed_links"]})
    return out


def filter_blob(blob, base):
    """B1's filter on a captured event dict: the final broadcast = communications without readers sent to several DDR ports."""
    import pickle as pk
    ev = pk.loads(blob)
    added = {t for t, e in ev.items() if e.event_type == "communication" and not e.issue_set and isinstance(e.target_location, (set, frozenset, list))
             and len(e.target_location) > 1}
    drop = {t for t in added if any(k in ev[t].event_name for k in W1_DROP)}
    for t in drop:
        for dep in ev[t].dependency_set:
            if ev.get(dep) is not None:
                ev[dep].issue_set.discard(t)
        del ev[t]
    tags = set(ev)
    bad = sum(1 for e in ev.values() for x in list(e.issue_set) + list(e.dependency_set) if x not in tags)
    if bad:
        raise RuntimeError(f"W1 filter left {bad} dangling references")
    want = 0 if base.startswith("ffn") else 72
    if len(drop) != want:
        raise RuntimeError(f"W1 filter dropped {len(drop)} messages from {base}, expected {want}")
    return pk.dumps(ev, protocol=pk.HIGHEST_PROTOCOL), len(drop), len(added)


def build_streams(wl, R):
    src = os.path.join(ROOT, "e4_overlap", "e4_out", "streams")
    dst = os.path.join(R, "e4_overlap", "e4_out", "streams")
    os.makedirs(dst, exist_ok=True)
    fm = fault_map(R)
    names = sorted(f[:-7] for f in os.listdir(src) if f.endswith(".pkl.gz") and not f.startswith("smoke_") and "_1x4" not in f)
    recapture = set()
    if wl == "w1":
        v = json.load(open(VARIANTS_W1, encoding="utf-8"))["rows"]
        for sq in (512, 2048, 8192):
            r = v[f"mha_sq{sq}_ch2x2"]
            if not r["same_variant"]:
                recapture.add((f"mha{sq}", r["best_w1"]))
    rec_bases = {b for b, _ in recapture}
    for n in names:
        base = n.split("__")[0]
        if base in rec_bases:
            continue
        if not os.path.exists(os.path.join(dst, n + ".pkl.gz")):
            shutil.copy2(os.path.join(src, n + ".pkl.gz"), os.path.join(dst, n + ".pkl.gz"))
    info = {"copied": [n for n in names if n.split("__")[0] not in rec_bases], "recaptured": {}}
    if recapture:
        # E4's capture (template unchanged) of the new winner, then E4's capture_r0 on every fault platform of that stream
        code = r'''
import json, sys
import e4_overlap as E, e4_grid as G
base, task, keys = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
E.STREAMS[base] = task
import os
if not os.path.exists(os.path.join(E.STREAM_DIR, base + ".pkl.gz")):
    print(json.dumps(E.capture(base), default=str))
for k, fl in keys:
    print(G.capture_r0(k, base, {"failed_nodes": [], "failed_links": [tuple(map(tuple, l)) for l in fl]}), flush=True)
'''
        for base, task in sorted(recapture):
            keys = [(k, f[1]["failed_links"]) for k, f in fm.items() if f[0] == base and (k + ".pkl.gz") in os.listdir(src)]
            log(f"{wl}: capture {base} from {task} and {len(keys)} fault platforms", wl)
            p = subprocess.run([sys.executable, "-c", code, base, task, json.dumps(keys)], cwd=os.path.join(R, "e4_overlap"),
                               capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
            open(os.path.join(HERE, "out", f"capture_{wl}_{base}.log"), "w", encoding="utf-8").write(p.stdout + p.stderr)
            if p.returncode:
                raise SystemExit(f"capture of {base} failed; see out/capture_{wl}_{base}.log")
            info["recaptured"][base] = {"task": task, "fault_keys": [k for k, _ in keys]}
    if wl == "w1":
        import e4_overlap as E          # the tree's (fault_map put it on sys.path): artifact modules for unpickling
        E._artifact_path()
        info["filtered"] = {}
        for f in sorted(os.listdir(dst)):
            n = f[:-7]
            p = os.path.join(dst, f)
            with gzip.open(p, "rb") as fh:
                rec = pickle.load(fh)
            if rec.get("w1rr_filtered"):
                info["filtered"][n] = rec["w1rr_filtered"]
                continue
            blob, nd, na = filter_blob(rec["events_blob"], n)
            rec["events_blob"] = blob
            rec["w1rr_filtered"] = {"dropped": nd, "final_broadcast": na, "sha_w1": hashlib.sha256(blob).hexdigest()[:16]}
            with gzip.open(p + ".part", "wb") as fh:
                pickle.dump(rec, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(p + ".part", p)
            info["filtered"][n] = rec["w1rr_filtered"]
    json.dump(info, open(os.path.join(R, "e4_overlap", "e4_out", "W1RR_STREAMS.json"), "w"), indent=1)
    log(f"{wl}: streams {len(os.listdir(dst))}; recaptured {sorted(info['recaptured'])}; "
        + (f"dropped per stream {sorted({v['dropped'] for v in info['filtered'].values()})}" if wl == "w1" else "unfiltered"), wl)


def build(wl):
    R = tree(wl)
    os.makedirs(R, exist_ok=True)
    copy_code(R)
    copy_work(R)
    copy_inputs(R)
    build_streams(wl, R)
    log(f"{wl}: tree {R} built", wl)


# ---------------------------------------------------------------- checks
def check(wl):
    R = tree(wl)
    code = r'''
import json, sys
sys.path.insert(0, "../cpcm")
import e4_overlap as E
E._artifact_path()
import event_notation, e4b_backend, cpcm
print(json.dumps({"tag_hash": getattr(event_notation, "__w1rr_patched__", False), "fast_hop": e4b_backend.event_driver_e4b.__name__,
                  "fast_cpcm": cpcm.simulate.__module__, "e4b_backend_file": e4b_backend.__file__, "cpcm_file": cpcm.__file__,
                  "event_notation_file": event_notation.__file__}))
'''
    p = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(R, "e4_overlap"), capture_output=True, text=True, env=env_of(wl))
    print(p.stdout, p.stderr[-2000:])
    r = json.loads(p.stdout.strip().splitlines()[-1])
    ok = r["tag_hash"] and r["fast_hop"] == "event_driver_fast" and r["fast_cpcm"] == "fast_cpcm" and all(
        os.path.abspath(r[k]).startswith(R) for k in ("e4b_backend_file", "cpcm_file", "event_notation_file"))
    log(f"{wl}: patch check {'ok' if ok else 'FAILED'} {r}", wl)
    return ok


REGRESS = r"""
# fast vs original event_driver_e4b on E4b children, tag hash in both: identical outputs required. E4b's W0 setting-1 inputs (T and
# solo caps of e4_overlap/e4b_out, read only): one 3-stream unchunked F4 configuration and the chunked a2 one, release d0, k = 1.5,
# seed 1; policies P1, fixed, P6, P7, P6n, P7n, P7m.
import gzip, hashlib, json, os, pickle, sys, time
import e4_overlap as E, e4_hop as H, e4b as B, e4b_s1 as S1
E._artifact_path()
import e4b_backend, fast_hop
W0 = sys.argv[1]
st = json.load(open(os.path.join(E.OUT, "grid1_state.json"), encoding="utf-8"))
for g, fs in st["faults"].items():
    for fl in fs.values():
        fl["failed_links"] = [tuple(map(tuple, l)) for l in fl["failed_links"]]
plats = S1.platforms(st)
s1 = json.load(open(os.path.join(W0, "setting1.json"), encoding="utf-8"))
T = {}
for kk, v in s1["T"].items():
    k, cz = kk.rsplit("|", 1)
    T[(k, None if cz == "None" else cz)] = v
def solo(k, cz):
    with gzip.open(os.path.join(W0, "sims", f"solo_{k}_{cz or 'none'}.pkl.gz"), "rb") as f:
        caps = {r["seed"]: r["cap"] for r in pickle.load(f)["runs"]}
    return B.solo_data(caps[1])
WALL = {"wall", "process_wall", "times", "wall_s"}
def strip(x):
    if isinstance(x, dict): return {k: strip(v) for k, v in x.items() if k not in WALL}
    if isinstance(x, (list, tuple)): return type(x)(strip(v) for v in x)
    return x
pick = [p for p in plats if p[0] == "a1_3s_512-2048-8192" and p[1] == "F4"] + [p for p in plats if p[0] == "a2_2s_mha-ffn" and p[1] == "nofault"]
items = []
for g, hw, cz, keys, fa, alias in pick:
    Tc = {i: T[(k, cz)] for i, k in enumerate(keys)}
    rel = {i: 0 for i in Tc}
    sd = {i: solo(k, cz) for i, k in enumerate(keys)}
    ref = {i: r for r, i in enumerate(sorted(Tc, key=lambda i: (Tc[i], i)))}
    d = {i: 1.5 * Tc[i] for i in Tc}
    base = {"cfg": E.CFG, "faults": fa, "streams": keys, "release": rel, "seeds": [1], "chunk": H.CHUNKS[cz] if cz else None, "audit": True}
    pp = {"solo": sd, "d": d, "T": Tc, "release": rel, "ref_rank": ref}
    items.append((f"{g}_{hw}_P1", dict(base, policy={"name": "P1"})))
    items.append((f"{g}_{hw}_fixed", dict(base, policy={"name": "fixed", "rank": {i: len(Tc) - 1 - i for i in Tc}})))
    for pn in ("P6", "P7", "P6n", "P7n", "P7m"):
        items.append((f"{g}_{hw}_{pn}", dict(base, policy=dict(pp, name=pn), shadow_rank=ref)))
out = []
for name, spec in items:
    res = {}
    for which in ("orig", "fast"):
        e4b_backend.event_driver_e4b = fast_hop.event_driver_fast if which == "fast" else e4b_backend._orig_event_driver_e4b
        sp = os.path.join(E.OUT, "_tmp", f"regress_{name}_{which}.pkl")
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        sink = sp + ".sink"
        pickle.dump(dict(spec, sink=sink), open(sp, "wb"))
        t0 = time.time()
        B.child(sp)
        r = pickle.load(open(sink, "rb"))
        os.remove(sink); os.remove(sp)
        res[which] = (time.time() - t0, hashlib.sha256(pickle.dumps(strip(r), protocol=4)).hexdigest())
    out.append({"item": name, "identical": res["orig"][1] == res["fast"][1], "orig_s": round(res["orig"][0], 1), "fast_s": round(res["fast"][0], 1)})
    print(json.dumps(out[-1]), flush=True)
print("REGRESS " + ("PASSED" if all(o["identical"] for o in out) else "FAILED"))
"""


def regress(wl):
    """E4b's policies through the fast backend (the speed-up regression covered DSTU's): run in the given tree, W0 inputs."""
    R = tree(wl)
    p = subprocess.run([sys.executable, "-c", REGRESS, os.path.join(ROOT, "e4_overlap", "e4b_out")], cwd=os.path.join(R, "e4_overlap"),
                       capture_output=True, text=True, env=env_of(wl))
    open(os.path.join(HERE, "out", f"regress_{wl}.log"), "w", encoding="utf-8").write(p.stdout + p.stderr)
    print(p.stdout[-3000:], p.stderr[-3000:])
    ok = "REGRESS PASSED" in p.stdout
    log(f"{wl}: regression fast vs original on E4b policies {'PASSED' if ok else 'FAILED'}", wl)
    return ok


# ---------------------------------------------------------------- E4a single-stream runs
E4A = r'''
import json, os, sys, time
import e4_overlap as E, e4_grid as G, e4_hop as H
fm = json.loads(open(sys.argv[1]).read())
jobs = int(sys.argv[2])
names = json.loads(open(sys.argv[3]).read())
items = []
for n in names["hop"]:
    k = n[len("single_"):-len("_hop")]
    fa = fm.get(k)
    items.append((n, H.HOP, {"streams": [k], "cfg": E.CFG, "faults": fa and {"failed_nodes": [], "failed_links": [tuple(map(tuple, l)) for l in fa["failed_links"]]},
                             "release": {0: 0}, "seeds": G.D1_SEEDS, "mode": "perhop"}))
for n in names["chunk"]:
    rest = n[len("chunk_single_"):]
    k, cz = rest.rsplit("_", 1)
    fa = fm.get(k)
    items.append((n, H.DIAG, {"streams": [k], "cfg": E.CFG, "faults": fa and {"failed_nodes": [], "failed_links": [tuple(map(tuple, l)) for l in fa["failed_links"]]},
                              "release": {0: 0}, "seeds": G.D1_SEEDS, "mode": "perhop", "chunk": H.CHUNKS[cz]}))
os.makedirs(os.path.join(E.OUT, "logs"), exist_ok=True)
t0 = time.time()
res = H.run_all(jobs, items)
keys = [n[len("single_"):-len("_hop")] for n in names["hop"]]
Thop = {k: E._mean([r["C"][0] - r["release"][0] for r in res[f"single_{k}_hop"]["runs"]]) for k in keys}
Thop_seeds = {k: [r["C"][0] - r["release"][0] for r in res[f"single_{k}_hop"]["runs"]] for k in keys}
E._write_json(os.path.join("hop", "hop_state.json"), {"T_hop": Thop, "T_hop_seeds": Thop_seeds, "single_seeds": G.D1_SEEDS,
              "note": "W1 rerun tree: only E4a's single-stream runs (grid configurations not rerun)", "singles_wall": time.time() - t0,
              "command": sys.argv, "artifact_commit": E._artifact_commit()})
print(json.dumps(Thop))
'''


def e4a(wl, jobs):
    R = tree(wl)
    src = os.path.join(ROOT, "e4_overlap", "e4_out")
    names = {"hop": sorted(f[:-7] for f in os.listdir(os.path.join(src, "hop", "sims")) if f.startswith("single_") and f.endswith("_hop.pkl.gz")),
             "chunk": sorted(f[:-7] for f in os.listdir(os.path.join(src, "diag", "sims")) if f.startswith("chunk_single_"))}
    fm = {k: v[1] for k, v in fault_map(R).items()}
    tmp = os.path.join(HERE, "out", f"_e4a_{wl}")
    os.makedirs(tmp, exist_ok=True)
    json.dump({k: {"failed_links": v["failed_links"]} for k, v in fm.items()}, open(os.path.join(tmp, "fm.json"), "w"))
    json.dump(names, open(os.path.join(tmp, "names.json"), "w"))
    log(f"{wl}: E4a singles: {len(names['hop'])} hop, {len(names['chunk'])} chunked", wl)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", E4A, os.path.join(tmp, "fm.json"), str(jobs), os.path.join(tmp, "names.json")],
                       cwd=os.path.join(R, "e4_overlap"), capture_output=True, text=True, env=env_of(wl))
    open(os.path.join(HERE, "out", f"e4a_{wl}.log"), "w", encoding="utf-8").write(p.stdout + p.stderr)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-3000:])
        raise SystemExit(f"E4a singles failed ({wl})")
    log(f"{wl}: E4a singles done, {time.time() - t0:.0f} s; T_hop {p.stdout.strip().splitlines()[-1][:300]}", wl)


# ---------------------------------------------------------------- the pipeline
MEM_STOP_GB = 3.0


def free_gb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30

def pool_b(R):
    """dstu/PROGRESS.md 9.2: keep pool B when, on no fault or F4 (Delta 0), all-link 64 KB beats D2D 64 KB or all-link 256 KB beats
    D2D 256 KB by >= 0.04 in L_hat. Read from the tree's pilot output (decided by dstu2.py's own pilot report)."""
    p = os.path.join(R, "dstu", "out", "stage2", "pilot.json")
    d = json.load(open(p, encoding="utf-8"))
    for k in ("pool_B_kept",):
        if k in d:
            return bool(d[k])
    raise SystemExit(f"pool B decision not found in {p}: decide by hand (keys {list(d)[:20]})")


def pipe(wl, start, stop_at=None):
    R = tree(wl)
    for i, (cwd, cmd) in enumerate(PIPE):
        if i < start or (stop_at is not None and i >= stop_at):
            continue
        if "POOL_B" in cmd:
            cmd = cmd.replace("POOL_B", "--pool-b" if pool_b(R) else "").strip()
        stamp = os.path.join(HERE, "out", f"pipe_{wl}_done", f"{i:02d}")
        if os.path.exists(stamp):
            log(f"{wl} [{i}] {cmd}: done before, skipped", wl)
            continue
        log(f"{wl} [{i}] start: (cwd {cwd}) python {cmd}", wl)
        t0 = time.time()
        lf = os.path.join(HERE, "out", f"pipe_{wl}_{i:02d}.out")
        low = [0, 99.0]
        with open(lf, "w", encoding="utf-8") as f:
            proc = subprocess.Popen([sys.executable] + cmd.split(), cwd=os.path.join(R, cwd), stdout=f, stderr=subprocess.STDOUT, env=env_of(wl))
            while proc.poll() is None:
                time.sleep(30)
                fr = free_gb()
                low[1] = min(low[1], fr)
                low[0] = low[0] + 1 if fr < MEM_STOP_GB else 0
                if low[0] >= 2:          # memory stop line: kill the step (and its children), stop the pipeline
                    subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    log(f"{wl} [{i}] {cmd}: STOP, free memory {fr:.1f} GB < {MEM_STOP_GB} GB twice; step killed", wl)
                    raise SystemExit(1)
            rc = proc.returncode
        wall = time.time() - t0
        log(f"{wl} [{i}] min free memory during the step {low[1]:.1f} GB", wl)
        tail = open(lf, encoding="utf-8", errors="replace").read().splitlines()[-3:]
        if rc != 0 and cmd in SOFT_STOP and any("E4bError" in x or x.startswith("[e4b] STOP:") for x in tail):
            log(f"{wl} [{i}] {cmd}: ended with the self-check stop (as under W0), {wall:.0f} s: {tail[-1][:300]}", wl)
        elif rc != 0:
            log(f"{wl} [{i}] {cmd}: FAILED (exit {rc}), {wall:.0f} s: {' | '.join(tail)[:500]}", wl)
            raise SystemExit(1)
        else:
            log(f"{wl} [{i}] {cmd}: ok, {wall:.0f} s", wl)
        os.makedirs(os.path.dirname(stamp), exist_ok=True)
        open(stamp, "w").write(f"{rc} {wall:.0f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check", "regress", "e4a", "pipe"])
    ap.add_argument("wl", choices=WLS)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--to", dest="stop_at", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.wl)
    elif a.cmd == "check":
        sys.exit(0 if check(a.wl) else 1)
    elif a.cmd == "regress":
        sys.exit(0 if regress(a.wl) else 1)
    elif a.cmd == "e4a":
        e4a(a.wl, a.jobs)
    elif a.cmd == "pipe":
        pipe(a.wl, a.start, a.stop_at)


if __name__ == "__main__":
    main()
