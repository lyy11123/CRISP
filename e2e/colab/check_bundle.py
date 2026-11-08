"""Clean-room check of a Colab bundle: unpack it into an empty temporary folder and run from there with `python -I` and no PYTHONPATH,
so a module that is imported but not packed fails here instead of on Colab. Used by build_b4_bundle.py; the W1 rerun bundle uses the same.

  python e2e/colab/check_bundle.py <bundle.tar.gz> [--keep]

Checks (each in its own process, cwd = the unpacked e2e/):
  smoke  e2e_pilot.run_one for each SMOKE run with CPCM_SA_STEPS=15, E2E_BACKEND_SEEDS=2 (search, final evaluation, backend);
         prints the child log tail on failure
  probe  the W1 prefill MHA probe T-hat (deterministic and seed-1 random tie) from the packed snapshot, compared with local_ref.json
Exit code 0 only when every check passes.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

SMOKE = [("dc_ln", "A3h", 0, 123), ("dc_ln", "A1", 0, 0)]

PROBE_CODE = r'''
import sys, json, random
sys.path.insert(0, sys.argv[2])
import e2e_b as EB, e2e_pilot as P
C = P.C
spec = json.load(open(sys.argv[1]))
ctx, beha, data, events, hp = EB.load_state(spec)
added = EB.broadcast(ctx, beha, data, events, hp)
dropped = len(P.w1_drop(events, added))
g = EB.compiled(events, hp)
print(json.dumps(dict(det=C.cpcm.simulate(g, "bb").T, tie1=C.cpcm.simulate(g, "bb", rng=random.Random(1)).T, dropped=dropped)))
'''

RUN_CODE = ("import sys,json;sys.path.insert(0,sys.argv[2]);import e2e_pilot as P;t=json.loads(sys.argv[1]);"
            "print(json.dumps(P.run_one(t[0],t[1],t[2],t[3],False)))")


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", E2E_B4="1", CPCM_SA_STAGE="s2", **extra)
    return env


def check(bundle, keep=False):
    tmp = tempfile.mkdtemp(prefix="bundle_check_")
    ok = True
    try:
        with tarfile.open(bundle, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        e2e = os.path.join(tmp, "e2e")
        os.makedirs(os.path.join(e2e, "out"), exist_ok=True)
        for t in SMOKE:
            name = f"{t[0]}_{t[1]}_g{t[2]}_{'l' if t[1] == 'A1' else 's'}{t[3]}"
            p = subprocess.run([sys.executable, "-I", "-c", RUN_CODE, json.dumps(list(t)), e2e], cwd=e2e, capture_output=True, text=True,
                               env=clean_env(E2E_SUB="check", CPCM_SA_STEPS="15", E2E_BACKEND_SEEDS="2"))
            good = p.returncode == 0 and '"ok"' in p.stdout
            print(f"[check] smoke {name}: {'ok' if good else 'FAILED'}", flush=True)
            if not good:
                ok = False
                print(p.stdout[-2000:], p.stderr[-2000:])
                log = os.path.join(e2e, "out", "check", "logs", name + ".log")
                if os.path.exists(log):
                    print("".join(open(log, encoding="utf-8", errors="replace").readlines()[-80:]))
        ref = json.load(open(os.path.join(e2e, "colab", "local_ref.json"), encoding="utf-8"))["probe"]
        pd = os.path.join(e2e, "colab", "probe")
        sp = os.path.join(tmp, "probe_spec.json")
        json.dump(dict(cfg="pf_mha_w1", run=os.path.join(pd, "run.pkl.gz"), snap=os.path.join(pd, "snap.pkl.gz")), open(sp, "w"))
        p = subprocess.run([sys.executable, "-I", "-c", PROBE_CODE, sp, e2e], cwd=e2e, capture_output=True, text=True, env=clean_env(E2E_SUB="probe"))
        if p.returncode:
            ok = False
            print("[check] probe FAILED\n", p.stdout[-2000:], p.stderr[-3000:])
        else:
            r = json.loads(p.stdout.strip().splitlines()[-1])
            same = r["det"] == ref["det"] and r["tie1"] == ref["tie1"] and r["dropped"] == ref["dropped"]
            ok &= same
            print(f"[check] probe det {r['det']} tie1 {r['tie1']} dropped {r['dropped']}: {'identical' if same else 'DIFFERS'} from local_ref", flush=True)
    finally:
        if keep:
            print(f"[check] kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("[check] BUNDLE CHECK PASSED" if ok else "[check] BUNDLE CHECK FAILED", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1], "--keep" in sys.argv) else 1)
