"""Launch a detached post-F/G batch and record its command and source version."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from e_run import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", default="e2e/post_fg/out/main")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--memory-hint-gib", type=float, default=2.0)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "scheduler_policy.json", dict(running_growth_fraction=0.25))
    env = dict(os.environ)
    settings = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                    NUMBA_NUM_THREADS="1", CRISP_E_ARTIFACT=str(ROOT / "e2e/work/artifact"))
    env.update(settings)
    command = [sys.executable, "-u", "-B", "e2e/fg_batch.py", "--manifest", args.manifest,
               "--out", str(out), "--workers", str(args.workers), "--reserve-gib", "3",
               "--memory-hint-gib", str(args.memory_hint_gib), "--name", args.name]
    log = out / f"{args.name}.log"
    with log.open("ab") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(name=args.name, pid=process.pid, command=command, cwd=str(ROOT),
                  created_utc=datetime.now(timezone.utc).isoformat(), log=str(log),
                  environment=settings, python=sys.version,
                  packages={n:importlib.metadata.version(n) for n in ("numpy", "numba", "llvmlite")},
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT / "e2e").glob("post_fg_*.py")})
    write_json(ROOT / f"{args.name}.start.json", record)
    print(f"{args.name}: PID {process.pid}, log {log}", flush=True)


if __name__ == "__main__":
    main()
