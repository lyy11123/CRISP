"""Run the supplement with the existing resumable 128-worker scheduler."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import hi_closeout_pool as C
from dstu_supplement_run import BASE, OUT, ROOT

C.BASE = BASE
C.OUT = OUT
C.REGISTRY = BASE / 'manifests/registry.json'


def category(job):
    spec = job['spec']
    return tuple(spec.get(key) for key in ('module', 'kind', 'model', 'hardware_profile', 'transport', 'run_type'))


C.category = category


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('check', 'launch', 'run'))
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--workers', type=int, default=128)
    args = p.parse_args()
    assert 1 <= args.workers <= 128
    if args.command == 'check':
        jobs, accepting = C.registered()
        compile(C.previous.wrapped_pool_source(), '<pool>', 'exec')
        print(json.dumps(dict(jobs=len(jobs), accepting=accepting, workers=args.workers, reserve_gib=3)))
        return 0
    if args.command == 'run':
        return C.Controller(args.out, args.workers).run()
    args.out.mkdir(parents=True, exist_ok=True)
    start = args.out / 'supplement.start.json'
    if start.exists():
        assert not C.pool.process_alive(C.previous.read(start)['pid'])
    C.write_json(args.out/'scheduler_policy.json', dict(running_growth_fraction=1.0))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMBA_NUM_THREADS='1', PYTHONUTF8='1', CRISP_E_ARTIFACT=str(ROOT/'e2e/work/artifact'))
    cmd = [sys.executable, '-u', '-B', 'e2e/dstu_supplement_pool.py', 'run', '--out', str(args.out), '--workers', str(args.workers)]
    options = dict(start_new_session=True) if sys.platform != 'win32' else dict(creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    with (args.out/'controller.log').open('ab') as handle:
        proc = C.previous.RAW_POPEN(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, **options)
    C.write_json(start, dict(pid=proc.pid, command=cmd, utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(pid=proc.pid, out=str(args.out))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
