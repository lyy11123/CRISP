"""Detached H/I/J batch, using the existing memory-aware process pool."""
import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from e_run import ROOT,write_json


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--workers',type=int,default=128)
    parser.add_argument('--memory-hint-gib',type=float,default=4)
    args=parser.parse_args()
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'scheduler_policy.json',dict(running_growth_fraction=.25))
    settings=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
                  CRISP_E_ARTIFACT=str(ROOT/'e2e/work/artifact'),PYTHONUTF8='1')
    command=[sys.executable,'-u','-B','e2e/fg_batch.py','--manifest',args.manifest,'--out',str(out),
             '--workers',str(args.workers),'--reserve-gib','3','--memory-hint-gib',str(args.memory_hint_gib),'--name',args.name]
    with (out/(args.name+'.log')).open('ab') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,**settings),stdin=subprocess.DEVNULL,
                                 stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/(args.name+'.start.json'),dict(pid=process.pid,command=command,cwd=str(ROOT),
               created_utc=datetime.now(timezone.utc).isoformat(),environment=settings,
               source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'e2e').glob('hi_*.py')}))
    print(f'{args.name}: PID {process.pid}, output {out}',flush=True)
