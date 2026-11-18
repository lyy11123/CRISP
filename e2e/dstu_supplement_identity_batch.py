"""Twelve independent saved-input checks; no backend or CPCM invocation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import subprocess
import sys

from dstu_supplement_run import BASE, ROOT, contexts, point_name
from fg_fault import write_json

OUT = BASE/'out/identity_check'


def check(spec):
    name = point_name(spec)
    started = datetime.now(timezone.utc).isoformat()
    with (OUT/(name+'.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-u', '-B', 'e2e/dstu_supplement_identity.py',
            'run', '--point', name, '--out', str(OUT)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    receipt = dict(point=name, rc=result.returncode, started_at_utc=started,
                   ended_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(OUT/(name+'.receipt.json'), receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['CRISP_E_ARTIFACT'] = str(ROOT/'e2e/work/artifact')
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(check, contexts()))
    write_json(OUT/'finish.json', dict(results=results, backend_calls=0, cpcm_calls=0,
                                      status='passed' if all(row['rc']==0 for row in results) else 'failed'))
    raise SystemExit(int(any(row['rc'] for row in results)))
