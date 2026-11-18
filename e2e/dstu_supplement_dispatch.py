"""Admit a measured seed-one batch, then the remainder of the approved supplement."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import statistics

import fg_fault as F
import post_fg_dstu as D
from dstu_supplement_run import BASE, OUT, plan_evaluations


def registry(manifest, accepting):
    F.write_json(BASE/'manifests/registry.json', dict(manifests=[
        'e2e/dstu_supplement/manifests/prepare.json',
        'e2e/dstu_supplement/manifests/gate.json',
        F.relative(manifest)], accepting=accepting))


def pilot():
    import dstu_supplement_gate as gate
    frozen = gate.collect(OUT)
    assert len(frozen['frozen_decisions']) == 36
    complete = BASE/'manifests/evaluate.json'
    jobs = plan_evaluations(OUT, complete)
    picked = [job for job in jobs if job['spec']['seed'] == 1 and
              (job['spec']['run_type'] == 'solo' or job['spec']['phase'] == .1)]
    ids = {job['id'] for job in picked}
    assert all(dep in ids or dep.endswith('__gate') for job in picked for dep in job['deps'])
    filename = BASE/'manifests/evaluate_pilot.json'
    F.write_json(filename, dict(jobs=picked, includes_in_full_batch=True,
        full_manifest_sha256=D.sha(complete), gate_decisions_sha256=D.sha(BASE/'evidence/gate_decisions.json')))
    registry(filename, True)
    print(json.dumps(dict(pilot_jobs=len(picked), full_jobs=len(jobs),
                         online_choices=frozen['counts'], frozen_before_backend=True)))


def remainder():
    pilot_jobs = F.read_json(BASE/'manifests/evaluate_pilot.json')['jobs']
    grouped, actual = defaultdict(list), Counter()
    for job in pilot_jobs:
        result = F.read_json(F.result_path(OUT, job['id']))
        receipt = F.read_json(OUT/'receipts'/(job['id']+'.json'))
        assert result['status'] in ('ok', 'reused') and receipt['rc'] == 0
        spec = result['spec']
        kind = (spec['model'], spec['hardware_profile'], spec['transport'], spec['run_type'], result['reuse_kind'])
        grouped[kind].append(result)
        actual[result['reuse_kind']] += 1
    estimates = [dict(category=list(key), samples=len(rows),
                     wall_seconds_median=statistics.median(row['wall_seconds'] for row in rows),
                     wall_seconds_max=max(row['wall_seconds'] for row in rows),
                     peak_rss_gib=max(row['peak_rss_gb'] for row in rows),
                     backend_calls=sum(row['backend_calls'] for row in rows)) for key, rows in grouped.items()]
    F.write_json(BASE/'evidence/pilot_measurements.json', dict(status='passed', jobs=len(pilot_jobs),
        accounting=dict(actual), estimates=estimates, updated_at_utc=datetime.now(timezone.utc).isoformat()))
    registry(BASE/'manifests/evaluate.json', False)
    print(json.dumps(dict(pilot_jobs=len(pilot_jobs), accounting=dict(actual),
                         longest_pilot_seconds=max(row['wall_seconds_max'] for row in estimates),
                         max_rss_gib=max(row['peak_rss_gib'] for row in estimates), action='remaining jobs admitted')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('pilot', 'remainder'))
    args = parser.parse_args()
    pilot() if args.command == 'pilot' else remainder()
