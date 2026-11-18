"""S256 supplement: fixed saved states, exact input reuse and paired executions."""
import argparse
import collections
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'

import fg_fault as F
import hi_closeout_s as S
import post_fg_dstu as D

ROOT = F.ROOT
BASE = ROOT / 'e2e/dstu_supplement'
OUT = BASE / 'out/main'
VERSION = 'dstu_supplement_v2'
PHASES = (.1, .5, .9)
ARMS = ('BB0', 'A256', 'S256')
SEEDS = (1, 2, 3)


def digest(value):
    return hashlib.sha256(json.dumps(F.plain(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def contexts():
    rows = []
    for hardware in ('wsc', 'face'):
        for state in ('A3hEIR', 'A3hR0', 'A1R0', 'A1EIR'):
            rows.append(dict(scope='S', model='gpt', hardware_profile=hardware, mapping_state=state))
    for scope, model in (('H', 'llama'), ('I', 'qwen3moe')):
        for state in ('healthy_A3h', 'D_A3hEIR'):
            rows.append(dict(scope=scope, model=model, hardware_profile='busybarn', mapping_state=state))
    return rows


def point_name(spec):
    return '_'.join(spec[k] for k in ('scope', 'hardware_profile', 'mapping_state'))


def context(spec):
    phase, seed = spec.get('phase', .1), spec.get('seed', 1)
    if spec['scope'] == 'S':
        old = dict(S.base_spec(spec['hardware_profile']), kind='s_joint', transport=spec.get('transport', 'S256'),
                   graph_job=f"S_{spec['hardware_profile']}__graph_{spec['mapping_state']}",
                   mapping_state=spec['mapping_state'], phase=phase, seed=seed)
        value = S.simulation_spec(old, S.OUT)
    else:
        import dstu_supplement_hi as H
        value = H.context(spec['model'], spec['mapping_state'], phase, seed)
        value['frozen_C4_deadlines'] = value['frozen_A256_deadlines']
    workload = value['workload_version'] if spec['scope'] != 'S' else 'W1'
    value.update(spec)
    value['workload_version'] = workload
    value.update(arm=spec.get('transport', 'S256'), source_root=str(ROOT),
                 run_type=spec.get('run_type', 'joint'), fresh_protocol='fresh_transform_v1',
                 reference={k: value['reference'][k] for k in ('T_P', 'T_D', 'KV_start', 'KV_end')})
    if spec.get('rank_mode') == 'recomputed':
        prep = F.read_json(spec['prepared_record'])
        value['rank'] = prep['rank_checks'][str(phase)]['recomputed_rank']
    return value


@contextmanager
def adapter(spec):
    if spec['scope'] == 'S':
        with S.simulation_adapter():
            yield
    else:
        import dstu_supplement_hi as H
        with H.simulation_adapter():
            yield


def graph_identity(spec, events, hp, release):
    identity = D.evaluation_identity(spec, events, hp, release)
    return dict(evaluation_identity=identity,
                initial_order_sha256=D.execution_order_digest(events, identity['graph_sha256']))


def mask_identity(metadata):
    return digest([dict(tag=tag, **value) for tag, value in metadata.items()])


def prepare(spec, out):
    inner = context(spec)
    with adapter(spec):
        d, hp, graph, streams = D.graph_inputs(inner, out)
        source_ids = {str(s): dict(semantic=D.graph_digest(ev), iteration=D.execution_order_digest(ev))
                      for s, ev in streams.items()}
        # chunk_by changes its input events. Reload the saved pickle for each
        # independent transport transform, preserving its actual set iteration.
        def fresh_streams():
            return {s: F.read_pickle(graph['files'][f'orig_{s}']) for s in (0, 1)}
        # That_hop builds its own activity graph; the saved event input is unchanged.
        masks, rank_checks, identities = {}, {}, {}
        cpcm_calls = 0
        for phase in PHASES:
            current = context(dict(spec, phase=phase))
            rank = {int(k): v for k, v in current['rank'].items()}
            key = digest(rank)
            if key not in masks:
                transformed, metadata, _, stats = D.transform(fresh_streams(), hp, 'S256', rank)
                before = D.execution_order_digest(transformed)
                predictions, prediction_seconds = {}, {}
                for stream in (0, 1):
                    filtered = {t: e for t, e in transformed.items() if t[1] - 10 == stream}
                    predicted, seconds, _ = d.that_hop(filtered, hp)
                    predictions[str(stream)], prediction_seconds[str(stream)] = int(predicted), seconds
                    cpcm_calls += 1
                assert D.execution_order_digest(transformed) == before
                masks[key] = dict(rank=rank, that=predictions, cpcm_seconds=prediction_seconds,
                                  mask_sha256=mask_identity(metadata), stats=stats)
            item = masks[key]
            deadlines = {s: 1.5 * item['that'][str(s)] + (current['tau'] if s else 0) for s in (0, 1)}
            order = sorted(deadlines, key=lambda s: (deadlines[s], s))
            normal = {s: order.index(s) for s in order}
            check = dict(frozen_rank=rank, recomputed_rank=normal, same=rank == normal,
                         A256_deadlines=current['frozen_C4_deadlines'], S256_deadlines=deadlines,
                         mask_sha256=item['mask_sha256'], tau=current['tau'])
            if normal != rank:
                transformed, metadata, _, stats = D.transform(fresh_streams(), hp, 'S256', normal)
                that = {s: int(d.that_hop({t: e for t, e in transformed.items() if t[1]-10 == s}, hp)[0])
                        for s in (0, 1)}
                cpcm_calls += 2
                second_deadlines = {s: 1.5 * that[s] + (current['tau'] if s else 0) for s in (0, 1)}
                second_order = sorted(second_deadlines, key=lambda s: (second_deadlines[s], s))
                second_rank = {s: second_order.index(s) for s in second_order}
                check.update(recomputed_mask_sha256=mask_identity(metadata), second_rank=second_rank,
                             flips_again=second_rank != normal, recomputed_chunk_stats=stats)
            rank_checks[str(phase)] = check
            for arm in ARMS:
                arm_spec = dict(current, arm=arm)
                events, metadata, _, stats = D.transform(fresh_streams(), hp, arm, rank)
                joint = graph_identity(arm_spec, events, hp, {0: 0, 1: current['tau']})
                solos = {str(s): graph_identity(arm_spec, {t: e for t, e in events.items() if t[1]-10 == s}, hp, {s: 0})
                         for s in (0, 1)}
                identities[f'{arm}_g{phase}'] = dict(joint=joint, solos=solos,
                    mask_sha256=mask_identity(metadata), chunk_stats=stats)
        files = [inner['graph_record']] + [graph['files'][f'orig_{s}'] for s in (0, 1)]
        return dict(status='ok', backend_calls=0, cpcm_calls=cpcm_calls,
                    source_graph_identity=source_ids, rank_checks=rank_checks, masks=masks,
                    identities=identities, fault=graph['fault'], reference=inner['reference'],
                    source_inputs=[dict(path=p, sha256=D.sha(F.path(p))) for p in files],
                    input_protocol='fresh_transform_v1', resolved_simulation_spec=inner)


def old_source(spec):
    if spec['scope'] != 'S':
        return None
    hw, state, arm = (spec[k] for k in ('hardware_profile', 'mapping_state', 'transport'))
    if arm == 'S256':
        if state != 'A3hEIR':
            return None
        suffix = f"g{spec['phase']}_s{spec['seed']}" if spec['run_type'] == 'joint' else f"solo{spec['stream']}_s{spec['seed']}"
        return S.POST / 'results' / f'dstu_{hw}_D_S256_{suffix}.json'
    if spec['run_type'] == 'joint':
        bits = ('1' if state.startswith('A3h') else '0') + ('1' if state.endswith('EIR') else '0') + ('1' if arm == 'A256' else '0')
        suffix = f"joint_{bits}_g{spec['phase']}_s{spec['seed']}"
    else:
        suffix = f"solo_{state}_{arm}_{spec['stream']}_s{spec['seed']}"
    return S.OUT / 'results' / f'S_{hw}__{suffix}.json'


def reuse_matches(old, inner, identity):
    if old['evaluation_identity'] != F.plain(identity['evaluation_identity']):
        return False
    if old['initial_order_sha256'] != identity['initial_order_sha256']:
        return False
    if old['fresh_protocol'] != 'fresh_transform_v1':
        return False
    old_spec = old.get('resolved_simulation_spec', old['spec'])
    if old_spec['reference'] != inner['reference']:
        return False
    return True


def evaluate(spec, out):
    inner = context(spec)
    with adapter(spec):
        _, hp, _, streams = D.graph_inputs(inner, out)
        events, metadata, _, stats = D.transform(streams, hp, inner['arm'], inner['rank'])
        mask = mask_identity(metadata)
        if inner['run_type'] == 'solo':
            events = {t: e for t, e in events.items() if t[1]-10 == inner['stream']}
            release = {inner['stream']: 0}
        else:
            release = {0: 0, 1: inner['tau']}
        identity = graph_identity(inner, events, hp, release)
        candidates = []
        old = old_source(spec)
        if old is not None and spec['rank_mode'] == 'frozen':
            candidates.append(('historical', old))
        candidates.extend(('within_batch', F.result_path(out, name)) for name in spec.get('prior_solos', []))
        result, attempts = None, []
        for source_kind, filename in candidates:
            previous = F.read_json(filename)
            same = reuse_matches(previous, inner, identity)
            attempts.append(dict(source=F.relative(filename), sha256=D.sha(filename), same_execution=same))
            if same:
                result = dict(previous, status='reused', backend_calls=0, simulation_seconds=0,
                              source_simulation_seconds=previous['simulation_seconds'], reused_from=F.relative(filename),
                              reused_sha256=D.sha(filename), reuse_kind=source_kind, reuse_identity_checked=True)
                break
        if result is None:
            result = D.simulate(inner, out)
            assert result['evaluation_identity'] == F.plain(identity['evaluation_identity'])
            assert result['initial_order_sha256'] == identity['initial_order_sha256']
            result.update(status='ok', reuse_kind='new')
        result.update(resolved_simulation_spec=inner, source_graph_sha256=inner['graph_sha256'],
                      transform_version=D.VERSION, transform_source_sha256=D.sha(ROOT/'e2e/post_fg_dstu.py'),
                      mask_sha256=mask, chunk_stats=stats, reuse_checks=attempts, cpcm_calls=0)
        if spec['run_type'] == 'joint':
            ps = F.read_json(F.result_path(out, spec['P_solo']))['completion']['0']
            ds = F.read_json(F.result_path(out, spec['D_solo']))['completion']['1']
            result['metrics'] = dict(result['metrics'], P_solo=ps, D_solo=ds,
                paired_delta_P=result['metrics']['P_completion']-ps,
                D_interference=result['metrics']['D_latency']/ds-1)
        return result


def child(spec, out):
    import e2e_pilot_peak
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if spec['kind'] == 'evaluate':
        assert D.sha(F.path(spec['gate_record'])) == spec['gate_sha256']
    result = prepare(spec, out) if spec['kind'] == 'prepare' else evaluate(spec, out)
    result.update(job_id=spec['job_id'], spec=spec, protocol_version=VERSION,
                  wall_seconds=time.perf_counter()-started, peak_rss_gb=e2e_pilot_peak.peak()/2**30,
                  started_at_utc=started_at, finished_at_utc=datetime.now(timezone.utc).isoformat())
    F.write_json(F.result_path(out, spec['job_id']), result)
    print(json.dumps({k: result[k] for k in ('job_id', 'status', 'backend_calls', 'wall_seconds', 'peak_rss_gb')}), flush=True)


def as_job(spec):
    return dict(id=spec['job_id'], kind=spec['kind'], spec=spec, deps=spec.get('depends_on', []),
                result=f"results/{spec['job_id']}.json", peak_rss_gb=1.8, estimated_seconds=40,
                argv=['{python}', '-u', '-B', 'e2e/dstu_supplement_run.py', 'child', '--spec', '{spec}', '--out', '{out}'])


def plan_prepare(destination):
    jobs = []
    for base in contexts():
        spec = dict(base, job_id=point_name(base)+'__prepare', kind='prepare', protocol_version=VERSION,
                    module='dstu_supplement_run', workload_version='W1', trace=False)
        jobs.append(as_job(spec))
    F.write_json(destination, dict(version=VERSION, jobs=jobs, backend_calls=0, graph_preparations=len(jobs)))
    return jobs


def plan_evaluations(out, destination):
    jobs, prior, counts = [], collections.defaultdict(list), collections.Counter()
    gate_summary = F.read_json(BASE/'evidence/gate_decisions.json')
    for base in contexts():
        prep_file = F.result_path(out, point_name(base)+'__prepare')
        prep = F.read_json(prep_file)
        gate_file = F.result_path(out, point_name(base)+'__gate')
        gate = F.read_json(gate_file)
        assert gate['status'] == 'ok' and len(gate['decisions']) == 3 and gate['backend_calls'] == 0
        gate_fields = dict(gate_record=F.relative(gate_file), gate_sha256=D.sha(gate_file))
        ranks = prep['rank_checks']
        assert all(row['same'] or 'recomputed_mask_sha256' in row for row in ranks.values())
        for arm in ARMS:
            modes = ['frozen'] + (['recomputed'] if arm == 'S256' and any(not r['same'] for r in ranks.values()) else [])
            for mode in modes:
                phase_solos, unique_solos = {}, {}
                for phase in PHASES:
                    if mode == 'recomputed' and ranks[str(phase)]['same']:
                        continue
                    rank = ranks[str(phase)]['frozen_rank' if mode == 'frozen' else 'recomputed_rank']
                    for stream in (0, 1):
                        for seed in SEEDS:
                            key = digest(rank), stream, seed
                            if key not in unique_solos:
                                name = f'{point_name(base)}__{arm}_{mode}_solo{stream}_g{phase}_s{seed}'
                                previous = prior[base['model'], base['hardware_profile'], arm, mode, stream, seed]
                                spec = dict(base, job_id=name, kind='evaluate', protocol_version=VERSION,
                                    module='dstu_supplement_run', workload_version='W1' if base['scope']=='S' else 'W1_8Q_1KV_v1', transport=arm, rank_mode=mode,
                                    run_type='solo', stream=stream, seed=seed, phase=phase, trace=False,
                                    prepared_record=F.relative(prep_file), prior_solos=list(previous),
                                    depends_on=list(previous)+[point_name(base)+'__gate'], **gate_fields)
                                jobs.append(as_job(spec)); previous.append(name); unique_solos[key] = name
                                counts[f'{arm}_solo_logical'] += 1
                            phase_solos[phase, stream, seed] = unique_solos[key]
                for phase in PHASES:
                    if mode == 'recomputed' and ranks[str(phase)]['same']:
                        continue
                    for seed in SEEDS:
                        name = f'{point_name(base)}__{arm}_{mode}_g{phase}_s{seed}'
                        ps, ds = (phase_solos[phase, s, seed] for s in (0, 1))
                        trace = base['scope'] != 'S' and phase == .1 and seed == 1
                        spec = dict(base, job_id=name, kind='evaluate', protocol_version=VERSION,
                            module='dstu_supplement_run', workload_version='W1' if base['scope']=='S' else 'W1_8Q_1KV_v1', transport=arm, rank_mode=mode,
                            run_type='joint', phase=phase, seed=seed, trace=trace,
                            prepared_record=F.relative(prep_file), P_solo=ps, D_solo=ds,
                            depends_on=[ps, ds, point_name(base)+'__gate'], **gate_fields)
                        jobs.append(as_job(spec)); counts[f'{arm}_joint_logical'] += 1
    F.write_json(destination, dict(version=VERSION, jobs=jobs, counts=dict(counts),
        gate_decisions_sha256=D.sha(BASE/'evidence/gate_decisions.json'),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        meaning='Logical records include exact-identity reused historical baselines. Actual backend calls counted in results.'))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('plan-prepare', 'plan-evaluate', 'child'))
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--spec', type=Path)
    p.add_argument('--manifest', type=Path)
    a = p.parse_args()
    if a.command == 'child':
        child(F.read_json(a.spec), a.out)
    else:
        name = 'prepare.json' if a.command == 'plan-prepare' else 'evaluate.json'
        target = a.manifest or BASE/'manifests'/name
        jobs = plan_prepare(target) if a.command == 'plan-prepare' else plan_evaluations(a.out, target)
        print(json.dumps(dict(jobs=len(jobs), manifest=str(target))))


if __name__ == '__main__':
    main()
