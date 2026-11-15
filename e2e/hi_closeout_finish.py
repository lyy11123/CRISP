"""Finish the closeout from the saved COS downloads; no experiment is launched."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv, gzip, hashlib, json, pickle, statistics
import hi_closeout_j_report as report

ROOT=report.ROOT
BASE=report.BASE
OUT=ROOT/'e2e/hi/out/closeout_j_20261115'
SOURCES={}
def read(path):
    path=Path(path); data=path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()]=dict(path=path.relative_to(ROOT).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    return json.loads(data)
def blob(path):
    path=Path(path); data=path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()]=dict(path=path.relative_to(ROOT).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    return pickle.loads(gzip.decompress(data))
def main():
    import e_gemini_analysis as saved_types
    saved_types.ARTIFACT=ROOT/"third_party/isca2026-busybarn-artifact"
    saved_types.pickle_types()
    finish=read(OUT/'controller_finish.json'); batch=read(OUT/'closeout.batch.json')
    start=read(OUT/'closeout.start.json')
    assert finish['rc']==0 and finish['completed']==finish['expected']==batch['completed']==80
    assert batch['failed']==batch['pending']==[] and batch['max_active']==1
    assert read(OUT/'live_workers.json')=={}
    entries=[]; checked_code={}
    for path in sorted((OUT/'results').glob('*.json')):
        r=read(path); name=r['job_id']; receipt=read(OUT/'receipts'/(name+'.json'))
        assert receipt['rc']==0 and r['status']=='ok'
        env=r.get('j_environment',r.get('environment'))
        for key,digest in env['code_sha256'].items():
            folder={'cpcm.py':'cpcm','cpcm_sa.py':'cpcm_sa','eir.py':'eir'}.get(key,'e2e')
            source=ROOT/folder/key
            assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,key
            checked_code[source.relative_to(ROOT).as_posix()]=digest
        row=dict(job_id=name,kind=r['kind'],steps=0,calibration=0,backend=0,random_cpcm=0,wall_seconds=r['wall_seconds'])
        if r['kind']=='search':
            raw=blob(OUT/'raw'/(name+'.pkl.gz'))
            assert len(raw['rows'])==raw['meta']['steps']==r['search_steps']==1000
            assert len(r['backend'])==len(r['T_hat_rand'])==10
            assert r['debug_checks'] is False and r['exact_rollback']
            row.update(steps=1000,calibration=len(raw['meta']['calibration']['dE']),backend=10,random_cpcm=10)
        elif r['kind'] in ('fault_l1','fault_rep'):
            raw=blob(report.local(r['sink'])); t=r['J3_timing']
            assert raw['spec']['checks'] is False
            row['backend']=t['backend_calls']
            if r['kind']=='fault_rep':
                n=r['spec']['steps']; assert len(raw['search']['rows'])==t['search_proposals']==n
                assert raw['T_L1_check']==raw['search']['T_start']
                row.update(steps=n,calibration=t['calibration_proposals'])
            else: assert t['backend_calls']==20
        entries.append(row)
    assert len(entries)==80
    counts=dict(Counter(r['kind'] for r in entries))
    assert counts==dict(j1=18,j5=6,search=20,fault_l1=9,fault_rep=27)
    totals={key:sum(r[key] for r in entries) for key in ['steps','calibration','backend','random_cpcm','wall_seconds']}
    assert totals['steps']==23870 and totals['backend']==650 and totals['random_cpcm']==200
    cumulative=list(csv.DictReader((BASE/'tables/j3_cumulative.csv').open(encoding='utf-8-sig')))
    assert len(cumulative)==27 and all(r['status']=='complete' for r in cumulative)
    oldbase=ROOT/'e2e/fg/out/main/results'
    prefix='G_busybarn_3x3_growth_d2d1__'
    stop_path=oldbase/(prefix+'stop_dc_mha_PF4_100.json'); stop=read(stop_path)
    assert stop['stop'] and stop['reason']=='within_epsilon'
    old_rows=[]
    for seed in (123,1,2):
        p=oldbase/(prefix+f'rep_dc_mha_PF4_s300_r{seed}.json'); old=read(p)
        assert old['status']=='reused' and old['previous_stop']==stop['job_id']; old_rows.append(old)
    audit=dict(status='passed',new_jobs=80,preserved_old_jobs=45,formal_timing_total=125,
        finish=finish,start=start,wall_seconds=(datetime.fromisoformat(finish['utc'])-datetime.fromisoformat(start['utc'])).total_seconds(),
        batch_wall_seconds=batch['wall_seconds'],process_hours=sum(r['wall_seconds'] for r in batch['runs'])/3600,
        max_active=batch['max_active'],min_available_gib=batch['min_available_gib'],peak_job_rss_gib=max(r['peak_rss_gb'] for r in batch['runs']),
        failed_jobs=0,counts=counts,totals=totals,jobs=entries,checked_runtime_sources=checked_code,
        GPT144_historical_300=dict(status='reused_after_stop100',stop=stop,records=old_rows,
            interpretation='J3 measures all independent budgets. Historical quality stopped at 100; new 300-step timing does not replace the quality table.'),
        additional_historical_inputs=[SOURCES[p.relative_to(ROOT).as_posix()] for p in [stop_path]+[oldbase/(prefix+f'rep_dc_mha_PF4_s300_r{s}.json') for s in (123,1,2)]],
        source_inputs=list(SOURCES.values()),new_experiments_in_postprocessing=0)
    report.json_write(BASE/'evidence/j_final_completion_audit.json',audit)
    eta=[]; actual=datetime.fromisoformat(finish['utc'])
    for p in sorted((BASE/'evidence').glob('j_eta_*.json')):
        d=json.loads(p.read_text()); predicted=d.get('estimated_finish_utc')
        eta.append(dict(recorded_utc=d['utc'],completed=d['completed'],predicted_finish_utc=predicted,
            actual_finish_utc=finish['utc'],actual_minus_prediction_minutes=(actual-datetime.fromisoformat(predicted)).total_seconds()/60 if predicted else None,
            remaining_hours=d['remaining_hours']))
    report.csv_write(BASE/'tables/j_eta_comparison.csv',eta)
    print(json.dumps({k:audit[k] for k in ['status','new_jobs','formal_timing_total','wall_seconds','process_hours','counts','totals']}))
if __name__=='__main__': main()
