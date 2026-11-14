"""Start the remaining J timings only after the quality queue has finished."""
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import hi_closeout_pool as shared
import hi_closeout_j_micro as micro
from e_run import ROOT,write_json

OUT=ROOT/"e2e/hi/out/closeout_j_20261115"
QUALITY=ROOT/"e2e/hi/out/closeout_20261115"
QUALITY_REGISTRY=ROOT/"e2e/hi/closeout/manifests/registry.json"
REGISTRY=ROOT/"e2e/hi/closeout/manifests/j_registry.json"


def ready():
    done=shared.previous.read(QUALITY/"controller_finish.json")
    assert done["rc"]==0 and done["completed"]==done["expected"]==549,done
    quality_registry=shared.previous.read(QUALITY_REGISTRY)
    assert quality_registry["accepting"] is False
    quality_jobs=[j for path in quality_registry["manifests"] for j in shared.previous.RAW_JOB_ITEMS(ROOT/path)]
    assert len(quality_jobs)==len({j["id"] for j in quality_jobs})==549
    assert not shared.previous.read(QUALITY/"live_workers.json")
    quality_pid=shared.previous.read(QUALITY/"closeout.start.json")["pid"]
    assert not shared.pool.process_alive(quality_pid)
    identities={}
    for job in quality_jobs:
        receipt=shared.previous.read(QUALITY/"receipts"/(job["id"]+".json"))
        identity=shared.pool.identity(job)
        assert receipt["rc"]==0 and receipt["id"]==job["id"] and receipt["identity"]==identity,job["id"]
        result=shared.previous.read(QUALITY/job["result"])
        assert result["status"] in ("ok","reused","complete","completed"),job["id"]
        assert result.get("job_id",result.get("candidate_id",job["id"]))==job["id"],job["id"]
        if "spec" in result:assert result["spec"]==job["spec"],job["id"]
        identities[job["id"]]=identity
    prepared=micro.plan(QUALITY)
    assert prepared["J1_ready"]==18 and prepared["J5_ready"]==6
    write_json(REGISTRY,dict(accepting=False,manifests=[
        "e2e/hi/closeout/manifests/j_micro.json","e2e/hi/closeout/manifests/j_search.json",
        "e2e/hi/closeout/manifests/j3.json"]))
    shared.REGISTRY=REGISTRY
    jobs,_=shared.registered()
    counts=Counter(j["kind"] for j in jobs)
    assert len(jobs)==80 and counts==dict(j1=18,j5=6,search=20,fault_l1=9,fault_rep=27),counts
    write_json(OUT/"idle_window_gate.json",dict(status="passed",quality_completion=done,
        quality_jobs=549,quality_registry=quality_registry,quality_job_identities=identities,
        quality_controller_pid=quality_pid,quality_controller_exited=True,j_kind_counts=dict(counts),
        jobs=80,workers=1,reserve_gib=3,utc=datetime.now(timezone.utc).isoformat(),
        quality_workers_remaining=0,old_formal_timing_jobs_preserved=45))


def verify_gate():
    gate=shared.previous.read(OUT/"idle_window_gate.json")
    assert gate["status"]=="passed" and gate["quality_jobs"]==549
    assert gate["jobs"]==80 and gate["workers"]==1
    assert gate["quality_completion"]==shared.previous.read(QUALITY/"controller_finish.json")
    assert gate["quality_controller_exited"] and gate["quality_workers_remaining"]==0


def wait_for_quality():
    while not (QUALITY/"controller_finish.json").exists():time.sleep(30)
    while shared.pool.process_alive(shared.previous.read(QUALITY/"closeout.start.json")["pid"]):time.sleep(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("wait","launch","run"))
    a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    shared.REGISTRY=REGISTRY
    if a.command=="run":
        verify_gate()
        return shared.Controller(OUT,1).run()
    if a.command=="wait":wait_for_quality()
    ready()
    start=OUT/"closeout.start.json"
    if start.exists():assert not shared.pool.process_alive(shared.previous.read(start)["pid"])
    write_json(OUT/"scheduler_policy.json",dict(running_growth_fraction=1.))
    env=dict(os.environ,CRISP_E_ARTIFACT=str(ROOT/"e2e/work/artifact"),PYTHONUTF8="1")
    cmd=[sys.executable,"-u","-B","e2e/hi_closeout_j_pool.py","run"]
    options=dict(start_new_session=True) if sys.platform!="win32" else dict(creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    with (OUT/"closeout.log").open("ab") as stream:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,**options)
    write_json(start,dict(pid=proc.pid,command=cmd,workers=1,utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(pid=proc.pid,jobs=80,workers=1)),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
