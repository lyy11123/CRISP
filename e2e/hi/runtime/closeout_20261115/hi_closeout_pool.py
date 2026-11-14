"""Append-only shared queue for the authorized H/I closeout packages."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import fg_batch as pool
import hi_next as previous
from e_run import ROOT,write_json

BASE=ROOT/"e2e/hi/closeout"
REGISTRY=BASE/"manifests/registry.json"
OUT=ROOT/"e2e/hi/out/closeout_20261115"


def category(job):
    s=job["spec"]
    return (s.get("module"),s.get("model"),s.get("start_arm"),s.get("steps"))+previous.RAW_CATEGORY(job)


def registered():
    registry=previous.read(REGISTRY)
    jobs=[j for path in registry["manifests"] for j in previous.RAW_JOB_ITEMS(ROOT/path)]
    assert len(jobs)==len({j["id"] for j in jobs})
    assert {d for j in jobs for d in j.get("deps",[])} <= {j["id"] for j in jobs}
    return jobs,registry["accepting"]


class Controller(previous.Controller):
    def __init__(self,out,workers):
        self.out,self.workers=out,workers
        self.jobs,self.accepting=registered()
        self.by_id={j["id"]:j for j in self.jobs}
        self.initial=set(self.by_id);self.calibrations=[]
        self.success,self.history,self.observed={ },[],defaultdict(list)
        self.live=previous.read(out/"live_workers.json") if (out/"live_workers.json").exists() else {}
        self.manifest=REGISTRY;self.release=out/"release.json"
        self.started=time.time()
        self.last_registry=None

    def refresh(self,jobs,by_id,ids,pending,completed,running,failed):
        signature=REGISTRY.stat().st_mtime_ns
        paths=previous.read(REGISTRY)["manifests"]
        signature=(signature,tuple((ROOT/p).stat().st_mtime_ns for p in paths))
        if signature==self.last_registry:return
        additions,self.accepting=registered()
        for job in additions:
            name=job["id"]
            if name in by_id:
                assert pool.identity(job)==pool.identity(by_id[name]),name
            else:
                jobs.append(job);by_id[name]=job;ids.add(name);pending[name]=job
                self.by_id[name]=job;self.initial.add(name)
        assert set(by_id)<={j["id"] for j in additions},"Queue entries cannot be removed"
        self.jobs=jobs
        self.save_release();self.last_registry=signature
        write_json(self.out/"admitted_jobs.json",dict(jobs=jobs,accepting=self.accepting,utc=datetime.now(timezone.utc).isoformat()))

    def spawn(self,argv,**kwargs):
        spec=previous.read(argv[argv.index("--spec")+1]);name=spec.get("job_id") or spec["candidate_id"]
        assert spec==self.by_id[name]["spec"]
        process=previous.RAW_POPEN(argv,**kwargs)
        self.live[name]=dict(pid=process.pid,started_unix=time.time(),identity=pool.identity(self.by_id[name]),log=str(kwargs["stdout"].name))
        write_json(self.out/"live_workers.json",self.live)
        return process

    def success_receipt(self,receipt):
        name=receipt["id"];job=self.by_id[name]
        assert receipt["identity"]==pool.identity(job)
        result=previous.read(self.out/job["result"])
        assert result.get("job_id",name)==name and result["status"] in ("ok","reused","complete","completed")
        if "spec" in result:assert result["spec"]==job["spec"]
        if name in self.success:return
        self.success[name]=receipt
        if result["status"]!="reused":self.observed[category(job)].append(receipt)
        self.save_release()

    def write(self,path,value):
        path=Path(path)
        if path.parent.name=="receipts":
            if value["rc"]==0:self.success_receipt(value)
            self.live.pop(value["id"],None)
            write_json(self.out/"live_workers.json",self.live)
        if path.name=="closeout.status.json":
            value.update(accepting_new_packages=self.accepting,utc=datetime.now(timezone.utc).isoformat(),
                failed_attempt_count=len(list((self.out/"failed_attempts").glob("*/*/receipt.json"))))
            with (self.out/"monitor_history.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(value)+"\n")
        write_json(path,value)

    def recover(self):
        for name,row in list(self.live.items()):
            job=self.by_id[name]
            assert row["identity"]==pool.identity(job)
            if pool.process_alive(row["pid"]):continue
            path=self.out/job["result"]
            if path.exists():
                result=previous.read(path)
                if result.get("status") in ("ok","reused","complete","completed"):
                    receipt=dict(id=name,identity=pool.identity(job),category=category(job),rc=0,
                        wall_seconds=result["wall_seconds"],peak_rss_gb=result.get("peak_rss_gb",0),result=job["result"],cached=False,
                        recovered_after_controller_restart=True)
                    self.write(self.out/"receipts"/(name+".json"),receipt)
            self.live.pop(name,None)
        write_json(self.out/"live_workers.json",self.live)

    def adopt(self,running,pending):
        """A controller restart observes existing workers without relaunching them."""
        for name,row in self.live.items():
            if name in self.success:continue
            job=self.by_id[name]
            running[name]=dict(process=AdoptedProcess(row["pid"],self.out/job["result"]),
                log=Path(row["log"]).open("a",encoding="utf-8"),job=job,
                peak=pool.process_rss(row["pid"]),started=time.monotonic()-(time.time()-row["started_unix"]))
            pending.pop(name,None)

    def run(self):
        self.out.mkdir(parents=True,exist_ok=True);(self.out/"receipts").mkdir(exist_ok=True)
        self.recover()
        for p in self.receipt_paths():
            r=previous.read(p)
            if r["rc"]==0:self.success_receipt(r)
        self.save_release()
        pool.category=category;pool.memory_key=category
        previous.category=category
        pool.memory_requirement=lambda job,*args:max(previous.RAW_MEMORY_REQUIREMENT(job,*args),job["peak_rss_gb"])
        pool.job_items=lambda path:self.jobs
        pool.write_json=self.write
        pool.subprocess=types.SimpleNamespace(Popen=self.spawn,STDOUT=subprocess.STDOUT)
        source=previous.wrapped_pool_source()
        old="    while pending or running:\n"
        new="    _hi.adopt(running,pending)\n    while pending or running or _hi.accepting:\n        _hi.refresh(jobs,jobs_by_id,ids,pending,completed,running,failed)\n"
        assert source.count(old)==1
        namespace=dict(vars(pool),_hi=self)
        exec(compile(source.replace(old,new),"<closeout shared queue>","exec"),namespace)
        rc=namespace["execute"](REGISTRY,self.out,self.workers,3,"closeout",4,self.release)
        write_json(self.out/"controller_finish.json",dict(rc=rc,completed=len(self.success),expected=len(self.jobs),utc=datetime.now(timezone.utc).isoformat()))
        return rc


class AdoptedProcess:
    def __init__(self,pid,result):
        self.pid,self.result=pid,result

    def poll(self):
        if pool.process_alive(self.pid):return None
        if not self.result.exists():return 92
        return 0 if previous.read(self.result).get("status") in ("ok","reused","complete","completed") else 91


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("check","launch","run"))
    p.add_argument("--out",type=Path,default=OUT);p.add_argument("--workers",type=int,default=128)
    a=p.parse_args();assert 1<=a.workers<=128
    if a.command=="check":
        jobs,accepting=registered()
        compile(previous.wrapped_pool_source(),"<pool>","exec")
        print(json.dumps(dict(status="passed",jobs=len(jobs),accepting=accepting,workers=a.workers,reserve_gib=3)))
        return 0
    if a.command=="run":return Controller(a.out,a.workers).run()
    a.out.mkdir(parents=True,exist_ok=True)
    start=a.out/"closeout.start.json"
    if start.exists():assert not pool.process_alive(previous.read(start)["pid"]),"Controller is already running"
    write_json(a.out/"scheduler_policy.json",dict(running_growth_fraction=1.0))
    env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMBA_NUM_THREADS="1",PYTHONUTF8="1",
        CRISP_E_ARTIFACT=str(ROOT/"e2e/work/artifact"))
    cmd=[sys.executable,"-u","-B","e2e/hi_closeout_pool.py","run","--out",str(a.out),"--workers",str(a.workers)]
    options=dict(start_new_session=True) if sys.platform!="win32" else dict(creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    with (a.out/"closeout.log").open("ab") as f:
        proc=previous.RAW_POPEN(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,**options)
    write_json(start,dict(pid=proc.pid,command=cmd,utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(pid=proc.pid,out=str(a.out))))
    return 0


if __name__=="__main__":raise SystemExit(main())
