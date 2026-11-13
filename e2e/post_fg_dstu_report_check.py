"""Validate reporting against historical inputs and a hand-calculated trace."""
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import post_fg_dstu as jobs
import post_fg_dstu_report as report


def write_gzip(path,content,jsonl=False):
    with gzip.open(path,"wt",encoding="utf-8") as handle:
        if jsonl:
            for row in content:handle.write(json.dumps(row)+"\n")
        else:json.dump(content,handle)


def main():
    checks=[]
    parent=jobs.ROOT/"e2e/post_fg/out"
    parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix="report_check_",dir=parent) as folder:
        folder=Path(folder);(folder/"results").mkdir();(folder/"traces").mkdir()
        manifest=jobs.read_json(jobs.ROOT/"e2e/post_fg/manifest/dstu_health.json")
        old=[s for s in manifest["jobs"] if s.get("reuse_job")]
        jobs.write_json(folder/"manifest.json",dict(jobs=old))
        for spec in old:
            result=jobs.reuse(spec)
            result.update(status="ok",spec=spec,wall_seconds=0,peak_rss_gb=0)
            jobs.write_json(folder/"results"/(spec["job_id"]+".json"),result)
        rows=report.source_rows(folder/"manifest.json",folder,"historical_check")
        checked=report.paired(rows)
        assert len(rows)==60 and len(checked)==72 and all(r["error_ns"]==0 for r in checked)
        checks.append("60 historical sources: 36 public-deadline calculations and 72 paired O/J identities")
        small=[]
        for arm,ps,ds,pj,dj in (("BB0",100,10,150,20),("A256",120,8,115,11)):
            base=dict(condition="fixture",hardware="fixture",arm=arm,seed=1)
            small.extend([dict(base,run_type="solo",stream=0,completion_ns=ps),dict(base,run_type="solo",stream=1,completion_ns=ds),
                          dict(base,job_id=arm,run_type="joint",phase=.5,P_completion_ns=pj,D_latency_ns=dj,Lhat_max=-.1)])
        report.paired(small)
        chosen=small[-1]
        assert chosen["P_O_ns"]==20 and chosen["P_J_ns"]==-5 and chosen["P_BB0_J_ns"]==50
        assert chosen["D_O_ns"]==-2 and chosen["D_J_ns"]==3 and chosen["D_BB0_J_ns"]==10
        checks.append("hand calculation: P difference -35 = 20 - 5 - 50; D difference -9 = -2 + 3 - 10")
        runs=[];p=[1,10,0,1];c=[0,10,0,2]
        for arm,last,ready,start,insertion in (("A256",10,10,12,True),("A256-M",8,8,9,False)):
            name=arm.replace("-","_")
            issue=dict(t=5,resource=[[0,0],[0,1]],stream=0,parent=p,chunk=1,head_ready=1,
                       same_stream_insertion=insertion,original_true_tie=True,original_key=[1,-1],original_min=[1,-1],lower_rank_ready=False)
            write_gzip(folder/"traces"/(name+".issues.gz"),[issue],True)
            message=dict(parents=[dict(parent=p,first_start=0,last_finish=last,link_service=10,bytes=8,stream=0,consumers=[c],links=1,shared_links=1,chunks=2)],
                         consumers=[dict(tag=c,ready=ready,start=start,end=start+3,device="fixture",location=[0,0])],occupancy=dict(co2co=10))
            write_gzip(folder/"traces"/(name+".messages.gz"),message)
            runs.append(dict(job_id=name,hardware="fixture",condition="healthy",arm=arm,phase=.1,seed=1,
                             frozen_rank={"0":1,"1":0},issue_trace=name+".issues.gz",message_trace=name+".messages.gz"))
        summaries,resources,parents,consumers,chain=report.trace_tables(runs,folder)
        assert len(chain)==1 and chain[0]["last_finish_delta_ns"]==-2 and chain[0]["consumer_start_delta_ns"]==-3
        assert chain[0]["baseline_last_dependency"] and chain[0]["earlier_message_and_consumer"]
        assert [r["same_stream_insertions"] for r in summaries]==[1,0]
        assert all(r["queue_wait_sum_ns"]==4 for r in summaries)
        checks.append("trace fixture: parent insertion 1 to 0, last chunk 2 ns earlier, consumer starts 3 ns earlier, resource wait 4 ns")
    result=dict(status="passed",checks=checks,new_backend_calls=0,
                fixture_scope="Temporary hand-calculated data removed; real quality reports await cloud results")
    jobs.write_json(jobs.ROOT/"e2e/post_fg/evidence/dstu_report_unit_checks.json",result)
    print(json.dumps(result))


if __name__=="__main__":main()
