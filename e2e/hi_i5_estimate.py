"""Use completed I5 pilot receipts to estimate the remaining formal jobs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import hi_h3_i2
import hi_next
from e_run import ROOT,write_json

HERE=ROOT/"e2e/hi"


def cost_class(spec):
    kind=spec["kind"]
    if kind=="joint_solo":return (kind,spec["backend"],spec["stream"])
    if kind=="joint_graph":return (kind,"hop" in spec["backends"])
    if kind=="joint":return (kind,spec["backend"])
    return (kind,)


def estimate(out,apply=False):
    manifest=HERE/"manifest/i5.json"
    data=hi_next.read(manifest);jobs=data["jobs"];by_id={j["id"]:j for j in jobs}
    observed=defaultdict(list);complete={};rows=[]
    for path in (out/"receipts").glob("*.json"):
        receipt=hi_next.read(path);job=by_id.get(receipt["id"])
        if job is None or receipt["rc"]!=0:continue
        assert receipt["identity"]==hi_next.pool.identity(job)
        result=hi_next.read(out/job["result"])
        assert result["spec"]==job["spec"] and result["status"] in ("ok","reused")
        observed[cost_class(job["spec"])].append(receipt)
        complete[job["id"]]=receipt
    assert len(complete)>=14,"The complete formal pilot must be present first"
    for job in jobs:
        key=cost_class(job["spec"])
        if key in observed:
            sample=observed[key];seconds=statistics.median(r["wall_seconds"] for r in sample);rss=max(r["peak_rss_gb"] for r in sample)
            basis="pilot same operation/backend/stream class"
        elif key==("joint","orig"):
            ps,ds=observed["joint_solo","orig",0],observed["joint_solo","orig",1]
            seconds=statistics.median(r["wall_seconds"] for r in ps)+statistics.median(r["wall_seconds"] for r in ds)
            rss=max(r["peak_rss_gb"] for r in ps)+max(r["peak_rss_gb"] for r in ds)
            basis="sum of measured original P and D solo costs; first original joint still unmeasured"
        else:raise ValueError(key)
        rows.append(dict(id=job["id"],class_key=list(key),completed=job["id"] in complete,estimated_seconds=seconds,
            memory_hint_gib=max(.6,1.5*rss),basis=basis))
        job["estimated_seconds"]=seconds;job["peak_rss_gb"]=max(.6,1.5*rss)
    remaining=[dict(j,deps=[d for d in j["deps"] if d not in complete]) for j in jobs if j["id"] not in complete]
    queues=[dict(available_slots=n,minutes=hi_h3_i2.queue_estimate(remaining,180,n)["queue_seconds"]/60) for n in (8,16,32,64)] if remaining else []
    result=dict(status="passed",completed=len(complete),remaining=len(remaining),remaining_process_minutes=sum(j["estimated_seconds"] for j in remaining)/60,
        queues=queues,assumptions="Healthy A3h hop joint is measured; original joint uses P+D solo sum. PF4 uses same-class healthy timing proxy. These are runtime estimates, not serial-paper measurements. Waiting for other pools to release reserved slots is additional.",jobs=rows,
        pilot_observations=[dict(class_key=list(k),n=len(v),median_seconds=statistics.median(r["wall_seconds"] for r in v),peak_rss_gib=max(r["peak_rss_gb"] for r in v)) for k,v in observed.items()],
        updated_manifest=apply)
    if apply:
        identities=[hi_next.pool.identity(j) for j in hi_next.read(manifest)["jobs"]]
        assert identities==[hi_next.pool.identity(j) for j in jobs]
        write_json(manifest,data)
        result["updated_manifest_sha256"]=hashlib.sha256(manifest.read_bytes()).hexdigest()
    write_json(HERE/"evidence/i5_pilot_estimate.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=HERE/"out/i5");p.add_argument("--apply",action="store_true")
    a=p.parse_args();r=estimate(a.out,a.apply)
    print(json.dumps({k:v for k,v in r.items() if k not in ("jobs","pilot_observations")},ensure_ascii=False,indent=2))
