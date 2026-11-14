"""Replace historical baseline reuse with measured fresh-transform baselines."""
import argparse
from pathlib import Path

from post_fg_dstu import read_json,write_json


def prepare(health_manifest,trace_manifest,destination):
    health=read_json(health_manifest);traces=read_json(trace_manifest)
    baseline=[];health_jobs=[]
    for original in health["jobs"]:
        spec=dict(original)
        if spec["arm"] in ("BB0","A256"):
            spec["historical_reference_job"]=spec.pop("reuse_job")
            spec["historical_reference_sha256"]=spec.pop("reuse_sha256")
            spec["fresh_protocol"]="fresh_transform_v1"
            baseline.append(spec)
        health_jobs.append(spec)
    repeats=[];trace_jobs=[]
    for original in traces["jobs"]:
        spec=dict(original)
        if spec["arm"]=="A256":
            spec["historical_reference_job"]=spec.pop("compare_old_job")
            spec["historical_reference_sha256"]=spec.pop("compare_old_sha256")
            spec["fresh_protocol"]="fresh_transform_v1"
            spec["compare_baseline_job"]=spec["job_id"].removesuffix("_trace")
            spec["depends_on"]=[spec["compare_baseline_job"]]
            repeats.append(spec)
        trace_jobs.append(spec)
    assert len(baseline)==60 and len(repeats)==3 and len(health_jobs)==180
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    description="Fresh original-graph load and transform, preserving actual dependency/issue set iteration; historical result differences retained"
    files={"dstu_health_fresh_cloud.json":dict(protocol=description,jobs=health_jobs,historical_reuse=0,batch_aliases=6,new_backend_calls=174),
           "dstu_trace_fresh_cloud.json":dict(protocol=description,jobs=trace_jobs,new_backend_calls=6),
           "dstu_fresh_baselines60_cloud.json":dict(protocol=description,jobs=baseline,new_backend_calls=60),
           "dstu_fresh_traces3_cloud.json":dict(protocol=description,jobs=repeats,new_backend_calls=3),
           "dstu_fresh_patch63_cloud.json":dict(protocol=description,jobs=baseline+repeats,new_backend_calls=63)}
    for name,data in files.items():write_json(dest/name,data)
    return dict(baselines=len(baseline),trace_repeats=len(repeats),health_logical=len(health_jobs),destination=str(dest))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--health-manifest",required=True)
    p.add_argument("--trace-manifest",required=True)
    p.add_argument("--destination",required=True)
    a=p.parse_args();print(prepare(a.health_manifest,a.trace_manifest,a.destination))
