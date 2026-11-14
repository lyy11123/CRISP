"""Read all 1256 saved job intervals and the combined server monitor."""
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
from hi_next_check import read,sha
from hi_h3_i2_check import write

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"e2e/hi"
BATCHES=(
    ("h3_i2_440","h3_i2",440,"h3_i2_440"),
    ("i3_108","i3",108,"i3_108"),
    ("h4_pf4","h4_pf4",264,"h4_pf4"),
    ("i4_light","i4_light",201,"i4_light"),
    ("i4_watos_dffn","i4_watos_dffn",23,"i4_watos_dffn"),
    ("h5","h5",110,"h5"),
    ("i5","i5",110,"i5"),
)


def identity(job):
    return hashlib.sha256(json.dumps({k:job[k] for k in ("id","kind","spec","result","argv")},sort_keys=True).encode()).hexdigest()


def date(stamp):return datetime.fromtimestamp(stamp,timezone(timedelta(hours=8))).isoformat()


def check():
    intervals,changes,groups,all_ids=[],[],[],set()
    for folder,prefix,count,manifest in BATCHES:
        out=HERE/"out"/folder
        jobs=read(HERE/"manifest"/(manifest+".json"))["jobs"]
        assert len(jobs)==count
        ids={j["id"] for j in jobs}
        scoped_ids={folder+"/"+name for name in ids}
        assert len(ids)==count and not (scoped_ids&all_ids)
        all_ids.update(scoped_ids)
        assert {p.stem for p in (out/"receipts").glob("*.json")}==ids
        finish=read(out/"controller_finish.json")
        batch=read(out/(prefix+".batch.json"))
        assert finish["rc"]==0 and finish["completed"]==finish["expected"]==batch["completed"]==count
        assert batch["failed"]==batch["pending"]==[]
        assert not list((out/"failed_attempts").glob("*/*/receipt.json"))
        rows=[]
        for job in jobs:
            name=job["id"];receipt=read(out/"receipts"/(name+".json"));r=read(out/job["result"])
            assert receipt["id"]==name and receipt["identity"]==identity(job) and receipt["rc"]==0
            assert receipt["result"]==job["result"] and not receipt["cached"]
            assert r["status"] in ("ok","reused") and read(out/"specs"/(name+".json"))==job["spec"]
            assert (r["candidate_id"] if job["kind"]=="search" else r["job_id"])==name
            logs=list((out/"logs").glob(name+".*.log"));assert len(logs)==1
            log=logs[0];start=int(log.stem.rsplit(".",1)[1])/1e9;end=start+receipt["wall_seconds"]
            assert end>start
            if log.stat().st_size==0:
                assert (job["kind"]=="fault_rep" and r["status"]=="reused") or job["kind"] in ("hi_joint_export","joint_graph")
            rows.append(dict(batch=folder,job_id=name,kind=job["kind"],status=r["status"],start_unix=start,receipt_approx_unix=end,
                start_beijing=date(start),receipt_approx_beijing=date(end),receipt_wall_seconds=receipt["wall_seconds"],result_wall_seconds=r["wall_seconds"],
                log_bytes=log.stat().st_size,log_sha256=sha(log),receipt_identity=receipt["identity"]))
            changes.extend(((start,1,folder+"/"+name),(end,-1,folder+"/"+name)))
        assert len(list((out/"logs").glob("*.log")))==count
        intervals.extend(rows)
        groups.append(dict(batch=folder,jobs=count,unique_successful_attempts=count,failed_attempts=0,
            start_beijing=min(r["start_beijing"] for r in rows),controller_finish=finish,
            receipt_interval_end_beijing=date(max(r["receipt_approx_unix"] for r in rows)),
            own_peak_active_recorded=batch["max_active"],receipt_process_hours=sum(r["receipt_wall_seconds"] for r in rows)/3600,
            result_process_hours=sum(r["result_wall_seconds"] for r in rows)/3600,minimum_available_gib=batch["min_available_gib"],
            empty_logs=sum(r["log_bytes"]==0 for r in rows),reused_jobs=sum(r["status"]=="reused" for r in rows),
            manifest_sha256=sha(HERE/"manifest"/(manifest+".json")),batch_sha256=sha(out/(prefix+".batch.json")),finish_sha256=sha(out/"controller_finish.json")))
    assert len(intervals)==len(all_ids)==1256
    active,maximum,set_at_peak= set(),0,[]
    peak_time=None
    for stamp,delta,name in sorted(changes,key=lambda x:(x[0],x[1])):
        if delta==1:
            assert name not in active;active.add(name)
        else:
            assert name in active;active.remove(name)
        if len(active)>maximum:
            maximum=len(active);set_at_peak=sorted(active);peak_time=stamp
    assert not active and maximum<=128
    monitor_path=HERE/"out/combined_monitor.jsonl"
    monitors=[json.loads(line) for line in monitor_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert monitors and [r["utc"] for r in monitors]==sorted(r["utc"] for r in monitors)
    for row in monitors:
        assert row["active"]==sum(b["active"] for b in row["batches"])
        assert row["active"]<=128
        assert all(not b["failed"] and b["failed_attempt_count"]==0 for b in row["batches"])
    last=monitors[-1];byname={b["name"]:b for b in last["batches"]}
    assert set(byname)=={f for f,_,_,_ in BATCHES} and last["active"]==0
    for folder,prefix,count,_ in BATCHES:
        row=byname[folder]
        assert row["completed"]==count and row["active"]==row["pending"]==0 and row["finished"]
        assert row["finish"]==read(HERE/"out"/folder/"controller_finish.json")
    early_path=HERE/"out/h3_i2_440/monitor_history.jsonl"
    early=[json.loads(line) for line in early_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert early and all(not row["failed"] and row["active"]<=128 for row in early)
    ids_without_batch=Counter(r["job_id"] for r in intervals)
    shared_names={name:[dict(batch=r["batch"],kind=r["kind"],receipt_identity=r["receipt_identity"]) for r in intervals if r["job_id"]==name] for name,n in ids_without_batch.items() if n>1}
    assert len(shared_names)==2 and all({r["batch"] for r in entries}=={"i4_light","i4_watos_dffn"} and len({r["receipt_identity"] for r in entries})==2 for entries in shared_names.values())
    reviews={}
    for filename in ("h3_i2_review.json","i3_review.json","h4_completion_review.json","i4_completion_review.json","i4_watos_dffn_completion_review.json","h5_completion_review.json","i5_completion_review.json"):
        path=HERE/"evidence"/filename;assert read(path)["status"]=="passed";reviews[filename]=sha(path)
    result=dict(status="passed",created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Seven newly executed H3/I2, I3, H4, I4-light, I4-WATOS-DFFN, H5 and I5 batches. Prior 348 healthy candidates are excluded from these 1256 jobs.",
        jobs=1256,unique_batch_and_job_ids=1256,unique_unscoped_job_ids=len(ids_without_batch),successful_receipts=1256,unique_job_logs=1256,duplicate_formal_attempts=0,failed_receipts=0,failed_attempts=0,
        reused_control_names_across_distinct_batches=shared_names,
        job_identity_note="Two independent I4 batches use the same WATOS configuration name for selection and fixed-fault controls; their receipt identities/specifications and output directories differ. Execution is identified by batch plus job ID.",
        reused_repair_jobs=sum(g["reused_jobs"] for g in groups),empty_logs=sum(g["empty_logs"] for g in groups),
        empty_logs_meaning="215 stopped repair records plus 14 export/graph jobs; all have unique successful receipts, not failed or missing executions.",
        approximate_max_overlap=maximum,first_peak_beijing=date(peak_time),first_peak_job_ids=set_at_peak,
        interval_definition="Filename Unix_ns is sampled immediately before Popen; receipt.wall_seconds begins after Popen and ends at controller collection. The end is approximate and includes polling/bookkeeping, so intervals describe occupied task slots rather than exact CPU time.",
        individual_dependency_tolerance_seconds=2.1,
        global_start_beijing=date(min(r["start_unix"] for r in intervals)),
        last_controller_finish_beijing=datetime.fromisoformat(max(g["controller_finish"]["utc"] for g in groups)).astimezone(timezone(timedelta(hours=8))).isoformat(),
        process_hours_from_results=sum(r["result_wall_seconds"] for r in intervals)/3600,
        process_hours_from_receipts=sum(r["receipt_wall_seconds"] for r in intervals)/3600,
        monitor_samples=len(monitors),monitor_first_utc=monitors[0]["utc"],monitor_last=last,
        monitor_sha256=sha(monitor_path),monitor_peak_active=max(r["active"] for r in monitors),monitor_peak_cpu_percent=max(r["cpu_percent"] for r in monitors),
        initial_pool_monitor=dict(file=early_path.relative_to(ROOT).as_posix(),sha256=sha(early_path),samples=len(early),first_utc=early[0]["utc"],last_utc=early[-1]["utc"],peak_active=max(r["active"] for r in early),peak_cpu_percent=max(r["cpu_percent"] for r in early),minimum_available_gib=min(r["available_gib"] for r in early),final_status_is_pre_finish_snapshot=True),
        monitor_minimum_available_gib=min(r["available_gib"] for r in monitors),all_pool_minimum_sampled_available_gib=min(g["minimum_available_gib"] for g in groups),
        memory_definition="Minima are observed samples; 3 GiB is the launch-budget reserve, not a claim about unsampled instants.",
        monitor_final_completed=sum(b["completed"] for b in last["batches"]),monitor_final_active=0,monitor_final_pending=0,
        independent_batch_review_sha256=reviews,batches=groups,intervals=intervals)
    assert result["reused_repair_jobs"]==215 and result["empty_logs"]==229 and result["monitor_final_completed"]==1256
    write(HERE/"evidence/full_execution_review.json",result)
    return result


if __name__=="__main__":
    r=check();print(json.dumps({k:r[k] for k in ("status","jobs","approximate_max_overlap","monitor_peak_active","monitor_final_completed","last_controller_finish_beijing")}))
