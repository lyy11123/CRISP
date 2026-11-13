"""Independent aggregation, saved-result, metadata-mode and receipt checks."""
import argparse
import copy
import json
import math
from pathlib import Path
import tempfile

from e_run import ROOT, write_json
from fg_batch import identity
import fg_report as report


def check(tables, run_out, manifest, output):
    summary=report.read(tables/"summary.json")
    specs=report.read(manifest)
    candidates=report.read(tables/"candidates.json")
    operators=report.read(tables/"operators.json")
    layers=report.read(tables/"layers.json")
    pairs=report.read(tables/"gemini_terminal_pairs.json")
    samples=report.read(tables/"backend_samples.json")
    ledger=report.read(tables/"job_timing.json")
    checks=[]
    def verified(name, condition):
        assert condition,name
        checks.append(name)

    verified("all primary candidates present",sum(r["kind"]!="terminal" for r in candidates)==len(specs))
    verified("all 56 Gemini terminal comparisons present",len(pairs)==56)
    verified("all primary groups have fourteen candidates",all(r["completed"]==14 for r in operators))
    verified("ten seed samples per saved state",len(samples)==len(candidates)*10)
    verified("backend components conserve total",all(r["compute_only_ns"]+r["communication_only_ns"]+r["overlap_ns"]==r["total_ns"] for r in samples))
    verified("nonnegative concurrent-service remainder",all(r["overlap_ns"]>=0 for r in samples))
    indexed={(r["config_id"],r["family"],r["fault_id"],r["method"],r["phase"],r["op"]):r for r in operators}
    for row in layers:
        values={phase:{op:indexed[row["config_id"],row["family"],row["fault_id"],row["method"],phase,op][row["protocol"]+"_ns"]
                       for op in report.OPS} for phase in report.PHASES}
        p,d=values["prefill"],values["decode"]
        assert math.isclose(row["prefill_ns"],2*p["ln"]+8*p["mha"]+p["proj"]+p["ffn"],rel_tol=1e-12)
        assert math.isclose(row["decode_ns"],d["ln"]+8*d["mha"]+d["proj"]+d["ffn"],rel_tol=1e-12)
        for steps in (512,2048):
            assert math.isclose(row[f"request_{steps}_ns"],row["prefill_ns"]+steps*row["decode_ns"],rel_tol=1e-12)
    checks.append("every layer and fixed-KV request independently reassembled")
    verified("one successful receipt per job",len(ledger)==len({r["job_id"] for r in ledger}))
    verified("receipt time total agrees",math.isclose(sum(r["wall_seconds"] for r in ledger)/3600,
             summary["observed_time"]["successful_process_hours"],rel_tol=1e-12))
    for row in report.read(tables/"scaling.json"):
        assert math.isclose(row["speedup"],row["baseline_ns"]/row["value_ns"],rel_tol=1e-12)
        assert math.isclose(row["efficiency"],row["speedup"]*64/row["cores"],rel_tol=1e-12)
    checks.append("scaling matches same-method same-protocol 64-core baseline")
    for row in pairs:
        assert math.isclose(row["terminal_vs_best"],row["terminal_ns"]/row["best_ns"]-1,rel_tol=1e-12,abs_tol=1e-12)
        if row["backend_reused"]:
            assert row["terminal_ns"]==row["best_ns"]
    checks.append("Gemini best/terminal comparisons and identical-state reuse agree")
    if summary["audit"]["mode"]=="results_and_snapshots":
        work=report.read(tables/"workload.json")
        workchecks=report.read(tables/"workload_checks.json")
        expected_work={(s["config_id"],s["phase"],s["op"]) for s in specs if s["family"]=="healthy"}
        verified("all healthy workload representatives parsed",{(r["config_id"],r["phase"],r["op"]) for r in work}==expected_work)
        verified("effective TensorCore work agrees across hardware and split configurations",len(workchecks)==8 and all(r["effective_tensor_work_equal_across_configurations"] for r in workchecks))
        verified("eight-head micrographs and aggregation multiplier confirmed",all(r["micro_mha_heads"]==8 and r["mha_aggregation_multiplier"]==8 for r in work if r["op"]=="mha"))
        if summary["audit"].get("static_resource_loads_parsed"):
            resources=report.read(tables/"resource_loads.json")
            expected_selected={r["backend_best_candidate"] for r in operators if r["stage"]!="E_reuse" and r["family"]=="healthy"}
            verified("resource work covers every selected healthy candidate",{r["candidate_id"] for r in resources}==expected_selected)
            verified("resource maximum and average use full resource count",all(math.isclose(r["mean_all_resources"]*r["available_resources"],r["total"],rel_tol=1e-12,abs_tol=1e-8) and r["used_resources"]<=r["available_resources"] for r in resources))

    # The known layer example detects an accidental extra decode LN or head multiplier.
    base=dict(config_id="fixture",stage="G",hardware_profile="busybarn",shape="2x3",split_mode="fixed",
              d2d_scale=1,family="healthy",fault_id="healthy",method="A3h")
    fixture=[dict(base,phase=phase,op=op,backend_best_ns=value)
             for phase,values in (("prefill",(2,3,5,7)),("decode",(11,13,17,19)))
             for op,value in zip(report.OPS,values)]
    layer=report.layer_rows(fixture)[0]
    verified("hand-calculated 40/151 ns layer and 77352/309288 ns requests",
             (layer["prefill_ns"],layer["decode_ns"],layer["request_512_ns"],layer["request_2048_ns"])==(40,151,77352,309288))

    with tempfile.TemporaryDirectory(prefix="fg_report_check_",dir=ROOT/"e2e/fg/out") as tmp:
        tmp=Path(tmp)
        assert tmp.resolve().is_relative_to(ROOT/"e2e/fg/out")
        (tmp/"results").mkdir()
        spec=next(s for s in specs if s["method"]=="A1")
        saved=report.read(run_out/"results"/(spec["candidate_id"]+".json"))
        file=tmp/"results"/(spec["candidate_id"]+".json")
        write_json(file,saved)
        metadata,_=report.candidate_audit([spec],tmp,metadata_only=True)
        full,_=report.candidate_audit([spec],tmp)
        verified("metadata mode omits only local snapshot absence",not metadata["issues"] and metadata["local_snapshot_absence_omitted"]==1)
        verified("full mode requires actual snapshot",any("snapshot" in r.get("issues",[]) for r in full["issues"]))
        saved["backend"][0]["seed"]=999
        write_json(file,saved)
        metadata,_=report.candidate_audit([spec],tmp,metadata_only=True)
        verified("metadata mode retains genuine seed errors",any("backend seeds" in r.get("issues",[]) for r in metadata["issues"]))

        run=tmp/"ledger"; (run/"receipts").mkdir(parents=True);(run/"results").mkdir()
        jobs=[]
        for i,seconds in enumerate((10,20)):
            name=f"test_{i}";s=dict(base,kind="test",job_id=name)
            job=dict(id=name,kind="test",spec=s,deps=[],result="results/"+name+".json")
            jobs.append(job)
            write_json(run/job["result"],dict(status="ok"))
            write_json(run/"receipts"/(name+".json"),dict(id=name,identity=identity(job),rc=0,wall_seconds=seconds,peak_rss_gb=1,result=job["result"]))
        write_json(tmp/"jobs.json",jobs)
        failure=dict(id="test_0",identity=identity(jobs[0]),rc=1,wall_seconds=3,peak_rss_gb=1)
        batch=dict(wall_seconds=20,workers=2,completed=2,failed=[],pending=[],max_active=2,min_available_gib=3,
                   runs=[report.read(p) for p in (run/"receipts").glob("*.json")]+[failure])
        write_json(run/"a.batch.json",batch);write_json(run/"b.batch.json",batch)
        write_json(tmp/"archive"/"patches"/"failure.json",failure)
        timing,totals=report.timing_tables(run,tmp/"jobs.json",tmp/"archive")
        verified("overlapping controllers do not duplicate successful work",len(timing["job_timing"])==2 and totals["successful_process_hours"]==30/3600)
        verified("identical archived batch record listed once",len(timing["controller_timing"])==1 and len(timing["controller_timing"][0]["source_batches"])==2)
        verified("same archived failure counted once",len(timing["retry_attempts"])==1 and totals["failed_attempt_process_hours"]==3/3600)
    result=dict(status="passed",checks=checks,check_count=len(checks),sources=dict(tables=report.source_path(tables),
                run_out=report.source_path(run_out),manifest=report.source_path(manifest)),
                counts=dict(candidates=len(candidates),operators=len(operators),layers=len(layers),gemini_pairs=len(pairs),backend_samples=len(samples),successful_jobs=len(ledger)))
    write_json(output,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables",required=True,type=Path)
    parser.add_argument("--run-out",required=True,type=Path)
    parser.add_argument("--manifest",type=Path,default=ROOT/"e2e/fg/manifest/search.json")
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    check(args.tables,args.run_out,args.manifest,args.out)
