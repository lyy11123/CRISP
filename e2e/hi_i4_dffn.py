"""One ready Qwen WATOS decode-FFN PF4 unit; preserve the running I4 batch."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import hi_i4 as base
import hi_h3_i2 as common
import hi_next as previous
import hi_fault
from e_run import ROOT,write_json

HERE=ROOT/"e2e/hi"
NAME="i4_watos_dffn"
OUT=HERE/"out/i4_watos_dffn"
MANIFEST=HERE/"manifest/i4_watos_dffn.json"
INPUTS=HERE/"manifest/i4_watos_dffn_healthy_inputs.json"
PARENTS=base.PARENTS+((base.MANIFEST,base.OUT,"i4_light.status.json"),)


def bind():
    base.NAME,base.OUT,base.MANIFEST,base.INPUTS,base.PARENTS=NAME,OUT,MANIFEST,INPUTS,PARENTS
    base.bind_shared()


def plan(out=OUT):
    specs=[]
    master={s["candidate_id"]:s for s in previous.read(HERE/"manifest/all_candidates_planned.json")["jobs"]}
    for s in hi_fault.jobs("qwen3moe","watos",out,"pf_ffn"):
        if s.get("phase_op") not in (None,"dc_ffn"):continue
        s.update(module="hi_i4_dffn",launch_status="authorized_I4_WATOS_decode_FFN_single_point")
        if s["kind"]=="fault_selection":
            s["candidate_ids"]=[n for n in s["candidate_ids"] if master[n]["phase"]=="decode" and master[n]["op"] in ("mha","ffn")]
            s["depends_on"]=[]
        specs.append(s)
    assert Counter(s["kind"] for s in specs)=={"fault_selection":1,"fault_select":1,"fault_pre":1,"fault_l1":1,"fault_stop":4,"fault_rep":15}
    needed={n for s in specs if s["kind"]=="fault_selection" for n in s["candidate_ids"]}
    assert len(needed)==28
    records,index={},{}
    for folder in base.SOURCES:
        folder=HERE/"out"/folder
        for p in (folder/"results").glob("*.json"):
            name=p.stem
            if name not in needed:continue
            r=previous.read(p)
            assert r["status"]=="ok" and all(r[k]==v for k,v in master[name].items())
            paths=dict(record=p,raw=folder/"raw"/(name+".pkl.gz"),snapshot=folder/"snapshots"/(name+".pkl.gz"))
            if not all(q.exists() for q in paths.values()):continue
            records[name]=r
            index[name]={k:q.relative_to(ROOT).as_posix() for k,q in paths.items()}
            index[name]["sha256"]={k:hashlib.sha256(q.read_bytes()).hexdigest() for k,q in paths.items()}
    missing=sorted(needed-set(records))
    if missing:
        result=dict(status="waiting_for_download_or_results",required=28,present=len(records),missing=missing)
        write_json(HERE/"evidence/i4_watos_dffn_readiness.json",result)
        return result
    rows=[r for r in records.values() if r["op"]=="ffn"]
    assert len(rows)==14 and sum(r["op"]=="mha" for r in records.values())==14
    selected=min(rows,key=lambda r:(statistics.mean(x["cycles"] for x in r["backend"]),r["candidate_order"]))
    backend=sum(selected["timing"]["evaluation"]["backend_individual_seconds"])
    search=float(selected["search_seconds"])
    write_json(INPUTS,dict(status="passed",count=28,candidates=index,selection="backend_mean_then_candidate_order",new_healthy_searches=0))
    jobs=[]
    for s in specs:
        kind=s["kind"]
        seconds=1. if kind in ("fault_selection","fault_select","fault_stop") else 20+backend*(2 if kind=="fault_l1" else 1)
        if kind=="fault_rep":seconds+=search*s["steps"]/1000
        jobs.append(dict(id=s["job_id"],kind=kind,spec=s,result=f"results/{s['job_id']}.json",deps=s["depends_on"],
            argv=["{python}","-u","-B","e2e/hi_i4_dffn.py","child","--spec","{spec}","--out","{out}"],
            estimated_seconds=seconds,peak_rss_gb=max(.6,2*selected["peak_rss_gb"]),priority=0))
    assert {d for j in jobs for d in j["deps"]}<={j["id"] for j in jobs}
    write_json(MANIFEST,dict(status="authorized",jobs=jobs,healthy_inputs=28,a3h_pf4_units=1,maximum_L2_trajectories=15,
        new_healthy_searches=0,additional_MHA_backend_calls=0))
    queue=common.queue_estimate(jobs,200,5)
    evidence=dict(status="passed",jobs=23,counts=dict(Counter(s["kind"] for s in specs)),healthy_inputs=28,complete_groups=2,
        control_jobs=6,healthy_replays=1,R0_L1_jobs=1,maximum_L2_trajectories=15,maximum_backend_samples=180,
        selected_candidate=selected["candidate_id"],selected_backend_ten_seconds=backend,selected_search_1000_seconds=search,
        estimated_upper_process_hours=sum(j["estimated_seconds"] for j in jobs)/3600,
        queue_minutes_all_L2_executed=queue["queue_seconds"]/60,planning_minutes=[queue["queue_seconds"]/60,1.5*queue["queue_seconds"]/60],
        estimate_basis="Selected full 14-candidate WATOS decode FFN healthy result; all fifteen independent L2 trajectories charged. Original stop can skip tiers.",
        new_healthy_searches=0,additional_MHA_backend_calls=0,
        parent_manifests=[m.relative_to(ROOT).as_posix() for m,_,_ in PARENTS],
        combined_process_rule="new active plus every unfinished H3I2, I3, H4 and I4-light job <=128",
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (MANIFEST,INPUTS,ROOT/"e2e/hi_i4_dffn.py")})
    write_json(HERE/"evidence/i4_watos_dffn_plan.json",evidence)
    text=["# I4 WATOS decode FFN 单点补充", "", "2026-11-14。独立补入刚完成14候选选优的Qwen WATOS decode FFN PF4点；已运行201项清单保持不变。", "",
        "输入28个健康候选：decode FFN的14项以及原有decode MHA的14项物理平台参照。MHA仅加载既有选优快照核对固定PF4边，不重复后端或搜索。两种硬件的prefill FFN、MF6仍留后续。", "",
        "沿用已验证的hi_i4/hi_h4/hi_fault和原fg_fault EIR流程：R0、L1、五种子123/1/2/3/4、独立30/100/300档、原2%停止、每状态十后端。共23调度作业：6控制、1健康重放、1R0/L1、最多15条实际L2；最多180后端样本。停档复用单独计数。", "",
        f"真实选优候选为 `{selected['candidate_id']}`，其完整十后端实测{backend/60:.2f}分钟，健康1000步搜索{search/60:.2f}分钟。全部档位执行的初始估计{queue['queue_seconds']/60:.1f}分钟，安排{evidence['planning_minutes'][0]:.1f}–{evidence['planning_minutes'][1]:.1f}分钟；早停会减少档位。包含搜索、评估和导出，下载汇总另计。", "",
        "输出独立 `out/i4_watos_dffn`。新池为H3/I2、I3、H4、I4-light所有未完成作业预留槽，合计最多128；扣除各池未来内存增长并预留3GiB。只用释放出的槽，不改停前池。", "", "```text", "python e2e/hi_i4_dffn.py check", "python e2e/hi_i4_dffn.py launch", "```", ""]
    (HERE/"I4_WATOS_DFFN_RUN_PLAN.md").write_text("\n".join(text),encoding="utf-8")
    compile(base.shared.wrapped_source(),"<I4 single-point shared pool>","exec")
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","check","child","launch","run"))
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--workers",type=int,default=128)
    args=parser.parse_args();args.out=args.out.resolve()
    assert 1<=args.workers<=128
    if args.command in ("plan","check"):
        result=plan(args.out);print(json.dumps(result,ensure_ascii=False,indent=2));return int(result["status"]!="passed")
    bind()
    if args.command=="child":base.child(previous.read(args.spec),args.out);return 0
    args.out.mkdir(parents=True,exist_ok=True)
    if args.command=="launch":
        start=args.out/(NAME+".start.json")
        if start.exists() and previous.pool.process_alive(previous.read(start)["pid"]):raise RuntimeError("I4 WATOS decode FFN controller already running")
        write_json(args.out/"scheduler_policy.json",dict(running_growth_fraction=.25))
        env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMBA_NUM_THREADS="1",CRISP_E_ARTIFACT=str(ROOT/"e2e/work/artifact"),PYTHONUTF8="1")
        command=[sys.executable,"-u","-B","e2e/hi_i4_dffn.py","run","--out",str(args.out),"--workers",str(args.workers)]
        options=dict(start_new_session=True) if sys.platform!="win32" else dict(creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        with (args.out/(NAME+".log")).open("ab") as stream:process=previous.RAW_POPEN(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,**options)
        write_json(start,dict(pid=process.pid,command=command,created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256={n:hashlib.sha256((ROOT/"e2e"/n).read_bytes()).hexdigest() for n in ("hi_i4_dffn.py","hi_i4.py","hi_h4.py","hi_fault.py","hi_next.py","fg_batch.py","fg_fault.py")}))
        print(json.dumps(dict(pid=process.pid,jobs=23,out=str(args.out))),flush=True);return 0
    return base.Controller(args.out,args.workers).run()


if __name__=="__main__":raise SystemExit(main())
