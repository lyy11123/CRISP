"""I3's 108 Qwen WATOS candidates beside the still-running H3/I2 pool."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import types

import hi_next as previous
import hi_h3_i2 as common
from e_run import ROOT, write_json

HERE = ROOT / "e2e/hi"
NAME = "i3"
MANIFEST = HERE / "manifest/i3_108.json"
RELEASE_PLAN = HERE / "manifest/i3_release_plan.json"
OLD_MANIFEST = HERE / "manifest/h3_i2_440.json"
OLD_OUT = HERE / "out/h3_i2_440"


def estimates(specs):
    candidate_path = HERE / "tables/h3_i2_candidates.csv"
    timing_path = HERE / "tables/h3_i2_timing_exclusive.csv"
    candidates = common.read_csv(candidate_path)
    timing = {r["candidate_id"]: r for r in common.read_csv(timing_path)}
    groups = defaultdict(list)
    for row in candidates:
        groups[common.key(row)].append(row)
    live_path = HERE / "evidence/h3_i2_long_jobs_latest.json"
    live = previous.read(live_path)
    live_A1_search = [r["progress"]["milestones"]["search_complete"] - r["progress"]["milestones"]["search_begin"]
        for r in live["rows"] if "_A1_" in r["id"] and "search_complete" in r["progress"].get("milestones", {})]
    def median(rows, field):
        return statistics.median(float(r[field]) for r in rows)
    def search(rows):
        return statistics.median(float(timing[r["candidate_id"]]["search_seconds"]) for r in rows)
    forecast, measured = [], []
    for spec in specs:
        target = common.key(spec)
        _, _, phase, op, method = target
        if target in groups:
            anchors = groups[target]
            seconds = median(anchors, "process_receipt_seconds")
            source, ratio = "same_Qwen_WATOS_phase_operator_method_complete_job", 1.0
            measured.extend(dict(candidate_id=r["candidate_id"], category=list(previous.category(dict(kind="search", spec=spec))),
                wall_seconds=float(r["process_receipt_seconds"]), peak_rss_gb=float(r["peak_rss_gib"])) for r in anchors)
        else:
            anchors = groups["qwen3moe", "watos", phase, op, "A3h"]
            other = groups.get(("qwen3moe", "busybarn", phase, op, method))
            base = groups["qwen3moe", "busybarn", phase, op, "A3h"]
            if other:
                ratio = search(other) / search(base)
                source = "Qwen_WATOS_A3h_nonsearch_plus_Qwen_BusyBarn_measured_method_search_ratio"
            elif phase == "prefill" and op == "ffn" and live_A1_search:
                ratio = statistics.median(live_A1_search) / search(base)
                source = "Qwen_WATOS_A3h_nonsearch_plus_live_Qwen_BusyBarn_A1_complete_search_ratio"
            else:
                other = groups["llama", "watos", phase, op, method]
                base = groups["llama", "watos", phase, op, "A3h"]
                ratio = search(other) / search(base)
                source = "Qwen_WATOS_A3h_nonsearch_plus_Llama_WATOS_measured_method_search_ratio"
            seconds = median(anchors, "process_receipt_seconds") + search(anchors)*(ratio-1)
        rss = max(float(r["peak_rss_gib"]) for r in anchors)
        forecast.append(dict(candidate_id=spec["candidate_id"], batch="I3", **dict(zip(common.KEYS,target)),
            estimated_seconds=seconds, measured_same_class=target in groups, ratio=ratio,
            estimated_rss_gib=rss, memory_hint_gib=1.25*rss, cost_source=source,
            anchors=";".join(r["candidate_id"] for r in anchors)))
    unique_measured = {r["candidate_id"]: r for r in measured}
    sources = [candidate_path,timing_path,live_path,HERE/"manifest/all_candidates_planned.json"]
    return forecast, list(unique_measured.values()), {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, live["utc"]


def assemble():
    manifest, release = previous.read(MANIFEST), previous.read(RELEASE_PLAN)
    return manifest["jobs"], set(release["initial_release_ids"]), release["calibrations"]


def wrapped_source():
    source = previous.wrapped_pool_source()
    substitutions = {
        "        external_slots = 0\n": "        external_slots = _hi.reserve_old()\n",
        "        budget = free - reserve_gib - reserved_growth\n":
        "        budget = free - reserve_gib - reserved_growth - _hi.old_growth_reserve_gib\n"}
    for old,new in substitutions.items():
        assert source.count(old)==1
        source=source.replace(old,new)
    return source


def plan():
    specs=[s for s in previous.read(HERE/"manifest/all_candidates_planned.json")["jobs"] if s["batch"]=="I3"]
    assert len(specs)==len({s["candidate_id"] for s in specs})==108
    assert Counter(s["method"] for s in specs)=={"A1":56,"A3h":52}
    old={j["id"] for j in previous.RAW_JOB_ITEMS(OLD_MANIFEST)}
    completed={r["candidate_id"] for r in common.read_csv(HERE/"tables/h3_i2_candidates.csv")}
    assert not old.intersection(s["candidate_id"] for s in specs)
    assert not completed.intersection(s["candidate_id"] for s in specs)
    forecast,observed,source_hashes,live_utc=estimates(specs)
    by_id={r["candidate_id"]:r for r in forecast}
    representatives={}
    for spec in specs:
        assert spec["model"]=="qwen3moe" and spec["hardware_profile"]=="watos"
        assert spec["search_steps"]==1000 and spec["backend_seeds"]==spec["random_cpcm_seeds"]==list(range(1,11))
        if spec["method"]=="A1":
            k=common.key(spec)
            if k not in representatives or spec["candidate_order"]<representatives[k]["candidate_order"]:
                representatives[k]=spec
    first={s["candidate_id"] for s in representatives.values()}
    calibrations=[dict(category=list(k),prerequisite_candidate=s["candidate_id"],release_after_success=[],
                      role="priority measurement; all same-class jobs released") for k,s in representatives.items()]
    jobs=[]
    for spec in specs:
        name=spec["candidate_id"];row=by_id[name]
        jobs.append(dict(id=name,kind="search",spec=spec,result=f"results/{name}.json",deps=[],
            argv=["{python}","-u","-B","e2e/hi_run.py","child","--spec","{spec}","--out","{out}"],
            estimated_seconds=row["estimated_seconds"],peak_rss_gb=row["memory_hint_gib"],
            priority=2 if name in first else 1 if spec["phase"]=="prefill" and spec["op"]=="ffn" else 0))
    write_json(MANIFEST,dict(status="authorized",counts={"I3":108},jobs=jobs,prior_observations=observed))
    write_json(RELEASE_PLAN,dict(all=True,initial_release_ids=[j["id"] for j in jobs],calibrations=calibrations,
        dependent_jobs=0,priority_rules="four A1 representatives, then prefill FFN, then longest complete jobs"))
    common.table(HERE/"tables/i3_estimates.csv",forecast)
    queue=common.queue_estimate(jobs,200,workers=84)
    summary=dict(status="passed",jobs=108,methods={"A1":56,"A3h":52},initially_released=108,dependent_jobs=0,
        measured_classes=4,new_classes=4,priority_representatives=4,process_hours=sum(j["estimated_seconds"] for j in jobs)/3600,
        queue_minutes_at_84_slots=queue["queue_seconds"]/60,planning_minutes=[queue["queue_seconds"]/60,2*queue["queue_seconds"]/60],
        longest_job_minutes=max(j["estimated_seconds"] for j in jobs)/60,
        prior_observations=observed,live_search_snapshot_utc=live_utc,source_sha256=source_hashes,
        combined_process_rule="new active + (440 - validated old successful receipts) <= 128",
        memory_rule="new free budget = system available - 3 GiB - own future growth reserve - old future growth reserve",
        includes="full 1000-step job, calibration, ten backend evaluations, ten random CPCM evaluations, export and reload",
        excludes="download, reports, later fault/joint experiments",old_run_untouched=True)
    write_json(HERE/"evidence/i3_estimates.json",summary)
    text=["# I3 Qwen WATOS 运行计划","","2026-11-14 新增授权108项：A1 56、A3h 52。与之前I1四项合为112，H/I健康候选将全部达到896。",
        "","108项使用冻结总清单里的I3 spec，原1000步、十后端与十次随机CPCM均保持。新目录 `out/i3_108`；旧 `out/h3_i2_440` 继续原样运行。",
        "","4个首次完整A1代表优先启动；其余prefill FFN优先，再按预计完整耗时从长到短填槽。所有108项立即释放，同类任务不等待首项完成。",
        "","新池每次启动前为旧440项中所有未取得身份匹配成功回执的任务保留名额。因此旧任务即使发生内存终止并重试，两池合计也不超过128。读取系统可用内存，扣除两池未来增长预算并预留3GiB。",
        "",f"初始按84个可用槽和200GiB作业预算排程：{summary['process_hours']:.2f}进程小时，预计{summary['planning_minutes'][0]:.1f}–{summary['planning_minutes'][1]:.1f}分钟。包含完整搜索、评估和导出；下载汇总另计。",
        "","A3h直接用此前同Qwen/WATOS算子的完整实测耗时。A1保留同算子A3h评估与其他成本，仅替换搜索比例；优先用Qwen主硬件的完整实测，prefill FFN使用本轮已完成搜索的实测进度。首次完整A1结果到达后更新对应类成本。",
        "","初始范围是排程值的1–2倍，用于安排并行争用与未测方法，不是统计区间。旧池继续释放槽位，新池会自动补满。",
        "","```text","python e2e/hi_i3.py check","python e2e/hi_i3.py launch","```",""]
    (HERE/"I3_RUN_PLAN.md").write_text("\n".join(text),encoding="utf-8")
    compile(wrapped_source(),"<I3 existing pool with shared slot reservation>","exec")
    return summary


class Controller(previous.Controller):
    def __init__(self,out,workers,old_out=OLD_OUT):
        previous.assemble=assemble
        super().__init__(out,workers)
        self.manifest=MANIFEST
        self.old_out=old_out
        self.old_jobs={j["id"]:j for j in previous.RAW_JOB_ITEMS(OLD_MANIFEST)}
        assert len(self.old_jobs)==440
        self.old_success=set()
        self.old_growth_reserve_gib=0.
        self.old_reserved_slots=440
        for row in previous.read(MANIFEST)["prior_observations"]:
            self.observed[tuple(row["category"])].append(row)

    def reserve_old(self):
        for name,job in self.old_jobs.items():
            if name in self.old_success:
                continue
            path=self.old_out/"receipts"/(name+".json")
            if not path.exists():
                continue
            receipt=previous.read(path)
            if receipt["rc"]!=0 or receipt["identity"]!=previous.pool.identity(job) or receipt["result"]!=job["result"]:
                continue
            result=previous.read(self.old_out/job["result"])
            if result["status"]=="ok" and all(result[k]==v for k,v in job["spec"].items()):
                self.old_success.add(name)
        self.old_reserved_slots=440-len(self.old_success)
        status=previous.read(self.old_out/"h3_i2.status.json")
        self.old_growth_reserve_gib=float(status["growth_reserve_gib"]) if self.old_reserved_slots else 0.
        return self.old_reserved_slots

    def write(self,path,value):
        path=Path(path)
        if path.name==NAME+".status.json":
            capacity=max(0,self.workers-self.old_reserved_slots)
            value.update(old_successful_receipts=len(self.old_success),old_reserved_slots=self.old_reserved_slots,
                allowed_new_active=capacity,combined_reserved_and_active=self.old_reserved_slots+value["active"],
                old_growth_reserve_gib=self.old_growth_reserve_gib)
            assert value["active"]<=capacity
            original=self.workers
            if capacity:
                self.workers=capacity
                estimate_status=dict(value,available_gib=max(0,value["available_gib"]-self.old_growth_reserve_gib))
                estimate=super().eta(estimate_status)
                self.workers=original
            else:
                estimate=dict(queue_seconds=None,planning_seconds=None,reason="All slots reserved for unfinished old jobs")
            overdue=[r["id"] for r in value["running"] if r["seconds"]>=self.by_id[r["id"]]["estimated_seconds"]]
            if overdue:
                estimate.update(queue_seconds=None,planning_seconds=None,estimated_finish_utc=None,overdue_ids=overdue)
            estimate["capacity_assumption"]=capacity
            value["estimate"]=estimate
            value["failed_attempt_count"]=len(list((self.out/"failed_attempts").glob("*/*/receipt.json")))
            write_json(self.out/"estimated_finish.json",estimate)
            write_json(path,value)
        else:
            super().write(path,value)

    def run(self):
        self.out.mkdir(parents=True,exist_ok=True)
        (self.out/"receipts").mkdir(exist_ok=True)
        self.recover()
        for path in self.receipt_paths():
            receipt=previous.read(path)
            if receipt["rc"]==0:self.success_receipt(receipt)
        self.save_release()
        pool=previous.pool
        pool.category,pool.memory_key=previous.category,previous.memory_key
        pool.memory_requirement=lambda job,*args:max(previous.RAW_MEMORY_REQUIREMENT(job,*args),job["peak_rss_gb"])
        pool.job_items=lambda path:self.jobs
        pool.write_json=self.write
        pool.subprocess=types.SimpleNamespace(Popen=self.spawn,STDOUT=subprocess.STDOUT)
        namespace=dict(vars(pool),_hi=self)
        exec(compile(wrapped_source(),"<I3 shared-slot existing pool>","exec"),namespace)
        rc=namespace["execute"](self.manifest,self.out,self.workers,3,NAME,4,self.release)
        write_json(self.out/"controller_finish.json",dict(rc=rc,completed=len(self.success),expected=108,
            utc=datetime.now(timezone.utc).isoformat()))
        return rc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("check","plan","launch","run"))
    parser.add_argument("--out",type=Path,default=HERE/"out/i3_108")
    parser.add_argument("--old-out",type=Path,default=OLD_OUT)
    parser.add_argument("--workers",type=int,default=128)
    args=parser.parse_args()
    assert 1<=args.workers<=128
    if args.command in ("check","plan"):
        print(json.dumps(plan(),ensure_ascii=False,indent=2));return 0
    args.out=args.out.resolve();args.old_out=args.old_out.resolve()
    assert args.out!=args.old_out
    args.out.mkdir(parents=True,exist_ok=True)
    if args.command=="launch":
        start=args.out/(NAME+".start.json")
        if start.exists() and previous.pool.process_alive(previous.read(start)["pid"]):
            raise RuntimeError("I3 controller already running")
        write_json(args.out/"scheduler_policy.json",dict(running_growth_fraction=.25))
        env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMBA_NUM_THREADS="1",
            CRISP_E_ARTIFACT=str(ROOT/"e2e/work/artifact"),PYTHONUTF8="1")
        command=[sys.executable,"-u","-B","e2e/hi_i3.py","run","--out",str(args.out),"--old-out",str(args.old_out),"--workers",str(args.workers)]
        options=dict(start_new_session=True) if sys.platform!="win32" else dict(creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        with (args.out/(NAME+".log")).open("ab") as stream:
            process=previous.RAW_POPEN(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,**options)
        write_json(start,dict(pid=process.pid,command=command,created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256={name:hashlib.sha256((ROOT/"e2e"/name).read_bytes()).hexdigest() for name in ("hi_i3.py","hi_next.py","fg_batch.py","hi_run.py")}))
        print(json.dumps(dict(pid=process.pid,jobs=108,out=str(args.out))),flush=True);return 0
    return Controller(args.out,args.workers,args.old_out).run()


if __name__=="__main__":
    raise SystemExit(main())
