"""Summarize the first serial J batch from saved JSON; run no experiment."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "e2e/hi"
REMOTE_ROOTS = ("/home/ubuntu/crisp_hi_20261114/", "/home/ubuntu/crisp_fg_20261113/",
                "/home/ubuntu/crisp_post_fg_20261113/")
J1_FIELDS = ("compile_seconds", "forward_complete_seconds", "analysis_seconds", "full_call_seconds")
FAST_FIELDS = ("pack_seconds", "kernel_seconds", "unpack_seconds", "total_seconds")
J4_FIELDS = ("input_load_platform_seconds", "shared_all256_chunk_seconds", "cpcm_deadline_rank_seconds",
             "selected_strategy_transform_seconds", "policy_and_initial_ready_list_seconds", "total_prepare_seconds")
J5_FIELDS = ("load_state_seconds", "platform_seconds", "baseline_metrics_seconds", "resident_decision_seconds",
             "frozen_graph_verification_seconds", "export_seconds", "complete_case_seconds")
SOURCE_FILES = {"j_timing.py": "e2e/j_timing.py", "post_fg_bb.py": "e2e/post_fg_bb.py",
                "cpcm.py": "cpcm/cpcm.py", "post_fg_split.py": "e2e/post_fg_split.py",
                "post_fg_dstu.py": "e2e/post_fg_dstu.py"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(path):
    text = str(path).replace("\\", "/")
    for prefix in REMOTE_ROOTS:
        if text.startswith(prefix):
            return ROOT / text[len(prefix):]
    return Path(path)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def jobs_from_manifest(path):
    source = read(path)
    items = source["jobs"] if isinstance(source, dict) else source
    jobs = []
    for item in items:
        if "spec" in item:
            jobs.append(item)
        else:
            name = item["job_id"]
            jobs.append(dict(id=name, kind=item["kind"], deps=item.get("depends_on", []), spec=item,
                             result=f"results/{name}.json", argv=["{python}", "-u", "-B",
                             f"e2e/{item.get('module', 'fg_run')}.py", "child", "--spec", "{spec}", "--out", "{out}"]))
    return jobs


def job_identity(job):
    payload = {key: job[key] for key in ("id", "kind", "spec", "result")}
    payload["argv"] = job.get("argv")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def check_file(path, expected, label, evidence):
    actual = sha(path)
    assert actual == expected, (label, str(path), actual, expected)
    evidence.append(dict(label=label, path=str(path.relative_to(ROOT)), sha256=actual, matched=True))


def audit(args):
    jobs = jobs_from_manifest(args.manifest)
    assert len(jobs) == len({job["id"] for job in jobs}) == 45
    assert Counter(job["kind"] for job in jobs) == dict(j1=18, j1_startup=6, j4=12, j5=9)
    batches = [read(path) for path in args.out.glob("*.batch.json")]
    batch = next(item for item in batches if item["completed"] == 45)
    assert batch["workers"] == 1 and batch["max_active"] == 1
    assert not batch["failed"] and not batch["pending"]
    assert {row["id"] for row in batch["runs"]} == {job["id"] for job in jobs}
    file_checks, receipts, records = [], [], []
    code = {name: sha(ROOT / rel) for name, rel in SOURCE_FILES.items()}
    check_file(args.runtime, code["j_timing.py"], "archived J runtime equals current source", file_checks)
    for job in jobs:
        record = read(args.out / job["result"])
        receipt = read(args.out / "receipts" / (job["id"] + ".json"))
        assert record["status"] == "ok" and record["spec"] == job["spec"]
        assert read(args.out / "specs" / (job["id"] + ".json")) == job["spec"]
        assert receipt["id"] == record["job_id"] == job["id"]
        assert receipt["rc"] == 0 and receipt["identity"] == job_identity(job)
        assert not receipt["cached"]
        assert record["environment"]["code_sha256"] == code
        assert set(record["environment"]["thread_settings"].values()) == {"1"}
        assert record["backend_calls"] == record["search_runs"] == 0
        receipts.append(dict(job_id=job["id"], kind=job["kind"], receipt_rc=0, identity=receipt["identity"],
                             result_sha256=sha(args.out / job["result"]),
                             process_seconds=record["wall_seconds"], receipt_seconds=receipt["wall_seconds"],
                             peak_rss_gib=record["peak_rss_gb"], all_recorded_source_hashes_match=True))
        records.append(record)
    preparations = [read(path) for path in sorted(args.prepare_out.glob("results/J1_prepare_*.json"))]
    assert len(preparations) == 3
    preparation_jobs = {job["id"]: job for job in jobs_from_manifest(args.manifest.with_name("J1_prepare.json"))}
    preparation_batch = next(read(path) for path in args.prepare_out.glob("*.batch.json"))
    assert preparation_batch["completed"] == 3 and not preparation_batch["failed"] and not preparation_batch["pending"]
    for row in preparations:
        assert row["status"] == "ok" and row["environment"]["code_sha256"] == code
        job = preparation_jobs[row["job_id"]]
        receipt = read(args.prepare_out / "receipts" / (row["job_id"] + ".json"))
        assert receipt["rc"] == 0 and receipt["identity"] == job_identity(job) and not receipt["cached"]
        assert row["spec"] == job["spec"]
        for key in ("snapshot", "raw"):
            source_key = "source_raw" if key == "raw" else key
            check_file(local(row["spec"][source_key]), row["input_sha256"][key], row["job_id"] + ":" + key, file_checks)
        check_file(local(row["graph_file"]), row["graph_file_sha256"], row["job_id"] + ":full graph", file_checks)
    graph_inputs = {row["graph_file_sha256"]: row["spec"]["input_id"] for row in preparations}
    for row in records:
        spec = row["spec"]
        if spec["kind"] in ("j1", "j1_startup"):
            assert row["input_sha256"] == spec["prepared_graph_sha256"]
            assert row["input_sha256"] in graph_inputs
            row["input_id"] = graph_inputs[row["input_sha256"]]
        elif spec["kind"] == "j4":
            path = ROOT / "e2e/fg/out/main/results" / (spec["graph_job"] + ".json")
            check_file(path, spec["graph_sha256"], spec["job_id"] + ":source metadata", file_checks)
            graph = read(path)
            for stream in (0, 1):
                check_file(ROOT / graph["files"][f"orig_{stream}"], spec["graph_files_sha256"][str(stream)],
                           spec["job_id"] + f":stream {stream}", file_checks)
            assert row["rank"] == spec["rank"] and row["deadlines"] == spec["frozen_C4_deadlines"]
        else:
            assert row["axis"] == spec["axis"] and row["graph_sha256"] == spec["graph_sha256"]
            assert row["factor"] == 2 and [c["axis"] for c in row["candidates"]] == ["N", "M"]
    isolation = read(args.isolation_evidence) if args.isolation_evidence else None
    source_inputs = read(args.input_evidence) if args.input_evidence else None
    if isolation:
        assert isolation["workers"] == 1 and not isolation["other_experiment_workers"]
        started = datetime.fromisoformat(isolation["j_start_utc"]).timestamp()
        assert all(item["batch_written_unix"] < started and not item["failed"]
                   for item in isolation["prior_batches"].values())
    if source_inputs:
        assert {row["case"] for row in source_inputs["cases"]} == {"FACE_proj", "G23_ffn", "G33_ffn"}
        for case in source_inputs["cases"]:
            assert len(case["files"]) == 5
            for item in case["files"]:
                assert local(item["source"]).stat().st_size == item["bytes"]
                check_file(local(item["source"]), item["sha256"], "server source input at archiving", file_checks)
    assert len({row["environment"]["host"] for row in records}) == 1
    assert len({row["instrumented_function_sha256"] for row in records if row["spec"]["kind"] == "j5"}) == 1
    evidence = dict(status="passed", jobs=len(records), counts=dict(Counter(row["spec"]["kind"] for row in records)),
                    manifest=dict(path=str(args.manifest.relative_to(ROOT)), sha256=sha(args.manifest)),
                    batch=batch, code_sha256=code, input_files=file_checks, receipts=receipts,
                    preparation_batch=preparation_batch,
                    j5_input_sidecar=source_inputs, serial_isolation=isolation,
                    timing_only=True, backend_calls=0, search_runs=0, deferred=["J2", "J3", "J1 Llama/Qwen", "J5 H/I"])
    return records, preparations, evidence


def summarize(kind, name, metric, values, process_unit="independent process"):
    assert len(values) == 3, (kind, name, metric, values)
    return dict(kind=kind, input=name, metric=metric, unit="seconds", process_count=3,
                median=statistics.median(values), minimum=min(values), maximum=max(values),
                process_unit=process_unit)


def summarize_records(records, fields, kind, group):
    groups = defaultdict(list)
    for row in records:
        groups[group(row)].append(row)
    return [summarize(kind, name, field, [row[field] for row in rows])
            for name, rows in sorted(groups.items()) for field in fields]


def build_tables(records, preparations):
    summaries, samples, process_rows, warmups, equality = [], [], [], [], []
    j1 = [row for row in records if row["spec"]["kind"] == "j1"]
    for row in j1:
        assert len(row["steady"]) == 5
        assert [item["repetition"] for item in row["steady"]] == list(range(1, 6))
        assert row["warmup"]["warmup"] and all(not item["warmup"] for item in row["steady"])
        base = dict(job_id=row["job_id"], input_id=row["input_id"], implementation=row["implementation"],
                    process_repeat=row["spec"]["repetition"])
        warmups.append(dict(base, **row["warmup"]))
        fields = J1_FIELDS + (FAST_FIELDS if row["implementation"] == "fast" else ())
        proc = dict(base, load_seconds=row["load_seconds"], platform_seconds=row["platform_seconds"],
                    process_seconds=row["wall_seconds"], peak_rss_gib=row["peak_rss_gb"],
                    **{key: row["warmup"][key] for key in ("events", "activities", "hops", "T")})
        for field in fields:
            proc[field] = statistics.median(item[field] for item in row["steady"])
        process_rows.append(proc)
        for item in row["steady"]:
            assert item["full_call_seconds"] >= sum(item[k] for k in J1_FIELDS[:3])
            if row["implementation"] == "fast":
                assert math.isclose(sum(item[k] for k in FAST_FIELDS[:3]), item["total_seconds"], rel_tol=1e-12)
                assert item["forward_complete_seconds"] >= item["total_seconds"]
            samples.append(dict(base, **item))
    for input_id in sorted({row["input_id"] for row in j1}):
        group = [row for row in j1 if row["input_id"] == input_id]
        assert len(group) == 6 and Counter(row["implementation"] for row in group) == dict(original=3, fast=3)
        calls = [item for row in group for item in [row["warmup"], *row["steady"]]]
        signatures = {(r["T"], r["events"], r["activities"], r["analysis_sha256"]) for r in calls}
        assert len(signatures) == 1
        T, events, activities, analysis_hash = signatures.pop()
        equality.append(dict(input_id=input_id, processes=6, compared_calls=len(calls), schedule_T=T,
                             events=events, activities=activities, critical_path_and_blocking_sha256=analysis_hash,
                             exact_T_and_analysis_equal=True))
        for implementation in ("original", "fast"):
            selected = [row for row in process_rows if row["input_id"] == input_id and row["implementation"] == implementation]
            fields = J1_FIELDS + (FAST_FIELDS if implementation == "fast" else ())
            for field in fields:
                summaries.append(summarize("J1 steady", input_id + ":" + implementation, field,
                                           [row[field] for row in selected], "median of five steady calls in one fresh process"))
            for field in ("load_seconds", "platform_seconds", "process_seconds"):
                summaries.append(summarize("J1 process", input_id + ":" + implementation, field,
                                           [row[field] for row in selected]))
    startup = []
    cold_caches = {row["job_id"]: row["numba_cache_dir"] for row in records
                   if row["spec"].get("startup_mode") == "first_jit"}
    assert len(cold_caches) == len(set(cold_caches.values())) == 3
    for row in records:
        if row["spec"]["kind"] == "j1_startup":
            assert not row["steady"] and row["warmup"]["warmup"] and row["cache_files"] and row["numba_signatures"]
            steady_reference = next(item for item in equality if item["input_id"] == row["input_id"])
            assert row["warmup"]["T"] == steady_reference["schedule_T"]
            assert row["warmup"]["analysis_sha256"] == steady_reference["critical_path_and_blocking_sha256"]
            if row["startup_mode"] == "disk_cache":
                assert row["numba_cache_dir"] == cold_caches[row["spec"]["cache_source_job"]]
            startup.append(dict(job_id=row["job_id"], mode=row["startup_mode"], process_repeat=row["spec"]["repetition"],
                                load_seconds=row["load_seconds"], platform_seconds=row["platform_seconds"],
                                process_seconds=row["wall_seconds"], **row["warmup"]))
    summaries += summarize_records(startup, J1_FIELDS + FAST_FIELDS + ("load_seconds", "platform_seconds", "process_seconds"),
                                   "J1 startup", lambda row: row["mode"])
    prepare_rows = [dict(input_id=row["spec"]["input_id"], **{k: row[k] for k in
                    ("pre_events", "full_events", "load_seconds", "initial_broadcast_seconds", "prepare_total_seconds")})
                    for row in preparations]
    j4 = [row for row in records if row["spec"]["kind"] == "j4"]
    j4_rows = [dict(job_id=row["job_id"], hardware=row["spec"]["hardware_profile"], arm=row["spec"]["arm"],
                   process_repeat=row["spec"]["repetition"], events=row["events"], ready_events=row["ready_events"],
                   process_seconds=row["wall_seconds"], peak_rss_gib=row["peak_rss_gb"],
                   **{key: row[key] for key in J4_FIELDS}) for row in j4]
    for row in j4_rows:
        row["resident_prepare_seconds"] = row["total_prepare_seconds"] - row["input_load_platform_seconds"]
        row["other_prepare_seconds"] = row["total_prepare_seconds"] - sum(row[key] for key in J4_FIELDS[:-1])
        assert row["other_prepare_seconds"] >= 0
    summaries += summarize_records(j4_rows, J4_FIELDS + ("resident_prepare_seconds", "other_prepare_seconds", "process_seconds"), "J4",
                                   lambda row: row["hardware"] + ":" + row["arm"])
    j5 = [row for row in records if row["spec"]["kind"] == "j5"]
    j5_rows, phase_rows = [], []
    for row in j5:
        item = dict(job_id=row["job_id"], case=row["case"], axis=row["axis"], process_repeat=row["spec"]["repetition"],
                    process_seconds=row["wall_seconds"], peak_rss_gib=row["peak_rss_gb"],
                    **{key: row[key] for key in J5_FIELDS})
        item["other_case_seconds"] = row["complete_case_seconds"] - sum(row[key] for key in J5_FIELDS[:-1])
        assert item["other_case_seconds"] >= 0
        j5_rows.append(item)
        for candidate in row["candidates"]:
            phase_rows.append(dict(job_id=row["job_id"], case=row["case"], axis=candidate["axis"],
                                   process_repeat=row["spec"]["repetition"], selected=candidate["axis"] == row["axis"],
                                   build_seconds=candidate["build_seconds"], **candidate["phases"]))
    summaries += summarize_records(j5_rows, J5_FIELDS + ("other_case_seconds", "process_seconds"), "J5", lambda row: row["case"])
    phase_keys = tuple(j5[0]["candidates"][0]["phases"])
    summaries += summarize_records(phase_rows, ("build_seconds",) + phase_keys, "J5 axis",
                                   lambda row: row["case"] + ":" + row["axis"])
    return dict(J_TIMING=summaries, J1_PROCESS=process_rows, J1_SAMPLES=samples, J1_WARMUP=warmups,
                J1_STARTUP=startup, J1_PREPARATION=prepare_rows, J1_EQUIVALENCE=equality,
                J4_PROCESS=j4_rows, J5_PROCESS=j5_rows, J5_AXIS_PHASES=phase_rows)


def render(tables, audit_data, records):
    index = {(row["kind"], row["input"], row["metric"]): row for row in tables["J_TIMING"]}
    def cell(kind, name, field, scale=1000):
        row = index[(kind, name, field)]
        return f"{row['median']*scale:.3f} [{row['minimum']*scale:.3f}, {row['maximum']*scale:.3f}]"
    env = records[0]["environment"]
    batch = audit_data["batch"]
    lines = ["# J 阶段第一批计时结果", "", "## 本批完成情况", "",
        f"本批 45/45 个作业成功：J1 稳态 18 个新进程、首次编译/磁盘缓存加载 6 个新进程，J4 准备计时 12 个、J5 切分计时 9 个。"
        f"批次墙钟 {batch['wall_seconds']:.3f} 秒，最多同时运行 1 个作业。所有作业均为计时与已有图的结构核对，没有新增 SA 搜索或后端质量评估。", "",
        f"机器：{env['cpu']}；{env['cpu_count']} 个逻辑 CPU；Python {env['python'].split()[0]}、NumPy {env['numpy']}、Numba {env['numba']}。"
        "OMP、MKL、OpenBLAS、Numba 均为 1 线程。服务器并行实验先结束，再进行这一批单进程串行计时；普通系统服务、腾讯云代理和已有三分钟 COS 备份仍在运行。", "",
        "作业清单、spec、receipt、5 个计时相关源文件 SHA-256、J1 输入快照/原始状态/完整图，以及 J4 输入文件已逐项核对。"
        + ("J5 的 15 个输入文件另与服务器归档时读取的 SHA-256 清单核对；这份摘要形成于运行结束后的归档，不替代运行中选中图的结构核对。" if audit_data["j5_input_sidecar"] else
           "J5 在运行中对选中图的语义 SHA-256 和切分轴逐项核对；其输入文件 SHA-256 未在本报告单独核对。"), "",
        "J2、J3 尚未开始；新模型的 J1 以及 H/I 新代表点的 J5 仍属后续任务。", "",
        "## J1：原实现与加速实现", "",
        "每个图、每种实现分别启动 3 个新进程；每进程先运行 1 次预热，再运行 5 次稳态调用。"
        "下表先取各进程 5 次的中位数，再给出 3 个进程的中位数 [最小, 最大]，单位毫秒；没有把同进程的 15 次调用当作 15 次独立测量。", "",
        "| 完整图 | 实现 | 编译事件图 | 前向推演（含数组转换） | 关键路径/阻塞分析 | 完整调用 |",
        "|---|---|---|---|---|---|"]
    inputs = [row["input_id"] for row in tables["J1_EQUIVALENCE"]]
    for name in inputs:
        for implementation, label in (("original", "原"), ("fast", "加速")):
            key = name + ":" + implementation
            lines.append("| " + " | ".join([name, label] + [cell("J1 steady", key, field) for field in J1_FIELDS]) + " |")
    lines += ["", "`编译事件图` 是把完整事件字典编译为 CPCM 输入图，不是 Numba 的机器码编译。"
              "`完整调用` 包括事件图编译、前向推演和关键路径/阻塞分析；不含单独记录的文件加载、平台创建和第一次完整广播准备。", "",
              "| 完整图 | 前向推演加速比 | 完整调用加速比 | events | 实际调度活动数 | Schedule.T | 原/快 CP 与 blocking |",
              "|---|---|---|---|---|---|---|"]
    for row in tables["J1_EQUIVALENCE"]:
        name = row["input_id"]
        ratios = [index[("J1 steady", name + ":original", field)]["median"] /
                  index[("J1 steady", name + ":fast", field)]["median"]
                  for field in ("forward_complete_seconds", "full_call_seconds")]
        lines.append(f"| {name} | {ratios[0]:.2f}× | {ratios[1]:.2f}× | {row['events']} | {row['activities']} | {row['schedule_T']} | 36 次调用一致 |")
    lines += ["", "加速比为两种实现的进程中位数之比。每个图的原/快 6 个进程、共 36 次调用（含预热）均得到相同的 Schedule.T、活动数及关键路径/阻塞分析哈希。", "",
              "| 完整图 | 数组打包 | Numba 内核 | 调度对象还原 |",
              "|---|---|---|---|"]
    for name in inputs:
        lines.append("| " + " | ".join([name] + [cell("J1 steady", name + ":fast", field) for field in FAST_FIELDS[:3]]) + " |")
    lines += ["", "这三项包含在前向推演时间内，不再加到完整调用上；各列独立取中位数，列中位数的和不必等于总项中位数。", "",
              "### 加载与第一次完整广播准备", "",
              "| 完整图 | 实现 | 完整图文件加载 | 平台建立 |",
              "|---|---|---|---|"]
    for name in inputs:
        for implementation in ("original", "fast"):
            lines.append("| " + " | ".join([name, implementation] + [cell("J1 process", name + ":" + implementation, field)
                         for field in ("load_seconds", "platform_seconds")]) + " |")
    lines += ["", "| 图准备（单次记录） | 原始 events | 完整 events | 加载快照/原始状态 ms | 第一次完整广播准备 ms | 图准备总计 ms |",
              "|---|---|---|---|---|---|"]
    for row in tables["J1_PREPARATION"]:
        lines.append(f"| {row['input_id']} | {row['pre_events']} | {row['full_events']} | {row['load_seconds']*1000:.3f} | {row['initial_broadcast_seconds']*1000:.3f} | {row['prepare_total_seconds']*1000:.3f} |")
    lines += ["", f"这三个准备作业先行生成含 W1 末端广播的完整图，3 项并行批次墙钟 {audit_data['preparation_batch']['wall_seconds']:.3f} 秒；"
              "这里只有各 1 次准备记录，作为加载/广播成本说明，未并入正式串行稳态统计。", "",
              "### 首次 JIT 与已有磁盘缓存的新进程", "",
              "同一 GPT64 decode FFN 完整图做 3 对新进程：每对先创建独立空缓存完成第一次 JIT，再由后一个进程读取前者磁盘缓存。"
              "每个进程只调用 1 次。单位毫秒，同样报告 3 个进程的中位数 [最小, 最大]。", "",
              "| 状态 | 编译事件图 | 内核调用（含 JIT 或缓存加载） | 前向推演完整 | 分析 | 完整调用 |",
              "|---|---|---|---|---|---|"]
    for mode in ("first_jit", "disk_cache"):
        lines.append("| " + " | ".join([mode] + [cell("J1 startup", mode, field) for field in
                     ("compile_seconds", "kernel_seconds", "forward_complete_seconds", "analysis_seconds", "full_call_seconds")]) + " |")
    lines += ["", "这里的内核调用时间包括首次编译或磁盘缓存载入，以及本次内核执行；当前计时没有把这些子项再分开。稳态表已排除首次预热。", "",
              "## J4：DSTU 控制准备", "",
              "从既有 P/D 图开始，先做 All256 切块并分别推演两条流，依照固定的 C4 规则得到期限与顺序，"
              "再生成 A256 或 S256 的事件、构造策略对象和初始 ready 列表。这里测的是 Python 控制准备，未执行后端队列，也不是硬件在线包仲裁延迟。单位毫秒。", "",
              "| 硬件/策略 | 输入+平台 | All256 切块 | CPCM 期限/顺序 | 所选策略变换 | 策略+ready | 常驻图准备总计 | 冷流程总计 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name in sorted({row["hardware"] + ":" + row["arm"] for row in tables["J4_PROCESS"]}):
        fields = J4_FIELDS[:-1] + ("resident_prepare_seconds", "total_prepare_seconds")
        lines.append("| " + " | ".join([name] + [cell("J4", name, field) for field in fields]) + " |")
    lines += ["", "A256 直接使用已经生成的 All256 图，所以所选策略变换项只含引用赋值；S256 才有第二次策略变换。"
              "`常驻图准备总计` 是冷流程减去输入/平台项，仍包含 All256、CPCM 期限计算和策略构造。", "",
              "## J5：Split-m2 决策与冷流程", "",
              "每例在新的进程里完整构造 N、M 两个候选：识别受损计算及合法轴、放置/路由评分、构造图及输入传输、处理末端广播、核对工作量和字节。"
              "随后按额外舍入计算量、链路传输字节、轴次序选定结果。决策时间包括两轴全部构造与选轴，不仅是一次轴评分。单位毫秒。", "",
              "| 代表点 | 选中轴 | 状态加载 | 两平台建立 | 健康工作量统计 | 两轴构造+选轴 | 选中图核对 | 导出 | 冷流程总计 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name in sorted({row["case"] for row in tables["J5_PROCESS"]}):
        axis = next(row["axis"] for row in tables["J5_PROCESS"] if row["case"] == name)
        lines.append("| " + " | ".join([name, axis] + [cell("J5", name, field) for field in J5_FIELDS]) + " |")
    lines += ["", "冷流程包括数据读取、健康和故障平台建立、基准统计、两轴决策、最终结构核对、结果写盘与少量准备代码。"
              "它不是运行中的单条消息处理时间。这里的冷流程从计时函数开始，不含解释器启动和先前模块导入；整个作业的进程时间另存于分项表。"
              "N/M 每轴六项构造耗时以及原始 9 个进程记录也保存在分项表。", "",
              "## 产物与复算", "",
              "- `tables/J_TIMING.csv`、`J_TIMING.json`：全部分项的 3 进程中位数和范围。",
              "- `tables/J1_PROCESS.csv`、`J1_SAMPLES.csv`、`J1_WARMUP.csv`：每进程统计、90 次稳态调用及18次预热。",
              "- `tables/J1_STARTUP.csv`、`J1_PREPARATION.csv`、`J1_EQUIVALENCE.csv`：6次启动、3次图准备、原快一致核对。",
              "- `tables/J4_PROCESS.csv`、`J5_PROCESS.csv`、`J5_AXIS_PHASES.csv`：控制准备与两轴构造分项。",
              "- `evidence/j_first_completion_audit.json`：45项receipt、源码与输入核对、串行运行记录。", "",
              "本文件由 `e2e/hi/j_report.py` 从保存结果生成；后处理不运行仿真。仓库根目录执行 `python e2e/hi/j_report.py` 可复算。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "out/j_first_serial")
    parser.add_argument("--prepare-out", type=Path, default=HERE / "out/j1_prepare")
    parser.add_argument("--manifest", type=Path, default=HERE / "j/J_FIRST_SERIAL.json")
    parser.add_argument("--runtime", type=Path, default=HERE / "runtime/j_20261114/j_timing.py")
    parser.add_argument("--isolation-evidence", type=Path, default=HERE / "evidence/j_serial_conditions.json")
    parser.add_argument("--input-evidence", type=Path, default=HERE / "evidence/j5_archival_input_hashes.json")
    args = parser.parse_args()
    records, preparations, evidence = audit(args)
    tables = build_tables(records, preparations)
    for name, rows in tables.items():
        table(HERE / "tables" / (name + ".csv"), rows)
    write(HERE / "J_TIMING.json", dict(summary=tables["J_TIMING"], tables=tables,
                                       environment=records[0]["environment"], code_sha256=evidence["code_sha256"],
                                       aggregation="median of three independent-process medians; J1 process median uses five steady calls"))
    write(HERE / "evidence/j_first_completion_audit.json", evidence)
    (HERE / "J_FIRST_REPORT.md").write_text(render(tables, evidence, records), encoding="utf-8")
    print(json.dumps(dict(status="passed", completed=evidence["jobs"], summary_rows=len(tables["J_TIMING"]),
                          report=str(HERE / "J_FIRST_REPORT.md")), ensure_ascii=False))


if __name__ == "__main__":
    main()
