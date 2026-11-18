"""Audit completed S256/gating records, reuse chains and delivery inputs; no simulation."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "e2e/dstu_supplement"
OUT = BASE / "out/main"
EXPECTED = dict(prepare=12, gate=12, evaluate=540)
TRANSPORTS = ("BB0", "A256", "S256")
PHASES, SEEDS = (.1, .5, .9), (1, 2, 3)
INPUTS = {}


def local(value):
    text = str(value).replace("\\", "/")
    if "/e2e/" in text:
        return ROOT / ("e2e/" + text.split("/e2e/", 1)[1])
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def relative(path):
    return local(path).resolve().relative_to(ROOT.resolve()).as_posix()


def source(path):
    path = local(path)
    content = path.read_bytes()
    INPUTS[relative(path)] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    return content


def read(path):
    return json.loads(source(path).decode("utf-8-sig"))


def sha(path):
    return hashlib.sha256(source(path)).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(x) for x in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def job_identity(job):
    payload = {key: job[key] for key in ("id", "kind", "spec", "result")}
    payload["argv"] = job.get("argv")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def instant(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def same_execution(left, right):
    assert left["evaluation_identity"] == right["evaluation_identity"]
    assert left["initial_order_sha256"] == right["initial_order_sha256"]
    assert left["fresh_protocol"] == right["fresh_protocol"] == "fresh_transform_v1"
    li = left.get("resolved_simulation_spec", left["spec"])
    ri = right.get("resolved_simulation_spec", right["spec"])
    assert li["reference"] == ri["reference"]
    for field in ("completion", "release", "simulated_total", "unhidden_compute", "unhidden_communication"):
        assert left[field] == right[field], field


def reuse_chain(filename):
    """Follow the three alias forms present in this run and its frozen inputs."""
    chain, seen = [], set()
    path = local(filename)
    while True:
        name = relative(path)
        assert name not in seen, name
        seen.add(name)
        record = read(path)
        chain.append(dict(path=name, sha256=sha(path), status=record["status"], backend_calls=record["backend_calls"],
                          simulation_seconds=record["simulation_seconds"]))
        if record["backend_calls"] == 1:
            assert record["status"] == "ok"
            return chain
        assert record["backend_calls"] == 0
        if "reused_from" in record:
            target = local(record["reused_from"])
            expected_sha = record["reused_sha256"] if "reused_sha256" in record else record["original_sha256"]
            assert sha(target) == expected_sha
        elif "reused_from_job" in record:
            target = path.parent / (record["reused_from_job"] + ".json")
        else:
            target = path.parent / (record["batch_alias_from"] + ".json")
            assert sha(target) == record["batch_alias_result_sha256"]
        previous = read(target)
        same_execution(record, previous)
        path = target


def inspect_job(job, out):
    name = job["id"]
    spec_path, receipt_path, result_path = (out / folder / (name + ".json") for folder in ("specs", "receipts", "results"))
    spec, receipt, result = read(spec_path), read(receipt_path), read(result_path)
    assert spec == job["spec"] == result["spec"]
    assert receipt["id"] == result["job_id"] == name
    assert receipt["identity"] == job_identity(job)
    assert receipt["rc"] == 0 and receipt["result"] == job["result"]
    assert result["status"] in ("ok", "reused")
    logs = sorted((out / "logs").glob(name + ".*.log"), key=lambda p: int(p.name.rsplit(".", 2)[1]))
    log = logs[-1]
    summaries = []
    for line in source(log).decode("utf-8", errors="replace").splitlines():
        if line.startswith("{") and line.endswith("}"):
            value = json.loads(line)
            if value.get("job_id") == name:
                summaries.append(value)
    summary = summaries[-1]
    for key in ("job_id", "status", "backend_calls", "wall_seconds"):
        assert summary[key] == result[key], (name, key)
    assert receipt["wall_seconds"] >= result["wall_seconds"]
    return result, dict(job_id=name, kind=job["kind"], status=result["status"], identity=receipt["identity"],
        result=relative(result_path), result_sha256=sha(result_path), receipt=relative(receipt_path), receipt_sha256=sha(receipt_path),
        spec=relative(spec_path), spec_sha256=sha(spec_path), log=relative(log), log_sha256=sha(log),
        log_started_utc=datetime.fromtimestamp(int(log.name.rsplit(".", 2)[1])/1e9, timezone.utc).isoformat(),
        task_seconds=result["wall_seconds"], receipt_seconds=receipt["wall_seconds"], peak_rss_gib=receipt["peak_rss_gb"],
        backend_calls=result["backend_calls"])


def state_key(spec):
    return tuple(spec[k] for k in ("scope", "model", "hardware_profile", "mapping_state"))


def audit_prepare(results):
    prepared = {}
    calls = 0
    for name, result in results.items():
        spec = result["spec"]
        if spec["kind"] != "prepare":
            continue
        assert result["backend_calls"] == 0 and result["cpcm_calls"] == 2
        assert set(result["rank_checks"]) == {str(x) for x in PHASES}
        assert all(r["same"] and r["frozen_rank"] == r["recomputed_rank"] for r in result["rank_checks"].values())
        assert len(result["identities"]) == 9 and len(result["masks"]) == 1
        for item in result["source_inputs"]:
            assert sha(item["path"]) == item["sha256"]
        prepared[state_key(spec)] = result
        calls += result["cpcm_calls"]
    assert len(prepared) == 12 and calls == 24
    return prepared, calls


def audit_gate(results):
    gates, decisions, frozen = {}, [], []
    for name, result in results.items():
        spec = result["spec"]
        if spec["kind"] != "gate":
            continue
        assert result["backend_calls"] == 0 and result["cpcm_prediction_calls"] == 1
        assert result["cpcm_validation_calls"] == 0 and result["cpcm_calls"] == 1
        assert result["decision_sha256"] == digest(result["decisions"])
        manifest = result["input_manifest"]
        assert manifest["identity_sha256"] == digest({k:v for k,v in manifest.items() if k != "identity_sha256"})
        assert spec["input_identity_sha256"] == manifest["identity_sha256"]
        for item in manifest["files"]:
            assert sha(item["path"]) == item["sha256"]
        for path, expected in manifest["code_sha256"].items():
            assert sha(path) == expected, (name, path)
        prediction = result["prediction"]
        assert prediction["predicted_KV_end_ns"] == max(row["end_ns"] for row in prediction["kv_events"])
        assert prediction["predicted_KV_start_ns"] == min(row["first_start_ns"] for row in prediction["kv_events"])
        by_activity = {row["activity_id"]: row for row in prediction["kv_activities"]}
        for event in prediction["kv_events"]:
            assert by_activity[event["terminal_activity"]]["end_ns"] == event["end_ns"]
            assert event["latest_activity_end_ns"] == max(by_activity[a]["end_ns"] for a in event["activity_ids"])
        assert {row["phase"] for row in result["decisions"]} == set(PHASES)
        for row in result["decisions"]:
            action = "S256" if row["shared_resource_count"] > 0 and row["tau_ns"] < row["predicted_KV_end_ns"] else "BB0"
            assert action == row["chosen_transport"]
            assert row["input_identity_sha256"] == manifest["identity_sha256"]
            decisions.append(dict(context=state_key(spec), **row))
        gates[state_key(spec)] = result
        frozen.append(instant(result["frozen_at_utc"]))
    assert len(gates) == 12 and len(decisions) == 36
    return gates, decisions, max(frozen)


def audit_fresh_identities(prepared, destination):
    checked = {}
    for context, old in prepared.items():
        spec = old["spec"]
        point = "_".join(spec[k] for k in ("scope", "hardware_profile", "mapping_state"))
        path = destination/"evidence/fresh_identity_checks"/(point+".json")
        result = read(path)
        assert result["status"] == "passed" and result["point"] == point
        assert result["backend_calls"] == result["cpcm_calls"] == result["new_searches"] == 0
        assert result["first_S256_mask_and_stats_match"] and result["normal_rank_same_all_phases"]
        assert sha(result["prepare_record"]) == result["prepare_sha256"]
        assert result["source_graph_identity"] == old["source_graph_identity"]
        for item in result["source_inputs"]:
            assert sha(item["path"]) == item["sha256"]
        first = next(iter(old["masks"].values()))
        assert first["mask_sha256"] == result["identities"]["S256_g0.1"]["mask_sha256"]
        assert first["stats"] == result["identities"]["S256_g0.1"]["chunk_stats"]
        checked[context] = result
    assert len(checked) == 12
    return checked


def audit_identity_delivery(fresh, destination):
    finish = read(destination/"out/identity_check/finish.json")
    assert finish["status"] == "passed" and finish["backend_calls"] == finish["cpcm_calls"] == 0
    records = {row["point"]: row for row in finish["results"]}
    assert len(records) == 12 and set(records) == {r["point"] for r in fresh.values()}
    runtime = {}
    for role, folder, version in (("production", "server_runtime", "dstu_supplement_v1"),
                                  ("identity_and_future_prepare", "server_runtime_identity", "dstu_supplement_v2")):
        path = destination/"dist"/folder/"e2e/dstu_supplement_run.py"
        content = source(path)
        assert ("VERSION = '"+version+"'").encode() in content
        runtime[role] = dict(path=relative(path), version=version, sha256=hashlib.sha256(content).hexdigest())
    for result in fresh.values():
        point = result["point"]
        row = read(destination/"out/identity_check"/(point+".receipt.json"))
        assert row == records[point] and row["rc"] == 0
        assert instant(row["started_at_utc"]) <= instant(result["generated_utc"]) <= instant(row["ended_at_utc"])
        log = source(destination/"out/identity_check"/(point+".log")).decode("utf-8")
        summary = json.loads(log.strip().splitlines()[-1])
        assert summary["point"] == point and summary["status"] == "passed"
        assert summary["backend_calls"] == summary["cpcm_calls"] == 0
        assert summary["wall_seconds"] == result["wall_seconds"]
        for path, expected in result["code_sha256"].items():
            saved = destination/"dist/server_runtime_identity"/path
            assert sha(saved if saved.exists() else path) == expected
    archives = []
    for name, expected, count in (
            ("S256_COMPLETE_20261118.tar.gz", "ee76101a132ea1394d0106ee4f0a44b121ae6f33dd5981b39c91c6fe19d03590", 2600),
            ("S256_IDENTITY_20261118.tar.gz", "50e32b7827e3b98fd6c60d6be52911075b2c808de735fe55e73bbce16b7c08c5", 42)):
        path = destination/"dist"/name
        metadata = read(str(path)+".json")
        assert metadata["sha256"] == sha(path) == expected and metadata["files"] == count
        assert path.stat().st_size == metadata["bytes"]
        archives.append(metadata)
    start = min(instant(row["started_at_utc"]) for row in records.values())
    end = max(instant(row["ended_at_utc"]) for row in records.values())
    return dict(status="passed", started_utc=start.isoformat(), completed_utc=end.isoformat(),
                elapsed_seconds=(end-start).total_seconds(), receipts=list(records.values()),
                runtime_sources=runtime, archives=archives, backend_calls=0, CPCM_calls=0)


def audit_evaluations(results, prepared, fresh, gates, latest_freeze, out):
    ledger, chains, groups = [], [], defaultdict(dict)
    counts, by_scope, source_roots = Counter(), Counter(), Counter()
    new_starts = []
    for name, result in results.items():
        spec = result["spec"]
        if spec["kind"] != "evaluate":
            continue
        context = state_key(spec)
        prep, gate = prepared[context], gates[context]
        arm, run_type, seed, phase = (spec[k] for k in ("transport", "run_type", "seed", "phase"))
        assert arm in TRANSPORTS and seed in SEEDS and phase in PHASES and spec["rank_mode"] == "frozen"
        assert result["cpcm_calls"] == 0
        assert sha(spec["gate_record"]) == spec["gate_sha256"]
        assert read(spec["gate_record"]) == gate
        assert read(spec["prepared_record"]) == prep
        inner = result["resolved_simulation_spec"]
        workload = "W1" if spec["scope"] == "S" else "W1_8Q_1KV_v1"
        assert spec["workload_version"] == inner["workload_version"] == workload
        assert inner["reference"] == prep["reference"]
        assert inner["rank"] == prep["rank_checks"][str(phase)]["frozen_rank"]
        assert result["fresh_protocol"] == "fresh_transform_v1"
        graph_record = read(inner["graph_record"])
        assert sha(inner["graph_record"]) == inner["graph_sha256"] == result["source_graph_sha256"]
        assert result["evaluation_identity"]["platform_sha256"] == graph_record["platform_sha256"]
        for stream in (0, 1):
            assert sha(graph_record["files"][f"orig_{stream}"]) == inner["graph_files_sha256"][str(stream)]
        if spec["scope"] != "S":
            assert inner["model_contract_sha256"] == graph_record["model_contract_sha256"]
        expected = fresh[context]["identities"][f"{arm}_g{phase}"]
        identity = expected["joint"] if run_type == "joint" else expected["solos"][str(spec["stream"])]
        expected_identity = dict(identity["evaluation_identity"], seed=seed)
        assert result["evaluation_identity"] == expected_identity
        assert result["initial_order_sha256"] == identity["initial_order_sha256"]
        assert result["mask_sha256"] == expected["mask_sha256"]
        assert result["chunk_stats"]["bytes_conserved"] and result["chunk_stats"]["tree_conserved"]
        assert result["transform_source_sha256"] == sha(ROOT / "e2e/post_fg_dstu.py")
        for attempt in result["reuse_checks"]:
            previous = read(attempt["source"])
            assert sha(attempt["source"]) == attempt["sha256"]
            expected_same = (previous["evaluation_identity"] == result["evaluation_identity"] and
                previous["initial_order_sha256"] == result["initial_order_sha256"] and
                previous["fresh_protocol"] == result["fresh_protocol"] and
                previous.get("resolved_simulation_spec", previous["spec"])["reference"] == inner["reference"])
            assert attempt["same_execution"] == expected_same
        current_path = out / "results" / (name + ".json")
        chain = reuse_chain(current_path)
        chains.append(dict(job_id=name, reuse_kind=result["reuse_kind"], chain=chain))
        source_roots[chain[-1]["path"]] += 1
        counts[run_type] += 1; counts[arm + "_" + run_type] += 1
        counts["actual_backend_calls"] += result["backend_calls"]
        counts["reuse_" + result["reuse_kind"]] += 1
        by_scope[spec["scope"] + "_" + arm + "_" + run_type] += 1
        started, finished = instant(result["started_at_utc"]), instant(result["finished_at_utc"])
        assert started <= finished and latest_freeze < started
        if result["backend_calls"]:
            assert result["status"] == "ok" and result["reuse_kind"] == "new" and len(chain) == 1
            assert result["backend_calls"] == 1
            new_starts.append(started)
        else:
            assert result["status"] == "reused" and result["reuse_identity_checked"] and len(chain) >= 2
        if run_type == "joint":
            metric = result["metrics"]
            ps, ds = results[spec["P_solo"]], results[spec["D_solo"]]
            for stream, solo in ((0, ps), (1, ds)):
                solo_spec = solo["spec"]
                assert state_key(solo_spec) == context and solo_spec["seed"] == seed
                assert solo_spec["transport"] == arm and solo_spec["stream"] == stream
                assert solo["mask_sha256"] == result["mask_sha256"]
            p, d = result["completion"]["0"], result["completion"]["1"] - inner["tau"]
            assert metric["P_completion"] == p and metric["D_latency"] == d
            assert metric["P_solo"] == ps["completion"]["0"] and metric["D_solo"] == ds["completion"]["1"]
            assert metric["paired_delta_P"] == p - metric["P_solo"]
            assert metric["D_interference"] == d / metric["D_solo"] - 1
            rp, rd = inner["reference"]["T_P"], inner["reference"]["T_D"]
            assert metric["Lhat_P"] == (p - 1.5*rp)/rp and metric["Lhat_D"] == (d - 1.5*rd)/rd
            assert metric["Lhat_max"] == max(metric["Lhat_P"], metric["Lhat_D"])
            key = context + (phase, seed)
            assert arm not in groups[key]
            groups[key][arm] = name
        ledger.append(dict(job_id=name, context=context, transport=arm, run_type=run_type,
            phase=phase, seed=seed, status=result["status"], backend_calls=result["backend_calls"], reuse_kind=result["reuse_kind"],
            started_at_utc=result["started_at_utc"], finished_at_utc=result["finished_at_utc"],
            actual_source=chain[-1]["path"], actual_source_sha256=chain[-1]["sha256"],
            actual_source_simulation_seconds=chain[-1]["simulation_seconds"]))
    assert len(ledger) == 540 and len(groups) == 108
    assert all(set(group) == set(TRANSPORTS) for group in groups.values())
    assert counts["joint"] == 324 and counts["solo"] == 216
    assert all(counts[arm + "_joint"] == 108 and counts[arm + "_solo"] == 72 for arm in TRANSPORTS)
    assert sum(by_scope[f"{scope}_{arm}_joint"] for scope in ("H", "I") for arm in ("BB0", "A256")) == 72
    assert sum(by_scope[f"{scope}_{arm}_solo"] for scope in ("H", "I") for arm in ("BB0", "A256")) == 48
    return ledger, chains, dict(counts), dict(by_scope), dict(source_roots), min(new_starts)


def controller_records(out, jobs):
    finish, live, batch, admitted = (read(out / name) for name in
        ("controller_finish.json", "live_workers.json", "closeout.batch.json", "admitted_jobs.json"))
    assert finish["rc"] == 0 and finish["completed"] == finish["expected"] == len(jobs) == 564
    assert live == {} and batch["completed"] == 564 and not batch["failed"] and not batch["pending"]
    assert {j["id"]:job_identity(j) for j in admitted["jobs"]} == {j["id"]:job_identity(j) for j in jobs}
    assert admitted["accepting"] is False
    monitors = [json.loads(line) for line in source(out / "monitor_history.jsonl").decode().splitlines() if line]
    by_pid = defaultdict(list)
    for record in monitors:
        by_pid[record["pid"]].append(record)
    segments = []
    for pid, records in by_pid.items():
        first, last = records[0], records[-1]
        start = instant(first["utc"]) - timedelta(seconds=first["wall_seconds"])
        segments.append(dict(pid=pid, reconstructed_start_utc=start.isoformat(), first_status_utc=first["utc"],
            last_status_utc=last["utc"], last_completed=last["completed"], last_failed=last["failed"],
            min_available_gib=min(r["min_available_gib"] for r in records), max_active=max(r["max_active"] for r in records)))
    first_start = min(instant(s["reconstructed_start_utc"]) for s in segments)
    end = instant(finish["utc"])
    failures = []
    for path in sorted((out / "failed_attempts").glob("*/*/receipt.json")):
        receipt = read(path)
        assert receipt["rc"] != 0
        log_path = local(receipt["log"])
        log = source(log_path).decode("utf-8", errors="replace")
        kind = "missing_reuse_input_before_backend" if "FileNotFoundError" in log else "other"
        failures.append(dict(job_id=receipt["id"], receipt=relative(path), receipt_sha256=sha(path),
            log=relative(log_path), log_sha256=sha(log_path), rc=receipt["rc"], memory_related=receipt["memory_related"],
            available_gib=receipt["available_gib"], classification=kind,
            finally_completed=read(out/"receipts"/(receipt["id"]+".json"))["rc"] == 0))
    return dict(finish=finish, segments=segments, estimated_first_controller_start_utc=first_start.isoformat(),
        completion_utc=end.isoformat(), full_elapsed_seconds=(end-first_start).total_seconds(),
        elapsed_definition="first status UTC minus its elapsed seconds through final controller completion; includes failed-input wait and restart",
        final_controller_segment_seconds=batch["wall_seconds"],
        min_available_gib=min([batch["min_available_gib"]]+[s["min_available_gib"] for s in segments]),
        max_active=max([batch["max_active"]]+[s["max_active"] for s in segments]), failures=failures,
        failed_attempts=len(failures), OOM_attempts=sum(r["memory_related"] for r in failures))


def extra_runs(destination):
    probes = []
    for model in ("llama", "qwen3moe"):
        path = destination/"out/probes/evidence"/f"baseline_protocol_check_{model}_healthy_A3h_replay.json"
        result = read(path)
        assert result["status"] == "passed" and result["new_backend_calls"] == 1
        assert result["historical_baseline_reusable"] is False
        probes.append(dict(model=model, source=relative(path), sha256=sha(path), backend_calls=1,
            wall_seconds=result["wall_seconds"], peak_rss_gib=result["peak_rss_gib"],
            replay=result["historical_A256_replay"]["exact_output_match"],
            fresh_and_old_actual_order_same=result["actual_order_comparison"]["same"]))
    diagnostic = read(destination / "evidence/diagnosis_audit.json")
    checks = read(destination / "evidence/diagnosis_checks.json")
    assert diagnostic["status"] == "complete" and diagnostic["deterministic_CPCM_calls"] == 8
    assert diagnostic["new_searches"] == diagnostic["backend_calls"] == 0 and checks["status"] == "passed"
    smoke_records = []
    for folder in (destination/"smoke", destination/"out/gate_smoke"):
        for path in sorted(folder.rglob("results/*.json")):
            result = read(path)
            smoke_records.append(dict(source=relative(path), sha256=sha(path),
                job_id=result["job_id"], cpcm_calls=result["cpcm_calls"], backend_calls=result["backend_calls"],
                validation_calls=result.get("cpcm_validation_calls", 0), scope="local development smoke, not a production record"))
    gate_attempts = read(destination/"evidence/gate_local_attempts.json")
    assert gate_attempts["total_CPCM_calls"] == 5 and gate_attempts["total_backend_calls"] == 0
    assert gate_attempts["accepted_final_smoke_CPCM_calls"] == 2 and gate_attempts["earlier_development_CPCM_calls"] == 3
    return dict(legacy_probes=probes, legacy_probe_backend_calls=sum(r["backend_calls"] for r in probes),
        accepted_local_diagnostic_CPCM_calls=8, local_smokes=smoke_records,
        local_smoke_CPCM_calls=sum(r["cpcm_calls"] for r in smoke_records),
        gate_development_CPCM_calls=3, gate_all_local_CPCM_calls=5,
        gate_local_attempts_source="e2e/dstu_supplement/evidence/gate_local_attempts.json",
        local_diagnostic_development_replays=dict(CPCM_calls=12, backend_calls=0, searches=0,
            source="assistant execution transcript before final diagnosis outputs",
            attempts=[dict(CPCM_calls=4, reason="route comparison encountered a new/recycled communication tag; corrected to separate identity changes"),
                      dict(CPCM_calls=8, reason="search-summary reader encountered reused budget record; corrected to skip status=reused")]))


def audit(out=OUT, destination=BASE):
    registry = read(destination/"manifests/registry.json")
    assert registry["accepting"] is False
    jobs = []
    for path in registry["manifests"]:
        jobs += read(path)["jobs"]
    assert len(jobs) == len({j["id"] for j in jobs}) == 564
    assert Counter(j["kind"] for j in jobs) == EXPECTED
    assert {dep for job in jobs for dep in job.get("deps", [])} <= {j["id"] for j in jobs}
    controller = controller_records(out, jobs)
    results, records = {}, []
    for job in jobs:
        result, record = inspect_job(job, out)
        results[job["id"]] = result; records.append(record)
    prepared, preparation_calls = audit_prepare(results)
    gates, decisions, frozen = audit_gate(results)
    fresh = audit_fresh_identities(prepared, destination)
    identity_delivery = audit_identity_delivery(fresh, destination)
    evaluations, chains, counts, scope_counts, roots, first_eval = audit_evaluations(results, prepared, fresh, gates, frozen, out)
    extras = extra_runs(destination)
    report = read(destination/"evidence/report_audit.json")
    assert report["status"] == "complete" and report["complete_groups"] == 108 and report["joint_records"] == 324
    assert report["gate_decisions"] == 36 and report["online_result_groups"] == 108
    version = read(destination/"evidence/diagnosis_runner_review.json")
    assert version["status"] == "passed" and all(all(row["checks"].values()) for row in version["model_label_equivalence"])
    repair_bundle = destination/"dist/s256_reuse_inputs.tar.gz"
    repair_metadata = read(str(repair_bundle) + ".json")
    assert repair_metadata["files"] == 30 and sha(repair_bundle) == repair_metadata["sha256"]
    assert repair_metadata["sha256"] == "f12a0cf905727996b88ca737b446aaa925ce5f857c222ecf5e47268397f2cae3"
    failures = [dict(event="CSV preparation-summary helper import failed on server", classification="entry_dispatch_failed_before_backend",
        effect="summary helper dependency was uploaded; no experimental job was started by the failed entry",
        evidence_source="parent conversation tool output; no standalone log was saved", counts_as_experiment_job_failure=False)]
    manifest = dict(status="passed", recorded_utc=datetime.now(timezone.utc).isoformat(),
        source_files=dict(sorted(INPUTS.items())), job_records=records, evaluation_records=evaluations,
        reuse_chains=chains, actual_backend_sources=roots,
        identity_fields=["saved input hashes", "model contract", "platform", "semantic graph", "actual input iteration",
            "rank", "release", "seed", "joint-derived chunk mask", "common reference", "transform source"],
        prepare_workload_label_note="Historical prepare spec W1 is a global label; four H/I label-equivalence checks prove unchanged W1_8Q_1KV_v1 generated code and model contract.",
        fresh_identity_records=[dict(point=r["point"], source=relative(destination/"evidence/fresh_identity_checks"/(r["point"]+".json")),
            original_prepare_identity_comparison=r["original_prepare_identity_comparison"]) for r in fresh.values()],
        identity_delivery=identity_delivery,
        excluded_prepare_fields="v1 prepare.identities: reused mutable event objects between transformations. Production evaluate/simulate both reloaded raw graph independently; formal identity comes from fresh_identity_checks.")
    manifest_path = destination/"evidence/input_and_reuse_manifest.json"
    write(manifest_path, manifest)
    result = dict(status="complete", completed_jobs=564, expected_jobs=564, job_counts=EXPECTED,
        complete_comparison_groups=108, joint_logical_records=324, solo_logical_records=216,
        S256_joint=108, S256_solo=72, HI_refreshed_baseline_joint=72, HI_refreshed_baseline_solo=48,
        evaluation_counts=counts, scope_evaluation_counts=scope_counts,
        actual_production_backend_calls=counts["actual_backend_calls"],
        total_new_backend_calls_including_legacy_probes=counts["actual_backend_calls"]+extras["legacy_probe_backend_calls"],
        production_CPCM=dict(prepare_rank_checks=preparation_calls, gate_predictions=len(gates), gate_validation=0,
                             total=preparation_calls+len(gates)), new_searches=0,
        fresh_identity_checks=dict(points=len(fresh), backend_calls=0, CPCM_calls=0,
            original_preparation_identity_rows=108,
            original_identity_rows_matching=sum(row["old_repeated_transform_identity_matches"] for r in fresh.values() for row in r["original_prepare_identity_comparison"]),
            all_540_execution_records_match_fresh_identities=True,
            first_S256_mask_stats_preserved_points=12,
            runtime_seconds=sum(r["wall_seconds"] for r in fresh.values()),
            independent_batch=identity_delivery,
            correction="Separate raw-stream reload before each transport; old prepare identities excluded without modifying old results or rerunning backends"),
        gate=dict(predictions=12, decisions=36, online_result_groups=108, action_counts=dict(Counter(r["chosen_transport"] for r in decisions)),
                  latest_freeze_utc=frozen.isoformat(), earliest_new_job_start_utc=first_eval.isoformat(),
                  freeze_leads_first_new_job_seconds=(first_eval-frozen).total_seconds(),
                  ordering_evidence="all 12 choices frozen before every evaluation job starts; backend invocation occurs later inside that job"),
        controller=controller, extra_runs=extras, helper_incidents=failures,
        input_files=len(manifest["source_files"]), input_and_reuse_manifest=relative(manifest_path),
        input_and_reuse_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        receipt_process_hours=sum(r["receipt_seconds"] for r in records)/3600,
        task_process_hours=sum(r["task_seconds"] for r in records)/3600,
        source_input_repair=dict(files=30, sha256=repair_metadata["sha256"], archive=relative(repair_bundle)),
        report_audit_sha256=sha(destination/"evidence/report_audit.json"),
        audit_backend_calls=0, audit_CPCM_calls=0, audit_searches=0,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), generated_utc=datetime.now(timezone.utc).isoformat())
    write(destination/"evidence/completion_audit.json", result)
    lines = ["# S256 补充最终核对", "",
        f"12项准备、12项门控预测、540项评估记录全部完成，共564项；另有两次历史A256回放。108个比较组均保留original、A256、S256三条联合结果，324条联合和216条逻辑单跑逐项配对。", "",
        f"正式评估实际调用后端{counts['actual_backend_calls']}次，另两次历史回放，总计{result['total_new_backend_calls_including_legacy_probes']}次。历史复用{counts['reuse_historical']}条，本批复用{counts['reuse_within_batch']}条；每条复用沿来源链核对到实际运行结果，未把重复引用计成新后端。", "",
        f"正式CPCM为24次S256排名检查和12次窗口预测。另计本机8个正式保存态诊断、{extras['local_smoke_CPCM_calls']}次最终保存冒烟调用，以及诊断脚本开发期间12次、门控开发期间3次重放；没有新增SA。", "",
        f"36个前验决定在第一项新评估开始前{(first_eval-frozen).total_seconds():.3f}秒全部冻结，三种后端种子共用相同动作；108条Online结果均取所选真实策略的完整P/D。预测耗时与服务时间分别报告。", "",
        f"池经历{len(controller['segments'])}个控制器进程，首次状态反推启动至最终完成共{controller['full_elapsed_seconds']/60:.2f}分钟，包含输入缺失、补传与重启等待。最终控制器段为{controller['final_controller_segment_seconds']/60:.2f}分钟。最大活跃{controller['max_active']}，最低可用内存{controller['min_available_gib']:.3f}GiB。", "",
        f"失败尝试保留{controller['failed_attempts']}份，其中内存事件{controller['OOM_attempts']}份；所有对应任务现均成功。初次CSV汇总辅助入口缺模块的错误另外记录，不计作实验运行失败。30份缺少的旧S256结果已补传并按归档摘要核对。", "",
        "最终核对发现旧prepare连续变换复用了会被chunk_by原地修改的事件对象，因而其中后续策略的identities字段错误。这些字段已从正式认证中排除；新增12项只读核对，每种策略都重新读取原始流后变换，540条实际执行身份全部匹配。正式evaluate在算身份后、simulate中分别重载原图，实际后端结果与复用判断未受影响。12点最初S256的掩码、统计与新鲜变换相同，原CPCM预测与排名检查保留。本次纠正新增CPCM和后端均为0。", "",
        f"主实验完成时间为北京时间{instant(controller['completion_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}；后续12项身份重建于{instant(identity_delivery['completed_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%H:%M:%S')}完成，单独耗时{identity_delivery['elapsed_seconds']:.3f}秒。两段时间分别记录。", "",
        "主实验实际运行v1源码保存在[server_runtime](dist/server_runtime/e2e/dstu_supplement_run.py)。以后每种变换重新读取原始流的修正为v2，源码保存在[server_runtime_identity](dist/server_runtime_identity/e2e/dstu_supplement_run.py)；身份重建入口为[dstu_supplement_identity.py](../dstu_supplement_identity.py)。两版本均保留并核对摘要，原564条结果与v1准备记录没有改写。旧prepare.identities的108行均不作为正式身份依据，逐条复核以新的12份fresh_identity_checks为准。", "",
        "检查包括manifest、落盘spec、回执身份、成功日志、输入文件摘要、图与执行次序、共同参考、切块掩码、配对单跑、门控冻结时间和完整报告覆盖。旧prepare的H/I负载标签差异经生成代码/配置等价核对，不影响实际模型图。", "",
        "[完整审计](evidence/completion_audit.json) · [输入、回执及复用链](evidence/input_and_reuse_manifest.json)", ""]
    (destination/"AUDIT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--destination", type=Path, default=BASE)
    args = parser.parse_args()
    value = audit(args.out, args.destination)
    print(json.dumps({k:value[k] for k in ("status", "completed_jobs", "complete_comparison_groups",
        "actual_production_backend_calls", "total_new_backend_calls_including_legacy_probes", "production_CPCM", "input_files")}, ensure_ascii=False, indent=2))
