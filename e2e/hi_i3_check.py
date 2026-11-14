"""Independent read-only checks for I3 beside the running H3/I2 batch."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import statistics

from hi_next_check import read, sha, job_identity, KEYS, SEEDS
from hi_h3_i2_check import write, completed_log, old_files

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "e2e/hi"


def plan_check():
    frozen = read(HERE / "manifest/all_candidates_planned.json")["jobs"]
    full = {s["candidate_id"]: s for s in frozen}
    chosen = {n: s for n, s in full.items() if s["batch"] == "I3"}
    previous = {n: s for n, s in full.items() if s["batch"] != "I3"}
    jobs = read(HERE / "manifest/i3_108.json")["jobs"]
    by_id = {j["id"]: j for j in jobs}
    assert len(full) == 896 and len(previous) == 788
    assert len(jobs) == len(by_id) == len(chosen) == 108 and set(by_id) == set(chosen)
    assert not set(chosen) & set(previous)
    assert Counter(s["method"] for s in chosen.values()) == {"A1": 56, "A3h": 52}
    groups = defaultdict(list)
    for job in jobs:
        spec = chosen[job["id"]]
        assert job["spec"] == spec and job["kind"] == "search"
        assert spec["model"] == "qwen3moe" and spec["hardware_profile"] == "watos"
        assert spec["op"] in ("mha", "ffn") and spec["search_steps"] == 1000
        assert spec["backend_seeds"] == spec["random_cpcm_seeds"] == SEEDS
        assert job["result"] == f"results/{job['id']}.json" and job["deps"] == []
        assert job["argv"] == ["{python}", "-u", "-B", "e2e/hi_run.py", "child", "--spec", "{spec}", "--out", "{out}"]
        assert job["estimated_seconds"] > 0 and job["peak_rss_gb"] > 0
        groups[tuple(spec[k] for k in KEYS)].append(spec)
    assert len(groups) == 8
    first = set()
    for key, specs in groups.items():
        if key[-1] == "A1":
            assert len(specs) == 14
            first.add(min(specs, key=lambda s: s["candidate_order"])["candidate_id"])
        else:
            assert len(specs) == 13
            existing = [s for s in previous.values() if tuple(s[k] for k in KEYS) == key]
            assert len(existing) == 1 and existing[0]["batch"] == "I1W"
    release = read(HERE / "manifest/i3_release_plan.json")
    assert release["all"] and set(release["initial_release_ids"]) == set(chosen)
    assert release["dependent_jobs"] == 0
    assert {r["prerequisite_candidate"] for r in release["calibrations"]} == first
    assert all(r["release_after_success"] == [] for r in release["calibrations"])
    assert {j["id"] for j in jobs if j["priority"] == 2} == first
    assert all(j["priority"] > 0 for j in jobs if j["spec"]["phase"] == "prefill" and j["spec"]["op"] == "ffn")
    contract = read(HERE / "models/qwen3moe/model_contract.json")
    core = {k: v for k, v in contract.items() if k not in ("generated_sources", "validation")}
    digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    assert core["head_groups"] == 4 and all(s["model_contract_sha256"] == digest for s in chosen.values())
    all_groups = Counter(tuple(s[k] for k in KEYS) for s in full.values())
    assert len(all_groups) == 64 and set(all_groups.values()) == {14}
    assert Counter(s["model"] for s in full.values()) == {"llama": 448, "qwen3moe": 448}

    # Import only; do not call launch, run, plan, or prepare any model.
    import hi_i3
    wrapped = hi_i3.wrapped_source()
    compile(wrapped, "<audited I3 pool>", "exec")
    source = (ROOT / "e2e/hi_i3.py").read_text(encoding="utf-8")
    assert "pool.subprocess=types.SimpleNamespace(Popen=self.spawn,STDOUT=subprocess.STDOUT)" in source
    assert 'namespace["execute"](self.manifest,self.out,self.workers,3,NAME,4,self.release)' in source
    assert "external_slots = _hi.reserve_old()" in wrapped
    assert "len(running) + external_slots >= workers" in wrapped
    assert "budget = free - reserve_gib - reserved_growth - _hi.old_growth_reserve_gib" in wrapped
    controller = hi_i3.Controller(HERE / "out/i3_108", 128)
    old_remaining = controller.reserve_old()  # Reads existing receipts/status only.
    assert old_remaining == 440 - len(controller.old_success)
    assert controller.old_out == HERE / "out/h3_i2_440" and controller.out != controller.old_out
    assert len(controller.jobs) == 108 and set(controller.by_id) == set(chosen)
    # Since an old unsuccessful ID owns at most one live attempt, reserving every
    # unfinished ID covers its running or retried attempt, even between status polls.
    for completed in range(441):
        reserved = 440 - completed
        capacity = max(0, 128 - reserved)
        if reserved <= 128:
            assert capacity + reserved == 128
        else:
            assert capacity == 0
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (
        ROOT / "e2e/hi_i3.py", ROOT / "e2e/hi_next.py", ROOT / "e2e/fg_batch.py",
        ROOT / "e2e/hi_run.py", ROOT / "e2e/hi_config.py", ROOT / "e2e/post_fg_bb.py",
        ROOT / "cpcm_sa/cpcm_sa.py", ROOT / "e2e/e_gemini.py")}
    result = dict(status="passed", created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Read-only frozen-spec, resource-sharing and runner review; no launch or simulation.",
        new_candidates=108, batches={"I3":108}, methods={"A1":56,"A3h":52},
        prior_planned_candidates=788, old_new_overlap=0, new_and_concurrent_candidates=548,
        healthy_candidates_after_completion=896, complete_groups_after_completion=64,
        groups=8, previous_I1_reused_candidates=4, initial_release=108, dependency_count=0,
        first_priority_candidates=sorted(first), main_backend_samples=1080, random_cpcm_samples=1080,
        terminal_states=0, model_contract_sha256=digest,
        combined_slot_rule="I3 active + (440 - identity-matched successful H3/I2 receipts) <= 128",
        combined_slot_reason="Every unfinished old candidate retains one slot, including an OOM retry; successful slots are released monotonically.",
        memory_rule="Current system available - 3 GiB - I3 future growth reserve - old pool growth reserve; old growth reserve is read from its status snapshot.",
        resource_observation=dict(source="local downloaded old results and status, not live SSH", old_successful_receipts=len(controller.old_success),
            old_reserved_slots=old_remaining, allowed_new_active=max(0,128-old_remaining),
            old_growth_reserve_gib=controller.old_growth_reserve_gib),
        old_controller_unchanged_by_new_entrypoint=True, stdout_binding_present=True,
        source_sha256=sources, manifest_sha256=sha(HERE / "manifest/i3_108.json"),
        candidate_identities={n:job_identity(s) for n,s in chosen.items()})
    write(HERE / "evidence/i3_plan_review.json", result)
    return result


def completion_check(out):
    plan = read(HERE / "evidence/i3_plan_review.json")
    jobs = read(HERE / "manifest/i3_108.json")["jobs"]
    chosen = {j["id"]: j["spec"] for j in jobs}
    assert {p.stem for p in (out / "results").glob("*.json")} == set(chosen)
    assert {p.stem for p in (out / "receipts").glob("*.json")} == set(chosen)
    sources = {Path(p).name:digest for p,digest in plan["source_sha256"].items()}
    launch = read(out / "i3.start.json")
    assert launch["command"][launch["command"].index("--workers") + 1] == "128"
    for filename,digest in launch["source_sha256"].items():
        assert sources[filename] == digest
    rows, failed_attempts = [], []
    for name,spec in sorted(chosen.items()):
        record = read(out / "results" / (name + ".json"))
        receipt = read(out / "receipts" / (name + ".json"))
        assert record["kind"] == "search" and record["status"] == "ok"
        assert record["graph_order_version"] == "hi_native_order_v1" and record["workload_version"] == "W1_8Q_1KV_v1"
        assert all(record[k] == v for k,v in spec.items())
        assert read(out / "specs" / (name + ".json")) == spec
        assert receipt["rc"] == 0 and receipt["identity"] == plan["candidate_identities"][name]
        assert receipt["result"] == f"results/{name}.json"
        with gzip.open(out / "raw" / (name + ".pkl.gz"), "rb") as stream:
            raw = pickle.load(stream)
        assert raw["meta"]["steps"] == 1000 and len(raw["rows"]) == 1000
        assert [r["step"] for r in raw["rows"]] == list(range(1,1001))
        assert raw["meta"]["sa_seed"] == (123 if spec["method"] == "A1" else spec["x"])
        assert record["exact_rollback"] and not record["debug_checks"]
        assert not raw["check2"] and not raw["check4"] and raw["check_revert"]["n"] == 0
        assert record["instrumentation"]["original_algorithm_ast_identical_after_removing_observations"]
        assert all(r["accepted"] or "t_revert" in r for r in raw["rows"])
        selected = raw["selected"]
        assert [r["seed"] for r in record["backend"]] == SEEDS and list(selected["backend"]) == SEEDS
        for row in record["backend"]:
            assert row["triple"] == list(selected["backend"][row["seed"]]) and row["cycles"] == row["triple"][0]
        assert len(record["T_hat_rand"]) == 10 and record["T_hat_rand"] == selected["T_hat_rand"]
        assert record["T_hat_rand_mean"] == statistics.mean(record["T_hat_rand"])
        assert record["T_hat"] == selected["T_bc"] and record["T_hat_pre"] == selected["T_pre"]
        assert record["workload_signature"] == selected["workload_signature"]
        assert [r["seed"] for r in record["backend_input_orders"]] == SEEDS
        assert record["model_contract_sha256"] == plan["model_contract_sha256"]
        for filename,digest in record["source_sha256"].items():
            assert sources[filename] == digest
        with gzip.open(out / "snapshots" / (name + ".pkl.gz"), "rb") as stream:
            snapshot = pickle.load(stream)
        assert len(snapshot) == 2 and isinstance(snapshot[0],bytes) and snapshot[0]
        assert set(snapshot[1]) == {"link_loads_dict","link_loads_count"}
        log, failures = completed_log(out,name,record)
        failed_attempts.extend(failures)
        rows.append(dict(candidate_id=name,receipt_identity=receipt["identity"],complete_1000_steps=True,
            backend_samples=10,random_cpcm_samples=10,completed_log=str(log.relative_to(ROOT)),
            result_sha256=sha(out / "results" / (name + ".json"))))
    assert set(read(out / "release.json")["ids"]) == set(chosen)
    finish = read(out / "controller_finish.json")
    assert finish["rc"] == 0 and finish["completed"] == finish["expected"] == 108
    batch = read(out / "i3.batch.json")
    assert batch["completed"] == 108 and not batch["failed"] and not batch["pending"]
    assert batch["workers"] <= 128 and batch["max_active"] <= 128
    all_specs = {s["candidate_id"]:s for s in read(HERE / "manifest/all_candidates_planned.json")["jobs"]}
    all_records = {}
    for folder in ("h1_main","next_344","h3_i2_440","i3_108"):
        for path in (HERE / "out" / folder / "results").glob("*.json"):
            record = read(path)
            if record["kind"] == "search":
                assert record["candidate_id"] not in all_records
                all_records[record["candidate_id"]] = record
    assert set(all_records) == set(all_specs) and len(all_records) == 896
    for name,record in all_records.items():
        assert all(record[k] == v for k,v in all_specs[name].items())
    groups = Counter(tuple(r[k] for k in KEYS) for r in all_records.values())
    assert len(groups) == 64 and set(groups.values()) == {14}
    resources = combined_resources(out)
    result = dict(status="passed",scope="Saved I3 outputs and 896 healthy candidate identities; no simulation.",
        new_candidates=108,all_healthy_candidates=896,complete_14_candidate_groups=64,
        new_backend_samples=1080,new_random_cpcm_samples=1080,new_terminal_states=0,
        final_failed_candidates=0,recorded_memory_retry_attempts=len(failed_attempts),
        completed_formal_logs=108,preserved_first348=old_files(),controller_completion=finish,
        candidates=rows,failed_attempts=failed_attempts,combined_resources=resources)
    write(HERE / "evidence/i3_review.json",result)
    return result


def combined_resources(out):
    """Audit both successful and retried attempts without treating slots as CPU use."""
    statuses = []
    for line in (out / "i3.log").read_text(encoding="utf-8").splitlines():
        if line.startswith('{"event": "status"'):
            row = json.loads(line)
            assert row["jobs"] == 108
            assert row["old_successful_receipts"] + row["old_reserved_slots"] == 440
            assert row["active"] + row["old_reserved_slots"] == row["combined_reserved_and_active"]
            assert row["combined_reserved_and_active"] <= 128
            assert row["allowed_new_active"] == 128 - row["old_reserved_slots"]
            statuses.append(row)
    assert statuses
    changes, attempts = [], []
    for folder, expected in (("h3_i2_440",440),("i3_108",108)):
        batch_out = HERE / "out" / folder
        batch_name = "h3_i2" if expected == 440 else "i3"
        batch = read(batch_out / (batch_name + ".batch.json"))
        assert batch["completed"] == expected and not batch["failed"] and not batch["pending"]
        jobs = read(HERE / "manifest" / (folder + ".json"))["jobs"]
        for job in jobs:
            name = job["id"]
            record = read(batch_out / job["result"])
            receipt = read(batch_out / "receipts" / (name + ".json"))
            log, failures = completed_log(batch_out,name,record)
            entries = [(log,receipt,"success")]
            entries += [(batch_out / "logs" / Path(r["log"]).name,r,"memory_retry") for r in failures]
            for path, r, state in entries:
                start = int(path.stem.rsplit(".",1)[1]) / 1e9
                end = start + r["wall_seconds"]
                changes.extend(((start,1),(end,-1)))
                attempts.append(dict(candidate_id=name,state=state,start_unix=start,end_unix=end))
    active, maximum = 0,0
    for stamp,delta in sorted(changes,key=lambda row:(row[0],row[1])):
        active += delta
        maximum = max(maximum,active)
    assert active == 0 and maximum <= 128
    minimum_free = min(read(HERE / "out" / folder / (name + ".batch.json"))["min_available_gib"]
        for folder,name in (("h3_i2_440","h3_i2"),("i3_108","i3")))
    return dict(status="passed",new_candidates=548,successful_attempts=548,
        recorded_attempts=len(attempts),i3_status_snapshots=len(statuses),
        max_recorded_reserved_plus_active=max(r["combined_reserved_and_active"] for r in statuses),
        approximate_peak_active_from_both_job_logs=maximum,minimum_sampled_available_gib=minimum_free,
        required_reserve_gib=3,
        interval_definition="Log filename Unix_ns is sampled just before Popen; receipt.wall_seconds starts just after Popen. Their sum approximates the receipt time, with launch bookkeeping error. These intervals include both successful and archived failed attempts.",
        slot_definition="Reserved-plus-active is an upper bound on occupied worker slots, not a CPU utilization measurement.",
        memory_definition="Available memory is sampled by the two pools; the 3 GiB value is the launch-budget reserve, not an assertion about unsampled instants.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","complete"))
    parser.add_argument("--out",type=Path,default=HERE / "out/i3_108")
    args = parser.parse_args()
    result = plan_check() if args.command == "plan" else completion_check(args.out)
    print(json.dumps({k:result[k] for k in ("status","new_candidates","initial_release","all_healthy_candidates") if k in result}))
