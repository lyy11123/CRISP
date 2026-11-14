"""Read-only authorization and completion checks for H3 + I2 (440 candidates)."""
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


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "e2e/hi"
BATCHES = {"H3": 108, "I2D": 166, "I2P": 166}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def planned():
    all_specs = read(HERE / "manifest/all_candidates_planned.json")["jobs"]
    chosen = {s["candidate_id"]: s for s in all_specs if s["batch"] in BATCHES}
    old = {}
    for folder in ("h1_main", "next_344"):
        for path in (HERE / "out" / folder / "results").glob("*.json"):
            record = read(path)
            if record["kind"] == "search":
                old[record["candidate_id"]] = record
    assert len(old) == 348 and len(chosen) == 440 and not set(chosen) & set(old)
    assert dict(Counter(s["batch"] for s in chosen.values())) == BATCHES
    assert Counter(s["batch"] for s in all_specs if s["candidate_id"] not in set(chosen) | set(old)) == {"I3": 108}
    return all_specs, chosen, old


def old_files():
    previous = read(HERE / "evidence/HIJ_NEXT_344_20261114_complete.tar.verified.json")["inventory"]["files"]
    prefixes = ("e2e/hi/out/h1_main/", "e2e/hi/out/next_344/")
    rows = [r for r in previous if r["path"].startswith(prefixes)]
    assert rows
    for row in rows:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], row["path"]
    return dict(status="unchanged", files=len(rows), baseline="HIJ_NEXT_344_20261114_complete.tar.verified.json")


def plan_check():
    _, chosen, old = planned()
    path = HERE / "manifest/h3_i2_440.json"
    jobs = read(path)["jobs"]
    job_by_id = {j["id"]: j for j in jobs}
    assert len(jobs) == 440 and set(job_by_id) == set(chosen)
    for job in jobs:
        spec = chosen[job["id"]]
        assert job["spec"] == spec and job["kind"] == "search"
        assert job["result"] == f"results/{job['id']}.json"
        assert job["argv"] == ["{python}", "-u", "-B", "e2e/hi_run.py", "child", "--spec", "{spec}", "--out", "{out}"]
        assert spec["search_steps"] == 1000 and spec["backend_seeds"] == SEEDS and spec["random_cpcm_seeds"] == SEEDS
        assert job["estimated_seconds"] > 0 and job["peak_rss_gb"] > 0
    measured = {tuple(r[k] for k in KEYS) for r in old.values()}
    groups = defaultdict(list)
    for spec in chosen.values():
        groups[tuple(spec[k] for k in KEYS)].append(spec)
    assert len(groups) == 32 and len(set(groups) & measured) == 8
    calibrations = {}
    for key, specs in groups.items():
        if key in measured:
            assert len(specs) == 13 and key[-1] == "A3h"
        else:
            assert len(specs) == 14
            first = min(specs, key=lambda s: s["candidate_order"])["candidate_id"]
            following = {s["candidate_id"] for s in specs} - {first}
            calibrations[first] = following
    initial = set(chosen)
    assert len(initial) == 440 and len(calibrations) == 24
    for job in jobs:
        assert job["deps"] == [], job["id"]
    release = read(HERE / "manifest/h3_i2_release_plan.json")
    assert set(release["initial_release_ids"]) == initial
    assert {r["prerequisite_candidate"] for r in release["calibrations"]} == set(calibrations)
    assert min(job_by_id[n]["priority"] for n in calibrations) > max(j["priority"] for j in jobs if j["id"] not in calibrations)
    assert all(j["priority"] > 0 for j in jobs if j["spec"]["model"] == "qwen3moe"
               and j["spec"]["phase"] == "prefill" and j["spec"]["op"] == "ffn")
    contracts = {}
    for model in ("llama", "qwen3moe"):
        full = read(HERE / "models" / model / "model_contract.json")
        core = {k: v for k, v in full.items() if k not in ("generated_sources", "validation")}
        digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
        assert all(s["model_contract_sha256"] == digest for s in chosen.values() if s["model"] == model)
        assert core["head_groups"] == (8 if model == "llama" else 4)
        contracts[model] = digest
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "e2e/hi_run.py", ROOT / "e2e/hi_config.py",
        ROOT / "e2e/post_fg_bb.py", ROOT / "cpcm_sa/cpcm_sa.py", ROOT / "e2e/e_gemini.py")}
    result = dict(status="passed", scope="Authorization, unchanged specs, all-jobs release and first-candidate priorities; no experiment execution.",
        created_utc=datetime.now(timezone.utc).isoformat(), candidates=440, batches=BATCHES,
        methods=dict(Counter(s["method"] for s in chosen.values())), reused_prior_candidates=348,
        excluded_I3=108, old_new_overlap=0, classes=32, measured_classes=8, new_classes=24,
        initial_release=440, dependency_count=0, main_backend_samples=4400, random_cpcm_samples=4400,
        terminal_states=8, terminal_backend_samples=80, model_contract_hashes=contracts,
        source_sha256=sources, manifest_sha256=sha(path),
        scheduling_policy="All 440 are released immediately; 24 first candidates have highest priority, followed by long Qwen prefill FFN jobs. No candidate waits for a same-class first result.",
        first_priority_candidates=sorted(calibrations), initial_release_ids=sorted(initial),
        candidate_identities={name: job_identity(spec) for name, spec in chosen.items()})
    write(HERE / "evidence/h3_i2_plan_review.json", result)
    return result


def completed_log(out, name, record):
    success, failed = [], []
    for path in sorted((out / "logs").glob(name + ".*.log")):
        lines = path.read_text(encoding="utf-8").splitlines()
        final = json.loads(lines[-1]) if lines and lines[-1].startswith('{"candidate_id":') and '"status": "ok"' in lines[-1] else None
        if final is not None and final.get("status") == "ok":
            assert final["candidate_id"] == name and final["seconds"] == record["wall_seconds"]
            success.append(path)
        else:
            failed.append(path)
    assert len(success) == 1, (name, "successful logs", len(success))
    failed_receipts = [read(p) for p in (out / "failed_attempts" / name).glob("*/receipt.json")]
    for path in failed:
        matching = [r for r in failed_receipts if Path(r["log"]).name == path.name]
        assert len(matching) == 1 and matching[0]["memory_related"], path
    assert len(failed) == len(failed_receipts)
    return success[0], failed_receipts


def partial_check(out):
    """Check saved early results and reload completed first-class snapshots only."""
    plan = read(HERE / "evidence/h3_i2_plan_review.json")
    _, chosen, old = planned()
    records = {p.stem: read(p) for p in (out / "results").glob("*.json") if not p.stem.endswith("_terminal")}
    assert records and len(records) < 440 and set(records) <= set(chosen)
    first_ids = set(plan["first_priority_candidates"])
    completed_first = first_ids & set(records)
    checked, snapshot_reviews, terminal_names = [], [], []
    source_hashes = {Path(p).name: digest for p, digest in plan["source_sha256"].items()}
    for relative, digest in plan["source_sha256"].items():
        assert sha(ROOT / relative) == digest
    import post_fg_split
    import e_run
    post_fg_split.setup()  # Register pickle classes; no platform construction.
    for name, record in sorted(records.items()):
        spec = chosen[name]
        assert record["status"] == "ok" and all(record[k] == v for k, v in spec.items())
        assert record["model_contract_sha256"] == plan["model_contract_hashes"][spec["model"]]
        assert record["graph_order_version"] == "hi_native_order_v1" and record["workload_version"] == "W1_8Q_1KV_v1"
        assert read(out / "specs" / (name + ".json")) == spec
        receipt = read(out / "receipts" / (name + ".json"))
        assert receipt["rc"] == 0 and receipt["identity"] == plan["candidate_identities"][name]
        assert record["exact_rollback"] and not record["debug_checks"]
        for filename, digest in record["source_sha256"].items():
            assert source_hashes[filename] == digest
        raw_path = out / "raw" / (name + ".pkl.gz")
        with gzip.open(raw_path, "rb") as stream:
            raw = pickle.load(stream)
        selected = raw["selected"]
        assert raw["meta"]["steps"] == 1000
        if spec["method"] == "Gemini":
            native = raw["gemini"]
            assert native["steps"] == 1000 and native["anneal_seed"] == spec["x"]
            assert native["LP_flag"] and native["hops_only"] and not native["dijkstra_routing"]
            assert native["rollback"] == "pre_move_snapshot"
            assert record["own_cost"] == native["best_energy"]
        else:
            assert len(raw["rows"]) == 1000 and [r["step"] for r in raw["rows"]] == list(range(1, 1001))
            assert not raw["check2"] and not raw["check4"] and raw["check_revert"]["n"] == 0
            assert record["instrumentation"]["original_algorithm_ast_identical_after_removing_observations"]
            assert record["own_cost"] == selected["own_cost"]
        assert [r["seed"] for r in record["backend"]] == list(selected["backend"]) == SEEDS
        assert len(record["T_hat_rand"]) == 10 and record["T_hat_rand"] == selected["T_hat_rand"]
        assert record["T_hat_rand_mean"] == statistics.mean(record["T_hat_rand"])
        assert record["T_hat"] == selected["T_bc"] and record["T_hat_pre"] == selected["T_pre"]
        assert record["backend_input_orders"] == selected["backend_input_orders"]
        assert [r["seed"] for r in record["backend_input_orders"]] == SEEDS
        assert record["workload_signature"] == selected["workload_signature"]
        for row in record["backend"]:
            assert row["triple"] == list(selected["backend"][row["seed"]]) and row["cycles"] == row["triple"][0]
        log, failures = completed_log(out, name, record)
        assert not failures
        if name in completed_first:
            path = out / "snapshots" / (name + ".pkl.gz")
            with gzip.open(path, "rb") as stream:
                blob, history = pickle.load(stream)
            assert set(history) == {"link_loads_dict", "link_loads_count"}
            behaviours, tensors, events = pickle.loads(blob)
            assert e_run.signature((behaviours, tensors, events, None)) == record["workload_signature"]
            method = "Gemini" if spec["method"] == "Gemini" else "A3h"
            h0 = read(HERE / "out/model_checks/results" /
                      f"check_{spec['model']}_{spec['hardware_profile']}_{spec['phase']}_{spec['op']}_{method}.json")
            semantic_rows = [(tag, b.beha_type, b.beha_datashape, sorted(b.needed_data_split_dict), sorted(b.produced_data_split_dict))
                             for tag, b in behaviours.items()]
            semantic_hash = hashlib.sha256(json.dumps(semantic_rows, sort_keys=True).encode()).hexdigest()
            assert semantic_hash == h0["semantic"]["semantic_sha256"]
            assert len(behaviours) == h0["semantic"]["behaviours"] and len(tensors) == h0["semantic"]["tensors"]
            assert h0["model_contract_sha256"] == record["model_contract_sha256"]
            assert Counter(e.comp_type for e in events.values() if e.event_type == "computation") == record["graph"]["computations"]
            snapshot_reviews.append(dict(candidate_id=name, snapshot_sha256=sha(path),
                reloaded_workload_signature=record["workload_signature"], semantic_sha256=semantic_hash,
                unchanged_H0_logical_graph=True, behaviours=len(behaviours), tensors=len(tensors),
                model_contract_sha256=record["model_contract_sha256"]))
            del behaviours, tensors, events
        if spec["method"] == "Gemini" and spec["x"] == 123:
            terminal_path = out / "results" / (name + "_terminal.json")
            terminal = read(terminal_path)
            assert terminal["kind"] == "terminal" and terminal["status"] == "ok" and terminal["from_search"] == name
            assert terminal["workload_signature"] == record["workload_signature"]
            assert terminal["random_cpcm_evaluations"] == 0 and set(terminal["backend"]) == {str(s) for s in SEEDS}
            assert [r["seed"] for r in terminal["backend_input_orders"]] == SEEDS
            terminal_names.append(name + "_terminal")
        checked.append(dict(candidate_id=name, receipt_identity=receipt["identity"], complete_1000_steps=True,
            backend_samples=10, random_cpcm_samples=10, result_sha256=sha(out / "results" / (name + ".json")),
            raw_sha256=sha(raw_path), source_sha256=record["source_sha256"], successful_log=str(log.relative_to(ROOT))))
    assert {p.stem for p in (out / "results").glob("*_terminal.json")} == set(terminal_names)
    result = dict(status="available_results_passed_batch_in_progress", scope="Read-only early results and first-candidate saved snapshot semantics; no graph construction, SA or backend simulation.",
        created_utc=datetime.now(timezone.utc).isoformat(), available_candidates=len(records), expected_candidates=440,
        reviewed_backend_samples=10*len(records), reviewed_random_cpcm_samples=10*len(records),
        separate_terminal_states=len(terminal_names), separate_terminal_backend_samples=10*len(terminal_names),
        completed_new_class_first_candidates=len(completed_first), planned_new_class_first_candidates=24,
        new_class_first_candidates_not_in_download=sorted(first_ids-set(records)),
        missing_candidate_count=440-len(records), per_batch=dict(Counter(chosen[n]["batch"] for n in records)),
        preserved_prior_candidates=348, old_output_files=old_files(), old_new_overlap=0,
        candidates=checked, representative_snapshot_checks=snapshot_reviews,
        method_note="A1 uses the already-verified BusyBarn/A3h logical template contract; Gemini is compared with its own H0 template contract.")
    write(HERE / "evidence/h3_i2_partial_review.json", result)
    return result


def completion_check(out):
    plan = read(HERE / "evidence/h3_i2_plan_review.json")
    _, chosen, old = planned()
    assert sha(HERE / "manifest/h3_i2_440.json") == plan["manifest_sha256"]
    previous = old_files()
    main_ids = set(chosen)
    terminal_parents = {n for n, s in chosen.items() if s["method"] == "Gemini" and s["x"] == 123}
    assert {p.stem for p in (out / "results").glob("*.json")} == main_ids | {n + "_terminal" for n in terminal_parents}
    assert not any((out / "results" / (n + ".json")).exists() for n in old)
    summaries, samples, starts, ends, failures, terminals = [], {}, {}, {}, [], []
    for name, spec in chosen.items():
        record = read(out / "results" / (name + ".json"))
        assert record["status"] == "ok" and all(record[k] == v for k, v in spec.items())
        assert read(out / "specs" / (name + ".json")) == spec
        receipt = read(out / "receipts" / (name + ".json"))
        assert receipt["rc"] == 0 and receipt["identity"] == plan["candidate_identities"][name]
        assert receipt["result"] == f"results/{name}.json"
        with gzip.open(out / "raw" / (name + ".pkl.gz"), "rb") as stream:
            raw = pickle.load(stream)
        assert raw["meta"]["steps"] == 1000 and record["exact_rollback"] and not record["debug_checks"]
        selected = raw["selected"]
        if spec["method"] == "Gemini":
            native = raw["gemini"]
            assert native["steps"] == 1000 and native["anneal_seed"] == spec["x"]
            assert native["LP_flag"] and native["hops_only"] and not native["dijkstra_routing"]
            assert native["rollback"] == "pre_move_snapshot"
        else:
            assert len(raw["rows"]) == 1000 and [r["step"] for r in raw["rows"]] == list(range(1, 1001))
            assert raw["meta"]["sa_seed"] == (123 if spec["method"] == "A1" else spec["x"])
            assert not raw["check2"] and not raw["check4"] and raw["check_revert"]["n"] == 0
            assert record["instrumentation"]["original_algorithm_ast_identical_after_removing_observations"]
            assert all(r["accepted"] or "t_revert" in r for r in raw["rows"])
        assert [r["seed"] for r in record["backend"]] == SEEDS and list(selected["backend"]) == SEEDS
        for row in record["backend"]:
            assert row["triple"] == list(selected["backend"][row["seed"]]) and row["cycles"] == row["triple"][0]
        assert len(record["T_hat_rand"]) == 10 and record["T_hat_rand"] == selected["T_hat_rand"]
        assert record["T_hat_rand_mean"] == statistics.mean(record["T_hat_rand"])
        assert record["T_hat"] == selected["T_bc"] and record["T_hat_pre"] == selected["T_pre"]
        assert record["workload_signature"] == selected["workload_signature"]
        assert [r["seed"] for r in record["backend_input_orders"]] == SEEDS
        for filename, digest in record["source_sha256"].items():
            canonical = next(v for k, v in plan["source_sha256"].items() if Path(k).name == filename)
            assert digest == canonical, (name, filename)
        paths = {kind: out / folder / (name + suffix) for kind, folder, suffix in (
            ("result", "results", ".json"), ("raw", "raw", ".pkl.gz"), ("snapshot", "snapshots", ".pkl.gz"))}
        with gzip.open(paths["snapshot"], "rb") as stream:
            snapshot = pickle.load(stream)
        assert len(snapshot) == 2 and isinstance(snapshot[0], bytes) and snapshot[0]
        assert set(snapshot[1]) == {"link_loads_dict", "link_loads_count"}
        log, prior_failures = completed_log(out, name, record)
        failures.extend(prior_failures)
        starts[name] = int(log.stem.rsplit(".", 1)[1]) / 1e9
        ends[name] = starts[name] + receipt["wall_seconds"]
        samples[name] = dict(backend=[r["cycles"] for r in record["backend"]], random_cpcm=record["T_hat_rand"])
        summaries.append(dict(candidate_id=name, receipt_identity=receipt["identity"],
            **{kind: sha(path) for kind, path in paths.items()}, complete_1000_steps=True,
            backend_samples=10, random_cpcm_samples=10, completed_log=str(log.relative_to(ROOT))))
        if name in terminal_parents:
            terminal = read(out / "results" / (name + "_terminal.json"))
            assert terminal["kind"] == "terminal" and terminal["status"] == "ok" and terminal["from_search"] == name
            assert set(terminal["backend"]) == {str(s) for s in SEEDS} and terminal["random_cpcm_evaluations"] == 0
            assert terminal["workload_signature"] == record["workload_signature"]
            assert [r["seed"] for r in terminal["backend_input_orders"]] == SEEDS
            assert (out / "snapshots" / (name + "_terminal.pkl.gz")).stat().st_size > 0
            terminals.append(name)
    release = read(out / "release.json")
    assert set(release["ids"]) == main_ids
    assert all(not j["deps"] for j in read(HERE / "manifest/h3_i2_440.json")["jobs"])
    groups = Counter(tuple(r[k] for k in KEYS) for r in list(old.values()) + list(chosen.values()))
    assert len(groups) == 60 and sum(n == 14 for n in groups.values()) == 56 and sum(n == 1 for n in groups.values()) == 4
    finish = read(out / "controller_finish.json")
    assert finish["rc"] == 0 and finish["completed"] == finish["expected"] == 440
    result = dict(status="passed", scope="Read-only saved outputs, receipts, source identity and launch order; no simulation.",
        new_candidates=440, preserved_prior_candidates=348, total_candidates=788,
        separately_authorized_I3_candidates=108, complete_14_candidate_groups=56, remaining_single_candidate_groups=4,
        new_backend_samples=4400, new_random_cpcm_samples=4400, new_terminal_states=len(terminals),
        new_terminal_backend_samples=80, final_failed_candidates=0, recorded_memory_retry_attempts=len(failures),
        old_output_files=previous, completed_formal_logs=440, initial_released_candidates=440, release_dependencies=0,
        timing_definition="Filename Unix_ns precedes Popen; plus receipt.wall_seconds approximates controller receipt time. The launch-order tolerance is 2.1 seconds for launch bookkeeping and polling.",
        candidates=summaries, separate_terminal_parents=terminals, failed_attempts=failures,
        first_priority_start_unix={name: starts[name] for name in plan["first_priority_candidates"]}, controller_completion=finish)
    write(HERE / "evidence/h3_i2_review.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "partial", "complete"))
    parser.add_argument("--out", type=Path, default=HERE / "out/h3_i2_440")
    args = parser.parse_args()
    result = plan_check() if args.command == "plan" else partial_check(args.out) if args.command == "partial" else completion_check(args.out)
    print(json.dumps({k: len(result[k]) if isinstance(result[k], list) else result[k]
                      for k in ("status", "candidates", "available_candidates", "new_candidates", "initial_release", "dependency_count") if k in result}))
