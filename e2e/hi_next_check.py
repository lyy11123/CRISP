"""Independently check the completed 344-job batch and its reports, without simulation."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "e2e/hi"
KEYS = ("model", "hardware_profile", "phase", "op", "method")
PROTOCOLS = ("backend", "own", "random_cpcm", "unified_det_cpcm")
SEEDS = list(range(1, 11))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, label):
    assert math.isclose(float(actual), expected, rel_tol=1e-11, abs_tol=1e-8), (label, actual, expected)


def job_identity(spec):
    name = spec["candidate_id"]
    payload = dict(id=name, kind=spec["kind"], spec=spec, result=f"results/{name}.json",
                   argv=["{python}", "-u", "-B", "e2e/hi_run.py", "child", "--spec", "{spec}", "--out", "{out}"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def check_release_order(next_out, new, destination):
    """Check launch order from recorded timestamps, separately from model timing."""
    ids = {s["candidate_id"] for s in new}
    logs = defaultdict(list)
    empty_logs = []
    for path in (next_out / "logs").glob("*.log"):
        name, stamp = path.stem.rsplit(".", 1)
        assert stamp.isdigit() and name in ids, path
        if path.stat().st_size == 0:
            empty_logs.append(dict(candidate_id=name, path=path.relative_to(ROOT).as_posix(),
                                   start_unix_ns=int(stamp), bytes=0, sha256=sha(path)))
        else:
            final = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
            record = read(next_out / "results" / (name + ".json"))
            assert final["candidate_id"] == name and final["status"] == "ok" and final["seconds"] == record["wall_seconds"]
            logs[name].append((int(stamp), path))
    assert set(logs) == ids and all(len(v) == 1 for v in logs.values())
    assert len(empty_logs) == 1 and empty_logs[0]["candidate_id"] == "H_llama_busybarn_prefill_mha_A1_g0_l1"
    controller_log = next_out / "next_344.log"
    controller_text = controller_log.read_text(encoding="utf-8")
    trace = controller_text[controller_text.index("Traceback (most recent call last):"):]
    trace = trace[:trace.index("\n{\"event\": \"start\"")]
    assert 'File "<HI authorized existing pool>", line 171, in execute' in trace
    assert "AttributeError: 'types.SimpleNamespace' object has no attribute 'STDOUT'" in trace
    prelaunch = read(next_out / "controller_start_before_stdout_fix.json")
    pool_path = ROOT / "e2e/fg_batch.py"
    assert sha(pool_path) == prelaunch["source_sha256"]["fg_batch.py"]
    pool_source = pool_path.read_text(encoding="utf-8")
    execute_ast = next(n for n in ast.parse(pool_source).body if isinstance(n, ast.FunctionDef) and n.name == "execute")
    source = ast.get_source_segment(pool_source, execute_ast) + "\n"
    wrapper_path = HERE / "runtime/next344_20261114/hi_next_before_eta_fix.py"
    wrapper_ast = ast.parse(wrapper_path.read_text(encoding="utf-8"))
    wrapped = next(n for n in wrapper_ast.body if isinstance(n, ast.FunctionDef) and n.name == "wrapped_pool_source")
    replacements = next(n.value for n in wrapped.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "replacements" for t in n.targets))
    for before, after in ast.literal_eval(replacements).items():
        assert source.count(before) == 1
        source = source.replace(before, after)
    source = source.replace('key = tuple(receipt["category"])', 'key = category(jobs_by_id[receipt["id"]])')
    source_lines = source.splitlines()
    assert '.open("w", encoding="utf-8")' in source_lines[169]
    assert 'process = subprocess.Popen(' in source_lines[170] and 'stderr=subprocess.STDOUT' in source_lines[170]
    empty = empty_logs[0]
    assert datetime.fromisoformat(prelaunch["created_utc"]).timestamp() <= empty["start_unix_ns"] / 1e9 < logs[empty["candidate_id"]][0][0] / 1e9
    prelaunch_evidence = dict(empty_logs=empty_logs, controller_traceback=trace,
        controller_log_sha256=sha(controller_log), frozen_pool_sha256=sha(pool_path),
        wrapped_execute_open_log_line=170, wrapped_execute_popen_line=171,
        source_explanation="The frozen execute opens the log at line 170. Evaluating stderr=subprocess.STDOUT at line 171 raises before the Popen call; this empty file does not represent a child search or failed experiment.")
    starts = {name: rows[0][0] / 1e9 for name, rows in logs.items()}
    receipts = {name: read(next_out / "receipts" / (name + ".json")) for name in ids}
    ends = {name: starts[name] + receipts[name]["wall_seconds"] for name in ids}
    classes = [row for batch in read(HERE / "manifest/next_batch_release_plan.json")["batches"]
               for row in batch["class_calibrations"]]
    assert len(classes) == 32 and len({r["prerequisite_candidate"] for r in classes}) == 32
    delayed = [r for r in classes if r["release_after_success"]]
    assert len(delayed) == 20 and all(len(r["release_after_success"]) == 13 for r in delayed)
    # The filename stamp precedes Popen; wall_seconds starts immediately after
    # Popen and ends when the scheduler observes exit (roughly a 2-second poll).
    # Their sum estimates receipt time, not the child's precise exit instant.
    tolerance = 2.1
    dependencies = []
    for row in delayed:
        first = row["prerequisite_candidate"]
        assert receipts[first]["rc"] == 0
        for following in row["release_after_success"]:
            margin = starts[following] - ends[first]
            assert margin >= -tolerance, (first, following, margin)
            dependencies.append(dict(prerequisite_candidate=first, following_candidate=following,
                prerequisite_log_start_unix_ns=logs[first][0][0],
                prerequisite_receipt_wall_seconds=receipts[first]["wall_seconds"],
                estimated_prerequisite_receipt_unix=ends[first], following_log_start_unix_ns=logs[following][0][0],
                start_after_estimated_receipt_seconds=margin))
    assert len(dependencies) == 260 and len({r["following_candidate"] for r in dependencies}) == 260
    job_rows = table(destination / "tables/next_job_completion_times.csv")
    assert len(job_rows) == 344 and {r["candidate_id"] for r in job_rows} == ids
    origin = datetime.fromisoformat(read(next_out / "initial_84.start.json")["created_utc"]).timestamp()
    for row in job_rows:
        name = row["candidate_id"]
        assert int(row["start_unix_ns"]) == logs[name][0][0]
        assert row["log"].replace("\\", "/") == logs[name][0][1].relative_to(ROOT).as_posix()
        close(row["receipt_wall_seconds"], receipts[name]["wall_seconds"], name + ":receipt_time")
        assert abs(datetime.fromisoformat(row["approximate_completed_utc"]).timestamp() - ends[name]) < 1e-5
        close(row["minutes_from_initial_launch"], (ends[name] - origin) / 60, name + ":completion_time")
    batches = table(destination / "tables/next_batch_completion_times.csv")
    assert len(batches) == 4
    for row in batches:
        subset = [s["candidate_id"] for s in new if ("I1" if s["batch"].startswith("I1") else s["batch"]) == row["batch"]]
        assert int(row["planned"]) == int(row["completed_records"]) == int(row["timed_records"]) == len(subset)
        assert row["state"] == "complete"
        last = max(subset, key=lambda n: ends[n])
        assert row["latest_timed_candidate"] == last
        close(row["minutes_from_initial_launch"], (ends[last] - origin) / 60, row["batch"] + ":completion_time")
    return dict(status="passed", total_log_files=345, unique_completed_formal_logs=344, formal_attempts_per_candidate=1,
        empty_prelaunch_files=1, prelaunch_controller_failure=prelaunch_evidence,
        first_candidates=32, classes_with_following_candidates=20, checked_dependencies=260,
        timestamp_tolerance_seconds=tolerance,
        minimum_start_after_estimated_receipt_seconds=min(r["start_after_estimated_receipt_seconds"] for r in dependencies),
        negative_margins_within_timestamp_tolerance=sum(r["start_after_estimated_receipt_seconds"] < 0 for r in dependencies),
        timing_definition="log filename Unix_ns is taken before Popen; receipt.wall_seconds starts after Popen and includes exit polling. Their sum approximates receipt time, not exact process exit. A 2.1-second tolerance covers launch bookkeeping and the approximately 2-second poll.",
        dependencies=dependencies)


def check_preserved_h1(pilot_out):
    previous = read(HERE / "evidence/HIJ_FIRST_BATCH_20261114_complete.tar.verified.json")["inventory"]["files"]
    prefix = pilot_out.relative_to(ROOT).as_posix() + "/"
    frozen = {r["path"]: r for r in previous if r["path"].startswith(prefix)}
    actual = {p.relative_to(ROOT).as_posix(): p for p in pilot_out.rglob("*") if p.is_file()
              and "__pycache__" not in p.parts and not p.name.endswith((".part", ".tmp"))}
    assert frozen and set(frozen) == set(actual)
    assert all(actual[name].stat().st_size == row["bytes"] and sha(actual[name]) == row["sha256"]
               for name, row in frozen.items())
    return dict(status="unchanged", files=len(frozen), baseline="HIJ_FIRST_BATCH_20261114_complete.tar.verified.json")


def check(next_out, pilot_out, destination, evidence):
    manifests = [HERE / "manifest" / (name + "_planned.json") for name in ("h2d", "h2p", "h1c", "i1")]
    new = [spec for path in manifests for spec in read(path)["jobs"]]
    old = read(HERE / "manifest/h1_main.json")["jobs"]
    universe = {s["candidate_id"]: s for s in read(HERE / "manifest/all_candidates_planned.json")["jobs"]}
    new_ids, old_ids = {s["candidate_id"] for s in new}, {s["candidate_id"] for s in old}
    assert len(new) == len(new_ids) == 344 and len(old) == len(old_ids) == 4
    assert not new_ids & old_ids
    assert all(spec == universe[spec["candidate_id"]] for spec in new + old)
    assert all(s["search_steps"] == 1000 and s["backend_seeds"] == SEEDS and s["random_cpcm_seeds"] == SEEDS for s in new + old)
    terminal_parents = {s["candidate_id"] for s in new if s["method"] == "Gemini" and s["x"] == 123}
    assert len(terminal_parents) == 8
    terminal_ids = {name + "_terminal" for name in terminal_parents}
    assert {p.stem for p in next_out.glob("results/*.json")} == new_ids | terminal_ids
    assert {p.stem for p in pilot_out.glob("results/*.json")} == old_ids
    for folder, suffix in (("raw", ".pkl.gz"), ("snapshots", ".pkl.gz"), ("receipts", ".json"), ("specs", ".json")):
        assert all(not (next_out / folder / (name + suffix)).exists() for name in old_ids), folder
    preserved_h1 = check_preserved_h1(pilot_out)
    release_order = check_release_order(next_out, new, destination)

    contracts = {}
    prior_review = read(HERE / "evidence/report_review.json")
    for model, groups in (("llama", 8), ("qwen3moe", 4)):
        path = HERE / "models" / model / "model_contract.json"
        relative = path.relative_to(ROOT).as_posix()
        frozen = next(row for row in prior_review["files"] if row["path"].replace("\\", "/") == relative)
        assert sha(path) == frozen["sha256"], model
        full = read(path)
        core = {k: v for k, v in full.items() if k not in ("generated_sources", "validation")}
        digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
        assert all(s["model_contract_sha256"] == digest for s in new + old if s["model"] == model)
        assert core["head_groups"] == groups and core["q_heads_per_graph"] == 8 and core["kv_heads_per_graph"] == 1
        assert core["prefill_chain"] == dict(norm=2, mha=groups, proj=1, ffn=1)
        assert core["decode_chain"] == dict(norm=1, mha=groups, proj=1, ffn=1)
        assert core["decode_ffn_includes_rmsnorm"] and not core["prefill_ffn_includes_rmsnorm"]
        contracts[model] = core

    prior_pilots = {r["candidate_id"]: r for r in read(HERE / "evidence/h1_graph_measured.json")["rows"]}
    values, checked, terminals = {}, [], {}
    for out, specs in ((pilot_out, old), (next_out, new)):
        for spec in specs:
            name = spec["candidate_id"]
            paths = {kind: out / folder / (name + suffix) for kind, folder, suffix in (
                ("result", "results", ".json"), ("raw", "raw", ".pkl.gz"), ("snapshot", "snapshots", ".pkl.gz"))}
            record = read(paths["result"])
            assert record["status"] == "ok" and all(record[k] == v for k, v in spec.items()), name
            assert read(out / "specs" / (name + ".json")) == spec, name
            receipt = read(out / "receipts" / (name + ".json"))
            assert receipt["id"] == name and receipt["rc"] == 0 and receipt["identity"] == job_identity(spec), name
            assert receipt["result"] == f"results/{name}.json"
            assert record["exact_rollback"] and not record["debug_checks"]
            with gzip.open(paths["raw"], "rb") as stream:
                raw = pickle.load(stream)
            with gzip.open(paths["snapshot"], "rb") as stream:
                snapshot = pickle.load(stream)
            assert len(snapshot) == 2 and isinstance(snapshot[0], bytes) and snapshot[0]
            assert set(snapshot[1]) == {"link_loads_dict", "link_loads_count"}
            assert raw["meta"]["steps"] == 1000
            selected = raw["selected"]
            if spec["method"] == "Gemini":
                meta = raw["gemini"]
                assert meta["steps"] == 1000 and meta["anneal_seed"] == spec["x"] and meta["initializer_seed"] == 123
                assert meta["LP_flag"] and meta["hops_only"] and not meta["dijkstra_routing"]
                assert meta["rollback"] == "pre_move_snapshot"
                assert record["selected_step"] == meta["best_step"] and 0 <= meta["best_step"] <= 1000
                assert meta["best_energy"] <= meta["terminal_energy"]
                own = primary = meta["best_energy"]
            else:
                assert len(raw["rows"]) == 1000 and [r["step"] for r in raw["rows"]] == list(range(1, 1001))
                assert raw["meta"]["arm"] == spec["method"] and raw["meta"]["g"] == spec["g"] and raw["meta"]["x"] == spec["x"]
                assert raw["meta"]["sa_seed"] == (123 if spec["method"] == "A1" else spec["x"])
                assert not raw["check2"] and not raw["check4"] and raw["check_revert"]["n"] == 0
                assert record["instrumentation"]["original_algorithm_ast_identical_after_removing_observations"]
                for row in raw["rows"]:
                    temperature = raw["meta"]["Tmax"] * math.exp(math.log(raw["meta"]["Tmin"] / raw["meta"]["Tmax"]) * row["step"] / 1000)
                    close(row["temp"], temperature, name + ":temperature")
                    assert row["accepted"] or "t_revert" in row
                best_energy = [r["best_E"] for r in raw["rows"]]
                assert all(b <= a for a, b in zip(best_energy, best_energy[1:]))
                assert selected["own_cost"] == best_energy[-1]
                assert record["selected_step"] == selected["step"] and 0 <= selected["step"] <= 1000
                own = selected["own_cost"]
                primary = selected["T_sel"] if spec["method"] == "A3h" else own
            assert [r["seed"] for r in record["backend"]] == SEEDS
            assert list(selected["backend"]) == SEEDS
            backend = [r["cycles"] for r in record["backend"]]
            assert all(v > 0 for v in backend)
            for item in record["backend"]:
                assert item["triple"] == list(selected["backend"][item["seed"]]) and item["cycles"] == item["triple"][0]
            ties = record["T_hat_rand"]
            assert len(ties) == 10 and ties == selected["T_hat_rand"] and all(v > 0 for v in ties)
            assert record["T_hat_rand_mean"] == statistics.mean(ties) == selected["T_hat_rand_mean"]
            assert record["T_hat"] == selected["T_bc"] and record["T_hat_pre"] == selected["T_pre"]
            assert record["workload_signature"] == selected["workload_signature"]
            for key in ("backend_individual_seconds", "random_cpcm_individual_seconds"):
                assert len(record["timing"]["evaluation"][key]) == 10
            if name in new_ids:
                assert record["own_cost"] == own
                for filename, source_hash in record["source_sha256"].items():
                    assert sha(HERE / "runtime/next344_20261114" / filename) == source_hash, (name, filename)
                orders = record["backend_input_orders"]
                assert orders == selected["backend_input_orders"] and [r["seed"] for r in orders] == SEEDS
                assert all(len(r["actual_iteration_sha256"]) == 64 for r in orders)
            hashes = {kind: sha(path) for kind, path in paths.items()}
            if name in old_ids:
                for key, field in (("result", "results"), ("raw", "raw"), ("snapshot", "snapshots")):
                    assert hashes[key] == prior_pilots[name]["input_sha256"][field], (name, key)
            values[name] = dict(spec=spec, backend=backend, ties=ties, mean=statistics.mean(backend),
                minimum=min(backend), maximum=max(backend), deviation=statistics.stdev(backend),
                primary=primary, own=own, deterministic=record["T_hat"], random=statistics.mean(ties),
                order=spec["candidate_order"])
            checked.append(dict(candidate_id=name, **hashes, receipt_identity=receipt["identity"],
                full_1000_steps=True, backend_samples=10, random_cpcm_samples=10,
                prior_h1_unchanged=name in old_ids, source_sha256=record["source_sha256"]))
            if name in terminal_parents:
                terminal = read(out / "results" / (name + "_terminal.json"))
                assert terminal["status"] == "ok" and terminal["kind"] == "terminal" and terminal["from_search"] == name
                assert terminal["candidate_id"] == name + "_terminal"
                assert all(terminal[k] == v for k, v in spec.items() if k not in ("kind", "candidate_id"))
                assert set(terminal["backend"]) == {str(s) for s in SEEDS}
                assert terminal["random_cpcm_evaluations"] == 0 and not terminal["timing"]["random_cpcm_individual_seconds"]
                assert terminal["workload_signature"] == record["workload_signature"]
                assert [r["seed"] for r in terminal["backend_input_orders"]] == SEEDS
                assert all(len(r["actual_iteration_sha256"]) == 64 for r in terminal["backend_input_orders"])
                assert len(terminal["timing"]["backend_individual_seconds"]) == 10
                terminal_snapshot = out / "snapshots" / (name + "_terminal.pkl.gz")
                assert terminal_snapshot.stat().st_size > 0
                terminals[name] = dict(samples=[terminal["backend"][str(s)][0] for s in SEEDS],
                    result_sha256=sha(out / "results" / (name + "_terminal.json")), snapshot_sha256=sha(terminal_snapshot))

    groups = defaultdict(list)
    for name, row in values.items():
        groups[tuple(row["spec"][k] for k in KEYS)].append(name)
    full = {key: names for key, names in groups.items() if key[:2] == ("llama", "busybarn")}
    assert len(full) == 24 and sum(map(len, full.values())) == 336 and len(groups) == 36
    for key, names in full.items():
        assert len(names) == 14 and {values[n]["order"] for n in names} == set(range(14))
        method = key[-1]
        expected = {(0, x) for x in (123, *range(1, 14))} if method == "Gemini" else {
            (g, x) for g in (0, 1) for x in (range(7) if method == "A1" else (123, 1, 2, 3, 4, 5, 6))}
        assert {(values[n]["spec"]["g"], values[n]["spec"]["x"]) for n in names} == expected

    candidate_rows = table(destination / "tables/next_candidates.csv")
    assert len(candidate_rows) == 348 and {r["candidate_id"] for r in candidate_rows} == set(values)
    for row in candidate_rows:
        item = values[row["candidate_id"]]
        assert all(row[k] == item["spec"][k] for k in KEYS)
        for field, field_value in (("backend_mean_ns", item["mean"]), ("own_selector_primary", item["primary"]),
                                   ("own_cost", item["own"]), ("deterministic_cpcm_ns", item["deterministic"]),
                                   ("random_cpcm_mean_ns", item["random"])):
            close(row[field], field_value, row["candidate_id"] + ":" + field)
    samples = table(destination / "tables/next_samples.csv")
    assert len(samples) == 3480 and len({(r["candidate_id"], r["seed"]) for r in samples}) == 3480
    for row in samples:
        item, index = values[row["candidate_id"]], int(row["seed"]) - 1
        assert int(row["backend_ns"]) == item["backend"][index] and int(row["random_cpcm_ns"]) == item["ties"][index]

    operator_rows = table(destination / "tables/next_operators.csv")
    assert len(operator_rows) == len(groups) and {tuple(r[k] for k in KEYS) for r in operator_rows} == set(groups)
    winners, selections = {}, []
    for row in operator_rows:
        key = tuple(row[k] for k in KEYS)
        names = groups[key]
        assert int(row["completed"]) == len(names) and int(row["planned"]) == 14
        if key not in full:
            assert len(names) == 1 and all(not row[p + "_candidate"] for p in PROTOCOLS)
            continue
        selectors = dict(backend=lambda n: (values[n]["mean"], values[n]["order"]),
            own=lambda n: (values[n]["primary"], values[n]["own"], values[n]["order"]),
            random_cpcm=lambda n: (values[n]["random"], values[n]["order"]),
            unified_det_cpcm=lambda n: (values[n]["deterministic"], values[n]["order"]))
        best_mean = min(values[n]["mean"] for n in names)
        for protocol, selector in selectors.items():
            name = min(names, key=selector)
            assert row[protocol + "_candidate"] == name, (key, protocol)
            item = values[name]
            for suffix, expected in (("mean_ns", item["mean"]), ("min_ns", item["minimum"]),
                    ("max_ns", item["maximum"]), ("stdev_ns", item["deviation"]),
                    ("selection_gap", item["mean"] / best_mean - 1)):
                close(row[protocol + "_" + suffix], expected, str(key) + ":" + protocol + ":" + suffix)
            winners[(*key, protocol)] = name
            selections.append(dict(group=list(key), protocol=protocol, selected_candidate=name))

    layers = table(destination / "tables/next_layers.csv")
    assert len(layers) == 12 and {(r["method"], r["protocol"]) for r in layers} == {
        (m, p) for m in ("A1", "A3h", "Gemini") for p in PROTOCOLS}
    layer_checks = []
    for row in layers:
        assert row["model"] == "llama" and row["hardware_profile"] == "busybarn"
        assert int(row["mha_multiplier"]) == 8 and int(row["batch"]) == 8
        assert int(row["prefill_norm_multiplier"]) == 2 and int(row["decode_norm_multiplier"]) == 1
        phase_values = {}
        for phase, norms in (("prefill", 2), ("decode", 1)):
            weights = dict(ln=norms, mha=8, proj=1, ffn=1)
            ids = {op: winners[("llama", "busybarn", phase, op, row["method"], row["protocol"])] for op in weights}
            assert json.loads(row[phase + "_sources"]) == ids
            operator_sum = sum(weights[op] * values[name]["mean"] for op, name in ids.items())
            byte_count = 8 * (2048 if phase == "prefill" else 1) * 8192 * 2
            handoff = 20 + byte_count / 256
            assert int(row[phase + "_handoff_bytes"]) == byte_count
            close(row[phase + "_operator_sum_ns"], operator_sum, phase + ":operator_sum")
            close(row[phase + "_handoff_ns"], handoff, phase + ":handoff")
            close(row[phase + "_ns"], operator_sum + handoff, phase + ":total")
            for suffix, metric in (("minima", "minimum"), ("maxima", "maximum")):
                expected = sum(weights[op] * values[name][metric] for op, name in ids.items()) + handoff
                close(row[phase + "_sum_of_component_" + suffix + "_ns"], expected, phase + ":" + suffix)
            phase_values[phase] = operator_sum + handoff
        close(row["TTFT_ms"], phase_values["prefill"] / 1e6, "TTFT")
        close(row["TPOT_ms"], phase_values["decode"] / 1e6, "TPOT")
        for length in (128, 512, 2048):
            close(row[f"request_additional_decode_{length}_ns"], phase_values["prefill"] + length * phase_values["decode"], f"request_{length}")
        layer_checks.append(dict(method=row["method"], protocol=row["protocol"], **phase_values))

    fullmodel = table(destination / "tables/next_fullmodel_reference.csv")
    layer_lookup = {(r["method"], r["protocol"]): r for r in layers}
    assert len(fullmodel) == 12 and {(r["method"], r["protocol"]) for r in fullmodel} == set(layer_lookup)
    for row in fullmodel:
        layer = layer_lookup[row["method"], row["protocol"]]
        assert int(row["layers"]) == 80 and row["crossgroup_connections"] == "B"
        assert row["equation"] == "constant_ns + B * per_connection_ns"
        constants, connections = {}, {}
        for phase in ("prefill", "decode"):
            constants[phase] = 80 * float(layer[phase + "_operator_sum_ns"])
            connections[phase] = float(layer[phase + "_handoff_ns"])
            close(row[phase + "_constant_ns"], constants[phase], phase + ":80_layer_constant")
            close(row[phase + "_per_connection_ns"], connections[phase], phase + ":connection")
        for length in (128, 512, 2048):
            close(row[f"request_additional_decode_{length}_constant_ns"], constants["prefill"] + length * constants["decode"], "request_constant")
            close(row[f"request_additional_decode_{length}_per_connection_ns"], connections["prefill"] + length * connections["decode"], "request_connection")

    terminal_rows = table(destination / "tables/next_gemini_terminal.csv")
    assert len(terminal_rows) == 8 and {r["candidate_id"] for r in terminal_rows} == terminal_parents
    for row in terminal_rows:
        name = row["candidate_id"]
        assert json.loads(row["samples"]) == terminals[name]["samples"]
        close(row["terminal_backend_mean_ns"], statistics.mean(terminals[name]["samples"]), name + ":terminal_mean")
        close(row["best_backend_mean_ns"], values[name]["mean"], name + ":best_mean")
    audit = read(destination / "evidence/next_completion_audit.json")
    for field, expected in dict(status="complete", next_planned=344, next_complete=344, reused_h1=4,
            healthy_candidates=348, backend_samples=3480, random_cpcm_samples=3480,
            gemini_terminal_states=8, gemini_terminal_backend_samples=80, complete_operator_groups=24).items():
        assert audit[field] == expected, field
    assert not audit["missing"] and audit["model_contracts"] == contracts
    audit_rows = {r["candidate_id"]: r for r in audit["candidates"]}
    assert set(audit_rows) == set(values)
    for row in checked:
        assert all(audit_rows[row["candidate_id"]][key] == row[key] for key in ("raw", "snapshot", "result"))
    assert any(b["completed"] == 344 and not b["failed"] and not b["pending"] for b in audit["batches"])
    finish = read(next_out / "controller_finish.json")
    assert finish["rc"] == 0 and finish["completed"] == finish["expected"] == 344
    assert not audit["controller"]["failed_attempts"]
    result = dict(status="passed", scope="Independent read-only checks; no SA, backend, graph simulation or graph reconstruction.",
        new_candidates=344, reused_h1=4, formal_candidates=348, main_llama_candidates=336,
        complete_operator_groups=24, candidates_per_complete_group=14, selection_checks=len(selections),
        layer_checks=len(layer_checks), backend_samples=3480, random_cpcm_samples=3480,
        separate_terminal_states=8, separate_terminal_backend_samples=80, candidates=checked,
        preserved_h1_directory=preserved_h1, release_order=release_order, fullmodel_reference_rows_checked=12,
        experiment_failures=0, prelaunch_controller_failures=1,
        new_runtime_sources_match_archived_execution_files=True, controller_completion=finish,
        selected_candidates=selections, assembled_layers=layer_checks, terminals=terminals,
        scientific_notes=["Full layer rows are Llama main-hardware serial sums, not measured full-model latency.",
            "Prefill weights 2/8/1/1 and decode weights 1/8/1/1; decode FFN already includes one RMSNorm.",
            "Handoff is an analytic adjacent-link contribution, kept separate from simulated operators.",
            "The 80-layer reference is 80 times operator sums plus B times per-connection transfer; B remains unspecified.",
            "WATOS and Qwen rows remain one-candidate pilots; no complete-layer values are inferred.",
            "Gemini raw metadata certifies the completed native 1000-step capture; A1/A3h also retain 1000 per-step rows."],
        report_files=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in (
            destination / "tables/next_candidates.csv", destination / "tables/next_samples.csv",
            destination / "tables/next_operators.csv", destination / "tables/next_layers.csv",
            destination / "tables/next_fullmodel_reference.csv", destination / "tables/next_job_completion_times.csv",
            destination / "tables/next_batch_completion_times.csv",
            destination / "tables/next_gemini_terminal.csv", destination / "evidence/next_completion_audit.json")])
    evidence.parent.mkdir(parents=True, exist_ok=True)
    temporary = evidence.with_suffix(evidence.suffix + ".part")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(evidence)
    print(json.dumps({k: result[k] for k in ("status", "new_candidates", "reused_h1", "selection_checks", "layer_checks")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next-out", type=Path, default=HERE / "out/next_344")
    parser.add_argument("--pilot-out", type=Path, default=HERE / "out/h1_main")
    parser.add_argument("--destination", type=Path, default=HERE)
    parser.add_argument("--evidence", type=Path, default=HERE / "evidence/next_344_review.json")
    args = parser.parse_args()
    check(args.next_out, args.pilot_out, args.destination, args.evidence)
