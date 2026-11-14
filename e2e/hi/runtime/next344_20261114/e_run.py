"""E candidates, fresh-process search/evaluation, and bounded batches.

python e2e/e_run.py manifest --out e2e/e/pilot.json --pilot
python e2e/e_run.py run --manifest e2e/e/pilot.json --out e2e/e/out/pilot --workers 4
"""
import argparse
import collections
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "e2e"
SA_SEEDS = [123, 1, 2, 3, 4, 5, 6]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def historical(phase, op, shape, method, g, x):
    if shape == "1x4":
        cfg, base = "ffn14", ROOT / "cpcm_sa/out/s2"
    elif op == "ffn":
        cfg = "pf_ffn" if phase == "prefill" else "ffn"
        base = HERE / "out" if phase == "prefill" else ROOT / "cpcm_sa/out/s2"
    else:
        cfg = ("pf_" if phase == "prefill" else "dc_") + op + ("_w1" if op == "mha" else "")
        base = HERE / "out/b4"
    name = f"{cfg}_{method}_g{g}_{'l' if method == 'A1' else 's'}{x}"
    return {"run": str((base / "runs" / (name + ".pkl.gz")).relative_to(ROOT)),
            "snapshot": str((base / "snaps" / (name + ".pkl.gz")).relative_to(ROOT))}


def candidates():
    items = []
    for shape in ("2x2", "1x4"):
        for phase in ("prefill", "decode"):
            for op in ("ln", "mha", "proj", "ffn"):
                for method in ("A1", "A3h", "Gemini"):
                    variants = [(0, s) for s in [123, *range(1, 14)]] if method == "Gemini" else [
                        (g, x) for g in (0, 1) for x in (range(7) if method == "A1" else SA_SEEDS)]
                    for order, (g, x) in enumerate(variants):
                        suffix = f"s{x}" if method == "Gemini" else f"g{g}_{'l' if method == 'A1' else 's'}{x}"
                        item = dict(candidate_id=f"{phase}_{op}_{shape}_{method}_{suffix}", phase=phase,
                                    op=op, shape=shape, method=method, g=g, x=x, candidate_order=order,
                                    search_steps=1000, backend_seeds=list(range(1, 11)), kind="search")
                        if method != "Gemini" and (shape == "2x2" or (phase, op) == ("decode", "ffn")):
                            item.update(kind="reeval", **historical(phase, op, shape, method, g, x))
                        items.append(item)
    return items


def pilot_candidates():
    # One representative per Gemini task, plus new A1/A3h shape classes and old snapshots.
    return [s for s in candidates() if s["candidate_order"] == 0 and (
        s["method"] == "Gemini" or s["shape"] == "1x4" or s["method"] == "A3h")]


def configure(spec, out):
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["CPCM_SA_STEPS"] = str(spec["search_steps"])
    import e2e_pilot as pilot
    c = pilot.C
    c.WORK = os.environ.get("CRISP_E_ARTIFACT", str(HERE / "work/artifact"))
    c.OUT = str(out)
    cfg = f"e_{spec['phase']}_{spec['op']}_{spec['shape']}"
    model = "intra_mapping_gpt_" + spec["phase"]
    c.CONFIGS[cfg] = (model, spec["op"], None, "ch" + spec["shape"])
    c.REGIME[cfg] = "E_W1"
    c.BACKEND_SEEDS = spec["backend_seeds"]
    c._artifact_path()
    if spec["op"] == "mha":
        pilot.install_w1()
    return pilot, c, cfg


def signature(snapshot):
    behaviours, data, _, _ = snapshot
    # Mapping-independent operators, tensor sizes and producer/consumer relationships.
    rows = []
    for tag, behaviour in sorted(behaviours.items()):
        rows.append((tag, behaviour.beha_type, behaviour.beha_datashape,
                     sorted(behaviour.producer_tags), sorted(behaviour.consumer_tags)))
    tensors = [(tag, tensor.data_shape, tensor.data_type) for tag, tensor in sorted(data.items())]
    encoded = json.dumps((rows, tensors), sort_keys=True).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "behaviours": len(behaviours),
            "types": dict(collections.Counter(b.beha_type for b in behaviours.values())),
            "tensors": len(data)}


def save_snapshot(path, snapshot):
    behaviours, data, events, platform = snapshot
    blob = pickle.dumps((behaviours, data, events), protocol=pickle.HIGHEST_PROTOCOL)
    history = {k: dict(getattr(platform, k)) for k in ("link_loads_dict", "link_loads_count")}
    temporary = path.with_suffix(path.suffix + ".part")
    with gzip.open(temporary, "wb") as file:
        pickle.dump((blob, history), file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_snapshot(path, c, cfg):
    with gzip.open(path, "rb") as file:
        blob, history = pickle.load(file)
    platform = copy.deepcopy(c.platform(cfg))
    for key, value in history.items():
        assert set(getattr(platform, key)) == set(value), f"saved {key} topology differs"
        getattr(platform, key).update(value)
    return (*pickle.loads(blob), platform)


def evaluate(snapshot, ctx, c, add_broadcast, event_driver, *, random_ties=True):
    workload = signature(snapshot)
    started = time.perf_counter()
    result = c.final_eval(snapshot, ctx, add_broadcast, event_driver)
    events, platform = snapshot[2:]
    graph = c.cpcm.compile_bb(events, platform)
    ties = [int(c.cpcm.simulate(graph, "bb", rng=random.Random(s)).T) for s in range(1, 11)] if random_ties else []
    tags = set(events)
    dangling = sum(len((event.dependency_set | event.issue_set) - tags) for event in events.values())
    assert dangling == 0, f"W1 dependencies: {dangling} absent event references"
    result.update(T_hat_rand_mean=sum(ties) / len(ties) if ties else None, T_hat_rand=ties,
                  evaluation_seconds=time.perf_counter() - started, workload_signature=workload,
                  graph_events=graph.n, dangling_dependencies=dangling)
    return result


def record(spec, result, search_seconds, wall_seconds, peak, checks):
    return dict(spec, status="ok", search_seconds=search_seconds,
                evaluation_seconds=result["evaluation_seconds"], wall_seconds=wall_seconds,
                peak_rss_gb=peak, T_hat=int(result["T_bc"]), T_hat_pre=int(result["T_pre"]),
                T_hat_rand_mean=result["T_hat_rand_mean"], T_hat_rand=result["T_hat_rand"],
                backend=[dict(seed=int(seed), cycles=values[0], triple=list(values))
                         for seed, values in result["backend"].items()],
                backend_seconds=result["backend_wall"], workload_signature=result["workload_signature"],
                checks=checks, graph_events=result["graph_events"])


def child(spec, out, recover=False):
    started = time.perf_counter()
    out = Path(out).resolve()
    for folder in ("results", "snapshots", "raw", "logs"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    pilot, c, cfg = configure(spec, out)
    import add_communication
    import event_driver
    add_broadcast, backend = add_communication.add_broadcast, event_driver.event_driver
    name = spec["candidate_id"]
    snap_path = out / "snapshots" / (name + ".pkl.gz")
    checks = {}
    search_seconds = 0.0
    if spec["kind"] == "reeval":
        with gzip.open(ROOT / spec["run"], "rb") as file:
            previous = pickle.load(file)
        snapshot = load_snapshot(ROOT / spec["snapshot"], c, cfg)
        result = evaluate(snapshot, previous["meta"]["ctx"], c, add_broadcast, backend)
        checks["T_pre_same"] = int(result["T_pre"]) == int(previous["selected"]["T_pre"])
        checks["T_bc_same"] = int(result["T_bc"]) == int(previous["selected"]["T_bc"])
        assert all(checks.values()), checks
    elif spec["method"] == "Gemini":
        import e_gemini
        import Stream_Mapping
        task = Path(c.WORK) / "models" / c.CONFIGS[cfg][0] / "py" / (
            f"{spec['op']}_sq2048_gemini_ch{spec['shape']}{c.SUF}.py")
        source = task.read_text(encoding="utf-8")
        if spec["phase"] == "prefill" and spec["op"] == "mha":
            source, replacements = re.subn(r"(?m)^heads = 7$", "heads = 8", source)
            assert replacements == 1
        # The template's diagnostic backend is discarded; evaluate exported states below.
        event_driver.event_driver = lambda **kwargs: (0, 0, 0)
        with e_gemini.install(Stream_Mapping.StreamOptimization, seed=spec["x"],
                              steps=spec["search_steps"], validate_rollback=True,
                              find_context=c._find_context) as capture:
            try:
                exec(compile(source, str(task), "exec"), {"__file__": str(task), "__name__": "__main__"})
            except e_gemini.SearchComplete:
                pass
        event_driver.event_driver = backend
        save_snapshot(snap_path, capture["best"])
        ctx = capture["ctx"]
        result = evaluate(capture["best"], ctx, c, add_broadcast, backend)
        restored = load_snapshot(snap_path, c, cfg)
        checks = {"snapshot_workload_same": signature(restored) == result["workload_signature"],
                  "snapshot_T_pre_same": int(c.cpcm_sim(restored[2], restored[3])[1].T) == int(result["T_pre"]),
                  "gemini": capture["meta"]}
        assert checks["snapshot_workload_same"] and checks["snapshot_T_pre_same"]
        search_seconds = capture["meta"]["sa_wall_s"]
        primary_wall = time.perf_counter() - started
        primary = record(spec, result, search_seconds, primary_wall, pilot.peak() / 2**30, checks)
        del capture["best"], restored
        if capture.get("terminal") is not None:
            terminal_start = time.perf_counter()
            save_snapshot(out / "snapshots" / (name + "_terminal.pkl.gz"), capture["terminal"])
            terminal = evaluate(capture["terminal"], ctx, c, add_broadcast, backend, random_ties=False)
            terminal_spec = dict(spec, kind="terminal", candidate_id=name + "_terminal")
            write_json(out / "results" / (name + "_terminal.json"), record(
                terminal_spec, terminal, 0.0, time.perf_counter() - terminal_start,
                pilot.peak() / 2**30, {"from_search": name}))
        with gzip.open(out / "raw" / (name + ".pkl.gz"), "wb") as file:
            pickle.dump({"meta": {"ctx": ctx, "cfg": cfg}, "selected": result,
                         "gemini": capture["meta"]}, file, protocol=pickle.HIGHEST_PROTOCOL)
        # Publishing the primary record marks both endpoints complete for resume.
        write_json(out / "results" / (name + ".json"), primary)
        return
    else:
        raw_path = out / "raw" / (name + ".pkl.gz")
        original_final = c.final_eval
        evaluated = {}

        def final_with_ties(snapshot, ctx, broadcast, driver):
            c.final_eval = original_final
            evaluated.update(evaluate(snapshot, ctx, c, broadcast, driver))
            return evaluated

        if not recover:
            c.final_eval = final_with_ties
            try:
                c.child(cfg, spec["method"], spec["g"], spec["x"], str(raw_path), str(snap_path), True)
            except SystemExit as exit_event:
                if exit_event.code not in (0, None):
                    raise
        with gzip.open(raw_path, "rb") as file:
            run = pickle.load(file)
        result = run["selected"]
        search_seconds = run["sa_wall"]
        restored = load_snapshot(snap_path, c, cfg)
        checks = {"snapshot_workload_same": signature(restored) == result["workload_signature"],
                  "snapshot_T_pre_same": int(c.cpcm_sim(restored[2], restored[3])[1].T) == int(result["T_pre"]),
                  "revert": {key: int(value) for key, value in run["check_revert"].items()},
                  "selected_step": int(run["selected"]["step"])}
        assert checks["snapshot_workload_same"] and checks["snapshot_T_pre_same"]
    elapsed = spec["original_wall_seconds"] if recover else time.perf_counter() - started
    write_json(out / "results" / (name + ".json"), record(
        spec, result, search_seconds, elapsed, None if recover else pilot.peak() / 2**30, checks))


def run_batch(manifest, out, workers):
    items = json.loads(Path(manifest).read_text(encoding="utf-8"))
    out = Path(out).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "specs").mkdir(exist_ok=True)
    started = time.perf_counter()

    def launch(spec):
        name = spec["candidate_id"]
        result = out / "results" / (name + ".json")
        if result.exists():
            old = json.loads(result.read_text(encoding="utf-8"))
            if all(old[key] == spec[key] for key in spec):
                return dict(candidate_id=name, rc=0, cached=True)
        spec_path = out / "specs" / (name + ".json")
        write_json(spec_path, spec)
        begin = time.perf_counter()
        with (out / "logs" / (name + ".log")).open("w", encoding="utf-8") as log:
            completed = subprocess.run([sys.executable, "-u", "-B", str(Path(__file__).resolve()), "child",
                                        "--spec", str(spec_path), "--out", str(out)],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = dict(candidate_id=name, rc=completed.returncode, wall_seconds=time.perf_counter() - begin)
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(launch, items))
    write_json(out / "batch.json", dict(workers=workers, wall_seconds=time.perf_counter() - started,
                                        runs=rows, python=sys.version, platform=sys.platform))
    return int(any(row["rc"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("manifest")
    plan.add_argument("--out", required=True, type=Path)
    plan.add_argument("--pilot", action="store_true")
    job = sub.add_parser("child")
    job.add_argument("--spec", required=True, type=Path)
    job.add_argument("--out", required=True, type=Path)
    job.add_argument("--recover-output", action="store_true",
                     help="Publish an existing A1/A3h run after an output-format failure")
    batch = sub.add_parser("run")
    batch.add_argument("--manifest", required=True, type=Path)
    batch.add_argument("--out", required=True, type=Path)
    batch.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    if args.command == "manifest":
        items = pilot_candidates() if args.pilot else candidates()
        write_json(args.out, items)
        print(len(items), dict(collections.Counter(item["kind"] for item in items)))
    elif args.command == "child":
        child(json.loads(args.spec.read_text(encoding="utf-8")), args.out, args.recover_output)
    else:
        return run_batch(args.manifest, args.out, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
