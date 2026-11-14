"""Remaining new-model J1 and J5 entries for an idle, single-worker J batch.

Preparation is a separate non-timing command. It never starts a runner.
"""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

import f_config
import hi_config
import hi_manifest
import hi_closeout_c as c_run
import j_timing as old

ROOT = hi_config.ROOT
DEST = ROOT / "e2e/hi/closeout"
OUT = ROOT / "e2e/hi/out/closeout_j_20261115"
GRAPH_DIR = DEST / "j_graphs"
MANIFEST = DEST / "manifests/j_micro.json"
VERSION = "hi_closeout_j_micro_v1"
INPUT_CASES = (("llama", "decode", "ffn"), ("qwen3moe", "decode", "ffn"), ("qwen3moe", "prefill", "ffn"))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


@contextmanager
def model_prepare():
    """Route graph construction to the already-validated model copies."""
    original = f_config.prepare
    f_config.prepare = hi_config.prepare
    try:
        yield
    finally:
        f_config.prepare = original


def graph_specs():
    with (ROOT / "e2e/hi/tables/h3_i2_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = {r["candidate_id"]: r for r in csv.DictReader(stream)}
    specs = []
    for model, phase, op in INPUT_CASES:
        matches = [s for s in hi_manifest.candidates() if s["model"] == model and s["phase"] == phase
                   and s["op"] == op and s["method"] == "A3h" and s["g"] == 0 and s["x"] == 123
                   and s["hardware_profile"] == "busybarn"]
        assert len(matches) == 1
        source = matches[0]
        record = ROOT / rows[source["candidate_id"]]["source"].replace("\\", "/")
        saved = old.read(record)
        assert saved["status"] == "ok" and all(saved[k] == v for k, v in source.items())
        folder = record.parent.parent
        input_id = f"{model}_{phase}_{op}"
        spec = dict(source, job_id="J1_prepare_" + input_id, kind="j1_prepare", module="hi_closeout_j_micro",
            input_id=input_id, source_record=rel(record),
            snapshot=rel(folder / "snapshots" / (source["candidate_id"] + ".pkl.gz")),
            source_raw=rel(folder / "raw" / (source["candidate_id"] + ".pkl.gz")),
            prepared_graph=rel(GRAPH_DIR / (input_id + ".pkl.gz")), protocol_version=VERSION)
        spec["input_sha256"] = {k: old.sha(ROOT / spec[k]) for k in ("source_record", "snapshot", "source_raw")}
        specs.append(spec)
    return specs


def prepare():
    rows = []
    for spec in graph_specs():
        with model_prepare():
            row = old.prepare_graph(spec, DEST / "j_prepare")
        row.update(spec=spec, job_id=spec["job_id"], status="ok", backend_calls=0, search_runs=0)
        old.write(DEST / "j_prepare/results" / (spec["job_id"] + ".json"), row)
        rows.append(row)
        print(json.dumps(dict(input=spec["input_id"], events=row["full_events"], seconds=row["prepare_total_seconds"])), flush=True)
    old.write(DEST / "evidence/j_micro_preparation.json", dict(status="passed", graphs=rows,
        formal_timing_runs=0, backend_calls=0, search_runs=0,
        fixed_candidate_rule="A3h g0/seed123; no quality selection; complete W1 graph"))
    return rows


def j1(spec, out):
    with model_prepare():
        result = old.j1(spec, out)
    import post_fg_bb as bb
    signatures = [str(s) for s in bb.forward.signatures]
    result.update(numba_signatures=signatures, prior_signature=spec["prior_signature"],
                  new_numba_signatures=sorted(set(signatures) - set(spec["prior_signature"])),
                  input_candidate=spec["input_candidate"], model=spec["model"],
                  preparation_excluded_from_steady_call=True)
    if spec["implementation"] == "fast":
        result["startup_interpretation"] = ("same signature as previous measured JIT/cache; this warmup is reported separately"
            if not result["new_numba_signatures"] else "new signature; first-call warmup contains actual compilation/load and is retained separately")
    return result


def j5(spec, out):
    started = time.perf_counter()
    split = c_run.split
    split.setup()
    local = c_run.private_out(ROOT / spec["quality_out"])
    t = time.perf_counter()
    prepare_spec = old.read(local / "results" / ("C_" + spec["model"] + "__split_prepare.json"))["spec"]
    with c_run.healthy_routes():
        state = c_run.load_split_state(prepare_spec, local)
    loaded = time.perf_counter() - t
    t = time.perf_counter()
    healthy_hp = split.platform(state["spec"])
    hp = split.platform(state["spec"], state["dead"])
    platforms = time.perf_counter() - t
    t = time.perf_counter()
    baseline = split.event_metrics(state["events"], healthy_hp)
    baseline_seconds = time.perf_counter() - t
    adapt, instrument_sha = old.instrumented_split(split)
    t = time.perf_counter()
    candidates = []
    for axis in spec["legal_axes"]:
        graph, info = adapt(state, hp, 2, axis)
        score = (info["metrics"]["tensor_rounded_flops"] - baseline["tensor_rounded_flops"],
                 info["metrics"]["communication"]["route_link_bytes"], 0 if axis == "N" else 1)
        candidates.append((score, axis, graph, info))
    score, axis, graph, info = min(candidates, key=lambda r: r[0])
    decision = time.perf_counter() - t
    t = time.perf_counter()
    digest = split.graph_digest(graph)
    assert digest == spec["graph_sha256"] and axis == spec["axis"]
    verification = time.perf_counter() - t
    t = time.perf_counter()
    split.write_pickle(out / "graphs" / (spec["job_id"] + ".pkl.gz"),
                       dict(events_blob=pickle.dumps(graph, pickle.HIGHEST_PROTOCOL), info=info))
    exported = time.perf_counter() - t
    return dict(model=spec["model"], factor=2, axis=axis, legal_axes=spec["legal_axes"],
        graph_sha256=digest, input_candidate=state["source_candidate"],
        load_state_seconds=loaded, platform_seconds=platforms, baseline_metrics_seconds=baseline_seconds,
        resident_decision_seconds=decision, frozen_graph_verification_seconds=verification,
        export_seconds=exported, complete_case_seconds=time.perf_counter() - started,
        candidates=[dict(axis=a, score=s, phases=i["j_phases"], build_seconds=i["build_seconds"])
                    for s, a, _, i in candidates], instrumented_function_sha256=instrument_sha,
        backend_calls=0, search_runs=0,
        scope="all legal m2 axes from the actual MF6 point; full construction/scoring/selection/export, no backend")


def child(spec, out):
    started = time.perf_counter()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert spec["serial_only"] and spec["research_repetition"] in (1, 2, 3)
    assert all(os.environ[k] == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"))
    result = {"j1": j1, "j5": j5}[spec["kind"]](spec, out)
    import e2e_pilot_peak
    result.update(job_id=spec["job_id"], spec=spec, kind=spec["kind"], status="ok", protocol_version=VERSION,
        wall_seconds=time.perf_counter()-started, peak_rss_gb=e2e_pilot_peak.peak()/2**30,
        environment=old.environment(), execution_source_sha256={n: old.sha(ROOT/"e2e"/n) for n in
            ("hi_closeout_j_micro.py", "hi_closeout_c.py", "hi_config.py", "j_timing.py", "post_fg_split.py", "post_fg_bb.py")})
    old.write(out / "results" / (spec["job_id"] + ".json"), result)
    print(json.dumps(dict(job_id=spec["job_id"], status="ok", wall_seconds=result["wall_seconds"])), flush=True)


def jobs(quality_out=c_run.OUT):
    items = []
    old_startup = ROOT / "e2e/hi/out/j_first_serial/results/J1_first_jit_r1.json"
    prior_signatures = old.read(old_startup)["numba_signatures"]
    def add(spec, seconds, rss):
        items.append(dict(id=spec["job_id"], kind=spec["kind"], spec=spec, deps=[],
            result=f"results/{spec['job_id']}.json", argv=["{python}", "-u", "-B", "e2e/hi_closeout_j_micro.py",
                "child", "--spec", "{spec}", "--out", "{out}"], estimated_seconds=seconds, peak_rss_gb=rss, priority=1))
    for source in graph_specs():
        path = DEST / "j_prepare/results" / (source["job_id"] + ".json")
        if not path.exists():
            continue
        prepared = old.read(path)
        assert prepared["status"] == "ok" and prepared["spec"] == source
        assert old.sha(ROOT / prepared["graph_file"]) == prepared["graph_file_sha256"]
        for implementation in ("original", "fast"):
            for repetition in range(1, 4):
                spec = dict(job_id=f"J1_{source['input_id']}_{implementation}_r{repetition}", module="hi_closeout_j_micro",
                    kind="j1", model=source["model"], implementation=implementation, research_repetition=repetition,
                    prepared_graph=prepared["graph_file"], prepared_graph_sha256=prepared["graph_file_sha256"],
                    input_candidate=source["candidate_id"], input_id=source["input_id"], prior_signature=prior_signatures,
                    prior_signature_source=rel(old_startup), prior_signature_source_sha256=old.sha(old_startup),
                    serial_only=True, protocol_version=VERSION, steady_repetitions=5, warmup_repetitions=1)
                add(spec, 60 if source["model"] == "qwen3moe" and source["phase"] == "prefill" else 10, 2.)
    local = c_run.private_out(quality_out)
    for model in hi_config.MODELS:
        representative_path = local / "representatives" / (model + ".json")
        prepared_path = local / "results" / ("C_" + model + "__split_prepare.json")
        if not representative_path.exists() or not prepared_path.exists():
            continue
        representative = old.read(representative_path)
        if not representative["applicable"]:
            continue
        prepared = old.read(prepared_path)
        graph = prepared["graphs"]["2"]
        for repetition in range(1, 4):
            spec = dict(job_id=f"J5_{model}_m2_r{repetition}", module="hi_closeout_j_micro", kind="j5", model=model,
                research_repetition=repetition, quality_out=rel(quality_out), legal_axes=representative["legal_axes"],
                axis=graph["axis"], graph_sha256=graph["graph_sha256"], graph=graph["graph"],
                source_prepare_record=rel(prepared_path), source_prepare_sha256=old.sha(prepared_path),
                serial_only=True, protocol_version=VERSION)
            add(spec, max(10., prepared["wall_seconds"]), 3.)
    return items


def plan(quality_out=c_run.OUT):
    rows = jobs(quality_out)
    result = dict(status="prepared_for_idle_serial_run", jobs=rows, preparation_backend_calls=0,
        worker_limit=1, quality_pool_must_finish=True, old_J_first_45_preserved=True,
        J1_expected=18, J1_ready=sum(j["kind"] == "j1" for j in rows),
        J5_maximum=6, J5_ready=sum(j["kind"] == "j5" for j in rows))
    old.write(MANIFEST, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "plan", "child"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--quality-out", type=Path, default=c_run.OUT)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "plan":
        r = plan(args.quality_out)
        print(json.dumps({k: v for k, v in r.items() if k != "jobs"}, indent=2))
    else:
        child(old.read(args.spec), args.out)
