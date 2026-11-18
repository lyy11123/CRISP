"""H/I saved-input adapter and A256 execution-order replay for the S256 supplement."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

from fg_fault import ROOT, read_json, read_pickle, write_json, platform_signature
import post_fg_dstu as dstu

VERSION = "dstu_supplement_hi_v1"
STATES = ("healthy_A3h", "D_A3hEIR")
MODELS = {"llama": ("h5", "H5_llama"), "qwen3moe": ("i5", "I5_qwen3moe")}


def context(model, state, phase=.5, seed=1):
    """Read the old input records and preserve their shared reference and C4 rank."""
    import hi_config as H
    stage, prefix = MODELS[model]
    assert state in STATES
    folder = ROOT / "e2e/hi/out" / stage / "results"
    graph_file = folder / f"{prefix}__graph_{state}.json"
    reference_file = folder / f"{prefix}__reference.json"
    graph, reference = read_json(graph_file), read_json(reference_file)
    assert graph["model_contract_sha256"] == H.digest(H.model_contract(model))
    arm = "A3h_A256" if state == "healthy_A3h" else "A3h_EIR_A256"
    old_file = folder / f"{prefix}__joint_{state}_{arm}_g{phase}_s{seed}.json"
    old = read_json(old_file)
    assert old["status"] == "ok"
    tau = int(round(reference["KV_start"] + phase * (reference["KV_end"] - reference["KV_start"])))
    assert tau == old["tau"]
    deadlines = {0: 1.5 * graph["that"]["0"], 1: tau + 1.5 * graph["that"]["1"]}
    order = sorted(deadlines, key=lambda stream: (deadlines[stream], stream))
    rank = {stream: order.index(stream) for stream in order}
    return dict(model=model, state=state, model_contract_sha256=graph["model_contract_sha256"],
                hardware_profile="busybarn", shape="2x2", split_mode="fixed", d2d_scale=1,
                condition=graph["spec"]["condition"], workload_version=H.WORKLOAD_VERSION,
                source_root=str(ROOT), graph_job=graph["job_id"],
                graph_record=graph_file.relative_to(ROOT).as_posix(),
                graph_sha256=dstu.sha(graph_file),
                graph_files_sha256={str(stream): dstu.sha(ROOT / graph["files"][f"orig_{stream}"])
                                    for stream in (0, 1)},
                reference_record=reference_file.relative_to(ROOT).as_posix(),
                reference_sha256=dstu.sha(reference_file), reference=reference,
                old_A256_result=old_file.relative_to(ROOT).as_posix(), old_A256_sha256=dstu.sha(old_file),
                rank=rank, frozen_A256_deadlines=deadlines, tau=tau, phase=phase, seed=seed,
                fresh_protocol="fresh_transform_v1", adapter_version=VERSION,
                arm="S256", run_type="joint", trace=False)


load_context = context


@contextmanager
def model_prepare():
    import f_config
    import hi_config
    previous = f_config.prepare
    f_config.prepare = hi_config.prepare
    try:
        yield
    finally:
        f_config.prepare = previous


def graph_inputs(spec, out):
    """Return (d, hp, graph, raw streams), as post_fg_dstu.simulate expects."""
    import fg_joint
    import f_config
    import hi_config as H
    source_root = Path(spec["source_root"])
    graph_file = source_root / spec["graph_record"]
    assert dstu.sha(graph_file) == spec["graph_sha256"]
    graph = read_json(graph_file)
    assert graph["model_contract_sha256"] == spec["model_contract_sha256"] == H.digest(H.model_contract(spec["model"]))
    setup = dict(spec, phase="prefill", op="mha", method="A3h", g=0, x=123)
    with model_prepare():
        H.prepare(setup, out)
        d = fg_joint.d_modules()
        hp = f_config.build_platform(setup, out, faults=graph["fault"], shape_override="2x4")
    assert platform_signature(hp) == graph["platform_sha256"]
    streams = {}
    for stream in (0, 1):
        filename = source_root / graph["files"][f"orig_{stream}"]
        assert dstu.sha(filename) == spec["graph_files_sha256"][str(stream)]
        streams[stream] = read_pickle(filename)
    return d, hp, graph, streams


@contextmanager
def simulation_adapter():
    original = dstu.graph_inputs
    dstu.graph_inputs = graph_inputs
    try:
        yield
    finally:
        dstu.graph_inputs = original


def order_comparison(left, right):
    fields = dict(event_order=list(left) == list(right), event_tags=set(left) == set(right))
    differences = {"dependencies": [], "successors": [], "targets": []}
    for tag in left.keys() & right.keys():
        a, b = left[tag], right[tag]
        if list(a.dependency_set) != list(b.dependency_set):
            differences["dependencies"].append(tag)
        if list(a.issue_set) != list(b.issue_set):
            differences["successors"].append(tag)
        if a.event_type == b.event_type == "communication":
            ta, tb = a.target_location, b.target_location
            if isinstance(ta, set) and isinstance(tb, set) and list(ta) != list(tb):
                differences["targets"].append(tag)
    fields.update({key + "_different": len(tags) for key, tags in differences.items()})
    fields["examples"] = {key: sorted(tags)[:3] for key, tags in differences.items()}
    fields["same"] = fields["event_order"] and fields["event_tags"] and not any(differences.values())
    return fields


def check(model, state, out, replay=False):
    """Compare actual inputs; optional replay invokes exactly one old A256 backend."""
    import hi_run
    import e2e_pilot_peak
    started = time.perf_counter()
    spec = context(model, state)
    d, hp, graph, streams = graph_inputs(spec, out)
    prepared = time.perf_counter()
    source_root = Path(spec["source_root"])
    loaded = {f"orig_{stream}": streams[stream] for stream in (0, 1)}
    loaded.update({f"hop_{stream}": read_pickle(source_root / graph["files"][f"hop_{stream}"])
                   for stream in (0, 1)})
    recorded = graph["actual_iteration_sha256"]
    loaded_order = {key: hi_run.execution_order_digest(events) for key, events in loaded.items()}
    # Historical fg_joint.joint_job loads two streams, merges, then serializes;
    # d_joint.run_backend deserializes that merged graph immediately before use.
    old_merged = dict(loaded["hop_0"])
    old_merged.update(loaded["hop_1"])
    old_backend_input = pickle.loads(pickle.dumps(old_merged, pickle.HIGHEST_PROTOCOL))
    fresh, _, _, chunk_stats = dstu.transform(streams, hp, "A256", spec["rank"])
    transformed = time.perf_counter()
    comparison = order_comparison(old_backend_input, fresh)
    canonical_same = dstu.graph_digest(old_backend_input) == dstu.graph_digest(fresh)
    # Old graph records cover per-stream reload order, not the final merged
    # backend input. Numerical agreement alone does not supply this missing hash.
    old_result = read_json(source_root / spec["old_A256_result"])
    result = dict(status="passed", model=model, state=state, adapter_version=VERSION,
                  spec=spec, source_graph_files=graph["files"],
                  source_graph_files_sha256={key: dstu.sha(source_root / value) for key, value in graph["files"].items()},
                  recorded_stream_iteration_sha256=recorded,
                  reloaded_stream_iteration_sha256=loaded_order,
                  stream_iteration_matches_record={key: loaded_order[key] == recorded[key] for key in loaded},
                  old_final_backend_iteration_recorded="initial_order_sha256" in old_result,
                  old_reconstructed_backend_order_sha256=dstu.execution_order_digest(old_backend_input),
                  fresh_A256_backend_order_sha256=dstu.execution_order_digest(fresh),
                  same_canonical_graph=canonical_same, actual_order_comparison=comparison,
                  fresh_chunk_stats=chunk_stats, rank=spec["rank"], tau=spec["tau"],
                  input_prepare_seconds=prepared - started, comparison_seconds=transformed - prepared,
                  historical_baseline_reusable=False,
                  baseline_action="new original/A256/S256 comparisons on one fresh_transform_v1 input path",
                  reason="historical final merged input iteration was not recorded; fresh and saved-hop paths are compared explicitly",
                  new_backend_calls=0)
    if replay:
        import e2e_pilot
        replay_started = time.perf_counter()
        simulated = d.run_backend(e2e_pilot.C, pickle.dumps(old_merged, pickle.HIGHEST_PROTOCOL), hp,
                                  "hop", {"name": "C4", "rank": spec["rank"]},
                                  {0: 0, 1: spec["tau"]}, spec["seed"])
        metrics = {"P_completion": simulated["streams"][0]["C"],
                   "D_latency": simulated["streams"][1]["C"] - spec["tau"]}
        expected = {key: old_result["metrics"][key] for key in metrics}
        result["historical_A256_replay"] = dict(metrics=metrics, expected=expected,
            exact_output_match=metrics == expected, elapsed_seconds=time.perf_counter() - replay_started,
            simulation=simulated, meaning="same saved-hop loading procedure, not a historical final-input hash")
        result["new_backend_calls"] = 1
    result.update(wall_seconds=time.perf_counter() - started, peak_rss_gib=e2e_pilot_peak.peak() / 2**30)
    filename = Path(out) / "evidence" / f"baseline_protocol_check_{model}_{state}{'_replay' if replay else ''}.json"
    write_json(filename, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("context", "check"))
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--state", choices=STATES, default="healthy_A3h")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "e2e/dstu_supplement")
    args = parser.parse_args()
    if args.command == "context":
        print(json.dumps(context(args.model, args.state), indent=2))
    else:
        value = check(args.model, args.state, args.out, args.replay)
        print(json.dumps({key: value[key] for key in ("status", "model", "state", "same_canonical_graph",
              "actual_order_comparison", "historical_baseline_reusable", "new_backend_calls", "wall_seconds", "peak_rss_gib")}, indent=2))


if __name__ == "__main__":
    main()
