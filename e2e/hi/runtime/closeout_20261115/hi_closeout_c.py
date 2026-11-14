"""Two H/I MF6 mechanism points over the frozen EIR and Split implementations.

``plan`` first writes six anchor/structural-selection jobs. Calling it after
those jobs finish appends the selected operators' EIR and m1/m2 evaluations.
All jobs use the closeout shared pool; C's selection/raw files have their own
subdirectory so PF4 and MF6 preparation cannot overwrite one another.
"""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import statistics
import time

import fg_fault as frozen
import hi_config
import hi_fault
import hi_manifest
import post_fg_split as split

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "e2e/hi/closeout"
OUT = ROOT / "e2e/hi/out/closeout_20261115"
INPUTS = DEST / "manifests/c_inputs.json"
MANIFEST = DEST / "manifests/c_mf6.json"
VERSION = "hi_closeout_c_v1"
OPERATORS = ("pf_ffn", "pf_proj")


def private_out(out):
    return Path(out) / "c_mf6"


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def index_inputs():
    with (ROOT / "e2e/hi/tables/h3_i2_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        candidates = {r["candidate_id"]: r for r in csv.DictReader(stream)}
    master = {s["candidate_id"]: s for s in hi_manifest.candidates()}
    index = {}
    for name, spec in master.items():
        if spec["hardware_profile"] != "busybarn" or spec["method"] != "A3h":
            continue
        if (spec["phase"], spec["op"]) not in (("decode", "mha"), ("prefill", "ffn"), ("prefill", "proj")):
            continue
        record = ROOT / candidates[name]["source"].replace("\\", "/")
        base = record.parent.parent
        files = dict(record=record, raw=base / "raw" / (name + ".pkl.gz"),
                     snapshot=base / "snapshots" / (name + ".pkl.gz"))
        saved = frozen.read_json(record)
        assert saved["status"] == "ok" and all(saved[k] == v for k, v in spec.items())
        index[name] = {k: rel(p) for k, p in files.items()}
        index[name]["sha256"] = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}
    assert len(index) == 84
    frozen.write_json(INPUTS, dict(count=len(index), candidates=index, protocol=VERSION))
    return index


@contextmanager
def healthy_routes():
    index = frozen.read_json(INPUTS)["candidates"]
    old_result, old_raw, old_choose = frozen.result_path, frozen.raw_path, frozen.choose_candidate
    def result_path(out, name):
        return ROOT / index[name]["record"] if name in index else old_result(out, name)
    def raw_path(out, name):
        return ROOT / index[name]["raw"] if name in index else old_raw(out, name)
    def choose(rows, out):
        chosen = old_choose(rows, out)
        source = index[chosen["candidate_id"]]
        chosen.update({k: source[k] for k in ("record", "raw", "snapshot")})
        return chosen
    frozen.result_path, frozen.raw_path, frozen.choose_candidate = result_path, raw_path, choose
    try:
        yield
    finally:
        frozen.result_path, frozen.raw_path, frozen.choose_candidate = old_result, old_raw, old_choose


def representative(spec, out):
    """Choose by workload and legal tensor structure, before any repair result."""
    faults = frozen.read_json(out / "faults" / (spec["config_id"] + ".json"))
    dead = tuple(faults["faults"]["MF6"]["failed_pes"][0])
    selection = frozen.read_json(out / "selection" / (spec["config_id"] + ".json"))
    inspections = []
    for op in OPERATORS:
        source = selection["operators"][op]["A3h"]
        _, _, snapshot, _, _ = frozen.load_healthy(frozen.op_spec(spec, op), source, out)
        behaviours, data, events, hp = snapshot
        orphan = [tag for tag, e in events.items() if e.event_type == "computation"
                  and tuple(e.comp_location[:2]) == dead]
        tensor = [tag for tag in orphan if events[tag].comp_device == "tensorcore"]
        terminal = [tag for tag in tensor if any(e.event_type == "communication"
                    and tag in e.dependency_set and not e.issue_set for e in events.values())]
        axes = []
        for axis in ("N", "M"):
            divisible = []
            for tag in tensor:
                b = behaviours[tag]
                dt = b.beha_datatag[1 if axis == "N" else 0]
                a, z = split.bounds(b.needed_data_split_dict[dt])[-1 if axis == "N" else 0]
                divisible.append((z - a + 1) % 2 == 0)
            legal = bool(tensor) and all(divisible) and not terminal
            axes.append(dict(axis=axis, legal=legal, all_orphan_tensor_dimensions_even=all(divisible),
                             terminal_tensor_broadcasts=len(terminal)))
            if legal:
                for tag in tensor:
                    split.pieces_for(tag, events[tag], behaviours[tag], 2, axis)
        legal_axes = [a["axis"] for a in axes if a["legal"]]
        inspections.append(dict(phase_op=op, source_candidate=source["candidate_id"],
            orphan_count=len(orphan), orphan_tensor_count=len(tensor),
            orphan_tensor_work_ns=sum(split.service(events[t], hp) for t in tensor),
            orphan_tensor_flops=sum(split.tensor_work(events[t].comp_datashape)[0] for t in tensor),
            terminal_tensor_broadcast_tags=[list(t) for t in terminal], axes=axes, legal_axes=legal_axes))
    # A predetermined operator order, not a choice made from observed speedups.
    eligible = [r for r in inspections if r["legal_axes"]]
    chosen = eligible[0] if eligible else None
    result = dict(model=spec["model"], config_id=spec["config_id"], failed_pe=list(dead),
        anchor=faults["anchor"], inspections=inspections, phase_op=chosen["phase_op"] if chosen else None,
        legal_axes=chosen["legal_axes"] if chosen else [], applicable=bool(chosen),
        rule="fixed preference pf_ffn then pf_proj among operators with orphan tensor work and legal m2 output axes; no repair/backend comparison",
        reason="legal orphan Tensor blocks" if chosen else "Neither candidate has splittable orphan Tensor blocks without partitioning terminal broadcasts",
        repair_evaluations_used_for_selection=0)
    frozen.write_json(out / "representatives" / (spec["model"] + ".json"), result)
    return result


def reused_pre(spec, out):
    """Reuse a same-candidate completed healthy replay, including its graph order."""
    source = frozen.selected_source(spec, out)
    folders = (ROOT / "e2e/hi/out/h4_pf4", ROOT / "e2e/hi/out/i4_light")
    for folder in folders:
        for path in (folder / "results").glob("*__pre_" + spec["phase_op"] + "_A3h.json"):
            r = frozen.read_json(path)
            if r["spec"]["config_id"] != spec["config_id"] or r["source_candidate"] != source["candidate_id"]:
                continue
            assert r["status"] == "ok" and r["spec"]["backend_seeds"] == spec["backend_seeds"]
            target = frozen.raw_path(out, spec["job_id"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / r["sink"], target)
            result = dict(r, job_id=spec["job_id"], spec=spec, sink=rel(target),
                          reused_healthy_record=rel(path), actual_backend_calls=0,
                          prior_wall_seconds=r["wall_seconds"], wall_seconds=0.)
            frozen.write_json(frozen.result_path(out, spec["job_id"]), result)
            return True
    return False


def load_split_state(spec, out):
    source = frozen.selected_source(spec, out)
    blob, _ = split.static.read_pickle(ROOT / source["snapshot"])
    behaviours, data, _ = pickle.loads(blob)
    pre = frozen.read_json(frozen.result_path(out, spec["pre_job"]))
    exported = split.static.read_pickle(ROOT / pre["exports"]["healthy"])
    events = pickle.loads(exported["events_blob"])
    dead = tuple(frozen.read_json(out / "faults" / (spec["config_id"] + ".json"))["faults"]["MF6"]["failed_pes"][0])
    assert all(tuple(events[tag].comp_location) == tuple(b.location) for tag, b in behaviours.items())
    return dict(case="C_" + spec["model"], config_id=spec["config_id"], op=spec["phase_op"],
        source_candidate=source["candidate_id"], snapshot=source["snapshot"],
        healthy_export=pre["exports"]["healthy"], spec=pre["spec"], dead=dead,
        behaviours=behaviours, data=data, events=events, T_pre=pre["T"], backend_pre=pre["backend_mean"])


def prepare_split(spec, out):
    split.setup()
    state = load_split_state(spec, out)
    healthy_hp = split.platform(state["spec"])
    hp = split.platform(state["spec"], state["dead"])
    assert set(healthy_hp.links_dict) == set(hp.links_dict)
    _, identity = split.adapt(state, healthy_hp, 1, "N", identity=True)
    baseline = split.event_metrics(state["events"], healthy_hp)
    representative_row = frozen.read_json(out / "representatives" / (spec["model"] + ".json"))
    choices, graphs = [], {}
    for factor in (1, 2):
        options = []
        axes = ["N"] if factor == 1 else representative_row["legal_axes"]
        for axis in axes:
            graph, info = split.adapt(state, hp, factor, axis)
            rank = (info["metrics"]["tensor_rounded_flops"] - baseline["tensor_rounded_flops"],
                    info["metrics"]["communication"]["route_link_bytes"], 0 if axis == "N" else 1)
            options.append((rank, axis, graph, info))
        rank, axis, graph, info = min(options, key=lambda r: r[0])
        for score, candidate_axis, _, candidate_info in options:
            choices.append(dict(factor=factor, axis=candidate_axis, selected=axis == candidate_axis,
                rounded_flop_delta=score[0], route_link_bytes=score[1], build_seconds=candidate_info["build_seconds"]))
        info["axis_selection_seconds"] = sum(row[3]["build_seconds"] for row in options)
        info["baseline_metrics"] = baseline
        rigid_events = dict(state["events"])
        for tag, event in state["events"].items():
            if event.event_type == "computation" and event.comp_device == "tensorcore" and tuple(event.comp_location[:2]) == state["dead"]:
                rigid_events[tag] = copy.copy(event)
                rigid_events[tag].comp_location = next(iter(hp.tensorcore_dict))
        info["rigid_tensor_load_bound"] = split.tensor_bounds(rigid_events, hp)
        times = [split.service(e, healthy_hp) for e in state["events"].values()
                 if e.event_type == "computation" and e.comp_device == "tensorcore"]
        info["rigid_block_time_distribution_ns"] = dict(min=min(times), max=max(times),
            mean=statistics.mean(times), counts=dict(Counter(map(str, times))))
        filename = out / "graphs" / f"C_{spec['model']}_m{factor}.pkl.gz"
        payload = dict(events_blob=pickle.dumps(graph, pickle.HIGHEST_PROTOCOL), info=info,
            source_spec=state["spec"], source_candidate=state["source_candidate"],
            source_export=state["healthy_export"], dead=state["dead"])
        split.write_pickle(filename, payload)
        frozen.write_json(out / "metadata" / f"C_{spec['model']}_m{factor}.json", info)
        graphs[str(factor)] = dict(graph=rel(filename), graph_sha256=split.graph_digest(graph), axis=axis,
                                  metrics=info["metrics"], decision_seconds=info["axis_selection_seconds"])
    return dict(model=spec["model"], phase_op=spec["phase_op"], graphs=graphs, choices=choices,
        identity_event_count=identity["DAG_events"], identity_exact=True, actual_backend_calls=0,
        axis_rule="minimum extra rounded FLOPs, then route-link bytes, then N before M",
        source_candidate=state["source_candidate"], source_export=state["healthy_export"])


def child(spec, out):
    out = Path(out).resolve()
    local = private_out(out)
    started = time.perf_counter()
    kind = spec["kind"]
    with healthy_routes(), hi_fault.model_adapter(spec, local):
        if kind in hi_fault.KINDS:
            if not (kind == "fault_pre" and reused_pre(spec, local)):
                frozen.child(spec, local)
            result = frozen.read_json(frozen.result_path(local, spec["job_id"]))
        elif kind == "c_representative":
            result = representative(spec, local)
        elif kind == "c_split_prepare":
            result = prepare_split(spec, local)
        elif kind == "c_split_eval":
            prepared = frozen.read_json(frozen.result_path(local, spec["prepare_job"]))
            graph = prepared["graphs"][str(spec["factor"])]
            actual = dict(spec, **{k: graph[k] for k in ("graph", "graph_sha256", "axis")})
            split.child(actual, local)
            result = frozen.read_json(frozen.result_path(local, spec["job_id"]))
            result["actual_backend_calls"] = 1
        else:
            raise ValueError(kind)
    if kind not in hi_fault.KINDS:
        result.update(job_id=spec["job_id"], kind=kind, status="ok", spec=spec,
                      wall_seconds=time.perf_counter() - started)
        import e2e_pilot_peak
        result["peak_rss_gb"] = e2e_pilot_peak.peak() / 2**30
        frozen.write_json(frozen.result_path(local, spec["job_id"]), result)
    frozen.write_json(out / "results" / (spec["job_id"] + ".json"), result)
    print(json.dumps(dict(job=spec["job_id"], status=result["status"], seconds=result["wall_seconds"])), flush=True)


def jobs(out=OUT):
    out = Path(out).resolve()
    local = private_out(out)
    index = index_inputs()
    result = []
    for model in hi_config.MODELS:
        candidates = [s for s in hi_manifest.candidates() if s["candidate_id"] in index and s["model"] == model]
        base = {k: candidates[0][k] for k in ("model", "hardware_profile", "shape", "split_mode", "d2d_scale", "config_id",
            "model_contract_sha256", "model_contract_version", "graph_order_version", "workload_version")}
        base.update(module="hi_closeout_c", protocol_version=VERSION, search_steps=1000,
                    backend_seeds=list(range(1, 11)), start_arm="A3h")
        def add(suffix, kind, deps, seconds=1., rss=.6, **fields):
            name = "C_" + model + "__" + suffix
            spec = dict(base, job_id=name, kind=kind, depends_on=deps, **fields)
            result.append(dict(id=name, kind=kind, spec=spec, result="results/" + name + ".json", deps=deps,
                argv=["{python}", "-u", "-B", "e2e/hi_closeout_c.py", "child", "--spec", "{spec}", "--out", "{out}"],
                estimated_seconds=seconds, peak_rss_gb=rss, priority=20 if kind == "fault_select" else 1))
            return name
        selection = add("selection", "fault_selection", [], candidate_ids=[s["candidate_id"] for s in candidates])
        anchor = add("anchor", "fault_select", [selection], seconds=40, rss=2.)
        representative_id = add("representative", "c_representative", [anchor], seconds=90, rss=3.)
        selected_path = local / "representatives" / (model + ".json")
        if not selected_path.exists():
            continue
        selected = frozen.read_json(selected_path)
        if not selected["applicable"]:
            continue
        phase_op = selected["phase_op"]
        op_fields = dict(frozen.op_spec({}, phase_op), phase_op=phase_op)
        records = [frozen.read_json(ROOT / index[s["candidate_id"]]["record"]) for s in candidates
                   if s["phase"] == op_fields["phase"] and s["op"] == op_fields["op"]]
        winner = min(records, key=lambda r: (statistics.mean(v["cycles"] for v in r["backend"]), r["candidate_order"]))
        backend = sum(winner["timing"]["evaluation"]["backend_individual_seconds"])
        search = winner["search_seconds"]
        rss = max(1., winner["peak_rss_gb"] * 2)
        pre = add("pre", "fault_pre", [representative_id], seconds=backend + 20, rss=rss, **op_fields)
        l1 = add("l1", "fault_l1", [pre], seconds=2 * backend + 30, rss=rss, **op_fields,
                 fault_id="MF6", pre_sink=rel(frozen.raw_path(local, pre)))
        stop = add("stop0", "fault_stop", [l1], **op_fields, fault_id="MF6", steps=0, l1_job=l1)
        for steps in frozen.STEPS:
            reps = [add(f"rep_s{steps}_r{seed}", "fault_rep", [stop], seconds=backend + search * steps / 1000 + 30,
                        rss=rss, **op_fields, fault_id="MF6", steps=steps, seed=seed, previous_stop=stop,
                        pre_sink=rel(frozen.raw_path(local, pre)), l1_sink=rel(frozen.raw_path(local, l1))) for seed in frozen.SEEDS]
            stop = add(f"stop{steps}", "fault_stop", reps, **op_fields, fault_id="MF6", steps=steps,
                       previous_stop=stop, rep_jobs=reps)
        prepared = add("split_prepare", "c_split_prepare", [pre], seconds=90, rss=rss, **op_fields, pre_job=pre)
        for factor in (1, 2):
            for seed in range(1, 11):
                add(f"m{factor}_s{seed}", "c_split_eval", [prepared], seconds=backend / 10 + 5, rss=rss,
                    **op_fields, factor=factor, seed=seed, prepare_job=prepared, method="Local-m1" if factor == 1 else "Split-m2")
    assert len({j["id"] for j in result}) == len(result)
    assert {d for j in result for d in j["deps"]} <= {j["id"] for j in result}
    return result


def plan(out=OUT):
    rows = jobs(out)
    previous = frozen.read_json(MANIFEST)["jobs"] if MANIFEST.exists() else []
    current = {r["id"]: r for r in rows}
    for old in previous:
        assert current[old["id"]] == old, old["id"]
    record = dict(status="authorized", protocol=VERSION, jobs=rows, counts=dict(Counter(j["kind"] for j in rows)),
        out=rel(out), private_out=rel(private_out(out)), healthy_inputs=84,
        maximum_new_backend_calls=402, maximum_L2_trajectories=30, fixed_fault_models=2,
        next="Run anchors/representatives, then call plan again to append only eligible point jobs")
    frozen.write_json(MANIFEST, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "child"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    if args.command == "child":
        child(frozen.read_json(args.spec), args.out)
    else:
        r = plan(args.out)
        print(json.dumps({k: v for k, v in r.items() if k != "jobs"}, indent=2))
