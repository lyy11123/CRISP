"""H4/I4 entry points over frozen EIR, with model-specific inputs and PF4/MF6.

The first H/I batch only checks this adapter. Fault execution requires the
completed fourteen-candidate groups and an explicit later-batch job manifest.
"""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import collections
from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
import time

import fg_fault as frozen
import hi_config
import hi_manifest

ROOT = Path(__file__).resolve().parents[1]
VERSION = "hi_fault_v1"
KINDS = ("fault_selection", "fault_select", "fault_pre", "fault_l1", "fault_rep", "fault_stop")


def matching_candidates(spec, phase="decode", op="mha", method="A3h"):
    rows = [row for row in hi_manifest.candidates() if row["model"] == spec["model"]
            and row["hardware_profile"] == spec["hardware_profile"] and row["phase"] == phase
            and row["op"] == op and row["method"] == method]
    assert len(rows) == 14
    assert all(row["config_id"] == hi_config.identity(spec) for row in rows)
    return rows


def fault_anchor_candidates(spec):
    return [row["candidate_id"] for row in matching_candidates(spec)]


def select_candidates(spec, out):
    expected = {row["candidate_id"]: row for row in hi_manifest.candidates()}
    for candidate in spec["candidate_ids"]:
        record = frozen.read_json(frozen.result_path(out, candidate))
        identity = expected[candidate]
        assert record["status"] == "ok"
        for key in ("model", "hardware_profile", "phase", "op", "method", "model_contract_sha256",
                    "model_contract_version", "graph_order_version", "workload_version"):
            assert record[key] == identity[key], (candidate, key)
        assert record["config_id"] == spec["config_id"]
        assert [sample["seed"] for sample in record["backend"]] == list(range(1, 11))
        assert len(record["T_hat_rand"]) == 10
    return _SELECT_CANDIDATES(spec, out)


_SELECT_CANDIDATES = frozen.select_candidates


def physical_faults(hp, busy, anchor_candidate):
    noncorner = sorted({tuple(location[:-1]) for location in hp.tensorcore_dict
                        if location[1] not in (0, 3, 12, 15)})
    node = min(noncorner, key=lambda candidate: (-busy.get(candidate, 0), candidate))
    faults = {
        "PF4": dict(failed_nodes=[], failed_links=[[[0, 11], [1, 8]], [[1, 8], [0, 11]]], failed_pes=[],
                    kind="fixed bidirectional D2D connection", rule="die0/core11 <-> die1/core8; shared by all P/D operators"),
        "MF6": dict(failed_nodes=[], failed_links=[], failed_pes=[list(node)], kind="PE only; router retained",
                    rule="A3h decode MHA backend-seed1 tensor+vector busy time; noncorner maximum; tie smallest die/core",
                    anchor_compute_busy_ns=busy.get(node, 0)),
    }
    totals = collections.Counter()
    for edge, link in hp.links_dict.items():
        totals[link.link_type] += link.bandwidth
    for label, fault in faults.items():
        removed = collections.Counter()
        for a, b in fault["failed_links"]:
            link = hp.links_dict[(tuple(a), tuple(b))]
            removed[link.link_type] += link.bandwidth
        fault.update(anchor_operator="dc_mha", anchor_candidate=anchor_candidate, backend_seed=1,
                     failed_compute_group_fraction=1 / len(hp.tensorcore_dict) if label == "MF6" else 0,
                     failed_compute_die_fraction=1 / 16 if label == "MF6" else 0,
                     failed_D2D_pair_fraction=0.5 if label == "PF4" else 0,
                     failed_bandwidth_fraction_by_type={kind: removed[kind] / total for kind, total in totals.items()})
    return faults


def select_faults(spec, out):
    source = frozen.fault_anchor_source(spec, out)
    _, c, snapshot, _, _ = frozen.load_healthy(frozen.op_spec(spec, "dc_mha"), source, out)
    import numpy as np
    import event_driver
    events, hp = copy.deepcopy(snapshot[2:])
    before = c.e3._e2()._snapshot_pre(events)
    random.seed(1)
    np.random.seed(1)
    triple = event_driver.event_driver(events_dict=events, hardware_platform=hp)
    captured = c.e3._e2()._extract(events, hp, before, triple, "seed1")
    busy = collections.Counter()
    for (_, _, location), record in captured["devices"].items():
        busy[tuple(location[:-1])] += sum(end - start for start, end, _ in record)
    faults = physical_faults(hp, busy, source["candidate_id"])
    result = dict(config_id=spec["config_id"], model=spec["model"], faults=faults, anchor=source,
                  anchor_backend=triple, model_contract_sha256=spec["model_contract_sha256"],
                  anchor_candidates=fault_anchor_candidates(spec), anchor_backend_calls=1,
                  fault_selection_version=VERSION)
    frozen.write_json(Path(out) / "faults" / (spec["config_id"] + ".json"), result)
    return result


@contextmanager
def model_adapter(spec, out):
    """Replace entry bindings only; the search, proposal and stop code stay intact."""
    import f_config
    setup = dict(spec, phase=spec.get("phase", "decode"), op=spec.get("op", "mha"), method="A1", g=0, x=0)
    _, c, _ = hi_config.prepare(setup, out)
    sys.path.insert(0, str(ROOT / "eir"))
    import eir
    import post_fg_bb as bb
    assert c.EXACT_REVERT and c.BROADCAST_T and c.BC_CACHE
    assert frozen.SEEDS == (123, 1, 2, 3, 4) and frozen.STEPS == (30, 100, 300)
    old = []
    def replace(module, name, value):
        old.append((module, name, getattr(module, name)))
        setattr(module, name, value)
    original_child, original_finalize = eir.child, eir.Engine.finalize

    def child_without_debug(inner):
        return original_child(dict(inner, checks=False))

    def original_finalization(engine, seeds=None):
        active = c.cpcm.simulate
        c.cpcm.simulate = bb.ORIGINAL
        try:
            return original_finalize(engine, seeds)
        finally:
            c.cpcm.simulate = active

    replace(f_config, "prepare", hi_config.prepare)
    replace(f_config, "task_path", hi_config.task_path)
    replace(frozen, "fault_anchor_candidates", fault_anchor_candidates)
    replace(frozen, "select_candidates", select_candidates)
    replace(frozen, "select_faults", select_faults)
    replace(eir, "child", child_without_debug)
    replace(eir.Engine, "finalize", original_finalization)
    replace(c.cpcm, "simulate", bb.simulate)
    # fg_fault.engine_job updates these frozen module bindings in its process.
    legacy = [(eir, name, getattr(eir, name)) for name in
              ("child_pre", "child_l1", "child_rep", "write_sink", "start_mapping", "fault_of", "GPT", "WORK", "BACKEND_SEEDS")]
    legacy += [(eir.Engine, "seed_broadcast", eir.Engine.seed_broadcast), (c, "WORK", c.WORK),
               (c.e3, "WORK", c.e3.WORK), (eir, "TASKS", dict(eir.TASKS))]
    try:
        yield dict(search_simulate="post_fg_bb.simulate", final_simulate="original cpcm.simulate",
                   debug_checks=False, exact_rollback=True, backend_seeds=list(range(1, 11)),
                   random_tie_behavior="unchanged original implementation when rng is provided",
                   stop_rule_sha256=hashlib.sha256(inspect.getsource(frozen.stop_job).encode()).hexdigest())
    finally:
        for module, name, value in reversed(legacy):
            setattr(module, name, value)
        for module, name, value in reversed(old):
            setattr(module, name, value)


def child(spec, out):
    assert spec["kind"] in KINDS
    assert spec["model_contract_sha256"] == hi_config.digest(hi_config.model_contract(spec["model"]))
    assert spec["config_id"] == hi_config.identity(spec)
    assert spec.get("fault_id") in (None, "PF4", "MF6")
    assert spec["backend_seeds"] == list(range(1, 11))
    if spec["kind"] == "fault_rep":
        assert spec["steps"] in frozen.STEPS and spec["seed"] in frozen.SEEDS
        assert spec["start_arm"] == "A3h"
    with model_adapter(spec, out):
        frozen.child(spec, out)


def jobs(model, hardware, out, mf6_operator):
    """Build the later H4/I4 list; generating this list does not run any job."""
    assert model in hi_config.MODELS and hardware in ("busybarn", "watos")
    assert mf6_operator in ("pf_ffn", "pf_proj")
    candidates = [row for row in hi_manifest.candidates() if row["model"] == model and row["hardware_profile"] == hardware]
    first = candidates[0]
    inherited = ("model", "hardware_profile", "shape", "split_mode", "d2d_scale", "config_id",
                 "model_contract_sha256", "model_contract_version", "graph_order_version", "workload_version")
    base = {key: first[key] for key in inherited}
    base.update(module="hi_fault", search_steps=1000, backend_seeds=list(range(1, 11)),
                protocol_version=VERSION, launch_status="planned_not_authorized_in_first_batch")
    result = []
    def add(suffix, kind, deps, **fields):
        name = first["config_id"] + "__" + suffix
        result.append(dict(base, job_id=name, kind=kind, depends_on=deps, **fields))
        return name
    selected = [row["candidate_id"] for row in candidates if row["method"] == "A3h"
                or (hardware == "busybarn" and row["phase"] == "decode" and row["method"] == "A1")]
    selection = add("selection", "fault_selection", selected, candidate_ids=selected)
    fault_selection = add("fault_select", "fault_select", [selection])
    phase_ops = frozen.OPS if hardware == "busybarn" else ("pf_mha", "pf_ffn", "dc_mha", "dc_ffn")
    for phase_op in phase_ops:
        arms = ("A1", "A3h") if hardware == "busybarn" and phase_op.startswith("dc_") else ("A3h",)
        for arm in arms:
            pre = add(f"pre_{phase_op}_{arm}", "fault_pre", [selection], **frozen.op_spec({}, phase_op),
                      phase_op=phase_op, start_arm=arm)
            faults = ("PF4", "MF6") if hardware == "busybarn" and phase_op == mf6_operator else ("PF4",)
            for fault in faults:
                l1 = add(f"l1_{phase_op}_{arm}_{fault}", "fault_l1", [pre, fault_selection],
                         **frozen.op_spec({}, phase_op), phase_op=phase_op, start_arm=arm, fault_id=fault,
                         pre_sink=frozen.relative(frozen.raw_path(out, pre)))
                if arm == "A1":
                    continue
                stop = add(f"stop_{phase_op}_{fault}_0", "fault_stop", [l1], phase_op=phase_op,
                           fault_id=fault, steps=0, l1_job=l1)
                for steps in frozen.STEPS:
                    reps = [add(f"rep_{phase_op}_{fault}_s{steps}_r{seed}", "fault_rep", [stop],
                                **frozen.op_spec({}, phase_op), phase_op=phase_op, start_arm=arm,
                                fault_id=fault, steps=steps, seed=seed, previous_stop=stop,
                                pre_sink=frozen.relative(frozen.raw_path(out, pre)),
                                l1_sink=frozen.relative(frozen.raw_path(out, l1))) for seed in frozen.SEEDS]
                    stop = add(f"stop_{phase_op}_{fault}_{steps}", "fault_stop", reps, phase_op=phase_op,
                               fault_id=fault, steps=steps, previous_stop=stop, rep_jobs=reps)
    return result


def check(out):
    """Exercise candidate identities, physical PE/link rules and frozen stop decisions."""
    import f_config
    import post_fg_split
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    identities, counts = [], []
    for model in hi_config.MODELS:
        for hardware in ("busybarn", "watos"):
            spec = dict(model=model, hardware_profile=hardware, shape="2x2", split_mode="fixed", d2d_scale=1)
            names = fault_anchor_candidates(spec)
            identities.append(dict(model=model, hardware=hardware, n=len(names), candidates=names))
            planned = jobs(model, hardware, out, "pf_ffn")
            counts.append(dict(model=model, hardware=hardware,
                a3h_fault_units=sum(row["kind"] == "fault_l1" and row["start_arm"] == "A3h" for row in planned),
                a1_r0_units=sum(row["kind"] == "fault_l1" and row["start_arm"] == "A1" for row in planned),
                maximum_l2_jobs=sum(row["kind"] == "fault_rep" for row in planned)))
    hp = post_fg_split.platform(dict(hardware_profile="busybarn", shape="2x2", split_mode="fixed", d2d_scale=1))
    faults = physical_faults(hp, {(0, 5): 10, (0, 6): 10, (0, 0): 100}, "fixture_anchor")
    assert faults["MF6"]["failed_pes"] == [[0, 5]]
    assert set(faults) == {"PF4", "MF6"}
    damaged = post_fg_split.platform(dict(hardware_profile="busybarn", shape="2x2", split_mode="fixed", d2d_scale=1), (0, 5))
    assert set(hp.links_dict) == set(damaged.links_dict)
    for kind in ("tensorcore", "vectorunit"):
        assert not any(tuple(location[:2]) == (0, 5) for location in damaged.modules_dict[kind])
    f_config_prepare = f_config.prepare
    spec = dict(model="llama", hardware_profile="busybarn", shape="2x2", split_mode="fixed", d2d_scale=1,
                phase="decode", op="ffn")
    with model_adapter(spec, out) as adapter:
        assert f_config.prepare is hi_config.prepare and f_config.task_path is hi_config.task_path
        assert frozen.fault_anchor_candidates(spec) == fault_anchor_candidates(spec)
        import cpcm
        import post_fg_bb as bb
        graph = cpcm.build([dict(tag="one", kind="comp", deps=[], res="cpu", dur=7)])
        assert cpcm.simulate is bb.simulate
        assert cpcm.simulate(graph).T == 7
        assert cpcm.simulate(graph, rng=random.Random(1)).T == bb.ORIGINAL(graph, rng=random.Random(1)).T
    assert f_config.prepare is f_config_prepare
    assert sum(r["a3h_fault_units"] for r in counts) == 26
    assert sum(r["maximum_l2_jobs"] for r in counts) == 390
    fixture = out / "stop_fixture"
    frozen.write_json(frozen.result_path(fixture, "l1"), dict(T_pre=100, T_R0=120, T_L1=120,
                      exports=dict(R0="fixture_R0", L1="fixture_L1")))
    zero_gain = frozen.stop_job(dict(steps=0, l1_job="l1"), fixture)
    assert not zero_gain["stop"] and zero_gain["T"] == 120
    frozen.write_json(frozen.result_path(fixture, "stop0"), zero_gain)
    stop_checks = [dict(case="L1 zero gain continues", result=zero_gain)]
    previous = "stop0"
    for steps, value, expected_stop in ((30, 110, False), (100, 108.5, True)):
        names = []
        for index, seed in enumerate(frozen.SEEDS):
            name = f"r{steps}_{seed}"
            names.append(name)
            frozen.write_json(frozen.result_path(fixture, name), dict(job_id=name, T=value, seed=seed,
                              backend_mean=10+index, exports=dict(final=f"fixture_{name}")))
        result = frozen.stop_job(dict(steps=steps, previous_stop=previous, rep_jobs=names), fixture)
        assert result["stop"] == expected_stop and result["seed"] == 2
        previous = f"stop{steps}"
        frozen.write_json(frozen.result_path(fixture, previous), result)
        stop_checks.append(dict(case=f"{steps}-step mean and backend-median representative", result=result))
    evidence = dict(status="passed", verification_scope="adapter/static/fixture only; no new model fault simulation",
        candidates=identities, counts=counts, physical_faults= faults, adapter=adapter,
        pe_retains_all_links=True, excluded_corner_despite_larger_busy=True,
        deterministic_small_graph_and_random_tie_dispatch=True, adapter_bindings_restored=True,
        stop_fixture_checks=stop_checks,
        pending=["complete 14-candidate groups and selected-state reload", "seed1 anchor backend on actual selected new-model MHA",
                 "healthy T equality in pre", "PF4 R0/L1 and MF6 chosen prefill operator", "30/100/300 five-seed repairs and stop exports"],
        fault_backend_calls=0, repair_searches=0)
    frozen.write_json(out / "fault_adapter_checks.json", evidence)
    print(json.dumps(dict(status="passed", a3h_fault_units=26, maximum_l2_jobs=390, fault_backend_calls=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("child", "check", "manifest"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "e2e/hi/out/fault_adapter")
    parser.add_argument("--model", choices=tuple(hi_config.MODELS))
    parser.add_argument("--hardware", choices=("busybarn", "watos"), default="busybarn")
    parser.add_argument("--mf6-operator", choices=("pf_ffn", "pf_proj"))
    args = parser.parse_args()
    if args.command == "child":
        child(frozen.read_json(args.spec), args.out)
    elif args.command == "check":
        check(args.out)
    else:
        rows = jobs(args.model, args.hardware, args.out, args.mf6_operator)
        selected = set(rows[0]["candidate_ids"])
        inputs = [row for row in hi_manifest.candidates() if row["candidate_id"] in selected]
        frozen.write_json(args.out / "fault_jobs_planned.json", dict(status="planned_only", jobs=inputs+rows,
                          healthy_inputs_included_for_scheduler_identity=True,
                          note="Run only in a later authorized batch; reuse matching successful health receipts in the same output root."))
