"""GPT S closeout: A1 EIR and the eight fixed mapping/repair/transport arms.

The healthy A1 reference is shared by all arms. Repair stopping uses the
corresponding A1 operator's healthy CPCM estimate. Old source modules are bound
only inside a job process; completed F/G and post-F/G results remain inputs.
"""
import argparse
import collections
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import fg_fault as fault
import fg_joint as joint

ROOT = fault.ROOT
OLD = ROOT / "e2e/fg/out/main"
POST = ROOT / "e2e/post_fg/out/main"
OUT = ROOT / "e2e/hi/out/closeout_20261115"
HERE = ROOT / "e2e/hi/closeout"
VERSION = "hi_closeout_s_v1"
READ = fault.read_json
WRITE = fault.write_json
REPAIR_SEEDS = (123, 1, 2, 3, 4)
PHASES = (.1, .5, .9)
STATES = ("A3hEIR", "A3hR0", "A1R0", "A1EIR")


def digest(filename):
    return hashlib.sha256(fault.path(filename).read_bytes()).hexdigest()


def receipt(out, name):
    return READ(fault.result_path(out, name))


def base_spec(hardware):
    cid = f"F_{hardware}_2x2_fixed_d2d1"
    return dict(stage="F", hardware_profile=hardware, shape="2x2", split_mode="fixed", d2d_scale=1.,
                config_id=cid, source_config_id=cid, module="hi_closeout_s", protocol_version=VERSION,
                workload_version="W1", backend_seeds=list(range(1, 11)))


def as_job(spec):
    kind = spec["kind"]
    estimates = {"s_l1": 90, "s_rep": 150, "s_stop": 1, "s_graph": 90,
                 "s_solo": 50, "s_joint": 50, "s_joint_reuse": 2}
    return dict(id=spec["job_id"], kind=kind, spec=spec, result=f"results/{spec['job_id']}.json",
                deps=spec["depends_on"], argv=["{python}", "-u", "-B", "e2e/hi_closeout_s.py", "child",
                "--spec", "{spec}", "--out", "{out}"], estimated_seconds=estimates[kind], peak_rss_gb=2., priority=2)


def repair_specs(out=OUT):
    result = []
    for hw in ("wsc", "face"):
        base = base_spec(hw)
        cid = base["config_id"]
        selection = READ(OLD / "selection" / (cid + ".json"))
        for op in joint.DC:
            prefix = f"S_{hw}_{op}_A1"
            source = selection["operators"][op]["A1"]
            pre = READ(OLD / "results" / (cid + f"__pre_{op}_A1.json"))
            fields = dict(base, **fault.op_spec({}, op), phase_op=op, start_arm="A1", fault_id="PF4",
                          healthy_source=source, pre_sink=pre["sink"], healthy_T=pre["T"],
                          fault_file=fault.relative(OLD / "faults" / (cid + ".json")),
                          input_sha256={value: digest(value) for value in
                                        [source[k] for k in ("record", "raw", "snapshot")] +
                                        [pre["sink"], fault.relative(OLD / "faults" / (cid + ".json"))]})
            def add(suffix, kind, deps, **kw):
                name = prefix + "__" + suffix
                result.append(dict(fields, job_id=name, kind=kind, depends_on=deps, **kw))
                return name
            l1 = add("l1", "s_l1", [])
            stop = add("stop0", "s_stop", [l1], steps=0, l1_job=l1)
            for steps in (30, 100, 300):
                reps = [add(f"s{steps}_r{seed}", "s_rep", [stop], steps=steps, seed=seed,
                            previous_stop=stop, l1_sink=fault.relative(fault.raw_path(out, l1)))
                        for seed in REPAIR_SEEDS]
                stop = add(f"stop{steps}", "s_stop", reps, steps=steps, previous_stop=stop, rep_jobs=reps)
    assert len(result) == 160
    return result


def a1_engine(spec, out):
    """Run the original complete L1/L2 entry, including the previously omitted A1 L1."""
    import f_config
    for filename, expected in spec["input_sha256"].items():
        assert digest(filename) == expected, filename
    source = spec["healthy_source"]
    source_record = READ(source["record"])
    setup = dict(source_record, method="A1", x=0)
    _, c, cfg = f_config.prepare(setup, out)
    if setup["op"] == "mha":
        import add_communication
        wrapper = add_communication.add_broadcast
        cells = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__))
        add_communication.add_broadcast = cells["orig"].cell_contents
    work = c.WORK
    sys.path.insert(0, str(ROOT / "eir"))
    import eir
    import post_fg_bb as bb
    c.WORK = eir.WORK = eir.e3.WORK = work
    eir.GPT = c.CONFIGS[cfg][0]
    eir.TASKS[spec["phase_op"]] = f_config.task_path(setup, c, cfg).stem
    eir.BACKEND_SEEDS = c.BACKEND_SEEDS = spec["backend_seeds"]
    mapping = fault.read_pickle(source["raw"])["selected"]["e3map"]
    physical = READ(spec["fault_file"])["faults"]["PF4"]
    assert physical["failed_links"] == [[[0, 11], [1, 8]], [[1, 8], [0, 11]]]
    eir.start_mapping = lambda start, op: mapping
    eir.fault_of = lambda op, fid: physical
    action = "l1" if spec["kind"] == "s_l1" else "rep"
    name = spec["job_id"]
    inner = dict(spec, name=name, kind=action, start="a1", op=spec["phase_op"], fault="PF4",
                 sink=str(fault.raw_path(out, name)), pre_sink=str(fault.path(spec["pre_sink"])), checks=False)
    if action == "rep":
        inner.update(method="L2Gsync", l1_sink=str(fault.path(spec["l1_sink"])))
    fault.install_broadcast_seed(inner)
    original = getattr(eir, "child_" + action)
    original_finalize = eir.Engine.finalize
    def finalize(engine, seeds=None):
        active = c.cpcm.simulate
        c.cpcm.simulate = bb.ORIGINAL
        try:
            return original_finalize(engine, seeds)
        finally:
            c.cpcm.simulate = active
    eir.Engine.finalize = finalize
    c.cpcm.simulate = bb.simulate
    assert c.EXACT_REVERT and c.BROADCAST_T and c.BC_CACHE
    def run(eng, inner_spec, sink, *args):
        original(eng, inner_spec, sink, *args)
        exports = {}
        if action == "l1":
            for state in ("R0", "L1"):
                eng.load(sink[state + "_snap"])
                exports[state] = fault.export_engine(eng, Path(out) / "exports" / (name + "__" + state + ".pkl.gz"), spec, sink["T_" + state])
            assert sink["T_pre"] == spec["healthy_T"]
        else:
            assert len(sink["search"]["rows"]) == spec["steps"]
            assert sink["T_L1_check"] == eir.read_sink(inner_spec["l1_sink"])["T_L1"]
            exports["final"] = fault.export_engine(eng, Path(out) / "exports" / (name + "__final.pkl.gz"), spec, sink["record"]["T"])
        sink["exports"] = exports
        sink["healthy_source"] = source
    setattr(eir, "child_" + action, run)
    eir.write_sink = lambda filename, data: fault.write_pickle(filename, dict(data, done=True))
    try:
        eir.child(inner)
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    sink = fault.read_pickle(fault.raw_path(out, name))
    result = dict(source_candidate=source["candidate_id"], exports=sink["exports"],
                  sink=fault.relative(fault.raw_path(out, name)), healthy_T=spec["healthy_T"],
                  stopping_reference="own healthy A1 deterministic CPCM", search_acceleration="deterministic bb only",
                  finalization="original CPCM and original ten-seed backend", exact_rollback=True, checks=False)
    mean = lambda state: statistics.mean(v[0] for v in state["backend"].values())
    if action == "l1":
        result.update(T_pre=sink["T_pre"], T_R0=sink["T_R0"], T_L1=sink["T_L1"],
                      R0_backend_mean=mean(sink["R0"]), L1_backend_mean=mean(sink["L1"]),
                      migration_R0=sink["R0"]["migration_vs_start"], migration=sink["L1"]["migration_vs_start"],
                      L1_evaluated=True, backend_calls=20,
                      timing={k: sink[k] for k in ("t_R0", "t_L1", "cpcm_L1")})
    else:
        result.update(T=sink["record"]["T"], backend_mean=mean(sink["record"]), steps=spec["steps"], seed=spec["seed"],
                      timing=sink["timing"], search_proposals=len(sink["search"]["rows"]),
                      migration=sink["record"]["migration_vs_start"], backend_calls=10)
    return result


def child(spec, out):
    out = fault.path(out)
    started = time.perf_counter()
    kind = spec["kind"]
    if kind == "s_rep" and receipt(out, spec["previous_stop"])["stop"]:
        prior = receipt(out, spec["previous_stop"])
        result = dict(status="reused", previous_stop=spec["previous_stop"], export=prior["export"], backend_calls=0)
    elif kind in ("s_l1", "s_rep"):
        result = a1_engine(spec, out)
    elif kind == "s_stop":
        result = fault.stop_job(spec, out)
        result["backend_calls"] = 0
    else:
        result = joint_child(spec, out)
    import e2e_pilot_peak
    result.update(job_id=spec["job_id"], kind=kind, status=result.get("status", "ok"), spec=spec,
                  wall_seconds=time.perf_counter() - started, peak_rss_gb=e2e_pilot_peak.peak() / 2**30)
    WRITE(fault.result_path(out, spec["job_id"]), result)
    print(json.dumps(dict(job_id=spec["job_id"], status=result["status"], seconds=result["wall_seconds"])), flush=True)


def graph_job(spec, out):
    import f_config
    import post_fg_dstu as dstu
    cid, state = spec["config_id"], spec["mapping_state"]
    if state != "A1EIR":
        suffix = "A1" if state == "A1R0" else state
        filename = OLD / "results" / (cid + "__graph_D_" + suffix + ".json")
        graph = READ(filename)
        graph_source = fault.relative(filename)
    else:
        p_source = READ(OLD / "results" / (cid + "__pre_pf_mha_A1.json"))["exports"]["healthy"]
        sources = {"P": {"file": p_source}}
        for op in joint.DC:
            sources[op] = {"file": receipt(out, f"S_{spec['hardware_profile']}_{op}_A1__stop300")["export"]}
        original_resolve, original_read = joint.resolve_export, joint.read_json
        joint.resolve_export = lambda root, ref: ref["file"]
        def read(filename):
            if Path(filename) == Path(out) / "faults" / (cid + ".json"):
                filename = OLD / "faults" / (cid + ".json")
            return original_read(filename)
        joint.read_json = read
        try:
            graph = joint.graph_job(dict(spec, sources=sources, condition="D", backends=["orig", "hop"]), out)
        finally:
            joint.resolve_export, joint.read_json = original_resolve, original_read
        graph_source = fault.relative(Path(out) / "graphs" / (spec["job_id"] + ".json"))
    setup = dict(spec, phase="prefill", op="mha", method="A3h", g=0, x=123)
    f_config.prepare(setup, out)
    d = joint.d_modules()
    hp = f_config.build_platform(setup, out, faults=graph["fault"], shape_override="2x4")
    assert fault.platform_signature(hp) == graph["platform_sha256"]
    # A1's original graph had no hop preparation. Score its exact saved streams
    # after All256, without rebuilding their routes or order.
    if not graph["that"]:
        import dstu2
        graph = dict(graph, that={})
        for stream in (0, 1):
            events = fault.read_pickle(graph["files"][f"orig_{stream}"])
            chunks, stats = dstu2.chunk_by(events, hp, dict(size=256 * 1024, only_d2d=False))
            assert stats["bytes_conserved"] and stats["tree_conserved"]
            value, wall, activities = d.that_hop(chunks, hp)
            graph["that"][str(stream)] = int(value)
        graph_source = fault.relative(Path(out) / "graphs" / (spec["job_id"] + ".json"))
    if graph_source.startswith(fault.relative(out)):
        WRITE(graph_source, graph)
    identities = {}
    for stream in (0, 1):
        events = fault.read_pickle(graph["files"][f"orig_{stream}"])
        identities[str(stream)] = dict(semantic_sha256=dstu.graph_digest(events),
                                      iteration_sha256=dstu.execution_order_digest(events))
    reference = READ(OLD / "results" / (cid + "__joint_reference.json"))
    ranks = []
    for phase in PHASES:
        tau = int(round(reference["KV_start"] + phase * (reference["KV_end"] - reference["KV_start"])))
        deadlines = {s: 1.5 * graph["that"][str(s)] + (tau if s else 0) for s in (0, 1)}
        ranks.append(sorted(deadlines, key=lambda s: (deadlines[s], s)))
    assert ranks[0] == ranks[1] == ranks[2], ranks
    return dict(graph_record=graph_source, graph_sha256=digest(graph_source), C4_order_by_phase=ranks,
                graph_files_sha256={str(s): digest(graph["files"][f"orig_{s}"]) for s in (0, 1)},
                original_graph_identity=identities, backend_calls=0,
                mapping_state=state, reference="same healthy A1/original reference for all eight arms")


def simulation_spec(spec, out):
    prepared = receipt(out, spec["graph_job"])
    graph = READ(prepared["graph_record"])
    reference = READ(OLD / "results" / (spec["config_id"] + "__joint_reference.json"))
    phase = spec.get("phase", .1)
    tau = int(round(reference["KV_start"] + phase * (reference["KV_end"] - reference["KV_start"])))
    deadlines = {s: 1.5 * graph["that"][str(s)] + (tau if s else 0) for s in (0, 1)}
    order = sorted(deadlines, key=lambda s: (deadlines[s], s))
    return dict(spec, arm=spec["transport"], run_type="solo" if spec["kind"] == "s_solo" else "joint",
                graph_record=prepared["graph_record"], graph_sha256=prepared["graph_sha256"],
                graph_files_sha256=prepared["graph_files_sha256"], source_root=str(ROOT),
                fresh_protocol="fresh_transform_v1", trace=False, tau=tau,
                rank={s: order.index(s) for s in order}, frozen_C4_deadlines=deadlines,
                reference={k: reference[k] for k in ("T_P", "T_D", "KV_start", "KV_end")})


@contextmanager
def simulation_adapter():
    import post_fg_dstu as dstu
    old = dstu.graph_inputs
    def graph_inputs(spec, out):
        import f_config
        graph = READ(spec["graph_record"])
        assert digest(spec["graph_record"]) == spec["graph_sha256"]
        setup = dict(spec, phase="prefill", op="mha", method="A3h", g=0, x=123)
        f_config.prepare(setup, out)
        d = joint.d_modules()
        hp = f_config.build_platform(setup, out, faults=graph["fault"], shape_override="2x4")
        assert fault.platform_signature(hp) == graph["platform_sha256"]
        streams = {}
        for s in (0, 1):
            filename = graph["files"][f"orig_{s}"]
            assert digest(filename) == spec["graph_files_sha256"][str(s)]
            streams[s] = fault.read_pickle(filename)
        return d, hp, graph, streams
    dstu.graph_inputs = graph_inputs
    try:
        yield dstu
    finally:
        dstu.graph_inputs = old


def initial_identity(dstu, spec, out):
    _, hp, graph, streams = dstu.graph_inputs(spec, out)
    events, _, _, _ = dstu.transform(streams, hp, spec["arm"], spec["rank"])
    release = {0: 0, 1: spec["tau"]}
    if spec["run_type"] == "solo":
        events = {t: e for t, e in events.items() if t[1] - 10 == spec["stream"]}
        release = {spec["stream"]: 0}
    identity = dstu.evaluation_identity(spec, events, hp, release)
    return identity, dstu.execution_order_digest(events, identity["graph_sha256"])


def joint_child(spec, out):
    if spec["kind"] == "s_graph":
        return graph_job(spec, out)
    inner = simulation_spec(spec, out)
    with simulation_adapter() as dstu:
        identity, iteration = initial_identity(dstu, inner, out)
        if spec["kind"] == "s_joint_reuse":
            old = READ(spec["reuse_record"])
            assert digest(spec["reuse_record"]) == spec["reuse_sha256"]
            assert old["fresh_protocol"] == "fresh_transform_v1"
            assert old["evaluation_identity"] == identity
            assert old["initial_order_sha256"] == iteration
            assert old["spec"]["reference"] == inner["reference"]
            result = dict(old, backend_calls=0, status="reused", reused_from=spec["reuse_record"],
                          reused_sha256=spec["reuse_sha256"], reuse_identity_checked=True)
        else:
            result = None
            if spec["kind"] == "s_solo":
                for name in spec["prior_solos"]:
                    prior = receipt(out, name)
                    if prior["evaluation_identity"] == identity and prior["initial_order_sha256"] == iteration:
                        result = dict(prior, backend_calls=0, status="reused", reused_from_job=name,
                                      reuse_identity_checked=True)
                        break
            if result is None:
                result = dstu.simulate(inner, out)
                assert result["evaluation_identity"] == identity
                assert result["initial_order_sha256"] == iteration
        result["resolved_simulation_spec"] = inner
        if spec["kind"] in ("s_joint", "s_joint_reuse"):
            psolo = receipt(out, spec["P_solo"])
            dsolo = receipt(out, spec["D_solo"])
            ps, ds = psolo["completion"]["0"], dsolo["completion"]["1"]
            result["metrics"].update(P_solo=ps, D_solo=ds, paired_delta_P=result["metrics"]["P_completion"] - ps,
                                     D_interference=result["metrics"]["D_latency"] / ds - 1)
        return result


def joint_specs():
    rows = []
    for hw in ("wsc", "face"):
        base = base_spec(hw)
        def add(suffix, kind, deps, **kw):
            name = "S_" + hw + "__" + suffix
            rows.append(dict(base, job_id=name, kind=kind, depends_on=list(dict.fromkeys(deps)), **kw))
            return name
        graphs, solos = {}, {}
        for state in STATES:
            deps = [f"S_{hw}_{op}_A1__stop300" for op in joint.DC] if state == "A1EIR" else []
            graphs[state] = add("graph_" + state, "s_graph", deps, mapping_state=state, condition="D", p_op="mha")
        # Include all earlier same-kind solos as dependencies; identities decide
        # whether a completed exact graph can be reused, never the state label.
        prior = collections.defaultdict(list)
        for state in STATES:
            for transport in ("BB0", "A256"):
                for stream in (0, 1):
                    for seed in (1, 2, 3):
                        key = transport, stream, seed
                        name = add(f"solo_{state}_{transport}_{stream}_s{seed}", "s_solo",
                                   [graphs[state]] + prior[key], mapping_state=state, condition="D",
                                   graph_job=graphs[state], transport=transport, stream=stream, seed=seed,
                                   prior_solos=list(prior[key]))
                        solos[state, transport, stream, seed] = name
                        prior[key].append(name)
        for bits in ("000", "100", "010", "001", "110", "101", "011", "111"):
            state = ("A3h" if bits[0] == "1" else "A1") + ("EIR" if bits[1] == "1" else "R0")
            transport = "A256" if bits[2] == "1" else "BB0"
            for phase in PHASES:
                for seed in (1, 2, 3):
                    ps, ds = solos[state, transport, 0, seed], solos[state, transport, 1, seed]
                    fields = dict(mapping_state=state, condition="D", graph_job=graphs[state], transport=transport,
                                  arm_bits=bits, phase=phase, seed=seed, P_solo=ps, D_solo=ds)
                    if bits in ("110", "111"):
                        source = POST / "results" / f"dstu_{hw}_D_{transport}_g{phase}_s{seed}.json"
                        fields.update(reuse_record=fault.relative(source), reuse_sha256=digest(source))
                    add(f"joint_{bits}_g{phase}_s{seed}", "s_joint_reuse" if bits in ("110", "111") else "s_joint",
                        [graphs[state], ps, ds], **fields)
    assert len(rows) == 248
    return rows


def source_files():
    files = set()
    for hw in ("wsc", "face"):
        cid = base_spec(hw)["config_id"]
        selection = OLD / "selection" / (cid + ".json")
        files.update((selection, OLD / "faults" / (cid + ".json")))
        selected = READ(selection)
        for op in joint.DC:
            source = selected["operators"][op]["A1"]
            files.update(fault.path(source[k]) for k in ("record", "raw", "snapshot"))
            pre = OLD / "results" / (cid + f"__pre_{op}_A1.json")
            files.add(pre); files.add(fault.path(READ(pre)["sink"]))
        files.add(OLD / "results" / (cid + "__joint_reference.json"))
        for state in ("A1", "A3hR0", "A3hEIR"):
            graph = OLD / "results" / (cid + "__graph_D_" + state + ".json")
            files.add(graph)
            files.update(fault.path(value) for value in READ(graph)["files"].values())
        ppre = OLD / "results" / (cid + "__pre_pf_mha_A1.json")
        files.add(ppre); files.add(fault.path(READ(ppre)["exports"]["healthy"]))
        for arm in ("BB0", "A256"):
            for phase in PHASES:
                for seed in (1, 2, 3):
                    files.add(POST / "results" / f"dstu_{hw}_D_{arm}_g{phase}_s{seed}.json")
    return [{"path": fault.relative(p), "sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(files)]


def plan(out=OUT):
    jobs = [as_job(s) for s in repair_specs(out)]
    inputs = source_files()
    WRITE(HERE / "s_inputs.json", dict(status="passed", files=inputs, bytes=sum(r["bytes"] for r in inputs)))
    payload = dict(version=VERSION, jobs=jobs, counts=dict(collections.Counter(j["kind"] for j in jobs)),
                   maximum_searches=120, maximum_backend_calls=1360, stopping_reference="own healthy A1 CPCM",
                   seeds=list(REPAIR_SEEDS), backend_seeds=list(range(1, 11)))
    WRITE(HERE / "s_repair_manifest.json", payload)
    followup = [as_job(s) for s in joint_specs()]
    WRITE(HERE / "s_joint_manifest.json", dict(version=VERSION, jobs=followup, counts=dict(collections.Counter(j["kind"] for j in followup)),
                                               joint_logical=144, new_joints=108, reused_joints=36, solo_logical=96))
    WRITE(HERE / "s_manifest.json", dict(version=VERSION, jobs=jobs + followup,
                                         counts=dict(collections.Counter(j["kind"] for j in jobs + followup))))
    return {k: v for k, v in payload.items() if k != "jobs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "child"))
    parser.add_argument("--spec")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(fault.path(args.out))))
    else:
        child(READ(args.spec), args.out)


if __name__ == "__main__":
    main()
