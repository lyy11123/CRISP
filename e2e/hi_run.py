"""H/I full candidates: frozen search, deterministic bb acceleration, full evaluation."""
import argparse
import ast
import collections
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import statistics
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

from e_run import ROOT, write_json, signature, save_snapshot, load_snapshot


def instrument_anneal(c, mark, progress):
    """Add observations only; retain the original loop, actions and temperature schedule."""
    source = inspect.getsource(c.make_anneal)
    original_tree = ast.parse(source)
    tree = ast.parse(source)
    anneal = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "anneal")
    def call(name, value):
        return ast.Expr(ast.Call(ast.Name(name, ast.Load()), [ast.Constant(value)], []))
    body = [call("_hi_mark", "anneal_enter")]
    for node in anneal.body:
        assigned = [n.id for n in node.targets if isinstance(n, ast.Name)] if isinstance(node, ast.Assign) else []
        if "steps" in assigned:
            body.append(call("_hi_mark", "normalization_enter"))
        if "wall0" in assigned:
            body.append(call("_hi_mark", "search_begin"))
        body.append(node)
        if "sa_wall" in assigned:
            body.append(call("_hi_mark", "search_complete"))
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "step":
            node.body.append(ast.Expr(ast.Call(ast.Name("_hi_progress", ast.Load()), [ast.Name("step", ast.Load())], [])))
    anneal.body = body
    namespace = dict(c.__dict__, _hi_mark=mark, _hi_progress=progress)
    observed_source = ast.unparse(ast.fix_missing_locations(tree))
    class RemoveObservations(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in ('_hi_mark','_hi_progress'):
                return None
            return self.generic_visit(node)
    restored_tree = RemoveObservations().visit(ast.parse(observed_source))
    assert ast.dump(original_tree, include_attributes=False) == ast.dump(restored_tree, include_attributes=False)
    exec(compile(tree, "<hi_full_anneal_observations>", "exec"), namespace)
    return namespace["make_anneal"], dict(
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        observed_sha256=hashlib.sha256(observed_source.encode()).hexdigest(),
        original_algorithm_ast_identical_after_removing_observations=True,
        loop="range(1, steps + 1)", changes="timestamps and progress messages only")


def graph_metrics(events, platform, c):
    graph = c.cpcm.compile_bb(events, platform)
    schedule = c.cpcm.simulate(graph, 'bb')
    computations = [e for e in events.values() if e.event_type == "computation"]
    comms = [e for e in events.values() if e.event_type == "communication"]
    return dict(events=len(events), activities=len(schedule.a_s),
                branch_hop_activity_references=len(computations) + len(graph.hop_link),
                graph_metric_version='schedule_activities_v2',
                dependency_edges=len(graph.pred), communication_hops=len(graph.hop_link),
                resources=graph.nres, computations=dict(collections.Counter(e.comp_type for e in computations)),
                communication_bytes=sum(int(e.comm_bytes) for e in comms),
                routed_link_bytes=sum(int(e.comm_bytes) * sum(len(p) for p in e.path_list) for e in comms))


def execution_order_digest(events):
    rows=[]
    for tag,event in events.items():
        target=getattr(event,'target_location',None)
        rows.append((tag,list(event.dependency_set),list(event.issue_set),
                     list(target) if isinstance(target,set) else target))
    return hashlib.sha256(json.dumps(rows,default=int).encode()).hexdigest()


def child(spec, out):
    assert spec["search_steps"] == 1000
    assert spec["backend_seeds"] == list(range(1, 11))
    assert spec["random_cpcm_seeds"] == list(range(1, 11))
    assert spec["kind"] == "search"
    started = time.perf_counter()
    out = Path(out).resolve()
    name = spec["candidate_id"]
    stage = out / ".inflight" / name
    for folder in ("results", "snapshots", "raw"):
        (stage / folder).mkdir(parents=True, exist_ok=True)
    milestones = {}
    def mark(label):
        milestones[label] = time.perf_counter() - started
        write_json(out / "progress" / (name + ".json"), dict(candidate_id=name, phase=label,
                   elapsed_seconds=milestones[label], milestones=milestones))
        print(json.dumps(dict(candidate_id=name, phase=label, elapsed_seconds=milestones[label])), flush=True)
    def progress(step):
        if step % 25 == 0:
            write_json(out / "progress" / (name + ".json"), dict(candidate_id=name, phase="search",
                       step=step, total_steps=1000, elapsed_seconds=time.perf_counter()-started,
                       search_elapsed_seconds=time.perf_counter()-started-milestones["search_begin"]))
            print(f"[hi] {name} step {step}/1000", flush=True)
    mark("configure_begin")
    import hi_config
    pilot, c, cfg = hi_config.prepare(spec, stage)
    assert spec['model_contract_sha256'] == hi_config.digest(hi_config.model_contract(spec['model']))
    assert c.EXACT_REVERT and c.BROADCAST_T and c.BC_CACHE
    import post_fg_bb as bb
    import add_communication
    import event_driver
    backend = event_driver.event_driver
    original_final = c.final_eval
    original_simulate = bb.ORIGINAL
    c.cpcm.simulate = bb.simulate
    mark("configure_complete")
    bc_calls = []
    original_bc = c.BcCache.graph
    def timed_bc(self, *args, **kwargs):
        t = time.perf_counter()
        result = original_bc(self, *args, **kwargs)
        bc_calls.append(dict(seconds=time.perf_counter()-t, phase="search" if "search_begin" in milestones else "initialization_calibration"))
        return result
    c.BcCache.graph = timed_bc
    evaluations = []
    def evaluate(snapshot, ctx, broadcast, driver, random_ties=True):
        mark("evaluation_begin")
        c.cpcm.simulate = original_simulate
        begin = time.perf_counter()
        calls, orders = [], []
        def timed_driver(**kwargs):
            index = len(calls)
            orders.append(dict(seed=spec['backend_seeds'][index],
                               actual_iteration_sha256=execution_order_digest(kwargs['events_dict'])))
            mark(f"backend_seed_{spec['backend_seeds'][index]}_begin")
            t = time.perf_counter()
            answer = driver(**kwargs)
            calls.append(time.perf_counter()-t)
            return answer
        work_signature = signature(snapshot)
        result = original_final(snapshot, ctx, broadcast, timed_driver)
        backend_end = time.perf_counter()
        mark("random_cpcm_begin")
        tie_begin = time.perf_counter()
        graph = c.cpcm.compile_bb(snapshot[2], snapshot[3])
        ties, times = [], []
        for seed in spec["random_cpcm_seeds"] if random_ties else []:
            t = time.perf_counter()
            ties.append(int(original_simulate(graph, "bb", rng=random.Random(seed)).T))
            times.append(time.perf_counter()-t)
        tie_elapsed = time.perf_counter()-tie_begin
        metrics = graph_metrics(snapshot[2], snapshot[3], c)
        result.update(T_hat_rand=ties, T_hat_rand_mean=statistics.mean(ties) if ties else None,
                      workload_signature=work_signature, graph=metrics,
                      evaluation_seconds=time.perf_counter()-begin,
                      backend_input_orders=orders,
                      timing=dict(backend_individual_seconds=calls, random_cpcm_individual_seconds=times,
                                  evaluation_prepare_and_backend_seconds=backend_end-begin,
                                  random_cpcm_with_compile_seconds=tie_elapsed,
                                  evaluation_other_seconds=time.perf_counter()-begin-sum(times)-sum(calls)))
        assert list(result["backend"]) == spec["backend_seeds"]
        tags = set(snapshot[2])
        assert all((e.dependency_set | e.issue_set) <= tags for e in snapshot[2].values())
        evaluations.append(result["timing"])
        c.cpcm.simulate = bb.simulate
        mark("evaluation_complete")
        return result
    c.final_eval = evaluate
    raw_path = stage / "raw" / (name + ".pkl.gz")
    snap_path = stage / "snapshots" / (name + ".pkl.gz")
    if spec["method"] == "Gemini":
        import e_gemini
        import Stream_Mapping
        event_driver.event_driver = lambda **kwargs: (0, 0, 0)
        task = hi_config.task_path(spec)
        with e_gemini.install(Stream_Mapping.StreamOptimization, seed=spec["x"], steps=1000,
                              validate_rollback=False, find_context=c._find_context) as capture:
            try:
                exec(compile(task.read_text(encoding="utf-8"), str(task), "exec"),
                     {"__file__": str(task), "__name__": "__main__"})
            except e_gemini.SearchComplete:
                pass
        event_driver.event_driver = backend
        save_snapshot(snap_path, capture["best"])
        result = evaluate(capture["best"], capture["ctx"], add_communication.add_broadcast, backend)
        run = dict(meta=dict(ctx=capture["ctx"], steps=1000, calibration=None), selected=result,
                   sa_wall=capture["meta"]["sa_wall_s"], gemini=capture["meta"])
        with gzip.open(raw_path, "wb") as stream:
            pickle.dump(run, stream, protocol=pickle.HIGHEST_PROTOCOL)
        instrumentation = dict(changes="native Gemini capture; no short trajectory entry")
        if capture["terminal"] is not None:
            save_snapshot(stage / "snapshots" / (name + "_terminal.pkl.gz"), capture["terminal"])
            terminal = evaluate(capture["terminal"], capture["ctx"], add_communication.add_broadcast, backend, False)
            write_json(stage / "results" / (name + "_terminal.json"), dict(
                candidate_id=name+"_terminal", status="ok", from_search=name,
                backend=terminal["backend"], timing=terminal["timing"]))
    else:
        c.make_anneal, instrumentation = instrument_anneal(c, mark, progress)
        try:
            c.child(cfg, spec["method"], spec["g"], spec["x"], str(raw_path), str(snap_path), False)
        except SystemExit as end:
            if end.code not in (0, None):
                raise
        with gzip.open(raw_path, "rb") as stream:
            run = pickle.load(stream)
        assert len(run["rows"]) == 1000 and run["meta"]["steps"] == 1000
        assert all(r["step"] == i for i, r in enumerate(run["rows"], 1))
        assert run["check_revert"]["n"] == 0 and not run["check2"] and not run["check4"]
        result = run["selected"]
    mark("export_complete")
    restored = load_snapshot(snap_path, c, cfg)
    assert signature(restored) == result["workload_signature"]
    assert int(c.cpcm_sim(restored[2], restored[3])[1].T) == int(result["T_pre"])
    mark("snapshot_reloaded")
    cal = run["meta"]["calibration"]
    timing = dict(milestones_seconds=milestones, sa_seconds=run["sa_wall"],
                  calibration_seconds=cal["wall"] if cal else 0,
                  first_broadcast_seconds=bc_calls[0]["seconds"] if bc_calls else None,
                  broadcast_total_seconds=sum(r["seconds"] for r in bc_calls),
                  broadcast_calls=len(bc_calls), evaluation=result["timing"],
                  complete_seconds=time.perf_counter()-started)
    if "anneal_enter" in milestones:
        timing.update(template_initialization_seconds=milestones["anneal_enter"]-milestones["configure_complete"],
                      initial_graph_and_calibration_seconds=milestones["search_begin"]-milestones["anneal_enter"],
                      best_restore_seconds=milestones["evaluation_begin"]-milestones["search_complete"],
                      export_seconds=milestones["export_complete"]-milestones["evaluation_complete"])
    record = dict(spec, status="ok", timing=timing, wall_seconds=time.perf_counter()-started,
                  search_seconds=run["sa_wall"], peak_rss_gb=pilot.peak()/2**30,
                  T_hat=int(result["T_bc"]), T_hat_pre=int(result["T_pre"]),
                  T_hat_rand=result["T_hat_rand"], T_hat_rand_mean=result["T_hat_rand_mean"],
                  backend=[dict(seed=int(seed), cycles=values[0], triple=list(values)) for seed, values in result["backend"].items()],
                  backend_seconds=result["backend_wall"], workload_signature=result["workload_signature"],
                  backend_input_orders=result['backend_input_orders'],
                  graph=result["graph"], selected_step=result.get("step", run.get("gemini", {}).get("best_step")),
                  own_cost=result.get('own_cost',run.get('gemini',{}).get('best_energy')),
                  instrumentation=instrumentation, exact_rollback=True, debug_checks=False,
                  evaluation_protocol="frozen final_eval: fresh deep copies and reset PRNG per seed; original backend and random-tie CPCM",
                  accelerated_forward_summary=dict(calls=len(bb.TIMING)),
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), Path(hi_config.__file__), Path(bb.__file__), Path(c.__file__))})
    write_json(stage / "results" / (name + ".json"), record)
    for folder in ("raw", "snapshots", "results"):
        (out / folder).mkdir(parents=True, exist_ok=True)
        for path in sorted((stage / folder).glob("*"), key=lambda p:p.name == name+".json"):
            path.replace(out / folder / path.name)
    print(json.dumps(dict(candidate_id=name, status="ok", seconds=record["wall_seconds"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("child",))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    child(json.loads(args.spec.read_text(encoding="utf-8")), args.out)


if __name__ == "__main__":
    main()
