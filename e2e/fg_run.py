"""Run one F/G search in a fresh process using the frozen E search protocol."""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import e_run
from e_run import ROOT, write_json


def resolved_spec(spec):
    spec = dict(spec)
    if spec.get("fault_file"):
        source = Path(spec["fault_file"])
        if not source.is_absolute():
            source = ROOT / source
        row = json.loads(source.read_text(encoding="utf-8"))["faults"][spec["fault_id"]]
        spec.update({key: row.get(key, []) for key in ("failed_nodes", "failed_links", "failed_pes")})
    return spec


def configure(spec, out):
    import f_config
    return f_config.prepare(spec, out)


def graph_metrics(snapshot, c):
    events, platform = snapshot[2:]
    begin = time.perf_counter()
    graph = c.cpcm.compile_bb(events, platform)
    compile_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    schedule = c.cpcm.simulate(graph, "bb")
    simulate_seconds = time.perf_counter() - begin
    paths = [graph.br_hi[b] - graph.br_lo[b] for b in range(len(graph.br_lo))]
    communications = [ev for ev in events.values() if ev.event_type != "computation"]
    comp = [ev for ev in events.values() if ev.event_type == "computation"]
    counts = Counter(ev.comp_type for ev in comp)
    return dict(graph_events=graph.n, dependency_edges=len(graph.pred),
                graph_hops=len(graph.hop_link), graph_resources=graph.nres,
                communication_bytes=int(sum(ev.comm_bytes for ev in communications)),
                routed_link_bytes=int(sum(ev.comm_bytes * sum(len(p) for p in ev.path_list)
                                          for ev in communications)),
                mean_branch_hops=statistics.mean(paths) if paths else 0,
                max_branch_hops=max(paths, default=0),
                computation_events_by_type=dict(counts),
                cpcm_compile_seconds=compile_seconds, cpcm_simulate_seconds=simulate_seconds,
                cpcm_measured_ticks=int(schedule.T))


def child(spec, out, recover=False):
    spec = resolved_spec(spec)
    out = Path(out).resolve()
    name = spec["candidate_id"]
    staging = out / ".inflight" / name
    original_evaluate, original_record = e_run.evaluate, e_run.record
    evaluated_best = {}

    def evaluate(snapshot, ctx, c, broadcast, backend, **options):
        started = time.perf_counter()
        digest = None
        if spec["method"] == "Gemini" and spec["x"] == 123:
            history = {key: dict(getattr(snapshot[3], key))
                       for key in ("link_loads_dict", "link_loads_count")}
            digest = hashlib.sha256(pickle.dumps((snapshot[:3], history), protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
        if options.get("random_ties") is False and digest is not None and digest == evaluated_best.get("digest"):
            row = copy.deepcopy(evaluated_best["result"])
            row.update(T_hat_rand=[], T_hat_rand_mean=None,
                       evaluation_seconds=time.perf_counter() - started,
                       evaluation_reused_from=name)
            row["backend_wall"] = [0.0] * len(row["backend_wall"])
            row["fg_metrics"].update(cpcm_compile_seconds=0.0, cpcm_simulate_seconds=0.0)
            return row
        row = original_evaluate(snapshot, ctx, c, broadcast, backend, **options)
        row["fg_metrics"] = graph_metrics(snapshot, c)
        if digest is not None and options.get("random_ties", True):
            evaluated_best.update(digest=digest, result=row)
        return row

    def record(spec_, result, *args):
        row = original_record(spec_, result, *args)
        row.update(fg_metrics=result["fg_metrics"], tick_seconds=1e-9,
                   backend_ns_mean=statistics.mean(r["cycles"] for r in row["backend"]))
        if "evaluation_reused_from" in result:
            row["evaluation_reused_from"] = result["evaluation_reused_from"]
        return row

    e_run.configure, e_run.evaluate, e_run.record = configure, evaluate, record
    e_run.child(spec, staging, recover=recover)
    # A primary result is published last; the scheduler uses it as completion.
    for folder in ("raw", "snapshots", "results"):
        (out / folder).mkdir(parents=True, exist_ok=True)
        paths = sorted((staging / folder).glob("*"), key=lambda p: p.name == name + ".json")
        for path in paths:
            path.replace(out / folder / path.name)
    print(json.dumps(dict(candidate_id=name, status="ok")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("child")
    worker.add_argument("--spec", required=True, type=Path)
    worker.add_argument("--out", required=True, type=Path)
    worker.add_argument("--recover-output", action="store_true")
    args = parser.parse_args()
    child(json.loads(args.spec.read_text(encoding="utf-8")), args.out, args.recover_output)


if __name__ == "__main__":
    main()
