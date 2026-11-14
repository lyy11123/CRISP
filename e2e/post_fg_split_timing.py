"""Measure Split graph construction separately from backend evaluation and SA."""
import time

ENTRY_STARTED = time.perf_counter()

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys

import post_fg_split as split


def run(run_out, prepared, manifest, out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    original_files = [manifest, *sorted((prepared / "graphs").glob("*.pkl.gz"))]
    original_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in original_files}
    official = {(s["case"], s["factor"]): s for s in split.read(manifest)["jobs"]}
    t = time.perf_counter()
    split.setup()
    numerical = split.numerical_checks()
    setup_seconds = time.perf_counter() - t
    case_rows, choices, selected = [], [], []
    for case, _, _ in split.CASES:
        case_started = time.perf_counter()
        t = time.perf_counter()
        # The old selection JSON names files relative to its own repository.
        current_root = split.ROOT
        split.ROOT = run_out.parents[3]
        state = split.read_case(run_out, case)
        split.ROOT = current_root
        read_seconds = time.perf_counter() - t
        t = time.perf_counter()
        healthy = split.platform(state["spec"])
        hp = split.platform(state["spec"], state["dead"])
        platform_seconds = time.perf_counter() - t
        assert all(tuple(loc[:2]) != state["dead"] for kind in ("tensorcore", "vectorunit")
                   for loc in hp.modules_dict[kind])
        assert set(hp.links_dict) == set(healthy.links_dict)
        t = time.perf_counter()
        _, identity = split.adapt(state, healthy, 1, "N", identity=True)
        identity_seconds = time.perf_counter() - t
        t = time.perf_counter()
        baseline = split.event_metrics(state["events"], healthy)
        baseline_seconds = time.perf_counter() - t
        for factor in (1, 2, 4):
            decision_started = time.perf_counter()
            candidates = []
            for axis in (("N",) if factor == 1 else ("N", "M")):
                t = time.perf_counter()
                graph, info = split.adapt(state, hp, factor, axis)
                build_wall = time.perf_counter() - t
                score = (info["metrics"]["tensor_rounded_flops"] - baseline["tensor_rounded_flops"],
                         info["metrics"]["communication"]["route_link_bytes"], 0 if axis == "N" else 1)
                candidates.append((score, axis, graph, info, build_wall))
            score, axis, graph, info, build_wall = min(candidates, key=lambda r: r[0])
            decision_wall = time.perf_counter() - decision_started
            candidate_seconds = sum(c[3]["build_seconds"] for c in candidates)
            for rank, other_axis, _, other, wall in candidates:
                choices.append(dict(case=case, factor=factor, axis=other_axis,
                                    selected=axis == other_axis, build_seconds=other["build_seconds"],
                                    build_wall_seconds=wall, rounded_flop_delta=rank[0],
                                    route_link_bytes=rank[1]))
            t = time.perf_counter()
            digest = split.graph_digest(graph)
            old = split.static.read_pickle(prepared / "graphs" / f"{case}_m{factor}.pkl.gz")
            old_digest = split.graph_digest(pickle.loads(old["events_blob"]))
            expected = official[(case, factor)]
            assert digest == old_digest == expected["graph_sha256"], (case, factor)
            assert axis == expected["axis"], (case, factor, axis)
            verify_seconds = time.perf_counter() - t
            info["axis_selection_seconds"] = candidate_seconds
            t = time.perf_counter()
            split.write_pickle(out / "graphs" / f"{case}_m{factor}.pkl.gz",
                               dict(events_blob=pickle.dumps(graph, pickle.HIGHEST_PROTOCOL), info=info,
                                    source_spec=state["spec"], source_candidate=state["source_candidate"],
                                    source_export=state["healthy_export"], dead=state["dead"]))
            write_seconds = time.perf_counter() - t
            row = dict(case=case, factor=factor, axis=axis, graph_sha256=digest,
                       official_graph_sha256=old_digest, equal=True,
                       selected_build_seconds=info["build_seconds"],
                       candidate_build_seconds=candidate_seconds, decision_wall_seconds=decision_wall,
                       verify_seconds=verify_seconds, write_seconds=write_seconds)
            selected.append(row)
            split.write(out / "metadata" / f"{case}_m{factor}.json", dict(timing=row, info=info))
        rows = [r for r in selected if r["case"] == case]
        case_rows.append(dict(case=case, read_seconds=read_seconds, platform_seconds=platform_seconds,
                              identity_seconds=identity_seconds, identity_events=identity["DAG_events"],
                              baseline_metrics_seconds=baseline_seconds,
                              candidate_build_seconds=sum(r["candidate_build_seconds"] for r in rows),
                              decision_wall_seconds=sum(r["decision_wall_seconds"] for r in rows),
                              selected_build_seconds=sum(r["selected_build_seconds"] for r in rows),
                              verify_seconds=sum(r["verify_seconds"] for r in rows),
                              write_graph_seconds=sum(r["write_seconds"] for r in rows),
                              complete_case_seconds=time.perf_counter() - case_started))
        print(json.dumps(case_rows[-1]), flush=True)
    assert len(choices) == 15 and len(selected) == 9
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in original_hashes.items())
    split.table(out / "candidate_timing.csv", choices)
    split.table(out / "selected_timing.csv", selected)
    split.table(out / "case_timing.csv", case_rows)
    import e2e_pilot_peak
    evidence = dict(status="passed", host=platform.node(), system=platform.platform(),
                    python=platform.python_version(), cpu_count=os.cpu_count(), command=sys.argv,
                    source_root=str(run_out.parents[3]), run_out=str(run_out),
                    official_manifest=str(manifest), official_files_unchanged=original_hashes,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    split_source_sha256=hashlib.sha256(Path(split.__file__).read_bytes()).hexdigest(),
                    setup_seconds=setup_seconds, complete_run_seconds=time.perf_counter() - started,
                    complete_entry_seconds=time.perf_counter() - ENTRY_STARTED,
                    peak_rss_gib=e2e_pilot_peak.peak() / 2**30, candidate_graphs=15,
                    selected_graphs=9, identity_graphs=3, numerical_checks=numerical,
                    backend_evaluations=0, search_runs=0, cases=case_rows, selected=selected)
    split.write(out / "timing.json", evidence)
    print(json.dumps({k: evidence[k] for k in ("status", "candidate_graphs", "selected_graphs",
                    "backend_evaluations", "complete_run_seconds", "complete_entry_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-out", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=split.ROOT / "e2e/post_fg/out/split_prepare")
    parser.add_argument("--manifest", type=Path, default=split.ROOT / "e2e/post_fg/manifest/split.json")
    parser.add_argument("--out", type=Path, default=split.ROOT / "e2e/post_fg/out/split_cloud_timing")
    args = parser.parse_args()
    run(args.run_out.resolve(), args.prepared.resolve(), args.manifest.resolve(), args.out.resolve())
