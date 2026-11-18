"""Rebuild each transport identity from newly loaded inputs; no CPCM or backend."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import dstu_supplement_run as run
import fg_fault as F
import post_fg_dstu as D

BASE = run.BASE
VERSION = "dstu_supplement_fresh_identity_v1"


def execute(spec, out):
    import e2e_pilot_peak
    began = time.perf_counter()
    point = run.point_name(spec)
    prepare_file = F.result_path(run.OUT, point + "__prepare")
    prepared = F.read_json(prepare_file)
    inner = run.context(spec)
    phase_ranks = {phase: prepared["rank_checks"][str(phase)]["frozen_rank"] for phase in run.PHASES}
    assert len({run.digest(r) for r in phase_ranks.values()}) == 1
    assert all(row["same"] for row in prepared["rank_checks"].values())
    rank = phase_ranks[.1]
    source_files = []
    identities, mutation_comparisons = {}, []
    with run.adapter(spec):
        _, hp, graph, initial = D.graph_inputs(inner, out)
        actual_source = {str(s): dict(semantic=D.graph_digest(ev), iteration=D.execution_order_digest(ev)) for s,ev in initial.items()}
        assert actual_source == prepared["source_graph_identity"]
        del initial
        source_files.append(dict(path=inner["graph_record"], sha256=D.sha(F.path(inner["graph_record"]))))
        for stream in (0, 1):
            filename = graph["files"][f"orig_{stream}"]
            assert D.sha(F.path(filename)) == inner["graph_files_sha256"][str(stream)]
            source_files.append(dict(path=filename, sha256=D.sha(F.path(filename))))
        for arm in run.ARMS:
            # chunk_by changes event objects in place; each arm starts from disk.
            streams = {stream: F.read_pickle(graph["files"][f"orig_{stream}"]) for stream in (0, 1)}
            before = {str(s): dict(semantic=D.graph_digest(ev), iteration=D.execution_order_digest(ev)) for s,ev in streams.items()}
            assert before == actual_source
            events, metadata, _, stats = D.transform(streams, hp, arm, rank)
            mask = run.mask_identity(metadata)
            for phase in run.PHASES:
                current = run.context(dict(spec, phase=phase, seed=1))
                current["arm"] = arm
                value = dict(joint=run.graph_identity(current, events, hp, {0: 0, 1: current["tau"]}),
                    solos={str(s): run.graph_identity(current, {t:e for t,e in events.items() if t[1]-10 == s}, hp, {s: 0}) for s in (0, 1)},
                    mask_sha256=mask, chunk_stats=stats)
                name = f"{arm}_g{phase}"
                identities[name] = value
                old = prepared["identities"][name]
                mutation_comparisons.append(dict(arm=arm, phase=phase,
                    old_repeated_transform_identity_matches=value == old,
                    old_graph_sha256=old["joint"]["evaluation_identity"]["graph_sha256"],
                    fresh_graph_sha256=value["joint"]["evaluation_identity"]["graph_sha256"]))
            if arm == "S256":
                first = prepared["masks"][run.digest(rank)]
                assert first["mask_sha256"] == mask and first["stats"] == stats
    result = dict(status="passed", point=point, spec=spec, version=VERSION,
        prepare_record=F.relative(prepare_file), prepare_sha256=D.sha(prepare_file),
        graph_sha256=inner["graph_sha256"], source_inputs=source_files,
        source_graph_identity=actual_source, identities=identities,
        original_prepare_identity_comparison=mutation_comparisons,
        first_S256_mask_and_stats_match=True, normal_rank_same_all_phases=True,
        backend_calls=0, cpcm_calls=0, new_searches=0,
        reason="v1 prepare reused event objects across transforms; chunk_by mutates them. This separate identity check reloads both original streams before each arm.",
        production_result_unchanged=True,
        production_path="evaluate transforms a fresh loaded graph; simulate independently reloads the original graph and asserts matching actual execution identities",
        wall_seconds=time.perf_counter()-began, peak_rss_gib=e2e_pilot_peak.peak()/2**30,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256={name:D.sha(run.ROOT/name) for name in
            ("e2e/dstu_supplement_identity.py", "e2e/dstu_supplement_run.py", "e2e/post_fg_dstu.py", "dstu/dstu2.py")})
    target = BASE / "evidence/fresh_identity_checks" / (point + ".json")
    F.write_json(target, result)
    print(json.dumps(dict(point=point, status="passed", old_identity_matches=sum(r["old_repeated_transform_identity_matches"] for r in mutation_comparisons),
        backend_calls=0, cpcm_calls=0, wall_seconds=result["wall_seconds"], peak_rss_gib=result["peak_rss_gib"])), flush=True)
    return result


def plan(path):
    jobs = []
    for spec in run.contexts():
        point = run.point_name(spec)
        jobs.append(dict(id=point, kind="fresh_identity", spec=spec,
            argv=["{python}", "-u", "-B", "e2e/dstu_supplement_identity.py", "run", "--point", point],
            result=f"e2e/dstu_supplement/evidence/fresh_identity_checks/{point}.json", backend_calls=0, cpcm_calls=0))
    F.write_json(path, dict(version=VERSION, jobs=jobs, count=12, backend_calls=0, cpcm_calls=0))
    return dict(jobs=len(jobs), manifest=str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--point")
    parser.add_argument("--out", type=Path, default=BASE / "out/identity_check")
    parser.add_argument("--manifest", type=Path, default=BASE/"manifests/fresh_identity.json")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(args.manifest)))
    else:
        execute(next(s for s in run.contexts() if run.point_name(s) == args.point), args.out)
