"""Compare original/Numba CPCM on the full W1 graph with cached-broadcast construction."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import time

from post_fg_speed import ROOT, plain
from e_run import write_json


def child(spec, out):
    import e_run
    import f_config
    import post_fg_bb as bb
    started = time.perf_counter()
    out = Path(out).resolve()
    pilot, c, cfg = f_config.prepare(spec, out)
    snap_path = Path(spec["snapshot"])
    raw_path = Path(spec["source_raw"])
    snapshot = e_run.load_snapshot(snap_path, c, cfg)
    with gzip.open(raw_path, "rb") as stream:
        saved = pickle.load(stream)
    behaviours, data, events, platform = snapshot
    t0 = time.perf_counter()
    full_events = c.BcCache(saved["meta"]["ctx"], platform).graph(behaviours, data, events, platform)
    bcast_seconds = time.perf_counter()-t0
    t0 = time.perf_counter()
    graph = c.cpcm.compile_bb(full_events, platform)
    compile_seconds = time.perf_counter()-t0
    equality = bb.compare(graph)
    cold = bb.TIMING[-1]
    original, accelerated = [], []
    for _ in range(spec.get("repeats", 3)):
        t0 = time.perf_counter()
        bb.ORIGINAL(graph)
        original.append(time.perf_counter()-t0)
        bb.simulate(graph)
        accelerated.append(bb.TIMING[-1])
    import e2e_pilot_peak
    row = dict(job_id=spec["job_id"], kind="speed_graph_full", spec=spec, status="ok",
               equivalence=equality, graph_scope="full_W1_BcCache_graph_from_saved_state",
               pre_events=len(events), full_events=len(full_events), full_hops=len(graph.hop_link),
               timing=dict(broadcast_seconds=bcast_seconds, compile_seconds=compile_seconds, cold=cold,
                           original=original, accelerated=accelerated),
               wall_seconds=time.perf_counter()-started, peak_rss_gb=e2e_pilot_peak.peak()/2**30)
    write_json(out / "results" / (spec["job_id"] + ".json"), plain(row))
    print(json.dumps(dict(job_id=spec["job_id"], status="ok", seconds=row["wall_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("child",))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    child(json.loads(args.spec.read_text(encoding="utf-8")), args.out)
