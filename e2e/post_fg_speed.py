"""Post-F/G bb equivalence and short-search timing, without backend repeats."""
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

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"

from e_run import ROOT, write_json


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in sorted(x.items(), key=lambda item: str(item[0]))}
    if isinstance(x, set):
        return [plain(v) for v in sorted(x, key=str)]
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True).encode()).hexdigest()


def short_anneal(c, limit):
    """Retain the frozen 1000-step cooling formula and stop its loop early.

    Changes are made to the loaded function, not the frozen source file. Timing
    fields measure the previously unattributed fingerprint/revert diagnostics.
    """
    source = inspect.getsource(c.make_anneal)
    tree = ast.parse(source)
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
             and n.target.id == "step"]
    assert len(loops) == 1
    loops[0].iter = ast.parse(f"range(1, min(steps, {int(limit)}) + 1)", mode="eval").body
    source = ast.unparse(ast.fix_missing_locations(tree))
    old = "fp0 = fingerprint() if exact and checks else None"
    assert source.count(old) == 1
    source = source.replace(old, "diag_start = time.perf_counter()\n            " + old +
                            "\n            fingerprint_seconds = time.perf_counter() - diag_start")
    old = "row['t_pre'] = t_pre"
    source = source.replace(old, old + "\n            row['t_fingerprint'] = fingerprint_seconds")
    old = "if exact and checks:\n"
    assert source.count(old) == 1
    source = source.replace(old, old + "                    diag_start = time.perf_counter()\n")
    old = "crev['state_mismatch'] += fingerprint() != fp0"
    source = source.replace(old, old + "\n                    row['t_revert_check'] = time.perf_counter() - diag_start")
    old = "sa_wall = time.time() - wall0"
    source = source.replace(old, old + "\n        sink['terminal_state'] = fingerprint()\n"
                            "        sink['terminal_schedule'] = {'T': st['sc'].T, 'W': [W(e) for e in range(st['g'].n)], 'CP': analysis().critical_path()}\n")
    namespace = dict(c.__dict__)
    exec(compile(source, "<post_fg_short_anneal>", "exec"), namespace)
    return namespace["make_anneal"], hashlib.sha256(source.encode()).hexdigest()


def state_record(snapshot):
    behaviours, data, events, hp = snapshot
    rows = []
    for tag, e in events.items():
        row = dict(tag=tag, deps=sorted(e.dependency_set, key=str), issues=sorted(e.issue_set, key=str))
        if e.event_type == "computation":
            a = e.commid_allocator
            row.update(location=e.comp_location, used_ids=sorted(a.used_ids), recycled=list(a.recycled), next_id=a.next_id)
        else:
            row.update(source=e.source_location, target=e.target_location, bytes=e.comm_bytes,
                       paths=e.path_list)
        rows.append(row)
    return plain(dict(events=rows, behaviour_locations={k: v.location for k, v in behaviours.items()},
                      loads=hp.link_loads_dict, counts=hp.link_loads_count))


def search(spec, out):
    import f_config
    started = time.perf_counter()
    config = dict(spec, search_steps=1000)
    pilot, c, cfg = f_config.prepare(config, out)
    import post_fg_bb as bb
    jit_start = time.perf_counter()
    if spec["execution"] == "fast":
        g = bb.cpcm.build([dict(tag="warm", kind="comp", deps=[], res="cpu", dur=1)])
        bb.simulate(g)
        c.cpcm.simulate = bb.simulate
    jit_seconds = time.perf_counter() - jit_start
    bb.TIMING.clear()
    selected = {}

    def final(snapshot, ctx, broadcast, backend):
        g, sc = c.cpcm_sim(snapshot[2], snapshot[3])
        selected.update(state=state_record(snapshot), T=sc.T,
                        W=[sc.S_last[e] - sc.r[e] for e in range(g.n)],
                        CP=c.cpcm.Analysis(sc).critical_path())
        return dict(T_pre=sc.T, backend_evaluations=0)

    c.final_eval = final
    c.make_anneal, source_sha = short_anneal(c, spec.get("trace_steps", 30))
    name = spec["job_id"]
    raw = out / "raw" / (name + ".pkl.gz")
    snap = out / "snapshots" / (name + ".pkl.gz")
    try:
        c.child(cfg, "A3h", spec.get("g", 0), spec.get("seed", 123), str(raw), str(snap),
                spec["execution"] == "checks")
    except SystemExit as end:
        if end.code not in (0, None):
            raise
    with gzip.open(raw, "rb") as stream:
        run = pickle.load(stream)
    rows = [{k: v for k, v in r.items() if not k.startswith("t_")} for r in run["rows"]]
    result = dict(trajectory=rows, calibration={k: v for k, v in run["meta"]["calibration"].items() if k != "wall"},
                  terminal_state=plain(run["terminal_state"]), terminal_schedule=plain(run["terminal_schedule"]),
                  selected=selected, best_step=run["selected"]["step"], check_revert=run["check_revert"],
                  cooling_steps=1000, executed_steps=len(rows), patched_function_sha256=source_sha)
    comparison = {k: result[k] for k in ("trajectory", "calibration", "terminal_state", "terminal_schedule", "selected", "best_step")}
    result["comparison_sha256"] = digest(comparison)
    result["timing"] = dict(sa_wall=run["sa_wall"], calibration=run["meta"]["calibration"]["wall"],
                            cold_jit=jit_seconds, complete_short_search=time.perf_counter()-started,
                            step_fields={k:sum(r.get(k, 0) for r in run["rows"]) for k in
                                         ("t_pre", "t_move", "t_target", "t_energy", "t_bcast", "t_revert", "t_snapshot", "t_fingerprint", "t_revert_check")},
                            accelerated_forward=list(bb.TIMING))
    result["coverage"] = dict(accepted=sum(r["accepted"] for r in rows), rejected=sum(not r["accepted"] for r in rows),
                              targeted=sum(r["targeted"] for r in rows))
    return result


def graph_job(spec, out):
    import e_run
    import f_config
    import post_fg_bb as bb
    pilot, c, cfg = f_config.prepare(spec, out)
    snap = Path(spec["snapshot"])
    if not snap.is_absolute():
        snap = ROOT / snap
    restored = e_run.load_snapshot(snap, c, cfg)
    started = time.perf_counter()
    g = c.cpcm.compile_bb(restored[2], restored[3])
    compile_seconds = time.perf_counter()-started
    comparison = bb.compare(g)
    cold = bb.TIMING[-1]
    repeats = spec.get("repeats", 3)
    original, accelerated = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        bb.ORIGINAL(g)
        original.append(time.perf_counter()-t0)
        bb.simulate(g)
        accelerated.append(bb.TIMING[-1])
    return dict(equivalence=comparison, timing=dict(compile_seconds=compile_seconds, cold=cold,
                original=original, accelerated=accelerated), snapshot=str(snap))


def synthetic():
    import post_fg_bb as bb
    build = bb.cpcm.build
    cases = []
    def comp(tag, res, dur, deps=()):
        return dict(tag=tag, kind="comp", res=res, dur=dur, deps=list(deps))
    def comm(tag, branches, duration, deps=()):
        return dict(tag=tag, kind="comm", branches=branches, dur=duration, deps=list(deps))
    cases.append(("shared_head", build([comm("a", [["q", "r"], ["q", "s"]], 2), comp("b", "cpu", 3, ["a"])]), None))
    cases.append(("last_issue_not_max", build([comm("a", [["q", "r"]], {"q": 9, "r": 2}), comp("b", "cpu", 1, ["a"])]), None))
    cases.append(("busy_heads", build([comm("a", [["q", "r"], ["s"]], 3), comm("b", [["q"], ["r", "s"]], 2), comp("c", "cpu", 1, ["a", "b"])]), None))
    cases.append(("compute_and_source_release", build([comp("a", "cpu", 5), comp("b", "cpu", 7), comm("c", [["q"]], 2, ["a"])]), {1: 5}))
    cases.append(("key3", build([comm("a", [["q", "r"], ["s", "r"]], 2), comm("b", [["r"]], 7)], [("q", "link", False), ("r", "link", False), ("s", "link", True)]), None))
    rng = random.Random(123)
    for i in range(80):
        events = []
        for e in range(rng.randrange(4, 25)):
            deps = rng.sample(range(e), min(e, rng.randrange(3)))
            if rng.random() < .35:
                events.append(comp(e, "cpu" + str(rng.randrange(3)), rng.randrange(1, 12), deps))
            else:
                branches = [rng.sample(["q" + str(q) for q in range(7)], rng.randrange(1, 6)) for _ in range(rng.randrange(1, 4))]
                events.append(comm(e, branches, rng.randrange(1, 9), deps))
        cases.append(("random_" + str(i), build(events), None))
    return [dict(name=name, **bb.compare(g, release)) for name, g, release in cases]


def child(spec, out):
    out = Path(out).resolve()
    for folder in ("raw", "snapshots", "results"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = graph_job(spec, out) if spec["kind"] == "speed_graph" else search(spec, out)
    import e2e_pilot_peak
    result.update(job_id=spec["job_id"], kind=spec["kind"], spec=spec, status="ok",
                  wall_seconds=time.perf_counter()-started, peak_rss_gb=e2e_pilot_peak.peak()/2**30)
    write_json(out / "results" / (spec["job_id"] + ".json"), plain(result))
    print(json.dumps(dict(job_id=spec["job_id"], seconds=result["wall_seconds"], status="ok")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "child"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        rows = synthetic()
        write_json(args.out, dict(status="passed", cases=rows, count=len(rows)))
        print(json.dumps(dict(status="passed", cases=len(rows))))
    else:
        child(json.loads(args.spec.read_text(encoding="utf-8")), args.out)


if __name__ == "__main__":
    main()
