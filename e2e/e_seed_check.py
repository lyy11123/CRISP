"""Trace short Gemini searches with 14 anneal seeds and one initialization per case."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from e_run import HERE, ROOT, configure, write_json


SEEDS = (123, *range(1, 14))
CASES = tuple((phase, op, shape) for phase in ("prefill", "decode") for op in ("ln", "proj") for shape in ("2x2", "1x4"))


class TraceComplete(Exception):
    """Stop the generated template before its final evaluation and result write."""


def plain(value):
    if isinstance(value, (tuple, list)):
        return [plain(part) for part in value]
    if isinstance(value, (set, frozenset)):
        return [plain(part) for part in sorted(value)]
    return value.item() if hasattr(value, "item") else value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def placement_digest(optimizer):
    rows = []
    for tag, event in sorted(optimizer.state[7].items()):
        if event.event_type == "computation":
            rows.append((plain(tag), "compute", plain(event.comp_location)))
        else:
            rows.append((plain(tag), "transfer", plain(event.source_location), plain(event.target_location)))
    return digest((int(optimizer.energy()), plain(optimizer.free_node), rows))


def check_case(case, steps, out):
    phase, op, shape = case
    started = time.perf_counter()
    _, c, cfg = configure({"phase": phase, "op": op, "shape": shape, "search_steps": steps, "backend_seeds": []}, out)
    import event_driver
    import numpy as np
    import Stream_Mapping
    import e_gemini

    optimizer_class = Stream_Mapping.StreamOptimization
    original_anneal = optimizer_class.anneal
    original_driver = event_driver.event_driver
    original_event = Stream_Mapping.update_event
    original_ddr = Stream_Mapping.update_offlinedata
    trials = []
    updates = None
    skipped_driver_calls = 0
    template = Path(c.WORK) / "models" / c.CONFIGS[cfg][0] / "py" / f"{op}_sq2048_gemini_ch{shape}{c.SUF}.py"
    source = template.read_bytes()

    def skip_driver(**kwargs):
        nonlocal skipped_driver_calls
        skipped_driver_calls += 1
        return 0, 0, 0

    def trace_event(*args, **kwargs):
        if updates is not None:
            updates.append({"behaviors": plain(kwargs["target_event_tags"]), "destinations": plain(kwargs["target_devices"])})
        return original_event(*args, **kwargs)

    def trace_ddr(*args, **kwargs):
        if updates is not None:
            updates.append({"tensor": plain(kwargs["target_data_tags"]), "split": plain(kwargs["target_data_split"]),
                            "destination": plain(kwargs["target_devices"])})
        return original_ddr(*args, **kwargs)

    def trace_anneal(optimizer):
        nonlocal updates
        initial = e_gemini.save_state(optimizer)
        initial_digest = placement_digest(optimizer)
        initial_movements = optimizer.movements
        native_move = optimizer.move
        trace = []

        def trace_move():
            nonlocal updates
            before = int(optimizer.energy())
            py_state = digest(random.getstate())
            numpy_state = np.random.get_state()
            np_state = digest([numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]])
            updates = []
            try:
                difference = native_move()
                trace.append({"kind": e_gemini.move_type(optimizer), "updates": updates, "energy_before": before,
                              "proposal_energy": int(optimizer.energy()), "python_rng": py_state, "numpy_rng": np_state})
                return difference
            finally:
                updates = None

        optimizer.move = trace_move
        try:
            # The final seed-123 repeat verifies deterministic replay after the other trials.
            for seed in (*SEEDS, 123):
                e_gemini.restore_state(optimizer, initial)
                optimizer.movements = initial_movements
                if placement_digest(optimizer) != initial_digest:
                    raise ValueError("Gemini initial placement differs after restoration")
                trace.clear()
                with e_gemini.install(optimizer_class, seed=seed, steps=steps, find_context=c._find_context) as capture:
                    try:
                        optimizer.anneal()
                    except e_gemini.SearchComplete:
                        pass
                meta = capture["meta"]
                if len(trace) != steps or any(not step["updates"] for step in trace):
                    raise ValueError("Missing native Gemini move evidence")
                after = [step["energy_before"] for step in trace[1:]] + [meta["terminal_energy"]]
                for step, energy in zip(trace, after):
                    step.update(accepted=(energy == step["proposal_energy"]), energy_after=energy)
                proposals = [(step["kind"], step["updates"]) for step in trace]
                trials.append({"seed": seed, "initial_placement": initial_digest, "proposal_digest": digest(proposals),
                               "best_energy": meta["best_energy"], "terminal_energy": meta["terminal_energy"],
                               "LP_flag": meta["LP_flag"], "trace": list(trace)})
        finally:
            optimizer.move = native_move
        raise TraceComplete

    optimizer_class.anneal = trace_anneal
    event_driver.event_driver = skip_driver
    Stream_Mapping.update_event = trace_event
    Stream_Mapping.update_offlinedata = trace_ddr
    try:
        exec(compile(source, str(template), "exec"), {"__file__": str(template), "__name__": "__main__"})
    except TraceComplete:
        pass
    finally:
        optimizer_class.anneal = original_anneal
        event_driver.event_driver = original_driver
        Stream_Mapping.update_event = original_event
        Stream_Mapping.update_offlinedata = original_ddr
    if len(trials) != len(SEEDS) + 1 or skipped_driver_calls != 1:
        raise ValueError("Generated Gemini template did not reach the expected anneal entry")
    repeated = trials.pop()
    unique = len({trial["proposal_digest"] for trial in trials})
    replay = repeated == trials[0]
    result = {"phase": phase, "op": op, "shape": shape, "steps": steps, "template_sha256": hashlib.sha256(source).hexdigest(),
              "unique_proposal_traces": unique, "seed123_replay_equal": replay, "backend_evaluations": 0,
              "wall_seconds": time.perf_counter() - started, "trials": trials}
    write_json(out / ("_".join(case) + ".json"), result)
    print(f"{' '.join(case)}: {unique}/{len(SEEDS)} distinct proposal traces, seed 123 replay {replay}", flush=True)
    return int(unique != len(SEEDS) or not replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "e/out/seed_check")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--case", nargs=3, metavar=("PHASE", "OP", "SHAPE"))
    args = parser.parse_args()
    if args.steps < 1 or args.workers < 1:
        parser.error("steps and workers must be positive")
    if args.case:
        case = tuple(args.case)
        if case not in CASES:
            parser.error(f"unknown case {case}")
        return check_case(case, args.steps, args.out.resolve())

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def launch(case):
        return subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--out", str(out),
                               "--steps", str(args.steps), "--case", *case], cwd=ROOT).returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(launch, CASES))
    rows = [json.loads((out / ("_".join(case) + ".json")).read_text(encoding="utf-8"))
            for case, rc in zip(CASES, results) if rc == 0]
    moves = Counter(step["kind"] for row in rows for trial in row["trials"] for step in trial["trace"])
    summary = {"cases_passed": len(rows), "cases_planned": len(CASES), "seeds": list(SEEDS), "steps_per_seed": args.steps,
               "wall_seconds": time.perf_counter() - started, "proposal_types": dict(moves),
               "failed_cases": [list(case) for case, rc in zip(CASES, results) if rc != 0]}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return int(any(results))


if __name__ == "__main__":
    raise SystemExit(main())
