"""F/G search candidates and dependency jobs; no simulation is run here."""

import argparse
from collections import Counter
import json
from pathlib import Path

from e_run import ROOT, write_json

PROTOCOL = "FG_W1_fixedLN_v1"
SA_SEEDS = (123, 1, 2, 3, 4, 5, 6)
BACKEND_SEEDS = list(range(1, 11))
OPS = ("ln", "mha", "proj", "ffn")


def scale_name(value):
    return format(value, "g").replace(".", "p")


def config_id(spec):
    return (f"{spec['stage']}_{spec['hardware_profile']}_{spec['shape']}_"
            f"{spec['split_mode']}_d2d{scale_name(spec['d2d_scale'])}")


def configurations():
    rows = [dict(stage="F", hardware_profile=p, shape="2x2", split_mode="fixed", d2d_scale=1.0)
            for p in ("wsc", "watos", "face")]
    rows += [dict(stage="G", hardware_profile="busybarn", shape=shape,
                  split_mode=split, d2d_scale=1.0)
             for shape in ("2x3", "3x3") for split in ("fixed", "growth")]
    return [row | {"config_id": config_id(row)} for row in rows]


def variants(method):
    if method == "Gemini":
        return [(0, x) for x in (123, *range(1, 14))]
    return [(g, x) for g in (0, 1)
            for x in (range(7) if method == "A1" else SA_SEEDS)]


def candidate_specs(configuration, family, methods=("A1", "A3h", "Gemini"),
                    phase_ops=None, run_out="e2e/fg/out/main"):
    pairs = phase_ops or [(phase, op) for phase in ("prefill", "decode") for op in OPS]
    for phase, op in pairs:
        for method in methods:
            for order, (g, x) in enumerate(variants(method)):
                suffix = f"s{x}" if method == "Gemini" else f"g{g}_{'l' if method == 'A1' else 's'}{x}"
                fault = "PF4" if family == "full" else "healthy"
                name = f"{configuration['config_id']}__{family}_{fault}_{phase}_{op}_{method}_{suffix}"
                spec = configuration | dict(candidate_id=name, kind="search", family=family,
                    phase=phase, op=op, method=method, g=g, x=x, candidate_order=order,
                    search_steps=1000, backend_seeds=BACKEND_SEEDS, fault_id=fault,
                    protocol_version=PROTOCOL, module="fg_run",
                    depends_on=[configuration['config_id'] + "__fault_select"] if family == "full" else [])
                if family == "full":
                    spec["fault_file"] = f"{run_out}/faults/{configuration['config_id']}.json"
                yield spec


def candidates(run_out="e2e/fg/out/main"):
    configurations_ = configurations()
    rows = [item for cfg in configurations_
            for item in candidate_specs(cfg, "healthy", run_out=run_out)]
    rows += [item for cfg in configurations_
             if cfg["stage"] == "F" or cfg["split_mode"] == "growth"
             for item in candidate_specs(cfg, "full", ("A3h",), run_out=run_out)]
    for scale in (0.5, 2.0):
        cfg = dict(stage="F_bw", hardware_profile="watos", shape="2x2", split_mode="fixed", d2d_scale=scale)
        cfg["config_id"] = config_id(cfg)
        rows.extend(candidate_specs(cfg, "bandwidth", ("A1", "A3h"),
                    [("decode", "mha"), ("decode", "ffn")], run_out))
    return rows


def pilot_candidates(run_out="e2e/fg/out/main"):
    return [s for s in candidates(run_out) if s["family"] == "healthy" and s["candidate_order"] == 0]


def jobs(specs, run_out):
    measured = json.loads((ROOT / "e2e/e/evidence/fg_estimates.json").read_text(encoding="utf-8"))
    durations = measured["per_job_duration_sources"]
    rows = []
    for spec in specs:
        key = f"{spec['phase']}/{spec['op']}/{spec['method']}"
        rows.append(dict(id=spec["candidate_id"], kind="search", spec=spec,
            deps=[spec["config_id"] + "__fault_select"] if spec["family"] == "full" else [],
            result=f"results/{spec['candidate_id']}.json",
            estimated_seconds=durations[key]["mean_job_seconds"],
            timing_basis=durations[key]["source_kind"], peak_rss_gb=1.0))
    return rows


def summary(specs):
    return dict(protocol_version=PROTOCOL, searches=len(specs),
                by_stage_family={f"{k[0]}/{k[1]}": v for k, v in
                    sorted(Counter((s["stage"], s["family"]) for s in specs).items())},
                gemini_terminal_states=sum(s["method"] == "Gemini" and s["x"] == 123 for s in specs),
                backend_calls=len(specs) * 10, random_tie_calls=len(specs) * 10,
                expected_primary_files=len(specs),
                note="Search candidates only; repair, paired solo and joint jobs are separate.")


def build(out, run_out):
    specs = candidates(run_out)
    healthy = [s for s in specs if s["family"] == "healthy"]
    pilot = pilot_candidates(run_out)
    assert len(specs) == 3024 and len(healthy) == 2352 and len(pilot) == 168
    assert len({s["candidate_id"] for s in specs}) == len(specs)
    assert summary(specs)["gemini_terminal_states"] == 56
    for name, rows in (("search", specs), ("healthy", healthy), ("pilot", pilot)):
        write_json(out / f"{name}.json", rows)
        write_json(out / f"{name}_jobs.json", jobs(rows, run_out))
    write_json(out / "configurations.json", configurations())
    write_json(out / "counts.json", summary(specs) | {"pilot_searches_included": len(pilot)})
    print(json.dumps(summary(specs), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "e2e/fg/manifest")
    parser.add_argument("--run-out", default="e2e/fg/out/main")
    args = parser.parse_args()
    build(args.out, args.run_out)


if __name__ == "__main__":
    main()
