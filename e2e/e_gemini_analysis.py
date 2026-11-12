"""Read saved E candidates and sum resource work; no search or simulation is run."""

import argparse
import ast
from collections import Counter, defaultdict
import configparser
import csv
import gzip
import json
import math
from pathlib import Path
import pickle
import statistics
import sys


ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "e2e"
ARTIFACT = ROOT / "third_party/isca2026-busybarn-artifact"
SUFFIX = "_bw256_co4x4_bw256_t128x64_failpattern0"


def pickle_types():
    # These are notation and device classes needed to decode saved objects and
    # evaluate the artifact's closed-form service-time formulas.
    for path in ("partition", "scheduling", "scheduling/communication/topology",
                 "platform/device", "platform/device/link", "platform/device/module"):
        sys.path.insert(0, str(ARTIFACT / "src" / path))
    from tensorcore import tensorcore
    from vectorunit import vectorunit
    return tensorcore, vectorunit


def read_pickle(path):
    with gzip.open(path, "rb") as file:
        return pickle.load(file)


def cfg(shape):
    path = HERE / "work/artifact/models/intra_mapping_gpt_prefill/cfg" / f"config_ch{shape}{SUFFIX}.cfg"
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    result = {}
    for section in parser.sections():
        result[section] = {}
        for key, value in parser[section].items():
            result[section][key] = ast.literal_eval(value) if value[:1].isdigit() or value.startswith("[") else value
    return result, path


def description(loads, devices):
    total = sum(loads.values())
    maximum = max(loads.values(), default=0)
    return {
        "total": total, "maximum": maximum, "mean_all_resources": total / devices,
        "max_to_mean": maximum / (total / devices) if total else 0,
        "used_resources": sum(v > 0 for v in loads.values()), "available_resources": devices,
        "busiest": [str(k) for k, v in sorted(loads.items(), key=lambda item: str(item[0])) if v == maximum],
        "per_resource": {str(k): v for k, v in sorted(loads.items(), key=lambda item: str(item[0]))},
    }


def link_key(source, target):
    # TupleIdx appends a parallel-link index to the destination in saved trees.
    return (tuple(source[:2]), tuple(target[:2]), *target[2:])


def link_class(link):
    source, target = link[:2]
    if source[1] >= 16:
        return "ddr2co"
    if target[1] >= 16:
        return "co2ddr"
    return "ch2ch" if source[0] != target[0] else "co2co"


def analyze(row, result_path, classes):
    record = json.loads(result_path.read_text(encoding="utf-8"))
    batch = result_path.parent.parent
    snapshot_path = ROOT / record["snapshot"] if record["kind"] == "reeval" else batch / "snapshots" / f"{row['candidate']}.pkl.gz"
    raw_path = ROOT / record["run"] if record["kind"] == "reeval" else batch / "raw" / f"{row['candidate']}.pkl.gz"
    snapshot_blob, _ = read_pickle(snapshot_path)
    behaviours, _, events = pickle.loads(snapshot_blob)
    run = read_pickle(raw_path)
    selected = run["selected"]
    mapping = selected["e3map"]
    # Historical E re-evaluation preserves its snapshot and deterministic result.
    assert int(selected["T_bc"]) == record["T_hat"]
    assert all(tuple(b.location) == tuple(mapping["beha_loc"][tag]) for tag, b in behaviours.items())
    config, cfg_path = cfg(row["shape"])
    units = {name: cls((0, 0, 0), config[name + "_cfg"]) for name, cls in zip(("tensorcore", "vectorunit"), classes)}
    comp = {name: Counter() for name in units}
    comp_counts = {name: Counter() for name in units}
    phase_loads = defaultdict(Counter)
    for event in events.values():
        if event.event_type != "computation":
            continue
        duration = int(units[event.comp_device].working_time(event.comp_datashape, event.comp_type, config["time_unit"]["frequency"]))
        comp[event.comp_device][tuple(event.comp_location)] += duration
        comp_counts[event.comp_device][tuple(event.comp_location)] += 1
        phase_loads[event.event_name.rsplit("_", 1)[0]][tuple(event.comp_location)] += duration
    link_cycles = defaultdict(Counter)
    link_bytes = defaultdict(Counter)
    ddr_reads = Counter()
    comm_count = 0
    for route_id, variants in mapping["routes"].items():
        nbytes = route_id[1]
        for source, tree, tag in variants:
            comm_count += 1
            if source[1] >= 16:
                ddr_reads[tuple(source)] += nbytes
            edges = {link_key(src, dst) for src, targets in tree for dst in targets}
            for edge in edges:
                kind = link_class(edge)
                settings = config[kind + "_cfg"]
                service = math.ceil(math.ceil(nbytes / settings[kind + "_bandwidth"]) * settings[kind + "_timeunit"]) + settings[kind + "_latency"]
                link_cycles[kind][edge] += service
                link_bytes[kind][edge] += nbytes
    ncomp = sum(sum(counts.values()) for counts in comp_counts.values())
    assert ncomp + comm_count == len(mapping["layout"]["order"])
    triples = [entry["triple"] for entry in record["backend"]]
    means = [statistics.mean(t[i] for t in triples) for i in range(3)]
    maximum_resource_work = max(max(loads.values()) for loads in (*comp.values(), *link_cycles.values()))
    assert maximum_resource_work <= min(t[0] for t in triples)
    result = {key: row[key] for key in ("shape", "phase", "op", "method", "candidate")}
    result.update(
        sources={"result": str(result_path.relative_to(ROOT)), "snapshot": str(snapshot_path.relative_to(ROOT)),
                 "raw": str(raw_path.relative_to(ROOT)), "config": str(cfg_path.relative_to(ROOT))},
        backend={"total_cycles_mean": means[0], "compute_only_cycles_mean": means[1],
                 "communication_only_cycles_mean": means[2], "compute_and_communication_cycles_mean": means[0] - means[1] - means[2],
                 "seeds": record["backend"]},
        computation={name: description(loads, 64) for name, loads in comp.items()},
        computation_phase={name: description(loads, 64) for name, loads in phase_loads.items()},
        ddr_read_bytes=description(ddr_reads, 32),
        link_service_cycles={kind: description(loads, {"ddr2co": 32, "co2ddr": 32, "co2co": 192,
                                                       "ch2ch": 16 if row["shape"] == "2x2" else 12}[kind]) for kind, loads in link_cycles.items()},
        link_bytes={kind: description(loads, {"ddr2co": 32, "co2ddr": 32, "co2co": 192,
                                             "ch2ch": 16 if row["shape"] == "2x2" else 12}[kind]) for kind, loads in link_bytes.items()},
        checks={"saved_mapping_matches_snapshot": True, "saved_T_bc_matches_E": True,
                "event_count_matches_saved_layout": True, "resource_work_bound_below_each_backend_total": True},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "E_SUMMARY.json")
    parser.add_argument("--json", type=Path, default=HERE / "e/evidence/gemini_attribution.json")
    parser.add_argument("--csv", type=Path, default=HERE / "e/tables/gemini_attribution.csv")
    args = parser.parse_args()
    classes = pickle_types()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    paths = {p.stem: p for name in ("pilot_remote", "full_remote") for p in (HERE / "e/out" / name / "results").glob("*.json")}
    rows = [analyze(row, paths[row["candidate"]], classes) for row in summary["operator_table"]
            if row["phase"] == "prefill" and row["op"] in ("mha", "ffn")]
    for shape in ("2x2", "1x4"):
        for op in ("mha", "ffn"):
            peers = [row for row in rows if (row["shape"], row["op"]) == (shape, op)]
            for unit in ("tensorcore", "vectorunit"):
                assert len({row["computation"][unit]["total"] for row in peers}) == 1
    result = {
        "method": "Static sums of saved computation assignments and final broadcast-inclusive communication routes; no backend, CPCM or SA execution.",
        "definitions": {
            "backend_triple": "total, time with computation but no communication, time with communication but no computation; complements are overlapping service time, not queue waiting",
            "resource_service": "Sum the artifact device/link working_time formulas for all saved events assigned to each resource; multicast shared links count once per event",
            "resource_maximum": "Static cumulative work lower bound for that resource; no issue times, queue wait or utilization is reconstructed",
            "routes": "Actual final communication sources/routes from selected.e3map, including final broadcast; historical platform.link_loads counters are not used",
            "device_counts": "64 tensorcores, 64 vectorunits, 32 DDR read ports, 192 directed on-die links, 16 (2x2) or 12 (1x4) directed parallel D2D links",
        },
        "source_definitions": [
            "cpcm_sa/cpcm_sa.py:337 final_eval saves final routes before backend and its three timing values",
            "e3_warmstart/e3_warmstart.py:128 routes_of saves actual sources and route trees",
            "third_party/isca2026-busybarn-artifact/src/backend/analytical/event_driver.py:760 pure-time accounting",
            "third_party/isca2026-busybarn-artifact/src/platform/device/module/tensorcore.py:29 working_time",
            "third_party/isca2026-busybarn-artifact/src/platform/device/module/vectorunit.py:48 working_time",
            "third_party/isca2026-busybarn-artifact/src/mapping/Stream_Mapping.py:379 hops_only energy omits max compute and link load terms",
            "e2e/work/artifact/models/intra_mapping_gpt_prefill/py/ffn_sq2048_gemini_ch1x4_bw256_co4x4_bw256_t128x64_failpattern0.py:57 ddr_random=True, dijkstra_routing=False, LP_flag=True, hops_only=True",
        ],
        "checks": {"selected_candidates": len(rows), "cross_method_total_computation_work_equal": True,
                   "simulations_executed": 0},
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flat = []
    for row in rows:
        item = {k: row[k] for k in ("shape", "op", "method", "candidate")}
        item.update({k: v for k, v in row["backend"].items() if k != "seeds"})
        for unit in ("tensorcore", "vectorunit"):
            item.update({unit + "_" + k: row["computation"][unit][k] for k in ("maximum", "mean_all_resources", "max_to_mean", "used_resources")})
        item.update({"ddr_read_bytes_" + k: row["ddr_read_bytes"][k] for k in ("total", "maximum", "max_to_mean", "used_resources")})
        for kind, stats in row["link_service_cycles"].items():
            item[kind + "_max_service_cycles"] = stats["maximum"]
        flat.append(item)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, list(dict.fromkeys(k for row in flat for k in row)))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(flat, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
