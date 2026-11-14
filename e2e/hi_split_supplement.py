"""Extend the three frozen POST-FG Split cases to backend seeds 4--10."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def manifest(source, graph_root, output):
    old = read(source)
    originals = {(s["case"], s["factor"]): s for s in old["jobs"] if s["seed"] == 1}
    assert len(old["jobs"]) == 27 and len(originals) == 9
    jobs, files = [], []
    for (_, _), original in originals.items():
        local_graph = ROOT / original["graph"]
        digest = hashlib.sha256(local_graph.read_bytes()).hexdigest()
        target = Path(graph_root) / local_graph.name
        files.append(dict(local_file=original["graph"], graph=target.as_posix(), file_sha256=digest,
                          graph_sha256=original["graph_sha256"]))
        for seed in range(4, 11):
            spec = dict(original, module="hi_split_supplement", seed=seed,
                        job_id=f"split_{original['case']}_m{original['factor']}_s{seed}",
                        graph=target.as_posix(), graph_file_sha256=digest,
                        supplement_version="HIJ_split_seeds4_10_v1")
            jobs.append(spec)
    assert len(jobs) == len({s["job_id"] for s in jobs}) == 63
    write(output, dict(protocol_version=old["protocol_version"],
                       scope="Frozen three cases; factors 1/2/4; backend seeds 4--10 only",
                       quality_searches=0, backend_calls=63, source_manifest=str(source),
                       source_manifest_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                       input_files=files, jobs=jobs))
    print(json.dumps(dict(manifest=str(output), jobs=63, graphs=9, search_runs=0)))


def child(spec, out):
    import post_fg_split as split
    filename = ROOT / spec["graph"]
    assert hashlib.sha256(filename.read_bytes()).hexdigest() == spec["graph_file_sha256"]
    split.child(spec, Path(out))


def report(old_out, new_out, target):
    import post_fg_split as split
    split.setup()
    old_specs = read(ROOT / "e2e/post_fg/manifest/split.json")["jobs"]
    baselines = {}
    baseline_samples = []
    for case, config, op in split.CASES:
        prior = ROOT / "e2e/fg/out/main"
        l1 = split.static.read_pickle(prior / "raw" / f"{config}__l1_{op}_A3h_MF6.pkl.gz")
        stop = read(prior / "results" / f"{config}__stop_{op}_MF6_300.json")
        repair = split.static.read_pickle(prior / "raw" / (stop["source_job"] + ".pkl.gz"))
        baselines[case] = {"R0": l1["R0"]["backend"], "rigid_EIR_representative": repair["record"]["backend"]}
        for method, values in baselines[case].items():
            assert set(values) == set(range(1, 11))
            baseline_samples.extend(dict(case=case, method=method, seed=seed, backend_ns=values[seed][0],
                                         source="saved F/G independent backend samples") for seed in range(1, 11))
    rows = []
    for spec in old_specs:
        if spec["seed"] != 1:
            continue
        for seed in range(1, 11):
            job = f"split_{spec['case']}_m{spec['factor']}_s{seed}"
            directory = old_out if seed <= 3 else new_out
            result = read(directory / "results" / (job + ".json"))
            assert result["status"] == "ok" and result["graph_sha256"] == spec["graph_sha256"]
            rows.append(dict(case=spec["case"], factor=spec["factor"], axis=spec["axis"],
                             seed=seed, backend_ns=result["backend_triple_ns"][0],
                             T_hat_ns=result["T_hat_ns"], backend_seconds=result["backend_seconds"],
                             wall_seconds=result["wall_seconds"], peak_rss_gb=result["peak_rss_gb"],
                             source=str(directory / "results" / (job + ".json"))))
    assert len(rows) == 90
    summaries = []
    for case, factor in sorted({(r["case"], r["factor"]) for r in rows}):
        selected = [r for r in rows if (r["case"], r["factor"]) == (case, factor)]
        values = [r["backend_ns"] for r in selected]
        original = next(s for s in old_specs if s["case"] == case and s["factor"] == factor)
        info = split.static.read_pickle(ROOT / original["graph"])["info"]
        bound = info["rigid_tensor_load_bound"]["lower_bound_ns"]
        r0 = statistics.mean(v[0] for v in baselines[case]["R0"].values())
        rigid = statistics.mean(v[0] for v in baselines[case]["rigid_EIR_representative"].values())
        summaries.append(dict(case=case, factor=factor, n=10, mean_ns=statistics.mean(values),
                              min_ns=min(values), max_ns=max(values), median_ns=statistics.median(values),
                              R0_mean_ns=r0, rigid_EIR_representative_mean_ns=rigid,
                              relative_to_R0=statistics.mean(values)/r0-1,
                              relative_to_rigid_EIR=statistics.mean(values)/rigid-1,
                              rigid_tensor_lower_bound_ns=bound,
                              below_rigid_bound=sum(v < bound for v in values),
                              all_below_rigid_bound=max(values) < bound,
                              worst_seed=max(selected, key=lambda r: r["backend_ns"])["seed"]))
    target.mkdir(parents=True, exist_ok=True)
    for filename, content in (("split_samples.csv", rows), ("split_summary.csv", summaries),
                              ("split_baselines.csv", baseline_samples)):
        with (target / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(content[0]))
            writer.writeheader()
            writer.writerows(content)
    write(target / "split_supplement.json", dict(status="passed", old_backend_calls=27,
          new_backend_calls=63, total_samples=90, summary=summaries, samples=rows, baselines=baseline_samples,
          baseline_selection="R0 and original F/G rigid EIR backend-median representative; each 10 backend seeds"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("manifest", "child", "report"))
    parser.add_argument("--source", type=Path, default=ROOT / "e2e/post_fg/manifest/split.json")
    parser.add_argument("--graph-root", default="e2e/post_fg/out/split_prepare/graphs")
    parser.add_argument("--manifest", type=Path, default=ROOT / "e2e/hi/j/split_supplement.json")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "e2e/hi/out/split_supplement")
    parser.add_argument("--old-out", type=Path, default=ROOT / "e2e/post_fg/out/main")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "e2e/hi/j/split_results")
    args = parser.parse_args()
    if args.command == "manifest":
        manifest(args.source, args.graph_root, args.manifest)
    elif args.command == "child":
        child(read(args.spec), args.out)
    else:
        report(args.old_out, args.out, args.report_dir)
