"""Release F/G categories after their existing pilot outputs pass checks.

Run once, or use --watch for a detached 30-second polling process. No simulation
is started here; fg_batch consumes the atomic release file.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from e_run import ROOT, write_json
from fg_batch import job_items
import fg_check
import fg_fault
from fg_joint import CHAIN

VERSION = "FG_release_v1"


class Pending(Exception):
    pass


def read(filename):
    return json.loads(Path(filename).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, name):
    require(isinstance(value, (int, float)) and math.isfinite(value) and value > 0, name)
    return value


def close(actual, expected, name):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), name)


def artifact(filename):
    p = Path(filename)
    p = p if p.is_absolute() else ROOT / p
    require(p.is_file() and p.stat().st_size > 0, f"missing artifact: {p}")


def result(jobs, out, name):
    filename = out / jobs[name]["result"]
    if not filename.exists():
        raise Pending(name)
    row = read(filename)
    require(row["job_id"] == name and row["kind"] == jobs[name]["kind"], f"result identity: {name}")
    require(row["status"] in ("ok", "reused"), f"result status: {name}")
    if row["status"] != "reused" or "spec" in row:
        require(row["spec"] == jobs[name]["spec"], f"result spec: {name}")
    positive(row["wall_seconds"], f"wall_seconds: {name}")
    return row


def full_ready(jobs, out, cid):
    row = result(jobs, out, cid + "__fault_select")
    stored = read(out / "faults" / (cid + ".json"))
    require(stored["config_id"] == cid and stored["faults"] == row["faults"], "fault file differs from result")
    fault = row["faults"]["PF4"]
    require(fault["failed_links"] == [[[0, 11], [1, 8]], [[1, 8], [0, 11]]]
            and not fault["failed_nodes"] and not fault["failed_pes"], "PF4 physical coordinates")
    return dict(fault="PF4", source_job=row["job_id"])


def repair_check(jobs, out, cid):
    target = cid + "__stop_pf_mha_PF4_300"
    result(jobs, out, target)
    pre_name, l1_name = cid + "__pre_pf_mha_A3h", cid + "__l1_pf_mha_A3h_PF4"
    pre, l1 = result(jobs, out, pre_name), result(jobs, out, l1_name)
    source = read(out / "selection" / (cid + ".json"))["operators"]["pf_mha"]["A3h"]["candidate_id"]
    for row in (pre, l1):
        require(row["source_candidate"] == source, "repair uses a different selected candidate")
        require(row["spec"]["backend_seeds"] == list(range(1, 11)), "repair backend seeds")
        artifact(row["sink"])
        for filename in row["exports"].values():
            artifact(filename)
    positive(pre["T"], "healthy CPCM"); positive(pre["backend_mean"], "healthy backend")
    close(l1["T_pre"], pre["T"], "repair healthy estimate")
    require(l1["L1_evaluated"] is True, "A3h L1 missing")
    for field in ("T_R0", "T_L1", "R0_backend_mean", "L1_backend_mean"):
        positive(l1[field], field)
    checked = []
    for steps in (0, 30, 100, 300):
        name = cid + f"__stop_pf_mha_PF4_{steps}"
        row = result(jobs, out, name)
        spec = jobs[name]["spec"]
        if steps:
            previous = result(jobs, out, spec["previous_stop"])
            reps = [result(jobs, out, r) for r in spec["rep_jobs"]]
            require(sorted(jobs[r]["spec"]["seed"] for r in spec["rep_jobs"]) == [1, 2, 3, 4, 123], "five SA seeds")
            for rep, rep_name in zip(reps, spec["rep_jobs"]):
                rs = jobs[rep_name]["spec"]
                require(rs["steps"] == steps and rs["l1_sink"] == l1["sink"]
                        and rs["pre_sink"] == pre["sink"], "independent L1 starting state")
                if previous["stop"]:
                    require(rep["status"] == "reused" and rep["previous_stop"] == spec["previous_stop"]
                            and rep["export"] == previous["export"], "repair early-stop reuse")
                else:
                    require(rep["status"] == "ok" and rep["seed"] == rs["seed"]
                            and rep["steps"] == steps and rep["source_candidate"] == source, "repair seed/state")
                    require(rs["backend_seeds"] == list(range(1, 11)), "repair backend seeds")
                    positive(rep["T"], "repair CPCM"); positive(rep["backend_mean"], "repair backend")
                    artifact(rep["sink"]); artifact(rep["exports"]["final"])
        expected = fg_fault.stop_job(spec, out)
        for field in ("stop", "level", "T", "T_pre", "export", "source_job", "reason"):
            require(row[field] == expected[field], f"stop {steps}: {field}")
        for field in ("seed", "backend_mean", "gain_from_previous"):
            require(row.get(field) == expected.get(field), f"stop {steps}: {field}")
        positive(row["T"], "selected CPCM"); artifact(row["export"])
        checked.append(dict(steps=steps, level=row["level"], reason=row["reason"], stop=row["stop"]))
    require(row["stop"] is True, "final repair not stopped")
    return dict(target=target, source_candidate=source, checks=checked,
                selected_level=row["level"], selected_source=row["source_job"])


def simulation_check(row, backend, policy, seed, streams, release):
    sim = row["simulation"]
    require((sim["backend"], sim["policy"], sim["seed"], sim["stop"]) == (backend, policy, seed, False), "simulation method/seed")
    require(sim["release"] == {str(k): v for k, v in release.items()}, "stream release times")
    require(set(sim["streams"]) == {str(s) for s in streams}, "simulation streams")
    for stream in streams:
        rec = sim["streams"][str(stream)]
        positive(rec["C"], "stream completion")
        require(rec["n"] > 0 and rec["n_done"] == rec["n"], "unfinished stream events")
        require(rec["first_start"] >= release[stream] and rec["C"] >= rec["first_start"], "stream timing")
    if backend == "hop":
        require(sim["audit"]["bad"] == 0 and sim["audit"]["n"] > 0, "hop arbitration audit")
    return sim


def joint_check(jobs, out, cid):
    target = cid + "__joint_healthy_CRISP_g0.5_s1"
    row = result(jobs, out, target); spec = jobs[target]["spec"]
    require((row["arm"], row["condition"], row["phase"], spec["seed"]) == ("CRISP", "healthy", .5, 1), "joint pilot identity")
    require(row["graph_job"] == spec["graph_job"] and row["reference_job"] == spec["reference_job"], "joint input jobs")
    graph = result(jobs, out, spec["graph_job"])
    require(graph["chain"] == CHAIN and not any(graph["fault"].values()), "healthy fixed-LN chain")
    require(graph["shared"]["compute"] == 0 and graph["shared"]["P_D_links"] == 0
            and graph["shared"]["KV_D_links"] > 0 and row["shared"] == graph["shared"], "joint shared resources")
    rows, cols = map(int, spec["shape"].split("x"))
    require(graph["kv"]["targets"] == 8 * rows * cols and graph["kv"]["messages"] > 0
            and graph["kv"]["unique_payload_bytes"] > 0 and graph["kv"]["cross_group_edges"] > 0, "cross-group KV")
    for filename in list(graph["files"].values()) + list(graph["source_exports"].values()):
        artifact(filename)
    for stream in ("0", "1"):
        size = graph["sizes"][stream]
        require(size["chunk"]["bytes_conserved"] and size["chunk"]["tree_conserved"], "chunk conservation")
        require(size["chunked_events"] >= size["events"] > 0, "chunk event count")
        positive(graph["that"][stream], "C4 own-graph CPCM")
    ref = result(jobs, out, spec["reference_job"])
    require(ref["backend_seeds"] == [1, 2, 3], "public reference seeds")
    ref_samples = {}
    for stream, field in ((0, "P_solos"), (1, "D_solos")):
        names = ref["spec"][field]
        require(len(names) == 3 and sorted(jobs[n]["spec"]["seed"] for n in names) == [1, 2, 3], "three public reference solos")
        ref_samples[stream] = [simulation_check(result(jobs, out, n), "orig", "P1", jobs[n]["spec"]["seed"], [stream], {stream: 0}) for n in names]
        close(ref["T_P" if stream == 0 else "T_D"], statistics.mean(s["streams"][str(stream)]["C"] for s in ref_samples[stream]), "reference completion mean")
    for index, field in enumerate(("KV_start", "KV_end")):
        close(ref[field], statistics.mean(s["kv_window"][index] for s in ref_samples[0]), "reference KV mean")
    require(0 <= ref["KV_start"] < ref["KV_end"] <= ref["T_P"], "reference KV interval")
    tau = int(round(ref["KV_start"] + .5 * (ref["KV_end"] - ref["KV_start"])))
    require(row["tau"] == tau, "common release tau")
    sim = simulation_check(row, "hop", "C4", 1, [0, 1], {0: 0, 1: tau})
    solos = {}
    for stream, field in ((0, "P_solo"), (1, "D_solo")):
        sr = result(jobs, out, spec[field])
        require(sr["graph_job"] == spec["graph_job"], "paired solo mapping")
        solo = simulation_check(sr, "hop", "P1", 1, [stream], {stream: 0})
        solos[stream] = solo["streams"][str(stream)]["C"]
        require(sim["streams"][str(stream)]["n"] == graph["sizes"][str(stream)]["chunked_events"], "simulated graph count")
    cp, dl = sim["streams"]["0"]["C"], sim["streams"]["1"]["C"] - tau
    expected = dict(P_completion=cp, D_latency=dl, paired_delta_P=cp-solos[0],
                    P_solo=solos[0], D_solo=solos[1], P_self_amplification=cp/solos[0],
                    D_self_amplification=dl/solos[1], Lhat_P=(cp-1.5*ref["T_P"])/ref["T_P"],
                    Lhat_D=(dl-1.5*ref["T_D"])/ref["T_D"])
    expected["Lhat_max"] = max(expected["Lhat_P"], expected["Lhat_D"])
    for field, value in expected.items():
        close(row["metrics"][field], value, field)
    return dict(target=target, tau=tau, kv_targets=graph["kv"]["targets"],
                graph_peak_rss_gb=graph["peak_rss_gb"], simulation_peak_rss_gb=row["peak_rss_gb"],
                arbitration_bad=sim["audit"]["bad"], metrics=expected)


def search_key(spec):
    return (spec["config_id"], spec["phase"], spec["op"], spec["method"])


def evaluate(check):
    try:
        return dict(status="passed", detail=check())
    except Pending as exc:
        return dict(status="waiting", missing=str(exc))
    except (AssertionError, KeyError, TypeError, ValueError, OSError, ZeroDivisionError) as exc:
        return dict(status="blocked", issue=f"{type(exc).__name__}: {exc}")


def search_check(specs, out):
    audit, _ = fg_check.check(specs, out, allow_incomplete=True)
    require(not audit["issues"], json.dumps(audit["issues"], ensure_ascii=False))
    missing = [n for n in fg_check.expected(specs) if not (out / "results" / (n + ".json")).exists()]
    if missing:
        raise Pending(", ".join(missing))
    return audit


def update(manifest, pilot_file, initial_file, out, release_file):
    jobs_list = job_items(manifest); jobs = {j["id"]: j for j in jobs_list}
    pilot = [j["spec"] for j in job_items(pilot_file)]
    initial = read(initial_file)
    identity = dict(jobs=[{k: j[k] for k in ("id", "kind", "spec", "deps", "result")}
                          for j in sorted(jobs_list, key=lambda j: j["id"])],
                    pilot=sorted(pilot, key=lambda s: s["candidate_id"]), initial_ids=sorted(initial["ids"]))
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    audit_file = release_file.with_name(release_file.stem + ".audit.json")
    previous = read(audit_file) if audit_file.exists() else {}
    same_run = previous.get("version") == VERSION and previous.get("manifest_sha256") == signature
    released = set(previous["released_ids"]) if same_run else set()
    passed = set(previous["passed_categories"]) if same_run else set()
    released.update(n for n in initial["ids"] if jobs[n]["kind"] != "search")
    released.update(s["candidate_id"] for s in pilot)
    groups = defaultdict(list)
    for spec in pilot:
        groups[search_key(spec)].append(spec)
    checks = {}
    for key, specs in sorted(groups.items()):
        label = "search/" + "/".join(key)
        checks[label] = evaluate(lambda specs=specs: search_check(specs, out))
        if checks[label]["status"] == "passed":
            passed.add(label)
    checks["pilot_all"] = evaluate(lambda: search_check(pilot, out))
    if checks["pilot_all"]["status"] == "passed":
        passed.add("pilot_all")
    config_ids = sorted({j["spec"]["config_id"] for j in jobs_list if j["kind"] == "fault_select"})
    for cid in config_ids:
        for kind, fn in (("full", full_ready), ("repair", repair_check), ("joint", joint_check)):
            label = kind + "/" + cid
            checks[label] = evaluate(lambda cid=cid, fn=fn: fn(jobs, out, cid))
            if checks[label]["status"] == "passed":
                passed.add(label)
    watos = next(s["config_id"] for s in pilot if s["hardware_profile"] == "watos")
    for job in jobs_list:
        s = job["spec"]; cid = s["config_id"]
        if job["kind"] == "search":
            source_key = search_key(s)
            if s["family"] == "bandwidth":
                source_key = (watos, *source_key[1:])
            category = "search/" + "/".join(source_key)
            if category in passed and (s["family"] != "full" or "full/" + cid in passed):
                released.add(job["id"])
        elif job["kind"].startswith("fault_") and "repair/" + cid in passed:
            released.add(job["id"])
        elif job["kind"].startswith("joint") and "joint/" + s.get("source_config_id", cid) in passed:
            released.add(job["id"])
    require(released <= set(jobs), "release contains unknown jobs")
    complete = len(released) == len(jobs) and "pilot_all" in passed
    release = dict(all=complete, ids=sorted(released))
    audit = dict(version=VERSION, manifest_sha256=signature, updated_utc=datetime.now(timezone.utc).isoformat(),
                 jobs=len(jobs), released=len(released), held=len(jobs)-len(released), all=complete,
                 released_ids=release["ids"], passed_categories=sorted(passed), checks=checks,
                 blocked={k: v for k, v in checks.items() if v["status"] == "blocked"},
                 note="Passed categories and released IDs are cumulative. Later check issues are recorded without revoking released jobs.")
    write_json(audit_file, audit)
    write_json(release_file, release)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--initial-release", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    initial = args.initial_release or args.manifest.with_name("initial_release.json")
    while True:
        audit = update(args.manifest, args.pilot, initial, args.out, args.release)
        print(json.dumps({k: audit[k] for k in ("updated_utc", "jobs", "released", "held", "all", "blocked")}, ensure_ascii=False), flush=True)
        if not args.watch or audit["all"]:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
