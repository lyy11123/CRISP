"""Read the 13:53 snapshot and project remaining work without changing jobs."""
import copy
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "e2e"))
import fg_estimate as est

SOURCE = ROOT / "e2e/fg/out/status_1353"
DEST = Path(__file__).resolve().parent
BASE = "G_busybarn_2x3_growth_d2d1"
TARGET = "G_busybarn_3x3_growth_d2d1"
jobs = est.read(SOURCE / "pipeline_jobs.json")
by_id = {j["id"]: j for j in jobs}
observed, rejected = est.observations(jobs, SOURCE)
assert not rejected
status = est.read(SOURCE / "pipeline.status.json")
running = {r["id"]: r for r in status["running"]}
released = set(est.read(SOURCE / "release.pipeline.audit.json")["released_ids"])
selected = {cid: observed[cid + "__selection"]["result"]["operators"] for cid in (BASE, TARGET)}


def selected_row(cid, op, arm):
    return observed[selected[cid][op][arm]["candidate_id"]]["result"]


ratios = {}
for op in selected[TARGET]:
    for arm in selected[TARGET][op]:
        a, b = (selected_row(cid, op, arm) for cid in (BASE, TARGET))
        ratios[(op, arm)] = dict(backend=sum(b["backend_seconds"]) / sum(a["backend_seconds"]),
                                search=b["search_seconds"] / a["search_seconds"],
                                hops=b["fg_metrics"]["graph_hops"] / a["fg_metrics"]["graph_hops"])
        pre = TARGET + "__pre_" + op + "_" + arm
        if pre in observed:
            ratios[(op, arm)]["measured_pre"] = observed[pre]["seconds"] / observed[pre.replace(TARGET, BASE)]["seconds"]


def joint_scale(spec):
    p_op = spec.get("p_op", "mha")
    terms = [("pf_" + p_op, 1)]
    if spec.get("stream") == 1:
        terms = []
    if spec.get("stream") != 0:
        terms += [("dc_ln", 1), ("dc_mha", 8), ("dc_proj", 1), ("dc_ffn", 1)]
    def size(cid):
        return sum(selected_row(cid, op, "A3h")["fg_metrics"]["graph_hops"] * n for op, n in terms)
    return size(TARGET) / size(BASE)


def repair_duration(job, case):
    spec = job["spec"]
    base_name = job["id"].replace(TARGET, BASE)
    source = observed[base_name]
    if source["reused"] and case != "all_tiers":
        return source["seconds"], "2x3 same early-stop pattern"
    op = spec["phase_op"]
    scale = ratios[(op, "A3h")]
    sample = source
    if sample["reused"]:
        candidates = [r for n, r in observed.items() if n.startswith(BASE + "__rep_" + op + "_")
                      and not r["reused"]]
        if candidates:
            sample = min(candidates, key=lambda r: abs(r["result"]["steps"] - spec["steps"]))
        else:
            r = selected_row(TARGET, op, "A3h")
            seconds = r["search_seconds"] * spec["steps"] / 1000 + sum(r["backend_seconds"]) + 30
            return seconds, "selected healthy SA-per-step and backend proxy; no 2x3 active tier"
    r = sample["result"]
    timing = r["timing"]
    sa = timing["t_L2_search"] * spec["steps"] / r["steps"]
    backend = timing["t_backend_offline"]
    rest = max(0, sample["seconds"] - timing["t_L2_search"] - backend)
    backend_scale = scale.get("measured_pre", scale["backend"])
    seconds = sa * scale["search"] + backend * backend_scale + rest * scale["hops"]
    return seconds, "2x3 L2 components scaled separately; latest matching pre ratio calibrates backend part"


def projected(job, case):
    name, spec, kind = job["id"], job["spec"], job["kind"]
    if name in observed:
        return 0., "complete"
    source_name = name.replace(TARGET, BASE)
    source = observed[source_name]
    elapsed = running.get(name, {}).get("seconds", 0)
    if kind == "search":
        fn = min if case == "fast" else max if case == "all_tiers" else statistics.mean
        same = [r for n, r in observed.items() if n.startswith(TARGET + "__full_")
                and all(r["result"].get(k) == spec.get(k) for k in ("phase", "op", "method", "g"))]
        if same:
            total = fn([r["seconds"] for r in same]) * (1.1 if case == "all_tiers" else 1)
            return max(60, total - elapsed), "matching 3x3 FULL duration, same initialization"
        healthy = [r for n, r in observed.items() if n.startswith(TARGET + "__healthy_")
                   and all(r["result"].get(k) == spec.get(k) for k in ("phase", "op", "method", "g"))]
        base_healthy = [r["seconds"] for n, r in observed.items() if n.startswith(BASE + "__healthy_")
                        and all(r["result"].get(k) == spec.get(k) for k in ("phase", "op", "method", "g"))]
        base_full = [r["seconds"] for n, r in observed.items() if n.startswith(BASE + "__full_")
                     and all(r["result"].get(k) == spec.get(k) for k in ("phase", "op", "method", "g"))]
        correction = fn(base_full) / statistics.mean(base_healthy)
        total = fn([r["seconds"] for r in healthy]) * correction * (1.1 if case == "all_tiers" else 1)
        return max(60, total - elapsed), "3x3 same-g healthy duration with measured 2x3 FULL/healthy ratio"
    if kind == "fault_rep":
        total, basis = repair_duration(job, case)
    elif kind in ("fault_pre", "fault_l1"):
        scale = ratios[(spec["phase_op"], spec["start_arm"])]
        measured_pre = TARGET + "__pre_" + spec["phase_op"] + "_" + spec["start_arm"]
        base_pre = measured_pre.replace(TARGET, BASE)
        multiplier = (observed[measured_pre]["seconds"] / observed[base_pre]["seconds"]
                      if measured_pre in observed else scale["backend"])
        total, basis = source["seconds"] * multiplier, "2x3 pre/L1 scaled by observed pre ratio or selected backend ratio"
    elif kind.startswith("joint") and kind != "joint_reference":
        same = [r for n, r in observed.items() if n.startswith(TARGET + "__") and by_id[n]["kind"] == kind
                and all(by_id[n]["spec"].get(k) == spec.get(k) for k in ("backend", "policy", "stream", "p_op"))]
        if same:
            fn = min if case == "fast" else max if case == "all_tiers" else statistics.mean
            total, basis = fn([r["seconds"] for r in same]), "matching 3x3 joint kind/backend/policy/stream; healthy sample used for pending faults"
        else:
            total, basis = source["seconds"] * joint_scale(spec), "2x3 formal joint task scaled by selected source graph-hop ratio"
    else:
        total, basis = source["seconds"], "2x3 same bookkeeping task"
    if case == "fast":
        total *= .8
    elif case == "all_tiers":
        total *= 1.2
    return max(2, total - elapsed), basis


# Reflect the actual release checks in addition to each task's data dependencies.
gated_jobs = copy.deepcopy(jobs)
gate_counts = {"repair": 0, "joint": 0}
for j in gated_jobs:
    if j["id"] not in released:
        if j["kind"].startswith("fault_"):
            j["deps"].append(TARGET + "__stop_pf_mha_PF4_300")
            gate_counts["repair"] += 1
        elif j["kind"].startswith("joint"):
            j["deps"].append(TARGET + "__joint_healthy_CRISP_g0.5_s1")
            gate_counts["joint"] += 1

cases = {}
for case in ("fast", "central", "all_tiers"):
    rows = {j["id"]: projected(j, case) for j in jobs}
    durations = {name: row[0] for name, row in rows.items()}
    projection = est.schedule(gated_jobs, durations, 128, running)
    # FULL search proceeds independently of the repair/joint release checks.
    full_left = max(durations[j["id"]] for j in jobs if j["kind"] == "search") / 3600
    cases[case] = dict(**projection, full_search_hours=full_left,
                       remaining_process_hours=sum(durations.values()) / 3600,
                       jobs=[dict(id=n, remaining_seconds=t, basis=b) for n, (t, b) in rows.items() if t])

report = dict(snapshot="2026-11-13 13:53:54 Asia/Shanghai", completed=len(observed), total=len(jobs),
              running=len(running), held=len(jobs)-len(released), added_release_dependencies=gate_counts,
              ratios={"/".join(k): v for k, v in ratios.items()}, scenarios=cases,
              assumptions=["Fast and central reuse the observed 2x3 stopping pattern; this is a scenario, not a known 3x3 result.",
                           "All-tiers runs every remaining 30/100/300 tier and adds 20 percent to non-search extrapolations.",
                           "Remaining FULL LN g1/MHA classes have no completed same-class 3x3 sample; LN g0 is not substituted for g1.",
                           "Non-search tasks use completed 3x3 samples when available, otherwise real 2x3 runs; pending work is never assigned zero.",
                           "Repair and joint release-check dependencies are included; maximum concurrency 128.",
                           "No memory throttling, future retry, archive/download time or reporting time is included."])
est.write(DEST / "remaining_scenarios.json", report)
print(json.dumps({k: {a: b for a, b in v.items() if a != "jobs"} for k,v in cases.items()}, indent=2))
