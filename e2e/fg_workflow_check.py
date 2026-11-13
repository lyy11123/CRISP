"""Independent arithmetic checks of the saved F/G workflow report."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from e_run import ROOT, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def same(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6), (a, b)


def check_backend_samples(records, faults, tables):
    samples = read(tables / "fault_backend_samples.json")
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["job_id"], sample["state"]].append(sample)
        same(sample["total_minus_unhidden_components_ns"], sample["backend_ns"]
             - sample["unhidden_compute_ns"] - sample["unhidden_communication_ns"])
    for (job, state), rows in groups.items():
        assert len(rows) == 10 and {r["backend_seed"] for r in rows} == set(range(1, 11))
        result = records[job]
        if state == "FULL":
            reference = sum(b["cycles"] for b in result["backend"]) / 10
        else:
            key = state + "_backend_mean" if state in ("R0", "L1") else "backend_mean"
            reference = result[key]
        same(sum(r["backend_ns"] for r in rows) / 10, reference)
    raw_count = sum(r["source_format"] == "raw" for r in samples)
    full_count = sum(r["state"] == "FULL" for r in samples)
    assert raw_count == 9690 and full_count == 400
    early_stop = []
    for row in faults:
        assert row["backend_sample_count"] == 10 * row["n_sa_states"]
        if row["scheme"] != "EIR":
            continue
        total = row["selected_t_decision_seconds_mean"]
        assert total is not None and total >= 0
        assert row["selected_t_finalize_seconds_mean"] >= 0
        assert row["selected_t_backend_offline_seconds_mean"] >= 0
        parts = ("t_R0", "t_L1") if row["stop_level"] in ("R0", "L1") else (
            "t_R0", "t_L1", "t_calibration", "t_L2_search", "t_finalize")
        same(total, sum(row["selected_" + part + "_seconds_mean"] for part in parts))
        if row["stop_level"] in ("R0", "L1"):
            early_stop.append(total)
    assert len(early_stop) == 77
    return dict(raw_backend_samples=raw_count, FULL_backend_samples=full_count,
                state_means_checked=len(groups), R0_L1_decision_times_checked=len(early_stop),
                R0_L1_decision_seconds_min=min(early_stop), R0_L1_decision_seconds_max=max(early_stop))


def check(run_out, tables):
    records = {f.stem: read(f) for f in (run_out / "results").glob("*.json")}
    faults = read(tables / "fault_summary.json")
    layers = read(tables / "fault_layers.json")
    joints = read(tables / "joint_runs.json")
    joint_summary = read(tables / "joint_summary.json")
    bandwidth = read(tables / "bandwidth_joint.json")
    assert len(faults) == 424 and len(layers) == 53 and len(joints) == 609 and len(joint_summary) == 203
    assert len(bandwidth) == 12
    examples = {}
    # WATOS/prefill MHA/PF4: independently read each stopped-tier seed.
    cid = "F_watos_2x2_fixed_d2d1"
    stop = records[cid + "__stop_pf_mha_PF4_300"]
    five = [records[f"{cid}__rep_pf_mha_PF4_{stop['level']}_r{seed}"] for seed in (123, 1, 2, 3, 4)]
    means = [r["backend_mean"] for r in five]
    median = sorted(means)[2]
    representative = min((r for r in five if r["backend_mean"] == median), key=lambda r: r["seed"])
    row = next(r for r in faults if r["config_id"] == cid and r["phase_op"] == "pf_mha"
               and r["fault_id"] == "PF4" and r["scheme"] == "EIR")
    same(row["backend_mean_ns"], sum(means) / 5)
    same(row["representative_backend_mean_ns"], median)
    assert row["representative_job"] == representative["job_id"]
    before = records[cid + "__pre_pf_mha_A3h"]["backend_mean"]
    same(row["backend_change_vs_healthy"], sum(means) / 5 / before - 1)
    examples["WATOS_pf_mha_PF4"] = dict(seed_means_ns=means, mean_ns=row["backend_mean_ns"],
        representative_ns=median, source_job=representative["job_id"], healthy_ns=before)
    # Decode contains one external LN; FFN already includes its internal LN.
    group = [r for r in faults if r["config_id"] == cid and r["fault_id"] == "PF4" and r["scheme"] == "EIR"]
    op = {(r["phase"], r["op"]): r for r in group}
    p = 2 * op["prefill", "ln"]["backend_mean_ns"] + 8 * op["prefill", "mha"]["backend_mean_ns"] + op["prefill", "proj"]["backend_mean_ns"] + op["prefill", "ffn"]["backend_mean_ns"]
    d = op["decode", "ln"]["backend_mean_ns"] + 8 * op["decode", "mha"]["backend_mean_ns"] + op["decode", "proj"]["backend_mean_ns"] + op["decode", "ffn"]["backend_mean_ns"]
    layer = next(r for r in layers if r["config_id"] == cid and r["fault_id"] == "PF4" and r["scheme"] == "EIR")
    same(layer["prefill_ns"], p); same(layer["decode_ns"], d)
    same(layer["request_2048_ns"], p + 2048 * d)
    examples["WATOS_PF4_EIR_layer"] = dict(prefill_ns=p, decode_ns=d, request_2048_ns=p + 2048 * d)
    # All terminal EIR choices: recompute epsilon, tier improvement and median representative.
    stop_checks = 0
    for final in (r for r in records.values() if r["kind"] == "fault_stop" and r["spec"]["steps"] == 300):
        s = final["spec"]; prefix = f"{s['config_id']}__"
        opid, fault = s["phase_op"], s["fault_id"]
        l1 = records[prefix + f"l1_{opid}_A3h_{fault}"]
        limit = l1["T_pre"] * 1.02
        level = "R0" if l1["T_R0"] <= limit else "L1"
        t = l1["T_" + level]
        stopped = t <= limit
        reason = "within_epsilon" if stopped else "continue"
        source = l1["job_id"]
        for step in (0, 30, 100, 300):
            if step and not stopped:
                selected = [records[prefix + f"rep_{opid}_{fault}_s{step}_r{seed}"] for seed in (123, 1, 2, 3, 4)]
                assert all(r["status"] == "ok" for r in selected)
                next_t = sum(r["T"] for r in selected) / 5
                epsilon = next_t <= limit
                plateau = step > 30 and (t - next_t) / t < 0.02
                reason = "within_epsilon" if epsilon else "tier_gain" if plateau else "budget" if step == 300 else "continue"
                stopped = epsilon or plateau or step == 300
                level, t = f"s{step}", next_t
                med = sorted(r["backend_mean"] for r in selected)[2]
                source = min((r for r in selected if r["backend_mean"] == med), key=lambda r: r["seed"])["job_id"]
            elif step:
                assert all(records[prefix + f"rep_{opid}_{fault}_s{step}_r{seed}"]["status"] == "reused" for seed in (123, 1, 2, 3, 4))
            actual = records[prefix + f"stop_{opid}_{fault}_{step}"]
            assert (actual["level"], actual["reason"], actual["stop"], actual["source_job"]) == (level, reason, stopped, source)
            same(actual["T"], t)
            stop_checks += 1
    # Joint absolute launch time and paired numbers from referenced standalone results.
    for row in joints:
        source = records[row["job_id"]]; spec = source["spec"]
        ref = records[spec["reference_job"]]
        psolo = records[spec["P_solo"]]["simulation"]["streams"]["0"]["C"]
        dsolo = records[spec["D_solo"]]["simulation"]["streams"]["1"]["C"]
        tau = round(ref["KV_start"] + spec["phase"] * (ref["KV_end"] - ref["KV_start"]))
        p = source["simulation"]["streams"]["0"]["C"]
        d = source["simulation"]["streams"]["1"]["C"] - tau
        same(row["paired_delta_P_ns"], p - psolo)
        same(row["D_self_amplification"], d / dsolo)
        same(row["Lhat_D"], (d - 1.5 * ref["T_D"]) / ref["T_D"])
        assert row["tau_ns"] == tau
    assert any(r["paired_delta_P_ns"] < 0 for r in joints)
    for row in joints:
        intersection = min(row["KV_end_ns"], row["actual_D_end_ns"]) - max(row["KV_start_ns"], row["actual_D_start_ns"])
        same(row["KV_D_interval_overlap_ns"], max(intersection, 0))
    for group in joint_summary:
        rows = [r for r in joints if all(r[k] == group[k] for k in ("config_id", "condition", "arm", "phase"))]
        assert len(rows) == 3 and {r["seed"] for r in rows} == {1, 2, 3}
        same(group["D_latency_ns_mean"], sum(r["D_latency_ns"] for r in rows) / 3)
    for row in bandwidth:
        base = next(r for r in bandwidth if r["arm"] == row["arm"] and r["d2d_scale"] == 1)
        same(row["D_latency_ns_ratio_to_1x"], row["D_latency_ns_mean"] / base["D_latency_ns_mean"])
        assert row["source_config_id"] == cid and row["condition"] == "healthy" and row["phase"] == 0.5
    assert all(r["fault_id"] == "PF4" for r in faults if r["scheme"] == "A3h-FULL")
    assert all(r["FULL_backend_mean_ns"] is None for r in faults if r["fault_id"] != "PF4")
    checks = read(tables / "workflow_checks.json")
    raw_checks = check_backend_samples(records, faults, tables) if checks["raw_backend_samples_read"] else None
    return dict(status="passed", stopping_rows_checked=stop_checks, fault_scheme_rows=len(faults), fault_layers=len(layers),
        joint_runs_checked=len(joints), joint_phase_groups_checked=len(joint_summary), bandwidth_groups_checked=len(bandwidth),
        FULL_only_PF4=True, negative_paired_deltas_preserved=True,
        zero_KV_overlap_count=sum(r["KV_D_interval_overlap_ns"] == 0 for r in joints),
        raw_backend_checks=raw_checks, examples=examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-out", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=ROOT / "e2e/fg/tables")
    parser.add_argument("--report", type=Path, default=ROOT / "e2e/fg/evidence/workflow_report_checks.json")
    args = parser.parse_args()
    result = check(args.run_out, args.tables)
    write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
