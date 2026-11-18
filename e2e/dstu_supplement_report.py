"""Read-only S256 paired statistics, whole-strategy oracle, and P/D Pareto tables."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "e2e/dstu_supplement"
TRANSPORTS = ("BB0", "A256", "S256")
TIE_ORDER = ("S256", "A256", "BB0")
CASE = ("scope", "model", "hardware", "mapping_state", "phase", "seed")
COMPARISONS = (("S256", "A256"), ("S256", "BB0"), ("A256", "BB0"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".part")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def save_json(path, value):
    save_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: canonical(value) if isinstance(value, (list, dict)) else value
                     for key, value in row.items()} for row in rows)
    save_text(path, "\ufeff" + output.getvalue())


def statistics_row(values):
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values), improved=sum(x < 0 for x in values),
                tied=sum(x == 0 for x in values), regressed=sum(x > 0 for x in values))


def key(row):
    return tuple(row[field] for field in CASE)


def planned_cases():
    states = [("S", "gpt", hardware, state) for hardware in ("wsc", "face")
              for state in ("A1R0", "A1EIR", "A3hR0", "A3hEIR")]
    states += [(scope, model, "busybarn", state) for scope, model in (("H", "llama"), ("I", "qwen3moe"))
               for state in ("healthy_A3h", "D_A3hEIR")]
    return {base + (phase, seed) for base in states for phase in (.1, .5, .9) for seed in (1, 2, 3)}


def finish_row(row):
    cp, latency = row["P_completion_ns"], row["D_latency_ns"]
    tp, td = row["reference_P_ns"], row["reference_D_ns"]
    assert cp > 0 and latency > 0 and tp > 0 and td > 0
    assert math.isclose(row["Lhat_P"], cp / tp - 1.5, abs_tol=1e-12)
    assert math.isclose(row["Lhat_D"], latency / td - 1.5, abs_tol=1e-12)
    assert row["Lhat_max"] == max(row["Lhat_P"], row["Lhat_D"])
    row.update(P_deadline_ns=1.5 * tp, D_deadline_absolute_ns=row["tau_ns"] + 1.5 * td,
               P_deadline_missed=cp > 1.5 * tp, D_deadline_missed=latency > 1.5 * td,
               either_deadline_missed=cp > 1.5 * tp or latency > 1.5 * td,
               Lhat_max_dominant="P" if row["Lhat_P"] > row["Lhat_D"] else "D" if row["Lhat_D"] > row["Lhat_P"] else "equal")
    if row["KV_start_ns"] is not None:
        row["KV_D_envelope_overlap_ns"] = max(0, min(row["KV_end_ns"], row["tau_ns"] + latency)
                                                 - max(row["KV_start_ns"], row["tau_ns"]))
    else:
        row["KV_D_envelope_overlap_ns"] = None
    return row


def collect_new(out):
    rows, input_files, rank_rows = [], [], []
    for path in sorted((Path(out) / "results").glob("*.json")):
        result = read(path)
        if result.get("status") not in ("ok", "reused"):
            continue
        spec = result["spec"]
        if spec.get("kind") == "prepare":
            for phase, check in result["rank_checks"].items():
                rank_rows.append(dict(scope=spec["scope"], model=spec["model"], hardware=spec["hardware_profile"],
                                      mapping_state=spec["mapping_state"], phase=float(phase),
                                      source_result=path.resolve().relative_to(ROOT).as_posix(), **check))
            continue
        if spec.get("run_type") != "joint":
            continue
        if spec.get("rank_mode", "frozen") != "frozen":
            continue
        inner = result["resolved_simulation_spec"]
        metric = result["metrics"]
        identity = result["evaluation_identity"]
        reference = inner["reference"]
        window = result["kv_window"]
        psolo_file = Path(out) / "results" / (spec["P_solo"] + ".json")
        psolo = read(psolo_file)
        solo_window = psolo["kv_window"]
        source = path.resolve().relative_to(ROOT).as_posix()
        row = dict(scope=spec["scope"], model=spec["model"], hardware=spec["hardware_profile"],
                   mapping_state=spec["mapping_state"], phase=float(spec["phase"]), seed=int(spec["seed"]),
                   transport=spec["transport"], job_id=spec["job_id"], status=result["status"],
                   source_result=source, source_sha256=sha(path),
                   graph_files_sha256=inner["graph_files_sha256"],
                   platform_sha256=identity["platform_sha256"],
                   fresh_protocol=result["fresh_protocol"],
                   initial_order_sha256=result["initial_order_sha256"],
                   evaluation_identity=identity, reference=reference, frozen_rank=inner["rank"],
                   tau_ns=inner["tau"], reference_P_ns=reference["T_P"], reference_D_ns=reference["T_D"],
                   P_completion_ns=metric["P_completion"], D_latency_ns=metric["D_latency"],
                   P_solo_ns=metric["P_solo"], D_solo_ns=metric["D_solo"],
                   Lhat_P=metric["Lhat_P"], Lhat_D=metric["Lhat_D"], Lhat_max=metric["Lhat_max"],
                   KV_start_ns=window[0] if window else None, KV_end_ns=window[1] if window else None,
                   P_solo_KV_start_ns=solo_window[0] if solo_window else None,
                   P_solo_KV_end_ns=solo_window[1] if solo_window else None,
                   P_solo_source=psolo_file.resolve().relative_to(ROOT).as_posix(),
                   P_solo_source_sha256=sha(psolo_file),
                   gate_record=spec["gate_record"], gate_sha256=spec["gate_sha256"],
                   started_at_utc=result["started_at_utc"],
                   backend_calls=result["backend_calls"])
        assert row["transport"] in TRANSPORTS
        assert row["fresh_protocol"] == "fresh_transform_v1" and row["initial_order_sha256"]
        rows.append(finish_row(row))
        input_files.append(dict(path=source, sha256=sha(path)))
    return rows, input_files, rank_rows


def full_groups(rows, legacy=False):
    grouped = defaultdict(dict)
    for row in rows:
        assert row["transport"] not in grouped[key(row)], (key(row), row["transport"])
        grouped[key(row)][row["transport"]] = row
    complete, missing = {}, []
    for case, values in sorted(grouped.items()):
        if set(values) != set(TRANSPORTS):
            missing.append(dict(zip(CASE, case), missing=sorted(set(TRANSPORTS) - set(values))))
            continue
        identities = []
        for row in values.values():
            identity = {field: row[field] for field in ("graph_files_sha256", "platform_sha256", "reference", "tau_ns", "frozen_rank")}
            if not legacy:
                identity["fresh_protocol"] = row["fresh_protocol"]
            identities.append(canonical(identity))
        assert len(set(identities)) == 1, case
        complete[case] = values
    return complete, missing


def pairs_of(groups):
    pairs = []
    for case, group in groups.items():
        for selected, baseline in COMPARISONS:
            a, b = group[selected], group[baseline]
            row = dict(zip(CASE, case), selected=selected, baseline=baseline,
                       selected_job=a["job_id"], baseline_job=b["job_id"],
                       P_change_pct=100 * (a["P_completion_ns"] / b["P_completion_ns"] - 1),
                       D_change_pct=100 * (a["D_latency_ns"] / b["D_latency_ns"] - 1),
                       Lhat_P_delta=a["Lhat_P"] - b["Lhat_P"], Lhat_D_delta=a["Lhat_D"] - b["Lhat_D"],
                       Lhat_max_delta=a["Lhat_max"] - b["Lhat_max"],
                       selected_Lhat_P=a["Lhat_P"], selected_Lhat_D=a["Lhat_D"], selected_Lhat_max=a["Lhat_max"],
                       baseline_Lhat_P=b["Lhat_P"], baseline_Lhat_D=b["Lhat_D"], baseline_Lhat_max=b["Lhat_max"],
                       selected_Lhat_max_dominant=a["Lhat_max_dominant"], baseline_Lhat_max_dominant=b["Lhat_max_dominant"],
                       selected_P_missed=a["P_deadline_missed"], selected_D_missed=a["D_deadline_missed"],
                       baseline_P_missed=b["P_deadline_missed"], baseline_D_missed=b["D_deadline_missed"],
                       selected_either_missed=a["either_deadline_missed"], baseline_either_missed=b["either_deadline_missed"])
            row["both_PD_improved"] = row["P_change_pct"] < 0 and row["D_change_pct"] < 0
            pairs.append(row)
    return pairs


def summaries(pairs, dimensions):
    grouped = defaultdict(list)
    for row in pairs:
        grouped[tuple(row[k] for k in (*dimensions, "selected", "baseline"))].append(row)
    rows = []
    for keys, selected in sorted(grouped.items()):
        result = dict(zip((*dimensions, "selected", "baseline"), keys), n=len(selected))
        for name in ("P_change_pct", "D_change_pct", "Lhat_P_delta", "Lhat_D_delta", "Lhat_max_delta",
                     "selected_Lhat_P", "selected_Lhat_D", "selected_Lhat_max"):
            for label, value in statistics_row([r[name] for r in selected]).items():
                if label != "n" and (not name.startswith("selected_Lhat") or label in ("mean", "median", "minimum", "maximum")):
                    result[name + "_" + label] = value
        for name in ("both_PD_improved", "selected_P_missed", "selected_D_missed", "baseline_P_missed", "baseline_D_missed",
                     "selected_either_missed", "baseline_either_missed"):
            result[name] = sum(row[name] for row in selected)
        rows.append(result)
    return rows


def oracle_and_pareto(groups):
    oracle, pareto = [], []
    for case, group in groups.items():
        minimum = min(r["Lhat_max"] for r in group.values())
        ties = [arm for arm in TIE_ORDER if group[arm]["Lhat_max"] == minimum]
        winner = ties[0]
        selected = group[winner]
        entry = dict(zip(CASE, case), selected_transport=winner, all_tied_transports=ties,
                     selected_job=selected["job_id"], objective="minimum saved Lhat_max across complete executed strategies",
                     P_completion_ns=selected["P_completion_ns"], D_latency_ns=selected["D_latency_ns"],
                     Lhat_P=selected["Lhat_P"], Lhat_D=selected["Lhat_D"], Lhat_max=minimum,
                     Lhat_max_dominant=selected["Lhat_max_dominant"],
                     P_deadline_missed=selected["P_deadline_missed"], D_deadline_missed=selected["D_deadline_missed"],
                     retrospective=True, tie_order=list(TIE_ORDER))
        for baseline in TRANSPORTS:
            entry[f"P_change_vs_{baseline}_pct"] = 100 * (selected["P_completion_ns"] / group[baseline]["P_completion_ns"] - 1)
            entry[f"D_change_vs_{baseline}_pct"] = 100 * (selected["D_latency_ns"] / group[baseline]["D_latency_ns"] - 1)
            entry[f"Lhat_max_vs_{baseline}_delta"] = minimum - group[baseline]["Lhat_max"]
        oracle.append(entry)
        for arm, row in group.items():
            dominators = [other for other, candidate in group.items() if other != arm
                          and candidate["P_completion_ns"] <= row["P_completion_ns"]
                          and candidate["D_latency_ns"] <= row["D_latency_ns"]
                          and (candidate["P_completion_ns"] < row["P_completion_ns"] or candidate["D_latency_ns"] < row["D_latency_ns"])]
            pareto.append(dict(zip(CASE, case), transport=arm, on_pareto_front=not dominators,
                               dominated_by=dominators, P_completion_ns=row["P_completion_ns"], D_latency_ns=row["D_latency_ns"],
                               Lhat_max=row["Lhat_max"], oracle_selected=arm == winner, tied_for_oracle=arm in ties,
                               source_result=row["source_result"], retrospective=True))
    return oracle, pareto


def read_gate_decisions(out):
    """Read choices already fixed by the separate prediction-only job."""
    rows = {}
    for path in sorted((Path(out) / "results").glob("*__gate.json")):
        record = read(path)
        if record["status"] not in ("ok", "passed"):
            continue
        manifest = record["input_manifest"]
        assert record["backend_calls"] == 0
        assert hashlib.sha256(canonical(record["decisions"]).encode()).hexdigest() == record["decision_sha256"]
        reference_file = ROOT / manifest["reference_record"]
        assert sha(reference_file) == manifest["reference_sha256"]
        reference_record = read(reference_file)
        reference = {k: reference_record[k] for k in ("T_P", "T_D", "KV_start", "KV_end")}
        prediction = {k: record["prediction"][k] for k in ("predicted_P_end_ns", "predicted_KV_start_ns", "predicted_KV_end_ns",
                                                          "shared_resource_count", "schedule_sha256", "predictor_seconds",
                                                          "graph_build_seconds", "cpcm_forward_seconds")}
        for decision in record["decisions"]:
            assert decision["input_identity_sha256"] == manifest["identity_sha256"]
            k = tuple(record[field] for field in ("scope", "model", "hardware_profile", "mapping_state")) + (float(decision["phase"]),)
            assert k not in rows
            rows[k] = dict(decision, decision_source=path.resolve().relative_to(ROOT).as_posix(),
                           decision_source_sha256=sha(path), prediction=prediction,
                           frozen_at_utc=record["frozen_at_utc"],
                           task_body_seconds=record["wall_seconds"], peak_rss_gb=record["peak_rss_gb"],
                           graph_files_sha256=manifest["graph_files_sha256"], reference=reference)
    return rows


def strategy_choices(groups, gate_decisions=None):
    """Each choice returns one whole executed strategy; only oracle rules inspect outcomes."""
    decisions = gate_decisions or {}
    rows = []
    for case, group in groups.items():
        definitions = [("Fixed-" + arm, [arm], arm, False, {}) for arm in TRANSPORTS]
        for label, objective, eligible in (
                ("Oracle-min-Lhatmax", "Lhat_max", list(TIE_ORDER)),
                ("Oracle-min-D", "D_latency_ns", list(TIE_ORDER)),
                ("Oracle-min-D-Ple3pct", "D_latency_ns",
                 [arm for arm in TIE_ORDER if group[arm]["P_completion_ns"] <= 1.03 * group["BB0"]["P_completion_ns"]])):
            optimum = min(group[arm][objective] for arm in eligible)
            ties = [arm for arm in eligible if group[arm][objective] == optimum]
            definitions.append((label, eligible, ties[0], True, dict(all_tied_transports=ties, objective=objective,
                                P_limit_ns=1.03 * group["BB0"]["P_completion_ns"] if label == "Oracle-min-D-Ple3pct" else None)))
        if case[:-1] in decisions:
            decision = decisions[case[:-1]]
            assert decision["tau_ns"] == group["BB0"]["tau_ns"]
            assert decision["graph_files_sha256"] == group["BB0"]["graph_files_sha256"]
            assert decision["reference"] == group["BB0"]["reference"]
            assert decision["chosen_transport"] in ("BB0", "S256")
            for executed in group.values():
                assert executed["gate_record"] == decision["decision_source"]
                assert executed["gate_sha256"] == decision["decision_source_sha256"]
                assert datetime.fromisoformat(executed["started_at_utc"]) >= datetime.fromisoformat(decision["frozen_at_utc"])
            definitions.append(("Online-CPCM-Gated", [decision["chosen_transport"]], decision["chosen_transport"], False,
                                dict(decision_source=decision["decision_source"], decision_source_sha256=decision["decision_source_sha256"],
                                     input_identity_sha256=decision["input_identity_sha256"], rule_version=decision["rule_version"],
                                     frozen_at_utc=decision["frozen_at_utc"],
                                     prediction=decision["prediction"],
                                     online_prediction_seconds=decision["prediction"]["predictor_seconds"],
                                     online_graph_build_seconds=decision["prediction"]["graph_build_seconds"],
                                     online_cpcm_forward_seconds=decision["prediction"]["cpcm_forward_seconds"],
                                     online_task_body_seconds=decision["task_body_seconds"],
                                     timing_source="this batch's separate new-process prediction jobs; one prediction reused for three phases and their backend seeds")))
        for label, eligible, chosen, retrospective, metadata in definitions:
            selected = group[chosen]
            row = dict(zip(CASE, case), selection=label, chosen_transport=chosen, eligible_transports=eligible,
                       retrospective=retrospective, chosen_job=selected["job_id"],
                       P_completion_ns=selected["P_completion_ns"], D_latency_ns=selected["D_latency_ns"],
                       Lhat_P=selected["Lhat_P"], Lhat_D=selected["Lhat_D"], Lhat_max=selected["Lhat_max"],
                       Lhat_max_dominant=selected["Lhat_max_dominant"],
                       P_deadline_missed=selected["P_deadline_missed"], D_deadline_missed=selected["D_deadline_missed"],
                       either_deadline_missed=selected["either_deadline_missed"], **metadata)
            for baseline in TRANSPORTS:
                row[f"P_change_vs_{baseline}_pct"] = 100 * (row["P_completion_ns"] / group[baseline]["P_completion_ns"] - 1)
                row[f"D_change_vs_{baseline}_pct"] = 100 * (row["D_latency_ns"] / group[baseline]["D_latency_ns"] - 1)
                row[f"Lhat_max_vs_{baseline}_delta"] = row["Lhat_max"] - group[baseline]["Lhat_max"]
            rows.append(row)
    return rows


def choice_summaries(rows, dimensions=()):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in (*dimensions, "selection"))].append(row)
    result = []
    for key_values, values in sorted(groups.items()):
        row = dict(zip((*dimensions, "selection"), key_values), n=len(values),
                   selected_counts=dict(Counter(r["chosen_transport"] for r in values)),
                   Lhat_P_mean=statistics.mean(r["Lhat_P"] for r in values),
                   Lhat_D_mean=statistics.mean(r["Lhat_D"] for r in values),
                   Lhat_max_mean=statistics.mean(r["Lhat_max"] for r in values),
                   Lhat_max_dominant_counts=dict(Counter(r["Lhat_max_dominant"] for r in values)),
                   P_missed=sum(r["P_deadline_missed"] for r in values), D_missed=sum(r["D_deadline_missed"] for r in values),
                   either_missed=sum(r["either_deadline_missed"] for r in values))
        if row["selection"] == "Online-CPCM-Gated":
            unique = {r["decision_source"]: r for r in values}
            row["unique_prediction_jobs"] = len(unique)
            for field in ("online_prediction_seconds", "online_graph_build_seconds", "online_cpcm_forward_seconds", "online_task_body_seconds"):
                measurements = [r[field] for r in unique.values()]
                row.update({field + "_median": statistics.median(measurements), field + "_minimum": min(measurements),
                            field + "_maximum": max(measurements), field + "_sum_once": sum(measurements)})
        for baseline in TRANSPORTS:
            for stream in ("P", "D"):
                field = f"{stream}_change_vs_{baseline}_pct"
                row.update({field + "_" + name: value for name, value in statistics_row([r[field] for r in values]).items() if name != "n"})
        result.append(row)
    return result


def choice_markdown(rows):
    lines = ["| 选择方法 | 组数 | 选择次数 | P相对original均值 | D相对original均值 | Lhat_P均值 | Lhat_D均值 | Lhat_max均值 | 主导P/D/并列 | 超期P/D/任一 |",
             "|---|---:|---|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        counts = ", ".join(f"{k}:{v}" for k, v in row["selected_counts"].items())
        dominant = row["Lhat_max_dominant_counts"]
        lines.append(f"| {row['selection']} | {row['n']} | {counts} | {row['P_change_vs_BB0_pct_mean']:+.3f}% | "
                     f"{row['D_change_vs_BB0_pct_mean']:+.3f}% | {row['Lhat_P_mean']:.6f} | {row['Lhat_D_mean']:.6f} | "
                     f"{row['Lhat_max_mean']:.6f} | {dominant.get('P',0)}/{dominant.get('D',0)}/{dominant.get('equal',0)} | "
                     f"{row['P_missed']}/{row['D_missed']}/{row['either_missed']} |")
    return lines


def online_window_comparison(groups, decisions):
    """Compare frozen predictions with saved backend windows; never change a choice."""
    samples = []
    for case, arms in groups.items():
        if case[:-1] not in decisions:
            continue
        decision = decisions[case[:-1]]
        bb, chosen = arms["BB0"], arms[decision["chosen_transport"]]
        pred = decision["prediction"]
        tau, shared = decision["tau_ns"], pred["shared_resource_count"]
        predicted_gate = decision["chosen_transport"] == "S256"
        end_error = pred["predicted_KV_end_ns"] - bb["P_solo_KV_end_ns"]
        actual_solo_test = shared > 0 and tau < bb["P_solo_KV_end_ns"]
        actual_joint_test = shared > 0 and tau < bb["KV_end_ns"]
        samples.append(dict(zip(CASE, case), chosen_transport=decision["chosen_transport"], tau_ns=tau,
                            predicted_KV_start_ns=pred["predicted_KV_start_ns"], predicted_KV_end_ns=pred["predicted_KV_end_ns"],
                            shared_resource_count=shared, BB0_solo_KV_start_ns=bb["P_solo_KV_start_ns"],
                            BB0_solo_KV_end_ns=bb["P_solo_KV_end_ns"], BB0_joint_KV_start_ns=bb["KV_start_ns"],
                            BB0_joint_KV_end_ns=bb["KV_end_ns"], chosen_joint_KV_start_ns=chosen["KV_start_ns"],
                            chosen_joint_KV_end_ns=chosen["KV_end_ns"], predicted_minus_BB0_solo_KV_end_ns=end_error,
                            end_prediction_direction="over" if end_error > 0 else "under" if end_error < 0 else "equal",
                            predicted_test_selects_S256=predicted_gate, actual_BB0_solo_end_test_selects_S256=actual_solo_test,
                            actual_BB0_joint_end_test_selects_S256=actual_joint_test,
                            same_as_actual_solo_end_test=predicted_gate == actual_solo_test,
                            same_as_actual_joint_end_test=predicted_gate == actual_joint_test,
                            BB0_KV_D_envelope_overlap_ns=bb["KV_D_envelope_overlap_ns"],
                            chosen_KV_D_envelope_overlap_ns=chosen["KV_D_envelope_overlap_ns"],
                            source_gate=decision["decision_source"], source_BB0_joint=bb["source_result"],
                            source_BB0_solo=bb["P_solo_source"], source_chosen_joint=chosen["source_result"],
                            analysis_use="post-hoc prediction/window comparison; frozen action is unchanged"))
    by_phase = defaultdict(list)
    for row in samples:
        by_phase[tuple(row[field] for field in CASE[:-1])].append(row)
    rows = []
    for phase_key, values in sorted(by_phase.items()):
        assert len({r["chosen_transport"] for r in values}) == 1
        first = values[0]
        row = dict(zip(CASE[:-1], phase_key), backend_seeds=[r["seed"] for r in values], n=len(values),
                   chosen_transport=first["chosen_transport"], tau_ns=first["tau_ns"],
                   predicted_KV_start_ns=first["predicted_KV_start_ns"], predicted_KV_end_ns=first["predicted_KV_end_ns"],
                   shared_resource_count=first["shared_resource_count"],
                   end_prediction_direction_counts=dict(Counter(r["end_prediction_direction"] for r in values)),
                   actual_solo_end_test_matches=sum(r["same_as_actual_solo_end_test"] for r in values),
                   actual_joint_end_test_matches=sum(r["same_as_actual_joint_end_test"] for r in values),
                   BB0_actual_envelope_overlap_count=sum(r["BB0_KV_D_envelope_overlap_ns"] > 0 for r in values),
                   chosen_actual_envelope_overlap_count=sum(r["chosen_KV_D_envelope_overlap_ns"] > 0 for r in values),
                   source_gate=first["source_gate"])
        for name in ("BB0_solo_KV_start_ns", "BB0_solo_KV_end_ns", "BB0_joint_KV_start_ns", "BB0_joint_KV_end_ns",
                     "chosen_joint_KV_start_ns", "chosen_joint_KV_end_ns", "predicted_minus_BB0_solo_KV_end_ns",
                     "BB0_KV_D_envelope_overlap_ns", "chosen_KV_D_envelope_overlap_ns"):
            numbers = [r[name] for r in values]
            row.update({name + "_mean": statistics.mean(numbers), name + "_minimum": min(numbers), name + "_maximum": max(numbers)})
        rows.append(row)
    return rows, samples


def online_costs(decisions):
    unique = {}
    for case, decision in decisions.items():
        name = decision["decision_source"]
        if name in unique:
            unique[name]["phases"].append(case[-1])
            continue
        prediction = decision["prediction"]
        unique[name] = dict(zip(CASE[:-2], case[:-1]), phases=[case[-1]],
            source=name, source_sha256=decision["decision_source_sha256"],
            prediction_seconds=prediction["predictor_seconds"], graph_build_seconds=prediction["graph_build_seconds"],
            cpcm_forward_seconds=prediction["cpcm_forward_seconds"], task_body_seconds=decision["task_body_seconds"],
            peak_rss_gb=decision["peak_rss_gb"], predicted_KV_end_ns=prediction["predicted_KV_end_ns"],
            cost_scope="one saved-P prediction per model/hardware/state; shared by phases and seeds; task body excludes interpreter startup")
    return list(unique.values())


def system_pairs(groups):
    """Pair S's M/R/Transport changes on the common release/reference and PF4."""
    inputs = {(row["hardware"], row["mapping_state"], row["transport"], row["phase"], row["seed"]): row
              for arms in groups.values() for row in arms.values() if row["scope"] == "S"}
    pairs, full = [], []
    def label(state, arm):
        return ("1" if state.startswith("A3h") else "0") + ("1" if state.endswith("EIR") else "0") + ("0" if arm == "BB0" else "1") + ("-" + arm if arm != "BB0" else "")
    def paired(a, b, effect):
        assert a["reference"] == b["reference"] and a["tau_ns"] == b["tau_ns"] and a["platform_sha256"] == b["platform_sha256"]
        result = dict(hardware=a["hardware"], phase=a["phase"], seed=a["seed"], effect=effect,
                      start_state=b["mapping_state"], end_state=a["mapping_state"], transport=a["transport"],
                      selected=label(a["mapping_state"], a["transport"]), baseline=label(b["mapping_state"], b["transport"]),
                      selected_job=a["job_id"], baseline_job=b["job_id"],
                      P_change_pct=100 * (a["P_completion_ns"] / b["P_completion_ns"] - 1),
                      D_change_pct=100 * (a["D_latency_ns"] / b["D_latency_ns"] - 1))
        for metric in ("Lhat_P", "Lhat_D", "Lhat_max"):
            result.update({metric + "_delta": a[metric] - b[metric], "selected_" + metric: a[metric], "baseline_" + metric: b[metric]})
        result["both_PD_improved"] = result["P_change_pct"] < 0 and result["D_change_pct"] < 0
        for name, field in (("P_missed", "P_deadline_missed"), ("D_missed", "D_deadline_missed"), ("either_missed", "either_deadline_missed")):
            result["selected_" + name], result["baseline_" + name] = a[field], b[field]
        return result
    for hw, state, transport, phase, seed in sorted(inputs):
        row = inputs[hw, state, transport, phase, seed]
        common = (hw, "A1R0", "BB0", phase, seed)
        if common in inputs:
            full.append(paired(row, inputs[common], "full_vs_000"))
        if state.startswith("A3h"):
            source = (hw, state.replace("A3h", "A1"), transport, phase, seed)
            if source in inputs:
                pairs.append(paired(row, inputs[source], "M"))
        if state.endswith("EIR"):
            source = (hw, state.replace("EIR", "R0"), transport, phase, seed)
            if source in inputs:
                pairs.append(paired(row, inputs[source], "R"))
        if transport != "BB0":
            source = (hw, state, "BB0", phase, seed)
            if source in inputs:
                pairs.append(paired(row, inputs[source], "Transport"))
    return pairs, full


def interference(groups):
    rows = []
    for case, group in groups.items():
        original = group["BB0"]
        for arm, row in group.items():
            for stream, metric in (("P", "P_completion_ns"), ("D", "D_latency_ns")):
                solo = row[f"{stream}_solo_ns"]
                old_solo = original[f"{stream}_solo_ns"]
                o, j, j0 = solo - old_solo, row[metric] - solo, original[metric] - old_solo
                change = row[metric] - original[metric]
                assert math.isclose(change, o + j - j0, abs_tol=1e-7)
                rows.append(dict(zip(CASE, case), transport=arm, stream=stream, joint_ns=row[metric], solo_ns=solo,
                                 original_joint_ns=original[metric], original_solo_ns=old_solo,
                                 service_change_O_ns=o, interference_J_ns=j, original_interference_J_ns=j0,
                                 joint_change_ns=change, decomposition_error_ns=change - (o + j - j0),
                                 original_interference_ratio=j0 / old_solo,
                                 KV_start_ns=row["KV_start_ns"], KV_end_ns=row["KV_end_ns"], tau_ns=row["tau_ns"],
                                 KV_D_envelope_overlap_ns=row["KV_D_envelope_overlap_ns"]))
    return rows


def group_title(row, fields):
    return "/".join(str(row[field]) for field in fields) or "全部"


def markdown_statistics(rows, fields=()):
    lines = ["| 范围 | 对比 | 配对数 | P 均值 | D 均值 | P/D 同时改善 | D 中位数 [最小,最大] |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {group_title(row, fields)} | {row['selected']}/{row['baseline']} | {row['n']} | "
                     f"{row['P_change_pct_mean']:+.3f}% | {row['D_change_pct_mean']:+.3f}% | {row['both_PD_improved']}/{row['n']} | "
                     f"{row['D_change_pct_median']:+.3f}% [{row['D_change_pct_minimum']:+.3f}%, {row['D_change_pct_maximum']:+.3f}%] |")
    return lines


def oracle_text(oracle, pareto):
    count = Counter(r["selected_transport"] for r in oracle)
    fronts = Counter(r["transport"] for r in pareto if r["on_pareto_front"])
    lines = ["从每个完整三策略组中，按保存的最大归一化迟约 Lhat_max 最小值选择一整条已执行结果；P、D 均取该行，"
             "不分别拼接两个最优值。精确并列按 S256、A256、BB0 顺序展示一行，表中保留全部并列。",
             f"共 {len(oracle)} 组：" + "、".join(f"{arm} 选中 {count[arm]} 次" for arm in TIE_ORDER) + "。",
             "这是知道三种策略结果后的事后 oracle，表示策略取舍空间；本列不表示在线控制器的成绩。",
             "P/D Pareto 同时最小化 P 完成时间和 D 延迟；一个策略在两者都不差且至少一者更小时，才支配另一策略。"
             + "、".join(f"{arm} 在前沿 {fronts[arm]}/{len(oracle)} 次" for arm in TRANSPORTS) + "。"]
    if oracle:
        lines += ["", "| oracle 相对固定策略 | P 配对均值 | D 配对均值 | Lhat_max 平均差 |", "|---|---:|---:|---:|"]
        for arm in TRANSPORTS:
            lines.append(f"| {arm} | {statistics.mean(r[f'P_change_vs_{arm}_pct'] for r in oracle):+.3f}% | "
                         f"{statistics.mean(r[f'D_change_vs_{arm}_pct'] for r in oracle):+.3f}% | "
                         f"{statistics.mean(r[f'Lhat_max_vs_{arm}_delta'] for r in oracle):+.6f} |")
    return lines


def existing(destination):
    destination = Path(destination)
    source = ROOT / "e2e/post_fg/tables/dstu_runs.csv"
    raw_rows = list(csv.DictReader(source.open(encoding="utf-8-sig", newline="")))
    selected = [r for r in raw_rows if r["arm"] in TRANSPORTS]
    solos, rows, sources = {}, [], []
    for row in selected:
        path = ROOT / row["source_result"]
        assert sha(path) == row["source_sha256"]
        result = read(path)
        assert result["status"] == "ok"
        sources.append(dict(path=row["source_result"], sha256=row["source_sha256"]))
        if row["run_type"] == "solo":
            stream = int(row["stream"])
            completion = result["completion"][str(stream)]
            assert float(row["completion_ns"]) == completion
            solos[row["hardware"], row["condition"], row["arm"], stream, int(row["seed"])] = completion
    for row in selected:
        if row["run_type"] != "joint":
            continue
        result = read(ROOT / row["source_result"])
        spec, metric = result["spec"], result["metrics"]
        for output, field in (("P_completion_ns", "P_completion"), ("D_latency_ns", "D_latency"),
                              ("Lhat_P", "Lhat_P"), ("Lhat_D", "Lhat_D"), ("Lhat_max", "Lhat_max")):
            assert float(row[output]) == metric[field]
        solo = {stream: solos[row["hardware"], row["condition"], row["arm"], stream, int(row["seed"])] for stream in (0, 1)}
        assert float(row["P_solo_ns"]) == solo[0] and float(row["D_solo_ns"]) == solo[1]
        normalized = dict(scope="POST_FG_36", model="gpt", hardware=row["hardware"],
                          mapping_state="healthy_A3h" if row["condition"] == "healthy" else "PF4_A3hEIR",
                          phase=float(row["phase"]), seed=int(row["seed"]), transport=row["arm"], job_id=row["job_id"],
                          source_result=row["source_result"], source_sha256=row["source_sha256"],
                          graph_files_sha256=spec["graph_files_sha256"], platform_sha256=result["evaluation_identity"]["platform_sha256"],
                          reference=spec["reference"], frozen_rank=spec["rank"], tau_ns=spec["tau"],
                          fresh_protocol=result.get("fresh_protocol"), initial_order_sha256=result.get("initial_order_sha256"),
                          evaluation_identity=result["evaluation_identity"],
                          reference_P_ns=spec["reference"]["T_P"], reference_D_ns=spec["reference"]["T_D"],
                          P_completion_ns=metric["P_completion"], D_latency_ns=metric["D_latency"],
                          P_solo_ns=solo[0], D_solo_ns=solo[1], Lhat_P=metric["Lhat_P"], Lhat_D=metric["Lhat_D"], Lhat_max=metric["Lhat_max"],
                          KV_start_ns=result["kv_window"][0], KV_end_ns=result["kv_window"][1])
        rows.append(finish_row(normalized))
    groups, missing = full_groups(rows, legacy=True)
    assert len(groups) == 36 and not missing and len(rows) == 108
    pairs = pairs_of(groups)
    overall = summaries(pairs, ())
    states = summaries(pairs, ("mapping_state",))
    hardware = summaries(pairs, ("hardware", "mapping_state"))
    phases = summaries(pairs, ("hardware", "mapping_state", "phase"))
    oracle, pareto = oracle_and_pareto(groups)
    choices = strategy_choices(groups)
    choice_summary = choice_summaries(choices)
    decomposition = interference(groups)
    expected = {"healthy_A3h": (-9.949, -1.802, 14), "PF4_A3hEIR": (-11.127, -1.396, 12)}
    checks = []
    for state, (p, d, both) in expected.items():
        summary = next(r for r in states if r["mapping_state"] == state and r["selected"] == "S256" and r["baseline"] == "A256")
        assert abs(summary["P_change_pct_mean"] - p) <= .0005 and abs(summary["D_change_pct_mean"] - d) <= .0005
        assert summary["both_PD_improved"] == both
        checks.append(dict(scope=state, expected_rounded_P=p, expected_rounded_D=d, actual=summary))
    all_s = next(r for r in overall if r["selected"] == "S256" and r["baseline"] == "A256")
    assert abs(all_s["P_change_pct_mean"] + 10.538) <= .0005 and abs(all_s["D_change_pct_mean"] + 1.599) <= .0005
    assert all_s["P_change_pct_improved"] == 36 and all_s["D_change_pct_improved"] == 26 and all_s["D_change_pct_regressed"] == 10
    assert all_s["Lhat_max_delta_improved"] == 36 and abs(all_s["D_change_pct_maximum"] - 17.952) <= .0005
    rounded_claims = []
    def check_claim(label, actual, expected):
        assert abs(actual - expected) <= .0005, (label, actual, expected)
        rounded_claims.append(dict(claim=label, expected_rounded=expected, actual=actual))
    pf4_s = next(r for r in states if r["mapping_state"] == "PF4_A3hEIR" and r["selected"] == "S256" and r["baseline"] == "A256")
    check_claim("PF4 maximum D regression vs A256", pf4_s["D_change_pct_maximum"], 14.810)
    for hw, state, expected_d in (("wsc", "healthy_A3h", 2.210), ("wsc", "PF4_A3hEIR", 2.065),
                                  ("face", "healthy_A3h", -5.813), ("face", "PF4_A3hEIR", -4.857)):
        item = next(r for r in hardware if r["hardware"] == hw and r["mapping_state"] == state and r["selected"] == "S256" and r["baseline"] == "A256")
        check_claim(f"{hw}/{state} mean D vs A256", item["D_change_pct_mean"], expected_d)
    for hw, p, d, improved, regressed in (("wsc", -1.057, -8.177, 7, 2), ("face", .217, -1.343, 4, 5)):
        item = next(r for r in hardware if r["hardware"] == hw and r["mapping_state"] == "PF4_A3hEIR" and r["selected"] == "S256" and r["baseline"] == "BB0")
        check_claim(f"{hw}/PF4 median P vs BB0", item["P_change_pct_median"], p)
        check_claim(f"{hw}/PF4 median D vs BB0", item["D_change_pct_median"], d)
        assert item["Lhat_max_delta_improved"] == improved and item["Lhat_max_delta_regressed"] == regressed
    for arm, p, d in (("A256", 12.081, -32.504), ("S256", -2.367, -22.507)):
        item = next(r for r in phases if r["hardware"] == "wsc" and r["mapping_state"] == "PF4_A3hEIR" and r["phase"] == .1 and r["selected"] == arm and r["baseline"] == "BB0")
        check_claim(f"wsc/PF4/g0.1/{arm} median P vs BB0", item["P_change_pct_median"], p)
        check_claim(f"wsc/PF4/g0.1/{arm} median D vs BB0", item["D_change_pct_median"], d)
    s_rows = list(csv.DictReader((ROOT / "e2e/hi/closeout/tables/s_144_coverage.csv").open(encoding="utf-8-sig")))
    index = {(r["hardware"], r["MRS"], float(r["phase"]), int(r["seed"])): r for r in s_rows}
    matched = []
    for row in rows:
        if row["mapping_state"] != "PF4_A3hEIR" or row["transport"] == "S256":
            continue
        other = index[row["hardware"], "110" if row["transport"] == "BB0" else "111", row["phase"], row["seed"]]
        fields = (("P_completion_ns", "P_completion"), ("D_latency_ns", "D_latency"), ("Lhat_P", "Lhat_P"),
                  ("Lhat_D", "Lhat_D"), ("Lhat_max", "Lhat_max"), ("tau_ns", "tau_D"),
                  ("P_solo_ns", "P_solo"), ("D_solo_ns", "D_solo"))
        assert all(row[a] == float(other[b]) for a, b in fields)
        matched.append(dict(post_fg_job=row["job_id"], system_job=other["job_id"], matched_numeric_fields=len(fields)))
    tables = destination / "tables"
    for name, content in (("existing_36_runs", rows), ("existing_36_pairs", pairs), ("existing_36_overall_means", overall),
                          ("existing_36_state_summary", states), ("existing_36_hardware_summary", hardware), ("existing_36_phase_summary", phases),
                          ("existing_36_oracle", oracle), ("existing_36_pareto", pareto), ("existing_36_solo_interference", decomposition)):
        table(tables / (name + ".csv"), content)
    table(tables / "existing_36_strategy_choices.csv", choices)
    table(tables / "existing_36_strategy_summary.csv", choice_summary)
    evidence = dict(status="passed", generated_utc=datetime.now(timezone.utc).isoformat(), source_csv=str(source.relative_to(ROOT)),
                    source_csv_sha256=sha(source), pair_groups=36, joint_records=108, solo_records=72,
                    source_records=sources, all_summary=all_s, state_checks=checks, rounded_claim_checks=rounded_claims,
                    system_numeric_matches=matched, system_numeric_field_matches=sum(r["matched_numeric_fields"] for r in matched),
                    oracle_counts=dict(Counter(r["selected_transport"] for r in oracle)),
                    strategy_summary=choice_summary,
                    pareto_counts=dict(Counter(r["transport"] for r in pareto if r["on_pareto_front"])),
                    decomposition_checks=len(decomposition), max_decomposition_error_ns=max(abs(r["decomposition_error_ns"]) for r in decomposition),
                    new_backend_calls=0, new_CPCM_calls=0, new_searches=0,
                    identity_note="Recomputed historical data and shared input records. Historical healthy S256 lacks a per-row final iteration hash; it is not a new-execution identity certificate.")
    save_json(destination / "evidence/existing_36_check.json", evidence)
    lines = ["# 已有 GPT 36 个配对复算", "", "范围是 POST-FG 的 GPT、WSC-LLM/FACE、健康/PF4、三相位、三种子，共36组。"
             "每组保留 original/BB0、A256、S256 三条实际结果；本表与待补齐的 S/H/I 108组分开。",
             "", "已逐条核对108份联合、72份单跑原始JSON的SHA-256与表中数值。"
             "P和D分别先算每对百分比，再等权平均；最大归一化迟约用有符号差值，真实超期按正值计。", ""]
    lines += markdown_statistics(overall)
    lines += ["", "按状态：", ""] + markdown_statistics(states, ("mapping_state",))
    lines += ["", "按硬件与状态：", ""] + markdown_statistics(hardware, ("hardware", "mapping_state"))
    lines += ["", "附件总体均值、改善数量和最大D回退均已复算一致。PF4 BB0/A256共36行与系统110/111的288个数值字段完全相同。",
              "旧健康S256记录没有逐行最终集合遍历摘要；原始字段和此前构图检查记录保留，本次复算不新增运行身份认证。", ""]
    lines += oracle_text(oracle, pareto)
    lines += ["", "三类事后选择与固定策略并排：", ""] + choice_markdown(choice_summary)
    lines += ["", "主导计数取每条结果中Lhat_P与Lhat_D的较大者；负值代表提前完成，只有大于0才计为超期。"
              "Lhat_max由P主导时，最小Lhat_max更接近保护P完成时间，不能将该选择直接描述成最大化D加速。"]
    for hw in ("wsc", "face"):
        chosen = [r for r in choices if r["mapping_state"] == "PF4_A3hEIR" and r["hardware"] == hw]
        lines += ["", f"{hw} 的 A3h+EIR（original对应系统110）：", ""] + choice_markdown(choice_summaries(chosen))
    lines += ["", "Oracle-min-D-Ple3pct只允许P完成时间不超过同输入original的103%。无约束与受约束的min-D均使用真实已完成结果，"
              "属于事后参照。旧36组未执行本轮前验门控，故不填造Online-CPCM-Gated历史成绩。"]
    lines += ["", "单跑变化与联合干扰分解共216条等式通过；负干扰值保留，表示该次联合服务次序比对应单跑更快。",
              "完整逐种子、分相位、超期计数、oracle及Pareto数据在tables/existing_36_*.csv。未运行新仿真。"]
    save_text(destination / "EXISTING_36_REPORT.md", "\n".join(lines) + "\n")
    return evidence


def report(out, destination):
    destination = Path(destination)
    rows, inputs, ranks = collect_new(out)
    groups, missing = full_groups(rows)
    expected_cases = planned_cases()
    observed_cases = {key(row) for row in rows}
    assert observed_cases <= expected_cases
    missing += [dict(zip(CASE, case), missing=list(TRANSPORTS)) for case in sorted(expected_cases - observed_cases)]
    pairs = pairs_of(groups)
    overall = summaries(pairs, ())
    by_model = summaries(pairs, ("model",))
    model = summaries(pairs, ("scope", "model", "hardware", "mapping_state"))
    phases = summaries(pairs, ("scope", "model", "hardware", "mapping_state", "phase"))
    oracle, pareto = oracle_and_pareto(groups)
    gate_decisions = read_gate_decisions(out)
    choices = strategy_choices(groups, gate_decisions)
    choice_summary = choice_summaries(choices)
    choice_details = choice_summaries(choices, ("scope", "model", "hardware", "mapping_state", "phase"))
    windows, window_samples = online_window_comparison(groups, gate_decisions)
    gate_costs = online_costs(gate_decisions)
    system_increment, system_full = system_pairs(groups)
    system_summary = summaries(system_increment, ("hardware", "effect", "transport", "start_state", "end_state"))
    system_full_summary = summaries(system_full, ("hardware", "transport", "end_state"))
    decomposition = interference(groups)
    for name, content in (("s256_runs", rows), ("s256_pairs", pairs), ("s256_overall_means", overall),
                          ("s256_state_summary", model), ("s256_phase_summary", phases), ("s256_solo_interference", decomposition),
                          ("s256_c4_rank_check", ranks), ("s256_oracle", oracle), ("s256_pareto", pareto)):
        table(destination / "tables" / (name + ".csv"), content)
    table(destination / "tables/s256_model_summary.csv", by_model)
    table(destination / "tables/s256_system_increments.csv", system_increment)
    table(destination / "tables/s256_system_increment_summary.csv", system_summary)
    table(destination / "tables/s256_system_full_vs_000.csv", system_full)
    table(destination / "tables/s256_system_full_summary.csv", system_full_summary)
    table(destination / "tables/s256_strategy_choices.csv", choices)
    table(destination / "tables/s256_strategy_summary.csv", choice_summary)
    table(destination / "tables/s256_strategy_phase_summary.csv", choice_details)
    table(destination / "tables/online_window_comparison.csv", windows)
    table(destination / "tables/online_window_samples.csv", window_samples)
    table(destination / "tables/online_gate_costs.csv", gate_costs)
    coverage = Counter(case[0] for case in groups)
    online_count = sum(r["selection"] == "Online-CPCM-Gated" for r in choices)
    complete = (set(groups) == expected_cases and not missing
                and set(gate_decisions) == {case[:-1] for case in expected_cases} and online_count == 108)
    evidence = dict(status="complete" if complete else "partial", generated_utc=datetime.now(timezone.utc).isoformat(),
                    expected_groups=dict(S=72, H=18, I=18), complete_group_counts=dict(coverage),
                    complete_groups=len(groups), joint_records=len(rows), missing_groups=missing, inputs=inputs,
                    all_complete=complete, new_backend_calls=0, new_CPCM_calls=0,
                    gate_decisions=len(gate_decisions), online_result_groups=online_count,
                    online_window_context_phases=len(windows), online_window_samples=len(window_samples),
                    gate_cost_jobs=len(gate_costs),
                    gate_prediction_seconds=sum(r["prediction_seconds"] for r in gate_costs),
                    gate_task_body_seconds=sum(r["task_body_seconds"] for r in gate_costs),
                    system_increment_pairs=len(system_increment), system_vs_000_pairs=len(system_full),
                    frozen_before_evaluation_checks=3 * online_count,
                    decomposition_checks=len(decomposition),
                    max_decomposition_error_ns=max((abs(r["decomposition_error_ns"]) for r in decomposition), default=None),
                    oracle_counts=dict(Counter(r["selected_transport"] for r in oracle)),
                    strategy_summary=choice_summary,
                    pareto_counts=dict(Counter(r["transport"] for r in pareto if r["on_pareto_front"])))
    save_json(destination / "evidence/report_audit.json", evidence)
    status = "已齐" if complete else "当前部分结果"
    lines = ["# S/H/I 的 S256 完整对照", "", f"{status}：{len(groups)}/108个三策略组、{len(rows)}条联合结果。"
             f"S {coverage['S']}/72、H {coverage['H']}/18、I {coverage['I']}/18。只有三种策略同输入齐备的组进入配对统计。",
             "", "S256是本轮主传输配置；A256和original/BB0保留对照。相同模型、硬件、状态、物理平台、输入图、"
             "共同参考、相位与种子逐行配对。P表示完成时间，D表示完成减释放，单位ns。", ""]
    lines += markdown_statistics(overall)
    lines += ["", "各模型分别等权汇总：", ""] + markdown_statistics(by_model, ("model",))
    lines += ["", "模型、硬件与状态：", ""] + markdown_statistics(model, ("scope", "model", "hardware", "mapping_state"))
    lines += ["", "完整相位、种子、正负变化和真实超期次数均列入表格。P/D各自平均，不将纳秒混成一个综合均值。",
              "相对A256和相对original的变化分别报告；旧POST-FG 36组另见EXISTING_36_REPORT.md。", ""]
    lines += ["S系统保持M×R×Transport：M为A1/A3h，R为R0/EIR，Transport为original/A256/S256。"
              "旧111代表A256，新111-S256另列；下面给出S256传输下映射与修复的条件增量，"
              "完整三种传输及相对000的结果保存在s256_system_*表。百分比均逐行配对，不把M、R与传输的百分比直接相加。", ""]
    lines += markdown_statistics([r for r in system_summary if r["effect"] in ("M", "R") and r["transport"] == "S256"],
                                  ("hardware", "effect", "start_state", "end_state"))
    lines += oracle_text(oracle, pareto)
    lines += ["", "固定策略、事后选择与前验选择并排：", ""] + choice_markdown(choice_summary)
    lines += ["", "主导计数逐行比较Lhat_P和Lhat_D，保留精确并列。最大值为负表示尚未超期；最小化这个值"
              "可能主要改善P，也可能主要改善D，必须结合主导计数与P/D两列解释。"]
    for hw in ("wsc", "face"):
        selected = [r for r in choices if r["scope"] == "S" and r["mapping_state"] == "A3hEIR" and r["hardware"] == hw]
        if selected:
            lines += ["", f"{hw} 的 A3h+EIR（original对应系统110）：", ""] + choice_markdown(choice_summaries(selected))
    lines += ["", "Oracle-min-D按D延迟最小选完整策略；Oracle-min-D-Ple3pct先要求P不超过同输入original的103%，再按D最小选择。"
              "两者使用事后结果，精确并列沿用S256、A256、BB0顺序且保留全部并列。",
              "Online-CPCM-Gated仅读取本轮后端运行前冻结的选择：存在共享有向资源且D释放早于CPCM预测P原图KV结束时刻时选S256，"
              "否则选original。每相位的动作在三个后端种子中相同，报表只取该既定动作对应的实际完整结果。"]
    if gate_costs:
        lines += ["", f"门控共完成{len(gate_costs)}次独立预测，供36个相位决策和108个后端种子结果复用，成本不按108重复累计。"
                  f"预测函数耗时中位{statistics.median(r['prediction_seconds'] for r in gate_costs):.3f}秒，"
                  f"其中CPCM前向中位{statistics.median(r['cpcm_forward_seconds'] for r in gate_costs):.3f}秒；"
                  f"含输入读取与准备的任务体中位{statistics.median(r['task_body_seconds'] for r in gate_costs):.3f}秒。"
                  "这是本轮并行新进程测量，任务体不含解释器启动，也不当作常驻部署的逐请求开销；分项见online_gate_costs.csv。"]
    save_text(destination / "S_AND_HI_S256_REPORT.md", "\n".join(lines) + "\n")
    applicability = ["# DSTU 的共享资源与重叠条件", "", f"本页使用{len(groups)}个完整组，按事先固定的三相位、模型、硬件与状态分组。"
                     "全部样本纳入，包含性能回退；没有按结果胜负筛选分组。", ""]
    applicability += markdown_statistics(phases, ("scope", "model", "hardware", "mapping_state", "phase"))
    applicability += ["", "g由共同健康A1参考的KV窗口定义。同一g在不同映射上可能形成不同实际重叠；"
                      "s256_runs.csv同时保存后端KV起止、D释放时刻与包络交集。包络交集用于解释执行现象，不等于每条链路持续发生竞争。",
                      "", "单跑来自完整联合图的相同切块选择，再按流过滤。O是同方案单跑相对original单跑的变化；"
                      "J是同方案联合减单跑。逐流核对 joint差=O+J−J_original。原策略干扰比例作为连续变量保留，未新增任意阈值。",
                      "", "oracle和Pareto使用已完成的三种策略结果描述可达取舍，不作为在线自适应执行成绩。"]
    applicability += ["Online-CPCM-Gated独立使用CPCM预测窗口与共享资源作前验选择；其预测窗口与这里解释机制的后端实际窗口分列保存。"]
    applicability += ["", f"当前窗口对照为{len(windows)}/36个模型/硬件/状态/相位组、{len(window_samples)}/108个种子结果。"
                      "online_window_comparison.csv按三个种子给出原始P单跑KV窗口、original联合KV窗口和既定动作联合窗口的均值与范围，"
                      "并记录预测结束偏早/偏晚、结束条件判断一致数及实际D/KV包络重叠数。逐种子值在online_window_samples.csv。",
                      "CPCM预测对象是原图P单跑，其结束偏差优先与original的P单跑比较；联合窗口用于解释D加入后的变化。"
                      "这些事后对照不改变已冻结动作。g=0.9只是共同参照中的晚相位，并不保证晚于每个当前映射的预测KV结束时刻。"]
    save_text(destination / "DSTU_APPLICABILITY_ANALYSIS.md", "\n".join(applicability) + "\n")
    final = ["# 论文表格输入", "", f"结果状态：{status}，完整三策略组{len(groups)}/108。", "",
             "| 文件 | 内容 |", "|---|---|",
             "| tables/s256_runs.csv | 逐策略、逐相位、逐种子结果、实际窗口及完整输入身份 |",
             "| tables/s256_pairs.csv | S256/A256、S256/original、A256/original逐行配对 |",
             "| tables/s256_overall_means.csv | 全部已完成组等权均值、中位数、范围、改善与回退计数 |",
             "| tables/s256_state_summary.csv | 分模型、硬件、映射与修复状态 |",
             "| tables/s256_model_summary.csv | GPT、Llama、Qwen分别汇总，显示各自实际范围 |",
             "| tables/s256_system_increment_summary.csv | M、R与Transport条件增量，保留A256与S256两套系统结果 |",
             "| tables/s256_system_full_summary.csv | S系统12种组合分别相对同输入000 |",
             "| tables/s256_phase_summary.csv | 三相位分组，全部样本纳入 |",
             "| tables/s256_solo_interference.csv | 单跑变化O、跨流干扰J及分解等式 |",
             "| tables/s256_oracle.csv | 按Lhat_max选完整已执行策略的事后最优参照，保留所有并列 |",
             "| tables/s256_pareto.csv | P/D双目标非支配策略，逐种子完整列出 |",
             "| tables/s256_strategy_choices.csv | 固定策略、三种事后选择及冻结前验门控的逐组完整结果 |",
             "| tables/s256_strategy_summary.csv | 各类选择的P/D相对original变化、迟约、超期及策略计数 |",
             "| tables/online_window_comparison.csv | 36个相位组的预测与三种子实际KV窗口、方向与重叠对照 |",
             "| tables/online_window_samples.csv | 108条窗口逐种子值与来源；不参与门控动作更改 |",
             "| tables/online_gate_costs.csv | 12次独立预测的准备/构图/前向/任务体成本，每个输入状态只计一次 |",
             "| tables/existing_36_*.csv | 历史GPT36组，单独报告 |", "",
             "图表先展示完整均值与P/D取舍，再用固定相位和实际共享重叠解释代表场景，最后列系统增量。"
             "S256结果带明确策略后缀，旧A256的111不被覆盖。所有表格均为原始结果后处理，没有新增搜索。"]
    save_text(destination / "FINAL_PAPER_TABLES.md", "\n".join(final) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("existing", "report"))
    parser.add_argument("--out", type=Path, default=BASE / "out/main")
    parser.add_argument("--destination", type=Path, default=BASE)
    args = parser.parse_args()
    result = existing(args.destination) if args.command == "existing" else report(args.out, args.destination)
    print(json.dumps({key: result[key] for key in result if key in ("status", "pair_groups", "complete_groups", "joint_records", "solo_records",
                     "oracle_counts", "pareto_counts", "decomposition_checks", "max_decomposition_error_ns")}, indent=2))


if __name__ == "__main__":
    main()
