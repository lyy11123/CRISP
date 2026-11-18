"""Diagnose frozen window gating from saved CSV/JSON only; no simulator imports."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "e2e/dstu_supplement"
CASE = ("scope", "model", "hardware", "mapping_state", "phase", "seed")


def read(name):
    with (OUT / "tables" / (name + ".csv")).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(name, rows):
    with (OUT / "tables" / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return tuple(row[field] for field in CASE)


def value(row, field):
    return float(row[field])


def pct(a, b):
    return 100 * (a / b - 1)


def label(row):
    model = {"gpt": "GPT", "llama": "Llama", "qwen3moe": "Qwen"}[row["model"]]
    state = {"healthy_A3h": "健康 A3h", "D_A3hEIR": "PF4 A3h-EIR"}.get(row["mapping_state"], row["mapping_state"])
    return f"{model}/{row['hardware']}/{state}/g={row['phase']}"


def main():
    inputs = ["online_window_samples", "s256_runs", "s256_strategy_choices", "s256_solo_interference"]
    windows, runs, choices, interference = [read(name) for name in inputs]
    arms = defaultdict(dict)
    for row in runs:
        source = (ROOT / row["source_result"]).read_bytes()
        assert hashlib.sha256(source).hexdigest() == row["source_sha256"]
        saved = json.loads(source)
        for field in ("P_completion", "D_latency", "P_solo", "D_solo", "Lhat_max"):
            column = field if field == "Lhat_max" else field + "_ns"
            assert value(row, column) == saved["metrics"][field]
        assert [value(row, "KV_start_ns"), value(row, "KV_end_ns")] == saved["kv_window"]
        arms[key(row)][row["transport"]] = row
    assert len(windows) == len(arms) == 108 and len(runs) == 324
    assert all(set(group) == {"BB0", "A256", "S256"} for group in arms.values())
    decomposition = {(key(row), row["transport"], row["stream"]): row for row in interference}
    samples = []
    for row in windows:
        group = arms[key(row)]
        bb, s = group["BB0"], group["S256"]
        solo_source = (ROOT / row["source_BB0_solo"]).read_bytes()
        assert hashlib.sha256(solo_source).hexdigest() == bb["P_solo_source_sha256"]
        solo_saved = json.loads(solo_source)
        assert [value(row, "BB0_solo_KV_start_ns"), value(row, "BB0_solo_KV_end_ns")] == solo_saved["kv_window"]
        chosen = group[row["chosen_transport"]]
        pred = value(row, "shared_resource_count") > 0 and value(row, "tau_ns") < value(row, "predicted_KV_end_ns")
        solo = value(row, "shared_resource_count") > 0 and value(row, "tau_ns") < value(row, "BB0_solo_KV_end_ns")
        joint = value(row, "shared_resource_count") > 0 and value(row, "tau_ns") < value(row, "BB0_joint_KV_end_ns")
        assert pred == (row["chosen_transport"] == "S256")
        assert solo == (row["actual_BB0_solo_end_test_selects_S256"] == "True")
        assert joint == (row["actual_BB0_joint_end_test_selects_S256"] == "True")
        item = dict(row)
        item.update(predicted_overlap=pred, actual_solo_end_overlap=solo, actual_joint_end_overlap=joint,
                    classification="TP" if pred and solo else "FP" if pred else "FN" if solo else "TN",
                    end_error_pct=pct(value(row, "predicted_KV_end_ns"), value(row, "BB0_solo_KV_end_ns")),
                    predicted_remaining_ns=value(row, "predicted_KV_end_ns") - value(row, "tau_ns"),
                    actual_solo_remaining_ns=value(row, "BB0_solo_KV_end_ns") - value(row, "tau_ns"),
                    S256_joint_KV_overlap_ns=value(s, "KV_D_envelope_overlap_ns"),
                    BB0_P_remaining_at_D_release_ns=value(bb, "P_completion_ns") - value(row, "tau_ns"),
                    S256_P_remaining_at_D_release_ns=value(s, "P_completion_ns") - value(row, "tau_ns"))
        for stream, metric in (("P", "P_completion_ns"), ("D", "D_latency_ns")):
            item[f"{stream}_S256_vs_BB0_pct"] = pct(value(s, metric), value(bb, metric))
            item[f"{stream}_online_vs_S256_pct"] = pct(value(chosen, metric), value(s, metric))
            item[f"{stream}_online_minus_fixed_common_BB0_pp"] = 100 * (value(chosen, metric) - value(s, metric)) / value(bb, metric)
            d = decomposition[(key(row), "S256", stream)]
            item[f"{stream}_O_ns"] = value(d, "service_change_O_ns")
            item[f"{stream}_J_difference_ns"] = value(d, "interference_J_ns") - value(d, "original_interference_J_ns")
            item[f"{stream}_joint_change_ns"] = value(s, metric) - value(bb, metric)
            assert abs(item[f"{stream}_joint_change_ns"] - item[f"{stream}_O_ns"] - item[f"{stream}_J_difference_ns"]) < 1e-6
        item["S256_minus_BB0_Lhat_max"] = value(s, "Lhat_max") - value(bb, "Lhat_max")
        item["online_minus_S256_Lhat_max"] = value(chosen, "Lhat_max") - value(s, "Lhat_max")
        samples.append(item)
    write("gate_diagnosis_samples", samples)

    by_phase = defaultdict(list)
    for row in samples:
        by_phase[key(row)[:-1]].append(row)
    phases = []
    numeric = ("end_error_pct", "BB0_solo_KV_end_ns", "BB0_joint_KV_end_ns", "BB0_KV_D_envelope_overlap_ns",
               "P_S256_vs_BB0_pct", "D_S256_vs_BB0_pct", "P_online_vs_S256_pct", "D_online_vs_S256_pct",
               "S256_minus_BB0_Lhat_max", "online_minus_S256_Lhat_max", "P_O_ns", "D_O_ns",
               "P_J_difference_ns", "D_J_difference_ns", "P_joint_change_ns", "D_joint_change_ns")
    for case, group in sorted(by_phase.items()):
        row = dict(zip(CASE[:-1], case), n=len(group), chosen_transport=group[0]["chosen_transport"],
                   tau_ns=group[0]["tau_ns"], predicted_KV_end_ns=group[0]["predicted_KV_end_ns"],
                   classification_counts=dict(Counter(r["classification"] for r in group)),
                   actual_BB0_overlap_count=sum(value(r, "BB0_KV_D_envelope_overlap_ns") > 0 for r in group),
                   S256_P_improved=sum(r["P_S256_vs_BB0_pct"] < 0 for r in group),
                   S256_D_improved=sum(r["D_S256_vs_BB0_pct"] < 0 for r in group),
                   S256_D_worse_seeds=",".join(r["seed"] for r in group if r["D_S256_vs_BB0_pct"] > 0),
                   S256_Lhat_max_improved=sum(r["S256_minus_BB0_Lhat_max"] < 0 for r in group))
        for name in numeric:
            values = [value(r, name) for r in group]
            row.update({name + "_mean": mean(values), name + "_min": min(values), name + "_max": max(values)})
        for stream in ("P", "D"):
            row[stream + "_overall_gap_contribution_pp"] = sum(r[stream + "_online_minus_fixed_common_BB0_pp"] for r in group) / len(samples)
        phases.append(row)
    write("gate_diagnosis_phases", phases)

    methods = []
    for mode in ("fixed_S256", "frozen_online", "actual_solo_end_diagnostic", "actual_joint_end_diagnostic"):
        selected = []
        for row in samples:
            pick = "S256"
            if mode == "frozen_online":
                pick = row["chosen_transport"]
            elif mode == "actual_solo_end_diagnostic":
                pick = "S256" if row["actual_solo_end_overlap"] else "BB0"
            elif mode == "actual_joint_end_diagnostic":
                pick = "S256" if row["actual_joint_end_overlap"] else "BB0"
            group = arms[key(row)]
            selected.append((pick, group[pick], group["BB0"], group["S256"]))
        result = dict(method=mode, n=len(selected), S256_count=sum(t[0] == "S256" for t in selected),
                      BB0_count=sum(t[0] == "BB0" for t in selected),
                      retrospective="diagnostic" in mode,
                      Lhat_max_mean=mean(value(t[1], "Lhat_max") for t in selected),
                      P_deadline_missed=sum(t[1]["P_deadline_missed"] == "True" for t in selected),
                      D_deadline_missed=sum(t[1]["D_deadline_missed"] == "True" for t in selected))
        for stream, metric in (("P", "P_completion_ns"), ("D", "D_latency_ns")):
            result[stream + "_vs_BB0_pct"] = mean(pct(value(t[1], metric), value(t[2], metric)) for t in selected)
            result[stream + "_vs_S256_pct"] = mean(pct(value(t[1], metric), value(t[3], metric)) for t in selected)
        methods.append(result)
    write("gate_diagnosis_counterfactual", methods)
    by_method = {row["method"]: row for row in methods}
    online, fixed, exact = (by_method[m] for m in ("frozen_online", "fixed_S256", "actual_solo_end_diagnostic"))
    contributions = []
    for name, classes in (("prediction_false_negative", {"FN"}), ("correct_solo_end_bypass", {"TN"})):
        group = [r for r in samples if r["classification"] in classes]
        row = dict(component=name, n=len(group))
        for stream in ("P", "D"):
            row[stream + "_overall_gap_pp"] = sum(r[stream + "_online_minus_fixed_common_BB0_pp"] for r in group) / len(samples)
        contributions.append(row)
    write("gate_diagnosis_gap_components", contributions)

    bypass = [r for r in samples if r["chosen_transport"] == "BB0"]
    enabled = [r for r in samples if r["chosen_transport"] == "S256"]
    no_window = [r for r in samples if value(r, "BB0_KV_D_envelope_overlap_ns") == 0 and r["S256_joint_KV_overlap_ns"] == 0]
    # Reconcile this independent calculation with the already published choice rows.
    previous = [r for r in choices if r["selection"] == "Online-CPCM-Gated"]
    assert len(previous) == 108
    for stream in ("P", "D"):
        assert abs(mean(value(r, stream + "_change_vs_BB0_pct") for r in previous) - online[stream + "_vs_BB0_pct"]) < 1e-10
    audit = dict(status="passed", backend_calls=0, CPCM_calls=0, new_experiments=0,
                 joint_JSON_sha256_and_metrics_verified=len(runs), solo_KV_window_checks=len(windows),
                 groups=len(samples), classification=dict(Counter(r["classification"] for r in samples)),
                 bypass_count=len(bypass), bypass_PD_dominated_by_S256=sum(r["P_S256_vs_BB0_pct"] < 0 and r["D_S256_vs_BB0_pct"] < 0 for r in bypass),
                 bypass_Lhat_max_worse=sum(r["online_minus_S256_Lhat_max"] > 0 for r in bypass),
                 enabled_count=len(enabled), enabled_D_regressions=sum(r["D_S256_vs_BB0_pct"] > 0 for r in enabled),
                 enabled_PD_dominated_by_BB0=sum(r["P_S256_vs_BB0_pct"] > 0 and r["D_S256_vs_BB0_pct"] > 0 for r in enabled),
                 enabled_actual_overlap_count=sum(value(r, "BB0_KV_D_envelope_overlap_ns") > 0 for r in enabled),
                 both_policies_without_KV_overlap=len(no_window),
                 no_window_P_improved=sum(r["P_S256_vs_BB0_pct"] < 0 for r in no_window),
                 no_window_D_improved=sum(r["D_S256_vs_BB0_pct"] < 0 for r in no_window),
                 source_sha256={"tables/" + name + ".csv": hashlib.sha256((OUT / "tables" / (name + ".csv")).read_bytes()).hexdigest() for name in inputs})
    for stream in ("P", "D"):
        gap = online[stream + "_vs_BB0_pct"] - fixed[stream + "_vs_BB0_pct"]
        corrected = online[stream + "_vs_BB0_pct"] - exact[stream + "_vs_BB0_pct"]
        audit[stream + "_gap_pp"] = gap
        audit[stream + "_prediction_correction_fraction"] = corrected / gap
        assert abs(sum(r[stream + "_overall_gap_pp"] for r in contributions) - gap) < 1e-10
    (OUT / "evidence/gate_diagnosis_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# 在线窗口门控为什么没有超过固定 S256", "", "2026-11-18。只读取本轮已经保存的 108 组、324 条联合结果及配对单跑、冻结决定和窗口表；新增 CPCM、后端和实验均为 0。原始动作、原始结果和已发布实验总表不改写。", "",
             "结论分两部分：Decode 的平均差距主要来自 Qwen 的两个错误旁路决定，涉及六个后端样本；Prefill 的差距主要来自旁路规则本身。KV 是否结束不能直接判定切换整个传输策略是否有收益。12个状态各预测一次，形成36个相位决定；每个决定对应三个后端种子，不是108次独立预测。", "",
             "## 1. 先量化差别", "",
             "每个样本先除以同输入 original（BB0），再对108组等权平均。P是Prefill完成时间，D是Decode完成减释放，负百分比表示更快。两个‘实际结束’行只从已有策略记录中重选，用于分离原因，是事后诊断，不是新的在线成绩。", "",
             "| 选择方式 | S256 / original | P 相对 original | D 相对 original | Lhat_max 均值 |", "|---|---:|---:|---:|---:|"]
    names = {"fixed_S256": "固定 S256", "frozen_online": "已冻结在线门控", "actual_solo_end_diagnostic": "同一规则换成实际 P 单跑 KV 结束", "actual_joint_end_diagnostic": "同一规则换成实际 original 联合 KV 结束"}
    for row in methods:
        lines.append(f"| {names[row['method']]} | {row['S256_count']} / {row['BB0_count']} | {row['P_vs_BB0_pct']:+.6f}% | {row['D_vs_BB0_pct']:+.6f}% | {row['Lhat_max_mean']:.6f} |")
    lines += ["", f"在线门控相对固定S256，P收益少 {audit['P_gap_pp']:.6f} 个百分点，D收益少 {audit['D_gap_pp']:.6f} 个百分点。如果直接逐样本算在线/S256，则P增加 {online['P_vs_S256_pct']:.6f}%、D增加 {online['D_vs_S256_pct']:.6f}%；两种数字的分母不同。四行的P/D超期数均为0/9。", "",
              "90条选择S256的样本与固定方案执行同一条结果；全部差别来自18条旁路。旁路中10条P/D都更差、8条是P/D取舍，没有一条P/D同时更好；15条的Lhat_max变差。", "",
              "## 2. 六个旁路场景的预测、实际和结果", "",
              "每行三种子；时间单位μs。实际结束范围来自original的P单跑，最接近CPCM预测对象；‘联合重叠’是original中D存续区间与KV时间包络有交集的种子数。最后两列是切回original后，相对固定S256的实际变化，正值表示门控更慢。", "",
              "| 场景 | D释放 | 预测KV结束 | 实际P单跑KV结束范围 | 联合重叠 | 在线/S256：P | 在线/S256：D |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in phases:
        if row["chosen_transport"] == "BB0":
            lines.append(f"| {label(row)} | {float(row['tau_ns'])/1000:.3f} | {float(row['predicted_KV_end_ns'])/1000:.3f} | {row['BB0_solo_KV_end_ns_min']/1000:.3f}–{row['BB0_solo_KV_end_ns_max']/1000:.3f} | {row['actual_BB0_overlap_count']}/3 | {row['P_online_vs_S256_pct_mean']:+.3f}% | {row['D_online_vs_S256_pct_mean']:+.3f}% |")
    lines += ["", "### 2.1 明确由预测偏早导致：Qwen健康与PF4修复，g=0.5", "",
              "共同预测667.358μs，D在706.849μs释放，规则认为KV已结束39.491μs；实际单跑结束792.342–922.874μs，释放时仍余85.493–216.025μs。预测提前15.774%–27.687%。六次original联合执行也都有实际KV包络重叠，因此这六次是明确的错关。", "",
              "| 状态 | 种子 | 实际单跑KV结束 μs | 预测误差 μs | 固定S256相对original：P | 固定S256相对original：D |", "|---|---:|---:|---:|---:|---:|"]
    for row in samples:
        if row["classification"] == "FN":
            lines.append(f"| {row['mapping_state']} | {row['seed']} | {value(row,'BB0_solo_KV_end_ns')/1000:.3f} | {value(row,'predicted_minus_BB0_solo_KV_end_ns')/1000:.3f} | {row['P_S256_vs_BB0_pct']:+.3f}% | {row['D_S256_vs_BB0_pct']:+.3f}% |")
    lines += ["", "六条改回S256后D全部改善，P五条改善、一条增加0.458%。原始预测与实际P单跑结束条件的分类是90次正确开启、12次正确旁路、6次错误旁路、0次错误开启。36个冻结决定均符合代码里的规则，不是条件写反、g硬编码或结果对接错误。", "",
              f"仅消除这六次误关，就恢复在线与固定方案之间 {audit['D_prediction_correction_fraction']*100:.2f}% 的D均值差、{audit['P_prediction_correction_fraction']*100:.2f}% 的P均值差。这里是本数据上的结果重组选取，未用后端数据重新训练或修改在线预测。", "",
              "CPCM在这次使用未切块P单跑的确定性bb日程；后端有随机平局与真实集合遍历顺序影响。已有记录能确定Qwen结束时间低估及其决策后果；125–256μs误差的内部来源，尚未由同输入、同仲裁轨迹的逐事件对照分开。本次不把误差进一步归到某条资源、某种平局机制或Numba实现。", "",
              "### 2.2 预测判对P单跑窗口，旁路仍损失收益：g=0.9", "",
              "Llama/Qwen四个晚相位场景的12条样本，P单跑KV确已在D释放前结束，但固定S256仍有10条P更快、7条D更快。Llama健康旁路保护了平均D（相对S256 −0.798%），却使P平均慢3.312%；Llama PF4修复旁路使P平均慢8.570%。因此应逐指标说明取舍，不能把全部18条都叫成同一种‘选错’。", "",
              "Llama两个状态g=0.9、seed2另有一层差异：P单跑结束1329.224μs早于释放1505.477μs，original联合结束却为1581.558/1581.687μs，仍重叠76.081/76.210μs。若按联合实际结束核对，就有8次错误旁路；相对预测对象P单跑是6次。联合/单跑执行顺序也会改变，不能把这两条联合与单跑的差直接当作纯D阻塞时间。", "",
              "## 3. 为什么预测窗口完全正确也不够", "",
              "当前规则只检查‘P/D有任意共享有向资源，并且D释放早于P的KV结束’。全部12状态都有115–167个共享资源，本批共享条件没有筛掉任何状态；这些资源属于整个P/D图，而时间条件只看KV消息。它没有估计共享资源在D关键时段的排队量，也没有比较策略切换带来的单流服务变化。", "",
              "更关键的是，旁路切换的是完整传输实现：original使用BusyBarn整消息后端，S256使用选择性切块与Hop/C4。KV结束之后P的其他部分仍可执行；即使没有P/D相互阻塞，策略变化也可能改变流自身的完成时间。它不是只把同一个实现上的‘KV抢占开关’关掉。", "",
              f"原始与S256联合都没有KV包络重叠的{len(no_window)}条中，S256仍有{audit['no_window_P_improved']}条P改善、{audit['no_window_D_improved']}条D改善。相反，门控启用的90条全部有实际KV包络重叠，仍有{audit['enabled_D_regressions']}条D退化，其中{audit['enabled_PD_dominated_by_BB0']}条P/D都不如original。‘有重叠’既不是收益保证，‘无重叠’也不是旁路收益保证。", "",
              "下面完整列出启用后至少一个种子D退化的场景。它们与固定S256使用同一结果，不贡献‘在线比固定差’的18条差距，但说明即使重叠预测正确，规则也不会自动择优。", "",
              "| 已启用S256的场景 | D退化种子 | S256/original：P均值 | S256/original：D均值 |", "|---|---|---:|---:|"]
    for row in phases:
        if row["chosen_transport"] == "S256" and row["S256_D_worse_seeds"]:
            lines.append(f"| {label(row)} | {row['S256_D_worse_seeds']} | {row['P_S256_vs_BB0_pct_mean']:+.3f}% | {row['D_S256_vs_BB0_pct_mean']:+.3f}% |")
    lines += ["", "### 3.1 Llama健康的实例：重叠存在，D仍变慢", "",
              "令O=T_solo(S256)−T_solo(original)，ΔJ=[T_joint(S256)−T_solo(S256)]−[T_joint(original)−T_solo(original)]。则联合时间变化精确等于O+ΔJ。这里的J是观测到的联合减单跑差，不把带负值的J解释成纯物理等待。", "",
              "| Llama健康 | D单跑变化O μs | 联合减单跑差的变化ΔJ μs | D联合变化 μs | D相对变化 |", "|---|---:|---:|---:|---:|"]
    for row in phases:
        if row["model"] == "llama" and row["mapping_state"] == "healthy_A3h" and row["phase"] in ("0.1", "0.5"):
            lines.append(f"| g={row['phase']} | {row['D_O_ns_mean']/1000:+.3f} | {row['D_J_difference_ns_mean']/1000:+.3f} | {row['D_joint_change_ns_mean']/1000:+.3f} | {row['D_S256_vs_BB0_pct_mean']:+.3f}% |")
    lines += ["", "这两组三个种子的D均退化；D是高优先级流，S256不切它，但BB0与Hop/C4下观测到不同的单跑服务结果。减少的联合减单跑差不足以抵消自身服务变化。仅看KV是否重叠，无法算出这个净收益。", "",
              "### 3.2 Qwen健康晚相位：KV不重叠，S256仍更快", "",
              "g=0.9时，D在953.766μs释放。三个种子中original的KV在763.971/922.874/922.874μs结束，S256在755.878/923.024/808.221μs结束，两种策略的KV都已结束；但完整P要到约3.3ms才完成。", "",
              "| 流 | original联合均值 μs | S256联合均值 μs | 单跑变化O μs | ΔJ μs | 联合净变化 μs | 配对百分比均值 |", "|---|---:|---:|---:|---:|---:|---:|"]
    qwen = next(row for row in phases if row["model"] == "qwen3moe" and row["mapping_state"] == "healthy_A3h" and row["phase"] == "0.9")
    qwen_arms = [arms[key(row)] for row in samples if row["model"] == "qwen3moe" and row["mapping_state"] == "healthy_A3h" and row["phase"] == "0.9"]
    for stream, metric in (("P", "P_completion_ns"), ("D", "D_latency_ns")):
        lines.append(f"| {stream} | {mean(value(a['BB0'],metric) for a in qwen_arms)/1000:.3f} | {mean(value(a['S256'],metric) for a in qwen_arms)/1000:.3f} | {qwen[stream+'_O_ns_mean']/1000:+.3f} | {qwen[stream+'_J_difference_ns_mean']/1000:+.3f} | {qwen[stream+'_joint_change_ns_mean']/1000:+.3f} | {qwen[stream+'_S256_vs_BB0_pct_mean']:+.3f}% |")
    lines += ["", "这里收益体现在观测到的单跑策略差，ΔJ反而抵消一部分；不能解释为避免KV竞争。即使KV结束预测完全准确，以‘无KV窗口’为理由切回original，也会放弃这些已有收益。", "",
              "## 4. 对结果解释的调整", "",
              "- Decode损失的主要来源是Qwen中相位的预测低估：两个错误旁路决定、六个后端样本解释87.59%的均值差。Prefill差距主要仍来自规则没有考虑完整传输策略的收益：把预测换成实际P单跑结束后仍剩75.29%的P均值差；Qwen无重叠实例可见单跑差是主要项。", "- 固定S256仍作为本轮主配置；当前Online-CPCM-Gated作为已冻结的简单窗口规则报告，不能写成已经优于固定策略的自适应选择器。Llama、GPT晚相位的反例照实保留。", "- 后续若改规则，需要预测策略对P/D的净影响，以及D经过具体共享资源的时序；只扩大KV时间阈值可以覆盖本批Qwen误关，却不解决重叠与策略收益不等价的问题。本次不调阈值、不修改动作、不补跑。", "",
              "## 5. 复算与数据", "",
              "从仓库顶层运行：`e2e/post_fg/dist/venv/Scripts/python.exe -B e2e/dstu_supplement_gate_analysis.py`。脚本仅使用Python标准库读取CSV/JSON并生成分析文件；324份联合JSON摘要/指标、108条单跑窗口与来源摘要及原选择表复算全部一致，没有导入仿真器、CPCM或服务器工具。", "",
              "- `tables/gate_diagnosis_samples.csv`：108条预测、实际窗口、策略差、O/ΔJ和原始结果路径。", "- `tables/gate_diagnosis_phases.csv`：36场景的三种子均值、范围、退化种子及对整体差距的贡献。", "- `tables/gate_diagnosis_counterfactual.csv`：固定、原在线和两个实际结束诊断对比。", "- `tables/gate_diagnosis_gap_components.csv`：六次预测误关与十二次P单跑窗口判断正确的旁路分别贡献多少差距。", "- `evidence/gate_diagnosis_audit.json`：输入摘要、统计计数、与原表等值及零新增调用记录。", "", "百分比是各配对百分比的均值；三个种子用于展示已有结果的方向和范围。上述差距分解是这108组数据的精确算术分解。"]
    (OUT / "ONLINE_GATE_DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
