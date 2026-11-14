"""Read saved H/I results for closeout P0; no simulation or search imports."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
HI = ROOT / "e2e/hi"
KEYS = ("model", "hardware_profile", "phase", "op", "method")
SEEDS = (123, 1, 2, 3, 4)
SOURCES = {}


def source(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    data = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
    return data.decode("utf-8-sig")


def read(path):
    return json.loads(source(path))


def csv_read(path):
    return list(csv.DictReader(source(path).splitlines()))


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                         for k, v in r.items()} for r in rows)


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def stats(values):
    return dict(n=len(values), median=statistics.median(values), p90=percentile(values, .9),
                minimum=min(values), maximum=max(values))


def selector_tables():
    candidates = csv_read(HI / "tables/h3_i2_candidates.csv")
    groups = defaultdict(list)
    for row in candidates:
        for k in ("backend_mean_ns", "random_cpcm_mean_ns", "deterministic_cpcm_ns", "own_selector_primary", "own_cost"):
            row[k] = float(row[k])
        row["candidate_order"] = int(row["candidate_order"])
        record = read(row["source"].replace("\\", "/"))
        assert record["candidate_id"] == row["candidate_id"]
        assert [v["seed"] for v in record["backend"]] == list(range(1, 11))
        assert statistics.mean(v["cycles"] for v in record["backend"]) == row["backend_mean_ns"]
        assert statistics.mean(record["T_hat_rand"]) == row["random_cpcm_mean_ns"]
        assert record["T_hat"] == row["deterministic_cpcm_ns"]
        if "own_cost" in record:
            assert record["own_cost"] == row["own_cost"]
        groups[tuple(row[k] for k in KEYS)].append(row)
    assert len(candidates) == 896 and len(groups) == 64
    selectors = dict(
        own=lambda r: (r["own_selector_primary"], r["own_cost"], r["candidate_order"]),
        unified_det_cpcm=lambda r: (r["deterministic_cpcm_ns"], r["candidate_order"]),
        random_cpcm=lambda r: (r["random_cpcm_mean_ns"], r["candidate_order"]))
    output = []
    for key, rows in sorted(groups.items()):
        assert len(rows) == 14 and {r["candidate_order"] for r in rows} == set(range(14))
        best = min(rows, key=lambda r: (r["backend_mean_ns"], r["candidate_order"]))
        for protocol, select in selectors.items():
            chosen = min(rows, key=select)
            output.append(dict(zip(KEYS, key), protocol=protocol, candidates=14,
                selected_candidate=chosen["candidate_id"], backend_best_candidate=best["candidate_id"],
                selected_backend_mean_ns=chosen["backend_mean_ns"], backend_best_mean_ns=best["backend_mean_ns"],
                selection_regret=chosen["backend_mean_ns"] / best["backend_mean_ns"] - 1,
                same_candidate=chosen["candidate_id"] == best["candidate_id"],
                zero_regret=chosen["backend_mean_ns"] == best["backend_mean_ns"],
                selector_values=select(chosen), selected_candidate_order=chosen["candidate_order"],
                backend_best_candidate_order=best["candidate_order"],
                selected_source=chosen["source"], backend_best_source=best["source"]))
    summaries = []
    for protocol in selectors:
        rows = [r for r in output if r["protocol"] == protocol]
        summaries.append(dict(protocol=protocol, **stats([r["selection_regret"] for r in rows]),
            same_candidate=sum(r["same_candidate"] for r in rows), zero_regret=sum(r["zero_regret"] for r in rows)))
    assert [r["same_candidate"] for r in summaries] == [16, 38, 53]
    return output, summaries


def joint_tables():
    rows, windows = [], []
    for model, folder in (("llama", "h5"), ("qwen3moe", "i5")):
        source_rows = csv_read(HI / f"tables/{folder}_joints.csv")
        assert len(source_rows) == 63
        base = HI / f"out/{folder}/results"
        for row in source_rows:
            r = read(base / (row["job_id"] + ".json"))
            s, sim = r["spec"], r["simulation"]
            solo = read(base / (s["D_solo"] + ".json"))
            psolo = read(base / (s["P_solo"] + ".json"))
            graph = read(base / (s["graph_job"] + ".json"))
            solo_graph = read(base / (solo["spec"]["graph_job"] + ".json"))
            reference = read(base / (s["reference_job"] + ".json"))
            assert s["seed"] == solo["spec"]["seed"] and s["backend"] == solo["spec"]["backend"]
            field = f"{s['backend']}_1"
            assert graph["platform_sha256"] == solo_graph["platform_sha256"]
            assert graph["stream_identity"][field] == solo_graph["stream_identity"][field]
            assert graph["actual_iteration_sha256"][field] == solo_graph["actual_iteration_sha256"][field]
            tau = int(round(reference["KV_start"] + s["phase"] * (reference["KV_end"] - reference["KV_start"])))
            assert r["tau"] == tau == sim["release"]["1"] == float(row["tau_ns"])
            cp, cd = (sim["streams"][str(i)]["C"] for i in (0, 1))
            ds = solo["simulation"]["streams"]["1"]["C"]
            ps = psolo["simulation"]["streams"]["0"]["C"]
            dlatency = cd - tau
            assert dlatency == float(row["D_latency"]) and ds == float(row["D_solo"])
            assert cp == float(row["P_completion"]) and ps == float(row["P_solo"])
            common = dict(model=model, condition=s["condition"], arm=s["arm"], backend=s["backend"],
                phase=s["phase"], seed=s["seed"], job_id=r["job_id"],
                source=(base / (r["job_id"] + ".json")).relative_to(ROOT).as_posix())
            rows.append(dict(**common, D_joint_latency_ns=dlatency, D_same_backend_solo_ns=ds,
                D_interference=dlatency / ds - 1, P_completion_ns=cp, P_same_backend_solo_ns=ps,
                P_interference=cp / ps - 1, D_solo_job=s["D_solo"], P_solo_job=s["P_solo"],
                tau_D_ns=tau, reference_job=s["reference_job"], paired_identity_verified=True))
            kv_start, kv_end = sim["kv_window"]
            intersection = max(0, min(kv_end, cd) - max(kv_start, tau))
            windows.append(dict(**common, tau_D_ns=tau, backend_KV_start_ns=kv_start,
                backend_KV_end_ns=kv_end, backend_D_completion_ns=cd,
                positive_window_overlap=intersection > 0, overlap_duration_ns=intersection,
                D_joint_latency_ns=dlatency, D_same_backend_solo_ns=ds,
                P_completion_ns=cp, reference_KV_start_ns=reference["KV_start"],
                reference_KV_end_ns=reference["KV_end"], reference_job=s["reference_job"],
                time_definition="backend absolute time; D alive interval [tau_D, D_completion]"))
    grouped = defaultdict(list)
    for r in rows:
        for phase in (r["phase"], "all"):
            grouped[r["model"], r["condition"], r["arm"], r["backend"], phase].append(r)
    summaries = []
    for (model, condition, arm, backend, phase), values in grouped.items():
        summaries.append(dict(model=model, condition=condition, arm=arm, backend=backend, phase=phase,
            **stats([r["D_interference"] for r in values]),
            negative_samples=sum(r["D_interference"] < 0 for r in values)))
    return rows, windows, summaries


def eir_tables():
    rows = []
    point_count = Counter()
    for prefix, folder in (("h4", "h4_pf4"), ("i4", "i4_light"), ("i4_watos_dffn", "i4_watos_dffn")):
        states = csv_read(HI / f"tables/{prefix}_states.csv")
        units = csv_read(HI / f"tables/{prefix}_units.csv")
        stops = csv_read(HI / f"tables/{prefix}_stop_decisions.csv")
        base = HI / f"out/{folder}/results"
        lookup = {(r["job_id"], r["state"]): r for r in states}
        for unit in units:
            if unit["start_arm"] != "A3h":
                continue
            level = unit["EIR_level"]
            point_count[level] += 1
            stop = next(r for r in stops if r["config_id"] == unit["config_id"]
                and r["phase_op"] == unit["phase_op"] and r["stop"] == "True" and r["decision"] == "evaluated")
            stop_record = read(base / (stop["job_id"] + ".json"))
            assert stop_record["source_job"] == unit["EIR_source_job"] and stop_record["level"] == level
            if level.startswith("s"):
                budget = int(level[1:])
                candidates = [r for r in states if r["config_id"] == unit["config_id"]
                    and r["phase_op"] == unit["phase_op"] and r["state"] == "L2" and int(r["budget"]) == budget]
                assert {int(r["repair_seed"]) for r in candidates} == set(SEEDS)
                for r in candidates:
                    saved = read(base / (r["job_id"] + ".json"))
                    assert saved["status"] == "ok" and saved["steps"] == budget
                    assert saved["T"] == float(r["T_hat_ns"]) and saved["backend_mean"] == float(r["backend_mean_ns"])
                    assert saved["migration"]["unique_block_bytes"] == int(r["migration_unique_block_bytes"])
                fixed = next(r for r in candidates if int(r["repair_seed"]) == 123)
                minimum = min(candidates, key=lambda r: (float(r["T_hat_ns"]), SEEDS.index(int(r["repair_seed"]))))
                median_value = statistics.median(float(r["backend_mean_ns"]) for r in candidates)
                median = min((r for r in candidates if float(r["backend_mean_ns"]) == median_value),
                             key=lambda r: int(r["repair_seed"]))
                assert median["job_id"] == unit["EIR_source_job"]
            else:
                candidates = [lookup[unit["EIR_source_job"], level]]
                fixed = minimum = median = candidates[0]
                saved = read(base / (fixed["job_id"] + ".json"))
                assert saved[level + "_backend_mean"] == float(fixed["backend_mean_ns"])
            for rule, chosen in (("fixed_seed123", fixed), ("minimum_T_hat", minimum), ("backend_median", median)):
                backend = float(chosen["backend_mean_ns"])
                rows.append(dict(model=unit["model"], hardware_profile=unit["hardware_profile"],
                    config_id=unit["config_id"], phase_op=unit["phase_op"], fault="PF4", start_arm="A3h",
                    stop_level=level, stop_reason=unit["EIR_reason"], candidates=len(candidates), rule=rule,
                    selected_job=chosen["job_id"], repair_seed=chosen["repair_seed"],
                    T_hat_ns=float(chosen["T_hat_ns"]), backend_mean_ns=backend,
                    healthy_backend_mean_ns=float(unit["healthy_backend_mean_ns"]),
                    R0_backend_mean_ns=float(unit["R0_backend_mean_ns"]),
                    relative_to_healthy=backend / float(unit["healthy_backend_mean_ns"]) - 1,
                    relative_to_R0=backend / float(unit["R0_backend_mean_ns"]) - 1,
                    relative_to_main_representative=backend / float(unit["EIR_backend_mean_ns"]) - 1,
                    same_as_main_representative=chosen["job_id"] == unit["EIR_source_job"],
                    migration_unique_block_bytes=int(chosen["migration_unique_block_bytes"]),
                    migration_moved_behas=int(chosen["migration_moved_behas"]),
                    migration_moved_behas_share=float(chosen["migration_moved_behas_share"]),
                    migration_ddr_read_moved_bytes=int(chosen["migration_ddr_read_moved_bytes"]),
                    migration_ddr_read_moved_share=float(chosen["migration_ddr_read_moved_share"]),
                    source=(base / (chosen["job_id"] + ".json")).relative_to(ROOT).as_posix()))
    assert len(rows) == 66
    summaries = [dict(rule=rule, **stats([r["relative_to_main_representative"] for r in rows if r["rule"] == rule]),
        same_as_main=sum(r["same_as_main_representative"] for r in rows if r["rule"] == rule))
        for rule in ("fixed_seed123", "minimum_T_hat", "backend_median")]
    return rows, summaries, dict(point_count)


def plot_windows(output, windows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    colors = {"A3h_EIR_no_DSTU": "#386cb0", "A3h_EIR_A256": "#e08214"}
    for ax, model, title in zip(axes, ("llama", "qwen3moe"), ("Llama", "Qwen3-MoE")):
        for arm, color in colors.items():
            offset = -.035 if arm.endswith("no_DSTU") else .035
            for g in (.1, .5, .9):
                selected = [r for r in windows if r["model"] == model and r["condition"] == "D" and r["arm"] == arm and r["phase"] == g]
                lo, hi, done = [statistics.median(r[k] for r in selected) / 1e6
                    for k in ("backend_KV_start_ns", "backend_KV_end_ns", "backend_D_completion_ns")]
                ax.plot([g + offset] * 2, [lo, hi], color=color, lw=5, solid_capstyle="butt")
                ax.plot(g + offset, done, "o", color=color, markersize=5)
                if arm.endswith("no_DSTU"):
                    ax.plot(g, selected[0]["tau_D_ns"] / 1e6, "v", color="black", markersize=6)
            ax.plot([], [], color=color, lw=5, label="original" if arm.endswith("no_DSTU") else "A256")
        ax.plot([], [], "kv", label="共同 D 释放时刻")
        ax.plot([], [], "ko", label="D 完成时刻")
        ax.set(title=title + " · PF4 修复后", xlabel="基于共同 A1 参考的释放相位 g", ylabel="后端绝对时刻（ms）",
               xticks=[.1, .5, .9], xlim=(-.02, 1.02))
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("实测 KV 窗口与共同释放时刻\n竖线：三种子 KV 起止中位数；圆点：D 完成中位数", fontsize=12)
    fig.tight_layout(rect=(0, .07, 1, 1))
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / "p0_kv_windows.png", dpi=180)
    fig.savefig(folder / "p0_kv_windows.svg")
    plt.close(fig)


def health_note():
    path = ROOT / "e2e/H_HEALTH_SUMMARY.md"
    old = path.read_text(encoding="utf-8")
    title = "## WATOS 与 Qwen 试点"
    replacement = "## WATOS 与 Qwen 历史试点\n\n以下表格保留 2026-11-14 第一批试点记录。后续 H/I 健康候选已全部完成（各 448，共 896）；完整十四候选结果见 [H3/I2/I3 汇总](hi/H3_I2_REPORT.md)，Qwen 主硬件完整层结果与 WATOS MHA/FFN 子集见 [I_HEALTH_SUMMARY.md](I_HEALTH_SUMMARY.md)。本节的“试点”描述对应当时进度。"
    if title in old:
        path.write_text(old.replace(title, replacement, 1), encoding="utf-8")


def report(output, selector_summary, interference, eir, eir_summary, point_count):
    lines = ["# H/I 收尾 P0：已有结果的四项补充", "",
        "本报告只读取既有 CSV 和 JSON，新增搜索 0、后端评估 0。候选与故障状态均沿用原结果，H5/I5 的输入和原 300 步主结果不变。", "",
        "## 完成状态", "",
        "健康候选 896/896、64 个十四候选组已齐。Llama PF4 12/12，Qwen PF4 在本次后处理基线中为 10/12；剩余两个 prefill FFN 点由 B 包补齐。新模型 MF6/Split、GPT S 八组合及剩余 J 计时按本轮计划执行，状态见收尾总览；旧 J 第一批 45 项保留，H6 后置。", "",
        "[原批次完整汇总](../HI_FOLLOWUP_SUMMARY.md)；[本轮计划](../../HI_CLOSEOUT_PLAN.md)。H_HEALTH_SUMMARY 的末节已标为历史试点并链接完整结果。", "",
        "## 64 组候选的选择损失", "",
        "每组固定模型、硬件、阶段、算子和方法，共 14 个候选。以各候选十种子后端均值计算所选值/组内最低值−1。所有浮点数使用原始精度；P90 使用排序后的线性插值。这是选择损失，不是 CPCM 的延迟估值误差。", "",
        "自身代价沿用原选择键：A3h 先看保存的搜索确定性估值、再看自身能量；A1/Gemini 看自身能量。统一确定性和随机 CPCM 看相应估值；最后均按原 candidate_order 破同分。", "",
        "| 选择方式 | 同一赢家 /64 | 零损失 /64 | 中位数 | P90 | 最大 |",
        "|---|---:|---:|---:|---:|---:|"]
    labels = dict(own="自身代价", unified_det_cpcm="统一确定性 CPCM", random_cpcm="十次随机 CPCM 均值")
    for r in selector_summary:
        lines.append(f"| {labels[r['protocol']]} | {r['same_candidate']} | {r['zero_regret']} | {r['median']:.3%} | {r['p90']:.3%} | {r['maximum']:.3%} |")
    lines += ["", "逐组 192 行见 [selector_regret.csv](tables/selector_regret.csv)，包含所选 ID、后端赢家 ID、两者后端均值与选择键。候选相同和延迟零损失分别计数；两个不同状态的后端均值相同时，后者仍算零损失。", "",
        "## 同后端单跑干扰与 KV 窗口", "",
        "共 126 个联合结果（H5/I5 各 63）。D 干扰 = D 联合延迟 / 同种子、同执行方式的 D 单跑延迟−1。配对核对了平台、完整流身份和实际集合遍历身份。负值保留：两次执行的随机仲裁轨迹不同，联合运行也可能略短。", "",
        "| 模型 | 条件 | 实验臂 | D 干扰中位数 [最小, 最大] | 负值 /9 |",
        "|---|---|---|---:|---:|"]
    for r in interference:
        if r["phase"] == "all":
            lines.append(f"| {r['model']} | {r['condition']} | {r['arm']} | {r['median']:+.3%} [{r['minimum']:+.3%}, {r['maximum']:+.3%}] | {r['negative_samples']} |")
    lines += ["", "修复后的 original 中，Llama 的 D 干扰中位数只有 +0.293%，Qwen 为 +4.224%；A256 后分别为 +0.058% 和 −0.384%。这说明两模型留给传输方案消除的干扰量不同，需结合 P 完成时间及执行方式本身的成本一起判断。", "",
        "![实际 KV 窗口与共同 D 释放时刻](figures/p0_kv_windows.png)", "",
        "图只展示 PF4 修复后 original 与 A256 的三种子中位数，全部七臂、三相位、三种子的原值保存在窗口表。横轴 g 来自共同健康 A1/original 参考，不是各方法自身 KV 已传完的比例。竖线与圆点来自实际后端，不是 CPCM 预测。", "",
        "窗口相交以 max(0,min(KV_end,D_completion)−max(KV_start,tau_D)) 计算，只表示时间相交，不把它当作排队阻塞量。D 存活窗口从统一释放时刻开始。修复后 g=0.9 时，original 的 KV/D 窗口相交样本为 Llama 1/3、Qwen 0/3；A256 分别为 3/3、2/3。这也说明相同 g 并不对应相同的实际 KV 进度。", "",
        "逐行原值：[joint_interference.csv](tables/joint_interference.csv)、[kv_overlap_windows.csv](tables/kv_overlap_windows.csv)；按实验臂及相位汇总：[joint_interference_summary.csv](tables/joint_interference_summary.csv)。", "",
        "## 同一停止档位内的代表状态选择", "",
        f"覆盖已有 22 个 A3h PF4 点，停止档位分布为 `{json.dumps(point_count, ensure_ascii=False)}`。每点固定原停止档位，分别选 seed123、最小确定性 CPCM、原报告的五种子后端中位代表；R0/L1 停止点三列使用同一状态。", "",
        "最小 CPCM 同分时按修复种子顺序 123、1、2、3、4；原后端中位代表沿用既有规则（中位后端值同分时取较小种子）。下表比较相对原主表代表状态的后端变化。", "",
        "| 选择方式 | 相同状态 /22 | 后端变化中位数 | 最小 | 最大 |",
        "|---|---:|---:|---:|---:|"]
    for r in eir_summary:
        lines.append(f"| {r['rule']} | {r['same_as_main']} | {r['median']:+.3%} | {r['minimum']:+.3%} | {r['maximum']:+.3%} |")
    lines += ["", "| 模型/硬件/算子 | 停止档位 | seed123 残差 | 最小 CPCM 残差 | 后端中位代表残差 |", "|---|---|---:|---:|---:|"]
    grouped = defaultdict(dict)
    for r in eir:
        grouped[r["model"], r["hardware_profile"], r["phase_op"], r["stop_level"]][r["rule"]] = r
    for (model, hardware, op, level), values in grouped.items():
        residuals = [values[k]["relative_to_healthy"] for k in ("fixed_seed123", "minimum_T_hat", "backend_median")]
        lines.append(f"| {model}/{hardware}/{op} | {level} | {residuals[0]:+.3%} | {residuals[1]:+.3%} | {residuals[2]:+.3%} |")
    lines += ["", "这里评价的是已有候选的不同选择办法。seed123 一列没有重新实现按自身结果停档的在线控制器；最小 CPCM 一列也需要先产生五个候选。不能把其选择质量配上单条轨迹的成本。", "",
        "[eir_selection_sensitivity.csv](tables/eir_selection_sensitivity.csv) 的 66 行保留每个状态的后端均值、相对健康残差、相对 R0 改善、唯一迁移块字节及行为/DDR 迁移。原主表及 H5/I5 仍使用原后端中位代表。", "",
        "## 复算与来源", "",
        "从仓库根目录运行：", "", "```powershell", "e2e/post_fg/dist/venv/Scripts/python.exe -B e2e/hi_closeout_p0.py", "```", "",
        "[p0_audit.json](evidence/p0_audit.json) 记录输入文件 SHA-256、原始 JSON 与 CSV 的逐项核对、生成时间和实际耗时。脚本只加载保存数据，不导入仿真/搜索模块。表格可独立复算上述统计。", ""]
    (output / "P0_POSTPROCESS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HI / "closeout")
    args = parser.parse_args()
    began = time.perf_counter()
    selector, selector_summary = selector_tables()
    interference, windows, interference_summary = joint_tables()
    eir, eir_summary, point_count = eir_tables()
    args.output.mkdir(parents=True, exist_ok=True)
    tables = dict(selector_regret=selector, selector_regret_summary=selector_summary,
        joint_interference=interference, joint_interference_summary=interference_summary,
        kv_overlap_windows=windows, eir_selection_sensitivity=eir, eir_selection_summary=eir_summary)
    for name, rows in tables.items():
        csv_write(args.output / "tables" / (name + ".csv"), rows)
    plot_windows(args.output, windows)
    health_note()
    report(args.output, selector_summary, interference_summary, eir, eir_summary, point_count)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == value for p, value in SOURCES.items())
    audit = dict(status="complete", created_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter() - began, new_searches=0, new_backend_calls=0,
        candidate_count=896, candidate_groups=64, selector_rows=len(selector),
        joint_rows=len(interference), window_rows=len(windows), fault_points=22,
        eir_selection_rows=len(eir), original_input_sha256=SOURCES,
        candidate_backend_random_and_deterministic_values_verified=True,
        paired_solo_platform_stream_iteration_and_seed_verified=True,
        all_final_stop_levels_and_eir_representatives_verified=True,
        original_inputs_unchanged_after_postprocessing=True,
        quantile="linear interpolation at (n-1)*q", output_counts={k:len(v) for k,v in tables.items()})
    json_write(args.output / "evidence/p0_audit.json", audit)
    print(json.dumps({k:v for k,v in audit.items() if k != "original_input_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
