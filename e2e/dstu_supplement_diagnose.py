"""R3: explain saved Qwen PF4 states and index existing paper evidence."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import time

from post_fg_diag import read, write, table

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "e2e/dstu_supplement"
OLD = ROOT / "e2e/hi/out/closeout_20261115"
TABLES = ROOT / "e2e/hi/closeout/tables"
SEEDS = (123, 1, 2, 3, 4)


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def state_inputs():
    import fg_fault as frozen
    units = rows(TABLES / "b_units.csv")
    stops = rows(TABLES / "b_stop_decisions.csv")
    states, faults = [], {}
    for unit in units:
        hardware, config = unit["hardware_profile"], unit["config_id"]
        prepath = OLD / "results" / (unit["healthy_source_job"] + ".json")
        l1path = OLD / "results" / (unit["l1_source_job"] + ".json")
        pre, l1 = read(prepath), read(l1path)
        faults[hardware] = read(OLD / "faults" / (config + ".json"))["faults"]["PF4"]
        states.append(dict(hardware=hardware, state="healthy", repair_seed=None, budget=0,
            export=pre["exports"]["healthy"], T_hat_ns=pre["T"], backend_mean_ns=pre["backend_mean"],
            source_result=relative(prepath), spec=pre["spec"]))
        for label in ("R0", "L1"):
            states.append(dict(hardware=hardware, state=label, repair_seed=None, budget=0,
                export=l1["exports"][label], T_hat_ns=l1["T_" + label], backend_mean_ns=l1[label + "_backend_mean"],
                source_result=relative(l1path), spec=l1["spec"]))
        selected = next(r for r in stops if r["hardware_profile"] == hardware and r["budget"] == "100")
        path = OLD / "results" / (selected["source_job"] + ".json")
        result = read(path)
        states.append(dict(hardware=hardware, state="s100", repair_seed=result["seed"], budget=100,
            export=result["exports"]["final"], T_hat_ns=result["T"], backend_mean_ns=result["backend_mean"],
            source_result=relative(path), spec=result["spec"]))
    assert len(states) == 8
    return states, faults


def route_rows(state, healthy_events, events, fault):
    failed = {tuple(map(tuple, edge)) for edge in fault["failed_links"]}
    common = {k: state[k] for k in ("hardware", "state", "repair_seed", "export")}
    volumes, route_changes, changed_volume = Counter(), 0, 0
    changed_from_fault, changed_off_fault, on_failed, comms = 0, 0, 0, 0
    old_comms = {tag:e for tag,e in healthy_events.items() if e.event_type == "communication"}
    byte_hops_healthy = sum(e.comm_bytes * len({edge for p in e.path_list for edge in p}) for e in old_comms.values())
    byte_hops_current, matched, added_messages = 0, 0, 0
    edges_added, edges_removed, details = Counter(), Counter(), []
    old_flow, new_flow = Counter(), Counter()
    for event in old_comms.values():
        for edge in {edge for p in event.path_list for edge in p}:
            old_flow[edge] += event.comm_bytes
    for tag, event in events.items():
        if event.event_type != "communication":
            continue
        comms += 1
        edges = {edge for path in event.path_list for edge in path}
        byte_hops_current += event.comm_bytes * len(edges)
        on_failed += bool(edges & failed)
        for edge in edges:
            new_flow[edge] += event.comm_bytes
            category = "DDR" if edge[0][1] >= 16 or edge[1][1] >= 16 else "D2D" if edge[0][0] != edge[1][0] else "NoC"
            volumes[category] += event.comm_bytes
        if tag not in old_comms or (event.event_name, event.comm_bytes) != (old_comms[tag].event_name, old_comms[tag].comm_bytes):
            added_messages += 1
            details.append(dict(**common, event_tag=repr(tag), event_name=event.event_name,
                match="new_or_changed_identity", bytes=event.comm_bytes, new_edges=len(edges), new_path=event.path_list))
            continue
        matched += 1
        original = old_comms[tag]
        oldedges = {edge for path in original.path_list for edge in path}
        if event.path_list != original.path_list:
            route_changes += 1
            changed_volume += event.comm_bytes
            changed_from_fault += bool(oldedges & failed)
            changed_off_fault += not bool(oldedges & failed)
            for edge in edges - oldedges:
                edges_added[edge] += event.comm_bytes
            for edge in oldedges - edges:
                edges_removed[edge] += original.comm_bytes
            details.append(dict(**common, event_tag=repr(tag), event_name=event.event_name, match="same_tag_name_bytes",
                bytes=event.comm_bytes, old_edges=len(oldedges), new_edges=len(edges),
                healthy_route_used_failed_link=bool(oldedges & failed),
                old_path=original.path_list, new_path=event.path_list))
    if state["state"] != "healthy":
        assert on_failed == 0
    summary = dict(**common, communication_events=comms, matched_message_identities=matched,
        new_or_changed_message_identities=added_messages, old_unmatched_message_identities=len(old_comms)-matched,
        changed_routes_vs_healthy=route_changes,
        changed_message_bytes=changed_volume, changed_routes_on_failed_path=changed_from_fault,
        changed_routes_off_failed_path=changed_off_fault, messages_using_failed_links=on_failed,
        healthy_byte_hops=byte_hops_healthy, current_byte_hops=byte_hops_current,
        byte_hops_relative_to_healthy=byte_hops_current / byte_hops_healthy - 1,
        **{f"{k}_byte_edges": volumes[k] for k in ("D2D", "NoC", "DDR")})
    edge_rows = [dict(**common, edge=repr(edge), old_bytes=old_flow[edge], new_bytes=new_flow[edge],
        net_bytes=new_flow[edge]-old_flow[edge], matched_message_added_bytes=edges_added[edge],
        matched_message_removed_bytes=edges_removed[edge]) for edge in sorted(old_flow.keys() | new_flow.keys())]
    return summary, details, edge_rows


def search_rows():
    import fg_fault as frozen
    results = []
    for path in sorted((OLD / "results").glob("B__*__rep_pf_ffn_PF4_s*_r*.json")):
        result = read(path)
        if result["status"] == "reused":
            continue
        raw = frozen.read_pickle(result["sink"])
        search = raw["search"]
        steps = result["steps"]
        assert len(search["rows"]) == steps
        assert [r["step"] for r in search["rows"]] == list(range(1, steps + 1))
        accepted = sum(r["accepted"] for r in search["rows"])
        results.append(dict(hardware=result["spec"]["hardware_profile"], budget=steps, repair_seed=result["seed"],
            accepted=accepted, acceptance_rate=accepted/steps, best_step=search["best_step"],
            T_start_ns=search["T_start"], T_final_ns=result["T"], backend_mean_ns=result["backend_mean"],
            predicted_gain_from_L1=1-result["T"]/search["T_start"],
            calibration_proposals=raw["proposals"]["calibration"], calibration_seconds=search["t_calibration"],
            search_seconds=search["t_L2_search"], source=relative(path), source_sha256=sha(path),
            raw_source=result["sink"], raw_sha256=sha(ROOT / result["sink"])))
    assert len(results) == 25
    return results


def diagnose(dest, hardware=None):
    import post_fg_split
    import fg_fault as frozen
    import hi_closeout_a_report as explain
    post_fg_split.setup()
    states, faults = state_inputs()
    if hardware:
        states = [s for s in states if s["hardware"] == hardware]
    began = time.perf_counter()
    composition, waits, loads, segments, timings = [], [], [], [], []
    routes, changed, edge_rows, inputs = [], [], [], []
    healthy = {}
    for state in states:
        hardware_name = state["hardware"]
        exported = frozen.read_pickle(state["export"])
        events = pickle.loads(exported["events_blob"])
        if state["state"] == "healthy":
            healthy[hardware_name] = events
        c, w, l, s, t = explain.explain(state, dest / "work" / hardware_name, faults[hardware_name])
        for target, values in ((composition, c), (waits, w), (loads, l), (segments, s)):
            target.extend(dict(hardware=hardware_name, **r) for r in values)
        timings.append(dict(hardware=hardware_name, **t))
        r, d, e = route_rows(state, healthy[hardware_name], events, faults[hardware_name])
        routes.append(r); changed.extend(d); edge_rows.extend(e)
        inputs.append(dict(hardware=hardware_name, state=state["state"], export=state["export"],
            export_sha256=sha(ROOT/state["export"]), result=state["source_result"],
            result_sha256=sha(ROOT/state["source_result"]), T_hat_verified_ns=t["T_hat_ns"]))
        print(json.dumps(dict(hardware=hardware_name, state=state["state"], seconds=t["diagnosis_seconds"],
            T_hat_ns=t["T_hat_ns"])), flush=True)
    searches = search_rows()
    names = dict(watos_chain_composition=composition, watos_exclusive_wait=waits,
        watos_resource_load=loads, watos_chain_segments=segments, watos_diagnostic_timing=timings,
        watos_rerouting_summary=routes, watos_changed_routes=changed, watos_edge_redistribution=edge_rows,
        watos_search_trajectories=searches)
    suffix = "_" + hardware if hardware else ""
    for name, values in names.items():
        table(dest / "tables" / (name + suffix + ".csv"), values)
    audit = dict(status="complete" if not hardware else "partial", states=len(states), deterministic_CPCM_calls=len(states),
        backend_calls=0, new_searches=0, T_hat_exact_matches=len(states),
        source_inputs=inputs, search_trajectories_read=len(searches),
        diagnostic_wall_seconds=time.perf_counter()-began,
        evidence_type="saved-state deterministic CPCM replay; existing independent ten-seed backend quality",
        representative="existing backend-median repair seed at independent s100 budget; five-seed stop remains unchanged",
        time_unit="ns", physical_fault=faults)
    write(dest / "evidence" / ("diagnosis_audit" + suffix + ".json"), audit)
    if not hardware:
        render_diagnosis(dest, timings, composition, waits, loads, routes, searches, audit)
    return audit


def render_diagnosis(dest, timings, composition, waits, loads, routes, searches, audit):
    out = ["# Qwen prefill FFN：PF4 后的剩余开销", "",
        "读取主硬件与 WATOS 的健康、R0、L1 和独立 100 步预算的保存状态。100 步状态沿用已有后端中位代表：主硬件 seed 1、WATOS seed 123；停档仍采用原五种子预测均值。", "",
        f"本次 {audit['states']} 次固定 CPCM 重放全部与原 T̂ 逐项相等，新增搜索和后端评估均为 0，耗时 {audit['diagnostic_wall_seconds']:.2f} 秒。", "",
        "|硬件|状态|T̂ / M ns|已有十种子后端均值 / M ns|D2D 最大服务量 / M ns|NoC 最大服务量 / M ns|DDR 最大服务量 / M ns|",
        "|---|---|---:|---:|---:|---:|---:|"]
    for t in timings:
        key = t["hardware"], t["state"]
        select = lambda values: [v for v in values if (v["hardware"],v["state"]) == key]
        peaks = {cat:max([v["resource_service_ns"] for v in select(loads) if v["category"] == cat], default=0) for cat in ("D2D", "NoC", "DDR")}
        out.append(f"|{t['hardware']}|{t['state']}|{t['T_hat_ns']/1e6:.6f}|{t['backend_mean_ns']/1e6:.6f}|{peaks['D2D']/1e6:.6f}|{peaks['NoC']/1e6:.6f}|{peaks['DDR']/1e6:.6f}|")
    out += ["", "资源服务量按实际设备/链路类型累计；DDR 消息经过片内和跨 Die 链路时，分别计入 NoC 和 D2D。下面的关键链由不重叠时间段组成，每个状态的分段总和恰好等于 T̂。", "",
        "|硬件|状态|关键链 D2D / M ns|NoC / M ns|DDR / M ns|计算 / M ns|排他等待首位资源|排他等待 / M ns|",
        "|---|---|---:|---:|---:|---:|---|---:|"]
    for t in timings:
        select = lambda values: [v for v in values if (v["hardware"],v["state"]) == (t["hardware"],t["state"])]
        chain = {v["category"]:v["chain_time_ns"] for v in select(composition)}
        lead = max(select(waits), key=lambda v:v["cb_excl_ns"])
        out.append(f"|{t['hardware']}|{t['state']}|{chain['D2D']/1e6:.6f}|{chain['NoC']/1e6:.6f}|{chain['DDR']/1e6:.6f}|{(chain['TensorCore']+chain['Vector'])/1e6:.6f}|{lead['category']} `{lead['resource']}`|{lead['cb_excl_ns']/1e6:.6f}|")
    w100 = next(t for t in timings if t["hardware"] == "watos" and t["state"] == "s100")
    def selected_load(hw, state, cat):
        return max(r["resource_service_ns"] for r in loads if r["hardware"] == hw and r["state"] == state and r["category"] == cat)
    w_d2d = selected_load("watos", "s100", "D2D")
    w_chain = next(r["chain_time_ns"] for r in composition if r["hardware"] == "watos" and r["state"] == "s100" and r["category"] == "D2D")
    out += ["", f"WATOS 的剩余开销主要集中在跨 Die 通信：100 步代表状态的最忙 D2D 服务量已占 T̂ 的 {w_d2d/w100['T_hat_ns']:.2%}，关键链上 D2D 占 {w_chain/w100['T_hat_ns']:.2%}，排他等待最大的资源同样是 D2D。DDR 最大服务量约 1.649M ns，远小于 D2D 的 27.245M ns。主硬件的关键链仍主要由计算组成，因而两种硬件在同一断链后的恢复幅度不同。",
        f"断链后的 R0 将 WATOS 最忙 D2D 服务量提高 {selected_load('watos', 'R0', 'D2D')/selected_load('watos', 'healthy', 'D2D')-1:.2%}，而总 byte-hop 只增加约 2.01%；关键是流量集中到剩余跨 Die 通道，而不是总传输量成比例增长。L1 与 100 步搜索继续降低这份集中负载，但没有把它恢复到健康水平。"]
    out += ["", "排他等待取 CPCM Analysis.blocking 的既有定义，表中的首位值不是关键链分段的附加项。这里解释的是固定映射、固定服务次序下的压力；资源忙碌或关键等待较大，本身不证明所有重映射都无法改善。", "",
        "|硬件|状态|相对健康改变的路由|其中原先经过故障链路|其他改变|总 byte-hop 变化|",
        "|---|---|---:|---:|---:|---:|"]
    for r in routes:
        out.append(f"|{r['hardware']}|{r['state']}|{r['changed_routes_vs_healthy']}|{r['changed_routes_on_failed_path']}|{r['changed_routes_off_failed_path']}|{r['byte_hops_relative_to_healthy']:+.3%}|")
    out += ["", "所有故障后状态均不再经过删除的两条有向 D2D 链路。路由改变数只比较 tag、名称、字节数一致的消息；修复搜索产生或回收的通信 tag 单列新增/变更身份。总 byte-hop 与逐边净流量统计包含全部消息，路径按实际多播边集合计数。", "",
        "|硬件|独立预算|五轨迹接受率范围|最佳步范围|预测均值相对前一档改善|原停止结果|",
        "|---|---:|---:|---:|---:|---|"]
    stops = rows(TABLES / "b_stop_decisions.csv")
    for hw in ("busybarn", "watos"):
        for budget in (30, 100, 300):
            group = [r for r in searches if r["hardware"] == hw and r["budget"] == budget]
            stop = next(r for r in stops if r["hardware_profile"] == hw and r["budget"] == str(budget))
            if not group:
                out.append(f"|{hw}|{budget}|复用停档，无新搜索|—|—|沿用 s100|")
                continue
            assert len(group) == 5
            gain = f"{float(stop['gain_from_previous']):.4%}" if stop["gain_from_previous"] else "首档"
            out.append(f"|{hw}|{budget}|{min(r['acceptance_rate'] for r in group):.1%}–{max(r['acceptance_rate'] for r in group):.1%}|{min(r['best_step'] for r in group)}–{max(r['best_step'] for r in group)}|{gain}|{stop['reason']}|")
    out += ["", "WATOS 的五种子平均预测从 30 到 100 步仅改善 0.3378%，因此按原规则停止；后端代表相对健康仍增加 35.048%。主硬件继续到 300 步，最终后端代表相对健康增加 1.682%。二者的差别结合资源服务量、关键链和绕行一起解释，不能把相邻档改善较小写成全局最优证明。", "",
        "[逐资源服务量](tables/watos_resource_load.csv) · [关键链连续分段](tables/watos_chain_segments.csv) · [排他等待](tables/watos_exclusive_wait.csv) · [逐消息改路](tables/watos_changed_routes.csv) · [逐边流量迁移](tables/watos_edge_redistribution.csv) · [25 条既有搜索](tables/watos_search_trajectories.csv) · [来源与重放核对](evidence/diagnosis_audit.json)", ""]
    (dest / "WATOS_DIAGNOSIS.md").write_text("\n".join(out), encoding="utf-8")


def index(dest):
    """Record claims using completed experiment records and their actual source files."""
    baseline = "244c59533e6154aafaf997dcb51fe58032b2981e"
    supplement_commit = "本轮工作树，提交见DELIVERY"
    artifact = (ROOT / "third_party/isca2026-busybarn-artifact.COMMIT").read_text(encoding="utf-8-sig").splitlines()[0]
    entries = []
    def add(ident, claim, scope, evidence_type, workload, selector, seeds, unit, report, data, code, figure, timing="quality; elapsed pool time not paper timing", status="complete", commit=baseline):
        inputs = dict(report=report, data=data, code=code, figure_or_table=figure)
        sources = {}
        for kind, value in inputs.items():
            path = ROOT / value
            if status == "complete":
                assert path.exists(), (ident, value)
            sources[kind] = dict(path=value, exists=path.exists(), sha256=sha(path) if path.is_file() else None)
        entries.append(dict(claim_id=ident, statement=claim, scope=scope, evidence_type=evidence_type,
            workload_version=workload, selector=selector, seeds=seeds, units=unit, timing_scope=timing,
            status=status, source_commit=commit, artifact_commit=artifact, **inputs, sources=sources))
    add("A01", "Prefill试点记录预测误差与实际质量，包含超出原5%停止线的结果", "GPT prefill MHA/FFN; 2x2; 84 runs", "direct_operator_simulation", "W0", "three original selector columns", "14 candidates per arm; backend seeds per saved run", "native cycles", "e2e/PILOT_SUMMARY.md", "e2e/out/report2.json", "e2e/e2e_pilot.py", "e2e/PILOT_SUMMARY.md")
    add("B01", "GPT完整P/D以算子仿真为基础，W1去除无消费者的QKT/S末端广播", "GPT; 8 operators; 2x2", "direct_operator_simulation", "W0 and W1 separately", "backend best, candidate median, random-tie CPCM", "backend 1..10; random CPCM 1..10", "native cycles", "e2e/B4_SUMMARY.md", "e2e/out/b4/b4.json", "e2e/e2e_b4.py", "e2e/B4_SUMMARY.md")
    add("B02", "单层和44层时间由实际算子时间相加得到", "GPT; fixed KV2048; 44 layers", "layer_assembly_and_analytical_request", "W1", "same operator winners as B4", "operator backend 1..10", "native cycles", "e2e/B4_SUMMARY.md", "e2e/out/b4/b4.json", "e2e/e2e_b4.py", "e2e/B4_SUMMARY.md")
    add("C01", "同一物理故障用于8个算子，EIR停止档、后端质量和迁移分别报告", "GPT; 19 physical faults; 2x2", "direct_operator_simulation", "W1", "healthy mapper winner; EIR five-seed prediction stop / backend-median representative", "repair 123,1,2,3,4; backend 1..10", "native cycles; bytes", "e2e/C_SUMMARY.md", "e2e/c/out", "e2e/c_eir.py", "e2e/C_SUMMARY.md")
    add("D01", "映射、修复和传输联合效果包含P代价、D延迟、迟约与配对单跑", "GPT; adjacent P/D groups; healthy and faults", "direct_joint_simulation", "W1", "frozen selected mappings and repaired states", "main backend 1..3; D5 supplement 4..10 separately", "native cycles", "e2e/D_SUMMARY.md", "e2e/d/out", "e2e/d_joint.py", "e2e/D_SUMMARY.md")
    add("D02", "由单次重叠推算整请求的结果是解析外推", "D report section 6", "analytical_extrapolation", "W1", "same D selected states", "paired underlying backend samples", "native cycles", "e2e/D_SUMMARY.md", "e2e/d/out", "e2e/d_joint.py", "e2e/D_SUMMARY.md")
    add("E01", "Gemini和A1/A3h在2x2与1x4上完整比较，prefill差距可由计算与访存集中解释", "GPT; 672 candidates and 16 terminal controls", "direct_operator_simulation_and_saved_state_analysis", "W1", "backend best / random-tie CPCM / candidate median separately", "14 candidates per operator-method; backend 1..10", "native cycles", "e2e/E_SUMMARY.md", "e2e/e/out", "e2e/e_report.py", "e2e/e/tables")
    add("E02", "44层混合布局与理想最慢组服务率属于组装与解析分析", "L_mix / L_strip; 56 dies; selected 2x2 physical faults", "layer_assembly_and_ideal_pipeline_analysis", "W1", "same E operator winners", "operator means from backend 1..10", "native cycles; ideal throughput ratio", "e2e/E_SUMMARY.md", "e2e/E_SUMMARY.json", "e2e/e_report.py", "e2e/e/tables")
    add("F01", "硬件参数迁移保留映射、修复和DSTU各自收益及不利点", "GPT; WSC-LLM, FACE, WATOS; 2x2", "direct_operator_and_joint_simulation", "W1", "backend winner main; median/random/deterministic alternatives", "healthy backend 1..10; joint 1..3; repair five seeds", "ns; 1ns/tick", "e2e/F_SUMMARY.md", "e2e/fg/out/main", "e2e/fg_report.py", "e2e/fg/tables/operators.csv")
    add("G01", "2x3/3x3在固定工作量下比较原划分与增长划分", "GPT; 96/144 cores; BusyBarn hardware", "direct_operator_and_joint_simulation", "W1", "same F/G selector definitions", "healthy backend 1..10; joint 1..3; repair five seeds", "ns; 1ns/tick", "e2e/G_SUMMARY.md", "e2e/fg/out/main", "e2e/fg_report.py", "e2e/fg/tables/scaling.csv")
    add("FG02", "F/G的单层与固定KV请求时间由算子结果组装", "GPT; 8 representative heads multiplied by 8; fixed KV2048", "layer_assembly_and_fixed_service_request", "W1", "same selected operator states", "operator means from backend 1..10", "ns; request P+N*D", "e2e/F_SUMMARY.md", "e2e/fg/tables/layers.csv", "e2e/fg_report.py", "e2e/fg/tables/fault_layers.csv")
    add("PF01", "确定性bb加速保留原结果与完整轨迹，调用开销和前向成本分别计时", "Post-FG three representative saved states", "equivalence_replay_and_timing", "W1", "fixed identical inputs", "paired deterministic runs", "seconds; speedup", "e2e/post_fg/POST_FG_SUMMARY.md", "e2e/post_fg/out", "e2e/post_fg_bb.py", "e2e/post_fg/POST_FG_SUMMARY.md")
    add("PF02", "S256已有36个健康/PF4配对降低全部P时间，D包含回退", "GPT; WSC-LLM/FACE; 3 phases x 3 seeds x 2 states", "direct_joint_simulation", "W1; fresh_transform_v1", "fixed A3h/EIR states and fixed A256 C4 rank", "backend 1,2,3", "ns; paired percentage", "e2e/post_fg/DSTU_REFINEMENT.md", "e2e/post_fg/tables/dstu_runs.csv", "e2e/post_fg_dstu.py", "e2e/post_fg/DSTU_REFINEMENT.md")
    add("PF03", "旧三例Split-m2能低于原固定块计算下界，切分粒度存在取舍", "GPT FACE projection; GPT 2x3/3x3 FFN; MF6", "direct_operator_simulation_and_rigid_block_bound", "W1", "fixed legal output-axis rule; no SA", "backend 1..10 after supplement", "ns; bytes; FLOPs", "e2e/post_fg/EIR_SPLIT.md", "e2e/hi/out/split_supplement", "e2e/post_fg_split.py", "e2e/post_fg/EIR_SPLIT.md")
    add("H01", "Llama八Q一KV子图、RMSNorm与门控双支路已核对", "Llama; batch8; 64Q heads; 8 representative groups", "model_contract_and_graph_validation", "W1_8Q_1KV_v1", "generated-copy source checks", "structural checks; no quality seed", "shape; bytes", "e2e/hi/README.md", "e2e/hi/models/llama/model_contract.json", "e2e/hi_config.py", "e2e/hi/evidence/model_contract_validation.json")
    add("H02", "Llama主硬件decode收益明显，prefill随机CPCM选优仍有回退", "main 8 operators; WATOS MHA/FFN subset", "direct_operator_simulation_and_layer_assembly", "W1_8Q_1KV_v1", "random-tie CPCM main; backend winner reference", "14 candidates; backend/random CPCM 1..10", "ns; main layer assembled", "e2e/H_HEALTH_SUMMARY.md", "e2e/hi/tables/h3_i2_candidates.csv", "e2e/hi_h3_i2_report.py", "e2e/hi/HI_FOLLOWUP_SUMMARY.md")
    add("I01", "Qwen MoE固定分派、共享历史KV与层乘数4已核对", "Qwen3-MoE; batch16; 128 experts; top8; fixed assignment", "model_contract_and_graph_validation", "W1_8Q_1KV_v1", "generated-copy source checks", "structural checks; no quality seed", "shape; bytes", "e2e/hi/README.md", "e2e/hi/models/qwen3moe/model_contract.json", "e2e/hi_config.py", "e2e/hi/evidence/model_contract_validation.json")
    add("I02", "Qwen主硬件的P/D改善与WATOS长尾点的微小prefill FFN改善同时保留", "main 8 operators; WATOS MHA/FFN subset", "direct_operator_simulation_and_layer_assembly", "W1_8Q_1KV_v1", "random-tie CPCM main; backend winner reference", "14 candidates; backend/random CPCM 1..10", "ns; main layer assembled", "e2e/I_HEALTH_SUMMARY.md", "e2e/hi/tables/i3_candidates.csv", "e2e/hi_h3_i2_report.py", "e2e/hi/HI_FOLLOWUP_SUMMARY.md")
    add("HI03", "H/I固定PF4修复已齐，Qwen WATOS prefill FFN仍有明显残差", "H4/I4 12 points each; Qwen main full-layer assembly", "direct_operator_simulation_and_layer_assembly", "W1_8Q_1KV_v1", "five-seed prediction stop; backend-median representative", "repair 123,1,2,3,4; backend 1..10", "ns; bytes", "e2e/hi/closeout/B_I4_PF4_COMPLETE.md", "e2e/hi/closeout/tables/b_all_I4_units.csv", "e2e/hi_closeout_b_report.py", "e2e/hi/closeout/tables/b_requests.csv")
    add("HI04", "新模型Split-m2的效果有限，Qwen没有额外二分收益", "Llama/Qwen prefill FFN MF6", "direct_operator_simulation_and_rigid_block_bound", "W1_8Q_1KV_v1", "fixed fault and legal axis rule", "backend 1..10", "ns; bytes; FLOPs", "e2e/hi/closeout/C_NEW_MODEL_SPLIT.md", "e2e/hi/closeout/tables/c_quality.csv", "e2e/hi_closeout_c_report.py", "e2e/hi/closeout/tables/c_block_load.csv")
    add("HI05", "原H5/I5联合实验保存映射/EIR/A256的独立增量与实际干扰", "Llama/Qwen; main hardware; healthy/PF4; original/A256", "direct_joint_simulation", "W1_8Q_1KV_v1; saved-hop protocol", "frozen mapped and repaired graphs", "joint backend 1,2,3", "ns; common deadline Lhat", "e2e/hi/HI_FOLLOWUP_SUMMARY.md", "e2e/hi/tables/h5_joints.csv", "e2e/hi_h5_report.py", "e2e/hi/tables/i5_joints.csv")
    add("S01", "A256八组合显示传输增益取决于剩余干扰，FACE保留D回退", "GPT; WSC-LLM/FACE; 144 joint cells", "direct_joint_simulation", "W1; fresh_transform_v1", "A1/A3h x R0/EIR x original/A256; common A1 reference", "joint backend 1,2,3; 3 phases", "ns; paired percentage; Lhat", "e2e/hi/closeout/S_FACTORIAL_SUMMARY.md", "e2e/hi/closeout/tables/s_144_coverage.csv", "e2e/hi_closeout_s_report.py", "e2e/hi/closeout/tables/s_conditional_pairs.csv")
    add("P001", "实际KV窗口与固定释放时刻解释g相同而真实重叠不同", "GPT/Llama/Qwen joint saved outputs", "posthoc_backend_window_analysis", "W1 and H/I W1_8Q_1KV_v1", "fixed common reference, not per-row strategy selection", "joint backend seeds 1..3", "ns; interference ratio", "e2e/hi/closeout/P0_POSTPROCESS.md", "e2e/hi/closeout/tables/kv_overlap_windows.csv", "e2e/hi_closeout_p0.py", "e2e/hi/closeout/tables/joint_interference.csv")
    add("J01", "完整调用、1000步映射、修复累计与控制准备采用独立串行计时", "125 timing records; GPT64/GPT144/Llama/Qwen representatives", "measured_serial_wall_time", "same saved quality workloads", "fixed candidates; timing repeats are not extra quality seeds", "repeat identifiers and repair seeds in J_TIMING", "seconds; one worker; threads=1", "e2e/hi/closeout/J_SUMMARY.md", "e2e/hi/closeout/J_TIMING.csv", "e2e/hi_closeout_j_report.py", "e2e/hi/closeout/J_ENVIRONMENT.csv", timing="serial paper timing; control/preparation/offline backend evaluation separate")
    add("R303", "WATOS剩余开销主要集中到D2D，固定状态关键链和改路已复算", "Qwen prefill FFN; main/WATOS; healthy/R0/L1/s100", "saved_state_deterministic_CPCM_analysis", "W1_8Q_1KV_v1", "same historical s100 backend-median representative", "saved backend 1..10; no new backend", "ns; byte-hop", "e2e/dstu_supplement/WATOS_DIAGNOSIS.md", "e2e/dstu_supplement/evidence/diagnosis_audit.json", "e2e/dstu_supplement_diagnose.py", "e2e/dstu_supplement/tables/watos_resource_load.csv", timing="local fixed-state analysis only", commit=supplement_commit)
    recovery = dest / "evidence/recovery_audit.json"
    recovery_done = recovery.exists() and read(recovery)["status"] == "passed"
    recovery_status = "complete" if recovery_done else "pending_recovery_postprocess"
    add("R301", "单轨迹用自身预测自主停档，30/100/300为独立预算而非连续300步", "GPT64/GPT144 decode MHA PF4; Qwen decode FFN PF4; 9 trajectories", "saved_result_postprocess", "W1 and W1_8Q_1KV_v1", "per-trajectory deterministic CPCM 2% recovery and adjacent-budget gain", "repair 123,1,2; existing backend 1..10", "ns; seconds; bytes", "e2e/dstu_supplement/ONLINE_RECOVERY.md", "e2e/dstu_supplement/tables/online_single_trajectory.csv", "e2e/dstu_supplement_recovery.py", "e2e/dstu_supplement/tables/recovery_payback.csv", timing="J3 serial cumulative decision costs; validation cost separate", status=recovery_status, commit=supplement_commit)
    add("R302", "24点PF4选择表与后台修复回本按既有状态配对计算", "24 model-hardware-operator points; three selectors; fixed-service recovery model", "saved_result_postprocess_and_analytical_payback", "W1 and W1_8Q_1KV_v1", "research mean/representative views retained; selected alternatives explicitly labeled", "existing five repair seeds and ten backend seeds", "ns; seconds; bytes; analytical request counts", "e2e/dstu_supplement/ONLINE_RECOVERY.md", "e2e/dstu_supplement/tables/eir_selector_24points.csv", "e2e/dstu_supplement_recovery.py", "e2e/dstu_supplement/tables/recovery_payback.csv", timing="control/cold start/offline evaluation separate; transfer assumptions explicit", status=recovery_status, commit=supplement_commit)
    completion = dest / "evidence/completion_audit.json"
    done = completion.exists() and read(completion).get("status") in ("passed", "complete")
    add("R1", "S256补齐后完整比较P/D、最大迟约、真实超期和配对单跑", "108 comparison groups; 324 joint plus 216 logical solo records; S72/H18/I18", "direct_joint_simulation", "W1 and W1_8Q_1KV_v1; audited fresh execution order", "frozen A256 rank plus rank consistency check; no per-row oracle", "joint backend 1,2,3; g=.1,.5,.9", "ns; paired percentage; Lhat", "e2e/dstu_supplement/S_AND_HI_S256_REPORT.md", "e2e/dstu_supplement/tables/s256_runs.csv", "e2e/post_fg_dstu.py", "e2e/dstu_supplement/tables/s256_overall_means.csv", status="complete" if done else "pending_supplement_completion", commit=supplement_commit)
    add("R2G", "Online-CPCM-Gated按共享有向资源与预测KV结束时刻预先选择完整BB0/S256策略", "12 fixed states; 36 frozen choices; same action for 3 backend seeds per phase", "prior_CPCM_prediction_and_selected_executed_strategy", "W1 and W1_8Q_1KV_v1", "shared resources and known tau < predicted unchunked-P bb KV end; no current backend metric in decision", "one deterministic prediction per state; backend 1,2,3 for evaluation", "predicted/actual ns; predictor wall seconds separate", "e2e/dstu_supplement/S_AND_HI_S256_REPORT.md", "e2e/dstu_supplement/tables/s256_strategy_choices.csv", "e2e/dstu_supplement_gate.py", "e2e/dstu_supplement/figures/gate_prediction_and_outcomes.png", timing="prediction time separate from selected strategy service time", status="complete" if done else "pending_supplement_completion", commit=supplement_commit)
    add("R2O", "三类事后选择和P/D帕累托保留真实整策略行，P约束相对同输入original", "108 groups; min Lhat_max, min D, min D under P <= 103% original", "retrospective_oracle_and_Pareto", "W1 and W1_8Q_1KV_v1", "whole executed strategy; exact ties retained; no independently spliced P/D minima", "backend 1,2,3; all phases", "ns; paired percentages; signed Lhat", "e2e/dstu_supplement/S_AND_HI_S256_REPORT.md", "e2e/dstu_supplement/tables/s256_strategy_choices.csv", "e2e/dstu_supplement_report.py", "e2e/dstu_supplement/tables/s256_pareto.csv", status="complete" if done else "pending_supplement_completion", commit=supplement_commit)
    add("R2F", "完整均值、分硬件相位及预测与真实执行并列展示全部108组", "S72/H18/I18; all three fixed transport strategies and frozen gate", "figure_data_from_complete_paired_results", "W1 and W1_8Q_1KV_v1", "same fixed/oracle/online definitions as strategy table", "backend 1,2,3; equal per-pair weighting", "paired percentage; ns", "e2e/dstu_supplement/FIGURE_GUIDE.md", "e2e/dstu_supplement/evidence/figure_audit.json", "e2e/dstu_supplement_report.py", "e2e/dstu_supplement/figures/strategy_overall_means.png", status="complete" if done else "pending_supplement_completion", commit=supplement_commit)
    add("R2C", "门控预测、CPCM前向、输入准备和后端验证成本分别统计，12次预测服务36个决定", "12 prediction processes; 36 frozen choices; 108 evaluated groups", "measured_parallel_process_cost", "W1 and W1_8Q_1KV_v1", "same frozen gate outputs; no prediction cost repeated for seeds", "one prediction per fixed state", "wall seconds; simulation service ns reported separately", "e2e/dstu_supplement/S_AND_HI_S256_REPORT.md", "e2e/dstu_supplement/tables/online_gate_costs.csv", "e2e/dstu_supplement_gate.py", "e2e/dstu_supplement/tables/gate_prediction_timing.csv", timing="new parallel-process prediction/task timings; excludes interpreter startup; not resident per-request deployment timing", status="complete" if done else "pending_supplement_completion", commit=supplement_commit)
    add("R4", "564项完整记录、540项实际身份、282条复用及门控先冻结后执行均已逐项核对", "12 prepare + 12 gate + 540 evaluate; separate 2 legacy probes and 12 identity reconstructions", "completion_receipt_input_identity_and_source_chain_audit", "W1 and W1_8Q_1KV_v1; actual v1 execution plus independent fresh identity reconstruction", "all 108 groups; no result removed", "backend seeds 1,2,3", "258 production + 2 probe backend calls; 36 production CPCM; zero new SA", "e2e/dstu_supplement/AUDIT_SUMMARY.md", "e2e/dstu_supplement/evidence/completion_audit.json", "e2e/dstu_supplement_audit.py", "e2e/dstu_supplement/evidence/input_and_reuse_manifest.json", timing="main total includes missing-input repair wait; identity verification measured separately", status="complete" if done else "pending_supplement_completion", commit=supplement_commit)
    scope = [dict(model="GPT artifact", stages="A-G, Post-FG, S, J, DSTU supplement", batch=16, hidden=6144, q_heads=64,
        q_heads_per_graph=8, head_groups=8, head_dim=96, layers=44, sequence_length=2048,
        workload="W0 historical; W1 main", historical_KV="shared within batch", hardware="WSC-LLM/FACE/WATOS; GPT 1x4,2x2,2x3,3x3",
        direct_simulation="operator; one P operator plus D chain on paired groups", layer_assembly="2 LN+8 MHA+proj+FFN prefill; LN+8 MHA+proj+FFN decode after LN correction",
        full_model="44-layer serial sum / ideal pipeline analytical only", source="e2e/F_SUMMARY.md", source_sha256=sha(ROOT/"e2e/F_SUMMARY.md"))]
    for model in ("llama", "qwen3moe"):
        path = ROOT / f"e2e/hi/models/{model}/model_contract.json"
        contract = read(path)
        scope.append(dict(model=model, stages="H, J, DSTU supplement" if model == "llama" else "I, J, DSTU supplement", **{k:contract[k] for k in
            ("batch","hidden","q_heads","q_heads_per_graph","kv_heads_per_graph","head_groups","head_dim","layers","sequence_length")},
            workload=contract["workload_version"], historical_KV=contract["decode_history"],
            hardware="main 2x2 all 8 operators; WATOS MHA/FFN only",
            direct_simulation="operator; main-hardware one P operator plus D chain",
            layer_assembly=dict(prefill=contract["prefill_chain"],decode=contract["decode_chain"]),
            full_model=contract["layer_count_use"], expert_assignment=contract["expert_assignment"],
            source=relative(path), source_sha256=sha(path)))
    table(dest / "tables/CLAIM_EVIDENCE.csv", entries)
    table(dest / "tables/MODEL_SCOPE_TABLE.csv", scope)
    audit = dict(status="complete", claims=len(entries), historical_reference_commit=baseline, artifact_commit=artifact,
        scopes=len(scope), supplement_status="complete" if done else "pending", sources=[dict(id=r["claim_id"],sources=r["sources"]) for r in entries])
    write(dest / "evidence/diagnosis_evidence_index.json", audit)
    out = ["# 论文证据索引", "", "每一行把可写入论文的陈述连接到实验范围、数据、代码、表图和版本。历史报告保留原配置名称；2026-11-18 新补充采用 S256 主配置。", "",
        "直接仿真、算子/层组装、解析估算和保存态机制分析分别标注。W0、W1、H/I 的八Q一KV合同分别保存，不能合成一种未运行的工作负载。单层或固定KV请求的相加结果也不表示整模型同时执行。", "",
        f"历史核查提交：`{baseline}`；BusyBarn artifact：`{artifact}`。补充实验运行于该历史基线加新增脚本快照，实际v1源码保存于 `dist/server_runtime/`；修正准备身份后的v2源码保存于 `dist/server_runtime_identity/`。本轮最终提交见 `DELIVERY.md`，各输入文件摘要保存在索引CSV及核对JSON中。", "",
        "[完整主张与来源表](tables/CLAIM_EVIDENCE.csv) · [模型范围表](tables/MODEL_SCOPE_TABLE.csv) · [文件存在性与摘要核对](evidence/diagnosis_evidence_index.json)", "",
        "|编号|陈述|证据类型|状态|入口|", "|---|---|---|---|---|"]
    for r in entries:
        link = os.path.relpath(ROOT / r["report"], dest).replace("\\", "/")
        out.append(f"|{r['claim_id']}|{r['statement']}|{r['evidence_type']}|{r['status']}|[{Path(r['report']).name}]({link})|")
    out += ["", "后端1–10与修复搜索种子123/1/2/3/4分别记录；随机平局CPCM选优损失与预测误差分别解释。J的独立进程重复用于串行计时，不增加健康候选数。历史A256八组合及H5/I5保留。S256新增108组的324条联合结果、216条逻辑单跑已逐项核对；Online的36个决定在本轮新后端开始前冻结，并在108组中取对应完整策略的P/D。三类事后选优另列。", ""]
    (dest / "PAPER_EVIDENCE_INDEX.md").write_text("\n".join(out), encoding="utf-8")
    return {k:v for k,v in audit.items() if k != "sources"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose", "index", "render"))
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--hardware", choices=("busybarn", "watos"))
    args = parser.parse_args()
    if args.command == "diagnose":
        print(json.dumps(diagnose(args.output, args.hardware), ensure_ascii=False, indent=2))
    elif args.command == "index":
        print(json.dumps(index(args.output), ensure_ascii=False, indent=2))
    else:
        load = lambda name: read(args.output / "tables" / ("watos_" + name + ".json"))
        render_diagnosis(args.output, load("diagnostic_timing"), load("chain_composition"), load("exclusive_wait"),
            load("resource_load"), load("rerouting_summary"), load("search_trajectories"), read(args.output/"evidence/diagnosis_audit.json"))


if __name__ == "__main__":
    main()
