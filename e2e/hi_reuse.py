"""Read-only I0 identity audit of the 28 saved Qwen decode FFN candidates.

No search or backend simulation is run.  In particular, a reconstructed event
order is not substituted for an unrecorded historical execution order.
"""
import argparse
import ast
import collections
import copy
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "e2e"
OLD = ROOT / "cpcm_sa/out/s2"
DEST = HERE / "hi/evidence"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, set):
        return sorted((plain(v) for v in value), key=repr)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def variants():
    for arm in ("A1", "A3h"):
        for g in (0, 1):
            for x in (range(7) if arm == "A1" else (123, 1, 2, 3, 4, 5, 6)):
                yield arm, g, x, f"moe_{arm}_g{g}_{'l' if arm == 'A1' else 's'}{x}"


def order_rows(events):
    """Actual iteration, deliberately without sorting sets or event entries."""
    return [dict(tag=t, dependencies=list(e.dependency_set), successors=list(e.issue_set),
                 targets=list(e.target_location) if e.event_type == "communication"
                 and isinstance(e.target_location, set) else None)
            for t, e in events.items()]


def compare_order(first, second):
    return dict(event_dictionary=list(first) == list(second),
                dependency_sets=sum(list(e.dependency_set) != list(second[t].dependency_set) for t, e in first.items()),
                issue_sets=sum(list(e.issue_set) != list(second[t].issue_set) for t, e in first.items()),
                multicast_target_sets=sum(list(e.target_location) != list(second[t].target_location)
                    for t, e in first.items() if e.event_type == "communication" and isinstance(e.target_location, set)))


def semantic_state(behaviours, data):
    return dict(behaviours=[dict(tag=t, name=b.beha_name, type=b.beha_type, shape=b.beha_datashape,
                    inputs=b.needed_data_split_dict, outputs=b.produced_data_split_dict,
                    producers=b.producer_tags, consumers=b.consumer_tags) for t, b in behaviours.items()],
                tensors=[dict(tag=t, name=d.data_name, shape=d.data_shape, type=d.data_type, bytes=d.data_size)
                         for t, d in data.items()])


def expert_structure(behaviours):
    rms = [b for b in behaviours.values() if b.beha_name.startswith("LN_rmsnorm")]
    norm_output = set(rms[-1].produced_data_split_dict)
    dispatch = [b for b in behaviours.values() if b.beha_type == "dispatch"]
    dispatch_outputs = sorted({t for b in dispatch for t in b.produced_data_split_dict})
    branches = {}
    for name in ("MLPmlp1", "MLPmlp2", "MLPmlp3"):
        operators = collections.defaultdict(set)
        counts = 0
        for b in behaviours.values():
            if b.beha_name.startswith(name + "_"):
                operators[b.beha_tag[:-1]].update(t for t in b.needed_data_split_dict if t[0] == 0)
                counts += 1
        branches[name] = dict(logical_operators=len(operators), behaviours=counts,
                              ordered_inputs=[sorted(ts) for _, ts in sorted(operators.items())])
    first, second = branches["MLPmlp1"]["ordered_inputs"], branches["MLPmlp2"]["ordered_inputs"]
    expected = [[dispatch_outputs[e % 16]] for e in range(128)]
    return dict(branches=branches, dispatch_behaviours=len(dispatch), dispatch_outputs=dispatch_outputs,
                rmsnorm_to_dispatch=all(set(b.needed_data_split_dict) == norm_output for b in dispatch),
                two_gates_share_expert_input=first == second,
                fixed_128_experts_mod16=first == expected,
                logical_experts_per_batch_item=dict(collections.Counter(str(ts[0]) for ts in first)))


def config_values(text):
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read_string(text)
    values = {}
    for section in cfg.sections():
        if section == "fg_metadata":
            continue
        row = {}
        for key, value in cfg.items(section):
            if (section, key) == ("tensorcore_cfg", "fg_compute_scale"):
                assert float(value) == 1
                continue
            if (section, key) == ("failures", "failed_pes"):
                assert ast.literal_eval(value) == []
                continue
            try:
                row[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                row[key] = value
        values[section] = row
    return values


def broadcast(state, ctx, c):
    from add_communication import add_broadcast
    b, d, e, h = state
    random.seed(c.BCAST_SEED)
    c.np.random.seed(c.BCAST_SEED)
    add_broadcast(data_tags=ctx["broadcast_datatags"],
                  ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]],
                  beha_dict=b, data_dict=d, hardware_platform=h, event_dict=e,
                  dijkstra_routing=ctx["dijkstra_routing"], alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])


def audit_one(arm, g, x, name):
    import hi_config
    import e_run
    started = time.perf_counter()
    spec = dict(model="qwen3moe", phase="decode", op="ffn", method=arm, g=g, x=x,
                hardware_profile="busybarn", shape="2x2", search_steps=1000)
    _, c, new_cfg = hi_config.prepare(spec, HERE / "hi/out/reuse_audit")
    c.OUT = str(OLD)
    old_run_path, old_snap_path = OLD / "runs" / (name + ".pkl.gz"), OLD / "snaps" / (name + ".pkl.gz")
    old_tie_path = HERE / "out/b0" / (name + ".json")
    with gzip.open(old_run_path, "rb") as handle:
        run = pickle.load(handle)
    ties = json.loads(old_tie_path.read_text(encoding="utf-8"))
    new_work, new_suffix = c.WORK, c.SUF
    c.WORK = str(ROOT / "cpcm_sa/work/artifact")
    c.SUF = "_bw256_co4x4_bw256_t128x64_failpattern0"
    old_state = c.load_snap("moe", arm, g, x)
    c.WORK, c.SUF = new_work, new_suffix
    new_state = e_run.load_snapshot(old_snap_path, c, new_cfg)
    semantics = semantic_state(*old_state[:2])
    old_script = ROOT / "cpcm_sa/work/artifact/models/intra_mapping_qwen3moe_decode/py" / (run["meta"]["task"] + ".py")
    old_cfg_path = old_script.parent.parent / "cfg/config_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0.cfg"
    new_script = hi_config.generate(spec)
    normalized_old = old_script.read_text(encoding="utf-8").replace(old_cfg_path.stem, "CONFIG").replace(old_script.stem, "OUTPUT")
    normalized_new = new_script.read_text(encoding="utf-8").replace(hi_config.config_path(spec).stem, "CONFIG").replace(new_script.stem, "OUTPUT")
    old_ast = ast.dump(ast.parse(normalized_old), include_attributes=False)
    new_ast = ast.dump(ast.parse(normalized_new), include_attributes=False)
    contract = hi_config.model_contract("qwen3moe")
    data = old_state[1]
    input_tensor = data[(0, 0)]
    rms = [b for b in old_state[0].values() if b.beha_name.startswith("LN_rmsnorm")]
    experts = expert_structure(old_state[0])
    static_checks = dict(
        generated_program_ast=old_ast == new_ast,
        hardware_values=config_values(old_cfg_path.read_text(encoding="utf-8")) == config_values(hi_config.config_path(spec).read_text(encoding="utf-8")),
        batch_hidden_dtype=input_tensor.data_shape == [contract["batch"], 1, contract["hidden"]] and input_tensor.data_type == contract["dtype"],
        one_embedded_rmsnorm=len(rms) == 7 and len({b.beha_tag[:-1] for b in rms}) == 1,
        rmsnorm_expert_inputs=experts["rmsnorm_to_dispatch"] and experts["two_gates_share_expert_input"],
        fixed_expert_assignment=experts["fixed_128_experts_mod16"],
        restored_operator_regions=semantic_state(*old_state[:2]) == semantic_state(*new_state[:2]),
        full_1000_steps=run["meta"]["steps"] == 1000 and len(run["rows"]) == 1000 and run["rows"][-1]["step"] == 1000,
        backend_seeds=list(run["meta"]["backend_seeds"]) == list(range(1, 11)) and sorted(run["selected"]["backend"]) == list(range(1, 11)),
        backend_triples=all(len(v) == 3 for v in run["selected"]["backend"].values()),
        ten_random_ties=len(ties["ties"]) == 10 and len(ties["t_tie_each"]) == 10,
        tie_deterministic_matches_saved=ties["det"] == ties["T_bc"] == run["selected"]["T_bc"],
        routing_context=run["meta"]["ctx"]["dijkstra_routing"] is True and [run["meta"]["ctx"][k] for k in ("alpha", "beta", "gamma")] == [100, 1, 100],
        broadcast_seed=run["meta"]["bcast_seed"] == 0,
    )
    ctx = run["meta"]["ctx"]
    broadcast(old_state, ctx, c)
    broadcast(new_state, ctx, c)
    before_backend = compare_order(old_state[2], new_state[2])
    # The historical protocol deep-copies a base once, then copies that base for
    # each seed. These are the graphs presented to the actual backend call.
    old_base = copy.deepcopy(old_state[2:])
    new_base = copy.deepcopy(new_state[2:])
    old_backend, old_hp = copy.deepcopy(old_base)
    new_backend, new_hp = copy.deepcopy(new_base)
    current_backend_order = compare_order(old_backend, new_backend)
    single_copy_change = compare_order(old_state[2], old_base[0])
    static_checks["saved_final_event_dictionary"] = list(old_state[2]) == run["selected"]["e3map"]["layout"]["order"]
    static_checks["saved_final_routes"] = c.e3.routes_of(old_state[2], c.e3._e2()._snapshot_pre(old_state[2])) == run["selected"]["e3map"]["routes"]
    static_checks["current_backend_iteration"] = current_backend_order == dict(event_dictionary=True, dependency_sets=0, issue_sets=0, multicast_target_sets=0)
    # Old archives capture event insertion order, but do not capture the set
    # iteration order that the historical backend actually saw after deepcopy.
    historical_iteration_recorded = False
    decision = "rerun"
    first_pilot = arm == "A3h" and g == 0 and x == 123
    return dict(name=name, model="qwen3moe", phase="decode", op="ffn", method=arm, g=g, x=x,
                decision=decision, new_batch="I1_main_decode" if first_pilot else "I2D",
                reuse_eligible=False, reuse_reason="historical_set_iteration_not_recorded",
                verified_fields=static_checks, all_reconstructable_fields_match=all(static_checks.values()),
                historical_backend_iteration_status="unknown_not_certifiable",
                historical_iteration_recorded=historical_iteration_recorded,
                current_full_graph_order=before_backend, current_backend_order=current_backend_order,
                one_deepcopy_order_change=single_copy_change,
                current_old_backend_order_sha256=digest(order_rows(old_backend)),
                current_hi_backend_order_sha256=digest(order_rows(new_backend)),
                semantic_sha256=digest(semantics), old_normalized_program_sha256=hashlib.sha256(old_ast.encode()).hexdigest(),
                new_normalized_program_sha256=hashlib.sha256(new_ast.encode()).hexdigest(),
                source=dict(run=str(old_run_path.relative_to(ROOT)), run_sha256=sha(old_run_path),
                            snapshot=str(old_snap_path.relative_to(ROOT)), snapshot_sha256=sha(old_snap_path),
                            ties=str(old_tie_path.relative_to(ROOT)), ties_sha256=sha(old_tie_path),
                            generated_old=str(old_script.relative_to(ROOT)), generated_old_sha256=sha(old_script),
                            generated_new=str(new_script.relative_to(ROOT)), generated_new_sha256=sha(new_script)),
                graph=dict(behaviours=len(old_state[0]), tensors=len(data), full_events=len(old_state[2]),
                           b0_events=ties["n_events"], rmsnorm_behaviours=len(rms), experts=experts),
                old_measurements=dict(sa_seconds=run["sa_wall"], calibration_seconds=run["meta"]["calibration"]["wall"],
                                      backend_seconds=sum(run["selected"]["backend_wall"]),
                                      backend_mean=sum(v[0] for v in run["selected"]["backend"].values()) / 10,
                                      T_bc=run["selected"]["T_bc"], random_tie_mean=sum(ties["ties"]) / 10),
                audit_seconds=time.perf_counter() - started)


def report(rows):
    verified = collections.Counter(k for r in rows for k, value in r["verified_fields"].items() if value)
    failures = {r["name"]: [k for k, v in r["verified_fields"].items() if not v]
                for r in rows if not r["all_reconstructable_fields_match"]}
    summary = dict(candidates=len(rows), reusable=sum(r["reuse_eligible"] for r in rows),
                   rerun=len(rows), I1_main_decode=1, I2D=27,
                   verification_pass_counts=dict(verified), reconstruction_differences=failures,
                   historical_backend_set_iteration="not recorded in any of the 28 archives",
                   new_backend_simulations=0, new_searches=0,
                   event_order_warning="Current reconstruction equality does not supply historical set iteration records.",
                   audit_seconds=sum(r["audit_seconds"] for r in rows))
    write_json(DEST / "i0_reuse_audit.json", dict(summary=summary, candidates=rows))
    columns = ["name", "method", "g", "x", "decision", "new_batch", "all_reconstructable_fields_match",
               "historical_backend_iteration_status", "dependency_order_changes_on_copy", "issue_order_changes_on_copy"]
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for r in rows:
        row = {k: r[k] for k in columns if k in r}
        row.update(dependency_order_changes_on_copy=r["one_deepcopy_order_change"]["dependency_sets"],
                   issue_order_changes_on_copy=r["one_deepcopy_order_change"]["issue_sets"])
        writer.writerow(row)
    (DEST / "i0_reuse_candidates.csv").write_text(buf.getvalue(), encoding="utf-8", newline="")
    lines = ["# I0：Qwen decode FFN 旧 28 候选核对\n",
             "28 份快照、1000 步搜索记录、十种子后端和十次随机平局 CPCM 记录齐全。本批复用 0 个，28 个按新入口重新搜索：A3h g0/seed123 计入 I1 主硬件 decode 试点，其余 27 个计入 I2D。旧结果原样保留。\n",
             "原因是旧记录只保存了最终 `event_dict` 插入顺序，没有保存后端实际读取的 `dependency_set`、`issue_set` 和多播目标集合遍历序列。重新加载和复制会改变部分集合的遍历次序；当前机器上重构相等也不能补出当时未保存的序列。因此，按本轮要求的逐字段完整身份核对，历史后端的该项记为 `unknown_not_certifiable`。这不是把未知项说成模型或 RMSNorm 有错。\n",
             "## 已核对的内容\n",
             "| 项目 | 通过数 / 28 |\n|---|---:|\n"]
    labels = dict(generated_program_ast="生成程序 AST（仅替换配置及输出文件名）", hardware_values="实际硬件配置值",
                  batch_hidden_dtype="batch=16、hidden=2048、bf16", one_embedded_rmsnorm="一次内嵌 RMSNorm（7 个行为）",
                  rmsnorm_expert_inputs="RMSNorm → Dispatch → 专家双支路", fixed_expert_assignment="128 专家固定 mod16 路由及每样本 8 专家",
                  restored_operator_regions="算子输入/输出区域与消费者", full_1000_steps="完整 1000 步",
                  backend_seeds="后端种子 1–10", backend_triples="后端三元组", ten_random_ties="10 次随机平局 CPCM",
                  tie_deterministic_matches_saved="B0 确定性估值与搜索记录", routing_context="Dijkstra 与 100/1/100 权重",
                  broadcast_seed="最终广播种子 0", saved_final_event_dictionary="保存的最终 event_dict 顺序",
                  saved_final_routes="保存的最终源与路由", current_backend_iteration="当前旧/新加载流程送入后端前的实际遍历")
    lines.extend(f"| {label} | {verified[key]} |\n" for key, label in labels.items())
    lines += ["\nQwen decode FFN 的原模板已经是 RMSNorm → Dispatch → 两条门控支路；残差读取原输入。新生成副本保留这一结构。128 个专家采用 `expert_idx % 16`，每个 batch 项对应 8 个专家。MHA 的头数改动不修改这张 FFN 图。\n",
              "\n新旧评估均是同进程内每个种子重新复制事件及平台并重设随机数，十种子分别执行；随机平局 CPCM 使用 `random.Random(1..10)`。不是十个独立子进程。这里没有新增后端仿真，也没有运行搜索。旧耗时仅随审计记录存档，不用于当前服务器的 J 计时或新批预计用时。\n",
              "\n## 逐候选记录\n",
              "| 候选 | 可重构字段 | 一次复制改变的依赖/后继集合 | 后续批次 |\n|---|---|---:|---|\n"]
    for r in rows:
        d = r["one_deepcopy_order_change"]
        lines.append(f"| {r['name']} | {'一致' if r['all_reconstructable_fields_match'] else '见差异表'} | {d['dependency_sets']}/{d['issue_sets']} | {r['new_batch']} |\n")
    lines += ["\nI 共需 448 个新健康候选：I1 为 8、I2D 为 166、I2P 为 166、I3 为 108；H+I 为 896 个新候选。本次只完成 I0，不启动 I1/I2D。\n",
              "\n证据：[`i0_reuse_audit.json`](evidence/i0_reuse_audit.json)、[`i0_reuse_candidates.csv`](evidence/i0_reuse_candidates.csv)。每份历史输入的 SHA-256、生成程序 AST 摘要、模型区域摘要、当前实际遍历摘要和逐字段判断均保留在 JSON 中。\n",
              "\n复算：`e2e/post_fg/dist/venv/Scripts/python.exe e2e/hi_reuse.py audit`。\n"]
    if failures:
        lines += ["\n## 当前重构差异\n\n", "```json\n", json.dumps(failures, ensure_ascii=False, indent=2), "\n```\n"]
    (HERE / "hi/I0_REUSE.md").write_text("".join(lines), encoding="utf-8", newline="\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "one", "refresh_experts"))
    parser.add_argument("--name")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.command == "refresh_experts":
        import hi_config
        _, c, _ = hi_config.prepare(dict(model="qwen3moe", phase="decode", op="ffn", method="A3h", g=0, x=123), HERE / "hi/out/reuse_audit")
        rows = []
        for _, _, _, name in variants():
            path = DEST / "i0_candidates" / (name + ".json")
            row = json.loads(path.read_text(encoding="utf-8"))
            with gzip.open(ROOT / row["source"]["snapshot"], "rb") as handle:
                blob, _ = pickle.load(handle)
            behaviours, _, _ = pickle.loads(blob)
            experts = expert_structure(behaviours)
            row["graph"].pop("tensor_branches", None)
            row["graph"]["experts"] = experts
            row["verified_fields"].update(rmsnorm_expert_inputs=experts["rmsnorm_to_dispatch"] and experts["two_gates_share_expert_input"],
                                          fixed_expert_assignment=experts["fixed_128_experts_mod16"])
            row["all_reconstructable_fields_match"] = all(row["verified_fields"].values())
            write_json(path, row)
            rows.append(row)
        print(json.dumps(report(rows)), flush=True)
        return
    rows = []
    for arm, g, x, name in variants():
        if args.name and name != args.name:
            continue
        row = audit_one(arm, g, x, name)
        write_json(DEST / "i0_candidates" / (name + ".json"), row)
        rows.append(row)
        print(json.dumps(dict(name=name, all_reconstructable_fields_match=row["all_reconstructable_fields_match"],
                              decision=row["decision"], seconds=row["audit_seconds"])), flush=True)
    if args.command == "audit":
        assert len(rows) == 28
        print(json.dumps(report(rows)), flush=True)


if __name__ == "__main__":
    main()
