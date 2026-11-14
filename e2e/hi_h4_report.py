"""Read H4 PF4 outputs; report actual repairs separately from stopped-task receipts."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import statistics

import f_config
import fg_batch
import hi_config
import hi_next_report as common
from hi_split_supplement import read, write

ROOT, HERE = common.ROOT, common.HERE
SEEDS, BUDGETS = (123, 1, 2, 3, 4), (30, 100, 300)
KINDS = dict(fault_selection=2, fault_select=2, fault_pre=16, fault_l1=16, fault_stop=48, fault_rep=180)
PF4 = [[[0, 11], [1, 8]], [[1, 8], [0, 11]]]


def path(value):
    p = Path(value.replace("\\", "/"))
    return p if p.is_absolute() else ROOT / p


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(spec):
    return spec["config_id"], spec["phase_op"], spec["start_arm"]


def fields(spec):
    return {k: spec[k] for k in ("model", "hardware_profile", "config_id", "phase_op", "start_arm")}


def state_rows(spec, label, record, result, samples):
    backend = record["backend"]
    assert set(backend) == set(range(1, 11)) and len(record["backend_wall"]) == 10
    assert not record["fault_violations"]
    values = [backend[seed][0] for seed in range(1, 11)]
    row = dict(job_id=spec["job_id"], **fields(spec), state=label, repair_seed=spec.get("seed"),
        budget=spec.get("steps", 0), T_hat_ns=record["T"], **common.stats(values, "backend"),
        deterministic_relative_error=record["T"]/statistics.mean(values)-1,
        route_count=len(record["routes"]), sum_comm_bytes_hops=record["sum_comm_bytes_hops"],
        distance=record["dist"], fault_violations=0,
        backend_call_seconds=sum(record["backend_wall"]), backend_total_seconds=record["t_backend"],
        backend_copy_and_prepare_seconds=record["t_backend"]-sum(record["backend_wall"]),
        final_cpcm_seconds=record["t_finalize"], source_candidate=result["source_candidate"])
    if "migration_vs_start" in record:
        row.update({"migration_"+k:v for k,v in record["migration_vs_start"].items()})
        row["migration_vs_R0"] = record.get("migration_vs_R0")
    for seed in range(1, 11):
        triple = backend[seed]
        samples.append(dict(job_id=spec["job_id"], **fields(spec), state=label,
            repair_seed=spec.get("seed"), budget=spec.get("steps", 0), backend_seed=seed,
            backend_ns=triple[0], unhidden_compute_ns=triple[1], unhidden_communication_ns=triple[2],
            backend_call_seconds=record["backend_wall"][seed-1]))
    return row


def parse_engine(spec, result, samples, load_pickle):
    source = path(result["sink"])
    raw = load_pickle(source)
    assert raw["backend_seeds"] == list(range(1, 11))
    kind = spec["kind"]
    if kind == "fault_pre":
        assert raw["faults"] == {"failed_nodes":[],"failed_links":[]}
    else:
        assert not raw["faults"]["failed_nodes"] and not raw["faults"]["failed_pes"]
        assert raw["faults"]["failed_links"] == PF4
    if kind == "fault_pre":
        entries = [("healthy", raw["record"])]
        assert raw["T"] == result["T"] == raw["record"]["T"]
        assert result["backend_mean"] == statistics.mean(v[0] for v in raw["record"]["backend"].values())
    elif kind == "fault_l1":
        entries = [("R0", raw["R0"])]
        assert result["T_R0"] == raw["T_R0"] == raw["R0"]["T"]
        assert result["R0_backend_mean"] == statistics.mean(v[0] for v in raw["R0"]["backend"].values())
        assert result["migration_R0"] == raw["R0"]["migration_vs_start"]
        assert result["L1_evaluated"] == (spec["start_arm"] == "A3h")
        if spec["start_arm"] == "A3h":
            entries.append(("L1",raw["L1"]))
            assert result["T_L1"] == raw["T_L1"] == raw["L1"]["T"]
            assert result["L1_backend_mean"] == statistics.mean(v[0] for v in raw["L1"]["backend"].values())
            assert result["migration"] == raw["L1"]["migration_vs_start"]
    else:
        assert kind == "fault_rep" and spec["steps"] in BUDGETS and spec["seed"] in SEEDS
        entries = [("L2",raw["record"])]
        assert result["T"] == raw["record"]["T"] and result["timing"] == raw["timing"]
        assert result["backend_mean"] == statistics.mean(v[0] for v in raw["record"]["backend"].values())
        assert result["migration"] == raw["record"]["migration_vs_start"]
        search = raw["search"]
        assert [r["step"] for r in search["rows"]] == list(range(1,spec["steps"]+1))
        assert raw["spec"]["method"] == "L2Gsync" and raw["spec"]["checks"] is False
        assert search["check_revert"]["n"] == 0
        assert raw["proposals"]["steps"] == spec["steps"]
        assert raw["T_L1_check"] == search["T_start"]
        assert search["best_T"] == result["T"]
    states = [state_rows(spec,label,record,result,samples) for label,record in entries]
    if kind == "fault_l1" and spec["start_arm"] == "A3h":
        states[0]["changed_routes_from_healthy"] = raw["n_changed_routes_R0"]
        states[1]["changed_routes_from_R0"] = raw["n_changed_routes_L1"]
        states[1]["failed_links_healthy_context"] = raw["link_stats"]
    timing = dict(job_id=spec["job_id"], **fields(spec), kind=kind, budget=spec.get("steps",0),
        repair_seed=spec.get("seed"), full_job_seconds=result["wall_seconds"],
        backend_calls=len(entries)*10, backend_call_seconds=sum(r["backend_call_seconds"] for r in states),
        backend_total_seconds=sum(r["backend_total_seconds"] for r in states),
        final_cpcm_seconds=sum(r["final_cpcm_seconds"] for r in states),
        calibration_seconds=0., search_seconds=0., restore_best_seconds=0.,
        R0_prepare_seconds=raw.get("t_R0",0.), L1_search_seconds=raw.get("t_L1",0.))
    if kind == "fault_rep":
        timing.update(calibration_seconds=raw["search"]["t_calibration"],search_seconds=raw["search"]["t_L2_search"],
            restore_best_seconds=raw["search"]["t_restore_best"],
            inherited_R0_prepare_seconds=raw["timing"]["t_R0"],inherited_L1_search_seconds=raw["timing"]["t_L1"],
            historical_decision_seconds=raw["timing"]["t_decision"],
            calibration_proposals=raw["proposals"]["calibration"],search_proposals=raw["proposals"]["steps"],
            accepted_proposals=sum(r["accepted"] for r in raw["search"]["rows"]),
            best_step=raw["search"]["best_step"],T_L1_reloaded_ns=raw["T_L1_check"],cpcm_calls=raw["cpcm_calls"])
    measured = ("backend_total_seconds","final_cpcm_seconds","calibration_seconds","search_seconds",
                "restore_best_seconds","R0_prepare_seconds","L1_search_seconds")
    timing["other_seconds"] = result["wall_seconds"]-sum(timing[k] for k in measured)
    assert timing["other_seconds"] >= -1e-6
    exported = {label:sha(path(filename)) for label,filename in result["exports"].items()}
    return states,timing,dict(raw_sha256=sha(source),export_sha256=exported)


def collect(out, manifest, progress, smoke, model="llama", kinds=KINDS):
    import post_fg_split
    post_fg_split.setup()
    jobs = {j["id"]:j for j in manifest["jobs"]}
    assert len(jobs) == sum(kinds.values()) and Counter(j["kind"] for j in jobs.values()) == kinds
    present, missing, samples, states, timing, checks = {}, [], [], [], [], []
    smoke_receipts = {r["id"]:r for r in read(out/"pilot_execution.json")} if smoke else {}
    for name,job in jobs.items():
        result_path,receipt_path = out/job["result"],out/"receipts"/(name+".json")
        if not result_path.exists() or (not smoke and not receipt_path.exists()):
            missing.append(name)
            continue
        result,spec = read(result_path),job["spec"]
        assert result["job_id"] == name and result["kind"] == spec["kind"]
        assert spec["model"] == model and spec.get("fault_id") in (None,"PF4")
        assert spec["model_contract_sha256"] == hi_config.digest(hi_config.model_contract(model))
        assert spec["backend_seeds"] == list(range(1,11))
        if smoke:
            receipt = smoke_receipts[name]
            assert receipt["rc"] == 0
        else:
            receipt = read(receipt_path)
            assert receipt["rc"] == 0 and receipt["identity"] == fg_batch.identity(job)
        check = dict(job_id=name,result_sha256=sha(result_path),status=result["status"],
                     receipt_source="local_pilot_execution" if smoke else "identity_checked_receipt")
        if result["status"] == "reused":
            assert spec["kind"] == "fault_rep" and result["previous_stop"] == spec["previous_stop"]
            stop = read(out/"results"/(spec["previous_stop"]+".json"))
            assert stop["stop"] and result["export"] == stop["export"]
            assert not (out/"raw"/(name+".pkl.gz")).exists()
        else:
            assert result["status"] == "ok"
            saved_spec = result["spec"]
            if smoke:
                assert {k:v for k,v in saved_spec.items() if k not in ("pre_sink","l1_sink")} == {k:v for k,v in spec.items() if k not in ("pre_sink","l1_sink")}
            else:
                assert saved_spec == spec
            if spec["kind"] in ("fault_pre","fault_l1","fault_rep"):
                sub,tm,evidence = parse_engine(saved_spec,result,samples,post_fg_split.static.read_pickle)
                states.extend(sub);timing.append(tm);check.update(evidence)
            elif spec["kind"] == "fault_select":
                assert set(result["faults"]) == {"PF4"} and result["anchor_backend_calls"] == 0
                fault = result["faults"]["PF4"]
                assert fault["failed_links"] == PF4 and not fault["failed_nodes"] and not fault["failed_pes"]
        present[name] = dict(result=result,spec=spec,receipt=receipt)
        checks.append(check)
    if not progress and not smoke:
        assert not missing
        assert {p.stem for p in (out/"results").glob("*.json")} == set(jobs)
        assert {p.stem for p in (out/"receipts").glob("*.json")} == set(jobs)
    return jobs,present,missing,states,samples,timing,checks


def stop_tables(jobs,present):
    stops,groups,trajectories = [],[],[]
    for name,job in jobs.items():
        spec = job["spec"]
        if spec["kind"] == "fault_rep":
            item = present.get(name)
            row = dict(job_id=name,**fields(spec),budget=spec["steps"],repair_seed=spec["seed"],
                previous_stop=spec["previous_stop"],status=item["result"]["status"] if item else "pending")
            if item:
                result = item["result"]
                row.update(wall_seconds=result["wall_seconds"],backend_calls=10 if result["status"] == "ok" else 0)
                if result["status"] == "ok":
                    row.update(T_hat_ns=result["T"],backend_mean_ns=result["backend_mean"],migration=result["migration"])
                else:
                    row["reused_export"] = result["export"]
            trajectories.append(row)
        if spec["kind"] != "fault_stop" or name not in present:
            continue
        result = present[name]["result"]
        propagated = False
        if spec["steps"] == 0:
            base = present[spec["l1_job"]]["result"]
            level = "R0" if base["T_R0"] <= base["T_pre"]*1.02 else "L1"
            T = base["T_"+level]
            stop = T <= base["T_pre"]*1.02
            assert result["level"] == level and result["T"] == T and result["stop"] == stop
            assert result["source_job"] == spec["l1_job"] and result["export"] == base["exports"][level]
            assert result["reason"] == ("within_epsilon" if stop else "continue")
        else:
            previous = present[spec["previous_stop"]]["result"]
            members = [present[r]["result"] for r in spec["rep_jobs"]]
            if previous["stop"]:
                propagated = True
                assert all(r["status"] == "reused" for r in members)
                for field in ("stop","level","T","T_pre","export","source_job","reason"):
                    assert result[field] == previous[field]
            else:
                assert all(r["status"] == "ok" for r in members)
                assert {r["seed"] for r in members} == set(SEEDS)
                T = statistics.mean(r["T"] for r in members)
                gain = (previous["T"]-T)/previous["T"] if spec["steps"] != 30 else None
                epsilon,plateau = T <= previous["T_pre"]*1.02,gain is not None and gain < .02
                median = statistics.median(r["backend_mean"] for r in members)
                representative = min((r for r in members if r["backend_mean"] == median),key=lambda r:r["seed"])
                assert result["T"] == T and result["gain_from_previous"] == gain
                assert result["stop"] == (epsilon or plateau or spec["steps"] == 300)
                assert result["reason"] == ("within_epsilon" if epsilon else "tier_gain" if plateau else "budget" if spec["steps"] == 300 else "continue")
                assert result["level"] == "s"+str(spec["steps"])
                assert result["source_job"] == representative["job_id"] and result["seed"] == representative["seed"]
                assert result["export"] == representative["exports"]["final"]
        stops.append(dict(job_id=name,model=spec["model"],hardware_profile=spec["hardware_profile"],
            config_id=spec["config_id"],phase_op=spec["phase_op"],budget=spec["steps"],
            decision="propagated_stop" if propagated else "evaluated",stop=result["stop"],level=result["level"],
            stop_mean_T_hat_ns=result["T"],T_pre_ns=result["T_pre"],reason=result["reason"],
            gain_from_previous=result.get("gain_from_previous"),representative_repair_seed=result.get("seed"),
            source_job=result["source_job"],export=result["export"],rule_recomputed=True))
    grouped = defaultdict(list)
    for row in trajectories:
        grouped[row["config_id"],row["phase_op"],row["budget"]].append(row)
    for (_,_,budget),members in grouped.items():
        first=members[0]
        actual=[r for r in members if r["status"] == "ok"]
        row=dict(config_id=first["config_id"],hardware_profile=first["hardware_profile"],phase_op=first["phase_op"],
            budget=budget,planned=5,actual_trajectories=len(actual),reused=sum(r["status"] == "reused" for r in members),
            pending=sum(r["status"] == "pending" for r in members),repair_seeds=[r["repair_seed"] for r in actual])
        if len(actual) == 5:
            row.update(T_hat_mean_ns=statistics.mean(r["T_hat_ns"] for r in actual),
                **common.stats([r["backend_mean_ns"] for r in actual],"trajectory_backend_mean"))
        groups.append(row)
    return stops,groups,trajectories


def unit_tables(jobs,present,states,stops,healthy_index=None):
    if healthy_index is None:
        healthy_index=read(HERE/"manifest/h4_healthy_inputs.json")["candidates"]
        assert len(healthy_index)==224
    bystate={(r["job_id"],r["state"]):r for r in states}
    pre={key(item["spec"]):item for item in present.values() if item["spec"]["kind"] == "fault_pre"}
    selected_stops={}
    for row in stops:
        if row["stop"] and row["decision"] != "propagated_stop":
            unit=(row["config_id"],row["phase_op"],"A3h")
            assert unit not in selected_stops
            selected_stops[unit]=row
    units=[]
    for name,item in present.items():
        spec,result=item["spec"],item["result"]
        if spec["kind"] != "fault_l1":continue
        healthy=pre[key(spec)]
        h=bystate[healthy["spec"]["job_id"],"healthy"]
        r0=bystate[name,"R0"]
        assert result["T_pre"] == h["T_hat_ns"]
        original=healthy_index[result["source_candidate"]]
        assert h["T_hat_ns"]==original["healthy_T_hat"]
        row=dict(**fields(spec),fault="PF4",source_candidate=result["source_candidate"],
            healthy_source_job=healthy["spec"]["job_id"],l1_source_job=name,
            T_pre_ns=h["T_hat_ns"],T_R0_ns=r0["T_hat_ns"],
            healthy_backend_mean_ns=h["backend_mean_ns"],R0_backend_mean_ns=r0["backend_mean_ns"],
            original_healthy_backend_mean_ns=original["backend_mean_ns"],
            healthy_replay_relative_to_original=h["backend_mean_ns"]/original["backend_mean_ns"]-1,
            R0_T_hat_relative_to_healthy=r0["T_hat_ns"]/h["T_hat_ns"]-1,
            R0_relative_to_healthy=r0["backend_mean_ns"]/h["backend_mean_ns"]-1,
            R0_migration={k.removeprefix("migration_"):v for k,v in r0.items() if k.startswith("migration_")},
            EIR_complete=False)
        if spec["start_arm"] == "A3h":
            l1=bystate[name,"L1"]
            row.update(T_L1_ns=l1["T_hat_ns"],L1_backend_mean_ns=l1["backend_mean_ns"],
                L1_T_hat_relative_to_healthy=l1["T_hat_ns"]/h["T_hat_ns"]-1,
                L1_relative_to_healthy=l1["backend_mean_ns"]/h["backend_mean_ns"]-1,
                L1_relative_to_R0=l1["backend_mean_ns"]/r0["backend_mean_ns"]-1)
            if key(spec) in selected_stops:
                stop=selected_stops[key(spec)]
                stage="L2" if stop["level"].startswith("s") else stop["level"]
                final=bystate[stop["source_job"],stage]
                row.update(EIR_complete=True,EIR_level=stop["level"],EIR_reason=stop["reason"],
                    EIR_stop_mean_T_hat_ns=stop["stop_mean_T_hat_ns"],EIR_selected_T_hat_ns=final["T_hat_ns"],
                    EIR_backend_mean_ns=final["backend_mean_ns"],EIR_backend_min_ns=final["backend_min_ns"],
                    EIR_backend_max_ns=final["backend_max_ns"],EIR_source_job=stop["source_job"],
                    EIR_repair_seed=stop["representative_repair_seed"],EIR_export=stop["export"],
                    EIR_stop_T_hat_relative_to_healthy=stop["stop_mean_T_hat_ns"]/h["T_hat_ns"]-1,
                    EIR_relative_to_healthy=final["backend_mean_ns"]/h["backend_mean_ns"]-1,
                    EIR_relative_to_R0=final["backend_mean_ns"]/r0["backend_mean_ns"]-1,
                    EIR_migration={k.removeprefix("migration_"):v for k,v in final.items() if k.startswith("migration_")})
        units.append(row)
    return units


def layer_tables(units):
    contract=hi_config.model_contract("llama")
    assert contract["head_groups"] == 8 and contract["decode_chain"]["norm"] == 1
    chosen={r["phase_op"]:r for r in units if r["hardware_profile"] == "busybarn" and r["start_arm"] == "A3h"}
    hp=f_config.hardware_parameters("busybarn")
    rows=[]
    for phase,prefix in (("prefill","pf"),("decode","dc")):
        weights=dict(ln=2 if phase=="prefill" else 1,mha=8,proj=1,ffn=1)
        if not all(prefix+"_"+op in chosen for op in weights):continue
        values={op:chosen[prefix+"_"+op] for op in weights}
        byte_count=contract["batch"]*(2048 if phase=="prefill" else 1)*contract["hidden"]*2
        handoff=hp["d2d_latency_ticks"]+byte_count/hp["d2d_bytes_per_tick_directed_link"]
        for stage in ("healthy","R0","L1","EIR"):
            if stage == "EIR" and not all(r["EIR_complete"] for r in values.values()):continue
            total=sum(weights[op]*values[op][stage+"_backend_mean_ns"] for op in weights)
            healthy=sum(weights[op]*values[op]["healthy_backend_mean_ns"] for op in weights)+handoff
            rows.append(dict(model="llama",hardware_profile="busybarn",phase=phase,state=stage,
                weights=weights,operator_sum_ns=total,handoff_bytes=byte_count,handoff_ns=handoff,
                layer_ns=total+handoff,relative_to_healthy=(total+handoff)/healthy-1,
                source_jobs={op:values[op]["EIR_source_job" if stage=="EIR" else "healthy_source_job" if stage=="healthy" else "l1_source_job"] for op in weights},
                measurement="serial arithmetic assembly from actual operator means; one faulty 2x2 group; healthy intergroup handoff term"))
    requests=[]
    lookup={(r["phase"],r["state"]):r for r in rows}
    for stage in ("healthy","R0","L1","EIR"):
        if not all((phase,stage) in lookup for phase in ("prefill","decode")):continue
        p,d=lookup["prefill",stage],lookup["decode",stage]
        healthy=lookup["prefill","healthy"]["layer_ns"]+2048*lookup["decode","healthy"]["layer_ns"]
        request=p["layer_ns"]+2048*d["layer_ns"]
        requests.append(dict(model="llama",hardware_profile="busybarn",state=stage,additional_decode_steps=2048,
            fixed_kv_length=2048,prefill_ns=p["layer_ns"],decode_ns=d["layer_ns"],request_ns=request,
            relative_to_healthy=request/healthy-1,measurement="P + 2048 D; one-layer reference, not 80-layer simulation"))
    return rows,requests


def execution(out,present,smoke):
    starts=[read(p) for p in sorted(out.glob("*start*.json"))]
    starts=[r for r in starts if "created_utc" in r]
    finish_path=out/"controller_finish.json"
    finish=read(finish_path) if finish_path.exists() else None
    result=dict(controller_finish=finish,starts=starts,smoke_only=smoke)
    if starts:
        first=min(r["created_utc"] for r in starts)
        result["first_start_utc"]=first
        if finish:
            result.update(completed_utc=finish["utc"],wall_seconds=(datetime.fromisoformat(finish["utc"])-datetime.fromisoformat(first)).total_seconds())
    observations=[]
    for p in sorted(out.glob("*.status*.json"))+sorted(out.glob("*.batch.json")):
        record=read(p)
        if "min_available_gib" in record:
            observations.append(dict(source=str(p.relative_to(ROOT)),own_max_active=record["max_active"],
                minimum_available_gib=record["min_available_gib"],
                combined_reserved_and_active_max=record.get("max_combined_active")))
    result["resources"]=observations
    result["failed_attempts"]=[dict(source=str(p.relative_to(ROOT)),**read(p)) for p in sorted(out.glob("failed_attempts/*/*/receipt.json"))]
    result.update(completed_job_process_hours=sum(r["result"]["wall_seconds"] for r in present.values())/3600,
        completed_receipt_process_hours=sum(r["receipt"]["wall_seconds"] for r in present.values())/3600)
    return result


def job_rows(jobs,present,timing):
    measured={r["job_id"]:r for r in timing}
    output=[]
    for name,job in jobs.items():
        spec=job["spec"]
        row=dict(job_id=name,kind=spec["kind"],hardware_profile=spec["hardware_profile"],
            phase_op=spec.get("phase_op"),start_arm=spec.get("start_arm"),budget=spec.get("steps"),
            repair_seed=spec.get("seed"),planned_seconds=job["estimated_seconds"],
            status=present[name]["result"]["status"] if name in present else "pending")
        if name in present:
            result=present[name]["result"]
            row.update(wall_seconds=result["wall_seconds"],receipt_seconds=present[name]["receipt"]["wall_seconds"],
                peak_rss_gib=result.get("peak_rss_gb"),actual_backend_calls=measured.get(name,{}).get("backend_calls",0))
        output.append(row)
    return output


def render(audit,units,groups,jobtable,layers,requests,timing):
    smoke=audit["smoke_only"]
    text=["# H4 PF4 汇总解析核对" if smoke else "# H4：Llama PF4 故障结果","",
        "本文件只解析本机短验证材料，不计入正式 H4 完成数。" if smoke else
        f"正式作业已完成 {audit['completed_jobs']}/264；其中实际 L2 修复 {audit['actual_repair_trajectories']} 条，停止后复用 {audit['reused_repair_jobs']} 条，剩余待处理 {audit['missing_jobs']} 条。",
        "本批只测试固定 PF4：`(die0,core11) ↔ (die1,core8)` 双向连接失效。主硬件八个 A3h P/D 算子、WATOS 四个 A3h MHA/FFN，以及供 H5 使用的主硬件四个 A1-R0。MF6 和新模型 Split 留在后续批次。",
        "既有 224 个健康候选按原后端十种子均值选择；原始搜索及快照直接复用，不增加健康搜索。健康状态在新进程重放十次后端，作为本批故障前参考。", "",
        "## 作业与后端次数","","| 内容 | 已完成 | 实际后端样本 |","|---|---|---|"]
    for kind,label in (("fault_pre","健康重放"),("fault_l1","R0/L1（A1仅R0）"),("fault_rep","实际L2修复")):
        subset=[r for r in jobtable if r["kind"]==kind and r["status"]=="ok"]
        text.append(f"| {label} | {len(subset)} | {sum(r['actual_backend_calls'] for r in subset)} |")
    text += [f"| 选优、故障设定、停档 | {sum(r['status']=='ok' and r['kind'] in ('fault_selection','fault_select','fault_stop') for r in jobtable)} | 0 |",
        f"| 停止后复用的修复任务 | {audit['reused_repair_jobs']} | 0 |","",
        f"本批已读取 {audit['backend_samples']} 个实际后端样本。最大计划为 16×10 健康 +12×20 A3h R0/L1 +4×10 A1-R0 +180×10 L2 =2240。复用记录仅传递先前选定状态，没有新搜索和后端。", "",
        "## 十二个 A3h PF4 点","",
        "时间为后端十种子均值（毫秒）。EIR 是停止规则选定的状态；若进入 L2，则为该档五条轨迹中后端均值居中的代表。它与五种子平均 CPCM 停档值分开保存。","",
        "| 硬件 | 算子 | 健康 ms | R0 ms | L1 ms | EIR ms | EIR层级/原因 | R0/健康−1 | EIR/健康−1 | EIR/R0−1 |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for row in units:
        if row["start_arm"]!="A3h":continue
        eir=f"{row['EIR_backend_mean_ns']/1e6:.5f}" if row["EIR_complete"] else "待停止判定"
        reason=row["EIR_level"]+"/"+row["EIR_reason"] if row["EIR_complete"] else "进行中"
        ratios=[f"{row[k]:+.2%}" if row["EIR_complete"] else "—" for k in ("EIR_relative_to_healthy","EIR_relative_to_R0")]
        text.append(f"| {row['hardware_profile']} | {row['phase_op']} | {row['healthy_backend_mean_ns']/1e6:.5f} | {row['R0_backend_mean_ns']/1e6:.5f} | {row['L1_backend_mean_ns']/1e6:.5f} | {eir} | {reason} | {row['R0_relative_to_healthy']:+.2%} | {' | '.join(ratios)} |")
    completed=[r for r in units if r["start_arm"]=="A3h" and r["EIR_complete"]]
    if len(completed)==12:
        cells={(r["hardware_profile"],r["phase_op"]):r for r in completed}
        levels=Counter(r["EIR_level"] for r in completed)
        improved=sum(r["EIR_relative_to_R0"]<0 for r in completed)
        main,watos=cells["busybarn","dc_ffn"],cells["watos","dc_ffn"]
        pmha,pproj,pffn=(cells["busybarn",op] for op in ("pf_mha","pf_proj","pf_ffn"))
        text += ["",f"EIR 相对 R0 改善 {improved}/12 点，其余点保留原值。按既定停止规则，{levels['R0']} 点使用 R0、{levels['L1']} 点使用 L1、{sum(v for k,v in levels.items() if k.startswith('s'))} 点进入 L2。主硬件 decode FFN 的健康差从 {main['R0_relative_to_healthy']:+.2%} 降至 {main['EIR_relative_to_healthy']:+.2%}，WATOS 同算子从 {watos['R0_relative_to_healthy']:+.2%} 降至 {watos['EIR_relative_to_healthy']:+.2%}。",
            f"2% 停止条件针对 CPCM 估值。主硬件 prefill MHA 停在 L1 时，估值相对健康为 {pmha['EIR_stop_T_hat_relative_to_healthy']:+.2%}，实际后端仍为 {pmha['EIR_relative_to_healthy']:+.2%}，因此不能把估值恢复写成实际延迟都已恢复到 2%。主硬件 prefill projection/FFN 的 L1 相对 R0 分别变化 {pproj['L1_relative_to_R0']:+.3%}/{pffn['L1_relative_to_R0']:+.3%}，最终都按规则保留 R0。",
            "R0 达到停止条件时就保留 R0，即使已经记录的 L1 后端更低，也不按后端事后改选。健康参考使用本批的新进程十种子重放；与原健康候选记录的差值在单位表中保留，没有混用两次均值。"]
    text += ["","## 五种子与三个预算档","",
        "种子为 `{123,1,2,3,4}`；30/100/300 步各自从同一 L1 状态重新开始，不是同一轨迹的检查点。跨级达到健康 CPCM 的 1.02 倍才停止；100/300 档还检查相邻档五种子平均估值改善是否小于 2%。30 档不使用这一相邻档停止条件。300 档到预算上限也停止。",
        "停档均值、每条轨迹的十后端、代表种子、复用状态和迁移量均逐项保留；报告脚本从原记录重新计算停止理由与中位代表，不改变选择。","",
        "| 硬件 | 算子 | 预算 | 实际轨迹 | 停止后复用 | 待处理 | 五轨迹平均 CPCM ns | 五个后端均值的中位数 ns |",
        "|---|---|---|---|---|---|---|---|"]
    for row in groups:
        if row["actual_trajectories"]+row["reused"]==0:continue
        text.append(f"| {row['hardware_profile']} | {row['phase_op']} | {row['budget']} | {row['actual_trajectories']} | {row['reused']} | {row['pending']} | {row.get('T_hat_mean_ns','—')} | {row.get('trajectory_backend_mean_median_ns','—')} |")
    text += ["","## A1-R0 与迁移","",
        "四个主硬件 decode A1-R0 为后续联合实验提供对照；本批没有 A1-EIR。迁移分别保留 moved_behas、实际 DDR 读源变化字节、唯一搬迁块字节与块数，不把累计读取量当作唯一权重搬迁量。","",
        "| 算子 | A1 健康 ms | A1-R0 ms | R0/健康−1 |","|---|---|---|---|"]
    for row in units:
        if row["start_arm"]=="A1":
            text.append(f"| {row['phase_op']} | {row['healthy_backend_mean_ns']/1e6:.5f} | {row['R0_backend_mean_ns']/1e6:.5f} | {row['R0_relative_to_healthy']:+.2%} |")
    text += ["","## 主硬件单层与 2048 步请求参考","",
        "主硬件同一 PF4 作用于八算子。Prefill 使用 2×RMSNorm+8×MHA+projection+FFN，decode 使用 RMSNorm+8×MHA+projection+FFN；decode FFN 已含另一次 RMSNorm。沿用 H 健康报告的相邻组 handoff 解析项（P 1,048,596 ns，D 532 ns），该项不施加组内 PF4。",
        "下表按实际算子均值串行相加，不是完整层联合仿真。N=2048 表示首 token 后追加 2048 个 decode 步，KV 固定2048；没有做80层容量与流水线仿真。WATOS缺LN/projection，不组装WATOS层。","",
        "| 状态 | Prefill ms | Decode ms | P+2048D s | 相对健康 |","|---|---|---|---|---|"]
    for row in requests:
        text.append(f"| {row['state']} | {row['prefill_ns']/1e6:.5f} | {row['decode_ns']/1e6:.5f} | {row['request_ns']/1e9:.6f} | {row['relative_to_healthy']:+.2%} |")
    if len(requests)==4:
        bystate={r["state"]:r for r in requests}
        d={r["state"]:r for r in layers if r["phase"]=="decode"}
        text += ["",f"主硬件 decode 参考从 R0 的 {d['R0']['relative_to_healthy']:+.2%} 健康差恢复到 {d['EIR']['relative_to_healthy']:+.2%}。P+2048D 参考从 {bystate['R0']['relative_to_healthy']:+.2%} 恢复到 {bystate['EIR']['relative_to_healthy']:+.2%}，EIR 相对 R0 缩短 {1-bystate['EIR']['request_ns']/bystate['R0']['request_ns']:.2%}。"]
    text += ["","## 实际用时","",
        f"已完成作业累计 {audit['execution']['completed_job_process_hours']:.4f} 进程小时；包括复用/停档进程的短开销。各并行作业时长相加不是实验墙钟。"]
    if "wall_seconds" in audit["execution"]:
        x=audit["execution"]
        text += [f"从 {x['first_start_utc']} 到 {x['completed_utc']}，墙钟 {x['wall_seconds']/60:.3f} 分钟。"]
    text += ["", "| 类别 | 作业数 | 实际后端调用秒 | 后端复制/准备秒 | 校准秒 | 搜索秒 | 恢复/最终CPCM秒 | 其余秒 |","|---|---|---|---|---|---|---|---|"]
    for kind in ("fault_pre","fault_l1","fault_rep"):
        subset=[r for r in timing if r["kind"]==kind]
        if not subset:continue
        sums=lambda field:sum(r[field] for r in subset)
        text.append(f"| {kind} | {len(subset)} | {sums('backend_call_seconds'):.3f} | {sums('backend_total_seconds')-sums('backend_call_seconds'):.3f} | {sums('calibration_seconds'):.3f} | {sums('search_seconds')+sums('L1_search_seconds'):.3f} | {sums('restore_best_seconds')+sums('final_cpcm_seconds'):.3f} | {sums('other_seconds')+sums('R0_prepare_seconds'):.3f} |")
    text += ["","L2 原记录的 t_R0/t_L1 是从起点继承的历史成本，已单列，未再次计入当前作业分项。每条修复计入自己的温度校准、搜索、恢复与最终估值；后端调用与复制准备分开。此处为共享服务器的批量成本，正式串行效率测量属于 J。","",
        "逐作业、十种子、状态、修复轨迹、停档、迁移和组装来源分别见 `h4_jobs.csv`、`h4_backend_samples.csv`、`h4_states.csv`、`h4_repair_trajectories.csv`、`h4_stop_decisions.csv`、`h4_units.csv`、`h4_layer_reference.csv`、`h4_request_reference.csv`。完整性记录在 `h4_completion_audit.json`。"]
    return text


def run(args):
    manifest=read(args.manifest)
    jobs,present,missing,states,samples,timing,checks=collect(args.out,manifest,args.progress,args.smoke)
    stops,groups,trajectories=stop_tables(jobs,present)
    units=unit_tables(jobs,present,states,stops)
    layers,requests=layer_tables(units)
    jobtable=job_rows(jobs,present,timing)
    runtime=execution(args.out,present,args.smoke)
    actual=sum(r["status"]=="ok" for r in trajectories)
    reused=sum(r["status"]=="reused" for r in trajectories)
    finish=runtime["controller_finish"]
    complete=not args.smoke and not missing and finish is not None and finish["rc"]==0 and finish["completed"]==264
    audit=dict(status="smoke_parse_passed" if args.smoke else "complete" if complete else "in_progress",
        smoke_only=args.smoke,expected_jobs=264,completed_jobs=len(present),missing_jobs=len(missing),missing=missing,
        formal_completed_jobs=0 if args.smoke else len(present),
        actual_repair_trajectories=actual,reused_repair_jobs=reused,backend_samples=len(samples),
        completed_A3h_fault_units=sum(r["EIR_complete"] for r in units),A1_R0_units=sum(r["start_arm"]=="A1" for r in units),
        stop_records=len(stops),evaluated_stop_records=sum(r["decision"]=="evaluated" for r in stops),
        propagated_stop_records=sum(r["decision"]=="propagated_stop" for r in stops),
        selected_healthy_inputs=224,new_healthy_searches=0,faults=["PF4"],MF6_included=False,Split_included=False,
        kinds=dict(Counter(item["spec"]["kind"] for item in present.values())),
        input_checks=checks,execution=runtime,manifest_sha256=sha(args.manifest),
        healthy_input_index_sha256=sha(HERE/"manifest/h4_healthy_inputs.json"),report_source_sha256=sha(__file__))
    assert len(samples)==sum(row["actual_backend_calls"] for row in jobtable if row["status"]!="pending")
    if not args.progress and not args.smoke:
        assert complete and actual+reused==180 and len(stops)==48 and len(units)==16
        assert audit["completed_A3h_fault_units"]==12 and audit["A1_R0_units"]==4
        assert len(samples)==440+actual*10 and len(layers)==8 and len(requests)==4
    destination=args.destination or (HERE/"evidence/h4_report_smoke" if args.smoke else HERE)
    tables=dict(jobs=jobtable,states=states,backend_samples=samples,timing=timing,repair_trajectories=trajectories,
        budget_groups=groups,stop_decisions=stops,units=units,layer_reference=layers,request_reference=requests)
    if args.smoke:
        import post_fg_split
        probe=read(args.out/"direct_repair_probe.json")
        assert probe["status"]=="passed" and "not a formal L2 run" in probe["scope"]
        probe_samples=[]
        probe_states,probe_timing,probe_evidence=parse_engine(probe["spec"],probe,probe_samples,post_fg_split.static.read_pickle)
        tables.update(probe_states=probe_states,probe_backend_samples=probe_samples,probe_timing=[probe_timing])
        audit["separate_probe"]=dict(scope=probe["scope"],backend_samples=len(probe_samples),formal_completed_increment=0,**probe_evidence)
    for name,rows in tables.items():common.table(destination/"tables"/("h4_"+name+".csv"),rows)
    write(destination/"evidence/h4_completion_audit.json",audit)
    destination.mkdir(parents=True,exist_ok=True)
    filename="H4_REPORT_PARSE_CHECK.md" if args.smoke else "H4_PF4_REPORT.md"
    (destination/filename).write_text("\n".join(render(audit,units,groups,jobtable,layers,requests,timing))+"\n",encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ("status","completed_jobs","actual_repair_trajectories","reused_repair_jobs","backend_samples","completed_A3h_fault_units")}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"out/h4_pf4")
    p.add_argument("--manifest",type=Path,default=HERE/"manifest/h4_pf4.json")
    p.add_argument("--destination",type=Path)
    p.add_argument("--progress",action="store_true")
    p.add_argument("--smoke",action="store_true")
    run(p.parse_args())


if __name__=="__main__":main()
