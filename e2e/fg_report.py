"""Read F/G results and saved graphs; write pilot and healthy-candidate tables.

This performs no search or backend simulation. Latency is reported per layer on
one Die group; the 56-Die assembly is not inferred from an unmeasured tiling.
"""

import argparse
import ast
from collections import Counter, defaultdict
import configparser
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics

from e_run import ROOT, write_json
from fg_check import check, estimate, read

OPS = ("ln", "mha", "proj", "ffn")
PHASES = ("prefill", "decode")
BASELINE_ID = "E_busybarn_2x2_fixed_d2d1"
ID_FIELDS = ("config_id", "stage", "hardware_profile", "shape", "split_mode", "d2d_scale", "family", "fault_id")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def source_path(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def distribution(values):
    mean = statistics.mean(values)
    return dict(n=len(values), min=min(values), q25=quantile(values, .25), median=statistics.median(values),
                q75=quantile(values, .75), max=max(values), mean=mean,
                std=statistics.pstdev(values), cv=statistics.pstdev(values)/mean if mean else 0)


def candidate_audit(specs, out, allow_incomplete=False, metadata_only=False):
    audit, records = check(specs, out, allow_incomplete)
    omitted = 0
    if metadata_only:
        issues = []
        for issue in audit["issues"]:
            if "issues" in issue and "snapshot" in issue["issues"]:
                omitted += 1
                rest = [value for value in issue["issues"] if value != "snapshot"]
                if rest:
                    issues.append(dict(issue, issues=rest))
            else:
                issues.append(issue)
        audit["issues"] = issues
    audit.update(mode="metadata_only" if metadata_only else "results_and_snapshots",
                 snapshot_presence_required=not metadata_only,
                 local_snapshot_absence_omitted=omitted,
                 snapshot_payload_scope="not parsed" if metadata_only else "one workload per configuration/operator and selected healthy candidates",
                 scope="Saved result fields and protocols; local snapshot presence is required only for full report mode.")
    return audit, records


def candidate_rows(records, out=None):
    rows = []
    for r in records.values():
        values = [v["cycles"] for v in r["backend"]]
        row = {k: r[k] for k in (*ID_FIELDS, "candidate_id", "candidate_order", "phase", "op", "method", "kind",
                                "search_steps", "wall_seconds", "search_seconds", "evaluation_seconds", "peak_rss_gb",
                                "T_hat", "T_hat_rand_mean")}
        row.update(backend_mean_ns=statistics.mean(values), backend_min_ns=min(values), backend_max_ns=max(values),
                   backend_seed_count=len(values), backend_seeds=[v["seed"] for v in r["backend"]],
                   backend_reused_from=r.get("evaluation_reused_from"),
                   workload_sha256=r["workload_signature"]["sha256"], **r["fg_metrics"])
        row.update(g=r["g"], search_seed_or_loss=r["x"],
                   backend_median_ns=statistics.median(values), backend_std_ns=statistics.pstdev(values),
                   backend_cv=statistics.pstdev(values)/statistics.mean(values),
                   random_cpcm_min_ns=min(r["T_hat_rand"]) if r["T_hat_rand"] else None,
                   random_cpcm_max_ns=max(r["T_hat_rand"]) if r["T_hat_rand"] else None,
                   random_cpcm_std_ns=statistics.pstdev(r["T_hat_rand"]) if r["T_hat_rand"] else None,
                   backend_seconds_total=sum(r["backend_seconds"]))
        if out:
            row["source_result"] = source_path(out/"results"/(r["candidate_id"]+".json"))
        rows.append(row)
    return rows


def operator_rows(candidates):
    grouped = defaultdict(list)
    for row in candidates:
        if row["kind"] != "terminal":
            grouped[tuple(row[k] for k in (*ID_FIELDS, "phase", "op", "method"))].append(row)
    rows = []
    for key, candidates_ in sorted(grouped.items()):
        row = dict(zip((*ID_FIELDS, "phase", "op", "method"), key))
        row.update(completed=len(candidates_), planned=14,
                   all_candidate_mean_ns=statistics.mean(c["backend_mean_ns"] for c in candidates_),
                   candidate_mean_min_ns=min(c["backend_mean_ns"] for c in candidates_),
                   candidate_mean_max_ns=max(c["backend_mean_ns"] for c in candidates_))
        row.update({"candidate_"+k+"_ns":v for k,v in distribution([c["backend_mean_ns"] for c in candidates_]).items()
                    if k not in ("n", "cv")})
        row["candidate_cv"] = distribution([c["backend_mean_ns"] for c in candidates_])["cv"]
        row["candidate_sources"] = [c["candidate_id"] for c in sorted(candidates_, key=lambda c:c["candidate_order"])]
        if len(candidates_) == 14:
            assert {c["candidate_order"] for c in candidates_} == set(range(14)), key
            best = min(candidates_, key=lambda c: (c["backend_mean_ns"], c["candidate_order"]))
            random = min(candidates_, key=lambda c: (c["T_hat_rand_mean"], c["candidate_order"]))
            deterministic = min(candidates_, key=lambda c: (c["T_hat"], c["candidate_order"]))
            row.update(backend_best_candidate=best["candidate_id"], backend_best_ns=best["backend_mean_ns"],
                       candidate_median_ns=statistics.median(c["backend_mean_ns"] for c in candidates_),
                       random_cpcm_candidate=random["candidate_id"], random_cpcm_ns=random["backend_mean_ns"],
                       random_selector_gap=random["backend_mean_ns"] / best["backend_mean_ns"] - 1,
                       deterministic_cpcm_candidate=deterministic["candidate_id"],
                       deterministic_cpcm_ns=deterministic["backend_mean_ns"],
                       deterministic_selector_gap=deterministic["backend_mean_ns"] / best["backend_mean_ns"] - 1,
                       selection_status="fourteen_candidates_complete")
        else:
            row["selection_status"] = "partial_candidates"
            if len(candidates_) == 1:
                row.update(pilot_candidate=candidates_[0]["candidate_id"], pilot_candidate_ns=candidates_[0]["backend_mean_ns"])
        rows.append(row)
    return rows


def baseline_operators(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    candidate_file=path.with_name("candidate_table.csv")
    with candidate_file.open(encoding="utf-8-sig",newline="") as file:
        candidates=list(csv.DictReader(file))
    result = []
    for row in rows:
        if row["shape"] != "2x2":
            continue
        result.append(dict(config_id=BASELINE_ID, stage="E_reuse", hardware_profile="busybarn", shape="2x2",
            split_mode="fixed", d2d_scale=1.0, family="healthy", fault_id="healthy", phase=row["phase"],
            op=row["op"], method=row["method"], completed=14, planned=14,
            backend_best_candidate=row["candidate"], backend_best_ns=float(row["cycles"]),
            candidate_median_ns=float(row["candidate_median"]),
            random_cpcm_candidate=row["random_cpcm_candidate"], random_cpcm_ns=float(row["random_cpcm_backend_cycles"]),
            random_selector_gap=float(row["random_cpcm_backend_cycles"])/float(row["cycles"])-1,
            deterministic_cpcm_candidate=row["det_cpcm_candidate"],
            deterministic_cpcm_ns=float(row["det_cpcm_backend_cycles"]),
            deterministic_selector_gap=float(row["det_cpcm_backend_cycles"])/float(row["cycles"])-1,
            selection_status="reused_E_fourteen_candidates", source=source_path(path),
            candidate_sources=[r["candidate_id"] for r in candidates if r["kind"]!="terminal"
                               and all(r[k]==row[k] for k in ("shape","phase","op","method"))],
            candidate_source_table=source_path(candidate_file)))
    return result


def layer_rows(operators):
    groups = defaultdict(dict)
    for row in operators:
        key = tuple(row[k] for k in (*ID_FIELDS, "method"))
        groups[key][row["phase"], row["op"]] = row
    rows = []
    for key, cells in sorted(groups.items()):
        if not all((phase, op) in cells for phase in PHASES for op in OPS):
            continue
        for protocol in ("backend_best", "candidate_median", "random_cpcm", "deterministic_cpcm", "pilot_candidate"):
            if not all(protocol + "_ns" in r for r in cells.values()):
                continue
            row = dict(zip((*ID_FIELDS, "method"), key))
            row.update(protocol=protocol, ln_count="fixed", micro_mha_heads=8, mha_multiplier=8,
                       model_heads=64, batch=16, hidden=6144, prefill_length=2048, decode_kv_length=2048,
                       measurement="operator backend results assembled into one layer")
            row["operator_sources"] = {phase+"_"+op:
                cells[phase,op].get(protocol+"_candidate", cells[phase,op].get("candidate_sources", []))
                for phase in PHASES for op in OPS}
            for phase in PHASES:
                row[phase + "_ns"] = sum((1 if phase == "decode" else 2) * cells[phase, op][protocol + "_ns"]
                    if op == "ln" else (8 if op == "mha" else 1) * cells[phase, op][protocol + "_ns"] for op in OPS)
            row["TTFT_us"] = row["prefill_ns"] / 1000
            row["TPOT_us"] = row["decode_ns"] / 1000
            for length in (512, 2048):
                row[f"request_{length}_ns"] = row["prefill_ns"] + length * row["decode_ns"]
            rows.append(row)
    lookup = {(r["config_id"], r["family"], r["fault_id"], r["method"], r["protocol"]): r for r in rows}
    for row in rows:
        if row["method"] == "A3h":
            for other in ("A1", "Gemini"):
                ref = lookup.get((row["config_id"], row["family"], row["fault_id"], other, row["protocol"]))
                if ref:
                    for metric in ("prefill", "decode", "request_512", "request_2048"):
                        row[f"{metric}_relative_to_{other}"] = row[metric+"_ns"] / ref[metric+"_ns"] - 1
        if row["stage"] == "G" and row["family"] == "healthy":
            ref = lookup.get((BASELINE_ID, "healthy", "healthy", row["method"], row["protocol"]))
            if ref:
                cores = math.prod(map(int, row["shape"].split("x"))) * 16
                for metric in ("prefill", "decode", "request_512", "request_2048"):
                    speedup = ref[metric+"_ns"] / row[metric+"_ns"]
                    row[metric+"_speedup_vs_64cores"] = speedup
                    row[metric+"_strong_scaling_efficiency"] = speedup / (cores / 64)
    return rows


def backend_samples(records, out):
    rows = []
    for r in records.values():
        identity = {k:r[k] for k in (*ID_FIELDS, "candidate_id", "phase", "op", "method", "kind")}
        for value in r["backend"]:
            total, comp, comm = value["triple"]
            rows.append(dict(identity, seed=value["seed"], total_ns=total, compute_only_ns=comp,
                             communication_only_ns=comm, overlap_ns=total-comp-comm,
                             source_result=source_path(out/"results"/(r["candidate_id"]+".json")),
                             reused_from=r.get("evaluation_reused_from")))
    return rows


def candidate_distributions(operators):
    return [{k:v for k,v in row.items() if k in (*ID_FIELDS, "phase", "op", "method", "completed", "planned")
             or k.startswith("candidate_") or k in ("all_candidate_mean_ns", "selection_status")}
            for row in operators if row["stage"] != "E_reuse"]


def terminal_pairs(records):
    rows = []
    for terminal in records.values():
        if terminal["kind"] != "terminal":
            continue
        best = records[terminal["checks"]["from_search"]]
        a = [v["cycles"] for v in best["backend"]]
        b = [v["cycles"] for v in terminal["backend"]]
        rows.append(dict({k:best[k] for k in (*ID_FIELDS,"phase","op","method")},
                         best_candidate=best["candidate_id"], terminal_candidate=terminal["candidate_id"],
                         best_ns=statistics.mean(a), terminal_ns=statistics.mean(b),
                         best_min_ns=min(a), best_max_ns=max(a), terminal_min_ns=min(b), terminal_max_ns=max(b),
                         terminal_vs_best=statistics.mean(b)/statistics.mean(a)-1,
                         best_T_hat=best["T_hat"], terminal_T_hat=terminal["T_hat"],
                         backend_reused=terminal.get("evaluation_reused_from") == best["candidate_id"],
                         seeds=[v["seed"] for v in best["backend"]]))
    return rows


def selected_tables(operators, records, candidates):
    by_id = {r["candidate_id"]:r for r in candidates}
    components, graphs = [], []
    for operator in operators:
        if operator["stage"] == "E_reuse" or "backend_best_candidate" not in operator:
            continue
        for protocol in ("backend_best", "random_cpcm", "deterministic_cpcm"):
            name = operator[protocol+"_candidate"]
            record, candidate = records[name], by_id[name]
            row = {k:operator[k] for k in (*ID_FIELDS,"phase","op","method")}
            row.update(protocol=protocol, candidate_id=name, source_result=candidate["source_result"])
            triples = [v["triple"] for v in record["backend"]]
            total, comp, comm = [statistics.mean(t[i] for t in triples) for i in range(3)]
            components.append(dict(row,total_ns=total,compute_only_ns=comp,communication_only_ns=comm,
                                   overlap_ns=total-comp-comm,compute_only_fraction=comp/total,
                                   communication_only_fraction=comm/total,overlap_fraction=(total-comp-comm)/total))
            graphs.append(dict(row, **record["fg_metrics"],
                               **{k:record[k] for k in ("search_seconds","evaluation_seconds","wall_seconds","peak_rss_gb")},
                               backend_seconds_total=sum(record["backend_seconds"])))
    return components, graphs


def scaling_rows(layers):
    baseline = {(r["method"],r["protocol"]):r for r in layers if r["config_id"] == BASELINE_ID}
    rows = []
    for r in layers:
        if r["stage"] != "G" or r["family"] != "healthy":
            continue
        ref = baseline[r["method"],r["protocol"]]
        cores = math.prod(map(int,r["shape"].split("x"))) * 16
        for metric in ("prefill", "decode", "request_512", "request_2048"):
            speedup = ref[metric+"_ns"] / r[metric+"_ns"]
            rows.append(dict(config_id=r["config_id"],split_mode=r["split_mode"],shape=r["shape"],cores=cores,
                             method=r["method"],protocol=r["protocol"],metric=metric,value_ns=r[metric+"_ns"],
                             baseline_config=BASELINE_ID,baseline_cores=64,baseline_ns=ref[metric+"_ns"],
                             speedup=speedup,efficiency=speedup/(cores/64),
                             source_operators=r["operator_sources"],baseline_source_operators=ref["operator_sources"]))
    return rows


def bandwidth_rows(operators):
    rows=[]
    chosen=[r for r in operators if r["hardware_profile"]=="watos" and r["phase"]=="decode"
            and r["op"] in ("mha","ffn") and r["method"] in ("A1","A3h")
            and r["family"] in ("healthy","bandwidth") and r["completed"]==14]
    refs={(r["op"],r["method"]):r for r in chosen if r["d2d_scale"]==1}
    for r in chosen:
        for protocol in ("backend_best","candidate_median","random_cpcm","deterministic_cpcm"):
            ref=refs[r["op"],r["method"]]
            value=r[protocol+"_ns"]
            rows.append(dict(config_id=r["config_id"],d2d_scale=r["d2d_scale"],phase="decode",op=r["op"],
                             method=r["method"],protocol=protocol,cycles_ns=value,reference_1x_ns=ref[protocol+"_ns"],
                             relative_to_1x=value/ref[protocol+"_ns"]-1,
                             candidate=r.get(protocol+"_candidate"),candidate_sources=r.get("candidate_sources",[]),
                             mapping="independently searched at each bandwidth"))
    return rows


def snapshot_workload(path, spec):
    import e2e_pilot
    c = e2e_pilot.C
    c.WORK = str(ROOT / "e2e/work/artifact")
    c._artifact_path()
    with gzip.open(path, "rb") as file:
        blob, _ = pickle.load(file)
    behaviours, data, events = pickle.loads(blob)
    effective, rounded = 0, 0
    counts, stage_counts, stage_ops, stage_flops = Counter(), Counter(), defaultdict(set), Counter()
    reduction_inputs = 0
    vector_inputs = Counter()
    for b in behaviours.values():
        counts[b.beha_type] += 1
        stage = b.beha_name.rsplit("_", 1)[0]
        stage_counts[stage] += 1
        stage_ops[stage].add(b.beha_tag[:-1])
        if b.device == "tensorcore":
            a, weight = b.beha_datashape[:2]
            k = int(a[-1]); outputs = math.prod(int(n) for n in a[:-1]) * int(weight[-1])
            logical = 2 * k * outputs
            executed = 2 * math.ceil(k / 128) * 128 * math.ceil(outputs / 64) * 64
            effective += logical; rounded += executed; stage_flops[stage] += logical
        if b.beha_type in ("matadd", "redsum", "redmax"):
            reduction_inputs += sum(math.prod(int(n) for n in shape) for shape in b.beha_datashape)
        if b.device == "vectorunit":
            vector_inputs[b.beha_type] += sum(math.prod(int(n) for n in shape) for shape in b.beha_datashape)
    heads = len(set().union(*(v for k, v in stage_ops.items() if k.startswith("MHA_qconv")))) if spec["op"] == "mha" else None
    inputs = [{"tag": list(tag), "name": d.data_name, "shape": list(map(int, d.data_shape)),
               "elements": math.prod(map(int, d.data_shape))} for tag, d in data.items() if d.data_name == "ifmap"]
    primary = inputs[0]
    assert primary["shape"][0] == 16 and primary["shape"][-1] == 6144, primary
    assert primary["shape"][-2] == (2048 if spec["phase"] == "prefill" else 1), primary
    if spec["op"] == "mha":
        assert heads == 8, (spec["candidate_id"], heads)
    return {**{k: spec[k] for k in (*ID_FIELDS, "phase", "op")},
            "source_candidate": spec["candidate_id"], "source_snapshot": str(path),
            "input_tensors": inputs, "model_heads":64, "micro_mha_heads":heads,
            "mha_aggregation_multiplier":8 if spec["op"]=="mha" else None,
            "tensor_effective_multiply_add_flops":effective, "tensor_grain_rounded_flops":rounded,
            "tensor_padding_fraction":rounded/effective-1 if effective else None,
            "tensor_bias_additions_in_flop_count":False,
            "reduction_events":sum(counts[k] for k in ("matadd","redsum","redmax")),
            "reduction_input_elements":reduction_inputs,
            "vector_input_elements_by_type":dict(vector_inputs),
            "computation_events_by_type":dict(counts), "computation_events_by_stage":dict(stage_counts),
            "effective_tensor_flops_by_stage":dict(stage_flops),
            "prebroadcast_events":len(events), "tensor_objects":len(data),
            "intermediate_tensor_elements_sum":sum(math.prod(map(int,d.data_shape)) for d in data.values())}


def workload_rows(specs, records, out):
    samples = {}
    for spec in specs:
        if spec["candidate_id"] in records and spec["family"] == "healthy":
            samples.setdefault((spec["config_id"],spec["phase"],spec["op"]),spec)
    rows = [snapshot_workload(out/"snapshots"/(s["candidate_id"]+".pkl.gz"),s) for s in samples.values()]
    by_op = defaultdict(set)
    for row in rows:
        by_op[row["phase"],row["op"]].add(row["tensor_effective_multiply_add_flops"])
    return rows, [{"phase":key[0],"op":key[1],"effective_tensor_work_equal_across_configurations":len(values)==1,
                   "effective_tensor_work_values":sorted(values)} for key,values in sorted(by_op.items())]


def resource_load_rows(operators, records, out):
    """Sum saved per-resource work using the configured service formulas."""
    import e_gemini_analysis as static
    import f_config
    classes = static.pickle_types()
    rows=[]
    for operator in operators:
        if operator["stage"]=="E_reuse" or operator["family"]!="healthy" or "backend_best_candidate" not in operator:
            continue
        name=operator["backend_best_candidate"]
        record=records[name]
        snapshot=out/"snapshots"/(name+".pkl.gz")
        raw=out/"raw"/(name+".pkl.gz")
        blob,_=static.read_pickle(snapshot)
        behaviours,_,events=pickle.loads(blob)
        selected=static.read_pickle(raw)["selected"]
        mapping=selected["e3map"]
        assert int(selected["T_bc"])==record["T_hat"]
        assert all(tuple(b.location)==tuple(mapping["beha_loc"][tag]) for tag,b in behaviours.items())
        parser=configparser.ConfigParser()
        parser.read_string(f_config.config_text(record))
        config={section:{k:ast.literal_eval(v) if v[:1].isdigit() or v.startswith("[") else v
                         for k,v in parser[section].items()} for section in parser.sections()}
        units={k:cls((0,0,0),config[k+"_cfg"]) for k,cls in zip(("tensorcore","vectorunit"),classes)}
        comp={k:Counter() for k in units}
        for event in events.values():
            if event.event_type=="computation":
                frequency=config["time_unit"]["frequency"]
                if event.comp_device=="tensorcore":
                    frequency*=config["tensorcore_cfg"]["fg_compute_scale"]
                duration=int(units[event.comp_device].working_time(event.comp_datashape,event.comp_type,frequency))
                comp[event.comp_device][tuple(event.comp_location)]+=duration
        links=defaultdict(Counter)
        link_bytes=defaultdict(Counter)
        ddr=Counter()
        communication_count=0
        for route_id,variants in mapping["routes"].items():
            nbytes=route_id[1]
            for source,tree,tag in variants:
                communication_count+=1
                if source[1]>=16:
                    ddr[tuple(source)]+=nbytes
                for edge in {static.link_key(src,dst) for src,targets in tree for dst in targets}:
                    kind=static.link_class(edge)
                    cfg=config[kind+"_cfg"]
                    links[kind][edge]+=math.ceil(math.ceil(nbytes/cfg[kind+"_bandwidth"])*cfg[kind+"_timeunit"])+cfg[kind+"_latency"]
                    link_bytes[kind][edge]+=nbytes
        assert sum(e.event_type=="computation" for e in events.values())+communication_count==len(mapping["layout"]["order"])
        nr,nc=map(int,record["shape"].split("x")); dies=nr*nc
        counts=dict(tensorcore=dies*16,vectorunit=dies*16,ddr_read_bytes=dies*8,
                    ddr2co=dies*8,co2ddr=dies*8,co2co=dies*48,ch2ch=4*(nr*(nc-1)+(nr-1)*nc))
        minimum=min(v["cycles"] for v in record["backend"])
        loads={**comp,**dict(links),"ddr_read_bytes":ddr}
        for resource,values in loads.items():
            desc=static.description(values,counts[resource])
            if resource!="ddr_read_bytes":
                assert desc["maximum"]<=minimum,(name,resource,desc["maximum"],minimum)
            rows.append(dict({k:record[k] for k in (*ID_FIELDS,"phase","op","method")},
                             candidate_id=name,protocol="backend_best",resource=resource,
                             unit="bytes" if resource=="ddr_read_bytes" else "ns",
                             **{k:v for k,v in desc.items() if k!="per_resource"},
                             link_bytes_total=sum(link_bytes[resource].values()) if resource in link_bytes else None,
                             source_snapshot=source_path(snapshot),source_raw=source_path(raw),
                             formula_source="f_config.config_text + artifact device/link working_time",
                             interpretation="static cumulative service work; no queue waiting reconstructed"))
    return rows


def timing_tables(out, workflow_manifest=None, run_root=None, batch_path=None):
    from fg_batch import identity, job_items
    jobs={j["id"]:j for j in job_items(workflow_manifest)} if workflow_manifest else {}
    ledger=[]
    attempts={}
    def attempt(row, path):
        if all(k in row for k in ("id","rc","wall_seconds")) and row["rc"]!=0:
            key=(row["id"],row.get("identity"),row["rc"],row["wall_seconds"])
            entry=attempts.setdefault(key,dict(job_id=row["id"],identity=row.get("identity"),rc=row["rc"],
                 wall_seconds=row["wall_seconds"],peak_rss_gb=row.get("peak_rss_gb"),sources=[]))
            entry["sources"].append(source_path(path))
    for path in sorted((out/"receipts").glob("*.json")):
        r=read(path)
        if r["rc"]!=0:
            attempt(r,path)
            continue
        job=jobs.get(r["id"])
        if job:
            assert identity(job)==r["identity"],r["id"]
            spec=job["spec"]
        else:
            spec=read(out/"specs"/(r["id"]+".json"))
        result=read(out/r["result"])
        ledger.append(dict(job_id=r["id"],identity=r["identity"],kind=spec["kind"],
                           **{k:spec.get(k) for k in (*ID_FIELDS,"phase","op","method","g","steps")},
                           rc=0,status=result["status"],wall_seconds=r["wall_seconds"],peak_rss_gb=r["peak_rss_gb"],
                           source_receipt=source_path(path),source_result=source_path(out/r["result"])))
    controllers=[]
    controller_records={}
    roots=[out,run_root] if run_root else [out]
    batch_files={p for root in roots for p in root.rglob("*.batch.json")}
    if batch_path:
        batch_files.add(batch_path)
    for path in sorted(batch_files):
        r=read(path)
        if "runs" not in r:
            continue
        for row in r["runs"]:
            attempt(row,path)
        digest=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
        if digest in controller_records:
            controller_records[digest]["source_batches"].append(source_path(path))
            continue
        record=dict(source_batch=source_path(path),source_batches=[source_path(path)],record_sha256=digest,
                                wall_seconds=r["wall_seconds"],
                                workers=r["workers"],completed=r["completed"],failed=r["failed"],pending=r["pending"],
                                max_active=r["max_active"],min_available_gib=r["min_available_gib"],
                                recorded_run_rows=len(r["runs"]),
                                accounting="controller rows overlap across resumptions; excluded from summed job time")
        controllers.append(record)
        controller_records[digest]=record
    if run_root:
        for patch_dir in run_root.rglob("patches"):
            for path in patch_dir.rglob("*.json"):
                r=read(path)
                if isinstance(r,dict):
                    attempt(r,path)
    assert len({r["job_id"] for r in ledger})==len(ledger)
    grouped=defaultdict(list)
    keys=(*ID_FIELDS,"kind","phase","op","method","g","steps","status")
    for r in ledger:
        grouped[tuple(r[k] for k in keys)].append(r)
    categories=[]
    for key,members in sorted(grouped.items(),key=lambda item:str(item[0])):
        categories.append(dict(zip(keys,key))|dict(jobs=len(members),process_hours=sum(r["wall_seconds"] for r in members)/3600,
                          **{"wall_seconds_"+k:v for k,v in distribution([r["wall_seconds"] for r in members]).items()},
                          peak_rss_gb_max=max(r["peak_rss_gb"] for r in members)))
    retries=list(attempts.values())
    summary=dict(successful_job_receipts=len(ledger),successful_process_hours=sum(r["wall_seconds"] for r in ledger)/3600,
                 failed_attempts=len(retries),failed_attempt_process_hours=sum(r["wall_seconds"] for r in retries)/3600,
                 retried_jobs=sorted({r["job_id"] for r in retries}&{r["job_id"] for r in ledger}),
                 source="one current successful receipt per job; archived nonzero-return attempts counted separately",
                 available_unique_batch_records=len(controllers),controller_times_are_overlapping=True)
    return dict(job_timing=ledger,timing_by_category=categories,controller_timing=controllers,retry_attempts=retries),summary


def observed_time(out, records, batch_path):
    result = dict(result_process_hours=sum(r["wall_seconds"] for r in records.values())/3600,
                  max_peak_rss_gb=max((r["peak_rss_gb"] for r in records.values()),default=0),
                  backend_seed_counts=sorted({len(r["backend"]) for r in records.values()}),
                  backend_actual_calls=sum(len(r["backend"]) for r in records.values() if "evaluation_reused_from" not in r),
                  backend_reused_samples=sum(len(r["backend"]) for r in records.values() if "evaluation_reused_from" in r))
    if batch_path:
        batch=read(batch_path)
        fresh=[r for r in batch["runs"] if not r.get("cached")]
        seconds=sum(r["wall_seconds"] for r in fresh)
        result.update(batch_file=str(batch_path),batch_wall_seconds=batch["wall_seconds"],
            worker_process_hours=seconds/3600,mean_active_workers=seconds/batch["wall_seconds"],
            configured_worker_limit=batch["workers"],max_observed_active=batch["max_active"],
            min_available_gib=batch["min_available_gib"],failed=batch["failed"],pending=batch["pending"],
            completed=batch["completed"],platform=batch["platform"],python=batch["python"])
    return result


def markdown(result):
    a=result["audit"]; t=result["observed_time"]
    lines=["# F/G 已完成候选汇总","",f"读取 {a['completed_records']}/{a['expected_records']} 条结果；核对问题 {len(a['issues'])} 项。",
        "", "检查方式："+("仅核对结果JSON；本地尚未下载的快照不记为实验失败，未作快照解析。" if a["mode"]=="metadata_only" else "核对结果JSON与快照文件，并解析工作量和静态资源工作。"),
        "", f"每条结果的后端种子数为 {t['backend_seed_counts']}。十四候选齐备才计算后端选优、候选中位数与随机平局 CPCM 选优；单候选小批仅列该候选的表现。",
        "", "层公式采用 E fixed：prefill = 2 LN + 8 MHA + proj + FFN；decode = LN + 8 MHA + proj + FFN，其中 decode FFN 已含一次 LN。MHA 微图八头、总层乘八，共六十四头。",
        "", "下表为一个 Die 组上的单层组装时间，单位微秒；请求时间仍固定 KV 长度 2048。", "",
        "| 配置 | 实验 | 故障 | 方法 | 候选处理 | TTFT / μs | TPOT / μs |", "|---|---|---|---|---|---:|---:|"]
    for row in result["layers"]:
        lines.append(f"| {row['config_id']} | {row['family']} | {row['fault_id']} | {row['method']} | {row['protocol']} | {row['TTFT_us']:.3f} | {row['TPOT_us']:.3f} |")
    lines += ["",f"保存结果累计耗时 {t['result_process_hours']:.2f} 进程小时，单进程最高峰值 {t['max_peak_rss_gb']:.3f} GiB。"]
    if "successful_job_receipts" in t:
        lines += [f"按每作业唯一成功receipt计时，共{t['successful_job_receipts']}项、{t['successful_process_hours']:.2f}进程小时；归档失败尝试{t['failed_attempts']}次、{t['failed_attempt_process_hours']:.2f}进程小时，单独统计。多次控制器记录不相加。"]
    if "batch_wall_seconds" in t:
        lines += [f"实际批次用时 {t['batch_wall_seconds']/60:.2f} 分钟，平均活跃进程 {t['mean_active_workers']:.1f}，最高同时 {t['max_observed_active']}；采样最低可用内存 {t['min_available_gib']:.2f} GiB。"]
    p=result.get("search_projection")
    if p:
        lines += ["",f"按已测类别估计，剩余搜索约 {p['remaining_process_hours']:.2f} 进程小时；按 {p['workers']} 并行、长任务优先排队约 {p['ideal_longest_first_hours']:.2f} 小时，尚未测得的搜索有 {p['remaining_jobs_unmeasured']} 个。修复、配对单跑和联合实验另计；调度还需加入任务依赖与实测内存。"]
    lines += ["", "完整模式中的工作量与取整量由保存快照解析：有效乘加量为各 TensorCore 事件 2×M×K×N 之和；取整量应用原 128×64 grain。matmac 的额外偏置加法未计入该乘加量。归约单列 matadd/redsum/redmax 的事件与输入元素。通信、路径和事件图指标来自运行记录。", "",
              "后端三元组分解为总时间、只有计算活动的时间、只有通信活动的时间；余项是计算与通信同时活动的时间。静态资源负载是保存事件的服务时间累加，未重建链路排队时间。", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--run-out",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--full-manifest",type=Path)
    parser.add_argument("--batch",type=Path)
    parser.add_argument("--workflow-manifest",type=Path)
    parser.add_argument("--run-root",type=Path,help="Local full archive root containing batch and archived attempt records")
    parser.add_argument("--workers",type=int,default=128)
    parser.add_argument("--allow-incomplete",action="store_true")
    parser.add_argument("--baseline-table",type=Path,default=ROOT/"e2e/e/tables/operator_table.csv")
    parser.add_argument("--metadata-only","--skip-snapshots",dest="metadata_only",action="store_true")
    parser.add_argument("--skip-static-loads",action="store_true")
    args=parser.parse_args()
    specs=[r.get("spec",r) for r in read(args.manifest)]
    audit,records=candidate_audit(specs,args.run_out,args.allow_incomplete,args.metadata_only)
    candidates_=candidate_rows(records,args.run_out)
    operators=operator_rows(candidates_)
    if args.baseline_table.exists(): operators+=baseline_operators(args.baseline_table)
    layers=layer_rows(operators)
    work,workchecks=([],[]) if args.metadata_only else workload_rows(specs,records,args.run_out)
    resources=[] if args.metadata_only or args.skip_static_loads else resource_load_rows(operators,records,args.run_out)
    audit["static_resource_loads_parsed"]=bool(resources)
    if args.skip_static_loads and not args.metadata_only:
        audit["snapshot_payload_scope"]="one workload per configuration/operator"
    components,graphs=selected_tables(operators,records,candidates_)
    times,time_summary=timing_tables(args.run_out,args.workflow_manifest,args.run_root,args.batch)
    tables=dict(candidates=candidates_,operators=operators,layers=layers,workload=work,workload_checks=workchecks,
                backend_samples=backend_samples(records,args.run_out),candidate_distribution=candidate_distributions(operators),
                gemini_terminal_pairs=terminal_pairs(records),scaling=scaling_rows(layers),bandwidth_search=bandwidth_rows(operators),
                backend_components=components,graph_metrics=graphs,resource_loads=resources,**times)
    result=dict(generated_utc=datetime.now(timezone.utc).isoformat(),audit=audit,observed_time=observed_time(args.run_out,records,None)|time_summary,
                sources=dict(manifest=source_path(args.manifest),run_output=source_path(args.run_out),
                             baseline_table=source_path(args.baseline_table),
                             manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                             artifact_commit=(ROOT/"third_party/isca2026-busybarn-artifact.COMMIT").read_text(encoding="utf-8-sig").splitlines()[0],
                             report_script=source_path(__file__),report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             workflow_manifest=source_path(args.workflow_manifest) if args.workflow_manifest else None,
                             run_root=source_path(args.run_root) if args.run_root else None),
                operators=operators,layers=layers,workloads=work,workload_checks=workchecks,
                table_rows={k:len(v) for k,v in tables.items()},
                definitions=dict(backend_components="total; computation only; communication only; total minus the latter two is simultaneous activity",
                                 candidate_distribution="distribution of fourteen candidate backend means; distinct from ten seeds within one candidate",
                                 scaling="same method and selection protocol, E 64-core 2x2 baseline, W1/fixed-LN/64 heads",
                                 workload="tensor multiply-add count; padding from 128x64 grain; reduction listed separately",
                                 timing="parallel process wall time, not J-stage isolated algorithm latency"),
                formula=dict(prefill="2 LN + 8 MHA + proj + FFN",decode="LN + 8 MHA + proj + FFN",
                             scope="one layer on one Die group",tick_seconds=1e-9))
    if args.full_manifest: result["search_projection"]=estimate(read(args.full_manifest),records,args.workers)
    args.out.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():
        write_csv(args.out/(name+".csv"),rows)
        write_json(args.out/(name+".json"),rows)
    write_json(args.out/"summary.json",result)
    (args.out/"SUMMARY.md").write_text(markdown(result),encoding="utf-8")
    print(json.dumps(dict(completed=audit["completed_records"],issues=audit["issues"],
                         workload_checks=workchecks,layers=len(layers)),ensure_ascii=False))
    return bool(audit["issues"] or any(not r["effective_tensor_work_equal_across_configurations"] for r in workchecks))


if __name__=="__main__":
    raise SystemExit(main())
