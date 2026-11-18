"""Freeze Online-CPCM-Gated choices from saved graphs and known release times."""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import fg_fault as F
import post_fg_dstu as D

ROOT = F.ROOT
BASE = ROOT / "e2e/dstu_supplement"
OUT = BASE / "out/main"
RULE_VERSION = "online_cpcm_gated_shared_and_KV_end_v1"
RULE_TEXT = "S256 iff shared_directed_resources > 0 and known_tau_ns < unchunked_P_bb_CPCM_KV_end_ns; otherwise BB0"
PHASES = (.1, .5, .9)


def digest(value):
    return hashlib.sha256(json.dumps(F.plain(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(F.plain(value), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def contexts():
    for hardware in ("wsc", "face"):
        for state in ("A3hEIR", "A3hR0", "A1R0", "A1EIR"):
            yield dict(scope="S", model="gpt", hardware_profile=hardware, mapping_state=state)
    for scope, model in (("H", "llama"), ("I", "qwen3moe")):
        for state in ("healthy_A3h", "D_A3hEIR"):
            yield dict(scope=scope, model=model, hardware_profile="busybarn", mapping_state=state)


def point_name(spec):
    return "_".join(spec[k] for k in ("scope", "hardware_profile", "mapping_state"))


def input_context(spec):
    """Read only saved graph metadata and the common, already-defined release reference."""
    files = []
    if spec["scope"] == "S":
        import hi_closeout_s as S
        prepared_file = S.OUT / "results" / f"S_{spec['hardware_profile']}__graph_{spec['mapping_state']}.json"
        prepared = F.read_json(prepared_file)
        graph_file = F.path(prepared["graph_record"])
        reference_file = S.OLD / "results" / (S.base_spec(spec["hardware_profile"])["config_id"]+"__joint_reference.json")
        inner = dict(S.base_spec(spec["hardware_profile"]))
        files.append(prepared_file)
    else:
        import dstu_supplement_hi as H
        import hi_config
        stage, prefix = H.MODELS[spec["model"]]
        folder = ROOT / "e2e/hi/out" / stage / "results"
        graph_file = folder / f"{prefix}__graph_{spec['mapping_state']}.json"
        reference_file = folder / f"{prefix}__reference.json"
        inner = dict(model=spec["model"], hardware_profile="busybarn", shape="2x2",
                     split_mode="fixed", d2d_scale=1, workload_version=hi_config.WORKLOAD_VERSION)
    graph, reference = F.read_json(graph_file), F.read_json(reference_file)
    if spec["scope"] != "S":
        assert graph["model_contract_sha256"] == hi_config.digest(hi_config.model_contract(spec["model"]))
        inner["model_contract_sha256"] = graph["model_contract_sha256"]
    files += [graph_file, reference_file]
    files += [F.path(graph["files"][f"orig_{stream}"]) for stream in (0, 1)]
    inner.update(spec)
    inner.update(source_root=str(ROOT), graph_record=F.relative(graph_file), graph_sha256=D.sha(graph_file),
        graph_files_sha256={str(stream): D.sha(F.path(graph["files"][f"orig_{stream}"])) for stream in (0, 1)},
        condition=graph.get("spec", {}).get("condition", "D"),
        workload_version="W1" if spec["scope"] == "S" else hi_config.WORKLOAD_VERSION,
        source_protocol="unchunked_saved_graph_before_backend")
    phase_taus = [dict(phase=phase, tau_ns=int(round(reference["KV_start"]+
                      phase*(reference["KV_end"]-reference["KV_start"])))) for phase in PHASES]
    manifest = dict(context={k:spec[k] for k in ("scope", "model", "hardware_profile", "mapping_state")},
        files=[dict(path=F.relative(p), sha256=D.sha(p)) for p in files],
        reference_record=F.relative(reference_file), reference_sha256=D.sha(reference_file),
        release_definition="pre-existing common healthy A1/original reference defines the experiment release times",
        reference_KV_start_ns=reference["KV_start"], reference_KV_end_ns=reference["KV_end"],
        phase_taus=phase_taus, kv_tag_source=F.relative(graph_file), kv_tags=graph["kv_tags"],
        kv_identification=graph["kv"], platform_sha256=graph["platform_sha256"],
        workload_version=inner["workload_version"],
        graph_record=F.relative(graph_file), graph_sha256=D.sha(graph_file),
        graph_files_sha256=inner["graph_files_sha256"],
        code_sha256={F.relative(ROOT/p):D.sha(ROOT/p) for p in
            ("e2e/dstu_supplement_gate.py", "e2e/post_fg_bb.py", "cpcm/cpcm.py",
             "e2e/post_fg_dstu.py", "e2e/dstu_supplement_hi.py", "e2e/hi_closeout_s.py",
             "e2e/f_config.py", "e2e/hi_config.py")})
    manifest["identity_sha256"] = digest(manifest)
    return inner, graph, phase_taus, manifest


def decide(shared_directed_resources, predicted_KV_end_ns, tau_ns):
    """The deployment rule has no phase label, seed or backend metric input."""
    return "S256" if shared_directed_resources > 0 and tau_ns < predicted_KV_end_ns else "BB0"


def predict(streams, hp, kv_tags, phase_taus, input_manifest, validate=False):
    """Evaluate one unchunked P schedule and freeze choices for the supplied taus."""
    import post_fg_bb as fast
    import cpcm
    started = time.perf_counter()
    p_events = streams[0]
    before = {str(s):D.execution_order_digest(events) for s, events in streams.items()}
    resources = {s:{edge for event in events.values() if event.event_type == "communication"
                     for edge in D.resource_set(event)} for s, events in streams.items()}
    shared = resources[0] & resources[1]
    tags = [tuple(tag) for tag in kv_tags]
    assert len(tags) == len(set(tags)) > 0
    assert all(tag in p_events and p_events[tag].event_type == "communication" for tag in tags)
    graph_started = time.perf_counter()
    compiled = cpcm.compile_bb(p_events, hp,
        path_lists={tag:[list(branch) for branch in event.path_list] for tag,event in p_events.items()
                    if event.event_type == "communication"},
        deps={tag:set(event.dependency_set) for tag,event in p_events.items()})
    graph_seconds = time.perf_counter()-graph_started
    forward_started = time.perf_counter()
    schedule = fast.simulate(compiled, "bb")
    forward_seconds = time.perf_counter()-forward_started
    kv_events, activities = [], []
    for tag in tags:
        event_id = compiled.id_of[tag]
        ids = schedule.ev_acts[event_id]
        assert ids and schedule.a_f[schedule.term[event_id]] == schedule.F[event_id]
        event = p_events[tag]
        kv_events.append(dict(tag=tag, event_id=event_id, name=event.event_name,
            source=event.source_location, targets=sorted(event.target_location), bytes=int(event.comm_bytes),
            first_start_ns=schedule.S[event_id], last_issue_ns=schedule.S_last[event_id],
            end_ns=schedule.F[event_id], terminal_activity=schedule.term[event_id],
            latest_activity_end_ns=max(schedule.a_f[index] for index in ids), activity_ids=ids))
        for index in ids:
            resource_id = schedule.a_res[index]
            activities.append(dict(activity_id=index, tag=tag, resource_id=resource_id,
                resource=compiled.res_name[resource_id], start_ns=schedule.a_s[index], end_ns=schedule.a_f[index],
                preceding_activity=schedule.a_par[index]))
    predicted_start = min(item["first_start_ns"] for item in kv_events)
    predicted_end = max(item["end_ns"] for item in kv_events)
    assert before == {str(s):D.execution_order_digest(events) for s, events in streams.items()}
    prediction = dict(predictor="post_fg_bb deterministic bb CPCM on unchunked P alone at release 0",
        predicted_P_end_ns=schedule.T, predicted_KV_start_ns=predicted_start, predicted_KV_end_ns=predicted_end,
        shared_resource_count=len(shared), shared_directed_resources=sorted(shared),
        P_resource_count=len(resources[0]), D_resource_count=len(resources[1]),
        kv_events=kv_events, kv_activities=activities, P_events=compiled.n, P_activities=len(schedule.a_s),
        P_resources=compiled.nres, graph_build_seconds=graph_seconds, cpcm_forward_seconds=forward_seconds,
        fast_forward_parts=fast.TIMING[-1], predictor_seconds=time.perf_counter()-started,
        schedule_sha256=digest({key:getattr(schedule,key) for key in
            ("r", "S", "S_last", "F", "term", "a_ev", "a_res", "a_s", "a_f", "a_par")}),
        source_graph_iteration_sha256=before, platform_sha256=F.platform_signature(hp))
    decisions = [dict(phase=item["phase"], tau_ns=item["tau_ns"],
        shared_resource_count=len(shared), predicted_KV_end_ns=predicted_end,
        chosen_transport=decide(len(shared),predicted_end,item["tau_ns"]), rule_version=RULE_VERSION,
        input_identity_sha256=input_manifest["identity_sha256"]) for item in phase_taus]
    checked = None
    if validate:
        validation_start = time.perf_counter()
        reference_schedule = fast.ORIGINAL(compiled, "bb")
        fields = ("r", "S", "S_last", "F", "term", "a_ev", "a_res", "a_s", "a_f", "a_par", "T", "ev_acts", "res_seq")
        assert all(getattr(schedule,key) == getattr(reference_schedule,key) for key in fields)
        checked = dict(status="passed", original_bb_schedule_fields=list(fields),
                       seconds=time.perf_counter()-validation_start, CPCM_calls=1)
        # Exercise the exact inequalities without a phase-specific branch.
        assert decide(0,predicted_end,0) == "BB0"
        assert decide(1,predicted_end,predicted_end) == "BB0"
        assert decide(1,predicted_end,predicted_end-1) == "S256"
    return dict(prediction=prediction, decisions=decisions, validation=checked,
                decision_sha256=digest(decisions), cpcm_prediction_calls=1,
                cpcm_validation_calls=1 if validate else 0, cpcm_calls=2 if validate else 1, backend_calls=0)


def child(spec, out, validate=False):
    import dstu_supplement_run as run
    import e2e_pilot_peak
    began = time.perf_counter()
    inner, graph, taus, inputs = input_context(spec)
    if "input_identity_sha256" in spec:
        assert inputs["identity_sha256"] == spec["input_identity_sha256"]
    with run.adapter(spec):
        _, hp, loaded_graph, streams = D.graph_inputs(inner, out)
        assert loaded_graph["kv_tags"] == graph["kv_tags"]
        result = predict(streams, hp, graph["kv_tags"], taus, inputs, validate)
    result.update(status="ok", job_id=point_name(spec)+"__gate", spec=spec,
        **{key:spec[key] for key in ("scope", "model", "hardware_profile", "mapping_state")},
        rule_version=RULE_VERSION, rule=RULE_TEXT, input_manifest=inputs,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-began, peak_rss_gb=e2e_pilot_peak.peak()/2**30,
        result_use="freeze before supplement backend jobs; selected-arm outcomes joined later")
    filename = F.result_path(out, result["job_id"])
    atomic_json(filename,result)
    print(json.dumps(dict(job_id=result["job_id"],status="ok",wall_seconds=result["wall_seconds"],
        peak_rss_gb=result["peak_rss_gb"],chosen=[row["chosen_transport"] for row in result["decisions"]],
        backend_calls=0,cpcm_prediction_calls=1,cpcm_validation_calls=result["cpcm_validation_calls"])),flush=True)
    return result


def plan(destination):
    jobs = []
    for base in contexts():
        _, _, _, manifest = input_context(base)
        spec = dict(base,kind="gate",module="dstu_supplement_gate",job_id=point_name(base)+"__gate",
                    rule_version=RULE_VERSION,input_identity_sha256=manifest["identity_sha256"])
        jobs.append(dict(id=spec["job_id"],kind="gate",spec=spec,deps=[],result="results/"+spec["job_id"]+".json",
            peak_rss_gb=1.8,estimated_seconds=30,
            argv=["{python}","-u","-B","e2e/dstu_supplement_gate.py","child","--spec","{spec}","--out","{out}"]))
    atomic_json(destination,dict(rule_version=RULE_VERSION,rule=RULE_TEXT,jobs=jobs,
        predictions=12,decisions=36,backend_calls=0,execution_order="all predictions freeze before any supplement backend job"))
    return jobs


def collect(out):
    rows, input_files, timings = [], [], []
    for base in contexts():
        filename = F.result_path(out,point_name(base)+"__gate")
        value = F.read_json(filename)
        assert value["status"] == "ok" and value["rule_version"] == RULE_VERSION
        assert value["decision_sha256"] == digest(value["decisions"])
        assert value["backend_calls"] == 0
        for row in value["decisions"]:
            assert row["chosen_transport"] == decide(row["shared_resource_count"],row["predicted_KV_end_ns"],row["tau_ns"])
            rows.append(dict(**base,**row,source=F.relative(filename),source_sha256=D.sha(filename),
                             frozen_at_utc=value["frozen_at_utc"]))
        input_files.append(dict(path=F.relative(filename),sha256=D.sha(filename),
                                input_identity_sha256=value["input_manifest"]["identity_sha256"]))
        timings.append(dict(**base,wall_seconds=value["wall_seconds"],peak_rss_gb=value["peak_rss_gb"],
            prediction_seconds=value["prediction"]["predictor_seconds"],
            cpcm_forward_seconds=value["prediction"]["cpcm_forward_seconds"],
            P_events=value["prediction"]["P_events"],P_activities=value["prediction"]["P_activities"]))
    assert len(rows)==36
    evidence=dict(status="passed",rule_version=RULE_VERSION,rule=RULE_TEXT,
        frozen_decisions=rows,decision_sha256=digest(rows),input_files=input_files,
        counts=dict(Counter(row["chosen_transport"] for row in rows)),predictions=12,backend_calls=0,timings=timings)
    atomic_json(BASE/"evidence/gate_decisions.json",evidence)
    import hi_closeout_p0 as table
    for filename,values in (("gate_decisions.csv",rows),("gate_prediction_timing.csv",timings)):
        final=BASE/"tables"/filename
        temporary=final.with_suffix(final.suffix+".part")
        table.csv_write(temporary,values)
        temporary.replace(final)
    print(json.dumps(dict(status="passed",predictions=12,decisions=36,counts=evidence["counts"])))
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","child","smoke","collect"))
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--manifest",type=Path,default=BASE/"manifests/gate.json")
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--context",default="S_wsc_A3hEIR")
    args=parser.parse_args()
    if args.command=="plan":
        print(json.dumps(dict(jobs=len(plan(args.manifest)),manifest=str(args.manifest))))
    elif args.command=="collect":
        collect(args.out)
    else:
        spec=F.read_json(args.spec) if args.spec else next(x for x in contexts() if point_name(x)==args.context)
        child(spec,args.out,validate=args.command=="smoke")


if __name__=="__main__":
    main()
