"""F/G selection, physical faults and frozen EIR stages.

One command executes one job. Historical EIR code is wrapped in this process.
"""
import argparse
import collections
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "e2e"
SEEDS = (123, 1, 2, 3, 4)
STEPS = (30, 100, 300)
OPS = tuple(f"{ph}_{op}" for ph in ("pf", "dc") for op in ("ln", "mha", "proj", "ffn"))
EMPTY_FAULT = {"failed_nodes": [], "failed_links": [], "failed_pes": []}


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def relative(value):
    return Path(value).resolve().relative_to(ROOT).as_posix()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [plain(x) for x in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def read_json(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def write_json(value, record):
    from e_run import write_json as write
    write(path(value), plain(record))


def read_pickle(value):
    with gzip.open(path(value), "rb") as f:
        return pickle.load(f)


def write_pickle(value, record):
    p = path(value)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    with gzip.open(tmp, "wb") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)


def job_id(spec):
    return spec.get("job_id", spec.get("candidate_id"))


def raw_path(out, name):
    return Path(out) / "raw" / (name + ".pkl.gz")


def result_path(out, name):
    return Path(out) / "results" / (name + ".json")


def op_spec(spec, op):
    ph, name = op.split("_", 1)
    return dict(spec, phase="prefill" if ph == "pf" else "decode", op=name)


def selected_source(spec, out):
    sel = read_json(Path(out) / "selection" / (spec["config_id"] + ".json"))
    return sel["operators"][spec["phase_op"]][spec["start_arm"]]


def choose_candidate(rows, out):
    assert len(rows) == 14, len(rows)
    mean = lambda r: statistics.mean(x["cycles"] for x in r["backend"])
    winner = min(rows, key=lambda r: (mean(r), r["candidate_order"]))
    rand_pick = min(rows, key=lambda r: (r["T_hat_rand_mean"], r["candidate_order"]))
    return {
        "candidate_id": winner["candidate_id"], "record": relative(result_path(out, winner["candidate_id"])),
        "raw": relative(raw_path(out, winner["candidate_id"])),
        "snapshot": relative(Path(out) / "snapshots" / (winner["candidate_id"] + ".pkl.gz")),
        "backend_mean": mean(winner), "candidate_median": statistics.median(mean(r) for r in rows),
        "random_CPCM_pick": rand_pick["candidate_id"], "random_CPCM_pick_backend_mean": mean(rand_pick),
    }


def fault_anchor_candidates(spec):
    from fg_manifest import candidate_specs
    return [s["candidate_id"] for s in candidate_specs(
        spec, "healthy", methods=("A3h",), phase_ops=[("decode", "mha")])]


def fault_anchor_source(spec, out):
    rows = [read_json(result_path(out, name)) for name in fault_anchor_candidates(spec)]
    return choose_candidate(rows, out)


def select_candidates(spec, out):
    groups = collections.defaultdict(list)
    for name in spec["candidate_ids"]:
        r = read_json(result_path(out, name))
        groups[(r["phase"], r["op"], r["method"])].append(r)
    selected = {"config_id": spec["config_id"], "selection": "backend_mean", "operators": {}}
    for (phase, op, arm), rows in groups.items():
        assert len(rows) == 14, (phase, op, arm, len(rows))
        key = ("pf" if phase == "prefill" else "dc") + "_" + op
        selected["operators"].setdefault(key, {})[arm] = choose_candidate(rows, out)
    write_json(Path(out) / "selection" / (spec["config_id"] + ".json"), selected)
    return selected


def load_healthy(spec, source, out):
    import f_config
    import e_run
    record = read_json(source["record"])
    source_spec = dict(record)
    source_spec.update({k: spec[k] for k in ("hardware_profile", "shape", "split_mode", "d2d_scale") if k in spec})
    pilot, c, cfg = f_config.prepare(source_spec, out)
    snap = e_run.load_snapshot(path(source["snapshot"]), c, cfg)
    raw = read_pickle(source["raw"])
    ctx = raw["meta"]["ctx"]
    import numpy as np
    import add_communication
    before = random.getstate(), np.random.get_state()
    random.seed(c.BCAST_SEED)
    np.random.seed(c.BCAST_SEED)
    beha, data, ev, hp = snap
    add_communication.add_broadcast(data_tags=ctx["broadcast_datatags"],
        ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]], beha_dict=beha,
        data_dict=data, hardware_platform=hp, event_dict=ev, dijkstra_routing=ctx["dijkstra_routing"],
        alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])
    random.setstate(before[0]); np.random.set_state(before[1])
    return pilot, c, snap, ctx, record


def select_faults(spec, out):
    source = fault_anchor_source(spec, out)
    pilot, c, snap, ctx, record = load_healthy(op_spec(spec, "dc_mha"), source, out)
    import numpy as np
    import event_driver
    ev, hp = copy.deepcopy(snap[2:])
    pre = c.e3._e2()._snapshot_pre(ev)
    random.seed(1); np.random.seed(1)
    triple = event_driver.event_driver(events_dict=ev, hardware_platform=hp)
    cap = c.e3._e2()._extract(ev, hp, pre, triple, "seed1")
    # C/E3 F1's busy-time rule, over all dies in this new platform.
    busy = collections.Counter()
    for (_, _, loc), rec in cap["devices"].items():
        busy[tuple(loc[:-1])] += sum(f-s for s, f, _ in rec)
    noncorner = sorted({tuple(loc[:-1]) for loc in hp.tensorcore_dict if loc[1] not in (0, 3, 12, 15)})
    node = min(noncorner, key=lambda n: (-busy[n], n))
    f1 = dict(EMPTY_FAULT, failed_nodes=[node], kind="noncorner core and router",
              rule="C/E3 F1: largest tensor+vector busy time; tie smallest (die,core)", compute_busy=busy[node])
    f6 = dict(EMPTY_FAULT, failed_pes=[node], kind="PE only", rule="C/E3 F6: same core as MF1, router retained")
    f8 = c.e3.choose_fault({"final": cap}, "F8")
    f8.setdefault("failed_pes", [])
    f4 = dict(EMPTY_FAULT, failed_links=[[[0,11],[1,8]],[[1,8],[0,11]]], kind="D2D bidirectional physical connection",
              rule="C PF4 fixed physical link, die 0 core 11 to die 1 core 8, both directions")
    faults = {"PF4": f4, "MF1": f1, "MF6": f6, "MF8": f8}
    n_dies = len({loc[0] for loc in hp.tensorcore_dict})
    for fid, f in faults.items():
        f["anchor_operator"] = "dc_mha"
        f["anchor_candidate"] = source["candidate_id"]
        f["backend_seed"] = 1
        f["failed_compute_group_fraction"] = 1/(n_dies*16) if fid in ("MF1", "MF6") else 0
        f["failed_compute_die_fraction"] = 1/16 if fid in ("MF1", "MF6") else 0
        f["failed_D2D_pair_fraction"] = 0.5 if fid == "PF4" else 0
        f["failed_DDR_port_fraction_die"] = 1/8 if fid == "MF8" else 0
        dead_nodes={tuple(n) for n in f["failed_nodes"]}
        dead_links={(tuple(a),tuple(b)) for a,b in f["failed_links"]}
        totals=collections.Counter();removed=collections.Counter()
        for (a,b),link in hp.links_dict.items():
            totals[link.link_type]+=link.bandwidth
            if a in dead_nodes or b in dead_nodes or (a,b) in dead_links:
                removed[link.link_type]+=link.bandwidth
        f["failed_bandwidth_fraction_by_type"]={kind:removed[kind]/total for kind,total in totals.items()}
    result = {"config_id":spec["config_id"], "faults":faults, "anchor":source, "anchor_backend":triple}
    write_json(Path(out)/"faults"/(spec["config_id"]+".json"), result)
    return result


def platform_signature(hp):
    links = sorted((str(k), l.link_type, l.bandwidth, l.latency, l.timeunit) for k,l in hp.links_dict.items())
    mods = sorted((t,str(loc)) for t,ms in hp.modules_dict.items() for loc in ms)
    tensor = sorted((str(loc),json.dumps(m.tensorcore_cfg,sort_keys=True),m.utilization_factor)
                    for loc,m in hp.tensorcore_dict.items())
    vector = sorted((str(loc),json.dumps(m.vectorunit_cfg,sort_keys=True))
                    for loc,m in hp.vectorunit_dict.items())
    return hashlib.sha256(repr((links,mods,hp.frequency,tensor,vector)).encode()).hexdigest()


def export_engine(eng, filename, spec, T_ref):
    state = eng.a.state
    ev = eng.bc.graph(state[5],state[6],state[7],eng.hp)
    T = int(eng.t_eval())
    assert T == int(T_ref), (T,T_ref)
    write_pickle(filename, {"spec":spec,"events_blob":pickle.dumps(ev,pickle.HIGHEST_PROTOCOL),
        "ctx":eng.ctx,"T":T,"platform_sha256":platform_signature(eng.hp),"n_events":len(ev)})
    return relative(filename)


def install_broadcast_seed(spec):
    """Use C's finalized non-chaining broadcast-cache rule in this F/G process."""
    import c_eir
    # C_SUMMARY: an empty producer/bytes bucket stays uncached; BcCache routes
    # that split from its actual producer. Other producers keep their own routes.
    c_eir.FALLBACK_OK.add((spec["start"], spec["op"], spec.get("fault")))
    c_eir.FALLBACK_LOG.clear()
    c_eir.install_seed_fallback(spec)


def engine_job(spec, out):
    for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    source = selected_source(spec,out)
    source_record = read_json(source["record"])
    import f_config
    prepare_spec = dict(source_record, method="A1", x=0)
    pilot,c,cfg = f_config.prepare(prepare_spec,out)
    if prepare_spec["op"] == "mha":
        # C seeds all original broadcast routes, then filters W1 in BcCache.graph.
        # E search filters both paths; restore only this process's broadcast hook.
        import add_communication
        wrapper = add_communication.add_broadcast
        closure = dict(zip(wrapper.__code__.co_freevars,wrapper.__closure__))
        add_communication.add_broadcast = closure["orig"].cell_contents
    work = c.WORK
    template = f_config.task_path(prepare_spec,c,cfg)
    sys.path.insert(0,str(ROOT/"eir"))
    import eir
    c.WORK = eir.WORK = eir.e3.WORK = work
    eir.GPT = c.CONFIGS[cfg][0]
    phase_op = spec["phase_op"]
    eir.TASKS[phase_op] = template.stem
    eir.BACKEND_SEEDS = c.BACKEND_SEEDS = spec.get("backend_seeds",list(range(1,11)))
    mapping = read_pickle(source["raw"])["selected"]["e3map"]
    fault = EMPTY_FAULT if spec["kind"] == "fault_pre" else read_json(Path(out)/"faults"/(spec["config_id"]+".json"))["faults"][spec["fault_id"]]
    eir.start_mapping = lambda start,op: mapping
    eir.fault_of = lambda op,fid: fault
    action = {"fault_pre":"pre","fault_l1":"l1","fault_rep":"rep"}[spec["kind"]]
    name = job_id(spec)
    inner = dict(spec, name=name, kind=action, start=spec["start_arm"].lower(), op=phase_op,
                 fault=spec.get("fault_id"), sink=str(path(raw_path(out,name))), checks=True)
    for key in ("pre_sink","l1_sink"):
        if key in inner:
            inner[key] = str(path(inner[key]))
    if action == "rep":
        inner.update(method="L2Gsync", steps=spec["steps"], seed=spec["seed"])
    install_broadcast_seed(inner)
    original = getattr(eir,"child_"+action)
    def run(eng, spec0, sink, *args):
        print(f"[fg_fault] {name} loaded engine; evaluating {action}",flush=True)
        if action=="l1" and spec["start_arm"]=="A1":
            pre=eir.read_sink(spec0["pre_sink"])
            sink["bc_seed"]=eng.seed_broadcast()
            sink["T_pre"],sink["T_R0"]=pre["T"],eng.t_eval()
            sink["R0_snap"]=eng.snapshot()
            rec=eng.finalize()
            sink["R0"]={**rec,"migration_vs_start":eir.migration_metrics(pre["record"],rec,pre["ddr_nodes"],pre["sizes"])}
        else:
            original(eng,spec0,sink,*args)
        exports = {}
        if action == "pre":
            exports["healthy"] = export_engine(eng,Path(out)/"exports"/(name+"__healthy.pkl.gz"),spec,sink["T"])
        elif action == "l1":
            for state in (("R0",) if spec["start_arm"]=="A1" else ("R0","L1")):
                eng.load(sink[state+"_snap"])
                exports[state] = export_engine(eng,Path(out)/"exports"/(name+"__"+state+".pkl.gz"),spec,sink["T_"+state])
        else:
            exports["final"] = export_engine(eng,Path(out)/"exports"/(name+"__final.pkl.gz"),spec,sink["record"]["T"])
        sink["exports"] = exports
    setattr(eir,"child_"+action,run)
    eir.write_sink = lambda filename,data: write_pickle(filename,data)
    print(f"[fg_fault] {name} injecting {source['candidate_id']} on {spec['shape']}",flush=True)
    try:
        eir.child(inner)
    except SystemExit as err:
        if err.code not in (0,None):
            raise
    sink = read_pickle(raw_path(out,name))
    result = {"source_candidate":source["candidate_id"],"exports":sink["exports"],"sink":relative(raw_path(out,name))}
    if "bc_seed" in sink:
        result["broadcast_seed"] = sink["bc_seed"]
    if action == "pre":
        result.update(T=sink["T"],backend_mean=statistics.mean(v[0] for v in sink["record"]["backend"].values()))
        assert int(sink["T"]) == int(source_record["T_hat"]), "healthy injection changed deterministic CPCM"
    elif action == "l1":
        result.update(T_pre=sink["T_pre"],T_R0=sink["T_R0"],
                      R0_backend_mean=statistics.mean(v[0] for v in sink["R0"]["backend"].values()),
                      migration_R0=sink["R0"]["migration_vs_start"],L1_evaluated=spec["start_arm"]=="A3h")
        if spec["start_arm"]=="A3h":
            result.update(T_L1=sink["T_L1"],L1_backend_mean=statistics.mean(v[0] for v in sink["L1"]["backend"].values()),
                          migration=sink["L1"]["migration_vs_start"])
    else:
        result.update(T=sink["record"]["T"],backend_mean=statistics.mean(v[0] for v in sink["record"]["backend"].values()),
                      steps=spec["steps"],seed=spec["seed"],timing=sink["timing"],migration=sink["record"]["migration_vs_start"])
    return result


def stop_job(spec,out):
    if spec["steps"] == 0:
        base = read_json(result_path(out,spec["l1_job"]))
        lim = base["T_pre"]*1.02
        state = "R0" if base["T_R0"] <= lim else "L1"
        return dict(stop=base["T_"+state]<=lim,level=state,T=base["T_"+state],T_pre=base["T_pre"],
                    export=base["exports"][state],source_job=spec["l1_job"],reason="within_epsilon" if base["T_"+state]<=lim else "continue")
    previous = read_json(result_path(out,spec["previous_stop"]))
    if previous["stop"]:
        return {k:v for k,v in previous.items() if k not in ("job_id","kind","wall_seconds","spec")}
    rows = [read_json(result_path(out,name)) for name in spec["rep_jobs"]]
    T = statistics.mean(r["T"] for r in rows)
    gain = (previous["T"]-T)/previous["T"] if spec["steps"] != 30 else None
    epsilon = T <= previous["T_pre"]*1.02
    plateau = gain is not None and gain < .02
    med = statistics.median(r["backend_mean"] for r in rows)
    rep = min((r for r in rows if r["backend_mean"]==med),key=lambda r:r["seed"])
    return dict(stop=epsilon or plateau or spec["steps"]==300,level="s"+str(spec["steps"]),T=T,T_pre=previous["T_pre"],
                export=rep["exports"]["final"],source_job=rep["job_id"],seed=rep["seed"],backend_mean=rep["backend_mean"],
                gain_from_previous=gain,reason="within_epsilon" if epsilon else "tier_gain" if plateau else "budget" if spec["steps"]==300 else "continue")


def child(spec,out):
    out = path(out)
    started = time.perf_counter()
    name = job_id(spec)
    if spec["kind"] == "fault_rep":
        previous = read_json(result_path(out,spec["previous_stop"]))
        if previous["stop"]:
            result = dict(job_id=name,kind=spec["kind"],status="reused",previous_stop=spec["previous_stop"],
                          export=previous["export"],wall_seconds=time.perf_counter()-started)
            write_json(result_path(out,name),result)
            return
    if spec["kind"] == "fault_selection":
        result=select_candidates(spec,out)
    elif spec["kind"] == "fault_select":
        result=select_faults(spec,out)
    elif spec["kind"] == "fault_stop":
        result=stop_job(spec,out)
    else:
        result=engine_job(spec,out)
    result.update(job_id=name,kind=spec["kind"],status="ok",wall_seconds=time.perf_counter()-started,spec=spec)
    import e2e_pilot_peak
    result["peak_rss_gb"] = e2e_pilot_peak.peak()/2**30
    write_json(result_path(out,name),result)
    print(f"[fg_fault] {name} {result['status']} {result['wall_seconds']:.2f}s",flush=True)


def jobs(configs, search_items, out):
    from fg_manifest import config_id, PROTOCOL
    out=path(out)
    result=[]
    for config in configs:
        cid=config_id(config)
        if config.get("d2d_scale",1)!=1 or (config["hardware_profile"]=="busybarn" and config["split_mode"]!="growth"):
            continue
        base=dict(config,config_id=cid,module="fg_fault",search_steps=1000,backend_seeds=list(range(1,11)),protocol_version=PROTOCOL)
        def add(suffix,kind,deps,**extra):
            name=cid+"__"+suffix
            result.append(dict(base,job_id=name,kind=kind,depends_on=deps,**extra))
            return name
        candidates=[s["candidate_id"] for s in search_items if config_id(s)==cid and s.get("family","healthy")=="healthy" and s["method"] in ("A1","A3h")]
        selection=add("selection","fault_selection",candidates,candidate_ids=candidates)
        fault_select=add("fault_select","fault_select",[selection])
        fids=("PF4","MF1","MF6","MF8") if config["hardware_profile"]!="busybarn" else ("PF4","MF6")
        for op in OPS:
            for arm in ("A1","A3h"):
                pre=add(f"pre_{op}_{arm}","fault_pre",[selection],**op_spec({},op),phase_op=op,start_arm=arm)
                for fid in fids:
                    l1=add(f"l1_{op}_{arm}_{fid}","fault_l1",[pre,fault_select],**op_spec({},op),phase_op=op,start_arm=arm,
                           fault_id=fid,pre_sink=relative(raw_path(out,pre)))
                    if arm=="A1":
                        continue
                    stop=add(f"stop_{op}_{fid}_0","fault_stop",[l1],phase_op=op,fault_id=fid,steps=0,l1_job=l1)
                    for steps in STEPS:
                        reps=[]
                        for seed in SEEDS:
                            reps.append(add(f"rep_{op}_{fid}_s{steps}_r{seed}","fault_rep",[stop],**op_spec({},op),phase_op=op,start_arm=arm,
                                fault_id=fid,steps=steps,seed=seed,previous_stop=stop,pre_sink=relative(raw_path(out,pre)),l1_sink=relative(raw_path(out,l1))))
                        stop=add(f"stop_{op}_{fid}_{steps}","fault_stop",reps,phase_op=op,fault_id=fid,steps=steps,previous_stop=stop,rep_jobs=reps)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["child"])
    parser.add_argument("--spec",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    child(read_json(args.spec),args.out)


if __name__=="__main__":
    main()
