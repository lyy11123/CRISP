"""Model-aware H/I joint inputs: contracts, cache messages and complete D chains.

This module constructs or exports graphs. It does not run a joint backend.
"""
import argparse
from collections import Counter, deque
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import hi_config as H
from e_run import load_snapshot, signature, write_json
from fg_fault import platform_signature, write_pickle

VERSION = "hi_joint_inputs_v1"
DC = ("dc_ln", "dc_mha", "dc_proj", "dc_ffn")


def legacy_graph_functions():
    os.environ["D_MAIN"] = str(H.ROOT)
    import d_joint
    return d_joint


def decode_chain_order(model):
    contract = H.model_contract(model)
    n = contract["decode_chain"]
    return ["dc_ln"] * n["norm"] + ["dc_mha"] * n["mha"] + ["dc_proj"] * n["proj"] + ["dc_ffn"] * n["ffn"]


def cache_messages(state, ctx, spec):
    """Select cache writebacks by tensor tag and producer, including Ktranspose."""
    assert spec["op"] == "mha"
    behaviours, data, events, hp = state
    # H0/I0 establishes that native broadcast_datatags is exactly cache_list.
    cache_tags = [tuple(t) for t in ctx["broadcast_datatags"]]
    assert len(cache_tags) == 2
    assert [data[t].data_name for t in cache_tags] == ["MHA_ktrans_ofmap", "MHA_vconv_ofmap"]
    contract = H.model_contract(spec["model"])
    length = contract["sequence_length"] if spec["phase"] == "prefill" else 1
    shapes = [[contract["batch"], contract["head_dim"], length],
              [contract["batch"], length, contract["head_dim"]]]
    ddr_nodes = {tuple(t[:-1]) for t in hp.ddr_dict}
    rows, selected = [], set()
    for tensor_tag, expected_shape in zip(cache_tags, shapes):
        tensor = data[tensor_tag]
        assert tensor.data_shape == expected_shape
        producers = tensor.generated_tag_splitted_dict
        expected = {p: Counter(int(math.prod(end - start + 1 for start, end in region) * tensor.data_bytes)
                               for region in regions) for p, regions in producers.items()}
        actual = {p: Counter() for p in producers}
        messages = []
        for tag, event in events.items():
            if event.event_type != "communication" or not isinstance(event.target_location, set):
                continue
            owner = (0,) + tag[1:-1]
            if owner not in producers or not event.target_location <= ddr_nodes or event.issue_set:
                continue
            assert owner in event.dependency_set
            actual[owner][int(event.comm_bytes)] += 1
            messages.append(dict(tag=tag, producer=owner, producer_name=behaviours[owner].beha_name,
                                 source=event.source_location, payload_bytes=int(event.comm_bytes),
                                 target_ports=len(event.target_location)))
            selected.add(tag)
        assert actual == expected, (tensor_tag, actual, expected)
        payload = sum(r["payload_bytes"] for r in messages)
        assert payload == tensor.data_size
        rows.append(dict(tensor_tag=tensor_tag, tensor_name=tensor.data_name, shape=tensor.data_shape,
                         logical_bytes=tensor.data_size, messages=messages, payload_bytes=payload,
                         ddr_copy_bytes=sum(r["payload_bytes"] * r["target_ports"] for r in messages)))
    return selected, dict(tensors=rows, logical_bytes=sum(r["logical_bytes"] for r in rows),
                          payload_bytes=sum(r["payload_bytes"] for r in rows),
                          messages=sum(len(r["messages"]) for r in rows),
                          ddr_copy_bytes=sum(r["ddr_copy_bytes"] for r in rows),
                          identification="cache_list equals broadcast_datatags, tensor tag, producer and exact region-byte multiset")


def export_graph(spec, state, ctx, source_kind):
    """Create an input bundle from an already broadcast-complete operator graph."""
    behaviours, data, events, hp = state
    contract = H.model_contract(spec["model"])
    assert data[(0, 0)].data_shape[0] == contract["batch"]
    rms = [b for b in behaviours.values() if b.beha_name.startswith("LN_rmsnorm")]
    if spec["op"] == "ffn":
        assert bool(rms) == (spec["phase"] == "decode")
    tags = set(events)
    assert all((e.dependency_set | e.issue_set) <= tags for e in events.values())
    kv_tags, kv = cache_messages(state, ctx, spec) if spec["op"] == "mha" else (set(), None)
    return dict(protocol=VERSION, model=spec["model"], phase=spec["phase"], op=spec["op"],
                model_contract_sha256=H.digest(contract), graph_order_version=H.GRAPH_ORDER_VERSION,
                source_kind=source_kind, events_blob=pickle.dumps(events, pickle.HIGHEST_PROTOCOL),
                ctx=ctx, platform_sha256=platform_signature(hp), workload_signature=signature(state),
                kv_tags=sorted(kv_tags), kv=kv)


def export_candidate(record_path, raw_path, snapshot_path, output):
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    assert record["status"] == "ok"
    assert record["model_contract_sha256"] == H.digest(H.model_contract(record["model"]))
    _, c, cfg = H.prepare(record, Path(output).parent)
    state = load_snapshot(snapshot_path, c, cfg)
    assert signature(state) == record["workload_signature"]
    with gzip.open(raw_path, "rb") as handle:
        ctx = pickle.load(handle)["meta"]["ctx"]
    import add_communication
    import numpy as np
    random.seed(c.BCAST_SEED)
    np.random.seed(c.BCAST_SEED)
    add_communication.add_broadcast(data_tags=ctx["broadcast_datatags"],
        ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]],
        beha_dict=state[0], data_dict=state[1], event_dict=state[2], hardware_platform=state[3],
        dijkstra_routing=ctx["dijkstra_routing"], alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])
    assert int(c.cpcm_sim(state[2], state[3])[1].T) == record["T_hat"]
    bundle = export_graph(record, state, ctx, "formal_candidate")
    bundle.update(candidate_id=record["candidate_id"],
                  source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in (record_path, raw_path, snapshot_path)})
    write_pickle(output, bundle)
    return bundle


def chain_graph(model, exports, die_map, sid=1):
    d = legacy_graph_functions()
    contract_sha = H.digest(H.model_contract(model))
    for op in DC:
        row = exports[op]
        assert row["model"] == model and row["phase"] == "decode" and row["op"] == op[3:]
        assert row["model_contract_sha256"] == contract_sha
    order = decode_chain_order(model)
    graph, previous, added_edges = {}, None, 0
    for index, op in enumerate(order):
        events = d.retag(d.remap_events(pickle.loads(exports[op]["events_blob"]), die_map), sid, index)
        sources = [tag for tag, event in events.items() if not event.dependency_set]
        sinks = [tag for tag, event in events.items() if not event.issue_set]
        if previous is not None:
            for tag in sources:
                events[tag].dependency_set.update(previous)
            for tag in previous:
                graph[tag].issue_set.update(sources)
            added_edges += len(sources) * len(previous)
        graph.update(events)
        previous = sinks
    count = {tag: len(event.dependency_set) for tag, event in graph.items()}
    queue = deque(tag for tag, degree in count.items() if degree == 0)
    visited = 0
    while queue:
        tag = queue.popleft()
        visited += 1
        for following in graph[tag].issue_set:
            count[following] -= 1
            if count[following] == 0:
                queue.append(following)
    assert visited == len(graph)
    return graph, dict(order=order, mha_groups=order.count("dc_mha"), external_norms=order.count("dc_ln"),
                       ffn_embedded_norms=1, stages=len(order), barrier_edges=added_edges,
                       events=len(graph), acyclic=True)


def redirect_kv(events, hp, tags, target_dies, ctx, bcast_seed=0):
    import numpy as np
    targets = {loc[:-1] for loc in hp.ddr_dict if loc[0] in target_dies}
    saved = random.getstate(), np.random.get_state()
    random.seed(bcast_seed)
    np.random.seed(bcast_seed)
    payload = link_bytes = copy_bytes = 0
    crossings = set()
    for tag in sorted(tags):
        event = events[tag]
        event.target_location = set(targets)
        event.paths = hp.record_dijkstra_multicast_path(
            comm_pairs=[[tag, event.source_location, set(targets), event.comm_bytes]],
            alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])[2][tag]
        event.path_list = event.get_paths(event.paths)
        links = {link for branch in event.path_list for link in branch}
        assert {branch[-1][1] for branch in event.path_list if branch} == targets
        assert links <= set(hp.links_dict)
        payload += int(event.comm_bytes)
        link_bytes += int(event.comm_bytes) * len(links)
        copy_bytes += int(event.comm_bytes) * len(targets)
        crossings.update(link for link in links if (link[0][0] in target_dies) != (link[1][0] in target_dies))
    random.setstate(saved[0])
    np.random.set_state(saved[1])
    return dict(messages=len(tags), payload_bytes=payload, routed_unique_link_bytes=link_bytes,
                ddr_copy_bytes=copy_bytes, target_ports=len(targets), crossing_links=len(crossings))


def initial_bundle(model, phase, op):
    """H0/I0 interface validation input; no search and no backend call."""
    import hi_model_check
    spec = dict(model=model, phase=phase, op=op, method="A3h", g=0, x=123,
                hardware_profile="busybarn", shape="2x2")
    _, c, _ = H.prepare(spec, H.HERE / "hi/out/joint_interfaces")
    path = H.generate(spec)
    ctx = {"__file__": str(path), "__name__": "__main__"}
    source = hi_model_check.source_prefix(path.read_text(encoding="utf-8"), mapped=True)
    exec(compile(source, str(path), "exec"), ctx)
    hi_model_check.semantic(ctx, spec)
    state = ctx["beha_dict"], ctx["data_dict"], ctx["initial_event_dict"], ctx["hardware_platform"]
    keys = ("broadcast_datatags", "layers_regions", "initialization_layer_tag", "dijkstra_routing", "alpha", "beta", "gamma")
    return export_graph(spec, state, {key: ctx[key] for key in keys}, "initial_graph_interface_check"), state[3]


def probe(model, output, candidate_root=None):
    started = time.perf_counter()
    from fg_joint import maps
    d = legacy_graph_functions()
    exports = {}
    for key, phase, op in [(key, "decode", key[3:]) for key in DC] + [("P", "prefill", "mha")]:
        name = f"H_llama_busybarn_{phase}_{op}_A3h_g0_s123"
        record = Path(candidate_root) / "results" / (name + ".json") if candidate_root else None
        if model == "llama" and record is not None and record.exists():
            export_path = H.HERE / "hi/out/joint_interfaces" / (name + ".pkl.gz")
            exports[key] = export_candidate(record, Path(candidate_root) / "raw" / (name + ".pkl.gz"),
                                            Path(candidate_root) / "snapshots" / (name + ".pkl.gz"), export_path)
        else:
            exports[key], local_hp = initial_bundle(model, phase, op)
    spec = dict(model=model, phase="prefill", op="mha", hardware_profile="busybarn", shape="2x2")
    import read_cfg
    import WAMIS_HD
    joint_cfg = H.config_path(spec).with_name("config_hi_joint_2x4.cfg")
    H.f_config.atomic_text(joint_cfg, H.f_config.config_text(spec, shape_override="2x4"))
    hp = WAMIS_HD.wamis_hdc(read_cfg.cfg_to_dict(str(joint_cfg)))
    mapping = maps("2x2")
    group_checks = {side: d.check_group(local_hp, hp, mapping[side]) for side in ("L", "R")}
    assert all(row["links_bad"] == 0 and row["modules_equal"] for row in group_checks.values())
    chain, chain_info = chain_graph(model, exports, mapping["R"])
    p = d.retag(d.remap_events(pickle.loads(exports["P"]["events_blob"]), mapping["L"]), 0)
    kv_tags = {(tag[0], 10, 0) + tag[3:] for tag in exports["P"]["kv_tags"]}
    kv = redirect_kv(p, hp, kv_tags, set(mapping["R"].values()), exports["P"]["ctx"])
    assert kv["payload_bytes"] == exports["P"]["kv"]["logical_bytes"]
    assert d.check_graph(p, hp) == d.check_graph(chain, hp) == dict(missing_compute=0, missing_links=0)
    kv_links = d.links_of({tag: p[tag] for tag in kv_tags})
    d_links = d.links_of(chain)
    non_kv_links = d.links_of({tag: event for tag, event in p.items() if tag not in kv_tags})
    result = dict(model=model, protocol=VERSION, status="ok", model_contract_sha256=H.digest(H.model_contract(model)),
                  source_kind="initial_graph_interface_check", new_searches=0, new_backend_simulations=0,
                  source_by_operator={key: row["source_kind"] for key, row in exports.items()},
                  chain=chain_info, cache_tensors=exports["P"]["kv"], kv_transfer=kv,
                  group_checks=group_checks, shared=dict(P_D_links=len(non_kv_links & d_links),
                                                        KV_D_links=len(kv_links & d_links)),
                  cross_stream_dependency_edges=0,
                  stream_meaning="independent P and D requests sharing resources; P cache completion does not release this D chain",
                  wall_seconds=time.perf_counter() - started)
    write_json(output, result)
    print(json.dumps(dict(model=model, status="ok", chain_stages=chain_info["stages"], kv_bytes=kv["payload_bytes"],
                          wall_seconds=result["wall_seconds"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("probe")
    check.add_argument("--model", choices=list(H.MODELS), required=True)
    check.add_argument("--out", type=Path, required=True)
    check.add_argument("--candidate-root", type=Path)
    export = sub.add_parser("export")
    for name in ("record", "raw", "snapshot", "out"):
        export.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "probe":
        probe(args.model, args.out, args.candidate_root)
    else:
        export_candidate(args.record, args.raw, args.snapshot, args.out)


if __name__ == "__main__":
    main()
