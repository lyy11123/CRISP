"""H0/I0 logical contracts and initial mapped graphs, without mapping search."""
import os
for _variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import ast
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time

import hi_config as H


def write(path, value):
    H.f_config.atomic_text(Path(path), json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def shape_work(behaviours):
    tensor = [b for b in behaviours.values() if b.beha_type in ("matmul","matmac")]
    flops = sum(2*math.prod(b.beha_datashape[0])*b.beha_datashape[1][-1] for b in tensor)
    vector = Counter()
    for b in behaviours.values():
        if b.beha_type not in ("matmul","matmac"):
            vector[b.beha_type] += math.prod(b.beha_datashape[0])
    return dict(tensor_effective_flops=flops, tensor_blocks=len(tensor), vector_elements_by_type=dict(vector))


def source_prefix(source, mapped):
    if not mapped:
        return source[:source.index("\nupdate_data(beha_dict=")]
    tree = ast.parse(source)
    first = next(n.lineno for n in tree.body if isinstance(n,ast.Assign)
                 and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
                 and n.value.func.id == "event_driver")
    return "\n".join(source.splitlines()[:first-1])+"\n"


def semantic(ctx, spec):
    data, behaviours = ctx["data_dict"], ctx["beha_dict"]
    p = H.MODELS[spec["model"]]
    checks = dict(batch=ctx["batch_size"] == p["batch"], hidden=ctx["hidden_states"] == p["hidden"])
    rows = [(tag,b.beha_type,b.beha_datashape,sorted(b.needed_data_split_dict),sorted(b.produced_data_split_dict))
            for tag,b in behaviours.items()]
    result = dict(behaviours=len(behaviours), tensors=len(data), checks=checks,
                  semantic_sha256=H.digest(rows), **shape_work(behaviours),
                  route=dict(dijkstra=ctx["dijkstra_routing"], LP=ctx["LP_flag"],
                             hops_only=ctx["hops_only"], ddr_random=ctx["ddr_random"]))
    if spec["op"] == "mha":
        names = Counter(d.data_name for d in data.values())
        counts = {k:names[f"MHA_{k}conv_ofmap"] for k in ("q","k","v")}
        checks["logical_heads"] = counts == dict(q=8,k=1,v=1)
        checks["output_heads"] = names["MHA_p_ofmap"] == 8
        checks["actual_q_loop"] = ctx["head_idx"] == 7 and ctx["heads"] == 8
        kv=[]
        for tag in ctx["cache_list"]:
            d=data[tag]
            kv.append(dict(tag=tag,name=d.data_name,shape=d.data_shape,bytes=d.data_size,
                           producers=[dict(tag=t,name=behaviours[t].beha_name) for t in d.generated_tag_splitted_dict]))
        checks["cache_pair"] = len(kv) == 2 and [r["name"] for r in kv] == ["MHA_ktrans_ofmap","MHA_vconv_ofmap"]
        length=2048 if spec["phase"] == "prefill" else 1
        checks["cache_shapes"] = [r["shape"] for r in kv] == [[p["batch"],p["head_dim"],length],[p["batch"],length,p["head_dim"]]]
        result.update(logical_heads=counts, kv_outputs=kv, kv_output_bytes=sum(r["bytes"] for r in kv),
                      d_chain_head_groups=p["head_groups"])
        if spec["phase"] == "decode":
            history=[]
            for name in ("MHA_qkt_weight","MHA_p_weight"):
                tensors=[(tag,d) for tag,d in data.items() if d.data_name == name]
                checks[name+"_one_shared_tensor"] = len(tensors) == 1
                tag,d=tensors[0]
                consumers=[b for b in behaviours.values() if tag in b.needed_data_split_dict]
                operations={b.beha_tag[:-1] for b in consumers}
                checks[name+"_eight_q_consumers"] = len(operations) == 8
                history.append(dict(tag=tag,name=name,shape=d.data_shape,bytes=d.data_size,
                                    consumer_head_operators=len(operations), consumer_compute_blocks=len(consumers)))
            result["historical_kv"] = history
            result["historical_kv_logical_bytes"] = sum(r["bytes"] for r in history)
    if spec["op"] == "ffn":
        rms=[b for b in behaviours.values() if b.beha_name.startswith("LN_")]
        checks["embedded_norm"] = bool(rms) == (spec["phase"] == "decode")
        result["embedded_rmsnorm_behaviours"] = len(rms)
        if spec["model"] == "llama" and spec["phase"] == "decode":
            for branch in ("mlp1","mlp2"):
                bs=[b for b in behaviours.values() if b.beha_name.startswith("MLP"+branch) and b.beha_type in ("matmul","matmac")]
                checks[branch+"_reads_norm"] = all(ctx["ln_data_tag"] in b.needed_data_split_dict for b in bs) and bool(bs)
        if spec["model"] == "qwen3moe":
            result.update(experts=ctx["expert_num"],top_k=ctx["top_k"],expert_dim=ctx["moe_ffn_dims"],
                          expert_batch_assignment="expert_idx % 16", dispatches=sum(b.beha_type=="dispatch" for b in behaviours.values()),
                          combines=sum(b.beha_type=="combine" for b in behaviours.values()))
    assert all(checks.values()), checks
    return result


def split_inventory(ctx):
    from post_fg_split import pieces_for, tensor_work
    events, behaviours = ctx["initial_event_dict"],ctx["beha_dict"]
    output=[]
    for tag,b in behaviours.items():
        e=events[tag]
        if e.comp_device != "tensorcore":
            continue
        terminal=[t for t in e.issue_set if events[t].event_type == "communication" and not events[t].issue_set]
        axes=[]
        for axis in ("N","M"):
            dt=b.beha_datatag[1 if axis=="N" else 0]
            dim=-1 if axis=="N" else 0
            regions=b.needed_data_split_dict[dt]
            width=max(r[dim][1] for r in regions)-min(r[dim][0] for r in regions)+1
            if width % 2 or terminal:
                continue
            parts=pieces_for(tag,e,b,2,axis)
            assert len(parts)==2
            assert sum(tensor_work(part["shape"])[0] for part in parts) == tensor_work(e.comp_datashape)[0]
            axes.append(axis)
        output.append(dict(tag=tag,name=e.event_name,shape=e.comp_datashape,
                           current_pe=e.comp_location[:2],legal_m2_axes=axes,
                           direct_terminal_broadcasts=len(terminal), effective_flops=tensor_work(e.comp_datashape)[0]))
    legal=[r for r in output if r["legal_m2_axes"]]
    return dict(tensor_blocks=len(output),legal_m2_blocks=len(legal),
                direct_terminal_broadcast_tensor_blocks=sum(bool(r["direct_terminal_broadcasts"]) for r in output),
                legal_tensor_flops=sum(r["effective_flops"] for r in legal),
                axes=dict(Counter(a for r in legal for a in r["legal_m2_axes"])),blocks=output,
                selection="prefilter only; choose at fixed MF6 PE after health selection, before repair")


def check(spec, out, mapped=True, fast=False):
    started=time.perf_counter()
    spec=H.normalized(spec)
    pilot,c,cfg=H.prepare(spec,Path(out).parent)
    task=H.generate(spec)
    source=task.read_text(encoding="utf-8")
    ctx={"__file__":str(task),"__name__":"__main__"}
    exec(compile(source_prefix(source,mapped),str(task),"exec"),ctx)
    built=time.perf_counter()
    row=dict(spec=spec,status="ok",stage="mapped_graph" if mapped else "logical_graph",
             source_sha256=hashlib.sha256(source.encode()).hexdigest(),
             model_contract_sha256=H.digest(H.model_contract(spec["model"])),
             graph_order_version=H.GRAPH_ORDER_VERSION,workload_version=H.WORKLOAD_VERSION,
             semantic=semantic(ctx,spec),build_seconds=built-started)
    if mapped:
        events,hp=ctx["initial_event_dict"],ctx["hardware_platform"]
        broadcast=set(events)-set(ctx["event_dict"])
        row["broadcast"]=dict(events=len(broadcast),bytes=sum(events[t].comm_bytes for t in broadcast),
                              event_names=dict(Counter(events[t].event_name.rsplit("_",1)[0] for t in broadcast)),
                              qkt_s_broadcasts=sum(any(k in events[t].event_name for k in ("qkt","s_softmax")) for t in broadcast))
        if spec["op"] == "mha":
            assert row["broadcast"]["qkt_s_broadcasts"] == 0
            assert ctx["broadcast_datatags"] == ctx["cache_list"]
            row["broadcast"]["native_qkt_s_requested"]=False
            row["broadcast"]["w1_qkt_s_removed"]=0
        comp=[e for e in events.values() if e.event_type=="computation"]
        comm=[e for e in events.values() if e.event_type=="communication"]
        tags=set(events)
        assert all((e.dependency_set|e.issue_set)<=tags for e in events.values())
        graph=c.cpcm.compile_bb(events,hp)
        row["graph"]=dict(events=len(events),computation_events=len(comp),communication_events=len(comm),
                          dependencies=len(graph.pred),hops=len(graph.hop_link),resources=graph.nres,
                          communication_bytes=sum(e.comm_bytes for e in comm),
                          all256_message_blocks=sum(math.ceil(e.comm_bytes/(256*1024)) for e in comm),
                          all256_branch_hop_blocks=sum(math.ceil(e.comm_bytes/(256*1024))*sum(map(len,e.path_list)) for e in comm),
                          tensor_service_ns=sum(int(hp.tensorcore_dict[e.comp_location].working_time(e.comp_datashape,e.comp_type,hp.frequency)) for e in comp if e.comp_device=="tensorcore"),
                          vector_service_ns=sum(int(hp.vectorunit_dict[e.comp_location].working_time(e.comp_datashape,e.comp_type,hp.frequency)) for e in comp if e.comp_device=="vectorunit"))
        if fast:
            import post_fg_bb
            row["fast_equivalence"]=post_fg_bb.compare(graph)
        if spec["phase"]=="prefill" and spec["op"] in ("ffn","proj"):
            row["split_m2"]=split_inventory(ctx)
        row["hardware"]=H.f_config.hardware_parameters(spec["hardware_profile"])
    row.update(wall_seconds=time.perf_counter()-started,peak_rss_gib=pilot.peak()/2**30)
    write(out,row)
    print(json.dumps(dict(out=str(out),status="ok",wall_seconds=row["wall_seconds"],peak_rss_gib=row["peak_rss_gib"])),flush=True)
    return row


def inventory(folder, out):
    results=[json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(folder).glob("*.json"))]
    mapped=[r for r in results if r.get("stage")=="mapped_graph"]
    rows=[]
    for r in mapped:
        s=r["spec"]
        rows.append({**{k:s[k] for k in ("model","phase","op","hardware_profile","method")},
                     **r["graph"], "behaviours":r["semantic"]["behaviours"],"tensors":r["semantic"]["tensors"],
                     "tensor_effective_flops":r["semantic"]["tensor_effective_flops"],
                     "broadcasts":r["broadcast"]["events"],"kv_output_bytes":r["semantic"].get("kv_output_bytes",0),
                     "build_seconds":r["build_seconds"],"wall_seconds":r["wall_seconds"],"peak_rss_gib":r["peak_rss_gib"]})
    if rows:
        b=io.StringIO(newline="")
        w=csv.DictWriter(b,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        H.f_config.atomic_text(Path(out),b.getvalue())
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    c=sub.add_parser("child")
    c.add_argument("--spec",required=True,type=Path)
    c.add_argument("--out",required=True,type=Path)
    c.add_argument("--logical",action="store_true")
    c.add_argument("--fast",action="store_true")
    m=sub.add_parser("inventory")
    m.add_argument("--folder",type=Path,required=True)
    m.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="child":
        check(json.loads(args.spec.read_text(encoding="utf-8")),args.out,not args.logical,args.fast)
    else:
        inventory(args.folder,args.out)


if __name__=="__main__":
    main()
