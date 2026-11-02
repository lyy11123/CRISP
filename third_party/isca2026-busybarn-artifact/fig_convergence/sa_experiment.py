"""
Unified SA experiment for bottleneck_with_brute.

Runs any operator × variant × barrier combination with configurable SA params.

Examples:
    python sa_experiment.py --operator ffn --variant busybarn
    python sa_experiment.py --operator ffn --variant busybarn --barrier
    python sa_experiment.py --operator ln  --variant gemini --steps 2000
    python sa_experiment.py --operator mha --variant busybarn --loss-ratio 1 0.1 1 0.1
    python sa_experiment.py --operator ffn --variant busybarn --hw-cfg path/to/config.cfg
    python sa_experiment.py --operator ffn --variant busybarn --hw-topology wamis_hdc_single
    python sa_experiment.py --operator ffn --variant busybarn --split-degree 1 2 1 4
"""
import os
import sys
import time
file_path = os.path.dirname(os.path.realpath(__file__))
file_real_path = os.path.realpath(__file__)
filename_with_extension = os.path.basename(file_real_path)
filename_without_extension = os.path.splitext(filename_with_extension)[0]
import argparse
import random
import pickle
from copy import deepcopy

import operator_setup
from operator_setup import (
    init_hardware, finalize_graph, build_operator,
    RESULTS_DIR, HW_CFG_PATH, TOPOLOGY_CLASSES, DEFAULT_SPLIT_DEGREES,
    MODEL_CONFIGS, set_model,
)

from add_communication import (
    build_event_v2 as build_event, add_broadcast_v2 as add_broadcast,
    reroute_dijkstra, apply_barrier_sync_all,
)
from Stream_Mapping import stream_mapping_v2 as stream_mapping
from event_driver import event_driver_v2 as event_driver
from timeline import devicedict_to_showlist


def _reset_hw_runtime(hw):
    """Reset mutable runtime state on all modules and links.

    event_driver mutates work_flag, work_endtime, work_record on each
    device/link.  This resets them so hw can be reused without deepcopy.
    """
    for dev_dict in hw.modules_dict.values():
        for dev in dev_dict.values():
            dev.work_flag = False
            dev.work_endtime = None
            dev.work_record = []
    for link in hw.links_dict.values():
        link.work_flag = False
        link.work_endtime = None
        link.work_record = []


DEFAULTS = {
    "busybarn": {
        "ffn":  dict(alpha=1,   beta=100, gamma=1,   loss_ratio=[2, 1, 1, 0.1],
                     random_threshold=0.3, lp=False, region_restricted=False,
                     hops_only=False, reroute=True),
        "ln":   dict(alpha=100, beta=1,   gamma=100, loss_ratio=[1, 1, 0, 1],
                     random_threshold=0.5, lp=False, region_restricted=False,
                     hops_only=False, reroute=True),
        "mha":  dict(alpha=1,   beta=100, gamma=1,   loss_ratio=[1, 0.1, 1, 1],
                     random_threshold=0.3, lp=False, region_restricted=False,
                     hops_only=False, reroute=True),
        "proj": dict(alpha=1,   beta=100, gamma=1,   loss_ratio=[1, 1, 1, 0.1],
                     random_threshold=0.3, lp=False, region_restricted=False,
                     hops_only=False, reroute=True),
    },
    "gemini": {
        "ffn":  dict(alpha=100, beta=1, gamma=100, loss_ratio=[1, 1, 1, 1],
                     random_threshold=0.3, lp=True, region_restricted=True,
                     hops_only=True, reroute=False),
        "ln":   dict(alpha=100, beta=1, gamma=100, loss_ratio=[1, 1, 0, 1],
                     random_threshold=0.5, lp=True, region_restricted=True,
                     hops_only=True, reroute=False),
        "mha":  dict(alpha=100, beta=1, gamma=100, loss_ratio=[1, 1, 1, 1],
                     random_threshold=0.3, lp=True, region_restricted=True,
                     hops_only=True, reroute=False),
        "proj": dict(alpha=100, beta=1, gamma=100, loss_ratio=[1, 1, 1, 1],
                     random_threshold=0.5, lp=True, region_restricted=True,
                     hops_only=True, reroute=False),
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified SA experiment: operator × variant × barrier")

    p.add_argument("--operator", required=True, choices=["ln", "mha", "proj", "ffn"])
    p.add_argument("--variant", required=True, choices=["busybarn", "gemini"])
    p.add_argument("--barrier", action="store_true", help="Apply barrier sync before final eval")

    p.add_argument("--hw-cfg", type=str, default=HW_CFG_PATH,
                   help="Path to hardware config file (default: wamis_hd_distributed.cfg)")
    p.add_argument("--hw-topology", type=str, default="wamis_hdc",
                   choices=list(TOPOLOGY_CLASSES.keys()),
                   help="Hardware topology class (default: wamis_hdc)")

    p.add_argument("--alpha", type=int, default=None)
    p.add_argument("--beta", type=int, default=None)
    p.add_argument("--gamma", type=int, default=None)
    p.add_argument("--loss-ratio", type=float, nargs=4, default=None,
                   metavar=("L1", "L2", "L3", "L4"))
    p.add_argument("--random-threshold", type=float, default=None)
    p.add_argument("--steps", type=float, default=1e3)
    p.add_argument("--t-max", type=float, default=10)
    p.add_argument("--t-min", type=float, default=1e-6)
    p.add_argument("--lp", action="store_true", default=None,
                   help="Enable LP flag (default: variant default)")
    p.add_argument("--no-lp", dest="lp", action="store_false")
    p.add_argument("--region-restricted", action="store_true", default=None)
    p.add_argument("--no-region-restricted", dest="region_restricted", action="store_false")
    p.add_argument("--hops-only", action="store_true", default=None)
    p.add_argument("--no-hops-only", dest="hops_only", action="store_false")
    p.add_argument("--reroute", action="store_true", default=None,
                   help="Post-SA dijkstra rerouting (default: variant default)")
    p.add_argument("--no-reroute", dest="reroute", action="store_false")
    p.add_argument("--greedy-flag", action="store_true", default=False)
    p.add_argument("--no-sa", action="store_true", default=False, help="Skip SA")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--model", type=str, default="qwen2.5-7b",
                   choices=list(MODEL_CONFIGS.keys()),
                   help="Model config (default: qwen2.5-7b)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size (default: 1)")
    p.add_argument("--seq-length", type=int, default=512,
                   help="Sequence length (default: 512)")
    p.add_argument("--split-degree", type=int, nargs='+', default=None,
                   help="Parallelism split degree (FFN: 4 ints, LN: 6 ints, MHA: 4 ints, Proj: 4 ints)")
    p.add_argument("--chiplets-per-layer", type=int, default=None,
                   help="Number of chiplets per layer (default: CHIPLETS_PER_LAYER from operator_setup)")
    p.add_argument("--output", type=str, default=None,
                   help="Output filename (default: auto-generated)")
    p.add_argument("--profile", action="store_true", default=False,
                   help="Collect and print per-phase wall-clock times")

    return p.parse_args()


def resolve_config(args):
    """Merge variant×operator defaults with CLI overrides."""
    defaults = DEFAULTS[args.variant][args.operator]
    cfg = {}
    for key in ["alpha", "beta", "gamma", "loss_ratio", "random_threshold",
                 "lp", "region_restricted", "hops_only", "reroute"]:
        cli_val = getattr(args, key)
        cfg[key] = cli_val if cli_val is not None else defaults[key]
    cfg["steps"] = args.steps
    cfg["t_max"] = args.t_max
    cfg["t_min"] = args.t_min
    cfg["greedy_flag"] = args.greedy_flag
    cfg["no_sa"] = args.no_sa
    cfg["barrier"] = args.barrier
    cfg["seed"] = args.seed
    return cfg


def main():
    args = parse_args()
    cfg = resolve_config(args)

    random.seed(cfg["seed"])

    set_model(args.model)
    operator_setup.BATCH_SIZE = args.batch_size
    operator_setup.SEQUENCE_LENGTH = args.seq_length

    barrier_suffix = "_barrier" if cfg["barrier"] else ""
    if args.output:
        out_name = args.output
    else:
        out_name = f"{args.operator}_distributed_{args.variant}{barrier_suffix}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    txt_path = os.path.join(RESULTS_DIR, f"{out_name}.txt")


    split_degree = tuple(args.split_degree) if args.split_degree else None

    profile = args.profile
    timings = {}

    def _tick():
        return time.perf_counter() if profile else 0

    t0 = _tick()
    hw = init_hardware(cfg_path=os.path.join(file_path, args.hw_cfg), topology=args.hw_topology)
    timings["init_hardware"] = _tick() - t0

    t0 = _tick()
    data_dict, beha_dict, broadcast_datatags = build_operator(args.operator, split_degree=split_degree)
    layers_regions = finalize_graph(
        beha_dict, data_dict, hw, greedy_flag=cfg["greedy_flag"],
        chiplets_per_layer=args.chiplets_per_layer)
    timings["build_operator"] = _tick() - t0

    t0 = _tick()
    event_dict = {}
    hops, cd, cl, tc, vu = build_event(
        beha_dict=beha_dict, data_dict=data_dict, hardware_platform=hw,
        event_dict=event_dict, dijkstra_routing=False,
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])
    timings["build_event"] = _tick() - t0

    mem_datatags = []

    saved_ll = dict(hw.link_loads_dict)
    saved_lc = dict(hw.link_loads_count)

    t0 = _tick()
    init_ed = deepcopy(event_dict)
    timings["deepcopy_event"] = _tick() - t0

    t0 = _tick()
    add_broadcast(
        data_tags=broadcast_datatags, ddr_chiplets=layers_regions[0],
        beha_dict=beha_dict, data_dict=data_dict, hardware_platform=hw,
        event_dict=init_ed, dijkstra_routing=False,
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])
    timings["add_broadcast"] = _tick() - t0

    t0 = _tick()
    init_tc = event_driver(events_dict=init_ed, hardware_platform=hw)
    timings["event_driver"] = _tick() - t0

    _reset_hw_runtime(hw)
    for k in hw.link_loads_dict:
        hw.link_loads_dict[k] = saved_ll[k]
        hw.link_loads_count[k] = saved_lc[k]

    t0 = _tick()
    if not cfg["no_sa"]:
        stream_mapping(
            hops=hops, communication_distances=cd,
            communication_loads_dict=cl, tensorcore_loads_dict=tc,
            vectorunit_loads_dict=vu, layers_regions=layers_regions,
            beha_dict=beha_dict, data_dict=data_dict,
            hardware_platform=hw, event_dict=event_dict,
            random_threshold=[cfg["random_threshold"]],
            LP_flag=cfg["lp"], region_restriced=cfg["region_restricted"],
            related=False, hops_only=cfg["hops_only"],
            loss_ratio=cfg["loss_ratio"],
            mem_enable=True, mem_threshold=[0.3],
            mem_datatags=mem_datatags, mem_random=True,
            dijkstra_routing=False,
            alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"],
            t_max=cfg["t_max"], t_min=cfg["t_min"], steps=cfg["steps"])
    timings["stream_mapping"] = _tick() - t0

    t0 = _tick()
    if cfg["reroute"]:
        hops, cd, cl = reroute_dijkstra(
            beha_dict=beha_dict, data_dict=data_dict,
            hardware_platform=hw, event_dict=event_dict,
            alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])
    timings["reroute_dijkstra"] = _tick() - t0

    t0 = _tick()
    final_dijkstra = cfg["reroute"]
    add_broadcast(
        data_tags=broadcast_datatags, ddr_chiplets=layers_regions[0],
        beha_dict=beha_dict, data_dict=data_dict, hardware_platform=hw,
        event_dict=event_dict, dijkstra_routing=final_dijkstra,
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])
    timings["add_broadcast"] += _tick() - t0

    t0 = _tick()
    if cfg["barrier"]:
        apply_barrier_sync_all(event_dict, data_dict)
    timings["barrier_sync"] = _tick() - t0

    t0 = _tick()
    time_cost = event_driver(events_dict=event_dict, hardware_platform=hw)
    timings["event_driver"] += _tick() - t0

    label = "SA + reroute_dijkstra" if cfg["reroute"] else "SA"
    print(f"{label} time cost: {time_cost}")

    t0 = _tick()
    records_lists, device_names = devicedict_to_showlist(
        hw.modules_dict["tensorcore"])
    records_lists, device_names = devicedict_to_showlist(
        hw.modules_dict["vectorunit"], records_lists, device_names)
    records_lists, device_names = devicedict_to_showlist(
        hw.links_dict, records_lists, device_names)
    pkl_path = os.path.join(RESULTS_DIR, f"{out_name}_records_lists.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump((records_lists, device_names), f)
    timings["save_records"] = _tick() - t0

    total_time = time_cost[0]



if __name__ == "__main__":
    main()
