"""
SA convergence tracking: runs SA with intermediate evaluation at regular intervals.

At every --eval-interval steps, snapshots the current best mapping and evaluates
the full pipeline (reroute + broadcast + barrier + event_driver) to get actual time cost.

Usage:
    python sa_convergence.py --operator ffn --variant busybarn --eval-interval 100 --steps 6000
    python sa_convergence.py --operator ffn --variant busybarn --eval-interval 100 --steps 6000 \
        --hw-topology wamis_hdc --hw-cfg ../src/platform/cfgs/wamis_hd_distributed.cfg
"""
import os
import sys
import argparse
import random
import time
import math
from copy import deepcopy

file_path = os.path.dirname(os.path.realpath(__file__))

from operator_setup import (
    init_hardware, finalize_graph, build_operator,
    RESULTS_DIR, HW_CFG_PATH, TOPOLOGY_CLASSES, DEFAULT_SPLIT_DEGREES,
)
from add_communication import (
    build_event_v2 as build_event, add_broadcast_v2 as add_broadcast,
    reroute_dijkstra, apply_barrier_sync_all,
)
from Stream_Mapping import StreamOptimizationV2 as StreamOptimization
from event_driver import event_driver_v2 as event_driver

import numpy as np


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


class StreamOptimizationWithEval(StreamOptimization):
    """StreamOptimization with periodic evaluation callback."""

    def __init__(self, *args, eval_callback=None, eval_interval=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_callback = eval_callback
        self.eval_interval = eval_interval

    def anneal(self):
        """Same as parent anneal() but calls eval_callback at step intervals."""
        random.seed(123)
        np.random.seed(123)

        step = 0
        self.start = time.time()

        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires Tmin > 0')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        T = self.Tmax
        E = self.energy()
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        self.best_link_loads = dict(self.hardware_platform.link_loads_dict)
        self.best_link_counts = dict(self.hardware_platform.link_loads_count)
        trials = accepts = improves = 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        while step < self.steps and not self.user_exit:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE >= 0.0:
                self.revert()
                E = prevEnergy
            else:
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
                    self.best_link_loads = dict(self.hardware_platform.link_loads_dict)
                    self.best_link_counts = dict(self.hardware_platform.link_loads_count)
            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(step, T, E, accepts / trials, improves / trials)
                    trials = accepts = improves = 0

            if (self.eval_callback and self.eval_interval > 0
                    and step % self.eval_interval == 0):
                self.eval_callback(
                    step, self.best_state, self.best_energy,
                    self.best_link_loads, self.best_link_counts)

        self.state = self.copy_state(self.best_state)
        for k in self.hardware_platform.link_loads_dict:
            self.hardware_platform.link_loads_dict[k] = self.best_link_loads[k]
            self.hardware_platform.link_loads_count[k] = self.best_link_counts[k]
        if self.save_state_on_exit:
            self.save_state()

        if self.verbose:
            print("Best energy: ", self.best_energy)
        return self.best_state, self.best_energy


def parse_args():
    p = argparse.ArgumentParser(
        description="SA convergence: SA with intermediate evaluation at regular intervals")

    p.add_argument("--operator", required=True, choices=["ln", "mha", "proj", "ffn"])
    p.add_argument("--variant", required=True, choices=["busybarn", "gemini"])
    p.add_argument("--barrier", action="store_true")

    p.add_argument("--hw-cfg", type=str, default=HW_CFG_PATH)
    p.add_argument("--hw-topology", type=str, default="wamis_hdc",
                   choices=list(TOPOLOGY_CLASSES.keys()))

    p.add_argument("--alpha", type=int, default=None)
    p.add_argument("--beta", type=int, default=None)
    p.add_argument("--gamma", type=int, default=None)
    p.add_argument("--loss-ratio", type=float, nargs=4, default=None)
    p.add_argument("--random-threshold", type=float, default=None)
    p.add_argument("--steps", type=float, default=6e3)
    p.add_argument("--t-max", type=float, default=10)
    p.add_argument("--t-min", type=float, default=1e-6)
    p.add_argument("--lp", action="store_true", default=None)
    p.add_argument("--no-lp", dest="lp", action="store_false")
    p.add_argument("--region-restricted", action="store_true", default=None)
    p.add_argument("--no-region-restricted", dest="region_restricted", action="store_false")
    p.add_argument("--hops-only", action="store_true", default=None)
    p.add_argument("--no-hops-only", dest="hops_only", action="store_false")
    p.add_argument("--reroute", action="store_true", default=None)
    p.add_argument("--no-reroute", dest="reroute", action="store_false")
    p.add_argument("--greedy-flag", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--split-degree", type=int, nargs='+', default=None)
    p.add_argument("--output", type=str, default=None)

    p.add_argument("--eval-interval", type=int, default=100,
                   help="Evaluate full pipeline every N SA steps (default: 100)")

    return p.parse_args()


def resolve_config(args):
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
    cfg["barrier"] = args.barrier
    cfg["seed"] = args.seed
    cfg["eval_interval"] = args.eval_interval
    return cfg


def main():
    args = parse_args()
    cfg = resolve_config(args)
    random.seed(cfg["seed"])

    barrier_suffix = "_barrier" if cfg["barrier"] else ""
    out_name = args.output or f"{args.operator}_distributed_{args.variant}{barrier_suffix}"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    split_degree = tuple(args.split_degree) if args.split_degree else None

    hw = init_hardware(cfg_path=os.path.join(file_path, args.hw_cfg),
                       topology=args.hw_topology)
    data_dict, beha_dict, broadcast_datatags = build_operator(
        args.operator, split_degree=split_degree)
    layers_regions = finalize_graph(
        beha_dict, data_dict, hw, greedy_flag=cfg["greedy_flag"])

    event_dict = {}
    hops, cd, cl, tc_loads, vu_loads = build_event(
        beha_dict=beha_dict, data_dict=data_dict, hardware_platform=hw,
        event_dict=event_dict, dijkstra_routing=False,
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])

    mem_datatags = []

    convergence = []

    saved_ll = dict(hw.link_loads_dict)
    saved_lc = dict(hw.link_loads_count)
    init_ed = deepcopy(event_dict)
    add_broadcast(
        data_tags=broadcast_datatags, ddr_chiplets=layers_regions[0],
        beha_dict=beha_dict, data_dict=data_dict, hardware_platform=hw,
        event_dict=init_ed, dijkstra_routing=False,
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])
    init_hw = deepcopy(hw)
    init_result = event_driver(events_dict=init_ed, hardware_platform=init_hw)
    convergence.append((0, init_result[0], None))
    for k in hw.link_loads_dict:
        hw.link_loads_dict[k] = saved_ll[k]
        hw.link_loads_count[k] = saved_lc[k]

    def eval_callback(step, best_state, best_energy,
                      best_link_loads, best_link_counts):
        t0 = time.time()
        eval_state = deepcopy(best_state)
        eval_beha = eval_state[5]
        eval_data = eval_state[6]
        eval_ed = eval_state[7]
        eval_hw = deepcopy(hw)
        for k in eval_hw.link_loads_dict:
            eval_hw.link_loads_dict[k] = best_link_loads.get(k, 0)
            eval_hw.link_loads_count[k] = best_link_counts.get(k, 0)

        if cfg["reroute"]:
            reroute_dijkstra(
                beha_dict=eval_beha, data_dict=eval_data,
                hardware_platform=eval_hw, event_dict=eval_ed,
                alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])

        add_broadcast(
            data_tags=broadcast_datatags, ddr_chiplets=layers_regions[0],
            beha_dict=eval_beha, data_dict=eval_data,
            hardware_platform=eval_hw, event_dict=eval_ed,
            dijkstra_routing=cfg["reroute"],
            alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])

        if cfg["barrier"]:
            apply_barrier_sync_all(eval_ed, eval_data)

        result = event_driver(events_dict=eval_ed, hardware_platform=eval_hw)
        dt = time.time() - t0
        convergence.append((step, result[0], best_energy))

    initial_state = [hops, cd, cl, tc_loads, vu_loads,
                     beha_dict, data_dict, event_dict]

    annealer = StreamOptimizationWithEval(
        state=initial_state,
        layers_regions=layers_regions,
        hardware_platform=hw,
        random_threshold=[cfg["random_threshold"]],
        LP_flag=cfg["lp"],
        region_restriced=cfg["region_restricted"],
        related=False,
        hops_only=cfg["hops_only"],
        loss_ratio=cfg["loss_ratio"],
        mem_enable=True,
        mem_threshold=[0.3],
        mem_datatags=mem_datatags,
        mem_random=True,
        dijkstra_routing=False,
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        gamma=cfg["gamma"],
        eval_callback=eval_callback,
        eval_interval=cfg["eval_interval"],
    )

    annealer.Tmax = cfg["t_max"]
    annealer.Tmin = cfg["t_min"]
    annealer.steps = cfg["steps"]
    annealer.copy_strategy = "deepcopy"
    annealer.verbose = False

    best_state, best_energy = annealer.anneal()

    beha_dict.clear()
    beha_dict.update(best_state[5])
    data_dict.clear()
    data_dict.update(best_state[6])
    event_dict.clear()
    event_dict.update(best_state[7])

    if cfg["reroute"]:
        reroute_dijkstra(
            beha_dict=beha_dict, data_dict=data_dict,
            hardware_platform=hw, event_dict=event_dict,
            alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])

    add_broadcast(
        data_tags=broadcast_datatags, ddr_chiplets=layers_regions[0],
        beha_dict=beha_dict, data_dict=data_dict, hardware_platform=hw,
        event_dict=event_dict, dijkstra_routing=cfg["reroute"],
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])

    if cfg["barrier"]:
        apply_barrier_sync_all(event_dict, data_dict)

    final_result = event_driver(events_dict=event_dict, hardware_platform=hw)
    total_steps = int(cfg["steps"])
    if convergence and convergence[-1][0] == total_steps:
        convergence[-1] = (total_steps, final_result[0], best_energy)
    else:
        convergence.append((total_steps, final_result[0], best_energy))
    conv_path = os.path.join(RESULTS_DIR, f"{out_name}_convergence.txt")
    with open(conv_path, 'w') as f:
        f.write(f"{'Step':>8}  {'Time_cost':>12}  {'Energy':>12}\n")
        for step, tc_val, energy in convergence:
            e_str = f"{energy:.6f}" if energy is not None else "N/A"
            f.write(f"{step:>8}  {tc_val:>12}  {e_str:>12}\n")


if __name__ == "__main__":
    main()
