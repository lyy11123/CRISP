
"""Simulated annealing (SA) based intra-die mapping: optimizes operator-to-core
placement with multi-objective loss balancing communication cost, load balance,
and failure tolerance."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
file_real_path = os.path.realpath(__file__)
filename_with_extension = os.path.basename(file_real_path)
filename_without_extension = os.path.splitext(filename_with_extension)[0]
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../'))
sys.path.append(os.path.join(file_path, '../../src/scheduling/communication/topology/'))
sys.path.append(os.path.join(file_path, '../../utils/'))
sys.path.append(os.path.join(file_path, '../scheduling/'))
sys.path.append(os.path.join(file_path, '../partition/'))


from net import net
from tlm import tlm2d
from read_cfg import cfg_to_dict
from Pre_Mapping import find_exclusive_dependents
from add_communication import update_event, update_offlinedata, add_mediumdata, _undo_init
from add_communication import update_event_v2, update_offlinedata_v2
from event_notation import event_notation
from beha_notation import beha_notation
from data_notation import tensor_notation


from typing import List, Tuple, Dict, Any
import random
random.seed(123)
import pickle
import time
import copy 
from copy import deepcopy
from collections import Counter
from simanneal import Annealer
import numpy as np
import math
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np


class StreamOptimization(Annealer):
    """Original SA optimizer for intra-die mapping. Reverts by re-calling update_event
    (full rebuild), unlike V2 which uses undo-log based revert."""

    def __init__(self, state: List[Any], layers_regions: Dict[int, List[int]],
                 hardware_platform: net,
                 random_threshold: List[float]=[0.3], LP_flag: bool=False, region_restriced: bool=True, related: bool=True, hops_only: bool=False, loss_ratio: List[float]=[1, 1, 1, 0.1], mem_enable: bool=True, mem_threshold: List[float]=[0.3], mem_datatags: List[Tuple[int]]=[], mem_random: bool=False,
                 dijkstra_routing: bool=False, alpha: int=100, beta: int=1, gamma: int=100):
        # state = [hops, comm_distances, comm_loads_dict, tc_loads_dict, vu_loads_dict, beha_dict, data_dict, event_dict]
        self.state = state
        # Normalization baseline: initial max values for each objective (avoid div-by-zero)
        self.origin = [state[0], state[1], max(state[2].values()), max(state[3].values()), max(state[4].values())]
        if self.origin[1] == 0:
            self.origin[1] = 1
        if self.origin[2] == 0:
            self.origin[2] = 1
        if self.origin[3] == 0:
            self.origin[3] = 1
        if self.origin[4] == 0:
            self.origin[4] = 1

        self.layers_regions = layers_regions

        # Build per-layer device candidate lists for SA move target selection
        self.comp_tags_list = list(state[5].keys())
        self.comp_position_dict = {}
        tensorcore_list = list(hardware_platform.modules_dict["tensorcore"].keys())
        vectorunit_list = list(hardware_platform.modules_dict["vectorunit"].keys())
        for layer_idx in layers_regions:
            self.comp_position_dict[layer_idx] = {}
            self.comp_position_dict[layer_idx]["tensorcore"] = []
            self.comp_position_dict[layer_idx]["vectorunit"] = []
            for tensorcore_idx in tensorcore_list:
                if tensorcore_idx[0] in layers_regions[layer_idx]:
                    self.comp_position_dict[layer_idx]["tensorcore"].append(tensorcore_idx)
            for vectorunit_idx in vectorunit_list:
                if vectorunit_idx[0] in layers_regions[layer_idx]:
                    self.comp_position_dict[layer_idx]["vectorunit"].append(vectorunit_idx)

        # Nodes with zero load on both TC and VU (candidates for LP_flag moves)
        self.free_node = set()
        for tensor_core_idx in state[3]:
            if state[3][tensor_core_idx] == 0 and state[4][tensor_core_idx] == 0:
                self.free_node.add(tensor_core_idx[:-1])
        self.free_node = list(self.free_node)

        # Pre-index offline (weight) data tags and their splits for DDR relocation moves
        self.offlinedata_list = []
        self.offlinedata_split_dict = {}
        for data_tag in state[6]:
            if data_tag[0] == 1:
                self.offlinedata_list.append(data_tag)
            self.offlinedata_split_dict[data_tag] = []
            for split_idx in self.state[6][data_tag].used_splitted_tag_dict:
                self.offlinedata_split_dict[data_tag].append(split_idx)

        # Per-layer DDR device list for offline data relocation
        self.ddr_list = {}
        for layer_idx in layers_regions:
            self.ddr_list[layer_idx] = []
            for ddr_idx in hardware_platform.ddr_dict:
                if ddr_idx[0] in layers_regions[layer_idx]:
                    self.ddr_list[layer_idx].append(ddr_idx)

        self.hardware_platform = hardware_platform

        # SA configuration flags
        self.random_threshold = random_threshold  # P(relocate) vs P(swap) per step
        self.LP_flag = LP_flag          # prefer idle nodes for placement
        self.region_restriced = region_restriced  # constrain swaps to adjacent layers
        self.related = related          # co-move exclusive dependents
        self.hops_only = hops_only      # use hop count as sole energy metric
        self.loss_ratio = loss_ratio    # weights: [comm_dist, max_comm, max_tc, max_vu]

        self.mem_enable = mem_enable
        self.mem_threshold = mem_threshold
        self.mem_datatags = mem_datatags
        self.mem_random = mem_random
        self.layer_idx = list(layers_regions.keys())[0] if layers_regions else 0

        # BALD routing parameters
        self.dijkstra_routing = dijkstra_routing
        self.alpha = alpha   # load-balance weight
        self.beta = beta     # distance weight
        self.gamma = gamma   # congestion weight

        self.movements = 0


    def move(self):
        # Decide move type: relocate a behavior/data (change_flag=True) or swap two behaviors
        self.change_flag = (random.random() < self.random_threshold[0])
        self.change_mem = (random.random() < self.mem_threshold[0])
        if self.change_flag:
            if self.change_mem and self.mem_enable:
                # --- Move type A: relocate an offline (weight) data split to a different DDR ---
                chosen_data = random.choice(self.offlinedata_list)
                chosen_split = random.choice(self.offlinedata_split_dict[chosen_data])
                self.last_data = chosen_data
                self.last_split = chosen_split
                self.last_position = self.state[6][chosen_data].generated_split_location[chosen_split][0] + (0, )

                location_candidates = []
                for belong_layer in self.state[6][chosen_data].belong_layer_set:
                    candidates = self.ddr_list[belong_layer[-1]]
                    location_candidates.extend(candidates)
                new_location = random.choice(location_candidates)

                self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] = update_offlinedata(
                    target_data_tags = chosen_data,
                    target_data_split = chosen_split,
                    target_devices = new_location,
                    hops = self.state[0],
                    communication_distances = self.state[1],
                    communication_loads_dict = self.state[2],
                    tensorcore_loads_dict = self.state[3],
                    vectorunit_loads_dict = self.state[4],
                    beha_dict = self.state[5],
                    data_dict = self.state[6],
                    hardware_platform = self.hardware_platform,
                    event_dict = self.state[7],
                    dijkstra_routing = self.dijkstra_routing,
                    alpha = self.alpha,
                    beta = self.beta,
                    gamma = self.gamma
                )

            else:
                # --- Move type B: relocate a compute behavior to a random device ---
                chosen_beha = random.choice(self.comp_tags_list)

                new_device = random.choice(self.comp_position_dict[chosen_beha[2]][self.state[7][chosen_beha].comp_device])
                device_load = self.state[3][new_device] if self.state[7][chosen_beha].comp_device == "tensorcore" else self.state[4][new_device]

                if self.LP_flag:
                    self.free_node = set()
                    for tensor_core_idx in self.state[3]:
                        if self.state[3][tensor_core_idx] == 0 and self.state[4][tensor_core_idx] == 0:
                            self.free_node.add(tensor_core_idx[:-1])
                    self.free_node = list(self.free_node)
                    if self.free_node:
                        while device_load > 0:
                            new_node = random.choice(self.free_node)
                            new_device = new_node + (0,)
                            device_load = self.state[3][new_device] if self.state[7][chosen_beha].comp_device == "tensorcore" else self.state[4][new_device]

                if self.related:
                    related_beha_tags = find_exclusive_dependents(beha_tag=chosen_beha, beha_dict=self.state[5])
                    chosen_beha_tags = [chosen_beha] + related_beha_tags
                    chosen_devices = [new_device] * len(chosen_beha_tags)
                    self.last_behas = [chosen_beha] + related_beha_tags
                    self.last_position = [self.state[7][chosen_beha].comp_location] * len(chosen_beha_tags)
                else:
                    chosen_beha_tags = [chosen_beha]
                    chosen_devices = [new_device]
                    self.last_behas = [chosen_beha]
                    self.last_position = [self.state[7][chosen_beha].comp_location]

                self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] = update_event(
                    target_event_tags = chosen_beha_tags,
                    target_devices = chosen_devices,
                    hops = self.state[0],
                    communication_distances = self.state[1],
                    communication_loads_dict = self.state[2],
                    tensorcore_loads_dict = self.state[3],
                    vectorunit_loads_dict = self.state[4],
                    beha_dict = self.state[5],
                    data_dict = self.state[6],
                    hardware_platform = self.hardware_platform,
                    event_dict = self.state[7],
                    dijkstra_routing = self.dijkstra_routing,
                    alpha = self.alpha,
                    beta = self.beta,
                    gamma = self.gamma
                )
        else:
            # --- Move type C: swap two behaviors' locations ---
            chosen_beha1 = random.choice(self.comp_tags_list)
            chosen_beha2 = random.choice(self.comp_tags_list)
            beha1_location = self.state[5][chosen_beha1].location
            beha2_location = self.state[5][chosen_beha2].location
            beha1_device = self.state[7][chosen_beha1].comp_location
            beha2_device = self.state[7][chosen_beha2].comp_location

            # Ensure swapped behaviors are in adjacent layers and within their assigned regions
            if self.region_restriced:
                while abs(chosen_beha1[2] - chosen_beha2[2]) > 1 or beha1_location[0] not in self.layers_regions[chosen_beha1[2]] or beha2_location[0] not in self.layers_regions[chosen_beha2[2]]:
                    chosen_beha1 = random.choice(self.comp_tags_list)
                    chosen_beha2 = random.choice(self.comp_tags_list)
                    beha1_location = self.state[5][chosen_beha1].location
                    beha2_location = self.state[5][chosen_beha2].location                    
                    beha1_device = self.state[7][chosen_beha1].comp_location
                    beha2_device = self.state[7][chosen_beha2].comp_location


            if self.related:
                related_beha_tags1 = find_exclusive_dependents(beha_tag=chosen_beha1, beha_dict=self.state[5])
                related_beha_tags2 = find_exclusive_dependents(beha_tag=chosen_beha2, beha_dict=self.state[5])
                chosen_beha_tags1 = [chosen_beha1] + related_beha_tags1
                chosen_beha_tags2 = [chosen_beha2] + related_beha_tags2
                chosen_beha_tags = chosen_beha_tags1 + chosen_beha_tags2
                chosen_devices = [beha2_device] * len(chosen_beha_tags1) + [beha1_device] * len(chosen_beha_tags2)
                self.last_behas = chosen_beha_tags
                self.last_position = [beha1_device] * len(chosen_beha_tags1) + [beha2_device] * len(chosen_beha_tags2)
            else:
                chosen_beha_tags = [chosen_beha1, chosen_beha2]
                chosen_devices = [beha2_device, beha1_device]
                self.last_behas = [chosen_beha1, chosen_beha2]
                self.last_position = [beha1_device, beha2_device]


            self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] = update_event(
                target_event_tags = [chosen_beha1, chosen_beha2],
                target_devices = [beha2_device, beha1_device],
                hops = self.state[0],
                communication_distances = self.state[1],
                communication_loads_dict = self.state[2],
                tensorcore_loads_dict = self.state[3],
                vectorunit_loads_dict = self.state[4],
                beha_dict = self.state[5], 
                data_dict = self.state[6],
                hardware_platform = self.hardware_platform,
                event_dict = self.state[7],
                dijkstra_routing = self.dijkstra_routing,
                alpha = self.alpha,
                beta = self.beta,
                gamma = self.gamma
            )

        self.movements += 1

        if self.mem_datatags:
            self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] = add_mediumdata(
                data_tags=self.mem_datatags,
                ddr_chiplets=self.layers_regions[self.layer_idx],
                hops=self.state[0],
                communication_distances=self.state[1],
                communication_loads_dict=self.state[2],
                tensorcore_loads_dict=self.state[3],
                vectorunit_loads_dict=self.state[4],
                beha_dict=self.state[5],
                data_dict=self.state[6],
                hardware_platform=self.hardware_platform,
                event_dict=self.state[7],
                dijkstra_routing=self.dijkstra_routing,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                random_flag=self.mem_random
            )


    def revert(self):
        """Undo the last move by re-calling update_event/update_offlinedata with saved positions."""
        if self.change_flag:
            if self.change_mem and self.mem_enable:
                self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] = update_offlinedata(
                    target_data_tags = self.last_data,
                    target_data_split = self.last_split,
                    target_devices = self.last_position,
                    hops = self.state[0],
                    communication_distances = self.state[1],
                    communication_loads_dict = self.state[2],
                    tensorcore_loads_dict = self.state[3],
                    vectorunit_loads_dict = self.state[4],
                    beha_dict = self.state[5],
                    data_dict = self.state[6],
                    hardware_platform = self.hardware_platform,
                    event_dict = self.state[7],
                    dijkstra_routing = self.dijkstra_routing,
                    alpha = self.alpha,
                    beta = self.beta,
                    gamma = self.gamma
                )
            else:
                self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] = update_event(
                    target_event_tags = self.last_behas,
                    target_devices = self.last_position,
                    hops = self.state[0],
                    communication_distances = self.state[1],
                    communication_loads_dict = self.state[2],
                    tensorcore_loads_dict = self.state[3],
                    vectorunit_loads_dict = self.state[4],
                    beha_dict = self.state[5],
                    data_dict = self.state[6],
                    hardware_platform = self.hardware_platform,
                    event_dict = self.state[7],
                    dijkstra_routing = self.dijkstra_routing,
                    alpha = self.alpha,
                    beta = self.beta,
                    gamma = self.gamma
                )
        else:
            self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] = update_event(
                target_event_tags = self.last_behas,
                target_devices = self.last_position,
                hops = self.state[0],
                communication_distances = self.state[1],
                communication_loads_dict = self.state[2],
                tensorcore_loads_dict = self.state[3],
                vectorunit_loads_dict = self.state[4],
                beha_dict = self.state[5],
                data_dict = self.state[6],
                hardware_platform = self.hardware_platform,
                event_dict = self.state[7],
                dijkstra_routing = self.dijkstra_routing,
                alpha = self.alpha,
                beta = self.beta,
                gamma = self.gamma
            )

        if self.mem_datatags:
            self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] = add_mediumdata(
                data_tags=self.mem_datatags,
                ddr_chiplets=self.layers_regions[self.layer_idx],
                hops=self.state[0],
                communication_distances=self.state[1],
                communication_loads_dict=self.state[2],
                tensorcore_loads_dict=self.state[3],
                vectorunit_loads_dict=self.state[4],
                beha_dict=self.state[5],
                data_dict=self.state[6],
                hardware_platform=self.hardware_platform,
                event_dict=self.state[7],
                dijkstra_routing=self.dijkstra_routing,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                random_flag=self.mem_random
            )


    def energy(self):
        # Multi-objective: weighted sum of normalized comm_dist, max_comm_load, max_tc_load, max_vu_load
        if self.hops_only:
            result = self.state[0]
        else:
            result = self.state[1] / self.origin[1] * self.loss_ratio[0] + \
                   max(self.state[2].values()) / self.origin[2] * self.loss_ratio[1] + \
                   max(self.state[3].values()) / self.origin[3] * self.loss_ratio[2] + \
                   max(self.state[4].values()) / self.origin[4] * self.loss_ratio[3]

        return result


    def anneal(self):
        """SA main loop with exponential cooling. Uses standard Metropolis acceptance criterion."""
        random.seed(123)
        np.random.seed(123)

        step = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        # Precompute exponential cooling factor
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        E = self.energy()
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials = accepts = improves = 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        # SA iteration: move -> evaluate dE -> accept/reject
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
            if dE > 0.0 and math.exp(-dE / T) < random.random():
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
            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials = accepts = improves = 0

        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        # Return best state and energy
        if self.verbose:
            print("Best energy: ", self.best_energy)
        return self.best_state, self.best_energy


# Entry point for original SA-based mapping (uses StreamOptimization with full-rebuild revert)
def stream_mapping(
    hops: int,
    communication_distances: float,
    communication_loads_dict: Dict[Tuple[int], float],
    tensorcore_loads_dict: Dict[Tuple[int], float],
    vectorunit_loads_dict: Dict[Tuple[int], float],
    layers_regions: Dict[int, List[int]],
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net,
    event_dict: Dict[Tuple[int], event_notation],
    random_threshold: List[float] = [0.3],
    LP_flag: bool = False,
    region_restriced: bool = True,
    related: bool = True,
    hops_only: bool = False,
    loss_ratio: List[float] = [1, 1, 1, 0.1],
    mem_enable: bool = True,
    mem_threshold: List[float] = [0.3],
    mem_datatags: List[Tuple[int]] = [],
    mem_random: bool = False,
    dijkstra_routing: bool = False,
    alpha: int = 100,
    beta: int = 1,
    gamma: int = 100,
    t_max: int=10,
    t_min: float=1e-8,
    steps: int=1e6,
):
    initial_state = [hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict, beha_dict, data_dict, event_dict]

    annealer = StreamOptimization(
        state=initial_state,
        layers_regions=layers_regions,
        hardware_platform=hardware_platform,
        random_threshold=random_threshold,
        LP_flag=LP_flag,
        region_restriced=region_restriced,
        related=related,
        hops_only=hops_only,
        loss_ratio=loss_ratio,
        mem_enable=mem_enable,
        mem_threshold=mem_threshold,
        mem_datatags=mem_datatags,
        mem_random=mem_random,
        dijkstra_routing=dijkstra_routing,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )

    annealer.Tmax = t_max 
    annealer.Tmin = t_min 
    annealer.steps = steps 

    annealer.copy_strategy = "deepcopy"
    annealer.verbose = True

    best_state, best_energy = annealer.anneal()
    print(f"Iteration {annealer.movements} completed. Best energy: {best_energy}")

    return best_state


class StreamOptimizationV2(Annealer):
    """Optimized SA optimizer with undo-log revert and incremental energy computation.
    Uses _apply_undo() for O(delta) revert instead of full update_event rebuild.
    Tracks cached max values (_max_comm, _max_tc, _max_vu) for incremental dE."""

    def __init__(self, state: List[Any], layers_regions: Dict[int, List[int]],
                 hardware_platform: net,
                 random_threshold: List[float]=[0.3], LP_flag: bool=False, region_restriced: bool=True, related: bool=True, hops_only: bool=False, loss_ratio: List[float]=[1, 1, 1, 0.1], mem_enable: bool=True, mem_threshold: List[float]=[0.3], mem_datatags: List[Tuple[int]]=[], mem_random: bool=False,
                 dijkstra_routing: bool=False, alpha: int=100, beta: int=1, gamma: int=100):
        # state = [hops, comm_distances, comm_loads_dict, tc_loads_dict, vu_loads_dict, beha_dict, data_dict, event_dict]
        self.state = state
        # Normalization baseline (guards against div-by-zero; empty comm_loads possible in dijkstra mode)
        self.origin = [state[0], state[1], max(state[2].values()) if state[2] else 0, max(state[3].values()), max(state[4].values())]
        if self.origin[1] == 0:
            self.origin[1] = 1
        if self.origin[2] == 0:
            self.origin[2] = 1
        if self.origin[3] == 0:
            self.origin[3] = 1
        if self.origin[4] == 0:
            self.origin[4] = 1

        # Cached max values for incremental energy computation (avoids full max() scan)
        self._max_comm = self.origin[2]
        self._max_tc = self.origin[3]
        self._max_vu = self.origin[4]

        self.layers_regions = layers_regions

        # Build per-layer device candidate lists for SA move target selection
        self.comp_tags_list = list(state[5].keys())
        self.comp_position_dict = {}
        tensorcore_list = list(hardware_platform.modules_dict["tensorcore"].keys())
        vectorunit_list = list(hardware_platform.modules_dict["vectorunit"].keys())
        for layer_idx in layers_regions:
            self.comp_position_dict[layer_idx] = {}
            self.comp_position_dict[layer_idx]["tensorcore"] = []
            self.comp_position_dict[layer_idx]["vectorunit"] = []
            for tensorcore_idx in tensorcore_list:
                if tensorcore_idx[0] in layers_regions[layer_idx]:
                    self.comp_position_dict[layer_idx]["tensorcore"].append(tensorcore_idx)
            for vectorunit_idx in vectorunit_list:
                if vectorunit_idx[0] in layers_regions[layer_idx]:
                    self.comp_position_dict[layer_idx]["vectorunit"].append(vectorunit_idx)

        self.free_node = set()
        for tensor_core_idx in state[3]:
            if state[3][tensor_core_idx] == 0 and state[4].get(tensor_core_idx, 0) == 0:
                self.free_node.add(tensor_core_idx[:-1])
        self.free_node = list(self.free_node)

        self.offlinedata_list = []
        self.offlinedata_split_dict = {}
        for data_tag in state[6]:
            if data_tag[0] == 1:
                self.offlinedata_list.append(data_tag)
            self.offlinedata_split_dict[data_tag] = []
            for split_idx in self.state[6][data_tag].used_splitted_tag_dict:
                self.offlinedata_split_dict[data_tag].append(split_idx)

        self.ddr_list = {}
        for layer_idx in layers_regions:
            self.ddr_list[layer_idx] = []
            for ddr_idx in hardware_platform.ddr_dict:
                if ddr_idx[0] in layers_regions[layer_idx]:
                    self.ddr_list[layer_idx].append(ddr_idx)

        self.hardware_platform = hardware_platform

        self.random_threshold = random_threshold
        self.LP_flag = LP_flag
        self.region_restriced = region_restriced
        self.related = related
        self.hops_only = hops_only
        self.loss_ratio = loss_ratio

        self.mem_enable = mem_enable
        self.mem_threshold = mem_threshold
        self.mem_datatags = mem_datatags
        self.mem_random = mem_random
        self.layer_idx = list(layers_regions.keys())[0] if layers_regions else 0

        self.dijkstra_routing = dijkstra_routing
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.movements = 0


    def _resolve_device(self, beha_tag, target_node):
        """Find a device of the behavior's own type at the target node."""
        dev_type = self.state[7][beha_tag].comp_device
        for d in self.comp_position_dict[beha_tag[2]][dev_type]:
            if d[:-1] == target_node:
                return d
        return self.state[7][beha_tag].comp_location


    @staticmethod
    def _incremental_max(dict_, cached_max, changed_old_vals):
        """Update cached max after a few dict keys changed.
        changed_old_vals: {key: old_value} from the undo log.
        Returns the new max value.
        """
        if not changed_old_vals:
            return cached_max
        new_max_candidate = cached_max
        need_rescan = False
        for k, old_v in changed_old_vals.items():
            new_v = dict_[k]
            if new_v > new_max_candidate:
                new_max_candidate = new_v
            if old_v >= cached_max and new_v < cached_max:
                need_rescan = True
        if new_max_candidate > cached_max:
            return new_max_candidate
        if need_rescan:
            return max(dict_.values()) if dict_ else 0
        return cached_max

    def _apply_undo(self, undo):
        """Restore state from an undo record -- no update_event call needed."""
        # 1. Remove comms created during move
        for tag in undo['created_comms']:
            if tag in self.state[7]:
                del self.state[7][tag]

        # 2. Restore comms deleted during move
        for tag, ev in undo['deleted_comms'].items():
            self.state[7][tag] = ev

        # 3. Restore issue_set / dependency_set
        for tag, s in undo['issue_sets'].items():
            if tag in self.state[7]:
                self.state[7][tag].issue_set = s
        for tag, s in undo['dep_sets'].items():
            if tag in self.state[7]:
                self.state[7][tag].dependency_set = s

        # 4. Restore beha and comp locations
        for tag, loc in undo['beha_locs'].items():
            self.state[5][tag].location = loc
        for tag, loc in undo['comp_locs'].items():
            self.state[7][tag].comp_location = loc

        # 5. Restore data structures
        for (dt, sp), lst in undo['data_gen_locs'].items():
            self.state[6][dt].generated_split_location[sp] = lst
        for (dt, sp), st in undo['data_used'].items():
            self.state[6][dt].used_splitted_tag_dict[sp] = st

        # 6. Restore scalars and load dicts
        self.state[0] = undo['hops']
        self.state[1] = undo['comm_dist']
        for k, v in undo['tc_loads'].items():
            self.state[3][k] = v
        for k, v in undo['vu_loads'].items():
            self.state[4][k] = v
        for k, v in undo['comm_loads'].items():
            self.state[2][k] = v

        # 7. Restore ID allocators
        for ct, (used, recycled, nid) in undo['allocators'].items():
            if ct in self.state[7]:
                a = self.state[7][ct].commid_allocator
                a.used_ids = used
                a.recycled = recycled
                a.next_id = nid

        # 8. Restore offline comm event properties (for update_offlinedata)
        for ct, (src, paths, plist, hops_val, cdist) in undo.get('offline_comms', {}).items():
            if ct in self.state[7]:
                self.state[7][ct].source_location = src
                self.state[7][ct].paths = paths
                self.state[7][ct].path_list = plist
                self.state[7][ct].hops = hops_val
                self.state[7][ct].communication_distances = cdist

        # 9. Restore platform link loads (dijkstra routing)
        if undo['platform_link_loads'] is not None:
            for k, v in undo['platform_link_loads'].items():
                self.hardware_platform.link_loads_dict[k] = v
        if undo['platform_link_counts'] is not None:
            for k, v in undo['platform_link_counts'].items():
                self.hardware_platform.link_loads_count[k] = v


    def move(self):
        """Perform one SA move. Returns dE (incremental) or None (requires full energy())."""
        self._undo = _undo_init()  # Fresh undo log for this move
        self.change_flag = (random.random() < self.random_threshold[0])
        self.change_mem = (random.random() < self.mem_threshold[0])
        if self.change_flag:
            if self.change_mem and self.mem_enable:
                chosen_data = random.choice(self.offlinedata_list)
                chosen_split = random.choice(self.offlinedata_split_dict[chosen_data])
                self.last_data = chosen_data
                self.last_split = chosen_split
                self.last_position = self.state[6][chosen_data].generated_split_location[chosen_split][0] + (0, )

                location_candidates = []
                for belong_layer in self.state[6][chosen_data].belong_layer_set:
                    candidates = self.ddr_list[belong_layer[-1]]
                    location_candidates.extend(candidates)
                new_location = random.choice(location_candidates)

                self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] = update_offlinedata_v2(
                    target_data_tags = chosen_data,
                    target_data_split = chosen_split,
                    target_devices = new_location,
                    hops = self.state[0],
                    communication_distances = self.state[1],
                    communication_loads_dict = self.state[2],
                    tensorcore_loads_dict = self.state[3],
                    vectorunit_loads_dict = self.state[4],
                    beha_dict = self.state[5],
                    data_dict = self.state[6],
                    hardware_platform = self.hardware_platform,
                    event_dict = self.state[7],
                    dijkstra_routing = self.dijkstra_routing,
                    alpha = self.alpha,
                    beta = self.beta,
                    gamma = self.gamma,
                    undo = self._undo
                )

            else:
                chosen_beha = random.choice(self.comp_tags_list)

                new_device = random.choice(self.comp_position_dict[chosen_beha[2]][self.state[7][chosen_beha].comp_device])
                device_load = self.state[3][new_device] if self.state[7][chosen_beha].comp_device == "tensorcore" else self.state[4][new_device]

                if self.LP_flag:
                    self.free_node = set()
                    for tensor_core_idx in self.state[3]:
                        if self.state[3][tensor_core_idx] == 0 and self.state[4].get(tensor_core_idx, 0) == 0:
                            self.free_node.add(tensor_core_idx[:-1])
                    self.free_node = list(self.free_node)
                    if self.free_node:
                        while device_load > 0:
                            new_node = random.choice(self.free_node)
                            new_device = new_node + (0,)
                            device_load = self.state[3][new_device] if self.state[7][chosen_beha].comp_device == "tensorcore" else self.state[4][new_device]

                if self.related:
                    related_beha_tags = find_exclusive_dependents(beha_tag=chosen_beha, beha_dict=self.state[5])
                    chosen_beha_tags = [chosen_beha] + related_beha_tags
                    target_node = new_device[:-1]
                    chosen_devices = [new_device] + [self._resolve_device(tag, target_node) for tag in related_beha_tags]
                    self.last_behas = [chosen_beha] + related_beha_tags
                    self.last_position = [self.state[7][tag].comp_location for tag in chosen_beha_tags]
                else:
                    chosen_beha_tags = [chosen_beha]
                    chosen_devices = [new_device]
                    self.last_behas = [chosen_beha]
                    self.last_position = [self.state[7][chosen_beha].comp_location]

                self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] = update_event_v2(
                    target_event_tags = chosen_beha_tags,
                    target_devices = chosen_devices,
                    hops = self.state[0],
                    communication_distances = self.state[1],
                    communication_loads_dict = self.state[2],
                    tensorcore_loads_dict = self.state[3],
                    vectorunit_loads_dict = self.state[4],
                    beha_dict = self.state[5],
                    data_dict = self.state[6],
                    hardware_platform = self.hardware_platform,
                    event_dict = self.state[7],
                    dijkstra_routing = self.dijkstra_routing,
                    alpha = self.alpha,
                    beta = self.beta,
                    gamma = self.gamma,
                    undo = self._undo
                )
        else:
            chosen_beha1 = random.choice(self.comp_tags_list)
            chosen_beha2 = random.choice(self.comp_tags_list)
            beha1_location = self.state[5][chosen_beha1].location
            beha2_location = self.state[5][chosen_beha2].location
            beha1_device = self.state[7][chosen_beha1].comp_location
            beha2_device = self.state[7][chosen_beha2].comp_location

            if self.region_restriced:
                while abs(chosen_beha1[2] - chosen_beha2[2]) > 1 or beha1_location[0] not in self.layers_regions[chosen_beha1[2]] or beha2_location[0] not in self.layers_regions[chosen_beha2[2]] or self.state[7][chosen_beha1].comp_device != self.state[7][chosen_beha2].comp_device:
                    chosen_beha1 = random.choice(self.comp_tags_list)
                    chosen_beha2 = random.choice(self.comp_tags_list)
                    beha1_location = self.state[5][chosen_beha1].location
                    beha2_location = self.state[5][chosen_beha2].location
                    beha1_device = self.state[7][chosen_beha1].comp_location
                    beha2_device = self.state[7][chosen_beha2].comp_location

            if self.related:
                related_beha_tags1 = find_exclusive_dependents(beha_tag=chosen_beha1, beha_dict=self.state[5])
                related_beha_tags2 = find_exclusive_dependents(beha_tag=chosen_beha2, beha_dict=self.state[5])
                chosen_beha_tags1 = [chosen_beha1] + related_beha_tags1
                chosen_beha_tags2 = [chosen_beha2] + related_beha_tags2
                chosen_beha_tags = chosen_beha_tags1 + chosen_beha_tags2
                self.last_behas = [chosen_beha1, chosen_beha2]
                self.last_position = [beha1_device, beha2_device]
            else:
                chosen_beha_tags = [chosen_beha1, chosen_beha2]
                self.last_behas = [chosen_beha1, chosen_beha2]
                self.last_position = [beha1_device, beha2_device]


            self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] = update_event_v2(
                target_event_tags = [chosen_beha1, chosen_beha2],
                target_devices = [beha2_device, beha1_device],
                hops = self.state[0],
                communication_distances = self.state[1],
                communication_loads_dict = self.state[2],
                tensorcore_loads_dict = self.state[3],
                vectorunit_loads_dict = self.state[4],
                beha_dict = self.state[5],
                data_dict = self.state[6],
                hardware_platform = self.hardware_platform,
                event_dict = self.state[7],
                dijkstra_routing = self.dijkstra_routing,
                alpha = self.alpha,
                beta = self.beta,
                gamma = self.gamma,
                undo = self._undo
            )

        self.movements += 1

        if self.mem_datatags:
            self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] = add_mediumdata(
                data_tags=self.mem_datatags,
                ddr_chiplets=self.layers_regions[self.layer_idx],
                hops=self.state[0],
                communication_distances=self.state[1],
                communication_loads_dict=self.state[2],
                tensorcore_loads_dict=self.state[3],
                vectorunit_loads_dict=self.state[4],
                beha_dict=self.state[5],
                data_dict=self.state[6],
                hardware_platform=self.hardware_platform,
                event_dict=self.state[7],
                dijkstra_routing=self.dijkstra_routing,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                random_flag=self.mem_random
            )
            return None  # add_mediumdata not tracked by undo; fall back to full energy()

        # --- Incremental energy: compute dE from undo log ---
        if self.hops_only:
            return self.state[0] - self._undo['hops']

        # Save old maxes in undo for revert
        self._undo['_old_max_comm'] = self._max_comm
        self._undo['_old_max_tc'] = self._max_tc
        self._undo['_old_max_vu'] = self._max_vu

        # Update cached maxes incrementally using only changed keys
        self._max_comm = self._incremental_max(self.state[2], self._max_comm, self._undo['comm_loads'])
        self._max_tc = self._incremental_max(self.state[3], self._max_tc, self._undo['tc_loads'])
        self._max_vu = self._incremental_max(self.state[4], self._max_vu, self._undo['vu_loads'])

        # Compute delta energy (new - old) using cached values
        old_e = (self._undo['comm_dist'] / self.origin[1] * self.loss_ratio[0] +
                 self._undo['_old_max_comm'] / self.origin[2] * self.loss_ratio[1] +
                 self._undo['_old_max_tc'] / self.origin[3] * self.loss_ratio[2] +
                 self._undo['_old_max_vu'] / self.origin[4] * self.loss_ratio[3])
        new_e = (self.state[1] / self.origin[1] * self.loss_ratio[0] +
                 self._max_comm / self.origin[2] * self.loss_ratio[1] +
                 self._max_tc / self.origin[3] * self.loss_ratio[2] +
                 self._max_vu / self.origin[4] * self.loss_ratio[3])
        return new_e - old_e


    def revert(self):
        """Restore state via undo log -- O(delta) cost, no update_event call."""
        # Restore cached maxes before applying undo
        if '_old_max_comm' in self._undo:
            self._max_comm = self._undo['_old_max_comm']
            self._max_tc = self._undo['_old_max_tc']
            self._max_vu = self._undo['_old_max_vu']
        self._apply_undo(self._undo)


    def energy(self):
        if self.hops_only:
            result = self.state[0]
        else:
            result = self.state[1] / self.origin[1] * self.loss_ratio[0] + \
                   (max(self.state[2].values()) if self.state[2] else 0) / self.origin[2] * self.loss_ratio[1] + \
                   max(self.state[3].values()) / self.origin[3] * self.loss_ratio[2] + \
                   max(self.state[4].values()) / self.origin[4] * self.loss_ratio[3]
        return result


    def anneal(self):
        random.seed(123)
        np.random.seed(123)

        """Minimizes the energy of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
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

        # SA iteration: move -> incremental dE -> greedy accept (dE < 0) or reject
        while step < self.steps and not self.user_exit:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E = self.energy()  # Fallback: full energy when move can't compute dE
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE >= 0.0:
                self.revert()  # Reject: restore via undo log
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
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials = accepts = improves = 0

        # Restore best state and platform link loads found during SA
        self.state = self.copy_state(self.best_state)
        for k in self.hardware_platform.link_loads_dict:
            self.hardware_platform.link_loads_dict[k] = self.best_link_loads[k]
            self.hardware_platform.link_loads_count[k] = self.best_link_counts[k]
        if self.save_state_on_exit:
            self.save_state()

        if self.verbose:
            print("Best energy: ", self.best_energy)
        return self.best_state, self.best_energy


# Entry point for optimized SA mapping (V2: undo-log revert, incremental energy, group Dijkstra)
def stream_mapping_v2(
    hops: int,
    communication_distances: float,
    communication_loads_dict: Dict[Tuple[int], float],
    tensorcore_loads_dict: Dict[Tuple[int], float],
    vectorunit_loads_dict: Dict[Tuple[int], float],
    layers_regions: Dict[int, List[int]],
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net,
    event_dict: Dict[Tuple[int], event_notation],
    random_threshold: List[float] = [0.3],
    LP_flag: bool = False,
    region_restriced: bool = True,
    related: bool = True,
    hops_only: bool = False,
    loss_ratio: List[float] = [1, 1, 1, 0.1],
    mem_enable: bool = True,
    mem_threshold: List[float] = [0.3],
    mem_datatags: List[Tuple[int]] = [],
    mem_random: bool = False,
    dijkstra_routing: bool = False,
    alpha: int = 100,
    beta: int = 1,
    gamma: int = 100,
    t_max: int=10,
    t_min: float=1e-8,
    steps: int=1e6,
):
    initial_state = [hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict, beha_dict, data_dict, event_dict]

    annealer = StreamOptimizationV2(
        state=initial_state,
        layers_regions=layers_regions,
        hardware_platform=hardware_platform,
        random_threshold=random_threshold,
        LP_flag=LP_flag,
        region_restriced=region_restriced,
        related=related,
        hops_only=hops_only,
        loss_ratio=loss_ratio,
        mem_enable=mem_enable,
        mem_threshold=mem_threshold,
        mem_datatags=mem_datatags,
        mem_random=mem_random,
        dijkstra_routing=dijkstra_routing,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )

    annealer.Tmax = t_max
    annealer.Tmin = t_min
    annealer.steps = steps

    annealer.copy_strategy = "deepcopy"
    annealer.verbose = True

    best_state, best_energy = annealer.anneal()
    print(f"Iteration {annealer.movements} completed. Best energy: {best_energy}")

    # Sync caller's dicts in-place from best_state (deepcopy creates new objects)
    beha_dict.clear()
    beha_dict.update(best_state[5])
    data_dict.clear()
    data_dict.update(best_state[6])
    event_dict.clear()
    event_dict.update(best_state[7])

    return best_state
