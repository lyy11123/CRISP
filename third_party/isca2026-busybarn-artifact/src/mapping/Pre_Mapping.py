
"""DAG construction for LLM workloads: builds the initial mapping DAG from
partitioned operators and hardware topology before SA optimization."""

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
from add_communication import update_event
from event_notation import event_notation
from beha_notation import beha_notation, beha_kind_dict
from data_notation import tensor_notation


from typing import List, Tuple, Dict, Any
import random
random.seed(123)
import pickle
from copy import deepcopy
from collections import Counter
from simanneal import Annealer
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np


def update_offline_data(
    data_dict: Dict[Tuple[int], tensor_notation]
):
    """Propagate split locations for offline data (inputs and weights) to sub-splits."""
    for data_tag in data_dict:
        if data_tag == (0, 0) or data_tag[0] == 1:
            dealing_generated_splitted_tag_dict = deepcopy(data_dict[data_tag].generated_splitted_tag_dict)
            for origin_split in dealing_generated_splitted_tag_dict:
                if origin_split in data_dict[data_tag].splitted_corresponding_dict:
                    for sub_split in data_dict[data_tag].splitted_corresponding_dict[origin_split]:
                        data_dict[data_tag].generated_splitted_tag_dict[sub_split] = data_dict[data_tag].generated_splitted_tag_dict[origin_split]  
                else:
                    continue


def update_online_data(
    data_dict: Dict[Tuple[int], tensor_notation],
    beha_dict: Dict[Tuple[int], beha_notation],
):
    """Expand coarse data splits to fine-grained sub-splits in behavior needed/produced dicts.
    Updates used_splitted_tag_dict and generated_splitted_tag_dict accordingly."""
    for beha_tag in beha_dict:
        # Expand needed (input) data splits
        for source_data_tag in beha_dict[beha_tag].needed_data_split_dict:
            copy_data_split_dict = deepcopy(beha_dict[beha_tag].needed_data_split_dict[source_data_tag])
            for source_data_split in copy_data_split_dict:
                if source_data_split in data_dict[source_data_tag].splitted_corresponding_dict and len(data_dict[source_data_tag].splitted_corresponding_dict[source_data_split]) > 1:
                    beha_dict[beha_tag].needed_data_split_dict[source_data_tag].remove(source_data_split)
                    beha_dict[beha_tag].needed_data_split_dict[source_data_tag] |= (data_dict[source_data_tag].splitted_corresponding_dict[source_data_split])
                    for split in data_dict[source_data_tag].splitted_corresponding_dict[source_data_split]:
                        if source_data_split in data_dict[source_data_tag].used_splitted_tag_dict:
                            if split in data_dict[source_data_tag].used_splitted_tag_dict:
                                data_dict[source_data_tag].used_splitted_tag_dict[split] |= (data_dict[source_data_tag].used_splitted_tag_dict[source_data_split])
                            else:
                                data_dict[source_data_tag].used_splitted_tag_dict[split] = data_dict[source_data_tag].used_splitted_tag_dict[source_data_split]

        # Expand produced (output) data splits and build reverse index (split -> producer)
        for target_data_tag in beha_dict[beha_tag].produced_data_split_dict:
            copy_data_split_dict = deepcopy(beha_dict[beha_tag].produced_data_split_dict[target_data_tag])
            for target_data_split in copy_data_split_dict:
                if target_data_split in data_dict[target_data_tag].splitted_corresponding_dict and len(data_dict[target_data_tag].splitted_corresponding_dict[target_data_split]) > 1:
                    beha_dict[beha_tag].produced_data_split_dict[target_data_tag].remove(target_data_split)
                    beha_dict[beha_tag].produced_data_split_dict[target_data_tag] |= (data_dict[target_data_tag].splitted_corresponding_dict[target_data_split])
                    for split in data_dict[target_data_tag].splitted_corresponding_dict[target_data_split]:
                        for producer_tag in data_dict[target_data_tag].generated_tag_splitted_dict:
                            if split in data_dict[target_data_tag].generated_tag_splitted_dict[producer_tag]:
                                if split in data_dict[target_data_tag].generated_splitted_tag_dict:
                                    data_dict[target_data_tag].generated_splitted_tag_dict[split].add(producer_tag)
                                else:
                                    data_dict[target_data_tag].generated_splitted_tag_dict[split] = {producer_tag}
                else:
                    for producer_tag in data_dict[target_data_tag].generated_tag_splitted_dict:
                        if target_data_split in data_dict[target_data_tag].generated_tag_splitted_dict[producer_tag]:
                            if target_data_split in data_dict[target_data_tag].generated_splitted_tag_dict:
                                data_dict[target_data_tag].generated_splitted_tag_dict[target_data_split].add(producer_tag)
                            else:
                                data_dict[target_data_tag].generated_splitted_tag_dict[target_data_split] = {producer_tag}        


def update_data(
    data_dict: Dict[Tuple[int], tensor_notation],
    beha_dict: Dict[Tuple[int], beha_notation],
):
    """Master data update: propagate offline splits then expand online behavior data references."""
    update_offline_data(data_dict)
    update_online_data(data_dict, beha_dict)


def find_exclusive_dependents(beha_tag, beha_dict):
    """
    Recursively find all consumers that depend only on the current behavior,
    then continue down the chain while each downstream behavior remains an
    exclusive dependent.

    Args:
        beha_tag (str): Tag of the current behavior.
        beha_dict (dict): Dictionary of all behavior instances. Keys are tags
            and values expose ``producer_tags`` and ``consumer_tags``.

    Returns:
        List[str]: Tags for all exclusive consumers in the chain, excluding
        the starting ``beha_tag``.
    """
    result = []

    def dfs(current_tag):
        for consumer_tag in beha_dict[current_tag].consumer_tags:
            consumer = beha_dict[consumer_tag]
            if consumer.producer_tags == {current_tag}:
                result.append(consumer_tag)
                dfs(consumer_tag)

    dfs(beha_tag)
    return result


def initialized_mapping(
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net,
    layers_regions: Dict[int, List[int]],
    greedy_flag: bool = False,
    mem_enable: bool = True
):
    """Create initial operator-to-core mapping before SA optimization.
    greedy_flag=True: load-balanced placement; False: random placement.
    Also assigns offline data (weights, activations) to DDR locations."""

    # Build per-layer available device lists
    comp_position_dict = {}
    tensorcore_list = list(hardware_platform.modules_dict["tensorcore"].keys())
    vectorunit_list = list(hardware_platform.modules_dict["vectorunit"].keys())
    if mem_enable:
        ddr_list = list(hardware_platform.ddr_dict.keys())

    if greedy_flag:
        tensorcore_workload_dict = {device_name: 0 for device_name in tensorcore_list}
        vectorunit_workload_dict = {device_name: 0 for device_name in vectorunit_list}
        if mem_enable:
            ddr_workload_dict = {device_name: 0 for device_name in ddr_list}

    for layer_idx in layers_regions:
        comp_position_dict[layer_idx] = {}
        comp_position_dict[layer_idx]["tensorcore"] = []
        comp_position_dict[layer_idx]["vectorunit"] = []
        for tensorcore_idx in tensorcore_list:
            if tensorcore_idx[0] in layers_regions[layer_idx]:
                comp_position_dict[layer_idx]["tensorcore"].append(tensorcore_idx)
        for vectorunit_idx in vectorunit_list:
            if vectorunit_idx[0] in layers_regions[layer_idx]:
                comp_position_dict[layer_idx]["vectorunit"].append(vectorunit_idx)

    # Initialize empty generated_split_location for all sub-splits
    for tensor_tag in data_dict:
        tensor = data_dict[tensor_tag]
        for source_split in tensor.splitted_corresponding_dict:
            for split in tensor.splitted_corresponding_dict[source_split]:
                if split not in tensor.generated_split_location:
                    tensor.generated_split_location[split] = []

    # Assign each behavior to a device type and pick an initial location
    for beha_tag in beha_dict:
        if beha_dict[beha_tag].beha_type in beha_kind_dict["tensorcore"]:
            beha_dict[beha_tag].device = "tensorcore"
        elif beha_dict[beha_tag].beha_type in beha_kind_dict["vectorunit"]:
            beha_dict[beha_tag].device = "vectorunit"
        else:
            raise ValueError(f"Unknown beha_type: {beha_dict[beha_tag].beha_type}")
        if beha_dict[beha_tag].location is not None:
            continue
        if greedy_flag:

            # if len(beha_dict[beha_tag].producer_tags) == 1:
            #     producer_bahe_tag, = beha_dict[beha_tag].producer_tags
            #     chosen_location = beha_dict[producer_bahe_tag].location
            #     if chosen_location:
            if beha_dict[beha_tag].device == "tensorcore":
                tensorcores = list(tensorcore_workload_dict.keys())
                random.shuffle(tensorcores)
                chosen_location = min(tensorcores, key=lambda k: tensorcore_workload_dict[k])
                # chosen_location = min(tensorcore_workload_dict, key=tensorcore_workload_dict.get)
                tensorcore_workload_dict[chosen_location] += hardware_platform.modules_dict["tensorcore"][chosen_location].working_time(
                    source_datashape=beha_dict[beha_tag].beha_datashape,
                    beha_type=beha_dict[beha_tag].beha_type,
                    frequency=hardware_platform.frequency
                )
            else:
                vectorunits = list(vectorunit_workload_dict.keys())
                random.shuffle(vectorunits)
                chosen_location = min(vectorunits, key=lambda k: vectorunit_workload_dict[k])
                # chosen_location = min(vectorunit_workload_dict, key=vectorunit_workload_dict.get)
                vectorunit_workload_dict[chosen_location] += hardware_platform.modules_dict["vectorunit"][chosen_location].working_time(
                    source_datashape=beha_dict[beha_tag].beha_datashape,
                    beha_type=beha_dict[beha_tag].beha_type,
                    frequency=hardware_platform.frequency
                )
        else:
            chosen_location = random.choice(comp_position_dict[beha_tag[2]][beha_dict[beha_tag].device])
        beha_dict[beha_tag].location = chosen_location
        for produced_data_tag in beha_dict[beha_tag].produced_data_split_dict:
            for produced_data_split in beha_dict[beha_tag].produced_data_split_dict[produced_data_tag]:
                data_dict[produced_data_tag].generated_split_location[produced_data_split].append(beha_dict[beha_tag].location[:-1]) 

        if greedy_flag:
            next_beha_list = find_exclusive_dependents(beha_tag, beha_dict)
            for next_beha_tag in next_beha_list:
                if beha_dict[next_beha_tag].location is not None:
                    if beha_dict[next_beha_tag].device == "tensorcore":
                        tensorcore_workload_dict[beha_dict[next_beha_tag].location] += hardware_platform.modules_dict["tensorcore"][beha_dict[next_beha_tag].location].working_time(
                            source_datashape=beha_dict[next_beha_tag].beha_datashape,
                            beha_type=beha_dict[next_beha_tag].beha_type,
                            frequency=hardware_platform.frequency
                        )
                    else:
                        vectorunit_workload_dict[beha_dict[next_beha_tag].location] += hardware_platform.modules_dict["vectorunit"][beha_dict[next_beha_tag].location].working_time(
                            source_datashape=beha_dict[next_beha_tag].beha_datashape,
                            beha_type=beha_dict[next_beha_tag].beha_type,
                            frequency=hardware_platform.frequency
                        )

                    for produced_data_tag in beha_dict[next_beha_tag].produced_data_split_dict:
                        for produced_data_split in beha_dict[next_beha_tag].produced_data_split_dict[produced_data_tag]:
                            data_dict[produced_data_tag].generated_split_location[produced_data_split].append(beha_dict[next_beha_tag].location[:-1]) 

    # --- Assign offline data (input activations and weights) to DDR locations ---
    if mem_enable:
        if greedy_flag:
            for large_split in data_dict[(0, 0)].splitted_corresponding_dict:
                for sub_split in data_dict[(0, 0)].splitted_corresponding_dict[large_split]:
                    # random.shuffle(ddrs)
                    data_dict[(0, 0)].generated_split_location[sub_split] = [ddr[:-1] for ddr in ddr_list]
            for data_tag in data_dict:
                if data_tag[0] == 1:
                    for large_split in data_dict[data_tag].splitted_corresponding_dict:
                        for sub_split in data_dict[data_tag].splitted_corresponding_dict[large_split]:
                            if sub_split in data_dict[data_tag].used_splitted_tag_dict:
                                # random.shuffle(ddrs)
                                chosen_ddr = min(ddr_workload_dict, key=ddr_workload_dict.get)
                                ddr_workload_dict[chosen_ddr] += np.prod([end - start + 1 for (start, end) in sub_split])
                                data_dict[data_tag].generated_split_location[sub_split] = [chosen_ddr[:-1]]
        else:
            for large_split in data_dict[(0, 0)].splitted_corresponding_dict:
                chosen_ddr = random.choice(ddr_list)[:-1]
                for sub_split in data_dict[(0, 0)].splitted_corresponding_dict[large_split]:
                    data_dict[(0, 0)].generated_split_location[sub_split] = [chosen_ddr]
            for data_tag in data_dict:
                if data_tag[0] == 1:
                    for large_split in data_dict[data_tag].splitted_corresponding_dict:
                        chosen_ddr = random.choice(ddr_list)[:-1]
                        for sub_split in data_dict[data_tag].splitted_corresponding_dict[large_split]:
                            if sub_split in data_dict[data_tag].used_splitted_tag_dict:
                                data_dict[data_tag].generated_split_location[sub_split] = [chosen_ddr]

    else:
        # No DDR: place offline data at all nodes in the first layer's region (broadcast)
        for large_split in data_dict[(0, 0)].splitted_corresponding_dict:
            data_dict[(0, 0)].generated_split_location[large_split] = [node_idx for node_idx in hardware_platform.nodes_set if node_idx[0] == layers_regions[0][0]]
            for sub_split in data_dict[(0, 0)].splitted_corresponding_dict[large_split]:
                data_dict[(0, 0)].generated_split_location[sub_split] = [node_idx for node_idx in hardware_platform.nodes_set if node_idx[0] == layers_regions[0][0]]

        for data_tag in data_dict:
            if data_tag[0] == 1:
                for large_split in data_dict[data_tag].splitted_corresponding_dict:
                    if large_split in data_dict[data_tag].used_splitted_tag_dict:
                        data_dict[data_tag].generated_split_location[large_split] = [beha_dict[used_tag].location[:-1] for used_tag in data_dict[data_tag].used_splitted_tag_dict[large_split]]
                    for sub_split in data_dict[data_tag].splitted_corresponding_dict[large_split]:
                        if sub_split in data_dict[data_tag].used_splitted_tag_dict:
                            data_dict[data_tag].generated_split_location[sub_split] = [beha_dict[used_tag].location[:-1] for used_tag in data_dict[data_tag].used_splitted_tag_dict[sub_split]]


def autoregreesive_dag(
    llm_layer_num: int,
    chiplets_per_layer: int
):
    """Build a cyclic DAG for autoregressive LLM decoding: layer N-1 -> layer 0."""
    model_dag = {}
    model_chiplets = {}
    for i in range(llm_layer_num - 1):
        model_dag[i] = [i + 1]
        model_chiplets[i] = chiplets_per_layer
    model_dag[llm_layer_num - 1] = [0] 
    model_chiplets[llm_layer_num - 1] = chiplets_per_layer

    return model_dag, model_chiplets


def vlm_dag(
    vit_layer_num: int,
    chiplets_per_vitlayer: int,
    llm_layer_num: int,
    chiplets_per_llmlayer: int
):
    """Build a DAG for VLM: ViT layers -> LLM layers with cyclic LLM decoding loop."""
    model_dag = {}
    model_chiplets = {}

    if vit_layer_num > 0:
        for i in range(vit_layer_num - 1):
            model_dag[i] = [i + 1]
            model_chiplets[i] = chiplets_per_vitlayer
        model_chiplets[vit_layer_num - 1] = chiplets_per_vitlayer

    if llm_layer_num > 0:
        if vit_layer_num > 0:
            model_dag[vit_layer_num - 1] = [vit_layer_num]
        for i in range(vit_layer_num, vit_layer_num + llm_layer_num - 1):
            model_dag[i] = [i + 1]
            model_chiplets[i] = chiplets_per_llmlayer
        model_dag[vit_layer_num + llm_layer_num - 1] = [vit_layer_num]
        model_chiplets[vit_layer_num + llm_layer_num - 1] = chiplets_per_llmlayer

    return model_dag, model_chiplets
