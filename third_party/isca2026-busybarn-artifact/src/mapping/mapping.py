
"""Core mapping logic: assigns partitioned behaviors to tensorcore and vector
units using greedy and priority-queue based allocation strategies."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../partition/'))
sys.path.append(os.path.join(file_path, '../platform/device/'))


from device import device
from beha_notation import beha_notation, beha_kind_dict
from data_notation import tensor_notation, tensor_slice_notation


from typing import List, Dict, Tuple, Set, Deque
from collections import deque
from copy import deepcopy
import random
random.seed(123)


def unit_pop(
    current_beha_tag: Tuple[int],
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation], 
    tensorcore_queue: Deque[Tuple[int]],
    vectorunit_queue: Deque[Tuple[int]]
):
    if beha_dict[current_beha_tag].beha_type in beha_kind_dict["tensorcore"]:
        beha_dict[current_beha_tag].location = tensorcore_queue.popleft()
        beha_dict[current_beha_tag].device = "tensorcore"
        for produced_data_tag in beha_dict[current_beha_tag].produced_data_split_dict:
            for produced_data_split in beha_dict[current_beha_tag].produced_data_split_dict[produced_data_tag]:
                data_dict[produced_data_tag].generated_split_location[produced_data_split].append(beha_dict[current_beha_tag].location[:-1])
        tensorcore_queue.append(beha_dict[current_beha_tag].location)
    elif beha_dict[current_beha_tag].beha_type in beha_kind_dict["vectorunit"]:
        beha_dict[current_beha_tag].location = vectorunit_queue.popleft()
        beha_dict[current_beha_tag].device = "vectorunit"
        for produced_data_tag in beha_dict[current_beha_tag].produced_data_split_dict:
            for produced_data_split in beha_dict[current_beha_tag].produced_data_split_dict[produced_data_tag]:
                data_dict[produced_data_tag].generated_split_location[produced_data_split].append(beha_dict[current_beha_tag].location[:-1])
        vectorunit_queue.append(beha_dict[current_beha_tag].location)
    else:
        raise ValueError("Unknown beha type")


def unit_pop_idx(
    current_beha_tag: Tuple[int],
    target_location: Tuple[int],
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation], 
    tensorcore_queue: Deque[Tuple[int]],
    vectorunit_queue: Deque[Tuple[int]]
):
    if beha_dict[current_beha_tag].beha_type in beha_kind_dict["tensorcore"]:
        beha_dict[current_beha_tag].location = target_location
        tensorcore_queue.remove(target_location)
        beha_dict[current_beha_tag].device = "tensorcore"
        for produced_data_tag in beha_dict[current_beha_tag].produced_data_split_dict:
            for produced_data_split in beha_dict[current_beha_tag].produced_data_split_dict[produced_data_tag]:
                data_dict[produced_data_tag].generated_split_location[produced_data_split].append(beha_dict[current_beha_tag].location[:-1])
        tensorcore_queue.append(beha_dict[current_beha_tag].location)
    elif beha_dict[current_beha_tag].beha_type in beha_kind_dict["vectorunit"]:
        beha_dict[current_beha_tag].location = target_location
        vectorunit_queue.remove(target_location)
        beha_dict[current_beha_tag].device = "vectorunit"
        for produced_data_tag in beha_dict[current_beha_tag].produced_data_split_dict:
            for produced_data_split in beha_dict[current_beha_tag].produced_data_split_dict[produced_data_tag]:
                data_dict[produced_data_tag].generated_split_location[produced_data_split].append(beha_dict[current_beha_tag].location[:-1])
        vectorunit_queue.append(beha_dict[current_beha_tag].location)
    else:
        raise ValueError("Unknown beha type", beha_dict[current_beha_tag].beha_type)        


def core_mapping(
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation], 
    hardware_platform: Dict[str, Dict[Tuple[int], device]],
    target_chiplets: List[int],
    target_layers: List[int] = None,
    ramdom_mapping: bool = False
):

    for tensor_tag in data_dict:
        tensor = data_dict[tensor_tag]
        for source_split in tensor.splitted_corresponding_dict:
            for split in tensor.splitted_corresponding_dict[source_split]:
                if split not in tensor.generated_split_location:
                    tensor.generated_split_location[split] = []


    available_tensorcore_queue = deque()
    for tensorcore_id in hardware_platform["tensorcore"]:
        if tensorcore_id[0] in target_chiplets:
            available_tensorcore_queue.append(tensorcore_id)
    available_tensorcore_queue_backup = deepcopy(available_tensorcore_queue)

    available_vectorunit_queue = deque()
    for vectorunit_id in hardware_platform["vectorunit"]:
        if vectorunit_id[0] in target_chiplets:
            available_vectorunit_queue.append(vectorunit_id)
    available_vectorunit_queue_backup = deepcopy(available_vectorunit_queue)

    last_oper_tag = 0
    for beha_tag in beha_dict:
        if target_layers:
            if beha_tag[2] not in target_layers:
                continue
        current_oper_tag = beha_tag[-2]
        if current_oper_tag != last_oper_tag:
            available_tensorcore_queue = deepcopy(available_tensorcore_queue_backup)
            available_vectorunit_queue = deepcopy(available_vectorunit_queue_backup)
            last_oper_tag = current_oper_tag

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

            '''
            offline greedy mapping: used device will be the last priority of the queue
            '''
        if ramdom_mapping:
            chosen_location = random.choice(available_tensorcore_queue)
            unit_pop_idx(
                current_beha_tag=beha_tag,
                target_location=chosen_location,
                beha_dict=beha_dict,
                data_dict=data_dict, 
                tensorcore_queue=available_tensorcore_queue,
                vectorunit_queue=available_vectorunit_queue
            )
        else:
            chosen_location = random.choice(available_tensorcore_queue)
            unit_pop_idx(
                current_beha_tag=beha_tag,
                target_location=chosen_location,
                beha_dict=beha_dict,
                data_dict=data_dict, 
                tensorcore_queue=available_tensorcore_queue,
                vectorunit_queue=available_vectorunit_queue
            )
            available_tensorcore_queue.remove(chosen_location)
            available_tensorcore_queue.append(chosen_location)   


    # for needed_tag in beha_dict[beha_tag].needed_tag_size_dict:
    #     if needed_tag == (-1,):
    return available_tensorcore_queue, available_vectorunit_queue      


def core_mapping_layers(
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation], 
    hardware_platform: Dict[str, Dict[Tuple[int], device]],
    target_chiplets: List[int],
    target_layers: List[int] = None,
    ramdom_mapping: bool = False
):

    for tensor_tag in data_dict:
        tensor = data_dict[tensor_tag]
        for source_split in tensor.splitted_corresponding_dict:
            for split in tensor.splitted_corresponding_dict[source_split]:
                if split not in tensor.generated_split_location:
                    tensor.generated_split_location[split] = []


    available_tensorcore_queue = deque()
    for tensorcore_id in hardware_platform["tensorcore"]:
        if tensorcore_id[0] in target_chiplets:
            available_tensorcore_queue.append(tensorcore_id)
    available_tensorcore_queue_backup = deepcopy(available_tensorcore_queue)

    available_vectorunit_queue = deque()
    for vectorunit_id in hardware_platform["vectorunit"]:
        if vectorunit_id[0] in target_chiplets:
            available_vectorunit_queue.append(vectorunit_id)
    available_vectorunit_queue_backup = deepcopy(available_vectorunit_queue)

    beha_tag_list = []
    last_oper_tag = 0
    for beha_tag in beha_dict:
        if target_layers:
            if beha_tag[2] not in target_layers:
                continue
            else:
                beha_tag_list.append(beha_tag)
        current_oper_tag = beha_tag[-2]
        if current_oper_tag != last_oper_tag:
            available_tensorcore_queue = deepcopy(available_tensorcore_queue_backup)
            available_vectorunit_queue = deepcopy(available_vectorunit_queue_backup)
            last_oper_tag = current_oper_tag

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

            '''
            offline greedy mapping: used device will be the last priority of the queue
            '''
        if ramdom_mapping:
            chosen_location = random.choice(available_tensorcore_queue)
            unit_pop_idx(
                current_beha_tag=beha_tag,
                target_location=chosen_location,
                beha_dict=beha_dict,
                data_dict=data_dict, 
                tensorcore_queue=available_tensorcore_queue,
                vectorunit_queue=available_vectorunit_queue
            )
        else:
            chosen_location = random.choice(available_tensorcore_queue)
            unit_pop_idx(
                current_beha_tag=beha_tag,
                target_location=chosen_location,
                beha_dict=beha_dict,
                data_dict=data_dict, 
                tensorcore_queue=available_tensorcore_queue,
                vectorunit_queue=available_vectorunit_queue
            )
            available_tensorcore_queue.remove(chosen_location)
            available_tensorcore_queue.append(chosen_location)   


    # for needed_tag in beha_dict[beha_tag].needed_tag_size_dict:
    #     if needed_tag == (-1,):
    return available_tensorcore_queue, available_vectorunit_queue, beha_tag_list     


def layer_to_chiplet_core_mapping(
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation], 
    hardware_platform: Dict[str, Dict[Tuple[int], device]],
    chiplet_list: List[int],
    layer_num: int,
    ramdom_mapping: bool = False
):

    cluster_number = len(chiplet_list)
    layer_per_cluster = layer_num // cluster_number

    tensorcore_queue_list = []
    vectorunit_queue_list = []
    beha_cluster_list = []
    for cluster_idx in range(cluster_number):
        target_chiplets = chiplet_list[cluster_idx]
        target_layers = []
        for intra_layer in range(layer_per_cluster):
            target_layers.append(cluster_idx * layer_per_cluster + intra_layer)
        tensorcore_queue, vectorunit_queue, beha_in_cluster = core_mapping_layers(
            beha_dict=beha_dict,
            data_dict=data_dict,
            hardware_platform=hardware_platform.modules_dict, 
            target_chiplets=target_chiplets,
            target_layers=target_layers,
            ramdom_mapping=ramdom_mapping
        )
        tensorcore_queue_list.append(tensorcore_queue)
        vectorunit_queue_list.append(vectorunit_queue)
        beha_cluster_list.append(beha_in_cluster)

    return chiplet_list[0][0], tensorcore_queue_list, vectorunit_queue_list, beha_cluster_list
