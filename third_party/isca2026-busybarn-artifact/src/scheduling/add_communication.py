
"""Communication inference: analyzes operator data dependencies to insert
communication events (unicast, multicast, reduction) and intermediate data
edges into the mapped schedule."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../src/partition/'))
sys.path.append(os.path.join(file_path, '../../src/scheduling/communication/topology/'))


from data_notation import tensor_notation, tensor_slice_notation, type_bytes
from beha_notation import beha_notation
from event_notation import event_notation, communication_notation, computation_notation
from net import net


from typing import List, Dict, Tuple, Set, Deque
from collections import defaultdict
import numpy as np
from copy import deepcopy
import random


# --- Behavior graph wiring ---

def add_beha_producers(
    beha_dict: Dict[Tuple[int], beha_notation]
):
    # Build reverse edges: for every consumer link, add the corresponding producer link.
    for beha_tag in beha_dict:
        for consumer_tag in beha_dict[beha_tag].consumer_tags:
            beha_dict[consumer_tag].producer_tags.add(beha_tag)


# --- Initial event graph construction (v1, without deduplication) ---

def build_event(
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net,
    event_dict: Dict[Tuple[int], event_notation],
    dijkstra_routing: bool = False,
    alpha: int = 100,
    beta: int = 1,
    gamma: int = 100
):
    """Create computation and communication events from behavior-to-data mappings.

    For each behavior, creates a computation event and checks its needed data
    splits. If data is remote, inserts a communication event with XY or Dijkstra
    routing. Returns (hops, comm_distances, comm_loads, tc_loads, vu_loads).
    """

    # Accumulated metrics returned to caller
    hops = 0
    communication_distances = 0
    communication_loads_dict = {link: 0 for link in hardware_platform.links_set}
    tensorcore_loads_dict = {tensorcore: 0 for tensorcore in hardware_platform.tensorcore_dict}
    vectorunit_loads_dict = {vectorunit: 0 for vectorunit in hardware_platform.vectorunit_dict}

    # Phase 1: create computation events and accumulate device loads
    for beha_tag in beha_dict:
        # TODO: add datatype by beha_dict[beha_tag].beha_datatag
        event_dict[beha_tag] = computation_notation(
            comp_name=beha_dict[beha_tag].beha_name,
            comp_tag=beha_tag, 
            comp_device=beha_dict[beha_tag].device,
            comp_location=beha_dict[beha_tag].location,
            comp_type=beha_dict[beha_tag].beha_type,
            comp_datashape=beha_dict[beha_tag].beha_datashape
        )
        computation_time = hardware_platform.modules_dict[event_dict[beha_tag].comp_device][event_dict[beha_tag].comp_location].working_time(
            source_datashape=event_dict[beha_tag].comp_datashape,
            beha_type=event_dict[beha_tag].comp_type,
            frequency=hardware_platform.frequency
        )
        if event_dict[beha_tag].comp_device ==  "tensorcore":
            tensorcore_loads_dict[event_dict[beha_tag].comp_location] += computation_time
        else:
            vectorunit_loads_dict[event_dict[beha_tag].comp_location] += computation_time

    # Phase 2: for each behavior, check data deps and create comm events if remote
    for beha_tag in beha_dict:
        communication_tag = (1,) + beha_tag[1:] + (event_dict[beha_tag].commid_allocator.allocate(),)
        target_position = beha_dict[beha_tag].location[:-1]
        for source_data_tag in beha_dict[beha_tag].needed_data_split_dict:
            for source_data_split in beha_dict[beha_tag].needed_data_split_dict[source_data_tag]:
                source_target_pair_distance = np.inf
                for source_position_candidate in data_dict[source_data_tag].generated_split_location[source_data_split]:
                    if source_position_candidate == target_position:
                        source_position = source_position_candidate
                        break
                    # Find nearest source candidate by distance
                    if dijkstra_routing:
                        current_pair_distance = hardware_platform.node_to_node_distance_dict[source_position_candidate][target_position]
                    else:
                        current_pair_distance = hardware_platform.node_to_node_manhattan_distance_dict[source_position_candidate][target_position]
                    if current_pair_distance < source_target_pair_distance:
                        source_target_pair_distance = current_pair_distance
                        source_position = source_position_candidate     

                if source_position == target_position:
                    # Data is local -- wire direct comp-to-comp dependency
                    for producer_tag in data_dict[source_data_tag].generated_splitted_tag_dict[source_data_split]:
                        if producer_tag == (-1, ):
                            continue
                        else:
                            event_dict[producer_tag].issue_set.add(beha_tag)
                            event_dict[beha_tag].dependency_set.add(producer_tag)

                else:
                    # Data is remote -- create a communication event and route it
                    event_dict[communication_tag] = communication_notation(
                        comm_name=beha_dict[beha_tag].beha_name,
                        comm_tag=communication_tag,
                        source_location=source_position,
                        target_location=target_position,
                        comm_bytes=int(np.prod([end - start + 1 for (start, end) in source_data_split])*type_bytes[data_dict[source_data_tag].data_type])
                    )
                    data_dict[source_data_tag].used_splitted_tag_dict[source_data_split].add(communication_tag)
                    data_dict[source_data_tag].used_splitted_tag_dict[source_data_split].remove(beha_tag)
                    communication_distances += hardware_platform.node_to_node_distance_function_dict[source_position][target_position](event_dict[communication_tag].comm_bytes)
                    event = event_dict[communication_tag]
                    # Route via Dijkstra (BALD) or XY
                    if dijkstra_routing:
                        dijkstra_path = hardware_platform.record_dijkstra_multicast_path(
                            comm_pairs=[[event.event_tag, event.source_location, {event.target_location}, event.comm_bytes]],
                            alpha=alpha, beta=beta, gamma=gamma, 
                        )[2]
                        event_dict[communication_tag].paths = dijkstra_path[event.event_tag]
                        event_dict[communication_tag].path_list = event.get_paths(dijkstra_path[event.event_tag])
                        event_dict[communication_tag].hops = hardware_platform.node_to_node_hop_dict[source_position][target_position]
                        hops += event_dict[communication_tag].hops * event_dict[communication_tag].comm_bytes
                        event_dict[communication_tag].communication_distances = hardware_platform.node_to_node_distance_function_dict[source_position][target_position](event_dict[communication_tag].comm_bytes)
                        communication_distances += event_dict[communication_tag].communication_distances
                        routed_path = dijkstra_path[event.event_tag]
                    else:
                        routed_path = hardware_platform.xy_paths[event.source_location][event.target_location]
                        xy_path = event.get_paths(routed_path)
                        event_dict[communication_tag].paths = routed_path
                        event_dict[communication_tag].path_list = xy_path
                        event_dict[communication_tag].hops = hardware_platform.node_to_node_manhattan_hops_dict[source_position][target_position]
                        hops += event_dict[communication_tag].hops * event_dict[communication_tag].comm_bytes
                        event_dict[communication_tag].communication_distances = hardware_platform.node_to_node_manhattan_distance_function_dict[source_position][target_position](event_dict[communication_tag].comm_bytes)
                        communication_distances += event_dict[communication_tag].communication_distances

                    for root in routed_path:
                        for leaf in routed_path[root]:
                            used_link = (root, leaf)
                            communication_loads_dict[used_link] += np.ceil(event.comm_bytes / hardware_platform.links_dict[used_link].bandwidth)                                        

                    for producer_tag in data_dict[source_data_tag].generated_splitted_tag_dict[source_data_split]:
                        if producer_tag == (-1, ):
                            continue
                        else:
                            event_dict[producer_tag].issue_set.add(communication_tag)
                            event_dict[communication_tag].dependency_set.add(producer_tag)
                    event_dict[communication_tag].issue_set.add(beha_tag)
                    event_dict[beha_tag].dependency_set.add(communication_tag)

                    communication_tag = communication_tag[:-1] + (event_dict[beha_tag].commid_allocator.allocate(),)

    return hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict


def update_event(
    target_event_tags: List[Tuple[int]],
    target_devices: List[Tuple[int]],
    hops: int,
    communication_distances: float,
    communication_loads_dict: Dict[Tuple[int], float],
    tensorcore_loads_dict: Dict[Tuple[int], float],
    vectorunit_loads_dict: Dict[Tuple[int], float],
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net,
    event_dict: Dict[Tuple[int], event_notation],
    dijkstra_routing: bool = False,
    alpha: int = 100,
    beta: int = 1,
    gamma: int = 100,
    mem_enable: bool = True
):
    """Relocate behaviors to new devices and rebuild affected communication.

    Used during SA: moves target_event_tags to target_devices, tears down
    all connected communication events, then rebuilds them from scratch.
    Three phases: (1) relocate + update loads, (2) teardown old comms,
    (3) rebuild comms for affected behaviors.
    """

    targets_number = len(target_event_tags)
    old_behas = set()
    for target_idx in range(targets_number):
        target_event_tag = target_event_tags[target_idx]
        target_device = target_devices[target_idx]

        old_position = event_dict[target_event_tag].comp_location
        old_computation_time = hardware_platform.modules_dict[event_dict[target_event_tag].comp_device][old_position].working_time(
            source_datashape=event_dict[target_event_tag].comp_datashape,
            beha_type=event_dict[target_event_tag].comp_type,
            frequency=hardware_platform.frequency
        )

        beha_dict[target_event_tag].location = target_device
        event_dict[target_event_tag].comp_location = target_device
        new_computation_time = hardware_platform.modules_dict[event_dict[target_event_tag].comp_device][event_dict[target_event_tag].comp_location].working_time(
            source_datashape=event_dict[target_event_tag].comp_datashape,
            beha_type=event_dict[target_event_tag].comp_type,
            frequency=hardware_platform.frequency
        )
        # Update device load accounting: subtract old, add new
        if event_dict[target_event_tag].comp_device == "tensorcore":
            tensorcore_loads_dict[old_position] -= old_computation_time
            tensorcore_loads_dict[target_device] += new_computation_time
        else:
            vectorunit_loads_dict[old_position] -= old_computation_time
            vectorunit_loads_dict[target_device] += new_computation_time

        # Update data location tracking for produced data splits
        for produced_data_tag in beha_dict[target_event_tag].produced_data_split_dict:
            for produced_data_split in beha_dict[target_event_tag].produced_data_split_dict[produced_data_tag]:
                data_dict[produced_data_tag].generated_split_location[produced_data_split].remove(old_position[:-1])
                data_dict[produced_data_tag].generated_split_location[produced_data_split].append(target_device[:-1])

        if mem_enable:
            pass
        else:
            for needed_data_tag in beha_dict[target_event_tag].needed_data_split_dict:
                if needed_data_tag[0] == 1:
                    for needed_data_split in beha_dict[target_event_tag].needed_data_split_dict[needed_data_tag]:
                        data_dict[needed_data_tag].generated_split_location[needed_data_split].remove(old_position[:-1])
                        data_dict[needed_data_tag].generated_split_location[needed_data_split].append(target_device[:-1])

        for issued_event_tag in event_dict[target_event_tag].issue_set:
            if issued_event_tag[0] == 1 and issued_event_tag in event_dict:
                for next_computed_event_tag in event_dict[issued_event_tag].issue_set:
                    if next_computed_event_tag[0] == 0:
                        old_behas.add(next_computed_event_tag)
                        event_dict[next_computed_event_tag].dependency_set.remove(issued_event_tag)
                    else:
                        raise ValueError(f"Communication event issued another communication event: {next_computed_event_tag} issued by {issued_event_tag}")
                event_dict[(0, ) + issued_event_tag[1:-1]].commid_allocator.recycle(issued_event_tag[-1])
                for root in event_dict[issued_event_tag].paths:
                    for leaf in event_dict[issued_event_tag].paths[root]:
                        used_link = (root, leaf)
                        communication_loads_dict[used_link] -= np.ceil(event_dict[issued_event_tag].comm_bytes / hardware_platform.links_dict[used_link].bandwidth)    
                hops -= event_dict[issued_event_tag].hops * event_dict[issued_event_tag].comm_bytes
                communication_distances -= event_dict[issued_event_tag].communication_distances
                del event_dict[issued_event_tag]
            elif issued_event_tag[0] == 0:
                old_behas.add(issued_event_tag)
                event_dict[issued_event_tag].dependency_set.remove(target_event_tag)
            else:
                pass
        event_dict[target_event_tag].issue_set = set()
        old_behas.add(target_event_tag)            

        # Clear stale refs from dependency_set of newly added old_behas
        for old_event_tag in old_behas:
            for denpendent_event_tag in event_dict[old_event_tag].dependency_set:
                if denpendent_event_tag[0] == 1 and denpendent_event_tag in event_dict:
                    for last_computed_event_tag in event_dict[denpendent_event_tag].dependency_set:
                        if last_computed_event_tag[0] == 0:
                            event_dict[last_computed_event_tag].issue_set.remove(denpendent_event_tag)
                        else:
                            raise ValueError(f"Communication event depended on another communication event: {last_computed_event_tag} depended by {denpendent_event_tag}")
                    event_dict[(0, ) + denpendent_event_tag[1:-1]].commid_allocator.recycle(denpendent_event_tag[-1])
                    for root in event_dict[denpendent_event_tag].paths:
                        for leaf in event_dict[denpendent_event_tag].paths[root]:
                            used_link = (root, leaf)
                            communication_loads_dict[used_link] -= np.ceil(event_dict[denpendent_event_tag].comm_bytes / hardware_platform.links_dict[used_link].bandwidth)
                    hops -= event_dict[denpendent_event_tag].hops * event_dict[denpendent_event_tag].comm_bytes
                    communication_distances -= event_dict[denpendent_event_tag].communication_distances
                    del event_dict[denpendent_event_tag]
                elif denpendent_event_tag[0] == 0:
                    event_dict[denpendent_event_tag].issue_set.remove(old_event_tag)
                else:
                    pass
            event_dict[old_event_tag].dependency_set = set()

    for beha_tag in old_behas:
        target_position = beha_dict[beha_tag].location[:-1]
        for source_data_tag in beha_dict[beha_tag].needed_data_split_dict:
            for source_data_split in beha_dict[beha_tag].needed_data_split_dict[source_data_tag]:
                source_target_pair_distance = np.inf
                for source_position_candidate in data_dict[source_data_tag].generated_split_location[source_data_split]:
                    if source_position_candidate == target_position:
                        source_position = source_position_candidate
                        break
                    # Track fwd comm in used_splitted_tag_dict so group routing includes it
                    if dijkstra_routing:
                        current_pair_distance = hardware_platform.node_to_node_distance_dict[source_position_candidate][target_position]
                    else:
                        current_pair_distance = hardware_platform.node_to_node_manhattan_distance_dict[source_position_candidate][target_position]
                    if current_pair_distance < source_target_pair_distance:
                        source_target_pair_distance = current_pair_distance
                        source_position = source_position_candidate     

                if source_position == target_position: 
                    for producer_tag in data_dict[source_data_tag].generated_splitted_tag_dict[source_data_split]:
                        if producer_tag == (-1, ):
                            continue
                        else:
                            event_dict[producer_tag].issue_set.add(beha_tag)
                            event_dict[beha_tag].dependency_set.add(producer_tag)

                else:
                    communication_tag = (1,) + beha_tag[1:] + (event_dict[beha_tag].commid_allocator.allocate(),)
                    event_dict[communication_tag] = communication_notation(
                        comm_name=beha_dict[beha_tag].beha_name,
                        comm_tag=communication_tag,
                        source_location=source_position,
                        target_location=target_position,
                        comm_bytes=int(np.prod([end - start + 1 for (start, end) in source_data_split])*type_bytes[data_dict[source_data_tag].data_type])
                    )
                    old_idx = set()
                    for all_idx in data_dict[source_data_tag].used_splitted_tag_dict[source_data_split]:
                        if all_idx not in event_dict:
                            old_idx.add(all_idx)
                    for old_idx_tag in old_idx:
                        data_dict[source_data_tag].used_splitted_tag_dict[source_data_split].remove(old_idx_tag)                 
                    data_dict[source_data_tag].used_splitted_tag_dict[source_data_split].add(communication_tag)
                    event = event_dict[communication_tag]
                    if dijkstra_routing:
                        dijkstra_path = hardware_platform.record_dijkstra_multicast_path(
                            comm_pairs=[[event.event_tag, event.source_location, {event.target_location}, event.comm_bytes]],
                            alpha=alpha, beta=beta, gamma=gamma, 
                        )[2]
                        event_dict[communication_tag].paths = dijkstra_path[event.event_tag]
                        event_dict[communication_tag].path_list = event.get_paths(dijkstra_path[event.event_tag])
                        event_dict[communication_tag].hops = hardware_platform.node_to_node_hop_dict[source_position][target_position]
                        hops += event_dict[communication_tag].hops * event_dict[communication_tag].comm_bytes
                        event_dict[communication_tag].communication_distances = hardware_platform.node_to_node_distance_function_dict[source_position][target_position](event_dict[communication_tag].comm_bytes)
                        communication_distances += event_dict[communication_tag].communication_distances
                        routed_path = dijkstra_path[event.event_tag]
                    else:
                        routed_path = hardware_platform.xy_paths[event.source_location][event.target_location]
                        xy_path = event.get_paths(routed_path)
                        event_dict[communication_tag].paths = routed_path
                        event_dict[communication_tag].path_list = xy_path
                        event_dict[communication_tag].hops = hardware_platform.node_to_node_manhattan_hops_dict[source_position][target_position]
                        hops += event_dict[communication_tag].hops * event_dict[communication_tag].comm_bytes
                        event_dict[communication_tag].communication_distances = hardware_platform.node_to_node_manhattan_distance_function_dict[source_position][target_position](event_dict[communication_tag].comm_bytes)
                        communication_distances += event_dict[communication_tag].communication_distances

                    for root in routed_path:
                        for leaf in routed_path[root]:
                            used_link = (root, leaf)
                            communication_loads_dict[used_link] += np.ceil(event.comm_bytes / hardware_platform.links_dict[used_link].bandwidth)                                        

                    for producer_tag in data_dict[source_data_tag].generated_splitted_tag_dict[source_data_split]:
                        if producer_tag == (-1, ):
                            continue
                        else:
                            event_dict[producer_tag].issue_set.add(communication_tag)
                            event_dict[communication_tag].dependency_set.add(producer_tag)
                    event_dict[communication_tag].issue_set.add(beha_tag)
                    event_dict[beha_tag].dependency_set.add(communication_tag)

    return hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict        


def update_offlinedata(
    target_data_tags: Tuple[int],
    target_data_split: Tuple[Tuple[int]],
    target_devices: Tuple[int],
    hops: int,    
    communication_distances: float,
    communication_loads_dict: Dict[Tuple[int], float],
    tensorcore_loads_dict: Dict[Tuple[int], float],
    vectorunit_loads_dict: Dict[Tuple[int], float],
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net, 
    event_dict: Dict[Tuple[int], event_notation],
    dijkstra_routing: bool = False,
    alpha: int = 100,
    beta: int = 1,
    gamma: int = 100,
    mem_enable: bool = True 
):
    if mem_enable:
        pass
    else:
        raise ValueError("update_offlinedata is not supported when mem_enable is False, please set mem_enable to True.")

    for communication_tag in data_dict[target_data_tags].used_splitted_tag_dict[target_data_split]:
        if communication_tag[0] == 1:
            old_paths = event_dict[communication_tag].paths
            for root in old_paths:
                for leaf in old_paths[root]:
                    used_link = (root, leaf)
                    communication_loads_dict[used_link] -= np.ceil(event_dict[communication_tag].comm_bytes / hardware_platform.links_dict[used_link].bandwidth)
            old_hops = event_dict[communication_tag].hops
            hops -= old_hops * event_dict[communication_tag].comm_bytes
            communication_distances -= event_dict[communication_tag].communication_distances

            event_dict[communication_tag].source_location = target_devices[:-1]
            source_location = event_dict[communication_tag].source_location
            target_location = event_dict[communication_tag].target_location
            if dijkstra_routing:
                dijkstra_path = hardware_platform.record_dijkstra_multicast_path(
                    comm_pairs=[[communication_tag, source_location, {target_location}, event_dict[communication_tag].comm_bytes]],
                    alpha=alpha, beta=beta, gamma=gamma, 
                )[2]
                event_dict[communication_tag].paths = dijkstra_path[communication_tag]
                event_dict[communication_tag].path_list = event_dict[communication_tag].get_paths(dijkstra_path[communication_tag])
                event_dict[communication_tag].hops = hardware_platform.node_to_node_hop_dict[source_location][target_location]
                hops += event_dict[communication_tag].hops * event_dict[communication_tag].comm_bytes
                event_dict[communication_tag].communication_distances = hardware_platform.node_to_node_distance_function_dict[source_location][target_location](event_dict[communication_tag].comm_bytes)
                communication_distances += event_dict[communication_tag].communication_distances
                routed_path = dijkstra_path[communication_tag]
            else:
                routed_path = hardware_platform.xy_paths[source_location][target_location]
                xy_path = event_dict[communication_tag].get_paths(routed_path)
                event_dict[communication_tag].paths = routed_path
                event_dict[communication_tag].path_list = xy_path
                event_dict[communication_tag].hops = hardware_platform.node_to_node_manhattan_hops_dict[source_location][target_location]
                hops += event_dict[communication_tag].hops * event_dict[communication_tag].comm_bytes
                event_dict[communication_tag].communication_distances = hardware_platform.node_to_node_manhattan_distance_function_dict[source_location][target_location](event_dict[communication_tag].comm_bytes)
                communication_distances += event_dict[communication_tag].communication_distances

            for root in routed_path:
                for leaf in routed_path[root]:
                    used_link = (root, leaf)
                    communication_loads_dict[used_link] += np.ceil(event_dict[communication_tag].comm_bytes / hardware_platform.links_dict[used_link].bandwidth)

        else:
            raise ValueError(f"Communication event {communication_tag} is not a communication event, but a computation event.")

    return hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict        


def add_mediumdata(
    data_tags: List[Tuple[int]],
    ddr_chiplets: List[int],
    hops: int,    
    communication_distances: float,
    communication_loads_dict: Dict[Tuple[int], float],
    tensorcore_loads_dict: Dict[Tuple[int], float],
    vectorunit_loads_dict: Dict[Tuple[int], float],    
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net, 
    event_dict: Dict[Tuple[int], event_notation],
    dijkstra_routing: bool = False,
    alpha: int = 100,
    beta: int = 1,
    gamma: int = 100, 
    random_flag: bool = False,
):

    available_ddrs = []
    ddrs_load = {}
    for ddr_idx in hardware_platform.ddr_dict:
        if ddr_idx[0] in ddr_chiplets:
            available_ddrs.append(ddr_idx)
            ddrs_load[ddr_idx] = 0

    """
    Add medium data to the data_dict and beha_dict, and update the event_dict accordingly.
    """
    for data_tag in data_tags:
        for producer_tag in data_dict[data_tag].generated_tag_splitted_dict:
            for data_split in data_dict[data_tag].generated_tag_splitted_dict[producer_tag]:
                producer_location = event_dict[producer_tag].comp_location[:-1]
                producer_beha_tag = producer_tag
                communication_tag = (1,) + producer_beha_tag[1:] + (event_dict[producer_beha_tag].commid_allocator.allocate(),)
                if random_flag:
                    candiate_position = random.choice(available_ddrs)
                    target_ddr = candiate_position[:-1]
                else:
                    if dijkstra_routing:
                        candiate_position = None
                        min_distance = np.inf
                        min_capacity = 0
                        for ddr_idx in available_ddrs:
                            current_distance = hardware_platform.node_to_node_distance_dict[producer_location][ddr_idx[:-1]]
                            if current_distance < min_distance:
                                min_distance = current_distance
                                candiate_position = ddr_idx
                            elif current_distance == min_distance:
                                if ddrs_load[ddr_idx] < min_capacity:
                                    min_capacity = ddrs_load[ddr_idx]
                                    candiate_position = ddr_idx
                            else:
                                continue
                        target_ddr = candiate_position[:-1]
                    else:
                        candiate_position = None
                        min_distance = np.inf
                        for ddr_idx in available_ddrs:
                            current_distance = hardware_platform.node_to_node_manhattan_distance_dict[producer_location][ddr_idx[:-1]]
                            if current_distance < min_distance:
                                min_distance = current_distance
                                candiate_position = ddr_idx
                            else:
                                continue
                        target_ddr = candiate_position[:-1]


                event_dict[communication_tag] = communication_notation(
                    comm_name=beha_dict[producer_beha_tag].beha_name,
                    comm_tag=communication_tag,
                    source_location=producer_location,
                    target_location=target_ddr,
                    comm_bytes=int(np.prod([end - start + 1 for (start, end) in data_split])*type_bytes[data_dict[data_tag].data_type])
                )
                ddrs_load[candiate_position] += event_dict[communication_tag].comm_bytes

                data_dict[data_tag].used_splitted_tag_dict[data_split].add(communication_tag)
                communication_distances += hardware_platform.node_to_node_distance_function_dict[producer_location][target_ddr](event_dict[communication_tag].comm_bytes)
                event = event_dict[communication_tag]
                if dijkstra_routing:
                    dijkstra_path = hardware_platform.record_dijkstra_multicast_path(
                        comm_pairs=[[event.event_tag, event.source_location, {event.target_location}, event.comm_bytes]],
                        alpha=alpha, beta=beta, gamma=gamma, 
                    )[2]
                    event_dict[communication_tag].paths = dijkstra_path[event.event_tag]
                    event_dict[communication_tag].path_list = event.get_paths(dijkstra_path[event.event_tag])
                    event_dict[communication_tag].hops = hardware_platform.node_to_node_hop_dict[producer_location][target_ddr]
                    hops += event_dict[communication_tag].hops * event_dict[communication_tag].comm_bytes
                    event_dict[communication_tag].communication_distances = hardware_platform.node_to_node_distance_function_dict[producer_location][target_ddr](event_dict[communication_tag].comm_bytes)
                    communication_distances += event_dict[communication_tag].communication_distances
                    routed_path = dijkstra_path[event.event_tag]
                else:
                    routed_path = hardware_platform.xy_paths[event.source_location][event.target_location]
                    xy_path = event.get_paths(routed_path)
                    event_dict[communication_tag].paths = routed_path
                    event_dict[communication_tag].path_list = xy_path
                    event_dict[communication_tag].hops = hardware_platform.node_to_node_manhattan_hops_dict[producer_location][target_ddr]
                    hops += event_dict[communication_tag].hops * event_dict[communication_tag].comm_bytes
                    event_dict[communication_tag].communication_distances = hardware_platform.node_to_node_manhattan_distance_function_dict[producer_location][target_ddr](event_dict[communication_tag].comm_bytes)
                    communication_distances += event_dict[communication_tag].communication_distances

                for root in routed_path:
                    for leaf in routed_path[root]:
                        used_link = (root, leaf)
                        communication_loads_dict[used_link] += np.ceil(event.comm_bytes / hardware_platform.links_dict[used_link].bandwidth)                                        

                event_dict[producer_tag].issue_set.add(communication_tag)
                event_dict[communication_tag].dependency_set.add(producer_tag)

    return hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict


def add_broadcast(
    data_tags: List[Tuple[int]],
    ddr_chiplets: List[int],
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net, 
    event_dict: Dict[Tuple[int], event_notation],
    dijkstra_routing: bool = False,
    alpha: int = 100,
    beta: int = 1,
    gamma: int = 100, 
):

    available_ddrs = []
    ddrs = set()
    for ddr_idx in hardware_platform.ddr_dict:
        if ddr_idx[0] in ddr_chiplets:
            available_ddrs.append(ddr_idx)
            ddrs.add(ddr_idx[:-1])

    """
    Add medium data to the data_dict and beha_dict, and update the event_dict accordingly.
    """
    for data_tag in data_tags:
        for producer_tag in data_dict[data_tag].generated_tag_splitted_dict:
            for data_split in data_dict[data_tag].generated_tag_splitted_dict[producer_tag]:
                producer_location = event_dict[producer_tag].comp_location[:-1]
                producer_beha_tag = producer_tag
                communication_tag = (1,) + producer_beha_tag[1:] + (event_dict[producer_beha_tag].commid_allocator.allocate(),)

                event_dict[communication_tag] = communication_notation(
                    comm_name=beha_dict[producer_beha_tag].beha_name,
                    comm_tag=communication_tag,
                    source_location=producer_location,
                    target_location=ddrs,
                    comm_bytes=int(np.prod([end - start + 1 for (start, end) in data_split])*type_bytes[data_dict[data_tag].data_type])
                )

                event = event_dict[communication_tag]
                if dijkstra_routing:
                    dijkstra_path = hardware_platform.record_dijkstra_multicast_path(
                        comm_pairs=[[event.event_tag, event.source_location, ddrs, event.comm_bytes]],
                        alpha=alpha, beta=beta, gamma=gamma, 
                    )[2]
                    event_dict[communication_tag].paths = dijkstra_path[event.event_tag]
                    event_dict[communication_tag].path_list = event.get_paths(dijkstra_path[event.event_tag])
                else:
                    routed_path = hardware_platform.xy_multicast_path(
                        source_node_idx = event.source_location,
                        target_nodes_set = ddrs
                    )
                    xy_path = event.get_paths(routed_path)
                    event_dict[communication_tag].paths = routed_path
                    event_dict[communication_tag].path_list = xy_path

                event_dict[producer_tag].issue_set.add(communication_tag)
                event_dict[communication_tag].dependency_set.add(producer_tag)


def _route_comm(event_dict, communication_tag, source_position, target_position,
                hardware_platform, dijkstra_routing, alpha, beta, gamma):
    """Route a communication event and return (hops_delta, dist_delta, routed_path)."""
    hops_delta = 0
    dist_delta = 0
    event = event_dict[communication_tag]
    if dijkstra_routing:
        dijkstra_path = hardware_platform.record_dijkstra_multicast_path(
            comm_pairs=[[event.event_tag, event.source_location, {event.target_location}, event.comm_bytes]],
            alpha=alpha, beta=beta, gamma=gamma,
        )[2]
        event.paths = dijkstra_path[event.event_tag]
        event.path_list = event.get_paths(dijkstra_path[event.event_tag])
        event.hops = hardware_platform.node_to_node_hop_dict[source_position][target_position]
        hops_delta += event.hops * event.comm_bytes
        event.communication_distances = hardware_platform.node_to_node_distance_function_dict[source_position][target_position](event.comm_bytes)
        dist_delta += event.communication_distances
        routed_path = dijkstra_path[event.event_tag]
    else:
        routed_path = hardware_platform.xy_paths[event.source_location][event.target_location]
        event.paths = routed_path
        event.path_list = event.get_paths(routed_path)
        event.hops = hardware_platform.node_to_node_manhattan_hops_dict[source_position][target_position]
        hops_delta += event.hops * event.comm_bytes
        event.communication_distances = hardware_platform.node_to_node_manhattan_distance_function_dict[source_position][target_position](event.comm_bytes)
        dist_delta += event.communication_distances
    return hops_delta, dist_delta, routed_path


def _undo_init():
    """Create an empty undo record."""
    return {
        'hops': None,
        'comm_dist': None,
        'tc_loads': {},
        'vu_loads': {},
        'comm_loads': {},
        'beha_locs': {},
        'comp_locs': {},
        'data_gen_locs': {},
        'data_used': {},
        'issue_sets': {},
        'dep_sets': {},
        'deleted_comms': {},
        'created_comms': set(),
        'allocators': {},
        'platform_link_loads': None,
        'platform_link_counts': None,
    }


def _apply_barrier_sync_for_group(event_dict, data_dict, data_tag, undo=None,
                                  exclude_comms=None):
    """Barrier-sync one inter-function comm group identified by data_tag.

    Collects all communication events (anchor + one-hop forwarding) carrying
    any split of data_tag and cross-wires them as a barrier:
      - Every comm waits for all comp events that directly feed any anchor comm
        in the group (group_senders).
      - Every comp event that depends on any comm in the group (group_receivers)
        is made to wait for all comms in the group.

    Args:
        undo: optional undo-log dict; issue_set / dep_set snapshots are saved
              with copy-on-first-write before any mutation (for SA revert).
        exclude_comms: optional set of comm tags to exclude from this group.
              Used to deduplicate comms shared across multiple data_tag groups
              (e.g., matadd comms registered under both ifmap and ofmap).
    """
    # Step 1: find anchor_comms via used_splitted_tag_dict
    anchor_comms = set()
    for split in data_dict[data_tag].used_splitted_tag_dict:
        for ct in data_dict[data_tag].used_splitted_tag_dict[split]:
            if ct[0] == 1 and ct in event_dict:
                anchor_comms.add(ct)
    if exclude_comms:
        anchor_comms -= exclude_comms
    if not anchor_comms:
        return

    # Step 2: expand one level to fwd_comms (anchor → fwd)
    fwd_comms = set()
    for ac in anchor_comms:
        for iss in event_dict[ac].issue_set:
            if iss[0] == 1 and iss in event_dict:
                fwd_comms.add(iss)
    if exclude_comms:
        fwd_comms -= exclude_comms
    group_comms = anchor_comms | fwd_comms

    # Step 3: group_senders = comp events in any anchor_comm's dependency_set
    group_senders = set()
    for ac in anchor_comms:
        for dep in event_dict[ac].dependency_set:
            if dep[0] == 0:
                group_senders.add(dep)

    # Step 4: group_receivers = comp events issued by any comm in the group
    group_receivers = set()
    # Step 6: cross-wire barrier, skipping edges that already exist
    for ct in group_comms:
        # Also find forward comms (children of anchor comms)
        for iss in event_dict[ct].issue_set:
            if iss[0] == 0:
                group_receivers.add(iss)

    # Step 5: exclude comp events that are both senders and receivers
    overlap = group_senders & group_receivers
    group_senders = group_senders - overlap
    group_receivers = group_receivers - overlap

    for ct in group_comms:
        for sc in group_senders:
            if sc not in event_dict[ct].dependency_set:
                if undo is not None and sc not in undo['issue_sets']:
                    undo['issue_sets'][sc] = set(event_dict[sc].issue_set)
                if undo is not None and ct not in undo['dep_sets']:
                    undo['dep_sets'][ct] = set(event_dict[ct].dependency_set)
                event_dict[ct].dependency_set.add(sc)
                event_dict[sc].issue_set.add(ct)
        for rc in group_receivers:
            if rc not in event_dict[ct].issue_set:
                if undo is not None and ct not in undo['issue_sets']:
                    undo['issue_sets'][ct] = set(event_dict[ct].issue_set)
                if undo is not None and rc not in undo['dep_sets']:
                    undo['dep_sets'][rc] = set(event_dict[rc].dependency_set)
                event_dict[ct].issue_set.add(rc)
                event_dict[rc].dependency_set.add(ct)


def apply_barrier_sync_all(event_dict, data_dict):
    """Apply barrier sync to all data_tag groups, preventing cycles.

    For each communication group (data_tag), the barrier enforces:
        all senders finish -> all comms execute -> all receivers start

    Cycle prevention: before adding barrier edges for each group, a reverse
    BFS from senders finds all their ancestors in the current event graph
    (including barrier edges committed by prior groups).  Any receiver that
    is an ancestor of a sender would create a cycle (receiver->...->sender
    ->comm->receiver) and is excluded from that group's barrier.
    """
    # ── Pass 1: collect anchor_comms and group_comms per data_tag ──
    group_anchors = {}
    group_comms_map = {}
    for dt in data_dict:
        anchors = set()
        for split in data_dict[dt].used_splitted_tag_dict:
            for ct in data_dict[dt].used_splitted_tag_dict[split]:
                if ct[0] == 1 and ct in event_dict:
                    anchors.add(ct)
        if not anchors:
            continue
        fwd = set()
        for ac in anchors:
            for iss in event_dict[ac].issue_set:
                if iss[0] == 1 and iss in event_dict:
                    fwd.add(iss)
        group_anchors[dt] = anchors
        group_comms_map[dt] = anchors | fwd

    # ── Pass 2: find shared comms (registered in multiple groups) ──
    comm_to_groups = {}
    for dt, anchors in group_anchors.items():
        for ct in anchors:
            comm_to_groups.setdefault(ct, []).append(dt)
    shared_comms = {ct for ct, dts in comm_to_groups.items() if len(dts) > 1}

    # Exclude shared comms from all but the last group that claims them.
    dt_list = sorted(group_anchors.keys())
    assigned = set()
    group_exclude_comms = {dt: set() for dt in dt_list}
    for dt in reversed(dt_list):
        my_shared = shared_comms & group_anchors[dt]
        group_exclude_comms[dt] = my_shared & assigned
        assigned |= my_shared

    # ── Pass 3: apply barrier per group with reachability-based exclusion ──
    for dt in dt_list:
        excl = group_exclude_comms[dt]
        anchors = group_anchors[dt] - excl
        if not anchors:
            continue
        fwd = set()
        for ac in anchors:
            for iss in event_dict[ac].issue_set:
                if iss[0] == 1 and iss in event_dict and iss not in excl:
                    fwd.add(iss)
        g_comms = anchors | fwd

        # Collect senders/receivers for this group
        senders = set()
        for ac in anchors:
            for dep in event_dict[ac].dependency_set:
                if dep[0] == 0:
                    senders.add(dep)
        receivers = set()
        # Cross-wire barrier: sender→comm and comm→receiver
        for ct in g_comms:
            for iss in event_dict[ct].issue_set:
                if iss[0] == 0:
                    receivers.add(iss)
        # Exclude within-group overlap (allreduce: comp is both sender & receiver)
        overlap = senders & receivers
        senders -= overlap
        receivers -= overlap

        # Cycle check A — exclude senders reachable from group comms.
        downstream = set()
        bfs_queue = list(g_comms)
        while bfs_queue:
            tag = bfs_queue.pop()
            if tag in downstream:
                continue
            downstream.add(tag)
            if tag in event_dict:
                for iss in event_dict[tag].issue_set:
                    if iss not in downstream:
                        bfs_queue.append(iss)
        senders -= downstream

        # Cycle check B — exclude receivers that are ancestors of senders
        ancestors = set()
        bfs_queue = list(senders) + list(g_comms)
        while bfs_queue:
            tag = bfs_queue.pop()
            if tag in ancestors:
                continue
            ancestors.add(tag)
            if tag in event_dict:
                for dep in event_dict[tag].dependency_set:
                    if dep not in ancestors:
                        bfs_queue.append(dep)
        receivers -= ancestors

        for ct in g_comms:
            for sc in senders:
                if sc not in event_dict[ct].dependency_set:
                    event_dict[ct].dependency_set.add(sc)
                    event_dict[sc].issue_set.add(ct)
            for rc in receivers:
                if rc not in event_dict[ct].issue_set:
                    event_dict[ct].issue_set.add(rc)
                    event_dict[rc].dependency_set.add(ct)


def build_event_v2(
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net,
    event_dict: Dict[Tuple[int], event_notation],
    dijkstra_routing: bool = False,
    alpha: int = 100,
    beta: int = 1,
    gamma: int = 100,
    barrier_sync: bool = False
):


    # communication_distances: total communication distance (message pass time)
    hops = 0
    communication_distances = 0
    if dijkstra_routing:
        communication_loads_dict = {}
    else:
        communication_loads_dict = {link: 0 for link in hardware_platform.links_set}
    tensorcore_loads_dict = {tensorcore: 0 for tensorcore in hardware_platform.tensorcore_dict}
    vectorunit_loads_dict = {vectorunit: 0 for vectorunit in hardware_platform.vectorunit_dict}

    for beha_tag in beha_dict:
        event_dict[beha_tag] = computation_notation(
            comp_name=beha_dict[beha_tag].beha_name,
            comp_tag=beha_tag,
            comp_device=beha_dict[beha_tag].device,
            comp_location=beha_dict[beha_tag].location,
            comp_type=beha_dict[beha_tag].beha_type,
            comp_datashape=beha_dict[beha_tag].beha_datashape
        )
        computation_time = hardware_platform.modules_dict[event_dict[beha_tag].comp_device][event_dict[beha_tag].comp_location].working_time(
            source_datashape=event_dict[beha_tag].comp_datashape,
            beha_type=event_dict[beha_tag].comp_type,
            frequency=hardware_platform.frequency
        )
        if event_dict[beha_tag].comp_device ==  "tensorcore":
            tensorcore_loads_dict[event_dict[beha_tag].comp_location] += computation_time
        else:
            vectorunit_loads_dict[event_dict[beha_tag].comp_location] += computation_time

    # Phase 1: collect local deps and remote requests
    remote_requests = defaultdict(list)

    for beha_tag in beha_dict:
        target_position = beha_dict[beha_tag].location[:-1]
        for source_data_tag in beha_dict[beha_tag].needed_data_split_dict:
            for source_data_split in beha_dict[beha_tag].needed_data_split_dict[source_data_tag]:
                candidates = data_dict[source_data_tag].generated_split_location[source_data_split]
                if not candidates:
                    continue
                source_target_pair_distance = np.inf
                source_position = candidates[0]
                for source_position_candidate in candidates:
                    if source_position_candidate == target_position:
                        source_position = source_position_candidate
                        break
                    if dijkstra_routing:
                        current_pair_distance = hardware_platform.node_to_node_distance_dict[source_position_candidate][target_position]
                    else:
                        current_pair_distance = hardware_platform.node_to_node_manhattan_distance_dict[source_position_candidate][target_position]
                    if current_pair_distance < source_target_pair_distance:
                        source_target_pair_distance = current_pair_distance
                        source_position = source_position_candidate

                if source_position == target_position:
                    for producer_tag in data_dict[source_data_tag].generated_splitted_tag_dict[source_data_split]:
                        if producer_tag == (-1,):
                            continue
                        event_dict[producer_tag].issue_set.add(beha_tag)
                        event_dict[beha_tag].dependency_set.add(producer_tag)
                else:
                    remote_requests[(source_data_tag, source_data_split, source_position)].append(
                        (beha_tag, target_position)
                    )

    # Phase 2: deduplicate — one load per (data_split, source), forward to other nodes
    comms_by_data_tag = defaultdict(list) if dijkstra_routing else None
    for (source_data_tag, source_data_split, source_position), requests in remote_requests.items():
        comm_bytes = int(np.prod([end - start + 1 for (start, end) in source_data_split]) * type_bytes[data_dict[source_data_tag].data_type])

        # Group behas by target node
        node_behas = defaultdict(list)
        for beha_tag, target_position in requests:
            node_behas[target_position].append(beha_tag)

        # Pick anchor node (closest to source, byte-aware)
        if dijkstra_routing:
            anchor_node = min(node_behas, key=lambda n: hardware_platform.node_to_node_distance_function_dict[source_position][n](comm_bytes))
        else:
            anchor_node = min(node_behas, key=lambda n: hardware_platform.node_to_node_manhattan_distance_dict[source_position][n])
        anchor_beha = node_behas[anchor_node][0]

        # Create one comm: source -> anchor
        anchor_comm_tag = (1,) + anchor_beha[1:] + (event_dict[anchor_beha].commid_allocator.allocate(),)
        event_dict[anchor_comm_tag] = communication_notation(
            comm_name=beha_dict[anchor_beha].beha_name,
            comm_tag=anchor_comm_tag,
            source_location=source_position,
            target_location=anchor_node,
            comm_bytes=comm_bytes
        )

        # Update data tracking
        data_dict[source_data_tag].used_splitted_tag_dict[source_data_split].add(anchor_comm_tag)
        for beha_tag, _ in requests:
            data_dict[source_data_tag].used_splitted_tag_dict[source_data_split].discard(beha_tag)

        # Route anchor comm
        if not dijkstra_routing:
            h, d, routed_path = _route_comm(
                event_dict, anchor_comm_tag, source_position, anchor_node,
                hardware_platform, dijkstra_routing, alpha, beta, gamma
            )
            hops += h
            communication_distances += d
            for root in routed_path:
                for leaf in routed_path[root]:
                    communication_loads_dict[(root, leaf)] += np.ceil(comm_bytes / hardware_platform.links_dict[(root, leaf)].bandwidth)
        else:
            comms_by_data_tag[source_data_tag].append(anchor_comm_tag)

        # Wire producer -> anchor comm
        for producer_tag in data_dict[source_data_tag].generated_splitted_tag_dict[source_data_split]:
            if producer_tag == (-1,):
                continue
            event_dict[producer_tag].issue_set.add(anchor_comm_tag)
            event_dict[anchor_comm_tag].dependency_set.add(producer_tag)

        # Anchor node behas depend on anchor comm
        for beha_tag in node_behas[anchor_node]:
            event_dict[anchor_comm_tag].issue_set.add(beha_tag)
            event_dict[beha_tag].dependency_set.add(anchor_comm_tag)

        # Forward to other nodes via co2co
        for target_node, beha_tags in node_behas.items():
            if target_node == anchor_node:
                continue
            fwd_beha = beha_tags[0]
            fwd_comm_tag = (1,) + fwd_beha[1:] + (event_dict[fwd_beha].commid_allocator.allocate(),)
            event_dict[fwd_comm_tag] = communication_notation(
                comm_name=beha_dict[fwd_beha].beha_name,
                comm_tag=fwd_comm_tag,
                source_location=anchor_node,
                target_location=target_node,
                comm_bytes=comm_bytes
            )

            if not dijkstra_routing:
                h, d, routed_path = _route_comm(
                    event_dict, fwd_comm_tag, anchor_node, target_node,
                    hardware_platform, dijkstra_routing, alpha, beta, gamma
                )
                hops += h
                communication_distances += d
                for root in routed_path:
                    for leaf in routed_path[root]:
                        communication_loads_dict[(root, leaf)] += np.ceil(comm_bytes / hardware_platform.links_dict[(root, leaf)].bandwidth)
            else:
                comms_by_data_tag[source_data_tag].append(fwd_comm_tag)

            # Forwarding depends on anchor comm
            event_dict[anchor_comm_tag].issue_set.add(fwd_comm_tag)
            event_dict[fwd_comm_tag].dependency_set.add(anchor_comm_tag)

            # Behas on this node depend on forwarding comm
            for beha_tag in beha_tags:
                event_dict[fwd_comm_tag].issue_set.add(beha_tag)
                event_dict[beha_tag].dependency_set.add(fwd_comm_tag)

    # Group Dijkstra routing: route each group on accumulated background from prior groups
    if dijkstra_routing and comms_by_data_tag:
        _accum_loads = {}
        _accum_counts = {}
        for _dt, _comm_tags in comms_by_data_tag.items():
            # Set final platform state from accumulated loads
            hardware_platform.dijkstra_offload()
            # Set accumulated background from prior groups so BALD sees contention
            for lk, v in _accum_loads.items():
                hardware_platform.link_loads_dict[lk] = v
            for lk, v in _accum_counts.items():
                hardware_platform.link_loads_count[lk] = v
            _comm_pairs = [
                [ct, event_dict[ct].source_location, {event_dict[ct].target_location}, event_dict[ct].comm_bytes]
                for ct in _comm_tags
            ]
            _max_load, _, _paths, _ = hardware_platform.record_dijkstra_multicast_path(
                comm_pairs=_comm_pairs, alpha=alpha, beta=beta, gamma=gamma,
                deterministic=True
            )
            for ct in _comm_tags:
                ev = event_dict[ct]
                ev.paths = _paths[ct]
                ev.path_list = ev.get_paths(_paths[ct])
                src = ev.source_location
                tgt = ev.target_location
                ev.hops = hardware_platform.node_to_node_hop_dict[src][tgt]
                hops += ev.hops * ev.comm_bytes
                ev.communication_distances = hardware_platform.node_to_node_distance_function_dict[src][tgt](ev.comm_bytes)
                communication_distances += ev.communication_distances
            # Platform now has accum + this group; read back as new accumulated state
            _accum_loads = dict(hardware_platform.link_loads_dict)
            _accum_counts = dict(hardware_platform.link_loads_count)
        # Platform already has final accumulated state from last iteration.
        communication_loads_dict = {lk: hardware_platform.link_loads_dict.get(lk, 0)
                                    for lk in hardware_platform.links_set}

    if barrier_sync:
        for data_tag in data_dict:
            _apply_barrier_sync_for_group(event_dict, data_dict, data_tag)

    return hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict


def update_event_v2(
    target_event_tags: List[Tuple[int]],
    target_devices: List[Tuple[int]],
    hops: int,
    communication_distances: float,
    communication_loads_dict: Dict[Tuple[int], float],
    tensorcore_loads_dict: Dict[Tuple[int], float],
    vectorunit_loads_dict: Dict[Tuple[int], float],
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net,
    event_dict: Dict[Tuple[int], event_notation],
    dijkstra_routing: bool = False,
    alpha: int = 100,
    beta: int = 1,
    gamma: int = 100,
    mem_enable: bool = True,
    undo: dict = None,
    barrier_sync: bool = False
):

    # --- Undo: save scalars and platform state at entry ---
    if undo is not None:
        undo['hops'] = hops
        undo['comm_dist'] = communication_distances
        if dijkstra_routing:
            undo['platform_link_loads'] = dict(hardware_platform.link_loads_dict)
            undo['platform_link_counts'] = dict(hardware_platform.link_loads_count)

    targets_number = len(target_event_tags)
    old_behas = set()
    for target_idx in range(targets_number):
        target_event_tag = target_event_tags[target_idx]
        target_device = target_devices[target_idx]

        old_position = event_dict[target_event_tag].comp_location
        old_computation_time = hardware_platform.modules_dict[event_dict[target_event_tag].comp_device][old_position].working_time(
            source_datashape=event_dict[target_event_tag].comp_datashape,
            beha_type=event_dict[target_event_tag].comp_type,
            frequency=hardware_platform.frequency
        )

        # --- Undo: save locations before mutation ---
        if undo is not None:
            if target_event_tag not in undo['beha_locs']:
                undo['beha_locs'][target_event_tag] = beha_dict[target_event_tag].location
            if target_event_tag not in undo['comp_locs']:
                undo['comp_locs'][target_event_tag] = event_dict[target_event_tag].comp_location

        beha_dict[target_event_tag].location = target_device
        event_dict[target_event_tag].comp_location = target_device
        new_computation_time = hardware_platform.modules_dict[event_dict[target_event_tag].comp_device][event_dict[target_event_tag].comp_location].working_time(
            source_datashape=event_dict[target_event_tag].comp_datashape,
            beha_type=event_dict[target_event_tag].comp_type,
            frequency=hardware_platform.frequency
        )

        # --- Undo: save load entries before mutation ---
        if event_dict[target_event_tag].comp_device == "tensorcore":
            # --- Undo: save data location list before mutation ---
            if undo is not None:
                if old_position not in undo['tc_loads']:
                    undo['tc_loads'][old_position] = tensorcore_loads_dict[old_position]
                if target_device not in undo['tc_loads']:
                    undo['tc_loads'][target_device] = tensorcore_loads_dict[target_device]
            tensorcore_loads_dict[old_position] -= old_computation_time
            tensorcore_loads_dict[target_device] += new_computation_time
        else:
            # --- Undo: save deleted comm event + its issue/dep sets ---
            if undo is not None:
                if old_position not in undo['vu_loads']:
                    undo['vu_loads'][old_position] = vectorunit_loads_dict[old_position]
                if target_device not in undo['vu_loads']:
                    undo['vu_loads'][target_device] = vectorunit_loads_dict[target_device]
            vectorunit_loads_dict[old_position] -= old_computation_time
            vectorunit_loads_dict[target_device] += new_computation_time

        for produced_data_tag in beha_dict[target_event_tag].produced_data_split_dict:
            for produced_data_split in beha_dict[target_event_tag].produced_data_split_dict[produced_data_tag]:
                # Clean stale refs in used_splitted_tag_dict
                if undo is not None:
                    _dk = (produced_data_tag, produced_data_split)
                    if _dk not in undo['data_gen_locs']:
                        undo['data_gen_locs'][_dk] = list(data_dict[produced_data_tag].generated_split_location[produced_data_split])
                data_dict[produced_data_tag].generated_split_location[produced_data_split].remove(old_position[:-1])
                data_dict[produced_data_tag].generated_split_location[produced_data_split].append(target_device[:-1])

        if mem_enable:
            pass
        else:
            for needed_data_tag in beha_dict[target_event_tag].needed_data_split_dict:
                if needed_data_tag[0] == 1:
                    for needed_data_split in beha_dict[target_event_tag].needed_data_split_dict[needed_data_tag]:
                        # --- Undo: save used_splitted_tag_dict before mutation ---
                        if undo is not None:
                            _dk = (needed_data_tag, needed_data_split)
                            if _dk not in undo['data_gen_locs']:
                                undo['data_gen_locs'][_dk] = list(data_dict[needed_data_tag].generated_split_location[needed_data_split])
                        data_dict[needed_data_tag].generated_split_location[needed_data_split].remove(old_position[:-1])
                        data_dict[needed_data_tag].generated_split_location[needed_data_split].append(target_device[:-1])

        # --- Teardown: delete downstream comms (issue_set) ---
        comms_to_delete = set()
        # --- Undo: save issue_set before clearing ---
        if undo is not None and target_event_tag not in undo['issue_sets']:
            undo['issue_sets'][target_event_tag] = set(event_dict[target_event_tag].issue_set)
        for issued_tag in list(event_dict[target_event_tag].issue_set):
            if issued_tag[0] == 1 and issued_tag in event_dict:
                comms_to_delete.add(issued_tag)
            elif issued_tag[0] == 0:
                old_behas.add(issued_tag)
                if undo is not None and issued_tag not in undo['dep_sets']:
                    undo['dep_sets'][issued_tag] = set(event_dict[issued_tag].dependency_set)
                event_dict[issued_tag].dependency_set.discard(target_event_tag)
        event_dict[target_event_tag].issue_set = set()
        old_behas.add(target_event_tag)

        # --- Teardown: delete upstream comms (dependency_set) of old_behas ---
        for old_event_tag in list(old_behas):
            if undo is not None and old_event_tag not in undo['dep_sets']:
                undo['dep_sets'][old_event_tag] = set(event_dict[old_event_tag].dependency_set)
            for dep_tag in list(event_dict[old_event_tag].dependency_set):
                if dep_tag[0] == 1 and dep_tag in event_dict:
                    if undo is not None and dep_tag not in undo['issue_sets']:
                        undo['issue_sets'][dep_tag] = set(event_dict[dep_tag].issue_set)
                    event_dict[dep_tag].issue_set.discard(old_event_tag)
                    if not event_dict[dep_tag].issue_set:
                        comms_to_delete.add(dep_tag)
                elif dep_tag[0] == 0:
                    if undo is not None and dep_tag not in undo['issue_sets']:
                        undo['issue_sets'][dep_tag] = set(event_dict[dep_tag].issue_set)
                    event_dict[dep_tag].issue_set.discard(old_event_tag)
            event_dict[old_event_tag].dependency_set = set()

        # --- Delete marked comms, following chains ---
        while comms_to_delete:
            comm_tag = comms_to_delete.pop()
            if comm_tag not in event_dict:
                continue
            ev = event_dict[comm_tag]
            # Disconnect downstream: behas lose dependency, child comms get deleted
            for child_tag in list(ev.issue_set):
                if child_tag[0] == 0 and child_tag in event_dict:
                    old_behas.add(child_tag)
                    if undo is not None and child_tag not in undo['dep_sets']:
                        undo['dep_sets'][child_tag] = set(event_dict[child_tag].dependency_set)
                    event_dict[child_tag].dependency_set.discard(comm_tag)
                elif child_tag[0] == 1 and child_tag in event_dict:
                    comms_to_delete.add(child_tag)
            # Disconnect upstream: producers lose this comm from issue_set
            for parent_tag in list(ev.dependency_set):
                if parent_tag in event_dict:
                    if undo is not None and parent_tag not in undo['issue_sets']:
                        undo['issue_sets'][parent_tag] = set(event_dict[parent_tag].issue_set)
                    event_dict[parent_tag].issue_set.discard(comm_tag)
                    if parent_tag[0] == 1 and not event_dict[parent_tag].issue_set:
                        comms_to_delete.add(parent_tag)
            # Subtract loads and delete
            for root in ev.paths:
                for leaf in ev.paths[root]:
                    lk = (root, leaf)
                    load_delta = np.ceil(ev.comm_bytes / hardware_platform.links_dict[lk].bandwidth)
                    if undo is not None and lk not in undo['comm_loads']:
                        undo['comm_loads'][lk] = communication_loads_dict.get(lk, 0)
                    communication_loads_dict[lk] = max(0, communication_loads_dict.get(lk, 0) - load_delta)
                    if dijkstra_routing:
                        if undo is not None and lk not in undo['platform_link_loads']:
                            undo['platform_link_loads'][lk] = hardware_platform.link_loads_dict.get(lk, 0)
                            undo['platform_link_counts'][lk] = hardware_platform.link_loads_count.get(lk, 0)
                        hardware_platform.link_loads_dict[lk] = max(0, hardware_platform.link_loads_dict.get(lk, 0) - load_delta)
                        hardware_platform.link_loads_count[lk] = max(0, hardware_platform.link_loads_count.get(lk, 0) - 1)
            hops -= ev.hops * ev.comm_bytes
            communication_distances -= ev.communication_distances
            # --- Undo: save allocator state before recycle ---
            _alloc_owner = (0,) + comm_tag[1:-1]
            if undo is not None and _alloc_owner not in undo['allocators']:
                _a = event_dict[_alloc_owner].commid_allocator
                undo['allocators'][_alloc_owner] = (set(_a.used_ids), list(_a.recycled), _a.next_id)
            event_dict[_alloc_owner].commid_allocator.recycle(comm_tag[-1])
            # --- Undo: track created comm ---
            if undo is not None:
                undo['deleted_comms'][comm_tag] = ev
                if comm_tag not in undo['issue_sets']:
                    undo['issue_sets'][comm_tag] = set(ev.issue_set)
                if comm_tag not in undo['dep_sets']:
                    undo['dep_sets'][comm_tag] = set(ev.dependency_set)
            del event_dict[comm_tag]

        for old_event_tag in old_behas:
            event_dict[old_event_tag].dependency_set = {
                d for d in event_dict[old_event_tag].dependency_set if d in event_dict
            }

    # remote_requests key: (source_data_tag, source_data_split, source_position)
    remote_requests = defaultdict(list)
    for beha_tag in old_behas:
        target_position = beha_dict[beha_tag].location[:-1]
        for source_data_tag in beha_dict[beha_tag].needed_data_split_dict:
            for source_data_split in beha_dict[beha_tag].needed_data_split_dict[source_data_tag]:
                candidates = data_dict[source_data_tag].generated_split_location[source_data_split]
                if not candidates:
                    continue
                source_target_pair_distance = np.inf
                source_position = candidates[0]
                for source_position_candidate in candidates:
                    if source_position_candidate == target_position:
                        source_position = source_position_candidate
                        break
                    if dijkstra_routing:
                        current_pair_distance = hardware_platform.node_to_node_distance_dict[source_position_candidate][target_position]
                    else:
                        current_pair_distance = hardware_platform.node_to_node_manhattan_distance_dict[source_position_candidate][target_position]
                    if current_pair_distance < source_target_pair_distance:
                        source_target_pair_distance = current_pair_distance
                        source_position = source_position_candidate

                if source_position == target_position:
                    for producer_tag in data_dict[source_data_tag].generated_splitted_tag_dict[source_data_split]:
                        if producer_tag == (-1,):
                            continue
                        if undo is not None and producer_tag not in undo['issue_sets']:
                            undo['issue_sets'][producer_tag] = set(event_dict[producer_tag].issue_set)
                        event_dict[producer_tag].issue_set.add(beha_tag)
                        event_dict[beha_tag].dependency_set.add(producer_tag)
                else:
                    remote_requests[(source_data_tag, source_data_split, source_position)].append(
                        (beha_tag, target_position)
                    )

    for (source_data_tag, source_data_split, source_position), requests in remote_requests.items():
        comm_bytes = int(np.prod([end - start + 1 for (start, end) in source_data_split]) * type_bytes[data_dict[source_data_tag].data_type])

        # --- Undo: save scalars and platform state at entry ---
        if undo is not None:
            _uk = (source_data_tag, source_data_split)
            if _uk not in undo['data_used']:
                undo['data_used'][_uk] = set(data_dict[source_data_tag].used_splitted_tag_dict[source_data_split])
        live = {t for t in data_dict[source_data_tag].used_splitted_tag_dict[source_data_split] if t in event_dict}
        data_dict[source_data_tag].used_splitted_tag_dict[source_data_split] = live

        node_behas = defaultdict(list)
        for beha_tag, target_position in requests:
            node_behas[target_position].append(beha_tag)

        # Check for existing comm serving this (source, split) → target_node; reuse if found
        existing_by_target = {}
        for ct in live:
            if ct[0] == 1 and ct in event_dict:
                ev = event_dict[ct]
                if ev.source_location == source_position:
                    existing_by_target[ev.target_location] = ct
                # fwd_comm: check parent is anchor from source_position
                for dep in ev.dependency_set:
                    if dep[0] == 1 and dep in event_dict and event_dict[dep].source_location == source_position:
                        existing_by_target[ev.target_location] = ct
                        break

        reused_nodes = []
        for target_node, beha_tags in node_behas.items():
            if target_node in existing_by_target:
                ec = existing_by_target[target_node]
                if undo is not None and ec not in undo['issue_sets']:
                    undo['issue_sets'][ec] = set(event_dict[ec].issue_set)
                for beha_tag in beha_tags:
                    event_dict[ec].issue_set.add(beha_tag)
                    event_dict[beha_tag].dependency_set.add(ec)
                reused_nodes.append(target_node)
        for rn in reused_nodes:
            del node_behas[rn]

        if not node_behas:
            continue

        # Need new comms — pick anchor node (byte-aware for dijkstra)
        if dijkstra_routing:
            anchor_node = min(node_behas, key=lambda n: hardware_platform.node_to_node_distance_function_dict[source_position][n](comm_bytes))
        else:
            anchor_node = min(node_behas, key=lambda n: hardware_platform.node_to_node_manhattan_distance_dict[source_position][n])
        anchor_beha = node_behas[anchor_node][0]

        # --- Undo: save allocator before allocate ---
        if undo is not None and anchor_beha not in undo['allocators']:
            _a = event_dict[anchor_beha].commid_allocator
            undo['allocators'][anchor_beha] = (set(_a.used_ids), list(_a.recycled), _a.next_id)
        anchor_comm_tag = (1,) + anchor_beha[1:] + (event_dict[anchor_beha].commid_allocator.allocate(),)
        event_dict[anchor_comm_tag] = communication_notation(
            comm_name=beha_dict[anchor_beha].beha_name,
            comm_tag=anchor_comm_tag,
            source_location=source_position,
            target_location=anchor_node,
            comm_bytes=comm_bytes
        )
        if undo is not None:
            undo['created_comms'].add(anchor_comm_tag)
        data_dict[source_data_tag].used_splitted_tag_dict[source_data_split].add(anchor_comm_tag)
        for beha_tag, _ in requests:
            data_dict[source_data_tag].used_splitted_tag_dict[source_data_split].discard(beha_tag)

        if not dijkstra_routing:
            h, d, routed_path = _route_comm(
                event_dict, anchor_comm_tag, source_position, anchor_node,
                hardware_platform, dijkstra_routing, alpha, beta, gamma
            )
            hops += h
            communication_distances += d
            for root in routed_path:
                for leaf in routed_path[root]:
                    if undo is not None and (root, leaf) not in undo['comm_loads']:
                        undo['comm_loads'][(root, leaf)] = communication_loads_dict[(root, leaf)]
                    communication_loads_dict[(root, leaf)] += np.ceil(comm_bytes / hardware_platform.links_dict[(root, leaf)].bandwidth)

        for producer_tag in data_dict[source_data_tag].generated_splitted_tag_dict[source_data_split]:
            if producer_tag == (-1,):
                continue
            if undo is not None and producer_tag not in undo['issue_sets']:
                undo['issue_sets'][producer_tag] = set(event_dict[producer_tag].issue_set)
            event_dict[producer_tag].issue_set.add(anchor_comm_tag)
            event_dict[anchor_comm_tag].dependency_set.add(producer_tag)

        for beha_tag in node_behas[anchor_node]:
            event_dict[anchor_comm_tag].issue_set.add(beha_tag)
            event_dict[beha_tag].dependency_set.add(anchor_comm_tag)

        for target_node, beha_tags in node_behas.items():
            if target_node == anchor_node:
                continue
            fwd_beha = beha_tags[0]
            # --- Undo: save allocator before allocate ---
            if undo is not None and fwd_beha not in undo['allocators']:
                _a = event_dict[fwd_beha].commid_allocator
                undo['allocators'][fwd_beha] = (set(_a.used_ids), list(_a.recycled), _a.next_id)
            fwd_comm_tag = (1,) + fwd_beha[1:] + (event_dict[fwd_beha].commid_allocator.allocate(),)
            event_dict[fwd_comm_tag] = communication_notation(
                comm_name=beha_dict[fwd_beha].beha_name,
                comm_tag=fwd_comm_tag,
                source_location=anchor_node,
                target_location=target_node,
                comm_bytes=comm_bytes
            )
            if undo is not None:
                undo['created_comms'].add(fwd_comm_tag)
            if dijkstra_routing:
                data_dict[source_data_tag].used_splitted_tag_dict[source_data_split].add(fwd_comm_tag)
            if not dijkstra_routing:
                h, d, routed_path = _route_comm(
                    event_dict, fwd_comm_tag, anchor_node, target_node,
                    hardware_platform, dijkstra_routing, alpha, beta, gamma
                )
                hops += h
                communication_distances += d
                for root in routed_path:
                    for leaf in routed_path[root]:
                        if undo is not None and (root, leaf) not in undo['comm_loads']:
                            undo['comm_loads'][(root, leaf)] = communication_loads_dict[(root, leaf)]
                        communication_loads_dict[(root, leaf)] += np.ceil(comm_bytes / hardware_platform.links_dict[(root, leaf)].bandwidth)

            event_dict[anchor_comm_tag].issue_set.add(fwd_comm_tag)
            event_dict[fwd_comm_tag].dependency_set.add(anchor_comm_tag)

            for beha_tag in beha_tags:
                event_dict[fwd_comm_tag].issue_set.add(beha_tag)
                event_dict[beha_tag].dependency_set.add(fwd_comm_tag)

    # Group Dijkstra routing: re-route each affected concurrent group independently
    if dijkstra_routing and undo is not None:
        affected_data_tags = {k[0] for k in remote_requests}
        if affected_data_tags:
            # Original platform state (not modified during teardown/rebuild for dijkstra mode)
            _orig_loads = undo['platform_link_loads']
            _orig_counts = undo['platform_link_counts']
            # Background = original minus all affected groups' original contributions
            _bg_loads = dict(_orig_loads)
            _bg_counts = dict(_orig_counts)
            _orig_comms = {}
            _new_comms = {}
            for _dt in affected_data_tags:
                _orig_ct = set()
                _new_ct = set()
                for sp in data_dict[_dt].used_splitted_tag_dict:
                    _dk = (_dt, sp)
                    if _dk in undo['data_used']:
                        _orig_ct.update(undo['data_used'][_dk])
                    else:
                        _orig_ct.update(data_dict[_dt].used_splitted_tag_dict[sp])
                    _new_ct.update(data_dict[_dt].used_splitted_tag_dict[sp])
                _orig_comms[_dt] = {ct for ct in _orig_ct if ct[0] == 1}
                _new_comms[_dt] = {ct for ct in _new_ct if ct[0] == 1 and ct in event_dict}
                # Subtract original comms' contributions from background
                for ct in _orig_comms[_dt]:
                    if ct in event_dict:
                        ev_s = event_dict[ct]
                    elif ct in undo['deleted_comms']:
                        ev_s = undo['deleted_comms'][ct]
                    else:
                        continue
                    for root in ev_s.paths:
                        for leaf in ev_s.paths[root]:
                            lk = (root, leaf)
                            _bg_loads[lk] -= hardware_platform.current_link_loads(ev_s.comm_bytes, hardware_platform.links_dict[lk])
                            _bg_counts[lk] -= 1
            # Save all link-indexed communication_loads_dict entries to undo (copy-on-first-write)
            for lk in communication_loads_dict:
                if lk not in undo['comm_loads']:
                    undo['comm_loads'][lk] = communication_loads_dict[lk]
            # Route each affected group on accumulated background (bg + previously re-routed groups)
            _grp_new_loads = dict(_bg_loads)
            _grp_new_counts = dict(_bg_counts)
            for _dt in affected_data_tags:
                if not _new_comms[_dt]:
                    continue
                # Route each group on accumulated background from prior groups
                hardware_platform.dijkstra_offload()
                # Set background = unaffected + previously re-routed affected groups
                for lk, v in _grp_new_loads.items():
                    if v > 0:
                        hardware_platform.link_loads_dict[lk] = v
                for lk, v in _grp_new_counts.items():
                    if v > 0:
                        hardware_platform.link_loads_count[lk] = v
                _comm_pairs = [
                    [ct, event_dict[ct].source_location, {event_dict[ct].target_location}, event_dict[ct].comm_bytes]
                    for ct in _new_comms[_dt]
                ]
                _max_load, _, _paths, _ = hardware_platform.record_dijkstra_multicast_path(
                    comm_pairs=_comm_pairs, alpha=alpha, beta=beta, gamma=gamma,
                    deterministic=True
                )
                for ct in _new_comms[_dt]:
                    ev = event_dict[ct]
                    old_h = ev.hops * ev.comm_bytes
                    old_d = ev.communication_distances
                    ev.paths = _paths[ct]
                    ev.path_list = ev.get_paths(_paths[ct])
                    src = ev.source_location
                    tgt = ev.target_location
                    ev.hops = hardware_platform.node_to_node_hop_dict[src][tgt]
                    ev.communication_distances = hardware_platform.node_to_node_distance_function_dict[src][tgt](ev.comm_bytes)
                    hops += ev.hops * ev.comm_bytes - old_h
                    communication_distances += ev.communication_distances - old_d
                # Read back platform state as new accumulated state
                _grp_new_loads = dict(hardware_platform.link_loads_dict)
                _grp_new_counts = dict(hardware_platform.link_loads_count)
            # Link loads accumulate across groups (prior groups' loads persist)
            hardware_platform.dijkstra_offload()
            for lk, v in _grp_new_loads.items():
                if v > 0:
                    hardware_platform.link_loads_dict[lk] = v
            for lk, v in _grp_new_counts.items():
                if v > 0:
                    hardware_platform.link_loads_count[lk] = v
            # Update communication_loads_dict from platform (link-indexed, same semantics as XY mode)
            for lk in communication_loads_dict:
                communication_loads_dict[lk] = hardware_platform.link_loads_dict.get(lk, 0)

    if barrier_sync:
        affected_data_tags = {k[0] for k in remote_requests}
        for data_tag in affected_data_tags:
            _apply_barrier_sync_for_group(event_dict, data_dict, data_tag, undo=undo)

    return hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict


def update_offlinedata_v2(
    target_data_tags: Tuple[int],
    target_data_split: Tuple[Tuple[int]],
    target_devices: Tuple[int],
    hops: int,
    communication_distances: float,
    communication_loads_dict: Dict[Tuple[int], float],
    tensorcore_loads_dict: Dict[Tuple[int], float],
    vectorunit_loads_dict: Dict[Tuple[int], float],
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net,
    event_dict: Dict[Tuple[int], event_notation],
    dijkstra_routing: bool = False,
    alpha: int = 100,
    beta: int = 1,
    gamma: int = 100,
    mem_enable: bool = True,
    undo: dict = None
):
    if mem_enable:
        pass
    else:
        raise ValueError("update_offlinedata is not supported when mem_enable is False, please set mem_enable to True.")

    if undo is not None:
        undo['hops'] = hops
        undo['comm_dist'] = communication_distances
        if dijkstra_routing:
            undo['platform_link_loads'] = dict(hardware_platform.link_loads_dict)
            undo['platform_link_counts'] = dict(hardware_platform.link_loads_count)

    for communication_tag in list(data_dict[target_data_tags].used_splitted_tag_dict[target_data_split]):
        if communication_tag[0] == 1:
            # Skip stale refs (comm deleted by update_event but not yet cleaned from used_splitted_tag_dict)
            if communication_tag not in event_dict:
                if undo is not None:
                    _uk = (target_data_tags, target_data_split)
                    if _uk not in undo['data_used']:
                        undo['data_used'][_uk] = set(data_dict[target_data_tags].used_splitted_tag_dict[target_data_split])
                data_dict[target_data_tags].used_splitted_tag_dict[target_data_split].discard(communication_tag)
                continue
            # --- Undo: save comm event state before mutation ---
            if undo is not None and communication_tag not in undo.get('offline_comms', {}):
                if 'offline_comms' not in undo:
                    undo['offline_comms'] = {}
                ev = event_dict[communication_tag]
                undo['offline_comms'][communication_tag] = (
                    ev.source_location, ev.paths, ev.path_list,
                    ev.hops, ev.communication_distances
                )

            old_paths = event_dict[communication_tag].paths
            if not dijkstra_routing:
                for root in old_paths:
                    for leaf in old_paths[root]:
                        used_link = (root, leaf)
                        if undo is not None and used_link not in undo['comm_loads']:
                            undo['comm_loads'][used_link] = communication_loads_dict[used_link]
                        communication_loads_dict[used_link] -= np.ceil(event_dict[communication_tag].comm_bytes / hardware_platform.links_dict[used_link].bandwidth)
            old_hops = event_dict[communication_tag].hops
            hops -= old_hops * event_dict[communication_tag].comm_bytes
            communication_distances -= event_dict[communication_tag].communication_distances

            event_dict[communication_tag].source_location = target_devices[:-1]
            source_location = event_dict[communication_tag].source_location
            target_location = event_dict[communication_tag].target_location
            if dijkstra_routing:
                dijkstra_path = hardware_platform.record_dijkstra_multicast_path(
                    comm_pairs=[[communication_tag, source_location, {target_location}, event_dict[communication_tag].comm_bytes]],
                    alpha=alpha, beta=beta, gamma=gamma,
                )[2]
                event_dict[communication_tag].paths = dijkstra_path[communication_tag]
                event_dict[communication_tag].path_list = event_dict[communication_tag].get_paths(dijkstra_path[communication_tag])
                event_dict[communication_tag].hops = hardware_platform.node_to_node_hop_dict[source_location][target_location]
                hops += event_dict[communication_tag].hops * event_dict[communication_tag].comm_bytes
                event_dict[communication_tag].communication_distances = hardware_platform.node_to_node_distance_function_dict[source_location][target_location](event_dict[communication_tag].comm_bytes)
                communication_distances += event_dict[communication_tag].communication_distances
                routed_path = dijkstra_path[communication_tag]
            else:
                routed_path = hardware_platform.xy_paths[source_location][target_location]
                xy_path = event_dict[communication_tag].get_paths(routed_path)
                event_dict[communication_tag].paths = routed_path
                event_dict[communication_tag].path_list = xy_path
                event_dict[communication_tag].hops = hardware_platform.node_to_node_manhattan_hops_dict[source_location][target_location]
                hops += event_dict[communication_tag].hops * event_dict[communication_tag].comm_bytes
                event_dict[communication_tag].communication_distances = hardware_platform.node_to_node_manhattan_distance_function_dict[source_location][target_location](event_dict[communication_tag].comm_bytes)
                communication_distances += event_dict[communication_tag].communication_distances

            if not dijkstra_routing:
                for root in routed_path:
                    for leaf in routed_path[root]:
                        used_link = (root, leaf)
                        if undo is not None and used_link not in undo['comm_loads']:
                            undo['comm_loads'][used_link] = communication_loads_dict[used_link]
                        communication_loads_dict[used_link] += np.ceil(event_dict[communication_tag].comm_bytes / hardware_platform.links_dict[used_link].bandwidth)

        else:
            raise ValueError(f"Communication event {communication_tag} is not a communication event, but a computation event.")

    return hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict


def add_broadcast_v2(
    data_tags: List[Tuple[int]],
    ddr_chiplets: List[int],
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net,
    event_dict: Dict[Tuple[int], event_notation],
    dijkstra_routing: bool = False,
    alpha: int = 100,
    beta: int = 1,
    gamma: int = 100,
):

    available_ddrs = []
    ddrs = set()
    for ddr_idx in hardware_platform.ddr_dict:
        if ddr_idx[0] in ddr_chiplets:
            available_ddrs.append(ddr_idx)
            ddrs.add(ddr_idx[:-1])

    """
    Add medium data to the data_dict and beha_dict, and update the event_dict accordingly.
    """
    for data_tag in data_tags:
        for producer_tag in data_dict[data_tag].generated_tag_splitted_dict:
            if producer_tag == (-1,):
                continue
            for data_split in data_dict[data_tag].generated_tag_splitted_dict[producer_tag]:
                producer_location = event_dict[producer_tag].comp_location[:-1]
                producer_beha_tag = producer_tag
                communication_tag = (1,) + producer_beha_tag[1:] + (event_dict[producer_beha_tag].commid_allocator.allocate(),)

                event_dict[communication_tag] = communication_notation(
                    comm_name=beha_dict[producer_beha_tag].beha_name,
                    comm_tag=communication_tag,
                    source_location=producer_location,
                    target_location=ddrs,
                    comm_bytes=int(np.prod([end - start + 1 for (start, end) in data_split])*type_bytes[data_dict[data_tag].data_type])
                )

                event = event_dict[communication_tag]
                if dijkstra_routing:
                    dijkstra_path = hardware_platform.record_dijkstra_multicast_path(
                        comm_pairs=[[event.event_tag, event.source_location, ddrs, event.comm_bytes]],
                        alpha=alpha, beta=beta, gamma=gamma,
                    )[2]
                    event_dict[communication_tag].paths = dijkstra_path[event.event_tag]
                    event_dict[communication_tag].path_list = event.get_paths(dijkstra_path[event.event_tag])
                else:
                    routed_path = hardware_platform.xy_multicast_path(
                        source_node_idx = event.source_location,
                        target_nodes_set = ddrs
                    )
                    xy_path = event.get_paths(routed_path)
                    event_dict[communication_tag].paths = routed_path
                    event_dict[communication_tag].path_list = xy_path

                event_dict[producer_tag].issue_set.add(communication_tag)
                event_dict[communication_tag].dependency_set.add(producer_tag)


def reroute_dijkstra(
    beha_dict: Dict[Tuple[int], beha_notation],
    data_dict: Dict[Tuple[int], tensor_notation],
    hardware_platform: net,
    event_dict: Dict[Tuple[int], event_notation],
    alpha: int = 1,
    beta: int = 0,
    gamma: int = 1
):
    """Re-route all communication events using dijkstra routing on accumulated background.

    Intended to be called after SA optimization (with XY routing) to get better
    path quality from BALD while keeping behavior locations unchanged.

    Returns (hops, communication_distances, communication_loads_dict).
    """
    # Collect all comm events and group by source_data_tag
    comms_by_data_tag = defaultdict(list)
    seen_comms = set()
    for dt in data_dict:
        for sp in data_dict[dt].used_splitted_tag_dict:
            for ct in data_dict[dt].used_splitted_tag_dict[sp]:
                if ct[0] == 1 and ct in event_dict and ct not in seen_comms:
                    comms_by_data_tag[dt].append(ct)
                    seen_comms.add(ct)
                    for iss in event_dict[ct].issue_set:
                        if iss[0] == 1 and iss in event_dict and iss not in seen_comms:
                            comms_by_data_tag[dt].append(iss)
                            seen_comms.add(iss)

    # Reset hops and communication_distances
    hops = 0
    communication_distances = 0

    hardware_platform.dijkstra_offload()
    for _dt, _comm_tags in comms_by_data_tag.items():
        if not _comm_tags:
            continue
        # Sort comms by comm_bytes descending — largest comms get best paths
        _comm_tags.sort(key=lambda ct: event_dict[ct].comm_bytes, reverse=True)
        for ct in _comm_tags:
            ev = event_dict[ct]
            src = ev.source_location
            tgt = ev.target_location
            if src == tgt:
                ev.paths = {}
                ev.path_list = []
                ev.hops = 0
                ev.communication_distances = 0
                continue
            paths_dict, _ = hardware_platform.allocate_path(
                src, tgt, ev.comm_bytes, alpha=alpha, beta=beta, gamma=gamma
            )
            ev.paths = paths_dict
            ev.path_list = ev.get_paths(paths_dict)
            ev.hops = hardware_platform.node_to_node_hop_dict[src][tgt]
            hops += ev.hops * ev.comm_bytes
            ev.communication_distances = hardware_platform.node_to_node_distance_function_dict[src][tgt](ev.comm_bytes)
            communication_distances += ev.communication_distances

    # Use dense link-indexed loads (same semantics as XY mode).
    communication_loads_dict = {lk: hardware_platform.link_loads_dict.get(lk, 0)
                                for lk in hardware_platform.links_set}

    return hops, communication_distances, communication_loads_dict
