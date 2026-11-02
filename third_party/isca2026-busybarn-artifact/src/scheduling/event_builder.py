
"""Event construction: builds computation and communication event graphs from
partitioned operators, mapping results, and hardware topology."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../partition/'))
sys.path.append(os.path.join(file_path, '../components/'))
sys.path.append(os.path.join(file_path, './communication/'))
sys.path.append(os.path.join(file_path, './communication/topology/'))


from Event import Event, ComputationEvent, CommunicationEvent
from notation import data_notation, data_parts, op_notation, function_notation, type_bytes
from Device import Device, ComputationDevice, Tensorcore, Vectorunit
from path_allocator import Dijkstra_path_allocation, cross_chiplet
from net import net
from mesh_2d import mesh_2d


from typing import List, Dict, Tuple, Set, Deque
from collections import defaultdict
import random


def aggregate_relationships(data):
    """Groups data splits by their shared set of target ops.

    Merges overlapping data-to-target mappings so that splits going to the
    same subset of targets are batched into one communication stream.
    Allows data replication: a split can appear in multiple groups if it
    serves different target subsets.
    """
    # Mapping from elements to sets of keys
    element_to_keys = defaultdict(set)

    # Build element to key mapping
    for key, elements in data:
        for element in elements:
            element_to_keys[element].add(key)

    # Reverse mapping from sets of keys to sets of elements
    keys_to_elements = defaultdict(set)
    for element, keys in element_to_keys.items():
        keys_to_elements[frozenset(keys)].add(element)

    # Convert dictionary to sorted list format
    result = [[set(keys), elements] for keys, elements in sorted(keys_to_elements.items(), key=lambda x: (len(x[0]), x[0]))]

    return result


def aggregate_identical(data):
    """Groups keys that map to exactly identical value sets.

    Unlike aggregate_relationships, does not split or replicate data --
    only merges keys whose target sets are identical. Produces fewer
    but larger messages.
    """
    # Mapping from elements to sets of keys
    elements_to_keys = defaultdict(set)

    # Build elements to key mapping
    for key, elements in data:
        elements_to_keys[frozenset(elements)].add(key)

    # Convert dictionary to sorted list format
    result = [[set(keys), set(elements)] for elements, keys in elements_to_keys.items()]

    return result


def event_builder(
    op_dict: Dict[int, op_notation],
    data_dict: Dict[Tuple[int, int], data_notation],
    block_candidates: Dict[int, Dict[str, Deque[int]]],
    dist_dict: Dict[Tuple[int, int], int],
    chiplet_net: net,
    cores_nets: List[net]
):
    """Build the complete event DAG from a mapped operator graph.

    Walks the op DAG in topological order. For each ready op, assigns it
    to a device, infers communication events from data dependencies (intra-
    chiplet via XY broadcast, inter-chiplet via Dijkstra + cross_chiplet
    decomposition), and wires dependency/issue edges. Returns events_dict.
    """

    # -- Initialization: partition ops into ready (no deps) and waiting --
    start_ops = []
    wait_ops = set()
    wait_ops_dependency_dict = defaultdict(set)

    start_ops = []
    for op_tag in op_dict:
        if op_dict[op_tag].dependencies == set():
            start_ops.append(op_dict[op_tag])
        else:
            wait_ops.add(op_tag)
    start_ops.sort(key=lambda x: x.op_tag)

    # Event tags: (0, id) for computation, (1, id) for communication
    computation_event_tag = (0, 0)
    communication_event_tag = (1, 0)
    events_dict = {}
    loop_cnt = 0
    loop_limit = len(op_dict)

    # -- Main loop: process ready ops level by level (BFS-like) --
    while start_ops != [] or wait_ops != set():

        # Collect communication pairs needed for this batch of ops
        # Format: {source_location: [[(data_tag, split_tag), {target_ops}], ...]}
        op_communication_pairs = {}
        computation_event_denpendencies_dict = defaultdict(set)
        for op in start_ops[:]:
            # Round-robin device assignment within the block's device pool
            op_device_tag = block_candidates[op.block_idx][op.op_type].popleft()
            block_candidates[op.block_idx][op.op_type].append(op_device_tag)
            # TODO: offload will be added here
            op.device_locations = {op.op_type: op_device_tag}
            op.locations.add(op_device_tag[0:2])
            for target_data_tag in op.target_data:
                split_data_tags = op.target_data[target_data_tag]
                for split_data_tag in split_data_tags:
                    data_dict[target_data_tag].split_data_dict[(target_data_tag[-1], split_data_tag)].locations.add(op_device_tag[0:2])

            '''
            get communications pairs for the current op
            '''
            op_source_data = op.source_data
            for source_data_tag in op_source_data:
                split_data_tags = op_source_data[source_data_tag]
                for split_data_tag in split_data_tags:
                    source_split_data_current_locations = data_dict[source_data_tag].split_data_dict[(source_data_tag[-1], split_data_tag)].locations
                    # TODO: make the offline data loaded without consideration of DDR
                    if source_data_tag[0] == 1:
                        data_dict[source_data_tag].split_data_dict[(source_data_tag[-1], split_data_tag)].locations.add(op_device_tag[0:2])
                    # NOTE: check the online data generated
                    if source_split_data_current_locations == set():
                        raise ValueError(f"Data {source_data_tag} part {split_data_tag} is not created yet")

                    if op_device_tag[0:2] not in source_split_data_current_locations:
                        # Data not local -- find nearest source by distance
                        source_split_data_current_location = None
                        for source_split_data_candidate_location in source_split_data_current_locations:
                            if source_split_data_current_location == None:
                                source_split_data_current_location = source_split_data_candidate_location
                            else:
                                # NOTE: The target data location is decided by the smallest distance
                                if dist_dict[(source_split_data_candidate_location, op_device_tag[0:2])] < dist_dict[(source_split_data_current_location, op_device_tag[0:2])]:
                                    source_split_data_current_location = source_split_data_candidate_location
                                else:
                                    continue
                        if source_split_data_current_location in op_communication_pairs:
                            op_communication_pairs[source_split_data_current_location].append([(source_data_tag, split_data_tag), {(op_device_tag[0], op_device_tag[1], op.op_tag)}])
                        else:
                            op_communication_pairs[source_split_data_current_location] = [[(source_data_tag, split_data_tag), {(op_device_tag[0], op_device_tag[1], op.op_tag)}]]
                    else:
                        # Data is local -- wire direct comp-to-comp dependency
                        last_computation_event_tag = {(0, tag) for tag in data_dict[source_data_tag].split_data_dict[(source_data_tag[-1], split_data_tag)].created}
                        computation_event_denpendencies_dict[(0, op.op_tag)] |= last_computation_event_tag
                        for previous_op_tag in last_computation_event_tag:
                            events_dict[previous_op_tag].issue_set.add((0, op.op_tag))

        # -- Create communication events for remote data transfers --
        for location in op_communication_pairs:
            # Merge streams sharing the same source and overlapping targets
            op_communication_pairs[location] = aggregate_relationships(op_communication_pairs[location])
            for stream in op_communication_pairs[location]:
                data_parts = stream[0]
                data_targets = stream[1]

                chiplet_target_location = {}
                chiplet_target_location[location[0]] = []
                chiplet_cores_dict = {}
                chiplet_ops_set = set()
                for target_location_list in data_targets:
                    if target_location_list[0] in chiplet_target_location:
                        chiplet_target_location[target_location_list[0]].append([(target_location_list[0], target_location_list[1]), {target_location_list[2]}])
                    else:
                        chiplet_target_location[target_location_list[0]] = [[(target_location_list[0], target_location_list[1]), {target_location_list[2]}]]
                    if target_location_list[0] in chiplet_cores_dict:
                        chiplet_cores_dict[target_location_list[0]].add(target_location_list[0:2])
                    else:
                        chiplet_cores_dict[target_location_list[0]] = {target_location_list[0:2]}
                    chiplet_ops_set.add((0, target_location_list[2]))
                # {chiplet_idx: {core_idx: {op_tag}}}
                chiplet_core_target_dict = {}
                for chiplet_idx in chiplet_target_location:
                    for core_target_op_list in chiplet_target_location[chiplet_idx]:
                        if chiplet_idx in chiplet_core_target_dict:
                            pass
                        else:
                            chiplet_core_target_dict[chiplet_idx] = {}
                        if core_target_op_list[0] in chiplet_core_target_dict[chiplet_idx]:
                            chiplet_core_target_dict[chiplet_idx][core_target_op_list[0]] |= (core_target_op_list[1])
                        else:
                            chiplet_core_target_dict[chiplet_idx][core_target_op_list[0]] = core_target_op_list[1]

                stream_size = 0
                data_created = set()
                for data_splits in data_parts:
                    stream_size += data_dict[data_splits[0]].split_data_size
                    data_created |= (data_dict[data_splits[0]].split_data_dict[(data_splits[0][1], data_splits[1])].created)

                for target_chiplet in chiplet_cores_dict:
                    if location[0] == target_chiplet:
                        # Intra-chiplet: XY broadcast within the core mesh
                        target_cores_nodes = {core_idx[1] for core_idx in chiplet_cores_dict[target_chiplet]}
                        core_path_beginner, core_path_ender, core_path_relationship = cores_nets[location[0]].xy_broadcast(
                            # core_path_beginner, core_path_ender, core_path_relationship = cores_nets[location[0]].record_dijkstra_broadcast(
                            source_node_idx=location[1],
                            target_nodes_idx=target_cores_nodes,
                            link_load=stream_size
                            )
                        events_dict[communication_event_tag] = CommunicationEvent(
                            event_tag=communication_event_tag,
                            source_node_idx=location,
                            pass_path=core_path_relationship,
                            pass_type="co2co",
                            message_bytes=stream_size,
                            chiplet_idx=location[0]
                        )
                        events_dict[communication_event_tag].add_dependency({(0, tag) for tag in data_created})
                        for data_splits in data_parts:
                            data_dict[data_splits[0]].split_data_dict[(data_splits[0][1], data_splits[1])].locations |= chiplet_cores_dict[target_chiplet]
                            for previous_created_tag in data_dict[data_splits[0]].split_data_dict[(data_splits[0][1], data_splits[1])].created:
                                events_dict[(0, previous_created_tag)].issue_set.add(communication_event_tag)
                        events_dict[communication_event_tag].add_issue(chiplet_ops_set)
                        for next_computation_tag in chiplet_ops_set:
                            computation_event_denpendencies_dict[next_computation_tag].add(communication_event_tag)
                        events_dict[communication_event_tag].issue_dict = chiplet_core_target_dict[target_chiplet]
                        communication_event_tag = (communication_event_tag[0], communication_event_tag[1] + 1)
                    else:
                        # Inter-chiplet: Dijkstra path at chiplet level,
                        # decomposed into co2co + ch2ch + co2co segments
                        chiplet_path = chiplet_net.record_dijkstra_path(
                            source_node_idx=location[0],
                            target_node_idx=target_chiplet,
                            link_load=stream_size
                            )
                        cross_chiplet_tasks_list, first_ch2ch_tag = cross_chiplet(
                            communication_tag=communication_event_tag,
                            dependencies={(0, tag) for tag in data_created},
                            issues=chiplet_ops_set,
                            source_core_idx=(location[0], location[1]),
                            target_core_idx=chiplet_cores_dict[target_chiplet], 
                            chiplet_path=chiplet_path,
                            work_loads=stream_size,
                            chiplet_width=chiplet_net.width,
                            chiplet_height=chiplet_net.height,
                            core_width=cores_nets[0].width,
                            core_height=cores_nets[0].height
                        )

                        computation_update_flag = True
                        for cross_chiplet_task_tag in cross_chiplet_tasks_list:
                            cross_chiplet_task = cross_chiplet_tasks_list[cross_chiplet_task_tag]
                            if cross_chiplet_task[0] == "co2co":
                                source_core_node = cross_chiplet_task[1][1]
                                target_cores_node = {core_idx[1] for core_idx in cross_chiplet_task[2]}
                                core_path_beginner, core_path_ender, core_path_relationship = cores_nets[cross_chiplet_task[1][0]].xy_broadcast(
                                    # core_path_beginner, core_path_ender, core_path_relationship = cores_nets[cross_chiplet_task[1][0]].record_dijkstra_broadcast(
                                    source_node_idx=source_core_node,
                                    target_nodes_idx=target_cores_node,
                                    link_load=cross_chiplet_task[3]
                                )
                                pass_path = core_path_relationship
                            else:
                                pass_path = (cross_chiplet_task[1], cross_chiplet_task[2])
                            events_dict[cross_chiplet_task_tag] = CommunicationEvent(
                                event_tag=cross_chiplet_task[4],
                                source_node_idx=cross_chiplet_task[1],
                                pass_path=pass_path,
                                pass_type=cross_chiplet_task[0],
                                message_bytes=cross_chiplet_task[3],
                                chiplet_idx=cross_chiplet_task[1][0] if cross_chiplet_task[0] == "co2co" else cross_chiplet_task[1]
                            )
                            events_dict[cross_chiplet_task[4]].add_dependency(cross_chiplet_task[5])
                            events_dict[cross_chiplet_task[4]].add_issue(cross_chiplet_task[6])
                            if cross_chiplet_task[4][1] < first_ch2ch_tag[1]:
                                for data_splits in data_parts:
                                    data_dict[data_splits[0]].split_data_dict[(data_splits[0][1], data_splits[1])].locations |= chiplet_cores_dict[target_chiplet]
                                    for previous_created_tag in data_dict[data_splits[0]].split_data_dict[(data_splits[0][1], data_splits[1])].created:
                                            events_dict[(0, previous_created_tag)].issue_set.add(cross_chiplet_task[4])
                            if cross_chiplet_task[4][1] > first_ch2ch_tag[1]:
                                for next_computation_tag in chiplet_ops_set:
                                    computation_event_denpendencies_dict[next_computation_tag].add(cross_chiplet_task[4])

                        communication_event_tag = (cross_chiplet_task[4][0], cross_chiplet_task[4][1] + 1)

        # -- Emit computation events and advance the frontier --
        new_start_ops = []
        for op in start_ops:
            for op_location in op.locations:
                events_dict[(0, op.op_tag)] = ComputationEvent(
                    event_tag=(0, op.op_tag),
                    node_idx=op_location,
                    node_type=op.op_type,
                    ops=op.op_cnts
                )
                events_dict[(0, op.op_tag)].add_dependency(computation_event_denpendencies_dict[(0, op.op_tag)])
                computation_event_tag = (computation_event_tag[0], computation_event_tag[1] + 1)
            for new_start_op in op.issues:
                wait_ops_dependency_dict[new_start_op].add(op.op_tag)
                if new_start_op in wait_ops and wait_ops_dependency_dict[new_start_op] == op_dict[new_start_op].dependencies:
                    new_start_ops.append(op_dict[new_start_op])
                    wait_ops.remove(new_start_op)
        start_ops = new_start_ops


        # for start_op in start_ops:
        loop_cnt += 1
        if loop_cnt > loop_limit:
            raise ValueError("The event builder is flying off")


    # Reset network state after event construction (remove accumulated loads)
    chiplet_net.init_nodes()
    chiplet_net.init_links()
    chiplet_net.dijkstra_offload()

    for core_net in cores_nets:
        core_net.init_nodes()
        core_net.init_links()
        core_net.dijkstra_offload()

    return events_dict


if __name__ == "__main__":

    sys.path.append(os.path.join(file_path, '../components/'))
    sys.path.append(os.path.join(file_path, '../partition/op/'))
    sys.path.append(os.path.join(file_path, '../partition/'))
    sys.path.append(os.path.join(file_path, '../mapping/'))

    from decoder import parallel_decoder, decoder_offline_data_build, decoder_medium_data_build
    from attention import attention_medium_data_build
    from ffn import ffn_medium_data_build
    from partition import split_list2degree

    import math


    decoder_data_type = ["fp16", "fp16", "fp16", "fp16"]
    batch_size = 1
    sequence_length = 128
    hidden_states = 2048
    head_num = 16
    head_dim = hidden_states // head_num
    ffn_dim = 8192
    online_data_tag = 0
    offline_data_tag = 0
    decoder_attn_split_list = [[1], [64, 64], [32, 32, 32, 32], [256, 256, 256, 256, 256, 256, 256, 256]]
    decoder_ffn_split_list = [[1], [32, 32, 32, 32], [256, 256, 256, 256, 256, 256, 256, 256], [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024], [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024], [256, 256, 256, 256, 256, 256, 256, 256]]
    next_decoder_attn_split_list = [[1], [64, 64], [32, 32, 32, 32], [256, 256, 256, 256, 256, 256, 256, 256]]
    func_name = "decoder"
    attn_data_split_degree = split_list2degree(decoder_attn_split_list)
    ffn_data_split_degree = split_list2degree(decoder_ffn_split_list)
    next_attn_data_split_degree = split_list2degree(next_decoder_attn_split_list)
    activation_data = data_notation(data_name="activation", data_kind=0, data_tag=online_data_tag, data_shape=[batch_size, sequence_length, hidden_states], data_type=decoder_data_type[0], data_split_degree=[attn_data_split_degree[0], sequence_length, math.gcd(attn_data_split_degree[-1], ffn_data_split_degree[2])])
    online_data_tag += 1
    next_offline_data_tag, attention_offline_data, ffn_offline_data = decoder_offline_data_build(
        offline_data_tag=offline_data_tag,
        data_name=func_name,
        hidden_states=hidden_states,
        ffn_dim=ffn_dim,
        ops_type_list=decoder_data_type,
        attn_data_split_degree=[1, head_num, attn_data_split_degree[-1]],
        ffn_data_split_degree=ffn_data_split_degree[2:],
    )
    next_online_data_tag, Q_data, K_data, KT_data, V_data, QKT_data, S_data, P_medium_data, P_data, Cproj_medium_data = attention_medium_data_build(
        online_data_tag=online_data_tag,
        data_name=func_name,
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_states=hidden_states,
        head_num=head_num,
        head_dim=head_dim,
        ops_type_list=[decoder_data_type[-1]],
        attn_split_degree=attn_data_split_degree
    ) 
    attn_medium_data = [Q_data, K_data, KT_data, V_data, QKT_data, S_data, P_medium_data, P_data, Cproj_medium_data] if len(decoder_attn_split_list[2]) > 1 else [Q_data, K_data, KT_data, V_data, QKT_data, S_data, P_data, Cproj_medium_data]
    next_online_data_tag, medium1_data, fc1_data, relu1_data, medium2_data = ffn_medium_data_build(
        medium_data_tag=next_online_data_tag, 
        data_name=func_name, 
        batch_size=batch_size, 
        sequence_length=sequence_length, 
        hidden_states=hidden_states, 
        ffn_dim=ffn_dim, 
        ops_type_list=decoder_data_type[0:1], 
        ffn_split_degree=ffn_data_split_degree
    )
    if len(decoder_ffn_split_list[2]) > 1 and len(decoder_ffn_split_list[4]) > 1:
        ffn_medium_data = [medium1_data, fc1_data, relu1_data, medium2_data]
    elif len(decoder_ffn_split_list[2]) > 1:
        ffn_medium_data = [medium1_data, fc1_data, relu1_data]
    elif len(decoder_ffn_split_list[4]) > 1:
        ffn_medium_data = [fc1_data, relu1_data, medium2_data]
    else:
        ffn_medium_data = [fc1_data, relu1_data]

    next_online_data_tag, online_medium_data = decoder_medium_data_build(
        online_data_tag=next_online_data_tag,
        data_name="decoder",
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_states=hidden_states,
        ffn_dim=ffn_dim,
        ops_type_list=decoder_data_type,
        attn_split_degree=attn_data_split_degree,
        ffn_split_degree=ffn_data_split_degree
    )    
    ofmap_data = data_notation(data_name="ofmap", data_kind=0, data_tag=next_online_data_tag, data_shape=[batch_size, sequence_length, hidden_states], data_type=decoder_data_type[-1], data_split_degree=[ffn_data_split_degree[0], sequence_length, math.gcd(attn_data_split_degree[-1], ffn_data_split_degree[-1], next_attn_data_split_degree[2])])
    online_data_tag += 1
    decoder_ops, decoder_next_ops_tag = parallel_decoder(
        attention_split_list=decoder_attn_split_list,
        ffn_split_list=decoder_ffn_split_list, 
        ops_type_list=decoder_data_type, 
        online_input_data=[activation_data], 
        attention_offline_data=attention_offline_data,
        attention_online_medium_data=attn_medium_data,
        ffn_offline_data=ffn_offline_data,
        ffn_online_medium_data=ffn_medium_data,
        online_medium_data=online_medium_data,
        online_output_data=[ofmap_data],
        hidden_states=hidden_states,
        head_num=head_num, 
        head_dim=head_dim, 
        block_idx=0,
        func_name="decoder",  
        op_tag=0,
        dependencies=set(),        
    )


    # for op in decoder_ops:
    decoder_data_dict = {}
    decoder_data_dict[(activation_data.data_kind, activation_data.data_tag)] = activation_data
    decoder_data_dict[(ofmap_data.data_kind, ofmap_data.data_tag)] = ofmap_data
    for data in attention_offline_data:
        if isinstance(data, list):
            for sub_data in data:
                decoder_data_dict[(sub_data.data_kind, sub_data.data_tag)] = sub_data
        else:
            decoder_data_dict[(data.data_kind, data.data_tag)] = data
    for data in ffn_offline_data:
        if isinstance(data, list):
            for sub_data in data:
                decoder_data_dict[(sub_data.data_kind, sub_data.data_tag)] = sub_data
        else:
            decoder_data_dict[(data.data_kind, data.data_tag)] = data
    for data in attn_medium_data:
        if isinstance(data, list):
            for sub_data in data:
                decoder_data_dict[(sub_data.data_kind, sub_data.data_tag)] = sub_data
        else:
            decoder_data_dict[(data.data_kind, data.data_tag)] = data
    for data in ffn_medium_data:
        if isinstance(data, list):
            for sub_data in data:
                decoder_data_dict[(sub_data.data_kind, sub_data.data_tag)] = sub_data
        else:
            decoder_data_dict[(data.data_kind, data.data_tag)] = data
    for data in online_medium_data:
        if isinstance(data, list):
            for sub_data in data:
                decoder_data_dict[(sub_data.data_kind, sub_data.data_tag)] = sub_data
        else:
            decoder_data_dict[(data.data_kind, data.data_tag)] = data


    from mapping import hardware_building, chiplet_mapping
    from allocation import hamiltonian_allocation

    transformerblock_num = 24

    hardware_description = {
        "chiplet_number": 25,
        "chiplet_rows": 5,
        "chiplet_columns": 5,
        "core_number": 25,
        "core_rows": 5,
        "core_columns": 5,
        "tensor_core_number": 1,
        "tensor_core_computation_power": 1024,
        "tensor_core_computation_latency": 1,
        "vector_unit_number": 1,
        "vector_unit_computation_power": 256,
        "vector_unit_computation_latency": 4,
        "ch2ch_latency": 100,
        "ch2ch_bandwidth": 100,
        "co2co_latency": 10,
        "co2co_bandwidth": 1024
    }    

    tensor_core_dict, vector_unit_dict, ch2ch_link_dict, co2co_link_dict, core_distance_dict = hardware_building(hardware_description)
    computation_devices = {"tensor": tensor_core_dict, "vector": vector_unit_dict}
    communication_devices = {"ch2ch": ch2ch_link_dict, "co2co": co2co_link_dict}
    hamiltonian_cluster = hamiltonian_allocation(
        num_x=hardware_description["chiplet_rows"], 
        num_y=hardware_description["chiplet_columns"],
        x_first=True
    )

    block_chiplet = chiplet_mapping(
        transformerblock_num=transformerblock_num,
        chipletcluster_list=hamiltonian_cluster,
        tensor_core_dict=tensor_core_dict,
        vector_unit_dict=vector_unit_dict
    )

    chiplet_net = mesh_2d(hardware_description["chiplet_columns"], hardware_description["chiplet_rows"])
    cores_nets = [mesh_2d(hardware_description["core_columns"], hardware_description["core_rows"]) for _ in range(hardware_description["chiplet_number"])]


    for split_x in decoder_data_dict[(0, 0)].split_data_dict:
        decoder_data_dict[(0, 0)].split_data_dict[split_x].locations.add((0, 0))

    events = event_builder(
        op_dict=decoder_ops, 
        data_dict=decoder_data_dict,
        block_candidates=block_chiplet,
        dist_dict=core_distance_dict,
        chiplet_net=chiplet_net,
        cores_nets=cores_nets
    )

    for event in events:
        print(event, events[event])
