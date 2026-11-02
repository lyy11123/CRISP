
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, './topology/'))
sys.path.append(os.path.join(file_path, '../../components/'))


from net import net
from Task import CommunicationTask

from typing import List, Tuple, Dict, Set
from copy import deepcopy
import random
from tqdm import tqdm


def path_to_linkspath(path: List[Tuple[int]]) -> List[Tuple[int, int]]:
    linkspath = []
    path_length  = len(path) - 2
    loop_range = range(path_length, -1, -1)
    for idx in loop_range:
        linkspath.append((path[idx], path[idx + 1]))
    return linkspath


def pair_path_allocation(task_list: List[CommunicationTask], network: net, iteration_number=100) -> List[List[Tuple[int, int]]]:

    backup_task_list = []
    backup_busytimes = float('inf')
    bakcup_path_list = []

    current_path_set = set()
    default_order_task_list = deepcopy(task_list)
    network.dijkstra_offload()

    for task_idx, task in enumerate(default_order_task_list):
        if task.built_flag is False:
            raise ValueError("Task dependencies are not built yet")
        source_node_idx = task.source_node_idx
        target_node_idx = task.target_node_idx
        source_node_coordinate = list(network.idx_to_coordinate(source_node_idx))
        target_node_coordinate = list(network.idx_to_coordinate(target_node_idx))
        hops_list = [abs(s - t) for s, t in zip(source_node_coordinate, target_node_coordinate)]
        task_list[task_idx].hops = sum(hops_list)
        task_path = network.record_dijkstra_path(source_node_idx, target_node_idx)
        task.path = path_to_linkspath(task_path)
        current_path_set.add(tuple(task_path))

    busy_times = max(network.link_loads_dict.values())
    backup_busytimes = busy_times
    backup_task_list = [deepcopy(default_order_task_list)]
    bakcup_path_list = [current_path_set]


    current_path_set = set()
    long_order_task_list = deepcopy(task_list)
    long_order_task_list = sorted(long_order_task_list, key=lambda x: x.hops, reverse=True)
    network.dijkstra_offload()

    for task in long_order_task_list:
        source_node_idx = task.source_node_idx
        target_node_idx = task.target_node_idx
        task_path = network.record_dijkstra_path(source_node_idx, target_node_idx)
        task.path = path_to_linkspath(task_path)
        current_path_set.add(tuple(task_path))

    if current_path_set in bakcup_path_list:
        pass
    else:
        busy_times = max(network.link_loads_dict.values())
        if busy_times < backup_busytimes:
            backup_busytimes = busy_times
            backup_task_list = [deepcopy(long_order_task_list)]
        elif busy_times == backup_busytimes:
            backup_task_list.append(deepcopy(long_order_task_list))
        else:
            pass

    for iter in range(iteration_number):

        current_path_set = set()
        default_order_task_list = deepcopy(long_order_task_list)
        random.shuffle(default_order_task_list)
        network.dijkstra_offload()

        for task_idx, task in enumerate(default_order_task_list):
            source_node_idx = task.source_node_idx
            target_node_idx = task.target_node_idx
            task_path = network.record_dijkstra_path(source_node_idx, target_node_idx)
            task.path = path_to_linkspath(task_path)
            current_path_set.add(tuple(task_path))

        if current_path_set in bakcup_path_list:
            continue
        else:
            busy_times = max(network.link_loads_dict.values())
            if busy_times < backup_busytimes:
                backup_busytimes = busy_times
                backup_task_list = [deepcopy(default_order_task_list)]
            elif busy_times == backup_busytimes:
                backup_task_list.append(deepcopy(default_order_task_list))
            else:
                pass

        current_path_set = set()
        long_order_task_list = deepcopy(default_order_task_list)
        long_order_task_list = sorted(long_order_task_list, key=lambda x: x.hops, reverse=True)
        network.dijkstra_offload()

        for task in long_order_task_list:
            source_node_idx = task.source_node_idx
            target_node_idx = task.target_node_idx
            task_path = network.record_dijkstra_path(source_node_idx, target_node_idx)
            task.path = path_to_linkspath(task_path)
            current_path_set.add(tuple(task_path))

        if current_path_set in bakcup_path_list:
            continue
        else:
            busy_times = max(network.link_loads_dict.values())
            if busy_times < backup_busytimes:
                backup_busytimes = busy_times
                backup_task_list = [deepcopy(long_order_task_list)]
            elif busy_times == backup_busytimes:
                backup_task_list.append(deepcopy(long_order_task_list))
            else:
                pass

    print("The best busy times is", backup_busytimes)

    return backup_task_list, backup_busytimes


def Dijkstra_path_allocation(task_list: List[int], network: net, iteration_number=100) -> List[List[Tuple[int, int]]]:


    backup_task_list = []
    backup_busytimes = float('inf')
    bakcup_path_list = []

    current_path_set = []
    default_order_task_list = deepcopy(task_list)
    network.init_nodes()
    network.init_links()
    network.dijkstra_offload()

    original_task = []
    for source_node_idx, target_nodes_idx, work_loads in default_order_task_list:
        task_source, task_end, task_path = network.record_dijkstra_broadcast(source_node_idx, target_nodes_idx, work_loads)
        if task_path == {}:
            raise ValueError("No path found")
        current_path_set.append([source_node_idx, task_path, work_loads, (source_node_idx, target_nodes_idx)])
        original_task.append([source_node_idx, target_nodes_idx, work_loads, abs(network.idx_to_coordinate(task_source)[0] - network.idx_to_coordinate(task_end)[0]) + abs(network.idx_to_coordinate(task_source)[1] - network.idx_to_coordinate(task_end)[1])])

    busy_times = max(network.link_loads_dict.values())
    backup_busytimes = busy_times
    backup_task_list = [deepcopy(default_order_task_list)]
    bakcup_path_list = [current_path_set]

    current_path_set = []
    long_order_task_list = deepcopy(original_task)
    long_order_task_list = sorted(long_order_task_list, key=lambda x: x[3], reverse=True)
    network.init_nodes()
    network.init_links()
    network.dijkstra_offload()

    for source_node_idx, target_nodes_idx, work_loads, maxdist in long_order_task_list:
        task_source, task_end, task_path = network.record_dijkstra_broadcast(source_node_idx, target_nodes_idx, work_loads)
        current_path_set.append([source_node_idx, task_path, work_loads, (source_node_idx, target_nodes_idx)])
        original_task.append([source_node_idx, target_nodes_idx, work_loads, abs(network.idx_to_coordinate(task_source)[0] - network.idx_to_coordinate(task_end)[0]) + abs(network.idx_to_coordinate(task_source)[1] - network.idx_to_coordinate(task_end)[1])])

    busy_times = max(network.link_loads_dict.values())
    if busy_times < backup_busytimes:
        backup_busytimes = busy_times
        bakcup_task_list = [deepcopy(long_order_task_list)]
        bakcup_path_list = [current_path_set]
    elif busy_times == backup_busytimes:
        backup_task_list.append(deepcopy(long_order_task_list))
        bakcup_path_list.append(current_path_set)
    else:
        pass

    for iter in range(iteration_number):

        current_path_set = []
        default_order_task_list = deepcopy(long_order_task_list)
        random.shuffle(default_order_task_list)
        network.dijkstra_offload()

        for source_node_idx, target_nodes_idx, work_loads, maxdist in default_order_task_list:
            task_source, task_end, task_path = network.record_dijkstra_broadcast(source_node_idx, target_nodes_idx, work_loads)
            current_path_set.append([source_node_idx, task_path, work_loads, (source_node_idx, target_nodes_idx)])
            original_task.append([source_node_idx, target_nodes_idx, work_loads, abs(network.idx_to_coordinate(task_source)[0] - network.idx_to_coordinate(task_end)[0]) + abs(network.idx_to_coordinate(task_source)[1] - network.idx_to_coordinate(task_end)[1])])

        if current_path_set in bakcup_path_list:
            continue
        else:
            busy_times = max(network.link_loads_dict.values())
            if busy_times < backup_busytimes:
                backup_busytimes = busy_times
                bakcup_task_list = [deepcopy(default_order_task_list)]
                bakcup_path_list = [current_path_set]
            elif busy_times == backup_busytimes:
                backup_task_list.append(deepcopy(default_order_task_list))
                bakcup_path_list.append(current_path_set)
            else:
                pass

        current_path_set = []
        long_order_task_list = deepcopy(default_order_task_list)
        long_order_task_list = sorted(long_order_task_list, key=lambda x: x[3], reverse=True)
        network.init_nodes()
        network.init_links()
        network.dijkstra_offload()

        for source_node_idx, target_nodes_idx, work_loads, maxdist in long_order_task_list:
            task_source, task_end, task_path = network.record_dijkstra_broadcast(source_node_idx, target_nodes_idx, work_loads)
            current_path_set.append([source_node_idx, task_path, work_loads, (source_node_idx, target_nodes_idx)])
            original_task.append([source_node_idx, target_nodes_idx, work_loads, abs(network.idx_to_coordinate(task_source)[0] - network.idx_to_coordinate(task_end)[0]) + abs(network.idx_to_coordinate(task_source)[1] - network.idx_to_coordinate(task_end)[1])])

        if current_path_set in bakcup_path_list:
            continue
        else:
            busy_times = max(network.link_loads_dict.values())
            if busy_times < backup_busytimes:
                backup_busytimes = busy_times
                bakcup_task_list = [deepcopy(long_order_task_list)]
                bakcup_path_list = [current_path_set]
            elif busy_times == backup_busytimes:
                backup_task_list.append(deepcopy(long_order_task_list))
                bakcup_path_list.append(current_path_set)
            else:
                pass

    return bakcup_path_list, backup_busytimes


def cross_chiplet(
    communication_tag: Tuple[int], dependencies: Set[int], issues: Set[int],
    source_core_idx: Tuple[int], target_core_idx: Set[Tuple[int]], chiplet_path: List[int], work_loads: int,
    chiplet_width: int, chiplet_height: int, 
    core_width: int, core_height: int
    ):

    cross_chiplet_transmission = {}
    for source_y in range(chiplet_height):
        for source_x in range(chiplet_width):
            source_chiplet_idx = source_y * chiplet_width + source_x
            for target_y in range(chiplet_height):
                for target_x in range(chiplet_width):
                    target_chiplet_idx = target_y * chiplet_width + target_x
                    if source_chiplet_idx == target_chiplet_idx:
                        continue
                    else:
                        x_dist = source_x - target_x
                        y_dist = source_y - target_y
                        if (abs(x_dist) == 1 and source_y == target_y) or (abs(y_dist) == 1 and source_x == target_x):
                            chiplet_neighbor = (source_chiplet_idx, target_chiplet_idx)
                            if x_dist == 1:
                                source_cores = [(source_chiplet_idx, core_width * core_row_idx) for core_row_idx in range(core_height)]
                                target_cores = [(target_chiplet_idx, core_width * (core_row_idx + 1) - 1) for core_row_idx in range(core_height)]
                            elif x_dist == -1:
                                source_cores = [(source_chiplet_idx, core_width * (core_row_idx + 1) - 1) for core_row_idx in range(core_height)]
                                target_cores = [(target_chiplet_idx, core_width * core_row_idx) for core_row_idx in range(core_height)]
                            elif y_dist == 1:
                                source_cores = [(source_chiplet_idx, core_column_idx) for core_column_idx in range(core_width)]
                                target_cores = [(target_chiplet_idx, core_width * (core_height - 1) + core_column_idx) for core_column_idx in range(core_width)]
                            else:
                                source_cores = [(source_chiplet_idx, core_width * (core_height - 1) + core_column_idx) for core_column_idx in range(core_width)]
                                target_cores = [(target_chiplet_idx, core_column_idx) for core_column_idx in range(core_width)]
                            cross_chiplet_transmission[chiplet_neighbor] = (source_cores, target_cores)
                        else:
                            continue

    current_communication_tag = communication_tag
    find_first_ch2ch = True
    cross_chiplet_tasks_list = {}
    chiplet_path_length = len(chiplet_path) - 1
    chiplet_dependencies = set()
    for source_chiplet_core_idx in cross_chiplet_transmission[(chiplet_path[0], chiplet_path[1])][0]:
        if source_chiplet_core_idx == source_core_idx:
            continue
        else:
            cross_chiplet_tasks_list[current_communication_tag] = ["co2co", source_core_idx, {source_chiplet_core_idx}, work_loads//len(cross_chiplet_transmission[(chiplet_path[0], chiplet_path[1])][0]), current_communication_tag, dependencies, set()]
            chiplet_dependencies.add(current_communication_tag)
            current_communication_tag = (current_communication_tag[0], current_communication_tag[1] + 1)
    for chiplet_path_idx in range(chiplet_path_length):
        source_chiplet_idx = chiplet_path[chiplet_path_idx]
        target_chiplet_idx = chiplet_path[chiplet_path_idx + 1]
        if find_first_ch2ch:
            find_first_ch2ch2_tag = current_communication_tag
            find_first_ch2ch = False
        cross_chiplet_tasks_list[current_communication_tag] = ["ch2ch", source_chiplet_idx, target_chiplet_idx, work_loads, current_communication_tag, chiplet_dependencies, set()]
        for com_tag in chiplet_dependencies:
            cross_chiplet_tasks_list[com_tag][-1].add(current_communication_tag)
        chiplet_dependencies = {current_communication_tag}
        current_communication_tag = (current_communication_tag[0], current_communication_tag[1] + 1)

        core_dependencies = set()
        source_cores, target_cores = cross_chiplet_transmission[(source_chiplet_idx, target_chiplet_idx)]
        if chiplet_path_idx == chiplet_path_length - 1:
            for target_chiplet_core_idx in target_cores:
                target_cores_idx = deepcopy(target_core_idx)
                if target_chiplet_core_idx in target_cores_idx:
                    target_cores_idx.remove(target_chiplet_core_idx)
                cross_chiplet_tasks_list[current_communication_tag] = ["co2co", target_chiplet_core_idx, target_cores_idx, work_loads//len(target_cores), current_communication_tag, chiplet_dependencies, issues]
                core_dependencies.add(current_communication_tag)
                current_communication_tag = (current_communication_tag[0], current_communication_tag[1] + 1)
            for com_tag in chiplet_dependencies:
                cross_chiplet_tasks_list[com_tag][-1] = core_dependencies

        else:
            next_source_cores, next_target_cores = cross_chiplet_transmission[(target_chiplet_idx, chiplet_path[chiplet_path_idx + 2])]
            transmit_cores = set(target_cores) & set(next_source_cores)
            gather_dependencies = set()
            for gather_core_idx in target_cores:
                for transmit_core_idx in transmit_cores:
                    if gather_core_idx == transmit_core_idx:
                        continue
                    else:
                        cross_chiplet_tasks_list[current_communication_tag] = ["co2co", gather_core_idx, transmit_cores, work_loads//(len(transmit_cores) * len(target_cores)), current_communication_tag, chiplet_dependencies, set()]
                        gather_dependencies.add(current_communication_tag)
                        current_communication_tag = (current_communication_tag[0], current_communication_tag[1] + 1)
            for com_tag in chiplet_dependencies:
                cross_chiplet_tasks_list[com_tag][-1] = gather_dependencies

            scatter_dependencies = set()
            for scatter_core_idx in next_source_cores:
                for transmit_core_idx in transmit_cores:
                    if transmit_core_idx == scatter_core_idx:
                        continue
                    else:
                        cross_chiplet_tasks_list[current_communication_tag] = ["co2co", transmit_core_idx, scatter_core_idx, work_loads//(len(transmit_cores) * len(next_source_cores)), current_communication_tag, gather_dependencies, set()]
                        scatter_dependencies.add(current_communication_tag)
                        current_communication_tag = (current_communication_tag[0], current_communication_tag[1] + 1)
            for com_tag in gather_dependencies:
                cross_chiplet_tasks_list[com_tag][-1] = scatter_dependencies

            chiplet_dependencies = scatter_dependencies


    return cross_chiplet_tasks_list, find_first_ch2ch2_tag


def path_relationship_to_linkspath(start_idx: int, path_relationship: Dict[Tuple[int], Tuple[int]]) -> List[Tuple[int]]:

    linkspaths = []
    root_set = {start_idx}
    relationship_links = deepcopy(path_relationship)
    while relationship_links:
        root_node = root_set.pop()
        if root_node not in relationship_links:
            continue
        for child_node in relationship_links[root_node]:
            linkspaths.append((root_node, child_node))
            root_set.add(child_node)
        del relationship_links[root_node]
    return linkspaths


if __name__ == "__main__":

    sys.path.append(os.path.join(file_path, './collective_communication/'))

    from mesh_2d import mesh_2d
    from alltoall import alltoall_base

    path_iteration_number = 100

    length = 5
    width = length
    height = length
    nodes_number = width * height

    network = mesh_2d(width, height)
    task_list = []
    for node_idx in range(nodes_number):
        for node_idy in range(nodes_number):
            if node_idx == node_idy:
                continue
            else:
                task_list.append([node_idx, {node_idy}, 1])

    task_list, busytimes = Dijkstra_path_allocation(task_list, network, path_iteration_number)
    print("The best busy times is", busytimes, "possible paths are", len(task_list))
    print("An example of the task list is", task_list[0])


    corss_chiplets_tasks = cross_chiplet((1, 23), {(1, 20), (1, 21), (1, 22)}, {(0, 7)}, (0, 12), {(6, 11), (6, 3)}, [0, 1, 6], 10, 5, 5, 5, 5)
    for task_tag in corss_chiplets_tasks:
        print(task_tag, corss_chiplets_tasks[task_tag])

    linkspath_example = path_relationship_to_linkspath(0, {0: [1, 4], 1: [2], 2: [3], 4: [5]})
    print(linkspath_example)
