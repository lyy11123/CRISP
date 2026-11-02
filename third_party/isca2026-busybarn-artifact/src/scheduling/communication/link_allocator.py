
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, './topology/'))
sys.path.append(os.path.join(file_path, '../../components/'))


from net import net
from Task import CommunicationTask
from Event import Event, CommunicationEvent

from typing import List, Tuple, Dict, Set
from copy import deepcopy
import random


def pair_link_allocate(tasks_list: List[CommunicationTask], network: net, iteration_number=10) -> List[List[Tuple[int, int]]]:

    flyingoff_value = (network.width * network.height) ** 4
    min_time_step = float('inf')
    best_steps = ()

    for task_list in tasks_list:

        for iteration in range(iteration_number):

            iteration_standard_task_list = deepcopy(task_list)
            if iteration > 0 and iteration % 2 == 0:
                random.shuffle(iteration_standard_task_list)
            elif iteration % 2 == 1:
                iteration_standard_task_list.sort(key=lambda x: (len(x.dependency_list), -x.hops))

            iteration_task_list = deepcopy(iteration_standard_task_list)
            iteration_task_set = set(iteration_task_list)

            time_step = 0
            steps = []

            while iteration_task_set:
                available_links = deepcopy(network.links_set)

                no_available_links = False
                debug_cnt = 0

                current_step_path = {}

                while no_available_links is False:
                    no_available_links = True
                    done_list = []
                    for task in iteration_task_list[:]:
                        task.update_dependency(done_list)
                        if task.path and (task.dependency_list == set()):
                            needed_link = task.path[-1]
                            if needed_link in available_links:
                                if task.task_tag not in current_step_path:
                                    current_step_path[task.task_tag] = [needed_link[0]]
                                current_step_path[task.task_tag].append(needed_link[1])
                                available_links.remove(needed_link)
                                task.path.pop()
                                if task.path == []:
                                    done_list.append(task.task_tag)
                                    iteration_task_set.remove(task)
                                    iteration_task_list.remove(task)
                                no_available_links = False
                                break
                            else:
                                continue
                        elif (task.path == []):
                            done_list.append(task.task_tag)
                            iteration_task_set.remove(task)
                            iteration_task_list.remove(task)
                            no_available_links = False
                            break
                        else:
                            continue

                    if debug_cnt > flyingoff_value:
                        raise ValueError("The scheduling algorithm is flying off")
                    debug_cnt += 1

                dual_path_dict = network.dualpath_dijkstra(current_step_path)

                if dual_path_dict:
                    time_step += 0.5
                    steps.append([current_step_path, dual_path_dict])
                else:
                    time_step += 1
                    steps.append([current_step_path, None])

                if time_step > flyingoff_value:
                    raise ValueError("The time step is flying off")

            if time_step < min_time_step and steps != []:
                min_time_step = time_step
                best_steps = [steps]
            elif time_step == min_time_step and steps not in best_steps:
                best_steps.append(steps)
            else:
                continue

    return min_time_step, best_steps


def dict_link_allocate(tasks_list: List[int], event_tag: Tuple[int], dependencies: Dict[Tuple[int], Set[int]], issues: Dict[int, Set[int]], iteration_number:int=10) -> List[CommunicationEvent]:

    iteration_list = []
    for _ in range(iteration_number):
        commuication_events_list = []
        random.shuffle(tasks_list)
        for tasks in tasks_list:
            for task in tasks:
                task_source_idx = task[0]
                task_route = task[1]
                task_bytes = task[2]
                task_pairs = task[3]
                task_path = []
                for pair_source in task_route:
                    for pair_target in task_route[pair_source]:
                        task_path.append((pair_source, pair_target))
                        communication_event = CommunicationEvent(event_tag, task_source_idx, task_path, "co2co", task_bytes)
                        communication_event.build_dependency(dependencies[task_pairs])
                        communication_event.add_issue(issues[task_pairs])
                commuication_events_list.append(communication_event)
                event_tag = (event_tag[0], event_tag[1] + 1)
        iteration_list.append(commuication_events_list)

    return iteration_list


def Dijkstra_link_allocation(task_list, net, iteration_number=100):

    def convert_paths(graph, start):
        def explore(node, path):
            if node not in graph or not graph[node]:
                return [path]
            results = []
            for next_node in graph[node]:
                results.extend(explore(next_node, path + [(node, next_node)]))
            return results

        return explore(start, [])

    task_paths_list = []
    for task in task_list:
        paths_list = []
        for task_path in task:
            print("*****", task_path[0], task_path[1])
            paths_list.append(convert_paths(task_path[1], task_path[0]))
        task_paths_list.append(paths_list)

    min_time_step = float('inf')
    for paths_list in task_paths_list:
        for iteration in range(iteration_number):
            iter_paths_list = deepcopy(paths_list)

            if iteration > 0 and iteration % 2 == 0:
                random.shuffle(iter_paths_list)
            elif iteration % 2 == 1:
                iter_paths_list.sort(key=lambda x: (sum(len(lst) for lst in x), -len(x)))


            iter_time_step = 0
            iter_steps = []
            while iter_paths_list:
                available_links = deepcopy(net.links_set)

                find_flag = True
                current_step = {}
                while find_flag:
                    find_flag = False
                    for broadcast_idx, broadcast_paths in enumerate(iter_paths_list[:]):
                        for paths_idx, paths in enumerate(broadcast_paths[:]):
                            if paths[0] in available_links:
                                available_links.remove(paths[0])
                                if (broadcast_idx, paths_idx) not in current_step:
                                    current_step[(broadcast_idx, paths_idx)] = [paths[0]]
                                else:
                                    current_step[(broadcast_idx, paths_idx)].append(paths[0])
                                popped_path = paths.pop(0)
                                if paths == []:
                                    broadcast_paths.remove(paths)
                                    if broadcast_paths == []:
                                        iter_paths_list.remove(broadcast_paths)
                                find_flag = True
                                break
                            else:
                                continue
                        for paths in broadcast_paths[:]:
                            if paths[0] == popped_path:
                                paths.pop(0)
                                if paths == []:
                                    broadcast_paths.remove(paths)
                                    if broadcast_paths == []:
                                        iter_paths_list.remove(broadcast_paths)
                        popped_path = ()

                iter_time_step += 1
                iter_steps.append(current_step)

            if iter_time_step < min_time_step:
                min_time_step = iter_time_step
                best_steps = [iter_steps]
            elif iter_time_step == min_time_step and iter_steps not in best_steps:
                best_steps.append(iter_steps)

        return min_time_step, best_steps


if __name__ == "__main__":

    sys.path.append(file_path)
    sys.path.append(os.path.join(file_path, './collective_communication/'))


    path_iteration_number = 100
    link_iteration_number = 10


    from mesh2d import mesh_2d
    from alltoall import alltoall_base
    from path_allocator import pair_path_allocation, Dijkstra_path_allocation


    width = 5
    height = 5
    number = width * height
    network = mesh_2d(width, height)

    allgather_pairs = []
    for i in range(number):
        target_set = set(range(number))
        target_set.remove(i)
        allgather_pairs.append([(1, i), i, target_set, 1024*1024])

    network.init_nodes()
    network.init_links()
    network.dijkstra_offload()
    max_link_load, total_link_load, paths = network.record_dijkstra_multicast_path(allgather_pairs)
    print(max_link_load, total_link_load)

    task_list = []
    for tag in paths:
        print(paths[tag])
        task_list.append([tag[1], paths[tag], 1, set()])
    for task in task_list:
        print(task)
    link_iteration_number = 10
    timestep, steps = Dijkstra_link_allocation([task_list], network, link_iteration_number)
    print(timestep, len(steps))
