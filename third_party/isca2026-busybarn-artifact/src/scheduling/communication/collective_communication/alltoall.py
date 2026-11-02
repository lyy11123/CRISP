
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../../../components/'))

from Task import CommunicationTask

from typing import List


def alltoall_base(alltoall_task_tag: int, alltoall_nodes_list: List[int], alltoall_packets: int) -> List[CommunicationTask]:

    alltoall_task_list = []
    iter_tag = alltoall_task_tag
    packet_per_timestep = alltoall_packets // len(alltoall_nodes_list)
    if alltoall_packets % len(alltoall_nodes_list) != 0:
        raise ValueError("alltoall_packets should be divisible by the number of nodes in alltoall_nodes_list")

    for source_node in alltoall_nodes_list:
        for target_node in alltoall_nodes_list:
            if source_node != target_node:
                alltoall_task = CommunicationTask(iter_tag, source_node, target_node, packet_per_timestep)
                alltoall_task.build_dependency(set())
                alltoall_task_list.append(alltoall_task)
                iter_tag += 1

    return alltoall_task_list, packet_per_timestep


if __name__ == "__main__":

    alltoall_task_tag = 0
    alltoall_nodes_list = [0, 1, 3, 7]

    alltoall_task_list, _ = alltoall_base(alltoall_task_tag, alltoall_nodes_list)

    for alltoall_task in alltoall_task_list:
        print(alltoall_task)
