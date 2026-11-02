
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../../../components/'))

from Task import CommunicationTask

from typing import List


def scatter_base(scatter_task_tag: int, scatter_source_nodes_list: List[int], scatter_target_nodes_list: List[int], scatter_packets: int) -> List[CommunicationTask]:

    scatter_task_list = []
    iter_tag = scatter_task_tag
    packet_per_timestep = scatter_packets // (len(scatter_source_nodes_list) * len(scatter_target_nodes_list))
    if scatter_packets % (len(scatter_source_nodes_list) * len(scatter_target_nodes_list)) != 0:
        raise ValueError("scatter_packets should be divisible by the number of nodes in scatter_nodes_list")

    for source_node in scatter_source_nodes_list:
        for target_node in scatter_target_nodes_list:
            if source_node != target_node:
                scatter_task = CommunicationTask(iter_tag, source_node, target_node, packet_per_timestep)
                scatter_task.build_dependency(set())
                scatter_task_list.append(scatter_task)
                iter_tag += 1

    return scatter_task_list, packet_per_timestep


if __name__ == "__main__":

    scatter_task_tag = 0
    scatter_source_nodes_list = [0, 1, 3, 7]
    scatter_target_nodes_list = [2, 6]

    scatter_task_list, _ = scatter_base(scatter_task_tag, scatter_source_nodes_list, scatter_target_nodes_list, 80)

    for scatter_task in scatter_task_list:
        print(scatter_task)
