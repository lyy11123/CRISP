
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../../../components/'))

from Task import CommunicationTask

from typing import List


def custom_base(custom_task_tag: int, custom_pairs_list: List[List[int]], custom_packets_perpair: int) -> List[CommunicationTask]:

    custom_task_list = []
    iter_tag = custom_task_tag
    packet_per_timestep = custom_packets_perpair

    for communication_pair in custom_pairs_list:
        source_node = communication_pair[0]
        target_node = communication_pair[1]
        custom_task = CommunicationTask(iter_tag, source_node, target_node, packet_per_timestep)
        custom_task.build_dependency(set())
        custom_task_list.append(custom_task)
        iter_tag += 1

    return custom_task_list, packet_per_timestep


if __name__ == "__main__":

    custom_task_tag = 0
    custom_pairs_list = [[1, 6], [1, 8]]

    custom_task_list, _ = custom_base(custom_task_tag, custom_pairs_list, 80)

    for custom_task in custom_task_list:
        print(custom_task)
