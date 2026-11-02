
"""WAMIS-HD topology: Wafer-scale Multi-die Integrated System with
high-bandwidth DDR links, two-level mesh routing, fault-tolerant XY-YX
backtracking, and distance-aware multicast for LLM inference."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../../platform/device/'))
sys.path.append(os.path.join(file_path, '../../../platform/device/link/'))
sys.path.append(os.path.join(file_path, '../../../platform/device/module/'))
sys.path.append(os.path.join(file_path, '../../../../utils/'))


from net import net, TupleIdx
from planar_2d import planar_2d
from device import device
from link import link
from ch2ch import ch2ch
from co2co import co2co
from ddr2co import ddr2co
from co2ddr import co2ddr
from module import module
from tensorcore import tensorcore
from vectorunit import vectorunit
from read_cfg import cfg_to_dict


from functools import lru_cache
from typing import List, Tuple, Set, Dict, Deque
from collections import deque, defaultdict
import itertools
import heapq
import numpy as np
import random
random.seed(123)
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import networkx as nx


# HBM on corner
class wamis_hdc(planar_2d):


    def __init__(self, platform_cfg: Dict[str, int]):
        super().__init__(platform_cfg=platform_cfg)


    def init_links(self):

        co_topleft_corner_coordinate = (0, 0)
        co_topright_corner_coordinate = (0, self.l1_width - 1)
        co_bottomleft_corner_coordinate = (self.l1_height - 1, 0)
        co_bottomright_corner_coordinate = (self.l1_height - 1, self.l1_width - 1)
        corner_coordinates = {co_topleft_corner_coordinate, co_topright_corner_coordinate,
                              co_bottomleft_corner_coordinate, co_bottomright_corner_coordinate}

        self.links_set = set()
        self.links_list = []
        self.available_neighbors = {}
        self.links_dict = {}

        for source_node_idx in self.nodes_set:
            # intra-chiplet
            source_node_coordinate = self.nodes_coordinate_dict[source_node_idx]
            top_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1], source_node_coordinate[2] - 1, source_node_coordinate[3])
            bottom_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1], source_node_coordinate[2] + 1, source_node_coordinate[3])
            left_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1], source_node_coordinate[2], source_node_coordinate[3] - 1)
            right_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1], source_node_coordinate[2], source_node_coordinate[3] + 1)
            if top_node_coordinate in self.nodes_coordinate_idx_dict and top_node_coordinate not in self.failed_nodes:
                target_node_idx = self.nodes_coordinate_idx_dict[top_node_coordinate]
                if (source_node_idx, target_node_idx) not in self.failed_links:
                    self.links_set.add((source_node_idx, target_node_idx))
                    if source_node_idx in self.available_neighbors:
                        self.available_neighbors[source_node_idx].add(target_node_idx)
                    else:
                        self.available_neighbors[source_node_idx] = {target_node_idx}
                    self.links_dict[(source_node_idx, target_node_idx)] = co2co(co2co_id=(source_node_idx, target_node_idx), co2co_cfg=self.co2co_cfg)
            if bottom_node_coordinate in self.nodes_coordinate_idx_dict and bottom_node_coordinate not in self.failed_nodes:
                target_node_idx = self.nodes_coordinate_idx_dict[bottom_node_coordinate]
                if (source_node_idx, target_node_idx) not in self.failed_links:
                    self.links_set.add((source_node_idx, target_node_idx))
                    if source_node_idx in self.available_neighbors:
                        self.available_neighbors[source_node_idx].add(target_node_idx)
                    else:
                        self.available_neighbors[source_node_idx] = {target_node_idx}
                    self.links_dict[(source_node_idx, target_node_idx)] = co2co(co2co_id=(source_node_idx, target_node_idx), co2co_cfg=self.co2co_cfg)
            if left_node_coordinate in self.nodes_coordinate_idx_dict and left_node_coordinate not in self.failed_nodes:
                target_node_idx = self.nodes_coordinate_idx_dict[left_node_coordinate]
                if (source_node_idx, target_node_idx) not in self.failed_links:
                    self.links_set.add((source_node_idx, target_node_idx))
                    if source_node_idx in self.available_neighbors:
                        self.available_neighbors[source_node_idx].add(target_node_idx)
                    else:
                        self.available_neighbors[source_node_idx] = {target_node_idx}
                    self.links_dict[(source_node_idx, target_node_idx)] = co2co(co2co_id=(source_node_idx, target_node_idx), co2co_cfg=self.co2co_cfg)
            if right_node_coordinate in self.nodes_coordinate_idx_dict and right_node_coordinate not in self.failed_nodes:
                target_node_idx = self.nodes_coordinate_idx_dict[right_node_coordinate]
                if (source_node_idx, target_node_idx) not in self.failed_links:
                    self.links_set.add((source_node_idx, target_node_idx))
                    if source_node_idx in self.available_neighbors:
                        self.available_neighbors[source_node_idx].add(target_node_idx)
                    else:
                        self.available_neighbors[source_node_idx] = {target_node_idx}
                    self.links_dict[(source_node_idx, target_node_idx)] = co2co(co2co_id=(source_node_idx, target_node_idx), co2co_cfg=self.co2co_cfg)

            # inter-chiplet
            if (source_node_coordinate[2], source_node_coordinate[3]) not in corner_coordinates and  \
                source_node_coordinate[2] == 0:
                target_node_coordinate = (source_node_coordinate[0] - 1, source_node_coordinate[1], self.l1_height - 1, source_node_coordinate[3])
                if target_node_coordinate in self.nodes_coordinate_idx_dict and target_node_coordinate not in self.failed_nodes:
                    target_node_idx = self.nodes_coordinate_idx_dict[target_node_coordinate]
                    if (source_node_idx, target_node_idx) not in self.failed_links:
                        self.links_set.add((source_node_idx, target_node_idx))
                        if source_node_idx in self.available_neighbors:
                            self.available_neighbors[source_node_idx].add(target_node_idx)
                        else:
                            self.available_neighbors[source_node_idx] = {target_node_idx}
                        self.links_dict[(source_node_idx, target_node_idx)] = ch2ch(ch2ch_id=(source_node_idx, target_node_idx), ch2ch_cfg=self.ch2ch_cfg)
            if (source_node_coordinate[2], source_node_coordinate[3]) not in corner_coordinates and  \
                source_node_coordinate[2] == self.l1_height - 1:
                target_node_coordinate = (source_node_coordinate[0] + 1, source_node_coordinate[1], 0, source_node_coordinate[3])
                if target_node_coordinate in self.nodes_coordinate_idx_dict and target_node_coordinate not in self.failed_nodes:
                    target_node_idx = self.nodes_coordinate_idx_dict[target_node_coordinate]
                    if (source_node_idx, target_node_idx) not in self.failed_links:
                        self.links_set.add((source_node_idx, target_node_idx))
                        if source_node_idx in self.available_neighbors:
                            self.available_neighbors[source_node_idx].add(target_node_idx)
                        else:
                            self.available_neighbors[source_node_idx] = {target_node_idx}
                        self.links_dict[(source_node_idx, target_node_idx)] = ch2ch(ch2ch_id=(source_node_idx, target_node_idx), ch2ch_cfg=self.ch2ch_cfg)
            if (source_node_coordinate[2], source_node_coordinate[3]) not in corner_coordinates and  \
                source_node_coordinate[3] == 0:
                target_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1] - 1, source_node_coordinate[2], self.l1_width - 1)
                if target_node_coordinate in self.nodes_coordinate_idx_dict and target_node_coordinate not in self.failed_nodes:
                    target_node_idx = self.nodes_coordinate_idx_dict[target_node_coordinate]
                    if (source_node_idx, target_node_idx) not in self.failed_links:
                        self.links_set.add((source_node_idx, target_node_idx))
                        if source_node_idx in self.available_neighbors:
                            self.available_neighbors[source_node_idx].add(target_node_idx)
                        else:
                            self.available_neighbors[source_node_idx] = {target_node_idx}
                        self.links_dict[(source_node_idx, target_node_idx)] = ch2ch(ch2ch_id=(source_node_idx, target_node_idx), ch2ch_cfg=self.ch2ch_cfg)
            if (source_node_coordinate[2], source_node_coordinate[3]) not in corner_coordinates and  \
                source_node_coordinate[3] == self.l1_width - 1:
                target_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1] + 1, source_node_coordinate[2], 0)
                if target_node_coordinate in self.nodes_coordinate_idx_dict and target_node_coordinate not in self.failed_nodes:
                    target_node_idx = self.nodes_coordinate_idx_dict[target_node_coordinate]
                    if (source_node_idx, target_node_idx) not in self.failed_links:
                        self.links_set.add((source_node_idx, target_node_idx))
                        if source_node_idx in self.available_neighbors:
                            self.available_neighbors[source_node_idx].add(target_node_idx)
                        else:
                            self.available_neighbors[source_node_idx] = {target_node_idx}
                        self.links_dict[(source_node_idx, target_node_idx)] = ch2ch(ch2ch_id=(source_node_idx, target_node_idx), ch2ch_cfg=self.ch2ch_cfg)
        self.links_list = list(self.links_set)


    def init_mem(self):

        self.l1_width_offset = self.l1_width + 2
        self.l1_height_offset = self.l1_height + 2
        self.ddr_dict = {}

        corner_ddr_coordinates = [
            [-1, 0], [0, -1],
            [-1, self.l1_width - 1], [0, self.l1_width],
            [self.l1_height, 0], [self.l1_height - 1, -1],
            [self.l1_height, self.l1_width - 1], [self.l1_height - 1, self.l1_width] 
        ]
        corner_node_coordinates = [
            [0, 0], [0, 0],
            [0, self.l1_width - 1], [0, self.l1_width - 1],
            [self.l1_height - 1, 0], [self.l1_height - 1, 0],
            [self.l1_height - 1, self.l1_width - 1], [self.l1_height - 1, self.l1_width - 1]
        ]

        for n in range(self.l2_height):
            for m in range(self.l2_width):
                for corner_idx, ddr_coordinate in enumerate(corner_ddr_coordinates):
                    # ddr_idx = (n * self.l2_width + m , ddr_coordinate[0] * self.l1_width + ddr_coordinate[1])
                    ddr_idx = (n * self.l2_width + m , int(self.l1_height * self.l1_width + corner_idx))
                    if ddr_idx in self.failed_nodes:
                        continue
                    else:
                        self.nodes_set.add(ddr_idx)
                        self.nodes_coordinate_dict[ddr_idx] = (n, m, ddr_coordinate[0], ddr_coordinate[1])
                        self.nodes_coordinate_idx_dict[(n, m, ddr_coordinate[0], ddr_coordinate[1])] = ddr_idx
                        for ddr_id in range(self.tpg_cfg["ddr_number"]):
                            ddr_location = ddr_idx + (ddr_id,)
                            ddr_device = device(device_type="ddr", device_id=ddr_location)
                            if ddr_id == 0:
                                self.node_device_list_dict[ddr_idx] = [ddr_location]
                            else:
                                self.node_device_list_dict[ddr_idx].append(ddr_location)
                            self.ddr_dict[ddr_location] = ddr_device

                    corener_node_coordinate = (n, m, corner_node_coordinates[corner_idx][0], corner_node_coordinates[corner_idx][1])
                    if corener_node_coordinate in self.nodes_coordinate_idx_dict:
                        corner_node_idx = self.nodes_coordinate_idx_dict[tuple(corener_node_coordinate)]
                    else:
                        raise ValueError(f"Cannot find corner node coordinate {corener_node_coordinate} in nodes_coordinate_idx_dict.")
                    if (ddr_idx, corner_node_idx) in self.failed_links:
                        raise ValueError(f"Failed link between ddr {ddr_idx} and corner node {corner_node_idx}.")
                    else:
                        self.links_set.add((ddr_idx, corner_node_idx))
                        if ddr_idx in self.available_neighbors:
                            self.available_neighbors[ddr_idx].add(corner_node_idx)
                        else:
                            self.available_neighbors[ddr_idx] = {corner_node_idx}
                        self.links_dict[(ddr_idx, corner_node_idx)] = ddr2co(ddr2co_id=(ddr_idx, corner_node_idx), ddr2co_cfg=self.ddr2co_cfg)
                    if (corner_node_idx, ddr_idx) in self.failed_links:
                        raise ValueError(f"Failed link between corner node {corner_node_idx} and ddr {ddr_idx}.")
                    else:
                        self.links_set.add((corner_node_idx, ddr_idx))
                        self.available_neighbors[corner_node_idx].add(ddr_idx)
                        self.links_dict[(corner_node_idx, ddr_idx)] = co2ddr(co2ddr_id=(corner_node_idx, ddr_idx), co2ddr_cfg=self.co2ddr_cfg)

        self.memory_dict["ddr"] = self.ddr_dict
        self.nodes_number = len(self.nodes_set)
        self.nodes_list = list(self.nodes_set)
        self.links_list = list(self.links_set)

        self.ddr_set = set()
        for ddr_idx in self.ddr_dict:
            self.ddr_set.add(ddr_idx[:-1])


    def update_topology(self):
        self.init_nodes()
        self.init_links()
        if self.mem_enable:
            self.init_mem()        
        self.init_duplicated_links()
        self.dijkstra_offload()
        self.node_to_node_distance()
        self.xy_initiliazation()
        self.xy_manhattan_distance()


    def xy_initiliazation(self):
        self.nodes_xycoor_set = set()
        # (chiplet_idx, core_idx) <--> (y, x)
        self.nodes_xycoor_idx_dict = {}
        self.nodes_idx_xycoor_dict = {}
        self.nodes_xycoor_xyidx_dict = {}
        self.nodes_xycoor_evenodd_dict = {}

        for node_idx in self.nodes_set:
            node_coordinate = self.nodes_coordinate_dict[node_idx]
            node_y = int(node_coordinate[0] * self.l1_height_offset + node_coordinate[2])
            node_x = int(node_coordinate[1] * self.l1_width_offset + node_coordinate[3])
            if self.mem_enable:
                node_y += 1
                node_x += 1
            node_xycoor_coordinate = (node_y, node_x)
            if self.mem_enable:
                node_xyidx = int(node_y * (self.l1_width_offset * self.l2_width) + node_x)
            else:
                node_xyidx = int(node_y * self.l1_width * self.l2_width + node_x)
            self.nodes_xycoor_set.add(node_xycoor_coordinate)
            self.nodes_xycoor_idx_dict[node_xycoor_coordinate] = node_idx
            self.nodes_idx_xycoor_dict[node_idx] = node_xycoor_coordinate
            self.nodes_xycoor_xyidx_dict[node_xycoor_coordinate] = node_xyidx
            # 0: even, 1: odd
            self.nodes_xycoor_evenodd_dict[node_xycoor_coordinate] = node_xyidx % 2

        self.links_xycoor_set = set()
        self.links_xycoor_idx_dict = {}
        self.links_idx_xycoor_dict = {}

        for link_idx in self.links_set:
            source_node_idx = link_idx[0]
            target_node_idx = link_idx[1]
            if len(link_idx) == 3:
                duplicated_idx = link_idx[2]
                link_xycoor_idx = (self.nodes_idx_xycoor_dict[source_node_idx], self.nodes_idx_xycoor_dict[target_node_idx], duplicated_idx)
            else:
                duplicated_idx = None
                link_xycoor_idx = (self.nodes_idx_xycoor_dict[source_node_idx], self.nodes_idx_xycoor_dict[target_node_idx])
            self.links_xycoor_set.add(link_xycoor_idx)
            self.links_xycoor_idx_dict[link_xycoor_idx] = link_idx
            self.links_idx_xycoor_dict[link_idx] = link_xycoor_idx

        self.available_xycoor_neighbors = {}
        for node_idx in self.available_neighbors:
            node_xycoor_idx = self.nodes_idx_xycoor_dict[node_idx]
            self.available_xycoor_neighbors[node_xycoor_idx] = set()
            for neighbor_idx in self.available_neighbors[node_idx]:
                self.available_xycoor_neighbors[node_xycoor_idx].add(self.nodes_idx_xycoor_dict[neighbor_idx])


    def xy_pointopoint_path(
        self,
        source_node_idx: Tuple[int],
        target_node_idx: Tuple[int]
    ) -> List[Tuple[int]]:
        '''
        Fault Tolerant XY-YX Routing Algorithm Supporting Backtracking Strategy for NoC
        '''
        sourc_node_xycoor = self.nodes_idx_xycoor_dict[source_node_idx]
        target_node_xycoor = self.nodes_idx_xycoor_dict[target_node_idx]
        target_y, target_x = target_node_xycoor

        xy_path = [source_node_idx]
        current_node_xycoor = sourc_node_xycoor
        # 0: xy  1: yx
        routing_mode = 0
        x_offset_mode = 1
        x_offset_cross = 3
        y_offset_mode = 1
        y_offset_cross = 3
        iter_cnt = 0
        while current_node_xycoor != target_node_xycoor:

            iter_cnt += 1
            if iter_cnt > 2*(self.nodes_number):
                return None

            current_y, current_x = current_node_xycoor
            dx = target_x - current_x
            dy = target_y - current_y
            current_candidate_links_set = set()
            top_candidate = (current_y - 1, current_x)
            if (current_node_xycoor, top_candidate) in self.links_xycoor_set:
                current_candidate_links_set.add(top_candidate)
            bottom_candidate = (current_y + 1, current_x)
            if (current_node_xycoor, bottom_candidate) in self.links_xycoor_set:
                current_candidate_links_set.add(bottom_candidate)
            left_candidate = (current_y, current_x - 1)
            if (current_node_xycoor, left_candidate) in self.links_xycoor_set:
                current_candidate_links_set.add(left_candidate)
            right_candidate = (current_y, current_x + 1)
            if (current_node_xycoor, right_candidate) in self.links_xycoor_set:
                current_candidate_links_set.add(right_candidate)
            top_candidate = (current_y - 3, current_x)
            if (current_node_xycoor, top_candidate) in self.links_xycoor_set:
                current_candidate_links_set.add(top_candidate)
            bottom_candidate = (current_y + 3, current_x)
            if (current_node_xycoor, bottom_candidate) in self.links_xycoor_set:
                current_candidate_links_set.add(bottom_candidate)
            left_candidate = (current_y, current_x - 3)
            if (current_node_xycoor, left_candidate) in self.links_xycoor_set:
                current_candidate_links_set.add(left_candidate)
            right_candidate = (current_y, current_x + 3)
            if (current_node_xycoor, right_candidate) in self.links_xycoor_set:
                current_candidate_links_set.add(right_candidate)                
            if not current_candidate_links_set:
                raise ValueError("No available link from current node to target node.")

            candidates_list = []
            if dx > 0:
                x_candidate = 1
                x_cross = 3
            elif dx < 0:
                x_candidate = -1
                x_cross = -3
            else:
                x_candidate = 0
            if dy > 0:
                y_candidate = 1
                y_cross = 3
            elif dy < 0:
                y_candidate = -1
                y_cross = -3
            else:
                y_candidate = 0

            if (routing_mode == 0 and self.nodes_xycoor_evenodd_dict[current_node_xycoor] == 0) or \
                (routing_mode == 1 and self.nodes_xycoor_evenodd_dict[current_node_xycoor] == 1):
                if x_candidate == 0 and y_candidate == 0:
                    raise ValueError("Source and target nodes are the same.")
                elif x_candidate == 0 and y_candidate != 0:
                    candidate = (current_y + y_candidate, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        continue
                    candidate = (current_y + y_cross, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        continue                    
                    candidate = (current_y, current_x + x_offset_mode)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            x_offset_mode = 0 - x_offset_mode
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue      
                    candidate = (current_y, current_x + x_offset_cross)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            x_offset_cross = 0 - x_offset_cross
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                        
                    candidate = (current_y, current_x - x_offset_mode)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            x_offset_mode = 0 - x_offset_mode                        
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue      
                    candidate = (current_y, current_x - x_offset_cross)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            x_offset_cross = 0 - x_offset_cross                        
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                                            
                    candidate = (current_y - y_candidate, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue      
                    candidate = (current_y - y_cross, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                                            
                elif x_candidate != 0 and y_candidate == 0:
                    candidate = (current_y, current_x + x_candidate)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        continue        
                    candidate = (current_y, current_x + x_cross)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        continue                                            
                    candidate = (current_y + y_offset_mode, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            y_offset_mode = 0 - y_offset_mode
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue      
                    candidate = (current_y + y_offset_cross, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            y_offset_cross = 0 - y_offset_cross
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                                            
                    candidate = (current_y - y_offset_mode, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            y_offset_mode = 0 - y_offset_mode
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue      
                    candidate = (current_y - y_offset_cross, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            y_offset_cross = 0 - y_offset_cross
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                                            
                    candidate = (current_y, current_x - x_candidate)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue      
                    candidate = (current_y, current_x - x_cross)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                                            
                else:
                    candidate = (current_y, current_x + x_candidate)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            pass
                        else:
                            candidates_list.append(candidate)
                            current_node_xycoor = candidates_list[0]
                            xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                            continue       
                    candidate = (current_y, current_x + x_cross)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            pass
                        else:
                            candidates_list.append(candidate)
                            current_node_xycoor = candidates_list[0]
                            xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                            continue                                                
                    candidate = (current_y + y_candidate, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            routing_mode = 1 - routing_mode
                            pass
                        else:
                            candidates_list.append(candidate)
                            current_node_xycoor = candidates_list[0]
                            xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                            routing_mode = 1 - routing_mode
                            continue
                    candidate = (current_y + y_cross, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            routing_mode = 1 - routing_mode
                            pass
                        else:                           
                            candidates_list.append(candidate)
                            current_node_xycoor = candidates_list[0]
                            xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                            routing_mode = 1 - routing_mode
                            continue                                            
                    candidate = (current_y - y_candidate, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            routing_mode = 1 - routing_mode
                            pass
                        else:
                            candidates_list.append(candidate)
                            current_node_xycoor = candidates_list[0]
                            xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                            routing_mode = 1 - routing_mode
                            continue
                    candidate = (current_y - y_cross, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            routing_mode = 1 - routing_mode
                            pass
                        else:
                            candidates_list.append(candidate)
                            current_node_xycoor = candidates_list[0]
                            xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                            routing_mode = 1 - routing_mode
                            continue                                            
                    candidate = (current_y, current_x - x_candidate)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue      
                    candidate = (current_y, current_x - x_cross)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                                            
            else:
                if x_candidate == 0 and y_candidate == 0:
                    raise ValueError("Source and target nodes are the same.")
                elif x_candidate == 0 and y_candidate != 0:
                    candidate = (current_y + y_candidate, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        continue          
                    candidate = (current_y + y_cross, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        continue                                            
                    candidate = (current_y, current_x + x_offset_mode)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            x_offset_mode = 0 - x_offset_mode
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue           
                    candidate = (current_y, current_x + x_offset_cross)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            x_offset_cross = 0 - x_offset_cross
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                        
                    candidate = (current_y, current_x - x_offset_mode)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            x_offset_mode = 0 - x_offset_mode
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue     
                    candidate = (current_y, current_x - x_offset_cross)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            x_offset_cross = 0 - x_offset_cross
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                                            
                    candidate = (current_y - y_candidate, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue  
                    candidate = (current_y - y_cross, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                                            
                elif x_candidate != 0 and y_candidate == 0:
                    candidate = (current_y, current_x + x_candidate)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        continue              
                    candidate = (current_y, current_x + x_cross)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        continue                                            
                    candidate = (current_y - y_offset_mode, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            y_offset_mode = 0 - y_offset_mode
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue     
                    candidate = (current_y - y_offset_cross, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            y_offset_cross = 0 - y_offset_cross
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                                            
                    candidate = (current_y + y_offset_mode, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            y_offset_mode = 0 - y_offset_mode
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue       
                    candidate = (current_y + y_offset_cross, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            y_offset_cross = 0 - y_offset_cross
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                                            
                    candidate = (current_y, current_x - x_candidate)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                
                    candidate = (current_y, current_x - x_cross)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                                            
                else:
                    candidate = (current_y + y_candidate, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            pass
                        else:
                            candidates_list.append(candidate)
                            current_node_xycoor = candidates_list[0]
                            xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                            continue            
                    candidate = (current_y + y_cross, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            pass
                        else:
                            candidates_list.append(candidate)
                            current_node_xycoor = candidates_list[0]
                            xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                            continue                                                
                    candidate = (current_y, current_x + x_candidate)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            routing_mode = 1 - routing_mode
                            pass
                        else:
                            candidates_list.append(candidate)
                            current_node_xycoor = candidates_list[0]
                            xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                            routing_mode = 1 - routing_mode
                            continue    
                    candidate = (current_y, current_x + x_cross)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            routing_mode = 1 - routing_mode
                            pass
                        else:                            
                            candidates_list.append(candidate)
                            current_node_xycoor = candidates_list[0]
                            xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                            routing_mode = 1 - routing_mode
                            continue                                            
                    candidate = (current_y, current_x - x_candidate)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            routing_mode = 1 - routing_mode
                            pass
                        else:                        
                            candidates_list.append(candidate)
                            current_node_xycoor = candidates_list[0]
                            xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                            routing_mode = 1 - routing_mode
                            continue       
                    candidate = (current_y, current_x - x_cross)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        if self.nodes_xycoor_idx_dict[candidate] in xy_path:
                            routing_mode = 1 - routing_mode
                            pass
                        else:                        
                            candidates_list.append(candidate)
                            current_node_xycoor = candidates_list[0]
                            xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                            routing_mode = 1 - routing_mode
                            continue                                            
                    candidate = (current_y - y_candidate, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue                        
                    candidate = (current_y - y_cross, current_x)
                    if candidate in self.nodes_xycoor_set and ((len(self.available_xycoor_neighbors[candidate]) > 1 and candidate != target_node_xycoor) or candidate == target_node_xycoor) and \
                        candidate in current_candidate_links_set:
                        candidates_list.append(candidate)
                        current_node_xycoor = candidates_list[0]
                        xy_path.append(self.nodes_xycoor_idx_dict[current_node_xycoor])
                        routing_mode = 1 - routing_mode
                        continue      

        def clean_backtrack(path):

            simp = []
            for p in path:
                simp.append(p)
                max_l = len(simp) // 2
                for l in range(1, max_l+1):
                    if simp[-2*l:-l] == simp[-l:]:
                        del simp[-l:]
                        break
            return simp


        return clean_backtrack(xy_path)


    def xy_multicast_path(
        self,
        source_node_idx: Tuple[int],
        target_nodes_set: Set[Tuple[int]]
    ) -> Dict[Tuple[int], List[Tuple[int]]]:

        xy_path = {}
        xy_path[source_node_idx] = []
        passed_ndoes_set = set()
        for target_node_idx in target_nodes_set:
            if self.mem_enable and (source_node_idx in self.ddr_set or target_node_idx in self.ddr_set):
                continue            
            if target_node_idx in passed_ndoes_set:
                continue
            source_target_path = self.xy_pointopoint_path(source_node_idx, target_node_idx)
            if source_target_path is None:
                return None
            source_target_path_length = len(source_target_path)
            for source_pair_idx in range(source_target_path_length - 1):
                source_pair = source_target_path[source_pair_idx]
                passed_ndoes_set.add(source_pair)
                target_pair = source_target_path[source_pair_idx + 1]
                if target_pair in passed_ndoes_set:
                    continue
                else:
                    passed_ndoes_set.add(target_pair)
                if source_pair in xy_path:
                    if target_pair not in xy_path[source_pair]:
                        xy_path[source_pair].append(target_pair)
                else:
                    xy_path[source_pair] = [target_pair]
        for target_node_idx in target_nodes_set:
            if self.mem_enable and (source_node_idx in self.ddr_set or target_node_idx in self.ddr_set):
                pass
            else:
                continue
            if target_node_idx in passed_ndoes_set:
                continue

            if source_node_idx in self.ddr_set:
                true_source_idx = list(self.available_neighbors[source_node_idx])[0]
            else:
                true_source_idx = source_node_idx
            if target_node_idx in self.ddr_set:
                true_target_idx = list(self.available_neighbors[target_node_idx])[0]
            else:
                true_target_idx = target_node_idx
            source_target_path = self.xy_pointopoint_path(true_source_idx, true_target_idx)
            if target_node_idx in source_target_path:
                pass
            else:
                source_target_path.append(target_node_idx)
            if source_node_idx in source_target_path:
                pass
            else:
                source_target_path.insert(0, source_node_idx)
            if source_target_path is None:
                return None
            source_target_path_length = len(source_target_path)
            for source_pair_idx in range(source_target_path_length - 1):
                source_pair = source_target_path[source_pair_idx]
                passed_ndoes_set.add(source_pair)
                target_pair = source_target_path[source_pair_idx + 1]
                if target_pair in passed_ndoes_set:
                    continue
                else:
                    passed_ndoes_set.add(target_pair)
                if source_pair in xy_path:
                    if target_pair not in xy_path[source_pair]:
                        xy_path[source_pair].append(target_pair)
                else:
                    xy_path[source_pair] = [target_pair]        


        for root in xy_path:
            new_leaves = []
            for leaf in xy_path[root]:
                new_leaves.append(TupleIdx(leaf, idx=None))
            xy_path[root] = new_leaves

        return xy_path


    def xy_manhattan_distance(self, profile_size: int = 1048576):

        self.node_to_node_manhattan_hops_dict = {}
        self.node_to_node_manhattan_distance_dict = {}
        self.node_to_node_manhattan_distance_function_dict = {}
        self.xy_paths = {}
        for source_node_idx in self.nodes_set:
            self.node_to_node_manhattan_hops_dict[source_node_idx] = {}
            self.node_to_node_manhattan_distance_dict[source_node_idx] = {}
            self.node_to_node_manhattan_distance_function_dict[source_node_idx] = {}
            self.xy_paths[source_node_idx] = {}
            for target_node_idx in self.nodes_set:
                if source_node_idx == target_node_idx:
                    continue
                else:
                    xy_path = self.xy_multicast_path(
                        source_node_idx=source_node_idx,
                        target_nodes_set={target_node_idx}
                    )
                    if xy_path is None:
                        self.xy_paths[source_node_idx][target_node_idx] = xy_path
                        self.node_to_node_manhattan_hops_dict[source_node_idx][target_node_idx] = None
                        self.node_to_node_manhattan_distance_dict[source_node_idx][target_node_idx] = None
                        self.node_to_node_manhattan_distance_function_dict[source_node_idx][target_node_idx] = None
                        continue
                    xy_links = []
                    for root in xy_path:
                        for leaf in xy_path[root]:
                            xy_links.append((root, leaf))
                    self.xy_paths[source_node_idx][target_node_idx] = xy_path
                    source_coord = self.nodes_idx_xycoor_dict[source_node_idx]
                    target_coord = self.nodes_idx_xycoor_dict[target_node_idx]
                    self.node_to_node_manhattan_hops_dict[source_node_idx][target_node_idx] = int(abs(source_coord[0] - target_coord[0]) + abs(source_coord[1] - target_coord[1]))
                    base_latency = sum(self.links_dict[lid].latency for lid in xy_links)
                    slope = sum(self.links_dict[lid].bandwidth / self.links_dict[lid].timeunit
                            for lid in xy_links)
                    self.node_to_node_manhattan_distance_function_dict[source_node_idx][target_node_idx] = (
                        lambda ms, base=base_latency, sl=slope: 
                            base + math.ceil(ms / sl)                        
                    )
                    self.node_to_node_manhattan_distance_dict[source_node_idx][target_node_idx] = self.node_to_node_manhattan_distance_function_dict[source_node_idx][target_node_idx](profile_size)


if __name__ == "__main__":

    random.seed(123)
    hardware_cfg = cfg_to_dict(os.path.join(file_path, "../../../platform/cfgs/wamis_hd.cfg"))
    network = wamis_hdc(hardware_cfg)
    paths = network.xy_multicast_path((0, 36), {(0, 35), (0, 20)})
    print(paths)
    _, _, paths, _ = network.record_dijkstra_multicast_path(comm_pairs=[[(1, 0, 0, 1, 25, 2), (0, 36), {(0, 35), (0, 20)}, 27248130]])
    print(paths[(1, 0, 0, 1, 25, 2)])
