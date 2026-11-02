
"""Two-level mesh (TLM) topology: hierarchical chiplet-of-cores interconnect
with inter-chiplet and intra-chiplet routing for wafer-scale systems."""

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


class tlm2d(planar_2d):
    """Two-Level Mesh: intra-chiplet co2co mesh + single inter-chiplet ch2ch link
    per edge (top/bottom/left/right center core of each chiplet)."""

    def __init__(self, platform_cfg: Dict[str, int]):
        super().__init__(platform_cfg=platform_cfg)


    def init_links(self):
        # Inter-chiplet bridge coordinates: center core on each edge
        co_width_half = self.l1_width // 2
        co_height_half = self.l1_height // 2
        co_top_edge_coordinate = (0, co_width_half)
        co_down_edge_coordinate = (self.l1_height - 1, co_width_half)
        co_left_edge_coordinate = (co_height_half, 0)
        co_right_edge_coordinate = (co_height_half, self.l1_width - 1)

        self.links_set = set()
        self.links_list = []
        self.available_neighbors = {}
        self.links_dict = {}
        self.chiplets_links_set = set()
        self.chiplet_links_dict = {}

        for source_node_idx in self.nodes_set:
            source_node_coordinate = self.nodes_coordinate_dict[source_node_idx]
            top_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1], source_node_coordinate[2] - 1, source_node_coordinate[3])
            bottom_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1], source_node_coordinate[2] + 1, source_node_coordinate[3])
            left_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1], source_node_coordinate[2], source_node_coordinate[3] - 1)
            right_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1], source_node_coordinate[2], source_node_coordinate[3] + 1)
            if top_node_coordinate in self.nodes_coordinate_idx_dict and top_node_coordinate not in self.failed_links:
                target_node_idx = self.nodes_coordinate_idx_dict[top_node_coordinate]
                self.links_set.add((source_node_idx, target_node_idx))
                if source_node_idx in self.available_neighbors:
                    self.available_neighbors[source_node_idx].add(target_node_idx)
                # --- Case 4: Moving Left (decreasing x) and Up (decreasing y) ---
                else:
                    self.available_neighbors[source_node_idx] = {target_node_idx}
                self.links_dict[(source_node_idx, target_node_idx)] = co2co(co2co_id=(source_node_idx, target_node_idx), co2co_cfg=self.co2co_cfg)
            if bottom_node_coordinate in self.nodes_coordinate_idx_dict and bottom_node_coordinate not in self.failed_links:
                target_node_idx = self.nodes_coordinate_idx_dict[bottom_node_coordinate]
                self.links_set.add((source_node_idx, target_node_idx))
                if source_node_idx in self.available_neighbors:
                    self.available_neighbors[source_node_idx].add(target_node_idx)
                else:
                    self.available_neighbors[source_node_idx] = {target_node_idx}
                self.links_dict[(source_node_idx, target_node_idx)] = co2co(co2co_id=(source_node_idx, target_node_idx), co2co_cfg=self.co2co_cfg)
            if left_node_coordinate in self.nodes_coordinate_idx_dict and left_node_coordinate not in self.failed_links:
                target_node_idx = self.nodes_coordinate_idx_dict[left_node_coordinate]
                self.links_set.add((source_node_idx, target_node_idx))
                if source_node_idx in self.available_neighbors:
                    self.available_neighbors[source_node_idx].add(target_node_idx)
                else:
                    self.available_neighbors[source_node_idx] = {target_node_idx}
                self.links_dict[(source_node_idx, target_node_idx)] = co2co(co2co_id=(source_node_idx, target_node_idx), co2co_cfg=self.co2co_cfg)
            if right_node_coordinate in self.nodes_coordinate_idx_dict and right_node_coordinate not in self.failed_links:
                target_node_idx = self.nodes_coordinate_idx_dict[right_node_coordinate]
                self.links_set.add((source_node_idx, target_node_idx))
                if source_node_idx in self.available_neighbors:
                    self.available_neighbors[source_node_idx].add(target_node_idx)
                else:
                    self.available_neighbors[source_node_idx] = {target_node_idx}
                self.links_dict[(source_node_idx, target_node_idx)] = co2co(co2co_id=(source_node_idx, target_node_idx), co2co_cfg=self.co2co_cfg)

            if (source_node_coordinate[2], source_node_coordinate[3]) == co_top_edge_coordinate:
                target_node_coordinate = (source_node_coordinate[0] - 1, source_node_coordinate[1], co_down_edge_coordinate[0], co_down_edge_coordinate[1])
                if target_node_coordinate in self.nodes_coordinate_idx_dict and target_node_coordinate not in self.failed_links:
                    target_node_idx = self.nodes_coordinate_idx_dict[target_node_coordinate]
                    self.links_set.add((source_node_idx, target_node_idx))
                    self.chiplets_links_set.add((source_node_idx[0], target_node_idx[0]))
                    self.chiplet_links_dict[source_node_idx[0], target_node_idx[0]] = (source_node_idx, target_node_idx)
                    if source_node_idx in self.available_neighbors:
                        self.available_neighbors[source_node_idx].add(target_node_idx)
                    else:
                        self.available_neighbors[source_node_idx] = {target_node_idx}
                    self.links_dict[(source_node_idx, target_node_idx)] = ch2ch(ch2ch_id=(source_node_idx, target_node_idx), ch2ch_cfg=self.ch2ch_cfg)
            if (source_node_coordinate[2], source_node_coordinate[3]) == co_down_edge_coordinate:
                target_node_coordinate = (source_node_coordinate[0] + 1, source_node_coordinate[1], co_top_edge_coordinate[0], co_top_edge_coordinate[1])
                if target_node_coordinate in self.nodes_coordinate_idx_dict and target_node_coordinate not in self.failed_links:
                    target_node_idx = self.nodes_coordinate_idx_dict[target_node_coordinate]
                    self.links_set.add((source_node_idx, target_node_idx))
                    self.chiplets_links_set.add((source_node_idx[0], target_node_idx[0]))
                    self.chiplet_links_dict[source_node_idx[0], target_node_idx[0]] = (source_node_idx, target_node_idx)
                    if source_node_idx in self.available_neighbors:
                        self.available_neighbors[source_node_idx].add(target_node_idx)
                    else:
                        self.available_neighbors[source_node_idx] = {target_node_idx}
                    self.links_dict[(source_node_idx, target_node_idx)] = ch2ch(ch2ch_id=(source_node_idx, target_node_idx), ch2ch_cfg=self.ch2ch_cfg)
            if (source_node_coordinate[2], source_node_coordinate[3]) == co_left_edge_coordinate:
                target_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1] - 1, co_right_edge_coordinate[0], co_right_edge_coordinate[1])
                if target_node_coordinate in self.nodes_coordinate_idx_dict and target_node_coordinate not in self.failed_links:
                    target_node_idx = self.nodes_coordinate_idx_dict[target_node_coordinate]
                    self.links_set.add((source_node_idx, target_node_idx))
                    self.chiplets_links_set.add((source_node_idx[0], target_node_idx[0]))
                    self.chiplet_links_dict[source_node_idx[0], target_node_idx[0]] = (source_node_idx, target_node_idx)
                    if source_node_idx in self.available_neighbors:
                        self.available_neighbors[source_node_idx].add(target_node_idx)
                    else:
                        self.available_neighbors[source_node_idx] = {target_node_idx}
                    self.links_dict[(source_node_idx, target_node_idx)] = ch2ch(ch2ch_id=(source_node_idx, target_node_idx), ch2ch_cfg=self.ch2ch_cfg)
            if (source_node_coordinate[2], source_node_coordinate[3]) == co_right_edge_coordinate:
                target_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1] + 1, co_left_edge_coordinate[0], co_left_edge_coordinate[1])
                if target_node_coordinate in self.nodes_coordinate_idx_dict and target_node_coordinate not in self.failed_links:
                    target_node_idx = self.nodes_coordinate_idx_dict[target_node_coordinate]
                    self.links_set.add((source_node_idx, target_node_idx))
                    self.chiplets_links_set.add((source_node_idx[0], target_node_idx[0]))
                    self.chiplet_links_dict[source_node_idx[0], target_node_idx[0]] = (source_node_idx, target_node_idx)
                    if source_node_idx in self.available_neighbors:
                        self.available_neighbors[source_node_idx].add(target_node_idx)
                    else:
                        self.available_neighbors[source_node_idx] = {target_node_idx}
                    self.links_dict[(source_node_idx, target_node_idx)] = ch2ch(ch2ch_id=(source_node_idx, target_node_idx), ch2ch_cfg=self.ch2ch_cfg)
        self.links_list = list(self.links_set)


    def update_topology(self):
        self.init_nodes()
        self.init_links()
        self.init_duplicated_links()
        self.dijkstra_offload()
        self.node_to_node_distance()
        self.ch_to_ch_distance()
        self.xy_manhattan_distance()


    def xy_intrachiplet_path(
        self,
        source_node_idx: Tuple[int],
        target_node_idx: Tuple[int]
    ) -> List[Tuple[int]]:

        source_ch_y, source_ch_x, source_co_y, source_co_x = self.idx_to_coordinate(source_node_idx)
        target_ch_y, target_ch_x, target_co_y, target_co_x = self.idx_to_coordinate(target_node_idx)
        if source_ch_x != target_ch_x or source_ch_y != target_ch_y:
            raise ValueError("Source and target nodes are not in the same chiplet.")

        path = []
        path.append(source_node_idx)
        current_x, current_y = source_co_x, source_co_y
        target_x, target_y = target_co_x, target_co_y
        # Flags for detecting failure in the horizontal or vertical segments.
        x_failed_flag = False
        y_failed_flag = False
        # --- Case 1: Moving Right (increasing x) and Down (increasing y) ---
        if current_x <= target_x and current_y <= target_y:
            # Try horizontal movement first (rightward)
            while current_x < target_x:
                path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x+1)))
                current_x += 1
            for failed_link in self.failed_links:
                if failed_link[0] in path and failed_link[1] in path:
                    x_failed_flag = True
                break
            if x_failed_flag:
                path = [source_node_idx]
                current_x, current_y = source_co_x, source_co_y
                if current_y == target_y and (source_ch_y, source_ch_x, current_y+1, current_x) not in self.nodes_coordinate_idx_dict:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                    current_y -= 1
                else:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y+1, current_x)))
                    current_y += 1                   
                # Now move horizontally toward target
                while current_x < target_x:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x+1)))
                    current_x += 1
                # Finally, adjust vertically to the target row
                while current_y < target_y:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y+1, current_x)))
                    current_y += 1
                # Now move vertically (upward)
                while current_y > target_y:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                    current_y -= 1                    
            else:
                x_path = [along_node for along_node in path]
                # Continue with vertical movement (downward)
                while current_y < target_y:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y+1, current_x)))
                    current_y += 1
                for failed_link in self.failed_links:
                    if failed_link[0] in path and failed_link[1] in path:
                        y_failed_flag = True
                    break
                if y_failed_flag:
                    if source_co_x == target_co_x:
                        path = [source_node_idx]
                        current_x, current_y = source_co_x, source_co_y
                        if (source_ch_y, source_ch_x, current_y, current_x+1) not in self.nodes_coordinate_idx_dict:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x-1)))
                            current_x -= 1
                        else:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x+1)))
                            current_x += 1
                        # Then proceed with vertical movement
                        while current_y < target_y:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y+1, current_x)))
                            current_y += 1
                        # Finally, adjust horizontally if needed
                        while current_x < target_x:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x+1)))
                            current_x += 1
                        # Horizontal movement first (leftward)
                        while current_x > target_x:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x-1)))
                            current_x -= 1
                    else:
                        # Otherwise, revert to the horizontal segment (minus its last node)
                        path = x_path[:-1]
                        current_y, current_x = self.idx_to_coordinate(x_path[-1])[2], self.idx_to_coordinate(x_path[-1])[3] - 1
                        # Continue vertical movement
                        while current_y < target_y:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y+1, current_x)))
                            current_y += 1
                        # And finish the horizontal movement if needed
                        while current_x < target_x:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x+1)))
                            current_x += 1                        
                else:
                    pass
        # --- Case 2: Moving Right (increasing x) and Up (decreasing y) ---
        elif current_x <= target_x and current_y > target_y:
            # Horizontal movement first (rightward)
            while current_x < target_x:
                path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x+1)))
                current_x += 1
            for failed_link in self.failed_links:
                if failed_link[0] in path and failed_link[1] in path:
                    x_failed_flag = True
                break
            if x_failed_flag:
                path = [source_node_idx]
                current_x, current_y = source_co_x, source_co_y
                path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                current_y -= 1                   
                while current_x < target_x:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x+1)))
                    current_x += 1
                while current_y > target_y:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                    current_y -= 1              
            else:
                x_path = [along_node for along_node in path]
                while current_y > target_y:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                    current_y -= 1
                for failed_link in self.failed_links:
                    if failed_link[0] in path and failed_link[1] in path:
                        y_failed_flag = True
                    break
                if y_failed_flag:
                    if source_co_x == target_co_x:
                        path = [source_node_idx]
                        current_x, current_y = source_co_x, source_co_y
                        if (source_ch_y, source_ch_x, current_y, current_x+1) not in self.nodes_coordinate_idx_dict:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x-1)))
                            current_x -= 1
                        else:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x+1)))
                            current_x += 1
                        while current_y > target_y:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                            current_y -= 1
                        while current_x < target_x:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x+1)))
                            current_x += 1
                        # Horizontal movement first (leftward)
                        while current_x > target_x:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x-1)))
                            current_x -= 1
                    else:
                        # and take a detour by shifting horizontally a little before
                        path = x_path[:-1]
                        current_y, current_x = self.idx_to_coordinate(x_path[-1])[2], self.idx_to_coordinate(x_path[-1])[3] - 1
                        while current_y > target_y:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                            current_y -= 1
                        while current_x < target_x:
                            path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x+1)))
                            current_x += 1
                else:
                    pass                        
        # --- Case 3: Moving Left (decreasing x) and Down (increasing y) ---
        elif current_x > target_x and current_y <= target_y:
            while current_x > target_x:
                path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x-1)))
                current_x -= 1
            for failed_link in self.failed_links:
                if failed_link[0] in path and failed_link[1] in path:
                    x_failed_flag = True
                break
            if x_failed_flag:
                path = [source_node_idx]
                current_x, current_y = source_co_x, source_co_y
                if current_y == target_y and (source_ch_y, source_ch_x, current_y+1, current_x) not in self.nodes_coordinate_idx_dict:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                    current_y -= 1
                else:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y+1, current_x)))
                    current_y += 1                   
                while current_x > target_x:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x-1)))
                    current_x -= 1
                while current_y < target_y:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y+1, current_x)))
                    current_y += 1
                while current_y > target_y:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                    current_y -= 1 
            else:
                x_path = [along_node for along_node in path]
                while current_y < target_y:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y+1, current_x)))
                    current_y += 1
                for failed_link in self.failed_links:
                    if failed_link[0] in path and failed_link[1] in path:
                        y_failed_flag = True
                    break
                if y_failed_flag:
                    # continuing with vertical and the rest of horizontal moves.
                    path = x_path[:-1]
                    current_y, current_x = self.idx_to_coordinate(x_path[-1])[2], self.idx_to_coordinate(x_path[-1])[3] + 1
                    while current_y < target_y:
                        path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y+1, current_x)))
                        current_y += 1
                    while current_x > target_x:
                        path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x-1)))
                        current_x -= 1                        
                else:
                    pass                               
        else:
            while current_x > target_x:
                path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x-1)))
                current_x -= 1
            for failed_link in self.failed_links:
                if failed_link[0] in path and failed_link[1] in path:
                    x_failed_flag = True
                break
            if x_failed_flag:
                path = [source_node_idx]
                current_x, current_y = source_co_x, source_co_y
                path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                current_y -= 1                   
                while current_x > target_x:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x-1)))
                    current_x -= 1
                while current_y > target_y:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                    current_y -= 1
            else:
                x_path = [along_node for along_node in path]
                while current_y > target_y:
                    path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                    current_y -= 1
                for failed_link in self.failed_links:
                    if failed_link[0] in path and failed_link[1] in path:
                        y_failed_flag = True
                    break
                if y_failed_flag:
                    path = x_path[:-1]
                    current_y, current_x = self.idx_to_coordinate(x_path[-1])[2], self.idx_to_coordinate(x_path[-1])[3] + 1
                    while current_y > target_y:
                        path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y-1, current_x)))
                        current_y -= 1
                    while current_x > target_x:
                        path.append(self.coordinate_to_idx((source_ch_y, source_ch_x, current_y, current_x-1)))
                        current_x -= 1                        
                else:
                    pass

        return path


    def xy_interchiplet_path(
        self,
        source_chiplet_idx: int,
        target_chiplet_idx: int
    ) -> List[int]:
        """
        Compute an inter-chiplet path from source_chiplet_idx to target_chiplet_idx 
        using an XY routing strategy (first horizontal then vertical or vice versa)
        with failure tolerance.

        The routing is performed using the chiplet coordinates:
        - The chiplet coordinate dictionary (self.chiplets_coordinate_dict) maps
            a chiplet index to a tuple (chiplet_row, chiplet_col).
        - The reverse mapping is found in self.chiplets_coordinate_idx_dict.
        The function checks for link failures (self.failed_ch2ch_links) along each
        segment. If a failure is detected, an alternate detour is taken.
        """
        # Get chiplet coordinates (row, col) for source and target
        source_ch_y, source_ch_x = self.chiplets_coordinate_dict[source_chiplet_idx]
        target_ch_y, target_ch_x = self.chiplets_coordinate_dict[target_chiplet_idx]

        # Start the path from the source chiplet index
        path = [source_chiplet_idx]
        current_x, current_y = source_ch_x, source_ch_y
        target_x, target_y = target_ch_x, target_ch_y

        x_failed_flag = False
        y_failed_flag = False

        if current_x <= target_x and current_y <= target_y:
            while current_x < target_x:
                next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x + 1)]
                path.append(next_chip)
                current_x += 1

            # Check horizontal segment for failure by scanning consecutive chiplet pairs
            for i in range(len(path) - 1):
                pair = (path[i], path[i + 1])
                if pair in self.failed_ch2ch_links or (pair[1], pair[0]) in self.failed_ch2ch_links:
                    x_failed_flag = True
                    break

            if x_failed_flag:
                # Detour: reset the path and try moving vertically first, then horizontally
                path = [source_chiplet_idx]
                current_x, current_y = source_ch_x, source_ch_y
                # Choose vertical detour: move down (since target is below source)
                next_chip = self.chiplets_coordinate_idx_dict[(current_y + 1, current_x)]
                path.append(next_chip)
                current_y += 1
                while current_x < target_x:
                    next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x + 1)]
                    path.append(next_chip)
                    current_x += 1
                while current_y < target_y:
                    # Detour by moving downward first
                    next_chip = self.chiplets_coordinate_idx_dict[(current_y + 1, current_x)]
                    path.append(next_chip)
                    current_y += 1
            else:
                # No horizontal failure encountered; store the horizontal segment
                x_path = path.copy()
                while current_y < target_y:
                    next_chip = self.chiplets_coordinate_idx_dict[(current_y + 1, current_x)]
                    path.append(next_chip)
                    current_y += 1
                # Check the vertical segment for failures (only check new vertical hops)
                for i in range(len(x_path) - 1, len(path) - 1):
                    pair = (path[i], path[i + 1])
                    if pair in self.failed_ch2ch_links or (pair[1], pair[0]) in self.failed_ch2ch_links:
                        y_failed_flag = True
                        break
                if y_failed_flag:
                    # If vertical movement failed, then depending on whether the horizontal
                    if source_ch_x == target_ch_x:
                        # When the column is already correct, try a small horizontal detour.
                        path = [source_chiplet_idx]
                        current_x, current_y = source_ch_x, source_ch_y
                        # Attempt a detour: move left if possible, else right.
                        detour_dir = -1 if current_x > 0 else 1
                        next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x + detour_dir)]
                        path.append(next_chip)
                        current_x += detour_dir
                        while current_y < target_y:
                            next_chip = self.chiplets_coordinate_idx_dict[(current_y + 1, current_x)]
                            path.append(next_chip)
                            current_y += 1
                        while current_x < target_x:
                            next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x + 1)]
                            path.append(next_chip)
                            current_x += 1
                        while current_x > target_x:
                            next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x - 1)]
                            path.append(next_chip)
                            current_x -= 1
                    else:
                        path = x_path[:-1]
                        last_chip = x_path[-1]
                        current_y, current_x = self.chiplets_coordinate_dict[last_chip]
                        # Detour: move horizontally one unit away from the failed link
                        detour_dir = -1 if current_x > 0 else 1
                        current_x += detour_dir
                        while current_y < target_y:
                            next_chip = self.chiplets_coordinate_idx_dict[(current_y + 1, current_x)]
                            path.append(next_chip)
                            current_y += 1
                        while current_x < target_x:
                            next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x + 1)]
                            path.append(next_chip)
                            current_x += 1

        elif current_x <= target_x and current_y > target_y:
            while current_x < target_x:
                next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x + 1)]
                path.append(next_chip)
                current_x += 1

            # Check horizontal steps for failures
            for i in range(len(path) - 1):
                pair = (path[i], path[i + 1])
                if pair in self.failed_ch2ch_links or (pair[1], pair[0]) in self.failed_ch2ch_links:
                    x_failed_flag = True
                    break

            if x_failed_flag:
                # Take a vertical detour first by moving up and then horizontal.
                path = [source_chiplet_idx]
                current_x, current_y = source_ch_x, source_ch_y
                # Detour: move upward first
                next_chip = self.chiplets_coordinate_idx_dict[(current_y - 1, current_x)]
                path.append(next_chip)
                current_y -= 1
                while current_x < target_x:
                    next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x + 1)]
                    path.append(next_chip)
                    current_x += 1
                while current_y > target_y:
                    next_chip = self.chiplets_coordinate_idx_dict[(current_y - 1, current_x)]
                    path.append(next_chip)
                    current_y -= 1
            else:
                x_path = path.copy()
                while current_y > target_y:
                    next_chip = self.chiplets_coordinate_idx_dict[(current_y - 1, current_x)]
                    path.append(next_chip)
                    current_y -= 1
                # Check vertical segment
                for i in range(len(x_path) - 1, len(path) - 1):
                    pair = (path[i], path[i + 1])
                    if pair in self.failed_ch2ch_links or (pair[1], pair[0]) in self.failed_ch2ch_links:
                        y_failed_flag = True
                        break
                if y_failed_flag:
                    # coordinate is unchanged, take an alternative detour.
                    if source_ch_x == target_ch_x:
                        # Use a horizontal detour when vertical moves fail
                        path = [source_chiplet_idx]
                        current_x, current_y = source_ch_x, source_ch_y
                        detour_dir = -1 if current_x > 0 else 1
                        next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x + detour_dir)]
                        path.append(next_chip)
                        current_x += detour_dir
                        while current_y > target_y:
                            next_chip = self.chiplets_coordinate_idx_dict[(current_y - 1, current_x)]
                            path.append(next_chip)
                            current_y -= 1
                        while current_x < target_x:
                            next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x + 1)]
                            path.append(next_chip)
                            current_x += 1
                        while current_x > target_x:
                            next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x - 1)]
                            path.append(next_chip)
                            current_x -= 1
                    else:
                        path = x_path[:-1]
                        last_chip = x_path[-1]
                        current_y, current_x = self.chiplets_coordinate_dict[last_chip]
                        detour_dir = -1 if current_x > 0 else 1
                        current_x += detour_dir
                        while current_y > target_y:
                            next_chip = self.chiplets_coordinate_idx_dict[(current_y - 1, current_x)]
                            path.append(next_chip)
                            current_y -= 1
                        while current_x < target_x:
                            next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x + 1)]
                            path.append(next_chip)
                            current_x += 1

        elif current_x > target_x and current_y <= target_y:
            while current_x > target_x:
                next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x - 1)]
                path.append(next_chip)
                current_x -= 1

            for i in range(len(path) - 1):
                pair = (path[i], path[i + 1])
                if pair in self.failed_ch2ch_links or (pair[1], pair[0]) in self.failed_ch2ch_links:
                    x_failed_flag = True
                    break

            if x_failed_flag:
                path = [source_chiplet_idx]
                current_x, current_y = source_ch_x, source_ch_y
                next_chip = self.chiplets_coordinate_idx_dict[(current_y + 1, current_x)]
                path.append(next_chip)
                current_y += 1
                while current_x > target_x:
                    next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x - 1)]
                    path.append(next_chip)
                    current_x -= 1
                while current_y < target_y:
                    next_chip = self.chiplets_coordinate_idx_dict[(current_y + 1, current_x)]
                    path.append(next_chip)
                    current_y += 1
            else:
                x_path = path.copy()
                while current_y < target_y:
                    next_chip = self.chiplets_coordinate_idx_dict[(current_y + 1, current_x)]
                    path.append(next_chip)
                    current_y += 1
                for i in range(len(x_path) - 1, len(path) - 1):
                    pair = (path[i], path[i + 1])
                    if pair in self.failed_ch2ch_links or (pair[1], pair[0]) in self.failed_ch2ch_links:
                        y_failed_flag = True
                        break
                if y_failed_flag:
                    path = x_path[:-1]
                    last_chip = x_path[-1]
                    current_y, current_x = self.chiplets_coordinate_dict[last_chip]
                    # Detour: try moving right (if possible) to bypass the vertical failure.
                    detour_dir = 1
                    current_x += detour_dir
                    while current_y < target_y:
                        next_chip = self.chiplets_coordinate_idx_dict[(current_y + 1, current_x)]
                        path.append(next_chip)
                        current_y += 1
                    while current_x > target_x:
                        next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x - 1)]
                        path.append(next_chip)
                        current_x -= 1

        else:
            while current_x > target_x:
                next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x - 1)]
                path.append(next_chip)
                current_x -= 1

            for i in range(len(path) - 1):
                pair = (path[i], path[i + 1])
                if pair in self.failed_ch2ch_links or (pair[1], pair[0]) in self.failed_ch2ch_links:
                    x_failed_flag = True
                    break

            if x_failed_flag:
                path = [source_chiplet_idx]
                current_x, current_y = source_ch_x, source_ch_y
                next_chip = self.chiplets_coordinate_idx_dict[(current_y - 1, current_x)]
                path.append(next_chip)
                current_y -= 1
                while current_x > target_x:
                    next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x - 1)]
                    path.append(next_chip)
                    current_x -= 1
                while current_y > target_y:
                    next_chip = self.chiplets_coordinate_idx_dict[(current_y - 1, current_x)]
                    path.append(next_chip)
                    current_y -= 1
            else:
                x_path = path.copy()
                while current_y > target_y:
                    next_chip = self.chiplets_coordinate_idx_dict[(current_y - 1, current_x)]
                    path.append(next_chip)
                    current_y -= 1
                for i in range(len(x_path) - 1, len(path) - 1):
                    pair = (path[i], path[i + 1])
                    if pair in self.failed_ch2ch_links or (pair[1], pair[0]) in self.failed_ch2ch_links:
                        y_failed_flag = True
                        break
                if y_failed_flag:
                    path = x_path[:-1]
                    last_chip = x_path[-1]
                    current_y, current_x = self.chiplets_coordinate_dict[last_chip]
                    # Detour: try moving right to bypass the failure.
                    current_x = current_x + 1
                    while current_y > target_y:
                        next_chip = self.chiplets_coordinate_idx_dict[(current_y - 1, current_x)]
                        path.append(next_chip)
                        current_y -= 1
                    while current_x > target_x:
                        next_chip = self.chiplets_coordinate_idx_dict[(current_y, current_x - 1)]
                        path.append(next_chip)
                        current_x -= 1

        return path


    def xy_pointopoint_path(
        self,
        source_node_idx: Tuple[int],
        target_node_idx: Tuple[int]
    ) -> List[Tuple[int]]:

        source_ch_y, source_ch_x, source_co_y, source_co_x = self.nodes_coordinate_dict[source_node_idx]
        target_ch_y, target_ch_x, target_co_y, target_co_x = self.nodes_coordinate_dict[target_node_idx]

        if source_ch_y == target_ch_y and source_ch_x == target_ch_x:
            # Same chiplet
            path = self.xy_intrachiplet_path(source_node_idx, target_node_idx)
            return path
        else:
            source_ch_idx = source_node_idx[0]
            target_ch_idx = target_node_idx[0]
            ch_paths = self.xy_interchiplet_path(source_ch_idx, target_ch_idx)
            ch_paths_length = len(ch_paths)
            ch_pairs_length = len(ch_paths) - 1
            co_paths = []
            for pair_idx in range(ch_pairs_length):
                ch_pair = (ch_paths[pair_idx], ch_paths[pair_idx + 1])
                co_pair = self.chiplet_links_dict[ch_pair]
                co_paths += co_pair
                if pair_idx < ch_pairs_length - 1:
                    next_ch_pair = (ch_paths[pair_idx + 1], ch_paths[pair_idx + 2])
                    next_co_pair = self.chiplet_links_dict[next_ch_pair]
                    co_paths += self.xy_intrachiplet_path(
                        co_pair[-1],
                        next_co_pair[0]
                    )
            ch_starting = co_paths[0]
            ch_ending = co_paths[-1]
            if source_node_idx == ch_starting:
                pass
            else:
                first_intra_path = self.xy_intrachiplet_path(source_node_idx, ch_starting)
                co_paths = first_intra_path + co_paths
            if target_node_idx == ch_ending:
                pass
            else:
                last_intra_path = self.xy_intrachiplet_path(ch_ending, target_node_idx)
                co_paths += last_intra_path
            true_co_paths = []
            for co in co_paths:
                if co not in true_co_paths:
                    true_co_paths.append(co)
            return true_co_paths


    def xy_multicast_path(
        self,
        source_node_idx: Tuple[int],
        target_nodes_set: Set[Tuple[int]]
    ) -> Dict[Tuple[int], List[Tuple[int]]]:

        xy_path = {}
        xy_path[source_node_idx] = []
        for target_node_idx in target_nodes_set:
            source_target_path = self.xy_pointopoint_path(source_node_idx, target_node_idx)
            source_target_path_length = len(source_target_path)
            for source_pair_idx in range(source_target_path_length - 1):
                source_pair = source_target_path[source_pair_idx]
                target_pair = source_target_path[source_pair_idx + 1]
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
                    xy_links = []
                    for root in xy_path:
                        for leaf in xy_path[root]:
                            xy_links.append((root, leaf))
                    self.xy_paths[source_node_idx][target_node_idx] = xy_path
                    self.node_to_node_manhattan_hops_dict[source_node_idx][target_node_idx] = len(xy_links)
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

    hardware_cfg = cfg_to_dict(os.path.join(file_path, "../../../platform/cfgs/template.cfg"))

    network = tlm2d(hardware_cfg)


    # network.draw_topology()
    ch_idx = 0
    co_number = network.l1_width * network.l1_height

    best_link = 800
    for iters in range(1):
        network.init_nodes()
        network.init_links()
        network.dijkstra_offload()
        alltoall_pairs = []
        alltoall_dict = {}
        task_idx = 0
        for i in range(co_number):
            for j in range(co_number):
                if i != j:
                    alltoall_pairs.append([(1, task_idx), (ch_idx, i), {(ch_idx, j)}, 1024*512])
                    alltoall_dict[(1, task_idx)] = [(ch_idx, i), {(ch_idx, j)}]
                    # alltoall_pairs.append([(2, task_idx), (ch_idx, i), {(ch_idx, j)}, 1024*512])
                    task_idx += 1

        max_link_load, total_link_load, paths, task_dict = network.record_dijkstra_multicast_path(
            comm_pairs=alltoall_pairs
        )
        max_link_load, total_link_load, paths, task_dict = network.iter_worse_tasks(
            original_comm_pairs_dict=task_dict,
            original_tag_paths_dict=paths,
            total_iterations=2000
        )        
        if max_link_load <= best_link:
            best_link = max_link_load
            print("iteration", iters, "Best link load:", best_link)
            path_cnt = {}
            for task_idx in paths:
                for source_node in paths[task_idx]:
                    for target_node in paths[task_idx][source_node]:
                        if (source_node, target_node) in path_cnt:
                            path_cnt[(source_node, target_node)] += 1
                        else:
                            path_cnt[(source_node, target_node)] = 1
            print("Max used link count:", max(path_cnt.values()))

        import pickle
        paths_file = os.path.join(file_path, f"../../../../tmp/alltoall_{network.l1_width}x{network.l1_height}_paths.pkl")
        with open(paths_file, "wb") as f:
            pickle.dump(paths, f)

        import pickle
        paths_file = os.path.join(file_path, f"../../../../tmp/alltoall_{network.l1_width}x{network.l1_height}_paths.pkl")
        paths = pickle.load(open(paths_file, "rb"))
        print("Example paths:", paths)

        timestep, kinds = network.dijkstra_link_allocation(comm_pairs=task_dict, multiple_paths=[paths], iteration_number=100)
        print("Max time step:", timestep, "with", len(kinds), "kinds of allocation")
