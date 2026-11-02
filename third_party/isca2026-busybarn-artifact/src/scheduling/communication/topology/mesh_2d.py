
"""2D mesh topology: XY/YX routing, Dijkstra path allocation, multicast tree
construction, and AllGather/AllToAll collective pattern generation."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))


from net import net

from functools import lru_cache
from typing import List, Tuple, Set, Dict, Deque
from collections import deque, defaultdict
import itertools
import heapq
import numpy as np
import random
from copy import deepcopy
import math

class mesh_2d(net):
    """Simple 2D mesh with flat node indexing (no chiplet hierarchy).
    Supports fault-tolerant link setup and quadrant-based link ordering."""

    def __init__(self, width: int, height: int, failed_nodes=set(), failed_links=set()):

        super().__init__()

        self.width = width
        self.height = height

        self.failed_nodes_set = failed_nodes
        self.failed_links_set = failed_links

        self.init_nodes()
        self.init_links()
        self.init_links_deque()
        self.dijkstra_offload()

        self.nodes_number = len(self.nodes_set)


    def init_nodes(self):

        self.nodes_set = set()
        self.nodes_coordinate_dict = {}
        self.nodes_coordinate_idx_dict = {}

        for j in range(self.height):
            for i in range(self.width):
                idx = j * self.width + i
                if idx in self.failed_nodes_set:
                    continue
                self.nodes_set.add(idx)
                # NOTE: (row_idx, column_idx) = (y, x)
                self.nodes_coordinate_dict[idx] = (j, i)
                self.nodes_coordinate_idx_dict[(j, i)] = idx


    def init_links(self):
        # Build bidirectional links with quadrant-based preferred direction ordering.
        # Each quadrant adds links in a spiral-outward order for Hamiltonian path coverage.
        self.links_set = set()
        self.available_links = {}
        self.available_neighbors = {}

        width_odd = self.width % 2
        height_odd = self.height % 2

        width_half = self.width // 2
        height_half = self.height // 2

        for y in range(self.height):
            for x in range(self.width):
                current_idx = y * self.width + x
                if current_idx in self.failed_nodes_set:
                    continue

                right_idx = current_idx + 1
                left_idx = current_idx - 1
                top_idx = current_idx - self.width
                bottom_idx = current_idx + self.width

                available_link = set()
                neighbor = set()
                # left_top: begin from right
                if (x < width_half and y < width_half) or (width_odd == 1 and x == width_half):
                    if x + 1 < self.width and right_idx not in self.failed_nodes_set and (current_idx, right_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, right_idx))
                        available_link.add((0, 1))
                        neighbor.add(right_idx)
                    if y + 1 < self.height and bottom_idx not in self.failed_nodes_set and (current_idx, bottom_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, bottom_idx))
                        available_link.add((1, 0))
                        neighbor.add(bottom_idx)
                    if x - 1 >= 0 and left_idx not in self.failed_nodes_set and (current_idx, left_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, left_idx))
                        available_link.add((0, -1))
                        neighbor.add(left_idx)
                    if y - 1 >= 0 and top_idx not in self.failed_nodes_set and (current_idx, top_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, top_idx))
                        available_link.add((-1, 0))
                        neighbor.add(top_idx)
                # right_top: begin from bottom
                elif (x >= width_half and y < height_half) or (height_odd == 1 and y == height_half):
                    if y + 1 < self.height and bottom_idx not in self.failed_nodes_set and (current_idx, bottom_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, bottom_idx))
                        available_link.add((1, 0))
                        neighbor.add(bottom_idx)
                    if x - 1 >= 0 and left_idx not in self.failed_nodes_set and (current_idx, left_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, left_idx))
                        available_link.add((0, -1))
                        neighbor.add(left_idx)
                    if y - 1 >= 0 and top_idx not in self.failed_nodes_set and (current_idx, top_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, top_idx))
                        available_link.add((-1, 0))
                        neighbor.add(top_idx)
                    if x + 1 < self.width and right_idx not in self.failed_nodes_set and (current_idx, right_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, right_idx))
                        available_link.add((0, 1))
                        neighbor.add(right_idx)
                # right_bottom: begin from left
                elif x >= width_half and y >= height_half:
                    if x - 1 >= 0 and left_idx not in self.failed_nodes_set and (current_idx, left_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, left_idx))
                        available_link.add((0, -1))
                        neighbor.add(left_idx)
                    if y - 1 >= 0 and top_idx not in self.failed_nodes_set and (current_idx, top_idx) not in self.failed_links_set: 
                        self.links_set.add((current_idx, top_idx))
                        available_link.add((-1, 0))
                        neighbor.add(top_idx)
                    if x + 1 < self.width and right_idx not in self.failed_nodes_set and (current_idx, right_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, right_idx))
                        available_link.add((0, 1))
                        neighbor.add(right_idx)
                    if y + 1 < self.height and bottom_idx not in self.failed_nodes_set and (current_idx, bottom_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, bottom_idx))
                        available_link.add((1, 0))
                        neighbor.add(bottom_idx)
                # left_bottom: begin from top
                else:
                    if y - 1 >= 0 and top_idx not in self.failed_nodes_set and (current_idx, top_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, top_idx))
                        available_link.add((-1, 0))
                        neighbor.add(top_idx)
                    if x + 1 < self.width and right_idx not in self.failed_nodes_set and (current_idx, right_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, right_idx))
                        available_link.add((0, 1))
                        neighbor.add(right_idx)
                    if y + 1 < self.height and bottom_idx not in self.failed_nodes_set and (current_idx, bottom_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, bottom_idx))
                        available_link.add((1, 0))
                        neighbor.add(bottom_idx)
                    if x - 1 >= 0 and left_idx not in self.failed_nodes_set and (current_idx, left_idx) not in self.failed_links_set:
                        self.links_set.add((current_idx, left_idx))
                        available_link.add((0, -1))
                        neighbor.add(left_idx)

                self.available_links[current_idx] = available_link
                self.available_neighbors[current_idx] = neighbor


    def add_right_link(self, current_idx: int):

        current_y, current_x = self.idx_to_coordinate(current_idx)
        right_idx = current_idx + 1
        if current_x + 1 < self.width and right_idx not in self.failed_nodes_set and (current_idx, right_idx) not in self.failed_links_set:
            return (0, 1), right_idx
        # left_bottom: begin from top
        else:
            return None, None


    def add_left_link(self, current_idx: int):

        current_y, current_x = self.idx_to_coordinate(current_idx)
        left_idx = current_idx - 1
        if current_x - 1 >= 0 and left_idx not in self.failed_nodes_set and (current_idx, left_idx) not in self.failed_links_set:
            return (0, -1), left_idx
        else:
            return None, None


    def add_top_link(self, current_idx: int):

        current_y, current_x = self.idx_to_coordinate(current_idx)
        top_idx = current_idx - self.width
        if current_y - 1 >= 0 and top_idx not in self.failed_nodes_set and (current_idx, top_idx) not in self.failed_links_set:
            return (-1, 0), top_idx
        else:
            return None, None


    def add_bottom_link(self, current_idx: int):

        current_y, current_x = self.idx_to_coordinate(current_idx)
        bottom_idx = current_idx + self.width
        if current_y + 1 < self.height and bottom_idx not in self.failed_nodes_set and (current_idx, bottom_idx) not in self.failed_links_set:
            return (1, 0), bottom_idx
        else:
            return None, None

    @lru_cache(maxsize=1000)
    def add_links(self, current_idx: int, add_strategy: str = "trbl"):
        # Return (link_set, available_link_deque, neighbor_deque) for a node
        # with links ordered according to the given strategy (trbl, tlbr, etc.)

        link_set = set()
        available_link = deque()
        neighbor = deque()

        top_link, top_neighbor = self.add_top_link(current_idx)
        right_link, right_neighbor = self.add_right_link(current_idx)
        bottom_link, bottom_neighbor = self.add_bottom_link(current_idx)
        left_link, left_neighbor = self.add_left_link(current_idx)

        if add_strategy == "trbl":
            if top_link:
                link_set.add((current_idx, top_neighbor))
                available_link.append(top_link)
                neighbor.append(top_neighbor)
            if right_link:
                link_set.add((current_idx, right_neighbor))
                available_link.append(right_link)
                neighbor.append(right_neighbor)
            if bottom_link:
                link_set.add((current_idx, bottom_neighbor))
                available_link.append(bottom_link)
                neighbor.append(bottom_neighbor)
            if left_link:
                link_set.add((current_idx, left_neighbor))
                available_link.append(left_link)
                neighbor.append(left_neighbor)
        elif add_strategy == "tlbr":
            if top_link:
                link_set.add((current_idx, top_neighbor))
                available_link.append(top_link)
                neighbor.append(top_neighbor)
            if left_link:
                link_set.add((current_idx, left_neighbor))
                available_link.append(left_link)
                neighbor.append(left_neighbor)
            if bottom_link:
                link_set.add((current_idx, bottom_neighbor))
                available_link.append(bottom_link)
                neighbor.append(bottom_neighbor)
            if right_link:
                link_set.add((current_idx, right_neighbor))
                available_link.append(right_link)
                neighbor.append(right_neighbor)
        elif add_strategy == "tbrl":
            if top_link:
                link_set.add((current_idx, top_neighbor))
                available_link.append(top_link)
                neighbor.append(top_neighbor)
            if bottom_link:
                link_set.add((current_idx, bottom_neighbor))
                available_link.append(bottom_link)
                neighbor.append(bottom_neighbor)
            if right_link:
                link_set.add((current_idx, right_neighbor))
                available_link.append(right_link)
                neighbor.append(right_neighbor)
            if left_link:
                link_set.add((current_idx, left_neighbor))
                available_link.append(left_link)
                neighbor.append(left_neighbor)
        elif add_strategy == "rltb":
            if right_link:
                link_set.add((current_idx, right_neighbor))
                available_link.append(right_link)
                neighbor.append(right_neighbor)
            if left_link:
                link_set.add((current_idx, left_neighbor))
                available_link.append(left_link)
                neighbor.append(left_neighbor)
            if top_link:
                link_set.add((current_idx, top_neighbor))
                available_link.append(top_link)
                neighbor.append(top_neighbor)
            if bottom_link:
                link_set.add((current_idx, bottom_neighbor))
                available_link.append(bottom_link)
                neighbor.append(bottom_neighbor)
        elif add_strategy == "location":

            width_odd = self.width % 2
            height_odd = self.height % 2

            width_half = self.width // 2
            height_half = self.height // 2

            current_x, current_y = self.idx_to_coordinate(current_idx)

            # left_top: begin from right
            if (current_x < width_half and current_y < width_half) or (width_odd == 1 and current_x == width_half):
                if right_link:
                    link_set.add((current_idx, right_neighbor))
                    available_link.append(right_link)
                    neighbor.append(right_neighbor)
                if bottom_link:
                    link_set.add((current_idx, bottom_neighbor))
                    available_link.append(bottom_link)
                    neighbor.append(bottom_neighbor)
                if left_link:
                    link_set.add((current_idx, left_neighbor))
                    available_link.append(left_link)
                    neighbor.append(left_neighbor)
                if top_link:
                    link_set.add((current_idx, top_neighbor))
                    available_link.append(top_link)
                    neighbor.append(top_neighbor)
            # right_top: begin from bottom
            elif (current_x >= width_half and current_y < height_half) or (height_odd == 1 and current_y == height_half):
                if bottom_link:
                    link_set.add((current_idx, bottom_neighbor))
                    available_link.append(bottom_link)
                    neighbor.append(bottom_neighbor)
                if left_link:
                    link_set.add((current_idx, left_neighbor))
                    available_link.append(left_link)
                    neighbor.append(left_neighbor)
                if top_link:
                    link_set.add((current_idx, top_neighbor))
                    available_link.append(top_link)
                    neighbor.append(top_neighbor)
                if right_link:
                    link_set.add((current_idx, right_neighbor))
                    available_link.append(right_link)
                    neighbor.append(right_neighbor)
            # right_bottom: begin from left
            elif current_x >= width_half and current_y >= height_half:
                if left_link:
                    link_set.add((current_idx, left_neighbor))
                    available_link.append(left_link)
                    neighbor.append(left_neighbor)
                if top_link:
                    link_set.add((current_idx, top_neighbor))
                    available_link.append(top_link)
                    neighbor.append(top_neighbor)
                if right_link:
                    link_set.add((current_idx, right_neighbor))
                    available_link.append(right_link)
                    neighbor.append(right_neighbor)
                if bottom_link:
                    link_set.add((current_idx, bottom_neighbor))
                    available_link.append(bottom_link)
                    neighbor.append(bottom_neighbor)
            else:
                if top_link:
                    link_set.add((current_idx, top_neighbor))
                    available_link.append(top_link)
                    neighbor.append(top_neighbor)
                if right_link:
                    link_set.add((current_idx, right_neighbor))
                    available_link.append(right_link)
                    neighbor.append(right_neighbor)
                if bottom_link:
                    link_set.add((current_idx, bottom_neighbor))
                    available_link.append(bottom_link)
                    neighbor.append(bottom_neighbor)
                if left_link:
                    link_set.add((current_idx, left_neighbor))
                    available_link.append(left_link)
                    neighbor.append(left_neighbor)
        elif add_strategy == "random":
            if top_link:
                link_set.add((current_idx, top_neighbor))
                available_link.append(top_link)
                neighbor.append(top_neighbor)
            if right_link:
                link_set.add((current_idx, right_neighbor))
                available_link.append(right_link)
                neighbor.append(right_neighbor)
            if bottom_link:
                link_set.add((current_idx, bottom_neighbor))
                available_link.append(bottom_link)
                neighbor.append(bottom_neighbor)
            if left_link:
                link_set.add((current_idx, left_neighbor))
                available_link.append(left_link)
                neighbor.append(left_neighbor)
            available_link_list = list(available_link)
            neighbor_list = list(neighbor)
            indices = list(range(len(available_link_list)))
            random.shuffle(indices)
            available_link = deque([available_link_list[i] for i in indices])
            neighbor = deque([neighbor_list[i] for i in indices])
        else:
            raise ValueError("Invalid add strategy")

        return link_set, available_link, neighbor


    def init_links_deque(self, add_strategy: str = "tbrl"):
        # Build ordered neighbor deques for each node (used by routing algorithms)

        for y in range(self.height):
            for x in range(self.width):
                current_idx = y * self.width + x
                if current_idx in self.failed_nodes_set:
                    continue

                link_set, available_link, neighbor = self.add_links(current_idx, add_strategy)
                self.available_links_deque[current_idx] = available_link
                self.available_neighbors_deque[current_idx] = neighbor


    def coordinate_in_network(self, x: int, y: int) -> bool:

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return True


    def dijkstra_offload(self):

        self.link_loads_dict = {link: 0 for link in self.links_set}


    def record_dijkstra_path(self, source_node_idx: int, target_node_idx: int, link_load: int = 1) -> List[int]:

        source_y, source_x = self.idx_to_coordinate(source_node_idx)
        target_y, target_x = self.idx_to_coordinate(target_node_idx)
        x_dist = abs(source_x - target_x)
        y_dist = abs(source_y - target_y)

        if x_dist >= y_dist:
            if source_x <= target_x and source_y <= target_y:
                directions = [(0, 1), (1, 0)]
            elif source_x <= target_x and source_y >= target_y:
                directions = [(0, 1), (-1, 0)]
            elif source_x >= target_x and source_y <= target_y:
                directions = [(0, -1), (1, 0)]
            else:
                directions = [(0, -1), (-1, 0)]
        else:
            if source_x <= target_x and source_y <= target_y:
                directions = [(1, 0), (0, 1)]
            elif source_x <= target_x and source_y >= target_y:
                directions = [(-1, 0), (0, 1)]
            elif source_x >= target_x and source_y <= target_y:
                directions = [(1, 0), (0, -1)]
            else:
                directions = [(-1, 0), (0, -1)]

        queue = []
        heapq.heappush(queue, (0, source_node_idx))
        distances = {node: float('inf') for node in self.nodes_set}
        distances[source_node_idx] = 0
        predecessors = {node: None for node in self.nodes_set}

        while queue:
            current_distance, current_node = heapq.heappop(queue)
            if current_node == target_node_idx:
                path = []
                node = target_node_idx
                while node is not None:
                    path.append(node)
                    node = predecessors[node]
                path.reverse()
                if path[0] != source_node_idx:
                    return []
                else:
                    path_length = len(path) - 1
                    for path_i in range(path_length):
                        link = (path[path_i], path[path_i + 1])
                        self.link_loads_dict[link] += link_load
                    return path

            for dy, dx in directions:
                neighbor_idx = current_node + dy * self.width + dx
                if neighbor_idx not in self.available_neighbors[current_node]:
                    continue
                new_distance = current_distance + 1 + self.link_loads_dict[(current_node, neighbor_idx)]
                if new_distance < distances[neighbor_idx]:
                    distances[neighbor_idx] = new_distance
                    predecessors[neighbor_idx] = current_node
                    heapq.heappush(queue, (new_distance, neighbor_idx))

        return []


    def record_dijkstra_broadcast(self, source_node_idx: int, target_nodes_idx: Set[int], link_load: int = 1) -> List[int]:

        source_y, source_x = self.idx_to_coordinate(source_node_idx)
        end_node_idx = max(target_nodes_idx, key=lambda x: (abs(self.idx_to_coordinate(x)[1] - source_x) + abs(self.idx_to_coordinate(x)[0] - source_y)))
        long_path = self.record_dijkstra_path(source_node_idx, end_node_idx, link_load)
        long_path_length = len(long_path) - 1

        iter_target_nodes_idx = deepcopy(target_nodes_idx)
        iter_target_nodes_idx.remove(end_node_idx)
        iter_path_nodes_idx = deepcopy(long_path)
        branch_list = defaultdict(list)
        for path_i in range(long_path_length):
            node_idx = long_path[path_i]
            branch_list[node_idx].append(long_path[path_i + 1])
        iter_target_nodes_idx.difference_update(set(long_path))
        while iter_target_nodes_idx != set():
            min_difference = np.inf
            for set_elem in iter_target_nodes_idx:
                for list_elem in iter_path_nodes_idx:
                    current_difference = (abs(self.idx_to_coordinate(set_elem)[0] - self.idx_to_coordinate(list_elem)[0]) + abs(self.idx_to_coordinate(set_elem)[1] - self.idx_to_coordinate(list_elem)[1]))
                    if current_difference < min_difference:
                        min_difference = current_difference
                        min_pair = (set_elem, list_elem)
            short_path = self.record_dijkstra_path(min_pair[1], min_pair[0], link_load)
            short_path_length = len(short_path) - 1
            for path_i in range(short_path_length):
                node_idx = short_path[path_i]
                branch_list[node_idx].append(short_path[path_i + 1])
            iter_target_nodes_idx.remove(min_pair[0])
            iter_path_nodes_idx.append(min_pair[0])

        return [source_node_idx, end_node_idx, branch_list]


    def record_dijkstra_broadcast(self, 
                                source_node_idx: int, 
                                target_nodes_idx: Set[int], 
                                link_load: int = 1
                                ) -> List:
        """
        Broadcast from ``source_node_idx`` to multiple targets in
        ``target_nodes_idx`` using Dijkstra with link-load-aware edge costs.
        Update link loads along the selected paths and return
        ``[source_node_idx, end_node_idx, branch_list]``.

        Example ``branch_list``:
        {
        4: [8],
        8: [9],
        9: [10],
        10: [11, 6],
        6: [2]
        }
        This means 4->8->9->10 is one path, node 10 branches to 11 and 6,
        and node 6 continues to 2.

        Returns:
            [
                source_node_idx,          # e.g. 4
                end_node_idx,             # e.g. 11, depending on caller needs
                branch_dict               # e.g. {4: [8], 8: [9], 9: [10], 10: [11, 6], 6: [2]}
            ]
        """

        distances = {node: float('inf') for node in self.nodes_set}
        distances[source_node_idx] = 0

        predecessors = {node: None for node in self.nodes_set}

        queue = []
        heapq.heappush(queue, (0, source_node_idx))

        found_targets_count = 0
        total_targets = len(target_nodes_idx)

        while queue:
            current_dist, current_node = heapq.heappop(queue)

            if current_dist > distances[current_node]:
                continue

            if current_node in target_nodes_idx:
                found_targets_count += 1
                if found_targets_count == total_targets:
                    break

            for neighbor_idx in self.available_neighbors[current_node]:
                edge_weight = link_load + self.link_loads_dict[(current_node, neighbor_idx)]
                new_dist = distances[current_node] + edge_weight

                if new_dist < distances[neighbor_idx]:
                    distances[neighbor_idx] = new_dist
                    predecessors[neighbor_idx] = current_node
                    heapq.heappush(queue, (new_dist, neighbor_idx))

        branch_dict: Dict[int, List[int]] = {}

        if target_nodes_idx:
            end_node_idx = next(iter(target_nodes_idx))
        else:
            return [source_node_idx, None, {}]

        for t_node in target_nodes_idx:
            if distances[t_node] == float('inf'):
                continue

            path = []
            cur = t_node
            while cur is not None:
                path.append(cur)
                cur = predecessors[cur]
            path.reverse()

            if not path or path[0] != source_node_idx:
                continue

            for i in range(len(path) - 1):
                p = path[i]
                c = path[i + 1]

                self.link_loads_dict[(p, c)] += link_load
                print((p, c), self.link_loads_dict[(p, c)])

                if p not in branch_dict:
                    branch_dict[p] = []
                if c not in branch_dict[p]:
                    branch_dict[p].append(c)

        return [source_node_idx, end_node_idx, branch_dict]


    def record_dijkstra_tree(self, source_node_idx: int, link_load: int = 1) -> Dict[int, List[int]]:
        """
        Run Dijkstra from ``source_node_idx`` across the full grid, build the
        shortest-path tree, and return it as ``{parent: [children, ...], ...}``.
        """

        distances = {node: float('inf') for node in self.nodes_set}
        distances[source_node_idx] = 0

        predecessors = {node: None for node in self.nodes_set}

        queue = []
        heapq.heappush(queue, (0, source_node_idx))

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor_idx in self.available_neighbors[current_node]:
                new_distance = current_distance + 1 + self.link_loads_dict.get((current_node, neighbor_idx), 0)

                if new_distance < distances[neighbor_idx]:
                    distances[neighbor_idx] = new_distance
                    predecessors[neighbor_idx] = current_node
                    heapq.heappush(queue, (new_distance, neighbor_idx))

        tree_dict = defaultdict(list)

        for node in self.nodes_set:
            pred = predecessors[node]
            if pred is not None:
                tree_dict[pred].append(node)

        for node in self.nodes_set:
            pred = predecessors[node]
            if pred is not None:
                self.link_loads_dict[(pred, node)] += link_load

        return dict(tree_dict)


    def record_dijkstra_tree_single_neighbor(
        self,
        source_node_idx: int,
        link_load: int = 1
    ) -> Dict[int, List[int]]:
        """
        A split-step variation of Dijkstra:
        - Relax only one neighbor of the current node at a time.
        - Use randomness to break ties between equal-distance nodes.
        - Still visit every reachable node and build a tree from the source.

        Returns:
            {parent_node: [child_node1, child_node2, ...], ...}
        """

        neighbors_copy = deepcopy(self.available_neighbors)
        for node_idx in neighbors_copy:
            if isinstance(neighbors_copy[node_idx], set):
                neighbors_copy[node_idx] = list(neighbors_copy[node_idx])

        distances = {node: float('inf') for node in self.nodes_set}
        distances[source_node_idx] = 0

        predecessors = {node: None for node in self.nodes_set}

        neighbor_ptr = {node: 0 for node in self.nodes_set}

        queue = []
        heapq.heappush(queue, (0, random.random(), source_node_idx))

        while queue:
            current_dist, _, current_node = heapq.heappop(queue)

            if current_dist > distances[current_node]:
                continue

            if neighbor_ptr[current_node] < len(neighbors_copy[current_node]):
                start_idx = neighbor_ptr[current_node]
                sub_neighbors = neighbors_copy[current_node][start_idx:]
                random.shuffle(sub_neighbors)
                neighbors_copy[current_node][start_idx:] = sub_neighbors

                neighbor_idx = neighbors_copy[current_node][neighbor_ptr[current_node]]
                neighbor_ptr[current_node] += 1

                new_dist = current_dist + 1 + self.link_loads_dict.get(
                    (current_node, neighbor_idx), 0
                )

                if new_dist < distances[neighbor_idx]:
                    distances[neighbor_idx] = new_dist
                    predecessors[neighbor_idx] = current_node
                    heapq.heappush(queue, (new_dist, random.random(), neighbor_idx))

            if neighbor_ptr[current_node] < len(neighbors_copy[current_node]):
                heapq.heappush(queue, (distances[current_node], random.random(), current_node))

        tree_dict = defaultdict(list)
        for node in self.nodes_set:
            parent = predecessors[node]
            if parent is not None:
                tree_dict[parent].append(node)

        for node in self.nodes_set:
            parent = predecessors[node]
            if parent is not None:
                self.link_loads_dict[(parent, node)] += link_load

        return dict(tree_dict)


    #         root: {node: float('inf') for node in self.nodes_set}
    def multi_root_single_neighbor_dijkstra_preserve_order_no_repeat(
        self, 
        link_load: int = 1
    ) -> Dict[int, Dict[int, List[int]]]:
        """
        Requirements:
        - Every node in ``self.nodes_set`` acts as a root.
        - Expand only one neighbor at a time for each queued node.
        - Each iteration must add one new node to the corresponding tree so
          no reachable node is skipped.
        - Break equal-distance ties with ``(dist, random.random())``.
        - Update ``link_loads_dict`` online. If a parent changes, roll back the
          old load and add it to the new edge.
        - Preserve randomized root processing order.

        Returns:
        multi_trees: { 
            rootA: { parentX: [child1, child2, ...], ... },
            rootB: { ... },
            ...
        }
        """

        neighbors_copy = deepcopy(self.available_neighbors)
        for n in neighbors_copy:
            if isinstance(neighbors_copy[n], set):
                neighbors_copy[n] = list(neighbors_copy[n])

        distances = {
            r: {n: float('inf') for n in self.nodes_set} 
            for r in self.nodes_set
        }
        adjacency = {
            r: defaultdict(list)
            for r in self.nodes_set
        }
        parent_of = {
            r: {n: None for n in self.nodes_set}
            for r in self.nodes_set
        }
        in_tree = {
            r: set() 
            for r in self.nodes_set
        }
        neighbor_ptr = {
            r: {n: 0 for n in self.nodes_set}
            for r in self.nodes_set
        }
        queues = {r: [] for r in self.nodes_set}

        for r in self.nodes_set:
            distances[r][r] = 0
            in_tree[r].add(r)
            heapq.heappush(queues[r], (0, random.random(), r))

        while True:
            if all(len(queues[r]) == 0 for r in self.nodes_set):
                break

            root_list = list(self.nodes_set)
            random.shuffle(root_list)

            for root in root_list:
                if not queues[root]:
                    continue

                curr_dist, _, curr_node = heapq.heappop(queues[root])
                if curr_dist > distances[root][curr_node]:
                    continue

                candidate_neighbors = []
                for nb in neighbors_copy[curr_node]:
                    # if True: candidate_neighbors.append(nb)
                    if nb not in in_tree[root]:
                        candidate_neighbors.append(nb)

                if not candidate_neighbors:
                    continue

                random.shuffle(candidate_neighbors)

                neighbor_idx = candidate_neighbors[0]

                old_dist = distances[root][neighbor_idx]
                new_dist = curr_dist + 1 + self.link_loads_dict.get((curr_node, neighbor_idx), 0)

                if new_dist < old_dist:
                    distances[root][neighbor_idx] = new_dist

                    old_parent = parent_of[root][neighbor_idx]
                    if old_parent is not None and neighbor_idx in in_tree[root]:
                        if neighbor_idx in adjacency[root][old_parent]:
                            adjacency[root][old_parent].remove(neighbor_idx)
                        self.link_loads_dict[(old_parent, neighbor_idx)] -= link_load
                        if self.link_loads_dict[(old_parent, neighbor_idx)] < 0:
                            self.link_loads_dict[(old_parent, neighbor_idx)] = 0

                    parent_of[root][neighbor_idx] = curr_node
                    adjacency[root][curr_node].append(neighbor_idx)
                    if neighbor_idx not in in_tree[root]:
                        in_tree[root].add(neighbor_idx)
                    self.link_loads_dict[(curr_node, neighbor_idx)] += link_load

                    heapq.heappush(
                        queues[root],
                        (new_dist, random.random(), neighbor_idx)
                    )

                other_candidates = [nb for nb in neighbors_copy[curr_node] if nb not in in_tree[root] and nb != neighbor_idx]
                if other_candidates:
                    heapq.heappush(
                        queues[root],
                        (distances[root][curr_node], random.random(), curr_node)
                    )

        multi_trees = {}
        for root in self.nodes_set:
            tree_dict = {}
            for p, clist in adjacency[root].items():
                if clist:
                    tree_dict[p] = clist
            multi_trees[root] = tree_dict

        return multi_trees


    def xy_path(self, source_node_idx: int, target_node_idx: int) -> List[int]:

        source_x, source_y = self.idx_to_coordinate(source_node_idx)
        target_x, target_y = self.idx_to_coordinate(target_node_idx)

        path = []
        path.append(source_node_idx)
        current_x, current_y = source_x, source_y
        if current_x <= target_x and current_y <= target_y:
            while current_x < target_x:
                path.append(self.coordinate_to_idx((current_x+1, current_y)))
                current_x += 1
            while current_y < target_y:
                path.append(self.coordinate_to_idx((current_x, current_y+1)))
                current_y += 1
        elif current_x <= target_x and current_y > target_y:
            while current_x < target_x:
                path.append(self.coordinate_to_idx((current_x+1, current_y)))
                current_x += 1
            while current_y > target_y:
                path.append(self.coordinate_to_idx((current_x, current_y-1)))
                current_y -= 1
        elif current_x > target_x and current_y <= target_y:
            while current_x > target_x:
                path.append(self.coordinate_to_idx((current_x-1, current_y)))
                current_x -= 1
            while current_y < target_y:
                path.append(self.coordinate_to_idx((current_x, current_y+1)))
                current_y += 1
        else:
            while current_x > target_x:
                path.append(self.coordinate_to_idx((current_x-1, current_y)))
                current_x -= 1
            while current_y > target_y:
                path.append(self.coordinate_to_idx((current_x, current_y-1)))
                current_y -= 1

        return path


    def xy_broadcast(self, source_node_idx: int, target_nodes_idx: Set[int], link_load: int = 1) -> List[int]:

        source_y, source_x = self.idx_to_coordinate(source_node_idx)
        end_node_idx = max(target_nodes_idx, key=lambda x: (abs(self.idx_to_coordinate(x)[1] - source_x) + abs(self.idx_to_coordinate(x)[0] - source_y)))
        long_path = self.xy_path(source_node_idx, end_node_idx)
        long_path_length = len(long_path) - 1

        iter_target_nodes_idx = deepcopy(target_nodes_idx)
        iter_target_nodes_idx.remove(end_node_idx)
        iter_path_nodes_idx = deepcopy(long_path)
        branch_list = defaultdict(list)
        for path_i in range(long_path_length):
            node_idx = long_path[path_i]
            branch_list[node_idx].append(long_path[path_i + 1])
        iter_target_nodes_idx.difference_update(set(long_path))
        while iter_target_nodes_idx != set():
            min_difference = np.inf
            for set_elem in iter_target_nodes_idx:
                for list_elem in iter_path_nodes_idx:
                    current_difference = (abs(self.idx_to_coordinate(set_elem)[0] - self.idx_to_coordinate(list_elem)[0]) + abs(self.idx_to_coordinate(set_elem)[1] - self.idx_to_coordinate(list_elem)[1]))
                    if current_difference < min_difference:
                        min_difference = current_difference
                        min_pair = (set_elem, list_elem)
            short_path = self.xy_path(min_pair[1], min_pair[0])
            short_path_length = len(short_path) - 1
            for path_i in range(short_path_length):
                node_idx = short_path[path_i]
                branch_list[node_idx].append(short_path[path_i + 1])
            iter_target_nodes_idx.remove(min_pair[0])
            iter_path_nodes_idx.append(min_pair[0])

        return [source_node_idx, end_node_idx, branch_list]


    def nodes_path_to_links_path(self, nodes_path: List[int]) -> List[Tuple[int, int]]:

        links_path = []
        nodes_number = len(nodes_path) - 1
        for i in range(nodes_number):
            links_path.append((nodes_path[i], nodes_path[i + 1]))
        return links_path


    def node_failure(self, fault_node_idx: int):

        fault_node_coordinate = self.idx_to_coordinate(fault_node_idx)
        self.nodes_set.remove(fault_node_idx)
        for links in self.links_set.copy():
            if fault_node_idx in links:
                self.links_set.remove(links)
                if hasattr(self, "link_loads_dict"):
                    del self.link_loads_dict[links]
        del self.nodes_coordinate_idx_dict[self.idx_to_coordinate(fault_node_idx)]
        del self.nodes_coordinate_dict[fault_node_idx]
        del self.available_links[fault_node_idx]
        for node_idx in self.available_links:
            current_coordinate = self.idx_to_coordinate(node_idx)
            for possible_link in self.available_links[node_idx].copy():
                availabe_coordinate = tuple(map(sum, zip(list(possible_link), list(current_coordinate))))
                if availabe_coordinate == fault_node_coordinate:
                    self.available_links[node_idx].remove(possible_link)
        del self.available_neighbors[fault_node_idx]
        for node_idx in self.available_neighbors:
            if fault_node_idx in self.available_neighbors[node_idx].copy():
                self.available_neighbors[node_idx].remove(fault_node_idx)


    def link_failure(self, fault_link: Tuple[int, int]):

        fault_reversed_link = fault_link[::-1]
        source_node_idx, target_node_idx = fault_link
        source_node_coordinate = self.idx_to_coordinate(source_node_idx)
        target_node_coordinate = self.idx_to_coordinate(target_node_idx)
        available_link_list = list(map(lambda x: x[1]-x[0], zip(list(source_node_coordinate), list(target_node_coordinate))))
        source_available_link = tuple(available_link_list)
        target_available_link = tuple([-x for x in available_link_list])

        self.links_set.remove(fault_link)
        self.links_set.remove(fault_reversed_link)

        if hasattr(self, "link_loads_dict"):
            del self.link_loads_dict[fault_link]
            del self.link_loads_dict[fault_reversed_link]

        self.available_links[source_node_idx].remove(source_available_link)
        self.available_links[target_node_idx].remove(target_available_link)
        self.available_neighbors[source_node_idx].remove(target_node_idx)
        self.available_neighbors[target_node_idx].remove(source_node_idx)


    def dijkstra_path(self, source_node_idx: int, target_node_idx) -> List[int]:

        source_y, source_x = self.idx_to_coordinate(source_node_idx)
        target_y, target_x = self.idx_to_coordinate(target_node_idx)
        x_dist = abs(source_x - target_x)
        y_dist = abs(source_y - target_y)

        if x_dist >= y_dist:
            if source_x <= target_x and source_y <= target_y:
                directions = [(0, 1), (1, 0)]
            elif source_x <= target_x and source_y >= target_y:
                directions = [(0, 1), (-1, 0)]
            elif source_x >= target_x and source_y <= target_y:
                directions = [(0, -1), (1, 0)]
            else:
                directions = [(0, -1), (-1, 0)]
        else:
            if source_x <= target_x and source_y <= target_y:
                directions = [(1, 0), (0, 1)]
            elif source_x <= target_x and source_y >= target_y:
                directions = [(-1, 0), (0, 1)]
            elif source_x >= target_x and source_y <= target_y:
                directions = [(1, 0), (0, -1)]
            else:
                directions = [(-1, 0), (0, -1)]

        queue = []
        heapq.heappush(queue, (0, source_node_idx))
        distances = {node: float('inf') for node in self.nodes_set}
        distances[source_node_idx] = 0
        predecessors = {node: None for node in self.nodes_set}

        while queue:
            current_distance, current_node = heapq.heappop(queue)
            if current_node == target_node_idx:
                path = []
                node = target_node_idx
                while node is not None:
                    path.append(node)
                    node = predecessors[node]
                path.reverse()
                if path[0] != source_node_idx:
                    return []
                else:
                    return path

            for dy, dx in directions:
                neighbor_idx = current_node + dy * self.width + dx
                if neighbor_idx not in self.available_neighbors[current_node]:
                    continue
                new_distance = current_distance + 1 
                if new_distance < distances[neighbor_idx]:
                    distances[neighbor_idx] = new_distance
                    predecessors[neighbor_idx] = current_node
                    heapq.heappush(queue, (new_distance, neighbor_idx))

        return []


    def generate_unique_combinations(self, original_dict):
        """
        Generate all unique combinations of dictionary values where each combination
        has unique elements across keys.

        Args:
        original_dict (dict): A dictionary where keys are identifiers and values are lists of possible elements.

        Returns:
        list: A list of dictionaries, each representing a unique combination without duplicate values.
        """
        keys = original_dict.keys()
        values_product = itertools.product(*(original_dict[key] for key in keys))

        valid_combinations = []
        for combination in values_product:
            if len(set(combination)) == len(combination):
                combination_dict = {key: [value] for key, value in zip(keys, combination)}
                valid_combinations.append(combination_dict)

        return valid_combinations


    # TODO: dualpath should be used in event_driven
    def dualpath_dijkstra(self, currentstep_path_dict: Dict[int, List[Tuple[int]]]) -> Dict[int, List[int]]:

        self.init_nodes()
        self.init_links()
        self.dijkstra_offload()

        for task_tag in currentstep_path_dict:
            task_path = currentstep_path_dict[task_tag]
            task_source = task_path[0]
            task_target = task_path[-1]
            task_length = len(task_path) - 1
            for path_i in range(task_length):
                loop_node_0_idx = task_path[path_i]
                loop_node_1_idx = task_path[path_i + 1]
                link = (loop_node_0_idx, loop_node_1_idx)
                neighbor_link = tuple(map(lambda x: x[1]-x[0], zip(list(self.idx_to_coordinate(loop_node_0_idx)), list(self.idx_to_coordinate(loop_node_1_idx)))))
                if link in self.links_set.copy():
                    self.links_set.remove(link)
                    self.available_links[loop_node_0_idx].remove(neighbor_link)
                    self.available_neighbors[loop_node_0_idx].remove(loop_node_1_idx)
                else:
                    self.init_nodes()
                    self.init_links()
                    self.dijkstra_offload()                    
                    return False


        copy_links_set = deepcopy(self.links_set)
        copy_available_links = deepcopy(self.available_links)
        copy_available_neighbors = deepcopy(self.available_neighbors)

        multipath_candidate_dict = {}
        for task_tag in currentstep_path_dict:
            task_source = currentstep_path_dict[task_tag][0]
            multipath_candidate_dict[task_source] = deepcopy(self.available_neighbors[task_source]) 
        multipath_candidates = self.generate_unique_combinations(multipath_candidate_dict)
        possible_multipath = []
        for multipath_candidate in multipath_candidates:
            multipath_dict = {}
            for task_tag in currentstep_path_dict:
                task_source = currentstep_path_dict[task_tag][0]
                task_newsource = multipath_candidate[task_source][0]
                task_target = currentstep_path_dict[task_tag][-1]
                multipath = self.dijkstra_path(task_newsource, task_target)
                if multipath:
                    multipath_dict[task_tag] = [task_source] + multipath
                else:
                    self.links_set = deepcopy(copy_links_set)
                    self.available_links = deepcopy(copy_available_links)
                    self.available_neighbors = deepcopy(copy_available_neighbors)  
                    multipath_dict = {}         
                    break
                multipath_length = len(multipath) - 1
                for path_i in range(multipath_length):
                    loop_node_0_idx = multipath[path_i]
                    loop_node_1_idx = multipath[path_i + 1]
                    link = (loop_node_0_idx, loop_node_1_idx)
                    neighbor_link = tuple(map(lambda x: x[1]-x[0], zip(list(self.idx_to_coordinate(loop_node_0_idx)), list(self.idx_to_coordinate(loop_node_1_idx)))))
                    if link in self.links_set.copy():
                        self.links_set.remove(link)
                        self.available_links[loop_node_0_idx].remove(neighbor_link)
                        self.available_neighbors[loop_node_0_idx].remove(loop_node_1_idx)
                    else:
                        self.links_set = deepcopy(copy_links_set)
                        self.available_links = deepcopy(copy_available_links)
                        self.available_neighbors = deepcopy(copy_available_neighbors)  
                        multipath_dict = {}         
                        break
            if multipath_dict != {}:
                possible_multipath.append(multipath_dict)
                self.links_set = deepcopy(copy_links_set)
                self.available_links = deepcopy(copy_available_links)
                self.available_neighbors = deepcopy(copy_available_neighbors) 

        if possible_multipath:
            return_multipath = possible_multipath[0]
            return_links_used = 0
            for ps in return_multipath:
                return_links_used += len(return_multipath[ps])

            for multipath in possible_multipath:
                links_used = 0
                for ps in multipath:
                    links_used += len(multipath[ps])
                if links_used < return_links_used:
                    return_multipath = multipath
                    return_links_used = links_used
        else:
            return_multipath = None

        self.init_nodes()
        self.init_links()
        self.dijkstra_offload()
        return return_multipath


    def onetree(self, root_idx: str, leaves: Deque[int]):

        tree_path = {root_idx: []}

        unchange_cnt = 0
        unchange_limit = len(leaves) ** 2
        while leaves:
            # the tree order is decided by the order of leaves
            candidate_leaf = leaves.popleft()
            leaf_find_root_flag = False
            for candidate_root in tree_path:
                if candidate_leaf in self.available_neighbors[candidate_root]:
                    tree_path[candidate_root].append(candidate_leaf)
                    tree_path[candidate_leaf] = []
                    leaf_find_root_flag = True
                    unchange_cnt = 0
                    break              
                else:
                    continue
            if not leaf_find_root_flag:
                leaves.append(candidate_leaf)

            unchange_cnt += 1
            if unchange_cnt > unchange_limit:
                raise ValueError("Cannot find a tree")

        return tree_path


    def multitree(self, root_leaves_dict: Dict[int, Set[int]], add_strategy: str = "trbl"):

        # the tree order will consider the order of available links and congestion
        root_list = list(root_leaves_dict.keys())
        '''
        trees:{
        0: [],
        1: [],
        ...
        }
        where []:
        [
        [(parent, children), (parent, children), ...], # timestep 1 
        ...
        ]

        '''
        trees = {root_idx : [[]] for root_idx in root_list}
        current_root = {root_idx : deque([root_idx]) for root_idx in root_list}
        time_step = 0

        while any(value != set() for value in root_leaves_dict.values()):

            timestep_rootidx_flag_list = [False for _ in root_list]
            for tree_root in trees:
                root_idx_deque = deque(sorted(current_root[tree_root], key=lambda x: len(self.available_neighbors_deque[x])))
                current_tree_addleaf_flag = False
                for current_root_idx in root_idx_deque:
                    current_root_addneighbor_flag = False
                    for available_neighbor in self.available_neighbors_deque[current_root_idx]:
                        if available_neighbor in root_leaves_dict[tree_root]:
                            trees[tree_root][time_step].append((current_root_idx, available_neighbor))
                            root_leaves_dict[tree_root].remove(available_neighbor)
                            current_root[tree_root].append(available_neighbor)
                            current_root_addneighbor_flag = True
                            break
                    if current_root_addneighbor_flag:
                        self.available_neighbors_deque[current_root_idx].remove(available_neighbor)
                        current_tree_addleaf_flag = True
                        break
                timestep_rootidx_flag_list[root_list.index(tree_root)] = current_tree_addleaf_flag

            if all(flag == False for flag in timestep_rootidx_flag_list):
                time_step += 1
                for tree_root in trees:
                    trees[tree_root].append([])
                self.init_links_deque(add_strategy=add_strategy)
            else:
                continue

        return trees


    def bfs_tree_from_source(self, source: int) -> Dict[int, int]:
        """
        Run BFS on the directed unweighted graph from ``source`` and return a
        predecessor map.

        ``predecessor[x] = p`` means that in the BFS tree rooted at ``source``,
        node ``p`` is the parent of ``x``. If ``x == source``,
        ``predecessor[x] = None``. Unreachable nodes also remain ``None``.

        This guarantees a unique ``source -> ... -> x`` path for each
        reachable node and avoids repeated branches within one BFS tree.
        """
        visited = {source}
        queue = deque([source])
        predecessor = {node: None for node in self.nodes_set}

        while queue:
            u = queue.popleft()
            for v in self.available_neighbors[u]:
                if v not in visited:
                    visited.add(v)
                    predecessor[v] = u
                    queue.append(v)

        return predecessor

    def extract_path(self, predecessor: Dict[int, int], source: int, target: int) -> List[int]:
        """
        Trace back from ``target`` to ``source`` using a BFS tree
        (or predecessor map) and return a simple path.
        Return ``[]`` when the target is unreachable.
        """
        if target == source:
            return [source]
        if predecessor[target] is None and source != target:
            return []

        path = []
        curr = target
        while curr is not None:
            path.append(curr)
            curr = predecessor[curr]
        path.reverse()
        return path

    def build_initial_paths(self) -> Dict[int, Dict[int, List[int]]]:
        """
        Build a BFS tree for every source node ``s`` and expand it into
        shortest paths ``s -> t``. This avoids duplicate nodes for a given
        source tree.

        Returns: ``all_paths[s][t] = [s, ..., t]``
        """
        all_paths = {}
        for s in self.nodes_set:
            pred_s = self.bfs_tree_from_source(s)
            all_paths[s] = {}
            for t in self.nodes_set:
                all_paths[s][t] = self.extract_path(pred_s, s, t)
        return all_paths

    def compute_max_edge_usage(self, all_paths: Dict[int, Dict[int, List[int]]]) -> int:
        """
        Count how often each directed edge ``(u -> v)`` is used in
        ``all_paths[s][t]`` and return the maximum count.

        :param all_paths: {s: {t: [s, ..., t]}}
        :return: ``max_count``, the maximum usage count over all edges
        """
        edge_count = {}
        for s in self.nodes_set:
            for t in self.nodes_set:
                path = all_paths[s][t]
                if len(path) < 2:
                    continue
                for i in range(len(path) - 1):
                    e = (path[i], path[i+1])
                    edge_count[e] = edge_count.get(e, 0) + 1
        if not edge_count:
            return 0
        return max(edge_count.values())

    def get_random_neighbor_whole_tree(
        self, 
        all_paths: Dict[int, Dict[int, List[int]]]
    ) -> Dict[int, Dict[int, List[int]]]:
        """
        Neighborhood move: randomly choose a source ``s``, rebuild the entire
        BFS tree for ``s``, and replace all paths ``s -> (*)``.
        This keeps a complete BFS tree for ``s`` without duplicate nodes.

        :param all_paths: current solution
        :return: new solution (deep copy)
        """
        import copy
        new_paths = copy.deepcopy(all_paths)

        s = random.choice(list(self.nodes_set))

        pred_s = self.bfs_tree_from_source(s)

        for t in self.nodes_set:
            new_paths[s][t] = self.extract_path(pred_s, s, t)

        return new_paths

    def simulated_annealing(
        self,
        init_paths: Dict[int, Dict[int, List[int]]],
        max_iter: int = 1000,
        start_temp: float = 10.0,
        alpha: float = 0.98
    ) -> Tuple[Dict[int, Dict[int, List[int]]], int]:
        """
        Use simulated annealing to replace one source BFS tree at a time and
        gradually reduce the maximum edge usage.

        :param init_paths: initial solution, such as from ``build_initial_paths()``
        :param max_iter: maximum number of iterations
        :param start_temp: initial temperature
        :param alpha: temperature decay rate
        :return: (best_solution, best_cost)
        """
        current_solution = init_paths
        current_cost = self.compute_max_edge_usage(current_solution)

        best_solution = current_solution
        best_cost = current_cost

        T = start_temp

        for iteration in range(max_iter):
            neighbor_solution = self.get_random_neighbor_whole_tree(current_solution)
            neighbor_cost = self.compute_max_edge_usage(neighbor_solution)

            if neighbor_cost < current_cost:
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                if neighbor_cost < best_cost:
                    best_cost = neighbor_cost
                    best_solution = neighbor_solution
            else:
                delta = neighbor_cost - current_cost
                accept_prob = math.exp(-delta / T)
                if random.random() < accept_prob:
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost

            T *= alpha


        # if best_cost <= (mesh_long + mesh_wide - 1):
        return best_solution, best_cost


if __name__ == "__main__":

    test_mesh2d = mesh_2d(5, 5)
    print("nodes_set:")
    print(test_mesh2d.nodes_set)
    print("links_set:")
    print(test_mesh2d.links_set)
    print("nodes_coordinate_dict:")
    print(test_mesh2d.nodes_coordinate_dict)
    print("nodes_coordinate_idx_dict:")
    print(test_mesh2d.nodes_coordinate_idx_dict)
    print("available_links:")
    print(test_mesh2d.available_links)
    print("______________________")
    print(test_mesh2d.available_links_deque)
    print("available_neighbors:")
    print(test_mesh2d.available_neighbors)
    print("______________________")
    print(test_mesh2d.available_neighbors_deque)


    test_mesh2d.init_links_deque(add_strategy="random")
    print("available_links_deque:")
    print(test_mesh2d.available_links_deque)


    # test_path = test_mesh2d.dijkstra_path(4, 0)
    test_path = test_mesh2d.record_dijkstra_path(0, 15)
    print("Test path:", test_path)
    multipath = test_mesh2d.dualpath_dijkstra({0: test_path})
    print("Mutipath:", multipath)
    test_path = test_mesh2d.record_dijkstra_broadcast(18, {11, 4, 21, 14})
    print("Broadcast path:", test_path)


    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ paper scheduling example")

    def find_dependencies(graph, start):
        def explore(node, path):
            if node not in graph or not graph[node]:
                return [path]
            results = []
            for next_node in graph[node]:
                results.extend(explore(next_node, path + [(node, next_node)]))
            return results

        return explore(start, [])

    test_mesh2d = mesh_2d(4, 3)
    source_node = 4
    target_node0 = 11
    target_node1 = 2
    # test_onetree = test_mesh2d.record_dijkstra_tree_single_neighbor(0)
    test_mesh2d.init_nodes()
    test_mesh2d.init_links()
    test_mesh2d.dijkstra_offload()
    print("two Dijkstra paths:")
    test_path_0 = test_mesh2d.record_dijkstra_path(source_node, target_node0)
    print(test_path_0)
    test_path_1 = test_mesh2d.record_dijkstra_path(source_node, target_node1)
    print(test_path_1)
    extra_path = test_mesh2d.dualpath_dijkstra({0: test_path_0})


    test_mesh2d = mesh_2d(3, 3)
    test_mesh2d.init_nodes()
    test_mesh2d.init_links()
    test_mesh2d.dijkstra_offload()
    # test_trees = test_mesh2d.multi_root_single_neighbor_dijkstra_preserve_order_no_repeat()
    final_solution, final_cost = test_mesh2d.simulated_annealing(
        test_mesh2d.build_initial_paths(),
        max_iter=2000, 
        start_temp=10.0, 
        alpha=0.98
    )
    print("Final solution:", final_solution)
    print("Final cost:", final_cost)
    raise
    test_path = test_mesh2d.record_dijkstra_broadcast(source_node, {target_node0, target_node1})
    print(test_path)
    test_paths = find_dependencies(test_path[2], source_node)
    print(test_paths)

    test_mesh2d.init_nodes()
    test_mesh2d.init_links()
    test_mesh2d.dijkstra_offload()
    test_mesh2d.link_failure((4, 5))
    test_path = test_mesh2d.record_dijkstra_broadcast(source_node, {target_node0, target_node1})
    print("failed (4, 5)", test_path)

    test_mesh2d.init_nodes()
    test_mesh2d.init_links()
    test_mesh2d.dijkstra_offload()
    test_mesh2d.node_failure(6)
    test_path = test_mesh2d.record_dijkstra_broadcast(source_node, {target_node0, target_node1})
    print("failed 6", test_path)

    test_mesh2d = mesh_2d(5, 5)
    test_mesh2d.init_nodes()
    test_mesh2d.init_links()
    test_mesh2d.dijkstra_offload()
    test_mesh2d.xy_path(0, 7)
    print("!@@##$$%%^^&&&")
    print(test_mesh2d.xy_path(4, 11))
    test_tree = test_mesh2d.onetree(0, deque([1, 2, 3, 5, 6, 7, 8]))
    print("Tree example:")
    print(test_tree)

    test_trees = test_mesh2d.multitree({10: {11, 12, 13, 15, 16, 17, 18}, 15: {10, 11, 12, 13, 16, 17, 18}})
    print("Trees example:")
    print(test_trees)

    test_mesh2d.init_links()
    test_mesh2d.init_nodes()
    test_mesh2d.dijkstra_offload()
    test_mesh2d.node_failure(5)
    test_mesh2d.link_failure((0, 1))
    print("nodes_set:")
    print(test_mesh2d.nodes_set)
    print("links_set:")
    print(test_mesh2d.links_set)
    print("nodes_coordinate_dict:")
    print(test_mesh2d.nodes_coordinate_dict)
    print("nodes_coordinate_idx_dict:")
    print(test_mesh2d.nodes_coordinate_idx_dict)
    print("available_links:")
    print(test_mesh2d.available_links)
    print("available_neighbors:")
    print(test_mesh2d.available_neighbors)
