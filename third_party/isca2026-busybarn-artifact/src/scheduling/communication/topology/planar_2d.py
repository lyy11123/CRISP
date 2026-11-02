
"""Planar 2D topology: base class for cfg-driven chiplet/core topologies with
explicit memory devices, link bandwidth, and fault injection support."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../../platform/device/'))
sys.path.append(os.path.join(file_path, '../../../platform/device/link/'))
sys.path.append(os.path.join(file_path, '../../../platform/device/module/'))
sys.path.append(os.path.join(file_path, '../../../platform/device/buffer/'))
sys.path.append(os.path.join(file_path, '../../../../utils/'))


from net import net
from device import device
from link import link
from ch2ch import ch2ch
from co2co import co2co
from module import module
from tensorcore import tensorcore
from vectorunit import vectorunit
from ddr import ddr
from sram import sram
from read_cfg import cfg_to_dict


from typing import List, Tuple, Set, Deque, Dict, Any
from collections import deque
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


class planar_2d(net):
    """Config-driven two-level topology (chiplets x cores) with hardware devices,
    fault injection, collective pattern generators, and topology visualization."""

    def __init__(self, platform_cfg: Dict[str, int]):
        super().__init__()

        self.load_cfg(platform_cfg)
        self.update_topology()


    def load_cfg(self, platform_cfg: Dict[str, int]):
        # Parse hardware config: topology dimensions, link params, failure sets, memory config
        self.original_cfg = platform_cfg

        self.frequency = platform_cfg["time_unit"]["frequency"]
        self.tpg_cfg = platform_cfg["topology"]
        self.failure_cfg = platform_cfg["failures"]
        self.tensor_cfg = platform_cfg["tensorcore_cfg"]
        self.vector_cfg = platform_cfg["vectorunit_cfg"]
        self.co2co_cfg = platform_cfg["co2co_cfg"]
        self.ch2ch_cfg = platform_cfg["ch2ch_cfg"]

        # l1 = intra-chiplet core grid, l2 = inter-chiplet grid
        self.l1_width = self.tpg_cfg["core_columns"]
        self.l1_height = self.tpg_cfg["core_rows"]
        self.l1_number = int(self.l1_width * self.l1_height)
        self.l2_width = self.tpg_cfg["chiplet_columns"]
        self.l2_height = self.tpg_cfg["chiplet_rows"]
        self.l2_number = int(self.l2_width * self.l2_height)

        self.failed_nodes = set(self.failure_cfg["failed_nodes"])
        self.failed_links = set(self.failure_cfg["failed_links"])
        self.failed_ch2ch_links = set((pair[0][0], pair[1][0]) for pair in self.failed_links if pair[0][0] != pair[1][0])

        self.l1_number -= len(self.failed_nodes)

        if "sram_cfg" in platform_cfg:
            self.sram_cfg = platform_cfg["sram_cfg"]
            self.ddr_cfg = platform_cfg["ddr_cfg"]
            self.co2ddr_cfg = platform_cfg["co2ddr_cfg"]
            self.ddr2co_cfg = platform_cfg["ddr2co_cfg"]
            self.mem_enable = True
        else:
            self.mem_enable = False


    def init_nodes(self):

        '''
        node: (l2_idx, l1_idx)
        node_coordinate: (l2_y, l2_x, l1_y, l1_x)
        '''
        self.nodes_set = set()
        self.chiplets_set = set()
        self.nodes_coordinate_dict = {}
        self.nodes_coordinate_idx_dict = {}
        self.l1_width_offset = self.l1_width
        self.l1_height_offset = self.l1_height
        self.l1_offset = int(self.l1_width * self.l1_height)
        self.chiplets_coordinate_dict = {}
        self.chiplets_coordinate_idx_dict = {}
        self.node_device_list_dict = {}
        self.tensorcore_dict = {}
        self.vectorunit_dict = {}
        self.sram_dict = {}

        for n in range(self.l2_height):
            for m in range(self.l2_width):
                chiplet_idx = n * self.l2_width + m
                self.chiplets_coordinate_dict[chiplet_idx] = (n, m)
                self.chiplets_coordinate_idx_dict[(n, m)] = chiplet_idx
                for j in range(self.l1_height):
                    for i in range(self.l1_width):
                        node_idx = (n * self.l2_width + m, j * self.l1_width + i)
                        if node_idx in self.failed_nodes:
                            continue
                        else:
                            self.nodes_set.add(node_idx)
                            self.chiplets_set.add(chiplet_idx)
                            self.nodes_coordinate_dict[node_idx] = (n, m, j, i)
                            self.nodes_coordinate_idx_dict[(n, m, j, i)] = node_idx
                            for tensor_id in range(self.tpg_cfg["tensorcore_number"]):
                                tensor_location = node_idx + (tensor_id,)
                                tensor_device = tensorcore(tensorcore_id=tensor_location, tensorcore_cfg=self.tensor_cfg)
                                if tensor_id == 0:
                                    self.node_device_list_dict[node_idx] = [tensor_location]
                                else:
                                    self.node_device_list_dict[node_idx].append(tensor_location)
                                self.tensorcore_dict[tensor_location] = tensor_device
                            for vector_id in range(self.tpg_cfg["vectorunit_number"]):
                                vector_location = node_idx + (vector_id,)
                                vector_device = vectorunit(vectorunit_id=vector_location, vectorunit_cfg=self.vector_cfg)
                                if self.tpg_cfg["tensorcore_number"] == 0:
                                    if vector_id == 0:
                                        self.node_device_list_dict[node_idx] = [vector_location]
                                    else:
                                        self.node_device_list_dict[node_idx].append(vector_location)
                                    self.vectorunit_dict[vector_location] = vector_device
                                else:
                                    self.node_device_list_dict[node_idx].append(vector_device)
                                    self.vectorunit_dict[vector_location] = vector_device
                            if self.mem_enable:
                                for sram_id in range(self.tpg_cfg["sram_number"]):
                                    sram_location = node_idx + (sram_id,)
                                    sram_device = sram(sram_id=sram_location, sram_cfg=self.sram_cfg)
                                    if self.tpg_cfg["tensorcore_number"] == 0 and self.tpg_cfg["vectorunit_number"] == 0:
                                        if sram_id == 0:
                                            self.node_device_list_dict[node_idx] = [sram_location]
                                        else:
                                            self.node_device_list_dict[node_idx].append(sram_location)
                                    else:
                                        self.node_device_list_dict[node_idx].append(sram_device)
                                    self.sram_dict[sram_location] = sram_device
        self.modules_dict = {"tensorcore": self.tensorcore_dict, "vectorunit": self.vectorunit_dict}
        if self.mem_enable:
            self.memory_dict = {"sram": self.sram_dict}

        self.nodes_number = len(self.nodes_set)
        self.nodes_list = list(self.nodes_set)
        self.chiplets_number = len(self.chiplets_set)
        self.chiplets_list = list(self.chiplets_set)


    def init_duplicated_links(self):
        # Mark all links as non-duplicated (0) unless already set by subclass
        for link_idx in self.links_set:
            if link_idx in self.duplicated_links:
                continue
            else:
                self.duplicated_links[link_idx] = 0


    def update_topology(self):
        self.init_nodes()
        self.init_links()
        self.init_duplicated_links()
        self.dijkstra_offload()
        self.node_to_node_distance()
        self.ch_to_ch_distance()


    def ch_to_ch_distance(self):
        # Compute minimum inter-chiplet hop counts from dijkstra paths
        self.ch_to_ch_hop_dict = {}
        for source_node in self.node_to_node_hop_dict:
            if source_node[0] not in self.ch_to_ch_hop_dict:
                self.ch_to_ch_hop_dict[source_node[0]] = {}
                self.ch_to_ch_hop_dict[source_node[0]][source_node[0]] = 0
            else:
                pass
            for target_node in self.node_to_node_hop_dict[source_node]:
                hop = 0
                if source_node[0] == target_node[0]:
                    continue
                else:
                    path = self.dijkstra_paths[(source_node, target_node)]
                    path_node_number = len(path) - 1
                    for link_idx in range(path_node_number):
                        if path[link_idx][0] != path[link_idx+1][0]:
                            hop += 1
                if hop > 0:
                    if target_node[0] in self.ch_to_ch_hop_dict[source_node[0]]:
                        if hop < self.ch_to_ch_hop_dict[source_node[0]][target_node[0]]:
                            self.ch_to_ch_hop_dict[source_node[0]][target_node[0]] = hop
                        else:
                            pass
                    else:
                        self.ch_to_ch_hop_dict[source_node[0]][target_node[0]] = hop
                else:
                    # skip anything unexpected
                    continue


    def draw_topology(self):

        self.draw_valid_nodes_links(self.nodes_set, self.links_set, f"{self.__class__.__name__}_topology.png")


    def draw_valid_nodes_links(self, 
                            valid_nodes_set, 
                            valid_link_set, 
                            pic_path="tlcm_topology.png",
                            rectangle_width=2, 
                            rectangle_height=2,
                            arrow_gap=1, 
                            pic_fontsize=28,
                            save_dpi=300,
                            link_labels=None):
        """
        Draw the topology for a given set of valid nodes and valid links,
        optionally labeling each link with a user-provided tag.

        Parameters
        ----------
        valid_nodes_set : set
            A set of node IDs (keys in self.nodes_coordinate_dict) that should be drawn.
        valid_link_set : set
            A set of links, where each link is a tuple (source, target) of node IDs.
        rectangle_width, rectangle_height : float
            The dimensions of each node's rectangle.
        arrow_gap : float
            The gap between nodes (and the outer margin), also used as an offset for labels.
        pic_fontsize : int
            Font size for node labels and link annotations.
        link_labels : dict or None
            A dictionary mapping (source_node, target_node) to a label string.
            Example: {(nodeA, nodeB): "12", (nodeC, nodeD): "X", ...}
            If None, no link labels are drawn.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import FancyArrowPatch

        # Precompute a color mapping for each unique label if link_labels is provided.
        label_to_color = {}
        if link_labels is not None:
            unique_labels = sorted(set(link_labels.values()))
            num_colors = len(unique_labels)
            colormap = plt.cm.hsv
            for idx, label_val in enumerate(unique_labels):
                label_to_color[label_val] = colormap(float(idx) / num_colors)

        # --- Local Helper Functions ---
        def get_node_draw_position(coordinate):
            """
            Compute the lower-left drawing position for a node given its coordinate.
            """
            chiplet_row, chiplet_col, core_row, core_col = coordinate
            chiplet_area_width  = self.l1_width  * (rectangle_width + arrow_gap) + arrow_gap
            chiplet_area_height = self.l1_height * (rectangle_height + arrow_gap) + arrow_gap
            node_x = core_col * (rectangle_width + arrow_gap) + arrow_gap
            node_y = core_row * (rectangle_height + arrow_gap) + arrow_gap
            x = chiplet_col * chiplet_area_width + node_x
            y = chiplet_row * chiplet_area_height + node_y
            return x, y

        def calculate_figure_size():
            """
            Calculate the overall figure size based on the chiplet and core grid.
            """
            chiplet_area_width  = self.l1_width  * (rectangle_width + arrow_gap) + arrow_gap
            chiplet_area_height = self.l1_height * (rectangle_height + arrow_gap) + arrow_gap
            total_width  = self.l2_width  * chiplet_area_width
            total_height = self.l2_height * chiplet_area_height
            return total_width, total_height

        def draw_node(ax, coordinate, text=None):
            """
            Draw a node as a rectangle.
            """
            x, y = get_node_draw_position(coordinate)
            if text is None:
                text = str(coordinate)
            rect = patches.Rectangle((x, y), rectangle_width, rectangle_height,
                                    edgecolor='black', facecolor='white', linewidth=2)
            ax.add_patch(rect)
            ax.text(x + rectangle_width/2, 
                    y + rectangle_height/2, 
                    text,
                    ha='center', va='center', 
                    fontsize=pic_fontsize)

        def compute_arrow_centers(source_coordinate, target_coordinate, ratio=3):
            """
            Compute the start and end center positions for an arrow between nodes.
            Returns (src_center_x, src_center_y, tgt_center_x, tgt_center_y).
            """
            x_src, y_src = get_node_draw_position(source_coordinate)
            x_tgt, y_tgt = get_node_draw_position(target_coordinate)
            if x_src == x_tgt:
                # Vertical link
                if y_src < y_tgt:
                    src_center_x = x_src + rectangle_width * 1 / ratio
                    tgt_center_x = x_tgt + rectangle_width * 1 / ratio
                    src_center_y = y_src + rectangle_height
                    tgt_center_y = y_tgt
                else:
                    src_center_x = x_src + rectangle_width * (ratio - 1) / ratio
                    tgt_center_x = x_tgt + rectangle_width * (ratio - 1) / ratio
                    src_center_y = y_src
                    tgt_center_y = y_tgt + rectangle_height
            elif y_src == y_tgt:
                # Horizontal link
                if x_src < x_tgt:
                    src_center_y = y_src + rectangle_height * 1 / ratio
                    tgt_center_y = y_tgt + rectangle_height * 1 / ratio
                    src_center_x = x_src + rectangle_width
                    tgt_center_x = x_tgt
                else:
                    src_center_y = y_src + rectangle_height * (ratio - 1) / ratio
                    tgt_center_y = y_tgt + rectangle_height * (ratio - 1) / ratio
                    src_center_x = x_src
                    tgt_center_x = x_tgt + rectangle_width
            else:
                raise ValueError("Source and target nodes must be aligned horizontally or vertically.")
            return src_center_x, src_center_y, tgt_center_x, tgt_center_y

        def draw_link(ax, source_coordinate, target_coordinate):
            """
            Draw an arrow from the source node to the target node.
            """
            same_chiplet = source_coordinate[0] == target_coordinate[0] and source_coordinate[1] == target_coordinate[1] 
            y_long_dis = True if (abs(source_coordinate[2] - target_coordinate[2]) > 1) else False
            x_long_dis = True if (abs(source_coordinate[3] - target_coordinate[3]) > 1) else False
            if same_chiplet and y_long_dis:
                if source_coordinate[2] % 2 == 0:
                    arrow_ratio = 6
                else:
                    arrow_ratio = 4
                arrow_color = "green"
                arrow_width = 2
            elif same_chiplet and x_long_dis:
                if source_coordinate[3] % 2 == 0:
                    arrow_ratio = 6
                else:
                    arrow_ratio = 4
                arrow_color = "green"
                arrow_width = 2
            else:
                arrow_ratio = 3
                arrow_color = "blue"
                arrow_width = 1
            src_center_x, src_center_y, tgt_center_x, tgt_center_y = compute_arrow_centers(source_coordinate, target_coordinate, ratio=arrow_ratio)
            if same_chiplet:
                arrow_style = dict(arrowstyle='->', linewidth=arrow_width, color=arrow_color)
            else:
                arrow_style = dict(arrowstyle='->', linewidth=4, color='black')
            ax.annotate('',
                        xy=(tgt_center_x, tgt_center_y),
                        xytext=(src_center_x, src_center_y),
                        arrowprops=arrow_style)

        def compute_label_centers(source_coordinate, target_coordinate):
            """
            Compute the start and end center positions for an arrow between nodes.
            Returns (src_center_x, src_center_y, tgt_center_x, tgt_center_y).
            """
            x_src, y_src = get_node_draw_position(source_coordinate)
            x_tgt, y_tgt = get_node_draw_position(target_coordinate)
            if x_src == x_tgt:
                # Vertical link
                if y_src < y_tgt:
                    src_center_x = x_src + rectangle_width * 4 / 7
                    tgt_center_x = x_tgt + rectangle_width * 4 / 7
                    src_center_y = y_src + rectangle_height
                    tgt_center_y = y_tgt
                else:
                    src_center_x = x_src + rectangle_width * 3 / 7
                    tgt_center_x = x_tgt + rectangle_width * 3 / 7
                    src_center_y = y_src
                    tgt_center_y = y_tgt + rectangle_height
            elif y_src == y_tgt:
                # Horizontal link
                if x_src < x_tgt:
                    src_center_y = y_src + rectangle_height * 4 / 7
                    tgt_center_y = y_tgt + rectangle_height * 4 / 7
                    src_center_x = x_src + rectangle_width
                    tgt_center_x = x_tgt
                else:
                    src_center_y = y_src + rectangle_height * 3 / 7
                    tgt_center_y = y_tgt + rectangle_height * 3 / 7
                    src_center_x = x_src
                    tgt_center_x = x_tgt + rectangle_width
            else:
                raise ValueError("Source and target nodes must be aligned horizontally or vertically.")
            return src_center_x, src_center_y, tgt_center_x, tgt_center_y

        def draw_label(ax, source_coordinate, target_coordinate, label):
            """
            Draw the label for a link near the arrow while applying an offset
            to avoid overlapping the arrow.
            """
            src_center_x, src_center_y, tgt_center_x, tgt_center_y = compute_label_centers(source_coordinate, target_coordinate)
            mid_x = 0.5 * (src_center_x + tgt_center_x)
            mid_y = 0.5 * (src_center_y + tgt_center_y)

            if src_center_x == tgt_center_x:
                if src_center_y < tgt_center_y:
                    offset = (-arrow_gap, 0)
                else:
                    offset = (arrow_gap, 0)
            elif src_center_y == tgt_center_y:
                if src_center_x < tgt_center_x:
                    offset = (0, -arrow_gap)
                else:
                    offset = (0, arrow_gap)
            else:
                offset = (0, 0)

            label_x = mid_x + offset[0]
            label_y = mid_y + offset[1]
            bbox_color = label_to_color.get(label, "yellow")
            ax.text(label_x, label_y, label,
                    ha='center', va='center',
                    fontsize=pic_fontsize-8,
                    bbox=dict(boxstyle="circle,pad=0.3", fc=bbox_color, alpha=0.3))

        def draw_curved_link(ax, source_coordinate, target_coordinate, rad, style):
            """
            Draw a curved arrow from source to target.
            rad > 0 bends one way; rad < 0 bends the opposite way.
            """
            # get the straight-line centers
            x1, y1, x2, y2 = compute_arrow_centers(source_coordinate, target_coordinate)

            # make a FancyArrowPatch with an arc connection
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                connectionstyle=f"arc3,rad={rad}",
                **style
            )
            ax.add_patch(arrow)

        # --- Main Drawing Code ---
        fig_width, fig_height = calculate_figure_size()
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Draw each valid node.
        for node in valid_nodes_set:
            if node in self.nodes_coordinate_dict:
                coordinate = self.nodes_coordinate_dict[node]
                if self.l2_height == 1 and self.l2_width == 1:
                    draw_node(ax, coordinate, text=str(node[1]))
                else:
                    draw_node(ax, coordinate, text=str(node))
            else:
                print("Warning: Node", node, "not found in nodes_coordinate_dict.")

        # Draw each valid link and its label.
        for link in valid_link_set:
            for link in valid_link_set:
                # 3-tuple = duplicated link
                if len(link) == 3:
                    src, tgt, dup_id = link
                    if src in self.nodes_coordinate_dict and tgt in self.nodes_coordinate_dict:
                        sc = self.nodes_coordinate_dict[src]
                        tc = self.nodes_coordinate_dict[tgt]
                        # choose same style logic as draw_link
                        style = dict(arrowstyle='->', linewidth=2, color='red', shrinkA=0, shrinkB=0, mutation_scale=25) \
                                if src[0]!=tgt[0] or src[1]!=tgt[1] else \
                                dict(arrowstyle='->', linewidth=1, color='green', shrinkA=0, shrinkB=0, mutation_scale=25)
                        # curvature: e.g. ±0.3 per duplicate index
                        rad = 0.5 * dup_id
                        draw_curved_link(ax, sc, tc, rad, style)
                    else:
                        print("Warning: Link", link, "has an invalid node.")
                # 2-tuple = straight link
                elif len(link) == 2:
                    source, target = link
                    if source in self.nodes_coordinate_dict and target in self.nodes_coordinate_dict:
                        sc = self.nodes_coordinate_dict[source]
                        tc = self.nodes_coordinate_dict[target]
                        draw_link(ax, sc, tc)
                        if link_labels and link in link_labels:
                            draw_label(ax, sc, tc, link_labels[link])
                    else:
                        print("Warning: Link", link, "has an invalid node.")
                else:
                    continue

        ax.set_xlim(0, fig_width)
        ax.set_ylim(0, fig_height)
        ax.set_aspect('equal', 'box')
        ax.axis('off')
        plt.savefig(pic_path, bbox_inches='tight', pad_inches=0, dpi=save_dpi)
        plt.close()


    # -- Collective communication pattern generators --

    def alltoall_pairs(self, package_size: int, chunk_number: int = 1, task_idx_offset: int = 0, ch_set: Set[Tuple[int]] = {0}, nodes_set: Set[Tuple[int]] = None):
        # Generate AllToAll comm pairs: every node sends a chunk to every other node

        comm_pairs = []
        task_idx = task_idx_offset
        alltoall_candidates = set()

        if nodes_set:
            alltoall_candidates = nodes_set
        else:
            if ch_set:
                for source_idx in self.nodes_set:
                    if source_idx[0] not in ch_set:
                        continue
                    else:
                        alltoall_candidates.add(source_idx)
            else:
                alltoall_candidates = deepcopy(self.nodes_set)

        alltoall_nodes_number = len(alltoall_candidates)
        message_size = np.ceil(package_size / alltoall_nodes_number)
        chunk_size = np.ceil(message_size / chunk_number)
        for chunk_idx in range(chunk_number):
            for source_idx in alltoall_candidates:
                for target_idx in alltoall_candidates:
                    if source_idx == target_idx:
                        continue
                    else:
                        comm_pairs.append([(1, task_idx), source_idx, {target_idx}, chunk_size])
                        task_idx += 1

        self.chunk_cost = self.current_link_loads(chunk_size, self.links_dict[self.links_list[0]])

        return comm_pairs


    def allgather_pairs(self, package_size: int, chunk_number: int = 1, task_idx_offset: int = 0, ch_set: Set[Tuple[int]] = {0}, nodes_set: Set[Tuple[int]] = None):
        # Generate AllGather comm pairs: each node multicasts its chunk to all others

        comm_pairs = []
        task_idx = task_idx_offset
        allgather_candidates = set()

        if nodes_set:
            allgather_candidates = nodes_set
        else:
            if ch_set:
                for source_idx in self.nodes_set:
                    if source_idx[0] not in ch_set:
                        continue
                    else:
                        allgather_candidates.add(source_idx)
            else:
                allgather_candidates = deepcopy(self.nodes_set)

        allgather_nodes_number = len(allgather_candidates)
        message_size = np.ceil(package_size / allgather_nodes_number)
        chunk_size = np.ceil(message_size / chunk_number)
        for chunk_idx in range(chunk_number):
            for source_idx in allgather_candidates:
                t_set = set()
                for target_idx in allgather_candidates:
                    if source_idx == target_idx:
                        continue
                    else:
                        t_set.add(target_idx)
                comm_pairs.append([(1, task_idx), source_idx, t_set, chunk_size])
                task_idx += 1

        self.chunk_cost = self.current_link_loads(chunk_size, self.links_dict[self.links_list[0]])

        return comm_pairs


    def allgather_xring_pairs(self, package_size: int, chunk_number: int = 1, task_idx_offset: int = 0, ch_set: Set[Tuple[int]] = {0}, nodes_set: Set[Tuple[int]] = None):

        comm_pairs = []
        task_idx = task_idx_offset
        allgather_candidates = set()

        if nodes_set:
            allgather_candidates = nodes_set
        else:
            if ch_set:
                for source_idx in self.nodes_set:
                    if source_idx[0] not in ch_set:
                        continue
                    else:
                        allgather_candidates.add(source_idx)
            else:
                allgather_candidates = deepcopy(self.nodes_set)

        allgather_nodes_number = len(allgather_candidates)
        message_size = np.ceil(package_size / allgather_nodes_number)
        chunk_size = np.ceil(message_size / chunk_number)

        for chunk_idx in range(chunk_number):
            for step_x in range(self.l1_width):
                for y_idx in range(self.l1_height):
                    source_y = y_idx
                    target_y = y_idx
                    for x_idx in range(self.l1_width):
                        source_x = x_idx
                        target_x = (x_idx + 1) % self.l1_width
                        source_co_idx = y_idx * self.l1_width + source_x
                        target_co_idx = target_y * self.l1_width + target_x    
                        source_idx = (0, source_co_idx)
                        target_idx = (0, target_co_idx)
                        comm_pairs.append([(1, task_idx), source_idx, {target_idx}, np.ceil(package_size / allgather_nodes_number/chunk_number)])

        self.chunk_cost = self.current_link_loads(chunk_size, self.links_dict[self.links_list[0]])

        return comm_pairs


    def allgather_xring_pairs(self, package_size: int, chunk_number: int = 1, task_idx_offset: int = 0, ch_set: Set[Tuple[int]] = {0}, nodes_set: Set[Tuple[int]] = None):

        comm_pairs = []
        task_idx = task_idx_offset
        allgather_candidates = set()

        if nodes_set:
            allgather_candidates = nodes_set
        else:
            if ch_set:
                for source_idx in self.nodes_set:
                    if source_idx[0] not in ch_set:
                        continue
                    else:
                        allgather_candidates.add(source_idx)
            else:
                allgather_candidates = deepcopy(self.nodes_set)

        allgather_nodes_number = len(allgather_candidates)
        message_size = np.ceil(package_size / allgather_nodes_number)
        chunk_size = np.ceil(message_size / chunk_number)

        for chunk_idx in range(chunk_number):
            for step_x in range(self.l1_width):
                for y_idx in range(self.l1_height):
                    source_y = y_idx
                    target_y = y_idx
                    for x_idx in range(self.l1_width):
                        source_x = x_idx
                        target_x = (x_idx + 1) % self.l1_width
                        source_co_idx = y_idx * self.l1_width + source_x
                        target_co_idx = target_y * self.l1_width + target_x    
                        source_idx = (0, source_co_idx)
                        target_idx = (0, target_co_idx)
                        comm_pairs.append([(1, task_idx), source_idx, {target_idx}, np.ceil(package_size / allgather_nodes_number/chunk_number)])

        self.chunk_cost = self.current_link_loads(chunk_size, self.links_dict[self.links_list[0]])

        return comm_pairs


    def allgather_yring_pairs(self, package_size: int, chunk_number: int = 1, task_idx_offset: int = 0, ch_set: Set[Tuple[int]] = {0}, nodes_set: Set[Tuple[int]] = None):
        # Ring-AllGather along Y dimension: each core forwards to its Y-neighbor
        comm_pairs = []
        task_idx = task_idx_offset
        allgather_candidates = set()

        if nodes_set:
            allgather_candidates = nodes_set
        else:
            if ch_set:
                for source_idx in self.nodes_set:
                    if source_idx[0] not in ch_set:
                        continue
                    else:
                        allgather_candidates.add(source_idx)
            else:
                allgather_candidates = deepcopy(self.nodes_set)

        allgather_nodes_number = len(allgather_candidates)
        message_size = np.ceil(package_size / allgather_nodes_number)
        chunk_size = np.ceil(message_size / chunk_number)

        for chunk_idx in range(chunk_number):
            for step_y in range(self.l1_height):
                for x_idx in range(self.l1_width):
                    source_x = x_idx
                    target_x = x_idx
                    for y_idx in range(self.l1_height):
                        source_y = y_idx
                        target_y = (y_idx + 1) % self.l1_height
                        source_co_idx = source_y * self.l1_width + source_x
                        target_co_idx = target_y * self.l1_width + target_x
                        source_idx = (0, source_co_idx)
                        target_idx = (0, target_co_idx)
                        comm_pairs.append([(1, task_idx), source_idx, {target_idx}, np.ceil(package_size / self.l1_height /chunk_number)])

        self.chunk_cost = self.current_link_loads(chunk_size, self.links_dict[self.links_list[0]])

        return comm_pairs
