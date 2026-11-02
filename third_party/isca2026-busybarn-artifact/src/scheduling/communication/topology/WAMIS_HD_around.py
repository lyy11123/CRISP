
"""WAMIS-HD edge variant: DDR on every edge core and ch2ch links on all edge
cores (including corners) for maximum inter-chiplet bandwidth."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../../platform/device/'))
sys.path.append(os.path.join(file_path, '../../../platform/device/link/'))
sys.path.append(os.path.join(file_path, '../../../platform/device/module/'))
sys.path.append(os.path.join(file_path, '../../../../utils/'))


from device import device
from ch2ch import ch2ch
from co2co import co2co
from ddr2co import ddr2co
from co2ddr import co2ddr
from read_cfg import cfg_to_dict

from WAMIS_HD import wamis_hdc

from typing import Dict
import random
random.seed(123)


class wamis_hd_around(wamis_hdc):


    def __init__(self, platform_cfg: Dict[str, int]):
        super().__init__(platform_cfg=platform_cfg)


    def init_links(self):

        self.links_set = set()
        self.links_list = []
        self.available_neighbors = {}
        self.links_dict = {}

        for source_node_idx in self.nodes_set:
            source_node_coordinate = self.nodes_coordinate_dict[source_node_idx]

            # ── intra-chiplet co2co (identical to wamis_hdc) ──
            top_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1], source_node_coordinate[2] - 1, source_node_coordinate[3])
            bottom_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1], source_node_coordinate[2] + 1, source_node_coordinate[3])
            left_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1], source_node_coordinate[2], source_node_coordinate[3] - 1)
            right_node_coordinate = (source_node_coordinate[0], source_node_coordinate[1], source_node_coordinate[2], source_node_coordinate[3] + 1)

            for neighbor_coordinate in [top_node_coordinate, bottom_node_coordinate, left_node_coordinate, right_node_coordinate]:
                if neighbor_coordinate in self.nodes_coordinate_idx_dict and neighbor_coordinate not in self.failed_nodes:
                    target_node_idx = self.nodes_coordinate_idx_dict[neighbor_coordinate]
                    if (source_node_idx, target_node_idx) not in self.failed_links:
                        self.links_set.add((source_node_idx, target_node_idx))
                        if source_node_idx in self.available_neighbors:
                            self.available_neighbors[source_node_idx].add(target_node_idx)
                        else:
                            self.available_neighbors[source_node_idx] = {target_node_idx}
                        self.links_dict[(source_node_idx, target_node_idx)] = co2co(co2co_id=(source_node_idx, target_node_idx), co2co_cfg=self.co2co_cfg)

            # ── inter-chiplet ch2ch (ALL edge cores, no corner exclusion) ──
            # top row -> chiplet above
            if source_node_coordinate[2] == 0:
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
            # bottom row -> chiplet below
            if source_node_coordinate[2] == self.l1_height - 1:
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
            # left column -> chiplet to the left
            if source_node_coordinate[3] == 0:
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
            # right column -> chiplet to the right
            if source_node_coordinate[3] == self.l1_width - 1:
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

        # Build DDR coordinates for every edge core
        # Each entry: (ddr_row, ddr_col, attached_core_row, attached_core_col)
        edge_entries = []
        # Top edge: DDR at (-1, x), attached to core (0, x)
        for x in range(self.l1_width):
            edge_entries.append((-1, x, 0, x))
        # Bottom edge: DDR at (l1_height, x), attached to core (l1_height-1, x)
        for x in range(self.l1_width):
            edge_entries.append((self.l1_height, x, self.l1_height - 1, x))
        # Left edge: DDR at (y, -1), attached to core (y, 0)
        for y in range(self.l1_height):
            edge_entries.append((y, -1, y, 0))
        # Right edge: DDR at (y, l1_width), attached to core (y, l1_width-1)
        for y in range(self.l1_height):
            edge_entries.append((y, self.l1_width, y, self.l1_width - 1))

        for n in range(self.l2_height):
            for m in range(self.l2_width):
                for edge_idx, (ddr_r, ddr_c, core_r, core_c) in enumerate(edge_entries):
                    ddr_idx = (n * self.l2_width + m, int(self.l1_height * self.l1_width + edge_idx))
                    if ddr_idx in self.failed_nodes:
                        continue

                    self.nodes_set.add(ddr_idx)
                    self.nodes_coordinate_dict[ddr_idx] = (n, m, ddr_r, ddr_c)
                    self.nodes_coordinate_idx_dict[(n, m, ddr_r, ddr_c)] = ddr_idx

                    for ddr_id in range(self.tpg_cfg["ddr_number"]):
                        ddr_location = ddr_idx + (ddr_id,)
                        ddr_device = device(device_type="ddr", device_id=ddr_location)
                        if ddr_id == 0:
                            self.node_device_list_dict[ddr_idx] = [ddr_location]
                        else:
                            self.node_device_list_dict[ddr_idx].append(ddr_location)
                        self.ddr_dict[ddr_location] = ddr_device

                    # Find the attached core node
                    core_node_coordinate = (n, m, core_r, core_c)
                    if core_node_coordinate in self.nodes_coordinate_idx_dict:
                        core_node_idx = self.nodes_coordinate_idx_dict[core_node_coordinate]
                    else:
                        raise ValueError(f"Cannot find edge core coordinate {core_node_coordinate} in nodes_coordinate_idx_dict.")

                    # ddr -> core link
                    if (ddr_idx, core_node_idx) in self.failed_links:
                        raise ValueError(f"Failed link between ddr {ddr_idx} and edge core {core_node_idx}.")
                    self.links_set.add((ddr_idx, core_node_idx))
                    if ddr_idx in self.available_neighbors:
                        self.available_neighbors[ddr_idx].add(core_node_idx)
                    else:
                        self.available_neighbors[ddr_idx] = {core_node_idx}
                    self.links_dict[(ddr_idx, core_node_idx)] = ddr2co(ddr2co_id=(ddr_idx, core_node_idx), ddr2co_cfg=self.ddr2co_cfg)

                    # core -> ddr link
                    if (core_node_idx, ddr_idx) in self.failed_links:
                        raise ValueError(f"Failed link between edge core {core_node_idx} and ddr {ddr_idx}.")
                    self.links_set.add((core_node_idx, ddr_idx))
                    self.available_neighbors[core_node_idx].add(ddr_idx)
                    self.links_dict[(core_node_idx, ddr_idx)] = co2ddr(co2ddr_id=(core_node_idx, ddr_idx), co2ddr_cfg=self.co2ddr_cfg)

        self.memory_dict["ddr"] = self.ddr_dict
        self.nodes_number = len(self.nodes_set)
        self.nodes_list = list(self.nodes_set)
        self.links_list = list(self.links_set)

        self.ddr_set = set()
        for ddr_idx in self.ddr_dict:
            self.ddr_set.add(ddr_idx[:-1])


if __name__ == "__main__":

    random.seed(123)
    hardware_cfg = cfg_to_dict(os.path.join(file_path, "../../../platform/cfgs/wamis_hd.cfg"))
    network = wamis_hd_around(hardware_cfg)

    print(f"Nodes: {network.nodes_number}")
    print(f"Links: {len(network.links_set)}")
    print(f"DDR nodes: {len(network.ddr_set)}")
    print(f"Chiplets: {network.chiplets_number}")

    # Count link types
    co2co_count = sum(1 for k in network.links_dict if isinstance(network.links_dict[k], co2co))
    ch2ch_count = sum(1 for k in network.links_dict if isinstance(network.links_dict[k], ch2ch))
    ddr2co_count = sum(1 for k in network.links_dict if isinstance(network.links_dict[k], ddr2co))
    co2ddr_count = sum(1 for k in network.links_dict if isinstance(network.links_dict[k], co2ddr))
    print(f"co2co links: {co2co_count}")
    print(f"ch2ch links: {ch2ch_count}")
    print(f"ddr2co links: {ddr2co_count}")
    print(f"co2ddr links: {co2ddr_count}")
