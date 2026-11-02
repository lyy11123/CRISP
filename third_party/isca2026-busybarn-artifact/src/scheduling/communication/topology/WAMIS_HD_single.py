
"""WAMIS-HD single-chiplet variant: star topology where cores connect to a
shared HBM per chiplet with optional ch2ch inter-chiplet links."""

#                 Supports single and multi-chiplet configs with ch2ch inter-chiplet links
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
from ch2ch import ch2ch
from co2ddr import co2ddr
from ddr2co import ddr2co
from read_cfg import cfg_to_dict


from typing import Dict
import math


class wamis_hdc_single(planar_2d):

    def __init__(self, platform_cfg: Dict[str, int]):
        super().__init__(platform_cfg=platform_cfg)

    def init_links(self):
        self.links_set = set()
        self.links_list = []
        self.available_neighbors = {}
        self.links_dict = {}

        # No co2co links within chiplet (star topology via HBM).
        # Add ch2ch links between adjacent chiplets' boundary cores.
        for source_node_idx in self.nodes_set:
            sc = self.nodes_coordinate_dict[source_node_idx]

            # top boundary -> chiplet above
            if sc[2] == 0:
                tc = (sc[0] - 1, sc[1], self.l1_height - 1, sc[3])
                self._add_ch2ch_if_valid(source_node_idx, tc)

            # bottom boundary -> chiplet below
            if sc[2] == self.l1_height - 1:
                tc = (sc[0] + 1, sc[1], 0, sc[3])
                self._add_ch2ch_if_valid(source_node_idx, tc)

            # left boundary -> chiplet left
            if sc[3] == 0:
                tc = (sc[0], sc[1] - 1, sc[2], self.l1_width - 1)
                self._add_ch2ch_if_valid(source_node_idx, tc)

            # right boundary -> chiplet right
            if sc[3] == self.l1_width - 1:
                tc = (sc[0], sc[1] + 1, sc[2], 0)
                self._add_ch2ch_if_valid(source_node_idx, tc)

        self.links_list = list(self.links_set)

    def _add_ch2ch_if_valid(self, source_node_idx, target_coordinate):
        if target_coordinate not in self.nodes_coordinate_idx_dict:
            return
        if target_coordinate in self.failed_nodes:
            return
        target_node_idx = self.nodes_coordinate_idx_dict[target_coordinate]
        if (source_node_idx, target_node_idx) in self.failed_links:
            return
        self.links_set.add((source_node_idx, target_node_idx))
        if source_node_idx in self.available_neighbors:
            self.available_neighbors[source_node_idx].add(target_node_idx)
        else:
            self.available_neighbors[source_node_idx] = {target_node_idx}
        self.links_dict[(source_node_idx, target_node_idx)] = ch2ch(
            ch2ch_id=(source_node_idx, target_node_idx), ch2ch_cfg=self.ch2ch_cfg
        )

    def init_mem(self):
        self.l1_width_offset = self.l1_width
        self.l1_height_offset = self.l1_height
        self.ddr_dict = {}
        l1_total = self.l1_height * self.l1_width

        # One shared HBM node per chiplet, connected to all cores in that chiplet
        for n in range(self.l2_height):
            for m in range(self.l2_width):
                chiplet_idx = n * self.l2_width + m
                ddr_idx = (chiplet_idx, l1_total)

                if ddr_idx in self.failed_nodes:
                    continue

                self.nodes_set.add(ddr_idx)
                self.nodes_coordinate_dict[ddr_idx] = (n, m, -1, 0)
                self.nodes_coordinate_idx_dict[(n, m, -1, 0)] = ddr_idx

                for ddr_id in range(self.tpg_cfg["ddr_number"]):
                    ddr_location = ddr_idx + (ddr_id,)
                    ddr_device = device(device_type="ddr", device_id=ddr_location)
                    if ddr_id == 0:
                        self.node_device_list_dict[ddr_idx] = [ddr_location]
                    else:
                        self.node_device_list_dict[ddr_idx].append(ddr_location)
                    self.ddr_dict[ddr_location] = ddr_device

                self.available_neighbors[ddr_idx] = set()

                # connect every core in this chiplet to its HBM
                for core_i in range(l1_total):
                    core_idx = (chiplet_idx, core_i)
                    if core_idx not in self.nodes_set:
                        continue

                    self.links_set.add((ddr_idx, core_idx))
                    self.available_neighbors[ddr_idx].add(core_idx)
                    self.links_dict[(ddr_idx, core_idx)] = ddr2co(
                        ddr2co_id=(ddr_idx, core_idx), ddr2co_cfg=self.ddr2co_cfg
                    )

                    self.links_set.add((core_idx, ddr_idx))
                    if core_idx in self.available_neighbors:
                        self.available_neighbors[core_idx].add(ddr_idx)
                    else:
                        self.available_neighbors[core_idx] = {ddr_idx}
                    self.links_dict[(core_idx, ddr_idx)] = co2ddr(
                        co2ddr_id=(core_idx, ddr_idx), co2ddr_cfg=self.co2ddr_cfg
                    )

        self.memory_dict["ddr"] = self.ddr_dict
        self.nodes_number = len(self.nodes_set)
        self.nodes_list = list(self.nodes_set)
        self.links_list = list(self.links_set)

        self.ddr_set = set()
        for ddr_idx in self.ddr_dict:
            self.ddr_set.add(ddr_idx[:-1])


    # node_to_node_distance: inherited from net.py (dijkstra-based)
    def update_topology(self):
        self.init_nodes()
        self.init_links()
        if self.mem_enable:
            self.init_mem()
        self.init_duplicated_links()
        self.dijkstra_offload()
        self.node_to_node_distance()


if __name__ == "__main__":
    import random
    random.seed(123)

    print("=== Single chiplet (1x1, 1 core) ===")
    cfg1 = cfg_to_dict(
        os.path.join(file_path, "../../../platform/cfgs/wamis_hd_single.cfg")
    )
    net1 = wamis_hdc_single(cfg1)
    print(f"  Nodes: {sorted(net1.nodes_set)}")
    print(f"  Links: {sorted(net1.links_set)}")
    print(f"  DDR set: {net1.ddr_set}")
    print(f"  Chiplets: {net1.chiplets_set}")

    print("\n=== Multi chiplet (2x2, 1 core each) ===")
    cfg2 = cfg_to_dict(
        os.path.join(file_path, "../../../platform/cfgs/wamis_hd_tpuv5e4.cfg")
    )
    net2 = wamis_hdc_single(cfg2)
    print(f"  Nodes: {sorted(net2.nodes_set)}")
    print(f"  Links ({len(net2.links_set)}):")
    for lnk in sorted(net2.links_set):
        print(f"    {lnk}  [{net2.links_dict[lnk].link_type}]")
    print(f"  DDR set: {net2.ddr_set}")
    print(f"  Chiplets: {net2.chiplets_set}")
    # Verify all-pairs reachability
    for src in net2.nodes_set:
        for dst in net2.nodes_set:
            if src == dst:
                continue
            assert (src, dst) in net2.dijkstra_paths, f"No path {src}->{dst}"
    print("  All-pairs reachability: OK")
