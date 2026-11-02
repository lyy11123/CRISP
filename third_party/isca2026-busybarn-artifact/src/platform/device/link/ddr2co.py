
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from link import link


from typing import List, Dict, Tuple, Set
import numpy as np


class ddr2co(link):

    def __init__(self, ddr2co_id: Tuple[Tuple[int]], ddr2co_cfg: Dict[str, int]):
        super().__init__(link_type="ddr2co", link_id=ddr2co_id)

        self.ddr2co_id = ddr2co_id
        self.parse_cfg(ddr2co_cfg)


    def parse_cfg(self, cfg):
        self.ddr2co_cfg = cfg
        self.ddr2co_latency = cfg["ddr2co_latency"]
        self.ddr2co_bandwidth = cfg["ddr2co_bandwidth"]
        self.ddr2co_timeunit = cfg["ddr2co_timeunit"]
        self.latency = cfg["ddr2co_latency"]
        self.bandwidth = cfg["ddr2co_bandwidth"]
        self.timeunit = cfg["ddr2co_timeunit"]


    def working_time(self, data_bytes: List[List[int]]):
        return int(np.ceil(np.ceil(data_bytes / self.bandwidth) * self.timeunit) + self.latency)


    def __str__(self):
        return f"{self.ddr2co_id} chiplet to chiplet link"

    def __repr__(self):
        return self.__str__()
