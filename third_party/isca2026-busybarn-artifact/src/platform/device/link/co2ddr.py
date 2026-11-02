
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from link import link


from typing import List, Dict, Tuple, Set
import numpy as np


class co2ddr(link):

    def __init__(self, co2ddr_id: Tuple[Tuple[int]], co2ddr_cfg: Dict[str, int]):
        super().__init__(link_type="co2ddr", link_id=co2ddr_id)

        self.co2ddr_id = co2ddr_id
        self.parse_cfg(co2ddr_cfg)


    def parse_cfg(self, cfg):
        self.co2ddr_cfg = cfg
        self.co2ddr_latency = cfg["co2ddr_latency"]
        self.co2ddr_bandwidth = cfg["co2ddr_bandwidth"]
        self.co2ddr_timeunit = cfg["co2ddr_timeunit"]
        self.latency = cfg["co2ddr_latency"]
        self.bandwidth = cfg["co2ddr_bandwidth"]
        self.timeunit = cfg["co2ddr_timeunit"]


    def working_time(self, data_bytes: List[List[int]]):
        return int(np.ceil(np.ceil(data_bytes / self.bandwidth) * self.timeunit) + self.latency)


    def __str__(self):
        return f"{self.co2ddr_id} chiplet to chiplet link"

    def __repr__(self):
        return self.__str__()
