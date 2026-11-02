
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from link import link


from typing import List, Dict, Tuple, Set
import numpy as np


class co2co(link):

    def __init__(self, co2co_id: Tuple[Tuple[int]], co2co_cfg: Dict[str, int]):
        super().__init__(link_type="co2co", link_id=co2co_id)

        self.co2co_id = co2co_id
        self.parse_cfg(co2co_cfg)


    def parse_cfg(self, cfg):
        self.co2co_cfg = cfg
        self.co2co_latency = cfg["co2co_latency"]
        self.co2co_bandwidth = cfg["co2co_bandwidth"]
        self.co2co_timeunit = cfg["co2co_timeunit"]
        self.latency = cfg["co2co_latency"]
        self.bandwidth = cfg["co2co_bandwidth"]
        self.timeunit = cfg["co2co_timeunit"]


    def working_time(self, data_bytes: List[List[int]]):
        return int(np.ceil(np.ceil(data_bytes / self.bandwidth) * self.timeunit) + self.latency)


    def __str__(self):
        return f"{self.co2co_id} core to core link"

    def __repr__(self):
        return self.__str__()
