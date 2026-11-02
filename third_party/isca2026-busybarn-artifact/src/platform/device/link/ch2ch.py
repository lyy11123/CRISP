
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from link import link


from typing import List, Dict, Tuple, Set
import numpy as np


class ch2ch(link):

    def __init__(self, ch2ch_id: Tuple[Tuple[int]], ch2ch_cfg: Dict[str, int]):
        super().__init__(link_type="ch2ch", link_id=ch2ch_id)

        self.ch2ch_id = ch2ch_id
        self.parse_cfg(ch2ch_cfg)


    def parse_cfg(self, cfg):
        self.ch2ch_cfg = cfg
        self.ch2ch_latency = cfg["ch2ch_latency"]
        self.ch2ch_bandwidth = cfg["ch2ch_bandwidth"]
        self.ch2ch_timeunit = cfg["ch2ch_timeunit"]
        self.latency = cfg["ch2ch_latency"]
        self.bandwidth = cfg["ch2ch_bandwidth"]
        self.timeunit = cfg["ch2ch_timeunit"]


    def working_time(self, data_bytes: List[List[int]]):
        return int(np.ceil(np.ceil(data_bytes / self.bandwidth) * self.timeunit) + self.latency)


    def __str__(self):
        return f"{self.ch2ch_id} chiplet to chiplet link"

    def __repr__(self):
        return self.__str__()
