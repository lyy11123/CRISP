
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from buffer  import buffer


from typing import List, Dict, Tuple, Set
import numpy as np


class ddr(buffer):

    def __init__(self, ddr_id: Tuple[int], ddr_cfg: Dict[str, int]):
        super().__init__(buffer_type="ddr", buffer_id=ddr_id)

        self.ddr_id = ddr_id
        self.parse_cfg(ddr_cfg)


    def parse_cfg(self, cfg):
        self.ddr_cfg = cfg
        self.ddr_capacity = cfg["ddr_capacity"]
