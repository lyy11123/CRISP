
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from buffer  import buffer


from typing import List, Dict, Tuple, Set
import numpy as np


class sram(buffer):

    def __init__(self, sram_id: Tuple[int], sram_cfg: Dict[str, int]):
        super().__init__(buffer_type="sram", buffer_id=sram_id)

        self.sram_id = sram_id
        self.parse_cfg(sram_cfg)


    def parse_cfg(self, cfg):
        self.sram_cfg = cfg
        self.sram_capacity = cfg["sram_capacity"]
