
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from module import module


from typing import List, Dict, Tuple, Set
import numpy as np


class tensorcore(module):

    def __init__(self, tensorcore_id: Tuple[int], tensorcore_cfg: Dict[str, int]):
        super().__init__(module_type="tensorcore", module_id=tensorcore_id)

        self.tensorcore_id = tensorcore_id
        self.utilization_factor = 2
        self.parse_cfg(tensorcore_cfg)


    def parse_cfg(self, cfg):
        self.tensorcore_cfg = cfg
        self.tensorcore_type = cfg["tensorcore_type"]
        self.tensorcore_grain = cfg["tensorcore_grain"]


    def working_time(self, source_datashape: List[List[int]], beha_type: str, frequency: int):
        source_data_number = len(source_datashape)
        source_data0_shape = source_datashape[0]
        source_data1_shape = source_datashape[1]
        if source_data_number == 3:
            source_data2_shape = source_datashape[2]        
        if self.tensorcore_type == "multree":
            if beha_type == "matmul" or beha_type == "matmac":
                inner_loop = np.ceil(source_data0_shape[-1] / self.tensorcore_grain[0])
                outer_loop = np.ceil(np.prod(source_data0_shape[:-1] + [source_data1_shape[-1]]) / self.tensorcore_grain[1])
                return np.ceil(inner_loop * outer_loop / frequency * self.utilization_factor)
            else:
                raise ValueError("Undefined behavior type for tensorcore")
        else:
            raise ValueError("Undefined tensorcore type")


    def __str__(self):
        return f"{self.tensorcore_id} tensorcore built as {self.tensorcore_type}"

    def __repr__(self):
        return self.__str__()
