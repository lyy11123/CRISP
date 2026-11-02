
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from module import module


from typing import List, Dict, Tuple, Set
import numpy as np


class vectorunit(module):

    def __init__(self, vectorunit_id: Tuple[int], vectorunit_cfg: Dict[str, int]):
        super().__init__(module_type="vectorunit", module_id=vectorunit_id)

        self.vectorunit_id = vectorunit_id
        self.parse_cfg(vectorunit_cfg)


    def parse_cfg(self, cfg):
        self.vectorunit_cfg = cfg
        self.vectorunit_type = cfg["vectorunit_type"]
        self.vectorunit_grain = cfg["vectorunit_grain"]

        self.eleadd_complexity = cfg["eleadd_complexity"]
        self.eleexp_complexity = cfg["eleexp_complexity"]
        self.elegelu_complexity = cfg["elegelu_complexity"]
        self.elesilu_complexity = 14
        self.elerope_complexity = 8
        self.elemul_complexity = cfg["elemul_complexity"]
        self.elepow2_complexity = cfg["elepow2_complexity"]
        self.elereleu_complexity = cfg["elerelu_complexity"]
        self.elesqrt_complexity = cfg["elesqrt_complexity"]
        self.matadd_complexity = cfg["matadd_complexity"]
        self.redmax_complexity = cfg["redmax_complexity"]
        self.redsum_complexity = cfg["redsum_complexity"]
        self.vecadd_complexity = cfg["vecadd_complexity"]
        self.vecdiv_complexity = cfg["vecdiv_complexity"]
        self.vecmac_complexity = cfg["vecmac_complexity"]
        self.vecmul_complexity = cfg["vecmul_complexity"]
        self.transpose_complexity = cfg["transpose_complexity"]
        self.lookup_complexity = cfg["lookup_complexity"]
        self.dispatch_complexity = 8
        self.combine_complexity = 1


    def working_time(self, source_datashape: List[List[int]], beha_type: str, frequency: int):
        source_data_number = len(source_datashape)
        source_data0_shape = source_datashape[0]
        if source_data_number == 2:
            source_data1_shape = source_datashape[1]
        if self.vectorunit_type == "simd":
            if beha_type == "eleadd":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.eleadd_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "eleexp": 
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.eleexp_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "elegelu":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.elegelu_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "elesilu":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.elesilu_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "elerope":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.elerope_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "elemul":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.elemul_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "elepow2":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.elepow2_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "elerelu":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.elereleu_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "elesqrt":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.elesqrt_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "matadd":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.matadd_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "redmax":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.redmax_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "redsum":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.redsum_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "vecadd":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.vecadd_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "vecdiv":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.vecdiv_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "vecmac":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.vecmac_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "vecmul":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.vecmul_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "transpose":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.transpose_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "lookup":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.lookup_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "dispatch":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.dispatch_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            elif beha_type == "combine":
                inner_loop = np.ceil(np.prod(source_data0_shape) * self.combine_complexity / self.vectorunit_grain[0])
                return np.ceil(inner_loop / frequency)
            else:
                raise ValueError("Undefined behavior type", beha_type, " for vectorunit")


        else:
            raise ValueError("Undefined vectorunit type")


    def __str__(self):
        return f"{self.vectorunit_id} vectorunit built as {self.vectorunit_type}"

    def __repr__(self):
        return self.__str__()
