
"""Partition variance operators into reduction and post-processing behaviors."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../'))


from oper_notation import oper_notation
from beha_notation import beha_notation
from data_notation import tensor_notation, tensor_slice_notation
from vecadd import Vecadd
from elepow2 import Elepow2
from redsum import Redsum
from elemul import Elemul
from eleadd import Eleadd
from elesqrt import Elesqrt


from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import itertools


class var(oper_notation):

    def __init__(self,
                 oper_name: str, oper_tag: int, 
                 source_data_tags: List[int], 
                 target_data_tags: List[int],
                 oper_split_list: List[Tuple[int]],
                 reduce_dims: List[int], 
                 beha_tag_offset: int = 0
                 ):

        self.reduce_dims = reduce_dims

        super().__init__(
            oper_name=oper_name+"_var", 
            oper_tag=oper_tag,
            source_data_tags=source_data_tags, 
            target_data_tags=target_data_tags, 
            oper_split_list=oper_split_list, 
            beha_tag_offset=beha_tag_offset
        )

        '''
        var_oper operator supports tensor variance calculation
        source_data_tags: [source_data_0, mean]
        the parallel degree is decided by all source_data_0 dimensions 
        source_data_0: [dim_0, ..., dim_m, dim_k]
        oper_split_list: [dim_0_degree, ..., dim_m_degree, dim_k_degree]
        data_type_list: [ofmap_data_type]
        '''


    def data_split(self):
        pass


    def consume_source_data_and_beha(self,  
                                     data_dict: Dict[int, tensor_notation], 
                                     beha_dict: Dict[int, beha_notation]
                                     ):
        pass


    def update_consumer_beha(self, 
                             data_dict: Dict[int, tensor_notation], 
                             beha_dict: Dict[int, beha_notation]
                             ):
        pass


    def update_target_data(self, 
                           data_dict: Dict[int, tensor_notation], 
                           beha_dict: Dict[int, beha_notation]                           
                           ):

        vecadd_next_oper_tag, next_online_data_tag, next_offline_data_tag, vecadd_next_beha_offset, vecadd_oper = Vecadd(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=self.source_data_tags,
            oper_split_list=self.oper_split_list,
            oper_tag=self.oper_tag,
            online_data_tag=self.online_data_tag,
            offline_data_tag=self.offline_data_tag,
            vec_dims=self.reduce_dims,
            beha_tag_offset=self.beha_tag_offset    
        )
        mean_minus_data_tag = vecadd_oper.target_data_tags[0]

        elepow2_next_oper_tag, elepow2_next_online_data_tag, elepow2_next_offline_data_tag, elepow2_next_beha_offset, elepow2_oper = Elepow2(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=[mean_minus_data_tag],
            oper_split_list=self.oper_split_list,
            oper_tag=self.oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            beha_tag_offset=vecadd_next_beha_offset,
        )            
        var_pow2_data_tag = elepow2_oper.target_data_tags[0]

        redsum_next_oper_tag, redsum_next_online_data_tag, redsum_next_offline_data_tag, redsum_next_beha_offset, redsum_oper = Redsum(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=[var_pow2_data_tag],
            oper_split_list=self.oper_split_list,
            oper_tag=self.oper_tag,
            online_data_tag=elepow2_next_online_data_tag,
            offline_data_tag=elepow2_next_offline_data_tag,
            reduce_dims=self.reduce_dims,
            beha_tag_offset=elepow2_next_beha_offset
        )
        var_sum_data_tag = redsum_oper.target_data_tags[0]

        elemul_next_oper_tag, elemul_next_online_data_tag, elemul_next_offline_data_tag, elemul_next_beha_offset, elemul_oper = Elemul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=[var_sum_data_tag],
            oper_split_list=self.oper_split_list[0:-1]+[(1,)],
            oper_tag=self.oper_tag,
            online_data_tag=redsum_next_online_data_tag,
            offline_data_tag=redsum_next_offline_data_tag,
            beha_tag_offset=redsum_next_beha_offset
        )
        var_avg_data_tag = elemul_oper.target_data_tags[0]

        eleadd_next_oper_tag, eleadd_next_online_data_tag, eleadd_next_offline_data_tag, eleadd_next_beha_offset, eleadd_oper = Eleadd(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=[var_avg_data_tag, self.source_data_tags[1]],
            oper_split_list=self.oper_split_list[0:-1]+[(1,)],
            oper_tag=self.oper_tag,
            online_data_tag=elemul_next_online_data_tag,
            offline_data_tag=elemul_next_offline_data_tag,
            beha_tag_offset=elemul_next_beha_offset
        )
        var_eps_data_tag = eleadd_oper.target_data_tags[0]

        elesqrt_next_oper_tag, elesqrt_next_online_data_tag, elesqrt_next_offline_data_tag, elesqrt_next_beha_offset, elesqrt_oper = Elesqrt(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=[var_eps_data_tag],
            oper_split_list=self.oper_split_list[0:-1]+[(1,)],
            oper_tag=self.oper_tag,
            online_data_tag=eleadd_next_online_data_tag,
            offline_data_tag=eleadd_next_offline_data_tag,
            beha_tag_offset=eleadd_next_beha_offset
        )
        var_sqrt_data_tag = elesqrt_oper.target_data_tags[0]
        self.target_data_tags = [var_sqrt_data_tag]

        return elesqrt_next_oper_tag, elesqrt_next_online_data_tag, elesqrt_next_offline_data_tag, elesqrt_next_beha_offset


def Var(
    data_dict: Dict[int, tensor_notation], 
    beha_dict: Dict[int, beha_notation],
    oper_name: str, 
    source_data_tags: List[int], 
    oper_split_list: List[Tuple[int]],
    oper_tag: int,
    online_data_tag: int,
    offline_data_tag: int,
    reduce_dims: List[int],
    beha_tag_offset: int = 0
):

    var_oper = var(
        oper_name=oper_name,
        oper_tag=oper_tag,
        source_data_tags=source_data_tags,
        target_data_tags=[],
        oper_split_list=oper_split_list,
        reduce_dims=reduce_dims,
        beha_tag_offset=beha_tag_offset
    )

    var_oper.online_data_tag = online_data_tag
    var_oper.offline_data_tag = offline_data_tag

    var_oper_next_oper_tag, next_online_data_tag, next_offline_data_tag, var_next_beha_offset = var_oper.update_target_data(
        data_dict=data_dict,
        beha_dict=beha_dict
    )

    return var_oper_next_oper_tag, next_online_data_tag, next_offline_data_tag, var_next_beha_offset, var_oper


if __name__ == "__main__":

    data_dict = {}
    beha_dict = {}

    data_dict[(0, 0)] = tensor_notation(
        data_name="data0",
        data_tag=(0, 0),
        data_shape=[16, 1024, 768], 
        data_type="float32"
    )
    data_dict[(0, 0)].dummy_generated_split()

    data_dict[(0, 1)] = tensor_notation(
        data_name="data1",
        data_tag=(0, 1),
        data_shape=[16, 1024, 1],
        data_type="float32"
    )
    data_dict[(0, 1)].dummy_generated_split()   


    var_next_oper_tag, next_online_data_tag, next_offline_data_tag, var_next_beha_offset, var_oper = Var(
        data_dict=data_dict,
        beha_dict=beha_dict,
        oper_name="",
        source_data_tags=[(0, 0), (0, 1)],
        oper_split_list=[(16,), (128, 256, 640), (768,)], 
        oper_tag=(5, 4),
        online_data_tag=(0, 2),
        offline_data_tag=(1, 0),
        reduce_dims=[2],
    )


    for data in data_dict:
        print(data_dict[data])

    print("===================================")
    for beha in beha_dict:
        print(beha_dict[beha])
