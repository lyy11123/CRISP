
"""Partition layer normalization operators into staged behaviors."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))
sys.path.append(file_path)


from oper_notation import oper_notation
from beha_notation import beha_notation
from data_notation import tensor_notation, tensor_slice_notation
from redsum import Redsum
from elemul import Elemul
from var import Var
from norm import Norm


from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import itertools


class layernorm(oper_notation):

    def __init__(self,
                 oper_name: str, oper_tag: int, 
                 source_data_tags: List[int], 
                 target_data_tags: List[int],
                 oper_split_list: List[Tuple[int]],
                 beha_tag_offset: int = 0,
                 regressive: bool = False,
                 regressive_data_tags: Tuple[int] = None
                 ):

        self.regressive = regressive
        self.regressive_data_tags = regressive_data_tags

        super().__init__(
            oper_name=oper_name+"_layernorm", 
            oper_tag=oper_tag,
            source_data_tags=source_data_tags, 
            target_data_tags=target_data_tags, 
            oper_split_list=oper_split_list, 
            beha_tag_offset=beha_tag_offset
        )

        '''
        layernorm operator supports tensor activation
        the parallel degree is decided by all source_data_0 dimensions 
        source_data_0: [dim_0, dim_1, dim_2]
        oper_split_list: [
            dim_0_degree, dim_1_degree, dim_2_degree, # mean
            dim_0_degree, dim_1_degree, dim_2_degree, # var
            dim_0_degree, dim_1_degree, dim_2_degree, # norm
        ]
        data_type_list: [ifmap_data_type, mean_data_type, var_data_type, gamma_data_type, beta_data_type, medium_data-type, ofmap_data_type]
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

        redsum_next_oper_tag, next_online_data_tag, next_offline_data_tag, redsum_next_beha_offset, redsum_oper = Redsum(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=self.source_data_tags,
            oper_split_list=self.oper_split_list[0:3],
            oper_tag=self.oper_tag,
            online_data_tag=self.online_data_tag,
            offline_data_tag=self.offline_data_tag,
            reduce_dims=[2],
            beha_tag_offset=self.beha_tag_offset
        )
        mean_sum_data_tag = redsum_oper.target_data_tags[0]

        elemul_next_oper_tag, next_online_data_tag, next_offline_data_tag, elemul_next_beha_offset, elemul_oper = Elemul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=[mean_sum_data_tag],
            oper_split_list=self.oper_split_list[3:6],
            oper_tag=self.oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            beha_tag_offset=redsum_next_beha_offset
        )
        mean_data_tag = elemul_oper.target_data_tags[0]

        var_next_oper_tag, next_online_data_tag, next_offline_data_tag, var_next_beha_offset, var_oper = Var(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=[self.source_data_tags[0], mean_data_tag],
            oper_split_list=self.oper_split_list[3:6],
            oper_tag=self.oper_tag, 
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            reduce_dims=[2], 
            beha_tag_offset=elemul_next_beha_offset
        )
        var_data_tag = var_oper.target_data_tags[0]

        norm_next_oper_tag, next_online_data_tag, next_offline_data_tag, norm_next_beha_offset, norm_oper = Norm(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=[self.source_data_tags[0], mean_data_tag, var_data_tag],
            oper_split_list=self.oper_split_list[6:],
            oper_tag=self.oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            reduce_dims=[2],
            beha_tag_offset=var_next_beha_offset,
            regressive=self.regressive,
            regressive_data_tags=self.regressive_data_tags
        )
        norm_data_tag = norm_oper.target_data_tags[0]
        self.target_data_tags = [norm_data_tag]

        self.ln_offline_data_tags = norm_oper.ln_offline_data_tags

        return norm_next_oper_tag, next_online_data_tag, next_offline_data_tag, norm_next_beha_offset


def Layernorm(
    data_dict: Dict[int, tensor_notation], 
    beha_dict: Dict[int, beha_notation],
    oper_name: str, 
    source_data_tags: List[int], 
    oper_split_list: List[Tuple[int]],
    oper_tag: int,
    online_data_tag: int,
    offline_data_tag: int,
    beha_tag_offset: int = 0,
    regressive: bool = False,
    regressive_data_tags: Tuple[int] = None
):

    layernorm_oper = layernorm(
        oper_name=oper_name,
        oper_tag=oper_tag,
        source_data_tags=source_data_tags,
        target_data_tags=[],
        oper_split_list=oper_split_list,
        beha_tag_offset=beha_tag_offset,
        regressive=regressive,
        regressive_data_tags=regressive_data_tags
    )

    layernorm_oper.online_data_tag = online_data_tag
    layernorm_oper.offline_data_tag = offline_data_tag

    layernorm_next_oper_tag, next_online_data_tag, next_offline_data_tag, layernorm_next_beha_offset = layernorm_oper.update_target_data(
        data_dict=data_dict,
        beha_dict=beha_dict
    )

    return layernorm_next_oper_tag, next_online_data_tag, next_offline_data_tag, layernorm_next_beha_offset, layernorm_oper


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


    layernorm_next_oper_tag, next_online_data_tag, next_offline_data_tag, layernorm_next_beha_offset, layernorm_oper = Layernorm(
        data_dict=data_dict,
        beha_dict=beha_dict,
        oper_name="",
        source_data_tags=[(0, 0)],
        oper_split_list=[
            (8, 8), (256, 256, 512,), (768,),
            (8, 8), (256, 256, 512,), (768,),
            (8, 8), (256, 256, 512,), (512, 256)
            ], 
        oper_tag=(5, 4),
        online_data_tag=(0, 2),
        offline_data_tag=(1, 0)
    )

    for data in data_dict:
        print(data_dict[data])

    print("========================================")
    for beha in beha_dict:
        print(beha_dict[beha])
