
"""Partition RMS normalization operators into staged behaviors."""

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
from elepow2 import Elepow2
from redsum import Redsum
from elemul import Elemul
from eleadd import Eleadd
from elesqrt import Elesqrt
from vecdiv import Vecdiv
from vecmac import Vecmac


from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import itertools


class rmsnorm(oper_notation):

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
            oper_name=oper_name+"_rmsnorm", 
            oper_tag=oper_tag,
            source_data_tags=source_data_tags, 
            target_data_tags=target_data_tags, 
            oper_split_list=oper_split_list, 
            beha_tag_offset=beha_tag_offset
        )

        '''
        rmsnorm operator supports tensor activation
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

        elepow2_next_oper_tag, elepow2_next_online_data_tag, elepow2_next_offline_data_tag, elepow2_next_beha_offset, elepow2_oper = Elepow2(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=[self.source_data_tags[0]],
            oper_split_list=self.oper_split_list[0:3],
            oper_tag=self.oper_tag,
            online_data_tag=self.online_data_tag,
            offline_data_tag=self.offline_data_tag,
            beha_tag_offset=self.beha_tag_offset,
        )            
        var_pow2_data_tag = elepow2_oper.target_data_tags[0]

        redsum_next_oper_tag, redsum_next_online_data_tag, redsum_next_offline_data_tag, redsum_next_beha_offset, redsum_oper = Redsum(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=[var_pow2_data_tag],
            oper_split_list=self.oper_split_list[0:3],
            oper_tag=self.oper_tag,
            online_data_tag=elepow2_next_online_data_tag,
            offline_data_tag=elepow2_next_offline_data_tag,
            reduce_dims=[2],
            beha_tag_offset=elepow2_next_beha_offset
        )
        var_sum_data_tag = redsum_oper.target_data_tags[0]

        elemul_next_oper_tag, elemul_next_online_data_tag, elemul_next_offline_data_tag, elemul_next_beha_offset, elemul_oper = Elemul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=[var_sum_data_tag],
            oper_split_list=self.oper_split_list[0:3][0:-1]+[(1,)],
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
            oper_split_list=self.oper_split_list[0:3][0:-1]+[(1,)],
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
            oper_split_list=self.oper_split_list[0:3][0:-1]+[(1,)],
            oper_tag=self.oper_tag,
            online_data_tag=eleadd_next_online_data_tag,
            offline_data_tag=eleadd_next_offline_data_tag,
            beha_tag_offset=eleadd_next_beha_offset
        )
        var_sqrt_data_tag = elesqrt_oper.target_data_tags[0]
        self.target_data_tags = [var_sqrt_data_tag]

        vecdiv_next_oper_tag, next_online_data_tag, next_offline_data_tag, vecdiv_next_beha_offset, vecdiv_oper = Vecdiv(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            source_data_tags=[self.source_data_tags[0], var_sqrt_data_tag],
            oper_split_list=self.oper_split_list[3:],
            oper_tag=self.oper_tag,
            online_data_tag=elesqrt_next_online_data_tag,
            offline_data_tag=elesqrt_next_offline_data_tag,
            vec_dims=[2],
            beha_tag_offset=elesqrt_next_beha_offset
        )
        var_div_data_tag = vecdiv_oper.target_data_tags[0]

        norm_weight_vec_dims = []
        for i in range(3):
            if i in [2]:
                continue
            else:
                norm_weight_vec_dims.append(i)
        if norm_weight_vec_dims == []:
            raise ValueError("norm_weight_vec_dims is empty")

        vecmac_next_oper_tag, next_online_data_tag, next_offline_data_tag, norm_cal_next_beha_offset, norm_cal = Vecmac(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.oper_name,
            oper_tag=self.oper_tag,
            source_data_tags=[var_div_data_tag, self.source_data_tags[-2], self.source_data_tags[-1]],
            oper_split_list=self.oper_split_list[3:],
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            vec_dims=norm_weight_vec_dims,
            beha_tag_offset=vecdiv_next_beha_offset
        )
        norm_data_tag = norm_cal.target_data_tags[0]
        self.target_data_tags = [norm_data_tag]

        self.rn_offline_data_tags = [self.source_data_tags[0], self.source_data_tags[1], self.source_data_tags[2]]

        return vecmac_next_oper_tag, next_online_data_tag, next_offline_data_tag, norm_cal_next_beha_offset


def rmsnorm_data(
    data_dict: Dict[int, tensor_notation],
    oper_name: str,
    oper_split_list: List[Tuple[int]],
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    data_type_list: List[str] = None,
    regressive: bool = False,
    regressive_data_tags: Tuple[int] = None
):

    if regressive:
        return regressive_data_tags, [], online_data_tag, offline_data_tag        
    else:
        oper_shape_list = [sum(oper_split) for oper_split in oper_split_list]
        gamma_shape = [1 for _ in oper_shape_list[-3:-1]] + [oper_shape_list[-1]]
        beta_shape = [1 for _ in oper_shape_list[-3:-1]] + [oper_shape_list[-1]]

        if data_type_list is None:
            data_type_list = ["bf16", "bf16"]

        gamma_data_tag = offline_data_tag
        gamma_data = tensor_notation(
            data_name=oper_name+"_gamma",
            data_tag=gamma_data_tag,
            data_shape=gamma_shape,
            data_type=data_type_list[0]
        )
        gamma_data.dummy_generated_split(dimension_split=[(1,) for _ in oper_shape_list[-3:-1]] + [oper_split_list[-1]])
        data_dict[gamma_data_tag] = gamma_data

        beta_data_tag = (offline_data_tag[0], offline_data_tag[1]+1)
        beta_data = tensor_notation(
            data_name=oper_name+"_beta",
            data_tag=beta_data_tag,
            data_shape=beta_shape,
            data_type=data_type_list[1]
        )
        beta_data.dummy_generated_split(dimension_split=[(1,) for _ in oper_shape_list[-3:-1]] + [oper_split_list[-1]])
        data_dict[beta_data_tag] = beta_data

        next_offline_data_tag = (beta_data_tag[0], beta_data_tag[1]+2)
        source_data_tag_list = [gamma_data_tag, beta_data_tag]

        return source_data_tag_list, [], online_data_tag, next_offline_data_tag


def Rmsnorm(
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

    source_data_tag_list, target_data_tag_list, next_online_data_tag, next_offline_data_tag = rmsnorm_data(
        data_dict=data_dict,
        oper_name=oper_name,
        oper_split_list=oper_split_list,
        online_data_tag=online_data_tag,
        offline_data_tag=offline_data_tag,
        data_type_list=None,
        regressive=regressive,
        regressive_data_tags=regressive_data_tags
    )

    norm_oper = rmsnorm(
        oper_name=oper_name,
        oper_tag=oper_tag,
        source_data_tags=source_data_tags + source_data_tag_list,
        target_data_tags=target_data_tag_list,
        oper_split_list=oper_split_list,
        beha_tag_offset=beha_tag_offset
    )

    norm_oper.online_data_tag = next_online_data_tag
    norm_oper.offline_data_tag = next_offline_data_tag

    norm_next_oper_tag, next_online_data_tag, next_offline_data_tag, norm_next_beha_offset = norm_oper.update_target_data(
        data_dict=data_dict,
        beha_dict=beha_dict
    )

    return norm_next_oper_tag, next_online_data_tag, next_offline_data_tag, norm_next_beha_offset, norm_oper


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


    rmsnorm_next_oper_tag, next_online_data_tag, next_offline_data_tag, rmsnorm_next_beha_offset, rmsnorm_oper = Rmsnorm(
        data_dict=data_dict,
        beha_dict=beha_dict,
        oper_name="",
        source_data_tags=[(0, 0)],
        oper_split_list=[
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
