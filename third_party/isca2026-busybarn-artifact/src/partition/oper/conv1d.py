
"""Partition 1D convolution operators into executable behaviors."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))
sys.path.append(file_path)


from oper_notation import oper_notation
from beha_notation import beha_notation
from data_notation import tensor_notation
from matadd import matadd


from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import itertools


class conv1d(oper_notation):

    def __init__(self,
                 oper_name: str, oper_tag: int, 
                 source_data_tags: List[int], 
                 target_data_tags: List[int], 
                 oper_split_list: List[Tuple[int]],
                 weight_dim: int, 
                 beha_tag_offset: int = 0,
                 reduction_split_list: List[Tuple[int]] = None
                 ):

        self.weight_dim = weight_dim
        if sum(oper_split_list[-1]) != weight_dim:
            raise ValueError(
                f"weight_dim {weight_dim} is not equal to the last dimension of oper_split_list {oper_split_list[-1]}"
            )
        self.bias = True if len(source_data_tags) == 3 else False

        self.reduction_split_list = reduction_split_list

        super().__init__(
            oper_name=oper_name+"_conv1d", 
            oper_tag=oper_tag,
            source_data_tags=source_data_tags,
            target_data_tags=target_data_tags,
            oper_split_list=oper_split_list, 
            beha_tag_offset=beha_tag_offset            
        )

        '''
        source_data: [dim_0, ..., dim_m, dim_k]
        weight: [dim_k, weight_dim]
        bias: [weight_dim]
        oper_split_list: [dim_0_degree, ..., dim_m_degree, dim_k_degree, weight_dim_degree]
        data_type_list: [weight_data_type, bias_data_type, partialsum_data_type, ofmap_data_type]
        '''


    def data_split(self):

        self.source_data_split_list = tuple(self.oper_split_list[:-1])
        self.source_data_used_tags_groups = self.generate_grouped_tags_bydim(
            dimensions=self.dim_degree_list, 
            group_by_dim=self.oper_dim_number-1
        )

        self.weight_split_list = tuple(self.oper_split_list[-2:])
        self.weight_used_tags_groups = self.generate_grouped_tags_bydims(
            dimensions=self.dim_degree_list, 
            group_by_dims=[i for i in range(self.oper_dim_number-2)]
        )

        self.bias_split_list = tuple(self.oper_split_list[-1:])
        self.bias_used_tags_groups = self.generate_grouped_tags_bydims(
            dimensions=self.dim_degree_list, 
            group_by_dims=[i for i in range(self.oper_dim_number-1)]
        )

        self.target_data_shape = [int(np.sum(dim_splitted)) for dim_splitted in self.oper_split_list[:-2]]+[self.weight_dim]
        self.target_oper_split_list = tuple(self.oper_split_list[:-2]+self.oper_split_list[-1:])


    def consume_source_data_and_beha(self,  
                                     data_dict: Dict[int, tensor_notation], 
                                     beha_dict: Dict[int, beha_notation]
                                     ):

        for source_data_used_tags_group in self.source_data_used_tags_groups:
            self.one_tag_group_update_data(
                source_data_tag=self.source_data_tags[0],
                source_data_split_list=self.source_data_split_list,
                source_data_used_tags_group=source_data_used_tags_group,
                data_dict=data_dict,
                beha_dict=beha_dict
            )           

        for weight_data_used_tags_group in self.weight_used_tags_groups:
            self.one_tag_group_update_data(
                source_data_tag=data_dict[self.source_data_tags[1]].data_tag,
                source_data_split_list=self.weight_split_list,
                source_data_used_tags_group=weight_data_used_tags_group,
                data_dict=data_dict,
                beha_dict=beha_dict
            )

        if self.bias:
            self.bias_data_tag = data_dict[self.source_data_tags[2]].data_tag

            if self.dim_degree_list[-2] == 1:
                for bias_data_used_tags_group in self.bias_used_tags_groups:
                    self.one_tag_group_update_data(
                        source_data_tag=self.bias_data_tag,
                        source_data_split_list=self.bias_split_list,
                        source_data_used_tags_group=bias_data_used_tags_group,
                        data_dict=data_dict,
                        beha_dict=beha_dict
                    )
            else:
                for bias_data_used_tags_group_idx, bias_data_used_tags_group in enumerate(self.bias_used_tags_groups):
                    if bias_data_used_tags_group_idx % self.dim_degree_list[-2] == 0:
                        self.one_tag_group_update_data(
                            source_data_tag=self.bias_data_tag,
                            source_data_split_list=self.bias_split_list,
                            source_data_used_tags_group=bias_data_used_tags_group,
                            data_dict=data_dict,
                            beha_dict=beha_dict
                        )
                    else:
                        continue


    def update_consumer_beha(self, 
                             data_dict: Dict[int, tensor_notation], 
                             beha_dict: Dict[int, beha_notation]
                             ):

        for beha_idx, beha_tag in enumerate(self.beha_tags):
            self.behaviors_dict[beha_tag] = beha_notation(
                beha_name=self.oper_name+"_"+str(beha_tag[-1]),
                beha_tag=beha_tag,
                beha_type="matmac" if self.bias_data_tag in self.needed_data_split_dict[beha_tag] else "matmul",
                needed_data_split_dict=self.needed_data_split_dict[beha_tag],
                needed_tag_size_dict=self.consumer_producer_dependencies_data[beha_tag]
            )
        beha_dict.update(self.behaviors_dict)


    def update_target_data(self, 
                           data_dict: Dict[int, tensor_notation], 
                           beha_dict: Dict[int, beha_notation]
                           ):

        if self.dim_degree_list[-2] == 1:
            ofmap = data_dict[self.target_data_tags[0]]
            ofmap.generated_split(
                dimension_split=self.target_oper_split_list,
                producertag_split=self.beha_tags
            )

        else:
            partialsum_used_tags_groups = self.generate_grouped_tags_bydim(
                dimensions=self.dim_degree_list,
                group_by_dim=self.oper_dim_number-2
            )
            partialsum_data_tags = []
            for partial_idx in range(self.dim_degree_list[-2]):
                partialsum_data_tag = self.target_data_tags[partial_idx]
                partialsum_data_tags.append(partialsum_data_tag)

                partial_sum = data_dict[partialsum_data_tag]
                partial_sum.generated_split(
                    dimension_split=self.target_oper_split_list,
                    producertag_split=partialsum_used_tags_groups[partial_idx]
                )

            partialsum_aggregate = matadd(
                oper_name=self.oper_name, 
                oper_tag=self.oper_tag,
                source_data_tags=partialsum_data_tags,
                target_data_tags=[self.target_data_tags[-1]],
                oper_split_list= self.reduction_split_list if self.reduction_split_list else self.target_oper_split_list,  
                beha_tag_offset=self.beha_tag_offset+self.degrees
            )            

            partialsum_aggregate.build_dependency(
                data_dict=data_dict,
                beha_dict=beha_dict
            )

            self.degrees += len(self.oper_split_list[-2])


def conv1d_data(
    data_dict: Dict[int, tensor_notation],
    oper_name: str,
    oper_split_list: List[Tuple[int]],
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    data_type_list: List[str] = None,
    bias: bool = True
):
    '''
    data_type_list: [weight, bias, psum..., ofmap]
    '''

    oper_shape_list = [sum(oper_split) for oper_split in oper_split_list]
    weight_shape = [oper_shape_list[-2], oper_shape_list[-1]]
    bias_shape = [oper_shape_list[-1]] if bias else None
    psum_number = len(oper_split_list[-2])
    psum_shape = oper_shape_list[:-2] + [oper_shape_list[-1]]

    if data_type_list is None:
        if bias:
            data_type_list = ["bf16"] * (psum_number + 3)
        else:
            data_type_list = ["bf16"] * (psum_number + 2)

    weight_data_tag = offline_data_tag
    weight_data = tensor_notation(
        data_name=oper_name+"_weight",
        data_tag=weight_data_tag,
        data_shape=weight_shape,
        data_type=data_type_list[0]
    )
    weight_data.dummy_generated_split()
    data_dict[weight_data_tag] = weight_data

    if bias:
        bias_data_tag = (offline_data_tag[0], offline_data_tag[1]+1)
        bias_data = tensor_notation(
            data_name=oper_name+"_bias",
            data_tag=bias_data_tag,
            data_shape=bias_shape,
            data_type=data_type_list[1]
        )
        bias_data.dummy_generated_split()
        data_dict[bias_data_tag] = bias_data

        next_datatype_idx = 2
        next_offline_data_tag = (offline_data_tag[0], offline_data_tag[1]+2)
    else:
        next_datatype_idx = 1
        next_offline_data_tag = (offline_data_tag[0], offline_data_tag[1]+1)

    if psum_number > 1:
        psum_data_tag_list = []
        for psum_idx in range(psum_number):
            psum_data_tag = (online_data_tag[0], online_data_tag[1]+psum_idx)
            psum_data = tensor_notation(
                data_name=oper_name+"_psum"+str(psum_idx),
                data_tag=psum_data_tag,
                data_shape=psum_shape,
                data_type=data_type_list[next_datatype_idx+psum_idx]
            )
            data_dict[psum_data_tag] = psum_data
            psum_data_tag_list.append(psum_data_tag)

        ofmap_data_tag = (online_data_tag[0], online_data_tag[1]+psum_number)
        ofmap_data = tensor_notation(
            data_name=oper_name+"_ofmap",
            data_tag=ofmap_data_tag,
            data_shape=psum_shape,
            data_type=data_type_list[-1]
        )
        data_dict[ofmap_data_tag] = ofmap_data

        next_online_data_tag = (online_data_tag[0], online_data_tag[1]+psum_number+1)
    else:
        ofmap_data_tag = online_data_tag
        ofmap_data = tensor_notation(
            data_name=oper_name+"_ofmap",
            data_tag=ofmap_data_tag,
            data_shape=psum_shape,
            data_type=data_type_list[next_datatype_idx]
        )
        data_dict[ofmap_data_tag] = ofmap_data

        next_online_data_tag = (online_data_tag[0], online_data_tag[1]+1)

    source_data_tag_list = [weight_data_tag, bias_data_tag] if bias else [weight_data_tag]
    target_data_tag_list = psum_data_tag_list + [ofmap_data_tag] if psum_number > 1 else [ofmap_data_tag]

    return source_data_tag_list, target_data_tag_list, next_online_data_tag, next_offline_data_tag


def Conv1d(
    data_dict: Dict[int, tensor_notation],
    beha_dict: Dict[int, beha_notation],
    oper_name: str,
    source_data_tags: List[int],
    oper_split_list: List[Tuple[int]],
    oper_tag: Tuple[int], 
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    weight_dim: int,
    beha_tag_offset: int = 0,
    reduction_split_list: List[Tuple[int]] = None
):

    source_data_tag_list, target_data_tag_list, next_online_data_tag, next_offline_data_tag = conv1d_data(
        data_dict=data_dict,
        oper_name=oper_name,
        oper_split_list=oper_split_list,
        online_data_tag=online_data_tag,
        offline_data_tag=offline_data_tag
    )

    conv1d_oper = conv1d(
        oper_name=oper_name, 
        oper_tag=oper_tag,
        source_data_tags=source_data_tags + source_data_tag_list, 
        target_data_tags=target_data_tag_list,
        oper_split_list=oper_split_list, 
        weight_dim=weight_dim,
        beha_tag_offset=beha_tag_offset,
        reduction_split_list=reduction_split_list
    )

    conv1d_oper.build_dependency(
        data_dict=data_dict,
        beha_dict=beha_dict
    )

    next_oper_tag = oper_tag[:-1] + (oper_tag[-1]+1,)
    next_beha_offset = beha_tag_offset + conv1d_oper.degrees

    return next_oper_tag, next_online_data_tag, next_offline_data_tag, next_beha_offset, conv1d_oper


if __name__ == "__main__":

    data_dict = {}
    beha_dict = {}

    data_dict[(0, 0)] = tensor_notation(
        data_name="data0",
        data_tag=(0, 0),
        data_shape=[1, 1024, 768],
        data_type="float32"
    )
    data_dict[(0, 0)].dummy_generated_split()


    conv1d_next_oper_tag, next_online_data_tag, next_offline_data_tag, conv1d_next_beha_offset, conv1d_oper = Conv1d(
        data_dict=data_dict,
        beha_dict=beha_dict,
        oper_name="",
        source_data_tags=[(0, 0)],
        oper_split_list=[(1,), (128, 256, 640), (128, 128, 512), (384, 384)],
        oper_tag=(5, 4),
        online_data_tag=(0, 2),
        offline_data_tag=(1, 0),
        weight_dim=768,
        beha_tag_offset=0,
        reduction_split_list=[(1,), (128, 256, 640), (384, 384)]
    )


    for data in data_dict:
        print(data_dict[data])

    print("***************************")
    for beha in beha_dict:
        print(beha_dict[beha])
