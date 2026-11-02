
"""Partition transpose operators into data-movement behaviors."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../'))


from oper_notation import oper_notation
from beha_notation import beha_notation
from data_notation import tensor_notation, tensor_slice_notation


from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import itertools


class transpose(oper_notation):

    def __init__(self,
                 oper_name: str, oper_tag: int, 
                 source_data_tags: List[int], 
                 target_data_tags: List[int],
                 oper_split_list: List[Tuple[int]],
                 beha_tag_offset: int = 0,
                 concat_dim_idx: int = None
                 ):

        self.concat_dim_idx = concat_dim_idx

        super().__init__(
            oper_name=oper_name+"_transpose", 
            oper_tag=oper_tag,
            source_data_tags=source_data_tags, 
            target_data_tags=target_data_tags, 
            oper_split_list=oper_split_list, 
            beha_tag_offset=beha_tag_offset
        )

        self.beha_tags = []
        for b_idx in range(self.dim_degree_list[0]):
            for h_idx in range(self.dim_degree_list[2]):
                for s_idx in range(self.dim_degree_list[1]):
                    beha_idx = h_idx + s_idx * self.dim_degree_list[2] + b_idx * self.dim_degree_list[2] * self.dim_degree_list[1]
                    self.beha_tags.append((*self.oper_tag, int(beha_idx + beha_tag_offset)))
        self.beha_tags = tuple(self.beha_tags)


    def data_split(self):

        self.source_data_split_list = tuple(self.oper_split_list)
        self.source_data_used_tags_groups = self.generate_grouped_tags_bydim(
            dimensions=self.dim_degree_list, 
            group_by_dim=-1
        )

        target_oper_split_list = self.oper_split_list[:-2] + [self.oper_split_list[-1]] + [self.oper_split_list[-2]]
        target_transposed_split_list = [oper_split for oper_split in target_oper_split_list]
        self.target_data_split_list = tuple(target_transposed_split_list)
        self.target_data_shape = [int(np.sum(dim_splitted)) for dim_splitted in target_transposed_split_list]


    def consume_source_data_and_beha(self,  
                                     data_dict: Dict[int, tensor_notation], 
                                     beha_dict: Dict[int, beha_notation]
                                     ):

        source_data = self.source_data_tags[0]
        source_data_used_tags_group = self.source_data_used_tags_groups

        self.one_tag_group_update_data(
            source_data_tag=source_data, 
            source_data_split_list=self.source_data_split_list,
            source_data_used_tags_group=source_data_used_tags_group,
            data_dict=data_dict,
            beha_dict=beha_dict
        )


    def update_consumer_beha(self, 
                             data_dict: Dict[int, tensor_notation], 
                             beha_dict: Dict[int, beha_notation]
                             ):

        for beha_idx, beha_tag in enumerate(self.beha_tags):
            self.behaviors_dict[beha_tag] = beha_notation(
                beha_name=self.oper_name+"_"+str(beha_tag[-1]),
                beha_tag=beha_tag,
                beha_type="transpose", 
                needed_data_split_dict=self.needed_data_split_dict[beha_tag],
                needed_tag_size_dict=self.consumer_producer_dependencies_data[beha_tag]
            )
        beha_dict.update(self.behaviors_dict)


    def update_target_data(self, 
                           data_dict: Dict[int, tensor_notation], 
                           beha_dict: Dict[int, beha_notation]
                           ):

        '''
        target data
        '''
        ofmap = data_dict[self.target_data_tags[0]]
        if self.concat_dim_idx:
            ofmap.concat_split(
                concat_dim_idx=2,
                concat_shape=[sum(concat_split) for concat_split in self.target_data_split_list],
                concat_dimension_split=self.target_data_split_list,
                concat_producertag_split=self.beha_tags
            )
        else:
            ofmap.generated_split(
                dimension_split=self.target_data_split_list,
                producertag_split=self.beha_tags
            )


def transpose_data(
    data_dict: Dict[int, tensor_notation],
    oper_name: str,
    oper_split_list: List[Tuple[int]],
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    data_type_list: List[str] = None,
    concat_dim_idx: int = None
):


    if concat_dim_idx:
        return [], [], online_data_tag, offline_data_tag
    else:
        target_oper_split_list = oper_split_list[:-2] + [oper_split_list[-1]] + [oper_split_list[-2]]
        oper_shape_list = [sum(oper_split) for oper_split in target_oper_split_list]

        if data_type_list is None:
            data_type_list = ["bf16"]

        ofmap_data_tag = online_data_tag
        ofmap_data = tensor_notation(
            data_name=oper_name+"_ofmap",
            data_tag=ofmap_data_tag,
            data_shape=oper_shape_list,
            data_type=data_type_list[0]
        )
        data_dict[ofmap_data_tag] = ofmap_data

        next_online_data_tag = (online_data_tag[0], online_data_tag[1]+1)

        return [], [ofmap_data_tag], next_online_data_tag, offline_data_tag


def Transpose(
    data_dict: Dict[int, tensor_notation], 
    beha_dict: Dict[int, beha_notation],
    oper_name: str, 
    source_data_tags: List[int], 
    oper_split_list: List[Tuple[int]],
    oper_tag: int,
    online_data_tag: int,
    offline_data_tag: int, 
    beha_tag_offset: int = 0,
    concat_dim_idx: int = None,
    concat_data_tag: Tuple[int] = None
):

    source_data_tag_list, target_data_tag_list, next_online_data_tag, next_offline_data_tag = transpose_data(
        data_dict=data_dict,
        oper_name=oper_name,
        oper_split_list=oper_split_list,
        online_data_tag=online_data_tag,
        offline_data_tag=offline_data_tag,
        concat_dim_idx=concat_dim_idx
    )

    transpose_oper = transpose(
        oper_name=oper_name,
        oper_tag=oper_tag,
        source_data_tags=source_data_tags,
        target_data_tags=target_data_tag_list + [concat_data_tag] if concat_data_tag else target_data_tag_list,
        oper_split_list=oper_split_list,
        beha_tag_offset=beha_tag_offset,
        concat_dim_idx=concat_dim_idx
    )

    transpose_oper.build_dependency(
        data_dict=data_dict,
        beha_dict=beha_dict
    )

    next_oper_tag = oper_tag[:-1] + (oper_tag[-1]+1,)
    next_beha_offset = beha_tag_offset + transpose_oper.degrees

    return next_oper_tag, next_online_data_tag, next_offline_data_tag, next_beha_offset, transpose_oper


if __name__ == "__main__":

    data_dict = {}
    beha_dict = {}

    data_dict[(0, 0)] = tensor_notation(
        data_name="data0",
        data_tag=(0, 0),
        data_shape=[1, 1024, 768],
        data_type="bf16"
    )
    data_dict[(0, 0)].dummy_generated_split()

    transpose_next_oper_tag, next_online_data_tag, next_offline_data_tag, transpose_next_beha_offset, transpose_oper_instance = Transpose(
        data_dict=data_dict,
        beha_dict=beha_dict,
        oper_name="",
        source_data_tags=[(0, 0)],
        oper_split_list=[(1,), (128, 256, 640), (256, 256, 256)],
        oper_tag=(5, 4),
        online_data_tag=(0, 1),
        offline_data_tag=(1, 0),
        beha_tag_offset=0
    )

    for data in data_dict:
        print(data_dict[data])

    print("========================================")
    for beha in beha_dict:
        print(beha_dict[beha])    


    print("*********************************")

    data_dict[(0, 4)] = tensor_notation(
        data_name="data0",
        data_tag=(0, 4),
        data_shape=[1, 1, 768],
        data_type="bf16"
    )
    data_dict[(0, 4)].dummy_generated_split()

    transpose_next_oper_tag, next_online_data_tag, next_offline_data_tag, transpose_next_beha_offset, transpose_oper_instance = Transpose(
        data_dict=data_dict,
        beha_dict=beha_dict,
        oper_name="",
        source_data_tags=[(0, 4)],
        oper_split_list=[(1,), (1,), (384, 384)],
        oper_tag=(6, 4),
        online_data_tag=(0, 2),
        offline_data_tag=(1, 0),
        beha_tag_offset=0,
        concat_dim_idx=1,
        concat_data_tag=(0, 1)
    )

    for data in data_dict:
        print(data_dict[data])

    print("========================================")
    for beha in beha_dict:
        print(beha_dict[beha])        
