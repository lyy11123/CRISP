
"""Partition the combine operator into behavior-level work units."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))


from oper_notation import oper_notation
from beha_notation import beha_notation
from data_notation import tensor_notation, tensor_slice_notation


from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import itertools


class combine(oper_notation):

    def __init__(self,
                 oper_name: str, oper_tag: int, 
                 source_data_tags: List[int], 
                 target_data_tags: List[int], 
                 oper_split_list: List[Tuple[int]],
                 combine_dims: List[int],
                 beha_tag_offset: int = 0,
                 ):

        self.combine_dims = combine_dims

        super().__init__(
            oper_name=oper_name+"_combine", 
            oper_tag=oper_tag,
            source_data_tags=source_data_tags, 
            target_data_tags=target_data_tags, 
            oper_split_list=oper_split_list, 
            beha_tag_offset=beha_tag_offset
        )

        '''
        vectorwise operator supports two source data, a tensor and a vector
        the parallel degree is decided by the tensor source data but shared by all source data: [dim0_degree, dim1_degree, ...]
        '''


    def data_split(self):
        self.vec_split_list = []
        for dim_idx in range(self.oper_dim_number):
            if dim_idx in self.combine_dims:
                self.vec_split_list.append((1,))
            else:
                self.vec_split_list.append(self.oper_split_list[dim_idx])
        self.vec_split_list = tuple(self.vec_split_list)

        self.vec_used_tags_groups = self.generate_grouped_tags_bydims(
            dimensions=self.dim_degree_list, 
            group_by_dims=self.combine_dims
        )


    def consume_source_data_and_beha(self, 
                                    data_dict: Dict[int, tensor_notation],
                                    beha_dict: Dict[int, beha_notation]
                                     ):

        for source_data_tag in self.source_data_tags:
            self.one_tag_group_update_data(
                source_data_tag=source_data_tag, 
                source_data_split_list=tuple(self.oper_split_list), 
                source_data_used_tags_group=self.beha_tags,
                data_dict=data_dict,
                beha_dict=beha_dict
            )


    def update_consumer_beha(self, 
                            data_dict: Dict[int, tensor_notation],
                            beha_dict: Dict[int, beha_notation]):
        for beha_idx, beha_tag in enumerate(self.beha_tags):
           self.behaviors_dict[beha_tag] = beha_notation(
               beha_name=self.oper_name+"_"+str(beha_tag[-1]),
               beha_tag=beha_tag,
               beha_type="combine", 
               needed_data_split_dict=self.needed_data_split_dict[beha_tag],
               needed_tag_size_dict=self.consumer_producer_dependencies_data[beha_tag]
           )
        beha_dict.update(self.behaviors_dict)


    def update_target_data(self, 
                            data_dict: Dict[int, tensor_notation],
                            beha_dict: Dict[int, beha_notation]
                            ):

        ofmap = data_dict[self.target_data_tags[0]]
        ofmap.generated_split(
            dimension_split=tuple(self.oper_split_list),
            producertag_split=self.beha_tags
        )


def combine_oper_data(
    data_dict: Dict[int, tensor_notation],
    oper_name: str,
    oper_split_list: List[Tuple[int]],
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    data_type_list: List[str] = None
):

    oper_shape_list = [sum(oper_split) for oper_split in oper_split_list]

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


def Combine(
    data_dict: Dict[int, tensor_notation],
    beha_dict: Dict[int, beha_notation],
    oper_name: str,
    source_data_tags: List[int],
    oper_split_list: List[Tuple[int]],
    oper_tag: Tuple[int], 
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    combine_dims: List[int],
    beha_tag_offset: int = 0        
):

    source_data_tag_list, target_data_tag_list, next_online_data_tag, next_offline_data_tag = combine_oper_data(
        data_dict=data_dict,
        oper_name=oper_name,
        oper_split_list=oper_split_list,
        online_data_tag=online_data_tag,
        offline_data_tag=offline_data_tag
    )

    combine_oper_instance = combine(
        oper_name=oper_name,
        oper_tag=oper_tag,
        source_data_tags=source_data_tags,
        target_data_tags=target_data_tag_list,
        oper_split_list=oper_split_list,    
        beha_tag_offset=beha_tag_offset,
        combine_dims=combine_dims
    )

    combine_oper_instance.build_dependency(
        data_dict=data_dict,
        beha_dict=beha_dict
    )

    next_oper_tag = oper_tag[:-1] + (oper_tag[-1]+1,)
    next_beha_offset = beha_tag_offset + combine_oper_instance.degrees

    return next_oper_tag, next_online_data_tag, next_offline_data_tag, next_beha_offset, combine_oper_instance


if __name__ == "__main__":

    data_dict = {}
    beha_dict = {}

    data_dict[(0, 0)] = tensor_notation(
        data_name="data0",
        data_tag=(0, 0),
        data_shape=[2, 1024, 768], 
        data_type="float32"
    )
    data_dict[(0, 0)].dummy_generated_split()

    data_dict[(0, 1)] = tensor_notation(
        data_name="data1",
        data_tag=(0, 1),
        data_shape=[1, 1024, 768],
        data_type="float32"
    )
    data_dict[(0, 1)].dummy_generated_split()   


    combine_next_oper_tag, next_online_data_tag, next_offline_data_tag, combine_next_beha_offset, combine_oper_instance = Combine(
        data_dict=data_dict,
        beha_dict=beha_dict,
        oper_name="",
        source_data_tags=[(0, 0), (0, 1)],
        oper_split_list=[(3, ), (1024, ), (384, 384)],
        oper_tag=(5, 4),
        online_data_tag=(0, 2),
        offline_data_tag=(1, 0),
        combine_dims=[0],
        beha_tag_offset=0
    )

    for data in data_dict:
        print(data_dict[data])

    print("===================================")
    for beha in beha_dict:
        print(beha_dict[beha])
