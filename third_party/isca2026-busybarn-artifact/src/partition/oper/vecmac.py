
"""Partition vector multiply-accumulate operators into behavior-level tasks."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../'))


from oper_notation import oper_notation
from vectorwise_oper import vectorwise_oper
from beha_notation import beha_notation
from data_notation import tensor_notation, tensor_slice_notation


from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import itertools


class vecmac(vectorwise_oper):

    def __init__(self,
                 oper_name: str, oper_tag: int, 
                 source_data_tags: List[int], 
                 target_data_tags: List[int], 
                 oper_split_list: List[Tuple[int]],
                 vec_dims: List[int],
                 beha_tag_offset: int = 0
                 ):

        super().__init__(
            oper_name=oper_name, 
            oper_tag=oper_tag,
            source_data_tags=source_data_tags,
            target_data_tags=target_data_tags,
            oper_split_list=oper_split_list,
            vec_dims=vec_dims,
            beha_tag_offset=beha_tag_offset,
            vectorwise_name="vecmac",
        )


    def consume_source_data_and_beha(self, 
                                    data_dict: Dict[int, tensor_notation],
                                    beha_dict: Dict[int, beha_notation]
                                     ):

        source_data_tag = self.source_data_tags[0]
        self.one_tag_group_update_data(
            source_data_tag=source_data_tag, 
            source_data_split_list=tuple(self.oper_split_list), 
            source_data_used_tags_group=self.beha_tags,
            data_dict=data_dict,
            beha_dict=beha_dict
        )

        vec_data_tag = self.source_data_tags[1]
        for vec_used_tags_group in self.vec_used_tags_groups:
            self.one_tag_group_update_data(
                source_data_tag=vec_data_tag, 
                source_data_split_list=self.vec_split_list, 
                source_data_used_tags_group=vec_used_tags_group,
                data_dict=data_dict,
                beha_dict=beha_dict
            )

        vec_data_tag = self.source_data_tags[2]
        for vec_used_tags_group in self.vec_used_tags_groups:
            self.one_tag_group_update_data(
                source_data_tag=vec_data_tag, 
                source_data_split_list=self.vec_split_list, 
                source_data_used_tags_group=vec_used_tags_group,
                data_dict=data_dict,
                beha_dict=beha_dict
            )


def vecmac_data(
    data_dict: Dict[int, tensor_notation],
    oper_name: str,
    oper_split_list: List[Tuple[int]],
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    data_type_list: List[str] = None
):
    """Create data for vector multiply-accumulate operation.

    Args:
        data_dict: Dictionary to store tensor data
        oper_name: Name of the operation
        oper_split_list: List of split sizes for each dimension
        online_data_tag: Tag for online data
        offline_data_tag: Tag for offline data
        data_type_list: List of data types for output tensors

    Returns:
        source_data_tags: List of source data tags
        target_data_tags: List of target data tags
        next_online_data_tag: Next available online data tag
        next_offline_data_tag: Next available offline data tag
    """
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


def Vecmac(
    data_dict: Dict[int, tensor_notation],
    beha_dict: Dict[int, beha_notation],
    oper_name: str,
    source_data_tags: List[int],
    oper_split_list: List[Tuple[int]],
    oper_tag: Tuple[int], 
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    vec_dims: List[int],
    beha_tag_offset: int = 0        
):
    """Initialize and build a vector multiply-accumulate operator.

    Args:
        data_dict: Dictionary containing tensor data
        beha_dict: Dictionary containing behavior data
        oper_name: Name of the operation
        source_data_tags: List of source data tags
        oper_split_list: List of split sizes for each dimension
        oper_tag: Tag for the operation
        online_data_tag: Tag for online data
        offline_data_tag: Tag for offline data
        vec_dims: List of dimensions for vector operation
        beha_tag_offset: Offset for behavior tags

    Returns:
        next_oper_tag: Next available operation tag
        next_online_data_tag: Next available online data tag
        next_offline_data_tag: Next available offline data tag
        next_beha_offset: Next available behavior tag offset
        vecmac_instance: The initialized vecmac operator instance
    """
    source_data_tag_list, target_data_tag_list, next_online_data_tag, next_offline_data_tag = vecmac_data(
        data_dict=data_dict,
        oper_name=oper_name,
        oper_split_list=oper_split_list,
        online_data_tag=online_data_tag,
        offline_data_tag=offline_data_tag
    )

    vecmac_instance = vecmac(
        oper_name=oper_name,
        oper_tag=oper_tag,
        source_data_tags=source_data_tags,
        target_data_tags=target_data_tag_list,
        oper_split_list=oper_split_list,    
        vec_dims=vec_dims,
        beha_tag_offset=beha_tag_offset
    )

    vecmac_instance.build_dependency(
        data_dict=data_dict,
        beha_dict=beha_dict
    )

    next_oper_tag = oper_tag[:-1] + (oper_tag[-1]+1,)
    next_beha_offset = beha_tag_offset + vecmac_instance.degrees

    return next_oper_tag, next_online_data_tag, next_offline_data_tag, next_beha_offset, vecmac_instance


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

    data_dict[(0, 1)] = tensor_notation(
        data_name="data1",
        data_tag=(0, 1),
        data_shape=[1, 1, 768],
        data_type="float32"
    )  
    data_dict[(0, 1)].dummy_generated_split()   

    data_dict[(0, 2)] = tensor_notation(
        data_name="data1",
        data_tag=(0, 2),
        data_shape=[1, 1, 768],
        data_type="float32"
    )  
    data_dict[(0, 2)].dummy_generated_split()   

    data_dict[(0, 3)] = tensor_notation(
        data_name="ofmap",
        data_tag=(0, 3),
        data_shape=[1, 1024, 768],
        data_type="float32"
    )

    test_vecmac = vecmac(
        oper_name="", 
        oper_tag=(5, 4),
        source_data_tags=[(0, 0), (0, 1), (0, 2)],
        target_data_tags=[(0, 3)],
        oper_split_list=[(1,), (128, 256, 640), (384, 384)], 
        vec_dims=[0, 1]
    )
    test_vecmac.build_dependency(
        data_dict=data_dict,
        beha_dict=beha_dict
    )

    for data in data_dict:
        print(data_dict[data])

    print("===================================")
    for beha in beha_dict:
        print(beha_dict[beha])
