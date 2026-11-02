
"""Partition reduction-sum operators into behavior-level tasks."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../'))


from oper_notation import oper_notation
from reduction_oper import reduction_oper
from beha_notation import beha_notation
from data_notation import tensor_notation, tensor_slice_notation


from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import itertools


class redsum(reduction_oper):

    def __init__(self,
                 oper_name: str, oper_tag: int, 
                 source_data_tags: List[int], 
                 target_data_tags: List[int],
                 oper_split_list: List[Tuple[int]],
                 reduce_dims: List[int], 
                 beha_tag_offset: int = 0
                 ):

        super().__init__(
            oper_name=oper_name, 
            oper_tag=oper_tag,
            source_data_tags=source_data_tags,
            target_data_tags=target_data_tags,
            oper_split_list=oper_split_list, 
            reduce_dims=reduce_dims,
            beha_tag_offset=beha_tag_offset,
            reduction_name="redsum", 
        )


def redsum_data(
    data_dict: Dict[int, tensor_notation],
    oper_name: str,
    oper_split_list: List[Tuple[int]],
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    data_type_list: List[str] = None,
    reduce_dims: List[int] = None
):
    """Create data for reduction sum operation.

    Args:
        data_dict: Dictionary to store tensor data
        oper_name: Name of the operation
        oper_split_list: List of split sizes for each dimension
        online_data_tag: Tag for online data
        offline_data_tag: Tag for offline data
        data_type_list: List of data types for output tensors
        reduce_dims: List of dimensions to reduce

    Returns:
        source_data_tags: List of source data tags
        target_data_tags: List of target data tags
        next_online_data_tag: Next available online data tag
        next_offline_data_tag: Next available offline data tag
    """
    oper_shape_list = [sum(oper_split) for oper_split in oper_split_list]
    for reduce_dim in reduce_dims:
        oper_shape_list[reduce_dim] = 1

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


def Redsum(
    data_dict: Dict[int, tensor_notation],
    beha_dict: Dict[int, beha_notation],
    oper_name: str,
    source_data_tags: List[int],
    oper_split_list: List[Tuple[int]],
    oper_tag: Tuple[int], 
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    reduce_dims: List[int],
    beha_tag_offset: int = 0        
):
    """Initialize and build a reduction sum operator.

    Args:
        data_dict: Dictionary containing tensor data
        beha_dict: Dictionary containing behavior data
        oper_name: Name of the operation
        source_data_tags: List of source data tags
        oper_split_list: List of split sizes for each dimension
        oper_tag: Tag for the operation
        online_data_tag: Tag for online data
        offline_data_tag: Tag for offline data
        reduce_dims: List of dimensions to reduce
        beha_tag_offset: Offset for behavior tags

    Returns:
        next_oper_tag: Next available operation tag
        next_online_data_tag: Next available online data tag
        next_offline_data_tag: Next available offline data tag
        next_beha_offset: Next available behavior tag offset
        redsum_instance: The initialized redsum operator instance
    """
    source_data_tag_list, target_data_tag_list, next_online_data_tag, next_offline_data_tag = redsum_data(
        data_dict=data_dict,
        oper_name=oper_name,
        oper_split_list=oper_split_list,
        online_data_tag=online_data_tag,
        offline_data_tag=offline_data_tag,
        reduce_dims=reduce_dims
    )

    redsum_instance = redsum(
        oper_name=oper_name, 
        oper_tag=oper_tag,
        source_data_tags=source_data_tags,
        target_data_tags=target_data_tag_list,
        oper_split_list=oper_split_list, 
        reduce_dims=reduce_dims,
        beha_tag_offset=beha_tag_offset
    )

    redsum_instance.build_dependency(
        data_dict=data_dict,
        beha_dict=beha_dict
    )

    next_oper_tag = oper_tag[:-1] + (oper_tag[-1]+1,)
    next_beha_offset = beha_tag_offset + redsum_instance.degrees

    return next_oper_tag, next_online_data_tag, next_offline_data_tag, next_beha_offset, redsum_instance


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
        data_name="ofmap",
        data_tag=(0, 1),
        data_shape=[1, 1024, 1],
        data_type="float32"
    )

    test_redmax = redsum(
        oper_name="", 
        oper_tag=(5, 4),
        source_data_tags=[(0, 0)],
        target_data_tags=[(0, 1)],
        oper_split_list=[(1,), (128, 128, 256, 512,), (768,)], 
        reduce_dims=[2],
    )
    test_redmax.build_dependency(
        data_dict=data_dict,
        beha_dict=beha_dict
    )

    for data in data_dict:
        print(data_dict[data])

    print("===================================")
    for beha in beha_dict:
        print(beha_dict[beha])
