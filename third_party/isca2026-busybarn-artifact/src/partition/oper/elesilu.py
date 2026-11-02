
"""Partition SiLU-style elementwise activation operators."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../'))


from oper_notation import oper_notation
from elementwise_oper import elementwise_oper
from beha_notation import beha_notation
from data_notation import tensor_notation, tensor_slice_notation


from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import itertools


class elesilu(elementwise_oper):

    def __init__(self,
                 oper_name: str, oper_tag: int, 
                 source_data_tags: List[int], 
                 target_data_tags: List[int],
                 oper_split_list: List[Tuple[int]],
                 beha_tag_offset: int = 0
                 ):

        super().__init__(
            oper_name=oper_name, 
            oper_tag=oper_tag,
            source_data_tags=source_data_tags, 
            target_data_tags=target_data_tags, 
            oper_split_list=oper_split_list, 
            beha_tag_offset=beha_tag_offset,
            elementwise_name="elesilu", 
        )


def elesilu_data(
    data_dict: Dict[int, tensor_notation],
    oper_name: str,
    oper_split_list: List[Tuple[int]],
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    data_type_list: List[str] = None
):
    """Create data for element-wise GELU operation.

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


def Elesilu(
    data_dict: Dict[int, tensor_notation],
    beha_dict: Dict[int, beha_notation],
    oper_name: str,
    source_data_tags: List[int],
    oper_split_list: List[Tuple[int]],
    oper_tag: Tuple[int], 
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    beha_tag_offset: int = 0
):
    """Initialize and build an element-wise GELU operator.

    Args:
        data_dict: Dictionary containing tensor data
        beha_dict: Dictionary containing behavior data
        oper_name: Name of the operation
        source_data_tags: List of source data tags
        oper_split_list: List of split sizes for each dimension
        oper_tag: Tag for the operation
        online_data_tag: Tag for online data
        offline_data_tag: Tag for offline data
        beha_tag_offset: Offset for behavior tags

    Returns:
        next_oper_tag: Next available operation tag
        next_online_data_tag: Next available online data tag
        next_offline_data_tag: Next available offline data tag
        next_beha_offset: Next available behavior tag offset
        elesilu_instance: The initialized elesilu operator instance
    """
    source_data_tag_list, target_data_tag_list, next_online_data_tag, next_offline_data_tag = elesilu_data(
        data_dict=data_dict,
        oper_name=oper_name,
        oper_split_list=oper_split_list,
        online_data_tag=online_data_tag,
        offline_data_tag=offline_data_tag
    )

    elesilu_instance = elesilu(
        oper_name=oper_name,
        oper_tag=oper_tag,
        source_data_tags=source_data_tags,
        target_data_tags=target_data_tag_list,
        oper_split_list=oper_split_list,    
        beha_tag_offset=beha_tag_offset
    )

    elesilu_instance.build_dependency(
        data_dict=data_dict,
        beha_dict=beha_dict
    )

    next_oper_tag = oper_tag[:-1] + (oper_tag[-1]+1,)
    next_beha_offset = beha_tag_offset + elesilu_instance.degrees

    return next_oper_tag, next_online_data_tag, next_offline_data_tag, next_beha_offset, elesilu_instance
