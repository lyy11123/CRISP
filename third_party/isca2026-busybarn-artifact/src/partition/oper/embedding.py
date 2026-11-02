
"""Partition embedding lookup operators into executable behaviors."""

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


class embedding(oper_notation):

    def __init__(self,
                 oper_name: str, oper_tag: int, 
                 source_data_tags: List[int], 
                 target_data_tags: List[int], 
                 oper_split_list: List[Tuple[int]],
                 hidden_dim: int,
                 beha_tag_offset: int = 0
                ):

        self.hidden_dim = hidden_dim

        super().__init__(
            oper_name=oper_name+"_embedding", 
            oper_tag=oper_tag,
            source_data_tags=source_data_tags,
            target_data_tags=target_data_tags,
            oper_split_list=oper_split_list, 
            beha_tag_offset=beha_tag_offset            
        )        

        '''
        source_data_tags[source_data]
        source_data: [dim_0, ..., dim_m, 1]
        oper_split_list: [dim_0_degree, ..., dim_m_degree, hidden_dim]
        NOTE: hidden_dim is not partitioned
        '''


    def data_split(self):

        self.source_data_split_list = tuple(self.oper_split_list[:-1] + [(1,)])
        self.source_data_used_tags_groups = self.generate_grouped_tags_bydim(
            dimensions=self.dim_degree_list, 
            group_by_dim=-1
        ) 


    def consume_source_data_and_beha(self, 
                                    data_dict: Dict[int, tensor_notation],
                                    beha_dict: Dict[int, beha_notation]
                                     ):

        source_data_tag = self.source_data_tags[0]
        self.one_tag_group_update_data(
            source_data_tag=source_data_tag, 
            source_data_split_list=self.source_data_split_list, 
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
               beha_type="lookup", 
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
            dimension_split=self.oper_split_list,
            producertag_split=self.beha_tags
        )


def embedding_data(
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


def Embedding(
    data_dict: Dict[int, tensor_notation],
    beha_dict: Dict[int, beha_notation],
    oper_name: str,
    source_data_tags: List[int],
    oper_split_list: List[Tuple[int]],
    oper_tag: Tuple[int],
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    hidden_dim: int,
    beha_tag_offset: int = 0
):
    """Initialize and build an embedding operator.

    Args:
        data_dict: Dictionary containing tensor data
        beha_dict: Dictionary containing behavior data
        oper_name: Name of the operation
        source_data_tags: List of source data tags
        oper_split_list: List of split sizes for each dimension
        oper_tag: Tag for the operation
        online_data_tag: Tag for online data
        offline_data_tag: Tag for offline data
        hidden_dim: Hidden dimension size
        beha_tag_offset: Offset for behavior tags

    Returns:
        next_oper_tag: Next available operation tag
        next_online_data_tag: Next available online data tag
        next_offline_data_tag: Next available offline data tag
        next_beha_offset: Next available behavior tag offset
        embedding_instance: The initialized embedding operator instance
    """
    source_data_tag_list, target_data_tag_list, next_online_data_tag, next_offline_data_tag = embedding_data(
        data_dict=data_dict,
        oper_name=oper_name,
        oper_split_list=oper_split_list,
        online_data_tag=online_data_tag,
        offline_data_tag=offline_data_tag
    )

    embedding_instance = embedding(
        oper_name=oper_name,
        oper_tag=oper_tag,
        source_data_tags=source_data_tags,
        target_data_tags=target_data_tag_list,
        oper_split_list=oper_split_list,
        hidden_dim=hidden_dim,
        beha_tag_offset=beha_tag_offset
    )

    embedding_instance.build_dependency(
        data_dict=data_dict,
        beha_dict=beha_dict
    )

    next_oper_tag = oper_tag[:-1] + (oper_tag[-1]+1,)
    next_beha_offset = beha_tag_offset + embedding_instance.degrees

    return next_oper_tag, next_online_data_tag, next_offline_data_tag, next_beha_offset, embedding_instance


if __name__ == "__main__":

    data_dict = {}
    beha_dict = {}

    data_dict[(0, 0)] = tensor_notation(
        data_name="input_data",
        data_tag=(0, 0),
        data_shape=[16, 1024, 1],
        data_type="int32"
    )
    data_dict[(0, 0)].dummy_generated_split()

    embedding_next_oper_tag, next_online_data_tag, next_offline_data_tag, embedding_next_beha_offset, embedding_oper = Embedding(
        data_dict=data_dict,
        beha_dict=beha_dict,
        oper_name="embedding",
        source_data_tags=[(0, 0)],
        oper_split_list=[(8, 8), (128, 256, 640), (768,)],
        oper_tag=(5, 4),
        online_data_tag=(0, 1),
        offline_data_tag=(1, 0),
        hidden_dim=768,
        beha_tag_offset=0
    )

    print("Data Tensors:")
    for data in data_dict:
        print(data_dict[data])

    print("\n========================================")

    print("Behavior Tensors:")
    for beha in beha_dict:
        print(beha_dict[beha])
