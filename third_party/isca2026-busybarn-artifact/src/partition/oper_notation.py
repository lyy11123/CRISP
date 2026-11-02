
"""Operator notation: wraps beha_notation and data_notation to represent a
complete operator (e.g., matmul, layernorm) with its split strategy and
expansion into per-core behaviors."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from beha_notation import beha_notation
from data_notation import tensor_notation, tensor_slice_notation


from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import itertools


class oper_notation(object):
    # Abstract base for an operator (e.g., matmul, layernorm).
    # Expands a split strategy into concrete beha_notation instances -- one per tile.
    # Subclasses implement data_split, consume_source_data_and_beha, update_consumer_beha,
    # and update_target_data.

    def __init__(self,
                 oper_name: str, oper_tag: Tuple[int],
                 source_data_tags: List[int],
                 target_data_tags: List[int],
                 oper_split_list: List[Tuple[int]],
                 beha_tag_offset: int = 0
                 ):

        self.oper_name = oper_name
        self.oper_tag = oper_tag
        self.source_data_tags = source_data_tags
        self.target_data_tags = target_data_tags
        self.oper_split_list = oper_split_list
        self.beha_tag_offset = beha_tag_offset

        self.source_data_number = len(source_data_tags)
        self.oper_dim_number = len(oper_split_list)
        # Cartesian product of per-dim splits gives all split combinations
        self.oper_split_degree_list = list(itertools.product(*oper_split_list))
        self.dim_degree_list = [len(dim_degree) for dim_degree in oper_split_list]
        self.degrees = int(np.prod(self.dim_degree_list))  # total number of behaviors
        self.beha_tags = tuple((*self.oper_tag, int(beha_tag + beha_tag_offset)) for beha_tag in range(self.degrees))
        self.needed_data_split_dict = {(*self.oper_tag, int(beha_tag + beha_tag_offset)): {} for beha_tag in range(self.degrees)}
        self.consumer_producer_dependencies_data = {}  # {consumer_tag: {producer_tag: bytes}}
        self.behaviors_dict = {}

        self.data_split()


    def one_tag_group_update_data(self,
        source_data_tag: Tuple[int],
        source_data_split_list: Tuple[Tuple[int]],
        source_data_used_tags_group: Tuple[Tuple[Tuple[int]]],
        data_dict: Dict[int, tensor_notation],
        beha_dict: Dict[int, beha_notation],
        ):
        # Wire one source tensor's consumer splits to this operator's behaviors,
        # updating dependency dicts and linking producer behaviors via add_consumer.
        consumer_producer_dependencies_data, consumer_producer_dependencies_splits = data_dict[source_data_tag].used_split(
            dimension_split=source_data_split_list, 
            consumertag_split=source_data_used_tags_group
        ) 

        if self.consumer_producer_dependencies_data == {}:
            self.consumer_producer_dependencies_data = consumer_producer_dependencies_data
        else:
            self.consumer_producer_dependencies_data = {**self.consumer_producer_dependencies_data, **consumer_producer_dependencies_data}

        for beha_tag in data_dict[source_data_tag].used_tag_splitted_dict:
            if beha_tag in self.needed_data_split_dict:
                if source_data_tag in self.needed_data_split_dict[beha_tag]: 
                    self.needed_data_split_dict[beha_tag][data_dict[source_data_tag].data_tag] |= data_dict[data_dict[source_data_tag].data_tag].used_tag_splitted_dict[beha_tag]
                else:
                    self.needed_data_split_dict[beha_tag][data_dict[source_data_tag].data_tag] = data_dict[data_dict[source_data_tag].data_tag].used_tag_splitted_dict[beha_tag]
                if beha_tag in consumer_producer_dependencies_data:
                    for producer_tag in consumer_producer_dependencies_data[beha_tag]:
                        if producer_tag == (-1, ):
                            continue
                        else:
                            beha_dict[producer_tag].add_consumer({beha_tag: [data_dict[source_data_tag].data_tag, consumer_producer_dependencies_splits[beha_tag][producer_tag]]})
                else:
                    pass


    def data_split(self):
        raise NotImplementedError("Self-defined operator must implement data_split method")


    def generate_grouped_tags_bydim(
            self,
            dimensions: List[int],
            group_by_dim: int,
            slice_beha_offset: int = 0
        ):
        # Group behavior tags by one dimension -- used for collective comm patterns
        if group_by_dim > 0:
            strides = [1] * len(dimensions)
            for i in range(len(dimensions) - 2, -1, -1):
                strides[i] = strides[i + 1] * dimensions[i + 1]

            groups = [[] for _ in range(dimensions[group_by_dim])]
            for coord in itertools.product(*[range(d) for d in dimensions]):
                groups[coord[group_by_dim]].append((*self.oper_tag, sum(c * s for c, s in zip(coord, strides)) + slice_beha_offset + self.beha_tag_offset))

            groups_tuple = []
            for group in groups:
                groups_tuple.append(tuple(group))
        else:
            groups_tuple = tuple(self.oper_tag + (i + slice_beha_offset + self.beha_tag_offset,) for i in range(self.degrees))

        return groups_tuple

    def generate_grouped_tags_bydims(
            self,
            dimensions: List[int],
            group_by_dims: List[int],
            slice_beha_offset: int = 0
        ):
        # Group behavior tags by multiple dimensions (multi-dim collective patterns)

        strides = [1] * len(dimensions)
        for i in range(len(dimensions) - 2, -1, -1):
            strides[i] = strides[i + 1] * dimensions[i + 1]

        group_strides = [1] * len(group_by_dims)
        for i in range(len(group_by_dims) - 2, -1, -1):
            group_strides[i] = group_strides[i + 1] * dimensions[group_by_dims[i + 1]]

        group_dimensions = [dimensions[i] for i in group_by_dims]
        group_number = int(np.prod(group_dimensions))
        groups = [[] for _ in range(group_number)]

        for coord in itertools.product(*[range(d) for d in dimensions]):
            group_idx = 0
            for i, dim in enumerate(group_by_dims):
                group_idx += coord[dim] * group_strides[i]
            groups[group_idx].append((*self.oper_tag, sum(c * s for c, s in zip(coord, strides)) + slice_beha_offset + self.beha_tag_offset))

        return groups


    def build_dependency(self,
                         data_dict: Dict[int, tensor_notation],
                         beha_dict: Dict[int, beha_notation]
                         ):
        # Three-phase dependency construction: consume inputs, create behaviors, produce outputs
        self.consume_source_data_and_beha(data_dict=data_dict, beha_dict=beha_dict)
        self.update_consumer_beha(data_dict=data_dict, beha_dict=beha_dict)
        return self.update_target_data(data_dict=data_dict, beha_dict=beha_dict)

    def consume_source_data_and_beha(self, data_dict: Dict[int, tensor_notation], beha_dict: Dict[int, beha_notation]):
        raise NotImplementedError("Self-defined operator must implement consume_source_data_and_beha method")           

    def update_consumer_beha(self, data_dict: Dict[int, tensor_notation], beha_dict: Dict[int, beha_notation]):
        raise NotImplementedError("Self-defined operator must implement update_consumer_beha method")

    def update_target_data(self, data_dict: Dict[int, tensor_notation], beha_dict: Dict[int, beha_notation]):
        raise NotImplementedError("Self-defined operator must implement update_target_data method")


    def __str__(self):
        return f"{self.oper_name} {self.oper_tag} Operation with {self.oper_split_list} Parallel Degree"

    def __repr__(self):
        return self.__str__()
