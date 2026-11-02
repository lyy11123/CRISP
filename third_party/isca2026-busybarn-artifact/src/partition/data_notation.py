
"""LR data notation: tensor_notation and tensor_slice_notation classes that
describe tensors and their slices across parallelism strategies, tracking
producer/consumer relationships and split-to-core location mappings."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))


from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import itertools
from copy import deepcopy


# Lookup: data type name -> bytes per element
type_bytes = {
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float8": 1,
    "fp8": 1,
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "fp16": 2,
    "float32": 4,
    "fp32": 4,
    "float64": 8,
    "fp64": 8,
}


def calculate_split_indices(original, split, offset=None):
    """
    Calculate the indices for each segment based on the original list and its split structure.

    :param original: List of original segment sizes.
    :param split: List of split sizes for each segment.
    :return: A list of lists of indices for each split segment.
    """
    if offset:
        pass
    else:
        offset = [0] * len(original)

    indices = []
    for original_segment, split_segment, offset_segment in zip(original, list(split), offset):
        segment_indices = []
        start_idx = 0
        for part in split_segment:
            end_idx = start_idx + part - 1
            segment_indices.append((start_idx + offset_segment, end_idx + offset_segment))
            start_idx = end_idx + 1
        indices.append(segment_indices)
    combined_indices = tuple(itertools.product(*tuple(indices)))
    return (tuple(combination) for combination in combined_indices)


def calculate_overlap(index_tuple1, index_tuple2):
    # Compute N-dim intersection volume and per-dim overlap intervals between two index ranges
    if len(index_tuple1) != len(index_tuple2):
        raise ValueError("The two tuples must have the same length.")

    total_overlap = 1
    overlap_intervals = []

    for (start1, end1), (start2, end2) in zip(index_tuple1, index_tuple2):
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        overlap_length = max(0, overlap_end - overlap_start + 1)

        if overlap_length > 0:
            overlap_intervals.append((overlap_start, overlap_end))
        else:
            overlap_intervals.append(())

        total_overlap *= overlap_length

    return total_overlap, tuple(overlap_intervals)


def subtract_nd_intervals(rect1, rect2, dim=0, current=[]):
    # Recursively subtract N-dim rect2 from rect1, returning remaining sub-rectangles
    if dim == len(rect1):
        if tuple(current) == rect2:
            return []
        else:
            return [tuple(current)]

    (a1, b1) = rect1[dim]
    (a2, b2) = rect2[dim]
    result = []

    if a2 > a1:
        left_part = current + [(a1, a2 - 1)]
        result.extend(subtract_nd_intervals(rect1, rect2, dim + 1, left_part))

    if b2 < b1:
        right_part = current + [(b2 + 1, b1)]
        result.extend(subtract_nd_intervals(rect1, rect2, dim + 1, right_part))

    if a2 <= b1 and b2 >= a1:
        overlap_part = current + [(max(a1, a2), min(b1, b2))]
        result.extend(subtract_nd_intervals(rect1, rect2, dim + 1, overlap_part))

    return result


class tensor_notation(object):
    # Represents a named tensor in the LR notation system.
    # Tracks how it is split across producers and consumers for parallelism.

    def __init__(
        self,
        data_name: str = "",
        data_tag: Tuple = (0, 0),
        data_shape: List = [1, 1, 1],
        data_type: str = "fp16",
        extra_info: str = ""
    ):

        self.data_name = data_name
        self.data_tag = data_tag
        self.data_shape = data_shape
        self.data_type = data_type
        self.extra_info = extra_info

        self.data_bytes = type_bytes[data_type]
        self.data_size = int(np.prod(data_shape) * type_bytes[data_type])
        self.data_dims = len(data_shape)

        # Which oper, layer, iter, model consume this tensor
        self.belong_oper_set = set()
        self.belong_layer_set = set()
        self.belong_iter_set = set()
        self.belong_model_set = set()

        '''
        {producer_tag: splitted_shape}
        '''
        self.generated_tag_splitted_dict = {}
        '''
        {splitted_shape: {consumer_tag}}        
        '''
        self.generated_splitted_tag_dict = {}
        '''
        {consumer_tag: {splitted_shape}}
        '''
        self.used_tag_splitted_dict = {}
        '''
        {splitted_shape: {consumer_tag}}
        '''
        self.used_splitted_tag_dict = {}

        '''
        {generated_split: {used_splits}}
        '''
        self.splitted_corresponding_dict = {}
        self.splitted_re_corrensponding_dict = {}
        '''
        {generated_split: {core_location}}
        '''
        self.generated_split_location = {}


    def generated_split(
        self, 
        dimension_split: Tuple[Tuple[int]],
        producertag_split: Tuple[Tuple[int]],
        dummy: bool = False,
        slice_offset: Tuple[int] = None,
        slice_shape: List[int] = None
    ):
        ''' build the producer tags for this tensor 

        Args:
            dimension_split: the splitted dimensions of the tensor
            producertag_split: the producer tags of the splitted tensor which matches the dimension_split
        Returns:
            self.generated_tag_splitted_dict: {producer_tag: splitted_shape}
            self.used_splitted_tag_dict: initilization the {splitted_shape: {consumer_tag}}        
        Example:
        >>> tensor.generated_split(
                dimension_split=((1,), (32, 64, 32), (384, 384)),
                producertag_split=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
            ) 
        '''

        current_data_shape = slice_shape if slice_shape else self.data_shape
        splitted_shapes = calculate_split_indices(current_data_shape, dimension_split, slice_offset)
        splitted_shapes_list = list(splitted_shapes)

        for i, splitted_shape in enumerate(splitted_shapes_list):
            if producertag_split[i] in self.generated_tag_splitted_dict:
                self.generated_tag_splitted_dict[producertag_split[i]].add(splitted_shape)
            else:
                self.generated_tag_splitted_dict[producertag_split[i]] = {splitted_shape}
            self.used_splitted_tag_dict[splitted_shape] = set()
            self.splitted_corresponding_dict[splitted_shape] = {splitted_shape}
            if dummy:
                self.generated_splitted_tag_dict[splitted_shape] = {(-1, )}


    def dummy_generated_split(
        self,
        dimension_split: List[Tuple[int]] = None,
        slice_offset: Tuple[int] = None,
        slice_shape: List[int] = None
        ):
        # Create producer splits with dummy (-1,) tags for externally-provided inputs
        if dimension_split is None:
            dimension_split = tuple((dim_shape,) for dim_shape in self.data_shape)
        tag_number = int(np.prod([len(dim_split) for dim_split in dimension_split]))
        producertag_split = [(-1, ) for _ in range(tag_number)]
        self.generated_split(
            dimension_split=dimension_split,
            producertag_split=producertag_split,
            dummy=True,
            slice_offset=slice_offset,
            slice_shape=slice_shape
        )


    def concat_split(
        self,
        concat_dim_idx: int, 
        concat_shape: List[int],
        concat_dimension_split: Tuple[Tuple[int]],
        concat_producertag_split: Tuple[Tuple[int]]
    ):
        ''' concat another tensor to this tensor which only support concat in one dimension

        Args:
            concat_dim_idx: the dimension index to be concated
            conccat_shape: the shape of the tensor to be concated
            concat_dimension_split: the splitted dimensions of the tensor to be concated
            concat_producertag_split: the producer tags of the splitted tensor to be concated which matches the concat_dimension_split
        Returns:
            self.generated_tag_splitted_dict: {producer_tag: concatted_splitted_shape} will be updated
            self.used_splitted_tag_dict: {concatted_splitted_shape: {consumer_tag}} will be updated
        Example:
        >>> tensor.concat_split(
               concat_dim_idx=1, 
                concat_shape=[1, 1, 768],
                concat_dimension_split=((1,), (1,), (384, 384)),
                concat_producertag_split=((1, 8), (1, 9))
            )
        '''

        concat_splitted_shapes = calculate_split_indices(concat_shape, concat_dimension_split)


        for i, concat_splitted_shape in enumerate(concat_splitted_shapes):
            concatted_splitted_shape = concat_splitted_shape
            changed_dim = concatted_splitted_shape[concat_dim_idx]
            concatted_splitted_shape = list(concatted_splitted_shape)
            concatted_splitted_shape[concat_dim_idx] = (changed_dim[0]+self.data_shape[concat_dim_idx], changed_dim[1]+self.data_shape[concat_dim_idx])
            concatted_splitted_shape = tuple(concatted_splitted_shape)
            self.generated_tag_splitted_dict[concat_producertag_split[i]] = {concatted_splitted_shape}
            self.used_splitted_tag_dict[concatted_splitted_shape] = set()
            self.splitted_corresponding_dict[concatted_splitted_shape] = {concatted_splitted_shape}

        self.data_shape[concat_dim_idx] += concat_shape[concat_dim_idx]
        self.data_size = int(np.prod(self.data_shape) * self.data_bytes)


    def get_all_dependent_shapes(self, shape, visited=None):
        """Recursively get all dependent shapes."""
        if visited is None:
            visited = set()

        if shape in visited:
            return set()

        visited.add(shape)
        result = {shape}

        if shape in self.splitted_re_corrensponding_dict:
            for dependent_shape in self.splitted_re_corrensponding_dict[shape]:
                result |= self.get_all_dependent_shapes(dependent_shape, visited)

        return result


    def used_split(
        self, 
        dimension_split: Tuple[Tuple[int]],
        consumertag_split: Tuple[Tuple[int]],
        slice_offset: Tuple[int] = None,
        slice_shape: List[int] = None
    ):
        ''' build the relationship of consumers and this tensor

        Args:
            dimension_split: the splitted dimensions of the tensor
            consumertag_split: the consumer tags of the splitted tensor which matches the dimension_split
        Returns:
            consumer_producer_dependencies: {comsumer_tag: {producer_tag: data_bytes}}        
        Example:
        >>> consumer_producer_dependencies = tensor.used_split(
                dimension_split=((1,), (96, 33), (192, 384, 192)),
                consumertag_split=((3, 0, 1, 2), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5))
            )   
        '''

        for consumer_tag in consumertag_split:
            if len(consumer_tag) == 5 or len(consumer_tag) == 6:
                belong_oper = consumer_tag[:4]
                self.belong_oper_set.add(belong_oper)
                belong_layer = consumer_tag[:3]
                self.belong_layer_set.add(belong_layer)
                belong_iter = consumer_tag[:2]
                self.belong_iter_set.add(belong_iter)
                belong_model = consumer_tag[:1]
                self.belong_model_set.add(belong_model)
            else:
                print(f"Warning: consumer_tag {consumer_tag} does not match the expected length of 5. Skipping.")

        current_data_shape = slice_shape if slice_shape else self.data_shape
        splitted_shapes = calculate_split_indices(current_data_shape, dimension_split, slice_offset)

        # For each consumer split, find overlapping producer splits and record data dependencies
        tag_split_dict = {}
        consumer_producer_dependencies = {}
        consumer_producer_parts = {}
        for i, splitted_shape in enumerate(splitted_shapes):
            tag_split_dict[consumertag_split[i]] = splitted_shape
            consumer_producer_dependency = {}
            consumer_producer_part = {}
            for producer_tag in self.generated_tag_splitted_dict:
                for producer_splitted_shape in self.generated_tag_splitted_dict[producer_tag]:
                    dependency_data_size, dependency_overlap_data = calculate_overlap(producer_splitted_shape, splitted_shape)
                    dependency_data_size = int(dependency_data_size * self.data_bytes)
                    if dependency_data_size > 0:
                        if producer_tag in consumer_producer_dependency:
                            consumer_producer_dependency[producer_tag] += dependency_data_size
                            consumer_producer_part[producer_tag].add(dependency_overlap_data)
                        else:
                            consumer_producer_dependency[producer_tag] = dependency_data_size
                            consumer_producer_part[producer_tag] = {dependency_overlap_data}

                        # Refine split tracking: subdivide existing splits along overlap boundaries
                        # so each sub-region knows exactly which consumers need it
                        used_splitted_tag_dict = deepcopy(self.used_splitted_tag_dict)
                        self.used_splitted_tag_dict = {}
                        for used_splitted_shape in used_splitted_tag_dict:
                            iter_overlap_size, iter_overlap_data = calculate_overlap(used_splitted_shape, dependency_overlap_data)
                            if iter_overlap_size > 0:
                                if iter_overlap_data not in self.splitted_re_corrensponding_dict:
                                    if iter_overlap_data == used_splitted_shape:
                                        pass
                                    else:
                                        self.splitted_corresponding_dict[used_splitted_shape] = {iter_overlap_data}
                                        self.splitted_re_corrensponding_dict[iter_overlap_data] = {used_splitted_shape}
                                        dependent_shapes = self.get_all_dependent_shapes(used_splitted_shape)
                                        self.splitted_re_corrensponding_dict[iter_overlap_data].update(dependent_shapes)
                                splitted_again_shapes = subtract_nd_intervals(used_splitted_shape, iter_overlap_data)
                                self.used_splitted_tag_dict[iter_overlap_data] = deepcopy(used_splitted_tag_dict[used_splitted_shape])
                                self.used_splitted_tag_dict[iter_overlap_data].add(consumertag_split[i])
                                if consumertag_split[i] in self.used_tag_splitted_dict:
                                    self.used_tag_splitted_dict[consumertag_split[i]].add(iter_overlap_data)
                                else:
                                    self.used_tag_splitted_dict[consumertag_split[i]] = {iter_overlap_data}
                                for splitted_again_shape in splitted_again_shapes:
                                    if splitted_again_shape not in self.splitted_re_corrensponding_dict:
                                        self.splitted_corresponding_dict[used_splitted_shape].add(splitted_again_shape)
                                        self.splitted_re_corrensponding_dict[splitted_again_shape] = {used_splitted_shape}
                                        dependent_shapes = self.get_all_dependent_shapes(used_splitted_shape)
                                        self.splitted_re_corrensponding_dict[splitted_again_shape].update(dependent_shapes)
                                    if splitted_again_shape not in self.used_splitted_tag_dict:
                                        self.used_splitted_tag_dict[splitted_again_shape] = deepcopy(used_splitted_tag_dict[used_splitted_shape])
                            else:
                                self.used_splitted_tag_dict[used_splitted_shape] = deepcopy(used_splitted_tag_dict[used_splitted_shape])

                            if used_splitted_shape in self.splitted_corresponding_dict and used_splitted_shape in self.splitted_re_corrensponding_dict:
                                for included_splitted_shape in self.splitted_re_corrensponding_dict[used_splitted_shape]:
                                    self.splitted_corresponding_dict[included_splitted_shape].remove(used_splitted_shape)
                                    for corresponding_splitted_shape in self.splitted_corresponding_dict[used_splitted_shape]:
                                        self.splitted_corresponding_dict[included_splitted_shape].add(corresponding_splitted_shape)

                    else:
                        continue
            consumer_producer_dependencies[consumertag_split[i]] = consumer_producer_dependency
            consumer_producer_parts[consumertag_split[i]] = consumer_producer_part

        # Update producer splits to reflect the refined sub-regions after consumer overlap analysis
        for producer_tag in self.generated_tag_splitted_dict:
            adpated_splitted_set = set()
            for producer_splitted_shape in self.generated_tag_splitted_dict[producer_tag]:
                if producer_splitted_shape in self.splitted_corresponding_dict:
                    adpated_splitted_set.update(self.splitted_corresponding_dict[producer_splitted_shape])
                else:
                    adpated_splitted_set.add(producer_splitted_shape)
            self.generated_tag_splitted_dict[producer_tag] = adpated_splitted_set

        return consumer_producer_dependencies, consumer_producer_parts


    def __repr__(self):
        return f'{self.data_tag} {self.data_name} {self.data_type} data {self.data_shape} produced by {self.generated_tag_splitted_dict}'

    def __str__(self):
        return self.__repr__()


class tensor_slice_notation(object):
    # A view into a sub-region of a tensor_notation, sharing the parent's split tracking dicts.
    # Operations on the slice (generated_split, used_split) delegate to the parent with offset.

    def __init__(
        self,
        tensor: tensor_notation,
        slice_tag: int = 0,
        slice_offset: Tuple[int] = None,
        slice_shape: List[int] = None
    ):

        self.tensor = tensor
        self.slice_tag = tensor.data_tag + (slice_tag,)
        self.slice_offset = slice_offset
        self.slice_shape = slice_shape
        self.data_shape = slice_shape

        self.data_name = tensor.data_name
        self.data_tag = tensor.data_tag
        self.data_type = tensor.data_type
        self.data_bytes = tensor.data_bytes
        self.data_dims = tensor.data_dims

        self.generated_tag_splitted_dict = tensor.generated_tag_splitted_dict
        self.generated_splitted_tag_dict = tensor.generated_splitted_tag_dict
        self.used_tag_splitted_dict = tensor.used_tag_splitted_dict
        self.used_splitted_tag_dict = tensor.used_splitted_tag_dict
        self.splitted_corresponding_dict = tensor.splitted_corresponding_dict
        self.splitted_re_corrensponding_dict = tensor.splitted_re_corrensponding_dict
        self.generated_split_location = tensor.generated_split_location


    def generated_split(
        self, 
        dimension_split: Tuple[Tuple[int]],
        producertag_split: Tuple[Tuple[int]],
        dummy: bool = False
    ):

        self.tensor.generated_split(
            dimension_split=dimension_split,
            producertag_split=producertag_split,
            dummy=dummy,
            slice_offset=self.slice_offset,
            slice_shape=self.slice_shape
        )


    def dummy_generated_split(
        self, 
        dimension_split: List[Tuple[int]] = None     
        ):

        self.tensor.dummy_generated_split(
            dimension_split=dimension_split,
            slice_offset=self.slice_offset,
            slice_shape=self.slice_shape
        )


    def used_split(
        self, 
        dimension_split: Tuple[Tuple[int]],
        consumertag_split: Tuple[Tuple[int]]
    ):
        return self.tensor.used_split(
            dimension_split=dimension_split,
            consumertag_split=consumertag_split,
            slice_offset=self.slice_offset,
            slice_shape=self.slice_shape
        )


    def __repr__(self):
        return f'{self.slice_tag} {self.data_name} {self.data_type} data {self.slice_shape} produced by {self.tensor.generated_tag_splitted_dict}'

    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":

    sys.path.append(os.path.join(file_path, "../../utils/"))

    import logging
    from logprint import setup_custom_levels, setup_logging_levels, log_message, logger


    test_used_logging_level = 1
    test_concat_logging_level = 2
    setup_custom_levels([test_used_logging_level, test_concat_logging_level, logging.DEBUG])

    setup_logging_levels([test_concat_logging_level, logging.DEBUG])


    tensor = tensor_notation(
        data_name="tensor", 
        data_tag=(0, 0),
        data_shape=[1, 128, 768],
        data_type="fp16", 
        extra_info=""
    )
    logger.debug(f"tenor: {tensor}")

    tensor.generated_split(
        dimension_split=((1,), (32, 64, 32), (384, 384)),
        producertag_split=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
    )
    log_message(test_used_logging_level, f"{tensor.generated_tag_splitted_dict}")

    consumer_producer_dependencies, _ = tensor.used_split(
        dimension_split=((1,), (32, 32, 32, 32), (384, 384)),
        consumertag_split=((1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9))
    )

    consumer_producer_dependencies, _ = tensor.used_split(
        dimension_split=((1,), (16, 32, 32, 32, 16), (384, 384)),
        consumertag_split=((2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9))
    )

    log_message(test_used_logging_level, f"@@@@@@@@@@@@@@@@")
    for splitted_shape in tensor.used_splitted_tag_dict:
        log_message(test_used_logging_level, f"{splitted_shape}, {tensor.used_splitted_tag_dict[splitted_shape]}")
    log_message(test_used_logging_level, f"@@@@@@@@@@@@@@@@")
    for consumer_tag in tensor.used_tag_splitted_dict:
        log_message(test_used_logging_level, f"{consumer_tag}, {tensor.used_tag_splitted_dict[consumer_tag]}")


    '''
    ####################################################################################################################
    '''
    tensor.concat_split(
        concat_dim_idx=1, 
        concat_shape=[1, 1, 768],
        concat_dimension_split=((1,), (1,), (384, 384)),
        concat_producertag_split=[(1, 8), (1, 9)]
    )
    logger.debug(f"tenor: {tensor}")
    log_message(test_concat_logging_level, f"{tensor.generated_tag_splitted_dict}")

    consumer_producer_dependencies, _ = tensor.used_split(
        dimension_split=((1,), (96, 33), (192, 384, 192)),
        consumertag_split=((3, 0, 1, 2), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5))
    )

    log_message(test_concat_logging_level, f"!!!!!!!!!!!!!!!!")
    for consumer_tag in consumer_producer_dependencies:
        log_message(test_concat_logging_level, f"{consumer_tag}, {consumer_producer_dependencies[consumer_tag]}")

    log_message(test_concat_logging_level, f"@@@@@@@@@@@@@@@@")
    for splitted_shape in tensor.used_splitted_tag_dict:
        log_message(test_concat_logging_level, f"{splitted_shape}, {tensor.used_splitted_tag_dict[splitted_shape]}")

    log_message(test_concat_logging_level, f"%%%%%%%%%%%%%%%%%")
    for consumer_tag in tensor.used_tag_splitted_dict:
        log_message(test_concat_logging_level, f"{consumer_tag}, {tensor.used_tag_splitted_dict[consumer_tag]}")

    log_message(test_concat_logging_level, f"********************")
    for splitted_shape in tensor.splitted_corresponding_dict:
        log_message(test_concat_logging_level, f"{splitted_shape}, {tensor.splitted_corresponding_dict[splitted_shape]}")
