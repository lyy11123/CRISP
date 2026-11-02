
"""LR behavior notation: describes computation tasks with their data dependencies,
compute costs, and core assignments used by the mapping and scheduling phases."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from data_notation import tensor_notation, tensor_slice_notation

from typing import List, Dict, Tuple, Set
from collections import defaultdict


'''
beha_size_dict = {
matmul: [source0_size, source1_size], 
matadd: [source0_size, source1_size],
matmac: [source0_size, source1_size, source2_size],
...
}
'''

# Maps hardware unit type -> set of behavior types that execute on it
beha_kind_dict = {
    "tensorcore": {"matmul", "matmac"},
    "vectorunit": {
        "eleadd", "eleexp", "elegelu", "elemul", "elepow2", "elerelu", "elesqrt", "elesilu", "elerope",
        "matadd", 
        "redmax", "redsum", 
        "vecadd", "vecdiv", "vecmac", "vecmul",
        "transpose",
        "lookup",
        "dispatch", "combine",
        }
}


class beha_notation(object):
    # A single compute behavior (e.g., one matmul tile on one core).
    # Tracks which data splits it needs (sources) and produces (targets),
    # plus its assigned core location and device type.

    def __init__(self,
                 beha_name: str, beha_tag: int,
                 beha_type: str,
                 needed_data_split_dict: Dict[Tuple[int], Set[Tuple[int]]],
                 needed_tag_size_dict: Dict[Tuple[int], int]
                 ):

        self.beha_name = beha_name
        self.beha_tag = beha_tag
        self.beha_type = beha_type
        ''' 

        Args:
            needed_data_split_dict: source data splitted part {(source_data_tag0): {split0, split1, ...}, ...}
        Returns:

        Example:
        >>> matadd_beha = beha_notation(
                beha_name="ffn",
                beha_tag=(5, 4, 16),
                beha_type="matmul",
                needed_data_split_dict={(0, 2): {((0, 0), (0, 127), (0, 383))}, (0, 3): {((0, 0), (0, 127), (0, 383))}, (0, 4): {((0, 0), (0, 127), (0, 383))}}
            )   
        '''
        self.needed_data_split_dict = needed_data_split_dict
        self.merge_tensor_blocks()
        self.needed_tag_size_dict = needed_tag_size_dict

        self.producer_tags = set()      # beha_tags that produce data this behavior needs
        self.consumer_tags = set()      # beha_tags that consume data this behavior produces
        self.produced_data_split_dict = {}

        self.source_tag_size_dict = {}

        self.location = None            # (y, x) core coordinate assigned by mapping
        self.device = None              # hardware unit type (tensorcore / vectorunit)


    def add_consumer(self, consumer_dict: Dict[Tuple[int], Set[Tuple[int]]]):
        # Register downstream behaviors that depend on this behavior's output splits
        for consumer_tag in consumer_dict:
            self.consumer_tags.add(consumer_tag)

            target_data_tag = consumer_dict[consumer_tag][0]
            if target_data_tag not in self.produced_data_split_dict:
                self.produced_data_split_dict[target_data_tag] = set()
            for target_data_split in consumer_dict[consumer_tag][1]:
                self.produced_data_split_dict[target_data_tag].add(target_data_split)


    def merge_tensor_blocks(self):
        # Compute bounding-box shape of all needed splits per source tensor
        self.beha_datashape = []
        self.beha_datatag = []
        for data_tag in self.needed_data_split_dict:
            blocks = list(self.needed_data_split_dict[data_tag])
            min_indices = [float('inf')] * len(blocks[0])
            max_indices = [-float('inf')] * len(blocks[0])
            for block in blocks:
                for dim, (start, end) in enumerate(block):
                    min_indices[dim] = min(min_indices[dim], start)
                    max_indices[dim] = max(max_indices[dim], end)
            shape = [max_idx - min_idx + 1 for min_idx, max_idx in zip(min_indices, max_indices)]
            self.beha_datashape.append(shape)        
            self.beha_datatag.append(data_tag)


    def add_location(self, data_dict: Dict[Tuple[int], tensor_notation]):

        pass


    def __str__(self):
        if self.location:
            return f"{self.beha_tag} {self.beha_name} Behavior is {self.beha_type} need {self.needed_data_split_dict} to {self.produced_data_split_dict} at {self.location}"
        else:
            return f"{self.beha_tag} {self.beha_name} Behavior is {self.beha_type} need {self.needed_data_split_dict} to {self.produced_data_split_dict}"

    def __repr__(self):
        return self.__str__()
