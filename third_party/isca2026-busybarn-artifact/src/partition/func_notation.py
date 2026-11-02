
"""Function notation: represents composite functions (MHA, FFN, LMHead) as
sequences of oper_notation objects with coordinated split strategies."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from oper_notation import oper_notation

from typing import List, Dict, Tuple, Set
from collections import defaultdict


class func_notation(object):
    # Represents a composite function (e.g., MHA, FFN, LMHead) composed of
    # multiple oper_notation stages with coordinated split strategies.
    # Tag hierarchy: (iter_tag, computation_tag, function_tag, operation_tag, behavior_tag)

    def __init__(self,
                 func_name: str, func_tag: int, oper_tag: int,
                 source_data_tags: List[int],
                 func_split_lists: List[Tuple[int]]
                 ):

        self.func_name = func_name
        self.func_tag = func_tag
        self.oper_tag = oper_tag
        self.source_data_tags = source_data_tags
        self.func_split_lists = func_split_lists
