
"""Construct partitioned language-model head functions."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))
sys.path.append(os.path.join(file_path, '../oper/'))


from func_notation import func_notation
from data_notation import tensor_notation, tensor_slice_notation
from beha_notation import beha_notation
from conv1d import Conv1d
from matmul import Matmul
from redmax import Redmax
from embedding import Embedding


from typing import List, Dict, Tuple, Set, Deque
import numpy as np


class lmhead(func_notation):

    def __init__(
        self,
        func_name: str, 
        func_tag: Tuple[int],
        oper_tag: Tuple[int],
        source_data_tags: List[int],
        func_split_lists: List[List[Tuple[int]]],
        vocub_size: int,
        hidden_states: int, 
        online_data_tag: Tuple[int],
        offline_data_tag: Tuple[int], 
        data_dict: Dict[int, tensor_notation],
        beha_dict: Dict[int, beha_notation]
    ):

        super().__init__(
            func_name=func_name, 
            func_tag=func_tag,
            oper_tag=oper_tag,
            source_data_tags=source_data_tags,
            func_split_lists=func_split_lists
        )    

        '''
        sequence_length = 1
        shape for split:
        [
        # lm_head func_split_lists[0]
        0: batch_size, 1: sequence_length, 2: hidden_states, 3: vocub_size, 
        # redmax func_split_lists[1]
        0: batch_size, 1: sequence_length, 2: hidden_states, 3: vocub_size,
        # embedding func_split_lists[2]
        0: batch_size, 1: sequence_length, 2: hidden_states, 
        ]
        '''

        self.vocub_size = vocub_size
        self.hidden_states = hidden_states
        next_oper_tag = oper_tag
        fmap_shape = data_dict[source_data_tags[0]].data_shape

        last_token_data = tensor_slice_notation(
            tensor=data_dict[source_data_tags[0]],
            slice_tag=hidden_states,
            slice_offset=[0, fmap_shape[1]-1, 0],
            slice_shape=[fmap_shape[0], 1, fmap_shape[2]]
        )
        data_dict[last_token_data.slice_tag] = last_token_data

        lm_head_next_oper_tag, next_online_data_tag, next_offline_data_tag, lm_head_next_beha_offset, self.lm_head_oper = Conv1d(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_lm_head",
            source_data_tags=[last_token_data.slice_tag],
            oper_split_list=func_split_lists[0],
            oper_tag=next_oper_tag,
            online_data_tag=online_data_tag,
            offline_data_tag=offline_data_tag,
            weight_dim=self.vocub_size,
            beha_tag_offset=0
        ) 
        self.lm_head_weight_data_tag = self.lm_head_oper.source_data_tags[1]
        self.lm_head_bias_data_tag = self.lm_head_oper.source_data_tags[2]
        logits_data_tag = self.lm_head_oper.target_data_tags[-1]

        redmax_next_oper_tag, next_online_data_tag, next_offline_data_tag, redmax_next_beha_offset, self.redmax_oper = Redmax(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_select",
            source_data_tags=[logits_data_tag],
            oper_split_list=func_split_lists[1],
            oper_tag=lm_head_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            reduce_dims=[2],
            beha_tag_offset=0
        )            
        select_data_tag = self.redmax_oper.target_data_tags[0]

        embedding_next_oper_tag, next_online_data_tag, next_offline_data_tag, embedding_next_beha_offset, self.embedding_oper = Embedding(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_embedding",
            source_data_tags=[select_data_tag],
            oper_split_list=func_split_lists[2],
            oper_tag=redmax_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            hidden_dim=self.hidden_states,
            beha_tag_offset=0
        )
        embedding_data_tag = self.embedding_oper.target_data_tags[0]
        self.target_data_tags = [embedding_data_tag]

        self.lmhead_next_oper_tag = embedding_next_oper_tag
        self.lmhead_next_online_data_tag = next_online_data_tag
        self.lmhead_next_offline_data_tag = next_offline_data_tag
        self.lmhead_next_beha_offset = embedding_next_beha_offset


    def prefill(self):

        return self.target_data_tags, self.lmhead_next_oper_tag, self.lmhead_next_online_data_tag, self.lmhead_next_offline_data_tag


    def decode(
        self, 
        iter_idx: int,
        func_tag: Tuple[int],
        oper_tag: Tuple[int],
        source_data_tags: List[int],
        func_split_lists: List[List[Tuple[int]]],
        online_data_tag: Tuple[int],
        offline_data_tag: Tuple[int],
        data_dict: Dict[int, tensor_notation],
        beha_dict: Dict[int, beha_notation]            
    ):

        next_oper_tag = oper_tag

        lm_head_next_oper_tag, next_online_data_tag, next_offline_data_tag, lm_head_next_beha_offset, self.lm_head_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_lm_head",
            source_data_tags=[source_data_tags[0], self.lm_head_weight_data_tag, self.lm_head_bias_data_tag],
            oper_split_list=func_split_lists[0],
            oper_tag=next_oper_tag,
            online_data_tag=online_data_tag,
            offline_data_tag=offline_data_tag,
            beha_tag_offset=0
        ) 
        logits_data_tag = self.lm_head_oper.target_data_tags[0]

        redmax_next_oper_tag, next_online_data_tag, next_offline_data_tag, redmax_next_beha_offset, self.redmax_oper = Redmax(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_select",
            source_data_tags=[logits_data_tag],
            oper_split_list=func_split_lists[1],
            oper_tag=lm_head_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            reduce_dims=[2],
            beha_tag_offset=0
        )            
        select_data_tag = self.redmax_oper.target_data_tags[0]

        embedding_next_oper_tag, next_online_data_tag, next_offline_data_tag, embedding_next_beha_offset, self.embedding_oper = Embedding(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_embedding",
            source_data_tags=[select_data_tag],
            oper_split_list=func_split_lists[2],
            oper_tag=redmax_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            hidden_dim=self.hidden_states,
            beha_tag_offset=0
        )
        embedding_data_tag = self.embedding_oper.target_data_tags[0]
        target_data_tags = [embedding_data_tag]

        lmhead_next_oper_tag = embedding_next_oper_tag
        lmhead_next_online_data_tag = next_online_data_tag
        lmhead_next_offline_data_tag = next_offline_data_tag

        return target_data_tags, lmhead_next_oper_tag, lmhead_next_online_data_tag, lmhead_next_offline_data_tag
