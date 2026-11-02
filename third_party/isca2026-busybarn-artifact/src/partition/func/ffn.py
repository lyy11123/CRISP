
"""Construct partitioned feed-forward network functions."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))
sys.path.append(os.path.join(file_path, '../oper/'))


from func_notation import func_notation
from data_notation import tensor_notation, tensor_slice_notation
from beha_notation import beha_notation
from layernorm import Layernorm
from conv1d import Conv1d
from matmul import Matmul
from elerelu import Elerelu
from matadd import Matadd


from typing import List, Dict, Tuple, Set, Deque
import numpy as np


class ffn(func_notation):

    def __init__(
        self,
        func_name: str, 
        func_tag: Tuple[int],
        oper_tag: Tuple[int],
        source_data_tags: List[int],
        func_split_lists: List[List[Tuple[int]]],
        hidden_states: int,
        ffn_dims: int,
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
        shape for split:
        [
        # LN func_split_lists[0]
        0: batch_size, 1: sequence_length, 2: hidden_states, 
        3: batch_size, 4: sequence_length, 5: hidden_states, 
        6: batch_size, 7: sequence_length, 8: hidden_states,
        # MLP1 func_split_lists[1]
        0: batch_size, 1: sequence_length, 2: hidden_states, 3: ffn_dims,
        # MLP2 func_split_lists[2]
        0: batch_size, 1: sequence_length, 2: ffn_dims, 3: hidden_states,
        # residual func_split_lists[3]
        0: batch_size, 1: sequence_length, 2: hidden_states,
        ]
        '''

        self.hidden_states = hidden_states
        self.ffn_dims = ffn_dims
        next_oper_tag = oper_tag

        ln_next_oper_tag, next_online_data_tag, next_offline_data_tag, ln_next_beha_offset, self.layernorm_oper = Layernorm(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_ln",
            source_data_tags=source_data_tags,
            oper_split_list=func_split_lists[0],
            oper_tag=next_oper_tag,
            online_data_tag=online_data_tag,
            offline_data_tag=offline_data_tag,
            beha_tag_offset=0
        )
        ln_data_tag = self.layernorm_oper.target_data_tags[0]
        self.ln_offline_data_tags = self.layernorm_oper.ln_offline_data_tags

        mlp1_next_oper_tag, next_online_data_tag, next_offline_data_tag, mlp1_next_beha_offset, self.mlp1_oper = Conv1d(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_mlp1", 
            source_data_tags=[ln_data_tag],
            oper_split_list=func_split_lists[1], 
            oper_tag=ln_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag, 
            weight_dim=ffn_dims,
            beha_tag_offset=0
        )
        self.mlp1_weight_data_tag = self.mlp1_oper.source_data_tags[1]
        self.mlp1_bias_data_tag = self.mlp1_oper.source_data_tags[2]
        mlp1_data_tag = self.mlp1_oper.target_data_tags[-1]

        act_next_oper_tag, next_online_data_tag, next_offline_data_tag, act_next_beha_offset, self.act_oper = Elerelu(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_act", 
            source_data_tags=[mlp1_data_tag],
            oper_split_list=[func_split_lists[1][0], func_split_lists[1][1], func_split_lists[1][3]], 
            oper_tag=mlp1_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            beha_tag_offset=0
        )
        act_data_tag = self.act_oper.target_data_tags[0]

        mlp2_next_oper_tag, next_online_data_tag, next_offline_data_tag, mlp2_next_beha_offset, self.mlp2_oper = Conv1d(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_mlp2", 
            source_data_tags=[act_data_tag],
            oper_split_list=func_split_lists[2], 
            oper_tag=act_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag, 
            weight_dim=hidden_states,
            beha_tag_offset=0
        )
        self.mlp2_weight_data_tag = self.mlp2_oper.source_data_tags[1]
        self.mlp2_bias_data_tag = self.mlp2_oper.source_data_tags[2]
        mlp2_data_tag = self.mlp2_oper.target_data_tags[-1]

        residual_next_oper_tag, next_online_data_tag, next_offline_data_tag, residual_next_beha_offset, self.residual_oper = Matadd(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_residual", 
            source_data_tags=[source_data_tags[0], mlp2_data_tag],
            oper_split_list=func_split_lists[3], 
            oper_tag=mlp2_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag, 
            beha_tag_offset=0
        )
        residual_data_tag = self.residual_oper.target_data_tags[0]
        self.target_data_tags = [residual_data_tag]

        self.ffn_next_oper_tag = residual_next_oper_tag
        self.ffn_next_online_data_tag = next_online_data_tag
        self.ffn_next_offline_data_tag = next_offline_data_tag
        self.ffn_next_beha_offset = residual_next_beha_offset

    def prefill(self):
        return self.target_data_tags, self.ffn_next_oper_tag, self.ffn_next_online_data_tag, self.ffn_next_offline_data_tag

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

        ln_next_oper_tag, next_online_data_tag, next_offline_data_tag, ln_next_beha_offset, self.layernorm_oper = Layernorm(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_ln",
            source_data_tags=source_data_tags,
            oper_split_list=func_split_lists[0],
            oper_tag=next_oper_tag,
            online_data_tag=online_data_tag,
            offline_data_tag=offline_data_tag,
            beha_tag_offset=0,
            regressive=True,
            regressive_data_tags=self.ln_offline_data_tags
        )
        ln_data_tag = self.layernorm_oper.target_data_tags[0]
        self.ln_offline_data_tags = self.layernorm_oper.ln_offline_data_tags

        mlp1_next_oper_tag, next_online_data_tag, next_offline_data_tag, mlp1_next_beha_offset, self.mlp1_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_mlp1", 
            source_data_tags=[ln_data_tag, self.mlp1_weight_data_tag, self.mlp1_bias_data_tag],
            oper_split_list=func_split_lists[1], 
            oper_tag=ln_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag, 
            beha_tag_offset=0,
            w_flag=True
        )
        mlp1_data_tag = self.mlp1_oper.target_data_tags[-1]

        act_next_oper_tag, next_online_data_tag, next_offline_data_tag, act_next_beha_offset, self.act_oper = Elerelu(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_act", 
            source_data_tags=[mlp1_data_tag],
            oper_split_list=[func_split_lists[1][0], func_split_lists[1][1], func_split_lists[1][3]], 
            oper_tag=mlp1_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            beha_tag_offset=0
        )
        act_data_tag = self.act_oper.target_data_tags[0]

        mlp2_next_oper_tag, next_online_data_tag, next_offline_data_tag, mlp2_next_beha_offset, self.mlp2_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_mlp2", 
            source_data_tags=[act_data_tag, self.mlp2_weight_data_tag, self.mlp2_bias_data_tag],
            oper_split_list=func_split_lists[2], 
            oper_tag=act_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag, 
            beha_tag_offset=0,
            w_flag=True
        )
        mlp2_data_tag = self.mlp2_oper.target_data_tags[-1]

        residual_next_oper_tag, next_online_data_tag, next_offline_data_tag, residual_next_beha_offset, self.residual_oper = Matadd(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_residual", 
            source_data_tags=[source_data_tags[0], mlp2_data_tag],
            oper_split_list=func_split_lists[3], 
            oper_tag=mlp2_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag, 
            beha_tag_offset=0
        )
        residual_data_tag = self.residual_oper.target_data_tags[0]
        target_data_tags = [residual_data_tag]

        ffn_next_oper_tag = residual_next_oper_tag
        ffn_next_online_data_tag = next_online_data_tag
        ffn_next_offline_data_tag = next_offline_data_tag

        return target_data_tags, ffn_next_oper_tag, ffn_next_online_data_tag, ffn_next_offline_data_tag
