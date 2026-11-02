
"""Construct partitioned multi-head attention functions."""

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
from transpose import Transpose
from matmul import Matmul
from softmax import Softmax
from matadd import Matadd


from typing import List, Dict, Tuple, Set, Deque
import numpy as np


class mha(func_notation):

    def __init__(
        self,
        func_name: str, 
        func_tag: Tuple[int],
        oper_tag: Tuple[int], 
        source_data_tags: List[int],
        func_split_lists: List[List[Tuple[int]]],
        head_num: int,
        head_dims: int, 
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
        # Q func_split_lists[1]
        0: batch_size, 1: sequence_length, 2: hidden_states, 3: hidden_states, 
        # K KT func_split_lists[2]
        0: batch_size, 1: sequence_length, 2: hidden_states, 3: hidden_states,
        # QKT func_split_lists[3]
        0: batch_size, 1: sequence_length, 2: head_dims, 3: sequence_length,
        # S (softmax) func_split_lists[4]
        0: batch_size, 1: sequence_length, 2: sequence_length,
        # V func_split_lists[5]
        0: batch_size, 1: sequence_length, 2: hidden_states, 3: hidden_states,
        # P func_split_lists[6]
        0: batch_size, 1: sequence_length, 2: sequence_length, 3: head_dims,
        # c_proj func_split_lists[7]
        0: batch_size, 1: sequence_length, 2: hidden_states, 3: hidden_states,
        # residual func_split_lists[8]
        0: batch_size, 1: sequence_length, 2: hidden_states,
        ]
        '''

        self.hidden_states = sum(func_split_lists[0][-1])
        self.head_num = head_num
        self.head_dims = head_dims
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

        qconv_next_oper_tag, next_online_data_tag, next_offline_data_tag, qconv_next_beha_offset, self.qconv_oper = Conv1d(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_qconv",
            source_data_tags=[ln_data_tag],
            oper_split_list=func_split_lists[1],
            oper_tag=ln_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            weight_dim=self.hidden_states,
            beha_tag_offset=0
        )
        self.q_weight_data_tag = self.qconv_oper.source_data_tags[1]
        self.q_bias_data_tag = self.qconv_oper.source_data_tags[2]
        Q_data_tag = self.qconv_oper.target_data_tags[-1]

        kconv_next_oper_tag, next_online_data_tag, next_offline_data_tag, kconv_next_beha_offset, self.kconv_oper = Conv1d(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_kconv",
            source_data_tags=[ln_data_tag],
            oper_split_list=func_split_lists[2],
            oper_tag=qconv_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            weight_dim=self.hidden_states,
            beha_tag_offset=0
        )
        self.k_weight_data_tag = self.kconv_oper.source_data_tags[1]
        self.k_bias_data_tag = self.kconv_oper.source_data_tags[2]
        K_data_tag = self.kconv_oper.target_data_tags[-1]

        ktrans_next_oper_tag, next_online_data_tag, next_offline_data_tag, ktrans_next_beha_offset, self.ktrans_oper = Transpose(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_ktrans",
            source_data_tags=[K_data_tag],
            oper_split_list=[func_split_lists[2][0], func_split_lists[2][1], func_split_lists[2][3]],
            oper_tag=kconv_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            beha_tag_offset=0
        )
        KT_data_tag = self.ktrans_oper.target_data_tags[0]
        self.KTCache_data_tag = KT_data_tag

        qkt_next_oper_tag, next_online_data_tag, next_offline_data_tag, qkt_next_beha_offset, self.qkt_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_qkt",
            source_data_tags=[Q_data_tag, KT_data_tag],
            oper_split_list=func_split_lists[3],
            oper_tag=ktrans_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            head_list=[head_num, head_num],
            beha_tag_offset=0
        )
        QKT_data_tag_list = [op.target_data_tags[-1] for op in self.qkt_oper]

        S_data_tag_list = []
        s_next_oper_tag = qkt_next_oper_tag
        s_next_beha_offset = 0
        for head_idx in range(head_num):
            s_next_oper_tag, next_online_data_tag, next_offline_data_tag, s_next_beha_offset, self.s_oper = Softmax(
                data_dict=data_dict,
                beha_dict=beha_dict,
                oper_name=func_name+"_s",
                source_data_tags=[QKT_data_tag_list[head_idx]],
                oper_split_list=func_split_lists[4],
                oper_tag=qkt_next_oper_tag,
                online_data_tag=next_online_data_tag,
                offline_data_tag=next_offline_data_tag,
                reduce_dims=[2],
                beha_tag_offset=s_next_beha_offset
            )
            S_data_tag = self.s_oper.target_data_tags[0]           
            S_data_tag_list.append(S_data_tag) 

        vconv_next_oper_tag, next_online_data_tag, next_offline_data_tag, vconv_next_beha_offset, self.vconv_oper = Conv1d(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_vconv",
            source_data_tags=[ln_data_tag],
            oper_split_list=func_split_lists[5],
            oper_tag=s_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            weight_dim=self.hidden_states,
            beha_tag_offset=0
        )
        self.v_weight_data_tag = self.vconv_oper.source_data_tags[1]
        self.v_bias_data_tag = self.vconv_oper.source_data_tags[2]
        V_data_tag = self.vconv_oper.target_data_tags[-1]
        self.VCache_data_tag = V_data_tag

        p_next_oper_tag, next_online_data_tag, next_offline_data_tag, p_next_beha_offset, self.p_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_p",
            source_data_tags=S_data_tag_list + [V_data_tag],
            oper_split_list=func_split_lists[6],
            oper_tag=vconv_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            head_list=[head_num],
            beha_tag_offset=0
        )
        P_data_tag = self.p_oper[0].target_data_tags[-1][:-1]

        cproj_next_oper_tag, next_online_data_tag, next_offline_data_tag, cproj_next_beha_offset, self.cproj_oper = Conv1d(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_cproj",
            source_data_tags=[P_data_tag],
            oper_split_list=func_split_lists[7],
            oper_tag=p_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            weight_dim=self.hidden_states,
            beha_tag_offset=0
        )
        self.cproj_weight_data_tag = self.cproj_oper.source_data_tags[1]
        self.cproj_bias_data_tag = self.cproj_oper.source_data_tags[2]
        cproj_data_tag = self.cproj_oper.target_data_tags[-1]

        residual_next_oper_tag, next_online_data_tag, next_offline_data_tag, residual_next_beha_offset, self.residual_oper = Matadd(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=func_name+"_residual",
            source_data_tags=[ln_data_tag, cproj_data_tag],
            oper_split_list=func_split_lists[8],
            oper_tag=cproj_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            beha_tag_offset=0
        )
        residual_data_tag = self.residual_oper.target_data_tags[0]
        self.target_data_tags = self.residual_oper.target_data_tags

        self.mha_next_oper_tag = residual_next_oper_tag
        self.mha_next_online_data_tag = next_online_data_tag
        self.mha_next_offline_data_tag = next_offline_data_tag
        self.mha_next_beha_offset = residual_next_beha_offset


    def prefill(self):

        return self.target_data_tags, self.mha_next_oper_tag, self.mha_next_online_data_tag, self.mha_next_offline_data_tag


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

        qconv_next_oper_tag, next_online_data_tag, next_offline_data_tag, qconv_next_beha_offset, self.qconv_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_qconv",
            source_data_tags=[ln_data_tag, self.q_weight_data_tag, self.q_bias_data_tag],
            oper_split_list=func_split_lists[1],
            oper_tag=ln_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            beha_tag_offset=0,
            w_flag=True
        )
        Q_data_tag = self.qconv_oper.target_data_tags[-1]

        kconv_next_oper_tag, next_online_data_tag, next_offline_data_tag, kconv_next_beha_offset, self.kconv_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_kconv",
            source_data_tags=[ln_data_tag, self.k_weight_data_tag, self.k_bias_data_tag],
            oper_split_list=func_split_lists[2],
            oper_tag=qconv_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            beha_tag_offset=0,
            w_flag=True
        )
        K_data_tag = self.kconv_oper.target_data_tags[-1]

        ktrans_next_oper_tag, next_online_data_tag, next_offline_data_tag, ktrans_next_beha_offset, self.ktrans_oper = Transpose(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_ktrans",
            source_data_tags=[K_data_tag],
            oper_split_list=[func_split_lists[2][0], func_split_lists[2][1], func_split_lists[2][3]],
            oper_tag=kconv_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            beha_tag_offset=0,
            concat_dim_idx=1,
            concat_data_tag=self.KTCache_data_tag
        )
        KT_data_tag = self.ktrans_oper.target_data_tags[0]

        qkt_next_oper_tag, next_online_data_tag, next_offline_data_tag, qkt_next_beha_offset, self.qkt_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_qkt",
            source_data_tags=[Q_data_tag, KT_data_tag],
            oper_split_list=func_split_lists[3],
            oper_tag=ktrans_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            head_list=[self.head_num, self.head_num],
            beha_tag_offset=0
        )
        QKT_data_tag_list = [op.target_data_tags[-1] for op in self.qkt_oper]

        S_data_tag_list = []
        s_next_oper_tag = qkt_next_oper_tag
        s_next_beha_offset = 0
        for head_idx in range(self.head_num):
            s_next_oper_tag, next_online_data_tag, next_offline_data_tag, s_next_beha_offset, self.s_oper = Softmax(
                data_dict=data_dict,
                beha_dict=beha_dict,
                oper_name=self.func_name+str(iter_idx)+"_s",
                source_data_tags=[QKT_data_tag_list[head_idx]],
                oper_split_list=func_split_lists[4],
                oper_tag=qkt_next_oper_tag,
                online_data_tag=next_online_data_tag,
                offline_data_tag=next_offline_data_tag,
                reduce_dims=[2],
                beha_tag_offset=s_next_beha_offset
            )
            S_data_tag = self.s_oper.target_data_tags[0]           
            S_data_tag_list.append(S_data_tag) 

        vconv_next_oper_tag, next_online_data_tag, next_offline_data_tag, vconv_next_beha_offset, self.vconv_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_vconv",
            source_data_tags=[ln_data_tag, self.v_weight_data_tag, self.v_bias_data_tag],
            oper_split_list=func_split_lists[5],
            oper_tag=s_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            beha_tag_offset=0,
            concat_dim_idx=1,
            concat_data_tag=self.VCache_data_tag,
            w_flag=True
        )
        V_data_tag = self.vconv_oper.target_data_tags[-1]

        p_next_oper_tag, next_online_data_tag, next_offline_data_tag, p_next_beha_offset, self.p_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_p",
            source_data_tags=S_data_tag_list + [V_data_tag],
            oper_split_list=func_split_lists[6],
            oper_tag=vconv_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            head_list=[self.head_num],
            beha_tag_offset=0
        )
        P_data_tag = self.p_oper[0].target_data_tags[-1][:-1]

        cproj_next_oper_tag, next_online_data_tag, next_offline_data_tag, cproj_next_beha_offset, self.cproj_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_cproj",
            source_data_tags=[P_data_tag, self.cproj_weight_data_tag, self.cproj_bias_data_tag],
            oper_split_list=func_split_lists[7],
            oper_tag=p_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            beha_tag_offset=0,
            w_flag=True
        )
        cproj_data_tag = self.cproj_oper.target_data_tags[-1]

        residual_next_oper_tag, next_online_data_tag, next_offline_data_tag, residual_next_beha_offset, self.residual_oper = Matadd(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name=self.func_name+str(iter_idx)+"_residual",
            source_data_tags=[ln_data_tag, cproj_data_tag],
            oper_split_list=func_split_lists[8],
            oper_tag=cproj_next_oper_tag,
            online_data_tag=next_online_data_tag,
            offline_data_tag=next_offline_data_tag,
            beha_tag_offset=0
        )
        residual_data_tag = self.residual_oper.target_data_tags[0]
        target_data_tags = [residual_data_tag]

        mha_next_oper_tag = residual_next_oper_tag
        mha_next_online_data_tag = next_online_data_tag
        mha_next_offline_data_tag = next_offline_data_tag

        return target_data_tags, mha_next_oper_tag, mha_next_online_data_tag, mha_next_offline_data_tag
