
"""Partition matrix multiply operators into tensorcore-oriented behaviors."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))
sys.path.append(file_path)


from oper_notation import oper_notation
from beha_notation import beha_notation
from data_notation import tensor_notation, tensor_slice_notation
from matadd import matadd


from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import itertools


class matmul(oper_notation):

    def __init__(self,
                 oper_name: str, oper_tag: int, 
                 source_data_tags: List[int], 
                 target_data_tags: List[int],
                 oper_split_list: List[Tuple[int]],
                 beha_tag_offset: int = 0,
                 concat_dim_idx: int = None,
                 w_flag: bool = False,
                 reduction_split_list: List[Tuple[int]] = None
                 ):

        self.concat_dim_idx = concat_dim_idx
        self.w_flag = w_flag
        self.reduction_split_list = reduction_split_list

        super().__init__(
            oper_name=oper_name+"_matmul", 
            oper_tag=oper_tag,
            source_data_tags=source_data_tags,
            target_data_tags=target_data_tags,
            oper_split_list=oper_split_list, 
            beha_tag_offset=beha_tag_offset
        )

        '''
        matmul operator supports two tensor matrix multiplication
        the parallel degree is decided by all source_data_0 dimensions and one source_data_1 dimension
        source_data_0: [dim_0, ..., dim_m, dim_k]
        source_data_1: [dim_0, ..., dim_k, dim_n]
        if source_data_2: [dim_n]
        oper_split_list: [dim_0_degree, ..., dim_m_degree, dim_k_degree, dim_n_degree]
        '''


    def data_split(self):

        source_data0_split_list = tuple(self.oper_split_list[:-1])
        source_data0_used_tags_groups = self.generate_grouped_tags_bydim(
            dimensions=self.dim_degree_list, 
            group_by_dim=self.oper_dim_number-1
        )

        if self.w_flag:
            source_data1_split_list = tuple(self.oper_split_list[-2:])
            source_data1_used_tags_groups = self.generate_grouped_tags_bydims(
                dimensions=self.dim_degree_list, 
                group_by_dims=[i for i in range(self.oper_dim_number-2)]
            )
        else:
            source_data1_split_list = tuple(self.oper_split_list[:-3]+self.oper_split_list[-2:])
            source_data1_used_tags_groups = self.generate_grouped_tags_bydim(
                dimensions=self.dim_degree_list,
                group_by_dim=self.oper_dim_number-3
            )

        self.source_data2_split_list = tuple(self.oper_split_list[-1:])
        self.source_data2_used_tags_groups = self.generate_grouped_tags_bydims(
            dimensions=self.dim_degree_list,
            group_by_dims=[i for i in range(self.oper_dim_number-1)]
        )        

        self.source_data_split_list_dict = {
            self.source_data_tags[0]: source_data0_split_list,
            self.source_data_tags[1]: source_data1_split_list
        }
        self.source_data_used_tag_groups_dict = {
            self.source_data_tags[0]: source_data0_used_tags_groups,
            self.source_data_tags[1]: source_data1_used_tags_groups
        }

        self.target_data_split_list = tuple(self.oper_split_list[:-2]+self.oper_split_list[-1:])
        self.target_data_shape = [int(np.sum(dim_splitted)) for dim_splitted in self.oper_split_list[:-2]]+[int(np.sum(dim_splitted)) for dim_splitted in self.oper_split_list[-1]]


    def consume_source_data_and_beha(self,  
                                     data_dict: Dict[int, tensor_notation], 
                                     beha_dict: Dict[int, beha_notation]
                                     ):

        for source_data in self.source_data_tags[0:1]:
            for source_data_used_tags_group in self.source_data_used_tag_groups_dict[source_data]:
                self.one_tag_group_update_data(
                    source_data_tag=source_data,
                    source_data_split_list=self.source_data_split_list_dict[source_data],
                    source_data_used_tags_group=source_data_used_tags_group,
                    data_dict=data_dict,
                    beha_dict=beha_dict
                )

        if self.w_flag:
            for source_data in self.source_data_tags[1:2]:
                for weight_data_used_tags_group in self.source_data_used_tag_groups_dict[source_data]:
                    self.one_tag_group_update_data(
                        source_data_tag=data_dict[self.source_data_tags[1]].data_tag,
                        source_data_split_list=self.source_data_split_list_dict[source_data],
                        source_data_used_tags_group=weight_data_used_tags_group,
                        data_dict=data_dict,
                        beha_dict=beha_dict
                    )

        else:
            for source_data in self.source_data_tags[1:2]:
                for source_data_used_tags_group in self.source_data_used_tag_groups_dict[source_data]:
                    self.one_tag_group_update_data(
                        source_data_tag=source_data,
                        source_data_split_list=self.source_data_split_list_dict[source_data],
                        source_data_used_tags_group=source_data_used_tags_group,
                        data_dict=data_dict,
                        beha_dict=beha_dict
                    )

        if len(self.source_data_tags) == 3:
            self.source_data2_data_tag = data_dict[self.source_data_tags[2]].data_tag

            if self.dim_degree_list[-2] == 1:
                for source_data2_data_used_tags_group in self.source_data2_used_tags_groups:
                    self.one_tag_group_update_data(
                        source_data_tag=self.source_data2_data_tag,
                        source_data_split_list=self.source_data2_split_list,
                        source_data_used_tags_group=source_data2_data_used_tags_group,
                        data_dict=data_dict,
                        beha_dict=beha_dict
                    )
            else:
                for source_data2_data_used_tags_group_idx, source_data2_data_used_tags_group in enumerate(self.source_data2_used_tags_groups):
                    if source_data2_data_used_tags_group_idx % self.dim_degree_list[-2] == 0:
                        self.one_tag_group_update_data(
                            source_data_tag=self.source_data2_data_tag,
                            source_data_split_list=self.source_data2_split_list,
                            source_data_used_tags_group=source_data2_data_used_tags_group,
                            data_dict=data_dict,
                            beha_dict=beha_dict
                        )
                    else:
                        continue


    def update_consumer_beha(self, 
                             data_dict: Dict[int, tensor_notation], 
                             beha_dict: Dict[int, beha_notation]
                             ):

        for beha_idx, beha_tag in enumerate(self.beha_tags):
            self.behaviors_dict[beha_tag] = beha_notation(
                beha_name=self.oper_name+"_"+str(beha_tag[-1]),
                beha_tag=beha_tag,
                beha_type="matmac" if len(self.source_data_tags) == 3 else "matmul",
                needed_data_split_dict=self.needed_data_split_dict[beha_tag], 
                needed_tag_size_dict=self.consumer_producer_dependencies_data[beha_tag]
            )
        beha_dict.update(self.behaviors_dict)


    def update_target_data(self, 
                           data_dict: Dict[int, tensor_notation], 
                           beha_dict: Dict[int, beha_notation]
                           ):

        if self.dim_degree_list[-2] == 1:
            ofmap = data_dict[self.target_data_tags[0]]
            if self.concat_dim_idx:
                ofmap.concat_split(
                    concat_dim_idx=self.concat_dim_idx,
                    concat_shape=[sum(concat_split) for concat_split in self.target_data_split_list],
                    concat_dimension_split=self.target_data_split_list,
                    concat_producertag_split=self.beha_tags
                )
            else:
                ofmap.generated_split(
                    dimension_split=self.target_data_split_list,
                    producertag_split=self.beha_tags
                )

        else:
            partialsum_used_tags_groups = self.generate_grouped_tags_bydim(
                dimensions=self.dim_degree_list, 
                group_by_dim=self.oper_dim_number-2
            )
            partialsum_data_tags = []
            for partial_idx in range(self.dim_degree_list[-2]):
                partialsum_data_tag = self.target_data_tags[partial_idx]
                partialsum_data_tags.append(partialsum_data_tag)

                partial_sum = data_dict[partialsum_data_tag]
                partial_sum.generated_split(
                    dimension_split=self.target_data_split_list,
                    producertag_split=partialsum_used_tags_groups[partial_idx]
                )

            partialsum_aggregate = matadd(
                oper_name=self.oper_name, 
                oper_tag=self.oper_tag,
                source_data_tags=partialsum_data_tags,
                target_data_tags=[self.target_data_tags[-1]],
                oper_split_list= self.reduction_split_list if self.reduction_split_list else self.target_data_split_list,             
                beha_tag_offset=self.degrees + self.beha_tag_offset,
                concat_dim_idx=self.concat_dim_idx
            )
            partialsum_aggregate.build_dependency(
                data_dict=data_dict,
                beha_dict=beha_dict
            )

            self.degrees += len(self.oper_split_list[-2])


def matmul_data(
    data_dict: Dict[int, tensor_notation],
    oper_name: str,
    oper_split_list: List[Tuple[int]],
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    data_type_list: List[str] = None,
    head_list: List[int] = None,
    slice_data_tags: List[Tuple[int]] = None,
    concat_dim_idx: int = None
):

    if head_list == None:
        oper_shape_list = [sum(oper_split) for oper_split in oper_split_list]
        psum_number = len(oper_split_list[-2])
        psum_shape = oper_shape_list[:-2] + [oper_shape_list[-1]]

        if data_type_list is None:
            data_type_list = ["bf16"] * (psum_number + 1)

        if psum_number > 1:
            psum_data_tag_list = []
            for psum_idx in range(psum_number):
                psum_data_tag = (online_data_tag[0], online_data_tag[1]+psum_idx)
                psum_data = tensor_notation(
                    data_name=oper_name+"_psum"+str(psum_idx),
                    data_tag=psum_data_tag,
                    data_shape=psum_shape,
                    data_type=data_type_list[psum_idx]
                )
                data_dict[psum_data_tag] = psum_data
                psum_data_tag_list.append(psum_data_tag)

            if concat_dim_idx:
                next_online_data_tag = (online_data_tag[0], online_data_tag[1]+psum_number)
            else:
                ofmap_data_tag = (online_data_tag[0], online_data_tag[1]+psum_number)
                ofmap_data = tensor_notation(
                    data_name=oper_name+"_ofmap",
                    data_tag=ofmap_data_tag,
                    data_shape=psum_shape,
                    data_type=data_type_list[-1]
                )
                data_dict[ofmap_data_tag] = ofmap_data

                next_online_data_tag = (online_data_tag[0], online_data_tag[1]+psum_number+1)
        else:
            if concat_dim_idx:
                next_online_data_tag = online_data_tag
            else:
                ofmap_data_tag = online_data_tag
                ofmap_data = tensor_notation(
                    data_name=oper_name+"_ofmap",
                    data_tag=ofmap_data_tag,
                    data_shape=psum_shape,
                    data_type=data_type_list[0]
                )
                data_dict[ofmap_data_tag] = ofmap_data

                next_online_data_tag = (online_data_tag[0], online_data_tag[1]+1)

        if concat_dim_idx:
            target_data_tag_list = psum_data_tag_list if psum_number > 1 else []
        else:
            target_data_tag_list = psum_data_tag_list + [ofmap_data_tag] if psum_number > 1 else [ofmap_data_tag]

        return [], target_data_tag_list, next_online_data_tag, offline_data_tag

    elif len(head_list) == 1:
        '''
        slice_data_tags: v_tag
        '''
        head_num = head_list[0]
        oper_shape_list = [sum(oper_split) for oper_split in oper_split_list]
        psum_number = len(oper_split_list[-2])
        psum_shape = oper_shape_list[:-2] + [oper_shape_list[-1]*head_num] 

        if data_type_list is None:
            data_type_list = ["bf16"] * (psum_number + 1)

        if psum_number > 1:
            psum_data_tag_list = []
            for psum_idx in range(psum_number):
                psum_data_tag = (online_data_tag[0], online_data_tag[1]+psum_idx)
                psum_data = tensor_notation(
                    data_name=oper_name+"_psum"+str(psum_idx),
                    data_tag=psum_data_tag,
                    data_shape=psum_shape,
                    data_type=data_type_list[psum_idx]
                )
                data_dict[psum_data_tag] = psum_data
                psum_data_tag_list.append(psum_data_tag)

            ofmap_data_tag = (online_data_tag[0], online_data_tag[1]+psum_number)
            ofmap_data = tensor_notation(
                data_name=oper_name+"_ofmap",
                data_tag=ofmap_data_tag,
                data_shape=psum_shape,
                data_type=data_type_list[-1]
            )
            data_dict[ofmap_data_tag] = ofmap_data

            next_online_data_tag = (online_data_tag[0], online_data_tag[1]+psum_number+1)
        else:
            ofmap_data_tag = online_data_tag
            ofmap_data = tensor_notation(
                data_name=oper_name+"_ofmap",
                data_tag=ofmap_data_tag,
                data_shape=psum_shape,
                data_type=data_type_list[0]
            )
            data_dict[ofmap_data_tag] = ofmap_data

            next_online_data_tag = (online_data_tag[0], online_data_tag[1]+1)

        for head_idx in range(head_num):
            v_slice = tensor_slice_notation(
                tensor=data_dict[slice_data_tags[0]],
                slice_tag=head_idx,
                slice_offset=(0, 0, oper_shape_list[-1]*head_idx),
                slice_shape=oper_shape_list[:-2]+[oper_shape_list[-1]]
            )
            data_dict[v_slice.slice_tag] = v_slice

            if psum_number > 1:
                for psum_data_tag in psum_data_tag_list:
                    psum_data_slice = tensor_slice_notation(
                        tensor=data_dict[psum_data_tag],
                        slice_tag=head_idx,
                        slice_offset=(0, 0, oper_shape_list[-1]*head_idx),
                        slice_shape=oper_shape_list[:-2]+[oper_shape_list[-1]]
                    )
                    data_dict[psum_data_slice.slice_tag] = psum_data_slice

            ofmap_slice = tensor_slice_notation(
                tensor=data_dict[ofmap_data_tag],
                slice_tag=head_idx,
                slice_offset=(0, 0, oper_shape_list[-1]*head_idx),
                slice_shape=oper_shape_list[:-2]+[oper_shape_list[-1]]
            )
            data_dict[ofmap_slice.slice_tag] = ofmap_slice

        target_data_tag_list = psum_data_tag_list + [ofmap_data_tag] if psum_number > 1 else [ofmap_data_tag]

        return [], target_data_tag_list, next_online_data_tag, offline_data_tag

    elif len(head_list) == 2:
        '''
        slice_data_tags: q_tag, kt_tag
        '''
        head_num = head_list[0]
        oper_shape_list = [sum(oper_split) for oper_split in oper_split_list]
        head_dim = oper_shape_list[-2]
        psum_number = len(oper_split_list[-2])
        psum_shape = oper_shape_list[:-2] + [oper_shape_list[-1]] 

        if data_type_list is None:
            data_type_list = ["bf16"] * ((psum_number + 1) * head_num)

        if psum_number > 1:
            target_data_tag_list = []
            for head_idx in range(head_num):
                q_slice = tensor_slice_notation(
                    tensor=data_dict[slice_data_tags[0]],
                    slice_tag=head_idx,
                    slice_offset=(0, 0, head_dim*head_idx),
                    slice_shape=oper_shape_list[:-1]
                )
                data_dict[q_slice.slice_tag] = q_slice
                kt_slice = tensor_slice_notation(
                    tensor=data_dict[slice_data_tags[1]],
                    slice_tag=head_idx,
                    slice_offset=(0, head_dim*head_idx, 0),
                    slice_shape=oper_shape_list[:-3] + [head_dim, oper_shape_list[-1]]
                )
                data_dict[kt_slice.slice_tag] = kt_slice

                for psum_idx in range(psum_number):
                    psum_data_tag = (online_data_tag[0], online_data_tag[1]+head_idx*(psum_number+1)+psum_idx)
                    psum_data = tensor_notation(
                        data_name=oper_name+"_head"+str(head_idx)+"_psum"+str(psum_idx),
                        data_tag=psum_data_tag,
                        data_shape=psum_shape,
                        data_type=data_type_list[head_idx*(psum_number+1)+psum_idx]
                    )
                    data_dict[psum_data_tag] = psum_data
                    target_data_tag_list.append(psum_data_tag)
                ofmap_data_tag = (online_data_tag[0], online_data_tag[1]+head_idx*(psum_number+1)+psum_number)
                ofmap_data = tensor_notation(
                    data_name=oper_name+"_head"+str(head_idx)+"_ofmap",
                    data_tag=ofmap_data_tag,
                    data_shape=psum_shape,
                    data_type=data_type_list[head_idx*(psum_number+1)+psum_number]
                )
                data_dict[ofmap_data_tag] = ofmap_data
                target_data_tag_list.append(ofmap_data_tag)
            next_online_data_tag = (online_data_tag[0], online_data_tag[1]+head_num*(psum_number+1))
        else:
            target_data_tag_list = []
            for head_idx in range(head_num):
                q_slice = tensor_slice_notation(
                    tensor=data_dict[slice_data_tags[0]],
                    slice_tag=head_idx,
                    slice_offset=(0, 0, head_dim*head_idx),
                    slice_shape=oper_shape_list[:-1]
                )
                data_dict[q_slice.slice_tag] = q_slice
                kt_slice = tensor_slice_notation(
                    tensor=data_dict[slice_data_tags[1]],
                    slice_tag=head_idx,
                    slice_offset=(0, head_dim*head_idx, 0),
                    slice_shape=oper_shape_list[:-3] + [head_dim, oper_shape_list[-1]]
                )
                data_dict[kt_slice.slice_tag] = kt_slice

                ofmap_data_tag = (online_data_tag[0], online_data_tag[1]+head_idx)
                ofmap_data = tensor_notation(
                    data_name=oper_name+"_head"+str(head_idx)+"_ofmap",
                    data_tag=ofmap_data_tag,
                    data_shape=psum_shape,
                    data_type=data_type_list[head_idx]
                )
                data_dict[ofmap_data_tag] = ofmap_data
                target_data_tag_list.append(ofmap_data_tag)
            next_online_data_tag = (online_data_tag[0], online_data_tag[1]+head_num)

        return [], target_data_tag_list, next_online_data_tag, offline_data_tag

    else:
        raise ValueError("head_list should be None or have length of 1 or 2")


def Matmul(
    data_dict: Dict[int, tensor_notation],
    beha_dict: Dict[int, beha_notation],
    oper_name: str,
    source_data_tags: List[int],
    oper_split_list: List[Tuple[int]],
    oper_tag: Tuple[int], 
    online_data_tag: Tuple[int],
    offline_data_tag: Tuple[int],
    head_list: List[int] = None,
    beha_tag_offset: int = 0,
    concat_dim_idx: int = None,
    concat_data_tag: Tuple[int] = None,
    w_flag: bool = False,
    reduction_split_list: List[Tuple[int]] = None
):

    if head_list == None:
        source_data_tag_list, target_data_tag_list, next_online_data_tag, next_offline_data_tag = matmul_data(
            data_dict=data_dict,
            oper_name=oper_name,
            oper_split_list=oper_split_list,
            online_data_tag=online_data_tag,
            offline_data_tag=offline_data_tag,
            data_type_list=None,
            head_list=None,
            slice_data_tags=None,
            concat_dim_idx=concat_dim_idx
        )

        matmul_oper = matmul(
            oper_name=oper_name, 
            oper_tag=oper_tag,
            source_data_tags=source_data_tags,
            target_data_tags=target_data_tag_list + [concat_data_tag] if concat_data_tag else target_data_tag_list,
            oper_split_list=oper_split_list,
            beha_tag_offset=beha_tag_offset,
            concat_dim_idx=concat_dim_idx,
            w_flag=w_flag,
            reduction_split_list=reduction_split_list
        )

        matmul_oper.build_dependency(
            data_dict=data_dict,
            beha_dict=beha_dict
        )

        next_oper_tag = oper_tag[:-1] + (oper_tag[-1]+1,)
        next_beha_offset = beha_tag_offset + matmul_oper.degrees

        return next_oper_tag, next_online_data_tag, next_offline_data_tag, next_beha_offset, matmul_oper

    elif len(head_list) == 1:
        source_data_tag_list, target_data_tag_list, next_online_data_tag, next_offline_data_tag = matmul_data(
            data_dict=data_dict,
            oper_name=oper_name,
            oper_split_list=oper_split_list,
            online_data_tag=online_data_tag,
            offline_data_tag=offline_data_tag,
            data_type_list=None,
            head_list=head_list,
            slice_data_tags=[source_data_tags[-1]]
        )

        head_num = head_list[0]

        matmul_list = []
        last_degrees = 0
        for head_idx in range(head_num):
            target_slice_tag_list = [target_data_tag + (head_idx,) for target_data_tag in target_data_tag_list]
            matmul_oper = matmul(
                oper_name=oper_name+"_head"+str(head_idx), 
                oper_tag=oper_tag,
                source_data_tags=[source_data_tags[head_idx], source_data_tags[-1] + (head_idx,)],
                target_data_tags=target_slice_tag_list,
                oper_split_list=oper_split_list,
                beha_tag_offset=beha_tag_offset + head_idx*last_degrees,
                w_flag=w_flag
            )
            matmul_oper.build_dependency(
                data_dict=data_dict,
                beha_dict=beha_dict
            )
            matmul_list.append(matmul_oper)
            last_degrees = matmul_oper.degrees

        next_oper_tag = oper_tag[:-1] + (oper_tag[-1]+1,)
        next_beha_offset = beha_tag_offset + last_degrees*head_num

        return next_oper_tag, next_online_data_tag, next_offline_data_tag, next_beha_offset, matmul_list

    elif len(head_list) == 2:
        source_data_tag_list, target_data_tag_list, next_online_data_tag, next_offline_data_tag = matmul_data(
            data_dict=data_dict,
            oper_name=oper_name,
            oper_split_list=oper_split_list,
            online_data_tag=online_data_tag,
            offline_data_tag=offline_data_tag,
            data_type_list=None,
            head_list=head_list,
            slice_data_tags=[source_data_tags[0], source_data_tags[1]]
        )

        head_num = head_list[0]

        matmul_list = []
        last_degrees = 0
        target_length = len(target_data_tag_list)
        target_head_length = target_length // head_num
        for head_idx in range(head_num):
            matmul_oper = matmul(
                oper_name=oper_name+"_head"+str(head_idx), 
                oper_tag=oper_tag,
                source_data_tags=[source_data_tags[0] + (head_idx,), source_data_tags[1] + (head_idx,)],
                target_data_tags=target_data_tag_list[target_head_length*head_idx:target_head_length*(head_idx+1)],
                oper_split_list=oper_split_list,
                beha_tag_offset=beha_tag_offset + head_idx*last_degrees,
                w_flag=w_flag
            )
            matmul_oper.build_dependency(
                data_dict=data_dict,
                beha_dict=beha_dict
            )
            matmul_list.append(matmul_oper)
            last_degrees = matmul_oper.degrees

        next_oper_tag = oper_tag[:-1] + (oper_tag[-1]+1,)
        next_beha_offset = beha_tag_offset + last_degrees*head_num

        return next_oper_tag, next_online_data_tag, next_offline_data_tag, next_beha_offset, matmul_list


if __name__ == "__main__":


    slice_number = 0

    if slice_number == 2:

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
            data_name="data1",
            data_tag=(0, 1),
            data_shape=[1, 768, 1024],
            data_type="float32"
        )
        data_dict[(0, 1)].dummy_generated_split()


        matmul_next_oper_tag, next_online_data_tag, next_offline_data_tag, matmul_next_beha_offset, matmul_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name="",
            source_data_tags=[(0, 0), (0, 1)],
            oper_split_list=[(1,), (512, 512), (128,), (1024,)],
            oper_tag=(5, 4),
            online_data_tag=(0, 2),
            offline_data_tag=(1, 0),
            head_list=[6, 6],
            beha_tag_offset=0
        )

        for data in data_dict:
            print(data_dict[data])

        print("***************************")
        for beha in beha_dict:
            print(beha_dict[beha])

    elif slice_number == 1:

        data_dict = {}
        beha_dict = {}

        data_dict[(0, 0)] = tensor_notation(
            data_name="V",
            data_tag=(0, 0),
            data_shape=[1, 1024, 768],
            data_type="float32"
        )
        data_dict[(0, 0)].dummy_generated_split()

        for head_idx in range(6):
            s_data = tensor_notation(
                data_name=f"head{head_idx}_S",
                data_tag=(0, 1+head_idx),
                data_shape=[1, 1024, 1024],
                data_type="float32"
            )
            data_dict[s_data.data_tag] = s_data
            data_dict[s_data.data_tag].dummy_generated_split()


        matmul_next_oper_tag, next_online_data_tag, next_offline_data_tag, matmul_next_beha_offset, matmul_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name="",
            source_data_tags=[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 0)],
            oper_split_list=[(1,), (512, 512), (512, 512), (128,)],
            oper_tag=(5, 4),
            online_data_tag=(0, 7),
            offline_data_tag=(1, 0),
            head_list=[6],
            beha_tag_offset=0
        )

        for data in data_dict:
            print(data_dict[data])

        print("***************************")
        for beha in beha_dict:
            print(beha_dict[beha])        

    else:

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
            data_name="data1",
            data_tag=(0, 1),
            data_shape=[1, 768, 1024],
            data_type="float32"
        )
        data_dict[(0, 1)].dummy_generated_split()

        matmul_next_oper_tag, next_online_data_tag, next_offline_data_tag, matmul_next_beha_offset, matmul_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name="",
            source_data_tags=[(0, 0), (0, 1)],
            oper_split_list=[(1,), (512, 512), (384, 192, 192), (1024,)],
            oper_tag=(5, 4),
            online_data_tag=(0, 2),
            offline_data_tag=(1, 0),
            head_list=None,
            beha_tag_offset=0,
            reduction_split_list=[(1,), (512, 512), (512, 256, 256)],
        )

        for data in data_dict:
            print(data_dict[data])

        print("***************************")
        for beha in beha_dict:
            print(beha_dict[beha])    

        print("==========================")

        data_dict[(0, 5)] = tensor_notation(
            data_name="new_input",
            data_tag=(0, 5),
            data_shape=[1, 1, 768],
            data_type="float32"
        )
        data_dict[(0, 5)].dummy_generated_split()

        data_dict[(0, 6)] = tensor_notation(
            data_name="bias",
            data_tag=(0, 6),
            data_shape=[1024],
            data_type="float32"
        )
        data_dict[(0, 6)].dummy_generated_split()

        matmul_next_oper_tag, next_online_data_tag, next_offline_data_tag, matmul_next_beha_offset, matmul_oper = Matmul(
            data_dict=data_dict,
            beha_dict=beha_dict,
            oper_name="",
            source_data_tags=[(0, 5), (0, 1), (0, 6)],
            oper_split_list=[(1,), (1,), (384, 384), (1024,)],
            oper_tag=(6, 4),
            online_data_tag=(0, 3),
            offline_data_tag=(1, 0),
            head_list=None,
            beha_tag_offset=2,
            concat_dim_idx=1,
            concat_data_tag=(0, 2)
        )

        for data in data_dict:
            print(data_dict[data])

        print("***************************")
        for beha in beha_dict:
            print(beha_dict[beha])    
