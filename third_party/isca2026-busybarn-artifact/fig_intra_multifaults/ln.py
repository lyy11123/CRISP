
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../'))


from cfg import ch_shapes, co_shapes, tesnorcore_grain_list, co_bws, ch_bws, failures


busybarn_template = '''import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
file_real_path = os.path.realpath(__file__)
filename_with_extension = os.path.basename(file_real_path)
filename_without_extension = os.path.splitext(filename_with_extension)[0]
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../utils/'))
sys.path.append(os.path.join(file_path, '../../src/partition/oper/'))
sys.path.append(os.path.join(file_path, '../../src/partition/func/'))
sys.path.append(os.path.join(file_path, '../../src/partition/'))
sys.path.append(os.path.join(file_path, '../../src/mapping/'))
sys.path.append(os.path.join(file_path, '../../src/scheduling/'))
sys.path.append(os.path.join(file_path, '../../src/backend/analytical/'))
sys.path.append(os.path.join(file_path, '../../endtoend/'))
sys.path.append(os.path.join(file_path, '../../src/scheduling/communication/topology/'))
sys.path.append(os.path.join(file_path, '../../tool/'))


from read_cfg import cfg_to_dict
from WAMIS_HD import wamis_hdc
from partition import generate_average_degree, whole_degree_to_dim_degrees
from add_communication import add_beha_producers, build_event, add_mediumdata, add_broadcast
from data_notation import tensor_notation
from layernorm import Layernorm
from Pre_Mapping import update_data, initialized_mapping, autoregreesive_dag, vlm_dag
from Loop_Mapping import LoopMapping, ZigZagMapping, RandomMapping, AllMapping
from Stream_Mapping import stream_mapping
from event_driver import event_driver
from timeline import plot_gantt_multiple, devicedict_to_showlist


import numpy as np 
import heapq
import itertools
from copy import deepcopy
import random
random.seed(123)
import pickle

hardware_cfg = cfg_to_dict(os.path.join(file_path, "../cfg/{cfg_name}.cfg"))
hardware_platform = wamis_hdc(hardware_cfg)

layer_num = 1
batch_size = 16
sequence_length = {sq}
hidden_states = 7168
head_num = 56
head_dims = hidden_states // head_num
vocub_size = 50257
ffn_dims = 28672

parallelism_degree = 64
chiplets_per_layer = 1
ddr_random = False
greedy_flag = {greedy_flag}
dijkstra_routing = True
alpha = 100
beta = 1
gamma = 100
SA_flag = True
random_threshold = [0.7]
LP_flag = False
region_restriced = True
related = False
hops_only = False
loss_ratio = {loss_ratio}
mem_enable = True
mem_threshold = [0.3]
t_max = 10
t_min = 1e-6
steps = 1e3

data_dict = {{}}
beha_dict = {{}}
event_dict = {{}}
activation_data_tag = (0, 0)

initialization_layer_tag = 0

data_dict[activation_data_tag] = tensor_notation(
    data_name="ifmap",
    data_tag=activation_data_tag,
    data_shape=[batch_size, sequence_length, hidden_states],
    data_type="bf16"
)
data_dict[activation_data_tag].dummy_generated_split()

decoder_out_data_tags = [activation_data_tag]
next_online_data_tag = (0, 1)
next_offline_data_tag = (1, 0)
layer_list = []


ln_split_parallelism_degree_0 = (1, 64, 1)
ln_split_parallelism_degree_1 = (1, 8, 8)

ln_parallelisms_list_0 = [
    generate_average_degree(batch_size, ln_split_parallelism_degree_0[0]),
    generate_average_degree(sequence_length, ln_split_parallelism_degree_0[1]),
    generate_average_degree(hidden_states, ln_split_parallelism_degree_0[2]),
    ]

ln_parallelisms_list_1 = [
    generate_average_degree(batch_size, ln_split_parallelism_degree_1[0]),
    generate_average_degree(sequence_length, ln_split_parallelism_degree_1[1]),
    generate_average_degree(hidden_states, ln_split_parallelism_degree_1[2]),
]

layernorm_next_oper_tag, next_online_data_tag, next_offline_data_tag, layernorm_next_beha_offset, layernorm_oper = Layernorm(
    data_dict=data_dict,
    beha_dict=beha_dict,
    oper_name="LN",
    source_data_tags=[activation_data_tag],
    oper_split_list=ln_parallelisms_list_0+ln_parallelisms_list_0+ln_parallelisms_list_1, 
    oper_tag=(0, 0, initialization_layer_tag, 0),
    online_data_tag=next_online_data_tag,
    offline_data_tag=next_offline_data_tag
)
ln_data_tag = layernorm_oper.target_data_tags[-1]

add_beha_producers(beha_dict=beha_dict)


update_data(beha_dict=beha_dict, data_dict=data_dict)
model_dag, model_chiplets = autoregreesive_dag(llm_layer_num=layer_num, chiplets_per_layer=chiplets_per_layer)
layers_regions = AllMapping(model_dag, model_chiplets, hardware_platform)

initialized_mapping(
    beha_dict=beha_dict,
    data_dict=data_dict,
    hardware_platform=hardware_platform,
    layers_regions=layers_regions,
    greedy_flag=greedy_flag
)

hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict = build_event(
    beha_dict=beha_dict,
    data_dict=data_dict,
    hardware_platform=hardware_platform, 
    event_dict=event_dict,
    dijkstra_routing=dijkstra_routing,
    alpha=alpha,
    beta=beta,
    gamma=gamma
)

broadcast_datatags = [ln_data_tag]
mem_datatags = []
if mem_datatags:
    hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict = add_mediumdata(
        data_tags=mem_datatags,
        ddr_chiplets=layers_regions[initialization_layer_tag],
        hops=hops,
        communication_distances=communication_distances,
        communication_loads_dict=communication_loads_dict,
        tensorcore_loads_dict=tensorcore_loads_dict,
        vectorunit_loads_dict=vectorunit_loads_dict,
        beha_dict=beha_dict,
        data_dict=data_dict,
        hardware_platform=hardware_platform,
        event_dict=event_dict,
        dijkstra_routing=dijkstra_routing,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        random_flag=ddr_random
    )


initial_event_dict = deepcopy(event_dict)
add_broadcast(
    data_tags=broadcast_datatags,
    ddr_chiplets=layers_regions[initialization_layer_tag],
    beha_dict=beha_dict,
    data_dict=data_dict,
    hardware_platform=hardware_platform,
    event_dict=initial_event_dict,
    dijkstra_routing=dijkstra_routing,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
)
initial_hardware_platform = deepcopy(hardware_platform)
time_cost = event_driver(
    events_dict=initial_event_dict,
    hardware_platform=initial_hardware_platform
)

if SA_flag:
    states = stream_mapping(
        hops=hops,
        communication_distances=communication_distances,
        communication_loads_dict=communication_loads_dict,
        tensorcore_loads_dict=tensorcore_loads_dict,
        vectorunit_loads_dict=vectorunit_loads_dict,
        layers_regions=layers_regions,
        beha_dict=beha_dict,
        data_dict=data_dict,
        hardware_platform=hardware_platform,
        event_dict=event_dict,
        random_threshold=random_threshold,
        LP_flag=LP_flag,
        region_restriced=region_restriced,
        related=related,
        hops_only=hops_only,
        loss_ratio=loss_ratio,
        mem_enable=mem_enable,
        mem_threshold=mem_threshold,
        mem_datatags=mem_datatags,
        mem_random=ddr_random,
        dijkstra_routing=dijkstra_routing,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        t_max=t_max,
        t_min=t_min,
        steps=steps,
    )

add_broadcast(
    data_tags=broadcast_datatags,
    ddr_chiplets=layers_regions[initialization_layer_tag],
    beha_dict=beha_dict,
    data_dict=data_dict,
    hardware_platform=hardware_platform,
    event_dict=event_dict,
    dijkstra_routing=dijkstra_routing,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
)


time_cost = event_driver(
    events_dict=event_dict,
    hardware_platform=hardware_platform
)
print(f"SA time cost: {{time_cost}}")
with open(os.path.join(file_path, "../results/{py_name}.txt"), "w") as result_f:
    result_f.write(f"SA time cost: {{time_cost}}")


records_lists, device_names = devicedict_to_showlist(hardware_platform.modules_dict["tensorcore"])
records_lists, device_names = devicedict_to_showlist(hardware_platform.modules_dict["vectorunit"], records_lists, device_names)
records_lists, device_names = devicedict_to_showlist(hardware_platform.links_dict, records_lists, device_names)

pkl_name = f"../results/{py_name}_records_lists.pkl"
pkl_path = os.path.join(file_path, pkl_name)
with open(pkl_path, 'wb') as f:
    pickle.dump((records_lists, device_names), f)
'''

sql_list = [512, 2048]

greedy_flag_list = [
    True,
    False,
]

loss_ratio_list = [
    [1, 1, 1, 1],
    [1, 1, 1, 0.1],
    [1, 1, 0.1, 0.1],
    [1, 0.1, 0.1, 0.1],
    [0.1, 1, 0.1, 0.1],
    [0.1, 0.1, 1, 0.1],
    [0.1, 0.1, 1, 1],
    [0.1, 1, 1, 1]
]


output_dir = os.path.join(file_path, './py/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for ch_shape_idx, [ch_row, ch_col] in enumerate(ch_shapes):
    ch_rc = int(ch_row * ch_col)
    for co_shape_idx, [co_row, co_col] in enumerate(co_shapes):
        co_rc = int(co_row * co_col)
        for tensorcore_grain in tesnorcore_grain_list:
            for ch_bw in ch_bws:
                for co_bw in co_bws:
                    for fail_idx, (failed_nodes, failed_links) in enumerate(failures):
                        cfg_name = f'config_ch{ch_row}x{ch_col}_bw{ch_bw}_co{co_row}x{co_col}_bw{co_bw}_t{tensorcore_grain[0]}x{tensorcore_grain[1]}_failpattern{fail_idx}'
                        for sq in sql_list:
                            for greedy_flag_idx, greedy_flag in enumerate(greedy_flag_list):
                                for loss_ratio_idx, loss_ratio in enumerate(loss_ratio_list):
                                    py_name = f'ln_sq{sq}_greedy{greedy_flag_idx}_lossratio{loss_ratio_idx}_ch{ch_row}x{ch_col}_bw{ch_bw}_co{co_row}x{co_col}_bw{co_bw}_t{tensorcore_grain[0]}x{tensorcore_grain[1]}_failpattern{fail_idx}'
                                    content = busybarn_template.format(
                                        cfg_name=cfg_name,
                                        sq=sq,
                                        greedy_flag=greedy_flag,
                                        loss_ratio=loss_ratio,
                                        py_name=py_name
                                    )
                                    path = os.path.join(output_dir, py_name+".py")
                                    with open(path, 'w') as f:
                                        f.write(content)
                                    print(f'Written {path}')


gemini_template = '''import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
file_real_path = os.path.realpath(__file__)
filename_with_extension = os.path.basename(file_real_path)
filename_without_extension = os.path.splitext(filename_with_extension)[0]
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../utils/'))
sys.path.append(os.path.join(file_path, '../../src/partition/oper/'))
sys.path.append(os.path.join(file_path, '../../src/partition/func/'))
sys.path.append(os.path.join(file_path, '../../src/partition/'))
sys.path.append(os.path.join(file_path, '../../src/mapping/'))
sys.path.append(os.path.join(file_path, '../../src/scheduling/'))
sys.path.append(os.path.join(file_path, '../../src/backend/analytical/'))
sys.path.append(os.path.join(file_path, '../../endtoend/'))
sys.path.append(os.path.join(file_path, '../../src/scheduling/communication/topology/'))
sys.path.append(os.path.join(file_path, '../../tool/'))


from read_cfg import cfg_to_dict
from WAMIS_HD import wamis_hdc
from partition import generate_average_degree, whole_degree_to_dim_degrees
from add_communication import add_beha_producers, build_event, add_mediumdata, add_broadcast
from data_notation import tensor_notation
from layernorm import Layernorm
from Pre_Mapping import update_data, initialized_mapping, autoregreesive_dag, vlm_dag
from Loop_Mapping import LoopMapping, ZigZagMapping, RandomMapping, AllMapping
from Stream_Mapping import stream_mapping
from event_driver import event_driver
from timeline import plot_gantt_multiple, devicedict_to_showlist


import numpy as np 
import heapq
import itertools
from copy import deepcopy
import random
random.seed(123)
import pickle

hardware_cfg = cfg_to_dict(os.path.join(file_path, "../cfg/{cfg_name}.cfg"))
hardware_platform = wamis_hdc(hardware_cfg)

layer_num = 1
batch_size = 16
sequence_length = {sq}
hidden_states = 7168
head_num = 56
head_dims = hidden_states // head_num
vocub_size = 50257
ffn_dims = 28672

parallelism_degree = 64
chiplets_per_layer = 1
ddr_random = True
greedy_flag = False
dijkstra_routing = False
alpha = 100
beta = 1
gamma = 100
SA_flag = True
random_threshold = [0.7]
LP_flag = True
region_restriced = False
related = False
hops_only = True
loss_ratio = [1, 1, 1, 1]
mem_enable = True
mem_threshold = [0.3]
t_max = 10
t_min = 1e-6
steps = 1e3

data_dict = {{}}
beha_dict = {{}}
event_dict = {{}}
activation_data_tag = (0, 0)

initialization_layer_tag = 0

data_dict[activation_data_tag] = tensor_notation(
    data_name="ifmap",
    data_tag=activation_data_tag,
    data_shape=[batch_size, sequence_length, hidden_states],
    data_type="bf16"
)
data_dict[activation_data_tag].dummy_generated_split()

decoder_out_data_tags = [activation_data_tag]
next_online_data_tag = (0, 1)
next_offline_data_tag = (1, 0)
layer_list = []


ln_split_parallelism_degree_0 = (1, 64, 1)
ln_split_parallelism_degree_1 = (1, 8, 8)

ln_parallelisms_list_0 = [
    generate_average_degree(batch_size, ln_split_parallelism_degree_0[0]),
    generate_average_degree(sequence_length, ln_split_parallelism_degree_0[1]),
    generate_average_degree(hidden_states, ln_split_parallelism_degree_0[2]),
    ]

ln_parallelisms_list_1 = [
    generate_average_degree(batch_size, ln_split_parallelism_degree_1[0]),
    generate_average_degree(sequence_length, ln_split_parallelism_degree_1[1]),
    generate_average_degree(hidden_states, ln_split_parallelism_degree_1[2]),
]

layernorm_next_oper_tag, next_online_data_tag, next_offline_data_tag, layernorm_next_beha_offset, layernorm_oper = Layernorm(
    data_dict=data_dict,
    beha_dict=beha_dict,
    oper_name="LN",
    source_data_tags=[activation_data_tag],
    oper_split_list=ln_parallelisms_list_0+ln_parallelisms_list_0+ln_parallelisms_list_1, 
    oper_tag=(0, 0, initialization_layer_tag, 0),
    online_data_tag=next_online_data_tag,
    offline_data_tag=next_offline_data_tag
)
ln_data_tag = layernorm_oper.target_data_tags[-1]

add_beha_producers(beha_dict=beha_dict)


update_data(beha_dict=beha_dict, data_dict=data_dict)
model_dag, model_chiplets = autoregreesive_dag(llm_layer_num=layer_num, chiplets_per_layer=chiplets_per_layer)
layers_regions = AllMapping(model_dag, model_chiplets, hardware_platform)

initialized_mapping(
    beha_dict=beha_dict,
    data_dict=data_dict,
    hardware_platform=hardware_platform,
    layers_regions=layers_regions,
    greedy_flag=greedy_flag
)

hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict = build_event(
    beha_dict=beha_dict,
    data_dict=data_dict,
    hardware_platform=hardware_platform, 
    event_dict=event_dict,
    dijkstra_routing=dijkstra_routing,
    alpha=alpha,
    beta=beta,
    gamma=gamma
)

broadcast_datatags = [ln_data_tag]
mem_datatags = []
if mem_datatags:
    hops, communication_distances, communication_loads_dict, tensorcore_loads_dict, vectorunit_loads_dict = add_mediumdata(
        data_tags=mem_datatags,
        ddr_chiplets=layers_regions[initialization_layer_tag],
        hops=hops,
        communication_distances=communication_distances,
        communication_loads_dict=communication_loads_dict,
        tensorcore_loads_dict=tensorcore_loads_dict,
        vectorunit_loads_dict=vectorunit_loads_dict,
        beha_dict=beha_dict,
        data_dict=data_dict,
        hardware_platform=hardware_platform,
        event_dict=event_dict,
        dijkstra_routing=dijkstra_routing,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        random_flag=ddr_random
    )


initial_event_dict = deepcopy(event_dict)
add_broadcast(
    data_tags=broadcast_datatags,
    ddr_chiplets=layers_regions[initialization_layer_tag],
    beha_dict=beha_dict,
    data_dict=data_dict,
    hardware_platform=hardware_platform,
    event_dict=initial_event_dict,
    dijkstra_routing=dijkstra_routing,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
)
initial_hardware_platform = deepcopy(hardware_platform)
time_cost = event_driver(
    events_dict=initial_event_dict,
    hardware_platform=initial_hardware_platform
)

if SA_flag:
    states = stream_mapping(
        hops=hops,
        communication_distances=communication_distances,
        communication_loads_dict=communication_loads_dict,
        tensorcore_loads_dict=tensorcore_loads_dict,
        vectorunit_loads_dict=vectorunit_loads_dict,
        layers_regions=layers_regions,
        beha_dict=beha_dict,
        data_dict=data_dict,
        hardware_platform=hardware_platform,
        event_dict=event_dict,
        random_threshold=random_threshold,
        LP_flag=LP_flag,
        region_restriced=region_restriced,
        related=related,
        hops_only=hops_only,
        loss_ratio=loss_ratio,
        mem_enable=mem_enable,
        mem_threshold=mem_threshold,
        mem_datatags=mem_datatags,
        mem_random=ddr_random,
        dijkstra_routing=dijkstra_routing,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        t_max=t_max,
        t_min=t_min,
        steps=steps,
    )

add_broadcast(
    data_tags=broadcast_datatags,
    ddr_chiplets=layers_regions[initialization_layer_tag],
    beha_dict=beha_dict,
    data_dict=data_dict,
    hardware_platform=hardware_platform,
    event_dict=event_dict,
    dijkstra_routing=dijkstra_routing,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
)


time_cost = event_driver(
    events_dict=event_dict,
    hardware_platform=hardware_platform
)
print(f"SA time cost: {{time_cost}}")
with open(os.path.join(file_path, "../results/{py_name}.txt"), "w") as result_f:
    result_f.write(f"SA time cost: {{time_cost}}")


records_lists, device_names = devicedict_to_showlist(hardware_platform.modules_dict["tensorcore"])
records_lists, device_names = devicedict_to_showlist(hardware_platform.modules_dict["vectorunit"], records_lists, device_names)
records_lists, device_names = devicedict_to_showlist(hardware_platform.links_dict, records_lists, device_names)

pkl_name = f"../results/{py_name}_records_lists.pkl"
pkl_path = os.path.join(file_path, pkl_name)
with open(pkl_path, 'wb') as f:
    pickle.dump((records_lists, device_names), f)
'''

sql_list = [512, 2048]


output_dir = os.path.join(file_path, './py/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for ch_shape_idx, [ch_row, ch_col] in enumerate(ch_shapes):
    ch_rc = int(ch_row * ch_col)
    for co_shape_idx, [co_row, co_col] in enumerate(co_shapes):
        co_rc = int(co_row * co_col)
        for tensorcore_grain in tesnorcore_grain_list:
            for ch_bw in ch_bws:
                for co_bw in co_bws:
                    for fail_idx, (failed_nodes, failed_links) in enumerate(failures):
                        cfg_name = f'config_ch{ch_row}x{ch_col}_bw{ch_bw}_co{co_row}x{co_col}_bw{co_bw}_t{tensorcore_grain[0]}x{tensorcore_grain[1]}_failpattern{fail_idx}'
                        for sq in sql_list:
                            py_name = f'ln_sq{sq}_gemini_ch{ch_row}x{ch_col}_bw{ch_bw}_co{co_row}x{co_col}_bw{co_bw}_t{tensorcore_grain[0]}x{tensorcore_grain[1]}_failpattern{fail_idx}'
                            content = gemini_template.format(
                                cfg_name=cfg_name,
                                sq=sq,
                                loss_ratio=loss_ratio,
                                py_name=py_name
                            )
                            path = os.path.join(output_dir, py_name+".py")
                            with open(path, 'w') as f:
                                f.write(content)
                            print(f'Written {path}')
