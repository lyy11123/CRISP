

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))


dij_template = '''import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../../src/scheduling/communication/topology/'))
sys.path.append(os.path.join(file_path, '../../utils/'))
sys.path.append(os.path.join(file_path, '../../src/backend/analytical/'))
sys.path.append(os.path.join(file_path, '../../src/scheduling/'))


from WAMIS_HD import wamis_hdc
from read_cfg import cfg_to_dict
from event_driver import event_driver
from event_notation import communication_notation


import random
random.seed(123)
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pickle


communication_size = {communication_size}
hardware_cfg = cfg_to_dict(os.path.join(file_path, "../cfg/{cfg_name}.cfg"))
network = wamis_hdc(hardware_cfg)

chunk_number = 1
allgather_pairs_list = network.allgather_pairs(package_size=communication_size, chunk_number=chunk_number, ch_set={{0}})
max_link_load, total_link_load, paths, task_dict = network.record_dijkstra_multicast_path(
    comm_pairs=allgather_pairs_list,
    alpha=100, beta=1, gamma=100,
)
# for path in paths:
#     print(paths[path])
# raise
max_link_load, total_link_load, paths, task_dict = network.iter_worse_tasks(
    original_comm_pairs_dict=task_dict,
    original_tag_paths_dict=paths,
    total_iterations=1000,
    dijkstra_paras={{
        "alpha": 1,
        "beta": 100,
        "gamma": 1,
        "shortest_init": True,
        "long_first": False,
        "perdist": False,
        "pertask": False,
        "funcdist": False,
    }}
)

event_dict = {{}}
for path in paths:
    event_dict[path[1]] = communication_notation(
        comm_name=path[1],
        comm_tag=path[1],
        source_location=task_dict[path][1],
        target_location=task_dict[path][2],
        comm_bytes=(communication_size//network.l1_number//chunk_number)
    )
    event_dict[path[1]].path_list = event_dict[path[1]].get_paths(paths[path])

min_time = np.inf
for scheduling_iter in tqdm(range(100)):
    network.update_topology()
    scheduling_iter_event_dict = deepcopy(event_dict)
    time_cost = event_driver(
        events_dict=scheduling_iter_event_dict,
        hardware_platform=network
    )
    if time_cost[0] < min_time:
        min_time = time_cost[0]
        target_devices = deepcopy(network.links_dict)
print("time_cost: ", min_time, "with ", (min_time / network.chunk_cost))

pickle.dump(
    target_devices,
    open(os.path.join(file_path, "../results/{py_name}.pkl"), "wb")
)

with open(os.path.join(file_path, "../results/{py_name}.txt"), "w", encoding="utf-8") as f:
    f.write(str(int(min_time)))

'''


xy_template = '''
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../../src/scheduling/communication/topology/'))
sys.path.append(os.path.join(file_path, '../../utils/'))
sys.path.append(os.path.join(file_path, '../../src/backend/analytical/'))
sys.path.append(os.path.join(file_path, '../../src/scheduling/'))


from WAMIS_HD import wamis_hdc
from read_cfg import cfg_to_dict
from event_driver import collective_event_driver, event_driver
from event_notation import event_notation, communication_notation


import random
random.seed(123)
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pickle


communication_size = {communication_size}
hardware_cfg = cfg_to_dict(os.path.join(file_path, "../cfg/{cfg_name}.cfg"))
network = wamis_hdc(hardware_cfg)

chunk_number = 1
allgather_pairs_list = network.allgather_pairs(package_size=communication_size, chunk_number=chunk_number, ch_set={{0}})

event_dict = {{}}
for pair in allgather_pairs_list:
    event_dict[pair[0]] = communication_notation(
    comm_name=pair[0],
    comm_tag=pair[0],
    source_location=pair[1],
    target_location=pair[2],
    comm_bytes=(communication_size//network.l1_number//chunk_number)
    )
    event_dict[pair[0]].path_list = event_dict[pair[0]].get_paths(network.xy_multicast_path(pair[1], pair[2]))

min_time = np.inf
for scheduling_iter in tqdm(range(10)):
    network.update_topology()
    scheduling_iter_event_dict = deepcopy(event_dict)
    time_cost = event_driver(
        events_dict=scheduling_iter_event_dict,
        hardware_platform=network
    )
    if time_cost[0] < min_time:
        min_time = time_cost[0]
        target_devices = deepcopy(network.links_dict)
print("time_cost: ", min_time, "with ", (min_time / network.chunk_cost))

pickle.dump(
    target_devices,
    open(os.path.join(file_path, "../results/{py_name}.pkl"), "wb")
)

with open(os.path.join(file_path, "../results/{py_name}.txt"), "w", encoding="utf-8") as f:
    f.write(str(int(min_time)))

'''


message_sizes = [
        1024,
        4 * 1024,
        16 * 1024,
        64 * 1024,
        256 * 1024,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
        1024 * 1024 * 1024,
        4 * 1024 * 1024 * 1024,
        16 * 1024 * 1024 * 1024,
]

message_str = [
    "1KB",
    "4KB",
    "16KB",
    "64KB",
    "256KB",
    "1MB",
    "4MB",
    "16MB",
    "64MB",
    "256MB",
    "1GB",
    "4GB",
    "16GB"
]

shapes = [
    (5, 5),
    (4, 6),
    (6, 4),
]

bws = [
    256,
]

shapes_failures = [
    [([], []),
    ([(0, 12)], []),
    ([], [((0, 12), (0, 13)), ((0, 13), (0, 12))]),],
    [([], []),
    ([(0, 8)], []),
    ([], [((0, 8), (0, 9)), ((0, 9), (0, 8))]),],
    [([], []),
    ([(0, 13)], []),
    ([], [((0, 13), (0, 14)), ((0, 14), (0, 13))]),],        
]

algorithms = ['dij', 'xy']


output_dir = os.path.join(file_path, './py/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for shape_idx, [r, c] in enumerate(shapes):
    rc = int(r * c)
    for bw in bws:
        failures = shapes_failures[shape_idx]
        for fail_idx, (failed_nodes, failed_links) in enumerate(failures):
            for communication_idx, communication_size in enumerate(message_sizes):
                for algorithm in algorithms:
                    if algorithm == 'dij':
                        template = dij_template
                    else:
                        template = xy_template
                    cfg_name = f'config_{r}x{c}_bw{bw}_failpatterm{fail_idx}'
                    py_name = f'allgather_{r}x{c}_bw{bw}_failpatterm{fail_idx}_{message_str[communication_idx]}_{algorithm}'
                    content = template.format(
                        communication_size=communication_size,
                        cfg_name=cfg_name,
                        py_name=py_name
                    )
                    path = os.path.join(output_dir, py_name+".py")
                    with open(path, 'w') as f:
                        f.write(content)
                    print(f'Written {path}')


ring_template = '''import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../../src/scheduling/communication/topology/'))
sys.path.append(os.path.join(file_path, '../../utils/'))
sys.path.append(os.path.join(file_path, '../../src/backend/analytical/'))
sys.path.append(os.path.join(file_path, '../../src/scheduling/'))


from WAMIS_HD import wamis_hdc
from read_cfg import cfg_to_dict
from event_driver import collective_event_driver, event_driver
from event_notation import event_notation, communication_notation


import random
random.seed(123)
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pickle


communication_size = {communication_size}
hardware_cfg = cfg_to_dict(os.path.join(file_path, "../cfg/{cfg_name}.cfg"))
network = wamis_hdc(hardware_cfg)

task_idx = 0
event_dict = {{}}

# x first
last_step = set()
current_step = set()
for step_x in range(network.l1_width - 1):
    last_step = current_step
    current_step = set()
    for step_y in range(network.l1_height):
        source_y = step_y
        target_y = step_y
        for x_idx in range(network.l1_width):
            source_x = x_idx
            target_x = (x_idx + 1) % network.l1_width
            source_idx = (0, int(source_x + source_y * network.l1_width))
            target_idx = (0, int(target_x + target_y * network.l1_width))
            task_id = (1, task_idx)
            event_dict[task_id] = communication_notation(
                comm_name=task_id,
                comm_tag=task_id,
                source_location=source_idx,
                target_location=target_idx,
                comm_bytes=(np.ceil(communication_size / network.l1_number))
            )
            event_dict[task_id].path_list = event_dict[task_id].get_paths(network.xy_multicast_path(source_idx, {{target_idx}}))
            for producer in last_step:
                event_dict[task_id].dependency_set.add(producer)
                event_dict[producer].issue_set.add(task_id)
            current_step.add(task_id)
            task_idx += 1

for step_y in range(network.l1_height - 1):
    last_step = current_step
    current_step = set()
    for step_x in range(network.l1_width):
        source_x = step_x
        target_x = step_x
        for y_idx in range(network.l1_height):
            source_y = y_idx
            target_y = (y_idx + 1) % network.l1_height
            source_idx = (0, int(source_x + source_y * network.l1_width))
            target_idx = (0, int(target_x + target_y * network.l1_width))
            task_id = (1, task_idx)
            event_dict[task_id] = communication_notation(
                comm_name=task_id,
                comm_tag=task_id,
                source_location=source_idx,
                target_location=target_idx,
                comm_bytes=(np.ceil(communication_size / network.l1_height))
            )
            event_dict[task_id].path_list = event_dict[task_id].get_paths(network.xy_multicast_path(source_idx, {{target_idx}}))
            for producer in last_step:
                event_dict[task_id].dependency_set.add(producer)
                event_dict[producer].issue_set.add(task_id)
            current_step.add(task_id)
            task_idx += 1

min_time = np.inf
for scheduling_iter in tqdm(range(1)):
    network.update_topology()
    scheduling_iter_event_dict = deepcopy(event_dict)
    time_cost = event_driver(
        events_dict=scheduling_iter_event_dict,
        hardware_platform=network
    )
    if time_cost[0] < min_time:
        min_time = time_cost[0]
        target_devices = deepcopy(network.links_dict)
print("time_cost: ", min_time)

pickle.dump(
    target_devices,
    open(os.path.join(file_path, "../results/{py_name}.pkl"), "wb")
)

with open(os.path.join(file_path, "../results/{py_name}.txt"), "w", encoding="utf-8") as f:
    f.write(str(int(min_time)))
'''

message_sizes = [
        1024,
        4 * 1024,
        16 * 1024,
        64 * 1024,
        256 * 1024,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
        1024 * 1024 * 1024,
        4 * 1024 * 1024 * 1024,
        16 * 1024 * 1024 * 1024,
]

message_str = [
    "1KB",
    "4KB",
    "16KB",
    "64KB",
    "256KB",
    "1MB",
    "4MB",
    "16MB",
    "64MB",
    "256MB",
    "1GB",
    "4GB",
    "16GB"
]

shapes = [
    (5, 5),
]

bws = [
    256,
]

shapes_failures = [
    [([], []),]      
]

algorithms = ['ring']

output_dir = os.path.join(file_path, './py/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for shape_idx, [r, c] in enumerate(shapes):
    rc = int(r * c)
    for bw in bws:
        failures = shapes_failures[shape_idx]
        for fail_idx, (failed_nodes, failed_links) in enumerate(failures):
            for communication_idx, communication_size in enumerate(message_sizes):
                for algorithm in algorithms:
                    if algorithm == 'ring':
                        template = ring_template
                    else:
                        raise ValueError(f"Unknown algorithm: {algorithm}")
                    cfg_name = f'config_{r}x{c}_bw{bw}_failpatterm{fail_idx}'
                    py_name = f'allgather_{r}x{c}_bw{bw}_failpatterm{fail_idx}_{message_str[communication_idx]}_{algorithm}'
                    content = template.format(
                        communication_size=communication_size,
                        cfg_name=cfg_name,
                        py_name=py_name
                    )
                    path = os.path.join(output_dir, py_name+".py")
                    with open(path, 'w') as f:
                        f.write(content)
                    print(f'Written {path}')


print('All python files generated.')
