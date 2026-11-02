
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))


template = '''[time_unit]
frequency = 1

[topology]
chiplet_number = 1
chiplet_rows = 1
chiplet_columns = 1
core_number = {core_number}
core_rows = {core_rows}
core_columns = {core_columns}
tensorcore_number = 1
vectorunit_number = 1
topology_type = tlm

[tensorcore_cfg]
tensorcore_type = multree
tensorcore_grain = [128, 64]
tensorcore_timeunit = 1

[vectorunit_cfg]
vectorunit_type = simd
vectorunit_grain = [1024]
vectorunit_timeunit = 1
eleadd_complexity = 1
eleexp_complexity = 2
elegelu_complexity = 8
elemul_complexity = 1
elepow2_complexity = 1
elerelu_complexity = 2
elesqrt_complexity = 4
matadd_complexity = 1
redmax_complexity = 1
redsum_complexity = 1
vecadd_complexity = 1
vecdiv_complexity = 4
vecmac_complexity = 1
vecmul_complexity = 1
transpose_complexity = 1
lookup_complexity = 1

[co2co_cfg]
co2co_latency = 1
co2co_bandwidth = {co2co_bandwidth}
co2co_timeunit = 1

[ch2ch_cfg]
ch2ch_latency = 20
ch2ch_bandwidth = 128
ch2ch_timeunit = 1

[failures]
failed_nodes = {failed_nodes}
failed_links = {failed_links}
'''  

shapes = [
    (5, 5),
    (4, 6),
    (6, 4),
]

bws = [
    128,
    256,
    512,
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


output_dir = os.path.join(file_path, './cfg/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for shape_idx, [r, c] in enumerate(shapes):
    rc = int(r * c)
    for bw in bws:
        failures = shapes_failures[shape_idx]
        for fail_idx, (failed_nodes, failed_links) in enumerate(failures):
            content = template.format(
                core_rows=r,
                core_columns=c,
                core_number=rc,
                co2co_bandwidth=bw,
                failed_nodes=failed_nodes,
                failed_links=failed_links
            )
            filename = f'config_{r}x{c}_bw{bw}_failpatterm{fail_idx}.cfg'
            path = os.path.join(output_dir, filename)
            with open(path, 'w') as f:
                f.write(content)
            print(f'Written {path}')

print('All configuration files generated.')
