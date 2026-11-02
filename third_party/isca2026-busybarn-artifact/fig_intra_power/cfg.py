
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))


template = '''[time_unit]
frequency = 1

[topology]
chiplet_number = {chiplet_number}
chiplet_rows = {chiplet_rows}
chiplet_columns = {chiplet_columns}
core_number = {core_number}
core_rows = {core_rows}
core_columns = {core_columns}
tensorcore_number = 1
vectorunit_number = 1
sram_number = 1
ddr_number = 1
topology_type = tlm

[tensorcore_cfg]
tensorcore_type = multree
tensorcore_grain = {tensorcore_grain}
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
ch2ch_bandwidth = {ch2ch_bandwidth}
ch2ch_timeunit = 1

[sram_cfg]
# 16 MB
sram_capacity = 16777216

[ddr_cfg]
# 32 GB = 4 * HBM2
ddr_capacity = 34359738368

[co2ddr_cfg]
co2ddr_latency = 50
co2ddr_bandwidth = {co2co_bandwidth}
co2ddr_timeunit = 1

[ddr2co_cfg]
ddr2co_latency = 50
ddr2co_bandwidth = {co2co_bandwidth}
ddr2co_timeunit = 1


[failures]
failed_nodes = {failed_nodes}
failed_links = {failed_links}
'''  

ch_shapes = [ 
    (1, 1),
]

co_shapes = [
    (4, 4),
]

tesnorcore_grain_list = [
    [64, 64],
    [128, 64],
    [128, 128],
 ]

co_bws = [
    256,
]

ch_bws = [
    256,
]


failures = [
    ([], []),
    ([(0, 5)], []),
]


output_dir = os.path.join(file_path, './cfg/')
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
                        content = template.format(
                            chiplet_number=ch_rc,
                            chiplet_rows=ch_row,
                            chiplet_columns=ch_col,
                            core_number=co_rc,
                            core_rows=co_row,
                            core_columns=co_col,
                            tensorcore_grain=tensorcore_grain,
                            co2co_bandwidth=co_bw,
                            ch2ch_bandwidth=ch_bw,
                            failed_nodes=failed_nodes,
                            failed_links=failed_links
                        )
                        filename = f'config_ch{ch_row}x{ch_col}_bw{ch_bw}_co{co_row}x{co_col}_bw{co_bw}_t{tensorcore_grain[0]}x{tensorcore_grain[1]}_failpattern{fail_idx}.cfg'
                        path = os.path.join(output_dir, filename)
                        with open(path, 'w') as f:
                            f.write(content)
                        print(f'Written {path}')


print('All configuration files generated.')
