
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
vectorunit_grain = {vectorunit_grain}
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
co2co_bandwidth = {bandwidth}
co2co_timeunit = 1

[ch2ch_cfg]
ch2ch_latency = 20
ch2ch_bandwidth = {bandwidth}
ch2ch_timeunit = 1

[sram_cfg]
# 16 MB
sram_capacity = 16777216

[ddr_cfg]
# 32 GB = 4 * HBM2
ddr_capacity = 34359738368

[co2ddr_cfg]
co2ddr_latency = 50
co2ddr_bandwidth = {bandwidth}
co2ddr_timeunit = 1

[ddr2co_cfg]
ddr2co_latency = 50
ddr2co_bandwidth = {bandwidth}
ddr2co_timeunit = 1


[failures]
failed_nodes = {failed_nodes}
failed_links = {failed_links}
'''

ch_shapes = [
    (2, 2),
    (1, 4),
]

core_configs = [
    ((16, 16), [32, 16],  [32],  1),
    ((8, 8),   [32, 64],  [128], 2),
    ((4, 4),   [64, 128], [512], 4),
]

base_bws = [64, 96, 128]

failures = [
    ([], []),
]


output_dir = os.path.join(file_path, "cfg")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for ch_row, ch_col in ch_shapes:
    ch_rc = int(ch_row * ch_col)
    for (co_row, co_col), tc_grain, vu_grain, bw_scale in core_configs:
        co_rc = int(co_row * co_col)
        for base_bw in base_bws:
            bw = base_bw * bw_scale
            for fail_idx, (failed_nodes, failed_links) in enumerate(failures):
                content = template.format(
                    chiplet_number=ch_rc,
                    chiplet_rows=ch_row,
                    chiplet_columns=ch_col,
                    core_number=co_rc,
                    core_rows=co_row,
                    core_columns=co_col,
                    tensorcore_grain=tc_grain,
                    vectorunit_grain=vu_grain,
                    bandwidth=bw,
                    failed_nodes=failed_nodes,
                    failed_links=failed_links,
                )
                filename = f'config_ch{ch_row}x{ch_col}_bw{bw}_co16x16_bw{bw}_t{co_row}x{co_col}.cfg'
                if fail_idx > 0:
                    filename = f'config_ch{ch_row}x{ch_col}_bw{bw}_co16x16_bw{bw}_t{co_row}x{co_col}_failpattern{fail_idx}.cfg'
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)
                print(f'Written {filepath}')


print('All configuration files generated.')
