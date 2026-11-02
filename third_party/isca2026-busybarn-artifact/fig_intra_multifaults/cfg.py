
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))


from makefaults import generate_cluster_failures


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
    (20, 20),
]

tesnorcore_grain_list = [
    [64, 64],
 ]

co_bws = [
    256,
]

ch_bws = [
    256,
]


failures = [
    ([
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25),
        (0, 40), (0, 41), (0, 42), (0, 43), (0, 44), (0, 45),
        (0, 60), (0, 61), (0, 62), (0, 63), (0, 64), (0, 65),
        (0, 80), (0, 81), (0, 82), (0, 83), (0, 84), (0, 85),
        (0, 100), (0, 101), (0, 102), (0, 103), (0, 104), (0, 105),
        (0, 121), (0, 122), (0, 123), (0, 124),
        (0, 400), (0, 401)
    ], []),
    ([(0, 4), (1, 0), (2, 9), (3, 6), (4, 7), (5, 6), (6, 0), (7, 0), (8, 2), (9, 3), (10, 3), (11, 12), (12, 16), (13, 5), (14, 1), (15, 17), (16, 5), (17, 2), (18, 12), (19, 13), (0, 8), (1, 0), (2, 17), (3, 2), (4, 12), (5, 14), (6, 10), (7, 11), (8, 15), (9, 14), (10, 17), (11, 13), (12, 16), (13, 15), (14, 0), (15, 9), (16, 10), (17, 15), (18, 5), (19, 10)], []),
    ([
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), 
        (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (0, 27),
        (0, 40), (0, 41), (0, 42), (0, 43), (0, 44), (0, 45), (0, 46), (0, 47),
        (0, 60), (0, 61), (0, 62), (0, 63), (0, 64), (0, 65), (0, 66), (0, 67),
        (0, 80), (0, 81), (0, 82), (0, 83), (0, 84), (0, 85), (0, 86), (0, 87),
        (0, 100), (0, 101), (0, 102), (0, 103), (0, 104), (0, 105), (0, 106), (0, 107),
        (0, 120), (0, 121), (0, 122), (0, 123), (0, 124), (0, 125), (0, 126), (0, 127),
        (0, 142), (0, 143), (0, 144), (0, 145),
        (0, 400), (0, 401)
    ], []),
    ([(0, 4), (1, 0), (2, 9), (3, 6), (4, 7), (5, 6), (6, 0), (7, 0), (8, 2), (9, 3), (10, 3), (11, 12), (12, 16), (13, 5), (14, 1), (15, 17), (16, 5), (17, 2), (18, 12), (19, 13), (0, 8), (1, 0), (2, 17), (3, 2), (4, 12), (5, 14), (6, 10), (7, 11), (8, 15), (9, 14), (10, 17), (11, 13), (12, 16), (13, 15), (14, 0), (15, 9), (16, 10), (17, 15), (18, 5), (19, 10), (0, 16), (1, 1), (2, 5), (3, 19), (4, 15), (5, 5), (6, 13), (7, 3), (8, 16), (9, 19), (10, 0), (11, 15), (12, 6), (13, 6), (14, 14), (15, 15), (16, 16), (17, 1), (18, 16), (19, 15)], []),
    ([
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), 
        (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (0, 27), (0, 28), 
        (0, 40), (0, 41), (0, 42), (0, 43), (0, 44), (0, 45), (0, 46), (0, 47), (0, 48), 
        (0, 60), (0, 61), (0, 62), (0, 63), (0, 64), (0, 65), (0, 66), (0, 67), (0, 68), 
        (0, 80), (0, 81), (0, 82), (0, 83), (0, 84), (0, 85), (0, 86), (0, 87), (0, 88), 
        (0, 100), (0, 101), (0, 102), (0, 103), (0, 104), (0, 105), (0, 106), (0, 107), (0, 108),
        (0, 120), (0, 121), (0, 122), (0, 123), (0, 124), (0, 125), (0, 126), (0, 127), (0, 128),
        (0, 140), (0, 141), (0, 142), (0, 143), (0, 144), (0, 145), (0, 146), (0, 147), (0, 148),
        (0, 160), (0, 161), (0, 162), (0, 163), (0, 164), (0, 165), (0, 166), (0, 167), (0, 168),
        (0, 180),
        (0, 400), (0, 401)
    ], []),
    ([(0, 4), (1, 0), (2, 9), (3, 6), (4, 7), (5, 6), (6, 0), (7, 0), (8, 2), (9, 3), (10, 3), (11, 12), (12, 16), (13, 5), (14, 1), (15, 17), (16, 5), (17, 2), (18, 12), (19, 13), (0, 8), (1, 0), (2, 17), (3, 2), (4, 12), (5, 14), (6, 10), (7, 11), (8, 15), (9, 14), (10, 17), (11, 13), (12, 16), (13, 15), (14, 0), (15, 9), (16, 10), (17, 15), (18, 5), (19, 10), (0, 16), (1, 1), (2, 5), (3, 19), (4, 15), (5, 5), (6, 13), (7, 3), (8, 16), (9, 19), (10, 0), (11, 15), (12, 6), (13, 6), (14, 14), (15, 15), (16, 16), (17, 1), (18, 16), (19, 15), (0, 4), (1, 10), (2, 16), (3, 9), (4, 2), (5, 9), (6, 0), (7, 13), (8, 18), (9, 13), (10, 14), (11, 0), (12, 3), (13, 6), (14, 14), (15, 16), (16, 10), (17, 6), (18, 19), (19, 3)], []),
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
