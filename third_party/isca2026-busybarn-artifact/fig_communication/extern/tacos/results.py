
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))
import numpy as np


message_sizes = [
    1024, 4096, 16384, 65536, 262144,
    1048576, 4194304, 16777216, 67108864, 268435456,
    1073741824, 4294967296, 17179869184
]

alpha = 1
beta = 256

r_list = []
for m in message_sizes:
    p = np.ceil(m / 25)
    t = alpha + np.ceil(p / beta)
    t = t * 12
    r_list.append(int(t))

print(r_list)
