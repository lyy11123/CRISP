
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))


import math
from typing import List

def proportional_split(weights: List[int], total: int) -> List[int]:
    """
    Split ``total`` into an integer list proportional to ``weights`` while
    keeping ``sum(result) == total`` and minimizing the differences among
    the implied scaling ratios ``result[i] / weights[i]``.

    Algorithm:
    1. Compute ``sum_w = sum(weights)``.
    2. Compute the ideal scaling factor ``factor = total / sum_w``.
    3. For each weight ``w_i``, compute the ideal floating allocation
       ``x_i = w_i * factor``.
    4. Take the floor of each value, ``f_i = floor(x_i)``, so
       ``floor_sum = sum(f_i) <= total``.
    5. Let ``rem = total - floor_sum`` and add 1 to the ``rem`` positions
       with the largest fractional parts ``(x_i - f_i)``.
    6. Return the final integer list.
    """
    n = len(weights)
    if total < 0:
        raise ValueError("total must be non-negative")
    sum_w = sum(weights)
    if sum_w == 0:
        return [0] * n

    factor = total / sum_w

    ideals = [w * factor for w in weights]
    floors = [math.floor(x) for x in ideals]
    result = floors.copy()

    rem = total - sum(floors)
    fracs = [(ideals[i] - floors[i], i) for i in range(n)]
    fracs_sorted = sorted(fracs, key=lambda x: x[0], reverse=True)

    for frac, idx in fracs_sorted[:rem]:
        result[idx] += 1

    return result

if __name__ == "__main__":
    weights = [4, 4, 4, 4, 4, 1]
    total = 44
    allocation = proportional_split(weights, total)
    print(allocation)
