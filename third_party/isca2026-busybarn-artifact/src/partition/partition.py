
"""Partition utilities: dimension splitting, degree generation, and workload
distribution across parallelism strategies (TP, PP, CP)."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))


from typing import List, Dict, Tuple, Set
import numpy as np
import random


def generate_average_degree(
    total: int,
    part_num: int,
    shuffle: bool = False
):
    # Divide total evenly into part_num pieces; remainder distributed to first (or random) slots
    base_value = total // part_num
    remainder = total % part_num

    result = [base_value] * part_num

    if shuffle:
        indices = list(range(part_num))
        random.shuffle(indices)
        for i in range(remainder):
            result[indices[i]] += 1
    else:
        for i in range(remainder):
            result[i] += 1

    return tuple(result)


def generate_random_degree(total: int, part_num: int) -> tuple:
    # Randomly split total into part_num positive integers that sum to total
    partitions = []
    remaining_total = total - part_num

    for i in range(part_num - 1):
        current_part = random.randint(0, remaining_total)
        partitions.append(current_part)
        remaining_total -= current_part 

    partitions.append(remaining_total)
    partitions = [x + 1 for x in partitions]
    random.shuffle(partitions)

    return tuple(partitions)


def regenerate_degree(input_tuple: tuple, limit: int) -> tuple:
    # Perturb two randomly chosen entries by a random delta (SA neighbor generation)
    if len(input_tuple) < 2:
        raise ValueError("tuple length should be greater than 1")

    index1, index2 = random.sample(range(len(input_tuple)), 2)
    num1, num2 = input_tuple[index1], input_tuple[index2]

    changed_data = random.randint(0, limit)
    while changed_data == 0 or changed_data > num1 or changed_data > num2:
        changed_data = random.randint(0, limit)    
    new_num1 = num1 + random.choice([changed_data, -changed_data])
    new_num2 = num2 + random.choice([changed_data, -changed_data])

    result = list(input_tuple)
    result[index1] = new_num1
    result[index2] = new_num2

    return tuple(result)


def whole_degree_to_dim_degrees(parallel_degree: int, dimension_num: int) -> List[Tuple[int]]:
    # Enumerate all ways to factor parallel_degree into dimension_num factors
    def find_factors(n):
        factors = []
        for i in range(1, n+1):
            if n % i == 0:
                factors.append(i)
        return factors

    factors = find_factors(parallel_degree)

    def get_combinations(prod, n, current_combination):
        if len(current_combination) == n:
            if prod == parallel_degree:
                yield tuple(current_combination)
            return
        for factor in factors:
            if prod * factor <= parallel_degree:
                yield from get_combinations(prod * factor, n, current_combination + [factor])

    return list(get_combinations(1, dimension_num, []))


def dim_degree_to_split(data_shape: List[int], dim_degree: Tuple[int], mode: str="avg") -> List[Tuple[int]]:
    # Convert per-dimension parallelism degrees into concrete split sizes for each dim
    dim_length = len(data_shape)
    if dim_length != len(dim_degree):
        raise ValueError("data shape and dim degrees should have the same length")

    result = []
    if mode == "avg":
        for i in range(dim_length):
            splits = generate_average_degree(data_shape[i], dim_degree[i])
            result.append(splits)
    else:
        for i in range(dim_length):
            result.append(generate_random_degree(data_shape[i], dim_degree[i]))

    return result


if __name__ == '__main__':

    sys.path.append(os.path.join(file_path, "../../utils/"))

    import logging
    from logprint import setup_custom_levels, setup_logging_levels, log_message, logger

    test_normal_log_level = 1
    test_loop_log_level = 2
    setup_custom_levels([test_normal_log_level, test_loop_log_level])
    setup_logging_levels([test_normal_log_level, test_loop_log_level])

    total = 1024
    part_num = 5
    avg_degree = generate_average_degree(total, part_num)
    log_message(test_normal_log_level, f"Average degree: {avg_degree}")
    rand_degree = generate_random_degree(total, part_num)
    log_message(test_normal_log_level, f"Random degree: {rand_degree}")
    changed_degree = regenerate_degree(avg_degree, total)
    log_message(test_normal_log_level, f"Changed degree: {changed_degree}")


    batch_size = 16
    sequence_length = 256
    hidden_states = 768

    dim_num = 3
    parallel_degrees = 100
    dim_degrees_list = whole_degree_to_dim_degrees(parallel_degrees, dim_num)
    log_message(test_loop_log_level, f"Dimension degrees: {dim_degrees_list}")
