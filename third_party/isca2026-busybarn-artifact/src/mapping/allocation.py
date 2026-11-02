
"""Resource allocation utilities: layer-to-chiplet assignment, rectangular
decomposition for die-group shapes, and mesh placement algorithms."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../components/'))
sys.path.append(os.path.join(file_path, '../partition/'))


from notation import data_notation, data_parts, op_notation, function_notation, type_bytes
from Device import Device, ComputationDevice, Tensorcore, Vectorunit


from typing import List, Dict, Tuple, Set
from itertools import groupby
import multiprocessing
from functools import partial
import numpy as np
import math
from tqdm import tqdm


def find_layer_per_chiplet(layer_number: int, chiplet_number: int):
    '''
    eg: find_layer_per_chiplet(36, 25)
        -> 
        24, 3, 2
    '''
    max_gcd = 0
    best_num = chiplet_number
    for i in range(chiplet_number, 0, -1):
        current_gcd = math.gcd(layer_number, i)
        if current_gcd > max_gcd:
            max_gcd = current_gcd
            best_num = i
        if current_gcd < max_gcd:
            break
    used_chiplets_number = best_num
    layer_group = layer_number // max_gcd
    chiplet_group = used_chiplets_number // max_gcd
    return used_chiplets_number, layer_group, chiplet_group


def split_list_by_sizes(lst, sizes):
    result = []
    start = 0
    for size in sizes:
        if start < len(lst): 
            end = start + size
            result.append(lst[start:end])
            start = end
        else:
            break 
    return result


def zigzag_allocation(
    num_x: int, num_y: int, 
    true_number: int = 0, x_first: bool = True
):
    def zigzag_indices(x_dim, y_dim, x_first=True): 
        indices = [] 
        if x_first:
            for y in range(y_dim): 
                if y % 2 == 0:
                    for x in range(x_dim): 
                        index = y * x_dim + x 
                        indices.append(index) 
                else:  
                    for x in range(x_dim - 1, -1, -1): 
                        index = y * x_dim + x 
                        indices.append(index) 
        else:
            for x in range(x_dim): 
                if x % 2 == 0:
                    for y in range(y_dim): 
                        index = y * x_dim + x 
                        indices.append(index) 
                else:  
                    for y in range(y_dim - 1, -1, -1): 
                        index = y * x_dim + x 
                        indices.append(index)
        return indices

    if true_number == 0:
        true_number = int(num_x * num_y)
    zigzag_list = zigzag_indices(num_x, num_y, x_first)

    cluster_path = []
    for node in zigzag_list[0:true_number]:
        cluster_path.append([node])

    return cluster_path


def hamiltonian_allocation(
    num_x: int, num_y: int, 
    x_first: bool = True
):

    x_odd = num_x % 2 == 1
    y_odd = num_y % 2 == 1

    if x_odd and y_odd:
        x_first = x_first
    elif y_odd:
        x_first = False
    elif x_odd:
        x_first = True
    else:
        x_first = x_first

    path = []

    if x_odd and y_odd:
        if x_first:
            for y in range(num_y):
                if y % 2 == 0:
                    if y == 0:
                        for x in range(num_x):
                            node = y * num_x + x
                            path.append(node)
                    elif y < num_y - 2:
                        for x in range(1, num_x):
                            node = y * num_x + x
                            path.append(node)
                    else:
                        y = num_y - 2
                        x = num_x - 1
                        node = y * num_x + x
                        path.append(node)
                        while x > 0:
                            x -= 1
                            node = y * num_x + x
                            path.append(node)
                            y += 1 
                            node = y * num_x + x
                            path.append(node)
                            x -= 1
                            node = y * num_x + x
                            path.append(node)
                            if x == 0:
                                break
                            else:
                                y -= 1
                                node = y * num_x + x
                                path.append(node)                       

                else:
                    if y < num_y - 2:
                        for x in range(num_x - 1, 0, -1):
                            node = y * num_x + x
                            path.append(node)
            x = 0
            for y in range(num_y - 2, 0, -1):
                node = y * num_x + x
                path.append(node)

        else:
            for x in range(num_x):
                if x % 2 == 0:
                    if x == 0:
                        for y in range(num_y):
                            node = y * num_x + x
                            path.append(node)
                    elif x < num_x - 2:
                        for y in range(1, num_y):
                            node = y * num_x + x
                            path.append(node)
                    else:
                        x = num_x - 2
                        y = num_y - 1
                        node = y * num_x + x
                        path.append(node)
                        while y > 0:
                            y -= 1
                            node = y * num_x + x
                            path.append(node)
                            x += 1 
                            node = y * num_x + x
                            path.append(node)
                            y -= 1
                            node = y * num_x + x
                            path.append(node)
                            if y == 0:
                                break
                            else:
                                x -= 1
                                node = y * num_x + x
                                path.append(node)                       

                else:
                    if x < num_x - 2:
                        for y in range(num_y - 1, 0, -1):
                            node = y * num_x + x
                            path.append(node)
            y = 0
            for x in range(num_x - 2, 0, -1):
                node = y * num_x + x
                path.append(node)

    else:
        if x_first:
            for y in range(num_y):
                if y % 2 == 0:
                    if y == 0:
                        for x in range(num_x):
                            node = y * num_x + x
                            path.append(node)
                    else:
                        for x in range(1, num_x):
                            node = y * num_x + x
                            path.append(node)
                else:
                    if y == num_y - 1:
                        for x in range(num_x - 1, -1, -1):
                            node = y * num_x + x
                            path.append(node)
                    else:
                        for x in range(num_x - 1, 0, -1):
                            node = y * num_x + x
                            path.append(node)

            x = 0
            for y in range(num_y - 2, 0, -1):
                node = y * num_x + x
                path.append(node)

        else:
            for x in range(num_x):
                if x % 2 == 0:
                    if x == 0:
                        for y in range(num_y):
                            node = y * num_x + x
                            path.append(node)
                    else:
                        for y in range(1, num_y):
                            node = y * num_x + x
                            path.append(node)
                else:
                    if x == num_x - 1:
                        for y in range(num_y - 1, -1, -1):
                            node = y * num_x + x
                            path.append(node)
                    else:
                        for y in range(num_y - 1, 0, -1):
                            node = y * num_x + x
                            path.append(node)

            y = 0
            for x in range(num_x - 2, 0, -1):
                node = y * num_x + x
                path.append(node)

    cluster_path = []
    for node in path:
        cluster_path.append([node])

    return cluster_path


def generate_rectangles(n: int, m: int, swap=True, side_limit: int = 4, points_limit: int = 6) -> list:
    '''
    This function will generate all possible rectangular shapes under an n x m rectangle.

    Args:
    n (int): height limitation of the rectangle 
    m (int): width limitation of the rectangle

    Returns:
    list of all possible rectangular shapes under an n x m rectangle

    Example:
    >>> generate_rectangles(2, 3)
    [(1, 2), (2, 1), (3, 1), (1, 1), (2, 3), (2, 2), (3, 2), (1, 3)]

    >>> generate_rectangles(2, 3, swap=False)
    [(1, 2), (2, 1), (1, 1), (2, 3), (2, 2), (1, 3)]
    '''

    # use set to avoid duplicate
    possible_rectangles = set()
    for height in range(1, n + 1):
        for width in range(1, m + 1):
            if height <= side_limit and width <= side_limit and height * width <= points_limit:
                possible_rectangles.add((height, width))
                if swap and width != height:
                    possible_rectangles.add((width, height))

    return list(possible_rectangles)


def find_combinations(rectangles: list, target: int, start=0, current=[], max_length=None) -> list: 
    '''
    Generate combinations that yield a total matching the target area.
    start and current is for recursion.

    Args:
    rectangles (list): list of all possible rectangular shapes under an n x m rectangle which is the reuslt of generate_rectangles()
    target (int): the whole number of nodes in the mesh

    Returns:
    iterator all possible combinations that yield a total matching the target area
    need to use list() to convert it to a list

    Example:
    >>> basic_combinations = list(find_combinations(possible_rectangles, n * m))
    [
     [(1, 2), (2, 1), (3, 2), (1, 3), (1, 3), (1, 3), (1, 3), (1, 3)], 
     [(1, 2), (2, 1), (3, 2), (3, 2), (1, 3), (1, 3), (1, 3)], 
     [(1, 2), (2, 1), (3, 2), (3, 2), (3, 2), (1, 3)
     ...
    ]
    '''

    if target == 0 and (max_length is None or len(current) <= max_length):
        yield list(current)
        return
    if start >= len(rectangles) or (max_length is not None and len(current) >= max_length):
        return
    # First, skip this rectangle.
    yield from find_combinations(rectangles, target, start + 1, current, max_length)
    width, height = rectangles[start]
    if width * height <= target:
        # Then, use this rectangle and keep the possibility of using it again.
        yield from find_combinations(rectangles, target - width * height, start, current + [rectangles[start]], max_length)


def remove_reverse_duplicates(lst):
    seen = set()
    result = []

    for sublst in lst:
        transpose_sublst = [reversed(subsublst) for subsublst in sublst]
        tup = tuple(sublst)
        reverse_tup = tuple(reversed(sublst))
        transpose_tup = tuple(transpose_sublst)
        transpose_reverse_tup = tuple(reversed(transpose_sublst))

        if tup not in seen and reverse_tup not in seen and transpose_tup not in seen and transpose_reverse_tup not in seen:
            result.append(sublst)
            seen.add(tup)
            seen.add(reverse_tup)
            seen.add(transpose_tup)
            seen.add(transpose_reverse_tup)

    return result


def filter_combinations_kickcontinuous(combinations: list) -> list:
    '''
    This function is used to eliminate combinations containing too much (1,1).

    Args:
    combinations: list of all combinations generated by top_n_percent_variance()

    Returns:
    hold the combinations that meet the requirements (no continuous (1,1) and no more than half (1,1))

    Example:
    >>> filter_combinations_kickone_includeone(variance_combinations)
    [
     [(1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)], 
     [(2, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (1, 3)], 
     [(2, 1), (3, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)], 
     ...
    ]
    '''

    def find_min_continuous(lst: list, target: list) -> int: 
        '''
        Counts the minimum number of consecutive occurrences of the target element in the given list.

        Args:
        lst: list of all combinations generated by top_n_percent_variance()
        target: the target element eg. (1,1)

        Returns:
        the number of continuous occurrences of the target element in the given list

        Example:
        >>> find_min_continuous(combs, (1,1))
        1
        '''

        continuous_counts = [len(list(group)) for key, group in groupby(lst) if key == target]
        filtered_counts = [count for count in continuous_counts if count > 1]
        return min(filtered_counts) if filtered_counts else 0

    filtedcombinations = []
    for combs in combinations:
        oneone_cnts = combs.count((1, 1))
        all_cnts = len(combs)
        # exclude combinations with too many (1,1), here cares about half of the total number
        if 2 * oneone_cnts < all_cnts:
            if find_min_continuous(combs, (1,1)) > 0 or find_min_continuous(combs, (1,2)) > 0 \
                or find_min_continuous(combs, (2,1)) > 0 or find_min_continuous(combs, (2,2)) > 0 \
                or find_min_continuous(combs, (3,1)) > 0 or find_min_continuous(combs, (1,3)) > 0 \
                or find_min_continuous(combs, (3,2)) > 0 or find_min_continuous(combs, (2,3)) > 0 \
                or find_min_continuous(combs, (4,1)) > 0 or find_min_continuous(combs, (1,4)) > 0:
                continue
            else:
                filtedcombinations.append(combs) 
        else:
            # NOTE: decide whether to hold so many (1,1)
            pass

    return filtedcombinations


def unique_permutations(lst: list) -> list:
    '''
    This function will generate all permutations possibilities for the combination.

    Args:
    lst (list): list of all possible rectangular shapes

    Returns:
    list of all permutations possibilities for the combination

    Example:
    >>> unique_permutations([(3, 1), (2, 2), (2, 2), (3, 2), (2, 2), (3, 2)])
    [
     [(3, 1), (2, 2), (2, 2), (2, 2), (3, 2), (3, 2)], 
     [(3, 1), (2, 2), (2, 2), (3, 2), (2, 2), (3, 2)], 
     [(3, 1), (2, 2), (2, 2), (3, 2), (3, 2), (2, 2)], 
     [(3, 1), (2, 2), (3, 2), (2, 2), (2, 2), (3, 2)], 
     ...    
    ]

    '''

    def backtrack(counter, path):
        # If the current path length is equal to the original list length, it means a permutation is formed.
        if len(path) == len(lst):
            yield path
            return

        for num in counter:
            # If the current element still has occurrences left.
            if counter[num] > 0:
                counter[num] -= 1

                # Recursively generate permutations for the remaining elements.
                for next_path in backtrack(counter, path + [num]):
                    yield next_path

                # Restore the count for the current element after backtracking.
                counter[num] += 1

    # Create a dictionary to count occurrences of each element in the list.
    counter = {}
    for num in lst:
        counter[num] = counter.get(num, 0) + 1

    return list(backtrack(counter, []))


def arrange_combinations(combinations):
    '''
    This function will generate all rearrangement possibilities for the all combinations.
    '''

    all_permutations = []
    for combination in tqdm(combinations):
        for perm in unique_permutations(combination):
            all_permutations.append(perm)
    return all_permutations


def is_inside_mesh(up_left, rect, n, m):
    '''
    Check if the rectangle is inside the mesh.

    Args:
    up_left: the up left point coordinate of the rectangle
    rect: the shape of rectangle which is to be placed
    n: the width of the mesh
    m: the height of the mesh

    Returns:
    whether the rectangle is inside the mesh

    Example:
    >>> is_inside_mesh([2, 3], [2, 2], 5, 5)
    True
    '''

    width, height = rect

    if up_left[0] < 0 or up_left[0] + width > n :
        return False
    elif up_left[1] < 0 or up_left[1] + height > m:
        return False
    else:
        return True


def does_overlap(placed_rects, new_coords):
    '''
    Check if the new rectangle overlaps any of the placed rectangles.

    Args:
    placed_rects: the placed coordinates
    new_coords: the coordinates of the new rectangle

    Returns:
    whether the new rectangle overlaps any of the placed rectangles

    Example:
    >>> does_overlap([(0, 0), (0, 1), (0, 2)], [(0, 2), (1, 2), (2, 2)])
    False    
    '''

    for rect in placed_rects:
        if any(coord in rect for coord in new_coords):
            return True
    return False


def is_adjacent(rect1, rect2):
    '''
    Check if two rectangles are adjacent.
    use abs for no orientaiton limited
    Args:
    rect1: the coordinates of the first rectangle
    rect2: the coordinates of the second rectangle

    Returns:
    whether two rectangles are adjacent

    Example:
    >>> is_adjacent([(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)])
    True
    '''

    for coord1 in rect1:
        for coord2 in rect2:
            if abs(coord1[0] - coord2[0]) == 1 and coord1[1] == coord2[1]:
                return True
            if abs(coord1[1] - coord2[1]) == 1 and coord1[0] == coord2[0]:
                return True
    return False


def get_coordinates_of_rectangle(up_left, rect_size):
    '''
    Get all coordinates covered by the rectangle.

    Args:
    upleft: the up left point coordinate of the rectangle
    rect_size: the shape of rectangle which is to be placed

    Returns:
    the coordinates of all nodes in the rectangle

    Example:
    >>> 

    '''

    width, height = rect_size
    return [(x, y) for x in range(up_left[0], up_left[0] + width) for y in range(up_left[1], up_left[1] + height)]


def possible_next_positions(last_coords, n, m, rect_size): 
    '''
    Get all possible positions for the next rectangle.

    Args:
    last_coords: the node coordinate of the all placed rectangle
    n: the width of the mesh
    m: the height of the mesh
    rect_size: the all possible coordinates of rectangle which can be placed

    Returns:
    the coordinates of all possible positions for a mapping

    Example:
    >>> possible_next_positions([[(0, 0), (0, 1), (1, 0), (1, 1)], [(0, 2), (1, 2)], [(1, 3), (1, 4), (2, 3), (2, 4)], [(3, 2), (3, 3), (4, 2), (4, 3)]], 5, 5, (2, 1))
    [[(2, 0), (2, 1), (3, 0), (3, 1)], [(3, 0), (3, 1), (4, 0), (4, 1)]]
    '''

    width, height = rect_size
    adjacent_positions = []

    for x in range(n - width + 1):
        for y in range(m - height + 1):
            potential_coords = [(i, j) for i in range(x, x + width) for j in range(y, y + height)]
            if is_adjacent(last_coords[-1], potential_coords) and is_inside_mesh((x, y), rect_size, n, m):
                xy_coordinates = get_coordinates_of_rectangle((x, y), rect_size)
                # Here we should check if the new rectangle overlaps any of all placed rectangles
                if not does_overlap(last_coords, xy_coordinates):
                    adjacent_positions.append(xy_coordinates)

    return adjacent_positions


def map_permutation_to_mesh(perm, n, m, am_only=True):
    '''
    Map a permutation (a specific order of a decomposition) to a mesh.

    Args:
    perm: a specific order of a decomposition
    n: the width of the mesh
    m: the height of the mesh

    Returns:
    the all possible mappings
    each mapping contatins the placed coordinates of all rectangles

    Example:
    >>> [(2, 2), (2, 1), (1, 2), (1, 1), (2, 2), (2, 1), (2, 1), (2, 2), (2, 1), (2, 1)]
    []
    '''    

    results = []

    def backtrack(placed_rects, index):
        """
        Backtrack to find all possible mappings.
        """

        if index == len(perm):
            # check if the last rectangle is adjacent to the first rectangle (autoregressive)
            if am_only:
                # NOTE: Here we think the last cluster can either be adjacent to the first cluster or the second last cluster or even both
                if is_adjacent(placed_rects[-1], placed_rects[0]):
                    results.append(placed_rects)
                return
            else:
                # if is_adjacent(placed_rects[-1], placed_rects[0]) or is_adjacent(placed_rects[-1], placed_rects[1]):
                if is_adjacent(placed_rects[-1], placed_rects[0]):
                    results.append(placed_rects)
                return

        rect = perm[index]
        start_positions = [get_coordinates_of_rectangle((0, 0), rect)] if not placed_rects else possible_next_positions(placed_rects, n, m, rect)

        if start_positions:
            for new_coords in start_positions:
                backtrack(placed_rects + [new_coords], index + 1)

    backtrack([], 0)
    return perm, results


def multiprocess_map(all_permutations, n, m, am_only=True):
    '''
    use multiprocess to speed up the process

    Args:
    all_permutations: all possible permutations
    n: the width of the mesh
    m: the height of the mesh

    Returns:
    the all possible mappings
    '''

    # set stationary parameters
    worker = partial(map_permutation_to_mesh, n=n, m=m, am_only=am_only)

    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(worker, all_permutations)

    # flatten the results
    clustershape_list = []
    placingnodes_list = []
    for re in results:
        if re[1]:
            for r in re[1]:
                clustershape_list.append(re[0])
                placingnodes_list.append(r)


    return clustershape_list, placingnodes_list


if __name__ == "__main__":

    sys.path.append(os.path.join(file_path, '../../utils/'))
    from save_load import save_data_to_pkl, load_data_from_pkl

    print("zigzag", zigzag_allocation(num_x=5, num_y=5, true_number=24, x_first=False))

    cycle = hamiltonian_allocation(5, 5, False)
    print("Hamiltonian loop:", cycle)

    possible_rectangles = generate_rectangles(5, 5, swap=True, side_limit=4, points_limit=6)
    basic_combinations = list(find_combinations(possible_rectangles, 25))
    print(len(basic_combinations))
    filted_combinations = filter_combinations_kickcontinuous(basic_combinations)
    print(len(filted_combinations))
    all_permutations = arrange_combinations(filted_combinations)
    print(len(all_permutations))
    clean_permutations = remove_reverse_duplicates(all_permutations)
    print(len(clean_permutations))

    _, mapping_example = multiprocess_map(clean_permutations, 5, 5)
    save_data_to_pkl(mapping_example, os.path.join(file_path, f"../../tmp/mapping_5x5.pkl"))
    save_data_to_pkl(clean_permutations, os.path.join(file_path, f"../../tmp/permutations_5x5.pkl"))
