
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))

from itertools import permutations
from typing import List
from collections import deque
from copy import deepcopy


def alltoall_mesh2d_tasks(width_number: int, height_number: int):

    def generate_combinations(lst: List[int]) -> List[List[int]]:
        '''
        get all combinations of elements in lst
        eg: lst = [0, 2, 5]
        return [(0, 2), (0, 5), (2, 0), (2, 5), (5, 0), (5, 2)]
        '''
        element_permutations = list(permutations(lst, 2))
        return element_permutations

    def idx_to_xy(idx):
        return (idx % width_number, idx // width_number)

    def xy_to_idx(x, y):
        return y * width_number + x

    def is_within_bounds(x, y):
        return 0 <= x < width_number and 0 <= y < height_number

    def are_neighbors(idx1, idx2):
        x1 = idx1 % width_number
        y1 = idx1 // width_number

        x2 = idx2 % width_number
        y2 = idx2 // width_number

        horizontal_neighbors = x1 == x2 and abs(y1 - y2) == 1
        vertical_neighbors = y1 == y2 and abs(x1 - x2) == 1

        return horizontal_neighbors or vertical_neighbors


    nodes_number = width_number * height_number
    nodes_idx_list = list(range(nodes_number))

    links_list = []
    for y in range(height_number):
        for x in range(width_number):
            if x + 1 < width_number:
                links_list.append([y * width_number + x, y * width_number + (x + 1)])
            if y + 1 < height_number:
                links_list.append([y * width_number + x, (y + 1) * width_number + x])
            if x - 1 >= 0:
                links_list.append([y * width_number + x, y * width_number + (x - 1)])
            if y - 1 >= 0:
                links_list.append([y * width_number + x, (y - 1) * width_number + x])

    all_indices = set(range(width_number * height_number))
    corner_indices = set()
    edge_indices = set()

    corners = [0, width_number - 1, (height_number - 1) * width_number, height_number * width_number - 1]
    corner_indices.update(corners)

    for i in range(1, width_number - 1):
        edge_indices.add(i)
        edge_indices.add(i + (height_number - 1) * width_number)

    for j in range(1, height_number - 1):
        edge_indices.add(j * width_number)
        edge_indices.add((width_number - 1) + j * width_number)  

    inner_indices = all_indices - corner_indices - edge_indices

    corner_nodes_idx_list = list(corner_indices)
    edge_nodes_idx_list = list(edge_indices)
    inner_nodes_idx_list = list(inner_indices)
    reordered_nodes_idx_list = corner_nodes_idx_list + edge_nodes_idx_list + inner_nodes_idx_list

    communication_pairs = generate_combinations(nodes_idx_list)

    def bfs(idx1, idx2):
        queue = deque([(idx1, [idx1])])
        visited = set([idx1])

        idx1_x, idx1_y = idx_to_xy(idx1)
        idx2_x, idx2_y = idx_to_xy(idx2)
        x_dist = abs(idx1_x - idx2_x)
        y_dist = abs(idx1_y - idx2_y)

        if x_dist > y_dist:
            if idx1_x <= idx2_x and idx1_y <= idx2_y:
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            elif idx1_x <= idx2_x and idx1_y >= idx2_y:
                directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
            elif idx1_x >= idx2_x and idx1_y <= idx2_y:
                directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            else:
                directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        else:
            if idx1_x <= idx2_x and idx1_y <= idx2_y:
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            elif idx1_x <= idx2_x and idx1_y >= idx2_y:
                directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            elif idx1_x >= idx2_x and idx1_y <= idx2_y:
                directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
            else:
                directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]

        while queue:
            current_idx, path = queue.popleft()
            current_x, current_y = idx_to_xy(current_idx)

            if current_idx == idx2:
                return path

            for dx, dy in directions:
                new_x, new_y = current_x + dx, current_y + dy
                new_idx = xy_to_idx(new_x, new_y)
                if is_within_bounds(new_x, new_y) and new_idx not in visited:
                    visited.add(new_idx)
                    queue.append((new_idx, path + [new_idx]))

        return []


    def adjacent_pairs(lst):
        lst_legnth = len(lst) - 1
        return [[lst[i], lst[i + 1]] for i in range(lst_legnth)]


    def find_busy_path(paths):
        busy_path = {}
        for path in paths:
            for link in path:
                link_tuple = tuple(link)
                if link_tuple in busy_path:
                    busy_path[link_tuple] += 1
                else:
                    busy_path[link_tuple] = 1
        busy_num = 0
        busy_link = []
        for link, count in busy_path.items():
            if count > busy_num:
                busy_num = count
                busy_link = [link]
            elif count == busy_num:
                busy_link.append(link)

        return busy_num, [list(link) for link in busy_link]


    communication_paths = []
    for source_idx, target_idx in communication_pairs:
        if are_neighbors(source_idx, target_idx):
            path = [source_idx, target_idx]
        else:
            path = bfs(source_idx, target_idx)
        if path:
            communication_paths.append(adjacent_pairs(path))
        else:
            raise ValueError(f"No path between {source_idx} and {target_idx}")
    print(communication_paths)
    busy_num, busy_link = find_busy_path(communication_paths)

    time_step = 0
    steps = []
    while communication_paths:
        current_step_links = []
        available_links = deepcopy(links_list)
        current_nodes = deepcopy(reordered_nodes_idx_list)

        no_available_links = False
        while no_available_links is False:
            no_available_links = True
            for current_node in current_nodes[:]:
                for task_path_idx, task_path in enumerate(communication_paths[:]):
                    if task_path:                        
                        task_link = task_path[0]
                        if task_link[0] == current_node and task_link in available_links:
                            current_step_links.append(task_link)
                            available_links.remove(task_link)
                            current_nodes.remove(current_node)
                            current_nodes.append(current_node)
                            task_path.remove(task_link)
                            no_available_links = False
                            break
                    else:
                        communication_paths.remove(task_path)
                if no_available_links is False:
                    break

        time_step += 1
        steps.append(current_step_links)
        for task_path in communication_paths[:]:
            if task_path:
                continue
            else:
                communication_paths.remove(task_path)

        if time_step > (width_number * height_number) ** 2:
            print(communication_paths)
            raise ValueError("The scheduling algorithm is flying off")


    return time_step, steps


if __name__ == "__main__":

    timesteps, steps = alltoall_mesh2d_tasks(3, 3)
    print(f"timesteps: {timesteps}, steps: {steps}")
