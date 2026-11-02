
import os
import sys

from numpy import indices
file_path = os.path.dirname(os.path.realpath(__file__))

import random


class mesh_2d(object):

    def __init__(self, height, width):

        self.height = height
        self.width = width
        self.node_num = height * width

        self.initial_links()
        self.initial_root()
        self.initial_target()

    def initial_links(self):

        self.coordinates = []
        self.source_links = []
        self.neighbour_coordiantes = []
        self.neighbours = []

        for i in range(self.height):
            for j in range(self.width):
                source_link = []
                neighbour_coordinate = []
                neighbour = []
                if i == 0:
                    source_link.append(None)
                    neighbour_coordinate.append([None, None])
                    neighbour.append(None)
                else:
                    source_link.append(1)
                    neighbour_coordinate.append([i-1, j])
                    neighbour.append((i-1) * self.width + j)
                if j == self.width - 1:
                    source_link.append(None)
                    neighbour_coordinate.append(None)
                    neighbour.append(None)
                else:
                    source_link.append(1)
                    neighbour_coordinate.append([i, j+1])
                    neighbour.append(i * self.width + j + 1)
                if i == self.height - 1:
                    source_link.append(None)
                    neighbour_coordinate.append(None)
                    neighbour.append(None)
                else:
                    source_link.append(1)
                    neighbour_coordinate.append([i+1, j])
                    neighbour.append((i+1) * self.width + j)
                if j == 0:
                    source_link.append(None)
                    neighbour_coordinate.append(None)
                    neighbour.append(None)
                else:
                    source_link.append(1)
                    neighbour_coordinate.append([i, j-1])
                    neighbour.append(i * self.width + j - 1)

                self.source_links.append(source_link)
                self.neighbour_coordiantes.append(neighbour_coordinate)
                self.neighbours.append(neighbour)

                coordinate = [i, j]
                self.coordinates.append(coordinate)


    def initial_root(self):
        self.trees = []
        self.current_parent = []        
        for i in range(self.node_num):
            self.trees.append([])
            self.trees[i].append([i])
            self.current_parent.append([i])


    def initial_target(self):
        self.targets = []
        for i in range(self.node_num):
            self.targets.append([i for i in range(self.node_num)])
            self.targets[i].remove(i)


    def build_tree(self, timestep):
        timestep_done_flag = [False for i in range(self.node_num)]
        for i in range(self.node_num):
            self.trees[i].append([])
        self.initial_links()

        old_parent_list = [[] for i in range(self.node_num)]
        new_parent_list = [[] for i in range(self.node_num)]

        while False in timestep_done_flag:

            ''' 
            according to the max distance from targets to sort the indices
            '''
            indices = list(range(self.height * self.width))
            ordered_list = indices

            for i in ordered_list:
                find_one_flag = 0
                if timestep_done_flag[i] == True:
                    continue
                else:
                    for parent_idx, parent in enumerate(self.current_parent[i]):
                        '''
                        calculate available links of each parent
                        '''
                        available_links = self.find_indices_of_ones(self.source_links[parent])
                        if available_links == []:
                            timestep_done_flag[i] = True
                            continue
                        else:                    
                            available_nodes = [self.neighbours[parent][ava_index] for ava_index in available_links]
                            '''
                            find just one branch
                            '''
                            next_node = self.find_first_common_element(available_nodes, self.targets[i])
                            if next_node == None:
                                if parent_idx == len(self.current_parent[i]) - 1:
                                    timestep_done_flag[i] = True
                            else:
                                self.trees[i][timestep].append([parent, next_node])
                                self.source_links[parent][available_links[available_nodes.index(next_node)]] = 0
                                self.targets[i].remove(next_node)
                                if parent not in old_parent_list[i]:
                                    old_parent_list[i].append(parent)
                                self.current_parent[i].remove(parent)
                                self.current_parent[i].append(parent)
                                '''
                                put here means one by one not one to two
                                eg. only support 1->2 not 1->2->3
                                '''
                                new_parent_list[i].append(next_node) 
                                timestep_done_flag[i] = False
                                find_one_flag = 1
                                break
                if find_one_flag == 1:
                    break
                else:
                    continue

        for i in range(self.node_num):
            for parent in old_parent_list[i]:
                self.current_parent[i].remove(parent)
            for parent in new_parent_list[i]:
                self.current_parent[i].append(parent)
            for parent in old_parent_list[i]:
                self.current_parent[i].append(parent)            

    @staticmethod
    def find_indices_of_ones(lst):
        return [index for index, value in enumerate(lst) if value == 1]

    @staticmethod
    def find_first_common_element(list1, list2):
        for element in list1:
            if element in list2:
                return element
        return None

    @staticmethod
    def sort_indices_by_max_distance_from_targets(origin, target, source_link, n, m):
        def get_max_distance_for_point_from_targets(point_indices, target_indices, m):

            def index_to_point(index, m):
                """
                Convert index to point coordinates.
                """
                return (index // m, index % m)

            def calculate_distance(point1, point2):
                return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])        

            max_distance = 0
            for point_index in point_indices:
                point = index_to_point(point_index, m)
                for target_index in target_indices:
                    target_point = index_to_point(target_index, m)
                    distance = calculate_distance(point, target_point)
                    if distance > max_distance:
                        max_distance = distance
            return max_distance

        indices = [i for i in range(n * m)]
        max_distances = [
            (index, 
            get_max_distance_for_point_from_targets(origin[index], target[index], m), 
            sum([source_link[p].count(1) for p in origin[index]])
            ) 
            for index in indices
        ]
        sorted_by_distance = sorted(max_distances, key=lambda x: (-x[1], x[2]))
        return [item[0] for item in sorted_by_distance]

    @staticmethod
    def draw_trees_in_one_dot(data):
        '''
        draw multitree to dot 
        '''


        def add_nodes_edges(tree_data, graph, tree_label):

            root = str(tree_data[0][0]) + "_" + tree_label
            graph.node(root, label=str(tree_data[0][0]))

            level_dict = {}
            level_dict[tree_data[0][0]] = -1
            for level_index, level in enumerate(tree_data[1:]):
                for pair in level:
                    parent, child = pair
                    if child in level_dict.keys():
                        pass
                    else:
                        level_dict[child] = level_index + 1
                    graph.edge(str(parent) + "_" + tree_label, str(child) + "_" + tree_label, minlen=str(level_index + 1 - level_dict[parent]))
                    graph.node(str(child) + "_" + tree_label, label=str(child))

        from graphviz import Digraph
        dot = Digraph(comment="Trees", format="dot")

        for idx, tree_data in enumerate(data):
            tree_label = "Tree" + str(tree_data[0][0])
            add_nodes_edges(tree_data, dot, tree_label)
            dot.node(tree_label, shape="plaintext")
            dot.edge(tree_label, str(tree_data[0][0]) + "_" + tree_label, minlen="1")

        with open(os.path.join(file_path, 'combined_tree.dot'), 'w') as f:
            f.write(dot.source)


    def walkthrough(self):
        '''
        generate all trees and leaves
        '''
        if self.node_num == 1:
            return 0
        else:
            t_max = self.node_num * (self.node_num - 1)
            for t in range(1, t_max):
                flag = 0
                self.build_tree(t)
                for i in range(self.node_num):
                    if self.targets[i]:
                        flag = 1
                        break                    
                    else:
                        continue
                if i == self.node_num - 1 and flag == 0:
                    break

            return t


if __name__ == "__main__":

    cost_list = []
    all_cost_list = []
    length = 5

    node_height = length
    node_width = length
    node_num = node_width * node_height

    mynet = mesh_2d(node_height, node_width)
    timecost = mynet.walkthrough()
    print("Time cost: ", timecost)
