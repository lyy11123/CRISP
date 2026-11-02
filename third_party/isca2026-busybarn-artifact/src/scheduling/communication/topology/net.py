
"""Base network class: provides node/link management, Dijkstra routing,
multicast tree construction, and the BALD (Balanced Allocation with Load
and Distance awareness) algorithm used by all concrete topologies."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../../platform/device/'))
sys.path.append(os.path.join(file_path, '../../../platform/device/link/'))
sys.path.append(os.path.join(file_path, '../../../platform/device/module/'))
sys.path.append(os.path.join(file_path, '../../../../utils/'))


from device import device
from link import link
from ch2ch import ch2ch
from co2co import co2co
from module import module
from tensorcore import tensorcore
from vectorunit import vectorunit
from read_cfg import cfg_to_dict


from typing import List, Tuple, Set, Deque, Dict, Any
from collections import deque
from functools import lru_cache
from typing import List, Tuple, Set, Dict, Deque
from collections import deque, defaultdict
import itertools
import heapq
import numpy as np
import random
random.seed(123)
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import inspect


class TupleIdx(tuple):
    """Tuple subclass that tracks the original tuple (.ori) and an optional
    duplicate-link index (.idx) for multi-link edges in the topology graph."""

    def __new__(cls, iterable, idx=None):

        original = tuple(iterable)
        if idx is not None:
            full = original + (idx,)
        else:
            full = original
        inst = super().__new__(cls, full)
        inst.ori = original
        inst.idx = idx

        return inst


class net(object):
    """Abstract base class for all network topologies. Provides Dijkstra
    shortest-path computation, BALD multicast routing, link load tracking,
    and distance-function precomputation."""

    def __init__(self):

        self.original_cfg = {}

        self.nodes_set = set()
        self.links_set = set()
        self.nodes_coordinate_dict = {}
        self.nodes_coordinate_idx_dict = {}

        self.available_links = {}
        self.available_neighbors = {}
        self.available_links_deque = {}
        self.duplicated_links = {}

        self.link_loads_dict = {link: 0 for link in self.links_set}
        self.links_list = list(self.links_set)


    def idx_to_coordinate(self, idx: int) -> Tuple:

        return self.nodes_coordinate_dict[idx]


    def coordinate_to_idx(self, coordinate: Tuple) -> int:

        return self.nodes_coordinate_idx_dict[coordinate]


    def show_nodes(self):

        for node_idx in self.nodes_set:
            print(node_idx, self.nodes_coordinate_dict[node_idx])


    def show_links(self):

        for link_idx in self.links_set:
            print(self.links_dict[link_idx])


    def dijkstra_offload(self):
        # Reset all link load counters to zero (called before routing)
        self.link_loads_dict = {link: 0 for link in self.links_set}
        self.link_loads_count = {link: 0 for link in self.links_set}


    def dijkstra_to_all_random(
        self,
        start_node_idx: int,
        use_random_tie_breaker: bool = True,
        shuffle_neighbors: bool = True,
        profile_size: int = 1048576,  # 1 MB default for link cost profiling
    ):
        # Single-source Dijkstra with random tie-breaking for load balance.
        # Returns (distances, all_links, predecessors, is_unique, fixed_paths, dijkstra_paths).

        distances = {node: float('inf') for node in self.nodes_set}
        distances[start_node_idx] = 0
        predecessors = {node: None for node in self.nodes_set}

        path_count = {node: 0 for node in self.nodes_set}
        is_unique = {node: True for node in self.nodes_set}
        path_count[start_node_idx] = 1

        if use_random_tie_breaker:
            queue = [(0, random.random(), start_node_idx)]
        else:
            queue = [(0, start_node_idx)]

        while queue:
            if use_random_tie_breaker:
                current_dist, _, current_node = heapq.heappop(queue)
            else:
                current_dist, current_node = heapq.heappop(queue)            

            if current_dist > distances[current_node]:
                continue

            neighbors = list(self.available_neighbors.get(current_node, []))
            if shuffle_neighbors:
                random.shuffle(neighbors)

            for neighbor in neighbors:
                for duplicated_idx in range(self.duplicated_links[(current_node, neighbor)] + 1):
                    if duplicated_idx == 0:
                        current_pair_idx = TupleIdx((current_node, neighbor), idx=None)
                    else:
                        current_pair_idx = TupleIdx((current_node, neighbor), idx=duplicated_idx)

                    new_dist = current_dist + self.links_dict[current_pair_idx].latency + self.current_link_loads(profile_size, self.links_dict[current_pair_idx])

                    # if record_flag:
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        if duplicated_idx == 0:
                           current_node_idx = TupleIdx(current_node, idx=None)
                        else:
                            current_node_idx = TupleIdx(current_node, idx=duplicated_idx)
                        predecessors[neighbor] = current_node_idx

                        path_count[neighbor] = path_count[current_node_idx]
                        is_unique[neighbor] = is_unique[current_node_idx]

                        if use_random_tie_breaker:
                            heapq.heappush(queue, (new_dist, random.random(), neighbor))
                        else:
                            heapq.heappush(queue, (new_dist, neighbor))

                    elif new_dist == distances[neighbor]:

                        if random.random() < 0.5:
                            distances[neighbor] = new_dist
                            if duplicated_idx == 0:
                                current_node_idx = TupleIdx(current_node, idx=None)
                            else:
                                current_node_idx = TupleIdx(current_node, idx=duplicated_idx)
                            predecessors[neighbor] = current_node_idx

                            path_count[neighbor] = path_count[current_node_idx.ori]
                            is_unique[neighbor] = is_unique[current_node_idx.ori]

                            if use_random_tie_breaker:
                                heapq.heappush(queue, (new_dist, random.random(), neighbor))
                            else:
                                heapq.heappush(queue, (new_dist, neighbor))

                        path_count[neighbor] += path_count[current_node_idx.ori]
                        if path_count[neighbor] > 1:
                            is_unique[neighbor] = False

        unreachable = [n for n, d in distances.items() if n != start_node_idx and d == float('inf')]
        if unreachable:
            raise RuntimeError(f"exsisting unreachable nodes: {unreachable[:10]} ...")

        fixed_paths = {}
        for remote in is_unique:
            if is_unique[remote] and remote != start_node_idx:
                remote_idx = TupleIdx(remote, idx=None)
                fixed_paths[(start_node_idx, remote)], _ = self.get_reverse_path(predecessors, start_node_idx, remote_idx)

        all_links = {}
        dijstra_paths = {}
        for remote in self.nodes_set:
            if remote != start_node_idx:
                remote_idx = TupleIdx(remote, idx=None)
                dijstra_paths[(start_node_idx, remote)], all_links[remote] = self.get_reverse_path(predecessors, start_node_idx, remote_idx)

        return distances, all_links, predecessors, is_unique, fixed_paths, dijstra_paths


    def get_reverse_path(self, predecessors, start_node, target_node):
        # Walk predecessor chain from target back to start, then reverse.
        path = []
        link_list = []
        cur = target_node
        while cur is not None:
            path.append(cur)
            root = cur.ori
            cur = predecessors[cur.ori]
            if cur:
                if cur.idx:
                    link_list.append((cur.ori, root, cur.idx)) 
                else:
                    link_list.append((cur.ori, root))               
        path.reverse()
        link_list.reverse()
        if path and path[0].ori == start_node:
            return path, link_list
        return []


    def get_clean_path(self, comm_tag, comm_pairs_dict, paths, start_node, target_nodes):
        # Prune unused branches from a multicast tree, keeping only paths to target_nodes.
        # Decrements link_loads for pruned links.
        path = {}
        for parent in paths:
            for child in paths[parent]:
                path[child.ori] = TupleIdx(parent, idx=child.idx)
        path[start_node] = None

        target_paths_list = []
        target_used_links = set()
        for target in target_nodes:
            target_path, got_used_links = self.get_reverse_path(path, start_node, target)
            # for used_link in got_used_links:
            target_paths_list.append(target_path)
            for i in range(len(target_path) - 1):
                target_used_links.add((target_path[i].ori, TupleIdx(target_path[i+1].ori, idx=target_path[i].idx)))

        for parent in paths:
            no_use_child = []
            for child in paths[parent]:
                if (parent, child) not in target_used_links:
                    no_use_child.append(child)
                    no_use_link = (parent, child.ori, child.idx) if child.idx else (parent, child)
                    self.link_loads_dict[no_use_link] -= self.current_link_loads(comm_pairs_dict[comm_tag][3], self.links_dict[no_use_link])
                    self.link_loads_count[no_use_link] -= 1
            for child in no_use_child:
                paths[parent].remove(child)

        clean_roots = []
        for parent in paths:
            if paths[parent] == []:
                clean_roots.append(parent)
        for clean_root in clean_roots:
            del paths[clean_root]

        return paths, target_paths_list


    def _precompute_edges(self, profile_size):
        """Pre-compute edge adjacency and link properties for fast Dijkstra."""
        edge_adj = {}
        link_lat = {}
        link_bw_tu = {}
        for node in self.nodes_set:
            edges = []
            for neighbor in self.available_neighbors.get(node, []):
                for di in range(self.duplicated_links[(node, neighbor)] + 1):
                    lid = TupleIdx((node, neighbor), idx=None if di == 0 else di)
                    pid = TupleIdx(node, idx=None if di == 0 else di)
                    lnk = self.links_dict[lid]
                    cost = lnk.latency + int(np.ceil(profile_size / lnk.bandwidth) * lnk.timeunit)
                    edges.append((neighbor, lid, cost, pid))
                    if lid not in link_lat:
                        link_lat[lid] = lnk.latency
                        link_bw_tu[lid] = lnk.bandwidth / lnk.timeunit
            edge_adj[node] = edges
        return edge_adj, link_lat, link_bw_tu

    def _dijkstra_fast(self, start, edge_adj):
        """Fast Dijkstra using pre-computed edge adjacency."""
        INF = float('inf')
        distances = {n: INF for n in self.nodes_set}
        distances[start] = 0
        predecessors = {n: None for n in self.nodes_set}
        path_count = {n: 0 for n in self.nodes_set}
        is_unique = {n: True for n in self.nodes_set}
        path_count[start] = 1

        queue = [(0, random.random(), start)]
        while queue:
            cd, _, cn = heapq.heappop(queue)
            if cd > distances[cn]:
                continue
            edges = list(edge_adj[cn])
            random.shuffle(edges)
            for nb, lid, ec, pid in edges:
                nd = cd + ec
                if nd < distances[nb]:
                    distances[nb] = nd
                    predecessors[nb] = pid
                    path_count[nb] = path_count[pid]
                    is_unique[nb] = is_unique[pid]
                    heapq.heappush(queue, (nd, random.random(), nb))
                elif nd == distances[nb]:
                    if random.random() < 0.5:
                        predecessors[nb] = pid
                        path_count[nb] = path_count[pid.ori]
                        is_unique[nb] = is_unique[pid.ori]
                        heapq.heappush(queue, (nd, random.random(), nb))
                    path_count[nb] += path_count[pid.ori]
                    if path_count[nb] > 1:
                        is_unique[nb] = False

        unreachable = [n for n, d in distances.items() if n != start and d == INF]
        if unreachable:
            raise RuntimeError(f"exsisting unreachable nodes: {unreachable[:10]} ...")
        return distances, predecessors, is_unique

    def _reconstruct_paths(self, start, predecessors, is_unique):
        """Reconstruct all paths and link lists from predecessor dict."""
        fixed_paths = {}
        all_links = {}
        dijkstra_paths = {}
        for remote in self.nodes_set:
            if remote == start:
                continue
            ri = TupleIdx(remote, idx=None)
            path, links = self.get_reverse_path(predecessors, start, ri)
            dijkstra_paths[(start, remote)] = path
            all_links[remote] = links
            if is_unique[remote]:
                fixed_paths[(start, remote)] = path
        return all_links, fixed_paths, dijkstra_paths

    def _build_dicts_from_links(self, node, all_links, distances,
                                link_lat, link_bw_tu):
        """Build hop/distance/function dicts for *node* from link lists."""
        self.node_to_node_hop_dict[node] = {}
        self.node_to_node_distance_dict[node] = distances
        self.node_to_node_distance_function_dict[node] = {}
        for leaf, links in all_links.items():
            self.node_to_node_hop_dict[node][leaf] = len(links)
            base_latency = sum(link_lat[lid] for lid in links)
            slope = sum(link_bw_tu[lid] for lid in links)
            self.node_to_node_distance_function_dict[node][leaf] = (
                lambda ms, base=base_latency, sl=slope:
                    base + math.ceil(ms / sl)
            )

    def allocate_path(self, source, target, comm_bytes,
                      alpha=1, beta=0, gamma=1):
        """Load-aware Dijkstra: find shortest path from source to target.

        Edge cost = transmission_cost + beta * existing_link_load + gamma * hop_distance_to_target.
        After finding the path, updates link_loads_dict and link_loads_count.

        Returns:
            paths_dict: {node: [TupleIdx(next_node)]} tree structure (for event.paths)
            path_links: list of link TupleIdx keys used (for load tracking)
        """
        INF = float('inf')
        dist = {n: INF for n in self.nodes_set}
        dist[source] = 0
        pred = {n: None for n in self.nodes_set}
        queue = [(0, source)]

        while queue:
            cd, cn = heapq.heappop(queue)
            if cd > dist[cn]:
                continue
            if cn == target:
                break
            for neighbor in self.available_neighbors.get(cn, []):
                for di in range(self.duplicated_links[(cn, neighbor)] + 1):
                    lid = TupleIdx((cn, neighbor), idx=None if di == 0 else di)
                    nid = TupleIdx(neighbor, idx=None if di == 0 else di)
                    tx_cost = self.current_link_loads(comm_bytes, self.links_dict[lid])
                    load_cost = self.link_loads_dict.get(lid, 0)
                    nd = cd + alpha * tx_cost + beta * load_cost
                    if nd < dist[neighbor]:
                        dist[neighbor] = nd
                        pred[neighbor] = (cn, nid, lid)
                        heapq.heappush(queue, (nd, neighbor))

        # Reconstruct path
        if dist[target] == INF:
            raise RuntimeError(f"No path from {source} to {target}")

        chain = []
        cur = target
        while pred[cur] is not None:
            prev_node, cur_idx, link_id = pred[cur]
            chain.append((prev_node, cur_idx, link_id))
            cur = prev_node
        chain.reverse()

        # Build paths_dict in the tree format: {parent_node: [TupleIdx(child)]}
        paths_dict = {}
        path_links = []
        for prev_node, cur_idx, link_id in chain:
            paths_dict.setdefault(prev_node, []).append(cur_idx)
            path_links.append(link_id)
            # Update link loads
            self.link_loads_dict[link_id] = self.link_loads_dict.get(link_id, 0) + \
                self.current_link_loads(comm_bytes, self.links_dict[link_id])
            self.link_loads_count[link_id] = self.link_loads_count.get(link_id, 0) + 1

        return paths_dict, path_links


    def node_to_node_distance(self, profile_size: int = 1048576):
        # Precompute all-pairs shortest paths, hop counts, and distance functions.
        # Populates dijkstra_paths, fixed_paths, and per-node distance dicts.
        self.node_to_node_hop_dict = {}
        self.node_to_node_distance_dict = {}
        self.node_to_node_distance_function_dict = {}
        self.fixed_paths = {}
        self.dijkstra_paths = {}
        for node in self.nodes_set:
            distances, all_links, predecessors, fixed_points, fixed_paths, dijstra_paths = self.dijkstra_to_all_random(start_node_idx=node, profile_size=profile_size)
            self.node_to_node_hop_dict[node] = {}
            self.node_to_node_distance_dict[node] = distances
            self.node_to_node_distance_function_dict[node] = {}
            for leaf, links in all_links.items():
                self.node_to_node_hop_dict[node][leaf] = len(links)
                base_latency = sum(self.links_dict[lid].latency for lid in links)
                slope = sum(self.links_dict[lid].bandwidth / self.links_dict[lid].timeunit
                            for lid in links)
                self.node_to_node_distance_function_dict[node][leaf] = (
                    lambda ms, base=base_latency, sl=slope:
                        base + math.ceil(ms / sl)
                )
            self.fixed_paths = {**self.fixed_paths, **fixed_paths}
            self.dijkstra_paths = {**self.dijkstra_paths, **dijstra_paths}


    def build_dijkstra_database_v2(self, profile_size: int = 1048576):
        """Optimized version of node_to_node_distance using fast Dijkstra helpers.

        Uses _precompute_edges / _dijkstra_fast / _reconstruct_paths /
        _build_dicts_from_links for better performance on large topologies.
        """

        self.node_to_node_hop_dict = {}
        self.node_to_node_distance_dict = {}
        self.node_to_node_distance_function_dict = {}
        self.fixed_paths = {}
        self.dijkstra_paths = {}

        nodes = list(self.nodes_set)
        if not nodes:
            return

        # Pre-compute edge adjacency and link properties once
        edge_adj, link_lat, link_bw_tu = self._precompute_edges(profile_size)

        for node in tqdm(nodes, desc="Building dijkstra paths"):
            distances, predecessors, is_unique = self._dijkstra_fast(
                node, edge_adj)
            all_links, fixed_paths, dp_out = self._reconstruct_paths(
                node, predecessors, is_unique)
            self._build_dicts_from_links(node, all_links, distances,
                                         link_lat, link_bw_tu)
            self.fixed_paths.update(fixed_paths)
            self.dijkstra_paths.update(dp_out)


    @lru_cache(maxsize=100)
    def current_link_loads(self,
                           message_size: int,
                           link_device: link,
                           ):
        # Transmission time in cycles: ceil(bytes / bandwidth) * timeunit
        return int(np.ceil(message_size / link_device.bandwidth) * link_device.timeunit)


    @lru_cache(maxsize=100)
    def current_link_latency(self,
                             message_size: int,
                             link_device: link
                             ):
        # Total link latency = fixed latency + transmission time
        return link_device.latency + int(np.ceil(message_size / link_device.bandwidth) * link_device.timeunit)


    @lru_cache(maxsize=100)
    def next_link_start(self,
                          current_starttime: int,
                          message_size: int,
                          current_link: link,
                          next_link: link
                        ):
        # Compute pipeline start time for the next link in a multi-hop path
        if current_link.bandwidth * current_link.timeunit >= next_link.bandwidth * next_link.timeunit:
            return current_starttime - self.current_link_loads(message_size, current_link)
        else:
            start_time = self.current_link_loads(message_size, current_link) - self.current_link_latency(message_size, next_link)
            return current_starttime - self.current_link_loads(message_size, current_link) + start_time


    def record_dijkstra_multicast_path(
        self,
        comm_pairs: List[int],
        alpha: int=100,
        beta: int=1,
        gamma: int=100,
        shortest_init: bool=True,
        long_first: bool=False,
        perdist: bool=False,
        pertask: bool=False,
        funcdist: bool=False,
        # TODO: runtime logic
        runtime: bool=False,
        deterministic: bool=False,
    ) -> Dict[Tuple[int], Dict[int, List[int]]]:
        """BALD multicast routing: builds load-balanced, distance-aware multicast
        trees for a set of communication pairs. alpha/beta/gamma control the
        trade-off between transmission cost, existing load, and hop distance.

        comm_pairs: [
            [comm_tag, source_node_idx, Set[targets_node_idx], link_load],
            ...
        ]

        Processed form:
        [
            [comm_tag, source_node_idx, Set[targets_node_idx], link_load, Set[other_node_idx], product_of_farthest_distance_and_load]
        ]
        """

        iter_comm_pairs = deepcopy(comm_pairs)
        if deterministic:
            iter_comm_pairs.sort(key=lambda x: x[0])
        else:
            random.shuffle(iter_comm_pairs)

        comm_pairs_dict = {}
        root_tag_dict = {}
        tag_neighbors_dict = {}
        tag_links_cost_dict = {}
        tag_paths_dict = {}
        tag_lastpair_dict = {}

        for comm_pair in list(iter_comm_pairs):
            comm_pairs_dict[comm_pair[0]] = list(comm_pair)
            comm_tag = comm_pair[0]
            source_node_idx = comm_pair[1]
            target_nodes_set = comm_pair[2]
            link_load = comm_pair[3]

            root_tag_dict.setdefault(source_node_idx, []).append(comm_tag)

            other_nodes_set = deepcopy(self.nodes_set)
            other_nodes_set.remove(source_node_idx)
            comm_pair.append(other_nodes_set)

            if comm_tag not in tag_links_cost_dict:
                tag_links_cost_dict[comm_tag] = {}

            if comm_tag not in tag_paths_dict:
                tag_paths_dict[comm_tag] = {}
                tag_paths_dict[comm_tag][source_node_idx] = []
            if comm_tag not in tag_lastpair_dict:
                tag_lastpair_dict[comm_tag] = {}


            # avaible_neighbors_list = list(self.available_neighbors[source_node_idx])
            if target_nodes_set:
                if funcdist:
                    neighbor_farest_distance = max(
                        self.node_to_node_distance_function_dict[source_node_idx][t](link_load) for t in target_nodes_set
                    )
                else:
                    neighbor_farest_distance = max(
                        self.node_to_node_distance_dict[source_node_idx][t] for t in target_nodes_set
                    )
            else:
                raise ValueError("Target nodes set cannot be empty for communication pairs.")

            tag_neighbors_dict.setdefault(comm_tag, []).append([source_node_idx, neighbor_farest_distance])

            target_nodes_set_copy = deepcopy(target_nodes_set)
            if shortest_init:
                for target in list(target_nodes_set):
                    if (source_node_idx, target) in self.fixed_paths:
                        fixed_path = self.fixed_paths[(source_node_idx, target)]
                        last_along = fixed_path[0]
                        tag_lastpair_dict[comm_tag][fixed_path[1]] = last_along
                        for along_node in fixed_path[1:]:
                            if along_node.ori in tag_paths_dict[comm_tag]:
                                last_along = along_node
                                continue

                            if last_along.ori in tag_paths_dict[comm_tag]:
                                if along_node.ori in tag_paths_dict[comm_tag][last_along]:
                                    last_along = along_node
                                    continue
                                else:
                                    pass
                            else:
                                pass

                            if along_node.ori in target_nodes_set:
                                target_nodes_set_copy.remove(along_node.ori)

                            tlcd = tag_links_cost_dict[comm_tag]
                            tlcd[(last_along, along_node)] = self.current_link_loads(link_load, self.links_dict[(last_along, along_node)])
                            if last_along.ori == source_node_idx:
                                pass
                            else:
                                tlcd[(last_along, along_node)] += tlcd[(tag_lastpair_dict[comm_tag][last_along], last_along)]

                            self.link_loads_dict[(last_along, along_node)] += self.current_link_loads(link_load, self.links_dict[(last_along, along_node)])
                            self.link_loads_count[(last_along, along_node)] += 1

                            other_nodes_set.remove(along_node.ori)

                            tag_paths_dict[comm_tag].setdefault(last_along, []).append(along_node)
                            tag_lastpair_dict[comm_tag][along_node] = last_along

                            last_along = along_node

                            # tag_paths_dict[comm_tag][source_node_idx].append(source_node_idx)
                            if target_nodes_set:
                                if funcdist:
                                    nfd = max(self.node_to_node_distance_function_dict[along_node.ori][t](link_load) for t in target_nodes_set)
                                else:
                                    nfd = min(self.node_to_node_distance_dict[along_node.ori][t] for t in target_nodes_set)
                            else:
                                nfd = 0
                            tag_neighbors_dict[comm_tag].append([along_node.ori, nfd])

                new_branch_list = []
                tnd = tag_neighbors_dict[comm_tag]
                tlcd = tag_links_cost_dict[comm_tag]
                for branch, branch_distance in tnd:
                    still_needed = False
                    for neighbor in self.available_neighbors[branch]:
                        if neighbor not in tlcd:
                            still_needed = True
                            break
                    if still_needed:
                        new_branch_list.append([branch, branch_distance])
                tag_neighbors_dict[comm_tag] = new_branch_list

            comm_pair[2] = target_nodes_set_copy
            if not comm_pair[2]:
                iter_comm_pairs.remove(comm_pair)
                continue

            # for comm_tag in tag_paths_dict:
            if not deterministic:
                random.shuffle(tag_neighbors_dict[comm_tag])

            if target_nodes_set:
                costs = []
                for target in target_nodes_set:
                    min_dist = float('inf')
                    for branch, _ in tag_neighbors_dict[comm_tag]:
                        if funcdist:
                            d = self.node_to_node_distance_function_dict[branch][target](link_load)
                        else:
                            d = self.node_to_node_distance_dict[branch][target]
                        if d < min_dist:
                            min_dist = d
                    costs.append(min_dist)
                comm_cost = max(costs) if costs else 0
            else:
                comm_cost = 0

            comm_pair.append(comm_cost)


        #     if tag_paths_dict[comm_tag][comm_pairs_dict[comm_tag][1]]:
        if not deterministic:
            random.shuffle(iter_comm_pairs)
        iter_comm_pairs.sort(key=lambda x: x[5], reverse=long_first)
        iter_comm_dist = []
        iter_comm_dist_dict = {}
        if perdist:
            for comm_pair in list(iter_comm_pairs):
                if comm_pair[5] in iter_comm_dist:
                    iter_comm_dist_dict[comm_pair[5]].append(comm_pair)
                else:
                    iter_comm_dist.append(comm_pair[5])
                    iter_comm_dist_dict[comm_pair[5]] = [comm_pair]
        else:
            iter_comm_dist = [1]
            iter_comm_dist_dict[1] = []
            for comm_pair in list(iter_comm_pairs):
                iter_comm_dist_dict[1].append(comm_pair)
        while iter_comm_dist:
            # iter_comm_pairs.sort(key=lambda x: x[5], reverse=True)
            for dist in iter_comm_dist[:]:
                current_iter_pairs = [cp for cp in iter_comm_dist_dict[dist]]
                #     random.shuffle(current_iter_pairs)
                while current_iter_pairs:
                    for comm_pair in list(current_iter_pairs):
                        if not comm_pair[2]:
                            comm_tag = comm_pair[0]
                            current_iter_pairs.remove(comm_pair)
                            path_source = comm_pairs_dict[comm_tag][1]
                            path_targets = comm_pairs_dict[comm_tag][2]
                            path_targets_idx = set()
                            for path_target in path_targets:
                                path_targets_idx.add(TupleIdx(path_target, None))
                            clean_paths, target_paths = self.get_clean_path(comm_tag, comm_pairs_dict, tag_paths_dict[comm_tag], path_source, path_targets_idx)
                            tag_paths_dict[comm_tag] = clean_paths
                            continue
                        else:
                            comm_tag = comm_pair[0]
                            source_node_idx = comm_pair[1]
                            target_nodes_set = comm_pair[2]
                            link_load = comm_pair[3]
                            other_nodes_set = comm_pair[4]

                            tnd = tag_neighbors_dict[comm_tag]
                            tlcd = tag_links_cost_dict[comm_tag]
                            tpd = tag_paths_dict[comm_tag]
                            tld = tag_lastpair_dict[comm_tag]

                            current_candidate = None
                            current_branch = None
                            current_priority = float('inf')

                            for branch, branch_distance in tnd:
                                avaible_neighbors_list = list(self.available_neighbors[branch])
                                if deterministic:
                                    avaible_neighbors_list.sort()
                                else:
                                    random.shuffle(avaible_neighbors_list)
                                for neighbor in avaible_neighbors_list:
                                    if funcdist:
                                        neighbor_distance = min(self.node_to_node_distance_function_dict[neighbor][t](link_load) for t in target_nodes_set)
                                    else:
                                        neighbor_distance = min(self.node_to_node_distance_dict[neighbor][t] for t in target_nodes_set)
                                    if neighbor in other_nodes_set:
                                        for duplicated_idx in range(self.duplicated_links[(branch, neighbor)] + 1):
                                            if duplicated_idx == 0:
                                                branch_neighbor_pair_idx = TupleIdx((branch, neighbor), idx=None)
                                                branch_idx = TupleIdx(branch, idx=None)
                                                neighbor_idx = TupleIdx(neighbor, idx=None)
                                            else:
                                                branch_neighbor_pair_idx = TupleIdx((branch, neighbor), idx=duplicated_idx)
                                                branch_idx = TupleIdx(branch, idx=duplicated_idx)
                                                neighbor_idx = TupleIdx(neighbor, idx=duplicated_idx)

                                            tlcd_cost = tlcd[(tld[branch], branch)] if branch != source_node_idx else 0 
                                            priority = (
                                                alpha * tlcd_cost 
                                                + beta * self.link_loads_dict[branch_neighbor_pair_idx]
                                                # + beta * self.link_loads_count[branch_neighbor_pair_idx]
                                                + gamma * neighbor_distance
                                            ) 
                                            if priority < current_priority:
                                                current_priority = priority
                                                current_candidate = neighbor
                                                current_candidate_idx = neighbor_idx
                                                current_branch = branch
                                                current_pair_idx = branch_neighbor_pair_idx
                            if current_candidate is not None:
                                if current_candidate in target_nodes_set:
                                    target_nodes_set.remove(current_candidate)

                                tlcd[current_pair_idx.ori] = self.current_link_loads(link_load, self.links_dict[current_pair_idx])
                                if current_branch == source_node_idx:
                                    pass
                                else:
                                    tlcd[current_pair_idx.ori] += tlcd[(tld[current_branch], current_branch)]
                                # tncd[current_candidate] = math.ceil(link_load / self.links_dict[(current_branch, current_candidate)].bandwidth) + self.links_dict[(current_branch, current_candidate)].latency + tncd[current_branch]
                                self.link_loads_dict[current_pair_idx] += self.current_link_loads(link_load, self.links_dict[current_pair_idx])
                                self.link_loads_count[current_pair_idx] += 1
                                other_nodes_set.remove(current_candidate)
                                tpd.setdefault(current_branch, []).append(current_candidate_idx)
                                tld[current_candidate] = current_branch

                                if target_nodes_set == set():
                                    continue

                                if target_nodes_set:
                                    if funcdist:
                                        candidate_farest_distance = max(
                                            self.node_to_node_distance_function_dict[current_candidate][t](link_load) for t in target_nodes_set
                                        )
                                    else:
                                        candidate_farest_distance = max(
                                            self.node_to_node_distance_dict[current_candidate][t] for t in target_nodes_set
                                     )
                                else:
                                    candidate_farest_distance = 0
                                tnd.append([current_candidate, candidate_farest_distance])

                                new_branch_list = []
                                for b, bd in tnd:
                                    still_needed = False
                                    for nb in self.available_neighbors[b]:
                                        if nb not in tlcd:
                                            still_needed = True
                                            break
                                    if still_needed:
                                        new_branch_list.append([b, bd])
                                tag_neighbors_dict[comm_tag] = new_branch_list

                                if target_nodes_set:
                                    costs = []
                                    for target in target_nodes_set:
                                        min_dist = float('inf')
                                        for b, _ in tag_neighbors_dict[comm_tag]:
                                            if funcdist:
                                                d = self.node_to_node_distance_function_dict[b][target](link_load)
                                            else:                                               
                                                d = self.node_to_node_distance_dict[b][target] * link_load
                                            if d < min_dist:
                                                min_dist = d
                                        costs.append(min_dist)
                                    comm_pair[5] = max(costs) if costs else 0
                                else:
                                    comm_pair[5] = 0

                                if pertask:
                                    break 
                iter_comm_dist.remove(dist)

        max_load = max(self.link_loads_count.values()) if self.link_loads_count else 0
        max_load_links = [link for link, count in self.link_loads_count.items() if count == max_load]

        # --- Final Return: maximum load count, total load, and the multicast tree structure ---
        return (
            max_load, 
            sum(self.link_loads_dict.values()), 
            tag_paths_dict,
            comm_pairs_dict
        )


    def backtrack_task(self, 
                       comm_tag: List[int], 
                       current_comm_pairs_dict: Dict[int, List[int]],
                       current_tag_paths_dict: Dict[Tuple[int], Dict[int, List[int]]]
                       ):
        # backtrack found_comm_tag task
        found_tag_paths = current_tag_paths_dict[comm_tag]
        for parent in found_tag_paths:
            for child in found_tag_paths[parent]:
                found_pair = (parent, child)
                self.link_loads_dict[found_pair] -= self.current_link_loads(current_comm_pairs_dict[comm_tag][3], self.links_dict[found_pair])
                self.link_loads_count[found_pair] -= 1


    # del current_tag_paths_dict[comm_tag]
    def backtrack_key_task(self, 
                        current_comm_pairs_dict: Dict[int, List],
                        current_tag_paths_dict: Dict[int, Dict[int, List[int]]],
                        tabu_candidate_list: List = [],
                        tabu_forbidden_set: Set = set(),
                        wrost_flag: bool = True) -> int:
        current_max = max(self.link_loads_count.values()) if self.link_loads_count else 0
        overloaded_links = [link for link, count in self.link_loads_count.items() if count == current_max]
        if not overloaded_links:
            return None
        # Choose one overloaded link.
        if tabu_candidate_list:
            chosen_link = random.choice(overloaded_links) if wrost_flag else random.choice(tabu_candidate_list)
        else:
            chosen_link = random.choice(overloaded_links) if wrost_flag else random.choice(self.links_list)

        # Locate a comm_tag whose multicast tree uses this link.
        found_comm_tag = None
        items_list = list(current_tag_paths_dict.items())
        for comm_tag, branches in random.sample(items_list, len(items_list)):
            # Optional: if only one target remains and a fixed path exists, skip this comm_tag.
            if comm_tag in tabu_forbidden_set:
                continue
            if len(current_comm_pairs_dict[comm_tag][2]) == 1:
                target_tmp = list(current_comm_pairs_dict[comm_tag][2])[0]
                if (current_comm_pairs_dict[comm_tag][1], target_tmp) in self.fixed_paths:
                    continue
            for parent, children in branches.items():
                if chosen_link[0] == parent and chosen_link[1] in children:
                    found_comm_tag = comm_tag
                    break
            if found_comm_tag is not None:
                break
        if found_comm_tag is None:
            return None

        # Backtrack the found_comm_tag task by removing its load contribution.
        found_tag_paths = current_tag_paths_dict[found_comm_tag]
        for parent in found_tag_paths:
            for child in found_tag_paths[parent]:
                if child.idx:
                    found_pair = (parent, child.ori, child.idx)
                else:
                    found_pair = (parent, child)
                self.link_loads_dict[found_pair] -= self.current_link_loads(current_comm_pairs_dict[found_comm_tag][3], self.links_dict[found_pair])
                self.link_loads_count[found_pair] -= 1

        # Optionally, if you want to completely remove the task from the tag paths, uncomment:
        del current_tag_paths_dict[found_comm_tag]

        # Return the found key, not the loop variable.
        return found_comm_tag


    def iter_worse_tasks(self, 
                        original_comm_pairs_dict: Dict[int, List],
                        original_tag_paths_dict: Dict[int, Dict[int, List[int]]],
                        total_iterations: int, 
                        dijkstra_paras: Dict[str, Any] = None,
                        ) -> Tuple[int, int, Dict[int, Dict[int, List[int]]], Dict[int, List]]:

        def smooth_transition(A: int, B: int, n: int):

            result = []

            for i in range(n):
                t = i / (n - 1) if n > 1 else 1
                value = A + (B - A) * (1 - math.cos(math.pi * t)) / 2
                result.append(round(value))

            return result

        """
        Iteratively tries to lower the maximum link load by backtracking and re-routing 
        only the multicast trees that contribute to overloaded links.
        """
        # Work on local copies of the task dictionaries.
        iter_comm_pairs_dict = deepcopy(original_comm_pairs_dict)
        iter_tag_paths_dict = deepcopy(original_tag_paths_dict)

        # Precompute an iteration schedule.
        iteration_loops = smooth_transition(np.ceil(np.sqrt(self.nodes_number)), 2, total_iterations)
        tabu_forbidden_length = int(np.ceil(np.sqrt(self.nodes_number)))
        tabu_forbidden_deque = deque(maxlen=tabu_forbidden_length)
        tabu_forbidden_set = set()
        tabu_candidate_list = []
        for iter_idx in tqdm(range(total_iterations)):
            current_max = max(self.link_loads_count.values()) if self.link_loads_count else 0
            overloaded_links = [link for link, count in self.link_loads_count.items() if count == current_max]
            if not overloaded_links:
                break

            # Backup the current link loads (using shallow copies here since values are numbers).
            backup_tag_paths = {comm_tag: iter_tag_paths_dict[comm_tag] for comm_tag in iter_tag_paths_dict}
            backup_link_loads = {link: self.link_loads_dict[link] for link in self.link_loads_count}
            backup_link_count = {link: self.link_loads_count[link] for link in self.link_loads_count}

            # Attempt backtracking several times according to the iteration schedule.
            backtracked_tasks = set()
            # for _ in range(iteration_loops[iter_idx]):
            while 1:
                random_tag = self.backtrack_key_task(iter_comm_pairs_dict, iter_tag_paths_dict, tabu_candidate_list, tabu_forbidden_set, wrost_flag=(random.random() < 0.9))
                if random_tag:
                    backtracked_tasks.add(random_tag)
                if len(backtracked_tasks) == iteration_loops[iter_idx]:
                    break

            # If no task was found for backtracking, exit the loop.
            if not backtracked_tasks:
                break

            # Recalculate multicast paths only for the tasks that were backtracked.
            comm_pairs_subset = [iter_comm_pairs_dict[comm_tag] for comm_tag in backtracked_tasks]
            new_link_load, new_total_load, new_paths, _ = self.record_dijkstra_multicast_path(
                comm_pairs=comm_pairs_subset, 
                **dijkstra_paras
            )
            new_overloaded = [link for link, count in self.link_loads_count.items() if count == new_link_load]

            # Accept the new state only if an improvement is observed.
            if new_link_load < current_max or (new_link_load == current_max and len(new_overloaded) < len(overloaded_links)):
                for comm_tag in new_paths:
                    if new_paths[comm_tag] != backup_tag_paths[comm_tag]:
                        tabu_candidate_list.append(comm_tag)
                    else:
                        if len(tabu_forbidden_deque) == tabu_forbidden_length:
                            tabu_forbidden_set.remove(tabu_forbidden_deque.popleft())
                        if comm_tag not in tabu_forbidden_set:
                            tabu_forbidden_set.add(comm_tag)
                            tabu_forbidden_deque.append(comm_tag)
                    iter_tag_paths_dict[comm_tag] = new_paths[comm_tag]                    
            else:
                # Otherwise, revert to the previous state.
                for comm_tag in backtracked_tasks:
                    iter_tag_paths_dict[comm_tag] = backup_tag_paths[comm_tag]
                    if len(tabu_forbidden_deque) == tabu_forbidden_length:
                        tabu_forbidden_set.remove(tabu_forbidden_deque.popleft())
                    if comm_tag not in tabu_forbidden_set:
                        tabu_forbidden_set.add(comm_tag)
                        tabu_forbidden_deque.append(comm_tag)
                for link in backup_link_count:
                    self.link_loads_dict[link] = backup_link_loads[link]
                    self.link_loads_count[link] = backup_link_count[link]

        return (max(self.link_loads_count.values()) if self.link_loads_count else 0,
                sum(self.link_loads_count.values()),
                iter_tag_paths_dict,
                original_comm_pairs_dict)


    def dijkstra_link_allocation(
            self, 
            comm_pairs: Dict[int, List],
            multiple_paths: Dict[Tuple[int], Dict[int, List[int]]],
            iteration_number: int = 100
            ): 
        '''
        this function limit:
        1. no dependencies
        2. each task has the same workload in a chiplet
        '''

        def convert_paths(graph, start):
            def explore(node, path):
                if node not in graph or not graph[node]:
                    return [path]
                results = []
                for next_node in graph[node]:
                    results.extend(explore(next_node, path + [(node, next_node)]))
                return results

            return explore(start, [])

        multiple_task_list = []
        multiple_task_idx_tag_dict = []
        for paths in multiple_paths:
            task_list = []
            task_idx_tag_dict = {}
            for comm_idx, comm_tag in enumerate(paths):
                task_list.append([comm_tag, paths[comm_tag], comm_pairs[comm_tag][1]])
                if len(comm_pairs[comm_tag][2]) == 1:
                    task_idx_tag_dict[comm_idx] = (comm_pairs[comm_tag][1][1], list(comm_pairs[comm_tag][2])[0][1])
                else:
                    task_idx_tag_dict[comm_idx] = comm_pairs[comm_tag][1]
            multiple_task_list.append(task_list)
            multiple_task_idx_tag_dict.append(task_idx_tag_dict)

        task_paths_list = []
        for task in multiple_task_list:
            paths_list = []
            for task_path in task:
                paths_list.append(convert_paths(task_path[1], task_path[2]))
            task_paths_list.append(paths_list)          

        min_time_step = float('inf')
        for multiple_paths_idx, paths_list in enumerate(task_paths_list):
            for iteration in range(iteration_number):
                raw_list = deepcopy(paths_list)
                idx_iter_paths_list = [(i, item) for i, item in enumerate(raw_list)]

                if iteration > 0 and iteration % 2 == 0:
                    random.shuffle(idx_iter_paths_list)
                elif iteration % 2 == 1:
                    idx_iter_paths_list.sort(
                       key=lambda pair: (sum(len(lst) for lst in pair[1]), -len(pair[1])), 
                       reverse=True
                    )

                idx_mapping = {new_idx: orig_idx for new_idx, (orig_idx, _) in enumerate(idx_iter_paths_list)}
                iter_paths_list = [item for _, item in idx_iter_paths_list]

                iter_time_step = 0
                iter_steps = []
                while idx_iter_paths_list:
                    available_links = deepcopy(self.links_set)

                    find_flag = True
                    current_step = {}
                    while find_flag:
                        find_flag = False
                        for broadcast_idx, broadcast_paths in idx_iter_paths_list[:]:
                            for paths_idx, paths in enumerate(broadcast_paths[:]):
                                if paths[0] in available_links:
                                    available_links.remove(paths[0])
                                    if (multiple_task_idx_tag_dict[multiple_paths_idx][broadcast_idx], paths_idx) not in current_step:
                                        current_step[(multiple_task_idx_tag_dict[multiple_paths_idx][broadcast_idx], paths_idx)] = [paths[0]]
                                    else:
                                        current_step[(multiple_task_idx_tag_dict[multiple_paths_idx][broadcast_idx], paths_idx)].append(paths[0])
                                    popped_path = paths.pop(0)
                                    if paths == []:
                                        broadcast_paths.remove(paths)
                                        if broadcast_paths == []:
                                            idx_iter_paths_list.remove((broadcast_idx, broadcast_paths))
                                    find_flag = True
                                    break
                                else:
                                    continue
                            for paths in broadcast_paths[:]:
                                if paths[0] == popped_path:
                                    paths.pop(0)
                                    if paths == []:
                                        broadcast_paths.remove(paths)
                                        if broadcast_paths == []:
                                            idx_iter_paths_list.remove((broadcast_idx, broadcast_paths))
                            popped_path = ()

                    iter_time_step += 1
                    iter_steps.append(current_step)

                if iter_time_step < min_time_step:
                    min_time_step = iter_time_step
                    best_steps = [iter_steps]
                elif iter_time_step == min_time_step and iter_steps not in best_steps:
                    best_steps.append(iter_steps)

            return min_time_step, best_steps


if __name__ == "__main__":

    link_idx = TupleIdx((0, 4), idx=None)
    link2_idx = TupleIdx((0, 4), idx=None)

    a = {}
    a[2] = link_idx
    a[3] = link2_idx
    print(a[2].idx)
