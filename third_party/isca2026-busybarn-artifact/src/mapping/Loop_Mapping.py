
"""Inter-die mapping: assigns transformer layers to chiplets using ZigZag,
random, and loop-based strategies for autoregressive decoding patterns."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../'))
sys.path.append(os.path.join(file_path, '../../src/scheduling/communication/topology/'))
sys.path.append(os.path.join(file_path, '../../utils/'))


from net import net
from planar_2d import planar_2d
from tlm import tlm2d
from read_cfg import cfg_to_dict


from typing import List, Tuple, Dict, Any
import random
import copy
from copy import deepcopy
from collections import Counter
from simanneal import Annealer
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np


class GroupAssignmentAnnealer(Annealer):
    """Phase 1 SA: assigns chiplets to layers minimizing usage frequency variance
    (even distribution across chiplets). Feeds into ReorderAssignmentAnnealer."""

    def __init__(self, layer_dag: Dict[int, List[int]], chiplet_per_layer: Dict[int, int], network: net):

        '''
        layer_dag: Dict[int, int]
            eg: {0: [1], 1: [2]}
        chiplet_per_layer: Dict[int, int]
            eg: {0: 2, 1: 4, 2: 4}
        '''

        self.layer_dag = layer_dag
        self.chiplet_per_layer = chiplet_per_layer
        self.layer_number = len(chiplet_per_layer)
        self.layers = list(chiplet_per_layer.keys())
        self.network = network

        initial_state = self.generate_initial_solution()
        super(GroupAssignmentAnnealer, self).__init__(initial_state)

        self.copy_strategy = "deepcopy"
        self.ideal_mean = sum([ln for ln in chiplet_per_layer.values()]) / self.layer_number


    def generate_initial_solution(self):

        solution = {}
        for layer_idx in self.chiplet_per_layer:
            chiplet_count = self.chiplet_per_layer[layer_idx]
            chiplets = random.sample(self.network.chiplets_list, chiplet_count)
            solution[layer_idx] = chiplets

        return solution


    def move(self):

        layer_idx = random.sample(self.layers, 1)[0]
        group = self.state[layer_idx]

        ch_idx = random.randrange(self.chiplet_per_layer[layer_idx])

        candidate_pool = self.network.chiplets_set - set(group)
        if candidate_pool:
            new_point = random.choice(list(candidate_pool))
        else:
            return

        group[ch_idx] = new_point


    def energy(self):
        """
        Compute the usage frequency ``freq[p]`` of each chiplet under the
        current state and return
        ``sum((freq[p] - ideal_mean)^2 for p in chiplets)`` as the energy.
        Smaller values indicate a more even distribution.
        """
        freq = Counter()
        for grp in self.state:
            freq.update(self.state[grp])
        for p in self.network.chiplets_set:
            if p not in freq:
                freq[p] = 0

        sq_sum = 0.0
        for p in self.network.chiplets_set:
            diff = freq[p] - self.ideal_mean
            sq_sum += diff * diff
        return sq_sum


class ReorderAssignmentAnnealer(Annealer):
    """Phase 2 SA: reorders chiplet assignments within layers to minimize
    inter-group + intra-group hop distances following the layer DAG."""

    def __init__(self, layer_dag: Dict[int, List[int]], chiplet_per_layer: Dict[int, int], network: net, initial_state: Dict[int, List[int]]):

        '''
        layer_dag: Dict[int, int]
            eg: {0: [1], 1: [2]}
        chiplet_per_layer: Dict[int, int]
            eg: {0: 2, 1: 4, 2: 4}
        ->
        state: Dict[int, List[int]] layer_idx -> [chiplets]
            eg: {0: [0, 1], 1: [2, 3, 4, 5], 2: [6, 7, 8, 9]}
        '''

        self.layer_dag = layer_dag
        self.chiplet_per_layer = chiplet_per_layer
        self.layer_number = len(chiplet_per_layer)
        self.layers = list(chiplet_per_layer.keys())
        self.network = network

        super(ReorderAssignmentAnnealer, self).__init__(initial_state)

        self.copy_strategy = "deepcopy"
        self.ideal_mean = 0


    def move(self):
        # Swap one chiplet between two randomly chosen layers
        layer_idx = random.sample(self.layers, 2)
        swap_layer_idx0 = layer_idx[0]
        swap_layer_idx1 = layer_idx[1]
        group0 = self.state[swap_layer_idx0]
        group1 = self.state[swap_layer_idx1]
        ch_idx0 = random.randrange(self.chiplet_per_layer[swap_layer_idx0])
        ch_idx1 = random.randrange(self.chiplet_per_layer[swap_layer_idx1])
        swap_ch_idx0 = group0[ch_idx0]
        swap_ch_idx1 = group1[ch_idx1]

        if swap_ch_idx0 in group1 or swap_ch_idx1 in group0:
            return 
        group0[ch_idx0] = swap_ch_idx1
        group1[ch_idx1] = swap_ch_idx0


    def energy(self):
        """Weighted sum of intra-group pairwise hops (2x) + inter-group min hops along DAG edges."""

        def inter_group_distance(group1, group2):
            dis = np.inf
            for ch1 in group1:
                for ch2 in group2:
                    current_dis = self.network.ch_to_ch_hop_dict[ch1][ch2]
                    if current_dis < dis:
                        dis = current_dis
            return dis

        def intra_group_distance(group):
            intra_pairs = list(combinations(group, 2))
            dis = 0
            for pair in intra_pairs:
                ch1, ch2 = pair
                dis += self.network.ch_to_ch_hop_dict[ch1][ch2]
            return dis

        dis_sum = 0
        for root in self.layer_dag:
            dis_sum += intra_group_distance(self.state[root]) * 2
            for leaf in self.layer_dag[root]:
                group1 = self.state[root]
                group2 = self.state[leaf]
                if len(group1) == 0 or len(group2) == 0:
                    continue
                dis_sum += inter_group_distance(group1, group2)

        return dis_sum


def draw_chiplet_mapping(best_state, mesh_dim=5, group_label_prefix="Group"):
    # Gather all unique chiplets used
    all_chiplets = set()
    for chips in best_state.values():
        all_chiplets.update(chips)
    max_chiplet_id = max(all_chiplets)
    assert mesh_dim * mesh_dim > max_chiplet_id, "Mesh dim too small for chiplet IDs!"

    # Chiplet positions
    chiplet_positions = {cid: (cid % mesh_dim, cid // mesh_dim) for cid in all_chiplets}

    plt.figure(figsize=(9, 9))
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xticks(range(mesh_dim))
    plt.yticks(range(mesh_dim))
    plt.xlim(-0.5, mesh_dim-0.5)
    plt.ylim(-0.5, mesh_dim-0.5)
    plt.grid(True, zorder=1, color='gray', alpha=0.4)
    plt.title('Chiplet 2D Mesh Mapping', fontsize=18)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)

    # Count chiplet assignments for highlighting
    chiplet_counts = {}
    for chips in best_state.values():
        for cid in chips:
            chiplet_counts[cid] = chiplet_counts.get(cid, 0) + 1

    # Draw chiplet positions (highlight if used >1 times)
    for cid, (x, y) in chiplet_positions.items():
        if chiplet_counts[cid] > 1:
            facecolor = 'orange'
            edgecolor = 'red'
            lw = 3
        else:
            facecolor = 'white'
            edgecolor = 'black'
            lw = 1.5
        plt.scatter(x, y, s=450, color=facecolor, edgecolor=edgecolor, zorder=3, linewidths=lw)
        plt.text(x, y, str(cid), ha='center', va='center', fontsize=15, weight='bold', color='black', zorder=4)

    # Draw group assignments
    colors = plt.cm.get_cmap('tab20', len(best_state))
    for idx, (grp, chips) in enumerate(best_state.items()):
        if len(chips) < 2:
            continue
        xys = [chiplet_positions[cid] for cid in chips]
        xs, ys = zip(*xys)
        plt.plot(xs, ys, color=colors(idx), linewidth=5, alpha=0.9, zorder=2)
        # Group label at the midpoint, with a white background for readability
        mx, my = np.mean(xs), np.mean(ys)
        plt.text(mx, my, f"{grp}", fontsize=11, fontweight='bold',
                 color=colors(idx), zorder=6,
                 bbox=dict(facecolor='white', edgecolor=colors(idx), boxstyle='round,pad=0.3', alpha=0.85))


    # Optional: add a legend for group lines (comment out if too crowded)
    plt.tight_layout()
    plt.show()


def LoopMapping(layer_dag: Dict[int, List[int]], chiplet_per_layer: Dict[int, int], network: planar_2d,
                t_max: int=10, t_min: int=1e-8, steps: int = 2e6) -> Dict[int, List[int]]:
    """Two-phase SA for inter-die mapping: Phase 1 balances chiplet usage, Phase 2 minimizes hops."""

    # Phase 1: balance chiplet usage across layers (coarse search, fewer steps)
    annealer = GroupAssignmentAnnealer(
        layer_dag=layer_dag,
        chiplet_per_layer=chiplet_per_layer,
        network=network
    )

    annealer.Tmax = t_max
    annealer.Tmin = t_min * 100
    annealer.steps = int(steps // 100)

    annealer.copy_strategy = "deepcopy"
    annealer.verbose = True

    best_state, best_energy = annealer.anneal()

    # Phase 2: minimize hop distances by reordering chiplets within groups
    annealer = ReorderAssignmentAnnealer(
        layer_dag=layer_dag,
        chiplet_per_layer=chiplet_per_layer,
        network=network,
        initial_state=best_state
    )

    annealer.Tmax = t_max
    annealer.Tmin = t_min
    annealer.steps = steps

    annealer.copy_strategy = "deepcopy"
    annealer.verbose = True

    best_state, best_energy = annealer.anneal()


    # for i, grp in enumerate(best_state):
    return best_state    


def ZigZagMapping(layer_dag: Dict[int, List[int]], chiplet_per_layer: Dict[int, int], network: planar_2d) -> Dict[int, List[int]]:
    """
    Zigzag mapping strategy for assigning chiplets to layers.
    Note: number of chiplet_per_layer can be smaller than 1 like 0.5
    """
    if sum(chiplet_per_layer.values()) > len(network.chiplets_list):
        raise ValueError("Total chiplets required exceeds available chiplets in the network.")
    state = {}
    zigzag_list = []
    for height_idx in range(network.l2_height):
        if height_idx % 2 == 0:
            row_start = height_idx * network.l2_width
            for width_idx in range(network.l2_width):
                zigzag_list.append(row_start + width_idx)
        else:
            row_start = (height_idx + 1) * network.l2_width - 1
            for width_idx in range(network.l2_width):
                zigzag_list.append(row_start - width_idx)
    ch_count = 0
    for layer_idx in layer_dag:
        ch_start = ch_count
        zigzag_idx = int(np.floor(ch_start))
        state[layer_idx] = [zigzag_list[zigzag_idx]]
        ch_end = ch_start + chiplet_per_layer[layer_idx] - 1e-6
        if np.floor(ch_end) == ch_start + 1:
            state[layer_idx].append(zigzag_list[zigzag_idx + 1])
        else:
            pass
        ch_count += chiplet_per_layer[layer_idx]
    return state


def RandomMapping(layer_dag: Dict[int, List[int]], chiplet_per_layer: Dict[int, int], network: planar_2d) -> Dict[int, List[int]]:
    """
    Random mapping strategy for assigning chiplets to layers.
    """
    if sum(chiplet_per_layer.values()) > len(network.chiplets_list):
        raise ValueError("Total chiplets required exceeds available chiplets in the network.")
    copy_list = deepcopy(network.chiplets_list)
    state = {}
    for layer_idx in layer_dag:
        state[layer_idx] = random.sample(copy_list, int(chiplet_per_layer[layer_idx]))
        for ch in state[layer_idx]:
            copy_list.remove(ch) 
    return state


def AllMapping(layer_dag: Dict[int, List[int]], chiplet_per_layer: Dict[int, int], network: planar_2d) -> Dict[int, List[int]]:
    """
    All mapping strategy for assigning chiplets to layers.
    """
    state = {}
    for layer_idx in layer_dag:
        state[layer_idx] = deepcopy(network.chiplets_list)
    return state


if __name__ == "__main__":
    random.seed(123)


    # for i in range(vit_number):
    llm_number = 24
    llm_chiplets = 0.5

    model_dag = {}
    model_chiplets = {}
    for i in range(llm_number - 1):
        model_dag[i] = [i + 1]
        model_chiplets[i] = llm_chiplets
    model_dag[llm_number - 1] = [0]
    model_chiplets[llm_number - 1] = llm_chiplets

    hardware_cfg = cfg_to_dict(os.path.join(file_path, "./tlm_test.cfg"))
    network = tlm2d(hardware_cfg)


    # for i, grp in enumerate(loop_state):
    zigzag_state = ZigZagMapping(model_dag, model_chiplets, network)
    print("Zigzag Mapping Result:")
    for i, grp in enumerate(zigzag_state):
        print(f"Group {i:2d}: {zigzag_state[grp]}")
