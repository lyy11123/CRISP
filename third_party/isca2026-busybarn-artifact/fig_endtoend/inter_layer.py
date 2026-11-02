"""
Inter-layer group generation, placement, and communication time computation.

Approach:
  1. Fix group number and shapes from hardware config (block tiling)
  2. Place fixed blocks onto the mesh:
     - BusyBarn: SA-optimized group ordering (from Hamiltonian_Mapping)
     - Gemini: zig-zag loop through the mesh
  3. Compute inter-group communication cost analytically on ch2ch links
"""

import os
import sys
import math
import pickle
import hashlib

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../src/mapping/'))

import random
from simanneal import Annealer

CACHE_PATH = os.path.join(file_path, "results", "inter_layer_cache.pkl")


def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def _cerebras_disabled(wafer_rows, wafer_cols, usable_count):
    """Compute disabled die positions for circular wafer mask."""
    center_r = (wafer_rows - 1) / 2.0
    center_c = (wafer_cols - 1) / 2.0
    positions = []
    for r in range(wafer_rows):
        for c in range(wafer_cols):
            dist = math.sqrt((r - center_r) ** 2 + (c - center_c) ** 2)
            positions.append((dist, r * wafer_cols + c))
    positions.sort()
    return sorted([p[1] for p in positions[usable_count:]])


HARDWARE = {
    'dojo': {
        'wafer_rows': 5,
        'wafer_cols': 5,
        'disabled_dies': [],
        'block_shape': (2, 2),
        'ch2ch_bw': 256,
        'ch2ch_latency': 20,
    },
    'cerebras': {
        'wafer_rows': 7,
        'wafer_cols': 12,
        'disabled_dies': _cerebras_disabled(7, 12, 63),
        'block_shape': (1, 3),
        'ch2ch_bw': 256,
        'ch2ch_latency': 20,
    },
    'manual': {
        'wafer_rows': 8,
        'wafer_cols': 8,
        'disabled_dies': [],
        'block_shape': (2, 2),
        'ch2ch_bw': 256,
        'ch2ch_latency': 20,
    },
}

MODELS = {
    'gpt':      {'hidden_dim': 6144, 'layers': 44, 'bytes_per_elem': 2},
    'opt':      {'hidden_dim': 6144, 'layers': 48, 'bytes_per_elem': 2},
    'qwen3moe': {'hidden_dim': 2048, 'layers': 48, 'bytes_per_elem': 2},
    'qwen':     {'hidden_dim': 5120, 'layers': 64, 'bytes_per_elem': 2},
    'qwen2moe': {'hidden_dim': 3584, 'layers': 28, 'bytes_per_elem': 2},
    'llama':    {'hidden_dim': 8192, 'layers': 80, 'bytes_per_elem': 2},
}


def _tile_blocks(wafer_rows, wafer_cols, block_rows, block_cols, disabled=None):
    """Tile wafer with non-overlapping blocks.

    Returns:
        full_blocks: list of groups (each a list of grid positions) that are complete blocks
        remainder:   list of leftover die positions that don't form a complete block
    """
    disabled = set(disabled or [])
    full_blocks = []
    used = set()

    for r in range(0, wafer_rows - block_rows + 1, block_rows):
        for c in range(0, wafer_cols - block_cols + 1, block_cols):
            positions = []
            valid = True
            for dr in range(block_rows):
                for dc in range(block_cols):
                    pos = (r + dr) * wafer_cols + (c + dc)
                    if pos in disabled:
                        valid = False
                        break
                    positions.append(pos)
                if not valid:
                    break
            if valid and len(positions) == block_rows * block_cols:
                full_blocks.append(positions)
                used.update(positions)

    remainder = [pos for pos in range(wafer_rows * wafer_cols)
                 if pos not in used and pos not in disabled]
    return full_blocks, remainder


def generate_groups(hw_name):
    """Generate fixed groups by tiling the mesh with blocks.

    Full blocks get their natural shape_idx (2×2→2, others→1).
    Remainder dies are grouped into target-sized chunks + leftover.

    Returns:
        groups: list of groups (each a list of grid positions)
        shape_indices: list of shape_idx per group
        group_sizes: list of die counts per group
    """
    hw = HARDWARE[hw_name]
    br, bc = hw['block_shape']
    target = br * bc

    full_blocks, remainder = _tile_blocks(
        hw['wafer_rows'], hw['wafer_cols'], br, bc, hw['disabled_dies'])

    if br == 2 and bc == 2:
        block_shape_idx = 2
    elif br == 1 and bc == 1:
        block_shape_idx = 0
    else:
        block_shape_idx = 1

    groups = list(full_blocks)
    shape_indices = [block_shape_idx] * len(full_blocks)

    for i in range(0, len(remainder), target):
        chunk = remainder[i:i + target]
        groups.append(chunk)
        if len(chunk) == 1:
            shape_indices.append(0)
        elif len(chunk) <= 2:
            shape_indices.append(1)
        else:
            coords = [(p // hw['wafer_cols'], p % hw['wafer_cols']) for p in chunk]
            rs = max(c[0] for c in coords) - min(c[0] for c in coords) + 1
            cs = max(c[1] for c in coords) - min(c[1] for c in coords) + 1
            if rs <= 2 and cs <= 2 and len(chunk) == 4:
                shape_indices.append(2)
            else:
                shape_indices.append(1)

    group_sizes = [len(g) for g in groups]
    return groups, shape_indices, group_sizes


def _group_center(group, wafer_cols):
    """Compute center of a group (for SA ordering)."""
    coords = [(p // wafer_cols, p % wafer_cols) for p in group]
    cr = sum(c[0] for c in coords) / len(coords)
    cc = sum(c[1] for c in coords) / len(coords)
    return (cr, cc)


def zigzag_order(groups, wafer_rows, wafer_cols):
    """Order groups by zig-zag (row-major snake) of their centers."""
    centers = [_group_center(g, wafer_cols) for g in groups]
    indexed = list(range(len(groups)))
    indexed.sort(key=lambda i: (
        int(centers[i][0] / 2) * 2,
        centers[i][1] if int(centers[i][0] / 2) % 2 == 0 else -centers[i][1]
    ))
    return [groups[i] for i in indexed]


class _GroupOrderAnnealer(Annealer):
    """SA to find the group ordering that minimizes total consecutive Manhattan distance."""

    def __init__(self, state, centers):
        self.centers = centers
        super().__init__(state)

    def move(self):
        i, j = random.sample(range(len(self.state)), 2)
        self.state[i], self.state[j] = self.state[j], self.state[i]

    def energy(self):
        total = 0
        for k in range(len(self.state) - 1):
            a = self.centers[self.state[k]]
            b = self.centers[self.state[k + 1]]
            total += abs(a[0] - b[0]) + abs(a[1] - b[1])
        return total


def sa_order(groups, hw_name, num_restarts=5, steps=500000):
    """Order groups using SA to minimize total inter-group path distance."""
    hw = HARDWARE[hw_name]
    n = len(groups)
    if n <= 2:
        return groups

    centers = [_group_center(g, hw['wafer_cols']) for g in groups]

    key_data = (hw_name, tuple(tuple(g) for g in groups), num_restarts, steps)
    cache_key = hashlib.md5(pickle.dumps(key_data)).hexdigest()
    cache = _load_cache()
    if cache_key in cache:
        best_state = cache[cache_key]
        return [groups[i] for i in best_state]

    best_state = None
    best_energy = float('inf')
    for _ in range(num_restarts):
        init = list(range(n))
        random.shuffle(init)
        annealer = _GroupOrderAnnealer(init, centers)
        annealer.set_schedule({'tmax': 100.0, 'tmin': 0.01, 'steps': steps, 'updates': 0})
        state, energy = annealer.anneal()
        if energy < best_energy:
            best_energy = energy
            best_state = state

    cache[cache_key] = best_state
    _save_cache(cache)
    return [groups[i] for i in best_state]


def generate_placement(hw_name, method='sa', num_restarts=5, steps=500000):
    """Generate groups with fixed shapes, then order them.

    Args:
        hw_name: 'dojo', 'cerebras', or 'manual'
        method: 'sa' (BusyBarn) or 'zigzag' (Gemini baseline)

    Returns:
        groups: ordered list of groups (each a list of grid positions)
        shape_indices: list of shape_idx per group (same regardless of ordering)
        group_sizes: list of die counts per group
    """
    groups, shape_indices, group_sizes = generate_groups(hw_name)
    hw = HARDWARE[hw_name]

    if method == 'sa':
        ordered = sa_order(groups, hw_name, num_restarts, steps)
    else:
        ordered = zigzag_order(groups, hw['wafer_rows'], hw['wafer_cols'])

    group_to_shape = {tuple(g): s for g, s in zip(groups, shape_indices)}
    ordered_shapes = [group_to_shape[tuple(g)] for g in ordered]
    ordered_sizes = [len(g) for g in ordered]

    return ordered, ordered_shapes, ordered_sizes


def _min_manhattan_distance(group_a, group_b, wafer_cols):
    """Minimum Manhattan distance between any die in group_a and any die in group_b."""
    min_dist = float('inf')
    for pa in group_a:
        ra, ca = pa // wafer_cols, pa % wafer_cols
        for pb in group_b:
            rb, cb = pb // wafer_cols, pb % wafer_cols
            dist = abs(ra - rb) + abs(ca - cb)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def compute_inter_time(groups, hw_name, hidden_dim, seq_len, bytes_per_elem):
    """Compute total inter-layer communication time across all group boundaries.

    For each consecutive group pair (i → i+1):
        hops = min Manhattan distance between groups
        data_size = hidden_dim × seq_len × bytes_per_elem
        transfer_time = hops × ch2ch_latency + data_size / ch2ch_bw

    Returns total inter-layer time (sum over all boundaries).
    """
    hw = HARDWARE[hw_name]
    wafer_cols = hw['wafer_cols']
    ch2ch_bw = hw['ch2ch_bw']
    ch2ch_latency = hw['ch2ch_latency']
    data_size = hidden_dim * seq_len * bytes_per_elem

    total = 0
    for i in range(len(groups) - 1):
        hops = _min_manhattan_distance(groups[i], groups[i + 1], wafer_cols)
        total += hops * ch2ch_latency + data_size / ch2ch_bw
    return total


if __name__ == '__main__':
    for hw_name in ['dojo', 'cerebras', 'manual']:
        hw = HARDWARE[hw_name]
        groups, shapes, sizes = generate_groups(hw_name)
        print(f"{hw_name}: {hw['wafer_rows']}x{hw['wafer_cols']}, "
              f"{len(sizes)} groups, sizes={sizes}, shapes={shapes}")

        zz_groups = zigzag_order(groups, hw['wafer_rows'], hw['wafer_cols'])
        zz_shapes = [shapes[groups.index(g)] for g in zz_groups]

        t = compute_inter_time(zz_groups, hw_name, 6144, 512, 2)
        t_dec = compute_inter_time(zz_groups, hw_name, 6144, 1, 2)
        print(f"  zigzag: inter_time prefill={t:.0f}, decode={t_dec:.0f}")
        print()
