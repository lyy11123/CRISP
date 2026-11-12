"""Fixed 7x8 layouts and byte counts for the GPT W1 experiment in E_PLAN.md."""

from dataclasses import asdict, dataclass


ROWS, COLS = 7, 8
LAYERS = 44
CORES_PER_DIE = 16
HIDDEN, FFN_HIDDEN, BATCH, SEQUENCE, KV_LENGTH, BYTES_PER_ELEMENT = 6144, 24576, 16, 2048, 2048, 2
D2D_BYTES_PER_CYCLE, D2D_LATENCY = 256, 20
SRAM_BYTES_PER_CORE = 16 * 1024**2
DDR_BYTES_PER_DIE_ASSUMED = 32 * 1024**3
DDR_PORTS_PER_DIE = 8
WEIGHT_MATRICES = BYTES_PER_ELEMENT * (4 * HIDDEN**2 + 2 * HIDDEN * FFN_HIDDEN)
BIAS_AND_LN = BYTES_PER_ELEMENT * (5 * HIDDEN + FFN_HIDDEN + 4 * HIDDEN)
SHARED_KV = 2 * KV_LENGTH * HIDDEN * BYTES_PER_ELEMENT
DAMAGE_GROUP = "G2"

Die = tuple[int, int]


@dataclass(frozen=True)
class Group:
    name: str
    shape: str
    dies: tuple[Die, ...]
    layer_start: int
    layer_stop: int

    @property
    def layers(self):
        return self.layer_stop - self.layer_start


@dataclass(frozen=True)
class Boundary:
    source_group: str
    target_group: str
    source_die: Die
    target_die: Die
    source_core: int
    target_core: int


@dataclass(frozen=True)
class Layout:
    name: str
    groups: tuple[Group, ...]
    boundaries: tuple[Boundary, ...]

    def transfer_cycles(self, phase):
        return len(self.boundaries) * boundary_cycles(phase)

    def describe(self):
        return {
            "mesh": [ROWS, COLS],
            "layers": LAYERS,
            "layer_range": "zero-based, stop excluded",
            "groups": [asdict(group) | {"layers": group.layers} for group in self.groups],
            "boundaries": [asdict(boundary) | {"hops": 1} for boundary in self.boundaries],
            "transfer_bytes_per_boundary": {phase: activation_bytes(phase) for phase in ("prefill", "decode")},
            "transfer_cycles": {phase: self.transfer_cycles(phase) for phase in ("prefill", "decode")},
            "d2d_bytes_per_cycle": D2D_BYTES_PER_CYCLE,
            "d2d_latency_cycles": D2D_LATENCY,
        }


def activation_bytes(phase):
    tokens = {"prefill": SEQUENCE, "decode": 1}[phase]
    return BATCH * tokens * HIDDEN * BYTES_PER_ELEMENT


def boundary_cycles(phase):
    return D2D_LATENCY + activation_bytes(phase) // D2D_BYTES_PER_CYCLE


def boundary_between(source, target):
    source_die, target_die = min(
        (a, b) for a in source.dies for b in target.dies
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    )
    direction = (target_die[0] - source_die[0], target_die[1] - source_die[1])
    # 4x4 core grid, first non-corner port on the facing edge of each Die.
    source_core, target_core = {(0, 1): (7, 4), (0, -1): (4, 7), (1, 0): (13, 1), (-1, 0): (1, 13)}[direction]
    return Boundary(source.name, target.name, source_die, target_die, source_core, target_core)


def make_layout(name, anchors):
    groups = []
    start = 0
    for index, (row, col, height, width) in enumerate(anchors):
        dies = tuple((r, c) for r in range(row, row + height) for c in range(col, col + width))
        layers = 4 if index < 2 else 3
        groups.append(Group(f"G{index}", f"{height}x{width}", dies, start, start + layers))
        start += layers
    die_list = [die for group in groups for die in group.dies]
    mesh = {(r, c) for r in range(ROWS) for c in range(COLS)}
    if start != LAYERS or len(die_list) != len(mesh) or set(die_list) != mesh:
        raise ValueError(f"Invalid coverage in {name}")
    boundaries = tuple(boundary_between(a, b) for a, b in zip(groups, groups[1:]))
    return Layout(name, tuple(groups), boundaries)


def layouts():
    mix = [(row, col, 2, 2) for row, cols in ((0, (0, 2, 4, 6)), (2, (6, 4, 2, 0)), (4, (0, 2, 4, 6)))
           for col in cols] + [(6, 4, 1, 4), (6, 0, 1, 4)]
    strip = [(row, col, 1, 4) for row in range(ROWS) for col in ((4, 0) if row % 2 else (0, 4))]
    return make_layout("L_mix", mix), make_layout("L_strip", strip)


def capacity_rows(layout):
    for group in layout.groups:
        die_count = len(group.dies)
        yield {
            "layout": layout.name, "group": group.name, "shape": group.shape, "layers": group.layers,
            "weight_matrix_bytes": group.layers * WEIGHT_MATRICES,
            "bias_and_ln_bytes": group.layers * BIAS_AND_LN,
            "weights_bytes": group.layers * (WEIGHT_MATRICES + BIAS_AND_LN),
            "shared_kv_bytes": group.layers * SHARED_KV,
            "optional_duplicate_k_bytes": group.layers * SHARED_KV // 2,
            "prefill_one_activation_buffer_bytes": activation_bytes("prefill"),
            "prefill_two_activation_buffers_bytes": 2 * activation_bytes("prefill"),
            "decode_one_activation_buffer_bytes": activation_bytes("decode"),
            "decode_two_activation_buffers_bytes": 2 * activation_bytes("decode"),
            "sram_bytes_per_core": SRAM_BYTES_PER_CORE,
            "sram_bytes_group_aggregate": die_count * CORES_PER_DIE * SRAM_BYTES_PER_CORE,
            "ddr_ports": die_count * DDR_PORTS_PER_DIE,
            "ddr_bytes_per_die_assumed": DDR_BYTES_PER_DIE_ASSUMED,
            "ddr_bytes_group_assumed": die_count * DDR_BYTES_PER_DIE_ASSUMED,
        }
