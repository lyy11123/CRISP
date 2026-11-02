# BusyBarn `src/` — Framework Internals

Simulator and optimizer for wafer-scale LLM inference, backing the ISCA 2026
paper *"Mapping and Communication Optimizations with Failure Tolerance for
Wafer-Scale LLM Inference."* This directory implements the BusyBarn framework
itself. Per-figure drivers live outside (`fig_*/`, `models/*/`) and import
everything here via `sys.path` hacks.

See `../CLAUDE.md` for reproduction commands and the top-level `../AGENTS.md`
for contribution guidelines.

---

## Paper → code map

| Paper section | Concept | Code |
|---|---|---|
| §4 Framework · LR notation | Location Relationship data/behavior/operator/function notation | `partition/` |
| §5.1 Inter-die Mapping | ZigZag / Hamiltonian Loop via SA | `mapping/Loop_Mapping.py`, `mapping/allocation.py` |
| §5.2 Intra-die Mapping | Hybrid-parallelism SA with 4-term loss | `mapping/Stream_Mapping.py`, `mapping/Pre_Mapping.py`, `mapping/mapping.py` |
| §6 BALD Communication | Path Profiling (Alg. 1) | `scheduling/communication/topology/net.py` (Dijkstra) |
| §6 BALD Communication | Path Scheduling (Alg. 2) | `scheduling/communication/path_allocator.py` |
| §6 BALD Communication | Heuristic Backtracking (Alg. 3) | `scheduling/communication/link_allocator.py` |
| §6.3 Fault Tolerance | Fault-aware link/node sets, LUT-style reroute | `scheduling/communication/topology/{mesh_2d,WAMIS_HD}.py` |
| §7 Evaluation method | Cycle-accurate analytical backend | `backend/analytical/event_driver.py` |

The string emitted at the end of each experiment — `SA time cost:
(compute_cycles, overlap_cycles, comm_cycles)` — is printed by
`backend/analytical/event_driver.py` and regex-scraped by every `*_pic.py`.

---

## Top-level layout

```
src/
├── partition/   LR notation: data, behavior, operator, function objects + LLM block builders
├── mapping/     SA-based inter-die and intra-die placement engines
├── scheduling/  Event DAG builder + communication inference + routing (topology + allocators)
├── backend/     Event-driven analytical simulator consuming the scheduled DAG
└── platform/    Hardware device models (cores, links, buffers) + cfg templates
```

Tag hierarchy used everywhere:
`(iter_tag, computation_tag, function_tag, operation_tag, behavior_tag)` — the
deeper you go in the hierarchy, the finer the work unit (a behavior is a
single tile executed on a single core).

---

## `partition/` — LR (Location Relationship) notation

Symbolic description of an LLM workload as data slices, producer/consumer
edges, and per-operator compute behaviors. This is the only layer that knows
the model.

| File | Role |
|---|---|
| `data_notation.py` | `tensor_notation` and `tensor_slice_notation`: track split dictionaries, producer/consumer splits, and generated-split locations. `type_bytes` maps dtype names to bytes/elem. Also holds overlap/split-index helpers. |
| `beha_notation.py` | `beha_notation`: one compute tile on one core. Holds `needed_data_split_dict`, `produced_data_split_dict`, `location`, `device` (`"tensorcore"`/`"vectorunit"`). `beha_kind_dict` routes op types to hardware unit kinds. |
| `oper_notation.py` | `oper_notation` base: expands a split strategy into per-tile `beha_notation` instances via Cartesian product of per-dim splits. Subclasses must implement `data_split`, `consume_source_data_and_beha`, `update_consumer_beha`, `update_target_data`. Also provides `generate_grouped_tags_bydim(s)` for collective-comm grouping. |
| `func_notation.py` | `func_notation` base: composite of multiple `oper_notation` stages sharing a split strategy. |
| `partition.py` | Utilities: `generate_average_degree`, `generate_random_degree`, `regenerate_degree` (SA neighbor move on degree tuples), `whole_degree_to_dim_degrees`, `dim_degree_to_split`. |
| `oper/` | Concrete operator kernels subclassing `oper_notation` — `matmul`, `conv1d`, `layernorm`, `rmsnorm`, `softmax`, `norm`, `transpose`, `embedding`, reductions (`redmax`, `redsum`), elementwise (`eleadd`, `elegelu`, `elemul`, `elerelu`, `elerope`, `elesilu`, `elesqrt`, `elepow2`, `eleexp`), vector ops (`vecadd`, `vecdiv`, `vecmac`), `matadd`, `dispatch`, `combine`, `var`, and the abstract bases `elementwise_oper.py`, `reduction_oper.py`, `vectorwise_oper.py`. |
| `func/` | LLM block assemblers: `mha.py`, `ffn.py`, `lmhead.py` — the units regex-tagged as `ln`/`mha`/`proj`/`ffn` in figure result files. |

---

## `mapping/` — SA placement

Two-stage simulated annealing. Both stages use the `simanneal.Annealer`
interface (custom `move`, `energy`, `copy_strategy`).

| File | Role |
|---|---|
| `mapping.py` | Utilities for queue-based core assignment. `unit_pop`/`unit_pop_idx` pop a (tensorcore or vectorunit) location from the per-chiplet deque for a behavior. `core_mapping`, `core_mapping_layers`, `layer_to_chiplet_core_mapping` drive initial random/greedy placement before SA. Seed fixed at 123. |
| `Pre_Mapping.py` | Pre-SA DAG construction: `update_offline_data` (weights/inputs), `update_online_data` (expand coarse splits), `find_exclusive_dependents`, build event dict from behaviors. Feeds Stream_Mapping's initial state. |
| `Loop_Mapping.py` | Inter-die mapping (paper §5.1). `GroupAssignmentAnnealer` (Phase 1 — chiplets → layers minimizing usage-frequency variance) and `ReorderAssignmentAnnealer` (Phase 2 — Hamiltonian loop). `planar_2d` network used as the chiplet-level graph. |
| `Stream_Mapping.py` | Intra-die mapping (paper §5.2). `StreamOptimization` (v1 — full rebuild on revert) and its V2 variant with undo-log based revert. State = `[hops, comm_distances, comm_loads_dict, tc_loads_dict, vu_loads_dict, beha_dict, data_dict, event_dict]`. Energy is the normalized 4-term loss (comm distance + max link/tensor-core/vector-unit workload), weights passed as `loss_ratio`. Supports `dijkstra_routing` flag plus BALD α/β/γ. |
| `allocation.py` | Resource-allocation primitives: `find_layer_per_chiplet` (GCD-based layer/chiplet division), `zigzag_allocation`, rectangular decomposition for die-group shapes, mesh placement. Also parallelizes with `multiprocessing` for large searches. |
| `tlm_test.cfg` | Tiny cfg used when `mapping/*.py` is run standalone. |

**On the `loss_ratio` sweep.** The 4-term loss has no single universally-best
weighting — which term dominates depends on the operator shape, hardware
aspect ratio, and fault pattern. For a small TC-bound matmul on a square
mesh, tensor-core load is the long pole; for a wide FFN with a failed node,
comm distance + link load dominate. Experiments therefore sweep several
`loss_ratio` presets per configuration (e.g. `lossratio0`…`lossratio6` in the
figure filenames) and the plot scripts' `find_min_abc` keeps the best
resulting `SA time cost`. Read the sweep as a probe of *which loss term is
the right proxy for wall-clock latency in this case*, not as tuning of a
single hyperparameter.

---

## `scheduling/` — event DAG + communication

Converts a mapped behavior DAG into a fully-timed event DAG of computation
and communication events, then routes the comm events through the network.

| File | Role |
|---|---|
| `event_notation.py` | `event_notation` base (tracks `dependency_set`/`issue_set`/`start_time`/`end_time`). `computation_notation`: event mapped to a `(y, x, device_idx)` location; owns an `IDAllocator` (heap-based, recycling) for its child comm IDs. `communication_notation`: src→dst p2p or multicast with `paths` (routing tree) and `path_list` (flat link hops). |
| `event_builder.py` | Top-level orchestrator. `aggregate_relationships` (splits → common-target batches, replication allowed) and `aggregate_identical` (merges only identical target sets) determine how data-split multicasts are grouped into streams. `event_builder` walks `op_dict` + `data_dict` to emit computation events, then inserts comm events via `path_allocator`. |
| `add_communication.py` | Communication inference: inspects each behavior's `needed`/`produced` splits, adds unicast/multicast/reduction comm events and intermediate data edges. `build_event`/`update_event` (v1) and `update_event_v2` + `update_offlinedata_v2` + `_undo_init` (v2 with undo-log based revert for SA speedup). `add_beha_producers` builds reverse edges. Supports XY routing or Dijkstra routing with BALD weights α/β/γ. |
| `alltoall.py` | `alltoall_mesh2d_tasks` — generates all-pair task lists for a 2D-mesh region, used by collective-communication experiments. |
| `communication/path_allocator.py` | **BALD Path Scheduling (Alg. 2)**. `pair_path_allocation` iterates tasks, computes hops, offloads Dijkstra, and picks paths balancing branch cost / link load / neighbor distance. |
| `communication/link_allocator.py` | **BALD Heuristic Backtracking (Alg. 3)**. `pair_link_allocate` runs timestep-by-timestep greedy link allocation with tabu candidates/forbiddens and probability ρ. |
| `communication/collective_communication/` | `alltoall.py` (base all-to-all task generator), `scatter.py`, `custom.py` — pattern templates for paper Fig. 8 experiments. |
| `communication/topology/net.py` | **Abstract `net`**. Core of BALD Path Profiling (Alg. 1): all-pairs Dijkstra with unique-path map (`dijkstra_offload`, `record_dijkstra_path`), multicast tree construction, link-load tracking, `TupleIdx` for multi-link edges. Parameters `alpha`/`beta`/`gamma` are the BALD priority weights. |
| `communication/topology/planar_2d.py` | Base 2D cfg-driven topology: chiplet/core grids, DDR/SRAM, failure injection. |
| `communication/topology/mesh_2d.py` | Flat 2D mesh (no chiplet hierarchy). XY/YX routing, Dijkstra, multicast tree, AllGather/AllToAll pattern generators. |
| `communication/topology/tlm.py` | **Two-Level Mesh**: intra-chiplet `co2co` mesh + single inter-chiplet `ch2ch` link per edge (center core of each side). The default topology for most figures. |
| `communication/topology/WAMIS_HD.py` / `WAMIS_HD_single.py` / `WAMIS_HD_around.py` | Wafer-scale multi-die variants with high-bandwidth DDR links, fault-tolerant XY-YX backtracking, distance-aware multicast. |

---

## `backend/analytical/event_driver.py` — timing simulator

Cycle-accurate event-driven driver. Maintains:

- `issued_events` keyed on `end_time`, plus a min-heap (`_end_heap`) for fast
  earliest-finish lookup.
- `working_computation_devices` / `working_communication_devices` release sets.
- Per-phase advance: release finished work → issue newly-ready events →
  advance time to next event end.

Entry points:

- `collective_event_driver(events_dict, hardware_platform)` → returns
  `(total_cycles, pure_comp_cycles, pure_comm_cycles)`. This is the tuple that
  plot scripts scrape via `SA time cost: (a, b, c)` regex, where
  `overlap = total − comp − comm`.

Device working-time logic lives in the hardware modules (see below).

---

## `platform/` — hardware device models

| File | Role |
|---|---|
| `device/device.py` | Abstract `device`: `work_flag`, `work_endtime`, `work_record`. All hardware classes inherit. |
| `device/storage.py` | Abstract buffer base. |
| `device/module/module.py` | Abstract compute module. |
| `device/module/tensorcore.py` | `tensorcore` (`multree` type): `working_time(source_datashape, beha_type, frequency)` models matmul/matmac cycles via grain-sized inner/outer loops. |
| `device/module/vectorunit.py` | `vectorunit` (`simd` type): per-op complexity factors (`eleadd_complexity`, `elegelu_complexity`, …) × grain size. |
| `device/buffer/sram.py`, `ddr.py`, `buffer.py` | Capacity-tracking SRAM/DDR buffers. |
| `device/link/link.py` | Abstract link. |
| `device/link/co2co.py`, `ch2ch.py`, `co2ddr.py`, `ddr2co.py` | Concrete link types with latency/bandwidth/timeunit parsed from cfg. |
| `cfgs/wamis_hd_distributed.cfg` | Default WAMIS-HD cfg: frequency, chiplet/core dims, `tensorcore_grain`, `vectorunit_grain`, per-op complexity, link latency/bandwidth, SRAM/DDR capacity, `[failures]` block for `failed_nodes`/`failed_links`. |
| `cfgs/wamis_hd_single.cfg`, `wamis_hd_round.cfg`, `wamis_hd_tpuv5e4.cfg` | Alternate hardware sizings used by specific figures. |

Each `fig_*/cfg.py` writes concrete `.cfg` variants of these templates;
`utils/read_cfg.py` (outside `src/`) parses them back into dicts consumed by
`net.__init__`, `tensorcore.parse_cfg`, etc.

---

## End-to-end pipeline for one experiment

A per-figure script (e.g. `fig_intra_power/py/mha_sq512_greedy0_lossratio0_ch1x1_bw256_co4x4_bw256_t128x128_failpattern0.py`) runs roughly:

1. **Parse cfg** → build `tlm2d` / `WAMIS_HD` `net` instance from
   `platform/cfgs/*.cfg`. (`utils/read_cfg.py`)
2. **Build LR notation** → instantiate `func/{mha,ffn,lmhead}.py` or raw
   `oper/*` operators with split lists. This produces `data_dict`,
   `beha_dict`. (`partition/`)
3. **Inter-die mapping** (optional per figure) → run `Loop_Mapping`
   annealers to assign layers to chiplets. (`mapping/Loop_Mapping.py`)
4. **Intra-die mapping** → `Pre_Mapping.update_offline_data/update_online_data`
   populate the initial state; `Stream_Mapping.StreamOptimization` runs SA
   over operator-to-core placement, calling `add_communication.update_event*`
   every move to re-evaluate the 4-term loss.
5. **Communication routing** → `net.dijkstra_offload` (profiling),
   `path_allocator.pair_path_allocation` (scheduling), optional
   `link_allocator.pair_link_allocate` (backtracking). Each pair of
   `(src, dst, bytes)` attaches a `communication_notation` path into the
   event DAG.
6. **Timing** → `backend/analytical/event_driver.collective_event_driver`
   consumes the event DAG and emits the `(compute, overlap, comm)` tuple,
   which the parent `*_pic.py` regex-scrapes to build the figure.

Faults flow through as the `[failures]` cfg block → `failed_nodes_set` /
`failed_links_set` on `net` → removed from `nodes_set` / `links_set` during
`init_links` → BALD reroutes around them and `event_driver` sees the reduced
graph. Comparing `variant=busybarn` vs `variant=gemini` is a runtime flag on
the SA (`Stream_Mapping`) and routing (`add_communication`) layers, not a
separate code path.
