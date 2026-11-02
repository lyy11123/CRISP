import os
import sys
import argparse
import random
import pickle
import time
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

file_path = os.path.dirname(os.path.realpath(__file__))
file_real_path = os.path.realpath(__file__)
filename_with_extension = os.path.basename(file_real_path)
filename_without_extension = os.path.splitext(filename_with_extension)[0]

for _p in [
    file_path,
    os.path.join(file_path, '../utils/'),
    os.path.join(file_path, '../src/partition/oper/'),
    os.path.join(file_path, '../src/partition/func/'),
    os.path.join(file_path, '../src/partition/'),
    os.path.join(file_path, '../src/mapping/'),
    os.path.join(file_path, '../src/scheduling/'),
    os.path.join(file_path, '../src/backend/analytical/'),
    os.path.join(file_path, '../src/scheduling/communication/topology/'),
    os.path.join(file_path, '../tool/'),
]:
    if _p not in sys.path:
        sys.path.append(_p)

from read_cfg import cfg_to_dict
from WAMIS_HD import wamis_hdc
from partition import generate_average_degree
from add_communication import add_beha_producers, build_event_v2 as build_event, add_broadcast_v2 as add_broadcast
from data_notation import tensor_notation
from conv1d import Conv1d
from matadd import Matadd
from Pre_Mapping import update_data, initialized_mapping, autoregreesive_dag
from Loop_Mapping import AllMapping
from event_driver import event_driver_v2 as event_driver

BATCH_SIZE         = 1
SEQUENCE_LENGTH    = 512
HIDDEN_STATES      = 3584
HEAD_NUM           = 28
CHIPLETS_PER_LAYER = 1
LAYER_NUM          = 1
PROJ_SPLIT_DEGREE  = (1, 4, 1, 4)
SEQ_DEGREE         = PROJ_SPLIT_DEGREE[1]
HEAD_DEGREE        = PROJ_SPLIT_DEGREE[3]
DIJKSTRA_ROUTING   = False
ALPHA, BETA, GAMMA = 100, 1, 100
HW_CFG_PATH = os.path.join(
    file_path, "../src/platform/cfgs/wamis_hd_distributed.cfg")


def _reset_hw(hp):
    for dev in hp.modules_dict["tensorcore"].values():
        dev.work_flag = False; dev.work_endtime = 0; dev.work_record = []
    for dev in hp.modules_dict["vectorunit"].values():
        dev.work_flag = False; dev.work_endtime = 0; dev.work_record = []
    for lnk in hp.links_dict.values():
        lnk.work_flag = False; lnk.work_endtime = 0; lnk.work_record = []


def _update_data_locs(beha_dict, data_dict, ddr_locs):
    for tag, tensor in data_dict.items():
        if tag == (0, 0) or tag[0] == 1:
            continue
        for split in tensor.generated_split_location:
            tensor.generated_split_location[split] = []
    for tag in ddr_locs:
        data_dict[tag].generated_split_location = deepcopy(ddr_locs[tag])
    for btag, beh in beha_dict.items():
        if beh.location is None:
            continue
        node = beh.location[:-1]
        for produced_tag, splits in beh.produced_data_split_dict.items():
            for split in splits:
                loc_list = data_dict[produced_tag].generated_split_location.setdefault(split, [])
                if node not in loc_list:
                    loc_list.append(node)


def _apply_proj_mapping(beha_dict, tc_list, core_to_vu, block_size,
                        proj_tc_tags, res_tags, P_proj):
    """
    Assign compute behaviors using per-seq-group TC permutations.

    Hardware: 1 chiplet, 4 cores (2×2 mesh), 1 TC + 1 VU per core.
    Conv1d proj: split (1,4,1,HEAD_DEGREE) → SEQ_DEGREE × HEAD_DEGREE TC behaviors.
    Matadd residual: split [1,4,HEAD_DEGREE] → SEQ_DEGREE × HEAD_DEGREE VU behaviors.

    P_proj: list of SEQ_DEGREE permutations of range(n_tc).
      block_size = HEAD_DEGREE // n_tc.
      tile (si, hi): fb = hi // block_size → tc_list[P_proj[si][fb]].

    Residual VU colocates with the proj TC that produces its data.
    """
    n_tc = len(tc_list)

    for li, btag in enumerate(proj_tc_tags):
        si = li // HEAD_DEGREE
        fb = (li % HEAD_DEGREE) // block_size
        beha_dict[btag].location = tc_list[P_proj[si][fb]]

    for li, btag in enumerate(res_tags):
        si = li // HEAD_DEGREE
        fb = (li % HEAD_DEGREE) // block_size
        beha_dict[btag].location = core_to_vu[tc_list[P_proj[si][fb]][:-1]]


def _eval_current_locs(fw):
    bd = fw["bd"]; dd = fw["dd"]; hw = fw["hw"]
    lr = fw["layers_regions"]
    bcast = fw["broadcast_tags"]
    ddr = fw["ddr_locs"]
    ill = fw["init_ll"]; ilc = fw["init_lc"]

    for tag in dd:
        for split in dd[tag].used_splitted_tag_dict:
            dd[tag].used_splitted_tag_dict[split] = {
                t for t in dd[tag].used_splitted_tag_dict[split] if t[0] != 1
            }

    _update_data_locs(bd, dd, ddr)

    for k in hw.link_loads_dict:
        hw.link_loads_dict[k] = ill[k]
        hw.link_loads_count[k] = ilc[k]

    ev = {}
    build_event(beha_dict=bd, data_dict=dd, hardware_platform=hw, event_dict=ev,
                dijkstra_routing=DIJKSTRA_ROUTING, alpha=ALPHA, beta=BETA, gamma=GAMMA)
    add_broadcast(data_tags=bcast, ddr_chiplets=lr[0],
                  beha_dict=bd, data_dict=dd, hardware_platform=hw, event_dict=ev,
                  dijkstra_routing=DIJKSTRA_ROUTING, alpha=ALPHA, beta=BETA, gamma=GAMMA)

    hw_copy = deepcopy(hw)
    _reset_hw(hw_copy)
    tc = event_driver(events_dict=ev, hardware_platform=hw_copy)
    return tc


def _build_framework():
    random.seed(123)
    hw = wamis_hdc(cfg_to_dict(HW_CFG_PATH))
    for tc in hw.modules_dict.get("tensorcore", {}).values():
        tc.utilization_factor = 1

    def _p(dim, deg): return generate_average_degree(dim, deg)
    B, S, H = BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_STATES
    d = PROJ_SPLIT_DEGREE
    proj_para = [_p(B,d[0]), _p(S,d[1]), _p(H,d[2]), _p(H,d[3])]

    dd = {}; bd = {}
    atag = (0, 0)
    dd[atag] = tensor_notation(
        data_name="ifmap", data_tag=atag,
        data_shape=[B, S, H], data_type="bf16")
    dd[atag].dummy_generated_split()

    ont = (0, 1); offt = (1, 0)
    o1, ont, offt, _, proj_o = Conv1d(
        data_dict=dd, beha_dict=bd, oper_name="MHA_proj",
        source_data_tags=[atag], oper_split_list=proj_para,
        oper_tag=(0, 0, 0, 0), online_data_tag=ont, offline_data_tag=offt,
        weight_dim=H, beha_tag_offset=0)
    proj_dt = proj_o.target_data_tags[-1]

    _, ont, offt, _, res_o = Matadd(
        data_dict=dd, beha_dict=bd, oper_name="MHA_residual",
        source_data_tags=[atag, proj_dt],
        oper_split_list=[proj_para[0], proj_para[1], proj_para[3]],
        oper_tag=o1, online_data_tag=ont, offline_data_tag=offt,
        beha_tag_offset=0)
    res_dt = res_o.target_data_tags[0]

    add_beha_producers(beha_dict=bd)
    update_data(beha_dict=bd, data_dict=dd)

    model_dag, model_chiplets = autoregreesive_dag(
        llm_layer_num=LAYER_NUM, chiplets_per_layer=CHIPLETS_PER_LAYER)
    layers_regions = AllMapping(model_dag, model_chiplets, hw)
    initialized_mapping(
        beha_dict=bd, data_dict=dd, hardware_platform=hw,
        layers_regions=layers_regions, greedy_flag=True)

    greedy_locs = {btag: bd[btag].location for btag in bd}

    ddr_locs = {}
    for tag in dd:
        if tag == (0, 0) or tag[0] == 1:
            ddr_locs[tag] = deepcopy(dd[tag].generated_split_location)

    cids   = sorted(set(k[0] for k in hw.modules_dict["tensorcore"]))
    ctcs   = {c: sorted(k for k in hw.modules_dict["tensorcore"] if k[0] == c) for c in cids}
    n_tc   = len(ctcs[cids[0]])
    tc_list = [tc for c in cids for tc in ctcs[c]]

    core_to_vu = {}
    for vu_key in hw.modules_dict["vectorunit"]:
        node = vu_key[:-1]
        if node not in core_to_vu:
            core_to_vu[node] = vu_key

    block_size = HEAD_DEGREE // n_tc

    return {
        "bd": bd, "dd": dd, "hw": hw,
        "layers_regions": layers_regions,
        "proj_o": proj_o, "res_o": res_o,
        "n_tc": n_tc, "tc_list": tc_list,
        "core_to_vu": core_to_vu,
        "block_size": block_size,
        "greedy_locs": greedy_locs,
        "ddr_locs": ddr_locs,
        "init_ll": dict(hw.link_loads_dict),
        "init_lc": dict(hw.link_loads_count),
        "broadcast_tags": [res_dt],
    }


def _worker(task):
    """
    Random search over per-seq-group TC permutations for Proj.

    P_proj: list of SEQ_DEGREE permutations of range(n_tc).
      block_size = HEAD_DEGREE // n_tc.
      tile (si, hi): fb = hi // block_size → tc_list[P_proj[si][fb]].
    Total search space: (n_tc!)^SEQ_DEGREE.

    Round 0 evaluates the greedy mapping.
    """
    worker_id, rounds_budget, seed = task
    fw = _build_framework()
    bd         = fw["bd"]
    proj_o     = fw["proj_o"]; res_o = fw["res_o"]
    tc_list    = fw["tc_list"]
    core_to_vu = fw["core_to_vu"]
    block_size = fw["block_size"]
    n_tc       = fw["n_tc"]
    rng = random.Random(seed)
    t0 = time.time()

    for btag, loc in fw["greedy_locs"].items():
        bd[btag].location = loc
    greedy_time = _eval_current_locs(fw)
    best_time = greedy_time
    best_perm = "greedy"

    for _ in range(1, rounds_budget):
        P_proj = [list(range(n_tc)) for _ in range(SEQ_DEGREE)]
        for p in P_proj: rng.shuffle(p)

        _apply_proj_mapping(
            bd, tc_list, core_to_vu, block_size,
            proj_o.beha_tags, res_o.beha_tags, P_proj)
        t = _eval_current_locs(fw)
        if t[0] < best_time[0]:
            best_time = t
            best_perm = P_proj

    elapsed = time.time() - t0
    print(f"  [worker {worker_id}] greedy={greedy_time[0]}  best={best_time[0]}"
          f"  rounds={rounds_budget}  {elapsed:.1f}s", flush=True)
    return best_time, best_perm, rounds_budget, greedy_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brute-force Proj distributed mapping search via per-seq-group TC permutations")
    parser.add_argument("--rounds", type=int, default=100,
                        help="Total evaluation budget across all workers (default: 100)")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    args = parser.parse_args()

    print("Computing greedy baseline ...")
    fw_main = _build_framework()
    bd_m = fw_main["bd"]
    for btag, loc in fw_main["greedy_locs"].items():
        bd_m[btag].location = loc
    baseline_time = _eval_current_locs(fw_main)
    print(f"Greedy baseline : {baseline_time[0]}")
    del fw_main

    rounds_per_worker = max(1, args.rounds // args.threads)
    tasks = [(i, rounds_per_worker, i * 9973 + 42) for i in range(args.threads)]
    print(f"Launching {args.threads} workers × {rounds_per_worker} rounds "
          f"(total ≤ {args.threads * rounds_per_worker})")

    t_start = time.time()
    all_results = []
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                all_results.append(future.result())
            except Exception as exc:
                print(f"Worker raised: {exc}", flush=True)

    elapsed = time.time() - t_start
    total_rounds = sum(r[2] for r in all_results)

    best = min(all_results, key=lambda r: r[0][0])
    best_time, best_perm, _, _ = best
    worker_greedy_times = [r[3][0] for r in all_results]

    print(f"\n=== Brute-force Proj Distributed Search Complete ===")
    print(f"Threads           : {args.threads}")
    print(f"Total rounds      : {total_rounds}")
    print(f"Elapsed           : {elapsed:.1f}s")
    print(f"Greedy baseline   : {baseline_time[0]}")
    print(f"Worker greedy chk : {worker_greedy_times}  (should all equal baseline)")
    print(f"Best found        : {best_time[0]}")
    if best_time[0] < baseline_time[0]:
        print(f"Improvement       : {(baseline_time[0] - best_time[0]) / baseline_time[0] * 100:.2f}%")
        print(f"Best perm         : P_proj={best_perm}")
    else:
        print("(no improvement found over greedy baseline)")
    print(f"Brute time cost: {best_time}")

    results_dir = os.path.join(file_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    txt_path = os.path.join(results_dir, f"{filename_without_extension}.txt")
    with open(txt_path, "w") as result_f:
        result_f.write(f"Best found: {best_time[0]}\n")
    pkl_path = os.path.join(results_dir, f"{filename_without_extension}_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "threads": args.threads,
            "total_rounds": total_rounds,
            "baseline_time": baseline_time,
            "best_time": best_time,
            "best_perm": best_perm,
            "per_worker_results": [(r[0], r[2], r[3]) for r in all_results],
            "elapsed": elapsed,
        }, f)
    print(f"Results saved to {pkl_path}")
