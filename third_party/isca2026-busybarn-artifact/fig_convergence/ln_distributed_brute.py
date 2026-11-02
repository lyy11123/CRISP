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
from rmsnorm import Rmsnorm
from Pre_Mapping import update_data, initialized_mapping, autoregreesive_dag
from Loop_Mapping import AllMapping
from event_driver import event_driver_v2 as event_driver

BATCH_SIZE         = 1
SEQUENCE_LENGTH    = 512
HIDDEN_STATES      = 3584
CHIPLETS_PER_LAYER = 1
LAYER_NUM          = 1
LN_SPLIT_0         = (1, 4, 1)
LN_SPLIT_1         = (1, 2, 2)
PHASE0_SEQ_DEGREE  = LN_SPLIT_0[1]
PHASE1_COUNT       = LN_SPLIT_1[1] * LN_SPLIT_1[2]
PHASE0_OPS         = 5
PHASE0_BEHA_COUNT  = PHASE0_OPS * PHASE0_SEQ_DEGREE
DIJKSTRA_ROUTING   = True
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


def _apply_ln_mapping(beha_dict, vu_list, all_ln_tags, P0, P1):
    """
    Assign all LN (Rmsnorm) VU behaviors using two permutations.

    Phase 0 (offsets 0..19): 5 child ops × 4 seq groups, split (1,4,1).
      P0[si] = VU index for seq group si.
    Phase 1 (offsets 20..27): 2 child ops × 4 groups, split (1,2,2).
      P1[bi] = VU index for group bi (bi = si*2 + hi, si∈{0,1}, hi∈{0,1}).
    """
    for btag in all_ln_tags:
        offset = btag[-1]
        if offset < PHASE0_BEHA_COUNT:
            si = offset % PHASE0_SEQ_DEGREE
            beha_dict[btag].location = vu_list[P0[si]]
        else:
            bi = (offset - PHASE0_BEHA_COUNT) % PHASE1_COUNT
            beha_dict[btag].location = vu_list[P1[bi]]


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

    ln_para_0 = [_p(B, LN_SPLIT_0[0]), _p(S, LN_SPLIT_0[1]), _p(H, LN_SPLIT_0[2])]
    ln_para_1 = [_p(B, LN_SPLIT_1[0]), _p(S, LN_SPLIT_1[1]), _p(H, LN_SPLIT_1[2])]

    dd = {}; bd = {}
    atag = (0, 0)
    dd[atag] = tensor_notation(
        data_name="ifmap", data_tag=atag,
        data_shape=[B, S, H], data_type="bf16")
    dd[atag].dummy_generated_split()

    ont = (0, 1); offt = (1, 0)
    _, ont, offt, _, rms_o = Rmsnorm(
        data_dict=dd, beha_dict=bd, oper_name="LN",
        source_data_tags=[atag], oper_split_list=ln_para_0 + ln_para_1,
        oper_tag=(0, 0, 0, 0), online_data_tag=ont, offline_data_tag=offt)
    ln_data_tag = rms_o.target_data_tags[-1]

    add_beha_producers(beha_dict=bd)
    update_data(beha_dict=bd, data_dict=dd)

    model_dag, model_chiplets = autoregreesive_dag(
        llm_layer_num=LAYER_NUM, chiplets_per_layer=CHIPLETS_PER_LAYER)
    layers_regions = AllMapping(model_dag, model_chiplets, hw)
    initialized_mapping(
        beha_dict=bd, data_dict=dd, hardware_platform=hw,
        layers_regions=layers_regions, greedy_flag=False)

    greedy_locs = {btag: bd[btag].location for btag in bd}

    ddr_locs = {}
    for tag in dd:
        if tag == (0, 0) or tag[0] == 1:
            ddr_locs[tag] = deepcopy(dd[tag].generated_split_location)

    all_ln_tags = sorted(btag for btag in bd if btag[:-1] == rms_o.oper_tag)

    vu_list = sorted(hw.modules_dict["vectorunit"].keys())

    return {
        "bd": bd, "dd": dd, "hw": hw,
        "layers_regions": layers_regions,
        "all_ln_tags": all_ln_tags,
        "n_vu": len(vu_list),
        "vu_list": vu_list,
        "greedy_locs": greedy_locs,
        "ddr_locs": ddr_locs,
        "init_ll": dict(hw.link_loads_dict),
        "init_lc": dict(hw.link_loads_count),
        "broadcast_tags": [ln_data_tag],
    }


def _worker(task):
    """
    Random search over VU permutations for LN.

    P0: permutation of range(n_vu=4) for phase 0 seq groups.
    P1: permutation of range(n_vu=4) for phase 1 (2seq × 2hidden) groups.
    Total search space: (4!)^2 = 576.

    Round 0 evaluates the greedy mapping.
    """
    worker_id, rounds_budget, seed = task
    fw = _build_framework()
    bd          = fw["bd"]
    all_ln_tags = fw["all_ln_tags"]
    vu_list     = fw["vu_list"]
    n_vu        = fw["n_vu"]
    rng = random.Random(seed)
    t0 = time.time()

    for btag, loc in fw["greedy_locs"].items():
        bd[btag].location = loc
    greedy_time = _eval_current_locs(fw)
    best_time = greedy_time
    best_perm = "greedy"

    for _ in range(1, rounds_budget):
        P0 = list(range(n_vu)); rng.shuffle(P0)
        P1 = list(range(n_vu)); rng.shuffle(P1)

        _apply_ln_mapping(bd, vu_list, all_ln_tags, P0, P1)
        t = _eval_current_locs(fw)
        if t[0] < best_time[0]:
            best_time = t
            best_perm = (P0, P1)

    elapsed = time.time() - t0
    print(f"  [worker {worker_id}] greedy={greedy_time[0]}  best={best_time[0]}"
          f"  rounds={rounds_budget}  {elapsed:.1f}s", flush=True)
    return best_time, best_perm, rounds_budget, greedy_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brute-force LN distributed mapping search via VU permutations")
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

    print(f"\n=== Brute-force LN Distributed Search Complete ===")
    print(f"Threads           : {args.threads}")
    print(f"Total rounds      : {total_rounds}")
    print(f"Elapsed           : {elapsed:.1f}s")
    print(f"Greedy baseline   : {baseline_time[0]}")
    print(f"Worker greedy chk : {worker_greedy_times}  (should all equal baseline)")
    print(f"Best found        : {best_time[0]}")
    if best_time[0] < baseline_time[0]:
        print(f"Improvement       : {(baseline_time[0] - best_time[0]) / baseline_time[0] * 100:.2f}%")
        print(f"Best perm         : P0={best_perm[0]}  P1={best_perm[1]}")
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
