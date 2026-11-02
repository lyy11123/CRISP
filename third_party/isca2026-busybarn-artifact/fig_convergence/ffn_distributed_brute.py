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
from elesilu import Elesilu
from matadd import Matadd
from Pre_Mapping import update_data, initialized_mapping, autoregreesive_dag
from Loop_Mapping import AllMapping
from event_driver import event_driver_v2 as event_driver

BATCH_SIZE         = 1
SEQUENCE_LENGTH    = 512
HIDDEN_STATES      = 3584
FFN_DIMS           = 18944
CHIPLETS_PER_LAYER = 4
LAYER_NUM          = 1
MLP1_SPLIT_DEGREE  = (1, 4, 1, 8)
FFN_DEGREE         = MLP1_SPLIT_DEGREE[3]
SEQ_DEGREE         = MLP1_SPLIT_DEGREE[1]
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


def _apply_dist_mapping(beha_dict, tc_list, core_to_vu, block_size,
                        m1_tags, m2_tags, m3_tc_tags,
                        act_tags, mul_tags, mlp3_vu_tags, res_tags,
                        P1, P2, P3):
    """
    Assign compute behaviors using per-seq-group TC permutations.

    Hardware layout: 1 chiplet, 4 cores (2×2 mesh), 1 TC + 1 VU per core.
    tc_list has 4 entries (one per core).

    P1/P2/P3: each is a list of SEQ_DEGREE permutations of range(n_tc).
      P1[si] = permutation of TC indices for seq group si, operator mlp1.

    Tile assignment:
      block_size = FFN_DEGREE // n_tc   (= 8 // 4 = 2)
      tile (si, fb): fb = (li % FFN_DEGREE) // block_size ∈ {0,1,2,3}
      → tc_list[P1[si][fb]]

    VU behaviors colocate on the same core as TC tile (si, fb=0) from P1.
    core_to_vu maps node (chiplet_idx, core_idx) → VU key on that node.
    """
    n_tc = len(tc_list)

    for li, btag in enumerate(m1_tags):
        si = li // FFN_DEGREE
        fb = (li % FFN_DEGREE) // block_size
        beha_dict[btag].location = tc_list[P1[si][fb]]

    for li, btag in enumerate(m2_tags):
        si = li // FFN_DEGREE
        fb = (li % FFN_DEGREE) // block_size
        beha_dict[btag].location = tc_list[P2[si][fb]]

    for li, btag in enumerate(m3_tc_tags):
        si = li // FFN_DEGREE
        fb = (li % FFN_DEGREE) // block_size
        beha_dict[btag].location = tc_list[P3[si][fb]]

    for li, btag in enumerate(act_tags):
        si = li // FFN_DEGREE
        anchor_tc = tc_list[P1[si][0]]
        beha_dict[btag].location = core_to_vu[anchor_tc[:-1]]

    for li, btag in enumerate(mul_tags):
        si = li // FFN_DEGREE
        anchor_tc = tc_list[P1[si][0]]
        beha_dict[btag].location = core_to_vu[anchor_tc[:-1]]

    for j, btag in enumerate(mlp3_vu_tags):
        anchor_tc = tc_list[P1[j][0]]
        beha_dict[btag].location = core_to_vu[anchor_tc[:-1]]

    for j, btag in enumerate(res_tags):
        anchor_tc = tc_list[P1[j][0]]
        beha_dict[btag].location = core_to_vu[anchor_tc[:-1]]


def _eval_current_locs(fw):
    """
    Evaluate whatever locations are currently set in beha_dict.
    Always does a FULL rebuild: _update_data_locs → build_event →
    add_broadcast → event_driver.  Returns total_time.
    """
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
    """Build all framework objects deterministically (random.seed(123))."""
    random.seed(123)
    hw = wamis_hdc(cfg_to_dict(HW_CFG_PATH))
    for tc in hw.modules_dict.get("tensorcore", {}).values():
        tc.utilization_factor = 1

    def _p(dim, deg): return generate_average_degree(dim, deg)
    B, S, H, F = BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_STATES, FFN_DIMS
    d = MLP1_SPLIT_DEGREE
    mlp1_para  = [_p(B,d[0]), _p(S,d[1]), _p(H,d[2]), _p(F,d[3])]
    act_para   = [_p(B,d[0]), _p(S,d[1]), _p(F,d[3])]
    mlp2_para  = [_p(B,d[0]), _p(S,d[1]), _p(H,d[2]), _p(F,d[3])]
    matadd_para= [_p(B,d[0]), _p(S,d[1]), _p(F,d[3])]
    mlp3_para  = [_p(B,d[0]), _p(S,d[1]), _p(F,d[3]), _p(H,d[2])]

    dd = {}; bd = {}
    atag = (0, 0)
    dd[atag] = tensor_notation(
        data_name="ifmap", data_tag=atag,
        data_shape=[B, S, H], data_type="bf16")
    dd[atag].dummy_generated_split()

    ont = (0, 1); offt = (1, 0)
    o1, ont, offt, _, m1o = Conv1d(
        data_dict=dd, beha_dict=bd, oper_name="MLPmlp1",
        source_data_tags=[atag], oper_split_list=mlp1_para,
        oper_tag=(0, 0, 0, 0), online_data_tag=ont, offline_data_tag=offt,
        weight_dim=F, beha_tag_offset=0)
    m1dt = m1o.target_data_tags[-1]

    o2, ont, offt, _, ao = Elesilu(
        data_dict=dd, beha_dict=bd, oper_name="MLP_act",
        source_data_tags=[m1dt], oper_split_list=act_para,
        oper_tag=o1, online_data_tag=ont, offline_data_tag=offt, beha_tag_offset=0)
    adt = ao.target_data_tags[0]

    o3, ont, offt, _, m2o = Conv1d(
        data_dict=dd, beha_dict=bd, oper_name="MLPmlp2",
        source_data_tags=[atag], oper_split_list=mlp2_para,
        oper_tag=o2, online_data_tag=ont, offline_data_tag=offt,
        weight_dim=F, beha_tag_offset=0)
    m2dt = m2o.target_data_tags[-1]

    o4, ont, offt, _, muo = Matadd(
        data_dict=dd, beha_dict=bd, oper_name="MLP_mul",
        source_data_tags=[adt, m2dt], oper_split_list=matadd_para,
        oper_tag=o3, online_data_tag=ont, offline_data_tag=offt, beha_tag_offset=0)
    mudt = muo.target_data_tags[0]

    o5, ont, offt, _, m3o = Conv1d(
        data_dict=dd, beha_dict=bd, oper_name="MLPmlp3",
        source_data_tags=[mudt], oper_split_list=mlp3_para,
        oper_tag=o4, online_data_tag=ont, offline_data_tag=offt,
        weight_dim=H, beha_tag_offset=0,
        reduction_split_list=[mlp3_para[0], mlp3_para[1], mlp3_para[3]])
    m3dt = m3o.target_data_tags[-1]

    _, ont, offt, _, ro = Matadd(
        data_dict=dd, beha_dict=bd, oper_name="MLP_residual",
        source_data_tags=[atag, m3dt],
        oper_split_list=[mlp3_para[0], mlp3_para[1], mlp3_para[3]],
        oper_tag=o5, online_data_tag=ont, offline_data_tag=offt, beha_tag_offset=0)
    res_dt = ro.target_data_tags[0]

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

    cids  = sorted(set(k[0] for k in hw.modules_dict["tensorcore"]))
    ctcs  = {c: sorted(k for k in hw.modules_dict["tensorcore"] if k[0] == c) for c in cids}
    n_tc  = len(ctcs[cids[0]])
    tc_list = [tc for c in cids for tc in ctcs[c]]

    core_to_vu = {}
    for vu_key in hw.modules_dict["vectorunit"]:
        node = vu_key[:-1]
        if node not in core_to_vu:
            core_to_vu[node] = vu_key

    block_size = FFN_DEGREE // n_tc

    mlp3_vu_start = SEQ_DEGREE * FFN_DEGREE
    mlp3_vu_tags = tuple((*m3o.oper_tag, mlp3_vu_start + j) for j in range(SEQ_DEGREE))

    return {
        "bd": bd, "dd": dd, "hw": hw,
        "layers_regions": layers_regions,
        "m1o": m1o, "ao": ao, "m2o": m2o, "muo": muo, "m3o": m3o, "ro": ro,
        "mlp3_vu_tags": mlp3_vu_tags,
        "cids": cids, "ctcs": ctcs,
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
    Random search over per-seq-group TC permutations.

    Hardware: 1 chiplet, 4 cores (2×2 mesh), 4 TCs total.
    MLP1_SPLIT_DEGREE = (1, 4, 1, 8): 4 seq groups × 8 FFN partitions = 32 TC behaviors.
    block_size = 2: every 2 consecutive FFN partitions form one block, mapped to one TC.

    P1/P2/P3: each is a list of SEQ_DEGREE independent permutations of range(n_tc=4).
    P1[si] shuffled → TC index for seq group si, block fb: tc_list[P1[si][fb]].

    Round 0 evaluates the greedy mapping (sanity check vs main-process baseline).
    """
    worker_id, rounds_budget, seed = task
    fw = _build_framework()
    bd       = fw["bd"]
    m1o      = fw["m1o"]; ao  = fw["ao"]; m2o = fw["m2o"]
    muo      = fw["muo"]; m3o = fw["m3o"]; ro  = fw["ro"]
    mvu      = fw["mlp3_vu_tags"]
    tc_list  = fw["tc_list"]
    core_to_vu = fw["core_to_vu"]
    block_size = fw["block_size"]
    n_tc     = fw["n_tc"]
    rng = random.Random(seed)
    t0 = time.time()

    for btag, loc in fw["greedy_locs"].items():
        bd[btag].location = loc
    greedy_time = _eval_current_locs(fw)
    best_time = greedy_time
    best_perm = "greedy"

    for _ in range(1, rounds_budget):
        P1 = [list(range(n_tc)) for _ in range(SEQ_DEGREE)]
        P2 = [list(range(n_tc)) for _ in range(SEQ_DEGREE)]
        P3 = [list(range(n_tc)) for _ in range(SEQ_DEGREE)]
        for p in P1: rng.shuffle(p)
        for p in P2: rng.shuffle(p)
        for p in P3: rng.shuffle(p)

        _apply_dist_mapping(
            bd, tc_list, core_to_vu, block_size,
            m1o.beha_tags, m2o.beha_tags, m3o.beha_tags,
            ao.beha_tags, muo.beha_tags, mvu, ro.beha_tags,
            P1, P2, P3)
        t = _eval_current_locs(fw)
        if t[0] < best_time[0]:
            best_time = t
            best_perm = (P1, P2, P3)

    elapsed = time.time() - t0
    print(f"  [worker {worker_id}] greedy={greedy_time[0]}  best={best_time[0]}"
          f"  rounds={rounds_budget}  {elapsed:.1f}s", flush=True)
    return best_time, best_perm, rounds_budget, greedy_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brute-force FFN distributed mapping search via per-seq-group TC permutations")
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

    print(f"\n=== Brute-force FFN Distributed Search Complete ===")
    print(f"Threads           : {args.threads}")
    print(f"Total rounds      : {total_rounds}")
    print(f"Elapsed           : {elapsed:.1f}s")
    print(f"Greedy baseline   : {baseline_time[0]}")
    print(f"Worker greedy chk : {worker_greedy_times}  (should all equal baseline)")
    print(f"Best found        : {best_time[0]}")
    if best_time[0] < baseline_time[0]:
        print(f"Improvement       : {(baseline_time[0] - best_time[0]) / baseline_time[0] * 100:.2f}%")
        print(f"Best perm         : P1={best_perm[0]}  P2={best_perm[1]}  P3={best_perm[2]}")
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
