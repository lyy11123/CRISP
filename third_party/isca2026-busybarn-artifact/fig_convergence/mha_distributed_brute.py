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
from transpose import Transpose
from elerope import Elerope
from matmul import Matmul
from softmax import Softmax
from Pre_Mapping import update_data, initialized_mapping, autoregreesive_dag
from Loop_Mapping import AllMapping
from event_driver import event_driver_v2 as event_driver

BATCH_SIZE         = 1
SEQUENCE_LENGTH    = 512
HIDDEN_STATES      = 3584
NUM_Q_HEADS        = 28
NUM_KV_HEADS       = 4
NUM_Q_PER_KV       = NUM_Q_HEADS // NUM_KV_HEADS
HEAD_DIMS          = HIDDEN_STATES // NUM_Q_HEADS
CHIPLETS_PER_LAYER = 1
LAYER_NUM          = 1
MHA_SPLIT_DEGREE   = (1, 4, 1, 1)
SEQ_DEGREE         = MHA_SPLIT_DEGREE[1]
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


def _apply_mha_mapping(beha_dict, tc_list, core_to_vu,
                       k_conv_opers, krope_opers, ktrans_opers,
                       v_conv_opers, q_conv_opers, qrope_opers,
                       qkt_opers, softmax_tags_per_head, p_opers,
                       P_k, P_v, P_q, P_qkt, P_p):
    """
    Assign compute behaviors using per-seq-group TC permutations.

    Hardware: 1 chiplet, 4 cores (2×2 mesh), 1 TC + 1 VU per core.
    tc_list has 4 entries (one per core).

    Each permutation P_x is a list of length SEQ_DEGREE (=4):
      P_x[si] = TC index for seq group si, operator class x.

    Since MHA_SPLIT_DEGREE = (1, 4, 1, 1), every operator has exactly
    SEQ_DEGREE=4 TC behaviors (one per seq group, with degree 1 in all
    other dims). All KV heads share P_k / P_v; all Q heads share P_q,
    P_qkt, P_p.  VU behaviors colocate on same core as their TC anchor.

    Softmax child VU behaviors: 20 per head (5 child ops × 4 seq groups).
    seq_group extracted as btag[-1] % SEQ_DEGREE (offsets 0..19 cycling).
    """
    for op in k_conv_opers:
        for si, btag in enumerate(op.beha_tags):
            beha_dict[btag].location = tc_list[P_k[si]]

    for op in krope_opers:
        for si, btag in enumerate(op.beha_tags):
            beha_dict[btag].location = core_to_vu[tc_list[P_k[si]][:-1]]

    for op in ktrans_opers:
        for si, btag in enumerate(op.beha_tags):
            beha_dict[btag].location = core_to_vu[tc_list[P_k[si]][:-1]]

    for op in v_conv_opers:
        for si, btag in enumerate(op.beha_tags):
            beha_dict[btag].location = tc_list[P_v[si]]

    for op in q_conv_opers:
        for si, btag in enumerate(op.beha_tags):
            beha_dict[btag].location = tc_list[P_q[si]]

    for op in qrope_opers:
        for si, btag in enumerate(op.beha_tags):
            beha_dict[btag].location = core_to_vu[tc_list[P_q[si]][:-1]]

    for op in qkt_opers:
        for si, btag in enumerate(op.beha_tags):
            beha_dict[btag].location = tc_list[P_qkt[si]]

    for tags in softmax_tags_per_head:
        for btag in tags:
            si = btag[-1] % SEQ_DEGREE
            beha_dict[btag].location = core_to_vu[tc_list[P_qkt[si]][:-1]]

    for op in p_opers:
        for si, btag in enumerate(op.beha_tags):
            beha_dict[btag].location = tc_list[P_p[si]]


def _eval_current_locs(fw):
    """
    Evaluate whatever locations are currently set in beha_dict.
    Full rebuild: _update_data_locs → build_event → add_broadcast →
    event_driver on deepcopy hw.  Returns total_time.
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
    B, S, H, D = BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_STATES, HEAD_DIMS
    d = MHA_SPLIT_DEGREE

    q_para   = [_p(B,d[0]), _p(S,d[1]), _p(H,d[2]), _p(D,d[3])]
    k_para   = [_p(B,d[0]), _p(S,d[1]), _p(H,d[2]), _p(D,d[3])]
    qkt_para = [_p(B,d[0]), _p(S,d[1]), _p(D,d[2]), _p(S,d[3])]
    s_para   = [_p(B,d[0]), _p(S,d[1]), _p(S,d[2])]
    v_para   = [_p(B,d[0]), _p(S,d[1]), _p(H,d[2]), _p(D,d[3])]
    p_para   = [_p(B,d[0]), _p(S,d[1]), _p(S,d[2]), _p(D,d[3])]

    dd = {}; bd = {}
    atag = (0, 0)
    dd[atag] = tensor_notation(
        data_name="ifmap", data_tag=atag,
        data_shape=[B, S, H], data_type="bf16")
    dd[atag].dummy_generated_split()

    ont = (0, 1); offt = (1, 0)
    p_next_oper_tag = (0, 0, 0, 0)
    cache_list = []
    kt_data_tags = []
    v_data_tags = []

    k_conv_opers = []
    krope_opers  = []
    ktrans_opers = []

    for kv_idx in range(NUM_KV_HEADS):
        k_next, ont, offt, _, k_o = Conv1d(
            data_dict=dd, beha_dict=bd, oper_name=f"MHA_k{kv_idx}conv",
            source_data_tags=[atag], oper_split_list=k_para,
            oper_tag=p_next_oper_tag, online_data_tag=ont, offline_data_tag=offt,
            weight_dim=D, beha_tag_offset=0)
        k_conv_opers.append(k_o)
        k_dt = k_o.target_data_tags[-1]

        kr_next, ont, offt, _, kr_o = Elerope(
            data_dict=dd, beha_dict=bd, oper_name=f"MHA_k{kv_idx}rope",
            source_data_tags=[k_dt],
            oper_split_list=[k_para[0], k_para[1], k_para[3]],
            oper_tag=k_next, online_data_tag=ont, offline_data_tag=offt,
            beha_tag_offset=0)
        krope_opers.append(kr_o)
        kr_dt = kr_o.target_data_tags[0]

        kt_next, ont, offt, _, kt_o = Transpose(
            data_dict=dd, beha_dict=bd, oper_name=f"MHA_k{kv_idx}trans",
            source_data_tags=[kr_dt],
            oper_split_list=[k_para[0], k_para[1], k_para[3]],
            oper_tag=kr_next, online_data_tag=ont, offline_data_tag=offt,
            beha_tag_offset=0)
        ktrans_opers.append(kt_o)
        kt_data_tags.append(kt_o.target_data_tags[0])
        cache_list.append(kt_o.target_data_tags[0])
        p_next_oper_tag = kt_next

    v_conv_opers = []

    for kv_idx in range(NUM_KV_HEADS):
        v_next, ont, offt, _, v_o = Conv1d(
            data_dict=dd, beha_dict=bd, oper_name=f"MHA_v{kv_idx}conv",
            source_data_tags=[atag], oper_split_list=v_para,
            oper_tag=p_next_oper_tag, online_data_tag=ont, offline_data_tag=offt,
            weight_dim=D, beha_tag_offset=0)
        v_conv_opers.append(v_o)
        v_data_tags.append(v_o.target_data_tags[-1])
        cache_list.append(v_o.target_data_tags[-1])
        p_next_oper_tag = v_next

    q_conv_opers = []
    qrope_opers  = []
    qkt_opers    = []
    s_opers      = []
    p_opers      = []

    for head_idx in range(NUM_Q_HEADS):
        kv_idx = head_idx // NUM_Q_PER_KV

        q_next, ont, offt, _, q_o = Conv1d(
            data_dict=dd, beha_dict=bd, oper_name=f"MHA_q{head_idx}conv",
            source_data_tags=[atag], oper_split_list=q_para,
            oper_tag=p_next_oper_tag, online_data_tag=ont, offline_data_tag=offt,
            weight_dim=D, beha_tag_offset=0)
        q_conv_opers.append(q_o)
        q_dt = q_o.target_data_tags[-1]

        qr_next, ont, offt, _, qr_o = Elerope(
            data_dict=dd, beha_dict=bd, oper_name=f"MHA_q{head_idx}rope",
            source_data_tags=[q_dt],
            oper_split_list=[q_para[0], q_para[1], q_para[3]],
            oper_tag=q_next, online_data_tag=ont, offline_data_tag=offt,
            beha_tag_offset=0)
        qrope_opers.append(qr_o)
        qr_dt = qr_o.target_data_tags[0]

        qkt_next, ont, offt, _, qkt_o = Matmul(
            data_dict=dd, beha_dict=bd, oper_name=f"MHA_qkt{head_idx}",
            source_data_tags=[qr_dt, kt_data_tags[kv_idx]],
            oper_split_list=qkt_para, oper_tag=qr_next,
            online_data_tag=ont, offline_data_tag=offt,
            head_list=None, beha_tag_offset=0,
            reduction_split_list=s_para)
        qkt_opers.append(qkt_o)
        qkt_dt = qkt_o.target_data_tags[-1]

        s_next, ont, offt, _, s_o = Softmax(
            data_dict=dd, beha_dict=bd, oper_name=f"MHA_s{head_idx}",
            source_data_tags=[qkt_dt], oper_split_list=s_para,
            oper_tag=qkt_next, online_data_tag=ont, offline_data_tag=offt,
            reduce_dims=[2], beha_tag_offset=0)
        s_opers.append(s_o)
        s_dt = s_o.target_data_tags[0]

        p_next_oper_tag, ont, offt, _, p_o = Matmul(
            data_dict=dd, beha_dict=bd, oper_name=f"MHA_p{head_idx}",
            source_data_tags=[s_dt, v_data_tags[kv_idx]],
            oper_split_list=p_para, oper_tag=s_next,
            online_data_tag=ont, offline_data_tag=offt,
            head_list=None, beha_tag_offset=0)
        p_opers.append(p_o)

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

    softmax_tags_per_head = []
    for s_o in s_opers:
        tags = tuple(btag for btag in bd if btag[:-1] == s_o.oper_tag)
        softmax_tags_per_head.append(tags)

    return {
        "bd": bd, "dd": dd, "hw": hw,
        "layers_regions": layers_regions,
        "k_conv_opers": k_conv_opers,
        "krope_opers":  krope_opers,
        "ktrans_opers": ktrans_opers,
        "v_conv_opers": v_conv_opers,
        "q_conv_opers": q_conv_opers,
        "qrope_opers":  qrope_opers,
        "qkt_opers":    qkt_opers,
        "softmax_tags_per_head": softmax_tags_per_head,
        "p_opers":      p_opers,
        "n_tc":         n_tc,
        "tc_list":      tc_list,
        "core_to_vu":   core_to_vu,
        "greedy_locs":  greedy_locs,
        "ddr_locs":     ddr_locs,
        "init_ll":  dict(hw.link_loads_dict),
        "init_lc":  dict(hw.link_loads_count),
        "broadcast_tags": cache_list,
    }


def _worker(task):
    """
    Random search over per-seq-group TC permutations.

    5 independent permutations of range(SEQ_DEGREE=4):
      P_k   – TC assignment for K Conv1d / RoPE / Transpose
      P_v   – TC assignment for V Conv1d
      P_q   – TC assignment for Q Conv1d / RoPE
      P_qkt – TC assignment for QKT Matmul + Softmax VU
      P_p   – TC assignment for P (SV) Matmul

    P_x[si] = TC index for seq group si under operator class x.
    Total search space: (4!)^5 = 7,962,624.

    Round 0 evaluates the greedy mapping (sanity check vs main-process baseline).
    """
    worker_id, rounds_budget, seed = task
    fw = _build_framework()
    bd            = fw["bd"]
    k_conv_opers  = fw["k_conv_opers"]
    krope_opers   = fw["krope_opers"]
    ktrans_opers  = fw["ktrans_opers"]
    v_conv_opers  = fw["v_conv_opers"]
    q_conv_opers  = fw["q_conv_opers"]
    qrope_opers   = fw["qrope_opers"]
    qkt_opers     = fw["qkt_opers"]
    softmax_tags_per_head = fw["softmax_tags_per_head"]
    p_opers       = fw["p_opers"]
    tc_list       = fw["tc_list"]
    core_to_vu    = fw["core_to_vu"]
    n_tc          = fw["n_tc"]
    rng = random.Random(seed)
    t0 = time.time()

    for btag, loc in fw["greedy_locs"].items():
        bd[btag].location = loc
    greedy_time = _eval_current_locs(fw)
    best_time = greedy_time
    best_perm = "greedy"

    for _ in range(1, rounds_budget):
        P_k   = list(range(n_tc)); rng.shuffle(P_k)
        P_v   = list(range(n_tc)); rng.shuffle(P_v)
        P_q   = list(range(n_tc)); rng.shuffle(P_q)
        P_qkt = list(range(n_tc)); rng.shuffle(P_qkt)
        P_p   = list(range(n_tc)); rng.shuffle(P_p)

        _apply_mha_mapping(
            bd, tc_list, core_to_vu,
            k_conv_opers, krope_opers, ktrans_opers,
            v_conv_opers, q_conv_opers, qrope_opers,
            qkt_opers, softmax_tags_per_head, p_opers,
            P_k, P_v, P_q, P_qkt, P_p)
        t = _eval_current_locs(fw)
        if t[0] < best_time[0]:
            best_time = t
            best_perm = (P_k, P_v, P_q, P_qkt, P_p)

    elapsed = time.time() - t0
    print(f"  [worker {worker_id}] greedy={greedy_time[0]}  best={best_time[0]}"
          f"  rounds={rounds_budget}  {elapsed:.1f}s", flush=True)
    return best_time, best_perm, rounds_budget, greedy_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brute-force MHA distributed mapping search via per-seq-group TC permutations")
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

    print(f"\n=== Brute-force MHA Distributed Search Complete ===")
    print(f"Threads           : {args.threads}")
    print(f"Total rounds      : {total_rounds}")
    print(f"Elapsed           : {elapsed:.1f}s")
    print(f"Greedy baseline   : {baseline_time[0]}")
    print(f"Worker greedy chk : {worker_greedy_times}  (should all equal baseline)")
    print(f"Best found        : {best_time[0]}")
    if best_time[0] < baseline_time[0]:
        print(f"Improvement       : {(baseline_time[0] - best_time[0]) / baseline_time[0] * 100:.2f}%")
        print(f"Best perm         : P_k={best_perm[0]}  P_v={best_perm[1]}"
              f"  P_q={best_perm[2]}  P_qkt={best_perm[3]}  P_p={best_perm[4]}")
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
