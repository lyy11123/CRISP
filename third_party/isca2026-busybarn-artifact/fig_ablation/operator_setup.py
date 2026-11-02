"""
Shared operator builders for bottleneck_with_brute experiments.

Provides build functions for FFN, LN, MHA, and Proj operators (Qwen2.5-7B config).
Each builder populates data_dict/beha_dict and returns a dict with
"broadcast_datatags" and operator objects (for brute-force mapping).

Usage (SA experiment):
    data_dict, beha_dict, broadcast_tags = build_operator("ffn")
    layers_regions = finalize_graph(beha_dict, data_dict, hw)

Usage (brute experiment, needs operator objects):
    data_dict = init_data_dict(); beha_dict = {}
    result = build_ffn(data_dict, beha_dict)
    broadcast_tags = result["broadcast_datatags"]
    mlp1_op = result["mlp1_op"]  # etc.
"""
import os
import sys


_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def setup_paths():
    """Add all required directories to sys.path (idempotent)."""
    for _p in [
        _FILE_PATH,
        os.path.join(_FILE_PATH, '../utils/'),
        os.path.join(_FILE_PATH, '../src/partition/oper/'),
        os.path.join(_FILE_PATH, '../src/partition/func/'),
        os.path.join(_FILE_PATH, '../src/partition/'),
        os.path.join(_FILE_PATH, '../src/mapping/'),
        os.path.join(_FILE_PATH, '../src/scheduling/'),
        os.path.join(_FILE_PATH, '../src/backend/analytical/'),
        os.path.join(_FILE_PATH, '../src/scheduling/communication/topology/'),
        os.path.join(_FILE_PATH, '../tool/'),
    ]:
        if _p not in sys.path:
            sys.path.append(_p)

setup_paths()

from read_cfg import cfg_to_dict
from WAMIS_HD import wamis_hdc
from WAMIS_HD_single import wamis_hdc_single
from WAMIS_HD_around import wamis_hd_around
from partition import generate_average_degree
from add_communication import add_beha_producers
from data_notation import tensor_notation
from conv1d import Conv1d
from elesilu import Elesilu
from matadd import Matadd
from rmsnorm import Rmsnorm
from matmul import Matmul
from softmax import Softmax
from transpose import Transpose
from elerope import Elerope
from Pre_Mapping import update_data, initialized_mapping, autoregreesive_dag
from Loop_Mapping import AllMapping


BATCH_SIZE = 1
SEQUENCE_LENGTH = 512


MODEL_CONFIGS = {
    "qwen2.5-7b": dict(
        HIDDEN_STATES=3584, NUM_Q_HEADS=28, NUM_KV_HEADS=4,
        FFN_DIMS=18944,
    ),
    "qwen2.5-32b": dict(
        HIDDEN_STATES=5120, NUM_Q_HEADS=40, NUM_KV_HEADS=8,
        FFN_DIMS=27648,
    ),
}

def set_model(name):
    """Set model constants from MODEL_CONFIGS."""
    global HIDDEN_STATES, NUM_Q_HEADS, NUM_KV_HEADS
    global NUM_Q_PER_KV, HEAD_DIMS, KV_DIMS, FFN_DIMS
    cfg = MODEL_CONFIGS[name]
    HIDDEN_STATES = cfg["HIDDEN_STATES"]
    NUM_Q_HEADS = cfg["NUM_Q_HEADS"]
    NUM_KV_HEADS = cfg["NUM_KV_HEADS"]
    NUM_Q_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
    HEAD_DIMS = HIDDEN_STATES // NUM_Q_HEADS
    KV_DIMS = NUM_KV_HEADS * HEAD_DIMS
    FFN_DIMS = cfg["FFN_DIMS"]

set_model("qwen2.5-7b")


LAYER_NUM = 1
CHIPLETS_PER_LAYER = 4

HW_CFG_PATH = os.path.join(
    _FILE_PATH, "../src/platform/cfgs/wamis_hd_distributed.cfg")
RESULTS_DIR = os.path.join(_FILE_PATH, "results")

TOPOLOGY_CLASSES = {
    "wamis_hdc": wamis_hdc,
    "wamis_hdc_single": wamis_hdc_single,
    "wamis_hd_around": wamis_hd_around,
}

DEFAULT_SPLIT_DEGREES = {
    "ffn":  (1, 4, 1, 8),
    "ln":   (1, 4, 1, 1, 2, 2),
    "mha":  (1, 4, 1, 1),
    "proj": (1, 4, 1, 4),
}


def init_hardware(cfg_path=None, topology="wamis_hdc"):
    """Create and return the hardware platform.

    Args:
        cfg_path: Path to hardware config file (default: wamis_hd_distributed.cfg).
        topology: Topology class name (default: wamis_hdc).
    """
    if cfg_path is None:
        cfg_path = HW_CFG_PATH
    topo_cls = TOPOLOGY_CLASSES[topology]
    hw = topo_cls(cfg_to_dict(cfg_path))
    for tc in hw.modules_dict.get("tensorcore", {}).values():
        tc.utilization_factor = 1
    return hw


def init_data_dict():
    """Create data_dict with the activation tensor at (0,0). Returns data_dict."""
    data_dict = {}
    atag = (0, 0)
    data_dict[atag] = tensor_notation(
        data_name="ifmap", data_tag=atag,
        data_shape=[BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_STATES],
        data_type="bf16")
    data_dict[atag].dummy_generated_split()
    return data_dict


def finalize_graph(beha_dict, data_dict, hardware_platform, greedy_flag=False,
                   chiplets_per_layer=None):
    """Run add_beha_producers, update_data, DAG construction, and initial mapping.

    Returns layers_regions.
    """
    if chiplets_per_layer is None:
        chiplets_per_layer = CHIPLETS_PER_LAYER
    add_beha_producers(beha_dict=beha_dict)
    update_data(beha_dict=beha_dict, data_dict=data_dict)
    model_dag, model_chiplets = autoregreesive_dag(
        llm_layer_num=LAYER_NUM, chiplets_per_layer=chiplets_per_layer)
    layers_regions = AllMapping(model_dag, model_chiplets, hardware_platform)
    initialized_mapping(
        beha_dict=beha_dict, data_dict=data_dict,
        hardware_platform=hardware_platform,
        layers_regions=layers_regions, greedy_flag=greedy_flag)
    return layers_regions


_ACTIVATION_TAG = (0, 0)
_INIT_LAYER = 0


def build_ffn(data_dict, beha_dict, split_degree=None):
    """Build FFN operator graph (mlp1 → act → mlp2 → mul → mlp3 → residual).

    Split: mlp1/mlp2 (1,4,1,8), mlp3 (1,4,8,1) with allreduce.
    Returns dict with broadcast_datatags and operator objects.
    """
    B, S, H, F = BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_STATES, FFN_DIMS
    if split_degree is None:
        split_degree = DEFAULT_SPLIT_DEGREES["ffn"]
    d = split_degree

    mlp1_par = [generate_average_degree(B, d[0]), generate_average_degree(S, d[1]),
                generate_average_degree(H, d[2]), generate_average_degree(F, d[3])]
    act_par = [mlp1_par[0], mlp1_par[1], mlp1_par[3]]
    mlp2_par = list(mlp1_par)
    matadd_par = [mlp1_par[0], mlp1_par[1], mlp1_par[3]]
    mlp3_par = [mlp1_par[0], mlp1_par[1], mlp1_par[3], mlp1_par[2]]

    ont = (0, 1); offt = (1, 0)

    ot1, ont, offt, _, mlp1_op = Conv1d(
        data_dict=data_dict, beha_dict=beha_dict, oper_name="MLPmlp1",
        source_data_tags=[_ACTIVATION_TAG], oper_split_list=mlp1_par,
        oper_tag=(0, 0, _INIT_LAYER, 0), online_data_tag=ont,
        offline_data_tag=offt, weight_dim=F, beha_tag_offset=0)
    mlp1_dt = mlp1_op.target_data_tags[-1]

    ot2, ont, offt, _, act_op = Elesilu(
        data_dict=data_dict, beha_dict=beha_dict, oper_name="MLP_act",
        source_data_tags=[mlp1_dt], oper_split_list=act_par, oper_tag=ot1,
        online_data_tag=ont, offline_data_tag=offt, beha_tag_offset=0)
    act_dt = act_op.target_data_tags[0]

    ot3, ont, offt, _, mlp2_op = Conv1d(
        data_dict=data_dict, beha_dict=beha_dict, oper_name="MLPmlp2",
        source_data_tags=[_ACTIVATION_TAG], oper_split_list=mlp2_par,
        oper_tag=ot2, online_data_tag=ont, offline_data_tag=offt,
        weight_dim=F, beha_tag_offset=0)
    mlp2_dt = mlp2_op.target_data_tags[-1]

    ot4, ont, offt, _, mul_op = Matadd(
        data_dict=data_dict, beha_dict=beha_dict, oper_name="MLP_mul",
        source_data_tags=[act_dt, mlp2_dt], oper_split_list=matadd_par,
        oper_tag=ot3, online_data_tag=ont, offline_data_tag=offt,
        beha_tag_offset=0)
    mul_dt = mul_op.target_data_tags[0]

    ot5, ont, offt, _, mlp3_op = Conv1d(
        data_dict=data_dict, beha_dict=beha_dict, oper_name="MLPmlp3",
        source_data_tags=[mul_dt], oper_split_list=mlp3_par,
        oper_tag=ot4, online_data_tag=ont, offline_data_tag=offt,
        weight_dim=H, beha_tag_offset=0,
        reduction_split_list=[mlp3_par[0], mlp3_par[1], mlp3_par[3]])
    mlp3_dt = mlp3_op.target_data_tags[-1]

    _, ont, offt, _, res_op = Matadd(
        data_dict=data_dict, beha_dict=beha_dict, oper_name="MLP_residual",
        source_data_tags=[_ACTIVATION_TAG, mlp3_dt],
        oper_split_list=[mlp3_par[0], mlp3_par[1], mlp3_par[3]],
        oper_tag=ot5, online_data_tag=ont, offline_data_tag=offt,
        beha_tag_offset=0)

    return {
        "broadcast_datatags": [res_op.target_data_tags[0]],
        "mlp1_op": mlp1_op, "act_op": act_op, "mlp2_op": mlp2_op,
        "mul_op": mul_op, "mlp3_op": mlp3_op, "res_op": res_op,
    }


def build_ln(data_dict, beha_dict, split_degree=None):
    """Build LN (RMSNorm) operator graph.

    Split phase 0: (1,4,1), phase 1: (1,2,2).
    Returns dict with broadcast_datatags and operator objects.
    """
    B, S, H = BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_STATES
    if split_degree is None:
        split_degree = DEFAULT_SPLIT_DEGREES["ln"]
    d0 = split_degree[:3]
    d1 = split_degree[3:]

    ln_par_0 = [generate_average_degree(B, d0[0]), generate_average_degree(S, d0[1]),
                generate_average_degree(H, d0[2])]
    ln_par_1 = [generate_average_degree(B, d1[0]), generate_average_degree(S, d1[1]),
                generate_average_degree(H, d1[2])]

    ont = (0, 1); offt = (1, 0)

    _, ont, offt, _, rms_op = Rmsnorm(
        data_dict=data_dict, beha_dict=beha_dict, oper_name="LN",
        source_data_tags=[_ACTIVATION_TAG],
        oper_split_list=ln_par_0 + ln_par_1,
        oper_tag=(0, 0, _INIT_LAYER, 0),
        online_data_tag=ont, offline_data_tag=offt)

    return {
        "broadcast_datatags": [rms_op.target_data_tags[-1]],
        "rms_op": rms_op,
    }


def build_mha(data_dict, beha_dict, split_degree=None):
    """Build MHA operator graph (GQA: 28 Q heads, 4 KV heads).

    Split: (1,4,1,1) — sequence-parallel, no head splitting.
    Returns dict with broadcast_datatags and operator objects.
    """
    B, S, H = BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_STATES
    if split_degree is None:
        split_degree = DEFAULT_SPLIT_DEGREES["mha"]
    d = split_degree

    q_par = [generate_average_degree(B, d[0]), generate_average_degree(S, d[1]),
             generate_average_degree(H, d[2]), generate_average_degree(HEAD_DIMS, d[3])]
    k_par = list(q_par)
    qkt_par = [generate_average_degree(B, d[0]), generate_average_degree(S, d[1]),
               generate_average_degree(HEAD_DIMS, d[2]), generate_average_degree(S, d[3])]
    s_par = [generate_average_degree(B, d[0]), generate_average_degree(S, d[1]),
             generate_average_degree(S, d[2])]
    v_par = list(q_par)
    p_par = [generate_average_degree(B, d[0]), generate_average_degree(S, d[1]),
             generate_average_degree(S, d[2]), generate_average_degree(HEAD_DIMS, d[3])]

    ont = (0, 1); offt = (1, 0)
    p_next_ot = (0, 0, _INIT_LAYER, 0)
    cache_list = []
    kt_data_tags = []
    v_data_tags = []

    k_conv_opers = []
    krope_opers = []
    ktrans_opers = []

    for kv_idx in range(NUM_KV_HEADS):
        k_ot, ont, offt, _, k_op = Conv1d(
            data_dict=data_dict, beha_dict=beha_dict,
            oper_name=f"MHA_k{kv_idx}conv",
            source_data_tags=[_ACTIVATION_TAG], oper_split_list=k_par,
            oper_tag=p_next_ot, online_data_tag=ont, offline_data_tag=offt,
            weight_dim=HEAD_DIMS, beha_tag_offset=0)
        k_conv_opers.append(k_op)
        k_dt = k_op.target_data_tags[-1]

        kr_ot, ont, offt, _, kr_op = Elerope(
            data_dict=data_dict, beha_dict=beha_dict,
            oper_name=f"MHA_k{kv_idx}rope",
            source_data_tags=[k_dt],
            oper_split_list=[k_par[0], k_par[1], k_par[3]],
            oper_tag=k_ot, online_data_tag=ont, offline_data_tag=offt,
            beha_tag_offset=0)
        krope_opers.append(kr_op)
        kr_dt = kr_op.target_data_tags[0]

        kt_ot, ont, offt, _, kt_op = Transpose(
            data_dict=data_dict, beha_dict=beha_dict,
            oper_name=f"MHA_k{kv_idx}trans",
            source_data_tags=[kr_dt],
            oper_split_list=[k_par[0], k_par[1], k_par[3]],
            oper_tag=kr_ot, online_data_tag=ont, offline_data_tag=offt,
            beha_tag_offset=0)
        ktrans_opers.append(kt_op)
        kt_data_tags.append(kt_op.target_data_tags[0])
        cache_list.append(kt_op.target_data_tags[0])
        p_next_ot = kt_ot

    v_conv_opers = []

    for kv_idx in range(NUM_KV_HEADS):
        v_ot, ont, offt, _, v_op = Conv1d(
            data_dict=data_dict, beha_dict=beha_dict,
            oper_name=f"MHA_v{kv_idx}conv",
            source_data_tags=[_ACTIVATION_TAG], oper_split_list=v_par,
            oper_tag=p_next_ot, online_data_tag=ont, offline_data_tag=offt,
            weight_dim=HEAD_DIMS, beha_tag_offset=0)
        v_conv_opers.append(v_op)
        v_data_tags.append(v_op.target_data_tags[-1])
        cache_list.append(v_op.target_data_tags[-1])
        p_next_ot = v_ot

    q_conv_opers = []
    qrope_opers = []
    qkt_opers = []
    s_opers = []
    p_opers = []

    for head_idx in range(NUM_Q_HEADS):
        kv_idx = head_idx // NUM_Q_PER_KV

        q_ot, ont, offt, _, q_op = Conv1d(
            data_dict=data_dict, beha_dict=beha_dict,
            oper_name=f"MHA_q{head_idx}conv",
            source_data_tags=[_ACTIVATION_TAG], oper_split_list=q_par,
            oper_tag=p_next_ot, online_data_tag=ont, offline_data_tag=offt,
            weight_dim=HEAD_DIMS, beha_tag_offset=0)
        q_conv_opers.append(q_op)
        q_dt = q_op.target_data_tags[-1]

        qr_ot, ont, offt, _, qr_op = Elerope(
            data_dict=data_dict, beha_dict=beha_dict,
            oper_name=f"MHA_q{head_idx}rope",
            source_data_tags=[q_dt],
            oper_split_list=[q_par[0], q_par[1], q_par[3]],
            oper_tag=q_ot, online_data_tag=ont, offline_data_tag=offt,
            beha_tag_offset=0)
        qrope_opers.append(qr_op)
        qr_dt = qr_op.target_data_tags[0]

        qkt_ot, ont, offt, _, qkt_op = Matmul(
            data_dict=data_dict, beha_dict=beha_dict,
            oper_name=f"MHA_qkt{head_idx}",
            source_data_tags=[qr_dt, kt_data_tags[kv_idx]],
            oper_split_list=qkt_par, oper_tag=qr_ot,
            online_data_tag=ont, offline_data_tag=offt,
            head_list=None, beha_tag_offset=0,
            reduction_split_list=s_par)
        qkt_opers.append(qkt_op)
        qkt_dt = qkt_op.target_data_tags[-1]

        s_ot, ont, offt, _, s_op = Softmax(
            data_dict=data_dict, beha_dict=beha_dict,
            oper_name=f"MHA_s{head_idx}",
            source_data_tags=[qkt_dt], oper_split_list=s_par,
            oper_tag=qkt_ot, online_data_tag=ont, offline_data_tag=offt,
            reduce_dims=[2], beha_tag_offset=0)
        s_opers.append(s_op)
        s_dt = s_op.target_data_tags[0]

        p_next_ot, ont, offt, _, p_op = Matmul(
            data_dict=data_dict, beha_dict=beha_dict,
            oper_name=f"MHA_p{head_idx}",
            source_data_tags=[s_dt, v_data_tags[kv_idx]],
            oper_split_list=p_par, oper_tag=s_ot,
            online_data_tag=ont, offline_data_tag=offt,
            head_list=None, beha_tag_offset=0)
        p_opers.append(p_op)

    return {
        "broadcast_datatags": cache_list,
        "k_conv_opers": k_conv_opers, "krope_opers": krope_opers,
        "ktrans_opers": ktrans_opers, "v_conv_opers": v_conv_opers,
        "q_conv_opers": q_conv_opers, "qrope_opers": qrope_opers,
        "qkt_opers": qkt_opers, "s_opers": s_opers, "p_opers": p_opers,
    }


def build_proj(data_dict, beha_dict, split_degree=None):
    """Build Proj operator graph (projection + residual add).

    Split: (1,4,1,4).
    Returns dict with broadcast_datatags and operator objects.
    """
    B, S, H = BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_STATES
    if split_degree is None:
        split_degree = DEFAULT_SPLIT_DEGREES["proj"]
    d = split_degree

    proj_par = [generate_average_degree(B, d[0]), generate_average_degree(S, d[1]),
                generate_average_degree(H, d[2]), generate_average_degree(H, d[3])]

    ont = (0, 1); offt = (1, 0)

    ot1, ont, offt, _, proj_op = Conv1d(
        data_dict=data_dict, beha_dict=beha_dict, oper_name="MHA_proj",
        source_data_tags=[_ACTIVATION_TAG], oper_split_list=proj_par,
        oper_tag=(0, 0, _INIT_LAYER, 0), online_data_tag=ont,
        offline_data_tag=offt, weight_dim=H, beha_tag_offset=0)
    proj_dt = proj_op.target_data_tags[-1]

    _, ont, offt, _, res_op = Matadd(
        data_dict=data_dict, beha_dict=beha_dict, oper_name="MHA_residual",
        source_data_tags=[_ACTIVATION_TAG, proj_dt],
        oper_split_list=[proj_par[0], proj_par[1], proj_par[3]],
        oper_tag=ot1, online_data_tag=ont, offline_data_tag=offt,
        beha_tag_offset=0)

    return {
        "broadcast_datatags": [res_op.target_data_tags[0]],
        "proj_op": proj_op, "res_op": res_op,
    }


OPERATOR_BUILDERS = {
    "ffn": build_ffn,
    "ln": build_ln,
    "mha": build_mha,
    "proj": build_proj,
}


def build_operator(operator_name, data_dict=None, beha_dict=None, split_degree=None):
    """Build operator graph by name. Returns (data_dict, beha_dict, broadcast_datatags)."""
    if data_dict is None:
        data_dict = init_data_dict()
    if beha_dict is None:
        beha_dict = {}
    builder = OPERATOR_BUILDERS[operator_name]
    result = builder(data_dict, beha_dict, split_degree=split_degree)
    return data_dict, beha_dict, result["broadcast_datatags"]
