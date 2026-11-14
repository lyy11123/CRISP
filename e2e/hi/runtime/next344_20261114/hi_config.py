"""H/I model copies and contracts; the artifact templates remain unchanged."""
import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path

import f_config

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "e2e"
VERSION = "hi_model_v1"
GRAPH_ORDER_VERSION = "hi_native_order_v1"
WORKLOAD_VERSION = "W1_8Q_1KV_v1"
MODELS = {
    "llama": dict(batch=8, hidden=8192, q_heads=64, head_dim=128,
                  q_heads_per_graph=8, kv_heads_per_graph=1, head_groups=8,
                  ffn_dim=28672, layers=80),
    "qwen3moe": dict(batch=16, hidden=2048, q_heads=32, head_dim=64,
                     q_heads_per_graph=8, kv_heads_per_graph=1, head_groups=4,
                     ffn_dim=6144, expert_dim=768, experts=128, top_k=8, layers=48),
}
LOSSES = [[1,1,1,1], [1,1,1,.1], [1,1,.1,.1], [1,.1,.1,.1],
          [.1,1,.1,.1], [.1,.1,1,.1], [.1,.1,1,1]]


def normalized(spec):
    s = f_config.normalized(spec)
    s.setdefault("model", "llama")
    assert s["model"] in MODELS
    assert s["shape"] == "2x2" and s["hardware_profile"] in ("busybarn", "watos")
    return s


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def model_contract(model):
    p = MODELS[model]
    return dict(model=model, model_contract_version=VERSION,
                graph_order_version=GRAPH_ORDER_VERSION, workload_version=WORKLOAD_VERSION,
                **p, sequence_length=2048, dtype="bf16", q_loop="range(heads), heads=8",
                decode_chain=dict(norm=1, mha=p["head_groups"], proj=1, ffn=1),
                prefill_chain=dict(norm=2, mha=p["head_groups"], proj=1, ffn=1),
                decode_ffn_includes_rmsnorm=True, prefill_ffn_includes_rmsnorm=False,
                llama_decode_gate_input="both MLPmlp1 and MLPmlp2 read ln_data_tag; residual reads activation_data_tag",
                kv_identity="cache_list tensor tags and generating behaviours, not GPT kconv/vconv names",
                kv_cache_length=2048,
                prefill_kv_output_bytes_per_group=2*p["batch"]*2048*p["head_dim"]*2,
                decode_new_kv_output_bytes_per_group=2*p["batch"]*p["head_dim"]*2,
                decode_history_kv_logical_bytes_per_group=2*2048*p["head_dim"]*2,
                decode_history="one shared Ktranspose/V offline tensor pair for eight Q heads; history is shared across batch; per-head biases and arithmetic retained",
                native_mha_broadcast="cache_list=[kt_data_tag,v_data_tag]; QKT/S are not appended",
                w1_filter="construct native broadcast with its routes, remove qkt/s_softmax added messages; expected zero for native H/I templates",
                expert_assignment="expert_idx % 16; one token dispatch per expert; eight experts per batch item" if model == "qwen3moe" else None,
                capacity_assumption="single-layer/operator simulation; capacity metadata is not an admission check",
                layer_count_use="analytical accumulation only; full model simultaneous residency not simulated")


def identity(spec):
    s = normalized(spec)
    return f"{s['model']}_{f_config.identity(s)}_{digest(model_contract(s['model']))[:10]}"


def model_name(spec):
    s = normalized(spec)
    return f"hi_{identity(s)}_{s['phase']}"


def suffix(spec):
    return f"_{VERSION}_{identity(spec)}"


def source_path(spec):
    s = normalized(spec)
    return f_config.artifact_root() / f"models/intra_mapping_{s['model']}_{s['phase']}/{s['op']}.py"


def template(spec):
    path = source_path(spec)
    variable = "gemini_template" if spec.get("method") == "Gemini" else "busybarn_template"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets))


def task_path(spec, c=None, cfg=None):
    s = normalized(spec)
    method = s.get("method", "A1")
    variant = "gemini" if method == "Gemini" else f"greedy{s.get('g',0)}_lossratio{s.get('x',0) if method == 'A1' else 0}"
    return f_config.artifact_root() / "models" / model_name(s) / "py" / f"{s['op']}_sq2048_{variant}_ch2x2{suffix(s)}.py"


def config_path(spec, shape_override=None):
    s = normalized(spec)
    path = f_config.artifact_root()/"models"/model_name(s)/"cfg"/f"config_ch2x2{suffix(s)}.cfg"
    f_config.atomic_text(path, f_config.config_text(s, shape_override))
    return path


def generated_source(spec):
    s = normalized(spec)
    target = task_path(s)
    original = template(s).format(cfg_name=config_path(s).stem, sq=2048,
                                  greedy_flag=s.get("g",0) == 0,
                                  loss_ratio=LOSSES[s.get("x",0) if s.get("method") == "A1" else 0],
                                  py_name=target.stem)
    source = original
    if s["op"] == "mha":
        assert source.count("for head_idx in range(head_num):") == 1
        source = source.replace("for head_idx in range(head_num):", "for head_idx in range(heads):")
        source = source.replace("heads = 7\n", "heads = 8\n")
        if s["phase"] == "decode":
            source = source.replace("from conv1d import Conv1d", "from conv1d import Conv1d\nfrom hi_config import shared_history_conv1d")
            for name in ("qkt", "p"):
                old = f"{name}_next_oper_tag, next_online_data_tag, next_offline_data_tag, {name}_next_beha_offset, {name}_oper = Conv1d("
                assert source.count(old) == 1
                source = source.replace(old, old.replace("= Conv1d(", "= shared_history_conv1d("))
    if (s["model"], s["phase"], s["op"]) == ("llama", "decode", "ffn"):
        start = source.index("mlp2_next_oper_tag, next_online_data_tag")
        end = source.index("mlp2_data_tag =", start)
        part = source[start:end]
        assert part.count("source_data_tags=[activation_data_tag]") == 1
        source = source[:start] + part.replace("source_data_tags=[activation_data_tag]", "source_data_tags=[ln_data_tag]") + source[end:]
    return original, source


def shared_history_conv1d(**kw):
    """Native decode arithmetic with one shared historical K or V tensor."""
    from conv1d import conv1d_data, conv1d
    data = kw["data_dict"]
    name = kw["oper_name"] + "_weight"
    existing = [tag for tag,tensor in data.items() if tensor.data_name == name]
    sources, targets, next_online, next_offline = conv1d_data(
        data_dict=data, oper_name=kw["oper_name"], oper_split_list=kw["oper_split_list"],
        online_data_tag=kw["online_data_tag"], offline_data_tag=kw["offline_data_tag"])
    if existing:
        assert len(existing) == 1
        original, shared = sources[0], existing[0]
        assert data[original].data_shape == data[shared].data_shape
        assert data[original].data_type == data[shared].data_type
        del data[original]
        sources[0] = shared
    oper = conv1d(oper_name=kw["oper_name"], oper_tag=kw["oper_tag"],
                  source_data_tags=kw["source_data_tags"] + sources,
                  target_data_tags=targets, oper_split_list=kw["oper_split_list"],
                  weight_dim=kw["weight_dim"], beha_tag_offset=kw.get("beha_tag_offset",0),
                  reduction_split_list=kw.get("reduction_split_list"))
    oper.build_dependency(data_dict=data, beha_dict=kw["beha_dict"])
    tag = kw["oper_tag"]
    return tag[:-1]+(tag[-1]+1,), next_online, next_offline, kw.get("beha_tag_offset",0)+oper.degrees, oper


def generate(spec):
    target = task_path(spec)
    _, source = generated_source(spec)
    f_config.atomic_text(target, source)
    (target.parent.parent/"results").mkdir(exist_ok=True)
    return target


def prepare(spec, out):
    s = normalized(spec)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["CPCM_SA_STEPS"] = str(s.get("search_steps",1000))
    import e2e_pilot as pilot
    c = pilot.C
    c.WORK, c.OUT, c.SUF = str(f_config.artifact_root()), str(out), suffix(s)
    cfg = f"hi_{identity(s)}_{s['phase']}_{s['op']}"
    ft = f_config.failures(s)
    c.CONFIGS[cfg] = (model_name(s), s["op"], ft if any(ft.values()) else None, "ch2x2")
    c.REGIME[cfg] = WORKLOAD_VERSION
    c.BACKEND_SEEDS = s.get("backend_seeds",list(range(1,11)))
    c._artifact_path()
    f_config.install_hardware()
    if s["op"] == "mha" and not getattr(pilot,"_hi_w1_installed",False):
        pilot.install_w1()
        pilot._hi_w1_installed = True
    generate(s)
    return pilot, c, cfg


def write_contracts():
    for model in MODELS:
        folder = HERE/"hi"/"models"/model
        contract = model_contract(model)
        source_rows, diffs = [], []
        for phase in ("decode","prefill"):
            for op in ("ln","mha","proj","ffn"):
                for method in ("A3h","Gemini"):
                    spec = dict(model=model, phase=phase, op=op, method=method, g=0, x=123)
                    original, source = generated_source(spec)
                    path = generate(spec)
                    source_rows.append(dict(phase=phase,op=op,method=method,
                                            template=str(source_path(spec).relative_to(ROOT)),
                                            template_sha256=hashlib.sha256(source_path(spec).read_bytes()).hexdigest(),
                                            generated=str(path.relative_to(ROOT)),
                                            generated_sha256=hashlib.sha256(source.encode()).hexdigest()))
                    diffs += [f"## {phase}/{op}/{method}\n", "```diff\n",
                              *difflib.unified_diff(original.splitlines(True), source.splitlines(True), fromfile="native formatted",tofile="H/I generated"), "```\n"]
        contract["generated_sources"] = source_rows
        f_config.atomic_text(folder/"model_contract.json",json.dumps(contract,ensure_ascii=False,indent=2)+"\n")
        f_config.atomic_text(folder/"generated_sources_diff.md", "# 生成副本的模型变更\n\n" + "".join(diffs))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["contracts"])
    parser.parse_args()
    write_contracts()
