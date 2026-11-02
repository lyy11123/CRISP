
import os
import sys
import json
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))

import re
import numpy as np

sql_list = [512, 2048, 8192]
ch_shapes = [(1,1),(1,4),(2,2)]
co_shapes = [(4,4)]
tensorcore_grain_list = [(128,64)]
failures = [([],[])]

pattern = re.compile(r"SA time cost:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")

folder_names = [
              "intra_mapping_gpt_prefill", "intra_mapping_gpt_decode",
              "intra_mapping_ch", "intra_mapping_opt_decode",
              "intra_mapping_qwen3moe_prefill", "intra_mapping_qwen3moe_decode",
              "intra_mapping_qwen_prefill", "intra_mapping_qwen_decode",
              "intra_mapping_qwen2moe_prefill", "intra_mapping_qwen2moe_decode",
              "intra_mapping_llama_prefill", "intra_mapping_llama_decode",
              ]

intra_model_dict = {}
total_missing = 0
total_checked = 0

for folder_name in folder_names:
    folder_path = os.path.join(file_path, f'../models/{folder_name}/results/')
    folder_missing = 0
    folder_checked = 0

    def find_min_abc(prefixes):
        global total_missing, total_checked, folder_missing, folder_checked
        best = None
        found_any = False
        for pre in prefixes:
            path = os.path.join(folder_path, f"./{pre}.txt")
            folder_checked += 1
            total_checked += 1
            if not os.path.isfile(path):
                folder_missing += 1
                total_missing += 1
                continue
            with open(path, "r") as f:
                matched = False
                for line in f:
                    m = pattern.search(line)
                    if m:
                        a,b,c = map(int, m.groups())
                        matched = True
                if not matched:
                    pass
                    continue
            found_any = True
            if best is None or a < best[0]:
                best = (a,b,c)
        if not found_any:
            pass
        return best or (np.inf, np.inf, np.inf)

    records = []
    for sq in sql_list:
        for ch in ch_shapes:
            ch_tag = f"{ch[0]}x{ch[1]}"
            for co in co_shapes:
                for tg in tensorcore_grain_list:
                    tg_tag = f"{tg[0]}x{tg[1]}"
                    for fail_idx, _ in enumerate(failures):
                        ln   = find_min_abc([f"ln_sq{sq}_greedy{gi}_lossratio{li}_ch{ch_tag}_bw256_co{co[0]}x{co[1]}_bw256_t{tg_tag}_failpattern{fail_idx}"
                                            for gi in (0,1) for li in range(7)])
                        mha  = find_min_abc([f"mha_sq{sq}_greedy{gi}_lossratio{li}_ch{ch_tag}_bw256_co{co[0]}x{co[1]}_bw256_t{tg_tag}_failpattern{fail_idx}"
                                            for gi in (0,1) for li in range(7)])
                        proj = find_min_abc([f"proj_sq{sq}_greedy{gi}_lossratio{li}_ch{ch_tag}_bw256_co{co[0]}x{co[1]}_bw256_t{tg_tag}_failpattern{fail_idx}"
                                            for gi in (0,1) for li in range(7)])
                        ffn  = find_min_abc([f"ffn_sq{sq}_greedy{gi}_lossratio{li}_ch{ch_tag}_bw256_co{co[0]}x{co[1]}_bw256_t{tg_tag}_failpattern{fail_idx}"
                                            for gi in (0,1) for li in range(7)])
                        busy_a = ln[0]*2 + mha[0]*8 + proj[0] + ffn[0]
                        busy_b = ln[1]*2 + mha[1]*8 + proj[1] + ffn[1]
                        busy_c = ln[2]*2 + mha[2]*8 + proj[2]

                        ga = gb = gc = 0
                        for tag,m in [("ln",2),("mha",8),("proj",1),("ffn",1)]:
                            pre = f"{tag}_sq{sq}_gemini_ch{ch_tag}_bw256_co{co[0]}x{co[1]}_bw256_t{tg_tag}_failpattern{fail_idx}"
                            a,b,c = find_min_abc([pre])
                            ga += a*m; gb += b*m; gc += c*m

                        has_inf = any(v == np.inf for v in [busy_a, ga])
                        if has_inf:
                            pass

                        records.append({
                            "sq":   sq,
                            "ch":   ch_tag,
                            "tg":   tg_tag,
                            "fail": fail_idx,
                            "busy": (busy_a, busy_b, busy_c),
                            "gem":  (ga,     gb,     gc),
                        })


    if folder_name == "intra_mapping_gpt_prefill":
        model_name = "gpt_prefill"
    elif folder_name == "intra_mapping_gpt_decode":
        model_name = "gpt_decode"
    elif folder_name == "intra_mapping_ch":
        model_name = "opt_prefill"
    elif folder_name == "intra_mapping_opt_decode":
        model_name = "opt_decode"
    elif folder_name == "intra_mapping_qwen3moe_prefill":
        model_name = "qwen3moe_prefill"
    elif folder_name == "intra_mapping_qwen3moe_decode":
        model_name = "qwen3moe_decode"
    elif folder_name == "intra_mapping_qwen_prefill":
        model_name = "qwen_prefill"
    elif folder_name == "intra_mapping_qwen_decode":
        model_name = "qwen_decode"
    elif folder_name == "intra_mapping_qwen2moe_prefill":
        model_name = "qwen2moe_prefill"
    elif folder_name == "intra_mapping_qwen2moe_decode":
        model_name = "qwen2moe_decode"
    elif folder_name == "intra_mapping_llama_prefill":
        model_name = "llama_prefill"
    elif folder_name == "intra_mapping_llama_decode":
        model_name = "llama_decode"
    else:
        raise ValueError(f"Unknown folder name: {folder_name}")

    intra_model_dict[model_name] = {}
    intra_model_dict[model_name][512] = []
    intra_model_dict[model_name][2048] = []
    intra_model_dict[model_name][8192] = []
    for record in records:
        if record['sq'] == 512:
            intra_model_dict[model_name][512].append([record['gem'][0], record['busy'][0]])
        elif record['sq'] == 2048:
            intra_model_dict[model_name][2048].append([record['gem'][0], record['busy'][0]])
        elif record['sq'] == 8192:
            intra_model_dict[model_name][8192].append([record['gem'][0], record['busy'][0]])



from allocate_layer import proportional_split
from inter_layer import (HARDWARE, MODELS, generate_placement, compute_inter_time)

sequence_list = [512, 2048, 8192]

placements = {}
for hw_name in HARDWARE:
    bb_groups, bb_shapes, bb_sizes = generate_placement(hw_name, method='sa',
                                                         num_restarts=5, steps=500000)
    gem_groups, gem_shapes, gem_sizes = generate_placement(hw_name, method='zigzag')

    placements[hw_name] = {
        'bb_groups': bb_groups, 'bb_shapes': bb_shapes, 'bb_sizes': bb_sizes,
        'gem_groups': gem_groups, 'gem_shapes': gem_shapes, 'gem_sizes': gem_sizes,
    }

speedup_dict = {}
for sequence in sequence_list:
    speedup_dict[sequence] = {}
    for model_name in MODELS:
        model = MODELS[model_name]
        speedup_dict[sequence][model_name] = {}

        for hw_name in HARDWARE:
            p = placements[hw_name]

            bb_layer_per_group = proportional_split(p['bb_sizes'], model['layers'])
            gem_layer_per_group = proportional_split(p['gem_sizes'], model['layers'])

            speedup_dict[sequence][model_name][hw_name] = {}
            for iter_name in ["prefill", "decode"]:
                data_name = f"{model_name}_{iter_name}"
                seq_len = sequence if iter_name == "prefill" else 1

                busybarn_intra = 0
                for group_idx in range(len(bb_layer_per_group)):
                    shape_idx = p['bb_shapes'][group_idx]
                    layer_num = bb_layer_per_group[group_idx]
                    busybarn_intra += intra_model_dict[data_name][sequence][shape_idx][1] * layer_num

                gemini_intra = 0
                for group_idx in range(len(gem_layer_per_group)):
                    shape_idx = p['gem_shapes'][group_idx]
                    layer_num = gem_layer_per_group[group_idx]
                    gemini_intra += intra_model_dict[data_name][sequence][shape_idx][0] * layer_num

                bb_inter = compute_inter_time(
                    p['bb_groups'], hw_name,
                    model['hidden_dim'], seq_len, model['bytes_per_elem'])
                gem_inter = compute_inter_time(
                    p['gem_groups'], hw_name,
                    model['hidden_dim'], seq_len, model['bytes_per_elem'])

                busybarn_time = busybarn_intra + bb_inter
                gemini_time = gemini_intra + gem_inter
                speedup = gemini_time / busybarn_time
                speedup_dict[sequence][model_name][hw_name][iter_name] = speedup

results_dir = os.path.join(file_path, "results")
os.makedirs(results_dir, exist_ok=True)
json_path = os.path.join(results_dir, "endtoend_speedup.json")
json_data = {str(k): v for k, v in speedup_dict.items()}
with open(json_path, "w") as f:
    json.dump(json_data, f, indent=2)
