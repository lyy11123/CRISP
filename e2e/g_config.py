"""F/G template partitioning with the fixed GPT W1 workload."""
import ast
import json
import random
import re
from types import SimpleNamespace


def dimensions(shape):
    rows, columns = map(int, shape.split("x"))
    return rows, columns


def splits(phase, op, shape, split_mode):
    rows, columns = dimensions(shape)
    growth = split_mode == "growth"
    lanes = rows * columns * 2 if growth else 8
    ffn_lanes = rows * columns if growth else 4
    if op == "ln":
        if phase == "prefill":
            return {"ln_split_parallelism_degree_0": (1, lanes * 8, 1),
                    "ln_split_parallelism_degree_1": (1, lanes, 8)}
        return {"ln_split_parallelism_degree_0": (1, 1, 1),
                "ln_split_parallelism_degree_1": (1, 1, 1)}
    if op == "mha":
        return {"mha_split_parallelism_degree": (1, lanes, 1, 1)}
    if op == "proj":
        return {"proj_split_parallelism_degree": (1, lanes, 1, 8) if phase == "prefill"
                else (1, 1, lanes, 8)}
    if op == "ffn":
        return {"mlp1_split_parallelism_degree": (1, ffn_lanes, 1, 16) if phase == "prefill"
                else (1, 1, ffn_lanes, 16)}
    raise ValueError(op)


def transform(source, spec):
    for name, value in splits(spec["phase"], spec["op"], spec["shape"], spec["split_mode"]).items():
        source, count = re.subn(rf"(?m)^{name} = .*?$", f"{name} = {value}", source)
        assert count == 1, (name, count)
    rows, columns = dimensions(spec["shape"])
    parallelism = rows * columns * 16 if spec["split_mode"] == "growth" else 64
    source, count = re.subn(r"(?m)^parallelism_degree = \d+$", f"parallelism_degree = {parallelism}", source)
    assert count == 1
    return source


def joint_layout(shape):
    rows, columns = dimensions(shape)
    p = {r * columns + c: r * 2 * columns + c for r in range(rows) for c in range(columns)}
    d = {r * columns + c: r * 2 * columns + c + columns for r in range(rows) for c in range(columns)}
    return {"shape": f"{rows}x{columns * 2}", "P": p, "D": d,
            "kv_targets": [(die, port) for die in d.values() for port in range(16, 24)]}


def check():
    import f_config
    from fg_manifest import pilot_candidates
    template_checks = []
    for spec in pilot_candidates():
        path = f_config.generate(spec)
        source = path.read_text(encoding="utf-8")
        # E's Gemini adapter changes the native prefill template's seven heads to eight.
        if spec["method"] == "Gemini" and spec["phase"] == "prefill" and spec["op"] == "mha":
            source, count = re.subn(r"(?m)^heads = 7$", "heads = 8", source)
            assert count == 1
        literals = {}
        for statement in ast.parse(source).body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                try:
                    literals[statement.targets[0].id] = ast.literal_eval(statement.value)
                except (ValueError, TypeError):
                    pass
        assert (literals["batch_size"], literals["sequence_length"], literals["hidden_states"], literals["head_num"]) == (16, 2048, 6144, 64)
        expected = splits(spec["phase"], spec["op"], spec["shape"], spec["split_mode"])
        for field, value in expected.items():
            assert literals[field] == value
        if spec["op"] == "mha":
            assert literals["heads"] == 8
        template_checks.append(dict(candidate_id=spec["candidate_id"], split_fields=expected,
                                    batch=16, sequence=2048, hidden=6144, total_heads=64,
                                    micro_heads=8 if spec["op"] == "mha" else None))
    platform_checks = []
    for shape in ("2x2", "2x3", "3x3"):
        hp = f_config.build_platform(dict(hardware_profile="busybarn", shape=shape, split_mode="fixed", phase="decode", op="ln"))
        from Loop_Mapping import AllMapping
        from Pre_Mapping import initialized_mapping
        from Stream_Mapping import StreamOptimization
        from partition import generate_average_degree
        regions = AllMapping({0: []}, {0: 1}, hp)
        rows, columns = dimensions(shape)
        expected_dies = list(range(rows * columns))
        assert sorted(regions[0]) == expected_dies
        used = {}
        for greedy in (False, True):
            random.seed(123)
            behaviours = {(0, i, 0, 0): SimpleNamespace(beha_type="matmul", location=None,
                beha_datashape=[[16, 128], [128, 64]], produced_data_split_dict={},
                producer_tags=set(), consumer_tags=set()) for i in range(rows * columns * 16)}
            data = {(0, 0): SimpleNamespace(splitted_corresponding_dict={})}
            initialized_mapping(behaviours, data, hp, regions, greedy_flag=greedy)
            used[str(greedy)] = sorted({b.location[0] for b in behaviours.values()})
            assert used[str(greedy)] == expected_dies
        state = [0, 0, {k: 0 for k in hp.links_dict}, {k: 0 for k in hp.tensorcore_dict},
                 {k: 0 for k in hp.vectorunit_dict}, {}, {}, {}]
        sa = StreamOptimization(state, regions, hp, LP_flag=True)
        assert len(sa.comp_position_dict[0]["tensorcore"]) == rows * columns * 16
        assert len(sa.free_node) == rows * columns * 16
        partition = {}
        for size, count in ((2048, rows * columns * 2), (6144, rows * columns * 2)):
            spans = generate_average_degree(size, count)
            assert len(spans) == count and min(spans) > 0 and sum(spans) == size
            partition[f"{size}/{count}"] = spans
        layout = joint_layout(shape)
        assert set(layout["P"].values()).isdisjoint(layout["D"].values())
        assert len(layout["kv_targets"]) == rows * columns * 8
        if shape == "2x2":
            assert layout["P"] == {0: 0, 1: 1, 2: 4, 3: 5}
            assert layout["D"] == {0: 2, 1: 3, 2: 6, 3: 7}
        joint_hp = f_config.build_platform(dict(shape=shape, phase="decode", op="ln"), shape_override=layout["shape"])
        assert len(joint_hp.tensorcore_dict) == rows * columns * 32
        assert all((*node, 0) in joint_hp.ddr_dict for node in layout["kv_targets"])
        p_dies, d_dies = set(layout["P"].values()), set(layout["D"].values())
        cross_links = [link for link in joint_hp.links_dict
                       if (link[0][0] in p_dies and link[1][0] in d_dies)
                       or (link[1][0] in p_dies and link[0][0] in d_dies)]
        assert len(cross_links) == rows * 4
        platform_checks.append(dict(shape=shape, initialization_used_dies=used,
            all_mapping_regions=regions, SA_tensor_candidates=len(sa.comp_position_dict[0]["tensorcore"]),
            Gemini_free_nodes=len(sa.free_node), partitions=partition, joint=layout,
            joint_tensor_cores=len(joint_hp.tensorcore_dict), joint_cross_directed_links=len(cross_links)))
    report = dict(template_count=len(template_checks), templates=template_checks, platforms=platform_checks)
    f_config.atomic_text(f_config.HERE / "f/tables/g_structure_checks.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps({"templates": len(template_checks), "platforms": [
        {k: v for k, v in row.items() if k not in ("partitions", "joint", "all_mapping_regions")}
        for row in platform_checks]}, indent=2))


if __name__ == "__main__":
    check()
