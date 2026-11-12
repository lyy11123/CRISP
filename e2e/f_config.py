"""F/G hardware parameters and isolated generated templates; historical sources stay unchanged.

python e2e/f_config.py tables
python e2e/f_config.py check
"""
import argparse
import copy
import configparser
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

import g_config

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "e2e"
BASE_SUFFIX = "_bw256_co4x4_bw256_t128x64_failpattern0"
PROTOCOL = "fg_v1"
TICK_SECONDS = 1e-9
DEFAULT_TFLOPS = 131.072
PROFILES = {
    "busybarn": dict(tflops=DEFAULT_TFLOPS, ddr_tb_s=2.048, ddr_capacity_bytes=34359738368,
                     d2d_tb_s=4.096, source="BusyBarn generated 4x4-core config; 1 ns interpretation"),
    "wsc": dict(tflops=261.12, ddr_tb_s=2, ddr_capacity_bytes=64_000_000_000,
                d2d_tb_s=2, source="WSC-LLM case 3, Table 1 and section 5.1.1"),
    "watos": dict(tflops=708, ddr_tb_s=2, ddr_capacity_bytes=70_000_000_000,
                  d2d_tb_s=4, source="WATOS configuration 3, Table II; per-Die total"),
    "face": dict(tflops=DEFAULT_TFLOPS, ddr_tb_s=3.28, ddr_capacity_bytes=128_000_000_000,
                 d2d_tb_s=5.69, source="FACE case 10, 8 HBM2E per Die; compute and NoC BusyBarn default"),
}


def artifact_root():
    return Path(os.environ.get("CRISP_E_ARTIFACT", HERE / "work/artifact")).resolve()


def normalized(spec):
    return dict(spec, hardware_profile=spec.get("hardware_profile", "busybarn"),
                split_mode=spec.get("split_mode", "fixed"), d2d_scale=float(spec.get("d2d_scale", 1)),
                phase=spec.get("phase", "decode"), op=spec.get("op", "mha"),
                shape=spec.get("shape", "2x2"))


def failures(spec):
    return {k: spec.get(k, []) for k in ("failed_nodes", "failed_links", "failed_pes")}


def identity(spec):
    s = normalized(spec)
    value = {k: s[k] for k in ("hardware_profile", "shape", "split_mode", "d2d_scale")}
    value.update(failures(s), protocol=PROTOCOL)
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]
    return f"{s['hardware_profile']}_{s['shape']}_{s['split_mode']}_{digest}"


def model_name(spec):
    s = normalized(spec)
    return f"fg_{identity(s)}_{s['phase']}"


def suffix(spec):
    return f"_{PROTOCOL}_{identity(spec)}"


def hardware_parameters(profile, d2d_scale=1):
    p = PROFILES[profile]
    return dict(hardware_profile=profile, tick_seconds=TICK_SECONDS,
                tflops_per_die=p["tflops"], flop_per_tick_die=p["tflops"] * 1000,
                flop_per_tick_core=p["tflops"] * 1000 / 16,
                tensor_compute_scale=p["tflops"] / DEFAULT_TFLOPS,
                tensor_grain=[128, 64], tensor_utilization_factor=2,
                vector_lanes_per_tick=1024, vector_scale=1, frequency_field=1,
                noc_bytes_per_tick=256, noc_latency_ticks=1,
                d2d_connections_per_edge=2, d2d_nominal_edges=4, d2d_directions=2,
                d2d_total_tb_s=p["d2d_tb_s"] * d2d_scale,
                d2d_bytes_per_tick_directed_link=p["d2d_tb_s"] * d2d_scale * 1000 / 16,
                d2d_latency_ticks=20, d2d_scale=d2d_scale,
                ddr_ports_per_die=8, ddr_tb_s_per_direction=p["ddr_tb_s"],
                ddr_bytes_per_tick_port_direction=p["ddr_tb_s"] * 1000 / 8,
                ddr_latency_ticks=50, ddr_read_write="independent read and write directions",
                ddr_max_combined_tb_s=2 * p["ddr_tb_s"],
                ddr_capacity_bytes=p["ddr_capacity_bytes"], sram_bytes_per_core=16777216,
                capacity_enforced=False, source=p["source"],
                d2d_interpretation="per-Die nominal four-edge bidirectional total; paper direction unspecified",
                capacity_note="configuration metadata; WAMIS constructs generic DDR devices without capacity admission")


def config_text(spec, shape_override=None):
    s = normalized(spec)
    shape = shape_override or s["shape"]
    rows, columns = g_config.dimensions(shape)
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    baseline = artifact_root() / "models/intra_mapping_gpt_decode/cfg" / f"config_ch2x2{BASE_SUFFIX}.cfg"
    cfg.read(baseline, encoding="utf-8")
    p = hardware_parameters(s["hardware_profile"], s["d2d_scale"])
    for key, value in dict(chiplet_number=rows * columns, chiplet_rows=rows, chiplet_columns=columns).items():
        cfg["topology"][key] = str(value)
    cfg["tensorcore_cfg"]["fg_compute_scale"] = str(p["tensor_compute_scale"])
    cfg["ch2ch_cfg"]["ch2ch_bandwidth"] = str(p["d2d_bytes_per_tick_directed_link"])
    for direction in ("ddr2co", "co2ddr"):
        cfg[direction + "_cfg"][direction + "_bandwidth"] = str(p["ddr_bytes_per_tick_port_direction"])
    cfg["ddr_cfg"]["ddr_capacity"] = str(p["ddr_capacity_bytes"])
    for key, value in failures(s).items():
        cfg["failures"][key] = repr(value)
    cfg["fg_metadata"] = {"protocol": PROTOCOL, "hardware_profile": s["hardware_profile"],
                          "split_mode": s["split_mode"], "tick_seconds": "0.000000001",
                          "ddr_read_write": "independent directions", "d2d_total": "four edges bidirectional"}
    output = io.StringIO()
    cfg.write(output)
    return output.getvalue()


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.part")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def config_path(spec, shape_override=None):
    s = normalized(spec)
    shape = shape_override or s["shape"]
    path = artifact_root() / "models" / model_name(s) / "cfg" / f"config_ch{shape}{suffix(s)}.cfg"
    atomic_text(path, config_text(s, shape_override))
    return path


def task_path(spec, c=None, cfg=None):
    s = normalized(spec)
    method = s.get("method", "A1")
    variant = "gemini" if method == "Gemini" else f"greedy{s.get('g', 0)}_lossratio{s.get('x', 0) if method == 'A1' else 0}"
    return artifact_root() / "models" / model_name(s) / "py" / f"{s['op']}_sq2048_{variant}_ch{s['shape']}{suffix(s)}.py"


def generate(spec):
    s = normalized(spec)
    target = task_path(s)
    method = s.get("method", "A1")
    variant = "gemini" if method == "Gemini" else f"greedy{s.get('g', 0)}_lossratio{s.get('x', 0) if method == 'A1' else 0}"
    source_path = artifact_root() / f"models/intra_mapping_gpt_{s['phase']}/py/{s['op']}_sq2048_{variant}_ch2x2{BASE_SUFFIX}.py"
    source = source_path.read_text(encoding="utf-8")
    source = source.replace(f"config_ch2x2{BASE_SUFFIX}.cfg", config_path(s).name)
    source = g_config.transform(source, s)
    atomic_text(target, source)
    (target.parent.parent / "results").mkdir(exist_ok=True)
    return target


def install_hardware():
    import WAMIS_HD
    import tensorcore
    cls = tensorcore.tensorcore
    if not hasattr(cls, "_fg_original_working_time"):
        cls._fg_original_working_time = cls.working_time

        def working_time(self, source_datashape, beha_type, frequency):
            return cls._fg_original_working_time(self, source_datashape, beha_type,
                                                  frequency * self.tensorcore_cfg.get("fg_compute_scale", 1))
        cls.working_time = working_time
    platform = WAMIS_HD.wamis_hdc
    if not hasattr(platform, "_fg_original_init"):
        platform._fg_original_init = platform.__init__
        platform._fg_original_xy = platform.xy_multicast_path

        def init(self, platform_cfg):
            platform._fg_original_init(self, platform_cfg)
            for node in platform_cfg["failures"].get("failed_pes", []):
                key = tuple(node)
                for devices in (self.tensorcore_dict, self.vectorunit_dict):
                    for device in list(devices):
                        if device[:2] == key:
                            del devices[device]

        def xy_multicast_path(self, source_node_idx, target_nodes_set):
            try:
                return platform._fg_original_xy(self, source_node_idx=source_node_idx, target_nodes_set=target_nodes_set)
            except TypeError as exc:
                if "NoneType" not in str(exc):
                    raise
                return None
        platform.__init__ = init
        platform.xy_multicast_path = xy_multicast_path


def prepare(spec, out):
    s = normalized(spec)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["CPCM_SA_STEPS"] = str(s.get("search_steps", 1000))
    import e2e_pilot as pilot
    c = pilot.C
    c.WORK = str(artifact_root())
    c.OUT = str(out)
    c.SUF = suffix(s)
    cfg = f"fg_{identity(s)}_{s['phase']}_{s['op']}"
    ft = failures(s)
    c.CONFIGS[cfg] = (model_name(s), s["op"], ft if any(ft.values()) else None, "ch" + s["shape"])
    c.REGIME[cfg] = "FG_W1"
    c.BACKEND_SEEDS = s.get("backend_seeds", list(range(1, 11)))
    c._artifact_path()
    install_hardware()
    if s["op"] == "mha" and not getattr(pilot, "_fg_w1_installed", False):
        pilot.install_w1()
        pilot._fg_w1_installed = True
    generate(s)
    return pilot, c, cfg


def build_platform(spec, out=None, faults=None, shape_override=None):
    s = normalized(spec)
    if faults is not None:
        s.update(faults)
    prepare(s, out or HERE / "fg/out/platform")
    import read_cfg
    import WAMIS_HD
    cfg = read_cfg.cfg_to_dict(str(config_path(s, shape_override)))
    cfg["failures"]["failed_nodes"] = [tuple(n) for n in cfg["failures"]["failed_nodes"]]
    cfg["failures"]["failed_links"] = [(tuple(a), tuple(b)) for a, b in cfg["failures"]["failed_links"]]
    return WAMIS_HD.wamis_hdc(cfg)


def tables():
    folder = HERE / "f/tables"
    rows = [hardware_parameters(name) for name in PROFILES]
    atomic_text(folder / "hardware_parameters.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(folder / "hardware_parameters.csv", buf.getvalue())
    text = "# F/G 硬件参数表\n\n统一采用 1 tick = 1 ns。每 Die 4×4 逻辑核；链路 latency 单位为 ns。\n\n"
    text += "| 配置 | TFLOPS/Die | FLOP/tick/核 | TensorCore缩放 | NoC B/tick,延迟 | D2D B/tick/有向链路,延迟 | DDR B/tick/端口/方向,延迟 | DDR容量字段 |\n|---|---:|---:|---:|---|---|---|---:|\n"
    for r in rows:
        text += f"| {r['hardware_profile']} | {r['tflops_per_die']} | {r['flop_per_tick_core']} | {r['tensor_compute_scale']:.9f} | 256,1 | {r['d2d_bytes_per_tick_directed_link']},20 | {r['ddr_bytes_per_tick_port_direction']},50 | {r['ddr_capacity_bytes']} B |\n"
    text += """
TensorCore 保留 128×64 grain 与 utilization_factor=2：默认每核 `2×128×64/2=8192 FLOP/tick`，每 Die 131072 FLOP/tick，即本次 1 ns 基准下的 131.072 TFLOPS。适配只把 TensorCore 工作时间公式中的 frequency 乘计算缩放；全局 frequency=1，向量速率保持默认 1024 lanes/tick，按原复杂度表计时。取整、padding 与两倍利用率开销均保留。FACE 的计算与 NoC 完整沿用默认值。

D2D 每边两条物理连接，均含两个方向。论文表值按每 Die 四边双向总预算解释，单条有向链路为 `TB/s×1000/16 B/tick`；这是所采用的分配假设。边缘 Die 只保留真实邻接边，不把其未连接边预算重新分配。

DDR 保留现有独立读写方向，每 Die 每方向八个端口均分标称带宽。故 WSC/WATOS 单方向 2 TB/s、读写同时饱和时合计 4 TB/s；FACE 单方向 3.28、合计 6.56 TB/s。该模型没有共享读写控制器。配对重叠检查分别观察两个方向的工作时间，CPCM 与后端使用相同链路。

SRAM 保留 16 MiB/core。DDR 容量字段分别为默认 32 GiB，以及论文的 64/70/128 GB（十进制）。当前 WAMIS 的 DDR 是 generic device，容量字段没有接入映射准入或后端服务约束；容量不作为本次性能改善来源。mem_enable 控制访存映射动作，不是容量检查。

来源：BusyBarn artifact 的 tensorcore.py/vectorunit.py 与 ch2ch/ddr2co/co2ddr 工作时间公式；WSC-LLM case 3 Table 1；WATOS configuration 3 Table II；FACE case 10 的 8 枚 HBM2E 与 5.69 TB/s D2D。原文路径与页面见 FG_PLAN.md §3.1。
"""
    atomic_text(folder / "hardware_parameters.md", text)
    return rows


def check():
    checks = []
    for profile in PROFILES:
        spec = dict(hardware_profile=profile, shape="2x2", split_mode="fixed", phase="decode", op="ln", method="A1", g=0, x=0)
        hp = build_platform(spec)
        p = hardware_parameters(profile)
        tc = next(iter(hp.tensorcore_dict.values()))
        vector = next(iter(hp.vectorunit_dict.values()))
        shapes = [[16, 128], [128, 64]]
        gemm = int(tc.working_time(shapes, "matmul", 1))
        expected = math.ceil(32 / p["tensor_compute_scale"])
        assert gemm == expected, (profile, gemm, expected)
        assert int(vector.working_time([[4096]], "eleadd", 1)) == 4
        d2d = [link for link in hp.links_dict.values() if link.link_type == "ch2ch"]
        reads = [link for link in hp.links_dict.values() if link.link_type == "ddr2co"]
        writes = [link for link in hp.links_dict.values() if link.link_type == "co2ddr"]
        assert len(d2d) == 16 and len(reads) == len(writes) == 32
        assert d2d[0].working_time(8000) == math.ceil(8000 / p["d2d_bytes_per_tick_directed_link"]) + 20
        assert reads[0].working_time(8000) == writes[0].working_time(8000) == math.ceil(8000 / p["ddr_bytes_per_tick_port_direction"]) + 50
        import cpcm
        from event_notation import computation_notation, communication_notation
        from event_driver import event_driver
        tensor_location = next(iter(hp.tensorcore_dict))
        vector_location = next(iter(hp.vectorunit_dict))
        tensor_event = computation_notation("gemm", (0,), "tensorcore", tensor_location, "matmul", shapes)
        vector_event = computation_notation("vector", (0,), "vectorunit", vector_location, "eleadd", [[4096]])
        link = next(k for k, v in hp.links_dict.items() if v.link_type == "ch2ch")
        transfer = communication_notation("d2d", (0,), link[0], link[1], 8000)
        transfer.path_list = [[link]]
        read_link = next(k for k, v in hp.links_dict.items() if v.link_type == "ddr2co")
        write_link = (read_link[1], read_link[0])
        read_event = communication_notation("read", (0,), read_link[0], read_link[1], 8000)
        write_event = communication_notation("write", (1,), write_link[0], write_link[1], 8000)
        read_event.path_list, write_event.path_list = [[read_link]], [[write_link]]
        cases = [("gemm", {tensor_event.event_tag: tensor_event}, gemm),
                 ("vector", {vector_event.event_tag: vector_event}, 4),
                 ("d2d", {transfer.event_tag: transfer}, hp.links_dict[link].working_time(8000)),
                 ("read_write_overlap", {read_event.event_tag: read_event, write_event.event_tag: write_event},
                  max(hp.links_dict[read_link].working_time(8000), hp.links_dict[write_link].working_time(8000)))]
        budgets = {}
        for label, events, expected_time in cases:
            estimate = int(cpcm.simulate(cpcm.compile_bb(events, hp), "bb").T)
            backend = int(event_driver(copy.deepcopy(events), copy.deepcopy(hp))[0])
            assert estimate == backend == expected_time, (profile, label, estimate, backend, expected_time)
            budgets[label] = dict(CPCM=estimate, backend=backend, formula=int(expected_time))
        checks.append(dict(profile=profile, gemm_ticks=gemm, vector_ticks=4,
                           d2d_links=len(d2d), ddr_ports=len(reads),
                           read_ticks=reads[0].working_time(8000), write_ticks=writes[0].working_time(8000),
                           read_write_overlap_ticks=max(reads[0].working_time(8000), writes[0].working_time(8000)),
                           CPCM_backend_budgets=budgets))
    atomic_text(HERE / "f/tables/hardware_checks.json", json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("tables", "check"))
    args = parser.parse_args()
    if args.command == "tables":
        tables()
    else:
        check()
