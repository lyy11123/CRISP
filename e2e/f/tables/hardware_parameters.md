# F/G 硬件参数表

统一采用 1 tick = 1 ns。每 Die 4×4 逻辑核；链路 latency 单位为 ns。

| 配置 | TFLOPS/Die | FLOP/tick/核 | TensorCore缩放 | NoC B/tick,延迟 | D2D B/tick/有向链路,延迟 | DDR B/tick/端口/方向,延迟 | DDR容量字段 |
|---|---:|---:|---:|---|---|---|---:|
| busybarn | 131.072 | 8192.0 | 1.000000000 | 256,1 | 256.0,20 | 256.0,50 | 34359738368 B |
| wsc | 261.12 | 16320.0 | 1.992187500 | 256,1 | 125.0,20 | 250.0,50 | 64000000000 B |
| watos | 708 | 44250.0 | 5.401611328 | 256,1 | 250.0,20 | 250.0,50 | 70000000000 B |
| face | 131.072 | 8192.0 | 1.000000000 | 256,1 | 355.625,20 | 410.0,50 | 128000000000 B |

TensorCore 保留 128×64 grain 与 utilization_factor=2：默认每核 `2×128×64/2=8192 FLOP/tick`，每 Die 131072 FLOP/tick，即本次 1 ns 基准下的 131.072 TFLOPS。适配只把 TensorCore 工作时间公式中的 frequency 乘计算缩放；全局 frequency=1，向量速率保持默认 1024 lanes/tick，按原复杂度表计时。取整、padding 与两倍利用率开销均保留。FACE 的计算与 NoC 完整沿用默认值。

D2D 每边两条物理连接，均含两个方向。论文表值按每 Die 四边双向总预算解释，单条有向链路为 `TB/s×1000/16 B/tick`；这是所采用的分配假设。边缘 Die 只保留真实邻接边，不把其未连接边预算重新分配。

DDR 保留现有独立读写方向，每 Die 每方向八个端口均分标称带宽。故 WSC/WATOS 单方向 2 TB/s、读写同时饱和时合计 4 TB/s；FACE 单方向 3.28、合计 6.56 TB/s。该模型没有共享读写控制器。配对重叠检查分别观察两个方向的工作时间，CPCM 与后端使用相同链路。

SRAM 保留 16 MiB/core。DDR 容量字段分别为默认 32 GiB，以及论文的 64/70/128 GB（十进制）。当前 WAMIS 的 DDR 是 generic device，容量字段没有接入映射准入或后端服务约束；容量不作为本次性能改善来源。mem_enable 控制访存映射动作，不是容量检查。

来源：BusyBarn artifact 的 tensorcore.py/vectorunit.py 与 ch2ch/ddr2co/co2ddr 工作时间公式；WSC-LLM case 3 Table 1；WATOS configuration 3 Table II；FACE case 10 的 8 枚 HBM2E 与 5.69 TB/s D2D。原文路径与页面见 FG_PLAN.md §3.1。
