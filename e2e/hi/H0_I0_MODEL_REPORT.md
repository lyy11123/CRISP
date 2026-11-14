# H0 / I0 模型与接口核对

2026-11-14。40 个初始映射图、16 组 BusyBarn/Gemini 语义比较、8 个 WATOS 检查全部通过。Gemini 另以 32 个独立进程完成每进程 2 步的接口检查，共 64 个提议，未调用独立后端，也不计正式候选。

[模型合同核对证据](evidence/model_contract_validation.json)、[完整图清单](tables/graph_inventory.csv)、[Llama 合同](models/llama/model_contract.json)、[Qwen 合同](models/qwen3moe/model_contract.json)。这些图是初始化状态，用于核对输入与执行成本；四个 H1 的正式搜索结果另外汇总。

## 1. 实际头数、归一化与 KV

两个模型的生成副本都把实际 `range(head_num)` 改为 `range(heads)`，并固定 `heads=8`。按输出张量和生成行为识别，每图恰有 8 个 Q、1 个 K、1 个 V，8 个注意力输出。Llama 层组装乘数为 8，Qwen 为 4；D 链由合同读取这个数字。

Llama decode FFN 的 mlp1、mlp2 现在都读取 RMSNorm 输出，残差仍读取原输入。两模型 decode FFN 各自已含一次归一化，因此独立 norm 的层乘数为 prefill 2、decode 1。主硬件八类算子的 BusyBarn/Gemini 行为顺序、类型、张量形状和输入输出标签摘要逐项相同。

原生 decode 把历史 K/V 当作 qkt/p 的离线矩阵，每个 Q 头原来各建一份。生成副本现在让 8 个 Q 共用同一对历史矩阵标签，历史 KV 仍沿用 batch 内共享的设定。计算行为、输出区域、每头 bias 和全部算术工作量均未改变；每个 Q 可以实际重复读取这对共享张量，逻辑容量不能当作链路传输量。四组改动核对见 [共享历史 KV 证据](evidence/history_kv_sharing.json)。

| 模型 | prefill 输出 K/V 合计 | decode 新 token K/V | decode 历史 K/V 逻辑量 | 每层头组 |
|---|---:|---:|---:|---:|
| Llama | 8,388,608 B | 4,096 B | 1,048,576 B | 8 |
| Qwen3-MoE | 8,388,608 B | 4,096 B | 524,288 B | 4 |

缓存从 `cache_list` 和生产者标签识别，实际张量名称为 `MHA_ktrans_ofmap` 与 `MHA_vconv_ofmap`，不借用 GPT 的 kconv/vconv 字符串筛选。prefill 形状为 `[batch,head_dim,2048]`、`[batch,2048,head_dim]`；decode 新 token 把 2048 换成 1。历史矩阵没有 batch 维，分别为 `[head_dim,2048]`、`[2048,head_dim]`。

原生末端广播把每块数据发向本组全部 32 个 DDR 端口。以下是本机初始图的实际消息提取；逐消息源 PE、目标端口、份数和链路字节均保存在对应 JSON。它尚不是 P→D 组间联合传输。

| 模型/阶段 | 消息 payload 合计 B | 按 32 个接收端累计 B | 去重有向链路字节 B |
|---|---:|---:|---:|
| [llama/decode](evidence/kv_llama_decode.json) | 4,096 | 131,072 | 294,912 |
| [llama/prefill](evidence/kv_llama_prefill.json) | 8,388,608 | 268,435,456 | 613,416,960 |
| [qwen3moe/decode](evidence/kv_qwen3moe_decode.json) | 4,096 | 131,072 | 290,816 |
| [qwen3moe/prefill](evidence/kv_qwen3moe_prefill.json) | 8,388,608 | 268,435,456 | 620,756,992 |

## 2. W1 广播与 Gemini 接口

Llama、Qwen 的 P/D 原生模板均设置 `broadcast_datatags=cache_list`，没有把 QKᵀ 或 S 加入末端广播名单。实际八张主硬件 MHA 图中，这类无读者最终写回消息均为 0；W1 筛选删除 0 条，保留 K/V 的真实写回。这个结果与 GPT 原模板需要删除 QKᵀ/S 的情况不同，已逐项写入模型合同。

Gemini 的 `LP_flag=True`、`hops_only=True`、XY 路由在 LN、MHA、projection、FFN 的 P/D 上均实际执行了能量求值和两步移动。16 个模型/算子组合中 seed123 与 seed1 的提议轨迹都不同；原生初始化仍固定 seed123，独立性来自搜索种子。此检查只确认接口和种子生效，不替代之后每算子 14 个完整候选。当前源码中的 LP_flag 分支选择空闲 PE，没有调用线性规划求解器。

精确回滚均由适配器恢复前状态并核对能量；原厂 revert 的状态字节是否相同另存原始审计，未用它替代适配器。原/快 CPCM 在 Llama decode MHA、FFN 的完整初始广播图上，Schedule 全部数组、T、关键路径与阻塞归因逐项相同。

## 3. 图规模与内存

表中 A3h 表示其 g0 初始图，未执行 SA。256 KiB 块数是每条通信按该粒度切分后的消息块数；多播各路径的逐跳引用数在完整 CSV 另列，二者不能混为同一个数量。RSS 包括进程初始化；两个带原/快比较的 decode 行还包括首次 Numba 加载。

| 模型 | 阶段/算子 | 行为 | BB/Gemini 事件 | BB 256 KiB 块 | BB 构图 s | BB 峰值 GiB |
|---|---|---:|---:|---:|---:|---:|
| llama | decode/ln | 7 | 18/18 | 11 | 1.725 | 0.218 |
| llama | decode/mha | 76 | 220/221 | 230 | 1.718 | 0.377 |
| llama | decode/proj | 80 | 302/303 | 670 | 1.986 | 0.221 |
| llama | decode/ffn | 247 | 910/913 | 5,847 | 2.608 | 0.387 |
| llama | prefill/ln | 448 | 2481/2482 | 5,877 | 7.054 | 0.243 |
| llama | prefill/mha | 608 | 2525/2519 | 25,676 | 4.399 | 0.240 |
| llama | prefill/proj | 128 | 959/960 | 15,408 | 3.930 | 0.225 |
| llama | prefill/ffn | 328 | 1105/1108 | 87,804 | 3.676 | 0.227 |
| qwen3moe | decode/ln | 7 | 18/18 | 11 | 1.580 | 0.218 |
| qwen3moe | decode/mha | 321 | 1274/1270 | 953 | 2.104 | 0.230 |
| qwen3moe | decode/proj | 80 | 302/303 | 222 | 1.957 | 0.221 |
| qwen3moe | decode/ffn | 1449 | 4795/4794 | 7,186 | 3.819 | 0.271 |
| qwen3moe | prefill/ln | 448 | 2481/2482 | 3,349 | 6.573 | 0.244 |
| qwen3moe | prefill/mha | 367 | 1210/1217 | 31,529 | 2.749 | 0.229 |
| qwen3moe | prefill/proj | 128 | 959/960 | 5,944 | 3.402 | 0.226 |
| qwen3moe | prefill/ffn | 2884 | 9561/9546 | 267,776 | 18.868 | 0.326 |

Qwen 专家数 128、top-k 8、专家维度 768；`expert_idx % 16` 固定分配，保留 Dispatch/Combine 和归约行为，没有增加动态路由器或专家失衡情景。单层图的容量字段继续只是配置记录，80/48 层求和不代表全模型权重同时驻留已被模拟。

## 4. WATOS 与 Split-m2 预筛

WATOS 的 8 个 P/D MHA/FFN 图与主硬件对应图计算语义相同，Dijkstra/DDR/LP 标志保持一致；参数逐字段等于冻结配置。每 Die 708 TFLOPS，TensorCore 为 44,250 FLOP/tick/核，主硬件为 8,192；Tensor grain 128×64 和原取整规则保留。DDR 每端口每方向 250 B/tick，D2D 每有向链路 250 B/tick；向量服务时间逐项相同。

先按张量区域检查 N 与 batch-M 两轴，再调用现有 m2 分片函数核对输出区域总量和有效 FLOP 守恒。下面四图的 Tensor 块都不直接驱动末端广播，均可使用已有变换，无须移除末端广播断言。

| 模型/算子 | Tensor 块 | N 合法 | batch-M 合法 | 覆盖 PE | 每 PE 合法块范围 |
|---|---:|---:|---:|---:|---:|
| llama/proj | 64 | 64 | 64 | 64 | 1–1 |
| llama/ffn | 192 | 192 | 192 | 64 | 3–3 |
| qwen3moe/proj | 64 | 64 | 64 | 64 | 1–1 |
| qwen3moe/ffn | 1536 | 1536 | 512 | 64 | 13–37 |

两模型都优先预留 prefill FFN 作为 MF6 机制点：它有更多合法 Tensor 工作量，projection 留作备选。表格对应初始化映射，正式 MF6 PE 还要按最终健康 decode MHA 的 seed1 忙碌排名固定；随后只看这个 PE 在候选 FFN/projection 上的孤儿工作量和合法轴决定代表点，先决定再运行修复。不会为获得更好结果更换物理故障，也不会把预筛当作 Split 后端结果。详见 [逐块预筛证据](evidence/split_m2_prefilter.json)。

## 5. 复现材料

- `hi_config.py contracts` 生成两个模型合同及源文件差异；核心合同摘要在正式候选身份中保持不变，本报告只追加验证证据。
- `hi/manifest/model_checks.json` 为 40 图清单；`hi_model_check.py child` 保存每图规模、语义、资源与 W1 结果。
- `hi/manifest/gemini_interface_checks.json` 为 16 个 Gemini 检查；每个种子在新进程中执行。
- `python e2e/hi/kv_inventory.py` 重建四图的 KV 消息清单，零 SA、零独立后端。
- `python e2e/hi/build_model_report.py` 从已保存结果重建本报告、40 行图清单和合同验证附件。
