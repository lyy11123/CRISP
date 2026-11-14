# I0：Qwen decode FFN 旧 28 候选核对
28 份快照、1000 步搜索记录、十种子后端和十次随机平局 CPCM 记录齐全。本批复用 0 个，28 个按新入口重新搜索：A3h g0/seed123 计入 I1 主硬件 decode 试点，其余 27 个计入 I2D。旧结果原样保留。
原因是旧记录只保存了最终 `event_dict` 插入顺序，没有保存后端实际读取的 `dependency_set`、`issue_set` 和多播目标集合遍历序列。重新加载和复制会改变部分集合的遍历次序；当前机器上重构相等也不能补出当时未保存的序列。因此，按本轮要求的逐字段完整身份核对，历史后端的该项记为 `unknown_not_certifiable`。这不是把未知项说成模型或 RMSNorm 有错。
## 已核对的内容
| 项目 | 通过数 / 28 |
|---|---:|
| 生成程序 AST（仅替换配置及输出文件名） | 28 |
| 实际硬件配置值 | 28 |
| batch=16、hidden=2048、bf16 | 28 |
| 一次内嵌 RMSNorm（7 个行为） | 28 |
| RMSNorm → Dispatch → 专家双支路 | 28 |
| 128 专家固定 mod16 路由及每样本 8 专家 | 28 |
| 算子输入/输出区域与消费者 | 28 |
| 完整 1000 步 | 28 |
| 后端种子 1–10 | 28 |
| 后端三元组 | 28 |
| 10 次随机平局 CPCM | 28 |
| B0 确定性估值与搜索记录 | 28 |
| Dijkstra 与 100/1/100 权重 | 28 |
| 最终广播种子 0 | 28 |
| 保存的最终 event_dict 顺序 | 28 |
| 保存的最终源与路由 | 28 |
| 当前旧/新加载流程送入后端前的实际遍历 | 28 |

Qwen decode FFN 的原模板已经是 RMSNorm → Dispatch → 两条门控支路；残差读取原输入。新生成副本保留这一结构。128 个专家采用 `expert_idx % 16`，每个 batch 项对应 8 个专家。MHA 的头数改动不修改这张 FFN 图。

新旧评估均是同进程内每个种子重新复制事件及平台并重设随机数，十种子分别执行；随机平局 CPCM 使用 `random.Random(1..10)`。不是十个独立子进程。这里没有新增后端仿真，也没有运行搜索。旧耗时仅随审计记录存档，不用于当前服务器的 J 计时或新批预计用时。

## 逐候选记录
| 候选 | 可重构字段 | 一次复制改变的依赖/后继集合 | 后续批次 |
|---|---|---:|---|
| moe_A1_g0_l0 | 一致 | 2/1 | I2D |
| moe_A1_g0_l1 | 一致 | 2/1 | I2D |
| moe_A1_g0_l2 | 一致 | 2/1 | I2D |
| moe_A1_g0_l3 | 一致 | 3/1 | I2D |
| moe_A1_g0_l4 | 一致 | 3/1 | I2D |
| moe_A1_g0_l5 | 一致 | 2/1 | I2D |
| moe_A1_g0_l6 | 一致 | 3/1 | I2D |
| moe_A1_g1_l0 | 一致 | 4/1 | I2D |
| moe_A1_g1_l1 | 一致 | 6/1 | I2D |
| moe_A1_g1_l2 | 一致 | 3/1 | I2D |
| moe_A1_g1_l3 | 一致 | 3/1 | I2D |
| moe_A1_g1_l4 | 一致 | 2/1 | I2D |
| moe_A1_g1_l5 | 一致 | 4/1 | I2D |
| moe_A1_g1_l6 | 一致 | 5/1 | I2D |
| moe_A3h_g0_s123 | 一致 | 5/2 | I1_main_decode |
| moe_A3h_g0_s1 | 一致 | 5/1 | I2D |
| moe_A3h_g0_s2 | 一致 | 7/1 | I2D |
| moe_A3h_g0_s3 | 一致 | 3/1 | I2D |
| moe_A3h_g0_s4 | 一致 | 5/1 | I2D |
| moe_A3h_g0_s5 | 一致 | 3/1 | I2D |
| moe_A3h_g0_s6 | 一致 | 4/1 | I2D |
| moe_A3h_g1_s123 | 一致 | 3/1 | I2D |
| moe_A3h_g1_s1 | 一致 | 4/1 | I2D |
| moe_A3h_g1_s2 | 一致 | 4/1 | I2D |
| moe_A3h_g1_s3 | 一致 | 5/1 | I2D |
| moe_A3h_g1_s4 | 一致 | 4/1 | I2D |
| moe_A3h_g1_s5 | 一致 | 4/1 | I2D |
| moe_A3h_g1_s6 | 一致 | 4/1 | I2D |

I 共需 448 个新健康候选：I1 为 8、I2D 为 166、I2P 为 166、I3 为 108；H+I 为 896 个新候选。本次只完成 I0，不启动 I1/I2D。

证据：[`i0_reuse_audit.json`](evidence/i0_reuse_audit.json)、[`i0_reuse_candidates.csv`](evidence/i0_reuse_candidates.csv)。每份历史输入的 SHA-256、生成程序 AST 摘要、模型区域摘要、当前实际遍历摘要和逐字段判断均保留在 JSON 中。

复算：`e2e/post_fg/dist/venv/Scripts/python.exe e2e/hi_reuse.py audit`。
