# MF6 固定计算块诊断

2026-11-13，Codex。只读保存图与原始评估；没有运行新搜索或后端仿真。

三例均沿用自身 MF6：一颗 PE 的 TensorCore/VectorUnit 不可用，路由器保留。EIR 在 s100 因相邻档改进不足停止；以下覆盖五个修复种子。

| 案例 | 健康Tensor块数 | 失效PE | R0峰值ns | EIR五种子峰值范围ns | 有效刚性下界ns | 峰值/下界 |
|---|---:|---|---:|---|---:|---:|
| FACE_proj | 64 | (1, 5) | 9,437,184 | 9,437,184–9,437,184 | 9,437,184 | 1.000000 |
| G23_ffn | 192 | (0, 14) | 37,748,736 | 37,711,872–37,748,736 | 37,711,872 | 1.000978 |
| G33_ffn | 288 | (6, 5) | 25,178,112 | 25,178,112–25,178,112 | 25,104,384 | 1.002937 |

FACE投影的Tensor块等时，使用 ceil(K/P_live)×c。两个FFN因序列片段取整包含两种接近的块时长，分别报告基本下界 max(max c, sum c/P_live)，以及计数下界 ceil(K/P_live)×min c，取两者较大值。计数下界的理由是：K块分给P_live个核，至少一个核有ceil(K/P_live)块，每块时长不低于min c。这是有效下界，没有假定这些不等时块完全相同。

三例Tensor负载已经达到或很接近该刚性下界。该结论约束计算负载；实际后端总延迟仍包含依赖、输入输出和链路争用。Vector负载单独保留在完整表中，未假定向量块也平衡。三个MF6均没有同条件FULL，PF4 FULL不用于此处。

原孤儿来自健康图，FACE为1 Tensor+1 Vector；2×3 FFN为2 Tensor+2 Vector（含16项部分和归约）；3×3 FFN为2 Tensor+1 Vector。Split须保留原归约，向量孤儿按原粒度迁移。

## 大图SA时间

| 相位 | 候选数 | SA中位秒 | 已记录分项中位秒 | 未归属中位秒 | 回滚检查中位次数 |
|---|---:|---:|---:|---:|---:|
| prefill MHA | 14 | 12293.748 | 7693.228 | 4635.880 | 865.5 |
| decode MHA | 14 | 680.398 | 534.499 | 133.536 | 513.5 |

普通t_energy不包含拒绝后为验证精确回滚再次执行的energy；剩余时间还含其它循环工作。这里仅列sa_wall减七类记录时间，不把差额全归到某个热点。WP2用短轨迹单独测量。上述均是先前并行环境的主机墙钟，不是芯片时间或J阶段串行计时。

## 文件

- [逐状态摘要](tables/mf6_summary.csv)、[逐核负载](tables/mf6_loads.csv)、[计算事件](tables/mf6_events.csv)
- [原始状态与SHA-256](evidence/mf6_sources.json)、[检查](evidence/mf6_checks.json)
- [SA逐候选分项](tables/sa_cost_breakdown.csv)

复算：`python -B e2e/post_fg_diag.py --run-out e2e/fg/out/main --out e2e/post_fg`。
