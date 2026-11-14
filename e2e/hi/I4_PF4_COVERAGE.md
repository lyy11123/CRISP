# I4 PF4 完成范围

当前本机正式结果已覆盖12/12个预定A3h PF4点：I4-light九点、WATOS decode FFN一点、本轮两种硬件prefill FFN两点。

| 硬件 | 算子 | 批次 | 最终EIR状态已齐 |
|---|---|---|---|
| busybarn | dc_ffn | i4_light | 是 |
| busybarn | dc_ln | i4_light | 是 |
| busybarn | dc_mha | i4_light | 是 |
| busybarn | dc_proj | i4_light | 是 |
| busybarn | pf_ffn | closeout_B | 是 |
| busybarn | pf_ln | i4_light | 是 |
| busybarn | pf_mha | i4_light | 是 |
| busybarn | pf_proj | i4_light | 是 |
| watos | dc_ffn | i4_watos_dffn | 是 |
| watos | dc_mha | i4_light | 是 |
| watos | pf_ffn | closeout_B | 是 |
| watos | pf_mha | i4_light | 是 |

三个批次共270个作业。实际L2 105 条、停档复用 75 条；后端共 1490 次。四个A1-R0对照属于原I4-light，未计入12个A3h点。

详细新结果、原十点合并、主硬件P/D与固定KV的P+2048D参考见 [B_I4_PF4_COMPLETE.md](closeout/B_I4_PF4_COMPLETE.md)。WATOS仅有MHA/FFN四类结果，不生成完整层。MF6/Split是C的独立工作包，不混入PF4覆盖。
