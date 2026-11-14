# I4 PF4 完成范围

当前本机正式结果覆盖 10/12 个预定 A3h PF4 点。两个独立批次的完成数和计时分别列出。

| 硬件 | 算子 | 来源 | 最终 EIR 状态已齐 |
|---|---|---|---|
| busybarn | dc_ffn | i4_light | 是 |
| busybarn | dc_ln | i4_light | 是 |
| busybarn | dc_mha | i4_light | 是 |
| busybarn | dc_proj | i4_light | 是 |
| busybarn | pf_ffn | planned_later | 否 |
| busybarn | pf_ln | i4_light | 是 |
| busybarn | pf_mha | i4_light | 是 |
| busybarn | pf_proj | i4_light | 是 |
| watos | dc_ffn | i4_watos_dffn | 是 |
| watos | dc_mha | i4_light | 是 |
| watos | pf_ffn | planned_later | 否 |
| watos | pf_mha | i4_light | 是 |

I4-light 本机作业数：201/201；WATOS decode FFN：23/23。A1-R0 四点属于 I4-light 对照，未计入12个 A3h PF4点。两个prefill FFN、MF6与Split仍未补齐；本页不生成完整prefill、请求或WATOS整层结果。
两批合计 224/224 个作业，实际 L2 80 条，停止后复用 70 条；后端样本共 1180 个。两个独立队列有时间重叠，不能相加其经过时间。
