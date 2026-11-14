# I4 PF4 就绪情况与初始估时

仅检查已有文件，没有启动I4实验。检查时间UTC 2026-11-14T09:56:10.058901+00:00。这里统计本机已恢复的结果，不代表云端此刻的实时完成数。

完整I4 PF4结构与H4相同：12个A3h算子点、主硬件4个decode A1-R0输入；264个调度作业，其中52个控制作业、16个健康重放、16个R0/L1、最多180条实际L2轨迹，后端样本上限2240。MF6另行准备，不在此清单内。

本机224个必需健康候选已有159个原结果/raw/snapshot；16组选优输入中11组已齐14候选。只有完整组才固定后端均值选优状态，缺项组不提前选优。

| 硬件 | 阶段 | 算子 | 方法 | 已有/14 |
|---|---|---|---|---:|
| busybarn | decode | ffn | A1 | 14/14 |
| busybarn | decode | ffn | A3h | 14/14 |
| busybarn | decode | ln | A1 | 14/14 |
| busybarn | decode | ln | A3h | 14/14 |
| busybarn | decode | mha | A1 | 14/14 |
| busybarn | decode | mha | A3h | 14/14 |
| busybarn | decode | proj | A1 | 14/14 |
| busybarn | decode | proj | A3h | 14/14 |
| busybarn | prefill | ffn | A3h | 1/14 |
| busybarn | prefill | ln | A3h | 14/14 |
| busybarn | prefill | mha | A3h | 14/14 |
| busybarn | prefill | proj | A3h | 14/14 |
| watos | decode | ffn | A3h | 1/14 |
| watos | decode | mha | A3h | 1/14 |
| watos | prefill | ffn | A3h | 1/14 |
| watos | prefill | mha | A3h | 1/14 |

主硬件7个A3h点（除prefill FFN）和4个A1-R0点的健康选优输入已经齐备。若后续按已就绪算子先运行，需要使选择文件和依赖只包含本批已齐的算子；不让缺项组参与选择。当前只完成就绪检查，未生成提交运行的I4清单。

按现有完整单作业实测、假定全部180条L2都执行，预计总计38.96进程小时。14/32/64/128槽估出的最长依赖链均约331.2分钟（5.52小时）：Qwen WATOS prefill FFN已有单候选的十后端约51.77分钟，健康重放、R0/L1、各独立档位依次推进，增加空闲核不能缩短同一算子这条依赖链。

这是一份全部档位执行的初始安排估计，不是预计一定会运行五个半小时。原2%停止规则会跳过后续档位。必须先看H4实际运行与Qwen健康选优完成后的图，再用I4真实R0/L1决定剩余档位；不改变停止规则来压缩时间。

估时包括搜索、十后端、导出，下载和汇总另计。缺项组仅用已经完成的同算子A3h候选作成本参照，未拿它替代正式14选优。完整计算与源结果SHA见 `evidence/i4_pf4_readiness.json`。

更新检查：`python e2e/hi_i4_prepare.py`。这个入口只读健康结果、写就绪与估时报告，没有启动实验的命令。
