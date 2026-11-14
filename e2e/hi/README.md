# H/I/J 实验结果

## 最新完成：H2D、H2P、H1c、I1

2026-11-14 14:05:57，新增 **344/344** 项全部完成：H2D 166、H2P 166、H1c 4、I1 8，实验失败 0。从 12:26:42 启动到全部完成共 **99 分 15 秒**。最高 128 个活跃进程，观测到 CPU 100%，最低可用内存 196.255 GiB；系统预留仍设为 3 GiB。

- [Llama 完整健康结果](../H_HEALTH_SUMMARY.md)：主硬件 8 算子 × 3 方法 × 14 候选，逐算子与单层结果、四种选优比较、固定 KV 请求与 80 层解析参考。
- [本批运行、试点与耗时](NEXT_BATCH_REPORT.md)：344 项执行清点、WATOS 与 Qwen 试点、逐项时间和 RSS、首候选估时与实际完成时间。
- [主结果图](figures/next_health.png)：随机平局 CPCM 选出的候选以十种子后端均值比较，白色菱形标出后端选优参考。
- [完整独立核对](evidence/next_344_review.json)：任务身份、完整搜索步数、原始样本、快照、首候选依赖、分组与层公式。

此前四个 H1 候选原样复用；目前 H 已完成 **340/448**，I 已完成 **8/448**。本轮未启动 H3、I2 及新模型故障、联合和后续 J 任务。A3h 相对 A1 的单层 decode 时间减少 40.42%；prefill 增加 2.65%（两者均按随机平局 CPCM 选优）。后端选优参考下 prefill 相差 −0.12%，因此 prefill 的结论是接近，而非普遍加速。

原始输出位于 `out/next_344/`。完整包为 [HIJ_NEXT_344_20261114_complete.tar.gz](dist/HIJ_NEXT_344_20261114_complete.tar.gz)，其中同时保留前一批结果、执行源码与实际引用的旧输入。完成清点、归档摘要和 COS 上传证据见 `evidence/next_344_delivery.json`。可从仓库顶层运行 `python e2e/hi_next_report.py`、`python e2e/hi_next_check.py`、`python e2e/hi_next_plot.py` 复算本批报告。

## 第一批记录（12:18 完成）

2026-11-14。本批按用户批准的 `../HIJ_STAGED_PLAN.md` 完成 P0、H0、I0 和 H1a/H1b 四个完整 A3h 试点；同时完成已有三例 Split 的 63 次后端补充及已有 GPT/J4/J5 的首批正式计时。

## 阅读顺序

1. [模型、W1、KV、Gemini 和 Split 预筛](H0_I0_MODEL_REPORT.md)：40 张图、16 对方法语义、WATOS 配置、实际头组与缓存字节。
2. [四个完整试点与下一批估时](H1_PILOT_REPORT.md)：4000 步、40 后端样本、40 次随机 CPCM、完整分项成本、H2D/H2P/H1c/I1 分片。
3. [旧 Qwen 28 候选核查](I0_REUSE.md)：可重构字段全部一致；历史实际集合遍历未保存，按本轮要求复用 0，之后重搜 28。
4. [J 首批正式计时](J_FIRST_REPORT.md)：已有 GPT 三图、启动成本、DSTU 准备、三例 Split 两轴构造；J2/J3 和新模型计时仍待后续输入。
5. [Split 十种子结果](j/split_results/SPLIT_SEEDS_1_10.md)：新增 63、合计 90 样本，m2 三例 30/30 低于原分块刚性下界。

## 本批已完成

| 工作 | 完成量 | 云端墙钟 |
|---|---:|---:|
| H0/I0 初始映射图 | 40/40 | 26.044 秒 |
| Gemini 接口 | 16/16，每项两个种子各两步 | 22.019 秒 |
| H1 正式健康候选 | 4/4 | 310.066 秒 |
| 既有 Split 种子补充 | 63/63 | 34.049 秒 |
| J1 完整图准备 | 3/3 | 344.062 秒，准备与其他任务有重叠 |
| J1/J4/J5 串行计时 | 45/45 | 770.194 秒 |

各批时间存在重叠，不相加当作整段墙钟。本批新增独立后端评估 103 次：H1 的 40 次及 Split 的 63 次；32 个 Gemini 两步子进程属于接口核对，不计完整健康候选。J 使用 CPCM 或图构造，没有新增质量后端样本。

第一批结束时，H 的 448 个正式候选完成 4 个，I 尚未运行；当时 H2D/H2P/H1c/I1 的规划时间约 24–48 分钟。后续完整实测显示 Qwen MoE prefill FFN 的十种子后端评估耗时较长，本轮实际用时 99 分 15 秒；估时修订过程与全部分项见上述最新报告。四份原计划清单保留于 `manifest/*_planned.json`。

## 模型与输入

- [Llama 模型合同](models/llama/model_contract.json)、[生成差异](models/llama/generated_sources_diff.md)。
- [Qwen3-MoE 模型合同](models/qwen3moe/model_contract.json)、[生成差异](models/qwen3moe/generated_sources_diff.md)。
- [40 图清单](tables/graph_inventory.csv)、[逐候选预计成本](tables/remaining_candidate_estimates.csv)。
- [D 链和 KV 接口](JOINT_INTERFACE.md)：Llama 8 个 MHA 组、Qwen 4 个；P/D 两流表示不同请求，仅共享资源，不新增跨流因果依赖。
- [故障入口](FAULT_ADAPTER.md)：固定 PF4、按既定规则选 MF6，真实新模型修复等待完整健康候选输入。

健康入口 `../hi_run.py` 直接使用完整冻结搜索与确定性 bb 加速，逐步额外调试关闭，精确回滚保留。后端与随机平局评估用原实现。本批四项原始结果保留；活动数量的旧标签更正写在 [图重载证据](evidence/h1_graph_measured.json)，报告使用真实调度活动数。运行时源码在 `runtime/h1_20261114/`；后续入口另加后端实际事件/集合遍历摘要，便于以后的复用核查。

## 文件与复现

原始任务在 `out/h1_main`、`out/model_checks`、`out/gemini_interfaces`、`out/split_supplement`、`out/j1_prepare`、`out/j_graphs`、`out/j_first_serial`。每批保留 spec、日志、结果及 receipt；H1 另有完整 raw 与选中快照。

本机完整归档：[HIJ_FIRST_BATCH_20261114_complete.tar.gz](dist/HIJ_FIRST_BATCH_20261114_complete.tar.gz)，33,690,348字节、3,698个文件均已通过SHA-256核对；[完成清点](evidence/first_batch_completion.json)、[COS成功记录](evidence/cos_complete_backup.json)。归档与原始输出保留本机和COS，不上传Git大文件。当前云目录为 `ubuntu@118.195.219.207:~/crisp_hi_20261114`；三分钟 COS 备份使用 `crisp/HIJ_20261114`，日志 `~/crisp_hi_cos_backup.log`。完整归档包含本批输出、运行源码、配置及引用的旧输入，原始 A–G/POST-FG 文件没有改写。

主要只读复算命令，从仓库顶层运行：

```powershell
python e2e/hi/build_model_report.py
python e2e/hi_result_check.py
python e2e/hi_report.py
python e2e/hi/j_report.py
python e2e/hi_split_supplement.py report
```

验证记录：[完整搜索协议](evidence/full_search_protocol.json)、[独立代码与结果复核](evidence/independent_review.json)、[报告与计划复核](evidence/report_review.json)。后续运行需按相应批次清单执行，不能把 `all_candidates_planned.json` 当成此次已授权的启动清单。
