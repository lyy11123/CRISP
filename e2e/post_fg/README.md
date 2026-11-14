# F/G 后续实验

本轮按 [交接方案](../CRISP_POST_FG_CODEX_HANDOFF.md) 开展固定分片诊断、确定性 CPCM 提速、DSTU 六方案比较和局部计算拆分。实施顺序、输入调整与运行记录分别见 [计划](../POST_FG_PLAN.md)、[服务器记录](../POST_FG_REMOTE_GUIDE.md) 和 [进度末尾](../PROGRESS.md)。本轮新增代码为 `e2e/post_fg_*.py`，A–G 源码和原始数据保持原样。

## 阅读顺序

| 内容 | 报告与数据 |
|---|---|
| 本轮五个问题、主要结果和下一步 | [总汇总](POST_FG_SUMMARY.md) |
| MF6 与固定计算块下界 | [MF6_DIAG.md](MF6_DIAG.md)、[逐核负载](tables/mf6_loads.csv) |
| bb 推演和正常执行提速 | [BB_SPEEDUP.md](BB_SPEEDUP.md)、[完整计时](tables/timing.csv) |
| DSTU 健康、PF4 与消息服务分析 | [DSTU_REFINEMENT.md](DSTU_REFINEMENT.md)、[逐运行结果](tables/dstu_runs.csv) |
| 故障孤儿块拆分 | [EIR_SPLIT.md](EIR_SPLIT.md)、[五方案对比](tables/split_all_methods.csv) |
| 连续服务策略实现的独立等价检查 | [DSTU_POLICY_SPEEDUP.md](DSTU_POLICY_SPEEDUP.md) |
| 系统八组合、H/I/J 的数量和条件 | [POST_FG_NEXT.md](POST_FG_NEXT.md) |

## 文件安排

- `manifest/`：实际提交的作业清单；文件中的源根目录指向当时云端 F/G 输入。健康初批与统一构图后的清单分别保留。
- `tables/`：逐种子数据、按案例汇总及机制分析，CSV/JSON 同时提供。两张大型消息表的完整JSON以 `.json.gz` 推送Git，CSV仍为完整逐行数据；本机另保留未压缩JSON。
- `figures/`：论文图及可编辑的 SVG、导出 PDF；每张图保留相应数值表。
- `evidence/`：输入、数学语义、等价、结果身份、下载与文件摘要检查。
- `out/main/`：完整新结果、receipt、spec、日志和 trace。`out/main/attempts/20261113_before_fresh_baselines/` 保存被替换的初批文件。
- `out/split_prepare/`：九张 Split 图及真实张量片段、字节、放置记录。
- `out/split_cloud_timing/`：云端构图与选轴计时，和原图中的 Windows 构图时间分开。
- `out/policy_validation/`：独立连续策略实现的完整逐记录比较；新增一次后端，单独列账。
- `dist/`：阶段传输包、最终完整压缩包和云端实际运行源码。

原始结果与压缩包保存在本机，不进入 Git；代码、报告、表图和检查证据推送 `origin/codex/e2e-handoff`。

本轮已于2026-11-14完成全部交付。完整包为 [POST_FG_20261113_complete.tar.gz](dist/POST_FG_20261113_complete.tar.gz)，4,639个文件全部核对通过；同一包已保存到COS的 `crisp/POST_FG_20261113/archive/`。[下载记录](evidence/complete_download.json)包含包大小、SHA-256及解包位置。

## 分析时的共同定义

时间表中明确区分模拟芯片的纳秒与主机计算的秒。D 联合延迟使用完成时刻减释放时刻；所有 DSTU 方法共用既有公共释放窗口和评价期限。只在同图、同构造次序下比较确定性调度轨迹。健康基线统一构图的补评原因见 DSTU 报告，历史 A–G 数值继续独立保存。

Split 的刚性下界约束固定块的计算负载；实际后端延迟包含依赖和通信。提速试点测的是保存图、30步SA与原30步EIR，下一批模型的完整用时由自己的代表作业测量。系统八组合与 H/I/J 目前只准备任务清单。
