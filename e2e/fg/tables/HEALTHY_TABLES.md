# F/G 健康、候选与规模表

这些表由`e2e/fg_report.py`读取已保存的结果生成，不运行搜索、CPCM推演或后端模拟。每张表同时保存CSV和JSON；JSON保留列表和来源字段的原始结构。

| 表 | 内容 |
|---|---|
| `candidates` | 每个候选或Gemini终态的十个后端种子统计、搜索和评估用时、图指标、峰值内存 |
| `backend_samples` | 每候选×后端种子的总时间、纯计算、纯通信、同时活动时间 |
| `operators` | 十四候选齐备后的后端选优、候选中位数、随机平局CPCM选优；另附确定性CPCM选优 |
| `candidate_distribution` | 十四个候选的后端均值分布；不与同一候选十个后端种子的波动混合 |
| `layers` | 单个Die组上的一层prefill/decode以及512/2048步请求组装 |
| `gemini_terminal_pairs` | 56对种子123的最好能量状态与结束状态，标记直接复用评估的相同状态 |
| `scaling` | G固定/增长切分，各方法及各选择方式对应E的64核基线；加速比及其除以核数比的效率 |
| `bandwidth_search` | WATOS 0.5×/1×/2×D2D下重新搜索的decode MHA/FFN；冻结映射联合结果在另一张表 |
| `backend_components` | 各选择方式所选候选的十种子平均时间组成 |
| `graph_metrics` | 所选候选的事件数、边数、消息和路径字节、逐跳活动、CPCM编译/推演时间、SA/评估/总时间、内存 |
| `workload` | 完整模式解析快照：输入尺寸、有效TensorCore乘加量、128×64粒度取整量、归约工作、向量输入元素 |
| `resource_loads` | 完整模式解析健康后端赢家：TensorCore、向量单元、DDR、片内和Die间链路的静态累计工作 |
| `job_timing` | 每个作业唯一成功receipt，跨控制器接续不重复计数 |
| `timing_by_category` | 成功receipt按配置、作业类型、初始化/预算等分别统计 |
| `controller_timing` | 已保存批次记录；相同副本合并，控制器经过时间因重叠不相加 |
| `retry_attempts` | 归档的非零返回码尝试，同一尝试在不同文件出现时只记一次 |

时间字段带`_ns`的是仿真纳秒；`wall_seconds`、`search_seconds`、`evaluation_seconds`等是本次并行运行在CPU上的秒数。它们不作为J阶段独立串行计时。均值/范围描述实际种子与候选；`std`采用总体标准差，四分位数按排序后位置`(n−1)×p`线性插值。

`relative_to_*`、`terminal_vs_best`、`*_selector_gap`表示比值减一；`speedup`是基线时间除以当前时间。`efficiency`再除以核数比。后端三元组的余项是计算和通信同时活动时间，不是排队归因。`resource_loads`是各资源服务时间的累加，不表示利用率或等待时间。

层公式固定为prefill `2 LN + 8 MHA + proj + FFN`，decode `LN + 8 MHA + proj + FFN`；decode FFN内部已有一次LN。MHA微图8头，组装乘8，共64头。请求组装保留KV长度2048，不模拟生成过程中KV增长。

仅结果JSON下载齐时加`--metadata-only`，它保留种子、配置和结果核对，只将本地缺快照排除在失败判断之外；`--skip-snapshots`与它同义。完整模式要求快照存在，进一步解析代表工作量及健康赢家资源工作。

```powershell
python -B e2e/fg_report.py --manifest e2e/fg/manifest/search.json --run-out e2e/fg/out/main --out e2e/fg/tables --workflow-manifest e2e/fg/dist/server_runtime/e2e/fg/out/pipeline_jobs.json --run-root e2e/fg/dist/server_runtime
python -B e2e/fg_report_check.py --tables e2e/fg/tables --run-out e2e/fg/out/main --out e2e/fg/evidence/healthy_report_checks.json
```

`summary.json`记录实际使用的源目录、清单摘要、artifact版本、汇总脚本摘要和生成时间。上述运行清单只用于身份核对与统计，不被修改。
