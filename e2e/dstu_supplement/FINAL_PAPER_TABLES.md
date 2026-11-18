# 论文表格输入

结果状态：已齐，完整三策略组108/108。

| 文件 | 内容 |
|---|---|
| tables/s256_runs.csv | 逐策略、逐相位、逐种子结果、实际窗口及完整输入身份 |
| tables/s256_pairs.csv | S256/A256、S256/original、A256/original逐行配对 |
| tables/s256_overall_means.csv | 全部已完成组等权均值、中位数、范围、改善与回退计数 |
| tables/s256_state_summary.csv | 分模型、硬件、映射与修复状态 |
| tables/s256_model_summary.csv | GPT、Llama、Qwen分别汇总，显示各自实际范围 |
| tables/s256_system_increment_summary.csv | M、R与Transport条件增量，保留A256与S256两套系统结果 |
| tables/s256_system_full_summary.csv | S系统12种组合分别相对同输入000 |
| tables/s256_phase_summary.csv | 三相位分组，全部样本纳入 |
| tables/s256_solo_interference.csv | 单跑变化O、跨流干扰J及分解等式 |
| tables/s256_oracle.csv | 按Lhat_max选完整已执行策略的事后最优参照，保留所有并列 |
| tables/s256_pareto.csv | P/D双目标非支配策略，逐种子完整列出 |
| tables/s256_strategy_choices.csv | 固定策略、三种事后选择及冻结前验门控的逐组完整结果 |
| tables/s256_strategy_summary.csv | 各类选择的P/D相对original变化、迟约、超期及策略计数 |
| tables/online_window_comparison.csv | 36个相位组的预测与三种子实际KV窗口、方向与重叠对照 |
| tables/online_window_samples.csv | 108条窗口逐种子值与来源；不参与门控动作更改 |
| tables/online_gate_costs.csv | 12次独立预测的准备/构图/前向/任务体成本，每个输入状态只计一次 |
| tables/existing_36_*.csv | 历史GPT36组，单独报告 |

图表先展示完整均值与P/D取舍，再用固定相位和实际共享重叠解释代表场景，最后列系统增量。S256结果带明确策略后缀，旧A256的111不被覆盖。所有表格均为原始结果后处理，没有新增搜索。
