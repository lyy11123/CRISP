# J：正式计时汇总

生成时间：2026-11-14T18:08:28.286927+00:00。只读取保存结果；本次后处理没有启动搜索或仿真。

已核对 45/125 项：旧 45 项保留，新计时最多 80 项。当前状态：partial。

| 批次 | 计划项 | 已核对 | 待完成 |
|---|---|---|---|
| old45 | 45 | 45 | 0 |
| micro | 24 | 0 | 24 |
| search | 20 | 0 | 20 |
| eir | 36 | 0 | 36 |

micro 包含新模型 J1 的 18 项与 C 两个真实代表点 J5 的最多 6 项；C 输入尚未进入清单的项列为 pending_C_representative。
J2a 是三个输入 × A1/A3h × 三次完整 1000 步，J2b 是 GPT144 prefill MHA × A1/A3h 各一次。J3 是三个输入 × 三个修复种子 × R0/L1 与独立 30/100/300 步。

## 计时如何相加

1. 离线映射：初始化、构图、校准、完整 SA、恢复最佳状态、导出和候选选择。J2 当前固定单候选，不将它乘十四后当成十四候选实测。十次随机 CPCM 单列；十种子后端是质量验证，只有采用后端选优的完整流程才把相应后端计入选择成本。
2. 运行时控制：R0/L1、每个 EIR 档位的校准/搜索/恢复、最终 CPCM、状态导出，以及 DSTU 准备和 Split 所有合法轴的构造/评分/选择。
3. 论文质量验证：独立后端调用和复制准备、恢复快照复查、诊断与计时记录分列。Python 仿真时间不是芯片上逐包仲裁延迟。

## J1：CPCM 完整调用与前向推演

每个独立进程预热一次后测五次稳态，先取进程内五次中位数，再报告三个进程中位数及最小/最大值。五次重复不当作五个独立进程。完整调用包含 compile、forward 和 analysis；fast 的 pack/kernel/unpack 已包含在 forward 中，不能再次相加。输入读取、平台创建和预热另列。
旧 GPT 的首次 JIT 与磁盘缓存各三次继续保留；新图记录实际编译签名与预热。只有签名相同才引用旧启动测量；新签名记录于 J_TIMING.json，预热含缓存/JIT，不冒充稳态。

| 输入 | 实现 | 计时项 | 进程数 | 中位秒 | 最小秒 | 最大秒 |
|---|---|---|---|---|---|---|
| GPT144_prefill_mha | fast | forward_complete_seconds | 3 | 0.548452 | 0.547396 | 0.552307 |
| GPT144_prefill_mha | fast | full_call_seconds | 3 | 1.9321 | 1.92673 | 1.93731 |
| GPT144_prefill_mha | original | forward_complete_seconds | 3 | 1.5596 | 1.55904 | 1.69062 |
| GPT144_prefill_mha | original | full_call_seconds | 3 | 2.9828 | 2.9654 | 3.25959 |
| GPT64_decode_ffn | fast | forward_complete_seconds | 3 | 0.00174795 | 0.001742 | 0.00174907 |
| GPT64_decode_ffn | fast | full_call_seconds | 3 | 0.00921918 | 0.00905334 | 0.00923355 |
| GPT64_decode_ffn | original | forward_complete_seconds | 3 | 0.00565038 | 0.00564347 | 0.00605481 |
| GPT64_decode_ffn | original | full_call_seconds | 3 | 0.0131271 | 0.0130914 | 0.0162667 |
| GPT64_prefill_mha | fast | forward_complete_seconds | 3 | 0.0388087 | 0.0386983 | 0.0393032 |
| GPT64_prefill_mha | fast | full_call_seconds | 3 | 0.296544 | 0.295711 | 0.3009 |
| GPT64_prefill_mha | original | forward_complete_seconds | 3 | 0.121593 | 0.120472 | 0.122458 |
| GPT64_prefill_mha | original | full_call_seconds | 3 | 0.354053 | 0.351379 | 0.355754 |

## J2：完整映射搜索

initial_graph_and_calibration_seconds 已包含校准；互斥表将校准取出，同时单列初始 A1 CPCM 监测。raw_sa_seconds 只覆盖循环，不含校准。algorithm_search_seconds 扣除循环 A1 CPCM 监测和进度回调，保留原始观测值供核对。搜索时间段与 raw_sa 的小差值记为 search_interval_other_seconds。
广播总时间包含在原时间段中，仅作内部明细。退火标记开销亦为内部明细，不再次从各段中扣减。互斥分项之和逐行等于完整作业时间；最外层入口与该时间的差值单列。
final_eval 没有独立的最终 CPCM 秒表，J2 将最终 CPCM、评估构图、深复制及图指标合列 evaluation_other_seconds，并明确 final_CPCM_seconds 为空。十次真实后端和十次随机 CPCM 调用分别计时；随机 CPCM 的编译包含在 evaluation_other_seconds。

J2 尚无完整结果；表项 pending。

GPT144 prefill MHA 明确 n=1；该行最小/最大相同只是同一次测量，不表示稳定性。所有 J2 质量结果仅随计时保存，不替换健康主表。

## J3：EIR 的分档与累计成本

每个修复种子先计一次 R0/L1，再分别从同一 L1 运行 30、100、300 步。累计到 300 步计 430 个正式提议以及三次独立校准；每档重新开始温度日程，不把 300 步轨迹当成前三档的全部代价。R0/L1 原数值也保存在后续 raw 中，但累计表忽略这些继承字段。
measured_control_with_finalize_export_seconds = 新增 R0/L1 或校准+搜索+恢复 + 最终 CPCM + 状态导出。后端调用及其复制准备独立计费。读取/初始化、原快照整理、未单独计时的诊断与记录保留为 other_job_seconds，不将它伪称为纯算法成本；完整 observed wall 同时列出。
J3 计时的三个修复种子是三次轨迹；研究质量表的五种子挑选是另一项工作。这里既不把三次或五次的最佳质量配上一条轨迹成本，也不把 5×中位数写成五种子的实测。停档及多候选状态选择未在 J3 单独重测，不能从此表补造其耗时。

J3 尚无齐备的累计档位；27 个累计行已列出 pending。

## J4 与 J5：控制准备与完整选轴

J4 的 total_prepare 包含读取平台、全链路 256 KB 构图、CPCM 截止时间/排序及初始 ready 列表。resident_prepare 去掉读取平台；这不是硬件每包服务时间。
J5 的 resident_decision 包含全部合法轴的构造、评分和选择；每轴 phases 包含在 build_seconds 内，而 build_seconds 又包含在 resident_decision 内。完整 case 另含输入、平台、基线指标、冻结图核对和导出，其他差值单列。

| 输入 | 分项 | n | 中位秒 | 最小秒 | 最大秒 |
|---|---|---|---|---|---|
| face:A256 | total_prepare_seconds | 3 | 38.9804 | 38.9153 | 47.0205 |
| face:A256 | resident_prepare_seconds | 3 | 36.1777 | 36.1046 | 43.9524 |
| face:S256 | total_prepare_seconds | 3 | 39.2257 | 38.3623 | 39.5939 |
| face:S256 | resident_prepare_seconds | 3 | 36.4327 | 35.6319 | 36.7527 |
| wsc:A256 | total_prepare_seconds | 3 | 38.9406 | 38.325 | 46.6454 |
| wsc:A256 | resident_prepare_seconds | 3 | 36.1271 | 35.5605 | 43.5815 |
| wsc:S256 | total_prepare_seconds | 3 | 38.8595 | 38.7163 | 39.0153 |
| wsc:S256 | resident_prepare_seconds | 3 | 36.0758 | 35.9562 | 36.2107 |
| FACE_proj | resident_decision_seconds | 3 | 0.43203 | 0.422996 | 0.440349 |
| FACE_proj | complete_case_seconds | 3 | 1.5944 | 1.58645 | 1.62005 |
| G23_ffn | resident_decision_seconds | 3 | 0.446709 | 0.446312 | 0.457502 |
| G23_ffn | complete_case_seconds | 3 | 3.17592 | 3.1171 | 3.19655 |
| G33_ffn | resident_decision_seconds | 3 | 1.35678 | 1.34431 | 1.46269 |
| G33_ffn | complete_case_seconds | 3 | 8.30117 | 8.2292 | 9.182 |

## 环境、来源与复算

J_INPUTS.json/CSV 保存逐项清单、实际完成状态、输入哈希、结果及回执身份；J_ENVIRONMENT.json/CSV 保存每个实际进程的 CPU、系统、Python/NumPy/Numba、线程设置、空闲采样与运行源码哈希。旧批与新批保持标签，不以旧环境代替尚未生成的新环境。
新批单进程与空闲证明读取 out 中保存的 batch 和现场证据；只有结果中的线程设置不能独立证明没有其他进程。已完成旧批的现场证据保存在 hi/evidence/j_serial_conditions.json。

当前环境记录 45 条；来源 SHA-256 161 项。所有已纳入结果均有成功回执，且回执身份与对应清单一致。

生成命令：

```text
e2e/post_fg/dist/venv/Scripts/python.exe -B e2e/hi_closeout_j_report.py --progress
```

去掉 --progress 时要求本地 125 项全部齐备。J_TIMING.csv 是逐进程宽表；tables/j_timing_summary.csv 是独立进程统计，j1_calls.csv 保留所有稳态/预热调用，j2_costs.csv 与 j3_costs.csv 保留分项，j3_cumulative.csv 保存各档累计，j5_axis_costs.csv 保存全轴成本。
