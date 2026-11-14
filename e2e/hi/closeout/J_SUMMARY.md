# J：正式计时汇总

生成时间：2026-11-14T23:10:10.128685+00:00。只读取保存结果；本次后处理没有启动搜索或仿真。

已核对 70/125 项：旧 45 项保留，新计时最多 80 项。当前状态：partial。

| 批次 | 计划项 | 已核对 | 待完成 |
|---|---|---|---|
| old45 | 45 | 45 | 0 |
| micro | 24 | 24 | 0 |
| search | 20 | 1 | 19 |
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
| llama_decode_ffn | fast | forward_complete_seconds | 3 | 0.00340779 | 0.00339084 | 0.00341215 |
| llama_decode_ffn | fast | full_call_seconds | 3 | 0.0153512 | 0.0152351 | 0.0155291 |
| llama_decode_ffn | original | forward_complete_seconds | 3 | 0.011178 | 0.0111769 | 0.0113756 |
| llama_decode_ffn | original | full_call_seconds | 3 | 0.0235192 | 0.0233416 | 0.0255998 |
| qwen3moe_decode_ffn | fast | forward_complete_seconds | 3 | 0.0706975 | 0.0703959 | 0.0738115 |
| qwen3moe_decode_ffn | fast | full_call_seconds | 3 | 0.135501 | 0.135174 | 0.157503 |
| qwen3moe_decode_ffn | original | forward_complete_seconds | 3 | 0.228241 | 0.223351 | 0.291271 |
| qwen3moe_decode_ffn | original | full_call_seconds | 3 | 0.29517 | 0.285833 | 0.399254 |
| qwen3moe_prefill_ffn | fast | forward_complete_seconds | 3 | 0.24111 | 0.234675 | 0.241378 |
| qwen3moe_prefill_ffn | fast | full_call_seconds | 3 | 0.667959 | 0.571734 | 0.68701 |
| qwen3moe_prefill_ffn | original | forward_complete_seconds | 3 | 0.73916 | 0.73697 | 0.755957 |
| qwen3moe_prefill_ffn | original | full_call_seconds | 3 | 1.06109 | 1.05697 | 1.0767 |

本轮新增三图的加速倍数按‘原实现三进程中位数 / 加速实现三进程中位数’计算：

| 输入 | 原前向秒 | 快前向秒 | 前向倍数 | 原完整调用秒 | 快完整调用秒 | 完整调用倍数 |
|---|---|---|---|---|---|---|
| llama_decode_ffn | 0.011178 | 0.00340779 | 3.28015 | 0.0235192 | 0.0153512 | 1.53208 |
| qwen3moe_decode_ffn | 0.228241 | 0.0706975 | 3.22842 | 0.29517 | 0.135501 | 2.17837 |
| qwen3moe_prefill_ffn | 0.73916 | 0.24111 | 3.06565 | 1.06109 | 0.667959 | 1.58855 |

本轮 9 个加速进程中，新增 Numba 签名为 0 个进程；当前签名与旧 JIT/缓存测量一致。每个新进程的首次调用保留为 warmup，启动开销单列，未并入稳态；签名一致与缓存是否命中是不同信息，本批没有单独记录缓存命中计数。完整调用还包含构图与分析，因此其提速小于只看前向推演的提速。

## J2：完整映射搜索

initial_graph_and_calibration_seconds 已包含校准；互斥表将校准取出，同时单列初始 A1 CPCM 监测。raw_sa_seconds 只覆盖循环，不含校准。algorithm_search_seconds 扣除循环 A1 CPCM 监测和进度回调，保留原始观测值供核对。搜索时间段与 raw_sa 的小差值记为 search_interval_other_seconds。
广播总时间包含在原时间段中，仅作内部明细。退火标记开销亦为内部明细，不再次从各段中扣减。互斥分项之和逐行等于完整作业时间；最外层入口与该时间的差值单列。
final_eval 没有独立的最终 CPCM 秒表，J2 将最终 CPCM、评估构图、深复制及图指标合列 evaluation_other_seconds，并明确 final_CPCM_seconds 为空。十次真实后端和十次随机 CPCM 调用分别计时；随机 CPCM 的编译包含在 evaluation_other_seconds。

| 输入 | 方法 | 分项 | n | 中位秒 | 最小秒 | 最大秒 |
|---|---|---|---|---|---|---|
| Qwen_decode_ffn | A1 | wall_seconds | 1 | 813.614 | 813.614 | 813.614 |
| Qwen_decode_ffn | A1 | calibration_seconds | 1 | 26.01 | 26.01 | 26.01 |
| Qwen_decode_ffn | A1 | algorithm_search_seconds | 1 | 364.111 | 364.111 | 364.111 |
| Qwen_decode_ffn | A1 | backend_calls_seconds | 1 | 377.4 | 377.4 | 377.4 |
| Qwen_decode_ffn | A1 | random_cpcm_calls_seconds | 1 | 5.611 | 5.611 | 5.611 |
| Qwen_decode_ffn | A1 | offline_generation_measured_seconds | 1 | 397.949 | 397.949 | 397.949 |

以下五项互不重叠，相加为完整作业时间。映射生成列包含已单独计时的初始化、校准、搜索、最佳状态恢复及导出；记录/复查列包含 A1 的 CPCM 监测、进度回调、快照重载和余下记账时间。最终 CPCM 与评估准备仍留在最后一列，不将它归入纯搜索或真实后端调用。

| 独立进程 | 映射生成 s | 记录/复查 s | 十次后端 s | 十次随机 CPCM s | 最终 CPCM/评估准备 s | 合计 s |
|---|---|---|---|---|---|---|
| J2_Qwen_decode_ffn_A1_r1 | 397.949 | 8.23842 | 377.4 | 5.611 | 24.415 | 813.614 |

A1 用聚合代价做搜索；额外的十次随机 CPCM 用于统一的候选选择比较，不属于 A1 的原生搜索循环。完整作业、外层入口、调度回执分别计时，进程启动和退出后的轮询差值不计作算法成本。每行是一条完整 1000 步轨迹；环境 helper 中通用的‘五次稳态’字符串只适用于 J1 的内核重复，J2 的实际次数以自身协议、轨迹和回执为准。

首个 Qwen decode FFN A1 独立进程已核对 1000 个正式提议、50 个校准提议、10 次后端与 10 次随机 CPCM；与历史同候选的 1000 条非计时轨迹逐项一致。详见[首项复查](evidence/j2_first_complete_review.json)。它的调度回执耗时为 816.234 秒（13.604 分钟），该类旧的并行实测估计为 1437.840 秒。该实测只更新同输入、同方法的剩余重复，不用于缩放其他图或方法的预计耗时。

十四候选整体费用另取历史同输入的完整记录累计：下表每行实际读取 14 个候选、14000 个正式提议、140 次后端及 140 次随机 CPCM。单位为各并行进程经历时间之和（进程分钟），其中包含当时的共享资源竞争；本轮 J2 表是空闲服务器上的固定候选串行实测。

| 历史十四候选输入 | 方法 | 生成含原监测 min | 随机 CPCM 选择评估 min | 后端选择评估及准备 min | 生成+随机 CPCM min | 生成+后端 min | 含全部研究评估 min |
|---|---|---|---|---|---|---|---|
| Llama_prefill_mha | A1 | 76.7739 | 0.286524 | 13.3011 | 77.0605 | 90.075 | 90.9072 |
| Llama_prefill_mha | A3h | 120.601 | 0.293815 | 14.2714 | 120.895 | 134.873 | 135.758 |
| Qwen_decode_ffn | A1 | 163.975 | 2.21844 | 162.887 | 166.194 | 326.862 | 330.097 |
| Qwen_decode_ffn | A3h | 283.198 | 1.34834 | 119.956 | 284.547 | 403.154 | 405.219 |

历史‘生成’由初始化至恢复最佳状态，再加导出组成；旧版 SA 记录包含原有 CPCM 监测和进度回调。随机 CPCM 选择评估含 140 次估值及其编译；后端选择评估含 140 次真实后端及其复制/最终 CPCM 等准备。两种选择流程分别与生成相加，包含两种选择评估的完整研究作业另列；最终对 14 个分数取最小值的程序耗时没有单独秒表。历史与本轮单候选各自保留计时方式，完整候选来源和 SHA-256 见 [j_offline_cost_inputs.json](evidence/j_offline_cost_inputs.json)。

GPT144 prefill MHA 计划每方法测 n=1；最小/最大相同表示同一次测量。其余输入每方法计划测三个独立进程，目前 n 随已完成结果更新。J2 质量结果随计时保存，健康主表继续使用原来的候选组。

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
| llama | resident_decision_seconds | 3 | 0.279373 | 0.279183 | 0.281947 |
| llama | complete_case_seconds | 3 | 1.42435 | 1.42231 | 1.44165 |
| qwen3moe | resident_decision_seconds | 3 | 2.39063 | 2.26909 | 2.40026 |
| qwen3moe | complete_case_seconds | 3 | 5.24129 | 5.04984 | 5.25785 |

新模型的 J5 使用 C 中已固定的真实 MF6 点。合法轴全部参与构造和评分：

| 模型 | 完整评估轴 | 选中轴 | 独立进程 | 常驻决策中位秒 | 含读入/平台/核对/导出秒 |
|---|---|---|---|---|---|
| llama | N/M | N | 3 | 0.279373 | 1.42435 |
| qwen3moe | N | N | 3 | 2.39063 | 5.24129 |

## 环境、来源与复算

J_INPUTS.json/CSV 保存逐项清单、实际完成状态、输入哈希、结果及回执身份；J_ENVIRONMENT.json/CSV 保存每个实际进程的 CPU、系统、Python/NumPy/Numba、线程设置、空闲采样与运行源码哈希。旧批与新批保持标签，不以旧环境代替尚未生成的新环境。
新批单进程与空闲证明读取 out 中保存的 batch 和现场证据；只有结果中的线程设置不能独立证明没有其他进程。已完成旧批的现场证据保存在 hi/evidence/j_serial_conditions.json。

当前环境记录 70 条；来源 SHA-256 271 项。所有已纳入结果均有成功回执，且回执身份与对应清单一致。

生成命令：

```text
e2e/post_fg/dist/venv/Scripts/python.exe -B e2e/hi_closeout_j_report.py --progress
```

去掉 --progress 时要求本地 125 项全部齐备。J_TIMING.csv 是逐进程宽表；tables/j_timing_summary.csv 是独立进程统计，j1_calls.csv 保留所有稳态/预热调用，j2_costs.csv 与 j3_costs.csv 保留分项，j3_cumulative.csv 保存各档累计，j5_axis_costs.csv 保存全轴成本。

本轮质量池 549/549 完成且控制器退出后，等待 28.955 秒启动 J。下载的串行现场记录截至 2026-11-14T22:58:42.498189+00:00，34 条状态记录的最大活跃实验数为 1；启动条件记录中质量实验剩余进程为 0。
本轮运行环境为 AMD EPYC 9K65（实例分配128 vCPU）、Linux 5.15、Python 3.13.18、NumPy 2.5.4、Numba 0.68.1；四项数值库线程设置均为1。现场记录随后续结果下载继续更新。

首24项的[独立复查记录](evidence/j_micro_24_review.json)逐项核对成功回执、输入文件与运行源码哈希：三个J1图的各36次调用具有相同T̂、活动规模和关键链/等待摘要；两个J5冻结图及六个重新导出图的事件语义摘要均一致。J5图的语义摘要与gzip文件字节SHA分别保存。复查没有启动CPCM、后端或搜索。
