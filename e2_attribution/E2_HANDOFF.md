# E2 交接说明：BusyBarn 后端的关键等待归因

> 给 Claude Code 的任务说明。研究方案、判定标准和结果解读在另一处讨论，这里只交代写代码和跑实验需要的全部上下文。不确定的地方先读源码确认，确认不了就停下来问用户，不要猜。

## 0. 背景与目标

论文方案 CRISP 有三个模块：CPCM（依赖感知的调度估计）、DSTU（按数据截止时间对“挡路”的流量做改路或推迟）、EIR（分级增量故障修复）。

E2 的目的是判断 DSTU 是否站得住，要回答：

1. 关键路径上的等待集中在哪些资源上？
2. 每段关键等待是被谁挡住的（阻塞者）？阻塞者是计算还是通信，属于哪个算子、哪类数据？
3. 阻塞者能不能让路：它自己的数据截止时间是否宽裕（能否推迟），有没有空闲的备选路径（能否改路）？
4. 附带两项：结果中未被掩盖的纯通信占总延迟的比例；decode 下从 DDR 读权重是否在关键路径上。

判定规则（来自方案文档）：如果阻塞者普遍既没有截止裕量、也没有备选路径，DSTU 就降为辅助手段。“普遍”的量化门槛建议取“这类阻塞造成的关键等待占全部关键等待的 70% 以上”，这是建议值，在汇总里如实报告数字即可，不要替用户下结论。

你的交付：一个脚本 `e2_attribution.py`，以及在下文任务清单上的运行结果。

## 1. 环境

**目录约定**（主文件夹 `E:\58Institute\PhD_TERM3\CRISP`，各项工作放在各自的子文件夹）：

| 位置 | 内容 |
|---|---|
| `third_party\isca2026-busybarn-artifact\` | BusyBarn artifact 的克隆，只读，不修改 |
| `e1_fidelity\e1_fidelity.py` | E1 脚本，E2 的挂接写法照它来 |
| `e2_attribution\` | 本说明所在；E2 的脚本 `e2_attribution.py` 和输出 `e2_out\` 都放这里 |

- Artifact：`https://github.com/redbird-arch/isca2026-busybarn-artifact`（ISCA 2026 BusyBarn），克隆到上表的 `third_party\` 下。
- 已在 Colab（Python 3.13.18）上跑通原版脚本，无需改代码。依赖：numpy、networkx、matplotlib、PyYAML、simanneal、tqdm。
- 生成实验脚本：在模型目录（如 `models/intra_mapping_gpt_decode/`）执行 `make generate_exp`。若在 Windows 上没有 make 和 bash，就按 Makefile 的顺序直接运行：`python cfg.py`、`python ln.py`、`python mha.py`、`python proj.py`、`python ffn.py`，再建 `results/` 目录。
- 单个任务内存约 100–300 MB（4×4 核的 Die 组）。后端单次评估 0.7–8 秒，取决于算子。
- 参考文件 `..\e1_fidelity\e1_fidelity.py`：演示了如何用 `runpy` 运行生成的任务脚本，在内存中替换函数，并在评估前后保存和恢复随机数状态。E2 可以照它的方式挂接。

## 2. 硬约束

- 不修改 artifact 仓库里的任何文件；需要改行为时，只在内存中替换（monkeypatch）。
- 每个进程只跑一个任务：任务脚本在模块级建立全局状态，同一进程跑多个会互相干扰。
- 不改 SA 和后端的逻辑；我们要分析的是作者流程本身的结果。
- 单次运行预计超过 1 小时的，先告诉用户再跑。
- 记录 artifact 的 git commit、随机种子和命令行，保证可复现。
- 范围只到 E2；不要开始实现 CPCM 或 DSTU。

## 3. 已知事实（读源码与实测所得）

**实验结构**
- 每个任务只映射一层中的一个算子（LN、MHA、投影、FFN 分开），放在 1×1、1×4 或 2×2 的 Die 组上，每 Die 4×4 核，batch 16。decode 时序列长度就是 KV 长度。
- 模板脚本流程：建计算图 → 初始映射 → `build_event` → `stream_mapping`（SA，1000 步，类 `StreamOptimization`，在 `src/mapping/Stream_Mapping.py`）→ `add_broadcast` → `event_driver` → 打印并写出 `SA time cost: (a, b, c)`。
- 三元组含义：a = 总延迟，b = 未被掩盖的纯计算时间，c = 未被掩盖的纯通信时间（`event_driver` 的返回值；画图脚本里重叠部分按 a − b − c 计算）。

**作者流程的两个特点（已实测确认）**
- 最终评估的是 SA 最后停留的状态，而不是能量最优的状态。原因是 SA 在原地修改传入的字典，而模板评估的正是这些字典。
- 端到端对 BusyBarn 的每个算子取 14 次运行（贪心初始化 g0/g1 × 损失权重 l0–l6）中总延迟最小的一个（`fig_endtoend/end_to_end.py` 的 `find_min_abc`）；Gemini 只跑 1 次。一个 block = 2×LN + 8×MHA + 投影 + FFN，算子之间直接相加。

**后端语义**（`src/backend/analytical/event_driver.py` 中的 `event_driver`，约第 282–527 行）
- 每个设备（张量单元、向量单元、链路）一次只服务一个事件，用 `work_flag` 和 `work_endtime` 表示占用。
- 每个设备的 `work_record` 是一个列表，元素为 `[开始, 结束, event_tag]`。链路在约第 446 行追加，计算设备在约第 366 行追加。
- 通信事件逐跳发放；同一时刻就绪的通信，按路径长度从长到短排序，排序前先 `random.shuffle`，所以同等长度之间的顺序随机，结果受随机种子影响（实测同一映射换种子，延迟平均波动约 1%，最大约 8%）。
- 事件对象（`src/scheduling/event_notation.py`）有 `event_type`、`event_name`、`event_tag`、`dependency_set`、`start_time`、`end_time`；通信事件另有 `source_location`、`target_location`、`comm_bytes`、`path_list`；计算事件有 `comp_device`、`comp_location`。注意 `path_list` 在仿真过程中会被逐个移除，仿真后是空的，实际走过的链路要从各链路的 `work_record` 里按 `event_tag` 反查。
- 需要你确认：多跳通信的下一跳，是否要等上一跳传完才能发放（存储转发），还是可以同时占用；以及事件的 `start_time`、`end_time` 分别对应第一跳和最后一跳的什么时刻。

**故障模型**（`src/scheduling/communication/topology/mesh_2d.py`）：坏节点整体删除（核与路由器一起），坏链路按有向边删除。E2 第一轮不涉及故障。

**已有实测**（GPT decode，序列 2048，2×2 Die 组）

| 任务 | 后端单次评估 | SA 一步 | 作者流程报出的总延迟 |
|---|---|---|---|
| FFN · BusyBarn（g0 l0） | 0.74 秒 | 36 毫秒 | 226,474 |
| FFN · Gemini | 1.06 秒 | 20 毫秒 | 313,695 |
| MHA · BusyBarn（g0 l0） | 8.3 秒 | 19 毫秒 | 26,086 |
| MHA · Gemini | 3.8 秒 | 4 毫秒 | 31,495 |

14 次运行中总延迟最小的：FFN 为 g0 l6（203,603），MHA 为 g0 l4（20,006）。

## 4. 要计算的量（定义）

对后端仿真得到的每个事件 e：

- 就绪时间 r(e)：所有依赖事件结束时间的最大值，没有依赖时为 0。
- 开始 s(e)、结束 f(e)、时长 d(e) = f(e) − s(e)：取仿真结果。通信事件若逐跳发放，额外记录每一跳在各链路上的区间。
- 等待 w(e) = s(e) − r(e)；多跳通信还要计算每一跳的等待（该跳开始时刻减去它可以开始的时刻）。

**关键链**：从结束最晚的事件开始往回走。在事件 e 上：
- 如果 e（或它的某一跳）有正的等待，前驱是“资源阻塞者”：在 e 需要的那个设备上、占用结束时刻恰好等于 e 开始时刻的那个事件，从该设备的 `work_record` 查得。边类型记为 resource。
- 否则前驱是结束最晚的那个依赖事件，边类型记为 data。

**阻塞者能否让路**（对关键链上每一段资源等待，阻塞者记为 b，被挡的事件记为 e）：
- 数据截止 D(b)：b 的所有消费者（依赖 b 的事件）在当前调度中开始时刻的最小值；没有消费者时取总延迟。
- 可推迟：r(e) + d(e) + d(b) ≤ D(b)，即先让 e 用完资源、b 再接着用，b 仍能在它的消费者开始前完成。
- 可改路（仅当 b 是通信时）：在当前网格上，b 的源和目的之间，长度不超过当前路径 + 2 跳、且避开被争用链路的路径中，是否存在一条，其所有链路在 [r(b), r(b) + d(b)] 内都空闲（按 `work_record` 判断）。分别报告“存在这种长度的路径”和“其中有空闲路径”。
- 分类：只能推迟、只能改路、两者皆可、都不行。按造成的关键等待时间加权统计各类占比。
- DSTU 收益上界：可推迟或可改路的那几类关键等待之和，除以总延迟。这只是上界：去掉一段等待后其他事件会移动。

**附带**
- 纯通信占比 c / a、纯计算占比 b / a。
- 关键链上 DDR 相关链路（co2ddr、ddr2co）上的时间占关键链总时长的比例。
- 关键等待按资源的集中度：等待最多的 1 个、5 个资源各占关键等待总量的比例。

**必须通过的自检**（写进汇总，不通过要报错停止）
1. 关键链终点的结束时刻等于 a。
2. 沿关键链累加（时长 + 等待）等于 a（允许整数取整误差）。
3. 所有事件满足 s(e) ≥ r(e)。

## 5. 实现建议

- 挂接方式：在用 `runpy` 运行任务脚本之前，替换 `event_driver` 模块里的 `event_driver` 函数（任务脚本是 `from event_driver import event_driver`，所以要在它导入之前先导入该模块并替换属性）。包装函数调用原函数后，深拷贝保存事件字典，以及仿真后的 `hardware_platform`（含各设备的 `work_record`）。只保留最后一次调用，即模板的最终评估。
- SA 过程中不调用后端，所以只有初始评估和最终评估两次调用。确认这一点后再取最后一次。
- 先在一个小任务上把数据结构打印出来看清楚（事件数量、`work_record` 样例、一个多跳通信的各跳区间），再写分析。
- 后端带随机性。第一轮直接分析模板最终评估那一次，同时记录当时的随机种子状态；如需看稳定性，再用 `e1_fidelity.py` 的方式对同一状态换 3 个种子各跑一遍。

## 6. 任务清单（先小后大）

1. 单任务打通并通过自检：`models/intra_mapping_gpt_decode/py/ffn_sq2048_greedy0_lossratio6_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0.py`（FFN 的 14 选 1 最优）。
2. 同配置下另外 5 个：`mha_..._greedy0_lossratio4_...`（MHA 最优）、`ffn_..._greedy0_lossratio0_...`、`mha_..._greedy0_lossratio0_...`、`ffn_sq2048_gemini_...`、`mha_sq2048_gemini_...`（其余参数同上）。
3. 扩展（每项先跑 BusyBarn 最优变体和 Gemini 各一个；“最优变体”需要先跑 14 个变体取总延迟最小，可以不带分析只跑原脚本）：Die 组 1×4；序列 512 和 8192；一个 prefill 模型 `models/intra_mapping_ch`（OPT prefill）；如有余力，一个 MoE decode 模型。

## 7. 输出

每个任务输出到 `e2_attribution\e2_out\`：
- `<task>_critical.csv`：关键链上每个事件的 tag、名称、类型、设备或链路、r、s、f、等待、前驱及边类型。
- `<task>_blockers.csv`：每段资源等待的被挡事件、阻塞者、资源、等待时长、D(b)、可推迟、存在备选路径、存在空闲备选路径。
- `<task>_summary.json`：三元组及纯通信占比、关键等待占总延迟比例、资源集中度、四类阻塞的时间占比、DSTU 收益上界、DDR 在关键链上的占比、三项自检结果、artifact commit 与种子。
- 终端最后打印一行 `[e2] {...}`，包含汇总里的主要字段，方便用户直接复制。

## 8. 遇到这些情况先停下来问

- 后端语义与第 3 节的描述不符（比如多跳不是逐跳发放）。
- 自检通不过，且原因不是取整。
- 关键链上找不到资源阻塞者（等待为正，但设备的 `work_record` 里没有在该时刻结束的占用）。
- 单个任务运行超过 1 小时。
