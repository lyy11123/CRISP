# CPCM 进度记录

只在末尾追加，不改写旧条目。交接说明：`CPCM_HANDOFF.md`。

## 2026-11-03 V0：working_time 核实、输入编译、手工小图（Claude Code）

**状态**：完成，通过（16 个用例、193 项检查，不一致 0）。

**环境与复现**
- artifact commit `299089ef266c54c6a1fa8c2f52df66e6c970b52a`（zip 下载）；Python 3.13.16（`C:\Users\YiyangLiu\AppData\Local\Programs\Python\Python313\python`）。
- 命令（在 `cpcm\` 下）：`python cpcm_v0.py`；`python cpcm_v0_backend.py 50`（后端交叉核对，种子 0–49）。
- 输出：`cpcm_out\v0.json`（每个用例的活动、事件时间、LST、D、W、CP、CB）、`cpcm_out\v0_backend.json`。
- V0 不涉及 artifact 任务脚本、不写 `results\`，没有建 `cpcm\work\`。

**第一步：`working_time` 是否只依赖静态量** —— 是，可以直接调用。
- 链路（`src/platform/device/link/{co2co,ch2ch,ddr2co,co2ddr}.py` 第 34 行）：`int(ceil(ceil(bytes/bandwidth)·timeunit) + latency)`，只用字节数和 cfg 常数。
- 张量单元（`module/tensorcore.py` 第 31 行）、向量单元（`module/vectorunit.py` 第 51 行）：只用 datashape、行为类型、频率和 cfg 常数；返回 `np.ceil(...)`（numpy float，值为整数）。
- 基类 `link`、`module` 没有覆盖；`src` 下没有对 `working_time` 的赋值替换。不读 `work_flag`、`work_endtime`、`work_record`。
- 任务脚本导入的是 `event_driver`（第 282 行这一版，调用点第 359、441 行），不是 `_v2`；该版不对计算时长取整，所以后端时间是值为整数的 numpy float。`compile_bb` 把时长转成 int 并检查无损，非整数或非正时长直接报错。

**实现（`cpcm\cpcm.py`）**
- `build(events, resources)`：通用构图；`compile_bb(events_dict, hp, path_lists=None, deps=None)`：从 `build_event` 之后、后端消耗 `dependency_set`/`path_list` 之前的对象编译。事件按静态拓扑序编号（Kahn，平局按 tag 排序）；依赖为 CSR；通信为“分支 → 逐跳（链路编号、时长、是否含三元组键）”的扁平列表；资源表登记平台上全部模块和链路（故障已由平台删除）。
- 正向三种模式，输出同一个 `Schedule`（活动 = 一次计算或一跳，按发放顺序编号，扩展图的每条边都从小编号指向大编号）：
  - `bb`：复现 Phase 1–4，平局规则 (剩余长度降序, 事件编号)。
  - `hop`：E4b 的按跳仲裁，键为 (prio[e], −ln, e)，prio 是外部固定数组；不给 prio 即 E4b 的 P1。
  - `order`：按资源强加服务顺序（V1 回放与手工换序用），非工作守恒。
- 反向（`Analysis`）：LST、slack、`critical_path()`（E2 的规则）、`wait(e)`、`blocking(ρ)` 给出 CB(ρ) 与 φ(b,·)、`deadline(b)`。
- `reroute(g, b, branches, link_dur)`：单消息改路评估，返回 T̂′、F′(b)、D(b)、是否 F′(b) ≤ D(b)。**目前是全量重算**，增量更新（方案第 5 项）还没写，计划在 V7 之前补上。

**V0 用例与结果**（时间单位 0.1 ms；R 在 0.2 ms 就绪用一个 2 单位的前驱计算表示）

| 用例 | 内容 | T̂ 及关键数字（与手算一致） |
|---|---|---|
| ex1_original_bb | 方案文档 ② 例子一，原方案 | FFN 开始 20（2.0 ms）；D(P)=20；CB(L1)=8，φ(P)=8；W(R)=8 |
| ex1_defer_P_order | 例子一，推迟 P（order 模式） | FFN 开始 22（2.2 ms）；φ(R)=10 < CB(L1)=12（L1 在 [0,2] 空闲） |
| ex1_reroute_P_L2 | 例子一，P 改走 L2（经 `reroute`） | FFN 开始 12（1.2 ms）；F′(P)=10 ≤ D(P)=20 |
| ex1_kv_mapA / mapB | 例子一 + 1.5 ms 就绪的 KV 写回 | 两映射每条链路聚合负载相同（L1 20、L2 10），FFN 开始 20 对 12 |
| ex2_fcfs_bb | 例子二，先到先服务 | up 20、down 40；D(P′)=40（4.0 ms） |
| ex2_defer_P2_order | 例子二，推迟 P′ | up 12、down 32；F(P′)=22 ≤ 32 |
| c1 | 三跳消息同 tick 发放全部空闲跳；长路径优先 | F(A)=5（存储转发会是 15）；B 等 5 |
| c2 | hold-and-wait：首跳先发、第二跳等 | A：S=0、S_last=20、F=25；首跳挡住 C 2 个单位 |
| c3 | 组播，共享首跳只发一次，一个分支被挡 | A 共 3 个活动；F(A)=17 |
| c4 | 结束时间 = 最后发放那一跳的结束 | 两跳分别在 9、5 结束，F(A)=5 |
| c5 | 同设备计算争用 | E2 推迟到 5 |
| c6a / c6b / c6c | bb 长路径优先 / hop 固定优先级 / hop 默认 P1 | Y 分别在 5、0、5 开始 |
| c7 | 与数据依赖矛盾的强加顺序 | 报死锁错误 |

**自检**
1. 后端交叉核对（`cpcm_v0_backend.py`）：把 10 个 `bb` 用例用桩对象喂给 artifact 原版 `event_driver`（只读导入，不改源码），种子 0–49。10/10 个用例中 CPCM 的结果属于后端的结果集合；8 个用例后端 50 个种子只有 1 种结果，c2、c3 有 2 种（随机平局），CPCM 分别与其中 22/50、29/50 个种子相同。
2. 同一批桩对象经 `compile_bb` 编译后的调度，与 `build` 构图的调度逐活动相同（10/10）。
3. 变异对照：把例子一中 P 的预期改成 [0,11]，`run_case` 报告不一致。
4. 反向遍历内置检查：所有活动 slack ≥ 0；关键链每一步连续（否则抛错）；`gaps`（等待 > 0 但资源前驱结束时刻 ≠ 开始时刻）在所有用例中为 0。

**对源码的新发现**（已核实的写明来源，推断标明）
- Phase 4 事件内分支排序的键是 `max(len(lst) for lst in pair)`（`event_driver.py` 第 412–415 行），`lst` 是链路键而不是分支，所以比较的是链路键元组的长度（2 或 3；三元组来自 `event_notation.get_paths` 中带 `ori, idx` 的 D2D 跳），等于“剩余部分含三元组键链路的分支先发”，并不是按分支长度。它只影响同一 tick 内哪一跳最后发放。
- 通信的 `end_time` 是最后发放那一跳的结束时刻（第 468 行），不是各跳结束的最大值；后端在 c4 上 50 个种子都给出 F=5（另一跳在 9 结束）。因此可能有非终止跳在 T̂ 之后才结束；反向遍历的时间上界取 T_end = max(T̂, 所有活动结束)，V0 中 c4 的 T̂=6、T_end=9。
- 推断（论证，并经上面的交叉核对支持，未穷举证明）：后端“每发一跳就重新洗牌、排序并从头扫描”的循环，在确定性平局下等价于“每个长度层按 (−最长剩余分支, 编号) 扫描一遍”，因为同一 tick 内链路只会由空闲变忙，被挡的事件重扫仍被挡。CPCM 按后者实现。
- 计算事件的同 tick 争用：当就绪表里没有任何带路径的通信事件时，Phase 4 不执行 shuffle，计算事件按就绪表顺序发放（c5 后端只有 1 种结果）；就绪表顺序由 Phase 2 遍历对象集合决定，属于地址依赖。CPCM 按事件编号。

**待用户决定的问题**
1. 几个定义按下面的方式实现，请确认或修改：
   - W(e) = S_last(e) − r(e)，即从就绪到最后一跳发放；
   - CB(ρ) 同时给原始和（关键链上在 ρ 处等待之和）与 E2 的“排他计入”两种；
   - D(b) 在 b 没有数据后继时取 T_end；
   - φ 按等待区间与服务区间的重叠计，order 模式下资源可能空闲，Σφ 可小于等待（ex1_defer_P_order：10 对 12）。
2. `hop` 模式在一轮内按链路编号处理，因此同一轮中一个事件发出多跳时，“最后一跳”按链路编号决定；E4b 按候选字典的插入顺序。只影响这种平局下的 F，V5 时会报告差异。
3. V1 回放要用 `order` 模式，需要从后端记录里取每个通信的终止跳（`term_res` 参数已留）。当前 `order` 实现每发一个活动扫描一遍待发集合，复杂度 O(A²)；MoE 约 4,782 个事件，跑 V1 前会先改成按资源的就绪队列。这一项不改变语义。
