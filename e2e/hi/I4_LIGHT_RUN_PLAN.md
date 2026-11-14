# I4 已就绪算子的 PF4 运行计划

2026-11-14。只补主硬件7个A3h算子点（prefill LN/MHA/projection及decode LN/MHA/projection/FFN）、WATOS prefill/decode MHA两个A3h点，另准备主硬件4个decode A1-R0。所有prefill FFN、WATOS decode FFN和MF6留在后续批次。

## 输入与方法

182个已有健康候选，13组各14项，result/raw/snapshot全部齐全并记录SHA-256。每组选十后端均值最小的健康状态，平局按candidate_order，记录随机CPCM所选状态用于比较。原结果直接引用，不重新搜索健康映射，不提前对缺项组做选择。

PF4固定双向物理D2D边die0/core11与die1/core8，不做映射相关重选。沿用H4已实测的hi_fault/fg_fault/EIR：R0、L1、L2G+sync，五修复种子123、1、2、3、4，独立30/100/300档均从同一L1启动；原2%停止和后端中位代表规则不变。确定性bb加速仅用于搜索，精确回滚、关闭逐步额外checks，最终评估仍用原实现、十后端种子1–10。

## 数量与保存

共201个调度作业：40控制作业（2选优、2固定故障定义、36停档判断）、13个健康重放、13个R0/L1、最多135条L2独立轨迹。停止后的复用记录计入完成作业，不计实际修复轨迹。最多1700后端样本：健康130、A3h R0/L1 180、A1-R0 40、L2最多1350。

新输出目录为 `out/i4_light`，清单 `manifest/i4_light.json`，健康引用 `manifest/i4_light_healthy_inputs.json`。代码只在新进程内绑定已验证H4包装的入口和目录，已有运行源码不变。

## 资源与初始时间

新池为H3/I2、I3和H4中所有未取得有效成功回执的作业预留名额；只占剩余槽，四池合计最多128进程。未运行但可重试的前池作业同样保留名额。系统可用内存扣除四池未来增长预算后预留3GiB。

用13组选中的完整健康候选实测耗时估算，全部135条L2都执行时约6.14进程小时，14至128槽均约58分钟，最长依赖链是主硬件decode FFN。实际自然停止将减少档位。该估计包括完整修复与后端评估和状态导出，下载、报告另计；没有把额外核数当作能加速单一算子后端的条件。实际Qwen MHA短验证记录单列在 `evidence/i4_light_smoke.json`，本机时间与云端分开。

```text
python e2e/hi_i4.py check
python e2e/hi_i4.py launch
```

批量由主控在真实Qwen MHA pre/R0/L1验证和独立检查通过后启动。当前批次不为填满CPU加入最长的prefill FFN修复链。

## 实际Qwen MHA验证

主硬件decode MHA实际五项验证通过：健康T̂=6090，精确等于选中的健康候选；PF4 R0=7746，L1=7656。原停止规则结果为L1、continue，stop=False。健康、R0、L1各十后端种子1–10均从raw核对，共30个样本。本机合计174.27秒，最大进程RSS 0.551GiB。这些是本机接口验证，独立于正式云端结果。
