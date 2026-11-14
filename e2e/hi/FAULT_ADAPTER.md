# H4/I4 故障入口准备

2026-11-14。第一批只完成入口实现和短接口核对；未启动新模型故障选择后端、L1 或 L2 搜索。实现为 `e2e/hi_fault.py`，旧 `fg_fault.py`、`eir.py`、CPCM 与 A–D 源码保持原样。

## 已实现

- 在单作业进程的上下文中，将原 F/G 的模型准备和模板路径绑定到 `hi_config.prepare`、`hi_config.task_path`。沿用原 `selection/pre/l1/rep/stop` 执行与输出。
- 候选选择要求同一模型、硬件、合同、图顺序、W1 版本一致；每组 14 个候选，每候选后端种子 1–10、十次随机平局 CPCM 记录齐备。选优仍采用原后端均值，并列保留随机 CPCM 选优信息。
- MF6 锚点来自 `hi_manifest` 中**同模型、同硬件**的 14 个 A3h decode MHA 候选，选后端均值最小者。重新在种子 1 后端的 Tensor+Vector 忙时上选非角落 PE，同值时选最小 `(die, core)`。不读取 GPT 的候选清单。
- 故障文件只生成 PF4 和 MF6。PF4 固定为 `(die0,core11) ↔ (die1,core8)`；MF6 仅移除选中 PE 的 Tensor/Vector，Router 与链路全部保留。不生成 MF1/MF8 任务。
- 搜索与 L1 确定性估值接入已验证 `post_fg_bb`。逐步调试 `checks=False`，原 `Engine.before/revert` 保留。最终独立后端与估值恢复原实现；显式带随机数生成器的 CPCM 调用仍进入原随机平局实现。
- H4/I4 使用原五修复种子 `{123,1,2,3,4}`，各自从同一 L1 启动的 `{30,100,300}` 档，以及原五种子平均停档和后端中位代表选择。没有修改 EIR 的 2% 规则。
- 可生成每模型 13 个 A3h 故障单元：主硬件 PF4 八个、MF6 一个、WATOS PF4 四个；两模型最多 390 个 L2 档位任务。另有主硬件 decode 四算子的 A1-R0，每模型四个。MF6 的 prefill 算子必须显式给出 `pf_ffn` 或 `pf_proj`；未设自动改选规则。

## 本机已经核对的内容

证据：`e2e/hi/evidence/fault_adapter_checks.json`；原接口运行材料：`e2e/hi/out/fault_adapter_smoke/`。

1. 两模型、两硬件的四组锚点均为 14 个不同候选，配置身份相同。
2. 真实 2×2 平台上，固定 PF4 边存在；给定忙时示例排除了更忙的角落核，两个同忙非角落 PE 按稳定坐标选中 `(0,5)`；删除该 PE 计算单元后链路集合完全相同。
3. 上下文绑定与恢复通过；小确定性图在加速路径结果一致，带随机数生成器的调用进入原随机平局分支。
4. 原停档函数的输入示例：L1 改进为零但仍高于健康 2% 时继续；30 档不因小于 2% 的跨级改进而停；100 档相邻预算增益不足 2% 时停；五种子后端中位代表不变。
5. 后续清单计数为 26 个 A3h 故障单元、最多 390 个 L2 作业。此次后端调用与修复搜索均为 **0**。

以上属于接口、物理结构和停档示例核对，不能当成新模型故障运行结果。

## 等健康 14 候选齐备后再验证

- 按完整候选组生成赢家，保存状态重载与健康 `T_hat` 相等。
- 实际新模型 decode MHA 的种子 1 后端排名与 MF6 PE 记录。
- PF4 的 R0/L1；实际 MF6 PE 在预选 prefill FFN/projection 上的孤儿与合法切分。
- 完整 30/100/300 档、五修复种子、十独立后端、停止结果及导出状态。
- 新模型完整图与原/快推演的一致性及一次修复拒绝回滚短检查。

MF6 预选信息不替代实际锚点上的合法性核对，也不因为某个 PE 不可拆而偷偷更换故障位置。

## 后续调用

```bash
python e2e/hi_fault.py check --out e2e/hi/out/fault_adapter_smoke
python e2e/hi_fault.py manifest --model llama --hardware busybarn --mf6-operator pf_ffn --out e2e/hi/out/main
python e2e/hi_fault.py child --spec <saved-job-spec.json> --out e2e/hi/out/main
```

`manifest` 仅生成计划。其清单同时带入健康输入任务，使现有调度器可检查依赖及匹配成功记录；后续获准批次在同一输出目录续跑，已成功健康候选按原身份复用。第一批不执行该故障清单。
