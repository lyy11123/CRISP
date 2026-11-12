# CRISP 项目记忆

本页继承 Claude 项目记忆中的长期决策。实验参数和结论查各阶段交接、`PROGRESS.md` 与结果表。研究方法以最新定版为准。

## 研究口径

- CPCM 的起点是资源负载与关键路径的差异。E3 中 FFN 最忙片内链路失效的延迟代价为 +3.0%，关键链路失效为 +31.8%。见 `e3_warmstart/PROGRESS.md`。
- BusyBarn 在建平台和映射前读入固定故障。FULL 是本研究构造的故障后完整重编译参照，遍历 14 个变体。EIR 比较恢复延迟与权重迁移量。按当前定版，跨级恢复目标是回到故障前估计延迟的 2% 以内，故障后下界用于分析。见 `eir/EIR_SUMMARY.md`、`eir/EIR_NEXT_SUMMARY.md` 和 `AGENTS.md`。
- DSTU 的默认方法是全链路 256 KB 切块和按跳 C4 优先级，`φ_res` 为可选增强。重叠流的主指标为 `L̂_max`，同时报告 `S_max`、`L_max` 与完成时间。E4b 中静态 EDF 与 BestFixed 在设定 2 打平，动态 slack 的增益受配置限制。见 `e4_overlap/E4B_SUMMARY.md`、`dstu/FINAL_RUNS_SUMMARY.md` 和 `e2e/w1_rerun/W1_RERUN_SUMMARY.md`。
- 端到端主负载使用 W1，MHA 最终广播去除 QKᵀ、S 消息。W0 用于复现与附录对照。KV 定义为 batch 16 内共享的历史 KV。B4 的 A3h 相对 A1 改善主要来自 decode，TPOT −31.8%，TTFT −1.3%。见 `e2e/B4_SUMMARY.md`。
- C 的 PE 故障使 pf_proj 与 pf_ffn 的最重 PE 工作量分别达到原来的 2 倍和 1.5 倍。现有映射和 FULL 沿用相同切块，这个下界适用于各映射臂。见 `e2e/C_SUMMARY.md` §4。
- BusyBarn 在同状态、同种子下存在进程间延迟波动。自检用结构一致性和多种子统计判定这类结果，确定性路径按原验收标准核对。把排查时间放在会改变 CRISP 结论的差异上。
- 端到端各阶段保留已确定的实验覆盖，用并行分片安排计算。论文计时按 J 阶段串行测量。

## 接续入口

- D 的主体与补充种子使用各自的比较口径，汇总见 `e2e/D_SUMMARY.md`，原始数据和复算入口见 `e2e/E2E_HANDOFF.md` §7。
- E 已完成并归档，Gemini prefill 对齐 8 头，完整层采用八份八头任务。672 个主候选、16 个结束状态对照及 6,880 次后端评估核对通过。A3h 相对 A1 的 44 层 decode 改善为混合 32.84%、条带 38.10%，请求 2048 改善为 28.83%、33.83%；完整结果和图见 `e2e/E_SUMMARY.md`。十九个 C 故障复用于混合布局 G2 的三层，EIR 相对 R0 改善十八个请求，MF6 增加 0.054%（44 层）。F/G 尚未启动；执行安排见更新后的 `e2e/FG_PLAN.md`。
- E 的追加分析使用现有结果：G0/G1 四层组受损时，A3h 的请求2048最慢组周期中位退化由 R0 +19.65% 降到 EIR +5.62%，理想服务率损失由16.42%降到5.32%；G2三层仅3/19故障超过健康四层瓶颈。Gemini prefill FFN 的最忙计算核工作量是A1/A3h的2.5倍，MHA呈现DDR与片内链路访问集中，证据见 `e2e/e/evidence/gemini_attribution.json`。F/G 方案已按用户 2026-11-12 更新修订，见 `e2e/FG_PLAN.md`：F 主搜索 1344、G 主搜索 1568、带宽扫描 112，共 3024；联合 585+24=609，配对单跑和五个 FFN 核对点另计。EIR 为 128 配置、640 配置×种子组合，最多 1920 个独立 L2 档位作业。后续获准运行后按依赖并行推进，128 为上限、按实测内存安排活跃数；时间用完整小批更新。当前只更新计划，未写 F/G 运行代码或启动实验。

Claude 记忆来源为 `busybarn-no-runtime-faults.md`、`crisp-design-entry-points.md`、`e4a-hop-arbitration-finding.md`、`e4b-deadline-arbitration-finding.md`、`e2e-w1-and-c-results.md`、`focus-on-own-method.md` 与 `e2e-coverage-over-time.md`。EIR 与 DSTU 的当前定义已与仓库中的后续定版对齐。
