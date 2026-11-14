# CRISP 项目记忆

本页继承 Claude 项目记忆中的长期决策。实验参数和结论查各阶段交接、`PROGRESS.md` 与结果表。研究方法以最新定版为准。

2026-11-14后续WP1–WP5及归档交付已完成。用户授权范围为 `e2e/CRISP_POST_FG_CODEX_HANDOFF.md` 的WP1–WP5，具体计划 `e2e/POST_FG_PLAN.md`；当前公网地址 `118.195.219.207`（11月13日旧地址 `1.13.251.45`），云端 `~/crisp_post_fg_20261113`，本机输出归 `e2e/post_fg/`。WP1三例MF6诊断、WP2的13项等价计时、Split27项以及DSTU健康/PF4评估全部完成。健康180项为174次后端和6项批内复用；最后S256/BB0/A256的90项PF4全部成功，274.280秒。云端主清单316项身份和结果核对通过（13计时、297次实际后端、6批内复用），成功作业8.505053进程小时；本机完整包124852269字节，4639文件长度及SHA-256全通过；输出在post_fg/out，实际云端源码在post_fg/dist/server_runtime，证据complete_download.json。用户获准的三分钟COS备份已启用，目的地 `cos://crisp/crisp/POST_FG_20261113`，09:33:20完成首个完整轮次；脚本 `e2e/post_fg_cos_backup.sh`，日志 `~/crisp_post_fg_cos_backup.log`。旧输入及失败记录实际位于 `e2e/post_fg/out/main/attempts/20261113_before_fresh_baselines/`，258原文件摘要已核对。详细代码、报告和表图已推送Codex分支05f3b25，归档证据随后追加；COS于09:45:05确认完整包上传成功。系统八组合与H/I/J仅准备下一批计划，尚未启动。

2026-11-13 F/G收尾：服务器 `119.45.93.240:~/crisp_fg_20261113` 的6813项已于16:53:45全部成功，另56个Gemini结束状态。最终记录 `e2e/fg/out/main/finish.batch.json`，代码c123b47。最后5项由finish控制器5并行补齐，298.18秒，最低可用内存214.76GiB；原OOM记录保留。完整包已保存到本机 `e2e/fg/dist/FG_20261113_complete.tar.gz`，解包至 `e2e/fg/out/main/`；所有实验文件和运行源码摘要与云端清单一致。详细F/G正文、CSV/JSON表、四类PNG/SVG图和复算证据已生成，入口 `e2e/fg/README.md`。服务器由用户关机，后处理在本机完成，按授权推送 `codex/e2e-handoff`。用户更新内存处理：有余量自动续跑；余量不足时通知，恢复后自动继续，不再重复确认。用户要求监控至完成时，当前任务持续检查服务器并完成后续下载/汇总/推送。

本轮策略结论：默认All256+C4继续保留；S256改善全部所测P完成时间，但健康D有4/18回退、PF4有6/18回退，作为可选取舍。消息连续优先未进入默认方法；本轮不追加时间门控或TieFIFO。Split-m2在三案例九次后端均低于固定原块的有效下界，保留给下一批跨模型PE验证。系统八组合默认S=A256；若整组采用S256，也都有本轮PF4的110/111共36项可复用，新增108项联合，单跑另计。

## 研究口径

- CPCM 的起点是资源负载与关键路径的差异。E3 中 FFN 最忙片内链路失效的延迟代价为 +3.0%，关键链路失效为 +31.8%。见 `e3_warmstart/PROGRESS.md`。
- BusyBarn 在建平台和映射前读入固定故障。FULL 是本研究构造的故障后完整重编译参照，遍历 14 个变体。EIR 比较恢复延迟与权重迁移量。按当前定版，跨级恢复目标是回到故障前估计延迟的 2% 以内，故障后下界用于分析。见 `eir/EIR_SUMMARY.md`、`eir/EIR_NEXT_SUMMARY.md` 和 `AGENTS.md`。
- DSTU 的默认方法是全链路 256 KB 切块和按跳 C4 优先级，`φ_res` 为可选增强。重叠流的主指标为 `L̂_max`，同时报告 `S_max`、`L_max` 与完成时间。E4b 中静态 EDF 与 BestFixed 在设定 2 打平，动态 slack 的增益受配置限制。见 `e4_overlap/E4B_SUMMARY.md`、`dstu/FINAL_RUNS_SUMMARY.md` 和 `e2e/w1_rerun/W1_RERUN_SUMMARY.md`。
- 端到端主负载使用 W1，MHA 最终广播去除 QKᵀ、S 消息。W0 用于复现与附录对照。KV 定义为 batch 16 内共享的历史 KV。B4 的 A3h 相对 A1 改善主要来自 decode，TPOT −31.8%，TTFT −1.3%。见 `e2e/B4_SUMMARY.md`。
- C 的 PE 故障使 pf_proj 与 pf_ffn 的最重 PE 工作量分别达到原来的 2 倍和 1.5 倍。现有映射和 FULL 沿用相同切块，这个下界适用于各映射臂。见 `e2e/C_SUMMARY.md` §4。
- BusyBarn 在同状态、同种子下存在进程间延迟波动。自检用结构一致性和多种子统计判定这类结果，确定性路径按原验收标准核对。把排查时间放在会改变 CRISP 结论的差异上。
- 端到端各阶段保留已确定的实验覆盖，用并行分片安排计算。论文计时按 J 阶段串行测量。

## 接续入口

- F/G收尾已获用户明确要求：完整输出下载至`e2e/fg/out/main/`，归档放`e2e/fg/dist/`，详细汇总为`F_SUMMARY.md`、`G_SUMMARY.md`及完整表格/图/复算记录；核对后推送`origin/codex/e2e-handoff`，已授权。具体内容与原始数据目录见`e2e/FG_REPORT_PLAN.md`。最终6813/6813已齐，56个工作量代表和168个健康赢家快照已解析；健康部分23项检查、修复/联合10090个原始样本及全部配对指标核对通过。

- D 的主体与补充种子使用各自的比较口径，汇总见 `e2e/D_SUMMARY.md`，原始数据和复算入口见 `e2e/E2E_HANDOFF.md` §7。
- E 已完成并归档，Gemini prefill 对齐 8 头，完整层采用八份八头任务。672 个主候选、16 个结束状态对照及 6,880 次后端评估核对通过。A3h 相对 A1 的 44 层 decode 改善为混合 32.84%、条带 38.10%，请求 2048 改善为 28.83%、33.83%；完整结果和图见 `e2e/E_SUMMARY.md`。十九个 C 故障复用于混合布局 G2 的三层，EIR 相对 R0 改善十八个请求，MF6 增加 0.054%（44 层）。F/G 已获准实施；执行安排见 `e2e/FG_PLAN.md`，实际运行状态见最新进度。
- E 的追加分析使用现有结果：G0/G1 四层组受损时，A3h 的请求2048最慢组周期中位退化由 R0 +19.65% 降到 EIR +5.62%，理想服务率损失由16.42%降到5.32%；G2三层仅3/19故障超过健康四层瓶颈。Gemini prefill FFN 的最忙计算核工作量是A1/A3h的2.5倍，MHA呈现DDR与片内链路访问集中，证据见 `e2e/e/evidence/gemini_attribution.json`。F/G 方案已按用户 2026-11-12 更新修订，见 `e2e/FG_PLAN.md`：F 主搜索 1344、G 主搜索 1568、带宽扫描 112，共 3024；联合 585+24=609，配对单跑和五个 FFN 核对点另计。EIR 为 128 配置、640 配置×种子组合，最多 1920 个独立 L2 档位作业。后续获准运行后按依赖并行推进，128 为上限、按实测内存安排活跃数；时间用完整小批更新。2026-11-13 用户已授权实现、本机冒烟、云端小批及直接接续正式批次，服务器新地址 `119.45.93.240`，环境保留。小批后报告结果和预计耗时，已授权任务按依赖继续。

Claude 记忆来源为 `busybarn-no-runtime-faults.md`、`crisp-design-entry-points.md`、`e4a-hop-arbitration-finding.md`、`e4b-deadline-arbitration-finding.md`、`e2e-w1-and-c-results.md`、`focus-on-own-method.md` 与 `e2e-coverage-over-time.md`。EIR 与 DSTU 的当前定义已与仓库中的后续定版对齐。
