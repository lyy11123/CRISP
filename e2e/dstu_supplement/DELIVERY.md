# 2026-11-18 S256 补充交付

本轮已完成。结果入口 [README.md](README.md)，三类选择并列表见 [S_AND_HI_S256_REPORT.md](S_AND_HI_S256_REPORT.md)，完整核对见 [AUDIT_SUMMARY.md](AUDIT_SUMMARY.md)。

## 完成情况

- 564项主队列全部成功：12准备、12前验CPCM窗口预测、540评估记录；另12项独立图身份核对。
- 108组三策略比较、324联合、216逻辑单跑全部齐备。新增真实后端258次，加2次历史回放共260次；270条历史复用、12条批内复用。新增SA为0。
- 36个门控动作在首项新评估开始前全部冻结，108条在线结果使用该动作对应的完整P/D；与三种事后选择及Pareto并排。
- 主计算北京时间15:14:25完成，独立核对15:23:03完成。完整数据已在本机，服务器后续不参与汇总；本轮没有代用户关机。
- P0已补到24点，单轨迹停档、解析恢复回本、WATOS诊断、33条论文证据索引、三张PNG/SVG图均完成。

## 本机保存位置

工作副本：`E:\58Institute\PhD_TERM3\CRISP_codex`。

| 位置 | 内容 |
|---|---|
| `e2e/dstu_supplement/out/main/` | 564项spec、回执、结果、日志、逐跳记录、失败尝试与完整状态历史 |
| `e2e/dstu_supplement/out/probes/` | 两次H/I历史A256回放 |
| `e2e/dstu_supplement/out/identity_check/` | 12次只读图身份核对的日志与回执 |
| `e2e/dstu_supplement/evidence/` | 输入、复用链、前验决定、逐文件验证和最终审计 |
| `e2e/dstu_supplement/tables/`、`figures/` | 完整逐种子数据、并列表、条件分析与论文图 |
| `e2e/dstu_supplement/dist/server_runtime/` | 实际主实验v1源码及初始部署包 |
| `e2e/dstu_supplement/dist/server_runtime_identity/` | v2准备修正与独立身份核对源码 |
| `e2e/dstu_supplement/dist/source_inputs/` | 363份精确外部输入快照，来源见archive_input_snapshot.json |

## 归档核对

| 包 | 文件数 | 字节数 | 来源 |
|---|---:|---:|---|
| `S256_COMPLETE_20261118.tar.gz` | 2600 | 26,898,520 | COS完整主运行包 |
| `S256_IDENTITY_20261118.tar.gz` | 42 | 32,597 | COS独立图身份核对包 |
| `DSTU_SUPPLEMENT_20261118_complete.tar.gz` | 3144 | 35,796,198 | 本机合并完整运行、报告、图、源码及输入快照 |

统一归档：`e2e/dstu_supplement/dist/DSTU_SUPPLEMENT_20261118_complete.tar.gz`。

SHA-256：`f171433b524aff72cf0373a9c893a14744f170a3b49b047c99e0e5c600a7feb6`。

全部包先验证整包SHA，再逐文件验证长度与SHA。统一包验证没有覆盖当前工作文件；本交付页与统一包验证记录在包生成后单独保存，避免自引用摘要。COS前缀为 `crisp/DSTU_SUPPLEMENT_20261118`，三分钟备份规则仍保留。

## GitHub

本轮在 `codex/e2e-handoff` 分支整理代码、报告、表图和证据，基线为 `244c59533e6154aafaf997dcb51fe58032b2981e`。推送至同名 `origin/codex/e2e-handoff`；最终提交以仓库日志及本机 `dist/git_delivery.json` 为准。大体积原始输出和压缩包留在本机/COS，Git保存复算入口与详细统计证据。

原A–D与CPCM/EIR/DSTU历史源码没有修改。准备阶段无效的重复变换身份字段、H/I全局负载标签及四次缺输入退出均已记录；所有540条正式输入由独立重新加载原图的核对认证，实际后端与统计无需重跑。
