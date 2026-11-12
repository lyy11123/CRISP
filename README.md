# CRISP 论文实验仓库

CRISP：面向晶圆级芯片上 LLM 推理的关键路径感知映射与故障修复（目标期刊 IEEE TC / TPDS）。本仓库是论文的测量实验目录，基线平台是 BusyBarn（ISCA 2026）artifact 的模拟器、硬件配置和模型集合。

论文方案、判定关口和结论记录在 claude.ai 上的在线方案文档里，不在本仓库。**Codex 从 [`AGENTS.md`](AGENTS.md) 接手。长期研究决策见 [`PROJECT_MEMORY.md`](PROJECT_MEMORY.md)，Claude 原工作说明见 [`CLAUDE.md`](CLAUDE.md)。**

## 目录与状态（2026-11-05）

| 位置 | 内容 | 状态 |
|---|---|---|
| `third_party/isca2026-busybarn-artifact/` | BusyBarn artifact，只读；commit 见 `third_party/isca2026-busybarn-artifact.COMMIT` | 已就位（commit 299089e） |
| `e1_fidelity/` | E1：聚合代价与后端延迟的秩相关 | 试点完成 |
| `e2_attribution/` | E2：关键等待归因、单资源重排与删除上限 | 完成 |
| `e3_warmstart/` | E3：故障后热启动 SA、迁移量与下界紧度 | 完成（含扩展，44 个“任务 × 故障”组合） |
| `e4_overlap/` | E4a：多流重叠下的链路冲突与仲裁收益上限；E4b：按跳截止时间仲裁原型 | 完成（汇总见 `E4A_SUMMARY.md`、`E4B_SUMMARY.md`） |
| `cpcm/` | CPCM：依赖感知的调度级延迟估计（第 2 步） | 完成（V0–V7，汇总见 `CPCM_SUMMARY.md`） |
| `cpcm_sa/` | CPCM 接入 SA 映射搜索（第 3 步） | 完成（本地 952 次 + Colab 224 次，汇总见 `CPCM_SA_SUMMARY.md`） |
| `eir/` | EIR 分级故障修复主实验（第 4 步） | 完成（1,240 次，汇总见 `EIR_SUMMARY.md`）；下一阶段增量实验 A、C、B 已完成（汇总见 `EIR_NEXT_SUMMARY.md`） |
| `dstu/` | DSTU：全链路切块 + 按跳优先级仲裁；CPCM 用于选切和前向试算 | 阶段 1、2、2C、块开销敏感性、全链路设定 1–3 与非零开销选切对照完成（汇总见 `DSTU_SUMMARY.md`、`STAGE2_SUMMARY.md`、`STAGE2C_SUMMARY.md`、`OVERHEAD_SUMMARY.md`、`SETTING3_ALL_SUMMARY.md`、`FINAL_RUNS_SUMMARY.md`） |
| `e2e/` | 端到端阶段：完整 prefill + decode、故障、联合实验、硬件、规模、补齐模型 | 进行中（交接 `E2E_HANDOFF.md`） |

每个实验文件夹里：
- `<实验>_HANDOFF.md`：交接说明（任务定义、步骤、自检、停止条件），执行时以它为准；
- `PROGRESS.md`：进度记录，**只在末尾追加**，不改写旧条目；最新状态和“待用户决定的问题”看最后一条；
- `*.py`：实验脚本；`_inject/sitecustomize.py`：注入到子进程的内存替换；
- `<实验>_out/`：输出（csv、json、log、汇总）。

## 仓库里没有的东西

为控制体积，下列内容不进仓库（见 `.gitignore`），只留在产生它们的机器上（主机：本地 Windows，`E:\58Institute\PhD_TERM3\CRISP_codex`）：

- `*.pkl`、`*.pkl.gz`：捕获的映射、事件图、修复后的事件对象（仅 E3/E4 就约 630 MB）；
- `e*/work/`、`cpcm/work/`、`cpcm_sa/work/`：各实验的 artifact 工作副本（从 `third_party` 复制，可重建，见各实验 `PROGRESS.md` 第 1 步）；
- `e4_overlap/e4_out/sims/`、`*_out/_tmp/`、`cpcm_sa/out/` 下的 `logs/`、`_smoke*/`、`_tmp/`：仿真缓存、日志与临时文件；
- `third_party` 中 `make generate_exp` 生成的 `cfg/`、`py/`、`results/`（artifact 自带的 `.gitignore` 已排除）。

需要这些文件的分析（如 `e2_attribution.py --from-dump`、E3/E4 的续跑与重算）只能在主机上做，或在新机器上按 `PROGRESS.md` 中记录的命令行重新生成。重新生成前先估计耗时，预计超过 1 小时先告知用户。

## Codex 工作副本

本副本从 Claude 的项目副本接手。项目规则与 Git 约定见 [AGENTS.md](AGENTS.md)。当前分支为 `codex/e2e-handoff`，以后 Codex 的 GitHub 推送使用 `origin` 上的 `codex/*` 分支。

继续实验时，先读对应的交接文档和 `PROGRESS.md` 最后的记录。D 阶段的结果和复算命令见 [端到端交接](e2e/E2E_HANDOFF.md)。
