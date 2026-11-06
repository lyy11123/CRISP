# CRISP 论文实验仓库

CRISP：面向晶圆级芯片上 LLM 推理的关键路径感知映射与故障修复（目标期刊 IEEE TC / TPDS）。本仓库是论文的测量实验目录，基线平台是 BusyBarn（ISCA 2026）artifact 的模拟器、硬件配置和模型集合。

论文方案、判定关口和结论记录在 claude.ai 上的在线方案文档里，不在本仓库。**项目说明、规则和 artifact 的已知事实写在 [`CLAUDE.md`](CLAUDE.md)，任何工具或 agent 开始工作前都要先读它。**

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
| `dstu/` | DSTU：CPCM 截止时间代替理想值 | 阶段 1、2 完成（汇总见 `DSTU_SUMMARY.md`、`STAGE2_SUMMARY.md`） |

每个实验文件夹里：
- `<实验>_HANDOFF.md`：交接说明（任务定义、步骤、自检、停止条件），执行时以它为准；
- `PROGRESS.md`：进度记录，**只在末尾追加**，不改写旧条目；最新状态和“待用户决定的问题”看最后一条；
- `*.py`：实验脚本；`_inject/sitecustomize.py`：注入到子进程的内存替换；
- `<实验>_out/`：输出（csv、json、log、汇总）。

## 仓库里没有的东西

为控制体积，下列内容不进仓库（见 `.gitignore`），只留在产生它们的机器上（主机：本地 Windows，`E:\58Institute\PhD_TERM3\CRISP`）：

- `*.pkl`、`*.pkl.gz`：捕获的映射、事件图、修复后的事件对象（仅 E3/E4 就约 630 MB）；
- `e*/work/`、`cpcm/work/`、`cpcm_sa/work/`：各实验的 artifact 工作副本（从 `third_party` 复制，可重建，见各实验 `PROGRESS.md` 第 1 步）；
- `e4_overlap/e4_out/sims/`、`*_out/_tmp/`、`cpcm_sa/out/` 下的 `logs/`、`_smoke*/`、`_tmp/`：仿真缓存、日志与临时文件；
- `third_party` 中 `make generate_exp` 生成的 `cfg/`、`py/`、`results/`（artifact 自带的 `.gitignore` 已排除）。

需要这些文件的分析（如 `e2_attribution.py --from-dump`、E3/E4 的续跑与重算）只能在主机上做，或在新机器上按 `PROGRESS.md` 中记录的命令行重新生成。重新生成前先估计耗时，超过 1 小时先问用户。

## 多工具协作约定

本仓库由多个 AI 编码工具（Claude Code、Codex、Gemini CLI、Cursor 等）轮流推进。**Claude Code 为主（main）**：负责维护本 README、`CLAUDE.md` 和 `main` 分支的整体状态；其他工具按下面的约定工作。

### 开始工作

```bash
git clone https://github.com/lyy11123/CRISP.git   # 第一次
git checkout main
git pull --rebase origin main                    # 每次开始前
```

然后依次读：`CLAUDE.md` → 当前实验的 `<实验>_HANDOFF.md` → 该实验 `PROGRESS.md` 的最后几条。不清楚当前该做什么时，问用户，不要自行选题。

### 工作规则（摘自 `CLAUDE.md`，以它为准）

- 只做当前交接说明里写的事；CPCM 按 `cpcm/` 下经用户确认的方案实现，DSTU 按 `dstu/DSTU_HANDOFF.md`，阶段 2 之后的新实验用户放行后才写代码；不要扩大范围。
- 不修改 `third_party/` 里的源码；需要改行为时在内存中替换函数。
- 一个进程只跑一个任务脚本；记录 artifact commit、随机种子和命令行。
- 单次运行预计超过 1 小时，先告诉用户。
- 每完成一步，在该实验 `PROGRESS.md` 末尾追加一条：日期、步骤、状态、命令、自检、关键数字、对源码的新发现、待用户决定的问题。只写确认过的事实，推测要标明。**条目标题后注明是哪个工具写的**，例如 `## 2026-11-03 第 5 步：……（Codex）`。
- 本机环境：Windows + PowerShell 5.1，Python 见各脚本；`make` 和 bash 不一定可用，按 Makefile 顺序直接运行 python 脚本。

### 提交与推送

- 同一时间只让一个工具改同一个实验文件夹，避免 `PROGRESS.md` 冲突。
- 小步提交，提交信息格式：`[<工具名>] <实验>: <做了什么>`，例如 `[Codex] e4: 第一批 F5 汇总`。
- 推送前先同步：

```bash
git pull --rebase origin main
git push origin main
```

- 改动较大或想让主会话审阅时，推到分支 `<工具名>/<主题>`（如 `codex/e4-batch2`）并开 PR，不要直接合入 `main`。
- 不提交 `.gitignore` 排除的文件；单个文件超过 50 MB 不要提交，先问用户。
- 不使用 `git push --force`、不改写 `main` 的历史。
- 解决 `PROGRESS.md` 冲突时两边条目都保留，按时间顺序排列。

### 与主会话交接

工具切换时，在当前实验的 `PROGRESS.md` 追加一条交接条目（做到哪一步、正在跑的进程与日志位置、下一步、待用户决定的问题），提交并推送。下一个工具从这条开始读。
