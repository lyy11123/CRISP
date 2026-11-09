# W1 补跑 Colab 指南（EIR 中涉及 MHA 的部分，最终版 2026-11-09）

用户 2026-11-08 放行（`e2e\E2E_HANDOFF.md` 第 4 节末尾）：EIR 放 Colab，DSTU 在本机跑（不在本指南里）。运行清单见 `W1_RERUN_INVENTORY.md` 第 1 节，比较规则见 `e2e\PROGRESS.md` 的“W1 补跑比较规则”。

规则照原实验：同映射来源、同故障坐标、同种子、同停止规则（ε = 2%），评估照原口径（BusyBarn 后端，10 个种子，对象按地址取哈希）。

用户 2026-11-09 定（`E2E_HANDOFF.md` 第 4 节第 8 条）：
- E3（BusyBarn）起点改用 W1 下按模板规则选出的 A1 g0 l1，用 `w1_e3start.py` 按 E3 的方式捕获；同一流程先重新捕获 l4，与 E3 原起点逐位相同。
- 7 类按 A3h 映射选坐标的故障（F4a、F5a、F8a 和 C 的四种），在 W1 的 A3h 起点上按原规则重选坐标，编号为 F4aW、F5aW、F8aW、CRandomW、CClusterW、CEdgeW、CScratchW，写在 `eir_faults.json` 里。新坐标只跑 L2-Global（+sync）的 30、100、300 步三档，以及 FULL-A3h。旧编号保留旧坐标，旧坐标的运行照常跑，作为 W0 对 W1 的同口径对照。唯一差别是工作负载 W1：最终广播去掉 QKᵀ、S 的消息，每次广播删 72 条，每次都核对条数。驱动是 `e2e\w1_rerun\w1_eir.py`：在内存里把 `eir.py`、`eir_next.py` 重定向到本文件夹的起点、故障表、artifact 工作副本和 Drive 上的输出目录，不改 `eir\` 的任何文件。

| 笔记本名 | 分片 | 内容 | 作业数 | 预计（7 个并行） |
|---|---|---|---|---|
| `CRISP_w1_prep` | `prep` | 起点评估 3，R0 + L1 23（含新坐标 7 个，即修复前的故障后延迟），故障选择 1，2 个参照作业；与本机逐位比较；故障坐标报告 | 29 | 约 15–25 分钟 |
| `CRISP_w1_eir_main_a3h` | `eir_main_a3h` | A3h 起点修复 180（F4、F5、F4a、F5a），对照 15 | 195 | 2.6–3.0 h |
| `CRISP_w1_eir_main_e3` | `eir_main_e3` | E3 起点（A1 g0 l1）修复 240（7 个故障），对照 15 | 255 | 3.6–4.2 h |
| `CRISP_w1_eir_full` | `eir_full` | FULL-A3h 112（F4、F5、F4a、F8a、复合故障 4 种，各 14 次），FULL-BusyBarn 98（7 个故障 × 14 个 A1 变体） | 210 | 3.7–4.3 h |
| `CRISP_w1_eir_next` | `eir_next` | next 修复 205（F8a 旧方法、新方法、加种子、复合故障），阶段 B 检查点 30 | 235 | 2.8–3.2 h |
| `CRISP_w1_eir_w1sel_rep` | `eir_w1sel_rep` | 新坐标的 L2Gsync：F4aW、F5aW、F8aW × 3 档 × 5 个种子（45）；四种复合故障 × 3 档 × 10 个种子（120） | 165 | 2.5–2.9 h |
| `CRISP_w1_eir_w1sel_full` | `eir_w1sel_full` | 新坐标的 FULL-A3h：7 × 14 | 98 | 2.6–3.0 h |

预计按 B4 的 Colab 验证折算：7 个并行时单次约为本机（14 个并行）记录的 1.25–1.45 倍。`CRISP_w1_prep` 先跑完并通过比较，其余 6 个同时开，墙钟约 4–4.5 h。新坐标两个分片的预计用 W0 的单次记录估算：修复 30、100、300 步的中位数分别为 208、约 300、418 s，FULL-A3h 中位数 531 s。

## 准备（本机，只做一次）

压缩包已在本机生成：`e2e\w1_rerun\dist\w1_eir_bundle.tar.gz`。生成时已在干净目录里自查：解到空的临时目录，用 `python -I`、不带 PYTHONPATH，跑冒烟（每种作业各一次，步数缩短）和探针（与本机逐位比较）。

把它上传到 Google Drive 的 `我的云端硬盘/CRISP_colab/`。7 个笔记本共用同一个压缩包和同一个输出目录 `CRISP_colab/w1_eir`。

## 每个笔记本的单元格

运行时类型选 CPU + 高 RAM。`<分片>` 换成上表的分片名。

**单元格 1：挂载、解压、探针**（约 1–2 分钟）

```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["OUT"] = "/content/drive/MyDrive/CRISP_colab/w1_eir"
!mkdir -p /content/crisp && tar -xzf /content/drive/MyDrive/CRISP_colab/w1_eir_bundle.tar.gz -C /content/crisp
!pip install -q numpy networkx simanneal tqdm PyYAML matplotlib
!nproc && free -g | head -2 && cat /content/crisp/BUNDLE_INFO.json | head -4
!cd /content/crisp && python e2e/w1_rerun/w1_eir.py probe --out $OUT
```

最后一行要是 `PROBE PASSED`：在 W1 的 A3h 起点上跑一次起点评估，T̂、路由、行为位置和删去条数（每次 72 条）都要与本机逐位相同。否则不要往下跑，把输出发回来。

**单元格 2：分片运行**（断线后重新执行单元格 1 和这一格即可续跑，已完成的作业会跳过）

```python
!cd /content/crisp && python e2e/w1_rerun/w1_eir.py run <分片> --out $OUT --workers 7 2>&1 | tee -a /content/w1_run.log | grep "\[w1\]\|PREP\|ALERT\|STOP"
```

- `prep` 跑完后自动与本机比较 29 个作业，并写故障坐标报告。比较规则：T̂、路由、行为位置、搜索轨迹逐位相同；故障选择和后端依赖对象地址，只看均值。最后要出现 `PREP: reference identical`，之后才能开其他 6 个笔记本。
- 其他分片开跑前会检查 prep 是否已完成，没完成就退出。
- 出现 `ALERT`（某个进程超过 1 小时）或 `STOP`，或者 `FAILED`，把输出发回来。

**单元格 3：汇总和打包**

```python
!cd /content/crisp && python e2e/w1_rerun/w1_eir.py report <分片> --out $OUT
!cd /content/crisp && python e2e/w1_rerun/w1_eir.py pack <分片> --out $OUT
```

`report` 打印完成数、单次墙钟、内存峰值和删去条数；`pack` 写 `$OUT/results_w1_<分片>.tar.gz`，内含 sink、日志和内存记录。

## 拷回本地

从 Drive 下载 7 个 `results_w1_<分片>.tar.gz`，放到仓库的 `colab_results/`（已被 git 忽略），然后：

```bash
mkdir -p colab_results/w1_eir && for f in colab_results/results_w1_*.tar.gz; do tar -xzf "$f" -C colab_results/w1_eir; done
```

## 需要知道的事

- **7 个笔记本共用输出目录**：作业名全局唯一，日志和规格文件按作业名存放，互不覆盖。
- **单元格 1 不能省**：这个压缩包的代码在 Linux 上还没跑过。
- **比较口径**：T̂、搜索轨迹不依赖对象地址，必须逐位相同；BusyBarn 后端依赖对象地址，只比均值。
- **Colab 上的耗时不能进论文**：论文要用的耗时以后串行重测。
- **内存**：EIR 单进程约 0.45 GB（W0 记录），7 个并行远低于 Colab 的 51 GB。
