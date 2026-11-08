# W1 补跑 Colab 指南（EIR、DSTU 中涉及 MHA 的部分）

**状态：草稿，未放行。** 驱动脚本 `e2e\w1_rerun\w1_rerun.py` 和打包脚本 `build_w1_bundle.py` 还没写；放行并定下 `W1_RERUN_INVENTORY.md` 第 4 节的三个问题后再写、再在本机冒烟、再打包。B4 的 Colab 验证（`e2e\colab\B4_COLAB_GUIDE.md`）通过前不打包正式分片。

运行清单、次数和耗时见 `W1_RERUN_INVENTORY.md`。规则照原实验：同映射来源、同故障、同种子、同停止规则（EIR ε = 2%），唯一差别是工作负载 W1。比较规则开跑前写进 `e2e\PROGRESS.md`。不改 `eir\`、`dstu\`、`e4_overlap\` 的源码：W1 过滤在内存中接入（EIR：包装 `add_communication.add_broadcast` 与 `BcCache.graph`，同 B4；DSTU：包装 `e4_overlap.load_stream`，对捕获的事件删去 72 条最终广播），提速版按跳后端（`e2e\fast_hop.py`）在内存中替换 `event_driver_e4b`。

| 笔记本名 | 内容 | 运行次数 | 预计（粗估，7 个并行） |
|---|---|---|---|
| `CRISP_w1_prep` | 探针；W1 起点、pre、L1；FFN 零删除自检；DSTU 变体重选与流过滤；2 个运行与本机逐位比较 | 约 20 + 变体评估 | 0.5–1 h |
| `CRISP_w1_eir_main_a3h` | main A3h 起点修复 180、对照 15 | 195 | 2.1–3.1 h |
| `CRISP_w1_eir_main_e3` | main e3 起点修复 240、对照 15 | 255 | 2.9–4.4 h |
| `CRISP_w1_eir_full` | FULL-A3h 112、FULL-BusyBarn 98 | 210 | 2.9–4.4 h |
| `CRISP_w1_eir_next` | next 修复 205、检查点 30 | 235 | 2.2–3.4 h |
| `CRISP_w1_dstu` | DSTU 全流水线（设定 1–3、阶段 2、2C、块开销、全链路设定 3、收尾） | 5,000 个以上进程（已知：设定 1–3 1,580，阶段 2 3,423，块开销 11.1 432） | 5.6–8.4 h，分 2–3 次连接 |

`CRISP_w1_prep` 先跑完，其余 5 个可同时开。并行数按 B4 并行数扫描的结果定（这里先按 7 估）。

## 准备（只做一次）

放行后，代码冻结、本机冒烟通过后，在本机生成压缩包：

```bash
python e2e/w1_rerun/build_w1_bundle.py
```

它写出 `e2e/w1_rerun/local_ref.json`（本机 2 个参照运行的 T̂、SA 轨迹摘要与后端结果），打包到 `e2e/w1_rerun/dist/w1_bundle.tar.gz`。输出最后一行有 `UNCOMMITTED CODE` 时先提交再打包。上传到 Google Drive 的 `我的云端硬盘/CRISP_colab/`。

## 每个笔记本的单元格

运行时类型选 CPU + 高 RAM。`<分片>` 是 `prep`、`eir_main_a3h`、`eir_main_e3`、`eir_full`、`eir_next`、`dstu` 之一。

**单元格 1：挂载、解压、环境探针**

```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["OUT"] = "/content/drive/MyDrive/CRISP_colab/w1"
!mkdir -p /content/crisp && tar -xzf /content/drive/MyDrive/CRISP_colab/w1_bundle.tar.gz -C /content/crisp
!pip install -q numpy networkx simanneal tqdm PyYAML matplotlib
!nproc && free -g | head -2 && cat /content/crisp/BUNDLE_INFO.json | head -4
!cd /content/crisp && python e2e/w1_rerun/w1_rerun.py probe --out $OUT
```

最后一行要是 `PROBE PASSED`：一个 EIR MHA 运行的 W1 删去条数（72）、FFN 删去条数（0）、确定性 T̂，和一个 DSTU 流的过滤后事件数、C4 T̂，都与本机逐位相同。否则不要往下跑，把输出发回来。

**单元格 2：分片运行**（断线后重新执行单元格 1 和这一格即可续跑，已完成的跳过）

```python
!cd /content/crisp && python e2e/w1_rerun/w1_rerun.py run <分片> --out $OUT --workers 7 2>&1 | tee -a /content/w1_run.log | grep "\[w1\]"
```

`prep` 分片还会与本机比较 2 个参照运行（T̂、SA 轨迹逐位相同；后端只比 10 种子均值，后端依赖对象地址），最后一行是 `PREP: reference identical` 才能开其他笔记本。

**单元格 3：汇总和打包**

```python
!cd /content/crisp && python e2e/w1_rerun/w1_rerun.py report <分片> --out $OUT
!cd /content/crisp && python e2e/w1_rerun/w1_rerun.py pack <分片> --out $OUT
```

`report` 打印失败数、单次墙钟、内存峰值和自检；`pack` 写 `$OUT/results_w1_<分片>.tar.gz`。

## 拷回本地

从 Drive 下载各分片的 `results_w1_<分片>.tar.gz`，放到仓库的 `colab_results/`（已被 git 忽略），然后：

```bash
mkdir -p colab_results/w1 && for f in colab_results/results_w1_*.tar.gz; do tar -xzf "$f" -C colab_results/w1; done
```

本机用 `w1_rerun.py compare` 按原实验的报告脚本生成 W1 的表，并与 W0 逐表对照。

## 需要知道的事

- **单元格 1 不能省**：e2e 的代码与这个新压缩包都还没有在 Linux 上跑过。
- **比较口径**：T̂ 与 SA 轨迹不依赖对象地址，必须逐位相同；BusyBarn 后端与按跳后端都依赖对象地址，只比均值。
- **Colab 上的耗时不能进论文**：论文要用的耗时以后串行重测。
- **DSTU 是一条流水线**：后面的实验读前面的结果，按原顺序依次跑，不能拆到多个笔记本；断线后从未完成的那一步续跑。
- **内存**：W0 下 EIR 单进程约 0.45 GB；DSTU 的单进程峰值在 `prep` 分片里测，超过 3 GB 停下。
