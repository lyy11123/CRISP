# C 的 Colab 指南（GPT 同一物理故障，W1）

- **放行与规则：** 用户 2026-11-09 放行（`E2E_HANDOFF.md` 第 5 节）。方案见 `C_PLAN.md`，比较规则与故障坐标见 `e2e\PROGRESS.md` 的“C 第 0 步”。
- **驱动：** `e2e\c_eir.py`。它在内存中把 EIR 接到 prefill 和 decode 的 8 个算子上，不改 `eir\`。

**开跑前已在本机完成：**

- 对 EIR decode 的 12 个已有结果做逐位回归，12/12 全部相同（覆盖起点评估、R0 + L1、L2Gsync 100 步、FULL-A3h 1000 步，以及节点、只坏 PE、DDR 端口、复合故障）。
- 8 个算子的冒烟。
- 本机 prep：16 个起点评估加 304 个 l1，完成 303 个，结果随压缩包分发。
- 失败 1 个：`l1_a3h_pf_mha_MCRandom`（EIR 的 `seed_broadcast` 配对出错，修法待用户定）。读它的作业暂缓，驱动会自动跳过并打印 `held back`：第 4 批 `h1_MF8+MCRandom` 中 prefill MHA 的 A3h 30、100 步修复（10 个），以及本机的 5 个 300 步修复。修法定了以后单独补跑。

**约束与分配**（用户 2026-11-09）：

- Colab 最多同时开 6 个高内存笔记本，每个 8 vCPU，跑 7 个并行。
- EIR 的 W1 补跑跑完后才开 C 的第 1 批。
- 本机从现在起同时跑（14 个并行：`localfull` 7 个 + `localrep` 7 个，两者合起来就是 `local`），分到两部分：
  - 全部 16 个 Colab 故障里单次最长的作业：prefill MHA、LN 的 300 步修复，以及 prefill MHA 的 FULL-A3h；
  - 3 个整故障（MCCluster、MCEdge、MCScratch）的全部作业。
- 本机另外先跑完 prep（16 个起点评估加 304 个 l1），它的结果放进压缩包。

**运行量（估）：**

| 部分 | 作业数 | 本机进程小时 | 墙钟 |
|---|---|---|---|
| 合计 | 6,856 | 约 1,040 | |
| 本机，含 prep | 1,736 | 约 444 | 约 26–31 h |
| Colab，24 个分片 | 5,120 | 约 600 | 4 批 × 4.3–5.4 h，加检查约 0.5 h，约 18–22 h |

- 估算用 B4 的单次墙钟折算；l1 已按本机实测（prefill LN 1,060–1,514 s）修正。
- Colab 按本机的 1.25–1.45 倍，未实测。
- 从现在算，Colab 约 4.5 h 后开始，两边都约在 26–31 h 内跑完。
- 若 Colab 先跑完，本机还没开始的整故障（先 MCScratch，再 MCEdge）改到 Colab，作为第 5 批；若本机先跑完，本机从第 4 批末尾往前接 Colab 的分片。挪动都记在下面的表里和 PROGRESS 里。

## 批次（每批 6 个笔记本；一批全部跑完再开下一批）

| 批 | 笔记本（分片名） | 每个的预计墙钟 |
|---|---|---|
| 1 | **先只开第 1 个 `check+h1_PF1+PF2`**（先做检查，再跑分片 `h1_PF1+PF2`）；它打印出 `CHECK PASSED` 后再开其余 5 个：`PF1_h2`、`PF2_h2`、`h1_PF4+PF5`、`PF4_h2`、`PF5_h2` | 检查约 0.5 h；分片 4.3–5.4 h |
| 2 | `h1_PF6+PF7`、`PF6_h2`、`PF7_h2`、`h1_PF8+PCRandom`、`PF8_h2`、`PCRandom_h2` | 4.3–5.4 h |
| 3 | `h1_PCCluster+PCEdge`、`PCCluster_h2`、`PCEdge_h2`、`h1_PCScratch+MF1`、`PCScratch_h2`、`MF1_h2` | 4.3–5.4 h |
| 4 | `h1_MF5+MF6`、`MF5_h2`、`MF6_h2`、`h1_MF8+MCRandom`、`MF8_h2`、`MCRandom_h2` | 4.3–5.4 h |
| 本机 | 3 个整故障 MCCluster、MCEdge、MCScratch；以及上面各分片中 prefill MHA、LN 的 300 步修复和 prefill MHA 的 FULL-A3h（这些作业不在 Colab 分片里） | |

**分片内容：**

| 分片 | 内容 |
|---|---|
| `<故障>_h2` | 该故障下 prefill LN 的 30、100 步修复与两种 FULL；prefill 投影、FFN 的全部作业 |
| `h1_<故障1>+<故障2>` | 两个故障下 prefill MHA 的 30、100 步修复与 FULL-BusyBarn；4 个 decode 算子的全部作业 |

故障编号：

- **PF*、PC*：** 集合 (a)，按物理位置固定，是主集合。
- **MF*、MC*：** 集合 (b)，按 decode MHA 的 W1 A3h 起点选。

**两边之间挪动的分片：**

| 分片 | 从 | 到 | 时间 |
|---|---|---|---|
| （暂无） | | | |

## 准备（只做一次）

把 `e2e\c\dist\c_bundle.tar.gz` 上传到 Google Drive 的 `我的云端硬盘/CRISP_colab/`。所有笔记本共用这个压缩包和同一个输出目录 `CRISP_colab/c`。

压缩包里有：

- 代码；
- 起点映射；
- 故障表；
- 本机 prep 的结果（`c_prep/prep`，修复作业要读它）；
- 回归所需的 EIR 已有结果。

生成时已在干净目录里自查：`python -I`，不带 PYTHONPATH，跑 `probe` 和 `check`。

## 每个笔记本的单元格

运行时类型选 CPU + 高 RAM。`<分片>` 换成上表的分片名。

**单元格 1：挂载、解压、放入 prep、探针**（约 2 分钟）

```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["OUT"] = "/content/drive/MyDrive/CRISP_colab/c"
!mkdir -p /content/crisp && tar -xzf /content/drive/MyDrive/CRISP_colab/c_bundle.tar.gz -C /content/crisp
!pip install -q numpy networkx simanneal tqdm PyYAML matplotlib
!mkdir -p $OUT && cp -rn /content/crisp/c_prep/prep $OUT/
!nproc && free -g | head -2 && cat /content/crisp/BUNDLE_INFO.json | head -6
!cd /content/crisp && python e2e/c_eir.py probe --out $OUT
```

最后一行要是 `PROBE PASSED`：decode MHA 的 A3h 起点评估一次，要与本机逐位相同。否则不要往下跑，把输出发回来。

**单元格 2：运行**（断线后重新执行单元格 1 和这一格即可续跑，已完成的作业会跳过）

第 1 批的第 1 个笔记本（`check+h1_PF1+PF2`），先检查：

```python
!cd /content/crisp && python e2e/c_eir.py check --out $OUT --workers 7 2>&1 | tee -a /content/c_check.log | grep "\[c\]\|PASSED\|FAILED"
```

- 先跑 EIR decode 的 12 个回归作业。
- 再重算 16 个起点评估和 PF4 的 16 个 l1，与本机逐位比较。
- 最后要出现 `REGRESSION PASSED` 和 `CHECK PASSED`。出现后再开第 1 批的其余 5 个笔记本，这个笔记本接着用下面的单元格跑分片 `h1_PF1+PF2`。

每个分片：

```python
!cd /content/crisp && python e2e/c_eir.py run <分片> --out $OUT --workers 7 2>&1 | tee -a /content/c_run.log | grep "\[c\]\|ALERT\|STOP"
```

- 出现 `STOP` 或 `FAILED` 时，把输出发回来。`STOP` 的原因：某个作业失败、非 prefill MHA/LN 的作业超过 1 小时、内存不足 3 GB。
- 出现 `ALERT`（prefill MHA 或 LN 的作业超过 3 小时）只是报警，可以继续，但也请告诉我。

**单元格 3：汇总和打包**

```python
!cd /content/crisp && python e2e/c_eir.py report <分片> --out $OUT
!cd /content/crisp && python e2e/c_eir.py pack <分片> --out $OUT
```

`pack` 写 `$OUT/results_c_<分片>.tar.gz`，内含结果、日志和内存记录。检查的结果在 `$OUT/_check`，不用打包。

## 拷回本地

从 Drive 下载各个 `results_c_<分片>.tar.gz`，放到仓库的 `colab_results/`（已被 git 忽略），然后：

```bash
mkdir -p colab_results/c && for f in colab_results/results_c_*.tar.gz; do tar -xzf "$f" -C colab_results/c; done
```

## 需要知道的事

- **共用输出目录：** 作业名全局唯一，日志、规格文件、运行记录都按作业名或分片名存放，互不覆盖。
- **比较口径：** T̂、路由、行为位置、搜索轨迹与对象地址无关，必须逐位相同；BusyBarn 后端依赖对象地址，只比均值。
- **Colab 上的耗时不能进论文。**
- **内存：** 单进程峰值，decode 约 0.45 GB，prefill MHA 约 0.8 GB（B4 记录）；7 个并行远低于 51 GB。
