# C 的 Colab 指南（GPT 同一物理故障，W1）

- **放行与规则：** 用户 2026-11-09 放行（`E2E_HANDOFF.md` 第 5 节）。方案见 `C_PLAN.md`，比较规则与故障坐标见 `e2e\PROGRESS.md` 的“C 第 0 步”。
- **驱动：** `e2e\c_eir.py`。它在内存中把 EIR 接到 prefill 和 decode 的 8 个算子上，不改 `eir\`。

**开跑前已在本机完成：**

- 对 EIR decode 的 12 个已有结果做逐位回归，12/12 全部相同（覆盖起点评估、R0 + L1、L2Gsync 100 步、FULL-A3h 1000 步，以及节点、只坏 PE、DDR 端口、复合故障）。
- 8 个算子的冒烟。
- 本机 prep：16 个起点评估加 304 个 l1，完成 303 个，结果随压缩包分发。
- 失败 1 个：`l1_a3h_pf_mha_MCRandom`（EIR 的 `seed_broadcast` 配对出错）。用户定为修法 (a′)，只在本机生效，见下文。

**约束与分配**（用户 2026-11-09 修订）：

- Colab 最多同时开 5 个 8 vCPU 高内存笔记本，每个跑 7 个并行。
- 耗时按本机已跑完的 C 作业实测：
  - l1 按算子取中位数（304 个）：prefill LN 1,290 s，MHA 1,000 s，投影 332 s，FFN 109 s；decode 66–123 s。
  - prefill MHA 的 FULL-A3h 56 次，中位数 1,644 s（B4 是 1,855 s）；其余 FULL 按 B4 × 0.886。
  - 修复 = l1 × 0.55 / 0.65 / 1.0（30 / 100 / 300 步）。300 步按 prefill MHA 实测 1,005–1,018 s 定，30 和 100 步按 EIR W1 decode MHA 的比例。
- Colab 系数 1.3（范围 1.15–1.5）。EIR W1 中 FULL decode MHA A3h：Colab 275–340 s，本机 240 s；FULL A1：Colab 152 s，本机 103 s。两边没有跑过逐个相同的作业，所以系数是按同类作业比的。
- **本机（14 个并行：`localfull` 7 个 + `localrep` 7 个）：**
  - 16 个 Colab 故障里单次最长的作业：prefill MHA、LN 的 300 步修复，prefill MHA 的 FULL-A3h；
  - 3 个整故障（MCCluster、MCEdge、MCScratch）的全部作业。
  - `localrep` 先跑完后，再开第二个 `localfull` 进程，与第一个共享作业表。作业靠认领文件区分，不会重复。
- **Colab：** 其余 5,110 个作业冻结在 `e2e\c\shards.json`，分成 15 个分片，每片约 4.9 h；5 个一批，共 3 批。

**运行量（估，实测折算）：**

| 部分 | 作业数 | 本机进程小时 | 墙钟 |
|---|---|---|---|
| 本机，余下（15:20 起） | 约 1,340 | 约 230 | 14 个并行，约 16.5 h |
| Colab，15 个分片 | 5,110 | 约 395（Colab 约 513） | 检查约 0.5 h + 3 批 × 约 4.9 h，约 15–15.5 h |

- 两边都约在明早 7–8 点跑完：Colab 若 16:40 开第 1 批，约 07:30–08:30 结束；本机约 07:30–08:00。
- 若 Colab 先跑完，就从本机的 `localfull` 末尾挪作业给 Colab，作为第 4 批；若本机先跑完，本机从第 3 批末尾往前接分片。挪动都记在下表和 PROGRESS 里。
- **修法 (a′)，已转本机：**
  - 失败的 prep `l1_a3h_pf_mha_MCRandom` 和读它的 15 个作业（prefill MHA、A3h、MCRandom 的 L2Gsync 30、100、300 步各 5 个种子）都在本机分片 `localfix` 中运行，不在 Colab 分片里。
  - 其中原分给 Colab 的 10 个（30、100 步）也挪到本机。
  - Colab 压缩包（git 0b60b2e）不含这个修法，也不需要含。

## 批次（每批 5 个笔记本；一批全部跑完再开下一批）

| 批 | 笔记本（分片名） | 每个的预计墙钟 |
|---|---|---|
| 1 | **先只开第 1 个 `check+c01`**（先做检查，再跑分片 `c01`）；它打印出 `CHECK PASSED` 后再开其余 4 个：`c02`、`c03`、`c04`、`c05` | 检查约 0.5 h；分片约 4.9 h（4.3–5.6 h） |
| 2 | `c06`、`c07`、`c08`、`c09`、`c10` | 约 4.9 h |
| 3 | `c11`、`c12`、`c13`、`c14`、`c15` | 约 4.9 h |

- 每个分片混有多个故障和算子：作业按“故障 × 算子 × 种类 × 档”成组，按最长处理时间优先装箱，使各片负载相等。
- 分片内容见 `shards.json`，运行时由驱动读出，不用手填。

**两边之间挪动的分片：**

| 分片 | 从 | 到 | 时间 |
|---|---|---|---|
| `a3h_pf_mha_MCRandom_L2Gsync_s30/s100_*`（10 个，原第 4 批计划，未进 shards.json） | Colab | 本机 `localfix` | 2026-11-09 17:00 |

## 准备（只做一次）

把 `e2e\c\dist\c_bundle.tar.gz` 上传到 Google Drive 的 `我的云端硬盘/CRISP_colab/`。所有笔记本共用这个压缩包和同一个输出目录 `CRISP_colab/c`。

压缩包里有：

- 代码；
- 起点映射；
- 故障表；
- 本机 prep 的结果（`c_prep/prep`，修复作业要读它）；
- 回归所需的 EIR 已有结果。

生成时已在干净目录里自查：`python -I`，不带 PYTHONPATH，跑 `probe` 和 `check`。2026-11-09 16:22 的版本（git 0b60b2e，105.9 MB）结果：`PROBE PASSED`、`REGRESSION PASSED`、32/32 与本机逐位相同、`CHECK PASSED`。

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

第 1 批的第 1 个笔记本（`check+c01`），先检查：

```python
!cd /content/crisp && python e2e/c_eir.py check --out $OUT --workers 7 2>&1 | tee -a /content/c_check.log | grep "\[c\]\|PASSED\|FAILED"
```

- 先跑 EIR decode 的 12 个回归作业。
- 再重算 16 个起点评估和 PF4 的 16 个 l1，与本机逐位比较。
- 最后要出现 `REGRESSION PASSED` 和 `CHECK PASSED`。出现后再开第 1 批的其余 4 个笔记本，这个笔记本接着用下面的单元格跑分片 `c01`。

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
- **单进程停线：** Colab 上 prefill LN 的 FULL-A3h 估约 1,330 × 1.3 ≈ 1,730 s，在 3 小时报警线以内。非 prefill MHA/LN 作业的 1 小时停线：最长是 prefill 投影的 FULL-A3h，估约 680 s。
