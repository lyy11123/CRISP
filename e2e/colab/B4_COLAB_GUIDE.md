# B4 Colab 验证指南

这个笔记本只做验证，不跑 B4 的正式结果：环境探针，B4 清单中的一小部分与本机逐位比较，以及并行数扫描。正式结果仍在本机跑；验证通过后由用户决定是否把剩余部分分到 Colab。规则与本机 B4 相同（`e2e\PROGRESS.md` 的“B4 第 0 步”）：`CPCM_SA_STAGE=s2`、1000 步、后端种子 1–10、线程数 1。

| 笔记本名 | 内容 | 运行次数 | 预计（粗估） |
|---|---|---|---|
| `CRISP_b4_check` | 探针；decode LN、投影 × A1/A3h × 2 个配置，W1 prefill MHA A1 g0 l0；并行数扫描 4、7、8 | 9 + 19 | 2–3 小时 |

本机参考：decode LN、投影单次约 1–2 分钟，W1 prefill MHA A1 约 6.5 分钟（4 个并行）。Colab 的 8 个 vCPU 通常只有 4 个物理核，所以 7、8 个并行时单次会更慢，这正是扫描要量的。

## 准备（只做一次）

代码冻结、本机已跑完 `colab_b4.py` 里 `SUBSET` 列出的 9 次后，在本机生成压缩包：

```bash
python e2e/colab/build_b4_bundle.py
```

它会写出 `e2e/colab/local_ref.json`（本机的 T̂、SA 轨迹摘要和后端结果）和探针快照，再打包到 `e2e/colab/dist/b4_bundle.tar.gz`。输出最后一行如果有 `UNCOMMITTED CODE`，先提交再打包。把压缩包上传到 Google Drive 的 `我的云端硬盘/CRISP_colab/`。

## 笔记本

新建笔记本 `CRISP_b4_check`，运行时类型选 CPU + 高 RAM。依次贴入下面的单元格。

**单元格 1：挂载、解压、环境探针**

```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["OUT"] = "/content/drive/MyDrive/CRISP_colab/b4_check"
!mkdir -p /content/crisp && tar -xzf /content/drive/MyDrive/CRISP_colab/b4_bundle.tar.gz -C /content/crisp
!pip install -q numpy networkx simanneal tqdm PyYAML matplotlib
!nproc && free -g | head -2 && cat /content/crisp/BUNDLE_INFO.json | head -4
!cd /content/crisp && python e2e/colab/colab_b4.py probe --out $OUT
```

最后一行要是 `PROBE PASSED`（一次 W1 prefill MHA 的确定性 T̂ 和种子 1 的随机平局 T̂ 都与本机逐位相同），否则不要往下跑，把输出发回来。输出里还有 nproc、内存、Python 与 numpy、networkx、simanneal 的版本，以及这次 T̂ 的编译和前向耗时（本机的值并列）。

**单元格 2：冒烟测试**（约 1–3 分钟）

```python
!cd /content/crisp && python e2e/colab/colab_b4.py smoke
```

一次 decode LN A3h 的 15 步运行、2 个后端种子，放在本地磁盘，不进结果。最后一行要是 `SMOKE PASSED`。失败时会打印子进程日志的最后 80 行，并保留 `/content/crisp/e2e/out`（不删），把输出发回来；之后要先 `!rm -rf /content/crisp/e2e/out` 才能跑单元格 3、4。

**单元格 3：验证子集**（断线后重新执行单元格 1 和这一格即可续跑，已完成的跳过）

```python
!cd /content/crisp && python e2e/colab/colab_b4.py run --out $OUT --workers 4 2>&1 | tee -a /content/b4_run.log | grep "\[colab\]"
```

9 次运行，然后逐个与本机比较：T̂（`T_pre`、`T_bc`、`T_sel`）、SA 轨迹（每步的能量与是否接受）、选中的步数，都要逐位相同；后端只按 10 种子均值比较（后端随对象地址变化，换平台可能落在不同的离散结果上）。最后一行是 `RUN: all T-hat identical` 才算通过。

**单元格 4：并行数扫描**（约 1–1.5 小时）

```python
!cd /content/crisp && python e2e/colab/colab_b4.py scan --out $OUT 2>&1 | tee -a /content/b4_scan.log | grep "\[colab\]"
```

依次以 4、7、8 个并行同时跑同一个 decode 投影 A3h 运行（每份各自的目录，互不复用），记录单次墙钟的中位与最大、总墙钟、运行中最低可用内存、单进程内存峰值，并核对每一份的 T̂ 与 SA 轨迹都与本机相同。

**单元格 5：汇总和打包**

```python
!cd /content/crisp && python e2e/colab/colab_b4.py report --out $OUT
!cd /content/crisp && python e2e/colab/colab_b4.py pack --out $OUT
```

`report` 打印探针、子集比较和扫描的要点，并写 `$OUT/b4_colab_report.json`；`pack` 写 `$OUT/results_b4.tar.gz`。

## 拷回本地

从 Drive 下载 `CRISP_colab/b4_check/results_b4.tar.gz`，放到仓库的 `colab_results/`（已被 git 忽略），然后：

```bash
mkdir -p colab_results/b4_check && tar -xzf colab_results/results_b4.tar.gz -C colab_results/b4_check
```

不要解到 `e2e/out/`：那是本机 B4 的输出目录。

## 需要知道的事

- **单元格 1、2 不能省**：CPCM 和 SA 已在 Colab 上跑过 `cpcm_sa` 的补充实验，在 Linux 上验证过；但 e2e 的代码（`e2e_pilot.py`、W1 过滤、LN/投影模板生成的脚本、`e2e_pilot.py` 与 `colab_b4.py` 的 Linux 内存读数分支）和这个新的压缩包都还没有在 Linux 上跑过。
- **比较口径**：T̂ 与 SA 轨迹不依赖对象地址，必须逐位相同；后端依赖对象地址（E0 起已知），只比均值，差异以本机后端的种子极差为参照。
- **Colab 上的耗时不能进论文**：资源不固定，论文要用的耗时以后串行重测；这里的墙钟只用来定 Colab 上的并行数。
- **Colab 时长限制**：断开后重新连接，执行单元格 1，再执行断开时的那一格。单元格 4 会从头重跑当前并行档（每档的目录在开始时清空）。
