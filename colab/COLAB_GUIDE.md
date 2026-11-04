# Colab 运行指南

这里跑 3 个笔记本，都是第 2 阶段的补充实验。规则与本地第 2 阶段相同：精确撤销 + 含缓存广播的 T̂，每臂 14 次，后端 10 个种子。运行时不碰本地的任何进程和目录。

| 笔记本名 | `--nb` | 任务 | 臂 | 运行次数 | 预计（7 个并行，粗估） |
|---|---|---|---|---|---|
| `CRISP_1a_moe_f4` | `1a` | Qwen3-MoE FFN + F4 | A0 A1 A2 A3h | 56 | 6–12 小时 |
| `CRISP_1b_moe_f5` | `1b` | Qwen3-MoE FFN + F5 | A0 A1 A2 A3h | 56 | 6–12 小时 |
| `CRISP_2_mha_sq` | `2` | GPT decode MHA，序列 512、8192 | A0 A1 A2 A3h | 112 | 4–7 小时 |

耗时是按本地第 2 阶段换算的（8 个 vCPU 通常只有 4 个物理核），第一个笔记本跑完 1–2 个任务后再看实际速度。三个笔记本互不依赖，可以同时开。

## 准备（只做一次）

在本机生成压缩包（已生成过且代码没变就不用重复）：

```bash
python colab/build_bundle.py
```

把 `colab/dist/colab_bundle.tar.gz` 上传到 Google Drive 的 `我的云端硬盘/CRISP_colab/`。

## 每个笔记本

新建笔记本，按上表命名。运行时类型选 CPU + 高 RAM。把下面 4 个单元格依次贴进去，只改第 1 个单元格的 `NB`。

**单元格 1：挂载并解压到本地磁盘**

```python
NB = "1a"   # 第 2 个笔记本改成 "1b"，第 3 个改成 "2"

from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["NB"] = NB
os.environ["OUT"] = f"/content/drive/MyDrive/CRISP_colab/{NB}"
!mkdir -p /content/crisp && tar -xzf /content/drive/MyDrive/CRISP_colab/colab_bundle.tar.gz -C /content/crisp
!pip install -q numpy networkx simanneal tqdm PyYAML matplotlib
!nproc && free -g | head -2 && cat /content/crisp/BUNDLE_INFO.json | head -3
```

**单元格 2：冒烟测试**（首次必跑，约 5–30 分钟）

```python
!cd /content/crisp && python colab/colab_run.py smoke --nb $NB --workers 4
```

最后一行要是 `SMOKE PASSED`，否则不要往下跑，把输出发回来。

**单元格 3：正式运行**（断线后重新执行单元格 1 和这一格即可续跑）

```python
!cd /content/crisp && python colab/colab_run.py run --nb $NB --out $OUT --workers 7 2>&1 | tee -a /content/run_$NB.log | grep -E "\[colab\]|FAILED|done in|not exempt"
```

- 输出目录是软链接到 Drive 的，每次运行完成后自动保存。
- 重启后会先检查已有结果：写到一半的文件被删除重跑，完整的直接跳过。最多损失断线时正在跑的那几次。
- 完整日志在 `$OUT/s2/logs/`。

**单元格 4：跑完后打包**

```python
!cd /content/crisp && python colab/colab_run.py report --nb $NB --out $OUT
!cd /content/crisp && python colab/colab_run.py pack --nb $NB --out $OUT
```

## 拷回本地

从 Drive 下载 `CRISP_colab/<NB>/results_<NB>.tar.gz`，放到仓库的 `colab_results/`（已被 git 忽略），然后：

```bash
mkdir -p colab_results/1a && tar -xzf colab_results/results_1a.tar.gz -C colab_results/1a
```

不要解到 `cpcm_sa/out/`：那是本地第 2 阶段的输出目录。

## 需要知道的事

- **未在 Linux 上试跑过**：本机只能在 Windows 上做冒烟测试，所以 Colab 上的单元格 2 不能省。
- **结果可比性**：模拟出来的延迟与机器速度无关，但后端有地址依赖，Linux 与 Windows 的具体数值可能落在不同的离散结果上。同一任务的 4 个臂都在同一个笔记本内跑，它们之间的比较不受影响。与本地结果跨平台比较时，以噪声范围为准。
- **Colab 上的耗时不能进论文**：资源不固定，论文要用的耗时数据以后串行重测。
- **自检 1 的 4% 超差**：`report` 会列出超差点，不会自动停，发回来由你决定。
- **Colab 时长限制**：长时间无操作或达到时长上限都会断开。断开后重新连接，再执行单元格 1 和 3。
