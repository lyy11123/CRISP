# D 远端运行指南（种子 4–10 的补充作业）

适用于任何 Linux（或 Windows）机器：用户租的服务器，或者 Colab。包由本机 `python e2e/d_remote.py build ...` 生成，放在 `e2e\d\dist\`。

**包里有什么：**

- 代码；
- C 的 artifact 工作副本（只含 GPT prefill、decode 的 py 与 cfg）；
- 导出的事件图；
- 作业清单 `e2e/d_jobs.json`（路径相对于包的根目录）；
- 本机的参照结果 `e2e/d_ref.json`。

不需要本仓库，也不需要 GPU。

## 1. 环境

- Python 3.13（本机 3.13.16；Colab 上 C 用的是 3.13.18），包：numpy、networkx、simanneal（与 C 相同）。
- **内存：**
  - 单进程峰值：本机实测 1.0–1.7 GB，连续 8 条链 2.2 GB。
  - 并行数 p 按“空闲内存 ÷ 2 GB”取，8 核机器取 7。
  - 默认停线：单进程 3 GB、系统空闲 3 GB；内存充足的机器可放宽到单进程 5 GB（`--peak-gb 5`）。
- 每个子进程都设 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`（代码自己设）。

## 2. 步骤（任何机器）

```bash
mkdir -p ~/d && cd ~/d && tar xzf /path/to/<bundle>.tar.gz
```

```bash
cd ~/d && python3 -I e2e/d_remote.py check --out ~/d/out_check
```

`check` 在这台机器上重跑包里指定的 2 个参照作业，和本机结果逐位比较：T̂、每次运行的总延迟、各流完成时刻。

- 最后一行为 `[remote] check: identical` 才继续。
- 不一致时，把 `~/d/out_check` 打包发回。

```bash
cd ~/d && nohup python3 -I e2e/d_remote.py run --out ~/d/out --workers 7 --shard 0 --nshards 1 > run.log 2>&1 &
```

- **多台机器：** 第 i 台用 `--shard i --nshards M`（i 从 0 开始），每台跑清单的 1/M。
- **中断后：** 用同样的命令重跑，已完成的作业（sink 存在）会跳过。
- **结束时** `run.log` 最后一行为 `[remote] done in ... h; failed 0: []`。
- **停线：** 内存超线的作业会被杀掉，并在 `run.log` 中记为 `killed`；单个作业超过 1 小时只在日志里看得到。遇到这两种情况都发回日志。

**取回结果：**

```bash
cd ~/d && tar czf results_<shard>.tar.gz out/sims out/logs out/runs_*.json run.log
```

把 `results_<shard>.tar.gz` 放到本机，交给 Claude 解到 `e2e\d\out\prod\sims\`。

## 3. Colab

1. 新建笔记本：运行时类型选 CPU、高内存（8 vCPU、约 51 GB）。
2. 挂载 Google Drive，把包上传到 Drive 的 `MyDrive/crisp_d/`：

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. 解包与检查：

```python
!mkdir -p /content/d && cd /content/d && tar xzf /content/drive/MyDrive/crisp_d/<bundle>.tar.gz
!cd /content/d && python3 -I e2e/d_remote.py check --out /content/d/out_check
```

4. 运行（第 i 个笔记本，共 M 个）。输出直接写到 Drive，断线重连后用同一条命令续跑：

```python
!cd /content/d && python3 -I e2e/d_remote.py run --out /content/drive/MyDrive/crisp_d/out_<i> --workers 7 --shard <i> --nshards <M> > /content/drive/MyDrive/crisp_d/run_<i>.log 2>&1
```

5. 结束后把 Drive 上的 `crisp_d/out_<i>` 与 `run_<i>.log` 下载到本机。

## 4. 预计墙钟

- **总量：** 由 `build` 打印，即作业数与本机进程小时；本机单次实测：按跳 pf_mha 单独 187–199 s、重叠 130–260 s。
- **墙钟** ≈ 本机进程小时 × s ÷ (M × p)。
  - s 为单次系数：Colab 1.15–1.5，租用机器按 CPU 单核性能，与本机相当时约 1。
  - M 为机器数，p 为每台并行数。
