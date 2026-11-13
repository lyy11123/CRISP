# F/G 服务器运行记录

**当前状态：2026-11-13 16:53:45，6813/6813全部完成。** 用户已关机，完整COS归档已下载到本机 `e2e/fg/dist/FG_20261113_complete.tar.gz`，解包到 `e2e/fg/out/main/` 并核对实验文件摘要。最终状态查看本机 `finish.batch.json`；数据与复算入口见 [fg/README.md](fg/README.md)。下面保留各次启动、接续和调度记录。

2026-11-13（Codex）。服务器 `ubuntu@119.45.93.240`，工作目录 `/home/ubuntu/crisp_fg_20261113`；Python 使用 `/home/ubuntu/crisp-venv/bin/python`。原实例重新开机，既有 SSH 主机标识与环境匹配。

**10:29接续记录（已结束）：**主控制器PID2161783，`pipeline.log`、`e2e/fg/out/main/pipeline.status.json`；放行进程PID2161782，`release_pipeline.log`、`release.pipeline.json`。运行清单为`e2e/fg/out/pipeline_jobs.json`，实际命令和代码摘要在`pipeline.start.json`。合计上限128，系统余量3GiB，未来增长预留系数0.25。

10:20实查6138/6813完成，只有G的3×3增长切分未齐；20个长搜索运行，CPU约16%、可用187GiB。原清单让故障选择等待全部224个A1/A3h健康候选，但实际只读取14个decode MHA A3h候选。`fg_pipeline.py`将该作业的外层调度依赖改为这14项，6813个作业的spec与身份全部不变；四个已完成配置的选优结果逐项一致。这样112个FULL搜索可在其余prefill搜索结束前开始。原始清单和结果保留，改动提交`c123b47`。

10:29移交时18个旧搜索继续运行（此前另两项已正常结束）；仅结束无工作进程的主控制器2601367及其放行进程2601366。旧owner241176/2429409继续收尾，由既有清理进程处理。三分钟COS备份沿用；新运行清单、源码、启动记录与状态均在原上传范围。

```bash
cd ~/crisp_fg_20261113
head -n 22 e2e/fg/out/main/pipeline.status.json
tail -n 3 pipeline.log
tail -n 2 release_pipeline.log
free -h
```

**08:39移交记录（已由10:29入口替代）：**主控制器PID2601367，`main.log`、`e2e/fg/out/main/main.status.json`；放行进程PID2601366，`release_main.log`、`release.main.json`。用户要求系统余量3GiB；当前将运行中预计增长的25%计为预留，首次启动仍按同类实测峰值扣额度。每轮读取`e2e/fg/out/main/scheduler_policy.json`，原子更新`running_growth_fraction`即可调整，无需再次移交。08:40实查128进程、CPU100%；08:46为127进程、CPU99%、可用46.49GiB。

```bash
cd ~/crisp_fg_20261113
head -n 22 e2e/fg/out/main/main.status.json
tail -n 3 main.log
tail -n 2 release_main.log
free -h
```

当前清单及状态见`main.start.json`、`e2e/fg/out/policy_owned_jobs.json`、`policy_owners.status.json`。移交保留59个在跑任务，旧控制器241176/2429409均不再派发；清理进程2601368在它们无工作进程并收齐59项receipt后结束旧空闲控制器。旧release文件不能改回all=true。

下面保留前次运行记录。

**08:27移交记录（已被上面的08:39入口替代）：**主控制器PID2429409，`continuation.log`、`e2e/fg/out/main/continuation.status.json`；放行进程PID2429408，`release_next.log`、`release.next.json`。旧PID241176只照看已运行的长任务，新的状态统计已包含这些任务。查看命令：

```bash
cd ~/crisp_fg_20261113
head -n 18 e2e/fg/out/main/continuation.status.json
tail -n 3 continuation.log
tail -n 2 release_next.log
free -h
```

内存估计已按配置/算子/方法/初始化细分；同类有实测时采用其峰值，系统预留仍为16GiB，合计上限128。08:27移交保留53个实际运行进程；新队列从原结果目录续跑，修好的MF6任务亦由新队列重试。旧放行进程已停止，旧release只含完成项及在跑项，不能改回all=true。清理进程PID2429410会在旧控制器无工作进程且54项移交记录收齐后结束旧空闲控制器。

三分钟COS上传已获用户授权，于08:22启用：`cos://crisp/crisp/FG_20261113`，日志`~/crisp_fg_cos_backup.log`。首次上传正在补齐已有文件；已有上传成功日志，且COS列表确认`meta/BUNDLE.full.json`存在。旧D定时任务保留。

## 小批

北京时间 2026-11-13 00:07:02 启动，PID `88017`。源码提交 `35711b9`，168 个健康搜索，每个 1000 步、后端种子 1–10、随机平局 CPCM 种子 1–10；另保存 56 个 Gemini 结束状态，相同状态复用评估。小批计入正式清单。

包 `FG_pilot_35711b9.tar.gz`：1,322,182 字节，SHA-256：

```text
7f04fa1e3a8d561f385dde5c8343765fb12cfda7ea96179553cdad2eba3d2eac
```

上传后核对 2392 个包内文件。完整源文件清单和逐文件摘要保存在服务器 `BUNDLE.json`；实际启动记录为 `pilot.start.json`。

实际启动命令：

```bash
cd ~/crisp_fg_20261113
nohup ~/crisp-venv/bin/python -u -B e2e/fg_batch.py \
  --manifest e2e/fg/manifest/pilot_jobs.json \
  --out e2e/fg/out/main --workers 128 --reserve-gib 16 --name pilot \
  > pilot.log 2>&1 < /dev/null &
echo $! > pilot.pid
```

调度每次按剩余内存和任务实测峰值安排作业，先排预计较长的任务。初始内存数据来自本机冒烟；云端运行持续记录实际峰值。每个子进程只执行一个任务，NumPy 底层线程均为 1。已观察到 128 个活跃子进程。

## 网页终端查看

```bash
cd ~/crisp_fg_20261113
tail -n 8 pilot.log
head -n 20 e2e/fg/out/main/pilot.status.json
ps -p "$(cat pilot.pid)" -o pid,etime,pcpu,rss,args
ps --ppid "$(cat pilot.pid)" -o pid= | wc -l
free -h
```

`completed` 统计主候选任务，目标 168；`failed` 为空表示没有失败任务。结果目录含 Gemini 终态，全部结束后共 224 份结果记录。小批结束时写 `e2e/fg/out/main/pilot.batch.json`。

核对命令：

```bash
~/crisp-venv/bin/python -B e2e/fg_check.py \
  --manifest e2e/fg/manifest/pilot.json --out e2e/fg/out/main \
  --report e2e/fg/out/pilot_check.json \
  --full-manifest e2e/fg/manifest/search.json
```

运行中查看已完成项时加 `--allow-incomplete`；最终核对使用上面的完整命令。

## 后续批次与恢复

用户已授权在小批通过、汇报预计耗时后直接继续正式运行。完整任务定义见 `fg/manifest/all_jobs.json`，共 6813 个调度作业，包括 3024 次健康/FULL/带宽搜索、最多 1920 个 EIR 独立档位、609 次联合，以及准备、停止判断和单跑。EIR 提前停止后的档位写复用记录，实际搜索量随恢复程度减少。

全量队列在小批长任务收尾期间接入，`--external-manifest` / `--external-status` 让两个控制器共用 128 个进程名额；小批保留自己的任务，全量读取其完成记录，避免重复启动。小批与全量共用 `e2e/fg/out/main`。

`fg_release.py` 每 30 秒核对已保存的小批结果，逐类开放相同配置/算子/方法的十四候选；FULL 使用对应健康 A3h 小批及已确定物理故障。正式候选选齐后执行代表修复和联合，核对停止规则、配对指标与发放审计，再开放该配置其余任务。完整任务清单始终为 6813 项，不改变种子或预算。

调度优先凑齐 WATOS、3×3 增长切分，再处理其他配置；已经可执行的修复/联合准备优先，同级按预计耗时排序。任务的 `priority`、`estimated_seconds` 只影响排队，不改变实验定义与续跑身份。运行清单由 `fg_estimate.py --updated-jobs` 生成，源清单保持原样。

恢复需保留 `results/`、`raw/`、`snapshots/`、`exports/`、`selection/`、`faults/`、`joint_cache/`，以及 `specs/`、`receipts/`、日志、清单和开放列表；这些文件共同记录候选与后续任务的输入。新的正式启动命令、PID 和源码版本在实际启动后追加。

完整输出当前保存在服务器磁盘，本机已开始分批接收结果记录。原D定时上传保留；F/G三分钟备份已于2026-11-13 08:22获准启用，见本页最新记录。

## 正式队列已启动

北京时间 **2026-11-13 00:39:52**，代码提交 `b2a1543`。全量控制器 PID **241176**，逐类核对进程 PID **241175**。完整清单 6813 项；启动前小批已完成 155/168，2515 项已达到开始条件，其余随前置结果核对通过自动接入。

包 `FG_full_20261113.tar.gz`，1,614,071 字节；上传核对2415个文件。SHA-256：

```text
c78ef2328ed2025055e7f26e8a32378127154e2d6a2634b8d52cfe067c860463
```

`BUNDLE.pilot.json` 与 `BUNDLE.full.json` 分别保留两次代码清单；`full.start.json` 保存完整实际命令、PID和时间。正式控制器参数：

```bash
~/crisp-venv/bin/python -u -B e2e/fg_batch.py \
  --manifest e2e/fg/out/full_jobs.json --out e2e/fg/out/main \
  --workers 128 --reserve-gib 16 --name full \
  --release-file e2e/fg/out/main/release.json \
  --external-manifest e2e/fg/manifest/pilot_jobs.json \
  --external-status e2e/fg/out/main/pilot.status.json
```

查看当前状态：

```bash
cd ~/crisp_fg_20261113
head -n 18 e2e/fg/out/main/full.status.json
tail -n 4 full.log
tail -n 2 release.log
free -h
```

`completed`包括已复用小批，`active`是全量队列自己启动的进程；`combined_active`加上仍运行的小批，合计上限128。`release.audit.json`记录各类核对结果，`waiting_for_release`为等待前置核对的作业。全流程最新测时文件是`e2e/fg/out/full_estimate_launch.json/md`，启动时仍有慢任务和修复未测，不能把已覆盖部分的小时数当作整批完成时间。


## 2026-11-13 16:54：实验全部完成

最终文件为 `e2e/fg/out/main/finish.batch.json`：6813/6813成功，无失败、无待运行项。最后五项于16:48:44以5并行补齐，16:53:45结束，298.18秒，最低可用214.76GiB。`finish.status.json`保留最后一次运行中采样，完成状态以最终batch和receipt为准。

原OOM记录在 `patches/20261113_finish/`，实际命令在 `finish.start.json`。完整归档 `FG_20261113_complete.tar.gz` 与逐文件摘要 `FG_20261113_files.json` 已生成，正下载本机。结果汇总/核对进展见 `PROGRESS.md` 末条和 `fg/evidence/completion_audit.json`。

用户更新的内存处理：有余量自动续跑，余量不足时通知并等待恢复，恢复后自动继续。SSH启动结束后继续读取状态；实验完成后接续已授权的下载、汇总和分支推送。
