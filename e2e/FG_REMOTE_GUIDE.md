# F/G 服务器运行记录

2026-11-13（Codex）。服务器 `ubuntu@119.45.93.240`，工作目录 `/home/ubuntu/crisp_fg_20261113`；Python 使用 `/home/ubuntu/crisp-venv/bin/python`。原实例重新开机，既有 SSH 主机标识与环境匹配。

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

后续共用一个队列。先开放独立健康与带宽搜索，以及代表修复/联合的依赖；各配置的正式候选选择完成后执行对应代表任务。核对结果并更新时间后，开放该配置余下任务。开放列表使用调度器的 `--release-file`，无需另开重叠队列。小批与全量共用 `e2e/fg/out/main`，已完成且定义相同的任务按 receipts 复用。

恢复需保留 `results/`、`raw/`、`snapshots/`、`exports/`、`selection/`、`faults/`、`joint_cache/`，以及 `specs/`、`receipts/`、日志、清单和开放列表；这些文件共同记录候选与后续任务的输入。新的正式启动命令、PID 和源码版本在实际启动后追加。

完整输出当前保存在服务器磁盘，本机已开始分批接收结果记录。原 D 阶段 COS 定时上传仍在；F/G 新目录的三分钟上传尚待用户授权，当前未启用。
