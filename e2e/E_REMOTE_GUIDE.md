# E 全量运行记录

**已完成并下载（2026-11-12）：** 632/632 正常退出，耗时 50 分 09.45 秒；本机数据在 `e2e/e/out/full_remote`，完整汇总见 `E_SUMMARY.md`。采样最低可用内存 184.85 GiB，OOM kill 为 0。下载和核对后尝试通过 Chrome 关机，电脑控制工具因无法确定当前浏览器地址而中止，未执行关机；服务器仍待在腾讯云控制台关闭。下文保留启动与查看命令。

2026-11-12 北京时间 20:31:22 启动，服务器 `119.45.118.66`，用户 `ubuntu`。目录 `/home/ubuntu/crisp_e_full_20261112`，进程池上限 128，启动时观察到 128 个子进程。主进程初始 PID 为 241350。

本批有 632 个任务：390 次新搜索、242 次旧快照重评。另有小批已完成的 40 个主候选和 16 个结束状态对照，最后一起汇总。

## 在腾讯云网页终端查看

```bash
cd ~/crisp_e_full_20261112
tail -n 10 full.log
find e2e/e/out/full/results -maxdepth 1 -name '*.json' | wc -l
ps -p "$(cat full.pid)" -o pid,etime,pcpu,rss,args
ps --ppid "$(cat full.pid)" -o pid= | wc -l
free -h
```

- `full.log` 每完成一个任务写一行；`rc: 0` 表示正常完成。
- 结果文件数应逐步达到 632。已经下载的小批 56 个结果另存，不在此数中。
- 子进程数在队列尚有作业时保持接近 128，最后几个长任务结束前会逐步减少。
- 最后一行查看当前内存；关注 `available`。

每个任务的日志位于 `e2e/e/out/full/logs/`，结果在 `results/`，映射在 `snapshots/`。整批退出后写出 `e2e/e/out/full/batch.json`。后台进程独立于 SSH 会话。

## 复现与续跑

源码版本 `251cc799`。输入包 `e2e/e/dist/E_full_632.tar.gz` 为 59,795,829 字节，SHA-256：

```text
d8e7f69fbacf1ba91554df167741ce3d15a06104a112c6ae7a523056f2b493f1
```

解包后核对了 `BUNDLE.json` 中的 817 个文件。任务清单为 `e2e/e/out/full_plan/remaining.json`，启动记录保存在 `full.start.json`。本批使用的命令：

```bash
nohup ~/crisp-venv/bin/python -u -B e2e/e_run.py run \
  --manifest e2e/e/out/full_plan/remaining.json \
  --out e2e/e/out/full --workers 128 > full.log 2>&1 < /dev/null &
echo $! > full.pid
```

上面是启动记录。需要续跑时沿用同一清单和输出目录，已完成且配置相同的任务会跳过；先确认原主进程已结束。旧日志另存一份后再启动。

本机接收位置为 `e2e/e/out/full_remote`，结果包和来源信息在 `e2e/e/out/full_remote_archive`。完整 E 结果由本批和 `e2e/e/out/pilot_remote` 合并，完成后运行 `e_check.py`、`e_report.py` 和 `e_plot.py`。
