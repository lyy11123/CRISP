# F/G 后续实验服务器记录

2026-11-13，Codex。用户确认计划、自检通过后直接执行，交接与任务范围见 [POST_FG_PLAN.md](POST_FG_PLAN.md)。

## 连接与目录

- SSH：`ubuntu@1.13.251.45`，128 vCPU；20:27可用内存约244GiB。
- 本轮工作目录：`/home/ubuntu/crisp_post_fg_20261113`。
- Python：`/home/ubuntu/crisp-venv/bin/python`，3.13.18；NumPy2.5.4、Numba0.68.1、llvmlite0.50.0。
- 旧F/G输入：`/home/ubuntu/crisp_fg_20261113/e2e/fg/out/main/`，只读取已有状态/结果。
- 新原始输出：`e2e/post_fg/out/main/`。每项保存spec、日志、结果与receipt；CPU/内存由fg_batch采样。
- 本机：`E:/58Institute/PhD_TERM3/CRISP_codex/e2e/post_fg/`，原始数据及传输包分别放out、dist。

源码包 `POST_FG_initial.tar.gz`：1694839字节、2437文件，SHA-256 `fe8a9b5b5ba792c0304570f06073679ac9859762f0e9e495b9543bfd25ab037c`，基线d5904de，逐文件摘要在云端BUNDLE.json。首次控制器启动缺少历史调度输入fg_estimates.json，尚未派发任务即退出；已从旧F/G目录补齐，原日志bb_graphs.log保留。打包入口已补上该依赖。

补充输入包 `POST_FG_followup_inputs.tar.gz`：1022586字节、28文件，SHA-256 `c286243bab97dff66d46528796cd7e824a4b29ff49e2ebce8a1c2de436c272e2`；包含新策略脚本、9张Split图与逐片记录，摘要在BUNDLE.followup_inputs.json。运行入口的实际源摘要另外记在每次start.json。

## 当前批次

1. `bb_graphs_r1`：PID1140280，广播前decode/prefill图两项已完成，36.01秒。完整Schedule/关键路径/阻塞归因一致；稳态加速3.013/2.872倍，仅对应广播前图。
2. `bb_searches`：PID1230466，6项短搜索，workers=1，保留1000步温度过程截30步。空闲服务器计时，不与质量批次同时跑。状态bb_searches.status.json，日志bb_searches.log；完成记录以batch.json为准。
3. 后接完整广播图2项、保存L1短EIR3项，仍串行计时。通过后执行WSC健康DSTU冒烟与FACE m1 Split冒烟，再按实测和依赖安排质量批次，最多128并行。

网页终端可读：

```bash
cd ~/crisp_post_fg_20261113
tail -n 3 e2e/post_fg/out/main/bb_searches.log
cat e2e/post_fg/out/main/bb_searches.status.json
free -h
```

运行与续跑使用同一入口，已完成且配置一致的项由receipt复用。例如：

```bash
cd ~/crisp_post_fg_20261113
~/crisp-venv/bin/python e2e/post_fg_launch.py \
  --manifest e2e/post_fg/manifest/search6_jobs.json \
  --name bb_searches --workers 1 --memory-hint-gib 5
```

同一批正在运行时不用重复启动。控制器每2秒检查进程与内存、每30秒保存状态。当前对话通过SSH持续跟踪；没有最终batch时，不把SSH命令返回视作实验结束。失败保留原记录，修正后使用同一清单或明确的新版本继续；内存不足时按用户授权等待可用量恢复后自动继续。

## 保存

旧D/F/G三分钟COS任务仍在。本轮新目录 `cos://crisp/crisp/POST_FG_20261113` 的定时备份需补充明确授权；问题已发送，尚未启用。云端结果保留，阶段完成后同步本机；全部完成后打完整包、逐文件核对并生成本轮报告。
