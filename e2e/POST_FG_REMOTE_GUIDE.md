# F/G 后续实验服务器记录

2026-11-13建立，2026-11-14恢复更新，Codex。用户确认计划、自检通过后直接执行，交接与任务范围见 [POST_FG_PLAN.md](POST_FG_PLAN.md)。

## 连接与目录

- 当前SSH：`ubuntu@118.195.219.207`，128 vCPU；11月13日旧地址为 `1.13.251.45`，当日20:27可用内存约244GiB。
- 本轮工作目录：`/home/ubuntu/crisp_post_fg_20261113`。
- Python：`/home/ubuntu/crisp-venv/bin/python`，3.13.18；NumPy2.5.4、Numba0.68.1、llvmlite0.50.0。
- 旧F/G输入：`/home/ubuntu/crisp_fg_20261113/e2e/fg/out/main/`，只读取已有状态/结果。
- 新原始输出：`e2e/post_fg/out/main/`。每项保存spec、日志、结果与receipt；CPU/内存由fg_batch采样。
- 本机：`E:/58Institute/PhD_TERM3/CRISP_codex/e2e/post_fg/`，原始数据及传输包分别放out、dist。

源码包 `POST_FG_initial.tar.gz`：1694839字节、2437文件，SHA-256 `fe8a9b5b5ba792c0304570f06073679ac9859762f0e9e495b9543bfd25ab037c`，基线d5904de，逐文件摘要在云端BUNDLE.json。首次控制器启动缺少历史调度输入fg_estimates.json，尚未派发任务即退出；已从旧F/G目录补齐，原日志bb_graphs.log保留。打包入口已补上该依赖。

补充输入包 `POST_FG_followup_inputs.tar.gz`：1022586字节、28文件，SHA-256 `c286243bab97dff66d46528796cd7e824a4b29ff49e2ebce8a1c2de436c272e2`；包含新策略脚本、9张Split图与逐片记录，摘要在BUNDLE.followup_inputs.json。运行入口的实际源摘要另外记在每次start.json。

## 已完成批次与当前归档

1. `bb_graphs_r1`：PID1140280，广播前decode/prefill图两项已完成，36.01秒。完整Schedule/关键路径/阻塞归因一致；稳态加速3.013/2.872倍，仅对应广播前图。
2. `bb_searches`：PID1230466，6/6完成，3507.07秒。workers=1，保留1000步温度过程截30步，空闲服务器串行计时。
3. `bb_remaining`：PID1515999，完整广播图2项、保存L1短EIR3项，5/5完成，610.13秒。WP2总计13项，后端调用0；完整等价与提速记录见BB_SPEEDUP.md。
4. `quality_smoke`：PID2990687，25/25完成，512.12秒，最高17活跃进程、最低可用234.52GiB。包括WSC六方案24项与FACE Split-m1的一项，均计入后续正式清单。
5. `split_main`：PID3527143，27/27完成，24.024秒，包含1项已完成结果。云端独立构图计时19.744秒，未新增后端。
6. `quality_health`：PID3754499，初批210/213成功，600.522秒，最高54活跃进程、最低可用219.73GiB。3项A256代表日志在仿真完成后与历史值比较失败，已定位为旧保存图与新构图的集合遍历顺序差异。完整初批归档已下载验证，原文件另存 `e2e/post_fg/out/main/attempts/20261113_before_fresh_baselines/`；目录内inventory记录258份原文件，另有inventory本身与当时源码，共260文件。
7. 11月13日22:38:54启动统一构图补评，顺序控制器PID1604232，先60项BB0/A256，再3条A256日志；11月14日09:25恢复时已确认全部成功。`dstu_fresh_baselines.batch.json`记录60/60、260.215秒；`dstu_fresh_verify.batch.json`记录63/63，其中60项复用、3项新日志核对，484.098秒。`dstu_fresh_patch.finished.json`记录新后端63次、总744.326秒、status=ok；最后文件于11月13日22:51写成。日志为 `out/main/dstu_fresh_patch.log`。最终健康清单为 `quality_health_fresh_cloud.json`，初批清单保留。健康质量180项中174次实际后端、6项批内复用；另6条代表日志独立记账。输入修正及证明见POST_FG_PLAN.md。
8. 11月14日09:26:57启动 `dstu_fault`，PID831370，清单 `dstu_fault_fresh_cloud.json`。从健康结果选定S256，与BB0/A256完成两硬件PF4确认：54项联合、36项P/D单跑，共90次实际后端；90/90成功、0失败，274.280秒，最高73活跃进程、最低可用210.446GiB。最终记录与日志分别为 `out/main/dstu_fault.batch.json`、`dstu_fault.log`，旧status保留最后一次轮询快照。

本轮实验均已完成。云端五份最终清单合计316项的manifest/spec/result/receipt核对全部passed：13项计时验证、297次实际后端、6项批内复用，成功作业累计8.505053进程小时。本机已取得 `post_fg/evidence/completion_audit.json`；完整总归档已保存本机post_fg/dist，4639文件全部核对通过，解包结果在out，实际云端源码在dist/server_runtime。独立执行器验证另列，不混入316项主清单。

独立连续策略优化验证另外执行1次完整WSC A256-M：333627条发放记录完全一致，输出在 `out/policy_validation/`，报告DSTU_POLICY_SPEEDUP.md。它未替换本轮正式策略实现，不并入健康180项。

网页终端可读：

```bash
cd ~/crisp_post_fg_20261113
ls -lt e2e/post_fg/out/main/*.status.json | head
cat e2e/post_fg/out/main/dstu_fresh_patch.finished.json
cat e2e/post_fg/out/main/dstu_fault.batch.json
tail -n 3 ~/crisp_post_fg_cos_backup.log
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

旧D/F/G三分钟COS任务仍在。用户已于11月14日授权本轮 `cos://crisp/crisp/POST_FG_20261113` 目录的三分钟备份，现已安装并启动；09:25首轮PID760004，09:25:53日志确认 `out/main/traces/dstu_face_healthy_H0_g0.9_s1.messages.json.gz` 上传成功。

```cron
*/3 * * * * /home/ubuntu/crisp_post_fg_20261113/e2e/post_fg_cos_backup.sh >> /home/ubuntu/crisp_post_fg_cos_backup.log 2>&1
```

授权与首轮证据保存于 `post_fg/evidence/cos_backup_enabled_20261114.json`。09:33:20日志已出现 `Post-F/G backup cycle finished`，完成首个完整上传轮次。最终总包已下载并逐文件核对通过；同一完整包于09:45:05上传COS成功，大小124852269字节。报告与表图已推Codex分支05f3b25，完整下载证据见post_fg/evidence/complete_download.json。

09:30扩充脚本以包含报告和最终包时，正在执行的首轮读到更新后的文件末尾并因EOF退出；经语法检查后续传，已上传文件跳过，09:33:20的完整轮次已确认该次续传成功。
