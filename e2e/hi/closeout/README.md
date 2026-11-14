# H/I 收尾工作目录

本轮按 [HI_CLOSEOUT_PLAN.md](../../HI_CLOSEOUT_PLAN.md) 执行。总览见 [CLOSEOUT_SUMMARY.md](CLOSEOUT_SUMMARY.md)。原 896 个健康候选、H5/I5、300 步修复主结果和旧 J 的 45 项计时继续使用原数据。

## 任务与结果

| 工作包 | 清单 | 输出/报告 |
|---|---|---|
| P0 | 已有结果的只读补充 | P0_POSTPROCESS.md |
| A | manifests/a.json，5 条独立 1000 步轨迹 | A_QWEN_DFFN_BUDGET.md |
| B | manifests/b.json，46 项 | B_I4_PF4_COMPLETE.md |
| C | manifests/c_mf6.json，90 项 | C_NEW_MODEL_SPLIT.md |
| S | s_repair_manifest.json + s_joint_manifest.json，共 408 项 | S_FACTORIAL_SUMMARY.md |
| J | manifests/j_micro.json + j_search.json + j3.json，共 80 项 | J_SUMMARY.md |

质量实验总清单为 `manifests/registry.json`，共 549 个作业。J 在质量实验全部结束后单进程运行；80 项新增计时与旧 45 项合并汇总。小批和复用占用各自在正式清单中的原 ID，不另加实验数。

服务器为 `ubuntu@175.27.169.196`，仓库 `/home/ubuntu/crisp_hi_20261114`，解释器 `/home/ubuntu/crisp-venv/bin/python`。质量结果在 `e2e/hi/out/closeout_20261115/`，J 结果在 `e2e/hi/out/closeout_j_20261115/`。每个输出目录包含清单中的结果、spec、日志、成功回执、活跃进程记录和监控历史。质量上限 128 个单线程进程、预留 3 GiB；J 上限为 1。

## 查看运行情况

在服务器仓库根目录运行：

```bash
cat e2e/hi/out/closeout_20261115/closeout.status.json
tail -n 1 e2e/hi/out/closeout_20261115/host_monitor.jsonl
tail -n 8 e2e/hi/out/closeout_20261115/closeout.log
/home/ubuntu/crisp-venv/bin/python -B e2e/hi_closeout_estimate.py --live
```

`controller_finish.json` 记录实际完成时间和最终数量；启动命令返回或 SSH 断开都不代表实验完成。ETA 按日期另存，原估计保留。质量剩余时间与后续 J 串行时间分别报告。

## 复算

以下命令从仓库顶层执行，只读取保存结果，不启动新的搜索或后端仿真。未齐时 A/B/C/J 可加 `--progress`。

```text
python -B e2e/hi_closeout_p0.py
python -B e2e/hi_closeout_a_report.py
python -B e2e/hi_closeout_b_report.py
python -B e2e/hi_closeout_c_report.py
python -B e2e/hi_closeout_s_report.py
python -B e2e/hi_closeout_j_report.py
```

表格保存在 `tables/`，图在 `figures/`，检查记录在 `evidence/`。准备图和冒烟原始文件保存在本机及归档，Git 提交代码、报告、表图、清单和检查记录。

## COS 备份与本机保存

三分钟备份继续使用 `cos://crisp/crisp/HIJ_20261114`。本轮结果位于该前缀的 `out/closeout_20261115/` 和 `out/closeout_j_20261115/`，完成包位于 `dist/`。

数据经 COS 下载到本机 `e2e/hi/dist/`，以归档侧文件的整包 SHA-256 和包内逐文件清单核对，再恢复至 `e2e/hi/`。SSH 用于运行和查看状态。临时下载地址不写入报告或仓库。
