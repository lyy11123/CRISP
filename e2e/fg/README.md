# F/G 结果文件

F/G 于北京时间 2026-11-13 16:53:45 完成。6813 个调度作业全部成功，另保存 56 个 Gemini 结束状态，共 6869 条结果记录。

## 阅读入口

- [F 阶段详细汇总](../F_SUMMARY.md)：硬件参数迁移、故障修复、P/D 联合与 D2D 带宽变化。
- [G 阶段详细汇总](../G_SUMMARY.md)：2×3、3×3 固定/增长切分、规模效率与优化成本。
- [完整数据表](tables/SUMMARY.md)：CSV/JSON 表及指标说明。
- [完成与归档核对](evidence/completion_audit.json)：任务数量、归档摘要和逐文件检查。

## 本机保存位置

| 路径 | 内容 |
|---|---|
| `out/main/results/` | 6869 条最终结果 |
| `out/main/receipts/`、`specs/`、`logs/` | 作业配置、退出状态与日志 |
| `out/main/snapshots/` | 3080 个候选状态快照 |
| `out/main/raw/`、`exports/` | 原始评估与修复状态 |
| `out/main/selection/`、`faults/`、`joint_cache/` | 选优、物理故障及联合输入 |
| `dist/FG_20261113_complete.tar.gz` | 从 COS 下载的完整原始归档 |
| `dist/FG_20261113_files.json` | 云端打包时的逐文件 SHA-256 清单 |
| `dist/server_runtime/` | 当时实际运行的源码、清单、控制器日志与重试记录 |
| `tables/`、`figures/`、`evidence/` | 分析表、PNG/SVG 图和核对记录 |

压缩包为 3,572,369,807 字节，SHA-256：

```text
c90c95ebe690f3c06b53709c084a066e7d9ac88eb5e99d083a1056c5e96444de
```

归档共 37909 个文件（包含文件清单）。清单中的 37907 个静态文件逐一匹配；打包脚本自身日志在清单生成后增加一行 26 字节的数量记录，已单列说明。全部实验输出和运行源码与清单一致。

完整原始包同时保存在 `cos://crisp/crisp/FG_20261113/archive/FG_20261113_complete.tar.gz`。服务器已由用户关机，本机后处理直接使用以上文件。GitHub 保存分析代码、文档、表格、图和核对记录；`out/` 与 `dist/` 沿用仓库的忽略设置。

## 重新生成汇总

在仓库顶层执行：

```powershell
python -B e2e/fg_report.py --manifest e2e/fg/manifest/search.json --run-out e2e/fg/out/main --out e2e/fg/tables --workflow-manifest e2e/fg/dist/server_runtime/e2e/fg/out/pipeline_jobs.json --run-root e2e/fg/dist/server_runtime
python -B e2e/fg_workflow_report.py --manifest e2e/fg/manifest/all_jobs.json --run-out e2e/fg/out/main --out e2e/fg/tables
python -B e2e/fg_report_check.py --tables e2e/fg/tables --run-out e2e/fg/out/main --out e2e/fg/evidence/healthy_report_checks.json
python -B e2e/fg_workflow_check.py --run-out e2e/fg/out/main
python -B e2e/fg_figures.py --tables e2e/fg/tables --out e2e/fg/figures
python -B e2e/fg_summary.py --tables e2e/fg/tables --out e2e --completion e2e/fg/evidence/completion_audit.json
```

这些命令读取已保存结果，生成分析表和正文。
