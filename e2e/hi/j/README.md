# 第一批 J 计时与 Split 种子补充

2026-11-14。执行入口为 `e2e/j_timing.py` 和 `e2e/hi_split_supplement.py`。此目录保存输入清单和汇总证据；原始日志、保存图及 Numba 私有缓存放在 `e2e/hi/out/`。

## 已完成的本机接口核对

`local_smoke_checks.json` 记录本机核对，不作为正式服务器耗时：

- GPT 64 核 decode FFN 的完整广播图可保存、重载；原与快实现各一次预热、五次完整 CPCM 调用，延迟与 CP/阻塞分析摘要相同。
- 空私有目录首次 JIT 与新进程磁盘缓存加载均成功；不清理公共缓存。
- WSC A256 重新构造控制输入，两个流的 CPCM 值和排序与冻结输入相同。
- FACE 的 m2 在 N/M 两轴候选构造、选轴后，保存图摘要与既有正式图相同。

## Split 补充：63 次后端

`split_supplement_cloud.json` 完全复用 POST-FG 三例的九张保存图，新增种子 4–10；不重新搜索、不修改旧种子 1–3。

```bash
python e2e/fg_batch.py --manifest e2e/hi/j/split_supplement_cloud.json --out e2e/hi/out/split_supplement --workers 128 --reserve-gib 3 --memory-hint-gib 2 --name split_supplement
```

此批可与 H1 同时按总内存和进程数安排。`hi_split_supplement.py report` 将本机旧 27 项和新 63 项汇总为九个十种子结果；图文件 SHA 与图内容 SHA 均核对。

2026-11-14 已完成 63/63，0 失败，墙钟 34.049 秒。十种子完整结果见 `split_results/SPLIT_SEEDS_1_10.md`：Split-m2 的 30/30 样本均低于三个既定案例的刚性下界。

## J0 / J1 / J4 / J5

J 正式计时使用空闲服务器、单个工作进程，且数值库与 Numba 均单线程。四个 H1 候选的并行成本不充当正式 J 计时。

```bash
python e2e/j_timing.py environment --out e2e/hi/j
python e2e/fg_batch.py --manifest e2e/hi/j/J1_prepare.json --out e2e/hi/out/j_prepare --workers 1 --reserve-gib 3 --name j_prepare
python e2e/j_timing.py finalize-j1 --preparation-out e2e/hi/out/j_prepare --out e2e/hi/j
python e2e/fg_batch.py --manifest e2e/hi/j/J1.json --out e2e/hi/out/j_timing --workers 1 --reserve-gib 3 --name j1
python e2e/fg_batch.py --manifest e2e/hi/j/J1_startup.json --out e2e/hi/out/j_timing --workers 1 --reserve-gib 3 --name j1_startup
python e2e/fg_batch.py --manifest e2e/hi/j/J4.json --out e2e/hi/out/j_timing --workers 1 --reserve-gib 3 --name j4
python e2e/fg_batch.py --manifest e2e/hi/j/J5.json --out e2e/hi/out/j_timing --workers 1 --reserve-gib 3 --name j5
```

- J1：三个既有 GPT 图 × 原/快实现 × 三个独立进程，共 18 项。每进程一次预热、五次稳态。记录保存图加载、平台准备、图编译、数组装入、内核、Schedule 恢复、CP/阻塞分析和完整调用。首次广播图准备单列，计时重复不再重复广播构图。另有三次首次 JIT、三次磁盘缓存加载，使用各次专属缓存目录。
- J4：WSC/FACE × A256/S256 × 三个独立进程，共 12 项。记录冻结图/平台加载、All256 基准切块、CPCM 期限及排名、被测方案切块、策略与初始就绪列表准备。A256 与 S256 的期限使用同一个 All256 基准。没有调用仿真后端，也不将其记成每包硬件仲裁延迟。
- J5：三例 × 三次 m2，共九项。每次同时构造 N/M 合法轴。记录状态加载、平台、基准工作量、驻留决策、选定图核对、导出及完整流程。通过内存中的函数副本加入分段时钟；原 Split 文件不改，选中图摘要与冻结图核对。构图细分包括孤儿/合法轴、落点与路由评分、输入传输、末端广播、语义核对/统计和算术记账。

J1 三图输入来自旧 F/G 根目录。其中 64 核 FFN 使用 `cpcm_sa/out/s2/{snaps,runs}/ffn_A3h_g0_s123.pkl.gz`，64 核 prefill MHA 使用 `e2e/out/b4/{snaps,runs}/pf_mha_w1_A3h_g0_s123.pkl.gz`；144 核图使用既有 G 的增长配置 prefill MHA g0/seed123。以文件清单中的 SHA 核对本机与云端输入。

J1 的新模型三图、J2/J3 和新模型两个 J5 点等待相应正式输入。此处只准备已获准的既有图项目。
