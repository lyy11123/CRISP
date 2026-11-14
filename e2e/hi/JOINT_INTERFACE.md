# H/I 联合输入接口核对

`hi_joint.py` 已提供实际可调用的候选导出、K/V 消息识别、完整 decode 链构造和跨组 K/V 路由。此次只构图及核对，没有运行联合后端。

## 输入与结果

| 项目 | Llama | Qwen3-MoE |
|---|---:|---:|
| D 链 MHA 头组 | 8 | 4 |
| D 链独立算子段 | 11 | 7 |
| 外置 RMSNorm / FFN 内嵌 RMSNorm | 1 / 1 | 1 / 1 |
| D 链事件 | 2,964 | 10,211 |
| 算子间新增屏障依赖 | 7,778 | 22,602 |
| P 侧 K/V 逻辑字节 | 8,388,608 | 8,388,608 |
| K/V 消息 | 16 | 8 |
| D 组 DDR 目标端口 | 32 | 32 |
| DDR 接收副本合计字节 | 268,435,456 | 268,435,456 |
| 重路由后按每消息唯一有向链路累加字节 | 635,437,056 | 742,391,808 |
| 跨组有向链路 | 4 | 4 |
| K/V 与 D 链共用链路 | 119 | 163 |
| P 非 K/V 通信与 D 链共用链路 | 0 | 0 |

Llama 的 P MHA、D MHA、D FFN 使用本次 H1 已完成的正式候选；导出时重载三个快照，完整广播图的确定性估值与已存结果逐项相同。LN、projection 只使用 H0 初始映射。Qwen 的五种输入均为 I0 初始映射。表内数字用于接口核对，没有用这组混合输入报告联合性能。

两模型的图均通过依赖闭合、无环、计算设备与路由链路存在性核对。K/V 从模型的缓存 tensor 标签和生产者识别，逐生产者核对张量分片字节与消息字节；包含 Ktranspose 输出，不再使用 GPT 的 `kconv/vconv` 名称筛选。逻辑字节、DDR 多播副本字节和实际链路工作量分别记录。

这组 P/D 表示共享资源的两个独立请求，跨流依赖边数为 0。P 的 K/V 完成不会释放这里的 D 链。D 链内部仍逐段串行依赖；后续 H5/I5 按既定相位独立释放 D 流。

## 可调用入口

- `export_candidate(record, raw, snapshot, output)`：读取新模型正式候选，核对合同、工作量和完整广播估值，导出完整事件图及缓存身份。
- `cache_messages(state, ctx, spec)`：通过缓存 tensor 与生产者获取 K/V 末端消息及字节明细。
- `decode_chain_order(model)`、`chain_graph(model, exports, die_map)`：按模型合同构造 D 链。Llama 为 `LN + 8×MHA + projection + FFN`，Qwen 为 `LN + 4×MHA + projection + FFN`。
- `redirect_kv(events, platform, tags, target_dies, ctx)`：把所选 K/V 消息重路由到目标组，记录多播份数及实际链路字节。

复用的历史函数为 `d_joint.retag/remap_events/check_group/check_graph/links_of` 及 `fg_joint.maps`；旧源码未改动。没有使用硬编码八头的旧 `CHAIN` 或按 GPT 名称筛选的旧 `build_kv`。

复算命令：

```powershell
e2e/post_fg/dist/venv/Scripts/python.exe e2e/hi_joint.py probe --model llama --candidate-root e2e/hi/out/h1_main --out e2e/hi/evidence/joint_interface_llama.json
e2e/post_fg/dist/venv/Scripts/python.exe e2e/hi_joint.py probe --model qwen3moe --out e2e/hi/evidence/joint_interface_qwen3moe.json
```

本机两项分别耗时 12.707 秒、17.654 秒，属于接口检查记录，不是 J 的正式串行计时。证据在 [`joint_interface_llama.json`](evidence/joint_interface_llama.json) 和 [`joint_interface_qwen3moe.json`](evidence/joint_interface_qwen3moe.json)。Llama 三份候选导出保存在 `hi/out/joint_interfaces/`。

H5/I5 的健康选优、PF4 修复状态与完整调度作业仍按后续批次接入；此次没有启动这些实验。
