# E 阶段图说明

`asis` 沿用 B 原式，decode FFN 内的一次 LN 与外加两次 LN 均计入。
两种 LN 计数的数值均见 `layer_table.csv` 和 `model_table.csv`。
A3h 是在原后端评估的 CRISP 映射。健康图使用各单元 14 个候选中的后端均值最优结果。

- `e_layer_latency` 比较 2×2、1×4 组上一个 GPT W1 block 的 prefill 与 decode 一步。各 phase 的纵轴单位独立。
- `e_model_latency` 比较 56 Die 混合和长条布局的 44 层组装时间，包含 13 次串行组间激活传输。组内和组间时间可在 `model_table.csv` 中查看。
- `e_request_latency` 按固定 KV 长度 2048 计算 `R(N) = P_model + N × D_model`。N 为 prefill 后追加的 decode 步数，标点对应 128、512、2048。
- `e_fault_latency` 采用 2048 个 decode 步，以同方法、同起点故障前结果为分母。PF4 为 G2 内双向 D2D 链路故障，PF1 为核和路由器故障，PF8 为 DDR 访问端口故障。左图只计受损的 3 层，右图保留另外 41 层及组间传输，两图的纵轴范围独立。

输入 `summary.json` 的 SHA-256 为 `5e379bfba20f5c172d03c561ebc04822210bebb5223d00ee4e31a0528600192f`。SVG 保留文本，字体为 DejaVu Sans。PNG 为 300 dpi。
