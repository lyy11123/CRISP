# 论文证据索引

每一行把可写入论文的陈述连接到实验范围、数据、代码、表图和版本。历史报告保留原配置名称；2026-11-18 新补充采用 S256 主配置。

直接仿真、算子/层组装、解析估算和保存态机制分析分别标注。W0、W1、H/I 的八Q一KV合同分别保存，不能合成一种未运行的工作负载。单层或固定KV请求的相加结果也不表示整模型同时执行。

历史核查提交：`244c59533e6154aafaf997dcb51fe58032b2981e`；BusyBarn artifact：`299089ef266c54c6a1fa8c2f52df66e6c970b52a`。补充实验运行于该历史基线加新增脚本快照，实际v1源码保存于 `dist/server_runtime/`；修正准备身份后的v2源码保存于 `dist/server_runtime_identity/`。本轮最终提交见 `DELIVERY.md`，各输入文件摘要保存在索引CSV及核对JSON中。

[完整主张与来源表](tables/CLAIM_EVIDENCE.csv) · [模型范围表](tables/MODEL_SCOPE_TABLE.csv) · [文件存在性与摘要核对](evidence/diagnosis_evidence_index.json)

|编号|陈述|证据类型|状态|入口|
|---|---|---|---|---|
|A01|Prefill试点记录预测误差与实际质量，包含超出原5%停止线的结果|direct_operator_simulation|complete|[PILOT_SUMMARY.md](../PILOT_SUMMARY.md)|
|B01|GPT完整P/D以算子仿真为基础，W1去除无消费者的QKT/S末端广播|direct_operator_simulation|complete|[B4_SUMMARY.md](../B4_SUMMARY.md)|
|B02|单层和44层时间由实际算子时间相加得到|layer_assembly_and_analytical_request|complete|[B4_SUMMARY.md](../B4_SUMMARY.md)|
|C01|同一物理故障用于8个算子，EIR停止档、后端质量和迁移分别报告|direct_operator_simulation|complete|[C_SUMMARY.md](../C_SUMMARY.md)|
|D01|映射、修复和传输联合效果包含P代价、D延迟、迟约与配对单跑|direct_joint_simulation|complete|[D_SUMMARY.md](../D_SUMMARY.md)|
|D02|由单次重叠推算整请求的结果是解析外推|analytical_extrapolation|complete|[D_SUMMARY.md](../D_SUMMARY.md)|
|E01|Gemini和A1/A3h在2x2与1x4上完整比较，prefill差距可由计算与访存集中解释|direct_operator_simulation_and_saved_state_analysis|complete|[E_SUMMARY.md](../E_SUMMARY.md)|
|E02|44层混合布局与理想最慢组服务率属于组装与解析分析|layer_assembly_and_ideal_pipeline_analysis|complete|[E_SUMMARY.md](../E_SUMMARY.md)|
|F01|硬件参数迁移保留映射、修复和DSTU各自收益及不利点|direct_operator_and_joint_simulation|complete|[F_SUMMARY.md](../F_SUMMARY.md)|
|G01|2x3/3x3在固定工作量下比较原划分与增长划分|direct_operator_and_joint_simulation|complete|[G_SUMMARY.md](../G_SUMMARY.md)|
|FG02|F/G的单层与固定KV请求时间由算子结果组装|layer_assembly_and_fixed_service_request|complete|[F_SUMMARY.md](../F_SUMMARY.md)|
|PF01|确定性bb加速保留原结果与完整轨迹，调用开销和前向成本分别计时|equivalence_replay_and_timing|complete|[POST_FG_SUMMARY.md](../post_fg/POST_FG_SUMMARY.md)|
|PF02|S256已有36个健康/PF4配对降低全部P时间，D包含回退|direct_joint_simulation|complete|[DSTU_REFINEMENT.md](../post_fg/DSTU_REFINEMENT.md)|
|PF03|旧三例Split-m2能低于原固定块计算下界，切分粒度存在取舍|direct_operator_simulation_and_rigid_block_bound|complete|[EIR_SPLIT.md](../post_fg/EIR_SPLIT.md)|
|H01|Llama八Q一KV子图、RMSNorm与门控双支路已核对|model_contract_and_graph_validation|complete|[README.md](../hi/README.md)|
|H02|Llama主硬件decode收益明显，prefill随机CPCM选优仍有回退|direct_operator_simulation_and_layer_assembly|complete|[H_HEALTH_SUMMARY.md](../H_HEALTH_SUMMARY.md)|
|I01|Qwen MoE固定分派、共享历史KV与层乘数4已核对|model_contract_and_graph_validation|complete|[README.md](../hi/README.md)|
|I02|Qwen主硬件的P/D改善与WATOS长尾点的微小prefill FFN改善同时保留|direct_operator_simulation_and_layer_assembly|complete|[I_HEALTH_SUMMARY.md](../I_HEALTH_SUMMARY.md)|
|HI03|H/I固定PF4修复已齐，Qwen WATOS prefill FFN仍有明显残差|direct_operator_simulation_and_layer_assembly|complete|[B_I4_PF4_COMPLETE.md](../hi/closeout/B_I4_PF4_COMPLETE.md)|
|HI04|新模型Split-m2的效果有限，Qwen没有额外二分收益|direct_operator_simulation_and_rigid_block_bound|complete|[C_NEW_MODEL_SPLIT.md](../hi/closeout/C_NEW_MODEL_SPLIT.md)|
|HI05|原H5/I5联合实验保存映射/EIR/A256的独立增量与实际干扰|direct_joint_simulation|complete|[HI_FOLLOWUP_SUMMARY.md](../hi/HI_FOLLOWUP_SUMMARY.md)|
|S01|A256八组合显示传输增益取决于剩余干扰，FACE保留D回退|direct_joint_simulation|complete|[S_FACTORIAL_SUMMARY.md](../hi/closeout/S_FACTORIAL_SUMMARY.md)|
|P001|实际KV窗口与固定释放时刻解释g相同而真实重叠不同|posthoc_backend_window_analysis|complete|[P0_POSTPROCESS.md](../hi/closeout/P0_POSTPROCESS.md)|
|J01|完整调用、1000步映射、修复累计与控制准备采用独立串行计时|measured_serial_wall_time|complete|[J_SUMMARY.md](../hi/closeout/J_SUMMARY.md)|
|R303|WATOS剩余开销主要集中到D2D，固定状态关键链和改路已复算|saved_state_deterministic_CPCM_analysis|complete|[WATOS_DIAGNOSIS.md](WATOS_DIAGNOSIS.md)|
|R301|单轨迹用自身预测自主停档，30/100/300为独立预算而非连续300步|saved_result_postprocess|complete|[ONLINE_RECOVERY.md](ONLINE_RECOVERY.md)|
|R302|24点PF4选择表与后台修复回本按既有状态配对计算|saved_result_postprocess_and_analytical_payback|complete|[ONLINE_RECOVERY.md](ONLINE_RECOVERY.md)|
|R1|S256补齐后完整比较P/D、最大迟约、真实超期和配对单跑|direct_joint_simulation|complete|[S_AND_HI_S256_REPORT.md](S_AND_HI_S256_REPORT.md)|
|R2G|Online-CPCM-Gated按共享有向资源与预测KV结束时刻预先选择完整BB0/S256策略|prior_CPCM_prediction_and_selected_executed_strategy|complete|[S_AND_HI_S256_REPORT.md](S_AND_HI_S256_REPORT.md)|
|R2O|三类事后选择和P/D帕累托保留真实整策略行，P约束相对同输入original|retrospective_oracle_and_Pareto|complete|[S_AND_HI_S256_REPORT.md](S_AND_HI_S256_REPORT.md)|
|R2F|完整均值、分硬件相位及预测与真实执行并列展示全部108组|figure_data_from_complete_paired_results|complete|[FIGURE_GUIDE.md](FIGURE_GUIDE.md)|
|R2C|门控预测、CPCM前向、输入准备和后端验证成本分别统计，12次预测服务36个决定|measured_parallel_process_cost|complete|[S_AND_HI_S256_REPORT.md](S_AND_HI_S256_REPORT.md)|
|R4|564项完整记录、540项实际身份、282条复用及门控先冻结后执行均已逐项核对|completion_receipt_input_identity_and_source_chain_audit|complete|[AUDIT_SUMMARY.md](AUDIT_SUMMARY.md)|

后端1–10与修复搜索种子123/1/2/3/4分别记录；随机平局CPCM选优损失与预测误差分别解释。J的独立进程重复用于串行计时，不增加健康候选数。历史A256八组合及H5/I5保留。S256新增108组的324条联合结果、216条逻辑单跑已逐项核对；Online的36个决定在本轮新后端开始前冻结，并在108组中取对应完整策略的P/D。三类事后选优另列。
