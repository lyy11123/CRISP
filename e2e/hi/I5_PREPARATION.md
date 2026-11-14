# I5 Qwen 联合验证准备

2026-11-14。复用已完成110项正式验证的H5联合方法，新入口为 `hi_i5.py`。I4轻批于18:53:13完成201项，完整文件已恢复本机并通过传输SHA核对；本机输入检查通过，已生成110项正式清单和14项正式试点清单，尚未启动I5。

Qwen D链为7段：一个外置LN、4个8Q+1KV的MHA头组、projection和含第二个RMSNorm的FFN。不会沿用Llama的8组，也不会再加第二个外置LN。P侧为Qwen prefill MHA代表子图及其真实Ktranspose/V跨组传输；P/D仍为独立请求，没有跨流释放依赖。

P输入已经按各自完整14候选核对：A1为g0/l1，十后端均值4208527.9ns；A3h为g0/seed3，均值3274634.7ns。三个源文件的SHA已记录在 `evidence/i5_P_healthy_selection.json`。公共参考仍用健康A1/original的三个P单跑和三个D单跑，属于配对单跑本身。

任务数量与H5相同：健康3臂、D组PF4四臂，各3相位×3种子，共63联合；42逻辑配对单跑预计按输入归并为39实际单跑，去重的3项必须通过平台、完整有序图、实际集合遍历三项核对。C4内部使用当前that排序，公共健康A1参考用于统一评价，保留两者区别。

所需I4输入为四decode算子的A1/A3h健康、A1/A3h R0，以及stop300返回的最终EIR代表，共20个逻辑导出引用。若原停止规则早停，stop300本身会指向有效最终状态；在stop300尚未生成时不以现有R0/L1或某一修复种子代替。

`manifest/i5_template.json` 保留为带引用的静态模板，不能提交运行。`evidence/i5_readiness.json` 已通过，20个D引用对应19个唯一导出文件；正式清单为 `manifest/i5.json`，试点为 `manifest/i5_pilot.json`，输入索引为 `manifest/i5_inputs.json`。云端启动还要求H5完整110作业通过。

Qwen的PF4最终EIR状态按I4停止结果选取：LN停在R0，MHA为300步种子1，projection为30步种子4，FFN为300步种子3。这里读取的是停止规则实际返回的代表导出，没有重新在修复种子间选一个最小T̂。输入文件及来源结果的完整摘要见 `evidence/i5_input_sha256.json`。

后续先执行健康A3h A256、g=0.5、seed1的14项正式试点闭包（2导出、2图、8单跑、1参考、1联合），实测时间/RSS后再估计剩余正式作业。试点数据计入全量并原位复用；试点控制记录移到pilot_run再启动完整清单，避免误报阶段结束。

新输出目录 `out/i5`；为H3/I2、I3、H4、I4轻批、WATOS decode FFN补充批以及H5所有未完成项预留名额，总计不超过128进程，系统预留3GiB。已完成父批按有效回执扣除为0；I3和WATOS decode FFN补充批仍在运行的部分继续预留。当前没有启动I5，也没有扩展MF6、S或J。

服务器先执行 `python -B e2e/hi_i5.py check`，再执行 `python -B e2e/hi_i5.py launch --manifest e2e/hi/manifest/i5_pilot.json --workers 128`。14项试点完成后用 `python -B e2e/hi_i5_estimate.py --apply` 更新剩余96项耗时/RSS估计；全量启动用 `python -B e2e/hi_i5.py launch --workers 128`，保留正式试点结果并将试点控制记录移入 `out/i5/pilot_run`。
