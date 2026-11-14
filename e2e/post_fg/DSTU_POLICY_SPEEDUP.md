# 消息连续策略的等价执行优化

2026-11-13，Codex。本轮正式质量实验继续使用原 `post_fg_dstu_policy.py`；独立实现为 `post_fg_dstu_policy_fast.py`，尚未替换正式清单。

原策略每轮对全部就绪消息重新遍历路径、分组，再在每次发放时扫描候选。构造的288块拥塞小图上，cProfile显示原on_tick累计占约61%，on_issue约5%。只把候选统计提前到每一轮会增加无效工作，因此最终实现采用两项缓存：

- 一个消息只有真正发放过链路后，下一轮才重新计算剩余路径首跳；on_issue标记、on_tick刷新、on_finish清理。多条链路在同轮发放时，trace仍读取事件当时的真实首跳。
- 每轮先保留候选快照和preferred；资源真正发放时才生成parent集合、最大rank、原键计数与最小键。后续查询复用该轮统计。owner仍只在实际发放后更新。

`key()`保持纯查询，不增加随机数调用。回退选择只在剩余路径严格更长时替换，等长时保留本轮shuffle中的首个候选。

## 本机验证

`post_fg_dstu_policy_bench.py smoke`运行真实fast_hop后端，8张有依赖/多分支小图×C4、TieFIFO、MsgCont×trace开关，共48项全部通过。两版本使用相同事件对象、重置其状态，以保留后端集合迭代所用对象身份；逐issue trace、完成时间、资源服务记录、计数、owner/preferred、ready_since及结束时RNG状态逐项一致。

另外，288块、2016次链路发放的构造拥塞图也逐项一致。关闭cProfile的本机单次配对测量为0.320068→0.263978秒，即1.212倍；开启cProfile时为1.542倍，该值只用于定位函数开销。

证据：`evidence/dstu_fast_policy_checks.json`、`evidence/dstu_fast_policy_contention.json`、`evidence/dstu_fast_policy_profile.json`。独立代码审查也已核对路径缓存的失效时机、同轮快照与owner更新。

## 完整WSC图验证

本轮在正式质量任务仍运行时新增一次完整WSC、A256-M、g=0.1、seed1验证。独立入口 `post_fg_dstu_policy_validate.py` 只将该进程的策略类替换为新实现，其他图构造、后端、trace与结果生成均沿用原runner。

- 基准为本轮fresh transform的A256-M完整作业；未用F/G历史A256状态替代它。
- 全部333627条链路发放记录逐条相同，解压内容SHA-256两者均为 `2dcc43366af96bcc4d97ea578687ef2fa4a0457694d785fb2134ff0eeb350c69`。
- 消息首末完成时间、消费者就绪/启动/结束、资源占用汇总全部相同；完成时间、指标、策略计数、audit、图身份等13项结果字段全部相同。
- 新作业后端371.549秒、全流程385.026秒；旧smoke后端399.479秒。这两次所处的并行负载不同，本轮只据此确认完整图等价，不报告隔离环境的加速倍数。
- 仅新增1次后端调用，结果单列于 `out/policy_validation/`，未改正式策略或清单；没有再追加其它完整图。

完整输出已下载并解包，归档为 `dist/policy_validation_20261113.tar.gz`，SHA-256为 `0cef554d861470682ce54930b6007ca81ef46b15553edb572334c62c3dc1a63a`，10个归档成员通过核对。证据：`out/policy_validation/verification.json`、`evidence/dstu_policy_validation_download.json`。历史A256与本轮fresh图构造的集合遍历差异另行处理，不和这份同输入策略实现核对混用。
