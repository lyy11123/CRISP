# H5 Llama 联合验证计划

2026-11-14。使用已经完成的Llama健康候选和H4 PF4导出状态，先执行一个健康A3h+A256的完整联合试点，再据其实际时间与内存安排剩余正式作业。

## 固定的场景

主硬件2×4晶圆上两个不相交2×2组。P侧为prefill MHA的8Q+1KV代表子图及真实跨组Ktranspose/V写回；D侧为完整decode链：一个外置LN、8个MHA头组、projection、FFN，FFN包含第二个RMSNorm。共11段，不另加第二个外置LN。P/D是独立请求，只共享资源，没有把P缓存完成设为本次D链的释放依赖。

相位为0.1、0.5、0.9，后端种子1、2、3。健康A1/original的三个P单跑和三个D单跑给出统一的KV窗口、释放相位与评价基准。这六项属于配对单跑，不另加重复参考。C4内部排序仍沿用当前图的that估值和1.5倍子期限；共同健康A1基准用于评价指标，二者分别保存。

## 条件与数量

健康三臂：A1无DSTU、A3h无DSTU、A3h A256。D组PF4四臂：A1-R0无DSTU、A3h-R0无DSTU、A3h-EIR无DSTU、A3h-EIR A256。每臂3相位×3种子，共63联合。

无DSTU沿用original整包后端，A256沿用全链路256KiB+C4按跳后端。比较的是整套执行方式；优先级自身的作用引用已有同后端消融。

42个逻辑配对P/D单跑按输入身份归并为39个实际单跑：D组同一PF4平台下，A3hR0与A3hEIR的P流及original后端相同，复用3个种子的P单跑。运行时还会核对平台签名、完整有序事件字段、实际集合遍历次序。健康与故障的P流不互相复用。

共110个正式调度作业：2个P图导出、5种组合图、39单跑、1公共参考、63联合。两种P输入都从完整14候选选优：A1为g0/l4，A3h为g0/seed123。D状态直接使用H4健康、R0或stop300导出的最终代表，不另挑修复种子或重新搜索。

## 入口与模型适配

新入口 `hi_h5.py` 复用fg_joint/d_joint图构造与后端，只在当前进程绑定hi_config、合同D链和精确KV张量识别。图构造显式建立左右组各自的local_hp并核对硬件，不调用旧probe里local_hp可能未赋值的分支。Ktranspose/V由生产者tag和真实区域字节识别，跨组重定向后验证载荷字节守恒。

输入清单 `manifest/h5_inputs.json`，完整任务 `manifest/h5.json`；输出 `out/h5`。图报告保留实际事件与块数、共享资源数、KV逻辑/链路字节、切块统计、完整D链结构；各结果保留原fg_joint simulation/metrics、配对单跑、公共参考及tau。

## 正式试点与资源

试点为健康A3h A256，g=0.5、seed1。其依赖闭包14作业：2导出、2图、8单跑（6个公共A1参考+2个本臂配对）、1参考汇总、1联合。全部使用正式ID，完成后继续全量时复用，不再运行一遍。当前清单中的时间与3GiB内存提示仅作启动占位，正式剩余耗时在试点完成后依据实测更新。

每轮给H3/I2、I3、H4、I4轻批和WATOS decode FFN补充批所有未完成作业预留名额。新池只用剩余槽，合计最多128进程；扣除各池未来增长预算后给系统保留3GiB。

```text
python e2e/hi_h5.py check
python e2e/hi_h5.py launch --manifest e2e/hi/manifest/h5_pilot.json
```

试点完成并报告实际时间/RSS后，主控才启动全量。由试点改为完整manifest时，旧controller start/status/batch/finish移入 `out/h5/pilot_run`，完成的receipt/spec/结果留原位，日志追加；不会把14项试点结束误写成110项全部结束。
