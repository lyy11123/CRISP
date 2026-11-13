# F/G 已完成候选汇总

读取 3080/3080 条结果；核对问题 0 项。

检查方式：核对结果JSON与快照文件，并解析工作量和静态资源工作。

每条结果的后端种子数为 [10]。十四候选齐备才计算后端选优、候选中位数与随机平局 CPCM 选优；单候选小批仅列该候选的表现。

层公式采用 E fixed：prefill = 2 LN + 8 MHA + proj + FFN；decode = LN + 8 MHA + proj + FFN，其中 decode FFN 已含一次 LN。MHA 微图八头、总层乘八，共六十四头。

下表为一个 Die 组上的单层组装时间，单位微秒；请求时间仍固定 KV 长度 2048。

| 配置 | 实验 | 故障 | 方法 | 候选处理 | TTFT / μs | TPOT / μs |
|---|---|---|---|---|---:|---:|
| E_busybarn_2x2_fixed_d2d1 | healthy | healthy | A1 | backend_best | 105491.379 | 352.135 |
| E_busybarn_2x2_fixed_d2d1 | healthy | healthy | A1 | candidate_median | 192694.797 | 473.842 |
| E_busybarn_2x2_fixed_d2d1 | healthy | healthy | A1 | random_cpcm | 105491.379 | 352.135 |
| E_busybarn_2x2_fixed_d2d1 | healthy | healthy | A1 | deterministic_cpcm | 105491.379 | 352.135 |
| E_busybarn_2x2_fixed_d2d1 | healthy | healthy | A3h | backend_best | 104842.526 | 237.544 |
| E_busybarn_2x2_fixed_d2d1 | healthy | healthy | A3h | candidate_median | 135715.889 | 291.598 |
| E_busybarn_2x2_fixed_d2d1 | healthy | healthy | A3h | random_cpcm | 104860.121 | 237.952 |
| E_busybarn_2x2_fixed_d2d1 | healthy | healthy | A3h | deterministic_cpcm | 104888.404 | 239.828 |
| E_busybarn_2x2_fixed_d2d1 | healthy | healthy | Gemini | backend_best | 232995.575 | 439.287 |
| E_busybarn_2x2_fixed_d2d1 | healthy | healthy | Gemini | candidate_median | 260173.147 | 538.313 |
| E_busybarn_2x2_fixed_d2d1 | healthy | healthy | Gemini | random_cpcm | 233006.065 | 439.287 |
| E_busybarn_2x2_fixed_d2d1 | healthy | healthy | Gemini | deterministic_cpcm | 234322.033 | 439.287 |
| F_face_2x2_fixed_d2d1 | full | PF4 | A3h | backend_best | 95431.817 | 213.266 |
| F_face_2x2_fixed_d2d1 | full | PF4 | A3h | candidate_median | 125161.499 | 255.267 |
| F_face_2x2_fixed_d2d1 | full | PF4 | A3h | random_cpcm | 95431.817 | 213.450 |
| F_face_2x2_fixed_d2d1 | full | PF4 | A3h | deterministic_cpcm | 96407.406 | 214.337 |
| F_face_2x2_fixed_d2d1 | healthy | healthy | A1 | backend_best | 96391.512 | 346.089 |
| F_face_2x2_fixed_d2d1 | healthy | healthy | A1 | candidate_median | 173523.802 | 396.823 |
| F_face_2x2_fixed_d2d1 | healthy | healthy | A1 | random_cpcm | 97203.815 | 350.390 |
| F_face_2x2_fixed_d2d1 | healthy | healthy | A1 | deterministic_cpcm | 96994.781 | 350.390 |
| F_face_2x2_fixed_d2d1 | healthy | healthy | A3h | backend_best | 95443.203 | 211.943 |
| F_face_2x2_fixed_d2d1 | healthy | healthy | A3h | candidate_median | 122516.614 | 253.594 |
| F_face_2x2_fixed_d2d1 | healthy | healthy | A3h | random_cpcm | 95471.050 | 213.578 |
| F_face_2x2_fixed_d2d1 | healthy | healthy | A3h | deterministic_cpcm | 95777.593 | 211.954 |
| F_face_2x2_fixed_d2d1 | healthy | healthy | Gemini | backend_best | 191675.665 | 355.052 |
| F_face_2x2_fixed_d2d1 | healthy | healthy | Gemini | candidate_median | 220829.617 | 431.474 |
| F_face_2x2_fixed_d2d1 | healthy | healthy | Gemini | random_cpcm | 192112.306 | 358.205 |
| F_face_2x2_fixed_d2d1 | healthy | healthy | Gemini | deterministic_cpcm | 195556.617 | 361.306 |
| F_watos_2x2_fixed_d2d1 | full | PF4 | A3h | backend_best | 60120.117 | 204.361 |
| F_watos_2x2_fixed_d2d1 | full | PF4 | A3h | candidate_median | 87050.534 | 262.550 |
| F_watos_2x2_fixed_d2d1 | full | PF4 | A3h | random_cpcm | 60148.984 | 204.361 |
| F_watos_2x2_fixed_d2d1 | full | PF4 | A3h | deterministic_cpcm | 62266.296 | 204.874 |
| F_watos_2x2_fixed_d2d1 | healthy | healthy | A1 | backend_best | 61842.795 | 332.443 |
| F_watos_2x2_fixed_d2d1 | healthy | healthy | A1 | candidate_median | 96228.826 | 413.056 |
| F_watos_2x2_fixed_d2d1 | healthy | healthy | A1 | random_cpcm | 61842.795 | 336.805 |
| F_watos_2x2_fixed_d2d1 | healthy | healthy | A1 | deterministic_cpcm | 61842.795 | 332.562 |
| F_watos_2x2_fixed_d2d1 | healthy | healthy | A3h | backend_best | 57209.404 | 194.839 |
| F_watos_2x2_fixed_d2d1 | healthy | healthy | A3h | candidate_median | 85708.308 | 260.332 |
| F_watos_2x2_fixed_d2d1 | healthy | healthy | A3h | random_cpcm | 57227.571 | 194.839 |
| F_watos_2x2_fixed_d2d1 | healthy | healthy | A3h | deterministic_cpcm | 58375.176 | 204.833 |
| F_watos_2x2_fixed_d2d1 | healthy | healthy | Gemini | backend_best | 119381.416 | 409.670 |
| F_watos_2x2_fixed_d2d1 | healthy | healthy | Gemini | candidate_median | 127948.427 | 496.066 |
| F_watos_2x2_fixed_d2d1 | healthy | healthy | Gemini | random_cpcm | 119769.683 | 409.670 |
| F_watos_2x2_fixed_d2d1 | healthy | healthy | Gemini | deterministic_cpcm | 122092.732 | 409.670 |
| F_wsc_2x2_fixed_d2d1 | full | PF4 | A3h | backend_best | 92890.235 | 243.498 |
| F_wsc_2x2_fixed_d2d1 | full | PF4 | A3h | candidate_median | 177939.728 | 313.827 |
| F_wsc_2x2_fixed_d2d1 | full | PF4 | A3h | random_cpcm | 92954.665 | 243.498 |
| F_wsc_2x2_fixed_d2d1 | full | PF4 | A3h | deterministic_cpcm | 93202.145 | 249.087 |
| F_wsc_2x2_fixed_d2d1 | healthy | healthy | A1 | backend_best | 99796.643 | 534.151 |
| F_wsc_2x2_fixed_d2d1 | healthy | healthy | A1 | candidate_median | 162559.636 | 725.685 |
| F_wsc_2x2_fixed_d2d1 | healthy | healthy | A1 | random_cpcm | 99804.750 | 534.151 |
| F_wsc_2x2_fixed_d2d1 | healthy | healthy | A1 | deterministic_cpcm | 99804.750 | 534.151 |
| F_wsc_2x2_fixed_d2d1 | healthy | healthy | A3h | backend_best | 90203.686 | 238.781 |
| F_wsc_2x2_fixed_d2d1 | healthy | healthy | A3h | candidate_median | 133127.334 | 294.599 |
| F_wsc_2x2_fixed_d2d1 | healthy | healthy | A3h | random_cpcm | 90203.686 | 238.781 |
| F_wsc_2x2_fixed_d2d1 | healthy | healthy | A3h | deterministic_cpcm | 91727.931 | 244.043 |
| F_wsc_2x2_fixed_d2d1 | healthy | healthy | Gemini | backend_best | 196626.232 | 482.852 |
| F_wsc_2x2_fixed_d2d1 | healthy | healthy | Gemini | candidate_median | 230280.694 | 565.752 |
| F_wsc_2x2_fixed_d2d1 | healthy | healthy | Gemini | random_cpcm | 196626.232 | 482.852 |
| F_wsc_2x2_fixed_d2d1 | healthy | healthy | Gemini | deterministic_cpcm | 196985.593 | 500.933 |
| G_busybarn_2x3_fixed_d2d1 | healthy | healthy | A1 | backend_best | 113207.657 | 349.795 |
| G_busybarn_2x3_fixed_d2d1 | healthy | healthy | A1 | candidate_median | 185591.139 | 431.578 |
| G_busybarn_2x3_fixed_d2d1 | healthy | healthy | A1 | random_cpcm | 113462.373 | 354.384 |
| G_busybarn_2x3_fixed_d2d1 | healthy | healthy | A1 | deterministic_cpcm | 113637.078 | 354.384 |
| G_busybarn_2x3_fixed_d2d1 | healthy | healthy | A3h | backend_best | 92834.755 | 199.338 |
| G_busybarn_2x3_fixed_d2d1 | healthy | healthy | A3h | candidate_median | 123752.705 | 260.759 |
| G_busybarn_2x3_fixed_d2d1 | healthy | healthy | A3h | random_cpcm | 93257.393 | 199.338 |
| G_busybarn_2x3_fixed_d2d1 | healthy | healthy | A3h | deterministic_cpcm | 93598.703 | 200.393 |
| G_busybarn_2x3_fixed_d2d1 | healthy | healthy | Gemini | backend_best | 229021.744 | 388.294 |
| G_busybarn_2x3_fixed_d2d1 | healthy | healthy | Gemini | candidate_median | 261137.856 | 468.110 |
| G_busybarn_2x3_fixed_d2d1 | healthy | healthy | Gemini | random_cpcm | 229021.744 | 388.294 |
| G_busybarn_2x3_fixed_d2d1 | healthy | healthy | Gemini | deterministic_cpcm | 229021.744 | 388.294 |
| G_busybarn_2x3_growth_d2d1 | full | PF4 | A3h | backend_best | 77978.761 | 195.236 |
| G_busybarn_2x3_growth_d2d1 | full | PF4 | A3h | candidate_median | 104160.371 | 255.208 |
| G_busybarn_2x3_growth_d2d1 | full | PF4 | A3h | random_cpcm | 77978.761 | 195.236 |
| G_busybarn_2x3_growth_d2d1 | full | PF4 | A3h | deterministic_cpcm | 78382.677 | 200.767 |
| G_busybarn_2x3_growth_d2d1 | healthy | healthy | A1 | backend_best | 77319.955 | 321.890 |
| G_busybarn_2x3_growth_d2d1 | healthy | healthy | A1 | candidate_median | 150226.311 | 404.676 |
| G_busybarn_2x3_growth_d2d1 | healthy | healthy | A1 | random_cpcm | 77339.400 | 322.882 |
| G_busybarn_2x3_growth_d2d1 | healthy | healthy | A1 | deterministic_cpcm | 78588.242 | 321.890 |
| G_busybarn_2x3_growth_d2d1 | healthy | healthy | A3h | backend_best | 77516.326 | 187.104 |
| G_busybarn_2x3_growth_d2d1 | healthy | healthy | A3h | candidate_median | 112498.640 | 250.457 |
| G_busybarn_2x3_growth_d2d1 | healthy | healthy | A3h | random_cpcm | 77535.976 | 187.104 |
| G_busybarn_2x3_growth_d2d1 | healthy | healthy | A3h | deterministic_cpcm | 78344.282 | 188.521 |
| G_busybarn_2x3_growth_d2d1 | healthy | healthy | Gemini | backend_best | 175508.727 | 372.575 |
| G_busybarn_2x3_growth_d2d1 | healthy | healthy | Gemini | candidate_median | 201246.139 | 418.150 |
| G_busybarn_2x3_growth_d2d1 | healthy | healthy | Gemini | random_cpcm | 175512.102 | 372.575 |
| G_busybarn_2x3_growth_d2d1 | healthy | healthy | Gemini | deterministic_cpcm | 178287.715 | 379.927 |
| G_busybarn_3x3_fixed_d2d1 | healthy | healthy | A1 | backend_best | 89664.184 | 308.596 |
| G_busybarn_3x3_fixed_d2d1 | healthy | healthy | A1 | candidate_median | 162674.032 | 379.664 |
| G_busybarn_3x3_fixed_d2d1 | healthy | healthy | A1 | random_cpcm | 90110.874 | 308.596 |
| G_busybarn_3x3_fixed_d2d1 | healthy | healthy | A1 | deterministic_cpcm | 89841.231 | 308.813 |
| G_busybarn_3x3_fixed_d2d1 | healthy | healthy | A3h | backend_best | 84733.170 | 173.508 |
| G_busybarn_3x3_fixed_d2d1 | healthy | healthy | A3h | candidate_median | 117373.845 | 235.611 |
| G_busybarn_3x3_fixed_d2d1 | healthy | healthy | A3h | random_cpcm | 84974.011 | 173.508 |
| G_busybarn_3x3_fixed_d2d1 | healthy | healthy | A3h | deterministic_cpcm | 84792.550 | 179.569 |
| G_busybarn_3x3_fixed_d2d1 | healthy | healthy | Gemini | backend_best | 181459.892 | 364.691 |
| G_busybarn_3x3_fixed_d2d1 | healthy | healthy | Gemini | candidate_median | 192281.628 | 406.949 |
| G_busybarn_3x3_fixed_d2d1 | healthy | healthy | Gemini | random_cpcm | 181459.892 | 364.691 |
| G_busybarn_3x3_fixed_d2d1 | healthy | healthy | Gemini | deterministic_cpcm | 181734.429 | 364.691 |
| G_busybarn_3x3_growth_d2d1 | full | PF4 | A3h | backend_best | 57427.944 | 170.242 |
| G_busybarn_3x3_growth_d2d1 | full | PF4 | A3h | candidate_median | 79120.032 | 231.970 |
| G_busybarn_3x3_growth_d2d1 | full | PF4 | A3h | random_cpcm | 57639.866 | 170.242 |
| G_busybarn_3x3_growth_d2d1 | full | PF4 | A3h | deterministic_cpcm | 57653.952 | 170.378 |
| G_busybarn_3x3_growth_d2d1 | healthy | healthy | A1 | backend_best | 56745.284 | 274.289 |
| G_busybarn_3x3_growth_d2d1 | healthy | healthy | A1 | candidate_median | 101622.280 | 359.098 |
| G_busybarn_3x3_growth_d2d1 | healthy | healthy | A1 | random_cpcm | 56759.170 | 274.289 |
| G_busybarn_3x3_growth_d2d1 | healthy | healthy | A1 | deterministic_cpcm | 56759.170 | 274.289 |
| G_busybarn_3x3_growth_d2d1 | healthy | healthy | A3h | backend_best | 56072.734 | 166.386 |
| G_busybarn_3x3_growth_d2d1 | healthy | healthy | A3h | candidate_median | 73233.002 | 227.141 |
| G_busybarn_3x3_growth_d2d1 | healthy | healthy | A3h | random_cpcm | 56074.290 | 166.386 |
| G_busybarn_3x3_growth_d2d1 | healthy | healthy | A3h | deterministic_cpcm | 56734.334 | 166.679 |
| G_busybarn_3x3_growth_d2d1 | healthy | healthy | Gemini | backend_best | 121480.891 | 314.333 |
| G_busybarn_3x3_growth_d2d1 | healthy | healthy | Gemini | candidate_median | 144360.173 | 365.087 |
| G_busybarn_3x3_growth_d2d1 | healthy | healthy | Gemini | random_cpcm | 122233.993 | 314.409 |
| G_busybarn_3x3_growth_d2d1 | healthy | healthy | Gemini | deterministic_cpcm | 122258.272 | 314.333 |

保存结果累计耗时 940.98 进程小时，单进程最高峰值 4.646 GiB。
按每作业唯一成功receipt计时，共6813项、1037.15进程小时；归档失败尝试2次、0.12进程小时，单独统计。多次控制器记录不相加。

完整模式中的工作量与取整量由保存快照解析：有效乘加量为各 TensorCore 事件 2×M×K×N 之和；取整量应用原 128×64 grain。matmac 的额外偏置加法未计入该乘加量。归约单列 matadd/redsum/redmax 的事件与输入元素。通信、路径和事件图指标来自运行记录。

后端三元组分解为总时间、只有计算活动的时间、只有通信活动的时间；余项是计算与通信同时活动的时间。静态资源负载是保存事件的服务时间累加，未重建链路排队时间。
