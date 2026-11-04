"""Markdown summary of one stage 2 part from out/s2/report_<part>.json. Usage: python part_md.py <part>"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
part = sys.argv[1]
r = json.load(open(os.path.join(HERE, "out", "s2", f"report_{part}.json"), encoding="utf-8"))
pct = lambda v: "—" if v is None else f"{100 * v:+.2f}%"
num = lambda v: f"{v:,.0f}"
tri = lambda t: " / ".join(num(x) for x in t)
print("| 配置 | 臂 | 后端最优 [min, max]（运行） | 相对 A1 | 三元组 | 自身代价最优（运行） | 相对 A1 | 三元组 | 14 次中位数 | 相对 A1 | 最差 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for t in r["table"]:
    print(f"| {t['cfg']} | {t['arm']} | {num(t['best_backend'])} [{num(t['min'])}, {num(t['max'])}]（{t['run']}） | {pct(t.get('best_backend_vs_A1'))} | "
          f"{tri(t['best_backend_triple'])} | {num(t['best_own'])}（{t['best_own_run']}） | {pct(t.get('best_own_vs_A1'))} | {tri(t['best_own_triple'])} | "
          f"{num(t['median'])} | {pct(t.get('median_vs_A1'))} | {num(t['worst'])} |")
print()
print("| 配置 | 臂 | 步数 | 最后接受 | 最后刷新最优 | Tmax | T̂ 对后端误差 中位 / 最大 | 缓存 T̂ 对完整广播 | 快照 / 移动 / 挑选 / 广播 / CPCM / 撤销 / 存最优（ms/步） | SA 墙钟 s |")
print("|---|---|---|---|---|---|---|---|---|---|")
for t in r["table"]:
    m = t["ms_per_step"]
    ca = t.get("err_Tsel_vs_Tbc_median")
    print(f"| {t['cfg']} | {t['arm']} | {t['steps']} | {t['last_accept_median']:g} | {t['last_best_median']:g} | {t['Tmax_median']:.5g} | "
          f"{pct(t['err_Tbc_vs_backend_median'])} / {100 * t['err_Tbc_vs_backend_maxabs']:.2f}% | {'—' if ca is None else f'{100 * ca:.2f}%'} | "
          f"{m['t_pre']:g} / {m['t_move']:g} / {m['t_target']:g} / {m['t_bcast']:g} / {m['t_energy']:g} / {m['t_revert']:g} / {m['t_snapshot']:g} | {t['sa_wall_median']:,.0f} |")
print()
print("噪声：" + "，".join(f"{c} {100 * v:.2f}%" for c, v in r["noise_by_config"].items()) + "。")
for c, v in r["E3"].items():
    if c in r["noise_by_config"]:
        print(f"E3 对照 {c}：R0 {num(v['R0'])}，FULL {num(v['FULL'])}。各臂后端最优相对 R0 / FULL：" + "；".join(
            f"{t['arm']} {pct(t.get('vs_E3_R0'))} / {pct(t.get('vs_E3_FULL'))}" for t in r["table"] if t["cfg"] == c) + "。")
ch = r["checks"]
print(f"自检：拒绝后重算 {ch['revert_n']} 次，E / T̂ / 状态不一致 {ch['revert_E']} / {ch['revert_T']} / {ch['revert_state']}；"
      f"自检 2 {ch['check2_n']} 个状态，不一致 {ch['check2_bad']}；自检 4 {ch['check4_n']} 次，不在关键链上 {ch['check4_off']}，T̂ 不同 {ch['check4_T']}。")
c3 = os.path.join(HERE, "out", "s2", f"check3_{part}.json")
if os.path.exists(c3):
    d = json.load(open(c3, encoding="utf-8"))
    print(f"自检 3：{sum(1 for x in d if x.get('equal'))}/{len(d)} 相同。")
