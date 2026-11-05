"""Phase B report (EIR_NEXT_HANDOFF.md sections 6, 8; rule in PROGRESS step 25). Replays stored data, no new runs. Reads out/next/b
(60 checkpoint runs), out/next/b_serial (serial timing), the stored l1, pre, independent runs and FULL-A3h at F4a. Writes
out/next/report_b.md, report_b.json.

  python report_b.py

Times: T in cycles, clock in ns, search in seconds. Weight copy: bytes / 256 cycles, one uncontended stream.
"""
import heapq
import json
import os
import statistics as st

import eir
import eir_next as en

OUT = en.OUT
FAULT = en.B_FAULT
METHODS = en.B_METHODS
SEEDS = en.B_SEEDS
TIERS = (30, 100, 300)
LABEL = {"L2Lsync": "L2L+sync", "L2Impact": "L2Impact", "L2Gsync": "L2G+sync"}
HS = (10, 60, 600, 3600, 86400)
RS = (1, 8)
CLOCKS = (0.5, 1.0)
SCALES = (1.0, 0.1)
SWITCHES = (0.0, 1e-3, 0.1, 10.0)
EPS = 0.02


def bk(b):
    return st.mean(v[0] for v in b.values())


def med(x):
    return st.median(list(x))


def f0(x):
    return f"{x:,.0f}"


def sec(x):
    if x < 1e-3:
        return f"{x * 1e6:.1f} µs"
    if x < 1:
        return f"{x * 1e3:.2f} ms"
    return f"{x:,.1f} s"


def moved_share(ref, now):
    return sum(tuple(v[:-1]) != tuple(now[k][:-1]) for k, v in ref.items()) / len(ref)


def load():
    D = {"pre": {}, "l1": {}, "ck": {}, "serial": {}, "full": {}}
    for o in en.OPS:
        p = eir.read_sink(eir.job_pre(en.MAIN, "a3h", o)["sink"])
        D["pre"][o] = {"T": p["record"]["T"], "bk": bk(p["record"]["backend"]), "weight_bytes": sum(p["sizes"].values())}
        l = eir.read_sink(en.job_l1(o, FAULT, out=en.l1_out(FAULT))["sink"])
        D["l1"][o] = {"T_R0": l["T_R0"], "T_L1": l["T_L1"], "bk_R0": bk(l["R0"]["backend"]), "bk_L1": bk(l["L1"]["backend"]),
                      "mig_L1": l["L1"].get("migration_vs_R0", {}).get("unique_block_bytes", 0),
                      "rho_L1": moved_share(p["record"]["beha_loc"], l["L1"]["beha_loc"])}
        for m in METHODS:
            for sd in SEEDS:
                d = eir.read_sink(en.b_job(o, m, sd)["sink"])
                D["ck"][o, m, sd] = d
        rows = []
        for g, x in eir.cs.variants("A3h"):
            s = eir.read_sink(eir.job_full(OUT, o, FAULT, "A3h", g, x)["sink"])
            sel = s["selected"]
            rows.append({"run": f"g{g}_{x}", "bk": bk(sel["backend"]), "T": sel["T_sel"], "own": sel["own_cost"], "sa_wall": s["sa_wall"]})
        D["full"][o] = rows
    with open(os.path.join(OUT, "b_serial", "serial.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    for o in en.OPS:
        l1s = [eir.read_sink(r["sink"]) for r in rows if r["op"] == o and r["what"] == "l1"]
        sr = {"t_R0": med(x["t_R0"] for x in l1s), "t_L1": med(x["t_L1"] for x in l1s), "t_R0_all": [x["t_R0"] for x in l1s], "t_L1_all": [x["t_L1"] for x in l1s]}
        for r in rows:
            if r["op"] != o:
                continue
            if r["what"] in METHODS:
                t = eir.read_sink(r["sink"])["timing"]
                sr[r["what"]] = {"t_cal": t["t_calibration"], "t_100": t["t_L2_search"], "t_fin": t["t_finalize"], "t_backend": t["t_backend_offline"], "wall": r["wall_s"]}
            elif r["what"] == "full":
                sr["full_sa"] = eir.read_sink(r["sink"])["sa_wall"]
                sr["full_wall"] = r["wall_s"]
        D["serial"][o] = sr
    return D


def self_check(D):
    bad = {"T_ne_best": 0, "last_ne_record": 0, "viol": 0, "stale": 0, "not_monotone": 0, "seen": 0}
    for d in D["ck"].values():
        rows = d["ck"]
        bad["seen"] += 1
        bad["T_ne_best"] += sum(x["T"] != x["best_T"] for x in rows)
        bad["last_ne_record"] += rows[-1]["T"] != d["record"]["T"]
        bad["viol"] += sum(x["fault_violations"] for x in rows)
        bad["stale"] += sum(x["stale"] for x in rows)
        bad["not_monotone"] += any(a["best_T"] < b["best_T"] for a, b in zip(rows, rows[1:]))
    return bad


# ---------------------------------------------------------------- handoff
def dijkstra(link_time, src):
    adj = {}
    for (u, v), (t0, _) in ((tuple(map(tuple, k)), x) for k, x in link_time.items()):
        adj.setdefault(u, []).append((v, t0))
    dist, h = {src: 0.0}, [(0.0, src)]
    while h:
        du, u = heapq.heappop(h)
        if du > dist[u]:
            continue
        for v, w in adj.get(u, ()):
            if du + w < dist.get(v, 1e30):
                dist[v] = du + w
                heapq.heappush(h, (du + w, v))
    return dist


def handoff(D, m, sd, i):
    """No-contention direct transfer: FFN's ifmap reads from the core of MHA's last MHA_p behaviour. See PROGRESS step 25."""
    f, a = D["ck"]["ffn", m, sd], D["ck"]["mha", m, sd]
    src = tuple(a["ck"][i]["handoff"]["output_locs"][0][:2])
    dist = dijkstra(f["link_time"], src)
    return max(dist[tuple(t[:2])] + b / 256 for t, b in f["ck"][i]["handoff"]["ifmap_reads"])


# ---------------------------------------------------------------- options
def tier_table(D):
    """(method, seed) -> list of tier rows for both tasks: T̂, backend, bytes against R0."""
    out = {}
    for m in METHODS:
        for sd in SEEDS:
            rows = [{"tier": "L1", "s": 0, "That": {o: D["l1"][o]["T_L1"] for o in en.OPS}, "bk": {o: D["l1"][o]["bk_L1"] for o in en.OPS}, "bytes": {o: D["l1"][o]["mig_L1"] for o in en.OPS}, "rho": D["l1"]["mha"]["rho_L1"]}]
            for i, s in enumerate(TIERS):
                rows.append({"tier": f"s{s}", "s": s, "That": {o: D["ck"][o, m, sd]["ck"][i]["best_T"] for o in en.OPS},
                             "bk": {o: bk(D["ck"][o, m, sd]["ck"][i]["backend"]) for o in en.OPS},
                             "bytes": {o: D["ck"][o, m, sd]["ck"][i]["migration_vs_R0"]["unique_block_bytes"] for o in en.OPS}, "h": handoff(D, m, sd, i),
                             "rho": D["ck"]["mha", m, sd]["ck"][i]["migration_vs_start"]["moved_behas_share"]})
            out[m, sd] = rows
    return out


def t_tier(D, o, m, s, scale):
    """Decision seconds of one task up to step s of this method's cooling (serial magnitude; L2 part in proportion to the steps)."""
    sr = D["serial"][o]
    t = sr["t_R0"] + sr["t_L1"]
    if s:
        x = sr[m]
        t += x["t_cal"] + x["t_100"] * s / 100
    return t * scale


def t_copy(bytes_, clock):
    return bytes_ / 256 * clock * 1e-9


def round_time(tm, tf, r, h, clock):
    return (r * tm + h + tf) * clock * 1e-9


def n_main(H, t, c, Ta, T0):
    return min(t, H) / T0 + max(H - t - c, 0) / Ta


def n_down(H, A, Ta):
    return int(max(H - A, 0) // Ta)


def best_T0(D, r, h, clock):
    return round_time(D["l1"]["mha"]["bk_R0"], D["l1"]["ffn"]["bk_R0"], r, h, clock)


# ---------------------------------------------------------------- deciders
def window_rule(rows, D, m, H, r, h_of, clock, scale, sw, version, M=0.0):
    """PROGRESS step 25. Joint tier, T̂ only, returns the index of the row deployed (the rule starts at L1). M: KV bytes of one
    MHA task, added after the rule was fixed (report_kv.py); 0 leaves the rule as written."""
    T0hat = round_time(D["l1"]["mha"]["T_R0"], D["l1"]["ffn"]["T_R0"], r, 0, clock)

    def cost(row):
        return t_copy(sum(row["bytes"].values()) + row["rho"] * M, clock) + sw

    def tall(s):
        return sum(t_tier(D, o, m, s, scale) for o in en.OPS)

    def n(t, c, Ta):
        return n_main(H, t, c, Ta, T0hat) if version == "main" else max(H - t - c, 0) / Ta

    def rt(That):
        return round_time(That["mha"], That["ffn"], r, 0, clock)

    prev = {o: D["l1"][o]["T_R0"] for o in en.OPS}
    i = 0
    while True:
        row = rows[i]
        now = n(tall(row["s"]), cost(row), rt(row["That"]))
        if i + 1 == len(rows):
            return i
        nxt = rows[i + 1]
        est = {o: row["That"][o] - max(0.0, prev[o] - row["That"][o]) for o in en.OPS}
        cont = n(tall(nxt["s"]), cost(row), rt(est))
        if cont <= now:
            return i
        prev = row["That"]
        i += 1


def fixed_rule(rows, D, o):
    """EIR_SUMMARY section 8, one task. Index into rows; -1 = R0."""
    thr = D["pre"][o]["T"] * (1 + EPS)
    if D["l1"][o]["T_R0"] <= thr:
        return -1
    if rows[0]["That"][o] <= thr:
        return 0
    i = 1
    while True:
        if rows[i]["That"][o] <= thr or i + 1 == len(rows):
            return i
        if i > 1 and (rows[i - 1]["That"][o] - rows[i]["That"][o]) / rows[i - 1]["That"][o] < EPS:
            return i
        i += 1


def realised(rows, D, m, idx, r, h, clock, scale, sw, H, version, M=0.0):
    """idx: dict task -> tier index (-1 = R0). All tasks are deployed together when the last search ends."""
    bk_ = {o: (D["l1"][o]["bk_R0"] if idx[o] < 0 else rows[idx[o]]["bk"][o]) for o in en.OPS}
    t = sum(t_tier(D, o, m, rows[max(idx[o], 0)]["s"], scale) if idx[o] >= 0 else D["serial"][o]["t_R0"] * scale for o in en.OPS)
    by = sum(rows[idx[o]]["bytes"][o] if idx[o] >= 0 else 0 for o in en.OPS) + (rows[idx["mha"]]["rho"] * M if idx["mha"] >= 0 else 0.0)
    Ta = round_time(bk_["mha"], bk_["ffn"], r, h, clock)
    T0 = best_T0(D, r, h, clock)
    if all(i < 0 for i in idx.values()):
        return H / T0 if version == "main" else n_down(H, t + sw, T0)
    c = t_copy(by, clock) + sw
    return n_main(H, t, c, Ta, T0) if version == "main" else n_down(H, t + c, Ta)


def full_options(D, clock):
    """FULL-A3h: T̂-chosen of 14 runs (online), median single run as the reference. Copy: all weights, an upper bound."""
    by = sum(D["pre"][o]["weight_bytes"] for o in en.OPS)
    picks = {o: min(D["full"][o], key=lambda x: (x["T"], x["own"])) for o in en.OPS}
    best = {o: min(D["full"][o], key=lambda x: x["bk"]) for o in en.OPS}
    single = {o: med(x["bk"] for x in D["full"][o]) for o in en.OPS}
    t1 = sum(D["serial"][o]["full_sa"] for o in en.OPS)
    return {"picks": picks, "best": best, "single": single, "t_single": t1, "bytes": by, "copy": t_copy(by, clock)}


def main():
    D = load()
    chk = self_check(D)
    tiers = tier_table(D)
    res = {"self_check": chk, "serial": D["serial"]}
    L = ["# EIR 下一阶段 阶段 B：双阶段时间轴与窗口决策\n",
         f"只报数字。A3h 起点，共享物理链路故障 `{FAULT}` = (0,13)↔(2,1)。{len(D['ck'])} 次检查点运行，s_max = 300，检查点 30、100、300，每点后端 10 个种子。**局限：一个物理故障、10 个种子配对、耗时是 Python 实现的串行量级，只作示范。**\n",
         f"自检：检查点 T ≠ best_T {chk['T_ne_best']}；最后一档 ≠ 该次记录 {chk['last_ne_record']}；故障违规 {chk['viol']}；过时 {chk['stale']}；best_T 不单调 {chk['not_monotone']}（共 {chk['seen']} 次运行）。\n"]
    # ---- per tier
    L.append("## 各档的 T̂、后端和迁移（10 个种子中位数）\n")
    L.append("| 任务 | 方法 | 档 | T̂ | 后端 | 相对故障前 | 唯一块 MB（相对 R0） | t_search 负载下 s |\n|---|---|---|---|---|---|---|---|")
    table = {}
    for o in en.OPS:
        pre, l = D["pre"][o], D["l1"][o]
        L.append(f"| {o} | — | 故障前 | {f0(pre['T'])} | {f0(pre['bk'])} | — | — | — |")
        L.append(f"| {o} | — | R0 | {f0(l['T_R0'])} | {f0(l['bk_R0'])} | {(l['bk_R0'] / pre['bk'] - 1) * 100:+.1f}% | 0.0 | — |")
        L.append(f"| {o} | — | L1 | {f0(l['T_L1'])} | {f0(l['bk_L1'])} | {(l['bk_L1'] / pre['bk'] - 1) * 100:+.1f}% | {l['mig_L1'] / 1e6:.1f} | — |")
        for m in METHODS:
            for i, s in enumerate(TIERS):
                rows = [D["ck"][o, m, sd]["ck"][i] for sd in SEEDS]
                x = {"That": med(r["best_T"] for r in rows), "bk": med(bk(r["backend"]) for r in rows), "bytes": med(r["migration_vs_R0"]["unique_block_bytes"] for r in rows), "t": med(r["t_search"] for r in rows),
                     "bk_min": min(bk(r["backend"]) for r in rows), "bk_max": max(bk(r["backend"]) for r in rows)}
                table[f"{o}_{m}_{s}"] = x
                L.append(f"| {o} | {LABEL[m]} | s={s} | {f0(x['That'])} | {f0(x['bk'])}（{f0(x['bk_min'])}–{f0(x['bk_max'])}） | {(x['bk'] / pre['bk'] - 1) * 100:+.1f}% | {x['bytes'] / 1e6:.1f} | {x['t']:.1f} |")
    res["tiers"] = table
    # ---- checkpoint vs independent
    L.append("\n## 在线检查点与独立运行的差距\n")
    L.append("独立 `+sync` 只有 s = 100，**没有 s = 30、300 的独立 `+sync`**。下面第一张表配对比较 s = 100（同种子）；第二张表把旧版 L2L、L2G（不带 sync，5 个种子）的独立 s = 30、100、300 放在一起，方法版本不同，只作量级参照。\n")
    L.append("| 任务 | 方法 | n | 检查点 s=100 后端中位数 | 独立 s=100 后端中位数 | 中位差（检查点 − 独立） | 中位差 / 故障前 | 胜 / 负 | 检查点 T̂ 中位数 | 独立 T̂ 中位数 |\n|---|---|---|---|---|---|---|---|---|---|")
    gap = {}
    for o in en.OPS:
        for m in METHODS:
            a, b, ta, tb = [], [], [], []
            for sd in SEEDS:
                p = en.job_rep(o, FAULT, m, sd)["sink"]
                if os.path.exists(p):
                    d = eir.read_sink(p)
                    a.append(bk(D["ck"][o, m, sd]["ck"][1]["backend"]))
                    b.append(bk(d["record"]["backend"]))
                    ta.append(D["ck"][o, m, sd]["ck"][1]["best_T"])
                    tb.append(d["record"]["T"])
            if not a:
                L.append(f"| {o} | {LABEL[m]} | 0 | — | 没有同种子独立运行 | — | — | — | — | — |")
                continue
            df = [x - y for x, y in zip(a, b)]
            gap[f"{o}_{m}"] = {"n": len(a), "ck": med(a), "ind": med(b), "diff": med(df), "ratio": med(df) / D["pre"][o]["bk"], "win": sum(x < 0 for x in df), "loss": sum(x > 0 for x in df), "T_ck": med(ta), "T_ind": med(tb)}
            g = gap[f"{o}_{m}"]
            L.append(f"| {o} | {LABEL[m]} | {g['n']} | {f0(g['ck'])} | {f0(g['ind'])} | {g['diff']:+,.0f} | {g['ratio'] * 100:+.1f}% | {g['win']} / {g['loss']} | {f0(g['T_ck'])} | {f0(g['T_ind'])} |")
    res["ck_vs_ind"] = gap
    L.append("\n| 任务 | 旧方法（独立，5 个种子） | s | 后端中位数 | 相对故障前 |\n|---|---|---|---|---|")
    for o in en.OPS:
        for m in ("L2L", "L2G"):
            for s in TIERS:
                vals = []
                for sd in eir.SA_SEEDS:
                    p = eir.job_rep(en.MAIN, "a3h", o, FAULT, m, s, sd)["sink"]
                    if os.path.exists(p):
                        vals.append(bk(eir.read_sink(p)["record"]["backend"]))
                if vals:
                    L.append(f"| {o} | {m} | {s} | {f0(med(vals))} | {(med(vals) / D['pre'][o]['bk'] - 1) * 100:+.1f}% |")
    # ---- serial
    L.append("\n## 串行抽测（单进程、机器空闲，只摸量级，不是论文数据）\n")
    L.append("| 任务 | t_R0 s（两次） | t_L1 s（两次） | 方法 | 校准 s | s=100 搜索 s | 恢复与最终 CPCM s | 后端 10 种子 s | FULL 1,000 步 SA s |\n|---|---|---|---|---|---|---|---|---|")
    for o in en.OPS:
        sr = D["serial"][o]
        for m in METHODS:
            x = sr[m]
            L.append(f"| {o} | {' / '.join(f'{v:.2f}' for v in sr['t_R0_all'])} | {' / '.join(f'{v:.2f}' for v in sr['t_L1_all'])} | {LABEL[m]} | {x['t_cal']:.1f} | {x['t_100']:.1f} | {x['t_fin']:.2f} | {x['t_backend']:.1f} | {sr['full_sa']:.1f} |")
    # ---- timeline
    L.append("\n## 恢复时间轴（中位种子量级，主用时钟 0.5 ns，T_交接 = 0，T_switch = 0）\n")
    L.append("每个方案：两任务依次搜索的 t_search（串行量级，L2 按步数比例），R0 在搜索期间继续服务。T_a = r·T_MHA + T_交接 + T_FFN，后端 10 个种子。FULL 的 T_copy 按全部权重算，是上界。\n")
    timeline = {}
    fo = full_options(D, 0.5)
    h_vals = [handoff(D, m, sd, i) for m in METHODS for sd in SEEDS for i in range(3)]
    h_med = med(h_vals)
    res["handoff_est"] = {"median": h_med, "min": min(h_vals), "max": max(h_vals), "ifmap_read_bytes": sum(b for _, b in D["ck"]["ffn", METHODS[0], SEEDS[0]]["ck"][0]["handoff"]["ifmap_reads"]), "n_reads": len(D["ck"]["ffn", METHODS[0], SEEDS[0]]["ck"][0]["handoff"]["ifmap_reads"])}
    L.append(f"T_交接 无竞争估算：中位 {f0(h_med)} 周期（{f0(min(h_vals))}–{f0(max(h_vals))}），FFN ifmap 读 {res['handoff_est']['n_reads']} 次共 {res['handoff_est']['ifmap_read_bytes']:,} 字节。这部分已经以 DDR 中转的形式算在 T_MHA、T_FFN 里，主表取 0。\n")

    def options(m):
        o = [("R0", 0.0, 0.0, D["l1"]["mha"]["bk_R0"], D["l1"]["ffn"]["bk_R0"])]
        rows = [tiers[m, sd] for sd in SEEDS]
        for i, name in enumerate(("L1", "s=30", "s=100", "s=300")):
            s = rows[0][i]["s"]
            o.append((f"{LABEL[m]} {name}" if s else "L1", sum(t_tier(D, x, m, s, 1.0) for x in en.OPS), med(sum(r[i]["bytes"].values()) for r in rows), med(r[i]["bk"]["mha"] for r in rows), med(r[i]["bk"]["ffn"] for r in rows)))
        return o

    for r in RS:
        L.append(f"\n### r = {r}\n")
        L.append("| 方案 | t_search s | T_copy | T_a（周期） | 相对 R0 | 单轮 | 相对 R0 超过的窗口 H*（主用） | H* 的轮数 | 停机版回本轮数 N* | 停机版 H* |\n|---|---|---|---|---|---|---|---|---|---|")
        T0 = best_T0(D, r, 0, 0.5)
        A0 = sum(D["serial"][o]["t_R0"] for o in en.OPS)
        rows_opt = list(options("L2Gsync")) + options("L2Lsync")[2:] + options("L2Impact")[2:]
        rows_opt += [("FULL 按 T̂ 自选，14 次依次", 14 * fo["t_single"], fo["bytes"], fo["picks"]["mha"]["bk"], fo["picks"]["ffn"]["bk"]),
                     ("FULL 按 T̂ 自选，14 次并行", fo["t_single"], fo["bytes"], fo["picks"]["mha"]["bk"], fo["picks"]["ffn"]["bk"]),
                     ("FULL 单次中位数（参考）", fo["t_single"], fo["bytes"], fo["single"]["mha"], fo["single"]["ffn"])]
        for name, t, by, tm, tf in rows_opt:
            Ta = round_time(tm, tf, r, 0, 0.5)
            c = t_copy(by, 0.5)
            row = {"t": t, "copy": c, "Ta_cycles": r * tm + tf, "rel_R0": Ta / T0 - 1}
            if name == "R0":
                row.update(Hstar=0.0, Nstar=0.0)
            elif Ta < T0:
                row["Hstar"] = t + c * T0 / (T0 - Ta)
                row["Nstar"] = (t + A0 * 0 + c - A0) / (T0 - Ta) if t + c > A0 else 0.0
                row["Hdown"] = t + c + row["Nstar"] * Ta
            timeline[f"r{r}_{name}"] = row
            hs = f"{sec(row['Hstar'])}" if "Hstar" in row else "不超过"
            nr = f"{row['Hstar'] / T0:,.0f}" if "Hstar" in row else "—"
            ns = f"{row['Nstar']:,.0f}" if "Nstar" in row else "—"
            hd = sec(row["Hdown"]) if "Hdown" in row else "—"
            L.append(f"| {name} | {t:,.1f} | {sec(c)} | {f0(r * tm + tf)} | {(Ta / T0 - 1) * 100:+.1f}% | {sec(Ta)} | {hs} | {nr} | {ns} | {hd} |")
    res["timeline"] = timeline
    # ---- adjacent boundaries and envelope
    L.append("\n## 回本窗口边界（相邻方案之间，后者开始超过前者的窗口）\n")
    L.append("主用版，方案链 R0 → L1 → L2G+sync s=30 → s=100 → s=300 → FULL 按 T̂ 自选（14 次依次），中位种子量级。边界按解析式求，不存在就写“无”（后者单轮不比前者短）。\n")
    L.append("| 时钟 | r | 搜索耗时 | R0→L1 | L1→s=30 | s=30→100 | s=100→300 | s=300→FULL |\n|---|---|---|---|---|---|---|---|")
    bound = {}
    for clock in CLOCKS:
        for r in RS:
            for scale in SCALES:
                chain = [(t * scale, t_copy(by, clock), round_time(tm, tf, r, 0, clock)) for _, t, by, tm, tf in options("L2Gsync")]
                chain.append((14 * fo["t_single"] * scale, t_copy(fo["bytes"], clock), round_time(fo["picks"]["mha"]["bk"], fo["picks"]["ffn"]["bk"], r, 0, clock)))
                T0 = chain[0][2]
                cells = []
                for (t1, c1, T1), (t2, c2, T2) in zip(chain, chain[1:]):
                    if T2 >= T1:
                        cells.append(None)
                        continue
                    # t1/T0 + (H-t1-c1)/T1 = t2/T0 + (H-t2-c2)/T2
                    H = ((t2 - t1) / T0 + (t1 + c1) / T1 - (t2 + c2) / T2) / (1 / T1 - 1 / T2)
                    cells.append(max(H, t2 + c2))
                bound[f"c{clock}_r{r}_x{scale}"] = cells
                L.append(f"| {clock} ns | {r} | ×{scale} | " + " | ".join("无" if x is None else sec(x) for x in cells) + " |")
    res["boundaries"] = bound
    # ---- window decision
    L.append("\n## 窗口决策与固定 ε = 2% 规则（回放，FFN 种子 i 配 MHA 种子 i，10 对）\n")
    L.append("规则见 `PROGRESS.md` 第 25 步，开跑前写定。判断只用 T̂，累计完成量用后端。主用版里搜索期间的 R0 完成量取 min(t_search, H) / T_R0；规则从 L1 开始，不会选“停在 R0”，所以窗口短于 L1 耗时时停机版可能是 0。下表：时钟 0.5 ns，T_交接 = 0，T_switch = 0。“窗口 / 固定”是配对比值的中位数，胜负按配对的累计完成量。FULL 是 14 次依次、按 T̂ 自选，事后最优档是每对种子事后挑的档（上限参照，在线拿不到）。\n")
    L.append("| 方法 | 版本 | r | 搜索耗时 | H | R0 | 窗口决策 | 固定 ε | 窗口 / 固定 | 胜 / 负 / 平 | 窗口选档（L1/30/100/300） | 固定选档（FFN；MHA） | 事后最优 | FULL |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    wd = {}
    cnt_names = ("L1", "s30", "s100", "s300")
    for m in METHODS:
        for version in ("main", "down"):
            for r in RS:
                for scale in SCALES:
                    for H in HS:
                        rec = _replay(D, tiers, m, version, r, scale, H, 0.5, 0.0, fo)
                        wd[f"{m}_{version}_r{r}_x{scale}_H{H}"] = rec
                        ls = "/".join(str(rec["win_tiers"].get(k, 0)) for k in cnt_names)
                        L.append(f"| {LABEL[m]} | {'主用' if version == 'main' else '停机'} | {r} | ×{scale} | {H:,} s | {rec['R0']:.4g} | {rec['window']:.5g} | {rec['fixed']:.5g} | {rec['ratio']:.4f} | {rec['w']} / {rec['l']} / {rec['t']} | {ls} | {rec['fixed_tiers']} | {rec['oracle']:.5g} | {rec['full']:.5g} |")
    res["window"] = wd
    # ---- sensitivity
    L.append("\n## 敏感性：T_switch、时钟、T_交接（L2G+sync，主用版，搜索 ×1，r = 1；T_switch 没有依据，这里只是参数）\n")
    L.append("| T_switch | 时钟 | T_交接 | H | 窗口决策 / R0 | 固定 ε / R0 | 窗口 / 固定 |\n|---|---|---|---|---|---|---|")
    sens = {}
    for sw in SWITCHES:
        for clock in CLOCKS:
            for hv in ("0", "est"):
                for H in (60, 3600):
                    rec = _replay(D, tiers, "L2Gsync", "main", 1, 1.0, H, clock, sw, full_options(D, clock), hv == "est")
                    sens[f"sw{sw}_c{clock}_h{hv}_H{H}"] = rec
                    L.append(f"| {sw} s | {clock} ns | {'估算' if hv == 'est' else '0'} | {H:,} s | {rec['window'] / rec['R0']:.4f} | {rec['fixed'] / rec['R0']:.4f} | {rec['ratio']:.4f} |")
    res["sensitivity"] = sens
    L.append("\n## 迁移量（扰动范围）\n")
    L.append("| 任务 | 方法 | s | 唯一块 MB | T_copy（0.5 ns，无竞争） | 占权重总量 |\n|---|---|---|---|---|---|")
    for o in en.OPS:
        for m in METHODS:
            for s in TIERS:
                by = table[f"{o}_{m}_{s}"]["bytes"]
                L.append(f"| {o} | {LABEL[m]} | {s} | {by / 1e6:.1f} | {sec(t_copy(by, 0.5))} | {by / D['pre'][o]['weight_bytes'] * 100:.1f}% |")
    L.append("\n自检：见开头。回放不产生新运行。")
    with open(os.path.join(OUT, "report_b.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "report_b.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print("\n".join(L))


def _replay(D, tiers, m, version, r, scale, H, clock, sw, fo, use_h=False):
    win, fix, ora, chosen, fixed_ch = [], [], [], {}, {}
    names = ("L1", "s30", "s100", "s300")
    h_med = med(handoff(D, m, sd, i) for sd in SEEDS for i in range(3)) if use_h else 0.0
    for sd in SEEDS:
        rows = tiers[m, sd]
        i = window_rule(rows, D, m, H, r, None, clock, scale, sw, version)
        chosen[names[i]] = chosen.get(names[i], 0) + 1
        win.append(realised(rows, D, m, {"ffn": i, "mha": i}, r, h_med, clock, scale, sw, H, version))
        fx = {o: fixed_rule(rows, D, o) for o in en.OPS}
        k = (("R0",) + names)[fx["ffn"] + 1] + "；" + (("R0",) + names)[fx["mha"] + 1]
        fixed_ch[k] = fixed_ch.get(k, 0) + 1
        fix.append(realised(rows, D, m, fx, r, h_med, clock, scale, sw, H, version))
        ora.append(max(realised(rows, D, m, {"ffn": j, "mha": j}, r, h_med, clock, scale, sw, H, version) for j in range(4)))
    T0 = best_T0(D, r, h_med, clock)
    if version == "main":
        R0 = H / T0
    else:
        R0 = n_down(H, sum(D["serial"][o]["t_R0"] for o in en.OPS) * scale + sw, T0)
    Tf = round_time(fo["picks"]["mha"]["bk"], fo["picks"]["ffn"]["bk"], r, h_med, clock)
    tf = 14 * fo["t_single"] * scale
    cf = t_copy(fo["bytes"], clock) + sw
    full = n_main(H, tf, cf, Tf, T0) if version == "main" else n_down(H, tf + cf, Tf)
    ratios = [a / b if b else float("nan") for a, b in zip(win, fix)]
    return {"R0": R0, "window": med(win), "fixed": med(fix), "ratio": med(x for x in ratios if x == x) if any(x == x for x in ratios) else float("nan"),
            "w": sum(a > b for a, b in zip(win, fix)), "l": sum(a < b for a, b in zip(win, fix)), "t": sum(a == b for a, b in zip(win, fix)),
            "win_tiers": chosen, "fixed_tiers": " ".join(f"{k}×{v}" for k, v in sorted(fixed_ch.items())), "oracle": med(ora), "full": full}


if __name__ == "__main__":
    main()
