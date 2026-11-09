"""Markdown tables of out/report_eir_w1.json (W1_RERUN_SUMMARY section 12)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
R = json.load(open(os.path.join(HERE, "out", "report_eir_w1.json"), encoding="utf-8"))
G = {(r["w"], r["start"], r["fault"]): r for r in R}
W1SEL = {"F4a": "F4aW", "F5a": "F5aW", "F8a": "F8aW", "CRandom": "CRandomW", "CCluster": "CClusterW", "CEdge": "CEdgeW", "CScratch": "CScratchW"}
pc = lambda a, b: 100 * (a / b - 1)
f0 = lambda x: "—" if x is None else f"{x:,.0f}"
def rel(r, x): return "—" if x is None else f"{f0(x)}（{pc(x, r['bk_pre']):+.1f}%）"
def cols(start, fid):
    a = G.get(("W0", start, fid)); b = G.get(("W1", start, fid)); c = G.get(("W1", start, W1SEL.get(fid, fid))) if start == "a3h" else None
    return a, b, c
def lv(r, k): return (r or {}).get("lv", {}).get(k) if r else None
def main_key(r):
    """main repair: L2Gsync, the stop-rule tier when three tiers exist, else s100."""
    if not r: return None
    k = r["stop"]["L2Gsync"][0]
    if k and k.startswith("L2Gsync"): return k
    if k in ("R0", "L1"): return k
    return "L2Gsync_100" if lv(r, "L2Gsync_100") else None
def cell_lv(r, k):
    x = lv(r, k)
    return "—" if not x else rel(r, x["bk"])
def stop_cell(r, m):
    if not r: return "—"
    k, why = r["stop"][m]
    if not k: return "—"
    x = r["lv"][k]
    return f"{k}（{why}）{rel(r, x['bk'])}"
def mig_cell(r, k):
    x = lv(r, k)
    if not x or not x.get("mig"): return "—"
    m = x["mig"]; return f"{100 * m['moved_behas_share']:.1f}% / {m['ddr_read_moved_bytes'] / 1e6:.1f} MB"
L = []
L.append("### 12.1 A3h 起点：故障前、R0、L1、主力修复与 FULL-A3h（后端，相对故障前）\n")
L.append("| 故障 | 口径 | 故障前 | R0 | L1 | L2Gsync s30 | s100 | s300 | FULL-A3h 按后端最优 | 14 次中位数 | 按 T̂ 自选 |\n|---|---|---|---|---|---|---|---|---|---|---|")
for fid in ["F4", "F5", "F4a", "F5a", "F8a", "CRandom", "CCluster", "CEdge", "CScratch"]:
    for name, r in zip(("W0", "W1 旧坐标", "W1 重选"), cols("a3h", fid)):
        if not r or (name == "W1 重选" and r["fault"] == fid): continue
        f = r.get("FULL_A3h") or {}
        L.append(f"| {r['fault']} | {name} | {f0(r['bk_pre'])} | {cell_lv(r, 'R0')} | {cell_lv(r, 'L1')} | {cell_lv(r, 'L2Gsync_30')} | {cell_lv(r, 'L2Gsync_100')} | {cell_lv(r, 'L2Gsync_300')} | {rel(r, f.get('best'))} | {rel(r, f.get('median'))} | {rel(r, f.get('by_T'))} |")
L.append("\n### 12.2 A3h 起点：主力修复（L2Gsync）相对 FULL-A3h，后端差（修复 − FULL）/ 故障前\n")
L.append("| 故障 | W0 | W1 旧坐标 | W1 重选 |\n|---|---|---|---|")
for fid in ["F4a", "F5", "F8a", "CRandom", "CCluster", "CEdge", "CScratch", "F4", "F5a"]:
    cells = []
    for i, r in enumerate(cols("a3h", fid)):
        if not r or (i == 2 and r["fault"] == fid): cells.append("—"); continue
        k = main_key(r); f = (r.get("FULL_A3h") or {})
        if not k or not f.get("best"): cells.append("—"); continue
        x = r["lv"][k]["bk"]
        cells.append(f"{k}: 对最优 {100 * (x - f['best']) / r['bk_pre']:+.1f} pt，对中位 {100 * (x - f['median']) / r['bk_pre']:+.1f} pt，对 T̂ 自选 {100 * (x - f['by_T']) / r['bk_pre']:+.1f} pt")
    L.append(f"| {fid} | " + " | ".join(cells) + " |")
L.append("\n### 12.3 停止级别（ε = 2%，make_summary.final_rule）与该级的迁移量（行为搬动占比 / DDR 读源）\n")
L.append("| 起点 | 故障 | 口径 | L2G 链 | 迁移 | L2Gsync 链 | 迁移 |\n|---|---|---|---|---|---|---|")
for start, fl in (("a3h", ["F4", "F5", "F4a", "F5a", "F8a", "CRandom", "CCluster", "CEdge", "CScratch"]), ("e3", ["F1", "F2", "F4", "F5", "F6", "F7", "F8"])):
    for fid in fl:
        for name, r in zip(("W0", "W1 旧坐标", "W1 重选"), cols(start, fid)):
            if not r or (name == "W1 重选" and r["fault"] == fid): continue
            if start == "e3" and name != "W0": name = "W1（l1 起点）"
            fm = (r.get("FULL_A3h") or {}).get("best_mig") if start == "a3h" else (r.get("FULL_BB") or {}).get("mig")
            fmc = "—" if not fm else f"{100 * fm['moved_behas_share']:.1f}% / {fm['ddr_read_moved_bytes'] / 1e6:.1f} MB"
            L.append(f"| {start} | {r['fault']} | {name} | {stop_cell(r, 'L2G')} | {mig_cell(r, r['stop']['L2G'][0])} | {stop_cell(r, 'L2Gsync')} | {mig_cell(r, r['stop']['L2Gsync'][0])} | {fmc} |")
L.append("\n### 12.4 E3 起点（BusyBarn 映射）：R0、L1、L2G 与 FULL-BusyBarn（W0：E3 的 l4 起点；W1：l1 起点）\n")
L.append("| 故障 | 口径 | 故障前 | R0 | L1 | L2G s30 | L2G s100 | L2G s300 | FULL-BusyBarn | FULL-BB 迁移 | R0 迁移 |\n|---|---|---|---|---|---|---|---|---|---|---|")
for fid in ["F1", "F2", "F4", "F5", "F6", "F7", "F8"]:
    for name, r in zip(("W0（l4）", "W1（l1）"), cols("e3", fid)[:2]):
        if not r: continue
        f = r.get("FULL_BB") or {}
        m = f.get("mig"); mm = "—" if not m else f"{100 * m['moved_behas_share']:.1f}% / {m['ddr_read_moved_bytes'] / 1e6:.1f} MB"
        L.append(f"| {fid} | {name} | {f0(r['bk_pre'])} | {cell_lv(r, 'R0')} | {cell_lv(r, 'L1')} | {cell_lv(r, 'L2G_30')} | {cell_lv(r, 'L2G_100')} | {cell_lv(r, 'L2G_300')} | {rel(r, f.get('bk'))}（{f.get('variant')}） | {mm} | {mig_cell(r, 'R0')} |")
print("\n".join(L))
