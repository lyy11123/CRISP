"""Read-only stage-2A enumeration: does the MHA-stage sub-deadline change the order against a competitor? Uses measured T and T^ only."""
import csv, itertools, json, collections
ROOT = r"E:\58Institute\PhD_TERM3\CRISP\dstu\out\solo_table.csv"
rows = {(r["stream"], r["chunk"]): r for r in csv.DictReader(open(ROOT, encoding="utf-8"))}
HW = {"nofault": ("mha2048", "ffn"), "F4": ("mha2048__5f4daae8", "ffn__5f4daae8"), "F5": ("mha2048__bbd7e98b", "ffn__bbd7e98b")}
T = {h: (float(rows[(m, "256d2d")]["T_ideal"]), float(rows[(f, "256d2d")]["T_ideal"])) for h, (m, f) in HW.items()}
Th = {h: (float(rows[(m, "256d2d")]["T_hat"]), float(rows[(f, "256d2d")]["T_hat"])) for h, (m, f) in HW.items()}
print("T", T, "That", Th)
KR = [1.1, 1.2, 1.5, 2, 3]
KX = [1.1, 1.2, 1.5, 2, 3]
RS = [1, 8]
tab = collections.defaultdict(lambda: collections.Counter())
win = {}
for hw in HW:
    for r_, kr, kx, xname, stag, basis in itertools.product(RS, KR, KX, ("X=MHA", "X=FFN"), (0.0, 0.25, 0.5, 1.0), ("cur", "prefault")):
        TM, TF = T[hw]
        M = r_ * TM
        ThM, ThF = Th[hw]
        Tbase = T["nofault"] if basis == "prefault" else T[hw]
        dreq = kr * (r_ * Tbase[0] + Tbase[1])
        TX = TM if xname == "X=MHA" else TF
        rx = stag * M           # competitor released while the MHA stage runs
        dx = rx + kx * (Tbase[0] if xname == "X=MHA" else Tbase[1])
        sub = {"CPCM": dreq - ThF, "ideal": dreq - TF, "stale": dreq - T["nofault"][1], "equal": dreq / 2, "ratio_0.3": 0.3 * dreq, "prop_stale": dreq * r_ * T["nofault"][0] / (r_ * T["nofault"][0] + T["nofault"][1])}
        key = (hw, r_, xname)
        tab[key]["n"] += 1
        c = sub["CPCM"] < dx
        for b, v in sub.items():
            if b != "CPCM" and (v < dx) != c:
                tab[key][b] += 1
        for b in sub:
            tab[key]["MHA_first_" + b] += sub[b] < dx
        if basis == "cur" and stag == 0 and kx == 1.5 and kr == 1.5:
            win[key] = {k: round(v) for k, v in sub.items()} | {"dx": round(dx)}
for k in sorted(tab, key=str):
    v = tab[k]
    print(k, {a: v[a] for a in ("n", "ideal", "stale", "equal", "ratio_0.3", "prop_stale")}, "MHA first: CPCM", v["MHA_first_CPCM"], "stale", v["MHA_first_stale"])
for k, v in sorted(win.items(), key=str):
    print(k, v)
