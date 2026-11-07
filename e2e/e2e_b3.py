"""B3 read-only check: which tensors each saved GPT mapping reads from DDR and writes back, by operator name. No search, no backend.

  E2E_B3=1 E2E_SUB=_b3 python e2e_b3.py     out/b3_io.json (LN and projection from the 30-step smoke folder)
"""
import gzip
import json
import os
import pickle
import re

os.environ["E2E_B3"] = "1"
import e2e_pilot as P  # noqa: E402

C = P.C
MAIN_OUT = os.path.join(P.HERE, "out")
SMOKE = os.path.join(MAIN_OUT, "_b3")
MAPS = [("pf_mha", "A3h", 0, 4, MAIN_OUT), ("mha", "A3h", 0, 2, P.DECODE_OUT), ("pf_ffn", "A3h", 0, 123, MAIN_OUT), ("ffn", "A3h", 0, 5, P.DECODE_OUT),
        ("pf_ln", "A3h", 0, 123, SMOKE), ("dc_ln", "A3h", 0, 123, SMOKE), ("pf_proj", "A3h", 0, 123, SMOKE), ("dc_proj", "A3h", 0, 123, SMOKE)]


def key(name):
    return re.sub(r"_\d+$", "", str(name))


def one(cfg, arm, g, x, base):
    n = C.run_name(cfg, arm, g, x)
    with gzip.open(os.path.join(base, "runs", n + ".pkl.gz"), "rb") as f:
        ctx = pickle.load(f)["meta"]["ctx"]
    with gzip.open(os.path.join(base, "snaps", n + ".pkl.gz"), "rb") as f:
        blob, hist = pickle.load(f)
    beha, data, events = pickle.loads(blob)
    hp = C.platform(cfg)
    ddr = {k[:-1] for k in hp.ddr_dict}
    rd, wr, comp, other = {}, {}, {}, {}

    def add(d, k, b):
        a = d.setdefault(k, [0, 0])
        a[0] += 1
        a[1] += b
    for t, e in events.items():
        if e.event_type == "computation":
            add(comp, key(e.event_name), 0)
        elif tuple(e.source_location) in ddr:
            add(rd, key(e.event_name), int(e.comm_bytes))
        else:
            add(other, key(e.event_name), int(e.comm_bytes))
    for tag in ctx["broadcast_datatags"]:
        for prod, splits in data[tag].generated_tag_splitted_dict.items():
            nm = key(beha[prod].beha_name) if prod in beha else str(prod)
            for sp in splits:
                b = 1
                for s0, s1 in sp:
                    b *= s1 - s0 + 1
                add(wr, nm, b * 2)
    # data tags of ifmap etc.: name, shape
    shapes = {str(t): (d.data_name, list(d.data_shape), d.data_type) for t, d in data.items() if d.data_name in ("ifmap",) or "weight" in str(d.data_name).lower()}
    return dict(mapping=n, cfg=cfg, read_from_ddr=rd, internal_comm=other, computation=comp, broadcast_written=wr, ifmap_and_weight_shapes=dict(list(shapes.items())[:14]))


res = []
for m in MAPS:
    C._artifact_path()
    res.append(one(*m))
with open(os.path.join(MAIN_OUT, "b3_io.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, default=str)
for r in res:
    print(r["mapping"])
    print(" read ", {k: v for k, v in sorted(r["read_from_ddr"].items())})
    print(" write", {k: v for k, v in sorted(r["broadcast_written"].items())})
    print(" comp ", {k: v[0] for k, v in sorted(r["computation"].items())})
    print(" shapes", list(r["ifmap_and_weight_shapes"].values())[:6])
