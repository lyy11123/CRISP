"""Why does XY routing survive node faults but not some on-chip link faults? Build WAMIS_HD platforms (no
task script) for single faults and count the (source, target) pairs the fault-tolerant XY router gives up
on, then check which of them a mapping actually needs.

  python xy_probe.py
"""
import collections
import gzip
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "work", "artifact")
for p in ["utils", "src/scheduling/communication/topology", "src/scheduling", "src/partition"]:
    sys.path.append(os.path.join(ART, p))
from read_cfg import cfg_to_dict  # noqa: E402
import WAMIS_HD  # noqa: E402

CFG = os.path.join(ART, "models", "intra_mapping_gpt_decode", "cfg",
                   "config_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0.cfg")
orig_xy = WAMIS_HD.wamis_hdc.xy_multicast_path


def xy_multicast_path(self, source_node_idx, target_nodes_set):  # same in-memory fix as e3_warmstart.py
    try:
        return orig_xy(self, source_node_idx=source_node_idx, target_nodes_set=target_nodes_set)
    except TypeError as exc:
        if "NoneType" not in str(exc):
            raise
        return None


WAMIS_HD.wamis_hdc.xy_multicast_path = xy_multicast_path


def platform(nodes=(), links=()):
    cfg = cfg_to_dict(CFG)
    cfg["failures"]["failed_nodes"] = [tuple(n) for n in nodes]
    cfg["failures"]["failed_links"] = [(tuple(a), tuple(b)) for a, b in links]
    return WAMIS_HD.wamis_hdc(cfg)


def none_pairs(hp):
    out = []
    for s, row in hp.node_to_node_manhattan_hops_dict.items():
        for t, v in row.items():
            if v is None:
                out.append((s, t))
    return out


def kind(hp, n):
    return "ddr" if n in hp.ddr_set else "core"


def main():
    base = platform()
    cores = sorted(n for n in base.nodes_set if n not in base.ddr_set)
    corner = {0, 3, 12, 15}
    res = {"no_fault": len(none_pairs(base))}
    # every single non-corner node fault, and every single bidirectional on-chip (co2co) link fault
    node_stats, link_stats = [], []
    for n in cores:
        if n[1] in corner:
            continue
        hp = platform(nodes=[n])
        node_stats.append((n, len(none_pairs(hp))))
    seen = set()
    for (u, v), l in sorted(base.links_dict.items(), key=str):
        if l.link_type != "co2co" or (v, u) in seen:
            continue
        seen.add((u, v))
        hp = platform(links=[(u, v), (v, u)])
        np_ = none_pairs(hp)
        link_stats.append(((u, v), len(np_), collections.Counter((kind(hp, s), kind(hp, t)) for s, t in np_)))
    res["node_faults"] = {"n": len(node_stats), "with_none_pairs": sum(1 for _, c in node_stats if c),
                          "max_none_pairs": max(c for _, c in node_stats)}
    res["co2co_link_faults"] = {"n": len(link_stats), "with_none_pairs": sum(1 for _, c, _ in link_stats if c),
                                "max_none_pairs": max(c for _, c, _ in link_stats),
                                "none_pair_kinds_total": {f"{a}->{b}": v for (a, b), v in
                                                          sum((k for _, _, k in link_stats), collections.Counter()).items()}}
    res["worst_links"] = [[str(k), c] for k, c, _ in sorted(link_stats, key=lambda x: -x[1])[:5]]
    res["node_faults_with_none"] = [[str(n), c] for n, c in node_stats if c][:10]
    # the Gemini cases of E3: which None pairs does the pre-fault mapping actually need?
    import e3_warmstart as e3
    cases = {"ffng": {"F3": None, "F1": None, "F4": None}, "mhag": {"F3": None, "F1": None, "F4": None}}
    for short in cases:
        with gzip.open(os.path.join(e3.OUT, f"{e3.TASKS[short]}_prefault.pkl.gz"), "rb") as f:
            pre = pickle.load(f)
        ev = pre["final"]["events"]
        needed = {(x["source"], x["target"]) for x in ev.values()
                  if x["type"] == "communication" and not isinstance(x["target"], set)}
        for fid in cases[short]:
            flt = e3.choose_fault(pre, fid)
            hp = platform(flt["failed_nodes"], flt["failed_links"])
            npairs = set(none_pairs(hp))
            dead = set(map(tuple, flt["failed_nodes"]))
            hit = [p for p in needed if p in npairs]
            cases[short][fid] = {"failed": str(flt["failed_links"] or flt["failed_nodes"]),
                                 "none_pairs": len(npairs), "needed_pairs": len(needed),
                                 "needed_pairs_with_dead_endpoint": sum(1 for s, t in needed if s in dead or t in dead),
                                 "needed_pairs_without_xy_path": len(hit), "examples": [str(p) for p in hit[:3]]}
    res["gemini_cases"] = cases
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
