"""Read-only S-B feasibility probe: 2x4 platform, die remap of 2x2 graphs, KV broadcast bytes. No experiment result."""
import sys, os, gzip, pickle, collections, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d_joint as d
c = d._sim_setup()
import WAMIS_HD
from read_cfg import cfg_to_dict
S = sys.argv[1]
src = os.path.join(c.WORK, "models", c.PREFILL, "cfg", d.CFG_NAME)
txt = open(src).read()
t24 = txt.replace("chiplet_number = 4", "chiplet_number = 8").replace("chiplet_columns = 2", "chiplet_columns = 4")
p24 = os.path.join(S, "config_ch2x4_probe.cfg"); open(p24, "w").write(t24)
out = {}
try:
    hp22 = WAMIS_HD.wamis_hdc(cfg_to_dict(src)); hp24 = WAMIS_HD.wamis_hdc(cfg_to_dict(p24))
    out["built_2x4"] = True
except Exception as e:
    out["built_2x4"] = repr(e); print(out); raise SystemExit
out["n_links"] = {"2x2": len(hp22.links_dict), "2x4": len(hp24.links_dict)}
out["module_types"] = {k: len(v) for k, v in hp24.modules_dict.items()}
out["dies_2x4"] = sorted({n[0] for n in hp24.nodes_set})
def dmap(off):  # 2x2 die n*2+m -> 2x4 die n*4+m+off
    return {n * 2 + m: n * 4 + m + off for n in (0, 1) for m in (0, 1)}
def rn(node, mp): return (mp[node[0]],) + tuple(node[1:])
res = {}
for off, grp in ((0, "P(left)"), (2, "D(right)")):
    mp = dmap(off); bad = []; same = 0
    for (a, b), L in hp22.links_dict.items():
        k = (rn(a, mp), rn(b, mp)); L2 = hp24.links_dict.get(k)
        if L2 is None or (L2.link_type, L2.bandwidth, L2.latency) != (L.link_type, L.bandwidth, L.latency): bad.append(str((a, b)))
        else: same += 1
    res[grp] = {"links_mapped_same": same, "links_missing_or_differ": len(bad), "examples": bad[:5]}
out["link_remap"] = res
inter = [(str(a), str(b), L.link_type) for (a, b), L in hp24.links_dict.items() if {a[0] % 4, b[0] % 4} == {1, 2} and a[0] // 4 == b[0] // 4]
out["inter_group_links"] = inter
# DDR ports per die (node index >= 16)
out["ddr_nodes_per_die_2x4"] = collections.Counter(n[0] for n in hp24.ddr_set) if hasattr(hp24, "ddr_set") else None
# events: remap pf_mha (P) and dc_mha (D) graphs, check every location / link exists
def check(name, off):
    ev = pickle.loads(pickle.load(gzip.open(f"d/out/d0/export/{name}.pkl.gz"))["d_export"]["events_blob"])
    mp = dmap(off); miss = collections.Counter(); kinds = collections.Counter()
    for t, e in ev.items():
        if e.event_type == "computation":
            loc = rn(e.comp_location, mp)
            if loc not in hp24.modules_dict[e.comp_device]: miss["comp_device"] += 1
        else:
            for br in e.path_list:
                for (a, b) in br:
                    if (rn(a, mp), rn(b, mp)) not in hp24.links_dict: miss["link"] += 1
            kinds["comm"] += 1
    return {"n": len(ev), "missing": dict(miss)}
out["remap_check"] = {"pf_mha_a3h(P,left)": check("x_pf_mha_a3h_pre_none", 0), "dc_mha_a3h(D,right)": check("x_dc_mha_a3h_pre_none", 2),
                      "pf_ffn_a3h(P,left)": check("x_pf_ffn_a3h_pre_none", 0), "dc_ffn_a3h(D,right)": check("x_dc_ffn_a3h_pre_none", 2)}
# KV: prefill MHA W1 final broadcast (events added by add_broadcast: to DDR targets), bytes by data name
ev = pickle.loads(pickle.load(gzip.open("d/out/d0/export/x_pf_mha_a3h_pre_none.pkl.gz"))["d_export"]["events_blob"])
ddr = hp22.ddr_set
by = collections.Counter(); cnt = collections.Counter(); tg = collections.Counter()
for t, e in ev.items():
    if e.event_type != "communication": continue
    T = e.target_location if isinstance(e.target_location, set) else {e.target_location}
    if T and all(tuple(x) in ddr for x in T):
        nm = re.sub(r"\d+", "#", str(e.event_name))
        by[nm] += int(e.comm_bytes); cnt[nm] += 1; tg[len(T)] += 1
out["to_ddr_messages_pf_mha"] = {"bytes_by_name": dict(by), "count_by_name": dict(cnt), "n_targets": dict(tg)}
evd = pickle.loads(pickle.load(gzip.open("d/out/d0/export/x_dc_mha_a3h_pre_none.pkl.gz"))["d_export"]["events_blob"])
rd = collections.Counter(); wr = collections.Counter()
for t, e in evd.items():
    if e.event_type != "communication": continue
    src_ddr = tuple(e.source_location) in ddr
    T = e.target_location if isinstance(e.target_location, set) else {e.target_location}
    nm = re.sub(r"\d+", "#", str(e.event_name))
    if src_ddr: rd[nm] += int(e.comm_bytes)
    if T and all(tuple(x) in ddr for x in T): wr[nm] += int(e.comm_bytes)
out["dc_mha_ddr_read_bytes"] = dict(rd); out["dc_mha_ddr_write_bytes"] = dict(wr)
print(json.dumps(out, indent=1, default=str))
