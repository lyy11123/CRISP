"""C diagnosis (anomaly 1, read-only): per (start, operator, fault) the SA per-unit compute load of the R0 snapshot (state[3] tensor units,
state[4] vector units), max relative to PF4 (a D2D fault, R0 moves no block), with T-hat before and after and the backend triple.
No search, no backend run.  Run from the repo root:  python e2e/c_diag_pe_load.py  -> e2e/c/out/diag_load.json
"""
import sys,os,pickle,collections,statistics as S,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import c_eir as c
c.redirect()
E=c.eir
R=[os.path.abspath("e2e/c/out"),os.path.abspath("colab_results/c")]
def l1(st,op,fid):
    for r in R:
        j=E.job_l1(os.path.join(r,"prep"),st,op,fid)
        if os.path.exists(j["sink"]): return E.read_sink(j["sink"])
def bk(b): return S.mean(v[0] for v in b.values())
FA=["PF4","PF6","PF7","PF1","PF2","MF1","MF6","PCRandom","PCCluster","PCEdge","PCScratch","MCRandom","MCCluster","MCEdge","MCScratch"]
out={}
for op in c.OPS:
  for st in ("a3h","a1"):
    rows={}
    for fid in FA:
        d=l1(st,op,fid)
        s=pickle.loads(d["R0_snap"]["blob"])
        tc=s[3]; vu=s[4]
        ev=s[7]
        comp=[e for e in ev.values() if e.event_type=="computation"]
        loc=collections.Counter((tuple(e.comp_location),e.comp_device) for e in comp)
        rows[fid]={"tc_max":float(max(tc.values())) if tc else 0,"tc_med":float(S.median(tc.values())) if tc else 0,"n_tc":len(tc),
                   "vu_max":float(max(vu.values())) if vu else 0,"n_comp":len(comp),"max_ev_per_unit":max(loc.values()),
                   "R0_rel":bk(d["R0"]["backend"])/d["T_pre"] if False else None,"T_pre":d["T_pre"],"T_R0":d["T_R0"],
                   "bk_R0":bk(d["R0"]["backend"]),"tri_R0":[S.mean(v[i] for v in d["R0"]["backend"].values()) for i in range(3)],
                   "shapes":sorted({str(e.comp_datashape) for e in comp})[:3], "devs":sorted({e.comp_device for e in comp})}
    out[f"{st}_{op}"]=rows
    b=rows["PF4"]; k="tc_max" if b["tc_max"]>0 else "vu_max"
    for f in FA: rows[f]["load_max_rel"]=rows[f][k]/b[k]; rows[f]["unit"]=k
    print(f"{st} {op:8s} n_comp {b['n_comp']} devs {b['devs']} {k} PF4 {b[k]:.3g} |", " ".join(f"{f}:{rows[f][k]/b[k]:.2f}/{rows[f]['T_R0']/rows[f]['T_pre']-1:+.2f}" for f in FA[1:]))
json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"c","out","diag_load.json"),"w"),indent=1,default=str)
