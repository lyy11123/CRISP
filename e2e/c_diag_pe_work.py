"""C diagnosis (anomaly 1, read-only): tensor-unit work per PE (MACs from each computation event's datashape) in the start (PF4 R0),
R0, the five 300-step L2Gsync seeds and the 14 FULL runs (winner, and the smallest over 14), relative to before the fault.
No search, no backend run.  Run from the repo root:  python e2e/c_diag_pe_work.py  -> e2e/c/out/diag_work.json
"""
import sys,os,pickle,collections,statistics as S,json,math
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import c_eir as c
c.redirect(); E=c.eir
R=[os.path.abspath("e2e/c/out"),os.path.abspath("colab_results/c")]
def rd(jf):
    for r in R:
        j=jf(r)
        if os.path.exists(j["sink"]): return E.read_sink(j["sink"])
def bk(b): return S.mean(v[0] for v in b.values())
res={}
for op in c.OPS:
  for st in ("a3h","a1"):
    l1=rd(lambda r:E.job_l1(os.path.join(r,"prep"),st,op,"PF4"))
    s=pickle.loads(l1["R0_snap"]["blob"])
    comp={k:e for k,e in s[7].items() if e.event_type=="computation"}
    tc={k:e for k,e in comp.items() if e.comp_device=="tensorcore"}
    def w(e):
        sh=e.comp_datashape
        return math.prod(sh[0])*sh[1][-1] if len(sh)>1 and len(sh[1])==2 else math.prod(sh[0])
    W={k:w(e) for k,e in tc.items()}
    kinds=collections.Counter((e.comp_type,str(e.comp_datashape)) for e in tc.values())
    big=max(W.values()) if W else 0
    nbig=sum(1 for v in W.values() if v==big)
    def load(bl):
        L=collections.Counter()
        for k,v in W.items():
            if k in bl: L[(bl[k][0],bl[k][1])]+=v
            elif tuple(k) in bl: L[(bl[tuple(k)][0],bl[tuple(k)][1])]+=v
        return L
    pre=load(l1["R0"]["beha_loc"]); m0=max(pre.values()) if pre else 0
    tot=sum(W.values())
    row={"n_tc_blocks":len(tc),"kinds":len(kinds),"n_largest":nbig,"largest_share_of_work":nbig*big/tot if tot else None,"max_pe_work_pre":m0,"mean_pe_work_pre":tot/64 if tot else 0,
         "lb63_balanced":(tot/63)/m0 if m0 else None}
    for fid in ("PF6","MF1","PCEdge"):
        x=rd(lambda r:E.job_l1(os.path.join(r,"prep"),st,op,fid))
        r0=load(x["R0"]["beha_loc"])
        row[f"{fid}_R0"]=max(r0.values())/m0 if m0 else None
        if st=="a3h":
            rp=[rd(lambda r,i=i:c.jobs_rep(r,op,fid)[10+i]) for i in range(5)]
            row[f"{fid}_s300"]=[round(max(load(z["record"]["beha_loc"]).values())/m0,2) for z in rp] if m0 else None
        fa=[rd(lambda r,i=i:c.jobs_full(r,op,fid,st.upper() if st=="a1" else "A3h")[i]) for i in range(14)]
        best=min(fa,key=lambda z:bk(z["selected"]["backend"]))
        row[f"{fid}_FULLbest"]=max(load(best["selected"]["e3map"]["beha_loc"]).values())/m0 if m0 else None
        row[f"{fid}_FULLmin14"]=min(max(load(z["selected"]["e3map"]["beha_loc"]).values()) for z in fa)/m0 if m0 else None
    res[f"{st}_{op}"]=row
    print(st,op,{k:(round(v,3) if isinstance(v,float) else v) for k,v in row.items()})
json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"c","out","diag_work.json"),"w"),indent=1)
