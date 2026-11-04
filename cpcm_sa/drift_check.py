import re, sys, os, glob, statistics as st, collections
sys.path.insert(0, r"E:\58Institute\PhD_TERM3\CRISP\cpcm_sa")
sys.argv=["x"]
import cpcm_sa as C
res=collections.defaultdict(list)
first=True
for cfg in ("ffn","mha","ffn_f5"):
  for arm in ("A3","A3CB","A3h"):
    for f in glob.glob(os.path.join(C.OUT,"runs",cfg+"_"+arm+"_g*_s*.pkl.gz")):
        b=os.path.basename(f)[:-7]
        g=int(re.search(r"_g(\d+)_",b).group(1)); x=int(b.rsplit("_s",1)[1])
        d=C.load_run(cfg,arm,g,x)
        if first:
            print({k:(type(v).__name__) for k,v in d.items()}); first=False
        rows=d.get("rows") or d.get("log") or d.get("steps")
        T0=d["meta"]["T0"]
        rec=T0; true=T0
        rej=drift=fake=0; dr=[]; lastacc=0; lastimp=0; best=T0; truebest_gap=[]
        for r in rows:
            if r["accepted"]:
                rec=true=r["T_new"]; lastacc=r["step"]
                if rec<best: best=rec; lastimp=r["step"]
            else:
                rej+=1
                if r["T_new"]<true: fake+=1
                tr=r.get("T_reverted")
                if tr is not None:
                    if tr!=rec: drift+=1
                    dr.append((tr-rec)/rec)
                    true=tr
        res[(cfg,arm)].append(dict(rej=rej,drift=drift,fake=fake,maxd=max(dr or [0]),medd=st.median(dr or [0]),end=dr[-1] if dr else 0,lastacc=lastacc,lastimp=lastimp))
for k,v in res.items():
    m=lambda f:st.median(x[f] for x in v)
    print(k,len(v),"rej",m("rej"),"drift",m("drift"),"fakeRej",m("fake"),"medDrift%%=%.1f maxDrift%%=%.1f endDrift%%=%.1f"%(100*m("medd"),100*m("maxd"),100*m("end")),"lastAcc",m("lastacc"),"lastImp",m("lastimp"))
