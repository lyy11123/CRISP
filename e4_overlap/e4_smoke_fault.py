"""Smoke test of the fault path (E3 R0 capture + simulation on a faulty platform) on MHA 2048 alone; the faults are
picked on the stream's own pre-fault graph. Writes nothing except e4_out/streams/smoke_*.pkl.gz and sims logs."""
import time

import e4_grid as G
import e4_overlap as E

rec = E.load_stream("mha2048")
f = G.choose_faults("smoke", dict(rec["final"], release={}))
for fid in ["F4", "F5"]:
    print(fid, f[fid]["kind"], f[fid]["failed_links"], flush=True)
    key = "smoke_" + G.fault_key("mha2048", f[fid])
    t = time.time()
    print(G.capture_r0(key, "mha2048", f[fid]), round(time.time() - t), flush=True)
    r = E.simulate("smoke_" + fid, [key], [{"name": "default", "release": {0: 0}}], [1], cfg=E.CFG,
                   faults={"failed_nodes": [], "failed_links": f[fid]["failed_links"]})
    c = r["runs"][0]["cap"]
    used = sum(1 for k in f[fid]["failed_links"] if tuple(map(tuple, k)) in c["links"])
    print(fid, c["result"], "records on failed links:", used, "links in platform:", len(c["link_types"]), flush=True)
