"""Self-check of e4_backend_diag mode "orig" against e4_backend.event_driver_e4 in one process (same seed, fresh
event objects each call): a1 512+8192 F4 Delta 0, default and prio_01; event_driver_e4 is run twice to show the
evaluator's own run-to-run spread (object-address dependence). Writes e4_out/diag/diag2_selfcheck.json."""
import copy
import json
import os
import random
import sys

import e4_grid as G
import e4_overlap as E

E._artifact_path()
from e4_backend import event_driver_e4  # noqa: E402
from e4_backend_diag import event_driver_diag  # noqa: E402

with open(os.path.join(E.OUT, "grid1_state.json"), encoding="utf-8") as f:
    fl = json.load(f)["faults"]["a1_2s_512-8192"]["F4"]
fl["failed_links"] = [tuple(map(tuple, l)) for l in fl["failed_links"]]
keys = [G.fault_key(k, fl) for k in ["mha512", "mha8192"]]
hp0 = E.build_platform(E.CFG, {"failed_nodes": [], "failed_links": fl["failed_links"]})
blobs = [E.load_stream(k)["events_blob"] for k in keys]
out = {}
for vname, pr in [("default", None), ("prio_01", {0: 0, 1: 1})]:
    res = {}
    for label, fn in [("e4_a", event_driver_e4), ("diag_orig", None), ("e4_b", event_driver_e4)]:
        ev = E.merged_events(blobs)
        hp = copy.deepcopy(hp0)
        random.seed(1)
        if fn is None:
            log = {}
            r = event_driver_diag(ev, hp, release={0: 0, 1: 0}, prio=pr, mode="orig", watch=0, log=log)
            res[label] = {"result": r, "n_deferrals": len(log["deferrals"])}
        else:
            r = fn(ev, hp, release={0: 0, 1: 0}, prio=pr)
            res[label] = {"result": r}
        res[label]["C"] = {i: max(e.end_time for t, e in ev.items() if t[1] - 10 == i) for i in (0, 1)}
    out[vname] = res
    print(vname, json.dumps(res), flush=True)
E._write_json(os.path.join("diag", "diag2_selfcheck.json"), out)
