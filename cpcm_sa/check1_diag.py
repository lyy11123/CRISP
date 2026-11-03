"""Self-check 1 follow-up: for an A0 run, re-evaluate its saved last state in a fresh process with backend seeds
1..N (broadcast seed 0, as the run) and with broadcast seeds 1..M (backend seed = broadcast seed), to see whether
the template's own result is one of the backend's modes.
Usage: python check1_diag.py <cfg> <g> <l> [N M]"""
import copy
import gzip
import json
import os
import pickle
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpcm_sa as C  # noqa: E402


def main(cfg, g, x, n=20, m=10):
    C._artifact_path()
    from add_communication import add_broadcast
    from event_driver import event_driver
    run = C.load_run(cfg, "A0", g, x)
    ctx = run["meta"]["ctx"]
    with gzip.open(os.path.join(C.OUT, "snaps", C.run_name(cfg, "A0", g, x) + ".pkl.gz"), "rb") as f:
        blob, hist = pickle.load(f)
    hp0 = C.platform(cfg)
    for k in ("link_loads_dict", "link_loads_count"):
        getattr(hp0, k).update(hist[k])
    state = pickle.loads(blob)

    def bcast(seed):
        beha, data, events = copy.deepcopy(state)
        hp = copy.deepcopy(hp0)
        random.seed(seed)
        np.random.seed(seed)
        add_broadcast(data_tags=ctx["broadcast_datatags"],
                      ddr_chiplets=ctx["layers_regions"][ctx["initialization_layer_tag"]],
                      beha_dict=beha, data_dict=data, hardware_platform=hp, event_dict=events,
                      dijkstra_routing=ctx["dijkstra_routing"], alpha=ctx["alpha"], beta=ctx["beta"], gamma=ctx["gamma"])
        bpaths = sorted(str(e.path_list) for t, e in events.items() if t not in state[2])
        return events, hp, bpaths

    def backend(events, hp, seed):
        ev, h = copy.deepcopy((events, hp))
        random.seed(seed)
        np.random.seed(seed)
        return int(event_driver(events_dict=ev, hardware_platform=h)[0])

    ev0, hp_b, bp0 = bcast(C.BCAST_SEED)
    res = {"cfg": cfg, "g": g, "x": x, "template": run["template_out"][0],
           "run_seeds_1_5": [v[0] for v in run["selected"]["backend"].values()],
           "bcast0_backend_seeds": [backend(ev0, hp_b, s) for s in range(1, n + 1)], "bcast_seed_k": []}
    for k in range(1, m + 1):
        evk, hpk, bpk = bcast(k)
        res["bcast_seed_k"].append({"seed": k, "same_broadcast_routes_as_seed0": bpk == bp0,
                                    "backend": backend(evk, hpk, k)})
    print(json.dumps(res))
    os.makedirs(os.path.join(C.OUT, "check1_diag"), exist_ok=True)
    with open(os.path.join(C.OUT, "check1_diag", f"{cfg}_g{g}_l{x}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), int(a[2]), *(int(v) for v in a[3:]))
