"""V0 self-check: run the artifact's own event_driver (unmodified, imported read-only) on the bb-mode hand graphs of
cpcm_v0.py through stub event/platform objects, over many seeds, and check that CPCM's deterministic bb result is
one of the backend's outcomes (the backend breaks ties randomly). Also a mutation control: a deliberately wrong
expectation must be reported by cpcm_v0.run_case.

Release times (no such input in the backend) become a dummy computation of that length on its own device.
Link durations: comm_bytes x per-link multiplier (multiplier 1 and bytes = duration, unless the case gives a
per-link duration dict, then bytes 1 and multiplier = duration).
Usage: python cpcm_v0_backend.py [n_seeds]   (writes cpcm_out/v0_backend.json)
"""
import copy
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "isca2026-busybarn-artifact", "src", "backend", "analytical"))
import cpcm  # noqa: E402
import cpcm_v0  # noqa: E402
import event_driver as bb_backend  # noqa: E402


class Dev:
    def __init__(self, mult=1):
        self.mult = mult
        self.work_flag = False
        self.work_endtime = 0
        self.work_record = []

    def working_time(self, data_bytes=None, source_datashape=None, beha_type=None, frequency=None):
        return data_bytes * self.mult if data_bytes is not None else source_datashape


class Plat:
    frequency = 1

    def __init__(self):
        self.modules_dict = {}
        self.links_dict = {}


class Ev:
    def __init__(self, tag, kind):
        self.event_tag = tag
        self.event_type = kind
        self.event_name = str(tag)
        self.dependency_set = set()
        self.issue_set = set()
        self.start_time = 0
        self.end_time = 0
        self.path_list = []


def link_key(name, key3):
    return (name, 0, 0) if key3 else (name, 0)


def to_backend(case):
    key3 = {r for r, k, f in case.get("res", []) if f}
    events = [dict(e) for e in case["events"]]
    for t, rel in case.get("release", {}).items():
        events.append({"tag": "rel_" + t, "kind": "comp", "res": "dev_rel_" + t, "dur": rel, "deps": []})
        next(e for e in events if e["tag"] == t)["deps"] = ["rel_" + t]
    tags = {e["tag"]: (i, 5) for i, e in enumerate(events)}     # tuples, never (0, 1): no "prefill" print
    hp = Plat()
    evd = {}
    for e in events:
        ev = Ev(tags[e["tag"]], "computation" if e["kind"] == "comp" else "communication")
        ev.dependency_set = {tags[d] for d in e["deps"]}
        if e["kind"] == "comp":
            ev.comp_device, ev.comp_location = "dev", e["res"]
            ev.comp_datashape, ev.comp_type = e["dur"], "x"
            hp.modules_dict.setdefault("dev", {})[e["res"]] = Dev()
        else:
            if isinstance(e["dur"], dict):
                ev.comm_bytes = 1
                for l, d in e["dur"].items():
                    hp.links_dict[link_key(l, l in key3)] = Dev(d)
            else:
                ev.comm_bytes = e["dur"]
                for br in e["branches"]:
                    for l in br:
                        hp.links_dict.setdefault(link_key(l, l in key3), Dev(1))
            ev.path_list = [[link_key(l, l in key3) for l in br] for br in e["branches"]]
        evd[ev.event_tag] = ev
    for ev in evd.values():
        for d in ev.dependency_set:
            evd[d].issue_set.add(ev.event_tag)
    names = {v: k for k, v in tags.items()}
    return evd, hp, names


def backend_outcome(evd, hp, names, seed):
    evd, hp = copy.deepcopy((evd, hp))
    random.seed(seed)
    T, _, _ = bb_backend.event_driver(evd, hp)
    acts = []
    for dev, mods in hp.modules_dict.items():
        for loc, m in mods.items():
            acts += [(names[tag], loc, s, f) for s, f, tag in m.work_record]
    for k, l in hp.links_dict.items():
        acts += [(names[tag], k[0], s, f) for s, f, tag in l.work_record]
    acts = tuple(sorted((a, b, int(s), int(f)) for a, b, s, f in acts if not str(a).startswith("rel_")))
    ends = tuple(sorted((names[t], int(e.end_time)) for t, e in evd.items() if not str(names[t]).startswith("rel_")))
    return int(T), acts, ends


def cpcm_outcome(case):
    res = case.get("res", []) + [(r, "link", False) for r in case.get("extra_res", [])]
    g = cpcm.build(case["events"], res)
    rel = {g.id_of[k]: v for k, v in case.get("release", {}).items()} or None
    sc = cpcm.simulate(g, "bb", release=rel)
    acts = tuple(sorted((g.tag[sc.a_ev[a]], g.res_name[sc.a_res[a]], sc.a_s[a], sc.a_f[a])
                        for a in range(len(sc.a_s))))
    ends = tuple(sorted((g.tag[e], sc.F[e]) for e in range(g.n)))
    return sc.T, acts, ends


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    rows = []
    ok_all = True
    for c in cpcm_v0.CASES:
        if c["mode"] != "bb" or "reroute" in c:
            continue
        evd, hp, names = to_backend(c)
        outs = {}
        for s in range(n_seeds):
            o = backend_outcome(evd, hp, names, s)
            outs.setdefault(o, []).append(s)
        mine = cpcm_outcome(c)
        hit = mine in outs
        ok_all &= hit
        rows.append({"case": c["name"], "backend_distinct_outcomes": len(outs),
                     "backend_T": sorted({o[0] for o in outs}), "cpcm_T": mine[0], "cpcm_in_backend_set": hit,
                     "seeds_matching_cpcm": len(outs.get(mine, []))})
        print(f"{'OK ' if hit else 'BAD'} {c['name']:28s} backend outcomes={len(outs)} T={sorted({o[0] for o in outs})}"
              f" cpcm T={mine[0]} (matches {len(outs.get(mine, []))}/{n_seeds} seeds)")
        if not hit:
            for o in outs:
                print("    backend:", o)
            print("    cpcm:   ", mine)
    # mutation control: a wrong expectation must be caught
    bad = copy.deepcopy(cpcm_v0.CASES[0])
    bad["acts"][("P", "L1")] = (0, 11)
    caught = not cpcm_v0.run_case(bad)["pass"]
    print("mutation control caught:", caught)
    with open(os.path.join(cpcm_v0.OUT, "v0_backend.json"), "w", encoding="utf-8") as f:
        json.dump({"n_seeds": n_seeds, "rows": rows, "mutation_caught": caught}, f, indent=1)
    return 0 if ok_all and caught else 1


if __name__ == "__main__":
    sys.exit(main())
