"""V1 self-checks on the captures: (1) mutation control: shifting one recorded activity by +1 must be reported;
(2) on the faithful replay, how much of the replay is non-trivial: activities that waited for their resource (s > earliest start),
communications whose end_time (last issued hop) is before their latest hop end, and the critical-path / CB of the
replayed schedule. Usage: python cpcm_v1_check.py ffn mha moe   (writes cpcm_out/v1_check.json)"""
import copy, gzip, json, os, pickle, sys
import cpcm, cpcm_v1

out = {}
for name in sys.argv[1:]:
    with gzip.open(os.path.join(cpcm_v1.OUT, f"v1_{name}_cap.pkl.gz"), "rb") as f:
        cap = pickle.load(f)
    g, run = cap["graph"], cap["runs"][0]
    bad = copy.deepcopy(run)
    k = max(bad["res"], key=lambda q: len(bad["res"][q]))
    s, f_, tag = bad["res"][k][-1]
    bad["res"][k][-1] = (s + 1, f_ + 1, tag)
    caught = cpcm_v1.replay_run(g, bad)["mismatches"] > 0
    seq, term, _ = cpcm_v1.replay_inputs(g, run)
    sc = cpcm.simulate_order(g, seq, term_res=term)                # faithful replay
    T_default = cpcm.simulate_order(g, seq).T                      # default terminal rule, for comparison
    an = cpcm.Analysis(sc)
    waited = sum(1 for a in range(len(sc.a_s)) if sc.a_s[a] > an.can(a))
    lastend = sum(1 for e in range(g.n) if g.kind[e] == cpcm.COMM and
                  run["events"][g.tag[e]][1] < max(sc.a_f[a] for a in sc.ev_acts[e]))
    cp = an.critical_path()
    cb = an.blocking()
    top = sorted(cb.items(), key=lambda x: -x[1])[:3]
    out[name] = {"mutation_caught": caught, "activities": len(sc.a_s), "waited": waited,
                 "comm_end_before_latest_hop": lastend,
                 "T_replay": sc.T, "T_default_terminal_rule": T_default, "T_backend": run["result"][0], "T_end": an.T_end,
                 "cp_rows": len(cp), "cp_resource_waits": sum(r["edge"] == "resource" for r in cp),
                 "cp_gaps": an.gaps, "cb_excl_total": sum(cb.values()),
                 "top3_cb_excl": [(str(g.res_name[q]), v) for q, v in top],
                 "idle_other_total": sum(an.idle.values())}
    print(name, out[name])
json.dump(out, open(os.path.join(cpcm_v1.OUT, "v1_check.json"), "w"), indent=1)
