"""Read-only stage-2B diagnostic: MHA's own critical chain under the fixed order MHA first, a2 256 KB, CPCM vs E4a's saved backend caps."""
import collections, gzip, json, math, os, pickle, sys
R = r"E:\58Institute\PhD_TERM3\CRISP"
sys.path[:0] = [os.path.join(R, "e4_overlap"), os.path.join(R, "cpcm")]
import cpcm, e4_overlap as E, e4b as B, e4_attr as A, e4_diag as DG, e4_grid as G, e4_hop as H, e4b_s1 as S1
E._artifact_path()


class An(cpcm.Analysis):
    def path_from(self, last):
        """critical_path's loop from a given event"""
        sc, g = self.sc, self.sc.g
        cur = sc.term[last]
        rows = []
        self.gaps = 0
        while cur >= 0:
            s, c = sc.a_s[cur], self.can(cur)
            e = sc.a_ev[cur]
            row = {"act": cur, "event": e, "res": sc.a_res[cur], "s": s, "f": sc.a_f[cur], "can": c, "wait": s - c}
            if s > c:
                row.update(edge="resource", pred_act=self.rprev[cur])
                nxt = self.rprev[cur]
            elif sc.a_par[cur] >= 0:
                nxt = sc.a_par[cur]
                row.update(edge="hop", pred_act=nxt)
            elif g.pred_ptr[e + 1] > g.pred_ptr[e]:
                p = max((g.pred[j] for j in range(g.pred_ptr[e], g.pred_ptr[e + 1])), key=lambda x: (sc.F[x], x))
                nxt = sc.term[p]
                row.update(edge="data", pred_act=nxt)
            else:
                nxt = -1
                row.update(edge="start", pred_act=-1)
            rows.append(row)
            cur = nxt
        rows.reverse()
        self._cp = rows
        self._attribute()
        return rows


def gid(tag):
    return (tag[1],) + B.group_of(B.rt(tag))


def conc(phi, added):
    """phi: {msg: value}; how many added blocks cover 50/80/100% of phi, taking messages by decreasing phi"""
    tot = sum(phi.values())
    out = {"msgs": len(phi), "total": tot}
    items = sorted(phi.items(), key=lambda x: -x[1])
    for q in (0.5, 0.8, 1.0):
        c, b, n = 0, 0, 0
        for m, v in items:
            if c >= q * tot:
                break
            c += v
            b += added.get(m, 0)
            n += 1
        out[f"q{int(q*100)}"] = {"msgs": n, "added_blocks": b}
    return out


def top(phi, n=10):
    return {m for m, _ in sorted(phi.items(), key=lambda x: -x[1])[:n]}


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    st, _ = S1.load()
    for g in st["faults"]:
        for fl in st["faults"][g].values():
            fl["failed_links"] = [tuple(map(tuple, l)) for l in fl["failed_links"]]
    fl = st["faults"]["a2_2s_mha-ffn"]["F4"]
    cases = {"nofault": (["mha2048", "ffn"], None), "F4": ([G.fault_key(k, fl) for k in ["mha2048", "ffn"]], {"failed_nodes": [], "failed_links": fl["failed_links"]})}
    out = {}
    for hw, (keys, fa) in cases.items():
        hp = E.build_platform(E.CFG, fa)
        raw = E.merged_events([B.load_stream(k)["events_blob"] for k in keys])
        size = {}
        for t, e in raw.items():
            if e.event_type == "communication":
                size[gid(t)] = int(e.comm_bytes)
        d2d = {gid(t) for t, e in raw.items() if e.event_type == "communication" and any(hp.links_dict[l].link_type == "ch2ch" for br in e.path_list for l in br)}
        added = {m: math.ceil(size[m] / 65536) - max(1, math.ceil(size[m] / 262144)) for m in d2d if size[m] > 65536}
        raw_kinds = {gid(t): sorted({hp.links_dict[l].link_type for br in e.path_list for l in br}) for t, e in raw.items() if e.event_type == "communication"}
        ev, cst = E.chunk_events(raw, hp, 256 * 1024, True)
        deps = {t: set(e.dependency_set) for t, e in ev.items()}
        paths = {t: [list(b) for b in e.path_list] for t, e in ev.items() if e.event_type == "communication"}
        g_ = cpcm.compile_bb(ev, hp, path_lists=paths, deps=deps)
        st_ = [g_.tag[e][1] - 10 for e in range(g_.n)]
        sc = cpcm.simulate(g_, "hop", prio=st_)
        an = An(sc, sink_deadline=[sc.T] * g_.n)
        last0 = max((e for e in range(g_.n) if st_[e] == 0), key=lambda e: (sc.F[e], e))
        rows = an.path_from(last0)
        phi_c = collections.Counter()
        for j, row in enumerate(rows):
            t_out = rows[j + 1]["s"] if j + 1 < len(rows) else sc.F[last0]
            phi_c[gid(g_.tag[row["event"]])] += t_out - row["s"]
        ownw = collections.Counter()
        for a in range(len(sc.a_s)):
            if st_[sc.a_ev[a]] != 0:
                continue
            lo, hi = an.can(a), sc.a_s[a]
            if hi <= lo:
                continue
            for u in sc.res_seq[sc.a_res[a]]:
                ov = min(sc.a_f[u], hi) - max(sc.a_s[u], lo)
                if ov > 0 and st_[sc.a_ev[u]] == 1:
                    ownw[gid(g_.tag[sc.a_ev[u]])] += ov
        kinds = collections.Counter()
        for j, row in enumerate(rows):
            t_out = rows[j + 1]["s"] if j + 1 < len(rows) else sc.F[last0]
            kinds[(st_[row["event"]], g_.res_kind[row["res"]])] += t_out - row["s"]
        ownshare = sum(v for m, v in phi_c.items() if m[0] == 10)
        othershare = sum(v for m, v in phi_c.items() if m[0] == 11)
        C_c = [max(sc.F[e] for e in range(g_.n) if st_[e] == i) for i in (0, 1)]
        # backend
        with gzip.open(os.path.join(E.OUT, "diag", "sims", f"chunk_a2_{hw}_256d2d_prio_01.pkl.gz"), "rb") as f:
            runs = pickle.load(f)["runs"]
        bk = []
        for r in runs:
            cap = r["cap"]
            sim = A.SimR(cap)
            last, rws, segs = DG.chain_for(sim, 0)
            phi = collections.Counter()
            kd = collections.Counter()
            for row in rws:
                if row["tag"] is not None:
                    phi[gid(row["tag"])] += row["t_out"] - row["s"]
                    kd[row["kind"]] += row["t_out"] - row["s"]
            ow = collections.Counter()
            for k_, X in sim.act.items():
                if X["tag"][1] != 10 or X["res"][0] != "link" or X["s"] <= X["can"]:
                    continue
                for ss, ff, tag in cap["links"][X["res"][1]]:
                    ov = min(ff, X["s"]) - max(ss, X["can"])
                    if ov > 0 and tag[1] == 11:
                        ow[gid(tag)] += ov
            C = [max(e["f"] for t, e in cap["events"].items() if t[1] - 10 == i) for i in (0, 1)]
            bk.append({"seed": r.get("seed"), "C": C, "phi": phi, "own": ow, "kinds": kd, "chain_rows": len(rws)})
        J_cb = [jac(top({m: v for m, v in phi_c.items() if m[0] == 11}), top({m: v for m, v in b["phi"].items() if m[0] == 11})) for b in bk]
        J_bb = [jac(top({m: v for m, v in bk[i]["phi"].items() if m[0] == 11}), top({m: v for m, v in bk[j]["phi"].items() if m[0] == 11})) for i in range(len(bk)) for j in range(i)]
        out[hw] = {
            "cst": {k: cst[k] for k in ("messages_chunked", "blocks_added", "n_events")},
            "eligible_d2d_gt64K_msgs": len(added), "added_all_64": sum(added.values()),
            "added_by_stream": {s: sum(v for m, v in added.items() if m[0] == s) for s in (10, 11)},
            "cpcm": {"C": C_c, "chain_time_by_stream_kind": {str(k): v for k, v in kinds.items()}, "phi_own": ownshare, "phi_other": othershare,
                     "conc_other": conc({m: v for m, v in phi_c.items() if m[0] == 11}, added)},
            "backend": [{"C": b["C"], "wait_by_kind": dict(b["kinds"]), "phi_own": sum(v for m, v in b["phi"].items() if m[0] == 10),
                         "phi_other": sum(v for m, v in b["phi"].items() if m[0] == 11),
                         "conc_other": conc({m: v for m, v in b["phi"].items() if m[0] == 11}, added)} for b in bk],
            "own_top": [(str(m), v, size.get(m), m in d2d, str(raw_kinds.get(m))) for m, v in ownw.most_common(8)],
            "own_share_noneligible": sum(v for m, v in ownw.items() if m not in added) / sum(ownw.values()),
            "own_share_nond2d": sum(v for m, v in ownw.items() if m not in d2d) / sum(ownw.values()),
            "own_share_nond2d_be": [sum(v for m, v in b["own"].items() if m not in d2d) / sum(b["own"].values()) for b in bk],
            "added_nond2d_256": sum(max(0, math.ceil(size[m] / 262144) - 1) for m in size if m not in d2d),
            "added_nond2d_64": sum(max(0, math.ceil(size[m] / 65536) - 1) for m in size if m not in d2d),
            "nond2d_gt256": sum(1 for m in size if m not in d2d and size[m] > 262144),
            "ownwait": {"cpcm": conc(ownw, added), "backend": [conc(b["own"], added) for b in bk],
                        "J10_cb": [jac(top(ownw), top(b["own"])) for b in bk], "J10_bb": [jac(top(bk[i]["own"]), top(bk[j]["own"])) for i in range(len(bk)) for j in range(i)],
                        "J30_cb": [jac(top(ownw, 30), top(b["own"], 30)) for b in bk], "J30_bb": [jac(top(bk[i]["own"], 30), top(bk[j]["own"], 30)) for i in range(len(bk)) for j in range(i)]},
            "jaccard_top10_other_cpcm_vs_backend": J_cb, "jaccard_top10_other_backend_pairs": J_bb,
        }
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
