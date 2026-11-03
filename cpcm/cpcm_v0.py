"""V0: hand-made graphs, CPCM vs hand calculation, event by event (cpcm/CPCM_HANDOFF.md section 3).

Time unit 0.1 ms in the plan-document examples (section 5 (2)): link 100 B/ms, message 100 B -> 10 units;
R ready at 0.2 ms is modelled as a predecessor computation Rpre of 2 units on its own device.
Every expectation below was worked out by hand from the mode semantics in cpcm.py's docstring (the derivations
are in the comments). Usage: python cpcm_v0.py   (writes cpcm_out/v0.json, exit code 1 on any mismatch)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpcm  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpcm_out")


def comp(tag, res, dur, deps=()):
    return {"tag": tag, "kind": "comp", "res": res, "dur": dur, "deps": list(deps)}


def comm(tag, branches, dur, deps=()):
    return {"tag": tag, "kind": "comm", "branches": branches, "dur": dur, "deps": list(deps)}


def ex1_events(p_link="L1", kv=None):
    ev = [comp("Rpre", "devA", 2), comm("R", [["L1"]], 10, ["Rpre"]), comm("P", [[p_link]], 10),
          comp("FFN", "devB", 10, ["R", "P"])]
    if kv:
        ev.append(comm("KV", [[kv]], 10))
    return ev


def ex2_events():
    # L2 failed: simply not in the resource table
    return [comp("Rpre", "devA", 2), comm("R", [["L1"]], 10, ["Rpre"]), comm("P2", [["L1"]], 10),
            comp("up", "devU", 20, ["R"]), comp("down", "devD", 10, ["up", "P2"])]


# Each case: graph, how to run it, and the hand results.
#   acts: {(event, resource): (s, f)}   events: {event: (r, S, S_last, F)}   T
#   lst: {(event, resource): LST}       D: {event: D}   cp: [(event, resource, edge, wait)]
#   cb: {resource: CB}                  phi: {resource: {event: phi}}       W: {event: W}
CASES = []

# --- Plan doc example 1, original mapping, BusyBarn FCFS (bb).
# t=0: Rpre on devA [0,2]; P is the only ready message, L1 free -> [0,10]. t=2: R ready, L1 busy until 10 ->
# retried at 10 -> [10,20]. t=20: FFN ready -> [20,30]. T=30.
# Backward (T_end 30): LST(FFN)=20; R->FFN: 20-10=10; P->FFN: 10, P->R (L1 order): LST(R)-10=0 -> 0;
# Rpre->R: 10-2=8. D(P)=D(R)=LSTev(FFN)=20 (plan doc: D(P)=2 ms); D(Rpre)=LSTev(R)=10; D(FFN)=T_end=30.
# CP from FFN: data <- R (F 20); R waited 2..10 on L1 behind P (P ends at 10) -> CB(L1)=8, phi(P)=8; P start.
CASES.append(dict(
    name="ex1_original_bb", doc="plan doc (2) example 1, original: P [0,1], R [1,2] on L1, FFN starts 2.0 ms",
    events=ex1_events(), mode="bb",
    acts={("Rpre", "devA"): (0, 2), ("P", "L1"): (0, 10), ("R", "L1"): (10, 20), ("FFN", "devB"): (20, 30)},
    evs={"Rpre": (0, 0, 0, 2), "P": (0, 0, 0, 10), "R": (2, 10, 10, 20), "FFN": (20, 20, 20, 30)}, T=30,
    lst={("FFN", "devB"): 20, ("R", "L1"): 10, ("P", "L1"): 0, ("Rpre", "devA"): 8},
    D={"P": 20, "R": 20, "Rpre": 10, "FFN": 30},
    cp=[("P", "L1", "start", 0), ("R", "L1", "resource", 8), ("FFN", "devB", "data", 0)],
    cb={"L1": 8}, phi={"L1": {"P": 8}}, W={"P": 0, "R": 8, "FFN": 0, "Rpre": 0}))

# --- Example 1, "defer P": imposed order R before P on L1 (order mode).
# R [2,12] (ready 2, L1 first in order); P: ready 0 but must follow R -> [12,22]; FFN [22,32] (plan doc 2.2 ms).
# Backward (T 32): FFN 22; P->FFN 12; R->FFN 12, R->P (L1 order): LST(P)-10=2 -> 2; Rpre 0.
# CP: FFN data <- P (F 22, latest pred); P can=0, s=12: resource edge to R (ends 12) -> wait 12 on L1;
# L1 served R over [2,12] only: phi(R)=10 (idle 0..2, not work-conserving); R data <- Rpre; Rpre start.
CASES.append(dict(
    name="ex1_defer_P_order", doc="example 1, defer P: R [0.2,1.2], P [1.2,2.2], FFN 2.2 ms",
    events=ex1_events(), mode="order", seq={"L1": ["R", "P"]},
    acts={("Rpre", "devA"): (0, 2), ("R", "L1"): (2, 12), ("P", "L1"): (12, 22), ("FFN", "devB"): (22, 32)},
    evs={"Rpre": (0, 0, 0, 2), "R": (2, 2, 2, 12), "P": (0, 12, 12, 22), "FFN": (22, 22, 22, 32)}, T=32,
    lst={("FFN", "devB"): 22, ("P", "L1"): 12, ("R", "L1"): 2, ("Rpre", "devA"): 0},
    D={"P": 22, "R": 22, "Rpre": 2},
    cp=[("Rpre", "devA", "start", 0), ("R", "L1", "data", 0), ("P", "L1", "resource", 12),
        ("FFN", "devB", "data", 0)],
    cb={"L1": 12}, phi={"L1": {"R": 10}}, W={"P": 12, "R": 0}))

# --- Example 1, P moved to the idle L2 (bb), evaluated through the reroute interface on the original mapping.
# P [0,10] on L2; R [2,12] on L1; FFN [12,22] (plan doc 1.2 ms). F'(P)=10 <= D(P)=20 -> ok (plan doc: passes).
CASES.append(dict(
    name="ex1_reroute_P_L2", doc="example 1, P on L2: FFN 1.2 ms; reroute check F'(P)=1 <= D(P)=2",
    events=ex1_events(), extra_res=["L2"], mode="bb", reroute=("P", [["L2"]], {"L2": 10}),
    acts={("Rpre", "devA"): (0, 2), ("P", "L2"): (0, 10), ("R", "L1"): (2, 12), ("FFN", "devB"): (12, 22)},
    evs={"P": (0, 0, 0, 10), "R": (2, 2, 2, 12), "FFN": (12, 12, 12, 22)}, T=22,
    reroute_expect={"F_b": 10, "D_b": 20, "ok": True, "T": 22}))

# --- Example 1 with a KV write-back (ready 1.5 ms) on the other link: identical per-link aggregate load,
# FFN start 2.0 vs 1.2 ms. Mapping A: P on L1, KV on L2. Mapping B: P on L2, KV on L1.
# A: L1 P [0,10], R [10,20]; L2 KV [15,25]; FFN [20,30]; T=30.
# B: L2 P [0,10]; L1 R [2,12], KV [15,25]; FFN [12,22]; T=max(22,25)=25.
CASES.append(dict(
    name="ex1_kv_mapA_bb", doc="example 1 + KV write-back, mapping A (P on L1, KV on L2)",
    events=ex1_events("L1", kv="L2"), mode="bb", release={"KV": 15},
    acts={("P", "L1"): (0, 10), ("R", "L1"): (10, 20), ("KV", "L2"): (15, 25), ("FFN", "devB"): (20, 30)},
    evs={"FFN": (20, 20, 20, 30), "KV": (15, 15, 15, 25)}, T=30, load={"L1": 20, "L2": 10}))
CASES.append(dict(
    name="ex1_kv_mapB_bb", doc="example 1 + KV write-back, mapping B (P on L2, KV on L1)",
    events=ex1_events("L2", kv="L1"), mode="bb", release={"KV": 15},
    acts={("P", "L2"): (0, 10), ("R", "L1"): (2, 12), ("KV", "L1"): (15, 25), ("FFN", "devB"): (12, 22)},
    evs={"FFN": (12, 12, 12, 22), "KV": (15, 15, 15, 25)}, T=25, load={"L1": 20, "L2": 10}))

# --- Plan doc example 2, FCFS (bb), L2 failed. P2 = P' (down-proj weight prefetch).
# t=0 Rpre [0,2], P2 [0,10] on L1; t=2 R waits for L1 -> [10,20]; up [20,40]; down needs up and P2 -> [40,50].
# Backward (T 50): down 40; up 20; R->up 10; P2->down 30, P2->R (L1) 10-10=0 -> 0; Rpre 8.
# D(P2)=LSTev(down)=40 (plan doc D(P')=4 ms); D(R)=LSTev(up)=20; D(up)=40.
# CP: down data <- up (F 40 > F(P2) 10); up data <- R; R wait 2..10 on L1 behind P2; P2 start.
CASES.append(dict(
    name="ex2_fcfs_bb", doc="plan doc (2) example 2, FCFS: up 2.0 ms, down 4.0 ms",
    events=ex2_events(), mode="bb",
    acts={("Rpre", "devA"): (0, 2), ("P2", "L1"): (0, 10), ("R", "L1"): (10, 20), ("up", "devU"): (20, 40),
          ("down", "devD"): (40, 50)},
    evs={"P2": (0, 0, 0, 10), "R": (2, 10, 10, 20), "up": (20, 20, 20, 40), "down": (40, 40, 40, 50)}, T=50,
    lst={("down", "devD"): 40, ("up", "devU"): 20, ("R", "L1"): 10, ("P2", "L1"): 0, ("Rpre", "devA"): 8},
    D={"P2": 40, "R": 20, "up": 40, "Rpre": 10},
    cp=[("P2", "L1", "start", 0), ("R", "L1", "resource", 8), ("up", "devU", "data", 0),
        ("down", "devD", "data", 0)],
    cb={"L1": 8}, phi={"L1": {"P2": 8}}, W={"R": 8}))

# --- Example 2, defer P': order R before P2 on L1.
# R [2,12]; P2 [12,22]; up [12,32]; down ready max(32,22)=32 -> [32,42] (plan doc: up 1.2, down 3.2, 2.2 <= 3.2).
# Backward (T 42): down 32; up 12; P2->down 22; R->up 2, R->P2 (L1) 22-10=12 -> 2; Rpre 0. D(P2)=32.
CASES.append(dict(
    name="ex2_defer_P2_order", doc="example 2, defer P': up 1.2 ms, down 3.2 ms, P' arrives 2.2 <= 3.2",
    events=ex2_events(), mode="order", seq={"L1": ["R", "P2"]},
    acts={("R", "L1"): (2, 12), ("P2", "L1"): (12, 22), ("up", "devU"): (12, 32), ("down", "devD"): (32, 42)},
    evs={"P2": (0, 12, 12, 22), "down": (32, 32, 32, 42)}, T=42,
    lst={("down", "devD"): 32, ("up", "devU"): 12, ("P2", "L1"): 22, ("R", "L1"): 2, ("Rpre", "devA"): 0},
    D={"P2": 32, "R": 12}))

# --- C1 multi-hop, no store-and-forward, length layering (bb). A: l1->l2->l3 (5 each); B: l2 only, B has the
# smaller id. t=0 level 3: A issues l1; level 2: A issues l2 (same tick); level 1: order (-1,B),(-1,A): B's head
# l2 is busy until 5 -> B retried at 5; A issues l3. A: all three hops [0,5], F=5 (store-and-forward would be 15).
# B [5,10]. Backward (T 10): B 5; A.l2 -> B (l2 order) 5-5=0; A.l3 5; A.l1 <- children l2 (0) -> 0.
# CP: B waited 0..5 on l2 behind A's l2 hop; A.l2 hop edge <- A.l1 start. CB(l2)=5, phi(A)=5.
CASES.append(dict(
    name="c1_multihop_layering_bb", doc="3-hop message issues all hops in one tick; longest path first",
    events=[comm("B", [["l2"]], 5), comm("A", [["l1", "l2", "l3"]], 5)], mode="bb",
    acts={("A", "l1"): (0, 5), ("A", "l2"): (0, 5), ("A", "l3"): (0, 5), ("B", "l2"): (5, 10)},
    evs={"A": (0, 0, 0, 5), "B": (0, 5, 5, 10)}, T=10,
    lst={("B", "l2"): 5, ("A", "l2"): 0, ("A", "l1"): 0, ("A", "l3"): 5},
    cp=[("A", "l1", "start", 0), ("A", "l2", "hop", 0), ("B", "l2", "resource", 5)],
    cb={"l2": 5}, phi={"l2": {"A": 5}}, W={"B": 5, "A": 0}))

# --- C2 hold-and-wait (bb). B: l2, 20 units (id 0). A: l1->l2, 5 units per hop (id 1). C: l1, ready 3 (id 2).
# t=0 level 2: A issues l1 [0,5]; level 1: B (id 0) takes l2 [0,20]; A's l2 busy -> A retried at 20.
# t=3: C ready, l1 busy (A's hop) until 5 -> C [5,10]. t=20: A's l2 [20,25]; F(A)=25, S=0, S_last=20.
# Backward (T 25): A.l2 20; B -> A.l2 (l2 order) 20-20=0; C 20; A.l1: child l2 -> 20, next on l1 (C) 20-5=15 -> 15.
# CP: A.l2 can = start of A.l1 = 0, s=20: waited on l2 behind B (ends 20); B start. CB(l2)=20, phi(B)=20.
CASES.append(dict(
    name="c2_hold_and_wait_bb", doc="first hop issued while the second waits; the early hop still blocks C",
    events=[comm("B", [["l2"]], 20), comm("A", [["l1", "l2"]], 5), comm("C", [["l1"]], 5)], mode="bb",
    release={"C": 3},
    acts={("B", "l2"): (0, 20), ("A", "l1"): (0, 5), ("A", "l2"): (20, 25), ("C", "l1"): (5, 10)},
    evs={"A": (0, 0, 20, 25), "B": (0, 0, 0, 20), "C": (3, 5, 5, 10)}, T=25,
    lst={("A", "l2"): 20, ("B", "l2"): 0, ("C", "l1"): 20, ("A", "l1"): 15},
    cp=[("B", "l2", "start", 0), ("A", "l2", "resource", 20)],
    cb={"l2": 20}, phi={"l2": {"B": 20}}, W={"A": 20, "C": 2, "B": 0}))

# --- C3 multicast (bb). C: l3, 12 units (id 0). A: branches l1->l2 and l1->l3, 5 per hop (id 1).
# t=0 level 2: A's branch 0 issues l1 [0,5]; branch 1 shares head l1 -> trimmed too (one l1 hop).
# level 1: C (id 0) takes l3 [0,12]; A: l2 free -> [0,5], l3 busy -> retried at 12 -> [12,17]. F(A)=17.
# Backward (T 17): A.l3 12, A.l2 12, A.l1 min(12,12)=12; C -> A.l3 (l3 order): 12-12=0.
CASES.append(dict(
    name="c3_multicast_bb", doc="multicast tree: shared head issued once, one branch blocked",
    events=[comm("C", [["l3"]], 12), comm("A", [["l1", "l2"], ["l1", "l3"]], 5)], mode="bb",
    acts={("C", "l3"): (0, 12), ("A", "l1"): (0, 5), ("A", "l2"): (0, 5), ("A", "l3"): (12, 17)},
    evs={"A": (0, 0, 12, 17), "C": (0, 0, 0, 12)}, T=17, n_acts={"A": 3},
    lst={("A", "l3"): 12, ("A", "l2"): 12, ("A", "l1"): 12, ("C", "l3"): 0},
    cp=[("C", "l3", "start", 0), ("A", "l3", "resource", 12)],
    cb={"l3": 12}, phi={"l3": {"C": 12}}))

# --- C4 BusyBarn's end time = end of the LAST ISSUED hop (bb). A: two branches, x (3-tuple key, 9 units) and
# l1 (5 units), both length 1. Inside one event the backend sorts branches with a remaining 3-tuple link first,
# so x is issued first, l1 last: F(A) = 5 although x ends at 9. Z (comp, 1 unit) depends on A: [5,6]. T=6.
CASES.append(dict(
    name="c4_last_issued_end_bb", doc="F(e) = end of the last issued hop, not the latest-ending hop",
    events=[comm("A", [["l1"], ["x"]], {"l1": 5, "x": 9}), comp("Z", "devZ", 1, ["A"])], mode="bb",
    res=[("x", "link", True)],
    acts={("A", "x"): (0, 9), ("A", "l1"): (0, 5), ("Z", "devZ"): (5, 6)},
    evs={"A": (0, 0, 0, 5), "Z": (5, 5, 5, 6)}, T=6))

# --- C5 compute contention (bb). E1, E2 on devX, 5 units, both ready at 0 (E1 smaller id); E3 on devY after E1.
# E1 [0,5]; E2 retried at 5 -> [5,10]; E3 [5,8]. T=10. Backward: E2 5; E1 -> E2 (devX) 0, -> E3 (10-3)-5=2 -> 0.
CASES.append(dict(
    name="c5_compute_contention_bb", doc="same-tick compute events on one device: id order, deferral",
    events=[comp("E1", "devX", 5), comp("E2", "devX", 5), comp("E3", "devY", 3, ["E1"])], mode="bb",
    acts={("E1", "devX"): (0, 5), ("E2", "devX"): (5, 10), ("E3", "devY"): (5, 8)},
    evs={"E2": (0, 5, 5, 10), "E3": (5, 5, 5, 8)}, T=10,
    lst={("E2", "devX"): 5, ("E1", "devX"): 0, ("E3", "devY"): 7},
    cp=[("E1", "devX", "start", 0), ("E2", "devX", "resource", 5)], cb={"devX": 5}, W={"E2": 5}))

# --- C6 bb vs hop. Y: l1 (id 0); X: l1->l2 (id 1), 5 per hop.
# bb: level 2 X issues l1 [0,5] and l2 [0,5] (level 1); Y retried at 5 -> [5,10].
# hop with prio Y=0 < X=1: round 1 candidates on l1: Y (ln 1, key (0,-1,0)), X (ln 2, key (1,-2,1)) -> Y wins
# [0,5]; X deferred to 5; t=5: X's l1 and, next round, l2 -> [5,10] both. T=10 either way.
c6 = [comm("Y", [["l1"]], 5), comm("X", [["l1", "l2"]], 5)]
CASES.append(dict(
    name="c6a_longest_first_bb", doc="bb: the longer message wins the shared first link",
    events=c6, mode="bb",
    acts={("X", "l1"): (0, 5), ("X", "l2"): (0, 5), ("Y", "l1"): (5, 10)}, evs={"Y": (0, 5, 5, 10)}, T=10))
CASES.append(dict(
    name="c6b_hop_priority", doc="hop: the fixed priority key decides each link",
    events=c6, mode="hop", prio={"Y": 0, "X": 1},
    acts={("Y", "l1"): (0, 5), ("X", "l1"): (5, 10), ("X", "l2"): (5, 10)}, evs={"X": (0, 5, 5, 10)}, T=10))
CASES.append(dict(
    name="c6c_hop_default_P1", doc="hop without priority = E4b P1 (-ln): same as bb here",
    events=c6, mode="hop",
    acts={("X", "l1"): (0, 5), ("X", "l2"): (0, 5), ("Y", "l1"): (5, 10)}, evs={"Y": (0, 5, 5, 10)}, T=10))

# --- C7 an imposed order that contradicts a data dependency must be rejected.
CASES.append(dict(
    name="c7_order_deadlock", doc="order mode rejects an order that deadlocks against the data edges",
    events=[comm("A", [["L1"]], 5), comm("B", [["L1"]], 5, ["A"])], mode="order", seq={"L1": ["B", "A"]},
    expect_error="deadlock"))


def run_case(c):
    res = c.get("res", []) + [(r, "link", False) for r in c.get("extra_res", [])]
    g = cpcm.build(c["events"], res)
    tid = g.id_of
    q = g.res_id
    release = {tid[k]: v for k, v in c.get("release", {}).items()} or None
    out = {"name": c["name"], "doc": c["doc"], "mode": c["mode"], "mismatches": []}
    mm = out["mismatches"]

    def check(what, got, exp):
        if got != exp:
            mm.append({"what": what, "got": got, "expected": exp})

    try:
        if c["mode"] == "order":
            seq = {q[r]: [tid[t] for t in evs] for r, evs in c["seq"].items()}
            sc = cpcm.simulate_order(g, seq, release=release)
        elif c["mode"] == "hop":
            prio = [c["prio"][g.tag[e]] for e in range(g.n)] if "prio" in c else None
            sc = cpcm.simulate(g, "hop", prio, release=release)
        else:
            sc = cpcm.simulate(g, "bb", release=release)
    except cpcm.CPCMError as ex:
        out["error"] = str(ex)
        if c.get("expect_error") and c["expect_error"] in str(ex):
            out["pass"] = True
        else:
            mm.append({"what": "error", "got": str(ex), "expected": c.get("expect_error")})
            out["pass"] = False
        return out
    if c.get("expect_error"):
        mm.append({"what": "error", "got": None, "expected": c["expect_error"]})

    an = cpcm.Analysis(sc)
    if "reroute" in c:
        b, brs, durs = c["reroute"]
        rr = cpcm.reroute(g, tid[b], [[q[l] for l in br] for br in brs], {q[l]: d for l, d in durs.items()},
                          base=an)
        for k, v in c["reroute_expect"].items():
            check(f"reroute.{k}", rr[k], v)
        sc = rr["schedule"]
        an = cpcm.Analysis(sc)
    acts = {(g.tag[sc.a_ev[a]], g.res_name[sc.a_res[a]]): a for a in range(len(sc.a_s))}
    if len(acts) != len(sc.a_s):
        mm.append({"what": "activities", "got": "an event used one resource twice", "expected": "unique"})
    for k, (s, f) in c["acts"].items():
        a = acts.get(k)
        check(f"act {k}", None if a is None else (sc.a_s[a], sc.a_f[a]), (s, f))
    for t, v in c.get("evs", {}).items():
        e = tid[t]
        check(f"event {t} (r,S,S_last,F)", (sc.r[e], sc.S[e], sc.S_last[e], sc.F[e]), tuple(v))
    check("T", sc.T, c["T"])
    for t, k in c.get("n_acts", {}).items():
        check(f"n_acts {t}", len(sc.ev_acts[tid[t]]), k)
    for k, v in c.get("lst", {}).items():
        check(f"LST {k}", an.lst[acts[k]], v)
    for t, v in c.get("D", {}).items():
        check(f"D({t})", an.deadline(tid[t]), v)
    for t, v in c.get("W", {}).items():
        check(f"W({t})", an.wait(tid[t]), v)
    rows = an.critical_path()
    out["cp"] = [(g.tag[r["event"]], g.res_name[r["res"]], r["edge"], r["wait"], r["s"], r["f"]) for r in rows]
    if "cp" in c:
        check("critical path", [x[:4] for x in out["cp"]], [tuple(x) for x in c["cp"]])
    cb = {g.res_name[k]: v for k, v in an.blocking().items()}
    out["cb"] = cb
    if "cb" in c:
        check("CB", cb, c["cb"])
    if "phi" in c:
        for r, exp in c["phi"].items():
            _, ph = an.blocking(q[r])
            check(f"phi {r}", {g.tag[b]: v for b, v in ph.items()}, exp)
    if "load" in c:
        load = {}
        for a in range(len(sc.a_s)):
            r = g.res_name[sc.a_res[a]]
            if r in c["load"]:
                load[r] = load.get(r, 0) + sc.a_f[a] - sc.a_s[a]
        check("aggregate link load", load, c["load"])
    out["gaps"] = an.gaps
    out["events"] = {g.tag[e]: {"r": sc.r[e], "S": sc.S[e], "S_last": sc.S_last[e], "F": sc.F[e],
                                "D": an.D[e], "W": an.wait(e)} for e in range(g.n)}
    out["acts"] = [{"event": g.tag[sc.a_ev[a]], "res": g.res_name[sc.a_res[a]], "s": sc.a_s[a], "f": sc.a_f[a],
                    "lst": an.lst[a], "slack": an.slack[a]} for a in range(len(sc.a_s))]
    out["T"] = sc.T
    out["pass"] = not mm
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    results = [run_case(c) for c in CASES]
    n_checks = 0
    for c in CASES:
        n_checks += (len(c.get("acts", {})) + len(c.get("evs", {})) + ("T" in c) + len(c.get("lst", {})) +
                     len(c.get("D", {})) + len(c.get("W", {})) + ("cp" in c) + ("cb" in c) + len(c.get("phi", {})) +
                     len(c.get("reroute_expect", {})) + ("load" in c) + len(c.get("n_acts", {})) +
                     ("expect_error" in c))
    summary = {"cases": len(results), "passed": sum(r["pass"] for r in results), "checks": n_checks,
               "mismatches": sum(len(r["mismatches"]) for r in results)}
    with open(os.path.join(OUT, "v0.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": results}, f, indent=1, default=str)
    for r in results:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['name']:28s} T={r.get('T')}  {r['doc']}")
        for m in r["mismatches"]:
            print("      ", m)
    print(summary)
    return 0 if summary["mismatches"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
