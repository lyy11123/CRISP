"""Bit-identical speed-up of e4_overlap/e4b_backend.event_driver_e4b (E2E_HANDOFF.md section 3). Same signature, same results,
same calls to random and to the policy; only the data structures change. e4_overlap is imported, not changed.

What the original does per tick, and how this file gets the same thing cheaper:
- start_events is a list that is stably re-sorted by start_time at the end of every tick; the scan and the ready list visit the
  events with start_time == now in list order. After the first tick the list is sorted, and the events at `now` sit at its front;
  events released during the tick are appended at its end. So the list is kept as buckets (start_time -> deque, in list order):
  at the end of a tick the surviving events of the front group go, by their new start_time, to the FRONT of that bucket (they were
  before every other event of that time) and those released during the tick to its BACK, each group in its own order. That is
  what the stable sort gives. The first tick (the initial list need not be sorted) uses a real stable sort.
- the ready list of each round is the previous round's list (in list order, before the shuffle) without the events whose path is
  used up, so it is not rebuilt from the whole list; `e in start_events` becomes a set lookup.
- from the second round of a tick on, only the events that won a link in the previous round are offered again: the others keep
  their heads, which are all on busy links by then, so they would only be deferred again to the same times (the shuffle still
  gets the whole ready list);
- candidates: one pass over the branches per event (first appearance of each first link, longest branch through it), in the
  original's order; with the default policy P1 the winner is found without building key tuples (same key, same tie rule).
random.shuffle gets the same list (same length, same order) at the same point, so the random sequence is unchanged.
"""
import collections
import heapq
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "e4_overlap"))
from e4_backend import stream_of  # noqa: E402
from e4b_backend import P1  # noqa: E402


def event_driver_fast(events_dict, hardware_platform, release=None, policy=None, released=None, audit=None,
                      shadow=None, shadow_log=None):
    policy = policy or P1()
    p1 = type(policy) is P1
    links_dict = hardware_platform.links_dict
    modules_dict = hardware_platform.modules_dict
    init_list = []
    wait_events = {}
    event_number = len(events_dict)
    release = release or {}
    if released is None:
        released = {}
    for event_tag in events_dict:
        ev = events_dict[event_tag]
        if ev.dependency_set == set():
            s = stream_of(event_tag)
            init_list.append(ev)
            ev.start_time = release.get(s, 0)
            released.setdefault(s, release.get(s, 0))
        else:
            wait_events[event_tag] = ev
    alive = set(init_list)            # the members of the original's start_events
    buckets = {}                      # start_time -> deque, list order (after the first tick)
    bheap = []
    first = True

    current_time = 0
    pure_comp_time = 0.0
    pure_comm_time = 0.0
    working_computation_devices = set()
    working_communication_devices = set()
    issued_events = {}
    events_record = {}
    iter_cnt = 0
    on_issue = policy.on_issue

    while alive or wait_events or issued_events:
        release_computation_devices = set()
        release_communication_devices = set()
        if working_computation_devices:
            for node_type, node_idx in working_computation_devices:
                if modules_dict[node_type][node_idx].work_endtime <= current_time:
                    modules_dict[node_type][node_idx].work_flag = False
                    release_computation_devices.add((node_type, node_idx))
        working_computation_devices.difference_update(release_computation_devices)
        if working_communication_devices:
            for node_idx in working_communication_devices:
                if links_dict[node_idx].work_endtime <= current_time:
                    links_dict[node_idx].work_flag = False
                    release_communication_devices.add(node_idx)
            working_communication_devices.difference_update(release_communication_devices)

        if first:
            front = [e for e in init_list if e.start_time == current_time]
        else:
            dq = buckets.pop(current_time, None)
            front = list(dq) if dq is not None else []
        back = []
        if current_time in issued_events:
            for finished_event in issued_events[current_time]:
                policy.on_finish(finished_event, current_time)
                for dep_event_tag in finished_event.issue_set:
                    w = wait_events[dep_event_tag]
                    w.dependency_set.remove(finished_event.event_tag)
                    if w.dependency_set == set():
                        w.start_time = current_time
                        back.append(w)
                        alive.add(w)
                        del wait_events[dep_event_tag]
            del issued_events[current_time]

        group = front + back
        for event in group:
            if current_time == event.start_time:
                if event.event_type == "computation":
                    dev = modules_dict[event.comp_device][event.comp_location]
                    if dev.work_flag:
                        event.start_time = dev.work_endtime
                    else:
                        computation_time = dev.working_time(source_datashape=event.comp_datashape,
                                                            beha_type=event.comp_type,
                                                            frequency=hardware_platform.frequency)
                        dev.work_flag = True
                        dev.work_endtime = current_time + computation_time
                        dev.work_record.append([current_time, current_time + computation_time, event.event_tag])
                        working_computation_devices.add((event.comp_device, event.comp_location))
                        event.end_time = dev.work_endtime
                        if event.end_time in issued_events:
                            issued_events[event.end_time].add(event)
                            events_record[event.end_time].add(event)
                        else:
                            issued_events[event.end_time] = {event}
                            events_record[event.end_time] = {event}
                        alive.discard(event)
                elif event.event_type == "communication":
                    if event.path_list == []:
                        event.path_list = event.get_paths(
                            hardware_platform.xy_multicast_path(
                                source_node_idx=event.source_location,
                                target_nodes_set=event.target_location if type(event.target_location) is set else {event.target_location}))

        event_starttime_dict = {}
        last_issued = {}
        base = [e for e in group if e.event_type == "communication" and current_time == e.start_time and e.path_list and e in alive]
        won = None          # events that won a link in the previous round (None: first round of the tick)
        while True:
            ready = [e for e in base if e.path_list]
            base = list(ready)
            random.shuffle(ready)
            policy.on_tick(current_time, ready)
            cands = {}
            won_now = set()
            for e in ready:
                # an event that won nothing in the previous round has the same heads, all on links now busy (each was busy or
                # was just issued): it would only be deferred again to the same times, and it adds no link the others need
                if won is not None and e not in won:
                    continue
                heads = {}
                for br in e.path_list:
                    if not br:
                        continue
                    link = br[0]
                    m = heads.get(link)
                    if m is None or len(br) > m:
                        heads[link] = len(br)
                for link, ln in heads.items():
                    c = cands.get(link)
                    if c is None:
                        cands[link] = [(e, ln)]
                    else:
                        c.append((e, ln))
            issued_any = False
            for link, cs in cands.items():
                L = links_dict[link]
                if L.work_flag:
                    we = L.work_endtime
                    for x, _ in cs:
                        st = x.start_time
                        if st is not None:
                            cand = we if we > st else None
                        else:
                            cand = we
                        if cand:
                            tg = x.event_tag
                            old = event_starttime_dict.get(tg)
                            if old is None:
                                if tg not in event_starttime_dict:
                                    event_starttime_dict[tg] = cand
                            elif cand < old:
                                event_starttime_dict[tg] = cand
                    continue
                if p1:
                    i0, ln0 = 0, cs[0][1]
                    for i in range(1, len(cs)):
                        if cs[i][1] > ln0:
                            i0, ln0 = i, cs[i][1]
                    e, ln = cs[i0]
                    if audit is not None:
                        k0 = (-ln0,)
                        audit.append((current_time, link, k0, k0, len(cs)))
                else:
                    keyed = [(policy.key(x, link, l, current_time), i, x, l) for i, (x, l) in enumerate(cs)]
                    k0, _, e, ln = min(keyed, key=lambda z: (z[0], z[1]))
                    if audit is not None:
                        audit.append((current_time, link, k0, min(policy.key(x, link, l, current_time) for x, l in cs),
                                      len(cs)))
                if shadow is not None and len(cs) > 1:
                    sk = min(((shadow.key(x, link, l, current_time), i, x) for i, (x, l) in enumerate(cs)),
                             key=lambda z: (z[0], z[1]))
                    shadow_log["n"] = shadow_log.get("n", 0) + 1
                    if sk[2] is not e:
                        shadow_log["differ"] = shadow_log.get("differ", 0) + 1
                    if stream_of(sk[2].event_tag) != stream_of(e.event_tag):
                        shadow_log["differ_stream"] = shadow_log.get("differ_stream", 0) + 1
                communication_time = L.working_time(data_bytes=e.comm_bytes)
                L.work_flag = True
                L.work_endtime = we = current_time + communication_time
                L.work_record.append([current_time, we, e.event_tag])
                working_communication_devices.add(link)
                on_issue(e, link, current_time)
                last_issued[e.event_tag] = link
                issued_any = True
                won_now.add(e)
                pl = e.path_list
                for br in pl[:]:
                    if br and br[0] == link:
                        br.pop(0)
                if [] in pl:
                    for br in pl[:]:
                        if br == []:
                            pl.remove(br)
                for x, _ in cs:
                    if x is not e:
                        st = x.start_time
                        if st is not None:
                            cand = we if we > st else None
                        else:
                            cand = we
                        if cand:
                            tg = x.event_tag
                            old = event_starttime_dict.get(tg)
                            if old is None:
                                if tg not in event_starttime_dict:
                                    event_starttime_dict[tg] = cand
                            elif cand < old:
                                event_starttime_dict[tg] = cand
            for e in ready:
                if e.path_list == [] and e in alive:
                    e.end_time = end = links_dict[last_issued[e.event_tag]].work_endtime
                    if end in issued_events:
                        issued_events[end].add(e)
                        events_record[end].add(e)
                    else:
                        issued_events[end] = {e}
                        events_record[end] = {e}
                    alive.discard(e)
                    event_starttime_dict.pop(e.event_tag, None)
            if not issued_any:
                break
            won = won_now
        if event_starttime_dict:
            for event in group:
                if event in alive and event.event_tag in event_starttime_dict:
                    event.start_time = event_starttime_dict[event.event_tag]

        if not alive and not wait_events and not issued_events:
            break
        if not alive and not issued_events:
            raise ValueError("Event-driven scheduler failed to converge")
        # the original's stable sort of start_events by start_time
        if first:
            first = False
            lst = [e for e in init_list if e in alive] + [e for e in back if e in alive]
            lst.sort(key=lambda x: x.start_time)
            for e in lst:
                dq = buckets.get(e.start_time)
                if dq is None:
                    buckets[e.start_time] = dq = collections.deque()
                    heapq.heappush(bheap, e.start_time)
                dq.append(e)
        else:
            fr = {}
            for e in front:
                if e in alive:
                    fr.setdefault(e.start_time, []).append(e)
            for t, es in fr.items():
                dq = buckets.get(t)
                if dq is None:
                    buckets[t] = collections.deque(es)
                    heapq.heappush(bheap, t)
                else:
                    dq.extendleft(reversed(es))
            for e in back:
                if e in alive:
                    t = e.start_time
                    dq = buckets.get(t)
                    if dq is None:
                        buckets[t] = dq = collections.deque()
                        heapq.heappush(bheap, t)
                    dq.append(e)
        while bheap and bheap[0] not in buckets:
            heapq.heappop(bheap)
        if not alive and issued_events:
            next_time = min(issued_events.keys())
        elif not issued_events and alive:
            next_time = buckets[bheap[0]][0].start_time     # the original's start_events[0].start_time (same object, so same type)
        else:
            next_starttime = buckets[bheap[0]][0].start_time
            next_endtime = min(issued_events.keys())
            next_time = next_starttime if next_starttime < next_endtime else next_endtime
        delta = next_time - current_time
        if delta > 0:
            if working_computation_devices and not working_communication_devices:
                pure_comp_time += delta
            elif working_communication_devices and not working_computation_devices:
                pure_comm_time += delta
        current_time = next_time
        iter_cnt += 1
        if iter_cnt > ((event_number) * 100):
            raise ValueError("Event-driven scheduler failed to converge")
    return int(current_time), int(pure_comp_time), int(pure_comm_time)
