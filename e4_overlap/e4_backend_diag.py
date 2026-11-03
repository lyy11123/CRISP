"""E4 diagnosis backends (user request 2026-11-02; measurement only, in memory; e4_backend.py and the artifact are
untouched). Copy of e4_backend.event_driver_e4 with two additions, marked `# DIAG:`:

  mode="orig": the original Phase 4 (length_count layering) plus a log of every deferral of a stream-`watch`
    event: at each tick, for a communication event whose remaining-branch first link is busy, the binding link
    (the one giving the deferred start) and its occupant (last work_record entry: tag, start, end), whether the
    occupant event was partially issued at that moment, the hops the watched event had already issued, and its
    previous deferral; computation events deferred in Phase 3 likewise (device occupant).
  mode="perhop": Phase 4 replaced by per-hop arbitration without length layering. Per tick, repeated rounds: every
    ready communication event offers the first link of each remaining branch; for every free link one candidate
    is chosen by (stream rank if prio, then longest remaining branch), random tie-break by a shuffle; winners are
    issued (shared multicast prefixes removed as in the original), losers and events on busy links are deferred
    to the earliest work_endtime of their busy first links (the original rule). Rounds repeat until nothing is
    issued, so the free hops of a branch are still issued in the same tick. Link occupancy formula, Phase 3 and
    time advance are the original's.
"""
import random

from e4_backend import stream_of


def event_driver_diag(events_dict, hardware_platform, release=None, gate=None, prio=None, released=None,
                      mode="orig", watch=0, log=None):
    start_events = []
    wait_events = {}
    event_number = len(events_dict)
    decode_flag = False
    release = release or {}
    gate = gate or {}
    if released is None:
        released = {}
    if log is None:
        log = {}
    log.setdefault("deferrals", [])
    ready_at = log.setdefault("ready_at", {})
    hops_issued = {}
    fully = set()
    last_def = {}
    remaining = {}
    held = {}
    for event_tag in events_dict:
        s = stream_of(event_tag)
        remaining[s] = remaining.get(s, 0) + 1
    for event_tag in events_dict:
        if events_dict[event_tag].dependency_set == set():
            s = stream_of(event_tag)
            if s in gate:
                held.setdefault(s, []).append(events_dict[event_tag])
                continue
            start_events.append(events_dict[event_tag])
            start_events[-1].start_time = release.get(s, 0)
            released.setdefault(s, release.get(s, 0))
            ready_at[event_tag] = release.get(s, 0)  # DIAG
        else:
            wait_events[event_tag] = events_dict[event_tag]

    current_time = 0
    pure_comp_time = 0.0
    pure_comm_time = 0.0
    working_computation_devices = set()
    working_communication_devices = set()
    issued_events = {}
    events_record = {}
    iter_cnt = 0
    if prio:
        def sort_key(pair):
            return (-prio[stream_of(pair.event_tag)],
                    max((len(lst) for lst in getattr(pair, 'path_list', []) or []), default=0))
    else:
        sort_key = None

    def occupant(link):  # DIAG
        L = hardware_platform.links_dict[link]
        s, f, tag = L.work_record[-1]
        return {"occ": tag, "occ_s": s, "occ_f": f, "occ_partial": tag not in fully and hops_issued.get(tag, 0) > 0}

    def record_issue(event, pp_path):
        communication_time = hardware_platform.links_dict[pp_path].working_time(data_bytes=event.comm_bytes)
        hardware_platform.links_dict[pp_path].work_flag = True
        hardware_platform.links_dict[pp_path].work_endtime = current_time + communication_time
        hardware_platform.links_dict[pp_path].work_record.append(
            [current_time, current_time + communication_time, event.event_tag])
        working_communication_devices.add(pp_path)
        hops_issued[event.event_tag] = hops_issued.get(event.event_tag, 0) + 1  # DIAG

    def finish_comm(event, issued_path):
        event.end_time = hardware_platform.links_dict[issued_path].work_endtime
        if event.end_time in issued_events:
            issued_events[event.end_time].add(event)
            events_record[event.end_time].add(event)
        else:
            issued_events[event.end_time] = {event}
            events_record[event.end_time] = {event}
        start_events.remove(event)
        fully.add(event.event_tag)  # DIAG

    while start_events or wait_events or issued_events:
        release_computation_devices = set()
        release_communication_devices = set()
        if working_computation_devices:
            for node_type, node_idx in working_computation_devices:
                if hardware_platform.modules_dict[node_type][node_idx].work_endtime <= current_time:
                    hardware_platform.modules_dict[node_type][node_idx].work_flag = False
                    release_computation_devices.add((node_type, node_idx))
        working_computation_devices.difference_update(release_computation_devices)
        if working_communication_devices:
            for node_idx in working_communication_devices:
                if hardware_platform.links_dict[node_idx].work_endtime <= current_time:
                    hardware_platform.links_dict[node_idx].work_flag = False
                    release_communication_devices.add(node_idx)
            working_communication_devices.difference_update(release_communication_devices)

        if current_time in issued_events:
            finished_events = issued_events[current_time]
            for finished_event in finished_events:
                remaining[stream_of(finished_event.event_tag)] -= 1
                for dep_event_tag in finished_event.issue_set:
                    wait_events[dep_event_tag].dependency_set.remove(finished_event.event_tag)
                    if wait_events[dep_event_tag].dependency_set == set():
                        wait_events[dep_event_tag].start_time = current_time
                        start_events.append(wait_events[dep_event_tag])
                        ready_at[dep_event_tag] = current_time  # DIAG
                        if decode_flag == False and dep_event_tag[1] == 1 and dep_event_tag[0] == 0:
                            print(f"prefill time cost: {int(current_time), int(pure_comp_time), int(pure_comm_time)}")
                            decode_flag = True
                        del wait_events[dep_event_tag]
            del issued_events[current_time]
            for s, prev in gate.items():
                if s in held and remaining.get(prev, 0) == 0:
                    for ev in held.pop(s):
                        ev.start_time = current_time
                        start_events.append(ev)
                        ready_at[ev.event_tag] = current_time
                    released[s] = current_time

        for event in start_events[:]:
            if current_time == event.start_time:
                if event.event_type == "computation":
                    dev = hardware_platform.modules_dict[event.comp_device][event.comp_location]
                    if dev.work_flag:
                        if stream_of(event.event_tag) == watch:  # DIAG
                            s_, f_, tg = dev.work_record[-1]
                            log["deferrals"].append({"t": current_time, "tag": event.event_tag, "kind": "compute",
                                                     "res": (event.comp_device, event.comp_location),
                                                     "new_start": dev.work_endtime, "occ": tg, "occ_s": s_, "occ_f": f_,
                                                     "occ_partial": False, "hops_done": 0,
                                                     "ready": ready_at.get(event.event_tag), "prev": None})
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
                        start_events.remove(event)
                elif event.event_type == "communication":
                    if event.path_list == []:
                        event.path_list = event.get_paths(
                            hardware_platform.xy_multicast_path(
                                source_node_idx=event.source_location,
                                target_nodes_set=event.target_location if type(event.target_location) is set else {event.target_location}))

        event_starttime_dict = {}
        bind = {}  # DIAG: tag -> occupant info of the binding busy link

        def defer(event, pp_path, length_count):
            L = hardware_platform.links_dict[pp_path]
            if event.start_time is not None:
                cand = L.work_endtime if L.work_endtime > event.start_time else None
            else:
                cand = L.work_endtime
            tg = event.event_tag
            if tg in event_starttime_dict and cand:
                if cand < event_starttime_dict[tg]:
                    event_starttime_dict[tg] = cand
                    if stream_of(tg) == watch:
                        bind[tg] = dict(occupant(pp_path), link=pp_path, length_count=length_count)
            elif cand:
                event_starttime_dict[tg] = cand
                if stream_of(tg) == watch:
                    bind[tg] = dict(occupant(pp_path), link=pp_path, length_count=length_count)

        if mode == "orig":
            if start_events:
                max_path_length = max(
                    max(len(lst) for lst in event.path_list) if event.event_type == "communication" and event.path_list else 0
                    for event in start_events)
            else:
                max_path_length = 0
            length_count = max_path_length
            while length_count > 0:
                no_available_pp_link = True
                random.shuffle(start_events)
                if sort_key is None:
                    start_events.sort(key=lambda pair: max((len(lst) for lst in getattr(pair, 'path_list', []) or []), default=0),
                                      reverse=True)
                else:
                    start_events.sort(key=sort_key, reverse=True)
                for event in start_events[:]:
                    if event.event_type != "communication":
                        continue
                    issued_path = None
                    if current_time == event.start_time:
                        random.shuffle(event.path_list)
                        event.path_list.sort(key=lambda pair: (max(len(lst) for lst in pair)), reverse=True)
                        for multicast_path in event.path_list[:]:
                            if len(multicast_path) != length_count:
                                continue
                            for path_link in multicast_path[:]:
                                pp_path = path_link
                                if hardware_platform.links_dict[pp_path].work_flag:
                                    defer(event, pp_path, length_count)
                                else:
                                    record_issue(event, pp_path)
                                    multicast_path.remove(path_link)
                                    issued_path = pp_path
                                    if multicast_path == []:
                                        event.path_list.remove(multicast_path)
                                    no_available_pp_link = False
                                break
                            if issued_path:
                                break
                        for multicast_path in event.path_list[:]:
                            if multicast_path == []:
                                event.path_list.remove(multicast_path)
                            elif multicast_path[0] == issued_path:
                                multicast_path.remove(path_link)
                                if multicast_path == []:
                                    event.path_list.remove(multicast_path)
                        if event.path_list == []:
                            finish_comm(event, issued_path)
                            if event.event_tag in event_starttime_dict:
                                del event_starttime_dict[event.event_tag]
                    if issued_path:
                        break
                if no_available_pp_link:
                    length_count -= 1
        else:  # DIAG: per-hop arbitration, no length layering
            last_issued = {}
            while True:
                ready = [e for e in start_events if e.event_type == "communication" and current_time == e.start_time
                         and e.path_list]
                random.shuffle(ready)
                cands = {}
                for e in ready:
                    rank = -prio[stream_of(e.event_tag)] if prio else 0
                    seen = set()
                    for br in e.path_list:
                        if not br:
                            continue
                        link = br[0]
                        if link in seen:
                            continue
                        seen.add(link)
                        ln = max(len(b) for b in e.path_list if b and b[0] == link)
                        cands.setdefault(link, []).append(((rank, ln), e))
                issued_any = False
                for link, cs in cands.items():
                    if hardware_platform.links_dict[link].work_flag:
                        for _, e in cs:
                            defer(e, link, None)
                        continue
                    cs.sort(key=lambda x: x[0], reverse=True)  # stable: shuffle order breaks ties
                    _, e = cs[0]
                    record_issue(e, link)
                    last_issued[e.event_tag] = link
                    issued_any = True
                    for br in e.path_list[:]:
                        if br and br[0] == link:
                            br.pop(0)
                    for br in e.path_list[:]:
                        if br == []:
                            e.path_list.remove(br)
                    for _, l2 in cs[1:]:
                        defer(l2, link, None)
                for e in ready:
                    if e.path_list == [] and e in start_events:
                        finish_comm(e, last_issued[e.event_tag])
                        event_starttime_dict.pop(e.event_tag, None)
                if not issued_any:
                    break
        for event in start_events:
            tg = event.event_tag
            if tg in event_starttime_dict:
                if stream_of(tg) == watch and tg in bind:  # DIAG
                    b = bind[tg]
                    log["deferrals"].append(dict(b, t=current_time, tag=tg, kind="link", new_start=event_starttime_dict[tg],
                                                 hops_done=hops_issued.get(tg, 0), ready=ready_at.get(tg),
                                                 prev=last_def.get(tg)))
                    last_def[tg] = (current_time, b["link"])
                event.start_time = event_starttime_dict[tg]

        if not start_events and not wait_events and not issued_events:
            break
        if not start_events and not issued_events:
            raise ValueError("Event-driven scheduler failed to converge")
        if not start_events and issued_events:
            next_time = min(issued_events.keys())
        elif not issued_events and start_events:
            start_events = sorted(start_events, key=lambda x: x.start_time)
            next_time = start_events[0].start_time
        else:
            start_events = sorted(start_events, key=lambda x: x.start_time)
            next_starttime = start_events[0].start_time
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
    if held:
        raise ValueError(f"gated streams never released: {sorted(held)}")
    return int(current_time), int(pure_comp_time), int(pure_comm_time)
