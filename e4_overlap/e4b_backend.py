"""E4b backend: the per-hop arbitration of e4_backend_diag.event_driver_diag (mode "perhop") with the priority
key supplied by a policy object; everything else (Phase 1-3, same-tick issue of every free hop, occupancy formula,
deferral rule, time advance) is the per-hop version's. In memory only.

Per tick, repeated rounds: every ready communication event offers the first link of each remaining branch (one
candidate per distinct first link, with ln = its longest remaining branch through that link). For every free link
the candidate with the smallest policy key wins (ties: the order of the round's shuffle, as in the per-hop
version); losers and events on busy links are deferred to the earliest work_endtime of their busy first links.

policy interface:
  key(event, link, ln, t) -> tuple (smaller first)
  on_issue(event, link, t)        after a hop is issued
  on_finish(event, t)             when an event retires (Phase 2)
  on_tick(t, ready_events)        once per arbitration round, before keys are taken (optional bookkeeping)
With policy = None the key is (-ln,), i.e. P1, the per-hop default.
audit: list to which (t, link, chosen key, min key over candidates recomputed after the choice, n candidates) is
appended for every issue (self-check 4).
"""
import random

from e4_backend import stream_of


class P1:
    def key(self, e, link, ln, t):
        return (-ln,)

    def on_issue(self, e, link, t):
        pass

    def on_finish(self, e, t):
        pass

    def on_tick(self, t, ready):
        pass


def event_driver_e4b(events_dict, hardware_platform, release=None, policy=None, released=None, audit=None,
                     shadow=None, shadow_log=None):
    """shadow: a second policy whose winner is computed on the same candidates (never used to decide); the count of
    issues where its winner differs from the actual one goes to shadow_log["differ"] (of shadow_log["n"])."""
    policy = policy or P1()
    start_events = []
    wait_events = {}
    event_number = len(events_dict)
    release = release or {}
    if released is None:
        released = {}
    for event_tag in events_dict:
        if events_dict[event_tag].dependency_set == set():
            s = stream_of(event_tag)
            start_events.append(events_dict[event_tag])
            start_events[-1].start_time = release.get(s, 0)
            released.setdefault(s, release.get(s, 0))
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

    def record_issue(event, pp_path):
        L = hardware_platform.links_dict[pp_path]
        communication_time = L.working_time(data_bytes=event.comm_bytes)
        L.work_flag = True
        L.work_endtime = current_time + communication_time
        L.work_record.append([current_time, current_time + communication_time, event.event_tag])
        working_communication_devices.add(pp_path)
        policy.on_issue(event, pp_path, current_time)

    def finish_comm(event, issued_path):
        event.end_time = hardware_platform.links_dict[issued_path].work_endtime
        if event.end_time in issued_events:
            issued_events[event.end_time].add(event)
            events_record[event.end_time].add(event)
        else:
            issued_events[event.end_time] = {event}
            events_record[event.end_time] = {event}
        start_events.remove(event)

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
            for finished_event in issued_events[current_time]:
                policy.on_finish(finished_event, current_time)
                for dep_event_tag in finished_event.issue_set:
                    wait_events[dep_event_tag].dependency_set.remove(finished_event.event_tag)
                    if wait_events[dep_event_tag].dependency_set == set():
                        wait_events[dep_event_tag].start_time = current_time
                        start_events.append(wait_events[dep_event_tag])
                        del wait_events[dep_event_tag]
            del issued_events[current_time]

        for event in start_events[:]:
            if current_time == event.start_time:
                if event.event_type == "computation":
                    dev = hardware_platform.modules_dict[event.comp_device][event.comp_location]
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
                        start_events.remove(event)
                elif event.event_type == "communication":
                    if event.path_list == []:
                        event.path_list = event.get_paths(
                            hardware_platform.xy_multicast_path(
                                source_node_idx=event.source_location,
                                target_nodes_set=event.target_location if type(event.target_location) is set else {event.target_location}))

        event_starttime_dict = {}

        def defer(event, pp_path):
            L = hardware_platform.links_dict[pp_path]
            if event.start_time is not None:
                cand = L.work_endtime if L.work_endtime > event.start_time else None
            else:
                cand = L.work_endtime
            tg = event.event_tag
            if tg in event_starttime_dict and cand:
                if cand < event_starttime_dict[tg]:
                    event_starttime_dict[tg] = cand
            elif cand:
                event_starttime_dict[tg] = cand

        last_issued = {}
        while True:
            ready = [e for e in start_events if e.event_type == "communication" and current_time == e.start_time
                     and e.path_list]
            random.shuffle(ready)
            policy.on_tick(current_time, ready)
            cands = {}
            for e in ready:
                seen = set()
                for br in e.path_list:
                    if not br:
                        continue
                    link = br[0]
                    if link in seen:
                        continue
                    seen.add(link)
                    ln = max(len(b) for b in e.path_list if b and b[0] == link)
                    cands.setdefault(link, []).append((e, ln))
            issued_any = False
            for link, cs in cands.items():
                if hardware_platform.links_dict[link].work_flag:
                    for e, _ in cs:
                        defer(e, link)
                    continue
                keyed = [(policy.key(e, link, ln, current_time), i, e, ln) for i, (e, ln) in enumerate(cs)]
                k0, _, e, ln = min(keyed, key=lambda x: (x[0], x[1]))  # ties: shuffle order
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
                record_issue(e, link)
                last_issued[e.event_tag] = link
                issued_any = True
                for br in e.path_list[:]:
                    if br and br[0] == link:
                        br.pop(0)
                for br in e.path_list[:]:
                    if br == []:
                        e.path_list.remove(br)
                for x, _ in cs:
                    if x is not e:
                        defer(x, link)
            for e in ready:
                if e.path_list == [] and e in start_events:
                    finish_comm(e, last_issued[e.event_tag])
                    event_starttime_dict.pop(e.event_tag, None)
            if not issued_any:
                break
        for event in start_events:
            if event.event_tag in event_starttime_dict:
                event.start_time = event_starttime_dict[event.event_tag]

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
    return int(current_time), int(pure_comp_time), int(pure_comm_time)
