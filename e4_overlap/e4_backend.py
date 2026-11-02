"""E4 copy of BusyBarn's `event_driver` (work copy src/backend/analytical/event_driver.py, lines 282-527,
artifact commit 299089ef). Every change is marked `# E4:`; with release=None, gate=None, prio=None the code
path, the RNG calls and the result are the original's.

E4 changes (in memory only; the artifact file is untouched):
  - release (approach (b) of the handoff, 3.1): {stream: time}. An event without dependencies of stream s gets
    start_time = release[s] instead of 0, so it enters the ready list at that time; nothing else changes.
  - gate (serial reference): {stream: previous stream}. The dependency-free events of a gated stream are held
    back and enter the ready list, with start_time = now, at the instant the last event of the previous stream
    retires (Phase 2). The release times actually used are returned in `released`.
  - prio (stream-level priority, user-confirmed measurement, handoff 3.3): {stream: rank}, rank 0 first.
    Phase 4's sort becomes (stream rank, then the original longest-remaining-path-first). Phase 3
    (computation dispatch) is unchanged.
The stream of an event is tag[1] - 10 (E4's merge puts 10 + stream id in tag position 1).
"""
import random


def stream_of(tag):
    return tag[1] - 10


def event_driver_e4(events_dict, hardware_platform, release=None, gate=None, prio=None, released=None):
    """Returns (total_cycles, pure_comp_cycles, pure_comm_cycles), as the original."""
    start_events = []
    wait_events = {}

    event_number = len(events_dict)

    decode_flag = False  # detects transition from prefill to decode phase

    # E4: per-stream bookkeeping for the gate and for `released`
    release = release or {}
    gate = gate or {}
    if released is None:
        released = {}
    remaining = {}
    held = {}
    for event_tag in events_dict:
        s = stream_of(event_tag)
        remaining[s] = remaining.get(s, 0) + 1

    # Partition events into ready (no deps) and waiting (has deps)
    for event_tag in events_dict:
        if events_dict[event_tag].dependency_set == set():
            s = stream_of(event_tag)  # E4
            if s in gate:  # E4: serial reference, held until the previous stream completes
                held.setdefault(s, []).append(events_dict[event_tag])
                continue
            start_events.append(events_dict[event_tag])
            start_events[-1].start_time = release.get(s, 0)  # E4: was 0
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

    # E4: Phase 4 sort key; None -> the original lambda
    if prio:
        def sort_key(pair):
            return (-prio[stream_of(pair.event_tag)],
                    max((len(lst) for lst in getattr(pair, 'path_list', []) or []), default=0))
    else:
        sort_key = None

    # --- Main simulation loop ---
    while start_events or wait_events or issued_events:
        last_time = current_time

        # Phase 1: Release devices whose work finished by current_time
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

        # Phase 2: Retire finished events, wake dependents, detect prefill->decode
        if current_time in issued_events:
            finished_events = issued_events[current_time]
            for finished_event in finished_events:
                remaining[stream_of(finished_event.event_tag)] -= 1  # E4
                for dep_event_tag in finished_event.issue_set:
                    wait_events[dep_event_tag].dependency_set.remove(finished_event.event_tag)
                    if wait_events[dep_event_tag].dependency_set == set():
                        wait_events[dep_event_tag].start_time = current_time
                        start_events.append(wait_events[dep_event_tag])
                        if decode_flag == False and dep_event_tag[1] == 1 and dep_event_tag[0] == 0:
                            print(f"prefill time cost: {int(current_time), int(pure_comp_time), int(pure_comm_time)}")
                            decode_flag = True
                        del wait_events[dep_event_tag]
            del issued_events[current_time]
            # E4: a gated stream starts when the stream it waits for has no event left
            for s, prev in gate.items():
                if s in held and remaining.get(prev, 0) == 0:
                    for ev in held.pop(s):
                        ev.start_time = current_time
                        start_events.append(ev)
                    released[s] = current_time

        # Phase 3: Issue ready comp events; lazily resolve comm paths
        for event in start_events[:]:
            if current_time == event.start_time:
                if event.event_type == "computation":
                    # Defer if device busy; otherwise dispatch
                    if hardware_platform.modules_dict[event.comp_device][event.comp_location].work_flag:
                        event.start_time = hardware_platform.modules_dict[event.comp_device][event.comp_location].work_endtime
                    else:
                        computation_time = hardware_platform.modules_dict[event.comp_device][event.comp_location].working_time(
                            source_datashape=event.comp_datashape,
                            beha_type=event.comp_type,
                            frequency=hardware_platform.frequency
                        )
                        hardware_platform.modules_dict[event.comp_device][event.comp_location].work_flag = True
                        hardware_platform.modules_dict[event.comp_device][event.comp_location].work_endtime = current_time + computation_time
                        hardware_platform.modules_dict[event.comp_device][event.comp_location].work_record.append(
                            [current_time, current_time + computation_time, event.event_tag]
                        )
                        working_computation_devices.add((event.comp_device, event.comp_location))
                        event.end_time = hardware_platform.modules_dict[event.comp_device][event.comp_location].work_endtime
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
                                target_nodes_set=event.target_location if type(event.target_location) is set else {event.target_location}
                            )
                        )

        # Phase 4: Link-level scheduling for comm events (longest paths first)
        no_available_pp_link = False
        if start_events:
            max_path_length = max(
                max(len(lst) for lst in event.path_list) if event.event_type == "communication" and event.path_list else 0
                for event in start_events
            )
        else:
            max_path_length = 0
        length_count = max_path_length
        event_starttime_dict = {}  # tracks deferred start times for blocked events
        while length_count > 0:
            no_available_pp_link = True
            random.shuffle(start_events)
            if sort_key is None:
                start_events.sort(
                    key=lambda pair: max((len(lst) for lst in getattr(pair, 'path_list', []) or []), default=0),
                    reverse=True
                )
            else:  # E4: stream priority first, then the original key
                start_events.sort(key=sort_key, reverse=True)
            for event in start_events[:]:
                if event.event_type != "communication":
                    continue
                issued_path = None
                if event.event_type == "communication" and current_time == event.start_time:
                    random.shuffle(event.path_list)
                    event.path_list.sort(
                        key=lambda pair: (max(len(lst) for lst in pair)),
                        reverse=True
                    )
                    for multicast_path in event.path_list[:]:
                        if len(multicast_path) == length_count:
                            pass
                        else:
                            continue
                        for path_link in multicast_path[:]:
                            pp_path = path_link
                            # Link busy: record earliest availability as deferred start
                            if hardware_platform.links_dict[pp_path].work_flag:
                                if event.start_time is not None:
                                    if hardware_platform.links_dict[pp_path].work_endtime > event.start_time:
                                        start_time_candidate = hardware_platform.links_dict[pp_path].work_endtime
                                    else:
                                        start_time_candidate = None
                                else:
                                    start_time_candidate = hardware_platform.links_dict[pp_path].work_endtime
                                if event.event_tag in event_starttime_dict and start_time_candidate:
                                    if start_time_candidate < event_starttime_dict[event.event_tag]:
                                        event_starttime_dict[event.event_tag] = start_time_candidate
                                elif start_time_candidate:
                                    event_starttime_dict[event.event_tag] = start_time_candidate
                                else:
                                    pass
                            else:
                                # Link free: issue one hop of the multicast path
                                communication_time = hardware_platform.links_dict[pp_path].working_time(
                                    data_bytes=event.comm_bytes
                                )
                                hardware_platform.links_dict[pp_path].work_flag = True
                                hardware_platform.links_dict[pp_path].work_endtime = current_time + communication_time
                                hardware_platform.links_dict[pp_path].work_record.append(
                                    [current_time, current_time + communication_time, event.event_tag]
                                )
                                working_communication_devices.add(pp_path)
                                multicast_path.remove(path_link)
                                issued_path = pp_path
                                if multicast_path == []:
                                    event.path_list.remove(multicast_path)
                                no_available_pp_link = False
                            break
                        if issued_path:
                            break
                    # Remove issued link from other multicast branches sharing it
                    for multicast_path in event.path_list[:]:
                        if multicast_path == []:
                            event.path_list.remove(multicast_path)
                        elif multicast_path[0] == issued_path:
                            multicast_path.remove(path_link)
                            if multicast_path == []:
                                event.path_list.remove(multicast_path)
                    # All paths consumed -> event fully issued
                    if event.path_list == []:
                        event.end_time = hardware_platform.links_dict[issued_path].work_endtime
                        if event.end_time in issued_events:
                            issued_events[event.end_time].add(event)
                            events_record[event.end_time].add(event)
                        else:
                            issued_events[event.end_time] = {event}
                            events_record[event.end_time] = {event}
                        start_events.remove(event)
                        if event.event_tag in event_starttime_dict:
                            del event_starttime_dict[event.event_tag]
                if issued_path:
                    break
                else:
                    continue
            if no_available_pp_link:
                length_count -= 1
        # Apply deferred start times for events blocked by busy links
        for event in start_events:
            if event.event_tag in event_starttime_dict:
                event.start_time = event_starttime_dict[event.event_tag]

        # Phase 5: Termination and time advancement
        if not start_events and not wait_events and not issued_events:
            break

        if not start_events and not issued_events:
            print(wait_events)
            raise ValueError("Event-driven scheduler failed to converge")

        # Advance to next event boundary
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

        # Track non-overlapping compute/comm time for breakdown analysis
        delta = next_time - current_time
        if delta > 0:
            if working_computation_devices and not working_communication_devices:
                pure_comp_time += delta
            elif working_communication_devices and not working_computation_devices:
                pure_comm_time += delta

        current_time = next_time

        iter_cnt += 1
        if iter_cnt > ((event_number) * 100):
            print("!!!!!!!!!!!!!!!!!!!!!!!!")
            for wait_tag in wait_events:
                print(wait_events[wait_tag])
            raise ValueError("Event-driven scheduler failed to converge")

    if held:  # E4: a gated stream never started
        raise ValueError(f"gated streams never released: {sorted(held)}")
    return int(current_time), int(pure_comp_time), int(pure_comm_time)
