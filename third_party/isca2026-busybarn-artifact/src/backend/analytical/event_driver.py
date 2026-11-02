
"""Event-driven analytical backend: processes computation and communication events
with cycle-accurate timing, tracking device utilization and communication contention."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../../scheduling/'))
sys.path.append(os.path.join(file_path, '../../scheduling/communication/topology/'))


from event_notation import event_notation
from tlm import tlm2d
from net import net


from typing import List, Set, Dict, Tuple, Deque
import random
import numpy as np
import heapq


def _heap_min(heap, valid_dict):
    """Return min key from heap that is still in valid_dict, or None."""
    while heap and heap[0] not in valid_dict:
        heapq.heappop(heap)
    return heap[0] if heap else None


def _issue_event(event, issued_events, _end_heap, events_record):
    """Add event to issued_events and push end_time onto heap."""
    et = event.end_time
    if et in issued_events:
        issued_events[et].add(event)
        events_record[et].add(event)
    else:
        issued_events[et] = {event}
        events_record[et] = {event}
        heapq.heappush(_end_heap, et)


def collective_event_driver(
    events_dict: Dict[Tuple[int], event_notation],
    hardware_platform: net
):
    """Simulate collective communication events using list-based scheduling.
    Returns (total_cycles, pure_comp_cycles, pure_comm_cycles)."""
    start_events = []
    wait_events = {}

    event_number = len(events_dict)

    # Partition events into ready (no deps) and waiting (has deps)
    for event_tag in events_dict:
        if events_dict[event_tag].dependency_set == set():
            start_events.append(events_dict[event_tag])
            start_events[-1].start_time = 0
        else:
            wait_events[event_tag] = events_dict[event_tag]

    current_time = 0
    pure_comp_time = 0.0
    pure_comm_time = 0.0

    working_computation_devices = set()   # currently busy compute modules
    working_communication_devices = set()  # currently busy network links
    issued_events = {}   # end_time -> set of in-flight events
    events_record = {}

    iter_cnt = 0

    # --- Main simulation loop: advance time until all events complete ---
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

        # Phase 2: Retire finished events and wake up dependent events
        if current_time in issued_events:
            finished_events = issued_events[current_time]
            for finished_event in finished_events:
                for dep_event_tag in finished_event.issue_set:
                    wait_events[dep_event_tag].dependency_set.remove(finished_event.event_tag)
                    if wait_events[dep_event_tag].dependency_set == set():
                        wait_events[dep_event_tag].start_time = current_time
                        start_events.append(wait_events[dep_event_tag])
                        del wait_events[dep_event_tag]
            del issued_events[current_time]

        # Phase 3: Issue ready computation events; resolve paths for comm events
        for event in start_events[:]:
            if current_time == event.start_time:
                if event.event_type == "computation":
                    # Defer if device is busy; otherwise dispatch
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
                    # Lazily resolve multicast routing paths on first encounter
                    if event.path_list == []:
                        event.path_list = event.get_paths(
                            hardware_platform.xy_multicast_path(
                                source_node_idx=event.source_location,
                                target_nodes_set=event.target_location if type(event.target_location) is set else {event.target_location}
                            )
                        )

        # Phase 4: Link-level scheduling for communication events.
        # Process longest paths first to reduce link contention.
        no_available_pp_link = False
        if start_events:
            max_path_length = max(
                max(len(lst) for lst in event.path_list) if event.path_list else 0
                for event in start_events
            )
        else:
            max_path_length = 0
        length_count = max_path_length
        event_starttime_dict = {}  # tracks deferred start times for blocked events
        while length_count > 0:
            no_available_pp_link = True
            random.shuffle(start_events)
            start_events.sort(
                key=lambda pair: max(len(lst) for lst in pair.path_list),
                reverse=True
            )
            for event in start_events[:]:
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

        # Advance to next event boundary (earliest start or end time)
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

        # Track non-overlapping compute and comm time for breakdown analysis
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

    return current_time, pure_comp_time, pure_comm_time


def event_driver(
    events_dict: Dict[Tuple[int], event_notation],
    hardware_platform: net
):
    """Simulate unicast computation/communication events with prefill/decode detection.
    Returns (total_cycles, pure_comp_cycles, pure_comm_cycles)."""
    start_events = []
    wait_events = {}

    event_number = len(events_dict)

    decode_flag = False  # detects transition from prefill to decode phase

    # Partition events into ready (no deps) and waiting (has deps)
    for event_tag in events_dict:
        if events_dict[event_tag].dependency_set == set():
            start_events.append(events_dict[event_tag])
            start_events[-1].start_time = 0
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
            start_events.sort(
                key=lambda pair: max((len(lst) for lst in getattr(pair, 'path_list', []) or []), default=0),
                reverse=True
            )
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


    return int(current_time), int(pure_comp_time), int(pure_comm_time)


def mem_event_driver(
    events_dict: Dict[Tuple[int], event_notation],
    hardware_platform: net
):
    """Memory-aware event driver with bandwidth pipelining across heterogeneous links.
    Adjusts start times when data crosses links with different bandwidths.
    Returns (total_cycles, pure_comp_cycles, pure_comm_cycles)."""
    start_events = []
    wait_events = {}

    event_number = len(events_dict)

    decode_flag = False

    # Partition events into ready (no deps) and waiting (has deps)
    for event_tag in events_dict:
        if events_dict[event_tag].dependency_set == set():
            start_events.append(events_dict[event_tag])
            start_events[-1].start_time = 0
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

        # Phase 3: Issue ready comp events; lazily resolve comm paths
        for event in start_events[:]:
            if current_time == event.start_time:
                if event.event_type == "computation":
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

        no_available_pp_link = False
        if start_events:
            max_path_length = max(
                max(len(lst) for lst in event.path_list) if event.event_type == "communication" and event.path_list else 0
                for event in start_events
            ) 
        else:
            max_path_length = 0
        length_count = max_path_length
        event_starttime_dict = {}
        while length_count > 0:
            no_available_pp_link = True
            random.shuffle(start_events)
            start_events.sort(
                key=lambda pair: max((len(lst) for lst in getattr(pair, 'path_list', []) or []), default=0),
                reverse=True
            )
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
                        current_path_length = len(multicast_path)
                        for link_idx, path_link in enumerate(multicast_path[:]):
                            pp_path = path_link
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
                                communication_time = hardware_platform.links_dict[pp_path].working_time(
                                    data_bytes=event.comm_bytes
                                )
                                hardware_platform.links_dict[pp_path].work_flag = True
                                hardware_platform.links_dict[pp_path].work_endtime = current_time + communication_time
                                hardware_platform.links_dict[pp_path].work_record.append(
                                    [current_time, current_time + communication_time, event.event_tag]
                                )
                                if link_idx < (current_path_length - 1):
                                    next_path = multicast_path[link_idx + 1]
                                    if hardware_platform.links_dict[pp_path].bandwidth < hardware_platform.links_dict[next_path].bandwidth:
                                        event.start_time = current_time + hardware_platform.links_dict[pp_path].latency + int(event.comm_bytes / hardware_platform.links_dict[pp_path].bandwidth - event.comm_bytes / hardware_platform.links_dict[next_path].bandwidth)
                                    else:
                                        event.start_time = current_time + hardware_platform.links_dict[pp_path].latency
                                else:
                                    pass
                                working_communication_devices.add(pp_path)
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
        for event in start_events:
            if event.event_tag in event_starttime_dict:
                event.start_time = event_starttime_dict[event.event_tag]

        if not start_events and not wait_events and not issued_events:
            break

        if not start_events and not issued_events:
            print(wait_events)
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
            print("!!!!!!!!!!!!!!!!!!!!!!!!")
            for wait_tag in wait_events:
                print(wait_events[wait_tag])
            raise ValueError("Event-driven scheduler failed to converge")


    return int(current_time), int(pure_comp_time), int(pure_comm_time)


def collective_event_driver_v2(
    events_dict: Dict[Tuple[int], event_notation],
    hardware_platform: net
):
    _ready = {}
    wait_events = {}

    event_number = len(events_dict)

    for event_tag in events_dict:
        if events_dict[event_tag].dependency_set == set():
            events_dict[event_tag].start_time = 0
            _ready[event_tag] = events_dict[event_tag]
        else:
            wait_events[event_tag] = events_dict[event_tag]

    current_time = 0
    pure_comp_time = 0.0
    pure_comm_time = 0.0

    working_computation_devices = set()
    working_communication_devices = set()
    issued_events = {}
    _end_heap = []
    events_record = {}

    iter_cnt = 0

    while _ready or wait_events or issued_events:
        last_time = current_time

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
                for dep_event_tag in finished_event.issue_set:
                    wait_events[dep_event_tag].dependency_set.remove(finished_event.event_tag)
                    if wait_events[dep_event_tag].dependency_set == set():
                        wait_events[dep_event_tag].start_time = current_time
                        _ready[dep_event_tag] = wait_events[dep_event_tag]
                        del wait_events[dep_event_tag]
            del issued_events[current_time]

        for event in list(_ready.values()):
            if current_time == event.start_time:
                if event.event_type == "computation":
                    if hardware_platform.modules_dict[event.comp_device][event.comp_location].work_flag:
                        event.start_time = hardware_platform.modules_dict[event.comp_device][event.comp_location].work_endtime
                    else:
                        computation_time = int(hardware_platform.modules_dict[event.comp_device][event.comp_location].working_time(
                            source_datashape=event.comp_datashape,
                            beha_type=event.comp_type,
                            frequency=hardware_platform.frequency
                        ))
                        hardware_platform.modules_dict[event.comp_device][event.comp_location].work_flag = True
                        hardware_platform.modules_dict[event.comp_device][event.comp_location].work_endtime = current_time + computation_time
                        hardware_platform.modules_dict[event.comp_device][event.comp_location].work_record.append(
                            [current_time, current_time + computation_time, event.event_tag]
                        )
                        working_computation_devices.add((event.comp_device, event.comp_location))
                        event.end_time = hardware_platform.modules_dict[event.comp_device][event.comp_location].work_endtime
                        _issue_event(event, issued_events, _end_heap, events_record)
                        del _ready[event.event_tag]

                elif event.event_type == "communication":
                    if event.path_list == []:
                        event.path_list = event.get_paths(
                            hardware_platform.xy_multicast_path(
                                source_node_idx=event.source_location,
                                target_nodes_set=event.target_location if type(event.target_location) is set else {event.target_location}
                            )
                        )

        no_available_pp_link = False
        if _ready:
            max_path_length = max(
                (max(len(lst) for lst in e.path_list) if e.path_list else 0)
                for e in _ready.values()
            )
        else:
            max_path_length = 0
        length_count = max_path_length
        event_starttime_dict = {}
        while length_count > 0:
            no_available_pp_link = True
            comm_events = [e for e in _ready.values() if e.event_type == "communication"]
            random.shuffle(comm_events)
            comm_events.sort(
                key=lambda pair: max(len(lst) for lst in pair.path_list) if pair.path_list else 0,
                reverse=True
            )
            for event in comm_events:
                issued_path = None
                if current_time == event.start_time:
                    random.shuffle(event.path_list)
                    event.path_list.sort(
                        key=lambda pair: (max(len(lst) for lst in pair)),
                        reverse=True
                    )
                    for multicast_path in event.path_list[:]:
                        if len(multicast_path) != length_count:
                            continue
                        for path_link in multicast_path[:]:
                            pp_path = path_link
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
                    for multicast_path in event.path_list[:]:
                        if multicast_path == []:
                            event.path_list.remove(multicast_path)
                        elif multicast_path[0] == issued_path:
                            multicast_path.remove(path_link)
                            if multicast_path == []:
                                event.path_list.remove(multicast_path)
                    if event.path_list == []:
                        event.end_time = hardware_platform.links_dict[issued_path].work_endtime
                        _issue_event(event, issued_events, _end_heap, events_record)
                        del _ready[event.event_tag]
                        if event.event_tag in event_starttime_dict:
                            del event_starttime_dict[event.event_tag]
                if issued_path:
                    break
                else:
                    continue
            if no_available_pp_link:
                length_count -= 1
        for event in _ready.values():
            if event.event_tag in event_starttime_dict:
                event.start_time = event_starttime_dict[event.event_tag]

        if not _ready and not wait_events and not issued_events:
            break
        if not _ready and not issued_events:
            print(wait_events)
            raise ValueError("Event-driven scheduler failed to converge")

        if not _ready and issued_events:
            next_time = _heap_min(_end_heap, issued_events)
        elif not issued_events and _ready:
            next_time = min(e.start_time for e in _ready.values())
        else:
            next_starttime = min(e.start_time for e in _ready.values())
            next_endtime = _heap_min(_end_heap, issued_events)
            next_time = next_starttime if next_starttime < next_endtime else next_endtime

        # Re-sort _ready by start_time (stable) to match original list-sort order
        _ready = dict(sorted(_ready.items(), key=lambda kv: kv[1].start_time))

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

    return current_time, pure_comp_time, pure_comm_time


def event_driver_v2(
    events_dict: Dict[Tuple[int], event_notation],
    hardware_platform: net
):
    _ready = {}
    wait_events = {}
    event_number = len(events_dict)
    decode_flag = False

    for event_tag in events_dict:
        if events_dict[event_tag].dependency_set == set():
            events_dict[event_tag].start_time = 0
            _ready[event_tag] = events_dict[event_tag]
        else:
            wait_events[event_tag] = events_dict[event_tag]

    current_time = 0
    pure_comp_time = 0.0
    pure_comm_time = 0.0
    working_computation_devices = set()
    working_communication_devices = set()
    issued_events = {}
    _end_heap = []
    events_record = {}
    iter_cnt = 0

    while _ready or wait_events or issued_events:
        last_time = current_time
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
                for dep_event_tag in finished_event.issue_set:
                    wait_events[dep_event_tag].dependency_set.remove(finished_event.event_tag)
                    if wait_events[dep_event_tag].dependency_set == set():
                        wait_events[dep_event_tag].start_time = current_time
                        _ready[dep_event_tag] = wait_events[dep_event_tag]
                        if decode_flag == False and dep_event_tag[1] == 1 and dep_event_tag[0] == 0:
                            print(f"prefill time cost: {int(current_time), int(pure_comp_time), int(pure_comm_time)}")
                            decode_flag = True
                        del wait_events[dep_event_tag]
            del issued_events[current_time]

        for event in list(_ready.values()):
            if current_time == event.start_time:
                if event.event_type == "computation":
                    if hardware_platform.modules_dict[event.comp_device][event.comp_location].work_flag:
                        event.start_time = hardware_platform.modules_dict[event.comp_device][event.comp_location].work_endtime
                    else:
                        computation_time = int(hardware_platform.modules_dict[event.comp_device][event.comp_location].working_time(
                            source_datashape=event.comp_datashape,
                            beha_type=event.comp_type,
                            frequency=hardware_platform.frequency
                        ))
                        hardware_platform.modules_dict[event.comp_device][event.comp_location].work_flag = True
                        hardware_platform.modules_dict[event.comp_device][event.comp_location].work_endtime = current_time + computation_time
                        hardware_platform.modules_dict[event.comp_device][event.comp_location].work_record.append(
                            [current_time, current_time + computation_time, event.event_tag]
                        )
                        working_computation_devices.add((event.comp_device, event.comp_location))
                        event.end_time = hardware_platform.modules_dict[event.comp_device][event.comp_location].work_endtime
                        _issue_event(event, issued_events, _end_heap, events_record)
                        del _ready[event.event_tag]

                elif event.event_type == "communication":
                    if event.path_list == []:
                        event.path_list = event.get_paths(
                            hardware_platform.xy_multicast_path(
                                source_node_idx=event.source_location,
                                target_nodes_set=event.target_location if type(event.target_location) is set else {event.target_location}
                            )
                        )

        no_available_pp_link = False
        if _ready:
            max_path_length = max(
                (max(len(lst) for lst in e.path_list) if e.event_type == "communication" and e.path_list else 0)
                for e in _ready.values()
            )
        else:
            max_path_length = 0
        length_count = max_path_length
        event_starttime_dict = {}
        while length_count > 0:
            no_available_pp_link = True
            comm_events = sorted(
                [e for e in _ready.values() if e.event_type == "communication"],
                key=lambda pair: max((len(lst) for lst in getattr(pair, 'path_list', []) or []), default=0),
                reverse=True
            )
            for event in comm_events:
                issued_path = None
                if current_time == event.start_time:
                    random.shuffle(event.path_list)
                    event.path_list.sort(
                        key=lambda pair: (max(len(lst) for lst in pair)),
                        reverse=True
                    )
                    for multicast_path in event.path_list[:]:
                        if len(multicast_path) != length_count:
                            continue
                        for path_link in multicast_path[:]:
                            pp_path = path_link
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
                    for multicast_path in event.path_list[:]:
                        if multicast_path == []:
                            event.path_list.remove(multicast_path)
                        elif multicast_path[0] == issued_path:
                            multicast_path.remove(path_link)
                            if multicast_path == []:
                                event.path_list.remove(multicast_path)
                    if event.path_list == []:
                        event.end_time = hardware_platform.links_dict[issued_path].work_endtime
                        _issue_event(event, issued_events, _end_heap, events_record)
                        del _ready[event.event_tag]
                        if event.event_tag in event_starttime_dict:
                            del event_starttime_dict[event.event_tag]
                if issued_path:
                    break
                else:
                    continue
            if no_available_pp_link:
                length_count -= 1
        for event in _ready.values():
            if event.event_tag in event_starttime_dict:
                event.start_time = event_starttime_dict[event.event_tag]

        if not _ready and not wait_events and not issued_events:
            break
        if not _ready and not issued_events:
            print(wait_events)
            raise ValueError("Event-driven scheduler failed to converge")

        if not _ready and issued_events:
            next_time = _heap_min(_end_heap, issued_events)
        elif not issued_events and _ready:
            next_time = min(e.start_time for e in _ready.values())
        else:
            next_starttime = min(e.start_time for e in _ready.values())
            next_endtime = _heap_min(_end_heap, issued_events)
            next_time = next_starttime if next_starttime < next_endtime else next_endtime

        # Re-sort _ready by start_time (stable) to match original list-sort order
        _ready = dict(sorted(_ready.items(), key=lambda kv: kv[1].start_time))

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

    return int(current_time), int(pure_comp_time), int(pure_comm_time)


def mem_event_driver_v2(
    events_dict: Dict[Tuple[int], event_notation],
    hardware_platform: net
):
    _ready = {}
    wait_events = {}
    event_number = len(events_dict)
    decode_flag = False

    for event_tag in events_dict:
        if events_dict[event_tag].dependency_set == set():
            events_dict[event_tag].start_time = 0
            _ready[event_tag] = events_dict[event_tag]
        else:
            wait_events[event_tag] = events_dict[event_tag]

    current_time = 0
    pure_comp_time = 0.0
    pure_comm_time = 0.0
    working_computation_devices = set()
    working_communication_devices = set()
    issued_events = {}
    _end_heap = []
    events_record = {}
    iter_cnt = 0

    while _ready or wait_events or issued_events:
        last_time = current_time
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
                for dep_event_tag in finished_event.issue_set:
                    wait_events[dep_event_tag].dependency_set.remove(finished_event.event_tag)
                    if wait_events[dep_event_tag].dependency_set == set():
                        wait_events[dep_event_tag].start_time = current_time
                        _ready[dep_event_tag] = wait_events[dep_event_tag]
                        if decode_flag == False and dep_event_tag[1] == 1 and dep_event_tag[0] == 0:
                            print(f"prefill time cost: {int(current_time), int(pure_comp_time), int(pure_comm_time)}")
                            decode_flag = True
                        del wait_events[dep_event_tag]
            del issued_events[current_time]

        for event in list(_ready.values()):
            if current_time == event.start_time:
                if event.event_type == "computation":
                    if hardware_platform.modules_dict[event.comp_device][event.comp_location].work_flag:
                        event.start_time = hardware_platform.modules_dict[event.comp_device][event.comp_location].work_endtime
                    else:
                        computation_time = int(hardware_platform.modules_dict[event.comp_device][event.comp_location].working_time(
                            source_datashape=event.comp_datashape,
                            beha_type=event.comp_type,
                            frequency=hardware_platform.frequency
                        ))
                        hardware_platform.modules_dict[event.comp_device][event.comp_location].work_flag = True
                        hardware_platform.modules_dict[event.comp_device][event.comp_location].work_endtime = current_time + computation_time
                        hardware_platform.modules_dict[event.comp_device][event.comp_location].work_record.append(
                            [current_time, current_time + computation_time, event.event_tag]
                        )
                        working_computation_devices.add((event.comp_device, event.comp_location))
                        event.end_time = hardware_platform.modules_dict[event.comp_device][event.comp_location].work_endtime
                        _issue_event(event, issued_events, _end_heap, events_record)
                        del _ready[event.event_tag]

                elif event.event_type == "communication":
                    if event.path_list == []:
                        event.path_list = event.get_paths(
                            hardware_platform.xy_multicast_path(
                                source_node_idx=event.source_location,
                                target_nodes_set=event.target_location if type(event.target_location) is set else {event.target_location}
                            )
                        )

        no_available_pp_link = False
        if _ready:
            max_path_length = max(
                (max(len(lst) for lst in e.path_list) if e.event_type == "communication" and e.path_list else 0)
                for e in _ready.values()
            )
        else:
            max_path_length = 0
        length_count = max_path_length
        event_starttime_dict = {}
        while length_count > 0:
            no_available_pp_link = True
            comm_events = [e for e in _ready.values() if e.event_type == "communication"]
            random.shuffle(comm_events)
            comm_events.sort(
                key=lambda pair: max((len(lst) for lst in getattr(pair, 'path_list', []) or []), default=0),
                reverse=True
            )
            for event in comm_events:
                issued_path = None
                if current_time == event.start_time:
                    random.shuffle(event.path_list)
                    event.path_list.sort(
                        key=lambda pair: (max(len(lst) for lst in pair)),
                        reverse=True
                    )
                    for multicast_path in event.path_list[:]:
                        if len(multicast_path) != length_count:
                            continue
                        current_path_length = len(multicast_path)
                        for link_idx, path_link in enumerate(multicast_path[:]):
                            pp_path = path_link
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
                                communication_time = hardware_platform.links_dict[pp_path].working_time(
                                    data_bytes=event.comm_bytes
                                )
                                hardware_platform.links_dict[pp_path].work_flag = True
                                hardware_platform.links_dict[pp_path].work_endtime = current_time + communication_time
                                hardware_platform.links_dict[pp_path].work_record.append(
                                    [current_time, current_time + communication_time, event.event_tag]
                                )
                                if link_idx < (current_path_length - 1):
                                    next_path = multicast_path[link_idx + 1]
                                    if hardware_platform.links_dict[pp_path].bandwidth < hardware_platform.links_dict[next_path].bandwidth:
                                        event.start_time = current_time + hardware_platform.links_dict[pp_path].latency + int(event.comm_bytes / hardware_platform.links_dict[pp_path].bandwidth - event.comm_bytes / hardware_platform.links_dict[next_path].bandwidth)
                                    else:
                                        event.start_time = current_time + hardware_platform.links_dict[pp_path].latency
                                working_communication_devices.add(pp_path)
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
                        event.end_time = hardware_platform.links_dict[issued_path].work_endtime
                        _issue_event(event, issued_events, _end_heap, events_record)
                        del _ready[event.event_tag]
                        if event.event_tag in event_starttime_dict:
                            del event_starttime_dict[event.event_tag]
                if issued_path:
                    break
                else:
                    continue
            if no_available_pp_link:
                length_count -= 1
        for event in _ready.values():
            if event.event_tag in event_starttime_dict:
                event.start_time = event_starttime_dict[event.event_tag]

        if not _ready and not wait_events and not issued_events:
            break
        if not _ready and not issued_events:
            print(wait_events)
            raise ValueError("Event-driven scheduler failed to converge")

        if not _ready and issued_events:
            next_time = _heap_min(_end_heap, issued_events)
        elif not issued_events and _ready:
            next_time = min(e.start_time for e in _ready.values())
        else:
            next_starttime = min(e.start_time for e in _ready.values())
            next_endtime = _heap_min(_end_heap, issued_events)
            next_time = next_starttime if next_starttime < next_endtime else next_endtime

        # Re-sort _ready by start_time (stable) to match original list-sort order
        _ready = dict(sorted(_ready.items(), key=lambda kv: kv[1].start_time))

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

    return int(current_time), int(pure_comp_time), int(pure_comm_time)
