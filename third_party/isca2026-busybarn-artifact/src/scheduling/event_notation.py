
"""Event notation: defines event_notation (computation) and communication_notation
objects that carry timing, path, and dependency information for the scheduler."""

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from typing import List, Set, Dict, Tuple, Deque
import heapq


# Base class for all events in the DAG (computation and communication).
# dependency_set: events that must finish before this one starts.
# issue_set: events that this one enables upon completion.
class event_notation(object):

    def __init__(self,
                 event_type: str,
                 event_name: str, event_tag: int
                 ):

        self.event_type = event_type
        self.event_name = event_name
        self.event_tag = event_tag

        self.dependency_set = set()
        self.issue_set = set()

        self.start_time = 0
        self.end_time = 0


# A point-to-point or multicast communication event between nodes.
# paths: tree-structured routing dict {node -> [next_nodes]}.
# path_list: flattened list of (src, dst) link hops for the event driver.
class communication_notation(event_notation):

    def __init__(self,
                 comm_name: str, comm_tag: int,
                 source_location: Tuple[int], target_location: Tuple[int],
                 comm_bytes: int
                 ):
        super(communication_notation, self).__init__(
            event_type="communication",
            event_name=comm_name, event_tag=comm_tag
        )

        self.source_location = source_location
        self.target_location = target_location
        self.comm_bytes = comm_bytes

        self.path_list = []
        # function path format
        self.paths = {}
        self.hops = 0
        self.communication_distances = 0


    def get_paths(self, paths_dict: Dict[Tuple[int], List[Tuple[int]]]):
        # DFS traversal of the routing tree to produce a flat list of link hops.
        def explore(node, path):
            if len(node) > 2:
                node = (node[0], node[1])
            if node not in paths_dict or not paths_dict[node]:
                return [path]
            results = []
            for next_node in paths_dict[node]:
                if next_node.idx:
                    results.extend(explore(next_node, path + [(node, next_node.ori, next_node.idx)]))
                else:
                    results.extend(explore(next_node, path + [(node, next_node)]))
            return results
        return explore(self.source_location, [])


    def __str__(self):
        return f"comm_name: {self.event_name}, comm_tag: {self.event_tag}, source_location: {self.source_location}, target_location: {self.target_location}, comm_bytes: {self.comm_bytes} needs {self.dependency_set} issues {self.issue_set} through {self.path_list}"

    def __repr__(self):
        return self.__str__()


# Min-heap based ID allocator with recycling. Used by computation_notation
# to assign unique IDs to child communication events; recycled IDs are
# reused lowest-first to keep the ID space compact during SA mutations.
class IDAllocator(object):

    def __init__(self, start_id=0):
        self.used_ids = set()
        self.recycled = []
        self.next_id = start_id


    def allocate(self):
        if self.recycled:
            new_id = heapq.heappop(self.recycled)
        else:
            new_id = self.next_id
            self.next_id += 1
        self.used_ids.add(new_id)
        return new_id


    def recycle(self, id_):
        if id_ in self.used_ids:
            self.used_ids.remove(id_)
            heapq.heappush(self.recycled, id_)
        else:
            raise ValueError(f"ID {id_} is not currently allocated and cannot be recycled.")


# A computation event mapped to a specific device (tensorcore or vectorunit)
# at a specific (y, x, device_idx) location. Owns a commid_allocator for
# creating child communication event IDs.
class computation_notation(event_notation):

    def __init__(self,
                 comp_name: str, comp_tag: int,
                 comp_device: str, comp_location: Tuple[int],
                 comp_type: str, comp_datashape: List[List[int]], comp_datatype: List[str] = None
                 ):
        super(computation_notation, self).__init__(
            event_type="computation",
            event_name=comp_name, event_tag=comp_tag
        )

        self.comp_device = comp_device
        self.comp_location = comp_location
        self.comp_type = comp_type
        self.comp_datashape = comp_datashape
        self.comp_datatype = comp_datatype

        # communication event ID allocator if defined by the computation event
        self.commid_allocator = IDAllocator()


    def __str__(self):
        return f"{self.event_tag} {self.event_name} {self.comp_type} on {self.comp_location} {self.comp_device} needs {self.dependency_set} issues {self.issue_set}"

    def __repr__(self):
        return self.__str__()
