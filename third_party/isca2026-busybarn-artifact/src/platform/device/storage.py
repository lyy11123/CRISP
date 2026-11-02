
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))


from typing import List, Dict, Tuple, Set


class storage(object):

    def __init__(self, device_type: str, device_id: Tuple[int]):

        self.device_type = device_type
        self.device_id = device_id

        self.capacity = None
        self.capacity_record = []

        self.data_phase_record = []
        self.phase_data_record = []
