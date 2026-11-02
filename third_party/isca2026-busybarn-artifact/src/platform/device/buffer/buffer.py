
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))

from storage import storage


from typing import List, Dict, Tuple, Set


class buffer(storage):

    def __init__(self, buffer_type: str, buffer_id: Tuple[int]):
        super().__init__(device_type="buffer", device_id=buffer_id)

        self.buffer_type = buffer_type
        self.buffer_id = buffer_id
