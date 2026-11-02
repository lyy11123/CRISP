
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))

from device import device


from typing import List, Dict, Tuple, Set


class link(device):

    def __init__(self, link_type: str, link_id: Tuple[int]):
        super().__init__(device_type="link", device_id=link_id)

        self.link_type = link_type
        self.link_id = link_id
