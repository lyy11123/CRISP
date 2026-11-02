
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))


from device import device


from typing import List, Dict, Tuple, Set


class module(device):

    def __init__(self, module_type: str, module_id: Tuple[int]):
        super().__init__(device_type="module", device_id=module_id)

        self.module_type = module_type
        self.module_id = module_id
