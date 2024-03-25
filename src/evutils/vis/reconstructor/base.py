
import torch

import subprocess
import numpy as np
from types import SimpleNamespace


def get_freer_gpu():
    memory_free_info = subprocess.check_output(['/bin/sh', '-c', 'nvidia-smi --query-gpu=memory.free --format=csv']).decode('ascii').split('\n')[1:-1]
    memory_free = [int(v.split()[0]) for v in memory_free_info]
    return np.argmax(memory_free)


class Reconstructor():
    DEFAULT_ARGS = {
        'device': "auto"
    }
    def __init__(self, height, width, args={}):
        self.args = {**Reconstructor.DEFAULT_ARGS, **args}
        

        if self.args['device'] == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:" + str(get_freer_gpu()))
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.args['device'])
        self.height = height
        self.width = width


    def get_frame(events):
        raise NotImplementedError
