
import torch

import subprocess
import numpy as np
from types import SimpleNamespace


def get_freer_gpu():
    memory_free_info = subprocess.check_output(['/bin/sh', '-c', 'nvidia-smi --query-gpu=memory.free --format=csv']).decode('ascii').split('\n')[1:-1]
    memory_free = [int(v.split()[0]) for v in memory_free_info]
    return np.argmax(memory_free)


class Reconstructor():
    '''
    Base class for reconstructing frames from events

    Parameters
    ----------
    height : int
        Height of the frame
    width : int
        Width of the frame
    args : dict, optional
        Additional arguments for the reconstructor, by default {}
    '''
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


    def get_frame(events: np.ndarray) -> np.ndarray:
        '''
        Reconstruct a frame from events

        Parameters
        ----------
        events : np.ndarray
            Array of events in the :class:`~evutils.types.Events` format


        Returns
        -------
        np.ndarray
            A numpy array with the frame (height, width, channels)

        '''
        raise NotImplementedError
