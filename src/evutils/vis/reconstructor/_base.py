"""Base classes and utilities for event-to-video reconstruction."""

import torch

import subprocess
import numpy as np
from types import SimpleNamespace

from typing import Union
def get_freer_gpu(cuda_string: bool = False) -> Union[int, str, None]:
    """Finds the GPU with the most free memory.

    Parameters
    ----------
    cuda_string : bool, optional
        If True, returns a string formatted as 'cuda:{id}', or 'cpu' if no GPU is found.
        If False, returns the integer ID of the freer GPU, or None if no GPU is found.
        By default False.

    Returns
    -------
    int or str or None
        The ID or formatted string of the GPU with the most free memory,
        or None/'cpu' if no GPU is available.

    """
    import subprocess
    try:
        memory_free_info = subprocess.check_output(['/bin/sh', '-c', 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'], encoding='utf-8').split('\n')
        memory_free = [int(v) for v in memory_free_info if v.strip()]
    except Exception as e:
        print(f"Error retrieving GPU memory info: {e}")
        return None
    most_free = max(range(len(memory_free)), key=lambda i: memory_free[i]) if memory_free else None
    if cuda_string:
        if most_free is None:
            return 'cpu'
        return f"cuda:{most_free}"
    return most_free

class Reconstructor():
    """Base class for reconstructing frames from events.

    Parameters
    ----------
    width : int
        Width of the frame
    height : int
        Height of the frame
    args : dict, optional
        Additional arguments for the reconstructor, by default {}

    """

    DEFAULT_ARGS = {
        'device': "auto"
    }
    def __init__(self, width: int, height: int, args: "dict | None" = None) -> None:
        if args is None: args = {}
        self.args = {**Reconstructor.DEFAULT_ARGS, **args}
        

        if self.args['device'] == "auto":
            if torch.cuda.is_available():
                self.device = torch.device(get_freer_gpu(cuda_string=True)) # type: ignore[arg-type]
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.args['device'])
        self.height = height
        self.width = width

    def gen_frame(self, events: np.ndarray) -> np.ndarray:
        """Reconstruct a frame from events.

        Parameters
        ----------
        events : np.ndarray
            Array of events in the :class:`~evutils.types.Events` format

        Returns
        -------
        np.ndarray
            A numpy array with the frame (height, width, channels)

        """
        raise NotImplementedError
