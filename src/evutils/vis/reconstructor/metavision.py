"""Event-to-video reconstruction using the Metavision ML model."""

import numpy as np
from evutils.torch import _try_import_torch
torch = _try_import_torch()
if torch:
    F = torch.nn.functional
else:
    F = None
from typing import Any

from ._base import Reconstructor

# metavision_core_ml is not on PyPI (ships with OpenEB/Metavision SDK), so it is
# never a declared dependency; the fallback class below raises on use.
try:
    from metavision_core_ml.utils.torch_ops import normalize_tiles, viz_flow  # type: ignore  # not on PyPI; may or may not be installed
    from metavision_core_ml.event_to_video.lightning_model import EventToVideoLightningModel  # type: ignore  # not on PyPI; may or may not be installed
    from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch, event_volume  # type: ignore  # not on PyPI; may or may not be installed

    class Metavision_Reconstructor(Reconstructor):
        """Reconstructor using the Metavision model.
        
        Parameters
        ----------
        height : int
            Height of the frame
        width : int
            Width of the frame
        args : dict, optional
            Additional arguments for the Metavision reconstructor, by default {}

        """

        def __init__(self, height: int, width: int, args: Any = None) -> None:
            super().__init__(height, width, args if args is not None else {})

            self.model = EventToVideoLightningModel.load_from_checkpoint("models/e2v.ckpt")
            self.model.eval().to(self.device)

        def gen_frame(self, e: np.ndarray) -> np.ndarray:
            """Reconstructs a single frame from the given events.

            Parameters
            ----------
            e : np.ndarray
                Array of events.

            Returns
            -------
            np.ndarray
                Reconstructed grayscale image frame as a numpy array of shape (height, width)
                with dtype uint8.

            """
            nbins = self.model.hparams.event_volume_depth

            if len(e) == 0:
                return np.zeros((self.height, self.width), dtype=np.uint8)


            events_th = event_cd_to_torch(e).to(self.device)
            start_times = torch.FloatTensor([e['t'][0]]).view(1,).to(self.device)

            durations = torch.FloatTensor([e['t'][-1] - e['t'][0]]).view(1,).to(self.device)
            tensor_th = event_volume(events_th, 1, self.height, self.width, start_times, durations, nbins, 'bilinear')
            tensor_th = F.interpolate(tensor_th, size=(self.height, self.width),
                                        mode='bilinear', align_corners=True)
            tensor_th = tensor_th.view(1, 1, nbins, self.height, self.width)


            state = self.model.model(tensor_th)
            gray = self.model.model.predict_gray(state).view(1, 1, self.height, self.width)
            gray = normalize_tiles(gray).view(self.height, self.width)
            gray_np = gray.detach().cpu().numpy() * 255
            return np.asarray(gray_np, dtype=np.uint8)
        
except ImportError:
    class Metavision_Reconstructor(Reconstructor):  # type: ignore[no-redef]
        """Reconstructor using the Metavision model (not available).

        Parameters
        ----------
        height : int
            Height of the frame
        width : int
            Width of the frame
        args : dict, optional
            Additional arguments for the Metavision reconstructor, by default {}

        """

        def __init__(self, height: int, width: int, args: Any = None) -> None:
            raise ImportError("Please install metavision_core_ml to use this class")
        
    


