import numpy as np
import torch

from .base import Reconstructor

try:
    from metavision_core_ml.utils.torch_ops import normalize_tiles, viz_flow
    from metavision_core_ml.event_to_video.lightning_model import EventToVideoLightningModel
    from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch, event_volume

    class Metavision_Reconstructor(Reconstructor):
        def __init__(self, device, height, width, args):
            super().__init__(device, height, width, args)


            self.model =  EventToVideoLightningModel.load_from_checkpoint("models/e2v.ckpt")
            self.model.eval().to(self.device)

        def gen_frame(self, e):

            nbins = self.model.hparams.event_volume_depth


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
            return np.uint8(gray_np)
        
except ImportError:
    class Metavision_Reconstructor(Reconstructor):
        def __init__(self, device, height, width, args={}):
            raise ImportError("Please install metavision_core_ml to use this class")
        
    


