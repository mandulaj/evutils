
import torch
import numpy as np

# from metavision_core_ml.utils.torch_ops import normalize_tiles, viz_flow
# from metavision_core_ml.event_to_video.lightning_model import EventToVideoLightningModel
# from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch, event_volume



from .rpg_e2vid.utils.loading_utils import load_model as rpg_load_model
from .rpg_e2vid.image_reconstructor import ImageReconstructor
from .rpg_e2vid.options.inference_options import set_inference_options as rpg_set_inference_options
from .rpg_e2vid.utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from .rpg_e2vid.utils.inference_utils import CropParameters, EventPreprocessor, IntensityRescaler, ImageFilter, ImageDisplay, ImageWriter, UnsharpMaskFilter


from types import SimpleNamespace
from .base import Reconstructor


def set_inference_options(params):
    rpg_set_inference_options(params)





class RPG_Reconstructor(Reconstructor):
    DEFAULT_ARGS = {
        'no_recurrent': False,
        'no_normalize': False,
        'color': False,
        'auto_hdr_median_filter_size': 10,
        'auto_hdr': False,
        'Imax': 1.0,
        'Imin': 0.0,
        'flip': False,
        'bilateral_filter_sigma': 0.0,
        'unsharp_mask_sigma': 1.0,
        'unsharp_mask_amount': 0.3,
        'hot_pixels_file': None,
        'display_wait_time': 1,
        'display_border_crop': 0,
        'num_bins_to_show': -1,
        'event_display_mode': 'red-blue',
        'show_events': False,
        'display': False,
        'compute_voxel_grid_on_cpu': False,
        'use_cuda': True,
        'dataset_name': 'reconstruction',
        'model_path': "models/E2VID_lightweight.pth.tar"
    }

    def __init__(self, height, width, args={}):
        args = {**RPG_Reconstructor.DEFAULT_ARGS, **args}


        super().__init__(height, width, args)

        sn_args = SimpleNamespace(**args)

        self.model = rpg_load_model(sn_args.model_path)
        self.model.eval().to(self.device)

        self.no_recurrent = args['no_recurrent']

        self.crop = CropParameters(self.width, self.height, self.model.num_encoders)
        self.last_states_for_each_channel = {'grayscale': None}

        self.event_preprocessor = EventPreprocessor(sn_args)
        self.intensity_rescaler = IntensityRescaler(sn_args)
        self.image_filter = ImageFilter(sn_args)
        self.unsharp_mask_filter = UnsharpMaskFilter(sn_args, device=self.device)
        # self.image_writer = ImageWriter(args)
        # self.image_display = ImageDisplay(args)

        # self.start_index = 0
        self.last_ts = 0
        self.reset_states = False


    def update_reconstruction(self, event_tensor):
        if isinstance(event_tensor, np.ndarray):
            event_tensor = torch.from_numpy(event_tensor)
        
        with torch.no_grad():

            events = event_tensor.unsqueeze(dim=0)
            events = events.to(self.device)

            events = self.event_preprocessor(events)

            # Resize tensor to [1 x C x crop_size x crop_size] by applying zero padding
            events_for_each_channel = {'grayscale': self.crop.pad(events)}
            reconstructions_for_each_channel = {}


            # Reconstruct new intensity image for each channel (grayscale + RGBW if color reconstruction is enabled)
            for channel in events_for_each_channel.keys():
                new_predicted_frame, states = self.model(events_for_each_channel[channel],
                                                        self.last_states_for_each_channel[channel])

                if self.no_recurrent or self.reset_states:
                    self.last_states_for_each_channel[channel] = None
                else:
                    self.last_states_for_each_channel[channel] = states

                # Output reconstructed image
                crop = self.crop

                # Unsharp mask (on GPU)
                new_predicted_frame = self.unsharp_mask_filter(new_predicted_frame)

                # Intensity rescaler (on GPU)
                new_predicted_frame = self.intensity_rescaler(new_predicted_frame)

                reconstructions_for_each_channel[channel] = new_predicted_frame[0, 0, crop.iy0:crop.iy1,
                                                                crop.ix0:crop.ix1].cpu().numpy()

            out = reconstructions_for_each_channel['grayscale']

            # Post-processing, e.g bilateral filter (on CPU)
            out = self.image_filter(out)

            # self.image_display(out, events)


            return out



    def gen_frame(self, e):
        if len(e) == 0:
            events = np.hstack((np.float32(np.array([self.last_ts])[None].T/1e6), np.array([0])[None].T, np.array([0])[None].T, np.array([0])[None].T))
            self.reset_states = True
        else:
            events = np.hstack((np.float32(e['t'][None].T/1e6), e['x'][None].T, e['y'][None].T, e['p'][None].T))
            self.last_ts = e['t'][-1]
            self.reset_states = False

        


        if self.args['compute_voxel_grid_on_cpu']:
            event_tensor = events_to_voxel_grid(events,
                                                num_bins=self.model.num_bins,
                                                width=self.width,
                                                height=self.height)
            event_tensor = torch.from_numpy(event_tensor)
        else:
            event_tensor = events_to_voxel_grid_pytorch(events,
                                                        num_bins=self.model.num_bins,
                                                        width=self.width,
                                                        height=self.height,
                                                        device=self.device)

        # num_events_in_window = events.shape[0]
        out = self.update_reconstruction(event_tensor)

        # self.start_index += num_events_in_window

        return out



