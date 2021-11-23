import torch
import numpy as np
from pytorchvideo.data import make_clip_sampler

import utils
from gaze_labeled_video_dataset import gaze_labeled_video_dataset
from gaze_video_data_module import VAL_TRANSFORM
from model import GazePredictionLightningModule


_PLOT_RESULTS = False

_DATA_PATH = r'data/GazeCom/movies_m2t_224x224/val'
_CHECKPOINT_PATH = r'lightning_logs/version_42/checkpoints/epoch=9-step=798.ckpt'
_CLIP_DURATION = 5
_VIDEO_SUFFIX = ''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = gaze_labeled_video_dataset(
    data_path=_DATA_PATH,
    #clip_sampler=make_clip_sampler("uniform", _CLIP_DURATION),
    clip_sampler=make_clip_sampler("random", _CLIP_DURATION),
    #video_sampler=torch.utils.data.SequentialSampler,
    video_sampler=torch.utils.data.RandomSampler,
    transform=VAL_TRANSFORM,
    #transform=None,
    video_file_suffix=_VIDEO_SUFFIX,
    decode_audio=False,
    decoder="pyav",
)

model = GazePredictionLightningModule.load_from_checkpoint(_CHECKPOINT_PATH).to(device=device)

samples = 4
for i in range(samples):
    sample = next(dataset)

    y_hat = model(sample['video'][None, :].to(device=device))[0]
    y = sample['frame_labels']

    # (C, F, H, W) -> (F, H, W, C)
    frames = torch.swapaxes(sample['video'].T, 0, 2)

    frames = frames.cpu().detach().numpy()
    em_data_hat = None
    y_hat = y_hat.cpu().detach().numpy()
    em_data = sample['em_data'].cpu().detach().numpy()
    y = y[:, None, :].tolist()

    if y_hat.shape[1] == 3:
        em_data_hat = y_hat[:, 2]
        y_hat = y_hat[:, :2]

    if _PLOT_RESULTS:
        utils.plot_frames_with_labels(frames, y_hat, em_data_hat, y, em_data, box_width=8)
    else:
        filepath = f'data/sample_outputs/version_42_{i}'
        np.savez(filepath, frames=frames, em_data_hat=em_data_hat, y_hat=y_hat, em_data=em_data, y=y)
