import torch
from pytorchvideo.data import make_clip_sampler

import utils
from gaze_labeled_video_dataset import gaze_labeled_video_dataset
from gaze_video_data_module import VAL_TRANSFORM
from model import GazePredictionLightningModule


_DATA_PATH = r'data/GazeCom/movies_m2t_224x224'
_CHECKPOINT_PATH = r'lightning_logs_ssh/version_39/checkpoints/epoch=9-step=399.ckpt'
_CLIP_DURATION = 2
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

model = GazePredictionLightningModule.load_from_checkpoint(_CHECKPOINT_PATH)


samples = 4
for _ in range(samples):
    sample = next(dataset)

    y_hat = model(sample['video'][None, :].to(device=device))[0]
    y = sample['frame_labels']

    # (C, F, H, W) -> (F, H, W, C)
    frames = torch.swapaxes(sample['video'].T, 0, 2)

    utils.plot_frames_with_labels(frames.cpu().detach().numpy(), y_hat.cpu().detach().numpy(),
                                  sample['em_data'].cpu().detach().numpy(), y[:, None, :].tolist(), box_width=8)
