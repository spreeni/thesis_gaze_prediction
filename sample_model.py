import torch
import os
import sys
import subprocess
import shutil
import numpy as np
from pytorchvideo.data import make_clip_sampler
from sklearn.preprocessing import OneHotEncoder

import utils
from metrics import score_gaussian_density
from gaze_labeled_video_dataset import gaze_labeled_video_dataset
from gaze_video_data_module import VAL_TRANSFORM
from model import GazePredictionLightningModule


#_PLOT_RESULTS = False
_OUTPUT_DIR = r"data/sample_outputs/version_301"
_MODE = 'train'

_DATA_PATH = f'data/GazeCom/movies_m2t_224x224/{_MODE}'
_DATA_PATH = f'data/GazeCom/movies_m2t_224x224/single_video/{_MODE}'
_DATA_PATH = f'data/GazeCom/movies_m2t_224x224/single_clip/{_MODE}'

_CHECKPOINT_PATH = r'data/lightning_logs/version_301/checkpoints/epoch=41-step=41.ckpt'

_SCALE_UP = True
CHANGE_DATA = True

_CLIP_DURATION = 2
_VIDEO_SUFFIX = ''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = gaze_labeled_video_dataset(
    data_path=_DATA_PATH,
    clip_sampler=make_clip_sampler("uniform", _CLIP_DURATION),
    video_sampler=torch.utils.data.SequentialSampler,
    #clip_sampler=make_clip_sampler("random", _CLIP_DURATION),
    #video_sampler=torch.utils.data.RandomSampler,
    transform=VAL_TRANSFORM,
    #transform=None,
    video_file_suffix=_VIDEO_SUFFIX,
    decode_audio=False,
    decoder="pyav",
    predict_change=CHANGE_DATA
)

model = GazePredictionLightningModule.load_from_checkpoint(_CHECKPOINT_PATH).to(device=device)

em_encoder = OneHotEncoder()
em_encoder.fit([[i] for i in range(4)])

samples = 4
for i in range(0, samples):
    sample = next(dataset)

    video_name = sample['video_name']
    observer = sample['observer']
    print(video_name, observer)

    y_hat = model(sample['video'][None, :].to(device=device))[0]
    y = sample['frame_labels']

    # (C, F, H, W) -> (F, H, W, C)
    frames = torch.swapaxes(sample['video'].T, 0, 2)

    frames = frames.cpu().detach().numpy()
    frames = np.interp(frames, (frames.min(), frames.max()), (0, 255)).astype('uint8')
    em_data_hat = None
    y_hat = y_hat.cpu().detach().numpy()
    em_data = sample['em_data'].cpu().detach().numpy()
    frame_indices = sample['frame_indices']
    y = y[:, None, :].cpu().detach().numpy()
    print(f"Frames {frame_indices[0]}-{frame_indices[-1]}")

    if CHANGE_DATA:
        y_hat = np.tanh(y_hat.cumsum(axis=0))
        y = np.tanh(y.cumsum(axis=0))

    print("y_hat.shape", y_hat.shape)
    if y_hat.shape[1] > 2:
        em_data_hat = y_hat[:, 2:]
        y_hat = y_hat[:, :2]
        em_data_hat = em_encoder.inverse_transform(em_data_hat).reshape((-1))

    save_dir = None if _OUTPUT_DIR is None else f'{_OUTPUT_DIR}/{i}'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if em_encoder:
        em_data = em_encoder.inverse_transform(em_data)
    else:
        em_data = em_data[:, None]

    if _SCALE_UP:
        y_hat = (y_hat + 1) * 112
        y = (y + 1) * 112

    print("y_hat")
    print(y_hat[:5])
    print("y")
    print(y[:5])
    if em_data_hat is not None:
        print("em_data_hat")
        print(em_data_hat[:20])
        print("em_data")
        print(em_data[:20])

    nss_orig = score_gaussian_density(video_name, y[0, :, :].astype(int), frame_ids=frame_indices)
    nss = score_gaussian_density(video_name, y_hat.astype(int), frame_ids=frame_indices)
    gaze_mid = np.ones(y_hat.shape, dtype=np.int32) * 112
    nss_mid = score_gaussian_density(video_name, gaze_mid, frame_ids=frame_indices)
    print("NSS original clip:", nss_orig)
    print("NSS prediction:", nss)
    print("NSS middle baseline:", nss_mid, "\n")

    utils.plot_frames_with_labels(frames, y_hat, em_data_hat, y, em_data, box_width=8, save_to_directory=save_dir)
    subprocess.call(f"/mnt/antares_raid/home/yannicsl/miniconda3/envs/thesis/bin/ffmpeg -framerate 10 -start_number 0 -i {i}/%03d.png -pix_fmt yuv420p {_MODE}_{i}.mp4", cwd=_OUTPUT_DIR, shell=True)
    shutil.rmtree(save_dir)
    with open(os.path.join(_OUTPUT_DIR, "metadata.txt"), "a") as f:
        f.write(f"{_MODE}_{i}: {video_name}+{observer}, Frames {frame_indices[0]}-{frame_indices[-1]}, nss (original, prediction, middle): ({nss_orig:.2f}, {nss:.2f}, {nss_mid:.2f})\n")
    #if _PLOT_RESULTS:
    #    utils.plot_frames_with_labels(frames, y_hat, em_data_hat, y, em_data, box_width=8)
    #else:
    #    filepath = f'data/sample_outputs/version_43/{i}'
    #    np.savez(filepath, frames=frames, em_data_hat=em_data_hat, y_hat=y_hat, em_data=em_data, y=y)
