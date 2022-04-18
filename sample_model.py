import torch
import os
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
_OUTPUT_DIR = r"data/sample_outputs/version_346"
_MODE = 'train'

_DATA_PATH = f'data/GazeCom/movies_m2t_224x224/{_MODE}'
_DATA_PATH = f'data/GazeCom/movies_m2t_224x224/all_videos_single_observer/{_MODE}'
#DATA_PATH = f'data/GazeCom/movies_m2t_224x224/single_video/{_MODE}'
#_DATA_PATH = f'data/GazeCom/movies_m2t_224x224/single_clip/{_MODE}'

_CHECKPOINT_PATH = r'data/lightning_logs/version_346/checkpoints/epoch=100-step=100.ckpt'

_SCALE_UP = True

_CLIP_DURATION = 5
_VIDEO_SUFFIX = ''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = gaze_labeled_video_dataset(
    data_path=_DATA_PATH,
    #clip_sampler=make_clip_sampler("uniform", _CLIP_DURATION),
    #video_sampler=torch.utils.data.SequentialSampler,
    clip_sampler=make_clip_sampler("random", _CLIP_DURATION),
    video_sampler=torch.utils.data.RandomSampler,
    transform=VAL_TRANSFORM,
    #transform=None,
    video_file_suffix=_VIDEO_SUFFIX,
    decode_audio=False,
    decoder="pyav",
)

model = GazePredictionLightningModule.load_from_checkpoint(_CHECKPOINT_PATH).to(device=device)

em_encoder = OneHotEncoder()
em_encoder.fit([[i] for i in range(4)])

samples_per_clip = 5
samples = 4
for i in range(0, samples):
    sample = next(dataset)

    video_name = sample['video_name']
    observer = sample['observer']
    print(video_name, observer)

    # Sample multiple scanpaths from same input to late average over metrics
    y_hats = [model(sample['video'][None, :].to(device=device))[0].cpu().detach().numpy() for i in range(samples_per_clip)]
    y = sample['frame_labels']

    # (C, F, H, W) -> (F, H, W, C)
    frames = torch.swapaxes(sample['video'].T, 0, 2)

    frames = frames.cpu().detach().numpy()
    frames = np.interp(frames, (frames.min(), frames.max()), (0, 255)).astype('uint8')
    em_data_hats = None
    em_data = sample['em_data'].cpu().detach().numpy()
    frame_indices = sample['frame_indices']
    y = y.cpu().detach().numpy()
    print(f"Frames {frame_indices[0]}-{frame_indices[-1]}")

    print("y_hat.shape", y_hats[0].shape)
    if y_hats[0].shape[1] > 2:
        em_data_hats = [y_hat[:, 2:] for y_hat in y_hats]
        y_hats = [y_hat[:, :2] for y_hat in y_hats]
        em_data_hats = [em_encoder.inverse_transform(em_data_hat).reshape((-1)) for em_data_hat in em_data_hats]

    save_dir = None if _OUTPUT_DIR is None else f'{_OUTPUT_DIR}/{i}'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if em_encoder:
        em_data = em_encoder.inverse_transform(em_data).reshape((-1))
    else:
        em_data = em_data

    if _SCALE_UP:
        y_hats = [(y_hat + 1) * 112 for y_hat in y_hats]
        y = (y + 1) * 112

    print("y_hat")
    print(y_hats[0][:5])
    print("y")
    print(y[:5])
    if em_data_hats is not None:
        print("em_data_hat")
        print(em_data_hats[0][:20])
        print("em_data")
        print(em_data[:20])

    nss_orig = score_gaussian_density(video_name, y.astype(int), frame_ids=frame_indices)
    nss_scores = [score_gaussian_density(video_name, y_hat.astype(int), frame_ids=frame_indices) for y_hat in y_hats]
    nss = np.array(nss_scores).mean()
    gaze_mid = np.ones(y.shape, dtype=np.int32) * 112
    nss_mid = score_gaussian_density(video_name, gaze_mid, frame_ids=frame_indices)
    print("NSS original clip:", nss_orig)
    print("NSS prediction:", nss, "all scores:", nss_scores)
    print("NSS middle baseline:", nss_mid, "\n")
    
    y_hats = np.stack(y_hats, axis=1)
    if em_data_hats is not None:
        em_data_hats = np.stack(em_data_hats, axis=1)

    utils.plot_frames_with_labels(frames, y, em_data, y_hats, em_data_hats, box_width=8, save_to_directory=save_dir)
    subprocess.call(f"/mnt/antares_raid/home/yannicsl/miniconda3/envs/thesis/bin/ffmpeg -framerate 10 -start_number 0 -i {i}/%03d.png -pix_fmt yuv420p {_MODE}_{i}.mp4", cwd=_OUTPUT_DIR, shell=True)
    shutil.rmtree(save_dir)
    with open(os.path.join(_OUTPUT_DIR, "metadata.txt"), "a") as f:
        f.write(f"{_MODE}_{i}: {video_name}+{observer}, Frames {frame_indices[0]}-{frame_indices[-1]}, nss (original, prediction, middle): ({nss_orig:.2f}, {nss:.2f}, {nss_mid:.2f})\n")
    #if _PLOT_RESULTS:
    #    utils.plot_frames_with_labels(frames, y_hat, em_data_hat, y, em_data, box_width=8)
    #else:
    #    filepath = f'data/sample_outputs/version_43/{i}'
    #    np.savez(filepath, frames=frames, em_data_hat=em_data_hat, y_hat=y_hat, em_data=em_data, y=y)
