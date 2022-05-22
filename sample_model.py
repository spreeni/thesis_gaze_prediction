import torch
import os
import subprocess
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorchvideo.data import make_clip_sampler
from sklearn.preprocessing import OneHotEncoder

import utils
import metrics
import metrics_nss
from gaze_labeled_video_dataset import gaze_labeled_video_dataset
from gaze_video_data_module import VAL_TRANSFORM
from model import GazePredictionLightningModule


#_PLOT_RESULTS = False
_OUTPUT_DIR = r"data/sample_outputs/version_471_single_clip"
_MODE = 'train'

_DATA_PATH = f'data/GazeCom/movies_m2t_224x224/{_MODE}'
_DATA_PATH = f'data/GazeCom/movies_m2t_224x224/all_videos_single_observer/{_MODE}'
_DATA_PATH = f'data/GazeCom/movies_m2t_224x224/single_video_all_observers/{_MODE}'
#_DATA_PATH = f'data/GazeCom/movies_m2t_224x224/single_video/{_MODE}'
_DATA_PATH = f'data/GazeCom/movies_m2t_224x224/single_clip/{_MODE}'
#_MODE += '_golf'
_CHECKPOINT_PATH = r'data/lightning_logs/version_471/checkpoints/epoch=101-step=101.ckpt'

_CALC_NSS = True

_SCALE_UP = True
_SHOW_SALIENCY = True
_PLOT_GAZE_CHANGE_HISTOGRAMS = True

_CLIP_DURATION = 5
_VIDEO_SUFFIX = ''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dataset():
    return gaze_labeled_video_dataset(
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
dataset = get_dataset()

model = GazePredictionLightningModule.load_from_checkpoint(_CHECKPOINT_PATH).to(device=device)
"""
model = GazePredictionLightningModule(lr=1e-6, batch_size=16, frames=round(_CLIP_DURATION * 29.97),
                                                        input_dims=(224, 224), out_channels=8,
                                                        predict_em=False,
                                                        fpn_only_use_last_layer=True,
                                                        rim_hidden_size=400,
                                                        rim_num_units=6, rim_k=4, rnn_cell='LSTM',
                                                        rim_layers=1, out_attn_heads=2,
                                                        p_teacher_forcing=0.3, n_teacher_vals=0,
                                                        weight_init='xavier_normal', mode='RIM', loss_fn='mse_loss',
                                                        lambda_reg_fix=6., lambda_reg_sacc=0.1, input_attn_heads=3, 
                                                        input_dropout=0.2, comm_dropout=0.2, channel_wise_attention=False)
"""
em_encoder = OneHotEncoder()
em_encoder.fit([[i] for i in range(4)])

if _CALC_NSS:
    df_nss = pd.DataFrame(columns=['video', 'observer', 'first_frame', 'last_frame', 'nss_orig', 'nss_pred', 'nss_middle', 'nss_rnd'])

samples_per_clip = 5 if not _CALC_NSS else 1
samples = 2 if not _CALC_NSS else 100
#for i, (video_name, observer, clip_start) in enumerate([
#    ('golf', 'AAW', 5.),
#    ('golf', 'AAW', 10.)
#]):
#    sample = dataset.get_clip(video_name, observer, clip_start=clip_start)
for i in range(0, samples):
    try:
        sample = next(dataset)
        #sample = dataset.get_clip('golf', 'AAW', clip_start=3. + i * _CLIP_DURATION)
        #sample = dataset.get_clip('doves', 'AAW', clip_start=3. + i * _CLIP_DURATION)
    except Exception:
        dataset = get_dataset()
        continue

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

    nss_orig = metrics_nss.score_gaussian_density(video_name, y.astype(int), frame_ids=frame_indices)
    nss_scores = [metrics_nss.score_gaussian_density(video_name, y_hat.astype(int), frame_ids=frame_indices) for y_hat in y_hats]
    nss = np.array(nss_scores).mean()
    gaze_mid = np.ones(y.shape, dtype=np.int32) * 112
    nss_mid = metrics_nss.score_gaussian_density(video_name, gaze_mid, frame_ids=frame_indices)
    gaze_rnd = np.random.randint(225, size=y.shape, dtype=np.int32)
    nss_rnd = metrics_nss.score_gaussian_density(video_name, gaze_rnd, frame_ids=frame_indices)
    print("NSS original clip:", nss_orig)
    print("NSS prediction:", nss, "all scores:", nss_scores)
    print("NSS middle baseline:", nss_mid)
    print("NSS random baseline:", nss_rnd, "\n")
    
    if _CALC_NSS:
        df_nss = df_nss.append({'video': video_name, 'observer': observer, 'first_frame': frame_indices[0], 'last_frame': frame_indices[-1],
                            'nss_orig': nss_orig, 'nss_pred': nss, 'nss_middle': nss_mid, 'nss_rnd': nss_rnd}, ignore_index=True)
        continue

    #y_hats = np.stack(y_hats, axis=1)
    if em_data_hats is not None:
        em_data_hats = np.stack(em_data_hats, axis=1)

    save_dir = None if _OUTPUT_DIR is None else f'{_OUTPUT_DIR}/{i}'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if _SHOW_SALIENCY:
        nss_calc = metrics_nss.NSSCalculator()
        nss_calc.load_gaussian_density(os.path.join('metrics', 'gaussian_density', f'{video_name}.npy'))
        density = nss_calc.gaussian_density[frame_indices[0]:frame_indices[-1] + 1, :, :]
        density = np.swapaxes(density, 1, 2)

        #nss_calc.save_animated_gaussian_density(os.path.join(_OUTPUT_DIR, f'{_MODE}_{i}_{video_name}_orig.png'), animate=False, gaze_data=y)
        nss_calc.save_animated_gaussian_density(os.path.join(_OUTPUT_DIR, f'{_MODE}_{i}_{video_name}_orig.png'), animate=False, frame_start=frame_indices[0], frame_end=frame_indices[-1])
        nss_calc.save_animated_gaussian_density(os.path.join(_OUTPUT_DIR, f'{_MODE}_{i}_{video_name}.png'), animate=False, gaze_data=y_hats)

        # Norm to [threshhold, 1]
        #density -= density.min(axis=(1, 2), keepdims=True)
        #density /= density.max(axis=(1, 2), keepdims=True)
        #density = density.clip(min=0.15)
        
        # Mask frames with gaussian density map
        #frames = (frames.astype(float) * density[:, :, :, None]).astype(int)

        color_overlay = (plt.cm.viridis(density) * 255)[:, :, :, :3]
        frames = (frames.astype(float) * 0.7 + color_overlay * 0.3).astype(int)

    # Calculate similarity in gaze change orientation and length distribution
    change_len, change_deg = utils.get_gaze_change_dist_and_orientation(y, absolute_values=True, normalize_gaze=True)
    change_data_hat = [utils.get_gaze_change_dist_and_orientation(y_hat, absolute_values=True, normalize_gaze=True) for y_hat in y_hats]
    metric = 'cosine_dist'
    change_len_similarity = np.mean(metrics.calc_similarity_gaze_change_distance(change_len, change_len_hat, metric) for change_len_hat, _ in change_data_hat)
    change_deg_similarity = np.mean(metrics.calc_similarity_gaze_change_distance(change_deg, change_deg_hat, metric) for _, change_deg_hat in change_data_hat)

    # Visualize predictions over video and in comparison with groundtruth
    utils.plot_frames_with_labels(frames, y, em_data, np.stack(y_hats, axis=1), em_data_hats, box_width=8, save_to_directory=save_dir)
    utils.create_movie_from_frames(_OUTPUT_DIR, str(i), f"{_MODE}_{i}.mp4", fps=10, width_px=1800, remote_machine=True,
                                   delete_frames=True)

    # Save metrics and metadata
    with open(os.path.join(_OUTPUT_DIR, "metadata.txt"), "a") as f:
        f.write(f"{_MODE}_{i}: {video_name}+{observer}, Frames {frame_indices[0]}-{frame_indices[-1]}, nss (original, prediction, middle): ({nss_orig:.2f}, {nss:.2f}, {nss_mid:.2f}), "
                f"change_dist_similarity: {change_len_similarity:.3f}, change_deg_similarity: {change_deg_similarity:.3f}\n")

    if _PLOT_GAZE_CHANGE_HISTOGRAMS:
        utils.plot_gaze_change_dist_and_orientation(change_len, change_deg, f"{_OUTPUT_DIR}/{_MODE}_{i}_change_similarity", use_plotly=True)
        utils.plot_gaze_change_dist_and_orientation(*change_data_hat[0], f"{_OUTPUT_DIR}/{_MODE}_{i}_change_similarity_pred", use_plotly=True)

    #if _PLOT_RESULTS:
    #    utils.plot_frames_with_labels(frames, y_hat, em_data_hat, y, em_data, box_width=8)
    #else:
    #    filepath = f'data/sample_outputs/version_43/{i}'
    #    np.savez(filepath, frames=frames, em_data_hat=em_data_hat, y_hat=y_hat, em_data=em_data, y=y)

if _CALC_NSS:
    diff_to_orig = df_nss.nss_orig - df_nss.nss_pred
    diff_to_mid = df_nss.nss_middle - df_nss.nss_pred
    diff_to_rnd = df_nss.nss_rnd - df_nss.nss_pred
    print(f"\nNSS (original - prediction): {diff_to_orig.mean():.2f}+-{diff_to_orig.std():.2f}")
    print(f"NSS (middle - prediction): {diff_to_mid.mean():.2f}+-{diff_to_mid.std():.2f}")
    print(f"NSS (random - prediction): {diff_to_rnd.mean():.2f}+-{diff_to_rnd.std():.2f}")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    df_nss.to_csv(os.path.join(_OUTPUT_DIR, f'{_MODE}_nss.csv'), index=False)