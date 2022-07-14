"""
Sample predictions from trained models. Visualize results and calculate metrics like NSS or gaze change similarities.
"""
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorchvideo.data import make_clip_sampler
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

import utils
import metrics
import metrics_nss
from gaze_labeled_video_dataset import gaze_labeled_video_dataset
from gaze_video_data_module import VAL_TRANSFORM
from model import GazePredictionLightningModule


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sample_model(checkpoint_path: str, output_dir: str, data_partition: str, clip_duration: float, mode: str = 'train',
                 calc_metrics: bool = False, show_saliency: bool = True, plot_gaze_change_histograms: bool = True,
                 teacher_forcing_on_inference: bool = False):
    """
    Sample from a pre-trained model from a partition of the dataset.

    Calculates NSS scores of ground truth, prediction and center/random baselines on given video.

    Either can be sampled to visualize model predictions as a video or to calculate metrics over many samples.

    Args:
        checkpoint_path:                Filepath to a checkpoint of the trained model
        output_dir:                     Directory to which outputs are written to
        data_partition:                 Partition of the dataset to use (['single_clip', 'single_video', 'single_video_all_observers', 'all_videos_single_observer', 'all_videos_all_observers'])
        clip_duration:                  Clip duration in seconds
        mode:                           Sampling from 'train' or 'val' set; default is 'train'
        calc_metrics:                   Flag for only calculating metrics; default is to visualize few samples
        show_saliency:                  Flag to overlay observer ground truths over video; default is True
        plot_gaze_change_histograms:    Flag to additionally plot histograms of gaze change distributions; default is True
        teacher_forcing_on_inference:   Flag to activate teacher forcing during model sampling; default is False
    """
    #print(f"mode: {mode}, calc_metrics: {calc_metrics}")
    if data_partition == 'all_videos_all_observers':
        _DATA_PATH = f'data/GazeCom/movies_m2t_224x224/{mode}'
    elif data_partition == 'all_videos_single_observer':
        _DATA_PATH = f'data/GazeCom/movies_m2t_224x224/all_videos_single_observer/{mode}'
    elif data_partition == 'single_video_all_observers':
        _DATA_PATH = f'data/GazeCom/movies_m2t_224x224/single_video_all_observers/{mode}'
    elif data_partition == 'single_video':
        _DATA_PATH = f'data/GazeCom/movies_m2t_224x224/single_video/{mode}'
    elif data_partition == 'single_clip':
        _DATA_PATH = f'data/GazeCom/movies_m2t_224x224/single_clip/{mode}'

    def get_dataset():
        return gaze_labeled_video_dataset(
            data_path=_DATA_PATH,
            #clip_sampler=make_clip_sampler("uniform", clip_duration),
            #video_sampler=torch.utils.data.SequentialSampler,
            clip_sampler=make_clip_sampler("random", clip_duration),
            video_sampler=torch.utils.data.RandomSampler,
            transform=VAL_TRANSFORM,
            #transform=None,
            video_file_suffix='',
            decode_audio=False,
            decoder="pyav",
        )
    dataset = get_dataset()

    model = GazePredictionLightningModule.load_from_checkpoint(checkpoint_path).to(device=device)
    """
    model = GazePredictionLightningModule(lr=1e-6, batch_size=16, frames=round(clip_duration * 29.97),
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

    if calc_metrics:
        df_nss = pd.DataFrame(columns=['video', 'observer', 'first_frame', 'last_frame', 'nss_orig', 'nss_pred', 'nss_middle', 'nss_rnd'])
        df_change_dist = pd.DataFrame(columns=['video', 'observer', 'first_frame', 'last_frame', 'change_distance_wasserstein', 'change_orientation_wasserstein'])
        changes_dist, changes_deg = [], []
        changes_dist_pred, changes_deg_pred = [], []

    # Sample n times from model. For metrics calculation increased number of samples is taken.
    samples_per_clip = 5 if not calc_metrics else 1
    samples = 2 if not calc_metrics else 100
    pbar = tqdm(range(0, samples))
    for i in pbar:
        try:
            if calc_metrics or data_partition == 'single_clip' or (data_partition == 'single_video_all_observers' and mode == 'val'):
                sample = next(dataset)
            elif mode == 'train':
                sample = dataset.get_clip('golf', 'AAW', clip_start=3. + i * clip_duration)
            else:
                sample = dataset.get_clip('doves', 'AAW', clip_start=3. + i * clip_duration)
        except Exception:
            dataset = get_dataset()
            continue

        video_name = sample['video_name']
        observer = sample['observer']
        pbar.set_description(f"{video_name} {observer}")
        #print(video_name, observer)

        # Sample multiple scanpaths from same input to late average over metrics
        y = sample['frame_labels']
        if teacher_forcing_on_inference:
            y_teacher = y[None, :].to(device=device)
        else:
            y_teacher = None
        y_hats = [model(sample['video'][None, :].to(device=device), y=y_teacher)[0].cpu().detach().numpy() for i in range(samples_per_clip)]

        # (C, F, H, W) -> (F, H, W, C)
        frames = torch.swapaxes(sample['video'].T, 0, 2)

        frames = frames.cpu().detach().numpy()
        frames = np.interp(frames, (frames.min(), frames.max()), (0, 255)).astype('uint8')
        em_data_hats = None
        em_data = sample['em_data'].cpu().detach().numpy()
        frame_indices = np.array(sample['frame_indices'])
        if data_partition == 'single_clip' and mode == 'train':
            frame_indices += 115
        y = y.cpu().detach().numpy()
        #print(f"Frames {frame_indices[0]}-{frame_indices[-1]}")

        #print("y_hat.shape", y_hats[0].shape)
        if y_hats[0].shape[1] > 2:
            em_data_hats = [y_hat[:, 2:] for y_hat in y_hats]
            y_hats = [y_hat[:, :2] for y_hat in y_hats]
            em_data_hats = [em_encoder.inverse_transform(em_data_hat).reshape((-1)) for em_data_hat in em_data_hats]

        if em_encoder:
            em_data = em_encoder.inverse_transform(em_data).reshape((-1))
        else:
            em_data = em_data

        y_hats = [(y_hat + 1) * 112 for y_hat in y_hats]
        y = (y + 1) * 112

        '''
        print("y_hat")
        print(y_hats[0][:5])
        print("y")
        print(y[:5])
        if em_data_hats is not None:
            print("em_data_hat")
            print(em_data_hats[0][:20])
            print("em_data")
            print(em_data[:20])
        '''

        # Calculate Normalized Scanpath Saliency (NSS) for ground truth, prediction and random/center baselines
        nss_orig = metrics_nss.score_gaussian_density(video_name, y.astype(int), frame_ids=frame_indices)
        nss_scores = [metrics_nss.score_gaussian_density(video_name, y_hat.astype(int), frame_ids=frame_indices) for y_hat in y_hats]
        nss = np.array(nss_scores).mean()
        gaze_mid = np.ones(y.shape, dtype=np.int32) * 112
        nss_mid = metrics_nss.score_gaussian_density(video_name, gaze_mid, frame_ids=frame_indices)
        gaze_rnd = np.random.randint(225, size=y.shape, dtype=np.int32)
        nss_rnd = metrics_nss.score_gaussian_density(video_name, gaze_rnd, frame_ids=frame_indices)
        #print("NSS original clip:", nss_orig)
        #print("NSS prediction:", nss, "all scores:", nss_scores)
        #print("NSS middle baseline:", nss_mid)
        #print("NSS random baseline:", nss_rnd, "\n")

        # Calculate similarity in gaze change orientation and length distribution
        change_len, change_deg = utils.get_gaze_change_dist_and_orientation(y, absolute_values=True, normalize_gaze=True)
        change_data_hat = [utils.get_gaze_change_dist_and_orientation(y_hat, absolute_values=True, normalize_gaze=True) for y_hat in y_hats]
        change_len_similarity = np.mean([metrics.calc_wasserstein_distance(change_len, change_len_hat) for change_len_hat, _ in change_data_hat])
        change_deg_similarity = np.mean([metrics.calc_wasserstein_distance(change_deg, change_deg_hat) for _, change_deg_hat in change_data_hat])
        #print("Gaze change distance distribution Wasserstein-distance to truth:", change_len_similarity)
        #print("Gaze change orientation distribution Wasserstein-distance to truth:", change_deg_similarity)

        if calc_metrics:
            df_nss = df_nss.append({'video': video_name, 'observer': observer, 'first_frame': frame_indices[0], 'last_frame': frame_indices[-1],
                                'nss_orig': nss_orig, 'nss_pred': nss, 'nss_middle': nss_mid, 'nss_rnd': nss_rnd}, ignore_index=True)
            df_change_dist = df_change_dist.append({'video': video_name, 'observer': observer, 'first_frame': frame_indices[0], 'last_frame': frame_indices[-1],
                                'change_distance_wasserstein': change_len_similarity, 'change_orientation_wasserstein': change_deg_similarity}, ignore_index=True)
            changes_dist.append(change_len)
            changes_deg.append(change_deg)
            changes_dist_pred.append(change_data_hat[0][0])
            changes_deg_pred.append(change_data_hat[0][1])
            continue    # Skip visualization as video if only metrics are calculated

        #y_hats = np.stack(y_hats, axis=1)
        if em_data_hats is not None:
            em_data_hats = np.stack(em_data_hats, axis=1)

        save_dir = None if output_dir is None else f'{output_dir}/{i}'
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        if show_saliency:
            nss_calc = metrics_nss.NSSCalculator()
            nss_calc.load_gaussian_density(os.path.join('metrics', 'gaussian_density', f'{video_name}.npy'))
            density = nss_calc.gaussian_density[frame_indices[0]:frame_indices[-1] + 1, :, :]

            #nss_calc.save_animated_gaussian_density(os.path.join(output_dir, f'{mode}_{i}_{video_name}_orig.png'), animate=False, gaze_data=y)
            nss_calc.save_animated_gaussian_density(os.path.join(output_dir, f'{mode}_{i}_{video_name}_orig.png'), animate=False, frame_start=frame_indices[0], frame_end=frame_indices[-1])
            nss_calc.save_animated_gaussian_density(os.path.join(output_dir, f'{mode}_{i}_{video_name}.png'), animate=False, gaze_data=y_hats)

            density = np.swapaxes(density, 1, 2)
            color_overlay = (plt.cm.viridis(density) * 255)[:, :, :, :3]
            frames = (frames.astype(float) * 0.7 + color_overlay * 0.3).astype(int)

        # Visualize predictions over video and in comparison with groundtruth
        utils.plot_frames_with_labels(frames, y, em_data, np.stack(y_hats, axis=1), em_data_hats, box_width=8, save_to_directory=save_dir)
        utils.create_movie_from_frames(output_dir, str(i), f"{mode}_{i}_{video_name}.mp4", fps=10, width_px=1800, remote_machine=True,
                                    delete_frames=True)

        # Save metrics and metadata
        with open(os.path.join(output_dir, "metadata.txt"), "a") as f:
            f.write(f"{mode}_{i}: {video_name}+{observer}, Frames {frame_indices[0]}-{frame_indices[-1]}, nss (original, prediction, middle): ({nss_orig:.2f}, {nss:.2f}, {nss_mid:.2f}), "
                    f"change_dist_similarity: {change_len_similarity:.3f}, change_deg_similarity: {change_deg_similarity:.3f}\n")

        if plot_gaze_change_histograms:
            utils.plot_gaze_change_dist_and_orientation(change_len, change_deg, f"{output_dir}/{mode}_{i}_change_similarity", use_plotly=True)
            utils.plot_gaze_change_dist_and_orientation(*change_data_hat[0], f"{output_dir}/{mode}_{i}_change_similarity_pred", use_plotly=True)

    if calc_metrics:
        diff_to_orig = df_nss.nss_orig - df_nss.nss_pred
        diff_to_mid = df_nss.nss_middle - df_nss.nss_pred
        diff_to_rnd = df_nss.nss_rnd - df_nss.nss_pred
        print(f"\nNSS prediction: {df_nss.nss_pred.mean():.2f}+-{df_nss.nss_pred.std():.2f}")
        print(f"\nNSS (original - prediction): {diff_to_orig.mean():.2f}+-{diff_to_orig.std():.2f}")
        print(f"NSS (middle - prediction): {diff_to_mid.mean():.2f}+-{diff_to_mid.std():.2f}")
        print(f"NSS (random - prediction): {diff_to_rnd.mean():.2f}+-{diff_to_rnd.std():.2f}")
        os.makedirs(output_dir, exist_ok=True)
        df_nss.to_csv(os.path.join(output_dir, f'{mode}_nss.csv'), index=False)

        all_changes_dist = np.concatenate(changes_dist)
        all_changes_dist_pred = np.concatenate(changes_dist_pred)
        all_changes_deg = np.concatenate(changes_deg)
        all_changes_deg_pred = np.concatenate(changes_deg_pred)
        df_all_changes = pd.DataFrame(dict(
            change_dist=all_changes_dist,
            change_dist_pred=all_changes_dist_pred,
            change_deg=all_changes_deg,
            change_deg_pred=all_changes_deg_pred
        ))
        df_all_changes.to_csv(os.path.join(output_dir, f'{mode}_all_changes.csv'), index=False)

        change_len_similarity = metrics.calc_wasserstein_distance(all_changes_dist, all_changes_dist_pred)
        change_deg_similarity = metrics.calc_wasserstein_distance(all_changes_deg, all_changes_deg_pred)
        utils.plot_gaze_change_dist_and_orientation(all_changes_dist, all_changes_deg, f"{output_dir}/{mode}_all_change_similarity", use_plotly=True)
        utils.plot_gaze_change_dist_and_orientation(all_changes_dist_pred,all_changes_deg_pred, f"{output_dir}/{mode}_all_change_similarity_pred", use_plotly=True)
        print(f"Wasserstein distance for change distance distribution (truth vs prediction): {change_len_similarity:.6f}")
        print(f"Wasserstein distance for change orientation distribution (truth vs prediction): {change_deg_similarity:.6f}")
        change_len_similarity_no_change = metrics.calc_wasserstein_distance(all_changes_dist, [0])
        change_deg_similarity_uniform = metrics.calc_wasserstein_distance(all_changes_deg, np.arange(360))
        print(f"Wasserstein distance for change distance distribution (truth vs no change): {change_len_similarity_no_change:.6f}")
        print(f"Wasserstein distance for change orientation distribution (truth vs uniform orientation): {change_deg_similarity_uniform:.6f}")
        df_change_dist = df_change_dist.append({'video': 'mean', 'change_distance_wasserstein': change_len_similarity, 'change_orientation_wasserstein': change_deg_similarity}, ignore_index=True)
        df_change_dist = df_change_dist.append({'video': 'baseline', 'change_distance_wasserstein': change_len_similarity_no_change, 'change_orientation_wasserstein': change_deg_similarity_uniform}, ignore_index=True)
        df_change_dist.to_csv(os.path.join(output_dir, f'{mode}_change_metrics.csv'), index=False)


if __name__ == '__main__':
    _CHECKPOINT_PATH = r'data/lightning_logs/version_534/checkpoints/epoch=149-step=149.ckpt'
    _OUTPUT_DIR = r'data/sample_outputs/version_534'
    _DATA_PARTITION = 'all_videos_single_observer'
    _CLIP_DURATION = 5

    _MODE = 'train'
    _CALC_METRICS = False

    _SHOW_SALIENCY = True
    _PLOT_GAZE_CHANGE_HISTOGRAMS = True
    _TEACHER_FORCING_ON_INFERENCE = True

    # Sample model with input from data partition and save results to _OUTPUT_DIR
    sample_model(_CHECKPOINT_PATH, _OUTPUT_DIR, _DATA_PARTITION, _CLIP_DURATION, _MODE, _CALC_METRICS, _SHOW_SALIENCY,
                 _PLOT_GAZE_CHANGE_HISTOGRAMS, _TEACHER_FORCING_ON_INFERENCE)
