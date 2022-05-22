import os
import sys

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
import pickle
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

import utils


SIGMA_X = 1.2  # deg
SIGMA_Y = 1.2  # deg
SIGMA_T = 26.25  # ms

#WIDTH_PX    = 1280
WIDTH_PX    = 224
#HEIGHT_PX   = 720
HEIGHT_PX   = 224
WIDTH_MM    = 400
HEIGHT_MM   = 225
DIST_MM     = 450

T_FRAME = 1/29.7 * 1000  # ms


class NSSCalculator:
    SIGMA_X = 1.2  # deg
    SIGMA_Y = 1.2  # deg
    SIGMA_T = 26.25  # ms

    def __init__(self):
        self.coeff = SIGMA_X**2 + SIGMA_Y**2 + SIGMA_T**2
        self.root = None
        self.vid_name = None
        self.observer_data = dict()
        self.kde = None
        self.gaussian_density = None
        self.score_mean = None
        self.score_std = None
        self.n_frames = 0

    def get_observer_data(self, vid_name, root):
        """
        Fetches gaze label data for all observers from the root directory
        """
        self.root = root
        self.vid_name = vid_name

        video_path = os.path.join(self.root, self.vid_name)
        n_frames = 0
        for root, dirs, files in os.walk(video_path):
            for filename in files:
                try:
                    observer = filename.split('_')[0]
                except Exception:
                    print(f"Encountered unexpected filename '{filename}'. Label files should have format <observer>_<video>.txt")
                    continue
                gaze, em_data = utils.read_label_file(os.path.join(video_path, filename), with_video_name=False)
                gaze = np.array(gaze).astype('int')
                em_data = np.array(em_data).astype('int')
                times = np.arange(len(gaze)) * T_FRAME
                if len(gaze) > n_frames:
                    n_frames = len(gaze)

                gaze_deg = np.array(utils.px_to_visual_angle(gaze[:, 0], gaze[:, 1], WIDTH_PX, HEIGHT_PX, WIDTH_MM, HEIGHT_MM, DIST_MM)).T

                self.observer_data[observer] = [times, gaze, gaze_deg, em_data]

            break  # Only look at immediate directory content
        self.n_frames = n_frames

    def get_stacked_observer_data(self, gaze_in_px=False):
        """
        Stacks gaze data for all observers in one array.

        Returns:
            Tuple: stacked time and gaze data, stacked eye movement classification data
        """
        assert len(self.observer_data) > 0, "No observer data loaded yet."

        if gaze_in_px:
            time_gaze = np.vstack([np.column_stack(self.observer_data[obs][:2]) for obs in self.observer_data])
        else:
            time_gaze = np.vstack([np.column_stack([self.observer_data[obs][0], self.observer_data[obs][2]]) for obs in self.observer_data])
        em_phases = np.concatenate([self.observer_data[obs][-1] for obs in self.observer_data])

        return time_gaze, em_phases

    @staticmethod
    def _calc_normalized_gaussian_density(gaze_px, n_frames):
        """
        Calculate discrete normalized (mean=0, std=1) gaussian density map for every pixel for every frame for given gaze data.

        Args:
            gaze_px:    Gaze data, array of shape (k, t, 2)
            n_frames:   Number of total frames

        Returns:

        """
        visual_angle_range_x = 2 * np.arctan(WIDTH_MM / 2. / DIST_MM) * 180 / np.pi
        visual_angle_range_y = 2 * np.arctan(HEIGHT_MM / 2. / DIST_MM) * 180 / np.pi
        sigmas = [SIGMA_T / T_FRAME, SIGMA_X * WIDTH_PX / visual_angle_range_x,
                  SIGMA_Y * HEIGHT_PX / visual_angle_range_y]

        point_map = np.zeros((n_frames, WIDTH_PX, HEIGHT_PX))
        if type(gaze_px) != list:
            gaze_px = [gaze_px]
        for gaze in gaze_px:
            for frame in range(len(gaze)):
                x, y = gaze[frame].astype(int).tolist()
                x = WIDTH_PX if x > WIDTH_PX else (x if x > 0 else 1)
                y = HEIGHT_PX if y > HEIGHT_PX else (y if y > 0 else 1)
                point_map[frame, x - 1, y - 1] += 1

        gaussian_density = gaussian_filter(point_map, sigmas)
        gaussian_density = (gaussian_density - gaussian_density.mean()) / gaussian_density.std()
        return gaussian_density

    def create_gaussian_density(self, export_path=None):
        """
        Calculate discrete gaussian density map for every pixel for every frame.

        Args:
            export_path:    In case the map shall be exported as a .npy file
        """
        assert len(self.observer_data) > 0, "No observer data loaded yet."

        gaze_px = [self.observer_data[obs][1] for obs in self.observer_data]
        self.gaussian_density = self._calc_normalized_gaussian_density(gaze_px, self.n_frames)

        if export_path is not None:
            np.save(export_path, self.gaussian_density)

    def load_gaussian_density(self, filepath):
        """
        Load discrete gaussian density map from .npy file.

        Args:
            filepath:   Filepath of .npy file
        """
        self.gaussian_density = np.load(filepath)

    def score_gaussian_density(self, gaze, frame_ids=None):
        """
        Calculate score for a scanpath on gaussian density map. Scores >0 show higher correlation, scores <0 show randomness.

        Args:
            gaze:       Gaze as pixel values
            frame_ids:  Frames that the gaze corresponds to. If omitted will assume that frame are from start of the video.

        Returns:
            The normalized scanpath saliency on the current clip.
        """
        assert self.gaussian_density is not None, "No observer data loaded yet."

        n_samples = len(gaze)
        if frame_ids is None:
            frame_ids = np.arange(n_samples)

        score = 0.
        gaze -= 1
        gaze[gaze < 0] = 0
        gaze[gaze[:, 0] > WIDTH_PX - 1, 0] = WIDTH_PX - 1
        gaze[gaze[:, 1] > HEIGHT_PX - 1, 1] = HEIGHT_PX - 1
        for i in range(len(gaze)):
            score += self.gaussian_density[frame_ids[i], gaze[i, 0], gaze[i, 1]]

        return score / n_samples

    def save_animated_gaussian_density(self, outpath, frame_start=0, frame_end=None, fps=29.97, animate=True, gaze_data=None):
        """
        Animate gaussian density per frame and save to a video file. Alternatively save mean over timeframe as picture.

        Args:
            outpath:        Output file
            frame_start:    (Optional) Starting frame, default is first frame
            frame_end:      (Optional) End frame, default is last frame
            fps:            (Optional) FPS of video, default is 29.97 (GazeCom)
            animate:        (Optional) If True, results get animated for each frame; if False output mean over timeframe as picture
            gaze_data:      (Optional) Gaze data to calculate density map from, default is density map from groundtruth

        Returns:

        """
        assert (self.gaussian_density is not None) or (gaze_data is not None), "No gaze data loaded yet."
        assert (frame_end is None) or (frame_end > frame_start), "End frame index needs to be larger then start frame index"
        if gaze_data is None:
            assert (frame_end is None) or ((frame_end >= 0) and (frame_end < self.gaussian_density.shape[0])), "End frame out of bounds"
            assert (frame_start >= 0) and (frame_start < self.gaussian_density.shape[0]), "Start frame out of bounds"
            density = self.gaussian_density
        else:
            if type(gaze_data) == list:
                n_frames = gaze_data[0].shape[-2]
            else:
                n_frames = gaze_data.shape[-2]
            assert (frame_end is None) or ((frame_end >= 0) and (frame_end < n_frames)), "End frame out of bounds"
            assert (frame_start >= 0) and (frame_start < n_frames), "Start frame out of bounds"
            density = self._calc_normalized_gaussian_density(gaze_data, n_frames)

        if frame_end is None:
            frame_end = density.shape[0] - 1

        fig = plt.figure(figsize=(10, 10))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        if animate:
            im = ax.imshow(density[frame_start, :, :], interpolation='none')

            # animation function.  This is called sequentially
            def animate(i):
                im.set_array(density[i, :, :])
                return [im]

            # call the animator.  blit=True means only re-draw the parts that have changed.
            anim = animation.FuncAnimation(fig, animate, frames=range(frame_start, frame_end+1),
                                           interval=1000/fps, blit=True, repeat=False)

            # save the animation as an mp4.  This requires ffmpeg or mencoder to be
            # installed.  The extra_args ensure that the x264 codec is used, so that
            # the video can be embedded in html5.  You may need to adjust this for
            # your system: for more information, see
            # http://matplotlib.sourceforge.net/api/animation_api.html
            anim.save(outpath, fps=round(fps))#, extra_args=['-vcodec', 'libx264'])
        else:
            im = ax.imshow(density[frame_start:frame_end+1, :, :].mean(axis=0), interpolation='none')
            fig.savefig(outpath, dpi=300)
        plt.close(fig)

    def fit_kde(self):
        """
        Fits a Kernel Density Estimator on time and gaze label data, then finds normalization parameters.
        This follows the process of https://jov.arvojournals.org/article.aspx?articleid=2121333

        Returns:
            The trained Kernel Density Estimator (Use this class for scores if you want normalization though)
        """
        assert len(self.observer_data) > 0, "No observer data loaded yet."

        self.kde = KernelDensity(kernel='gaussian', bandwidth=np.sqrt(self.coeff))

        time_gaze, em_phases = self.get_stacked_observer_data()
        non_saccades = em_phases != 2
        self.kde.fit(time_gaze[non_saccades, :])
        self._set_normalization_params()
        return self.kde

    def _set_normalization_params(self):
        """
        Finds the mean and std of the trained KDE by sampling over the complete input space.

        Here the resolution (n_samples) has a huge impact on the runtime (O = n_samples³)
        """
        assert self.kde is not None, "No Kernel Density Estimator trained yet."

        x_lims, y_lims = utils.px_to_visual_angle(np.array([0, WIDTH_PX - 1]), np.array([0, HEIGHT_PX - 1]), WIDTH_PX,
                                                  HEIGHT_PX, WIDTH_MM, HEIGHT_MM, DIST_MM)
        time_gaze, _ = self.get_stacked_observer_data()

        n_samples = 30
        x = np.linspace(x_lims.min(), x_lims.max(), n_samples)
        y = np.linspace(y_lims.min(), y_lims.max(), n_samples)
        t = np.linspace(time_gaze[:, 0].min(), time_gaze[:, 0].max(), n_samples)
        sample_vals = np.array(np.meshgrid(t, x, y)).T.reshape(-1, 3)

        scores = np.exp(self.kde.score_samples(sample_vals))
        self.score_mean = scores.mean()
        self.score_std = scores.std()
        print(f"{self.vid_name}, mean: {self.score_mean}, std: {self.score_std}")

    def _normalize_scores(self, scores):
        """
        Normalize scores by substracting the mean and dividing by the std.

        Args:
            scores: An array or scalar of NSS scores

        Returns:
            Normalized scores
        """
        assert self.score_mean is not None and self.score_std is not None,\
            "Normalization parameters have not been calculated yet."
        return (scores - self.score_mean) / self.score_std

    def score_kde(self, gaze, t, score_each_sample=False):
        """
        Calculate NSS scores for given data on the trained label data.

        Args:
            gaze:               gaze locatings in degrees of visual angle
            t:                  timesteps in ms
            score_each_sample:  Flag to get a score for each individual sample vs one score for all

        Returns:
            NSS score for the given data on the trained label data
        """
        assert self.kde is not None, "No Kernel Density Estimator trained yet."

        X = np.column_stack([t, gaze])
        if not score_each_sample:
            return self._normalize_scores(np.exp(self.kde.score(X)))
        else:
            return self._normalize_scores(np.exp(self.kde.score_samples(X)))

    def save_to_file(self, filepath):
        assert len(self.observer_data) > 0, "No observer data loaded yet."
        assert self.kde is not None, "No Kernel Density Estimator trained yet."

        with open(filepath, "wb") as output_file:
            pickle.dump(self, output_file)

    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, "rb") as input_file:
            return pickle.load(input_file)
    """
    def save_trained_kde(self, filepath):
        assert self.kde is not None, "No Kernel Density Estimator trained yet."

        with open(filepath, "wb") as output_file:
            pickle.dump(self.kde, output_file)

    def load_trained_kde(self, filepath):
        with open(filepath, "rb") as input_file:
            self.kde = pickle.load(input_file)
    """


def train_kde_on_all_vids(rootdir):
    for root, dirs, files in os.walk(rootdir):
        pbar = tqdm(dirs)
        for video_name in pbar:
            pbar.set_description("Processing '%s'" % video_name)
            nss = NSSCalculator()
            nss.get_observer_data(video_name, rootdir)
            nss.fit_kde()
            nss.save_to_file(os.path.join('metrics', 'kernel_density_estimator', f'{video_name}.pickle'))
        break  # Only look at immediate directory content


def train_gaussian_density_on_all_vids(rootdir):
    for root, dirs, files in os.walk(rootdir):
        pbar = tqdm(dirs)
        for video_name in pbar:
            pbar.set_description("Processing '%s'" % video_name)
            nss = NSSCalculator()
            nss.get_observer_data(video_name, rootdir)
            nss.create_gaussian_density(export_path=os.path.join('metrics', 'gaussian_density', f'{video_name}.npy'))
        break  # Only look at immediate directory content


def animate_saliency_on_all_vids(rootdir):
    for root, dirs, files in os.walk(rootdir):
        pbar = tqdm(dirs)
        for video_name in pbar:
            pbar.set_description("Processing '%s'" % video_name)
            nss = NSSCalculator()
            nss.load_gaussian_density(os.path.join('metrics', 'gaussian_density', f'{video_name}.npy'))
            nss.save_animated_gaussian_density(os.path.join('plots', 'GazeCom', 'density_maps', f'{video_name}.mp4'), animate=True)
            nss.save_animated_gaussian_density(os.path.join('plots', 'GazeCom', 'density_maps', f'{video_name}.png'), animate=False)
        break  # Only look at immediate directory content


def score_gaussian_density(video, gaze, frame_ids=None):
    nss = NSSCalculator()
    nss.load_gaussian_density(os.path.join('metrics', 'gaussian_density', f'{video}.npy'))
    return nss.score_gaussian_density(gaze, frame_ids)


if __name__ == "__main__":
    root = 'data/GazeCom/deepEM_classifier/ground_truth_framewise'
    root = 'data/GazeCom/movies_m2t_224x224/label_data'
    #train_kde_on_all_vids(root)
    #train_gaussian_density_on_all_vids(root)
    #animate_saliency_on_all_vids(root)

    # Test vs random data
    nss = NSSCalculator()
    nss.load_gaussian_density(os.path.join('metrics', 'gaussian_density', 'doves.npy'))

    gaze, em_data = utils.read_label_file(os.path.join(root, 'doves', 'AAW_doves.txt'), with_video_name=False)
    gaze = np.array(gaze).astype('int')
    em_data = np.array(em_data).astype('int')
    times = np.arange(len(gaze)) * T_FRAME

    gaze_deg = np.array(utils.px_to_visual_angle(gaze[:, 0], gaze[:, 1], WIDTH_PX, HEIGHT_PX, WIDTH_MM, HEIGHT_MM, DIST_MM)).T
    g_rand = np.random.randint(0, 224, gaze.shape)
    g_mid = np.ones(gaze.shape, dtype=np.int32) * 112
    t_rand = np.random.randn(*times.shape)
    #print("score:", nss.score_kde(gaze_deg, times))
    #print("score random:", nss.score_kde(g_rand, t_rand))
    print("score:", nss.score_gaussian_density(gaze))
    print("score random:", nss.score_gaussian_density(g_rand))
    print("score mid:", nss.score_gaussian_density(g_mid))
