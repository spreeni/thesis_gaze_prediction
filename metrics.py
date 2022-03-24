import os
import numpy as np
import pickle
from sklearn.neighbors import KernelDensity
from tqdm.auto import tqdm

import utils


SIGMA_X = 1.2  # deg
SIGMA_Y = 1.2  # deg
SIGMA_T = 26.25  # ms

WIDTH_PX    = 1280
HEIGHT_PX   = 720
WIDTH_MM    = 400
HEIGHT_MM   = 225
DIST_MM     = 450

T_FRAME = 1/29.7 * 1000  # ms


class NSSCalculator:
    SIGMA_X = 1.2  # deg
    SIGMA_Y = 1.2  # deg
    SIGMA_T = 26.25  # ms

    def __init__(self, vid_name, root):
        self.coeff = SIGMA_X**2 + SIGMA_Y**2 + SIGMA_T**2
        self.root = root
        self.vid_name = vid_name
        self.observer_data = dict()
        self.kde = None
        self.score_mean = None
        self.score_std = None

        self.get_observer_data()

    def get_observer_data(self):
        """
        Fetches gaze label data for all observers from the root directory
        """
        video_path = os.path.join(self.root, self.vid_name)
        for root, dirs, files in os.walk(video_path):
            for filename in files:
                try:
                    observer = filename.split('_')[0]
                    gaze, em_data = utils.read_label_file(os.path.join(video_path, filename), with_video_name=True)
                    gaze = np.array(gaze).astype('int')
                    em_data = np.array(em_data).astype('int')
                    times = np.arange(len(gaze)) * T_FRAME

                    gaze_deg = np.array(utils.px_to_visual_angle(gaze[:, 0], gaze[:, 1], WIDTH_PX, HEIGHT_PX, WIDTH_MM, HEIGHT_MM, DIST_MM)).T

                    self.observer_data[observer] = [times, gaze_deg, em_data]
                except Exception:
                    print(f"Encountered unexpected filename '{filename}'. Label files should have format <observer>_<video>.txt")
            break  # Only look at immediate directory content

    def get_stacked_observer_data(self):
        """
        Stacks gaze data for all observers in one array.

        Returns:
            Tuple: stacked time and gaze data, stacked eye movement classification data
        """
        assert len(self.observer_data) > 0, "No observer data loaded yet."

        time_gaze = np.vstack([np.column_stack(self.observer_data[obs][:2]) for obs in self.observer_data])
        em_phases = np.concatenate([self.observer_data[obs][-1] for obs in self.observer_data])

        return time_gaze, em_phases

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
        for video_name in tqdm(dirs):
            nss = NSSCalculator(video_name, rootdir)
            nss.fit_kde()
            nss.save_to_file(os.path.join('metrics', 'kernel_density_estimator', f'{video_name}.pickle'))
        break  # Only look at immediate directory content


root = 'data/GazeCom/deepEM_classifier/ground_truth_framewise'
#train_kde_on_all_vids(root)


# Test vs random data
#nss = NSSCalculator('doves', root)
#nss.fit_kde()
nss = NSSCalculator.load_from_file('metrics/kernel_density_estimator/doves.pickle')

gaze, em_data = utils.read_label_file(os.path.join(root, 'doves', 'AAW_doves.txt'), with_video_name=True)
gaze = np.array(gaze).astype('int')
em_data = np.array(em_data).astype('int')
times = np.arange(len(gaze)) * T_FRAME

gaze_deg = np.array(utils.px_to_visual_angle(gaze[:, 0], gaze[:, 1], WIDTH_PX, HEIGHT_PX, WIDTH_MM, HEIGHT_MM, DIST_MM)).T
g_rand = np.random.randn(*gaze_deg.shape)
t_rand = np.random.randn(*times.shape)
print("score:", nss.score_kde(gaze_deg, times))
print("score random:", nss.score_kde(g_rand, t_rand))
