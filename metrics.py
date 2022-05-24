import numpy as np
import scipy.spatial.distance as dist


def calc_similarity(x1, x2, metric):
    similarity = None
    if metric == 'histogram_intersection':
        similarity = np.minimum(x1, x2).sum() / x1.sum()
    elif metric == 'canberra':
        similarity = dist.canberra(x1, x2)
    elif metric == 'manhattan':
        similarity = dist.cityblock(x1, x2)
    elif metric == 'cosine_dist':
        similarity = dist.cosine(x1, x2)
    elif metric == 'euclidean':
        similarity = dist.euclidean(x1, x2)

    return similarity


def calc_similarity_gaze_change_distance(change_len1, change_len2, metric):
    counts_len1, _ = np.histogram(change_len1, bins=np.linspace(0, 2., 100))
    counts_len2, _ = np.histogram(change_len2, bins=np.linspace(0, 2., 100))

    # Normalize to probability density
    counts_len1 /= len(change_len1)
    counts_len2 /= len(change_len2)

    return calc_similarity(counts_len1, counts_len2, metric)


def calc_similarity_gaze_change_orientation(change_deg1, change_deg2, metric):
    counts_deg1, _ = np.histogram(change_deg1, bins=np.linspace(0, 360, 100))
    counts_deg2, _ = np.histogram(change_deg2, bins=np.linspace(0, 360, 100))

    # Normalize to probability density
    counts_deg1 /= len(change_deg1)
    counts_deg2 /= len(change_deg2)

    return calc_similarity(counts_deg1, counts_deg2, metric)
