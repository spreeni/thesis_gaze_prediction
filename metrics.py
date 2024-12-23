"""
Similarity metric implementations between two distributions. This is used to quantify the similarity between
distributions in frame-wise gaze change distance and direction.
"""
from typing import Optional

import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import wasserstein_distance


def calc_wasserstein_distance(x1, x2) -> float:
    """
    Calculates the Wasserstein, or "earth-movers" distance between the distributions of values of two arrays.

    Args:
        x1: Unstructured array of values
        x2: Unstructured array of values

    Returns:
        Distance as float, where 0 is a perfect match. 1 means that each element would have to be moved by 1 on average.
    """
    return wasserstein_distance(x1, x2)


def calc_similarity(x1, x2, metric) -> Optional[float]:
    """
    Calculates a distance between two structured distributions (histogram bin counts).

    Args:
        x1:     Histogram bin counts
        x2:     Histogram bin counts
        metric: Name of metric, can be one of ['histogram_intersection', 'canberra', 'manhattan',
                'cosine_dist', 'euclidean']

    Returns:
        Distance as float
    """
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


def calc_similarity_gaze_change_distance(change_len1, change_len2, metric='histogram_intersection', nbins=20) -> Optional[float]:
    """
    Calculates similarity between the distributions of two measurements of gaze change distances.

    Args:
        change_len1:    Unstructured array of gaze change distances
        change_len2:    Unstructured array of gaze change distances
        metric:         Name of metric to use
        nbins:          Number of bins

    Returns:
        Distance as float
    """
    counts_len1, _ = np.histogram(change_len1, bins=np.linspace(0, 2., nbins))
    counts_len2, _ = np.histogram(change_len2, bins=np.linspace(0, 2., nbins))

    # Normalize to probability density
    counts_len1 = counts_len1 / len(change_len1)
    counts_len2 = counts_len2 / len(change_len2)

    return calc_similarity(counts_len1, counts_len2, metric)


def calc_similarity_gaze_change_orientation(change_deg1, change_deg2, metric='histogram_intersection', nbins=8) -> Optional[float]:
    """
        Calculates similarity between the distributions of two measurements of gaze change directions.

        Args:
            change_deg1:    Unstructured array of gaze change directions
            change_deg2:    Unstructured array of gaze change directions
            metric:         Name of metric to use
            nbins:          Number of bins

        Returns:
            Distance as float
        """
    counts_deg1, _ = np.histogram(change_deg1, bins=np.linspace(0, 360, nbins))
    counts_deg2, _ = np.histogram(change_deg2, bins=np.linspace(0, 360, nbins))

    # Normalize to probability density
    counts_deg1 = counts_deg1 / len(change_deg1)
    counts_deg2 = counts_deg2 / len(change_deg2)

    return calc_similarity(counts_deg1, counts_deg2, metric)
