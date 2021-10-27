import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import os
import numpy as np

logger = logging.getLogger(__name__)


def read_label_file(
        file_path: str,
        with_video_name: bool = False,
        with_EM_data: bool = True
) -> Tuple[List[Tuple[int, int]], Optional[List[int]]]:
    """
    Reads a label file and returns a list of gaze data and EM-phase data

    Args:
        file_path:          filepath to label file
        with_video_name:    flag if label file contains video name in first column
        with_EM_data:       flag if label file contains eye movement classification data in third column

    Returns:
        gaze data:      List of x,y gaze positions on screen
        EM-phase data:  List of EM-phases for each frame
    """
    assert os.path.isfile(file_path), f"{file_path} not found."

    dtype = []
    if with_video_name:
        dtype.append(('video', '<U30'))
    dtype.append(('frame', '<i8'))
    if with_EM_data:
        dtype.append(('EM_phase', '<i8'))
    dtype.extend([('x_gaze', '<i8'), ('y_gaze', '<i8')])

    labels = np.loadtxt(
        file_path,
        dtype=np.dtype(dtype)
    )

    em_data = labels['EM_phase'].tolist() if 'EM_phase' in labels.dtype.names else None
    return (labels[['x_gaze', 'y_gaze']].tolist(), em_data)


def get_observer_from_label_path(label_path: str) -> str:
    file_name = os.path.basename(label_path)
    assert len(file_name) > 0 and '_' in file_name, f"{label_path} is not a valid label-file path."

    return file_name.split('_')[0]
