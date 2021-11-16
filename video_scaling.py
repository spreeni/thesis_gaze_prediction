import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pathlib import Path
import subprocess
import numpy as np
import numpy.lib.recfunctions as rfn

import utils

logger = logging.getLogger(__name__)


def resize_videos_and_labels(video_dir: str, label_dir: str, out_video_dir: str, out_label_dir: str, height: int,
                             width: Optional[int] = None):
    """
    Rescales video and label files to the given dimensions.

    If no width is passed, the aspect ratio is kept.

    Args:
        video_dir:      Input video folder - should only contain video files and directories
        label_dir:      Input label folder - should only contain label files and directories
        out_video_dir:  Output video folder - structure from input will be recreated
        out_label_dir:  Output label folder - structure from input will be recreated
        height:         New height of the video in pixel
        width:          (optional) New width of the video in pixel
    """
    old_width, old_height, new_width, new_height = (None, None, None, None)

    # iterate through videos
    root_video = Path(video_dir)
    root_video_out = Path(out_video_dir)
    for video_path in root_video.rglob('*'):
        if video_path.is_file():
            out_path = str(root_video_out.joinpath(video_path.parent.relative_to(root_video)).joinpath(video_path.name))
            _resize_video(str(video_path), out_path, height, width)

            if old_width is None:
                old_width, old_height = utils.get_video_dimensions(str(video_path))
                new_width, new_height = utils.get_video_dimensions(out_path)

    # iterate through labels
    root_label = Path(label_dir)
    root_label_out = Path(out_label_dir)
    for label_path in root_label.rglob('*.txt'):
        out_path = root_label_out.joinpath(label_path.parent.relative_to(root_label)).joinpath(label_path.name)
        _resize_label(str(label_path), str(out_path), old_width, old_height, new_width, new_height)


def _resize_video(video_path: str, out_path: str, height: int, width: Optional[int] = None):
    """
    Scales video file to a new size.

    Args:
        video_path: filepath of video file
        out_path:   filepath where the new scaled video file is written to
        height:     new video height in px
        width:      (optional) new video width in px
    """
    # Create output directories if not existent
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if width is None:
        width = -1      # keep aspect ratio
    subprocess.run(f"ffmpeg -i {video_path} -vf scale={width}:{height} {out_path}")


def _resize_label(label_path: str, out_path: str, old_width: int, old_height: int, new_width: int, new_height: int):
    """
    Rescales gaze positions in label file to the given dimensions.

    Args:
        label_path: filepath of label file
        out_path:   filepath where the new scaled label file is written to
        old_width:  old video width in px
        old_height: old video height in px
        new_width:  new video width in px
        new_height: new video height in px
    """
    # read label contents
    gaze_data, em_data = utils.read_label_file(label_path, with_video_name=True)
    gaze_data, em_data = np.array(gaze_data), np.array(em_data)

    # scale gaze locations
    gaze_data[:, 0] = gaze_data[:, 0] * new_width / old_width
    gaze_data[:, 1] = gaze_data[:, 1] * new_height / old_height

    # create new label array
    n_frames = len(em_data)
    arr = np.empty((n_frames, 4))
    arr[:, 0] = np.arange(n_frames)     # frame indices
    arr[:, 1] = em_data                 # em-data
    arr[:, 2:] = gaze_data              # gaze locations

    structured_dt = np.dtype([('frame', '<f8'), ('label', '<f8'), ('x_angle', '<f8'), ('y_angle', '<f8')])
    struct_arr = np.array(rfn.unstructured_to_structured(arr), dtype=structured_dt)

    # Create output directories if not existent
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_path, struct_arr, fmt=['%d', '%d', '%d', '%d'])
