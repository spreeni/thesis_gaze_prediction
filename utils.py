import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import os
from pathlib import Path
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy.io.arff import loadarff
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


EM_MAP = {
    0: 'UNKNOWN/NOISE',
    1: 'FIXATION',
    2: 'SACCADE',
    3: 'SMOOTH PURSUIT'
}

EM_COLOR_MAP = {
    0: 'white',
    1: 'yellow',
    2: 'red',
    3: 'blue'
}


def read_label_file(
        file_path: str,
        with_video_name: bool = False,
        with_EM_data: bool = True
) -> Tuple[List[Tuple[int, int]], Optional[List[int]]]:
    """
    Reads a GazeCom label file and returns a list of gaze data and EM-phase data

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


def plot_frames_with_labels(
        frames: np.ndarray,
        avg_gaze_locations: np.ndarray,
        avg_em_data: Optional[np.ndarray] = None,
        gaze_locations: Optional[List[List[Tuple]]] = None,
        em_data: Optional[List[List]] = None,
        fps=30.,
        box_width=25,
        show_time=True,
        display_speed=0.05,
        fig_width=12,
        save_to_directory=None
):
    """
    Visualizes video frames with bounding boxes for gaze labels.

    Can be used to plot averaged gaze data per frame together with raw gaze data.

    Note: Can also be used to plot gaze data with multiple predictions - then set groundtruth as average gaze data and predicted as raw gaze data.

    Args:
        frames:             Frames as array of shape (n_frames, height, width, channels)
        avg_gaze_locations: Averaged gaze locations per frame as array of shape (n_frames, 2)
        avg_em_data:        Averaged eye-movement classification data per frame as array of shape (n_frames)
        gaze_locations:     List of raw gaze locations (multiple per frame possible)
        em_data:            List of raw eye-movement classification data (multiple per frame possible)
        fps:                Frames per second im Video
        box_width:          Annotation box width
        show_time:          Toggle to display frame time in title
        display_speed:      Speed with which frames are displayed
        fig_width:          Width of plot figure
        save_to_directory:  Directory to which plots are to be saved to. If given, will not display plots
    """
    num_frames = frames.shape[0]
    if avg_em_data is not None:
        assert len(avg_gaze_locations) == len(
            avg_em_data), f"Number of gaze locations and eye data classification labels needs to be the same: {len(avg_gaze_locations)} != {len(avg_em_data)}."

    fig = plt.figure(figsize=(fig_width, fig_width*frames.shape[1]/frames.shape[2]))
    ax = plt.Axes(fig, [0., -0.05, 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    for i_frame in range(min(num_frames, len(avg_gaze_locations))):
        frame = frames[i_frame]
        avg_gaze = avg_gaze_locations[i_frame]

        # Plot frame
        ax.clear()
        ax.imshow(frame)

        # Plot averaged label
        color = EM_COLOR_MAP[avg_em_data[i_frame]] if avg_em_data is not None else 'r'
        avg_label_box = patches.Rectangle(avg_gaze - box_width / 2., box_width, box_width,
                                          linewidth=1.4, edgecolor='black', facecolor=color)
        ax.add_patch(avg_label_box)

        # Plot raw labels
        raw_box_width = round(0.5 * box_width)
        if gaze_locations is not None:
            assert num_frames == len(
                gaze_locations), f"Number of frames and lists of raw gaze locations needs to be the same."
            for i, gaze in enumerate(gaze_locations[i_frame]):
                if em_data is not None:
                    color = EM_COLOR_MAP[em_data[i_frame][i]]
                else:
                    color = plt.get_cmap('tab10').colors[i]
                label_box = patches.Rectangle(np.array(gaze) - raw_box_width / 2., raw_box_width, raw_box_width,
                                              linewidth=0.8, edgecolor='black', facecolor=color)#'none')
                ax.add_patch(label_box)

        # Update title
        title = f"Frame {i_frame} ({i_frame/fps:.2f}s)" if show_time else f"Frame {i_frame}"
        ax.set_title(title)

        # Update figure
        if not save_to_directory:
            plt.pause(1/fps/display_speed)
        else:
            plt.savefig(f'{save_to_directory}/{i_frame:03d}.png', dpi=300)


def get_video_frames_from_file(video_path: str) -> Tuple[np.ndarray, float]:
    """
    Retrieves video frames and FPS from video file

    Args:
        video_path: file path of the video file

    Returns:
        Tuple of frames as array of shape (n_frames, height, width, channels) and FPS
    """
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    success = True
    frames = []
    while success:
        success, image = vidcap.read()
        if success:
            frames.append(image)

    return np.array(frames), fps


def plot_gazecom_frames_with_labels(video_path: str, label_path: str, raw_label_path: str):
    """
    Visualizes video frames with bounding boxes for gaze labels for GazeCom dataset

    Args:
        video_path:     video file path (needs to be a file)
        label_path:     label file path with frame-wise labels
        raw_label_path: raw label file path with all labels
    """
    # Load frame data
    print("load video data")
    frames, fps = get_video_frames_from_file(video_path)

    # Load frame-wise averaged label data
    print("load frame-wise label data")
    avg_gaze, avg_em_data = read_label_file(label_path, with_video_name=False)
    avg_gaze = np.array(avg_gaze).astype('int')
    avg_em_data = np.array(avg_em_data).astype('int')

    # Load raw label data
    print("load raw label data")
    raw_label_arr, meta = loadarff(raw_label_path)

    raw_gaze = np.empty((len(raw_label_arr), 2))
    raw_gaze[:, 0] = raw_label_arr['x'].astype('int')
    raw_gaze[:, 1] = raw_label_arr['y'].astype('int')
    raw_em_data = raw_label_arr['handlabeller_final'].astype('int')
    raw_em_data[raw_em_data == 4] = 0

    # Collect raw label data per frame
    n_frames = frames.shape[0]
    f_label = 250.  # frequency eye tracker
    # n_entries_to_frame = len(raw_label_arr) / n_frames
    n_entries_to_frame = f_label / fps

    raw_gaze_per_frame = []
    raw_em_data_per_frame = []

    for i in range(n_frames):
        lbound = round(i * n_entries_to_frame)
        ubound = round((i + 1) * n_entries_to_frame)

        # take majority for EM classification
        raw_em_data_per_frame.append(raw_em_data[lbound:ubound].tolist())

        # take mean for gaze position
        raw_gaze_per_frame.append(raw_gaze[lbound:ubound].tolist())

    # Visualize labels on video data
    plot_frames_with_labels(frames, avg_gaze, avg_em_data=avg_em_data, gaze_locations=raw_gaze_per_frame,
                            em_data=raw_em_data_per_frame, fps=fps, display_speed=0.5)


def get_video_dimensions(filepath: str) -> Tuple[int, int]:
    """
    Returns video dimensions for video file.

    Args:
        filepath:   filepath of video file

    Returns:
        Integer tuple of (width_px, height_px)
    """
    vcap = cv2.VideoCapture(filepath)

    width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    return int(width), int(height)


def store_frames_to_png(video_path: str, export_dir: str):
    """
    Stores all frames from a video file as png-images in export_dir.

    Args:
        video_path:     filepath of video
        export_dir:     directory in which frames will be saved
    """
    # Create export_dir if not existant
    os.makedirs(export_dir, exist_ok=True)

    # Open video
    vidcap = cv2.VideoCapture(video_path)

    success, image = vidcap.read()
    frame = 0

    while success:
        # save frame as PNG file
        cv2.imwrite(os.path.join(export_dir, f"frame_{frame:05d}.png"), image)

        success, image = vidcap.read()
        frame += 1


def videos_to_frames(video_dir: str, out_frames_dir: str):
    """
    Extracts frames from video files in video_dir and saves them as png-images in out_frames_dir.

    Args:
        video_dir:          directory where video files are stored
        out_frames_dir:     directory where frames will be saved
    """
    root_video = Path(video_dir)
    root_frames = Path(out_frames_dir)
    for video_path in tqdm(root_video.rglob('*')):
        if video_path.is_file():
            if root_frames is not None:
                store_frames_to_png(str(video_path), str(root_frames.joinpath(video_path.stem)))


def px_to_visual_angle(x, y, width_px: int, height_px: int, width: int, height: int, dist: int, x_left_to_right: bool=True,
                       y_top_to_bottom: bool=True):
    """
    Converts (x,y) pixel values to visual angle values.

    A positive visual angle for the x-axis means the observer is looking to the right.
    A positive visual angle for the y-axis means the observer is looking upwards.

    (x, y, height_px, height_px) are pixel values
    (width, height, dist) are length measures (e.g. mm, needs to be same for all 3 values)
    """
    x_dist = (x / width_px - 0.5) * width
    y_dist = -(y / height_px - 0.5) * height

    if not x_left_to_right: x_dist *= -1
    if not y_top_to_bottom: y_dist *= -1

    x_angle = np.arctan(x_dist / dist) * 180 / np.pi
    y_angle = np.arctan(y_dist / dist) * 180 / np.pi

    return x_angle, y_angle


def px_to_visual_angle_in_structured_arr(arr, x_colname, y_colname, width_px, height_px, width, height, dist, x_left_to_right=True,
                       y_top_to_bottom=True):
    """
    Converts (x,y) pixel values to visual angle values within a structured array (Is the case for GazeCom arff files).

    A positive visual angle for the x-axis means the observer is looking to the right.
    A positive visual angle for the y-axis means the observer is looking upwards.

    (height_px, height_px) are pixel values
    (width, height, dist) are length measures (e.g. mm, needs to be same for all 3 values
    """
    x_angle, y_angle = px_to_visual_angle(arr[x_colname], arr[y_colname], width_px, height_px, width, height, dist, x_left_to_right=x_left_to_right,
                       y_top_to_bottom=y_top_to_bottom)

    arr_unstruct = rfn.structured_to_unstructured(arr)

    new_arr_unstruct = np.c_[arr_unstruct, x_angle]
    new_arr_unstruct = np.c_[new_arr_unstruct, y_angle]

    new_dt = np.dtype(arr.dtype.descr + [('x_angle', '<f8'), ('y_angle', '<f8')])

    return np.array(rfn.unstructured_to_structured(new_arr_unstruct), dtype=new_dt)
