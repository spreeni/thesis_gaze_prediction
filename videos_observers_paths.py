# Modified version of Facebook's
# https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/labeled_video_paths.py

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import os
import pathlib
from typing import List, Optional, Tuple

from iopath.common.file_io import g_pathmgr


class VideosObserversPaths:
    """
    VideosObserversPaths contains pairs of video path and an observer.
    """

    @classmethod
    def from_path(cls, data_path: str) -> VideosObserversPaths:
        """
        Factory function that creates a VideosObserversPaths object depending on the path
        type.
        - If it is a directory path it uses the VideosObserversPaths.from_directory function.
        - If it's a file it uses the VideosObserversPaths.from_csv file.
        Args:
            file_path (str): The path to the file to be read.
        """

        if g_pathmgr.isfile(data_path):
            return VideosObserversPaths.from_csv(data_path)
        elif g_pathmgr.isdir(data_path):
            return VideosObserversPaths.from_directory(data_path)
        else:
            raise FileNotFoundError(f"{data_path} not found.")

    @classmethod
    def from_csv(cls, file_path: str) -> VideosObserversPaths:
        """
        Factory function that creates a VideosObserversPaths object by reading a file with the
        following format:
            <path> <integer_label>
            ...
            <path> <integer_label>

        Args:
            file_path (str): The path to the file to be read.
        """
        assert g_pathmgr.exists(file_path), f"{file_path} not found."
        videos_and_observers = []
        with g_pathmgr.open(file_path, "r") as f:
            for path_label in f.read().splitlines():
                line_split = path_label.rsplit(None, 1)

                assert len(line_split) == 2, f"{file_path}: Unexpected format."

                video_name, observer = line_split

                videos_and_observers.append((video_name, observer))

        assert (
            len(videos_and_observers) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(videos_and_observers)

    @classmethod
    def from_directory(cls, dir_path: str) -> VideosObserversPaths:
        """
        Factory function that creates a VideosObserversPaths object by parsing the structure
        of the given directory's subdirectories. It expects the directory label format to be
        the following:
             dir_path/label_data/<video_name>/<observer_name>_<video_name>.txt
             dir_path/video_data/<video_name>                 # Video file
             dir_path/video_data/<video_name>/frame_0001.jpg  # Video as individual frames

        Args:
            dir_path (str): Root directory to the video and label data directories .
        """
        label_path = os.path.join(dir_path, 'label_data')
        assert g_pathmgr.exists(label_path), f"{label_path} not found."

        # Find all video names based on directory names. Then the available observers for a video
        # are inferred from the label file names.
        # TODO: Add file extensions in case of video files?
        videos_and_observers: list[tuple[str, str]] = []

        video_names: list[str] = sorted(
            (f.name for f in pathlib.Path(label_path).iterdir() if f.is_dir())
        )

        for video_name in video_names:
            observers = sorted((f.name.replace(f'_{video_name}.txt', '') for f in
                                pathlib.Path(os.path.join(label_path, video_name)).iterdir() if f.is_file()))

            videos_and_observers.extend([(video_name, observer) for observer in observers])

        assert (
            len(videos_and_observers) > 0
        ), f"Failed to load dataset from {label_path}."
        return cls(videos_and_observers, path_prefix=dir_path)

    def __init__(
        self, videos_and_observers: List[Tuple[str, str]], path_prefix=""
    ) -> None:
        """
        Args:
            videos_and_observers [(str, str)]: a list of tuples containing the video
                name and observer name.
        """
        self._videos_and_observers = videos_and_observers
        self._path_prefix = path_prefix

    def path_prefix(self, prefix):
        self._path_prefix = prefix

    path_prefix = property(None, path_prefix)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        """
        Args:
            index (int): the path and observer index.

        Returns:
            The path and observer tuple for the given index.
        """
        video_name, observer = self._videos_and_observers[index]

        video_path = os.path.join(self._path_prefix, 'video_data', video_name)
        label_path = os.path.join(self._path_prefix, 'label_data', video_name, f'{observer}_{video_name}.txt')

        return (video_path, label_path)

    def __len__(self) -> int:
        """
        Returns:
            The number of video paths and label pairs.
        """
        return len(self._videos_and_observers)