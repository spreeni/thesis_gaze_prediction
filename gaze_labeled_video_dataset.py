# Modified version of Facebook's
# https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/labeled_video_dataset.py

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch.utils.data
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.data.utils import MultiProcessSampler
from sklearn.preprocessing import OneHotEncoder

from videos_observers_paths import VideosObserversPaths
import utils

logger = logging.getLogger(__name__)


class GazeLabeledVideoDataset(torch.utils.data.IterableDataset):
    """
    LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as either an encoded video
    (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(

            self,
            video_and_label_paths: VideosObserversPaths,
            clip_sampler: ClipSampler,
            video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
            transform: Optional[Callable[[dict], Any]] = None,
            decode_audio: bool = True,
            decoder: str = "pyav",
            predict_change: bool = False,
    ) -> None:
        """
        Args:
            video_and_label_paths (List[Tuple[str, str]]): List containing
                    combinations of video file paths and label file_paths. If video paths are a
                    folder it's interpreted as a frame video, otherwise it must be an encoded
                    video.

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            decode_audio (bool): If True, also decode audio from video.

            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.

            predict_change (bool): If True, predicted data is relative change and not absolute positions.
        """
        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._video_and_label_paths = video_and_label_paths
        self._decoder = decoder
        self._predict_change = predict_change

        self.em_encoder = OneHotEncoder()
        self.em_encoder.fit([[i] for i in range(4)])

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_observer_sampler = video_sampler(
                self._video_and_label_paths, generator=self._video_random_generator
            )
        else:
            self._video_observer_sampler = video_sampler(self._video_and_label_paths)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._next_clip_start_time = 0.0
        self.video_path_handler = VideoPathHandler()

        self._loaded_frame_labels = None
        self._loaded_em_data = None

    @property
    def video_observer_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_observer_sampler

    @property
    def num_video_observer_combinations(self):
        """
        Returns:
            Number of video-observer combinations in dataset.
        """
        return len(self._video_and_label_paths)

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len({video_path for video_path, _ in self._video_and_label_paths})

    @property
    def num_observers(self):
        """
        Returns:
            Number of observers in dataset.
        """
        return len({utils.get_observer_from_label_path(label_path) for _, label_path in self._video_and_label_paths})

    @property
    def video_observer_combinations(self) -> VideosObserversPaths:
        """
        Returns:
            List of video-observer combinations in dataset.
        """
        return self._video_and_label_paths

    @property
    def videos(self) -> List[str]:
        """
        Returns:
            List of videos in dataset.
        """
        return list({video_path for video_path, _ in self._video_and_label_paths})

    @property
    def observers(self) -> List[str]:
        """
        Returns:
            List of observers in dataset.
        """
        return list({utils.get_observer_from_label_path(label_path) for _, label_path in self._video_and_label_paths})

    def _build_sample_dict(self, loaded_clip, loaded_frame_labels, loaded_em_data, video_name, observer,
                           video_index=None, clip_index=None, aug_index=None):
        frames = loaded_clip["video"]
        frame_indices = loaded_clip["frame_indices"]
        audio_samples = loaded_clip["audio"]
        frame_labels = torch.tensor(loaded_frame_labels, dtype=torch.float32)[frame_indices]
        em_data = torch.tensor(loaded_em_data, dtype=torch.int8)[
            frame_indices] if loaded_em_data else None

        # Normalize gaze location labels to range [-1, 1]
        max_h, max_w = frames.shape[-2:]
        frame_labels[:, 0] = torch.clamp(frame_labels[:, 0], min=0, max=max_h - 1)
        frame_labels[:, 1] = torch.clamp(frame_labels[:, 1], min=0, max=max_w - 1)
        
        if self._predict_change:
            # Padding to avoid -1 and 1 in arctanh
            frame_labels_change = ((frame_labels + 1) / torch.tensor([(max_h + 2) / 2., (max_w + 2) / 2.])) - 1.

            # Use arctanh/tanh conversion to stay within image bounds
            frame_labels_change = torch.atanh(frame_labels_change)
            frame_labels_change[1:, :] -= torch.roll(frame_labels_change, 1, dims=0)[1:, :]

        frame_labels = (frame_labels / torch.tensor([max_h / 2., max_w / 2.])) - 1.

        # One-hot encode eye movement class labels to vector of [NOISE, FIXATION, SACCADE, SMOOTH PURSUIT]
        if loaded_em_data:
            em_data = torch.tensor(self.em_encoder.transform(em_data[:, None]).toarray(), dtype=torch.float32)

        sample_dict = {
            "video": frames,
            "video_name": video_name,
            "video_index": video_index,
            "clip_index": clip_index,
            "aug_index": aug_index,
            "observer": observer,
            "frame_labels": frame_labels,
            "frame_indices": frame_indices,
            **({"em_data": em_data} if em_data is not None else {}),
            **({"frame_labels_change": frame_labels_change} if self._predict_change else {}),
            **({"audio": audio_samples} if audio_samples is not None else {}),
        }

        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        return sample_dict

    def get_clip(self, video_name: str, observer: str, clip_start: float, clip_end: Optional[float] = None) -> Dict:
        """
        Helper function to sample custom clip data. Returns dictionary analog to __next__

        Args:
            video_name: Name of video
            observer:   Name of observer
            clip_start: Clip start time in seconds
            clip_end:   (Optional) Clip end time in seconds, default is clip_duration in clip sampler

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'video_name': <video_name>,
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                    'observer': <observer_abbreviation>,
                    'frame_labels': <label_tensor>
                    'em_data': <em_label_tensor> (optional)
                    'audio': <audio_data> (optional)
                }
        """
        video_path, label_path = self._video_and_label_paths.get_paths_for_video_observer(video_name, observer)
        video = self.video_path_handler.video_from_path(
            video_path,
            decode_audio=self._decode_audio,
            decoder=self._decoder,
        )

        # Load frame labels and em phase data from label file
        loaded_frame_labels, loaded_em_data = utils.read_label_file(label_path, with_video_name=False)

        if clip_end is None:
            clip_end = clip_start + self._clip_sampler._clip_duration
        loaded_clip = video.get_clip(clip_start, clip_end)

        try:  # TODO: Check that labels exist for frame_indices
            sample_dict = self._build_sample_dict(loaded_clip, loaded_frame_labels, loaded_em_data, video_name,
                                                  observer)
        except Exception as e:
            logger.debug(
                "Failed to select label data with error: {}".format(e)
            )

        # User can force dataset to continue by returning None in transform.
        if sample_dict is None:
            raise Exception("Error: Transform returned None on requested clip.")

        return sample_dict

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'video_name': <video_name>,
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                    'observer': <observer_abbreviation>,
                    'frame_labels': <label_tensor>
                    'em_data': <em_label_tensor> (optional)
                    'audio': <audio_data> (optional)
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_observer_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video, label_path, video_index = self._loaded_video_label
            else:
                video_index = next(self._video_sampler_iter)
                try:
                    video_path, label_path = self._video_and_label_paths[video_index]
                    video = self.video_path_handler.video_from_path(
                        video_path,
                        decode_audio=self._decode_audio,
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video, label_path, video_index)

                    # Load frame labels and em phase data from label file
                    self._loaded_frame_labels, self._loaded_em_data = utils.read_label_file(label_path,
                                                                                            with_video_name=False)

                except Exception as e:
                    logger.debug(
                        "Failed to load video with error: {}; trial {}".format(
                            e,
                            i_try,
                        )
                    )
                    continue

            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self._clip_sampler(
                self._next_clip_start_time, video.duration, label_path
            )

            # Only load the clip once and reuse previously stored clip if there are multiple
            # views for augmentations to perform on the same clip.
            if aug_index == 0:
                self._loaded_clip = video.get_clip(clip_start, clip_end)

            self._next_clip_start_time = clip_end

            video_is_null = (
                    self._loaded_clip is None or self._loaded_clip["video"] is None
            )
            if is_last_clip or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                self._loaded_video_label[0].close()
                self._loaded_video_label = None
                self._next_clip_start_time = 0.0

                if video_is_null:
                    logger.debug(
                        "Failed to load clip {}; trial {}".format(video.name, i_try)
                    )
                    continue

            try:  # TODO: Check that labels exist for frame_indices
                observer = utils.get_observer_from_label_path(label_path)
                sample_dict = self._build_sample_dict(self._loaded_clip, self._loaded_frame_labels,
                                                      self._loaded_em_data, video.name,
                                                      observer, video_index, clip_index, aug_index)
            except Exception as e:
                logger.debug(
                    "Failed to select label data with error: {}; trial {}".format(
                        e,
                        i_try,
                    )
                )
                continue

            # User can force dataset to continue by returning None in transform.
            if sample_dict is None:
                continue

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self


def gaze_labeled_video_dataset(
        data_path: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        video_path_prefix: str = "",
        video_file_suffix: str = "",
        decode_audio: bool = True,
        decoder: str = "pyav",
        predict_change: bool = False,
) -> GazeLabeledVideoDataset:
    """
    A helper function to create ``LabeledVideoDataset`` object for Ucf101 and Kinetics datasets.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.

        video_file_suffix (str): Video file suffix. Empty string in case video exists as a
                directory of frame images.

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video.

        predict_change (bool): If True, predicted data is relative change and not absolute positions

    """
    videos_and_observers = VideosObserversPaths.from_path(data_path, video_file_suffix)
    # videos_and_observers.path_prefix = video_path_prefix
    dataset = GazeLabeledVideoDataset(
        videos_and_observers,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
        predict_change=predict_change,
    )
    return dataset
