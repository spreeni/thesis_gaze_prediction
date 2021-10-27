# Guided by tutorial on https://pytorchvideo.org/docs/tutorial_classification

import os
import pytorch_lightning
import torch.utils.data
from pytorchvideo.data import make_clip_sampler

# load transforms
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

from gaze_labeled_video_dataset import gaze_labeled_video_dataset


class GazeVideoDataModule(pytorch_lightning.LightningDataModule):
    # Dataset configuration
    _DATA_PATH = root_path = r'C:\Projects\uni\master_thesis\datasets\GazeCom\movies_mpg_frames'
    #csv_path = r'C:\Projects\uni\master_thesis\datasets\GazeCom\movies_mpg_frames\test_pytorchvideo.txt'
    _CLIP_DURATION = 2  # Duration of sampled clip for each video
    _BATCH_SIZE = 8
    _NUM_WORKERS = 8  # Number of parallel processes fetching data

    def train_dataloader(self):
        """
        Create the train partition from the list of video labels
        in {self._DATA_PATH}/train
        """
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            #UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            #RandomShortSideScale(min_size=256, max_size=320),
                            #RandomCrop(244),
                            #RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
        train_dataset = gaze_labeled_video_dataset(
            data_path=os.path.join(self._DATA_PATH, "train"),
            clip_sampler=make_clip_sampler("random", self._CLIP_DURATION),
            video_sampler=torch.utils.data.RandomSampler,
            transform=train_transform,
            decode_audio=False,
            decoder="pyav",
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def val_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            #UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                        ]
                    ),
                ),
            ]
        )
        val_dataset = gaze_labeled_video_dataset(
            data_path=os.path.join(self._DATA_PATH, "val"),
            clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
            video_sampler=torch.utils.data.RandomSampler,
            transform=val_transform,
            decode_audio=False,
            decoder="pyav",
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )
