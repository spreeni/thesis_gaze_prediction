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
    RandomHorizontalFlip,
    Grayscale
)

from gaze_labeled_video_dataset import gaze_labeled_video_dataset


class GazeVideoDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, data_path, clip_duration=2, batch_size=8, num_workers=8):
        """
        Initializes a GazeVideoDataModule which can be used in Pytorch Lightning

        Args:
            data_path:      Data root directory, should contain a "train" and "val" directory
            clip_duration:  Duration of sampled clip for each video in seconds
            batch_size:     Batch size per model run
            num_workers:    Number of parallel processes fetching data
        """
        super().__init__()

        # Dataset configuration
        self._DATA_PATH = data_path
        self._CLIP_DURATION = clip_duration
        self._BATCH_SIZE = batch_size
        self._NUM_WORKERS = num_workers

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
                            #RandomHorizontalFlip(p=0.5),
                            #Grayscale()
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
            #transform=None,
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
        Create the validation partition from the list of video labels
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
                            #Grayscale()
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
            #transform=None,
            decode_audio=False,
            decoder="pyav",
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )
