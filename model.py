import torch
import torch.nn.functional as F
import pytorch_lightning

from RIM import RIM

from gaze_video_data_module import GazeVideoDataModule
from feature_extraction import FeatureExtractor, FPN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_RIM_model(device, features=512):
    return RIM(
        device=device,
        input_size=features,
        hidden_size=4,
        num_units=6,
        k=2,
        rnn_cell='LSTM',
        n_layers=4,
        bidirectional=False
    )


class GazePredictionLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, lr=0.09, batch_size=16, frames=30, input_dims=(244, 244), out_channels=16):
        super().__init__()

        self.learning_rate = lr
        self.batch_size = batch_size

        # Feature Pyramid Network for feature extraction
        self.backbone = FeatureExtractor(device, input_dims, self.batch_size)
        self.fpn = FPN(device, in_channels_list=self.backbone.in_channels, out_channels=out_channels, only_use_last_layer=True)

        # Dry run to get input size for RIM
        inp = torch.randn(self.batch_size * frames, 3, *input_dims)
        with torch.no_grad():
            out = self.fpn(self.backbone(inp))
        print(f"FPN produces {out.shape[-1]} different Features")
        self.rim = make_RIM_model(device, features=out.shape[-1])
        #self.out_pool = torch.nn.LazyLinear(out_features=2)
        self.out_pool = torch.nn.Linear(in_features=24, out_features=2)

    def forward(self, x):
        # Reshaping as feature extraction expects tensor of shape (B, C, H, W)
        batch_size, ch, frames, *input_dim = x.shape
        x = torch.swapaxes(x, 1, 2)
        x = x.reshape(batch_size * frames, 3, *input_dim)

        # Feature extraction in feature pyramid network
        x = self.backbone(x)
        x = self.fpn(x)

        # De-tangle batches and frames again
        features = x.shape[1]
        x = x.reshape(batch_size, frames, features)
        x = torch.swapaxes(x, 0, 1)         # RIM expects tensor of shape (seq, B, features)

        # Sequential processing in RIM
        out, h, c = self.rim(x)
        # TODO: Improve reduction from different RIM units and hidden units to gaze predictions
        out = self.out_pool(out)
        out = torch.swapaxes(out, 0, 1)     # Swap batch and sequence again
        return out

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        # TODO: Include EM-classification in data
        y_hat = self.forward(batch["video"])

        # Compute mean squared error loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        # TODO: Implement specialized loss
        loss = F.mse_loss(y_hat, batch["frame_labels"])

        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item(), batch_size=batch["video"].shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch["video"])
        loss = F.mse_loss(y_hat, batch["frame_labels"])
        self.log("val_loss", loss, batch_size=batch["video"].shape[0])
        return loss

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# Dataset configuration
_DATA_PATH_FRAMES = r'data/GazeCom/movies_m2t_224x224'
#csv_path = r'C:\Projects\uni\master_thesis\datasets\GazeCom\movies_mpg_frames\test_pytorchvideo.txt'
_CLIP_DURATION = 5  # Duration of sampled clip for each video in seconds
_BATCH_SIZE = 16
_NUM_WORKERS = 8  # Number of parallel processes fetching data
_OUT_CHANNELS = 16

regression_module = GazePredictionLightningModule(batch_size=_BATCH_SIZE, input_dims=(244, 244), out_channels=_OUT_CHANNELS)
data_module = GazeVideoDataModule(data_path=_DATA_PATH_FRAMES, video_file_suffix='', batch_size=_BATCH_SIZE, num_workers=_NUM_WORKERS)
#data_module = GazeVideoDataModule(data_path=_DATA_PATH, video_file_suffix='.m2t', batch_size=_BATCH_SIZE, num_workers=_NUM_WORKERS)

#early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True, mode='auto')
trainer = pytorch_lightning.Trainer(gpus=[0], max_epochs=40, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=False, fast_dev_run=False)
                      #progress_bar_refresh_rate=10, early_stop_callback=early_stop_callback)

trainer.tune(regression_module, data_module)
trainer.fit(regression_module, data_module)
