from logging import log
from os import X_OK
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning

from RIM import RIM

from gaze_video_data_module import GazeVideoDataModule
from feature_extraction import FeatureExtractor, FPN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_tensor_as_video(logger, frames, name, fps=5, interpolate_range=True):
    """
    Adds tensor as video to tensorboard logs.

    Expects frames in shape (T, C, H, W).
    """
    logger.add_histogram(f"{name}_hist", frames)
    if interpolate_range:
        frames = np.interp(frames, (frames.min(), frames.max()), (0, 255)).astype('uint8')

    # (T,C,H,W) -> (N,T,C,H,W)
    if len(frames.shape) == 3:  # grayscale
        frames = frames[:, None, :, :]
    frames = torch.from_numpy(frames[None, :])
    #frames = torch.swapaxes(frames, 3, 4)
    #frames = torch.swapaxes(frames, 2, 3)

    logger.add_video(
        tag=name,
        vid_tensor=frames,
        fps=fps)


class GazePredictionLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, lr=3e-2, batch_size=16, frames=30, input_dims=(244, 244), out_channels=16, predict_em=False,
                 fpn_only_use_last_layer=True, em_loss_scaling=3600, rim_hidden_size=8, rim_num_units=6, rim_k=3,
                 rnn_cell='LSTM', rim_layers=3, use_attention_layer=True, attention_heads=8):
        super().__init__()

        self.learning_rate = lr
        self.batch_size = batch_size
        self.predict_em = predict_em
        self.em_loss_scaling = em_loss_scaling

        self.save_hyperparameters()

        # Feature Pyramid Network for feature extraction
        self.backbone = FeatureExtractor(device, input_dims, self.batch_size)
        self.fpn = FPN(device, in_channels_list=self.backbone.in_channels, out_channels=out_channels,
                       only_use_last_layer=fpn_only_use_last_layer)

        # Dry run to get input size for RIM
        inp = torch.randn(self.batch_size * frames, 3, *input_dims)
        with torch.no_grad():
            out = self.fpn(self.backbone(inp))
        print(f"FPN produces {out.shape[-1]} different Features")

        self.rim = RIM(
            device=device,
            input_size=out.shape[-1],
            hidden_size=rim_hidden_size,
            num_units=rim_num_units,
            k=rim_k,
            rnn_cell=rnn_cell,
            n_layers=rim_layers,
            bidirectional=False
        )

        # Dry run to get input size for end layer
        inp = torch.randn(frames, self.batch_size, out.shape[-1], device=device)
        with torch.no_grad():
            out, _, _ = self.rim(inp)
        out_features = 3 if self.predict_em else 2
        self.attention_layer = None
        if use_attention_layer:
            self.attention_layer = torch.nn.TransformerEncoderLayer(d_model=out.shape[-1], nhead=attention_heads, device=device)
        # self.out_pool = torch.nn.LazyLinear(out_features=out_features, device=device)
        self.out_pool = torch.nn.Linear(in_features=out.shape[-1], out_features=out_features, device=device)

    def forward(self, x, log_features=False):
        batch_size, ch, frames, *input_dim = x.shape
        x = torch.swapaxes(x, 1, 2)
        if log_features:
            for i in range(min(batch_size, 2)): #(N,T,C,H,W)
                log_tensor_as_video(self.trainer.logger.experiment, x[i].cpu().detach().numpy(), f"vids_epoch_{self.current_epoch}", fps=5, interpolate_range=True)
        
        # Reshaping as feature extraction expects tensor of shape (B, C, H, W)
        x = x.reshape(batch_size * frames, 3, *input_dim)

        # Feature extraction in feature pyramid network
        x = self.backbone(x)
        if log_features:
            x, ch_data = self.fpn(x, return_channels=True)
            for key in ch_data:
                key_data = ch_data[key].cpu().detach().numpy()
                channels = ch_data[key].shape[1]
                for ch in range(channels):
                    log_tensor_as_video(self.trainer.logger.experiment, key_data[:, ch, :, :], f"fpn_{key}_{ch}_epoch_{self.current_epoch}", fps=5, interpolate_range=True)
        else:
            x = self.fpn(x)

        # De-tangle batches and frames again
        features = x.shape[1]
        x = x.reshape(batch_size, frames, features)
        x = torch.swapaxes(x, 0, 1)         # RIM expects tensor of shape (seq, B, features)

        # Sequential processing in RIM
        out, h, c = self.rim(x)
        # TODO: Improve reduction from different RIM units and hidden units to gaze predictions
        out = self.attention_layer(out)
        out = self.out_pool(out)
        out = torch.swapaxes(out, 0, 1)     # Swap batch and sequence again
        return out

    def loss(self, y_hat, batch):
        #return F.mse_loss(y_hat[:, :, :2], batch['frame_labels'])
        not_noise = batch['em_data'] != 0
        loss = F.mse_loss(y_hat[:, :, :2][not_noise], batch['frame_labels'][not_noise])
        if self.predict_em:
            loss += F.mse_loss(y_hat[:, :, 2][not_noise], batch['em_data'][not_noise]) * self.em_loss_scaling
        return loss

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        # TODO: Include EM-classification in data
        if self.current_epoch % 25 == 0:
            y_hat = self.forward(batch["video"], log_features=True)
            for name, param in self.fpn.fpn.named_parameters():
                self.trainer.logger.experiment.add_histogram(f"fpn_{name}_epoch_{self.current_epoch}", param)
        else:
            y_hat = self.forward(batch["video"])

        # Compute mean squared error loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        # TODO: Implement specialized loss
        loss = self.loss(y_hat, batch)

        # Log the train loss to Tensorboard
        self.log("train_loss", loss, batch_size=batch["video"].shape[0], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch["video"])
        loss = self.loss(y_hat, batch)
        self.log("val_loss", loss, batch_size=batch["video"].shape[0])
        self.log("hp_metric", loss)
        return loss

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train_model(data_path: str, clip_duration: float, batch_size: int, num_workers: int, out_channels: int,
                only_tune: bool = False, predict_em=True, fpn_only_use_last_layer=False, em_loss_scaling=1,
                rim_hidden_size=8, rim_num_units=6, rim_k=3, rnn_cell='LSTM', rim_layers=6, train_checkpoint=None):
    """
    Train or tune the model on the data in data_path.
    """
    if train_checkpoint:
        regression_module = GazePredictionLightningModule.load_from_checkpoint(train_checkpoint).to(device=device)
    else:
        regression_module = GazePredictionLightningModule(batch_size=batch_size, frames=round(clip_duration * 29.97),
                                                        input_dims=(244, 244), out_channels=out_channels,
                                                        predict_em=predict_em,
                                                        fpn_only_use_last_layer=fpn_only_use_last_layer,
                                                        em_loss_scaling=em_loss_scaling, rim_hidden_size=rim_hidden_size,
                                                        rim_num_units=rim_num_units, rim_k=rim_k, rnn_cell=rnn_cell,
                                                        rim_layers=rim_layers)
    data_module = GazeVideoDataModule(data_path=data_path, video_file_suffix='', batch_size=batch_size,
                                      clip_duration=clip_duration, num_workers=num_workers)
    # data_module = GazeVideoDataModule(data_path=data_path, video_file_suffix='.m2t', batch_size=batch_size, clip_duration=clip_duration, num_workers=num_workers)

    if only_tune:
        # Find maximum batch size that fits into memory
        trainer = pytorch_lightning.Trainer(gpus=[0], max_epochs=1, auto_lr_find=False, auto_scale_batch_size=True)
        trainer.tune(regression_module, data_module)

        # Find best initial learning rate with optimal batch size
        trainer = pytorch_lightning.Trainer(gpus=[0], max_epochs=1, auto_lr_find=True, auto_scale_batch_size=False)
        trainer.tune(regression_module, data_module)
    else:
        # early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True, mode='auto')
        trainer = pytorch_lightning.Trainer(gpus=[0], max_epochs=150, auto_lr_find=False, auto_scale_batch_size=False,
                                            fast_dev_run=False, log_every_n_steps=10)  # , early_stop_callback=early_stop_callback)

        trainer.fit(regression_module, data_module)


if __name__ == '__main__':
    # Dataset configuration
    _DATA_PATH_FRAMES = r'data/GazeCom/movies_m2t_224x224/single_video'
    # csv_path = r'C:\Projects\uni\master_thesis\datasets\GazeCom\movies_mpg_frames\test_pytorchvideo.txt'
    _CLIP_DURATION = 2  # Duration of sampled clip for each video in seconds
    _BATCH_SIZE = 8
    _NUM_WORKERS = 1  # Number of parallel processes fetching data
    _OUT_CHANNELS = 16

    train_model(_DATA_PATH_FRAMES, _CLIP_DURATION, _BATCH_SIZE, _NUM_WORKERS, _OUT_CHANNELS, only_tune=False, predict_em=False,
                 fpn_only_use_last_layer=True)#, train_checkpoint="lightning_logs/version_52/checkpoints/epoch=87-step=86.ckpt")
