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

    Expects frames in shape (T, C, H, W) or (T, H, W) for one-channel tensors.
    """
    logger.add_histogram(f"{name}_hist", frames)
    if interpolate_range:
        frames = np.interp(frames, (frames.min(), frames.max()), (0, 255)).astype('uint8')

    # (T,C,H,W) -> (N,T,C,H,W)
    if len(frames.shape) == 3:  # grayscale
        frames = frames[:, None, :, :]
    frames = torch.from_numpy(frames[None, :])

    logger.add_video(
        tag=name,
        vid_tensor=frames,
        fps=fps)


class GazePredictionLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, lr, batch_size, frames, input_dims, out_channels, predict_em,
                 fpn_only_use_last_layer, rim_hidden_size, rim_num_units, rim_k,
                 rnn_cell, rim_layers, attention_heads, p_teacher_forcing, n_teacher_vals):
        super().__init__()

        self.learning_rate = lr
        self.batch_size = batch_size
        self.predict_em = predict_em
        self.p_teacher_forcing = p_teacher_forcing
        self.n_teacher_vals = n_teacher_vals

        self.save_hyperparameters()

        # Feature Pyramid Network for feature extraction
        self.backbone = FeatureExtractor(device, input_dims, self.batch_size)
        self.fpn = FPN(device, in_channels_list=self.backbone.in_channels, out_channels=out_channels,
                       only_use_last_layer=fpn_only_use_last_layer)

        # Dry run to get input size for RIM
        inp = torch.randn(self.batch_size * frames, 3, *input_dims)
        with torch.no_grad():
            out = self.fpn(self.backbone(inp))
        n_features = out.shape[-1]
        print(f"FPN produces {n_features} different Features")

        self.out_features = 6 if self.predict_em else 2
        if n_teacher_vals > 0:
            n_features += n_teacher_vals * self.out_features

        self.rim = RIM(
            device=device,
            input_size=n_features,
            hidden_size=rim_hidden_size,
            num_units=rim_num_units,
            k=rim_k,
            rnn_cell=rnn_cell,
            n_layers=rim_layers,
            bidirectional=False
        )

        # Dry run to get input size for end layer
        inp = torch.randn(frames, self.batch_size, n_features, device=device)
        with torch.no_grad():
            out, _, _ = self.rim(inp)
        embed_dim = out.shape[-1]

        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim, attention_heads)

        # here comes the hack, because MultiheadAttention maps to embed_dim
        factory_kwargs = {'device': self.multihead_attn.in_proj_weight.device, 'dtype':  self.multihead_attn.in_proj_weight.dtype}
        self.multihead_attn.out_proj = torch.nn.modules.linear.NonDynamicallyQuantizableLinear(embed_dim, self.out_features, bias=self.multihead_attn.in_proj_bias is not None, **factory_kwargs)
        self.multihead_attn._reset_parameters()

    def forward(self, x, y=None, log_features=False):
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

        # Process each time step in RIM and Multiattention layer (teacher forcing is applied)
        xs = list(torch.split(x, 1, dim = 0))
        outputs = []
        h, c = None, None
        for i, x in enumerate(xs):
            # If teacher forcing activated, extend features with possible teacher values
            if self.n_teacher_vals > 0:
                x = torch.nn.ConstantPad1d((0, self.n_teacher_vals * self.out_features), 0)(x)
                if y is not None and i != 0:
                    y_prev = y[:, i-1, :]

                    # Repeat label values n_teacher_vals times
                    teacher_vals = torch.tile(y_prev, (1, 1, self.n_teacher_vals))
                    output_vals = torch.tile(output, (1, 1, self.n_teacher_vals))

                    # Create random mask over batch
                    random_mask = torch.FloatTensor(x.shape[:2]).uniform_().to(device=device) < self.p_teacher_forcing

                    # Add ground truth or output of previous iteration to input features of next iteration
                    x[:, :, -self.n_teacher_vals * self.out_features:][random_mask, :] = teacher_vals[random_mask, :]
                    x[:, :, -self.n_teacher_vals * self.out_features:][~random_mask, :] = output_vals[~random_mask, :]

            x, h, c = self.rim(x, h=h, c=c)
            output, attn_output_weights = self.multihead_attn(x, x, x)
            outputs.append(output)
        out = torch.cat(outputs, dim = 0)

        out = torch.swapaxes(out, 0, 1)     # Swap batch and sequence again
        return out

    def loss(self, y_hat, batch):
        #return F.mse_loss(y_hat[:, :, :2], batch['frame_labels'])
        not_noise = batch['em_data'][:, :, 0] == 0
        loss = F.mse_loss(y_hat[:, :, :2][not_noise], batch['frame_labels'][not_noise])
        if self.predict_em:
            loss += F.cross_entropy(y_hat[:, :, 2:][not_noise], batch['em_data'][not_noise])
        return loss

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        # TODO: Include EM-classification in data
        if self.current_epoch == 0:
            for name, param in self.fpn.fpn.named_parameters():
                print(f"fpn_{name}_epoch_{self.current_epoch}", param.shape)
            for name, param in self.rim.named_parameters():
                print(f"rim_{name}_epoch_{self.current_epoch}", param.shape)
            for name, param in self.multihead_attn.named_parameters():
                print(f"attn_{name}_epoch_{self.current_epoch}", param.shape)
        if self.current_epoch % 25 == 0:
            y_hat = self.forward(batch["video"], y=batch['frame_labels'], log_features=True)
            for name, param in self.fpn.fpn.named_parameters():
                self.trainer.logger.experiment.add_histogram(f"fpn_{name}_epoch_{self.current_epoch}", param)
            for name, param in self.rim.named_parameters():
                self.trainer.logger.experiment.add_histogram(f"rim_{name}_epoch_{self.current_epoch}", param)
            for name, param in self.multihead_attn.named_parameters():
                self.trainer.logger.experiment.add_histogram(f"attn_{name}_epoch_{self.current_epoch}", param)
        else:
            y_hat = self.forward(batch["video"], y=batch['frame_labels'])
        if self.current_epoch % 10 == 0:
            print("y_hat:\n", y_hat[0, :, :2])
            print("y:\n", batch['frame_labels'][0])
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
                lr=1e-6, only_tune: bool = False, predict_em=False, fpn_only_use_last_layer=True,
                rim_hidden_size=400, rim_num_units=6, rim_k=4, rnn_cell='LSTM', rim_layers=1, 
                attention_heads=2, p_teacher_forcing=0.3, n_teacher_vals=10, train_checkpoint=None):
    """
    Train or tune the model on the data in data_path.
    """
    if train_checkpoint:
        regression_module = GazePredictionLightningModule.load_from_checkpoint(train_checkpoint).to(device=device)
    else:
        regression_module = GazePredictionLightningModule(lr=lr, batch_size=batch_size, frames=round(clip_duration * 29.97),
                                                        input_dims=(224, 224), out_channels=out_channels,
                                                        predict_em=predict_em,
                                                        fpn_only_use_last_layer=fpn_only_use_last_layer,
                                                        rim_hidden_size=rim_hidden_size,
                                                        rim_num_units=rim_num_units, rim_k=rim_k, rnn_cell=rnn_cell,
                                                        rim_layers=rim_layers, attention_heads=attention_heads,
                                                        p_teacher_forcing=p_teacher_forcing, n_teacher_vals=n_teacher_vals)
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
        tb_logger = pytorch_lightning.loggers.TensorBoardLogger("data/lightning_logs", name='')
        early_stop_callback = pytorch_lightning.callbacks.early_stopping.EarlyStopping(monitor='train_loss', min_delta=0.002, patience=15, verbose=True, mode='min')
        trainer = pytorch_lightning.Trainer(gpus=[0], max_epochs=51, auto_lr_find=False, auto_scale_batch_size=False, logger=tb_logger,
                                            fast_dev_run=False, log_every_n_steps=1, callbacks=[early_stop_callback])

        trainer.fit(regression_module, data_module)


if __name__ == '__main__':
    # Dataset configuration
    _DATA_PATH_FRAMES = r'data/GazeCom/movies_m2t_224x224/single_video'
    _DATA_PATH_FRAMES = r'data/GazeCom/movies_m2t_224x224/single_clip'
    # csv_path = r'C:\Projects\uni\master_thesis\datasets\GazeCom\movies_mpg_frames\test_pytorchvideo.txt'
    _CLIP_DURATION = 2  # Duration of sampled clip for each video in seconds
    _BATCH_SIZE = 16
    _NUM_WORKERS = 1 # For one video
    #_NUM_WORKERS = 8  # Number of parallel processes fetching data
    _OUT_CHANNELS = 16

    train_model(_DATA_PATH_FRAMES, _CLIP_DURATION, _BATCH_SIZE, _NUM_WORKERS, _OUT_CHANNELS, only_tune=False, predict_em=False,
                 fpn_only_use_last_layer=True)#, train_checkpoint="lightning_logs/version_52/checkpoints/epoch=87-step=86.ckpt")
