"""
Implementation of the proposed model in Pytorch Lightning.

The model forward pass works as follows:
    1. Visual features are extracted per frame with a Feature Pyramid Network
    2. The temporal state of the prediction gets processed in Recurrent Independent Mechanisms (or alternatively a single LSTM)
        2.1 In each step Teacher Forcing can be applied by randomly appending the previous model prediction or the
            previous ground truth label to the RIM input
    3. The hidden state of the RIM units gets condensed into a gaze prediction with Multihead-Attention
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pytorch_lightning
from pytorch_lightning.callbacks import LearningRateMonitor
from torchinfo import summary
from sklearn.preprocessing import OneHotEncoder

from RIM import RIM, RIMCell, GroupLinearLayer

from gaze_video_data_module import GazeVideoDataModule
from feature_extraction import FeatureExtractor, FPN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_DEBUG_INFO = False  # Flag to log storage-intensive feature channels of Feature Pyramid network and input clips


def log_tensor_as_video(model, frames, name, fps=5, interpolate_range=True):
    """
    Adds tensor as video to tensorboard logs.

    Expects frames in shape (T, C, H, W) or (T, H, W) for one-channel tensors.
    """
    model.trainer.logger.experiment.add_histogram(f"{name}_hist", frames, model.global_step)
    if interpolate_range:
        frames = np.interp(frames, (frames.min(), frames.max()), (0, 255)).astype('uint8')

    # (T,C,H,W) -> (N,T,C,H,W)
    if len(frames.shape) == 3:  # grayscale
        frames = frames[:, None, :, :]
    frames = torch.from_numpy(frames[None, :])
    
    model.trainer.logger.experiment.add_video(
        tag=f"{name}_epoch_{model.global_step}",
        vid_tensor=frames,
        fps=fps)


def log_tensor_as_image(model, frame, name, interpolate_range=True, dataformats='CHW'):
    """
    Adds tensor as image to tensorboard logs.

    Expects frame in shape (C, H, W) or (H, W) for one-channel tensors.
    """
    if interpolate_range:
        frame = np.interp(frame, (frame.min(), frame.max()), (0, 255)).astype('uint8')
    
    model.trainer.logger.experiment.add_image(
        tag=f"{name}_epoch_{model.global_step}",
        img_tensor=frame,
        dataformats=dataformats)


class GazePredictionLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, lr, batch_size, frames, input_dims, out_channels, predict_em,
                 backbone_model, fpn_only_use_last_layer, rim_hidden_size, rim_num_units, rim_k,
                 rnn_cell, rim_layers, out_attn_heads, p_teacher_forcing, n_teacher_vals,
                 weight_init, mode, loss_fn, lambda_reg_fix, lambda_reg_sacc, input_attn_heads,
                 input_dropout, comm_dropout, channel_wise_attention):
        super().__init__()

        self.rim_hidden_size = rim_hidden_size
        self.learning_rate = lr
        self.batch_size = batch_size
        self.predict_em = predict_em
        self.p_teacher_forcing = p_teacher_forcing
        self.n_teacher_vals = n_teacher_vals
        self.channel_wise_attention = channel_wise_attention
        self.loss_fn = loss_fn
        self.lambda_reg_fix = lambda_reg_fix
        self.lambda_reg_sacc = lambda_reg_sacc

        self.mode = mode

        self.em_encoder = OneHotEncoder()
        self.em_encoder.fit([[i] for i in range(4)])

        self.save_hyperparameters()

        # Feature Pyramid Network for feature extraction
        self.backbone = FeatureExtractor(device, input_dims, self.batch_size, model=backbone_model)
        self.fpn = FPN(device, in_channels_list=self.backbone.in_channels, out_channels=out_channels,
                       separate_channels=self.channel_wise_attention, only_use_last_layer=fpn_only_use_last_layer)

        # Dry run to get input size for RIM
        inp = torch.randn(self.batch_size * frames, 3, *input_dims)
        with torch.no_grad():
            out = self.fpn(self.backbone(inp))
        n_features = out.shape[-1]
        print(f"FPN produces {n_features} different Features")

        self.out_features = 6 if self.predict_em else 2
        
        if self.mode == 'LSTM':
            n_features_lstm = n_features
            if n_teacher_vals > 0:
                n_features_lstm += n_teacher_vals * self.out_features
            self.lstm = torch.nn.LSTM(input_size=n_features_lstm, hidden_size=rim_hidden_size, device=device, bidirectional=False)
        else:
            self.rim = RIM(
                device=device,
                input_size=n_features,
                hidden_size=rim_hidden_size,
                num_units=rim_num_units,
                k=rim_k,
                rnn_cell=rnn_cell,
                n_layers=rim_layers,
                bidirectional=False,
                num_input_heads=input_attn_heads,
                input_dropout=input_dropout,
                comm_dropout=comm_dropout,
                #input_key_size=512,
                #input_query_size=512,
                #input_value_size=1600,
                #comm_value_size=100
                p_teacher_forcing = p_teacher_forcing,
                n_teacher_vals = n_teacher_vals,
                out_features=self.out_features
            )

        # Dry run to get input size for end layer
        inp = torch.randn(frames, self.batch_size, 1 if not self.channel_wise_attention else out_channels, n_features, device=device)
        with torch.no_grad():
            if self.mode == 'LSTM':
                inp = torch.randn(frames, self.batch_size, n_features_lstm, device=device)
                out = self.lstm(inp)[0]
            else:
                out = self.rim(inp)[0]
            
        embed_dim = out.shape[-1]

        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim, out_attn_heads, device=device)

        # here comes the hack, because MultiheadAttention maps to embed_dim
        factory_kwargs = {'device': self.multihead_attn.in_proj_weight.device, 'dtype':  self.multihead_attn.in_proj_weight.dtype}
        self.multihead_attn.out_proj = torch.nn.modules.linear.NonDynamicallyQuantizableLinear(embed_dim, self.out_features, bias=self.multihead_attn.in_proj_bias is not None, **factory_kwargs)

        # Freeze parameters except for attention
        #for param in self.fpn.parameters():
        #    param.requires_grad = False
        #for param in self.rim.parameters():
        #    param.requires_grad = False
        #for param in self.multihead_attn.parameters():
        #    param.requires_grad = False
        
        #self.reset_parameters()
        self.multihead_attn._reset_parameters()

        self.sample_param_vals = dict()
        self.prev_param_vals = dict()

    def reset_parameters(self):
        linear_weight_init = {'xavier_normal': torch.nn.init.xavier_normal_, 'kaiming_uniform': torch.nn.init.kaiming_uniform_}
        linear_weight_init = linear_weight_init.get(self.hparams.weight_init, lambda x: None) #lambda x: None means that if hparams gives unknown key then the weight_init is weight_init(x) = None -> stick to default weights

        def weights_init(m):
            if isinstance(m, torch.nn.Linear):
                linear_weight_init(m.weight.data)
            elif isinstance(m, GroupLinearLayer):
                linear_weight_init(m.w.data)

            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.)

        def weights_init_rnn(m):
            def orthogonal_rnn_weight_init(m):
                if isinstance(m, GroupLinearLayer):
                    torch.nn.init.orthogonal_(m.w.data)

            if isinstance(m, RIMCell):
                m.rnn.apply(orthogonal_rnn_weight_init)
            
        self.apply(weights_init)
        self.apply(weights_init_rnn)
        self.multihead_attn._reset_parameters()

    def forward(self, x, y=None, em_data=None, log_features=False):
        batch_size, ch, frames, *input_dim = x.shape
        x = torch.swapaxes(x, 1, 2)
        if log_features:
            for i in range(min(batch_size, 2)): #(N,T,C,H,W)
                log_tensor_as_video(self, x[i].cpu().detach().numpy(), f"vids_{i}", fps=5, interpolate_range=True)
        
        # Reshaping as feature extraction expects tensor of shape (B, C, H, W)
        x = x.reshape(batch_size * frames, 3, *input_dim)

        # Feature extraction in feature pyramid network
        x = self.backbone(x)
        if log_features:
            x, ch_data = self.fpn(x, return_channels=True)
            for key in ch_data:
                channels = ch_data[key].shape[1]
                key_data = ch_data[key].reshape(
                        batch_size, frames, channels, *ch_data[key].shape[-2:]
                    ).cpu().detach().numpy()
                for i in range(min(batch_size, 2)):
                    for ch in range(channels):
                        log_tensor_as_video(self, key_data[i, :, ch, :, :], f"fpn_{key}_{i}_ch{ch}", fps=5, interpolate_range=True)
        else:
            x = self.fpn(x)

        # De-tangle batches and frames again
        features = x.shape[-1]
        x = x.reshape(batch_size, frames, -1, features)
        out_channels = x.shape[2]
        x = torch.swapaxes(x, 0, 1)         # RIM expects tensor of shape (seq, B, features)

        # Process each time step in RIM and Multiattention layer (teacher forcing is applied)
        xs = list(torch.split(x, 1, dim = 0))
        outputs = []
        rim_activations = []
        output = None

        if self.mode == 'LSTM':
            h = torch.randn(1, batch_size, self.rim_hidden_size, device=device)
            c = torch.randn(1, batch_size, self.rim_hidden_size, device=device)
        else:
            h, c = None, None
        for i, x in enumerate(xs):
            if self.mode == 'LSTM':
                # If teacher forcing activated, extend features with possible teacher values
                if self.n_teacher_vals > 0:
                    x = torch.nn.ConstantPad1d((0, self.n_teacher_vals * self.out_features), 0)(x)

                    if i != 0:
                        # Add output of previous iteration to input features of next iteration
                        output_repeated = torch.tile(output, (1, 1, self.n_teacher_vals * out_channels))
                        x[:, :, :, -self.n_teacher_vals * self.out_features:] = output_repeated.reshape(1, batch_size, out_channels, -1)

                        if y is not None:
                            y_prev = y[:, i-1, :]

                            # Repeat label values n_teacher_vals times
                            y_prev_repeated = torch.tile(y_prev, (1, 1, self.n_teacher_vals * out_channels)).reshape(1, batch_size, out_channels, -1)

                            # Create random mask over batch
                            random_mask = torch.FloatTensor(x.shape[:2]).uniform_().to(device=device) < self.p_teacher_forcing

                            # Add ground truth for random mask to input features of next iteration
                            x[:, :, :, -self.n_teacher_vals * self.out_features:][random_mask, :, :] = y_prev_repeated[random_mask, :, :]
                
                x, (h, c) = self.lstm(x[:, :, 0, :], (h, c))
            else:
                y_prev = None
                if y is not None and i != 0:
                    y_prev = y[:, i-1, :].unsqueeze(0)
                x, mask, h, c = self.rim(x, h=h, c=c, y_prev=y_prev, y_hat_prev=output)
                rim_activations.append(mask)
            output, attn_output_weights = self.multihead_attn(x, x, x)
            output = torch.tanh(output)
            outputs.append(output)
        out = torch.cat(outputs, dim = 0)

        # Log RIM unit activations
        if self.mode != 'LSTM' and log_features:
            if em_data is not None:
                em_data = em_data.cpu().detach().numpy()
                for i in range(batch_size):
                    log_tensor_as_image(self, em_data[i].T, f"em_phases{i}", interpolate_range=False, dataformats='HW')

            rim_activations = torch.cat(rim_activations, dim=0).swapaxes(0, 1)
            for i in range(batch_size):
                log_tensor_as_image(self, rim_activations[i].T, f"rim_activations{i}", interpolate_range=False, dataformats='HW')

        out = torch.swapaxes(out, 0, 1)     # Swap batch and sequence again
        return out

    def loss(self, y_hat, batch, train_step=False):
        not_noise = batch['em_data'][:, :, 0] == 0
        saccades = batch['em_data'][:, :, 2] == 1
        fix_sp = not_noise & ~saccades

        # Gaze regression loss: loss_fn can be mse_loss/l1_loss/smooth_l1_loss
        loss_fn = getattr(F, self.loss_fn)
        loss = loss_fn(y_hat[:, :, :2][not_noise], batch['frame_labels'][not_noise])

        # Gaze regularization loss: During saccades gaze should move much, otherwise jitter is punished
        # First timepoint is ignored as it does not have a previous comparison
        if train_step:
            self.log("train_loss_mse", loss, prog_bar=True)
        fix_sp[0, 0] = False
        saccades[0, 0] = False
        #loss_reg_fix_sp = self.lambda_reg_fix * F.mse_loss(y_hat[:, :, :2][fix_sp], y_hat[:, :, :2][torch.roll(fix_sp, -1, 1)])
        #loss_reg_sacc = -self.lambda_reg_sacc * F.l1_loss(y_hat[:, :, :2][saccades], y_hat[:, :, :2][torch.roll(saccades, -1, 1)])
        loss_reg_fix_sp = 1./self.lambda_reg_fix + F.mse_loss(y_hat[:, :, :2][fix_sp], y_hat[:, :, :2][torch.roll(fix_sp, -1, 1)])
        loss_reg_sacc = 1./self.lambda_reg_sacc + F.l1_loss(y_hat[:, :, :2][saccades], y_hat[:, :, :2][torch.roll(saccades, -1, 1)])
        if train_step:
            self.log("reg_loss_fix", loss_reg_fix_sp, prog_bar=True)
            self.log("reg_loss_sacc", loss_reg_sacc, prog_bar=True)
        #loss += loss_reg_fix_sp + loss_reg_sacc
        loss *= loss_reg_fix_sp / loss_reg_sacc

        # Eye movement classification loss
        if self.predict_em:
            loss_em = F.cross_entropy(y_hat[:, :, 2:][not_noise], batch['em_data'][not_noise])
            self.log("gaze_loss", loss, prog_bar=True)
            self.log("em_phase_loss", loss_em, prog_bar=True)
            loss += loss_em
        return loss

    def save_and_plot_param_changes(self, param_name, curr_param_val):
        # Calculates param change in current step and plots histogram, then saves current param values
        if param_name in self.prev_param_vals:
            change = curr_param_val  - self.prev_param_vals[param_name]
            self.trainer.logger.experiment.add_histogram(f"{param_name}_change", change, self.global_step)
            self.trainer.logger.experiment.add_scalar(f"{param_name}_change_min", change.min(), self.global_step)
            self.trainer.logger.experiment.add_scalar(f"{param_name}_change_max", change.max(), self.global_step)
            self.trainer.logger.experiment.add_scalar(f"{param_name}_change_std", change.std(), self.global_step)
        self.prev_param_vals[param_name] = curr_param_val.detach().clone()

    def plot_sample_param_values(self):
        # Plot evolution of param values over epochs
        for name in self.sample_param_vals:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            vals = np.array(self.sample_param_vals[name]).T
            n_epochs = vals.shape[1]
            epochs = list(range(n_epochs))
            for y, val in enumerate(vals):
                ax.plot([y]*n_epochs, epochs, val)

            self.trainer.logger.experiment.add_figure(f"{name}_params", fig)

    def save_sample_param_values(self, name, vals):
        # Save first 20 param values over epochs
        if not name in self.sample_param_vals:
            self.sample_param_vals[name] = []
        self.sample_param_vals[name].append(vals.detach().cpu().flatten()[:20].numpy())

    def on_after_backward(self):
        # Log histograms of model parameters, gradients and parameter changes
        for name, param in self.named_parameters():
            if not name.startswith('backbone.') and LOG_DEBUG_INFO:
                # Log param values
                self.trainer.logger.experiment.add_histogram(name, param, self.global_step)
                self.save_sample_param_values(name, param)

                # Log param changes
                self.save_and_plot_param_changes(name, param)

                # Log gradients
                if param.grad is not None:
                    self.trainer.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.global_step)
                    self.save_sample_param_values(f"{name}_grad", param.grad)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        if self.global_step == 0:
            for name, param in self.named_parameters():
                if not name.startswith('backbone.'):
                    print(f"{name}_epoch_{self.global_step}", param.shape)
        
        # Very experimental as a time point
        if self.global_step == 45 and LOG_DEBUG_INFO:
            print("plotting param values!")
            self.plot_sample_param_values()

        if self.predict_em:
            y = torch.concat((batch['frame_labels'], batch['em_data']), dim=-1)
        else:
            y = batch['frame_labels']

        # Log features every 20th step
        if self.global_step % 20 == 0 and LOG_DEBUG_INFO:
            y_hat = self.forward(batch["video"], y=y, em_data=batch['em_data'], log_features=True)
        else:
            y_hat = self.forward(batch["video"], y=y)

        if self.global_step % 20 == 0:
            print("y_hat:\n", y_hat[0, :, :2])
            print("y:\n", batch['frame_labels'][0])
        # Compute mean squared error loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        # TODO: Implement specialized loss
        loss = self.loss(y_hat, batch, train_step=True)

        # Log the train loss and batch size to Tensorboard
        self.log("batch_size", batch["video"].shape[0], prog_bar=True)
        self.log("train_loss", loss, batch_size=batch["video"].shape[0], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch["video"])
        loss = self.loss(y_hat, batch)
        self.log("batch_size_val", batch["video"].shape[0], prog_bar=True)
        self.log("val_loss", loss, batch_size=batch["video"].shape[0])
        self.log("hp_metric", loss)
        return loss

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        #return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train_model(data_path: str, clip_duration: float, batch_size: int, num_workers: int, out_channels: int,
                lr=1e-6, only_tune: bool = False, predict_em=True, backbone_model='mobilenet_v3_large',
                fpn_only_use_last_layer=True, rim_hidden_size=400, rim_num_units=6, rim_k=4, rnn_cell='LSTM', rim_layers=1,
                out_attn_heads=2, p_teacher_forcing=0.3, n_teacher_vals=50, weight_init='xavier_normal', 
                gradient_clip_val=1., gradient_clip_algorithm='norm', mode='RIM', loss_fn='mse_loss',
                lambda_reg_fix=1000, lambda_reg_sacc=10, input_attn_heads=2, input_dropout=0.,
                comm_dropout=0., channel_wise_attention=False, train_checkpoint=None):
    """
    Train or tune the model on the data in data_path.
    """
    if train_checkpoint:
        regression_module = GazePredictionLightningModule.load_from_checkpoint(train_checkpoint).to(device=device)
    else:
        regression_module = GazePredictionLightningModule(lr=lr, batch_size=batch_size, frames=round(clip_duration * 29.97),
                                                        input_dims=(224, 224), out_channels=out_channels,
                                                        predict_em=predict_em, backbone_model=backbone_model,
                                                        fpn_only_use_last_layer=fpn_only_use_last_layer,
                                                        rim_hidden_size=rim_hidden_size,
                                                        rim_num_units=rim_num_units, rim_k=rim_k, rnn_cell=rnn_cell,
                                                        rim_layers=rim_layers, out_attn_heads=out_attn_heads,
                                                        p_teacher_forcing=p_teacher_forcing, n_teacher_vals=n_teacher_vals, 
                                                        weight_init=weight_init, mode=mode, loss_fn=loss_fn,
                                                        lambda_reg_fix=lambda_reg_fix, lambda_reg_sacc=lambda_reg_sacc,
                                                        input_attn_heads=input_attn_heads, input_dropout=input_dropout,
                                                        comm_dropout=comm_dropout, channel_wise_attention=channel_wise_attention)
    data_module = GazeVideoDataModule(data_path=data_path, video_file_suffix='', batch_size=batch_size,
                                      clip_duration=clip_duration, num_workers=num_workers)
    # data_module = GazeVideoDataModule(data_path=data_path, video_file_suffix='.m2t', batch_size=batch_size, clip_duration=clip_duration, num_workers=num_workers)

    #summary(regression_module, input_size=(batch_size, 3, round(clip_duration * 29.97), 224, 224))

    if only_tune:
        # Find maximum batch size that fits into memory
        trainer = pytorch_lightning.Trainer(gpus=[0], max_epochs=1, auto_lr_find=False, auto_scale_batch_size=True)
        trainer.tune(regression_module, data_module)

        # Find best initial learning rate with optimal batch size
        trainer = pytorch_lightning.Trainer(gpus=[0], max_epochs=1, auto_lr_find=True, auto_scale_batch_size=False)
        trainer.tune(regression_module, data_module)
    else:
        tb_logger = pytorch_lightning.loggers.TensorBoardLogger("data/lightning_logs", name='')
        early_stop_callback = pytorch_lightning.callbacks.early_stopping.EarlyStopping(monitor='train_loss', min_delta=0.0005, patience=20, verbose=True, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)
        trainer = pytorch_lightning.Trainer(gpus=[0], max_epochs=201, auto_lr_find=False, auto_scale_batch_size=False, logger=tb_logger,
                                            fast_dev_run=False, log_every_n_steps=1, callbacks=[], gradient_clip_val=gradient_clip_val,
                                            gradient_clip_algorithm=gradient_clip_algorithm, stochastic_weight_avg=False)#, track_grad_norm=2)

        trainer.fit(regression_module, data_module)


if __name__ == '__main__':
    # Dataset configuration
    _DATA_PATH_FRAMES = r'data/GazeCom/movies_m2t_224x224'
    _DATA_PATH_FRAMES = r'data/GazeCom/movies_m2t_224x224/all_videos_single_observer'
    _DATA_PATH_FRAMES = r'data/GazeCom/movies_m2t_224x224/single_video_all_observers'
    #_DATA_PATH_FRAMES = r'data/GazeCom/movies_m2t_224x224/single_video'
    #_DATA_PATH_FRAMES = r'data/GazeCom/movies_m2t_224x224/single_clip'
    # csv_path = r'C:\Projects\uni\master_thesis\datasets\GazeCom\movies_mpg_frames\test_pytorchvideo.txt'
    _CLIP_DURATION = 5  # Duration of sampled clip for each video in seconds
    _BATCH_SIZE = 16
    _NUM_WORKERS = 1 # For one video
    #_NUM_WORKERS = 12  # Number of parallel processes fetching data
    _OUT_CHANNELS = 8

    train_model(_DATA_PATH_FRAMES, _CLIP_DURATION, _BATCH_SIZE, _NUM_WORKERS, _OUT_CHANNELS, only_tune=False, predict_em=False,
                 fpn_only_use_last_layer=True)#, train_checkpoint="lightning_logs/version_52/checkpoints/epoch=87-step=86.ckpt")
