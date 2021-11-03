import torch
import torch.nn.functional as F
import pytorch_lightning
import torchvision

from RIM import RIM

from gaze_video_data_module import GazeVideoDataModule


def make_headless_resnet_2plus1d():
    resnet_2plus1d = torchvision.models.video.r2plus1d_18(pretrained=True)

    # Freeze parameters
    for param in resnet_2plus1d.parameters():
        param.requires_grad = False

    # Return without last pooling layers
    return torch.nn.Sequential(*(list(resnet_2plus1d.children())[:-2]))


def make_RIM_model():
    return RIM(
        # device='cuda',
        device='cpu',
        input_size=512,
        hidden_size=3,
        num_units=6,
        k=4,
        rnn_cell='LSTM',
        n_layers=4,
        bidirectional=False
    )


class GazePredictionLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()

        resnet = make_headless_resnet_2plus1d()
        rim = make_RIM_model()
        self.model = torch.nn.Sequential(resnet, rim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(batch["video"])

        # Compute mean squared error loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = F.mse_loss(y_hat, batch["label"])

        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.mse_loss(y_hat, batch["label"])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-1)


# Dataset configuration
_DATA_PATH = r'C:\Projects\uni\master_thesis\datasets\GazeCom\movies_mpg_frames'
#csv_path = r'C:\Projects\uni\master_thesis\datasets\GazeCom\movies_mpg_frames\test_pytorchvideo.txt'
_CLIP_DURATION = 2  # Duration of sampled clip for each video in seconds
_BATCH_SIZE = 8
_NUM_WORKERS = 0  # Number of parallel processes fetching data

regression_module = GazePredictionLightningModule()
data_module = GazeVideoDataModule(data_path=_DATA_PATH, num_workers=_NUM_WORKERS)
trainer = pytorch_lightning.Trainer()
trainer.fit(regression_module, data_module)
