import torch
import timm

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import torchvision.models as models

class FeatureExtractor(torch.nn.Module):
    """
    Bottom-Up feature extractor based on Mobilenet v3 Large. Extracts the features of layer 2-7

    Example Usage:
        inp = torch.randn(batch_size, 3, *input_dim)
        extractor = FeatureExtractor(input_dim, batch_size)
        features = extractor(inp)
    """
    def __init__(self, device, input_dim=(224, 224), batch_size=4, model='mobilenetv3_large_100'):
        super().__init__()
        self.device = device

        # Get a ResNet backbone
        if model == 'mobilenetv3_large_100':
            m = timm.create_model('mobilenetv3_large_100', pretrained=True)
        elif model == 'efficientnet_b0':
            m = models.efficientnet_b0(pretrained=True)
        else:
            raise f"Unknown model name '{model}' given to FeatureExtractor"

        # Freeze parameters in backbone (only train top-down convolutions)
        for param in m.parameters():
            param.requires_grad = False

        # Extract features after each main layer
        if model == 'mobilenetv3_large_100':
            return_nodes = {f'blocks.{i}': str(i) for i in range(1, 7)}
        elif model == 'efficientnet_b0':
            return_nodes = {f'features.{i}': str(i) for i in range(1, 8)}
        self.body = create_feature_extractor(
            m, return_nodes=return_nodes).to(device=self.device)

        # Dry run to get number of channels for FPN
        inp = torch.randn(batch_size, 3, *input_dim, device=self.device)
        with torch.no_grad():
            out = self.body(inp)
        self._in_channels_list = [o.shape[1] for o in out.values()]

    def forward(self, x):
        return self.body(x.to(device=self.device))

    @property
    def in_channels(self):
        return self._in_channels_list


class FPN(torch.nn.Module):
    """
    Top-down Feature Pyramid Network. Processes feature maps from different layers from a bottom-up process

    in_channels can be given, but are constant for the Mobilenet v3 Large backbone

    only_use_last_layer specifies if only the bottom-most layer features should be output or a concatenation of all
    resulting layer features
    """
    def __init__(self, device, in_channels_list=None, out_channels=16, only_use_last_layer=False, separate_channels=True):
        super().__init__()

        # Build FPN
        if in_channels_list is None:
            in_channels_list = [24, 40, 80, 112, 160, 960]  # for 6 layers of mobilenetv3_large_100
        self.out_channels = out_channels
        self.only_use_last_layer = only_use_last_layer
        self.separate_channels = separate_channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels).to(device=device)

    def forward(self, x, return_channels=False):
        x = self.fpn(x)
        ch_data = x.copy()
        flatten_start_d = 2 if self.separate_channels else 1
        if self.only_use_last_layer:  # Output last layer of FPN
            ch_data = {list(ch_data.keys())[0]: ch_data[list(ch_data.keys())[0]]}
            x = x[list(x.keys())[0]].flatten(start_dim=flatten_start_d)
        else:                         # Concatenate outputs of all FPN layers
            x = torch.cat([t.flatten(start_dim=flatten_start_d) for t in x.values()], flatten_start_d)
        if return_channels:
            return x, ch_data
        else:
            return x
