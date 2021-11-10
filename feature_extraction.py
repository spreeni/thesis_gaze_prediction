import torch
import timm

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from torchinfo import summary


_OUTPUT_CHANNELS = 4
_BATCH_SIZE = 4
_INPUT_DIM = (720, 1280)


class MobileNetV3WithFPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Get a MobileNet v3 backbone
        m = timm.create_model('mobilenetv3_large_100', pretrained=True)

        # Freeze parameters
        for param in m.parameters():
            param.requires_grad = False

        # Extract 6 main layers
        self.body = create_feature_extractor(
            m, return_nodes={f'blocks.{i}': str(i)
                             for i in range(1, 7)})
        # Dry run to get number of channels for FPN
        inp = torch.randn(_BATCH_SIZE, 3, *_INPUT_DIM)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]

        # Build FPN
        self.out_channels = 1
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool())

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)

        # Concatenate all outputs
        #x = torch.cat([t.flatten(start_dim=1) for t in x.values()], 1)
        return x


### Testing the FPN ###
m = MobileNetV3WithFPN()
summary(m, input_size=(_BATCH_SIZE, 3, *_INPUT_DIM))

inp = torch.randn(_BATCH_SIZE, 3, *_INPUT_DIM)
out = m(inp)

print("\n".join([f"{l}: {out[l].shape}" for l in out]))

feature_tensor = torch.cat([t.flatten(start_dim=1) for t in out.values()], 1)
print(feature_tensor.shape)
