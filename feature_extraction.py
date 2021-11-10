import torch
import timm
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

import numpy as np
from torchinfo import summary

class MobileNetV3WithFPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Get a resnet50 backbone
        m = timm.create_model('mobilenetv3_large_100', pretrained=True)

        # Freeze parameters
        for param in m.parameters():
            param.requires_grad = False

        # Extract 6 main layers
        self.body = create_feature_extractor(
            m, return_nodes={f'blocks.{i}': str(i)
                             for i in range(1, 7)})
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, 720, 1280)
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
        return x


m = MobileNetV3WithFPN()
summary(m, input_size=(2, 3, 720, 1280))

inp = torch.randn(2, 3, 720, 1280)
out = m(inp)

print("\n".join([f"{l}: {out[l].shape}" for l in out]))

tensors = list(out.values())
tensors_reshaped = []
for t in tensors:
    b, c, h, w = t.shape
    t = t.reshape(b, c*h*w)
    tensors_reshaped.append(t)

feature_tensor = torch.cat(tensors_reshaped, 1)
print(feature_tensor.shape)
