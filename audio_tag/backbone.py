# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from sedt.backbone import FrozenBatchNorm2d



class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, fix_backbone: bool, num_channels: int, pooling):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            # freeze backbone while training audio tag classifier
            if fix_backbone:
                parameter.requires_grad_(False)

        return_layers = {'layer4': "0"}
        return_layers[f"{pooling}pool_"] = str(int(return_layers["layer4"]) + 1)
        self.weak_label = torch.nn.Sequential(
            torch.nn.Linear(num_channels, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 10)
        )
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x):
        xs = self.body(x)
        x = xs['1'].flatten(1)  # cnn_at
        at = self.weak_label(x)
        at = at.sigmoid()
        return at



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 fix_backbone: bool,
                 dilation: bool,
                 pooling: str,
                 pretrained: bool
                 ):
        backbone = nn.Sequential()
        backbone.add_module('conv0', nn.Conv2d(1, 3, 1))
        resnet = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=FrozenBatchNorm2d)
        for name, module in resnet.named_children():
            if "avgpool" in name :
                if "max" in pooling :
                    backbone.add_module('maxpool_', nn.AdaptiveMaxPool2d(output_size=(1, 1)))
                else:
                    backbone.add_module('avgpool_', module)
            else:
                backbone.add_module(name, module)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, fix_backbone, num_channels, pooling)




def build_backbone(args):
    model = Backbone(args.backbone, args.fix_backbone, args.dilation, args.pooling, args.pretrained)
    return model
