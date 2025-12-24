from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision import models

Backbone = Literal["smallcnn", "resnet18", "mobilenet_v3_small"]


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    num_classes: int = 2
    backbone: Backbone = "smallcnn"
    pretrained: bool = True


class SmallCNN(nn.Module):
    """
    小型 CNN：足够应付试跑的 OK/NG 二分类，同时便于做 Grad-CAM（有明确的 last_conv）。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.features = nn.Sequential(
            nn.Conv2d(cfg.in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.last_conv = self.features[-3]  # Conv2d(64->128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias,
    )
    if pretrained:
        with torch.no_grad():
            new_conv.weight[:] = conv.weight.mean(dim=1, keepdim=True)
    return new_conv


def _build_resnet18(cfg: ModelConfig) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    model = models.resnet18(weights=weights)

    if cfg.in_channels != 3:
        model.conv1 = _adapt_first_conv(model.conv1, cfg.in_channels, cfg.pretrained)

    model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
    model.last_conv = model.layer4[-1].conv2
    return model


def _build_mobilenet_v3_small(cfg: ModelConfig) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    if cfg.in_channels != 3:
        # features[0] 是 ConvNormActivation，内部 [0] 是 Conv2d
        conv = model.features[0][0]
        model.features[0][0] = _adapt_first_conv(conv, cfg.in_channels, cfg.pretrained)

    # 分类头
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, cfg.num_classes)

    # 找最后一个 conv 以便 Grad-CAM
    last_conv = None
    for m in reversed(model.features):
        if isinstance(m, nn.Conv2d):
            last_conv = m
            break
        if hasattr(m, "block") and isinstance(m.block[-1], nn.Conv2d):  # ConvNormActivation
            last_conv = m.block[-1]
            break
        if isinstance(m, nn.Sequential):
            for sub in reversed(m):
                if isinstance(sub, nn.Conv2d):
                    last_conv = sub
                    break
            if last_conv is not None:
                break
    model.last_conv = last_conv
    return model


def build_model(
    *,
    in_channels: int = 1,
    num_classes: int = 2,
    backbone: Backbone = "smallcnn",
    pretrained: bool = True,
) -> nn.Module:
    cfg = ModelConfig(in_channels=in_channels, num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    if backbone == "smallcnn":
        return SmallCNN(cfg)
    if backbone == "resnet18":
        return _build_resnet18(cfg)
    if backbone == "mobilenet_v3_small":
        return _build_mobilenet_v3_small(cfg)
    raise ValueError(f"Unsupported backbone: {backbone}")


