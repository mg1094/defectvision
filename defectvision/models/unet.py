"""U-Net 分割模型"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双层卷积块：Conv-BN-ReLU x 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """下采样块：MaxPool + DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    """上采样块：Upsample + Concat + DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net 分割模型

    Args:
        in_channels: 输入通道数（灰度图为1，RGB为3）
        num_classes: 输出类别数（二分类分割为1，多类分割为N）
        base_channels: 基础通道数
        bilinear: 是否使用双线性插值上采样
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(c * 8, c * 16 // factor)

        self.up1 = Up(c * 16, c * 8 // factor, bilinear)
        self.up2 = Up(c * 8, c * 4 // factor, bilinear)
        self.up3 = Up(c * 4, c * 2 // factor, bilinear)
        self.up4 = Up(c * 2, c, bilinear)
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetSmall(nn.Module):
    """轻量版 U-Net：更少通道数，适合小数据集/快速实验"""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)

        self.up1 = Up(c * 12, c * 4, bilinear=True)  # 8 + 4 = 12
        self.up2 = Up(c * 6, c * 2, bilinear=True)  # 4 + 2 = 6
        self.up3 = Up(c * 3, c, bilinear=True)  # 2 + 1 = 3
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)  # c
        x2 = self.down1(x1)  # c*2
        x3 = self.down2(x2)  # c*4
        x4 = self.down3(x3)  # c*8

        x = self.up1(x4, x3)  # c*4
        x = self.up2(x, x2)  # c*2
        x = self.up3(x, x1)  # c
        logits = self.outc(x)
        return logits


def build_unet(
    in_channels: int = 1,
    num_classes: int = 1,
    variant: str = "unet",
    base_channels: int = 64,
) -> nn.Module:
    """
    构建 U-Net 模型

    Args:
        in_channels: 输入通道数
        num_classes: 输出类别数
        variant: 模型变体 (unet / unet_small)
        base_channels: 基础通道数

    Returns:
        U-Net 模型
    """
    if variant == "unet":
        return UNet(in_channels, num_classes, base_channels)
    elif variant == "unet_small":
        return UNetSmall(in_channels, num_classes, base_channels)
    else:
        raise ValueError(f"Unknown variant: {variant}")

