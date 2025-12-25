"""异常检测模型：AutoEncoder / VAE"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AEConfig:
    """AutoEncoder 配置"""

    in_channels: int = 1
    latent_dim: int = 128
    base_channels: int = 32


class ConvAutoEncoder(nn.Module):
    """
    卷积自编码器用于异常检测

    训练时只用 OK 样本，测试时通过重建误差检测异常
    """

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.base_channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.in_channels, c, 4, 2, 1),  # /2
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1),  # /4
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),  # /8
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1),  # /16
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck
        self.fc_encode = nn.Linear(c * 8 * 8 * 8, cfg.latent_dim)  # 假设输入 128x128
        self.fc_decode = nn.Linear(cfg.latent_dim, c * 8 * 8 * 8)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1),  # x2
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1),  # x4
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1),  # x8
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, cfg.in_channels, 4, 2, 1),  # x16
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc_encode(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.cfg.base_channels * 8, 8, 8)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class ConvVAE(nn.Module):
    """
    变分自编码器 (VAE)

    相比 AE，VAE 的潜在空间更规整，异常检测效果通常更好
    """

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.base_channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.in_channels, c, 4, 2, 1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_mu = nn.Linear(c * 8 * 8 * 8, cfg.latent_dim)
        self.fc_logvar = nn.Linear(c * 8 * 8 * 8, cfg.latent_dim)
        self.fc_decode = nn.Linear(cfg.latent_dim, c * 8 * 8 * 8)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, cfg.in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.cfg.base_channels * 8, 8, 8)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE 损失函数：重建损失 + KL 散度

    Args:
        recon: 重建图像
        x: 原始图像
        mu: 均值
        logvar: 对数方差
        beta: KL 权重

    Returns:
        total_loss, recon_loss, kl_loss
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def build_anomaly_model(
    in_channels: int = 1,
    latent_dim: int = 128,
    base_channels: int = 32,
    variant: str = "ae",
) -> nn.Module:
    """
    构建异常检测模型

    Args:
        in_channels: 输入通道数
        latent_dim: 潜在空间维度
        base_channels: 基础通道数
        variant: 模型变体 (ae / vae)

    Returns:
        AutoEncoder 或 VAE 模型
    """
    cfg = AEConfig(in_channels=in_channels, latent_dim=latent_dim, base_channels=base_channels)
    if variant == "ae":
        return ConvAutoEncoder(cfg)
    elif variant == "vae":
        return ConvVAE(cfg)
    else:
        raise ValueError(f"Unknown variant: {variant}")

