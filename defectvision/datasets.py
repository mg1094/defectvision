"""数据集加载与增强模块"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataConfig:
    """数据加载配置"""

    data_dir: Path
    image_size: int = 128
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False


def build_transforms(image_size: int, train: bool = False) -> transforms.Compose:
    """
    构建数据变换管道。

    Args:
        image_size: 目标图像尺寸
        train: 是否为训练模式（启用数据增强）

    Returns:
        transforms.Compose 实例
    """
    aug = []
    if train:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        ]

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def build_dataloaders(cfg: DataConfig) -> Tuple[Dict[str, DataLoader], Dict[int, str]]:
    """
    构建训练/验证/测试数据加载器。

    Args:
        cfg: 数据配置

    Returns:
        (loaders, idx_to_class): 数据加载器字典和类别映射
    """
    train_tfm = build_transforms(cfg.image_size, train=True)
    eval_tfm = build_transforms(cfg.image_size, train=False)

    train_dir = cfg.data_dir / "train"
    val_dir = cfg.data_dir / "val"
    test_dir = cfg.data_dir / "test"

    # 检查目录是否存在
    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Data directory not found: {d}")

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tfm)
    val_ds = datasets.ImageFolder(root=str(val_dir), transform=eval_tfm)
    test_ds = datasets.ImageFolder(root=str(test_dir), transform=eval_tfm)

    # 优化 DataLoader 配置
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
    }

    # 多 worker 时启用优化选项
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        loader_kwargs["persistent_workers"] = cfg.persistent_workers

    loaders = {
        "train": DataLoader(train_ds, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_ds, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_ds, shuffle=False, **loader_kwargs),
    }

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    return loaders, idx_to_class
