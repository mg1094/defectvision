"""异常检测数据集：只用 OK 样本训练"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass(frozen=True)
class AnomalyDataConfig:
    """异常检测数据配置"""

    data_dir: Path
    image_size: int = 128
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False


class AnomalyDataset(Dataset):
    """
    异常检测数据集

    训练时只用 OK 样本
    测试时包含 OK + NG 样本，用于计算异常分数

    目录结构：
    data_dir/
    ├── train/
    │   └── ok/
    ├── val/
    │   ├── ok/
    │   └── ng/
    └── test/
        ├── ok/
        └── ng/
    """

    def __init__(
        self,
        root: Path,
        image_size: int = 128,
        train: bool = False,
        include_ng: bool = False,
    ):
        self.root = root
        self.image_size = image_size
        self.train = train

        # 收集图像
        self.samples: List[tuple] = []  # (path, label)

        ok_dir = root / "ok"
        if ok_dir.exists():
            for p in sorted(ok_dir.glob("*.png")):
                self.samples.append((p, 0))  # 0 = OK
            for p in sorted(ok_dir.glob("*.jpg")):
                self.samples.append((p, 0))

        if include_ng:
            ng_dir = root / "ng"
            if ng_dir.exists():
                for p in sorted(ng_dir.glob("*.png")):
                    self.samples.append((p, 1))  # 1 = NG
                for p in sorted(ng_dir.glob("*.jpg")):
                    self.samples.append((p, 1))

        # 变换
        aug = []
        if train:
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=10),
            ]

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def build_anomaly_dataloaders(cfg: AnomalyDataConfig) -> Dict[str, DataLoader]:
    """构建异常检测数据加载器"""
    loaders = {}

    # 训练集：只用 OK
    train_dir = cfg.data_dir / "train"
    if train_dir.exists():
        train_ds = AnomalyDataset(train_dir, cfg.image_size, train=True, include_ng=False)
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

    # 验证集/测试集：包含 OK + NG
    for split in ["val", "test"]:
        split_dir = cfg.data_dir / split
        if split_dir.exists():
            ds = AnomalyDataset(split_dir, cfg.image_size, train=False, include_ng=True)
            loaders[split] = DataLoader(
                ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
            )

    return loaders

