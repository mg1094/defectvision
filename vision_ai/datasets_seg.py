"""分割任务数据集"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass(frozen=True)
class SegDataConfig:
    """分割数据配置"""

    data_dir: Path
    image_size: int = 128
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False


class SegmentationDataset(Dataset):
    """
    分割数据集：图像 + Mask 对

    目录结构：
    data_dir/
    ├── train/
    │   ├── images/
    │   │   ├── 000000.png
    │   │   └── ...
    │   └── masks/
    │       ├── 000000.png
    │       └── ...
    ├── val/
    └── test/
    """

    def __init__(
        self,
        root: Path,
        image_size: int = 128,
        train: bool = False,
    ):
        self.root = root
        self.image_size = image_size
        self.train = train

        self.images_dir = root / "images"
        self.masks_dir = root / "masks"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        # 收集图像文件
        self.image_files: List[Path] = sorted(self.images_dir.glob("*.png"))
        if len(self.image_files) == 0:
            self.image_files = sorted(self.images_dir.glob("*.jpg"))

        # 数据增强
        self.img_transform = self._build_img_transform(train)

    def _build_img_transform(self, train: bool) -> transforms.Compose:
        tfms = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
        return transforms.Compose(tfms)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        # 加载图像
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        # 加载 mask
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
        else:
            # 无 mask 则为全零（OK 样本）
            mask = torch.zeros(self.image_size, self.image_size)

        # 数据增强（训练时随机翻转）
        if self.train and torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if self.train and torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[1])
            mask = torch.flip(mask, dims=[0])

        return img, mask.unsqueeze(0)  # mask: 1xHxW


def build_seg_dataloaders(cfg: SegDataConfig) -> Dict[str, DataLoader]:
    """构建分割数据加载器"""
    loaders = {}

    for split in ["train", "val", "test"]:
        split_dir = cfg.data_dir / split
        if not split_dir.exists():
            continue

        ds = SegmentationDataset(
            root=split_dir,
            image_size=cfg.image_size,
            train=(split == "train"),
        )

        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

    return loaders

