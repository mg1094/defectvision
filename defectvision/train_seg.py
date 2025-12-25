"""分割模型训练脚本"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from defectvision.datasets_seg import SegDataConfig, build_seg_dataloaders
from defectvision.logger import setup_logger
from defectvision.models.unet import build_unet
from defectvision.utils import device_from_arg, ensure_dir, get_device_info, save_ckpt, save_json


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class DiceBCELoss(nn.Module):
    """组合损失：Dice + BCE"""

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(pred, target) + (1 - self.dice_weight) * self.bce(
            pred, target
        )


def _compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """计算分割指标"""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target_bin = target.float()

    # Flatten
    pred_flat = pred_bin.view(-1)
    target_flat = target_bin.view(-1)

    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()

    # Dice / F1
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    # IoU
    iou = tp / (tp + fp + fn + 1e-8)

    # Precision / Recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    # Pixel Accuracy
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "pixel_acc": pixel_acc,
    }


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    """评估一个 epoch"""
    model.eval()
    total_loss = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "pixel_acc": 0.0}
    n_batches = 0

    for x, mask in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            pred = model(x)
            loss = criterion(pred, mask)

        total_loss += loss.item()
        batch_metrics = _compute_metrics(pred, mask)
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k]
        n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, **{k: 0.0 for k in metrics_sum}}

    return {
        "loss": total_loss / n_batches,
        **{k: v / n_batches for k, v in metrics_sum.items()},
    }


def _train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0}
    n_batches = 0

    for x, mask in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            pred = model(x)
            loss = criterion(pred, mask)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        batch_metrics = _compute_metrics(pred, mask)
        metrics_sum["dice"] += batch_metrics["dice"]
        metrics_sum["iou"] += batch_metrics["iou"]
        n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0}

    return {
        "loss": total_loss / n_batches,
        "dice": metrics_sum["dice"] / n_batches,
        "iou": metrics_sum["iou"] / n_batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="分割模型训练")
    parser.add_argument("--data", type=str, required=True, help="分割数据集目录")
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10, help="早停耐心轮数")
    parser.add_argument("--model", type=str, default="unet_small", choices=["unet", "unet_small"])
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练")
    parser.add_argument("--no-tensorboard", action="store_true", help="禁用 TensorBoard")
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    logger = setup_logger("defectvision", log_file=out_dir / "train_seg.log")

    logger.info(f"Device info: {get_device_info()}")

    torch.manual_seed(args.seed)

    device = device_from_arg(args.device)
    logger.info(f"Using device: {device}")

    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if use_amp:
            logger.info("Mixed precision training (AMP) enabled")

    # TensorBoard
    writer: Optional[SummaryWriter] = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))

    # 数据
    loaders = build_seg_dataloaders(
        SegDataConfig(
            data_dir=Path(args.data),
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    )
    logger.info(f"Train samples: {len(loaders['train'].dataset)}")
    logger.info(f"Val samples: {len(loaders['val'].dataset)}")

    # 模型
    model = build_unet(
        in_channels=1,
        num_classes=1,
        variant=args.model,
    ).to(device)
    logger.info(f"Model: {args.model}")

    # 训练配置
    criterion = DiceBCELoss(dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if use_amp else None

    best_dice = -1.0
    best_epoch = 0
    metrics_history = {}

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_epoch(model, loaders["train"], criterion, optimizer, device, scaler, use_amp)
        val_metrics = _eval_epoch(model, loaders["val"], criterion, device, use_amp)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        metrics_history[str(epoch)] = {
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "lr": current_lr,
        }

        logger.info(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("Dice/train", train_metrics["dice"], epoch)
            writer.add_scalar("Dice/val", val_metrics["dice"], epoch)
            writer.add_scalar("IoU/val", val_metrics["iou"], epoch)
            writer.add_scalar("LearningRate", current_lr, epoch)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            save_ckpt(
                out_dir / "best.pt",
                {
                    "model_state_dict": model.state_dict(),
                    "model_variant": args.model,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_dice": val_metrics["dice"],
                },
            )
            logger.info(f"  -> New best model saved (val_dice={val_metrics['dice']:.4f})")

        if epoch - best_epoch >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t0

    # 测试评估
    if "test" in loaders:
        logger.info("Evaluating on test set...")
        test_metrics = _eval_epoch(model, loaders["test"], criterion, device, use_amp)
        logger.info(
            f"Test | loss={test_metrics['loss']:.4f} dice={test_metrics['dice']:.4f} "
            f"iou={test_metrics['iou']:.4f} recall={test_metrics['recall']:.4f}"
        )
        save_json(out_dir / "test.json", test_metrics)

    save_json(out_dir / "metrics.json", {"metrics": metrics_history, "best_dice": best_dice, "elapsed_sec": elapsed})

    if writer is not None:
        writer.close()

    logger.info(f"Training completed in {elapsed:.1f}s. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

