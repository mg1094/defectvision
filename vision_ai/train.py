"""训练脚本：支持混合精度、早停、TensorBoard 等"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vision_ai.datasets import DataConfig, build_dataloaders
from vision_ai.logger import setup_logger
from vision_ai.model import build_model
from vision_ai.utils import ensure_dir, get_device_info, save_ckpt, save_json


@dataclass
class EpochStats:
    """单个 epoch 的统计信息"""

    loss: float
    acc: float


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """计算准确率"""
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


@torch.no_grad()
def _eval_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> EpochStats:
    """评估一个 epoch"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        total_loss += float(loss.item())
        total_acc += _accuracy(logits, y)
        n_batches += 1

    if n_batches == 0:
        return EpochStats(loss=0.0, acc=0.0)
    return EpochStats(loss=total_loss / n_batches, acc=total_acc / n_batches)


def _train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
) -> EpochStats:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_acc += _accuracy(logits, y)
        n_batches += 1

    if n_batches == 0:
        return EpochStats(loss=0.0, acc=0.0)
    return EpochStats(loss=total_loss / n_batches, acc=total_acc / n_batches)


@torch.no_grad()
def _confusion_matrix(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int = 2,
) -> torch.Tensor:
    """计算混淆矩阵"""
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = torch.argmax(model(x), dim=1)
        for t, p in zip(y.view(-1), preds.view(-1)):
            cm[int(t), int(p)] += 1
    return cm


def _format_cm(cm: torch.Tensor, idx_to_class: Dict[int, str]) -> str:
    """格式化混淆矩阵为字符串"""
    header = "      " + "  ".join([f"pred={idx_to_class[i]:>2}" for i in range(cm.shape[1])])
    lines = [header]
    for i in range(cm.shape[0]):
        row = "  ".join([f"{int(cm[i, j]):6d}" for j in range(cm.shape[1])])
        lines.append(f"true={idx_to_class[i]:>2} {row}")
    return "\n".join(lines)


def _safe_div(num: float, den: float) -> float:
    """安全除法，避免除零"""
    return float(num) / float(den) if den != 0 else 0.0


def _cm_metrics(cm: torch.Tensor, idx_to_class: Dict[int, str]) -> Dict[str, object]:
    """从混淆矩阵计算各种指标"""
    num_classes = cm.shape[0]
    per_class: Dict[str, Dict[str, float]] = {}
    recalls, precisions, f1s = [], [], []

    for i in range(num_classes):
        tp = float(cm[i, i].item())
        fp = float(cm[:, i].sum().item() - tp)
        fn = float(cm[i, :].sum().item() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
        per_class[idx_to_class[i]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum().item()),
        }
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    acc = _safe_div(float(cm.trace().item()), float(cm.sum().item()))
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    balanced_acc = macro_recall

    ng_recall: Optional[float] = per_class.get("ng", {}).get("recall") if "ng" in per_class else None

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "ng_recall": ng_recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 OK/NG 缺陷二分类（合成数据/真实数据均可）")
    parser.add_argument("--data", type=str, required=True, help="数据集目录（包含 train/val/test 子目录）")
    parser.add_argument("--out", type=str, required=True, help="输出目录，例如 ./runs/exp1")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5, help="早停耐心轮数（验证集未提升时停止）")
    parser.add_argument(
        "--backbone",
        type=str,
        default="smallcnn",
        choices=["smallcnn", "resnet18", "mobilenet_v3_small"],
    )
    parser.add_argument("--no-pretrained", action="store_true", help="关闭预训练权重")
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练（仅 CUDA）")
    parser.add_argument("--no-tensorboard", action="store_true", help="禁用 TensorBoard 日志")
    args = parser.parse_args()

    # 设置输出目录和日志
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    logger = setup_logger("vision_ai", log_file=out_dir / "train.log")

    logger.info(f"Device info: {get_device_info()}")

    # 设置随机种子
    torch.manual_seed(int(args.seed))

    # 设备配置
    from vision_ai.utils import device_from_arg

    device = device_from_arg(args.device)
    logger.info(f"Using device: {device}")

    # CUDA 优化
    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if use_amp:
            logger.info("Mixed precision training (AMP) enabled")

    # TensorBoard
    writer: Optional[SummaryWriter] = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
        logger.info(f"TensorBoard logs: {out_dir / 'tensorboard'}")

    # 数据加载
    loaders, idx_to_class = build_dataloaders(
        DataConfig(
            data_dir=Path(args.data),
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
            persistent_workers=int(args.num_workers) > 0,
        )
    )
    logger.info(f"Classes: {idx_to_class}")
    logger.info(f"Train samples: {len(loaders['train'].dataset)}")
    logger.info(f"Val samples: {len(loaders['val'].dataset)}")
    logger.info(f"Test samples: {len(loaders['test'].dataset)}")

    # 模型
    model = build_model(
        in_channels=1,
        num_classes=len(idx_to_class),
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
    ).to(device)
    logger.info(f"Model: {args.backbone}, pretrained={not args.no_pretrained}")

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs))
    scaler = GradScaler() if use_amp else None

    best_val_acc = -1.0
    best_epoch = 0
    metrics: Dict[str, Dict[str, float]] = {}

    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        train_stats = _train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler=scaler, use_amp=use_amp
        )
        val_stats = _eval_one_epoch(model, loaders["val"], criterion, device, use_amp=use_amp)

        # 每个 epoch 结束后更新学习率（修复：之前在每个 batch 后调用是错误的）
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        metrics[str(epoch)] = {
            "train_loss": train_stats.loss,
            "train_acc": train_stats.acc,
            "val_loss": val_stats.loss,
            "val_acc": val_stats.acc,
            "lr": current_lr,
        }

        logger.info(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats.loss:.4f} train_acc={train_stats.acc:.4f} | "
            f"val_loss={val_stats.loss:.4f} val_acc={val_stats.acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        # TensorBoard 记录
        if writer is not None:
            writer.add_scalar("Loss/train", train_stats.loss, epoch)
            writer.add_scalar("Loss/val", val_stats.loss, epoch)
            writer.add_scalar("Accuracy/train", train_stats.acc, epoch)
            writer.add_scalar("Accuracy/val", val_stats.acc, epoch)
            writer.add_scalar("LearningRate", current_lr, epoch)

        # 保存最佳模型
        if val_stats.acc > best_val_acc:
            best_val_acc = val_stats.acc
            best_epoch = epoch
            save_ckpt(
                out_dir / "best.pt",
                {
                    "model_state_dict": model.state_dict(),
                    "idx_to_class": idx_to_class,
                    "image_size": int(args.image_size),
                    "backbone": args.backbone,
                    "pretrained": not args.no_pretrained,
                    "epoch": epoch,
                    "val_acc": val_stats.acc,
                },
            )
            logger.info(f"  -> New best model saved (val_acc={val_stats.acc:.4f})")

        # 早停
        if epoch - best_epoch >= int(args.patience):
            logger.info(f"Early stopping: patience={args.patience} reached at epoch {epoch}")
            break

    elapsed = time.time() - t0
    save_json(out_dir / "metrics.json", {"metrics": metrics, "best_val_acc": best_val_acc, "elapsed_sec": elapsed})

    # 测试评估
    logger.info("Evaluating on test set...")
    test_stats = _eval_one_epoch(model, loaders["test"], criterion, device, use_amp=use_amp)
    cm = _confusion_matrix(model, loaders["test"], device, num_classes=len(idx_to_class))
    metrics_cm = _cm_metrics(cm, idx_to_class)

    ng_recall_str = f" ng_recall={metrics_cm['ng_recall']:.4f}" if metrics_cm.get("ng_recall") is not None else ""
    logger.info(
        f"Test | loss={test_stats.loss:.4f} acc={test_stats.acc:.4f} "
        f"bal_acc={metrics_cm['balanced_accuracy']:.4f} macro_f1={metrics_cm['macro_f1']:.4f}{ng_recall_str}"
    )
    logger.info(f"Confusion matrix:\n{_format_cm(cm, idx_to_class)}")

    # TensorBoard 记录测试指标
    if writer is not None:
        writer.add_scalar("Test/accuracy", test_stats.acc, 0)
        writer.add_scalar("Test/loss", test_stats.loss, 0)
        writer.add_scalar("Test/balanced_accuracy", metrics_cm["balanced_accuracy"], 0)
        writer.add_scalar("Test/macro_f1", metrics_cm["macro_f1"], 0)
        if metrics_cm.get("ng_recall") is not None:
            writer.add_scalar("Test/ng_recall", metrics_cm["ng_recall"], 0)
        writer.close()

    save_json(
        out_dir / "test.json",
        {
            "test_loss": test_stats.loss,
            "test_acc": test_stats.acc,
            "confusion_matrix": cm.cpu().tolist(),
            "idx_to_class": idx_to_class,
            "metrics": metrics_cm,
        },
    )

    logger.info(f"Training completed in {elapsed:.1f}s. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
