"""异常检测模型训练脚本"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from defectvision.datasets_anomaly import AnomalyDataConfig, build_anomaly_dataloaders
from defectvision.logger import setup_logger
from defectvision.models.autoencoder import build_anomaly_model, vae_loss
from defectvision.utils import device_from_arg, ensure_dir, get_device_info, save_ckpt, save_json


def _compute_anomaly_scores(
    model: nn.Module,
    loader,
    device: torch.device,
    use_vae: bool = False,
) -> Tuple[List[float], List[int]]:
    """计算所有样本的异常分数"""
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)

            if use_vae:
                recon, _, _ = model(x)
            else:
                recon, _ = model(x)

            # 异常分数 = 重建误差 (MSE per sample)
            mse = F.mse_loss(recon, x, reduction="none")
            mse = mse.view(mse.size(0), -1).mean(dim=1)

            scores.extend(mse.cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())

    return scores, labels


def _compute_threshold(scores: List[float], labels: List[int], target_recall: float = 0.95) -> float:
    """根据目标召回率计算阈值"""
    # 收集 NG 样本的分数
    ng_scores = [s for s, l in zip(scores, labels) if l == 1]
    if len(ng_scores) == 0:
        return float(np.percentile(scores, 95))

    # 排序并找到能达到目标召回率的阈值
    ng_scores_sorted = sorted(ng_scores)
    idx = int((1 - target_recall) * len(ng_scores_sorted))
    return ng_scores_sorted[max(0, idx)]


def _train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_vae: bool = False,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    n_batches = 0

    for x, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            if use_vae:
                recon, mu, logvar = model(x)
                loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=1.0)
            else:
                recon, _ = model(x)
                loss = F.mse_loss(recon, x)
                recon_loss = loss
                kl_loss = torch.tensor(0.0)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        if use_vae:
            total_kl_loss += kl_loss.item()
        n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon_loss / n_batches,
        "kl_loss": total_kl_loss / n_batches if use_vae else 0.0,
    }


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    use_vae: bool = False,
) -> Dict[str, float]:
    """评估一个 epoch（计算 AUROC）"""
    scores, labels = _compute_anomaly_scores(model, loader, device, use_vae)

    # 计算指标
    result = {"mean_score": float(np.mean(scores))}

    if len(set(labels)) > 1:  # 有 OK 和 NG
        auroc = roc_auc_score(labels, scores)
        result["auroc"] = auroc

        # 各类平均分数
        ok_scores = [s for s, l in zip(scores, labels) if l == 0]
        ng_scores = [s for s, l in zip(scores, labels) if l == 1]
        result["ok_mean"] = float(np.mean(ok_scores)) if ok_scores else 0.0
        result["ng_mean"] = float(np.mean(ng_scores)) if ng_scores else 0.0

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="异常检测模型训练（只用OK样本）")
    parser.add_argument("--data", type=str, required=True, help="数据集目录")
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=128, help="潜在空间维度")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20, help="早停耐心轮数")
    parser.add_argument("--model", type=str, default="vae", choices=["ae", "vae"])
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    logger = setup_logger("defectvision", log_file=out_dir / "train_anomaly.log")

    logger.info(f"Device info: {get_device_info()}")

    torch.manual_seed(args.seed)

    device = device_from_arg(args.device)
    logger.info(f"Using device: {device}")

    use_amp = args.amp and device.type == "cuda"
    use_vae = args.model == "vae"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # TensorBoard
    writer: Optional[SummaryWriter] = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))

    # 数据
    loaders = build_anomaly_dataloaders(
        AnomalyDataConfig(
            data_dir=Path(args.data),
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    )
    logger.info(f"Train samples (OK only): {len(loaders['train'].dataset)}")
    if "val" in loaders:
        logger.info(f"Val samples (OK + NG): {len(loaders['val'].dataset)}")

    # 模型
    model = build_anomaly_model(
        in_channels=1,
        latent_dim=args.latent_dim,
        base_channels=32,
        variant=args.model,
    ).to(device)
    logger.info(f"Model: {args.model}, latent_dim={args.latent_dim}")

    # 训练配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if use_amp else None

    best_auroc = -1.0
    best_epoch = 0
    metrics_history = {}

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_epoch(model, loaders["train"], optimizer, device, use_vae, scaler, use_amp)
        scheduler.step()

        val_metrics = {}
        if "val" in loaders:
            val_metrics = _eval_epoch(model, loaders["val"], device, use_vae)

        current_lr = scheduler.get_last_lr()[0]
        metrics_history[str(epoch)] = {
            "train_loss": train_metrics["loss"],
            "train_recon_loss": train_metrics["recon_loss"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": current_lr,
        }

        log_msg = f"Epoch {epoch:03d} | loss={train_metrics['loss']:.4f} recon={train_metrics['recon_loss']:.4f}"
        if use_vae:
            log_msg += f" kl={train_metrics['kl_loss']:.4f}"
        if "auroc" in val_metrics:
            log_msg += f" | val_auroc={val_metrics['auroc']:.4f}"
        logger.info(log_msg)

        if writer is not None:
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/recon", train_metrics["recon_loss"], epoch)
            if use_vae:
                writer.add_scalar("Loss/kl", train_metrics["kl_loss"], epoch)
            if "auroc" in val_metrics:
                writer.add_scalar("AUROC/val", val_metrics["auroc"], epoch)
            writer.add_scalar("LearningRate", current_lr, epoch)

        # 保存最佳模型
        current_auroc = val_metrics.get("auroc", train_metrics["loss"] * -1)
        if current_auroc > best_auroc:
            best_auroc = current_auroc
            best_epoch = epoch
            save_ckpt(
                out_dir / "best.pt",
                {
                    "model_state_dict": model.state_dict(),
                    "model_variant": args.model,
                    "latent_dim": args.latent_dim,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "auroc": val_metrics.get("auroc", 0.0),
                },
            )
            logger.info(f"  -> New best model saved (auroc={val_metrics.get('auroc', 0.0):.4f})")

        if epoch - best_epoch >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t0

    # 测试评估
    if "test" in loaders:
        logger.info("Evaluating on test set...")
        test_metrics = _eval_epoch(model, loaders["test"], device, use_vae)

        # 计算最佳阈值
        scores, labels = _compute_anomaly_scores(model, loaders["test"], device, use_vae)
        threshold = _compute_threshold(scores, labels, target_recall=0.95)

        # 基于阈值计算指标
        predictions = [1 if s > threshold else 0 for s in scores]
        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        test_metrics["threshold"] = threshold
        test_metrics["precision"] = precision
        test_metrics["recall"] = recall
        test_metrics["f1"] = f1

        logger.info(
            f"Test | auroc={test_metrics.get('auroc', 0.0):.4f} "
            f"threshold={threshold:.6f} "
            f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
        )
        save_json(out_dir / "test.json", test_metrics)

    save_json(out_dir / "metrics.json", {"metrics": metrics_history, "best_auroc": best_auroc, "elapsed_sec": elapsed})

    if writer is not None:
        writer.close()

    logger.info(f"Training completed in {elapsed:.1f}s. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

