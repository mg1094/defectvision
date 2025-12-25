"""YOLO 目标检测训练脚本（基于 ultralytics）"""

import argparse
from pathlib import Path

from defectvision.logger import get_logger, setup_logger
from defectvision.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 缺陷检测训练")
    parser.add_argument("--data", type=str, required=True, help="数据集 YAML 配置文件路径")
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="预训练模型 (yolov8n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--image-size", type=int, default=640, help="图片尺寸")
    parser.add_argument("--device", type=str, default="0", help="设备 (0, 1, cpu)")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心轮数")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续训练")
    args = parser.parse_args()

    setup_logger("defectvision")
    logger = get_logger("defectvision")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics not installed. Install with:\n"
            "  pip install ultralytics"
        )

    # 检查数据配置
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    logger.info("=" * 50)
    logger.info("YOLO Defect Detection Training")
    logger.info("=" * 50)
    logger.info(f"Data config: {data_yaml}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {out_dir}")

    # 加载模型
    model = YOLO(args.model)

    # 训练
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.image_size,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(out_dir),
        name="train",
        exist_ok=True,
        resume=args.resume,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )

    # 验证
    logger.info("\nValidating on test set...")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
    )

    logger.info("\nTest Results:")
    logger.info(f"  mAP50: {metrics.box.map50:.4f}")
    logger.info(f"  mAP50-95: {metrics.box.map:.4f}")

    # 导出最佳模型
    best_model_path = out_dir / "train" / "weights" / "best.pt"
    if best_model_path.exists():
        logger.info(f"\nBest model saved to: {best_model_path}")

        # 导出 ONNX
        logger.info("Exporting to ONNX...")
        model = YOLO(str(best_model_path))
        onnx_path = model.export(format="onnx", imgsz=args.image_size, simplify=True)
        logger.info(f"ONNX exported to: {onnx_path}")

    logger.info("\nTraining completed!")


if __name__ == "__main__":
    main()

