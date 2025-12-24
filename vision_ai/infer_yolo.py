"""YOLO 目标检测推理脚本"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from vision_ai.logger import get_logger, setup_logger
from vision_ai.utils import ensure_dir

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 缺陷类别颜色
COLORS = {
    "scratch": (0, 165, 255),  # 橙色
    "spot": (0, 0, 255),  # 红色
    "crack": (255, 0, 255),  # 紫色
    "dent": (255, 0, 0),  # 蓝色
}


def _iter_images(img_dir: Path) -> Iterable[Path]:
    """递归遍历目录下的所有图片"""
    for p in sorted(img_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: List[int],
    scores: List[float],
    class_names: List[str],
) -> np.ndarray:
    """
    在图像上绘制检测结果

    Args:
        image: BGR 图像
        boxes: 边界框 [N, 4] (x1, y1, x2, y2)
        classes: 类别索引列表
        scores: 置信度列表
        class_names: 类别名称列表

    Returns:
        绘制后的图像
    """
    result = image.copy()

    for box, cls_id, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        color = COLORS.get(cls_name, (0, 255, 0))

        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = f"{cls_name} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(result, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(result, label, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), thickness)

    return result


class YOLODetector:
    """YOLO 检测器封装"""

    def __init__(self, model_path: Path, device: str = "0", conf: float = 0.25, iou: float = 0.45):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

        self.logger = get_logger("vision_ai")
        self.logger.info(f"Loading YOLO model: {model_path}")

        self.model = YOLO(str(model_path))
        self.device = device
        self.conf = conf
        self.iou = iou
        self.class_names = self.model.names

        self.logger.info(f"Classes: {self.class_names}")

    def predict(self, image: np.ndarray) -> Dict:
        """
        执行检测

        Args:
            image: BGR 图像

        Returns:
            检测结果字典
        """
        t0 = time.perf_counter()

        results = self.model.predict(
            image,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

        latency = (time.perf_counter() - t0) * 1000

        boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else np.array([])
        classes = results.boxes.cls.cpu().numpy().astype(int).tolist() if len(results.boxes) > 0 else []
        scores = results.boxes.conf.cpu().numpy().tolist() if len(results.boxes) > 0 else []

        # 统计各类缺陷数量
        defect_counts = {}
        for cls_id in classes:
            cls_name = self.class_names[cls_id]
            defect_counts[cls_name] = defect_counts.get(cls_name, 0) + 1

        return {
            "boxes": boxes,
            "classes": classes,
            "scores": scores,
            "class_names": [self.class_names[c] for c in classes],
            "defect_counts": defect_counts,
            "total_defects": len(boxes),
            "latency_ms": latency,
        }

    def predict_and_draw(self, image: np.ndarray) -> tuple:
        """预测并绘制结果"""
        result = self.predict(image)
        vis = draw_detections(
            image,
            result["boxes"],
            result["classes"],
            result["scores"],
            list(self.class_names.values()),
        )
        return vis, result


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 缺陷检测推理")
    parser.add_argument("--model", type=str, required=True, help="YOLO 模型路径 (.pt)")
    parser.add_argument("--source", type=str, required=True, help="图片/目录/视频路径")
    parser.add_argument("--out", type=str, default=None, help="输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--device", type=str, default="0", help="设备 (0, cpu)")
    parser.add_argument("--save-csv", action="store_true", help="保存检测结果到 CSV")
    parser.add_argument("--show", action="store_true", help="显示结果窗口")
    args = parser.parse_args()

    setup_logger("vision_ai")
    logger = get_logger("vision_ai")

    # 加载模型
    detector = YOLODetector(
        model_path=Path(args.model),
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )

    source = Path(args.source)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        ensure_dir(out_dir)

    # 处理图片
    if source.is_file():
        image_paths = [source]
    elif source.is_dir():
        image_paths = list(_iter_images(source))
    else:
        raise ValueError(f"Invalid source: {source}")

    logger.info(f"Processing {len(image_paths)} images...")

    # CSV 结果
    csv_rows = []
    total_defects = 0

    for img_path in image_paths:
        # 读取图片
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Failed to read: {img_path}")
            continue

        # 检测
        vis, result = detector.predict_and_draw(image)

        total_defects += result["total_defects"]

        # 日志
        logger.info(
            f"{img_path.name}: {result['total_defects']} defects, "
            f"{result['latency_ms']:.1f}ms"
        )

        # 保存结果
        if out_dir:
            out_path = out_dir / f"{img_path.stem}_det{img_path.suffix}"
            cv2.imwrite(str(out_path), vis)

        # CSV 行
        csv_rows.append({
            "filename": str(img_path.name),
            "total_defects": result["total_defects"],
            "scratch": result["defect_counts"].get("scratch", 0),
            "spot": result["defect_counts"].get("spot", 0),
            "crack": result["defect_counts"].get("crack", 0),
            "dent": result["defect_counts"].get("dent", 0),
            "latency_ms": round(result["latency_ms"], 1),
        })

        # 显示
        if args.show:
            cv2.imshow("YOLO Detection", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q") or key == 27:
                break

    if args.show:
        cv2.destroyAllWindows()

    # 保存 CSV
    if args.save_csv and out_dir:
        csv_path = out_dir / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["filename", "total_defects", "scratch", "spot", "crack", "dent", "latency_ms"],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"Results saved to: {csv_path}")

    # 统计
    logger.info("=" * 50)
    logger.info("Detection Summary:")
    logger.info(f"  Total images: {len(image_paths)}")
    logger.info(f"  Total defects: {total_defects}")
    logger.info(f"  Avg defects/image: {total_defects / max(1, len(image_paths)):.2f}")


if __name__ == "__main__":
    main()

