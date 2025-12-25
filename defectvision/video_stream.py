"""实时视频流检测：支持摄像头、视频文件、RTSP 流"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from defectvision.datasets import build_transforms
from defectvision.logger import get_logger, setup_logger
from defectvision.model import build_model
from defectvision.utils import device_from_arg, load_ckpt


class VideoDetector:
    """视频流缺陷检测器"""

    def __init__(
        self,
        ckpt_path: Path,
        device: str = "auto",
        threshold: float = 0.5,
    ):
        self.logger = get_logger("defectvision")
        self.device = device_from_arg(device)
        self.threshold = threshold

        # 加载模型
        self.logger.info(f"Loading model from: {ckpt_path}")
        ckpt = load_ckpt(ckpt_path, map_location=str(self.device))

        self.idx_to_class: Dict[int, str] = ckpt["idx_to_class"]
        self.image_size: int = int(ckpt["image_size"])
        backbone: str = ckpt.get("backbone", "smallcnn")
        pretrained: bool = ckpt.get("pretrained", True)

        self.model = build_model(
            in_channels=1,
            num_classes=len(self.idx_to_class),
            backbone=backbone,
            pretrained=pretrained,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # 预处理
        self.transform = build_transforms(self.image_size, train=False)

        # 找到 NG 类别索引
        self.ng_idx = self._find_class_idx("ng")

        self.logger.info(f"Model loaded: {backbone}, classes={list(self.idx_to_class.values())}")

    def _find_class_idx(self, name: str) -> int:
        """查找类别索引"""
        for k, v in self.idx_to_class.items():
            if v.lower() == name.lower():
                return k
        return max(self.idx_to_class.keys())

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """预处理帧"""
        from PIL import Image

        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # 应用变换
        x = self.transform(pil)
        return x.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> Dict:
        """
        预测单帧

        Args:
            frame: BGR 格式的 OpenCV 帧

        Returns:
            预测结果字典
        """
        t0 = time.perf_counter()

        x = self.preprocess(frame)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        pred_idx = int(probs.argmax(dim=1).item())
        pred_class = self.idx_to_class[pred_idx]
        pred_prob = float(probs[0, pred_idx].item())
        ng_prob = float(probs[0, self.ng_idx].item())

        # 基于阈值判定
        is_ng = ng_prob >= self.threshold

        latency = (time.perf_counter() - t0) * 1000

        return {
            "prediction": pred_class,
            "confidence": pred_prob,
            "ng_prob": ng_prob,
            "is_ng": is_ng,
            "latency_ms": latency,
        }

    def draw_result(
        self,
        frame: np.ndarray,
        result: Dict,
        show_fps: bool = True,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        在帧上绘制结果

        Args:
            frame: 原始帧
            result: 预测结果
            show_fps: 是否显示 FPS
            fps: 当前 FPS

        Returns:
            绘制后的帧
        """
        h, w = frame.shape[:2]

        # 状态颜色
        if result["is_ng"]:
            color = (0, 0, 255)  # 红色 - NG
            status = "NG"
        else:
            color = (0, 255, 0)  # 绿色 - OK
            status = "OK"

        # 绘制边框
        thickness = max(2, int(min(h, w) * 0.01))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)

        # 绘制状态标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, min(h, w) / 500)
        label = f"{status} ({result['confidence']:.1%})"

        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 2)
        cv2.rectangle(frame, (0, 0), (text_w + 20, text_h + 20), color, -1)
        cv2.putText(frame, label, (10, text_h + 10), font, font_scale, (255, 255, 255), 2)

        # 绘制 FPS
        if show_fps and fps > 0:
            fps_label = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_label, (10, h - 10), font, font_scale * 0.8, (255, 255, 255), 2)

        # 绘制延迟
        latency_label = f"Latency: {result['latency_ms']:.1f}ms"
        (lw, lh), _ = cv2.getTextSize(latency_label, font, font_scale * 0.6, 1)
        cv2.putText(frame, latency_label, (w - lw - 10, h - 10), font, font_scale * 0.6, (255, 255, 255), 1)

        return frame


def run_video_detection(
    detector: VideoDetector,
    source: str,
    output_path: Optional[Path] = None,
    show: bool = True,
    max_fps: float = 30.0,
) -> None:
    """
    运行视频流检测

    Args:
        detector: 检测器实例
        source: 视频源（0=摄像头, 文件路径, RTSP URL）
        output_path: 输出视频路径（可选）
        show: 是否显示窗口
        max_fps: 最大处理帧率
    """
    logger = get_logger("defectvision")

    # 打开视频源
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        logger.info(f"Opening camera: {source}")
    else:
        cap = cv2.VideoCapture(source)
        logger.info(f"Opening video: {source}")

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    logger.info(f"Video size: {width}x{height}, FPS: {input_fps:.1f}")

    # 输出视频
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, min(input_fps, max_fps), (width, height))
        logger.info(f"Output video: {output_path}")

    # 统计
    frame_count = 0
    ng_count = 0
    total_latency = 0.0
    fps_counter = 0
    fps_time = time.time()
    current_fps = 0.0

    min_frame_time = 1.0 / max_fps

    try:
        while True:
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # 预测
            result = detector.predict(frame)
            frame_count += 1
            total_latency += result["latency_ms"]

            if result["is_ng"]:
                ng_count += 1

            # 绘制结果
            vis_frame = detector.draw_result(frame.copy(), result, show_fps=True, fps=current_fps)

            # 显示
            if show:
                cv2.imshow("Vision AI - Defect Detection", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q 或 ESC 退出
                    break

            # 保存
            if writer:
                writer.write(vis_frame)

            # 计算 FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()

            # 帧率控制
            elapsed = time.time() - frame_start
            if elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    # 统计结果
    logger.info("=" * 50)
    logger.info("Detection Summary:")
    logger.info(f"  Total frames: {frame_count}")
    logger.info(f"  NG frames: {ng_count} ({ng_count / max(1, frame_count) * 100:.1f}%)")
    logger.info(f"  Average latency: {total_latency / max(1, frame_count):.1f} ms")
    logger.info(f"  Average FPS: {frame_count / max(1, time.time() - fps_time):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="实时视频流缺陷检测")
    parser.add_argument("--ckpt", type=str, required=True, help="模型检查点路径")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="视频源：0=默认摄像头, 1=第二摄像头, 文件路径, RTSP URL",
    )
    parser.add_argument("--output", type=str, default=None, help="输出视频路径（可选）")
    parser.add_argument("--threshold", type=float, default=0.5, help="NG 判定阈值")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cpu/cuda/mps)")
    parser.add_argument("--no-show", action="store_true", help="不显示窗口（用于服务器）")
    parser.add_argument("--max-fps", type=float, default=30.0, help="最大处理帧率")
    args = parser.parse_args()

    setup_logger("defectvision")
    logger = get_logger("defectvision")

    # 创建检测器
    detector = VideoDetector(
        ckpt_path=Path(args.ckpt),
        device=args.device,
        threshold=args.threshold,
    )

    # 运行检测
    output_path = Path(args.output) if args.output else None

    logger.info(f"Starting video detection...")
    logger.info(f"  Source: {args.source}")
    logger.info(f"  Threshold: {args.threshold}")
    logger.info(f"  Press 'q' or ESC to quit")

    run_video_detection(
        detector=detector,
        source=args.source,
        output_path=output_path,
        show=not args.no_show,
        max_fps=args.max_fps,
    )


if __name__ == "__main__":
    main()

