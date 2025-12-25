"""YOLO 实时视频流检测"""

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from defectvision.infer_yolo import COLORS, YOLODetector, draw_detections
from defectvision.logger import get_logger, setup_logger
from defectvision.utils import ensure_dir


def run_video_detection(
    detector: YOLODetector,
    source: str,
    output_path: Optional[Path] = None,
    show: bool = True,
    max_fps: float = 30.0,
) -> None:
    """
    运行 YOLO 视频流检测

    Args:
        detector: YOLO 检测器
        source: 视频源
        output_path: 输出视频路径
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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    logger.info(f"Video size: {width}x{height}, FPS: {input_fps:.1f}")

    # 输出视频
    writer = None
    if output_path:
        ensure_dir(output_path.parent)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, min(input_fps, max_fps), (width, height))
        logger.info(f"Output video: {output_path}")

    # 统计
    frame_count = 0
    total_defects = 0
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

            # 检测
            result = detector.predict(frame)
            frame_count += 1
            total_defects += result["total_defects"]
            total_latency += result["latency_ms"]

            # 绘制结果
            vis = draw_detections(
                frame,
                result["boxes"],
                result["classes"],
                result["scores"],
                list(detector.class_names.values()),
            )

            # 绘制统计信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            y_offset = 25

            # 缺陷数量
            defects_text = f"Defects: {result['total_defects']}"
            if result["total_defects"] > 0:
                color = (0, 0, 255)  # 红色
            else:
                color = (0, 255, 0)  # 绿色
            cv2.putText(vis, defects_text, (10, y_offset), font, font_scale, color, 2)

            # FPS
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(vis, fps_text, (10, y_offset + 25), font, font_scale, (255, 255, 255), 2)

            # 延迟
            latency_text = f"Latency: {result['latency_ms']:.1f}ms"
            cv2.putText(vis, latency_text, (10, y_offset + 50), font, font_scale, (255, 255, 255), 2)

            # 显示
            if show:
                cv2.imshow("YOLO Defect Detection", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            # 保存
            if writer:
                writer.write(vis)

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
    logger.info(f"  Total defects: {total_defects}")
    logger.info(f"  Avg defects/frame: {total_defects / max(1, frame_count):.2f}")
    logger.info(f"  Avg latency: {total_latency / max(1, frame_count):.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 实时视频流缺陷检测")
    parser.add_argument("--model", type=str, required=True, help="YOLO 模型路径")
    parser.add_argument("--source", type=str, default="0", help="视频源 (0=摄像头, 路径, RTSP)")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--device", type=str, default="0", help="设备")
    parser.add_argument("--no-show", action="store_true", help="不显示窗口")
    parser.add_argument("--max-fps", type=float, default=30.0, help="最大帧率")
    args = parser.parse_args()

    setup_logger("defectvision")
    logger = get_logger("defectvision")

    # 创建检测器
    detector = YOLODetector(
        model_path=Path(args.model),
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )

    output_path = Path(args.output) if args.output else None

    logger.info("Starting YOLO video detection...")
    logger.info(f"  Source: {args.source}")
    logger.info(f"  Confidence: {args.conf}")
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

