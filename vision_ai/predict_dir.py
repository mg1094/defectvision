"""批量推理脚本：扫描目录 → 推理 → 输出 CSV 报表"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from vision_ai.gradcam import compute_gradcam, overlay_cam, prepare_image
from vision_ai.logger import get_logger, setup_logger
from vision_ai.model import build_model
from vision_ai.utils import device_from_arg, ensure_dir, load_ckpt

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(img_dir: Path) -> Iterable[Path]:
    """递归遍历目录下的所有图片"""
    for p in sorted(img_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _find_class_idx(idx_to_class: Dict[int, str], name: str) -> int:
    """查找类别名称对应的索引"""
    for k, v in idx_to_class.items():
        if v.lower() == name.lower():
            return k
    # fallback：若无明确名称，使用最大索引
    return max(idx_to_class.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="批量推理目录，并输出 CSV + 可选 Grad-CAM")
    parser.add_argument("--ckpt", type=str, required=True, help="训练保存的 best.pt")
    parser.add_argument("--images", type=str, required=True, help="待推理图片目录（递归扫描）")
    parser.add_argument("--out-csv", type=str, required=True, help="输出 CSV 路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="判定 NG 的概率阈值（基于 prob_ng）")
    parser.add_argument("--save-cam-dir", type=str, default=None, help="若提供，则保存 Grad-CAM 叠加图")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--alpha", type=float, default=0.4, help="Grad-CAM 叠加透明度")
    args = parser.parse_args()

    setup_logger("vision_ai")
    logger = get_logger("vision_ai")

    # 加载模型
    device = device_from_arg(args.device)
    logger.info(f"Using device: {device}")

    ckpt = load_ckpt(Path(args.ckpt), map_location=str(device))
    idx_to_class: Dict[int, str] = ckpt["idx_to_class"]
    image_size: int = int(ckpt["image_size"])
    backbone: str = ckpt.get("backbone", "smallcnn")
    pretrained: bool = ckpt.get("pretrained", True)

    model = build_model(
        in_channels=1,
        num_classes=len(idx_to_class),
        backbone=backbone,
        pretrained=pretrained,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(f"Model loaded: {backbone}, classes={list(idx_to_class.values())}")

    # 检查输入目录
    img_dir = Path(args.images)
    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    # Grad-CAM 输出目录
    cam_dir = Path(args.save_cam_dir) if args.save_cam_dir else None
    if cam_dir:
        ensure_dir(cam_dir)
        logger.info(f"Grad-CAM will be saved to: {cam_dir}")

    # 查找 NG/OK 类别索引
    ng_idx = _find_class_idx(idx_to_class, "ng")
    ok_idx = _find_class_idx(idx_to_class, "ok") if "ok" in [v.lower() for v in idx_to_class.values()] else None

    # 批量推理
    rows: List[Dict[str, object]] = []
    image_paths = list(_iter_images(img_dir))
    logger.info(f"Found {len(image_paths)} images to process")

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                x, pil_img = prepare_image(img_path, image_size)
                x = x.to(device)

                logits = model(x)
                probs = logits.softmax(dim=1)
                prob_ng = float(probs[0, ng_idx].item())
                prob_ok = float(probs[0, ok_idx].item()) if ok_idx is not None else float(1.0 - prob_ng)
                pred_idx = int(probs.argmax(dim=1).item())
                pred_label = idx_to_class[pred_idx]
                decision = "ng" if prob_ng >= float(args.threshold) else "ok"

                rows.append(
                    {
                        "filename": str(img_path.relative_to(img_dir)),
                        "pred_label": pred_label,
                        "prob_ng": round(prob_ng, 4),
                        "prob_ok": round(prob_ok, 4),
                        "decision": decision,
                    }
                )

                # 可选：保存 Grad-CAM
                if cam_dir is not None:
                    cam = compute_gradcam(model, x, target_class=pred_idx)
                    vis = overlay_cam(pil_img, cam, alpha=float(args.alpha))
                    out_path = cam_dir / (img_path.stem + "_cam.png")
                    ensure_dir(out_path.parent)
                    vis.save(out_path)

                # 进度日志
                if (i + 1) % 100 == 0 or (i + 1) == len(image_paths):
                    logger.info(f"Processed {i + 1}/{len(image_paths)} images")

            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                rows.append(
                    {
                        "filename": str(img_path.relative_to(img_dir)),
                        "pred_label": "ERROR",
                        "prob_ng": 0.0,
                        "prob_ok": 0.0,
                        "decision": "error",
                    }
                )

    # 写 CSV
    out_csv = Path(args.out_csv)
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "pred_label", "prob_ng", "prob_ok", "decision"])
        writer.writeheader()
        writer.writerows(rows)

    # 统计
    n = len(rows)
    n_ng = sum(1 for r in rows if r["decision"] == "ng")
    n_ok = sum(1 for r in rows if r["decision"] == "ok")
    n_error = sum(1 for r in rows if r["decision"] == "error")

    logger.info(f"Results saved to: {out_csv}")
    logger.info(f"Summary: total={n}, ng={n_ng}, ok={n_ok}, errors={n_error}, threshold={args.threshold}")

    if cam_dir is not None:
        logger.info(f"Grad-CAM images saved to: {cam_dir}")


if __name__ == "__main__":
    main()
