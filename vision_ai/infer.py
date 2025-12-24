"""单图推理脚本：预测 + Grad-CAM 可视化"""

import argparse
from pathlib import Path
from typing import Dict

import torch

from vision_ai.gradcam import compute_gradcam, overlay_cam, prepare_image
from vision_ai.logger import get_logger, setup_logger
from vision_ai.model import build_model
from vision_ai.utils import device_from_arg, load_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="单图推理 + Grad-CAM 可视化")
    parser.add_argument("--ckpt", type=str, required=True, help="训练保存的 best.pt")
    parser.add_argument("--image", type=str, required=True, help="待推理的图片路径")
    parser.add_argument("--out", type=str, required=True, help="输出可视化文件路径")
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

    # 加载并预处理图片
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    x, pil_img = prepare_image(image_path, image_size)
    x = x.to(device)

    # 推理
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1)
        top_prob, top_idx = probs.max(dim=1)

    pred_cls = idx_to_class[int(top_idx.item())]
    prob_value = float(top_prob.item())

    # Grad-CAM
    cam = compute_gradcam(model, x, int(top_idx.item()))
    vis = overlay_cam(pil_img, cam, alpha=float(args.alpha))

    # 保存结果
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path)

    logger.info(f"Prediction: class={pred_cls}, prob={prob_value:.4f}")
    logger.info(f"Grad-CAM saved to: {out_path}")

    # 打印所有类别的概率
    for i, cls_name in idx_to_class.items():
        logger.info(f"  {cls_name}: {float(probs[0, i].item()):.4f}")


if __name__ == "__main__":
    main()
