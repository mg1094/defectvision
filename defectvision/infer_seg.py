"""分割模型推理脚本"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from defectvision.logger import get_logger, setup_logger
from defectvision.models.unet import build_unet
from defectvision.utils import device_from_arg, ensure_dir, load_ckpt


def _prepare_image(path: Path, image_size: int) -> tuple:
    """加载并预处理图片"""
    pil = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    x = tfm(pil).unsqueeze(0)
    return x, pil


def _overlay_mask(
    pil_img: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple = (255, 0, 0),
) -> Image.Image:
    """将预测 mask 叠加到原图"""
    # 将 mask 调整到原图尺寸
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_pil = mask_pil.resize(pil_img.size, Image.BILINEAR)
    mask_arr = np.array(mask_pil) / 255.0

    # 创建彩色叠加
    base = np.array(pil_img.convert("RGB")).astype(np.float32)
    overlay = np.zeros_like(base)
    for i, c in enumerate(color):
        overlay[:, :, i] = c

    # 混合
    mask_3d = mask_arr[:, :, np.newaxis]
    result = base * (1 - mask_3d * alpha) + overlay * mask_3d * alpha
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="分割模型推理")
    parser.add_argument("--ckpt", type=str, required=True, help="模型检查点")
    parser.add_argument("--image", type=str, required=True, help="输入图片")
    parser.add_argument("--out", type=str, required=True, help="输出图片路径")
    parser.add_argument("--out-mask", type=str, default=None, help="输出 mask 路径（可选）")
    parser.add_argument("--threshold", type=float, default=0.5, help="二值化阈值")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--alpha", type=float, default=0.5, help="叠加透明度")
    args = parser.parse_args()

    setup_logger("defectvision")
    logger = get_logger("defectvision")

    device = device_from_arg(args.device)
    logger.info(f"Using device: {device}")

    # 加载模型
    ckpt = load_ckpt(Path(args.ckpt), map_location=str(device))
    image_size = int(ckpt["image_size"])
    model_variant = ckpt.get("model_variant", "unet_small")

    model = build_unet(in_channels=1, num_classes=1, variant=model_variant).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(f"Model loaded: {model_variant}")

    # 加载图片
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    x, pil_img = _prepare_image(image_path, image_size)
    x = x.to(device)

    # 推理
    with torch.no_grad():
        pred = model(x)
        pred_sigmoid = torch.sigmoid(pred)
        pred_bin = (pred_sigmoid > args.threshold).float()

    # 转为 numpy
    mask = pred_bin.squeeze().cpu().numpy()
    prob_mask = pred_sigmoid.squeeze().cpu().numpy()

    # 统计
    defect_ratio = mask.sum() / mask.size
    logger.info(f"Defect pixel ratio: {defect_ratio:.4f}")

    # 保存结果
    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    if defect_ratio > 0.001:
        vis = _overlay_mask(pil_img, mask, alpha=args.alpha, color=(255, 0, 0))
        logger.info("Defect detected!")
    else:
        vis = pil_img.convert("RGB")
        logger.info("No defect detected (OK)")

    vis.save(out_path)
    logger.info(f"Result saved to: {out_path}")

    # 可选：保存 mask
    if args.out_mask:
        mask_path = Path(args.out_mask)
        ensure_dir(mask_path.parent)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(pil_img.size, Image.NEAREST)
        mask_pil.save(mask_path)
        logger.info(f"Mask saved to: {mask_path}")


if __name__ == "__main__":
    main()

