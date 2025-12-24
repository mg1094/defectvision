"""异常检测模型推理脚本"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from vision_ai.logger import get_logger, setup_logger
from vision_ai.models.autoencoder import build_anomaly_model
from vision_ai.utils import device_from_arg, ensure_dir, load_ckpt


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


def _create_diff_image(orig: torch.Tensor, recon: torch.Tensor) -> Image.Image:
    """创建差异可视化图"""
    # 反归一化
    orig = (orig * 0.5 + 0.5).clamp(0, 1)
    recon = (recon * 0.5 + 0.5).clamp(0, 1)

    diff = torch.abs(orig - recon)
    diff = diff.squeeze().cpu().numpy()

    # 归一化到 0-255
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    diff = (diff * 255).astype(np.uint8)

    return Image.fromarray(diff)


def _create_comparison_image(
    orig_pil: Image.Image,
    recon: torch.Tensor,
    diff: Image.Image,
    image_size: int,
) -> Image.Image:
    """创建对比图：原图 | 重建 | 差异"""
    # 反归一化重建图
    recon_np = (recon.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1).numpy()
    recon_np = (recon_np * 255).astype(np.uint8)
    recon_pil = Image.fromarray(recon_np)

    # 调整原图尺寸
    orig_resized = orig_pil.convert("L").resize((image_size, image_size))

    # 拼接
    comparison = Image.new("L", (image_size * 3, image_size))
    comparison.paste(orig_resized, (0, 0))
    comparison.paste(recon_pil, (image_size, 0))
    comparison.paste(diff, (image_size * 2, 0))

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="异常检测模型推理")
    parser.add_argument("--ckpt", type=str, required=True, help="模型检查点")
    parser.add_argument("--image", type=str, required=True, help="输入图片")
    parser.add_argument("--out", type=str, required=True, help="输出图片路径（对比图）")
    parser.add_argument("--threshold", type=float, default=None, help="异常阈值（可选，从 test.json 读取）")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "mps", "auto"])
    args = parser.parse_args()

    setup_logger("vision_ai")
    logger = get_logger("vision_ai")

    device = device_from_arg(args.device)
    logger.info(f"Using device: {device}")

    # 加载模型
    ckpt = load_ckpt(Path(args.ckpt), map_location=str(device))
    image_size = int(ckpt["image_size"])
    model_variant = ckpt.get("model_variant", "vae")
    latent_dim = ckpt.get("latent_dim", 128)

    model = build_anomaly_model(
        in_channels=1,
        latent_dim=latent_dim,
        base_channels=32,
        variant=model_variant,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(f"Model loaded: {model_variant}, latent_dim={latent_dim}")

    # 尝试读取阈值
    threshold = args.threshold
    if threshold is None:
        test_json = Path(args.ckpt).parent / "test.json"
        if test_json.exists():
            import json
            with open(test_json) as f:
                test_data = json.load(f)
                threshold = test_data.get("threshold", 0.01)
                logger.info(f"Loaded threshold from test.json: {threshold}")
        else:
            threshold = 0.01  # 默认阈值
            logger.info(f"Using default threshold: {threshold}")

    # 加载图片
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    x, pil_img = _prepare_image(image_path, image_size)
    x = x.to(device)

    # 推理
    with torch.no_grad():
        if model_variant == "vae":
            recon, _, _ = model(x)
        else:
            recon, _ = model(x)

        # 计算异常分数
        mse = F.mse_loss(recon, x, reduction="none")
        anomaly_score = mse.mean().item()

    # 判定
    is_anomaly = anomaly_score > threshold
    status = "ANOMALY (NG)" if is_anomaly else "NORMAL (OK)"

    logger.info(f"Anomaly score: {anomaly_score:.6f}")
    logger.info(f"Threshold: {threshold:.6f}")
    logger.info(f"Result: {status}")

    # 创建可视化
    diff_img = _create_diff_image(x, recon)
    comparison = _create_comparison_image(pil_img, recon, diff_img, image_size)

    # 保存
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    comparison.save(out_path)

    logger.info(f"Comparison image saved to: {out_path}")
    logger.info("Layout: [Original] [Reconstruction] [Difference]")


if __name__ == "__main__":
    main()

