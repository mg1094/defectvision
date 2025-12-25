"""Grad-CAM 可视化工具模块"""

from pathlib import Path
from typing import Tuple

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from defectvision.datasets import build_transforms


def prepare_image(path: Path, image_size: int) -> Tuple[torch.Tensor, Image.Image]:
    """
    加载图片并预处理为模型输入张量。

    Args:
        path: 图片路径
        image_size: 目标尺寸

    Returns:
        (tensor, pil_image): 预处理后的张量和原始 PIL 图像
    """
    pil = Image.open(path).convert("RGB")
    tfm = build_transforms(image_size, train=False)
    x = tfm(pil).unsqueeze(0)  # 1xCxHxW
    return x, pil


def compute_gradcam(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    """
    计算 Grad-CAM 热力图。

    Args:
        model: 模型（需有 last_conv 属性）
        x: 输入张量 (1, C, H, W)
        target_class: 目标类别索引

    Returns:
        cam: 归一化后的热力图 (H, W)
    """
    feats: torch.Tensor = torch.empty(0)
    grads: torch.Tensor = torch.empty(0)

    def _forward_hook(_, __, output):
        nonlocal feats
        feats = output.detach()

    def _backward_hook(_, grad_input, grad_output):
        nonlocal grads
        grads = grad_output[0].detach()

    handle_fwd = model.last_conv.register_forward_hook(_forward_hook)
    handle_bwd = model.last_conv.register_full_backward_hook(_backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        score = logits[:, target_class].sum()
        score.backward()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()

    # CAM = ReLU( sum_k (alpha_k * A_k) ), alpha_k = mean spatial grad
    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = (weights * feats).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam_min, cam_max = cam.min(), cam.max()
    if float(cam_max - cam_min) > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    return cam.squeeze(0).squeeze(0)  # HxW


def overlay_cam(
    pil_img: Image.Image,
    cam: torch.Tensor,
    alpha: float = 0.4,
) -> Image.Image:
    """
    将 Grad-CAM 热力图叠加到原图上。

    Args:
        pil_img: 原始 PIL 图像
        cam: 热力图 (H, W)
        alpha: 叠加透明度

    Returns:
        叠加后的 PIL 图像
    """
    heatmap = cm.jet(cam.cpu().numpy())[..., :3]  # HxWx3
    heatmap = (heatmap * 255.0).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).resize(pil_img.size, resample=Image.BILINEAR)
    base = pil_img.convert("RGB")
    return Image.blend(base, heatmap, alpha=alpha)

