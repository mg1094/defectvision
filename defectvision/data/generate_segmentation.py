"""分割任务数据集生成器：图像 + 缺陷 Mask"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _draw_base_part(rng: np.random.Generator, size: int) -> Image.Image:
    """生成 OK 零件底图"""
    bg = rng.integers(190, 235)
    base = Image.new("L", (size, size), int(bg))
    draw = ImageDraw.Draw(base)

    part_color = int(rng.integers(40, 90))
    pad = int(rng.integers(int(size * 0.12), int(size * 0.22)))
    x0, y0 = pad, pad
    x1, y1 = size - pad, size - pad

    if rng.random() < 0.6:
        draw.ellipse((x0, y0, x1, y1), fill=part_color)
    else:
        r = int(rng.integers(int(size * 0.06), int(size * 0.12)))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=r, fill=part_color)

    # 光照 + 噪声
    gx = np.linspace(0, 1, size, dtype=np.float32)
    gy = np.linspace(0, 1, size, dtype=np.float32)
    gridx, gridy = np.meshgrid(gx, gy)
    shade = (gridx * 0.6 + gridy * 0.4)
    shade = (shade - shade.mean()) * float(rng.uniform(6.0, 14.0))
    arr = np.array(base, dtype=np.float32) + shade
    noise = rng.normal(0.0, rng.uniform(1.5, 4.0), size=(size, size)).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="L").filter(
        ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.8)))
    )
    return img


def _apply_random_defect(rng: np.random.Generator, img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """随机应用缺陷并返回 (image, mask)"""
    size = img.size[0]
    out = img.copy()
    draw = ImageDraw.Draw(out)
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)

    defect_type = rng.choice(["scratch", "spot", "crack", "dent"])

    if defect_type == "scratch":
        n = 1 if rng.random() < 0.7 else 2
        for _ in range(n):
            x0 = int(rng.integers(int(size * 0.2), int(size * 0.8)))
            y0 = int(rng.integers(int(size * 0.2), int(size * 0.8)))
            length = int(rng.integers(int(size * 0.2), int(size * 0.5)))
            angle = rng.uniform(0, 2 * np.pi)
            x1 = int(np.clip(x0 + length * np.cos(angle), 0, size - 1))
            y1 = int(np.clip(y0 + length * np.sin(angle), 0, size - 1))
            w = int(rng.integers(1, 4))
            col = int(rng.integers(120, 200))
            draw.line((x0, y0, x1, y1), fill=col, width=w)
            mask_draw.line((x0, y0, x1, y1), fill=255, width=w + 2)
        out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.0))))

    elif defect_type == "spot":
        n = int(rng.integers(2, 8))
        for _ in range(n):
            r = int(rng.integers(2, int(size * 0.04) + 3))
            cx = int(rng.integers(int(size * 0.18), int(size * 0.82)))
            cy = int(rng.integers(int(size * 0.18), int(size * 0.82)))
            col = int(rng.integers(0, 35))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
            mask_draw.ellipse((cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1), fill=255)

    elif defect_type == "crack":
        col = int(rng.integers(5, 40))
        w = int(rng.integers(1, 3))
        points = []
        x = int(rng.integers(int(size * 0.25), int(size * 0.75)))
        y = int(rng.integers(int(size * 0.25), int(size * 0.75)))
        points.append((x, y))
        steps = int(rng.integers(4, 8))
        for _ in range(steps):
            x = int(np.clip(x + rng.integers(-int(size * 0.15), int(size * 0.15)), 0, size - 1))
            y = int(np.clip(y + rng.integers(-int(size * 0.15), int(size * 0.15)), 0, size - 1))
            points.append((x, y))
        draw.line(points, fill=col, width=w)
        mask_draw.line(points, fill=255, width=w + 2)
        out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.7))))

    else:  # dent
        bg = int(np.array(out, dtype=np.uint8).mean())
        pad = int(rng.integers(int(size * 0.10), int(size * 0.18)))
        side = rng.choice(["top", "bottom", "left", "right"])
        dent_w = int(rng.integers(int(size * 0.10), int(size * 0.20)))
        dent_h = int(rng.integers(int(size * 0.06), int(size * 0.14)))
        if side in ["top", "bottom"]:
            x0 = int(rng.integers(pad, size - pad - dent_w))
            y0 = pad if side == "top" else size - pad - dent_h
            rect = (x0, y0, x0 + dent_w, y0 + dent_h)
        else:
            y0 = int(rng.integers(pad, size - pad - dent_w))
            x0 = pad if side == "left" else size - pad - dent_h
            rect = (x0, y0, x0 + dent_h, y0 + dent_w)
        draw.rectangle(rect, fill=bg)
        mask_draw.rectangle(rect, fill=255)
        out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.6))))

    return out, mask


def _generate_split(
    rng: np.random.Generator,
    out_dir: Path,
    split: str,
    n_total: int,
    defect_ratio: float,
    size: int,
) -> Tuple[int, int]:
    """生成单个 split"""
    images_dir = out_dir / split / "images"
    masks_dir = out_dir / split / "masks"
    _ensure_dir(images_dir)
    _ensure_dir(masks_dir)

    n_defect = int(round(n_total * defect_ratio))
    n_ok = n_total - n_defect

    idx = 0

    # OK 样本（mask 全零）
    for _ in range(n_ok):
        img = _draw_base_part(rng, size)
        mask = Image.new("L", (size, size), 0)
        img.save(images_dir / f"{idx:06d}.png", format="PNG")
        mask.save(masks_dir / f"{idx:06d}.png", format="PNG")
        idx += 1

    # 缺陷样本
    for _ in range(n_defect):
        base = _draw_base_part(rng, size)
        img, mask = _apply_random_defect(rng, base)
        img.save(images_dir / f"{idx:06d}.png", format="PNG")
        mask.save(masks_dir / f"{idx:06d}.png", format="PNG")
        idx += 1

    return n_ok, n_defect


def main() -> None:
    parser = argparse.ArgumentParser(description="生成分割任务数据集（图像 + Mask）")
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    parser.add_argument("--image-size", type=int, default=128, help="图片尺寸")
    parser.add_argument("--train", type=int, default=1000, help="训练样本数")
    parser.add_argument("--val", type=int, default=200, help="验证样本数")
    parser.add_argument("--test", type=int, default=200, help="测试样本数")
    parser.add_argument("--defect-ratio", type=float, default=0.6, help="缺陷样本占比")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    rng = _rng(args.seed)

    splits = [
        ("train", int(args.train)),
        ("val", int(args.val)),
        ("test", int(args.test)),
    ]

    for split_name, n_total in splits:
        n_ok, n_defect = _generate_split(
            rng=rng,
            out_dir=out_dir,
            split=split_name,
            n_total=n_total,
            defect_ratio=float(args.defect_ratio),
            size=int(args.image_size),
        )
        print(f"[{split_name}] ok={n_ok}, defect={n_defect}")

    # 元信息
    meta = out_dir / "meta.txt"
    meta.write_text(
        "\n".join([
            "type=segmentation",
            f"seed={args.seed}",
            f"image_size={args.image_size}",
            f"defect_ratio={args.defect_ratio}",
            f"train={args.train}",
            f"val={args.val}",
            f"test={args.test}",
        ]) + "\n",
        encoding="utf-8",
    )

    print(f"[done] wrote dataset to: {out_dir}")


if __name__ == "__main__":
    main()

