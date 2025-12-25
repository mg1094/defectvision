"""目标检测数据集生成器：图像 + YOLO 格式标注"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class BBox:
    """边界框"""

    class_id: int
    x_center: float  # 归一化中心 x
    y_center: float  # 归一化中心 y
    width: float  # 归一化宽度
    height: float  # 归一化高度


DEFECT_CLASSES = ["scratch", "spot", "crack", "dent"]


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
    pad = int(rng.integers(int(size * 0.08), int(size * 0.15)))
    x0, y0 = pad, pad
    x1, y1 = size - pad, size - pad

    if rng.random() < 0.6:
        draw.ellipse((x0, y0, x1, y1), fill=part_color)
    else:
        r = int(rng.integers(int(size * 0.04), int(size * 0.08)))
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


def _apply_scratch(rng: np.random.Generator, img: Image.Image) -> Tuple[Image.Image, BBox]:
    """划痕缺陷"""
    size = img.size[0]
    out = img.copy()
    draw = ImageDraw.Draw(out)

    x0 = int(rng.integers(int(size * 0.2), int(size * 0.7)))
    y0 = int(rng.integers(int(size * 0.2), int(size * 0.7)))
    length = int(rng.integers(int(size * 0.15), int(size * 0.35)))
    angle = rng.uniform(0, 2 * np.pi)
    x1 = int(np.clip(x0 + length * np.cos(angle), 0, size - 1))
    y1 = int(np.clip(y0 + length * np.sin(angle), 0, size - 1))
    w = int(rng.integers(2, 5))
    col = int(rng.integers(120, 200))
    draw.line((x0, y0, x1, y1), fill=col, width=w)

    out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 0.8))))

    # 计算 bbox
    bx0, by0 = min(x0, x1) - w, min(y0, y1) - w
    bx1, by1 = max(x0, x1) + w, max(y0, y1) + w
    bx0, by0 = max(0, bx0), max(0, by0)
    bx1, by1 = min(size, bx1), min(size, by1)

    bbox = BBox(
        class_id=0,
        x_center=(bx0 + bx1) / 2 / size,
        y_center=(by0 + by1) / 2 / size,
        width=(bx1 - bx0) / size,
        height=(by1 - by0) / size,
    )

    return out, bbox


def _apply_spot(rng: np.random.Generator, img: Image.Image) -> Tuple[Image.Image, BBox]:
    """黑点缺陷"""
    size = img.size[0]
    out = img.copy()
    draw = ImageDraw.Draw(out)

    r = int(rng.integers(4, int(size * 0.06) + 5))
    cx = int(rng.integers(int(size * 0.2), int(size * 0.8)))
    cy = int(rng.integers(int(size * 0.2), int(size * 0.8)))
    col = int(rng.integers(0, 35))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)

    # bbox
    margin = 2
    bx0, by0 = max(0, cx - r - margin), max(0, cy - r - margin)
    bx1, by1 = min(size, cx + r + margin), min(size, cy + r + margin)

    bbox = BBox(
        class_id=1,
        x_center=(bx0 + bx1) / 2 / size,
        y_center=(by0 + by1) / 2 / size,
        width=(bx1 - bx0) / size,
        height=(by1 - by0) / size,
    )

    return out, bbox


def _apply_crack(rng: np.random.Generator, img: Image.Image) -> Tuple[Image.Image, BBox]:
    """裂纹缺陷"""
    size = img.size[0]
    out = img.copy()
    draw = ImageDraw.Draw(out)

    col = int(rng.integers(5, 40))
    w = int(rng.integers(1, 3))

    points = []
    x = int(rng.integers(int(size * 0.25), int(size * 0.75)))
    y = int(rng.integers(int(size * 0.25), int(size * 0.75)))
    points.append((x, y))

    steps = int(rng.integers(3, 6))
    for _ in range(steps):
        x = int(np.clip(x + rng.integers(-int(size * 0.12), int(size * 0.12)), 0, size - 1))
        y = int(np.clip(y + rng.integers(-int(size * 0.12), int(size * 0.12)), 0, size - 1))
        points.append((x, y))

    draw.line(points, fill=col, width=w)
    out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.6))))

    # bbox
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    margin = w + 2
    bx0, by0 = max(0, min(xs) - margin), max(0, min(ys) - margin)
    bx1, by1 = min(size, max(xs) + margin), min(size, max(ys) + margin)

    bbox = BBox(
        class_id=2,
        x_center=(bx0 + bx1) / 2 / size,
        y_center=(by0 + by1) / 2 / size,
        width=(bx1 - bx0) / size,
        height=(by1 - by0) / size,
    )

    return out, bbox


def _apply_dent(rng: np.random.Generator, img: Image.Image) -> Tuple[Image.Image, BBox]:
    """缺口缺陷"""
    size = img.size[0]
    out = img.copy()
    draw = ImageDraw.Draw(out)

    bg = int(np.array(out, dtype=np.uint8).mean())
    pad = int(rng.integers(int(size * 0.10), int(size * 0.15)))

    side = rng.choice(["top", "bottom", "left", "right"])
    dent_w = int(rng.integers(int(size * 0.08), int(size * 0.15)))
    dent_h = int(rng.integers(int(size * 0.05), int(size * 0.10)))

    if side in ["top", "bottom"]:
        x0 = int(rng.integers(pad, size - pad - dent_w))
        y0 = pad if side == "top" else size - pad - dent_h
        rect = (x0, y0, x0 + dent_w, y0 + dent_h)
    else:
        y0 = int(rng.integers(pad, size - pad - dent_w))
        x0 = pad if side == "left" else size - pad - dent_h
        rect = (x0, y0, x0 + dent_h, y0 + dent_w)

    draw.rectangle(rect, fill=bg)
    out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.5))))

    # bbox
    bx0, by0, bx1, by1 = rect

    bbox = BBox(
        class_id=3,
        x_center=(bx0 + bx1) / 2 / size,
        y_center=(by0 + by1) / 2 / size,
        width=(bx1 - bx0) / size,
        height=(by1 - by0) / size,
    )

    return out, bbox


DEFECT_FUNCS = [_apply_scratch, _apply_spot, _apply_crack, _apply_dent]


def _generate_sample(
    rng: np.random.Generator,
    size: int,
    max_defects: int = 3,
) -> Tuple[Image.Image, List[BBox]]:
    """生成单个样本（可能包含多个缺陷）"""
    img = _draw_base_part(rng, size)
    bboxes = []

    # 随机数量的缺陷
    n_defects = int(rng.integers(1, max_defects + 1))

    for _ in range(n_defects):
        defect_idx = int(rng.integers(0, len(DEFECT_FUNCS)))
        defect_func = DEFECT_FUNCS[defect_idx]
        img, bbox = defect_func(rng, img)
        bboxes.append(bbox)

    return img, bboxes


def _save_yolo_annotation(path: Path, bboxes: List[BBox]) -> None:
    """保存 YOLO 格式标注"""
    lines = []
    for bbox in bboxes:
        lines.append(f"{bbox.class_id} {bbox.x_center:.6f} {bbox.y_center:.6f} {bbox.width:.6f} {bbox.height:.6f}")
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def _generate_split(
    rng: np.random.Generator,
    out_dir: Path,
    split: str,
    n_total: int,
    size: int,
    max_defects: int,
) -> int:
    """生成单个 split"""
    images_dir = out_dir / split / "images"
    labels_dir = out_dir / split / "labels"
    _ensure_dir(images_dir)
    _ensure_dir(labels_dir)

    total_defects = 0

    for i in range(n_total):
        img, bboxes = _generate_sample(rng, size, max_defects)
        img.save(images_dir / f"{i:06d}.png", format="PNG")
        _save_yolo_annotation(labels_dir / f"{i:06d}.txt", bboxes)
        total_defects += len(bboxes)

    return total_defects


def main() -> None:
    parser = argparse.ArgumentParser(description="生成目标检测数据集（YOLO 格式）")
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    parser.add_argument("--image-size", type=int, default=640, help="图片尺寸")
    parser.add_argument("--train", type=int, default=1000, help="训练样本数")
    parser.add_argument("--val", type=int, default=200, help="验证样本数")
    parser.add_argument("--test", type=int, default=200, help="测试样本数")
    parser.add_argument("--max-defects", type=int, default=3, help="每张图最大缺陷数")
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

    total_defects = 0
    for split_name, n_total in splits:
        n_defects = _generate_split(
            rng=rng,
            out_dir=out_dir,
            split=split_name,
            n_total=n_total,
            size=int(args.image_size),
            max_defects=int(args.max_defects),
        )
        print(f"[{split_name}] images={n_total}, defects={n_defects}")
        total_defects += n_defects

    # 生成 YOLO 配置文件
    yaml_content = f"""# Vision AI Defect Detection Dataset
path: {out_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: scratch
  1: spot
  2: crack
  3: dent

nc: 4
"""
    (out_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

    # 元信息
    meta = out_dir / "meta.txt"
    meta.write_text(
        "\n".join([
            "type=detection",
            "format=yolo",
            f"classes={','.join(DEFECT_CLASSES)}",
            f"seed={args.seed}",
            f"image_size={args.image_size}",
            f"max_defects={args.max_defects}",
            f"train={args.train}",
            f"val={args.val}",
            f"test={args.test}",
            f"total_defects={total_defects}",
        ]) + "\n",
        encoding="utf-8",
    )

    print(f"[done] wrote dataset to: {out_dir}")
    print(f"       YOLO config: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()

