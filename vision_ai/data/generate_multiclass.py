"""多类缺陷数据集生成器：ok/scratch/spot/crack/dent"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class DefectConfig:
    """缺陷类型配置"""

    name: str
    ratio: float  # 在 NG 中的占比


DEFECT_TYPES: List[DefectConfig] = [
    DefectConfig("scratch", 0.30),
    DefectConfig("spot", 0.30),
    DefectConfig("crack", 0.20),
    DefectConfig("dent", 0.20),
]


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

    # 光照渐变
    gx = np.linspace(0, 1, size, dtype=np.float32)
    gy = np.linspace(0, 1, size, dtype=np.float32)
    gridx, gridy = np.meshgrid(gx, gy)
    shade = (gridx * 0.6 + gridy * 0.4)
    shade = (shade - shade.mean()) * float(rng.uniform(6.0, 14.0))
    arr = np.array(base, dtype=np.float32) + shade

    # 噪声
    noise = rng.normal(0.0, rng.uniform(1.5, 4.0), size=(size, size)).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="L").filter(
        ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.8)))
    )
    return img


def _apply_scratch(rng: np.random.Generator, img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """划痕缺陷 + mask"""
    size = img.size[0]
    out = img.copy()
    draw = ImageDraw.Draw(out)
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)

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
    return out, mask


def _apply_spot(rng: np.random.Generator, img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """黑点/污渍缺陷 + mask"""
    size = img.size[0]
    out = img.copy()
    draw = ImageDraw.Draw(out)
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)

    n = int(rng.integers(2, 8))
    for _ in range(n):
        r = int(rng.integers(2, int(size * 0.04) + 3))
        cx = int(rng.integers(int(size * 0.18), int(size * 0.82)))
        cy = int(rng.integers(int(size * 0.18), int(size * 0.82)))
        col = int(rng.integers(0, 35))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
        mask_draw.ellipse((cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1), fill=255)

    if rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 0.8))))
    return out, mask


def _apply_crack(rng: np.random.Generator, img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """裂纹缺陷 + mask"""
    size = img.size[0]
    out = img.copy()
    draw = ImageDraw.Draw(out)
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)

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
    return out, mask


def _apply_dent(rng: np.random.Generator, img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """缺口缺陷 + mask"""
    size = img.size[0]
    out = img.copy()
    draw = ImageDraw.Draw(out)
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)

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


DEFECT_FUNCS = {
    "scratch": _apply_scratch,
    "spot": _apply_spot,
    "crack": _apply_crack,
    "dent": _apply_dent,
}


def _save_png(img: Image.Image, path: Path) -> None:
    img.save(path, format="PNG", optimize=True)


def _generate_split(
    rng: np.random.Generator,
    out_dir: Path,
    split: str,
    n_total: int,
    ok_ratio: float,
    size: int,
    save_mask: bool = False,
) -> dict:
    """生成单个 split 的数据"""
    n_ok = int(round(n_total * ok_ratio))
    n_ng = n_total - n_ok

    # 按比例分配各类缺陷
    defect_counts = {}
    remaining = n_ng
    for i, cfg in enumerate(DEFECT_TYPES):
        if i == len(DEFECT_TYPES) - 1:
            defect_counts[cfg.name] = remaining
        else:
            count = int(round(n_ng * cfg.ratio))
            defect_counts[cfg.name] = count
            remaining -= count

    stats = {"ok": 0}

    # 创建目录
    ok_dir = out_dir / split / "ok"
    _ensure_dir(ok_dir)

    for defect_name in DEFECT_FUNCS.keys():
        _ensure_dir(out_dir / split / defect_name)
        if save_mask:
            _ensure_dir(out_dir / split / f"{defect_name}_mask")
        stats[defect_name] = 0

    # 生成 OK 样本
    for i in range(n_ok):
        img = _draw_base_part(rng, size)
        _save_png(img, ok_dir / f"{i:06d}.png")
        stats["ok"] += 1

    # 生成各类缺陷样本
    for defect_name, count in defect_counts.items():
        defect_func = DEFECT_FUNCS[defect_name]
        defect_dir = out_dir / split / defect_name
        mask_dir = out_dir / split / f"{defect_name}_mask" if save_mask else None

        for i in range(count):
            base = _draw_base_part(rng, size)
            img, mask = defect_func(rng, base)
            _save_png(img, defect_dir / f"{i:06d}.png")
            if mask_dir:
                _save_png(mask, mask_dir / f"{i:06d}.png")
            stats[defect_name] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="生成多类缺陷数据集（ok/scratch/spot/crack/dent）")
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    parser.add_argument("--image-size", type=int, default=128, help="图片尺寸")
    parser.add_argument("--train", type=int, default=2000, help="训练样本总数")
    parser.add_argument("--val", type=int, default=400, help="验证样本总数")
    parser.add_argument("--test", type=int, default=400, help="测试样本总数")
    parser.add_argument("--ok-ratio", type=float, default=0.4, help="OK 占比")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save-mask", action="store_true", help="同时保存缺陷 mask（用于分割任务）")
    args = parser.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    rng = _rng(args.seed)

    splits = [
        ("train", int(args.train)),
        ("val", int(args.val)),
        ("test", int(args.test)),
    ]

    all_stats = {}
    for split_name, n_total in splits:
        stats = _generate_split(
            rng=rng,
            out_dir=out_dir,
            split=split_name,
            n_total=n_total,
            ok_ratio=float(args.ok_ratio),
            size=int(args.image_size),
            save_mask=args.save_mask,
        )
        all_stats[split_name] = stats
        print(f"[{split_name}] {stats}")

    # 保存元信息
    meta = out_dir / "meta.txt"
    lines = [
        f"type=multiclass",
        f"classes=ok,scratch,spot,crack,dent",
        f"seed={args.seed}",
        f"image_size={args.image_size}",
        f"ok_ratio={args.ok_ratio}",
        f"save_mask={args.save_mask}",
    ]
    for split_name, stats in all_stats.items():
        for cls_name, count in stats.items():
            lines.append(f"{split_name}_{cls_name}={count}")
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote dataset to: {out_dir}")


if __name__ == "__main__":
    main()

