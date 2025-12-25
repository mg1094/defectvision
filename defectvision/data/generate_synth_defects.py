import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class SplitCfg:
    name: str
    n: int


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _draw_base_part(
    rng: np.random.Generator,
    size: int,
) -> Image.Image:
    """
    生成一个“OK”零件底图：浅色背景 + 深色圆盘/矩形零件 + 轻微光照/噪声。
    """
    bg = rng.integers(190, 235)
    base = Image.new("L", (size, size), int(bg))
    draw = ImageDraw.Draw(base)

    # 零件：圆盘或圆角矩形
    part_color = int(rng.integers(40, 90))
    pad = int(rng.integers(int(size * 0.12), int(size * 0.22)))
    x0, y0 = pad, pad
    x1, y1 = size - pad, size - pad
    if rng.random() < 0.6:
        draw.ellipse((x0, y0, x1, y1), fill=part_color)
    else:
        r = int(rng.integers(int(size * 0.06), int(size * 0.12)))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=r, fill=part_color)

    # 轻微光照渐变
    gx = np.linspace(0, 1, size, dtype=np.float32)
    gy = np.linspace(0, 1, size, dtype=np.float32)
    gridx, gridy = np.meshgrid(gx, gy)
    shade = (gridx * 0.6 + gridy * 0.4)  # 0..1
    shade = (shade - shade.mean()) * float(rng.uniform(6.0, 14.0))
    arr = np.array(base, dtype=np.float32) + shade

    # 传感器噪声
    noise = rng.normal(0.0, rng.uniform(1.5, 4.0), size=(size, size)).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="L").filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.8))))
    return img


def _apply_defect(
    rng: np.random.Generator,
    img: Image.Image,
) -> Image.Image:
    """
    在“零件”上叠加缺陷，形成 NG：
    - scratch: 划痕
    - spot: 黑点/脏污
    - dent: 缺口/缺边
    - crack: 裂纹（多段折线）
    """
    size = img.size[0]
    out = img.copy()
    draw = ImageDraw.Draw(out)

    defect_type = rng.choice(["scratch", "spot", "dent", "crack"], p=[0.35, 0.30, 0.20, 0.15])

    if defect_type == "scratch":
        # 划痕：一条或两条细长线
        n = 1 if rng.random() < 0.75 else 2
        for _ in range(n):
            x0 = int(rng.integers(int(size * 0.2), int(size * 0.8)))
            y0 = int(rng.integers(int(size * 0.2), int(size * 0.8)))
            x1 = int(np.clip(x0 + rng.integers(-int(size * 0.45), int(size * 0.45)), 0, size - 1))
            y1 = int(np.clip(y0 + rng.integers(-int(size * 0.45), int(size * 0.45)), 0, size - 1))
            w = int(rng.integers(1, 4))
            col = int(rng.integers(120, 200))  # 更亮一点，模拟反光划痕
            draw.line((x0, y0, x1, y1), fill=col, width=w)
        out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.6, 1.4))))

    elif defect_type == "spot":
        # 黑点/污渍：小圆点或一团
        n = int(rng.integers(2, 8))
        for _ in range(n):
            r = int(rng.integers(1, int(size * 0.03) + 2))
            cx = int(rng.integers(int(size * 0.18), int(size * 0.82)))
            cy = int(rng.integers(int(size * 0.18), int(size * 0.82)))
            col = int(rng.integers(0, 35))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
        if rng.random() < 0.5:
            out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.0))))

    elif defect_type == "dent":
        # 缺口/缺边：在边缘挖掉一块（用背景色覆盖）
        bg = int(np.array(out, dtype=np.uint8).mean())
        pad = int(rng.integers(int(size * 0.10), int(size * 0.18)))
        # 从四边随机选一个
        side = rng.choice(["top", "bottom", "left", "right"])
        dent_w = int(rng.integers(int(size * 0.08), int(size * 0.18)))
        dent_h = int(rng.integers(int(size * 0.05), int(size * 0.12)))
        if side in ["top", "bottom"]:
            x0 = int(rng.integers(pad, size - pad - dent_w))
            y0 = pad if side == "top" else size - pad - dent_h
            draw.rectangle((x0, y0, x0 + dent_w, y0 + dent_h), fill=bg)
        else:
            y0 = int(rng.integers(pad, size - pad - dent_w))
            x0 = pad if side == "left" else size - pad - dent_h
            draw.rectangle((x0, y0, x0 + dent_h, y0 + dent_w), fill=bg)
        out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.7))))

    else:  # crack
        # 裂纹：多段折线，较暗
        col = int(rng.integers(5, 40))
        w = int(rng.integers(1, 3))
        points = []
        x = int(rng.integers(int(size * 0.25), int(size * 0.75)))
        y = int(rng.integers(int(size * 0.25), int(size * 0.75)))
        points.append((x, y))
        steps = int(rng.integers(3, 7))
        for _ in range(steps):
            x = int(np.clip(x + rng.integers(-int(size * 0.18), int(size * 0.18)), 0, size - 1))
            y = int(np.clip(y + rng.integers(-int(size * 0.18), int(size * 0.18)), 0, size - 1))
            points.append((x, y))
        draw.line(points, fill=col, width=w)
        out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 0.9))))

    return out


def _save_png(img: Image.Image, path: Path) -> None:
    img.save(path, format="PNG", optimize=True)


def _write_split(
    *,
    rng: np.random.Generator,
    out_dir: Path,
    split: SplitCfg,
    size: int,
    ok_ratio: float,
) -> Tuple[int, int]:
    n_ok = int(round(split.n * ok_ratio))
    n_ng = split.n - n_ok

    ok_dir = out_dir / split.name / "ok"
    ng_dir = out_dir / split.name / "ng"
    _ensure_dir(ok_dir)
    _ensure_dir(ng_dir)

    # OK
    for i in range(n_ok):
        img = _draw_base_part(rng, size)
        _save_png(img, ok_dir / f"{i:06d}.png")

    # NG
    for i in range(n_ng):
        img = _draw_base_part(rng, size)
        img = _apply_defect(rng, img)
        _save_png(img, ng_dir / f"{i:06d}.png")

    return n_ok, n_ng


def main() -> None:
    parser = argparse.ArgumentParser(description="生成合成工业质检缺陷数据集（OK/NG）")
    parser.add_argument("--out", type=str, required=True, help="输出目录，例如 ./datasets/synth_defects")
    parser.add_argument("--image-size", type=int, default=128, help="图片尺寸（正方形）")
    parser.add_argument("--train", type=int, default=2000, help="训练样本数（总数）")
    parser.add_argument("--val", type=int, default=400, help="验证样本数（总数）")
    parser.add_argument("--test", type=int, default=400, help="测试样本数（总数）")
    parser.add_argument("--ok-ratio", type=float, default=0.5, help="OK 占比（0~1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于复现）")
    args = parser.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    # 方便外部工具识别这是一个数据集目录
    (out_dir / ".keep").touch(exist_ok=True)

    rng = _rng(args.seed)

    splits = [
        SplitCfg("train", int(args.train)),
        SplitCfg("val", int(args.val)),
        SplitCfg("test", int(args.test)),
    ]

    total_ok, total_ng = 0, 0
    for s in splits:
        n_ok, n_ng = _write_split(
            rng=rng,
            out_dir=out_dir,
            split=s,
            size=int(args.image_size),
            ok_ratio=float(args.ok_ratio),
        )
        total_ok += n_ok
        total_ng += n_ng

    meta = out_dir / "meta.txt"
    meta.write_text(
        "\n".join(
            [
                f"seed={args.seed}",
                f"image_size={args.image_size}",
                f"ok_ratio={args.ok_ratio}",
                f"train={args.train}",
                f"val={args.val}",
                f"test={args.test}",
                f"total_ok={total_ok}",
                f"total_ng={total_ng}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[done] wrote dataset to: {out_dir}")
    print(f"       total_ok={total_ok}, total_ng={total_ng}")


if __name__ == "__main__":
    # 让 `python defectvision/data/generate_synth_defects.py ...` 也能跑
    os.environ.setdefault("PYTHONUTF8", "1")
    main()


