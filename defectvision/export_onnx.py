"""ONNX 导出脚本"""

import argparse
from pathlib import Path

import torch

from defectvision.logger import get_logger, setup_logger
from defectvision.model import build_model
from defectvision.utils import device_from_arg, ensure_dir, load_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 ONNX (支持动态 batch)")
    parser.add_argument("--ckpt", type=str, required=True, help="训练保存的 best.pt")
    parser.add_argument("--out", type=str, required=True, help="输出 onnx 路径，如 ./runs/exp1/model.onnx")
    parser.add_argument("--dynamic-batch", action="store_true", help="启用动态 batch 维度")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--simplify", action="store_true", help="使用 onnx-simplifier 简化模型")
    args = parser.parse_args()

    setup_logger("defectvision")
    logger = get_logger("defectvision")

    # 加载模型
    device = device_from_arg(args.device)
    logger.info(f"Using device: {device}")

    ckpt = load_ckpt(Path(args.ckpt), map_location=str(device))
    idx_to_class = ckpt["idx_to_class"]
    image_size = int(ckpt["image_size"])
    backbone = ckpt.get("backbone", "smallcnn")
    pretrained = ckpt.get("pretrained", True)

    model = build_model(
        in_channels=1,
        num_classes=len(idx_to_class),
        backbone=backbone,
        pretrained=pretrained,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(f"Model loaded: {backbone}, classes={list(idx_to_class.values())}")

    # 准备输出路径
    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    # 导出 ONNX
    dummy = torch.randn(1, 1, image_size, image_size, device=device)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if args.dynamic_batch else None

    logger.info(f"Exporting ONNX: opset={args.opset}, dynamic_batch={args.dynamic_batch}")

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=int(args.opset),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    logger.info(f"ONNX exported to: {out_path}")

    # 可选：简化模型
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify

            logger.info("Simplifying ONNX model...")
            onnx_model = onnx.load(str(out_path))
            simplified_model, check = simplify(onnx_model)
            if check:
                onnx.save(simplified_model, str(out_path))
                logger.info("ONNX model simplified successfully")
            else:
                logger.warning("ONNX simplification check failed, keeping original model")
        except ImportError:
            logger.warning("onnx-simplifier not installed, skipping simplification")
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")

    # 验证导出的模型
    try:
        import onnx

        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed")
    except ImportError:
        logger.info("onnx package not installed, skipping validation")
    except Exception as e:
        logger.warning(f"ONNX validation failed: {e}")

    # 打印模型信息
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model size: {file_size_mb:.2f} MB")
    logger.info(f"Input shape: (batch, 1, {image_size}, {image_size})")
    logger.info(f"Output: logits with {len(idx_to_class)} classes")


if __name__ == "__main__":
    main()
