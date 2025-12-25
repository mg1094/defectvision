"""TensorRT 引擎导出脚本（从 ONNX 转换）"""

import argparse
from pathlib import Path

from defectvision.logger import get_logger, setup_logger
from defectvision.utils import ensure_dir


def export_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 1,
    workspace_size_gb: float = 4.0,
) -> None:
    """
    将 ONNX 模型转换为 TensorRT 引擎

    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 TensorRT 引擎路径
        fp16: 启用 FP16 精度
        int8: 启用 INT8 精度（需要校准数据）
        max_batch_size: 最大 batch size
        workspace_size_gb: GPU 工作空间大小 (GB)
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT not installed. Please install TensorRT:\n"
            "  pip install tensorrt\n"
            "Or for CUDA 12: pip install tensorrt --extra-index-url https://pypi.nvidia.com"
        )

    logger = get_logger("defectvision")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # 创建 builder
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 解析 ONNX
    logger.info(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    # 配置 builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_size_gb * (1 << 30)))

    if fp16 and builder.platform_has_fast_fp16:
        logger.info("Enabling FP16 mode")
        config.set_flag(trt.BuilderFlag.FP16)

    if int8 and builder.platform_has_fast_int8:
        logger.info("Enabling INT8 mode")
        config.set_flag(trt.BuilderFlag.INT8)
        # 注意：INT8 需要校准器，这里暂不实现

    # 设置优化配置
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape

    # 动态 batch 支持
    min_shape = (1,) + tuple(input_shape[1:])
    opt_shape = (max_batch_size // 2 + 1,) + tuple(input_shape[1:])
    max_shape = (max_batch_size,) + tuple(input_shape[1:])

    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # 构建引擎
    logger.info("Building TensorRT engine (this may take a while)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # 保存引擎
    ensure_dir(engine_path.parent)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    file_size_mb = engine_path.stat().st_size / (1024 * 1024)
    logger.info(f"TensorRT engine saved to: {engine_path}")
    logger.info(f"Engine size: {file_size_mb:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="将 ONNX 模型转换为 TensorRT 引擎")
    parser.add_argument("--onnx", type=str, required=True, help="输入 ONNX 模型路径")
    parser.add_argument("--out", type=str, required=True, help="输出 TensorRT 引擎路径 (.engine)")
    parser.add_argument("--fp16", action="store_true", default=True, help="启用 FP16 精度（默认开启）")
    parser.add_argument("--no-fp16", action="store_true", help="禁用 FP16")
    parser.add_argument("--int8", action="store_true", help="启用 INT8 精度（需要校准）")
    parser.add_argument("--max-batch-size", type=int, default=8, help="最大 batch size")
    parser.add_argument("--workspace", type=float, default=4.0, help="GPU 工作空间大小 (GB)")
    args = parser.parse_args()

    setup_logger("defectvision")
    logger = get_logger("defectvision")

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    engine_path = Path(args.out)
    fp16 = args.fp16 and not args.no_fp16

    logger.info(f"Converting ONNX to TensorRT: {onnx_path} -> {engine_path}")
    logger.info(f"Settings: fp16={fp16}, int8={args.int8}, max_batch={args.max_batch_size}")

    export_tensorrt(
        onnx_path=onnx_path,
        engine_path=engine_path,
        fp16=fp16,
        int8=args.int8,
        max_batch_size=args.max_batch_size,
        workspace_size_gb=args.workspace,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()

