"""TensorRT 高性能推理脚本"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from defectvision.logger import get_logger, setup_logger
from defectvision.utils import ensure_dir


class TensorRTInference:
    """TensorRT 推理引擎封装"""

    def __init__(self, engine_path: Path, device_id: int = 0):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}\n"
                "Install with: pip install tensorrt pycuda"
            )

        self.logger = get_logger("defectvision")
        self.logger.info(f"Loading TensorRT engine: {engine_path}")

        # 加载引擎
        self.trt = trt
        self.cuda = cuda

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()

        # 获取输入/输出信息
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        input_shape = self.engine.get_tensor_shape(self.input_name)
        output_shape = self.engine.get_tensor_shape(self.output_name)

        self.logger.info(f"Input: {self.input_name} {input_shape}")
        self.logger.info(f"Output: {self.output_name} {output_shape}")

        # 分配 GPU 内存
        self.d_input = None
        self.d_output = None
        self.h_output = None
        self.stream = cuda.Stream()

    def _allocate_buffers(self, batch_size: int, input_shape: Tuple[int, ...]) -> None:
        """分配 GPU 缓冲区"""
        input_size = int(np.prod(input_shape)) * np.dtype(np.float32).itemsize
        output_shape = self.engine.get_tensor_shape(self.output_name)
        # 替换动态维度
        output_shape = (batch_size,) + tuple(output_shape[1:])
        output_size = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize

        self.d_input = self.cuda.mem_alloc(input_size)
        self.d_output = self.cuda.mem_alloc(output_size)
        self.h_output = np.empty(output_shape, dtype=np.float32)

    def infer(self, images: np.ndarray) -> np.ndarray:
        """
        执行推理

        Args:
            images: 输入图像 (N, C, H, W)，float32，已归一化

        Returns:
            logits: 输出 logits (N, num_classes)
        """
        batch_size = images.shape[0]
        input_shape = images.shape

        # 分配缓冲区
        if self.d_input is None:
            self._allocate_buffers(batch_size, input_shape)

        # 设置输入形状（动态 batch）
        self.context.set_input_shape(self.input_name, input_shape)

        # 拷贝输入到 GPU
        self.cuda.memcpy_htod_async(self.d_input, images.astype(np.float32), self.stream)

        # 设置张量地址
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        # 执行推理
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 拷贝输出到 CPU
        self.cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output.copy()

    def __del__(self):
        """释放资源"""
        if self.d_input is not None:
            self.d_input.free()
        if self.d_output is not None:
            self.d_output.free()


def preprocess_image(
    image_path: Path,
    image_size: int,
) -> Tuple[np.ndarray, Image.Image]:
    """预处理图像"""
    pil = Image.open(image_path).convert("L")
    pil_resized = pil.resize((image_size, image_size), Image.BILINEAR)

    # 转为 numpy 并归一化
    arr = np.array(pil_resized, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # 归一化到 [-1, 1]
    arr = arr[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

    return arr, pil


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorRT 高性能推理")
    parser.add_argument("--engine", type=str, required=True, help="TensorRT 引擎路径")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--image-size", type=int, default=128, help="图片尺寸")
    parser.add_argument("--classes", type=str, default="ok,ng", help="类别名称，逗号分隔")
    parser.add_argument("--benchmark", action="store_true", help="运行性能基准测试")
    parser.add_argument("--iterations", type=int, default=100, help="基准测试迭代次数")
    args = parser.parse_args()

    setup_logger("defectvision")
    logger = get_logger("defectvision")

    engine_path = Path(args.engine)
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine not found: {engine_path}")

    # 加载引擎
    engine = TensorRTInference(engine_path)

    # 类别名称
    classes = args.classes.split(",")
    idx_to_class = {i: c.strip() for i, c in enumerate(classes)}

    # 加载图片
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    x, pil_img = preprocess_image(image_path, args.image_size)

    # 推理
    logits = engine.infer(x)

    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    pred_idx = int(np.argmax(probs[0]))
    pred_class = idx_to_class.get(pred_idx, str(pred_idx))
    pred_prob = float(probs[0, pred_idx])

    logger.info(f"Prediction: {pred_class} ({pred_prob:.4f})")
    for i, cls_name in idx_to_class.items():
        logger.info(f"  {cls_name}: {float(probs[0, i]):.4f}")

    # 性能基准测试
    if args.benchmark:
        logger.info(f"\nRunning benchmark ({args.iterations} iterations)...")

        # 预热
        for _ in range(10):
            engine.infer(x)

        # 计时
        times = []
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            engine.infer(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        times = np.array(times)
        logger.info(f"Latency (ms): mean={times.mean():.2f}, std={times.std():.2f}")
        logger.info(f"Latency (ms): min={times.min():.2f}, max={times.max():.2f}")
        logger.info(f"Throughput: {1000 / times.mean():.1f} FPS")


if __name__ == "__main__":
    main()

