"""工具函数模块"""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointError(Exception):
    """模型检查点加载错误"""

    pass


def ensure_dir(p: Path) -> None:
    """确保目录存在，不存在则创建"""
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    """保存对象为 JSON 文件"""

    def _default(x: Any) -> Any:
        if is_dataclass(x):
            return asdict(x)
        raise TypeError(f"Object of type {type(x)} is not JSON serializable")

    try:
        path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, default=_default) + "\n",
            encoding="utf-8",
        )
    except (IOError, OSError) as e:
        raise IOError(f"Failed to save JSON to {path}: {e}") from e


def load_json(path: Path) -> Any:
    """加载 JSON 文件"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (IOError, OSError) as e:
        raise IOError(f"Failed to load JSON from {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e


def device_from_arg(arg: str) -> torch.device:
    """根据参数返回设备"""
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {arg}")


def load_ckpt(path: Path, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    """
    加载模型检查点，带版本兼容性检查。

    Args:
        path: 检查点路径
        map_location: 设备映射

    Returns:
        检查点字典

    Raises:
        CheckpointError: 加载失败或格式不兼容
    """
    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")

    try:
        ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint {path}: {e}") from e

    # 检查必要字段
    required_fields = ["model_state_dict", "idx_to_class", "image_size"]
    missing = [f for f in required_fields if f not in ckpt]
    if missing:
        raise CheckpointError(f"Checkpoint missing required fields: {missing}")

    return ckpt


def save_ckpt(path: Path, payload: Dict[str, Any]) -> None:
    """
    保存模型检查点。

    Args:
        path: 保存路径
        payload: 检查点内容
    """
    try:
        ensure_dir(path.parent)
        torch.save(payload, str(path))
    except Exception as e:
        raise IOError(f"Failed to save checkpoint to {path}: {e}") from e


def get_device_info() -> str:
    """获取设备信息字符串"""
    info = []
    info.append(f"PyTorch: {torch.__version__}")
    info.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        info.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
        info.append(f"CUDA version: {torch.version.cuda}")
    info.append(f"MPS available: {torch.backends.mps.is_available()}")
    return " | ".join(info)
