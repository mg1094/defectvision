# defectvision

工业质检缺陷检测 Vision 项目，支持：

- **四种检测模式**：分类 / 分割 / 异常检测 / 目标检测
- **实时视频流**：摄像头 / RTSP / 视频文件
- **高性能部署**：TensorRT 加速（10x 推理速度）
- **REST API 服务**：FastAPI + Web UI

## ✨ 特性

| 功能 | 说明 |
|------|------|
| 🏷️ 分类 | OK/NG 二分类或多类缺陷（scratch/spot/crack/dent） |
| 🎯 分割 | 像素级缺陷定位（U-Net） |
| 🔍 异常检测 | 只用 OK 样本训练（VAE/AutoEncoder） |
| 📦 目标检测 | YOLO 多目标缺陷定位 |
| 📹 视频流 | 实时摄像头/RTSP 检测 |
| ⚡ TensorRT | GPU 高性能推理（FP16/INT8） |
| 🌐 REST API | FastAPI 服务 + 内置 Web UI |
| 📊 可视化 | Grad-CAM 热力图 / TensorBoard |

## 🚀 快速开始

```bash
# 安装基础依赖
uv sync

# 安装服务依赖（FastAPI）
uv sync --extra server

# 安装 YOLO 目标检测
uv sync --extra yolo

# 安装 TensorRT 依赖（需要 NVIDIA GPU）
uv sync --extra tensorrt

# 安装所有依赖
uv sync --extra all
```

---

## 1️⃣ 分类模式

```bash
# 生成数据
uv run defect-generate --out ./datasets/binary

# 训练
uv run defect-train --data ./datasets/binary --out ./runs/cls --backbone resnet18 --epochs 20

# 推理
uv run defect-infer --ckpt ./runs/cls/best.pt --image ./test.png --out ./result.png
```

## 2️⃣ 分割模式

```bash
# 生成数据
uv run defect-generate-seg --out ./datasets/seg

# 训练 U-Net
uv run defect-train-seg --data ./datasets/seg --out ./runs/seg --epochs 50

# 推理
uv run defect-infer-seg --ckpt ./runs/seg/best.pt --image ./test.png --out ./result.png
```

## 3️⃣ 异常检测

```bash
# 生成数据
uv run defect-generate --out ./datasets/anomaly --ok-ratio 0.5

# 训练 VAE（只用 OK 样本）
uv run defect-train-anomaly --data ./datasets/anomaly --out ./runs/anomaly --model vae

# 推理
uv run defect-infer-anomaly --ckpt ./runs/anomaly/best.pt --image ./test.png --out ./result.png
```

## 4️⃣ YOLO 目标检测

YOLO 可以同时检测多个缺陷并标注位置，适合复杂场景。

```bash
# 生成数据（YOLO 格式：图像 + txt 标注）
uv run defect-generate-det --out ./datasets/det --train 1000 --val 200 --test 200

# 训练 YOLOv8
uv run defect-train-yolo --data ./datasets/det/data.yaml --out ./runs/yolo --epochs 100

# 图片推理
uv run defect-infer-yolo --model ./runs/yolo/train/weights/best.pt --source ./test.png --out ./results/

# 目录批量推理
uv run defect-infer-yolo --model ./runs/yolo/train/weights/best.pt --source ./datasets/det/test/images --out ./results/ --save-csv
```

### YOLO 模型选择

| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| YOLOv8n | 3.2M | ⚡️ 最快 | ⭐️⭐️ | 边缘设备/实时 |
| YOLOv8s | 11.2M | ⚡️ 快 | ⭐️⭐️⭐️ | 平衡选择 |
| YOLOv8m | 25.9M | 中等 | ⭐️⭐️⭐️⭐️ | 高精度 |
| YOLOv8l | 43.7M | 较慢 | ⭐️⭐️⭐️⭐️⭐️ | 最高精度 |

---

## 📹 实时视频流检测

支持摄像头、RTSP 流、视频文件的实时缺陷检测。

### 分类模型视频流

```bash
# 摄像头实时检测
uv run defect-video --ckpt ./runs/cls/best.pt --source 0

# RTSP 流检测
uv run defect-video --ckpt ./runs/cls/best.pt --source "rtsp://192.168.1.100:554/stream"

# 视频文件检测
uv run defect-video --ckpt ./runs/cls/best.pt --source ./video.mp4 --output ./result.mp4
```

### YOLO 视频流检测

```bash
# 摄像头实时目标检测
uv run defect-video-yolo --model ./runs/yolo/train/weights/best.pt --source 0

# RTSP 流目标检测
uv run defect-video-yolo --model ./runs/yolo/train/weights/best.pt --source "rtsp://ip:port/stream"

# 视频文件目标检测
uv run defect-video-yolo --model ./runs/yolo/train/weights/best.pt --source ./video.mp4 --output ./result.mp4
```

### 视频流参数

| 参数 | 说明 |
|------|------|
| `--source 0` | 默认摄像头 |
| `--source 1` | 第二摄像头 |
| `--source rtsp://...` | RTSP 流 |
| `--source video.mp4` | 视频文件 |
| `--threshold 0.5` | NG 判定阈值（分类） |
| `--conf 0.25` | 置信度阈值（YOLO） |
| `--max-fps 30` | 最大处理帧率 |
| `--no-show` | 不显示窗口（服务器） |

---

## ⚡ TensorRT 部署

TensorRT 可将推理速度提升 **5-10 倍**，适合生产环境部署。

### 导出流程

```bash
# 1. 导出 ONNX
uv run defect-export --ckpt ./runs/cls/best.pt --out ./runs/cls/model.onnx --dynamic-batch

# 2. 转换为 TensorRT（需要 NVIDIA GPU）
uv run defect-export-trt --onnx ./runs/cls/model.onnx --out ./runs/cls/model.engine --fp16

# 3. TensorRT 推理
uv run defect-infer-trt --engine ./runs/cls/model.engine --image ./test.png --classes ok,ng
```

### 性能基准测试

```bash
uv run defect-infer-trt --engine ./model.engine --image ./test.png --benchmark --iterations 1000
```

### TensorRT 参数

| 参数 | 说明 |
|------|------|
| `--fp16` | 启用 FP16 精度（默认开启，速度快 2x） |
| `--int8` | 启用 INT8 精度（需要校准数据） |
| `--max-batch-size` | 最大 batch size（默认 8） |
| `--workspace` | GPU 工作空间大小 GB（默认 4） |

---

## 🌐 REST API 服务

内置 FastAPI 服务，提供 REST API + Web UI。

### 启动服务

```bash
uv run defect-server --ckpt ./runs/cls/best.pt --port 8000
```

### 访问

- **Web UI**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

### API 接口

#### `POST /predict`

上传图片进行预测。

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test.png" \
  -F "gradcam=true"
```

返回：

```json
{
  "prediction": "ng",
  "confidence": 0.9823,
  "probabilities": {
    "ok": 0.0177,
    "ng": 0.9823
  },
  "latency_ms": 12.5,
  "gradcam_base64": "iVBORw0KGgo..."
}
```

#### `POST /predict/image`

返回 Grad-CAM 叠加图（PNG 图片）。

```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@test.png" \
  -o result.png
```

#### `GET /info`

获取模型信息。

```bash
curl http://localhost:8000/info
```

### Web UI 功能

- 📷 拖拽上传图片
- 🔍 实时预测结果
- 🎨 Grad-CAM 可视化
- 📊 概率分布显示

---

## 🐳 容器化部署（Docker）

适合把 **推理服务**（FastAPI + Web UI）快速部署到服务器/工控机。

### 方式 A：Docker 直接运行

1) 构建镜像：

```bash
docker build -t defectvision:latest .
```

2) 准备权重（示例：把分类模型权重放到 `./weights/best.pt`）：

```bash
mkdir -p weights
cp ./runs/cls/best.pt ./weights/best.pt
```

3) 启动服务（容器内默认读取 `/weights/best.pt`）：

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/weights:/weights:ro" \
  defectvision:latest
```

访问：
- Web UI：`http://localhost:8000`
- API Docs：`http://localhost:8000/docs`

### 方式 B：Docker Compose

1) 准备权重：

```bash
mkdir -p weights
cp ./runs/cls/best.pt ./weights/best.pt
```

2) 启动：

```bash
docker compose up --build
```

> 如果你想换权重，只要替换 `./weights/best.pt` 并重启容器即可。

### 说明

- 默认镜像安装了 `server` + `yolo` 依赖（便于后续扩展 YOLO 推理/视频流）。
- 若你要在容器里跑视频流（OpenCV 窗口显示），通常不推荐；生产环境建议无 UI、只走 API。

---

## 📁 项目结构

```
defectvision/
├── model.py              # 分类模型
├── models/
│   ├── unet.py           # 分割模型
│   └── autoencoder.py    # 异常检测模型
├── train.py              # 分类训练
├── train_seg.py          # 分割训练
├── train_anomaly.py      # 异常检测训练
├── train_yolo.py         # YOLO 训练
├── infer.py              # 分类推理
├── infer_seg.py          # 分割推理
├── infer_anomaly.py      # 异常检测推理
├── infer_yolo.py         # YOLO 推理
├── infer_tensorrt.py     # TensorRT 推理
├── video_stream.py       # 分类视频流检测
├── video_yolo.py         # YOLO 视频流检测
├── export_onnx.py        # ONNX 导出
├── export_tensorrt.py    # TensorRT 导出
├── server.py             # FastAPI 服务
├── predict_dir.py        # 批量推理
├── gradcam.py            # Grad-CAM
├── datasets*.py          # 数据集
└── data/
    ├── generate_synth_defects.py  # OK/NG 数据
    ├── generate_multiclass.py     # 多类缺陷
    ├── generate_segmentation.py   # 分割数据
    └── generate_detection.py      # YOLO 格式数据
```

## 🔧 CLI 命令一览

| 命令 | 说明 |
|------|------|
| **数据生成** | |
| `defect-generate` | OK/NG 二分类数据 |
| `defect-generate-multiclass` | 多类缺陷数据 |
| `defect-generate-seg` | 分割数据（图像 + mask） |
| `defect-generate-det` | YOLO 目标检测数据 |
| **分类** | |
| `defect-train` | 分类训练 |
| `defect-infer` | 单图推理 + Grad-CAM |
| `defect-predict` | 批量推理 |
| **分割** | |
| `defect-train-seg` | U-Net 训练 |
| `defect-infer-seg` | 分割推理 |
| **异常检测** | |
| `defect-train-anomaly` | VAE/AE 训练 |
| `defect-infer-anomaly` | 异常检测推理 |
| **YOLO 目标检测** | |
| `defect-train-yolo` | YOLO 训练 |
| `defect-infer-yolo` | YOLO 推理 |
| **视频流** | |
| `defect-video` | 分类视频流检测 |
| `defect-video-yolo` | YOLO 视频流检测 |
| **部署** | |
| `defect-export` | ONNX 导出 |
| `defect-export-trt` | TensorRT 导出 |
| `defect-infer-trt` | TensorRT 推理 |
| `defect-server` | 启动 REST API 服务 |

## 📊 模式选择指南

| 场景 | 推荐模式 |
|------|----------|
| NG 样本充足，判断好坏 | 分类（二分类） |
| 需要区分缺陷类型 | 分类（多类） |
| 需要定位缺陷位置 | 分割 / YOLO |
| 一张图多个缺陷 | YOLO 目标检测 |
| NG 样本稀少/未知 | 异常检测 |
| 实时产线检测 | 视频流 |
| 生产环境高性能 | TensorRT |
| 快速集成/演示 | REST API |

## 🖥️ 硬件配置建议

| 任务 | CPU | GPU 4GB | GPU 8GB+ |
|------|-----|---------|----------|
| 分类训练 | ✅ 慢 | ✅ | ✅ |
| 分割训练 | ⚠️ 很慢 | ✅ | ✅ |
| 异常检测 | ⚠️ 很慢 | ✅ | ✅ |
| TensorRT | ❌ | ✅ | ✅ |
| REST API | ✅ | ✅ | ✅ |

## 🛠️ 开发

```bash
uv sync --extra dev
uv run ruff format .
uv run ruff check .
uv run pytest
```

## License

MIT
