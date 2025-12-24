"""FastAPI 服务端：提供 REST API 进行缺陷检测"""

import io
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from vision_ai.datasets import build_transforms
from vision_ai.gradcam import compute_gradcam, overlay_cam
from vision_ai.logger import get_logger, setup_logger
from vision_ai.model import build_model
from vision_ai.utils import device_from_arg, load_ckpt


class InferenceEngine:
    """推理引擎封装"""

    def __init__(
        self,
        ckpt_path: Path,
        device: str = "auto",
    ):
        self.logger = get_logger("vision_ai")
        self.device = device_from_arg(device)
        self.logger.info(f"Loading model from: {ckpt_path}")
        self.logger.info(f"Using device: {self.device}")

        # 加载检查点
        ckpt = load_ckpt(ckpt_path, map_location=str(self.device))
        self.idx_to_class: Dict[int, str] = ckpt["idx_to_class"]
        self.image_size: int = int(ckpt["image_size"])
        backbone: str = ckpt.get("backbone", "smallcnn")
        pretrained: bool = ckpt.get("pretrained", True)

        # 构建模型
        self.model = build_model(
            in_channels=1,
            num_classes=len(self.idx_to_class),
            backbone=backbone,
            pretrained=pretrained,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # 预处理
        self.transform = build_transforms(self.image_size, train=False)

        self.logger.info(f"Model loaded: {backbone}, classes={list(self.idx_to_class.values())}")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """预处理图像"""
        x = self.transform(image.convert("RGB"))
        return x.unsqueeze(0).to(self.device)

    def predict(
        self,
        image: Image.Image,
        with_gradcam: bool = False,
    ) -> Dict:
        """
        执行预测

        Args:
            image: PIL 图像
            with_gradcam: 是否生成 Grad-CAM

        Returns:
            预测结果字典
        """
        t0 = time.perf_counter()

        x = self.preprocess(image)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

        pred_idx = int(probs.argmax(dim=1).item())
        pred_class = self.idx_to_class[pred_idx]
        pred_prob = float(probs[0, pred_idx].item())

        # 所有类别的概率
        class_probs = {
            self.idx_to_class[i]: float(probs[0, i].item())
            for i in range(len(self.idx_to_class))
        }

        result = {
            "prediction": pred_class,
            "confidence": pred_prob,
            "probabilities": class_probs,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

        # Grad-CAM
        if with_gradcam:
            cam = compute_gradcam(self.model, x, pred_idx)
            cam_overlay = overlay_cam(image, cam, alpha=0.4)
            result["gradcam"] = cam_overlay

        return result


# 全局引擎实例
_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    """获取推理引擎实例"""
    global _engine
    if _engine is None:
        raise RuntimeError("Inference engine not initialized. Call init_engine() first.")
    return _engine


def init_engine(ckpt_path: str, device: str = "auto") -> None:
    """初始化推理引擎"""
    global _engine
    setup_logger("vision_ai")
    _engine = InferenceEngine(Path(ckpt_path), device)


def create_app(ckpt_path: str, device: str = "auto"):
    """创建 FastAPI 应用"""
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError:
        raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn python-multipart")

    # 初始化引擎
    init_engine(ckpt_path, device)

    app = FastAPI(
        title="Vision AI 缺陷检测 API",
        description="工业质检缺陷检测服务：上传图片，返回预测结果和 Grad-CAM 可视化",
        version="0.2.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """主页：显示上传界面"""
        return get_html_page()

    @app.get("/health")
    async def health():
        """健康检查"""
        return {"status": "ok", "model_loaded": _engine is not None}

    @app.get("/info")
    async def info():
        """获取模型信息"""
        engine = get_engine()
        return {
            "classes": list(engine.idx_to_class.values()),
            "image_size": engine.image_size,
            "device": str(engine.device),
        }

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),
        gradcam: bool = True,
    ):
        """
        图像分类预测

        - **file**: 上传的图片文件
        - **gradcam**: 是否返回 Grad-CAM 可视化
        """
        # 读取图片
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

        # 预测
        engine = get_engine()
        result = engine.predict(image, with_gradcam=gradcam)

        # 如果有 Grad-CAM，转为 base64
        if "gradcam" in result:
            import base64

            buf = io.BytesIO()
            result["gradcam"].save(buf, format="PNG")
            buf.seek(0)
            result["gradcam_base64"] = base64.b64encode(buf.read()).decode("utf-8")
            del result["gradcam"]

        return result

    @app.post("/predict/image")
    async def predict_image(
        file: UploadFile = File(...),
    ):
        """返回 Grad-CAM 叠加图（直接返回图片）"""
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

        engine = get_engine()
        result = engine.predict(image, with_gradcam=True)

        # 返回图片
        buf = io.BytesIO()
        result["gradcam"].save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    return app


def get_html_page() -> str:
    """返回简单的 HTML 前端页面"""
    return """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision AI 缺陷检测</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e8e8e8;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        
        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle {
            text-align: center;
            color: #8892b0;
            margin-bottom: 40px;
        }
        
        .upload-zone {
            background: rgba(255, 255, 255, 0.05);
            border: 2px dashed rgba(0, 217, 255, 0.3);
            border-radius: 16px;
            padding: 60px 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 30px;
        }
        
        .upload-zone:hover {
            border-color: #00d9ff;
            background: rgba(0, 217, 255, 0.05);
        }
        
        .upload-zone.dragover {
            border-color: #00ff88;
            background: rgba(0, 255, 136, 0.1);
        }
        
        .upload-icon {
            font-size: 4rem;
            margin-bottom: 20px;
        }
        
        .upload-text {
            font-size: 1.2rem;
            color: #8892b0;
        }
        
        #fileInput {
            display: none;
        }
        
        .results {
            display: none;
            gap: 30px;
        }
        
        .results.show {
            display: grid;
            grid-template-columns: 1fr 1fr;
        }
        
        @media (max-width: 768px) {
            .results.show {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(10px);
        }
        
        .card h3 {
            color: #00d9ff;
            margin-bottom: 16px;
            font-size: 1.1rem;
        }
        
        .image-container {
            width: 100%;
            aspect-ratio: 1;
            border-radius: 12px;
            overflow: hidden;
            background: #0a0a0f;
        }
        
        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        
        .prediction {
            margin-top: 20px;
        }
        
        .prediction-label {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 8px;
        }
        
        .prediction-label.ok {
            color: #00ff88;
        }
        
        .prediction-label.ng {
            color: #ff4757;
        }
        
        .confidence {
            font-size: 1.2rem;
            color: #8892b0;
        }
        
        .probabilities {
            margin-top: 20px;
        }
        
        .prob-bar {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }
        
        .prob-label {
            width: 80px;
            font-size: 0.9rem;
        }
        
        .prob-track {
            flex: 1;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin: 0 12px;
        }
        
        .prob-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .prob-value {
            width: 60px;
            text-align: right;
            font-size: 0.9rem;
            color: #8892b0;
        }
        
        .latency {
            text-align: center;
            color: #8892b0;
            font-size: 0.9rem;
            margin-top: 20px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(0, 217, 255, 0.2);
            border-top-color: #00d9ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Vision AI</h1>
        <p class="subtitle">工业质检缺陷检测系统</p>
        
        <div class="upload-zone" id="uploadZone">
            <div class="upload-icon">📷</div>
            <p class="upload-text">点击或拖拽图片到此处上传</p>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>正在分析...</p>
        </div>
        
        <div class="results" id="results">
            <div class="card">
                <h3>📸 原图 / Grad-CAM</h3>
                <div class="image-container">
                    <img id="resultImage" src="" alt="Result">
                </div>
            </div>
            
            <div class="card">
                <h3>📊 预测结果</h3>
                <div class="prediction">
                    <div class="prediction-label" id="predLabel">-</div>
                    <div class="confidence" id="confidence">-</div>
                </div>
                <div class="probabilities" id="probabilities"></div>
                <div class="latency" id="latency"></div>
            </div>
        </div>
    </div>
    
    <script>
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        
        uploadZone.addEventListener('click', () => fileInput.click());
        
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });
        
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                processFile(file);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                processFile(file);
            }
        });
        
        async function processFile(file) {
            loading.classList.add('show');
            results.classList.remove('show');
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/predict?gradcam=true', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                alert('预测失败: ' + error.message);
            } finally {
                loading.classList.remove('show');
            }
        }
        
        function displayResults(data) {
            results.classList.add('show');
            
            // 显示 Grad-CAM 图
            if (data.gradcam_base64) {
                document.getElementById('resultImage').src = 'data:image/png;base64,' + data.gradcam_base64;
            }
            
            // 预测标签
            const predLabel = document.getElementById('predLabel');
            predLabel.textContent = data.prediction.toUpperCase();
            predLabel.className = 'prediction-label ' + data.prediction.toLowerCase();
            
            // 置信度
            document.getElementById('confidence').textContent = 
                `置信度: ${(data.confidence * 100).toFixed(1)}%`;
            
            // 概率条
            const probsDiv = document.getElementById('probabilities');
            probsDiv.innerHTML = '';
            
            for (const [cls, prob] of Object.entries(data.probabilities)) {
                const bar = document.createElement('div');
                bar.className = 'prob-bar';
                bar.innerHTML = `
                    <span class="prob-label">${cls}</span>
                    <div class="prob-track">
                        <div class="prob-fill" style="width: ${prob * 100}%"></div>
                    </div>
                    <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
                `;
                probsDiv.appendChild(bar);
            }
            
            // 延迟
            document.getElementById('latency').textContent = 
                `推理耗时: ${data.latency_ms.toFixed(1)} ms`;
        }
    </script>
</body>
</html>
"""


def main() -> None:
    """启动 FastAPI 服务"""
    import argparse

    parser = argparse.ArgumentParser(description="启动 Vision AI 缺陷检测服务")
    parser.add_argument("--ckpt", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cpu/cuda/mps)")
    parser.add_argument("--reload", action="store_true", help="开发模式（自动重载）")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not installed. Install with: pip install uvicorn")

    # 创建应用
    app = create_app(args.ckpt, args.device)

    print(f"\n🚀 Starting Vision AI Server...")
    print(f"   Model: {args.ckpt}")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"   API Docs: http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

