FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 系统依赖：OpenCV 运行时需要
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 仅复制安装所需文件，减少构建层体积
COPY pyproject.toml README.md ./
COPY vision_ai ./vision_ai

# 安装项目（包含 FastAPI + YOLO 依赖）
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[server,yolo]"

EXPOSE 8000

# 默认启动 REST API 服务，挂载权重到 /weights
ENTRYPOINT ["vision-server"]
CMD ["--ckpt", "/weights/best.pt", "--host", "0.0.0.0", "--port", "8000"]

