# ============================================================
# LangChain-AI-Stack Dockerfile
# 多阶段构建，支持国内 pip 镜像 + 大模型文件外部挂载
# ============================================================

# ── 阶段 1：构建 wheels（可选优化，减小最终镜像体积）─────────────
FROM python:3.11-slim AS builder

# 安装编译依赖（paddlepaddle / faiss 等需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ── 阶段 2：最终运行时镜像 ───────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Arthur <arthur@example.com>"
LABEL description="LangChain-AI-Stack - 智能面试助手"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # ── 国内 pip 镜像 ──────────────────────────────
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
    # ── 模型缓存目录（运行时挂载） ─────────────────
    MODELSCOPE_CACHE=/models/funasr \
    ULTRALYTICS_HOME=/models/yolo \
    PADDLEOCR_HOME=/models/paddleocr \
    HF_HOME=/models/huggingface \
    # ── 运行时配置 ────────────────────────────────
    SPEECH_ENGINE=sensevoice \
    MOONSHOT_MODEL=moonshot-v1-8k \
    # ── JavaScript 运行时（uvicorn ASGI 需要） ────
    NODE_OPTIONS="--max-old-space-size=4096"

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV / PaddleOCR 依赖
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # 语音处理依赖
    libsndfile1 \
    ffmpeg \
    # 文档处理依赖
    # （libreoffice 等可根据需要添加，这里用 Python 原生库）
    # 网络工具（健康检查等）
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── 预装轻量依赖（不包含重型模型库）──
# 注意：paddlepaddle / paddleocr / funasr / ultralytics
# 全部在 requirements-docker.txt 中管理，镜像会安装，
# 但模型权重在首次启动时下载到挂载的 /models 目录
COPY requirements-docker.txt* /tmp/

# 使用国内镜像安装 Python 依赖
RUN pip install --no-cache-dir \
    --prefix=/opt/local \
    -r /tmp/requirements-docker.txt 2>&1 | \
    grep -v "^Requirement already satisfied" | \
    grep -v "^Downloading\|Processing\|Building\|Installing" || true

# 复制项目源码
WORKDIR /app
COPY src/ src/
COPY *.py ./
COPY mcp_config.json* ./
COPY yolov8n.pt* ./

# ── 非 root 用户（安全最佳实践）──
RUN groupadd -r appgroup && useradd -r -g appgroup appuser && \
    mkdir -p /models /app/logs && \
    chown -R appuser:appgroup /app /models

USER appuser

# ── 暴露端口 ─────────────────────────────────────────────────
EXPOSE 8000

# ── 健康检查 ─────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ── 启动命令 ─────────────────────────────────────────────────
# 启动时自动下载缺失的模型（首次运行）
# uvicorn 以 ASGI app 运行，支持热重载（开发模式）
ENTRYPOINT []
CMD ["sh", "-c", \
    "echo '[Docker] 启动模型预下载检查...' && \
     python -c \" \
import os, sys; \
print('MODELSCOPE_CACHE:', os.environ.get('MODELSCOPE_CACHE')); \
print('ULTRALYTICS_HOME:', os.environ.get('ULTRALYTICS_HOME')); \
\" && \
     python -c \" \
try:
    from ultralytics import YOLO; m=YOLO('yolov8n.pt'); print('[OK] YOLO 模型就绪')
except Exception as e:
    print('[WARN] YOLO 模型待下载:', e); \
\" && \
     echo '[Docker] 启动 uvicorn...' && \
     exec uvicorn src.main:app \
       --host 0.0.0.0 \
       --port 8000 \
       --log-level info \
       --access-log \
     "]
