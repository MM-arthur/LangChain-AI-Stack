# LangChain-AI-Stack 部署指南

> 容器化部署方案 · 2026-04-21

---

## 目录

1. [方案概述](#方案概述)
2. [LangServe 是什么](#langserve-是什么)
3. [架构设计](#架构设计)
4. [快速部署](#快速部署)
5. [模型文件处理策略](#模型文件处理策略)
6. [环境变量配置](#环境变量配置)
7. [Dockerfile 说明](#dockerfile-说明)
8. [docker-compose 生产部署](#docker-compose-生产部署)
9. [GPU 部署](#gpu-部署)
10. [运维与监控](#运维与监控)
11. [常见问题](#常见问题)

---

## 方案概述

### 项目技术栈

| 组件 | 技术 | 备注 |
|------|------|------|
| Agent 框架 | LangGraph + LangChain | 核心业务逻辑 |
| Web 框架 | FastAPI + uvicorn | 已有的 API 层 |
| 语音识别 | Funasr Paraformer | 约 944MB，ModelScope 下载 |
| 行为分析 | YOLOv8n | 约 6MB，运行时下载 |
| OCR | PaddleOCR + PaddlePaddle | 较重，镜像内安装 |
| LLM | Moonshot AI ( moonshot-v1-8k ) | 需 API Key |

### 部署挑战

```
PaddleOCR/PaddlePaddle  → 镜像内安装（重量级 Python 包）
Funasr Paraformer 模型  → ~944MB，运行时从 ModelScope 拉取
YOLOv8n 权重            → ~6MB，运行时从 Ultralytics CDN 拉取
Sentence-Transformers   → ~500MB，HuggingFace 下载
FAISS                   → 镜像内安装
```

**核心原则：镜像只装包，模型文件走 volume 挂载或首次启动下载。**

---

## LangServe 是什么

### 定义

LangServe 是 LangChain 官方出品的 **将 LangChain Runnables/Chains 部署为 REST API** 的库。它基于 FastAPI，给你自动生成：

- `/invoke` — 单次调用
- `/batch` — 批量调用
- `/stream` — 流式响应
- `/playground/` — 可交互调试页面
- 自动 OpenAPI 文档

### LangServe vs. 自建 FastAPI

| 对比项 | LangServe | 当前项目（自建 FastAPI） |
|--------|-----------|------------------------|
| API 风格 | 标准化 `/invoke /batch/stream` | 自定义 `/ws/chat/...` |
| 接入 LangChain 对象 | 一行 `add_routes` | 需手动封装 |
| WebSocket | 不支持 | ✅ 支持（`/ws/chat/`） |
| 文件上传 | 需额外处理 | ✅ 原生支持 |
| 适用场景 | 纯 LangChain Chain/Runnable | 复杂多模态 Agent |

### 结论

> 当前项目的 API 大量使用了 **文件上传、WebSocket 通话、行为分析** 等自定义逻辑，**LangServe 并不能直接替代**。  
> LangServe 更适合纯 LLM 调用场景（问答、摘要、翻译等轻量 Runnable）。  
> 当前项目 **保留已有的 FastAPI 架构**，Docker 部署也基于此。

**但如果未来需要把某些 LangChain Chain 单独暴露给外部使用，可以这样用 LangServe：**

```python
# 单独部署一个 langserve app，只服务某个 chain
from fastapi import FastAPI
from langserve import add_routes
from src.multi_agent import some_chain  # 单个 LangChain Runnable

app = FastAPI(title="LangChain-AI-Stack API")
add_routes(app, some_chain, path="/chain/chat")
```

---

## 架构设计

```
                           ┌─────────────────────┐
  浏览器 / 客户端            │   Docker Container   │
                           │                     │
  WebSocket ──────────────▶ │  ┌───────────────┐  │
  HTTP API  ──────────────▶ │  │  uvicorn      │  │
  文件上传   ──────────────▶ │  │  src.main:app │  │
                           │  └───────┬───────┘  │
                           │          │          │
                           │  ┌───────▼───────┐  │
                           │  │ AgentSingleton│  │  ← 进程启动时编译一次
                           │  │ SessionManager│  │
                           │  └───────┬───────┘  │
                           └──────────┼──────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                  │
             ┌──────▼──────┐  ┌───────▼────┐  ┌───────▼─────┐
             │ Funasr      │  │ YOLOv8n    │  │ Moonshot    │
             │ (语音识别)   │  │ (行为分析)  │  │ LLM API     │
             │ ~944MB      │  │ ~6MB       │  │ (外网调用)   │
             └─────────────┘  └────────────┘  └─────────────┘
                   │               │
          ┌────────▼────┐  ┌───────▼──────┐
          │ ModelScope  │  │ HuggingFace │
          │ (国内镜像)   │  │ / Ultralytics│
          └─────────────┘  └─────────────┘
```

### 模型挂载策略

```
宿主机目录                    Docker 容器内路径
─────────────────────────────────────────────────────
/data/models/funasr    →   /models/funasr      (MODELSCOPE_CACHE)
/data/models/yolo      →   /models/yolo        (ULTRALYTICS_HOME)
/data/models/huggingface→   /models/huggingface (HF_HOME)
/data/models/paddleocr →   /models/paddleocr   (PADDLEOCR_HOME)
```

首次容器启动时，模型自动下载到挂载目录；后续重启直接从本地加载，无需再次下载。

---

## 快速部署

### 前置条件

```bash
# 安装 Docker（Linux）
curl -fsSL https://get.docker.com | sh
sudo systemctl enable docker
sudo systemctl start docker

# 安装 Docker Compose V2
sudo apt-get install docker-compose-v2  # Ubuntu/Debian
# 或
brew install docker-compose            # macOS
```

### 构建镜像

```bash
cd /path/to/LangChain-AI-Stack

# 构建（使用 BuildKit 加速构建）
DOCKER_BUILDKIT=1 docker build \
  --tag langchain-ai-stack:latest \
  --progress=plain \
  .

# 查看镜像大小
docker images langchain-ai-stack:latest
```

### 运行容器（开发模式）

```bash
# 创建模型缓存目录
mkdir -p /data/models/{funasr,yolo,huggingface,paddleocr}

# 首次运行（模型会在容器启动时自动下载）
docker run -d \
  --name langchain-ai-stack-dev \
  -p 8000:8000 \
  -p 8080:8080 \
  -v /data/models:/models \
  -e MOONSHOT_API_KEY=your_api_key_here \
  -e TAVILY_API_KEY=your_tavily_key_here \
  -e SPEECH_ENGINE=sensevoice \
  --restart unless-stopped \
  langchain-ai-stack:latest

# 查看日志
docker logs -f langchain-ai-stack-dev

# 进入容器调试
docker exec -it langchain-ai-stack-dev bash
```

### 验证部署

```bash
# API 健康检查
curl http://localhost:8000/api/health

# WebSocket 测试
wscat -c ws://localhost:8000/ws/chat/test-session
# 发送：{"type": "chat", "content": "你好"}
```

---

## 模型文件处理策略

### 为什么不能 COPY 到镜像

| 模型 | 路径 | 大小 | 处理方式 |
|------|------|------|---------|
| Funasr Paraformer | `~/.cache/modelscope/hub/` | ~944MB | 挂载 `/models/funasr` |
| YOLOv8n | `ultralytics` 自动下载 | ~6MB | 挂载 `/models/yolo` |
| PaddleOCR | `~/.paddleocr/` | ~200MB | 挂载 `/models/paddleocr` |
| Sentence-Transformers | `~/.cache/huggingface/` | ~500MB | 挂载 `/models/huggingface` |

直接 COPY 造成的问题：
- 镜像体积膨胀 2~3 GB
- 模型更新需要重新 build 镜像
- 不同环境（GPU/CPU）模型不通用

### 预下载脚本（可选，提前下载模型）

在宿主机上执行：

```bash
# 创建模型下载脚本
cat > /data/models/download_models.sh << 'EOF'
#!/bin/bash
set -e

echo "=== 预下载 LangChain-AI-Stack 模型 ==="

# YOLOv8n
echo "[1/3] 下载 YOLOv8n..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Funasr Paraformer（从 ModelScope）
echo "[2/3] 下载 Funasr Paraformer..."
python3 -c "from funasr import AutoModel; AutoModel(model='paraformer-zh', disable_update=True)"

# Sentence-Transformers
echo "[3/3] 下载 Sentence-Transformers..."
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

echo "=== 模型下载完成 ==="
EOF

mkdir -p /data/models && cd /data/models
docker run --rm \
  -v /data/models:/models \
  -e MODELSCOPE_CACHE=/models/funasr \
  -e ULTRALYTICS_HOME=/models/yolo \
  -e HF_HOME=/models/huggingface \
  -e PADDLEOCR_HOME=/models/paddleocr \
  langchain-ai-stack:latest \
  bash /data/models/download_models.sh
```

### 模型缓存复用

```bash
# 停止容器
docker stop langchain-ai-stack

# 重新启动（模型已存在于宿主机 /data/models/）
docker start langchain-ai-stack

# 日志中看到 [OK] 即表示模型从本地加载，跳过下载
```

---

## 环境变量配置

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `MOONSHOT_API_KEY` | **是** | - | Moonshot AI API 密钥 |
| `TAVILY_API_KEY` | 否 | - | Tavily 搜索 API |
| `SPEECH_ENGINE` | 否 | `sensevoice` | `sensevoice` 或 `paddlespeech` |
| `MOONSHOT_MODEL` | 否 | `moonshot-v1-8k` | Moonshot 模型名 |
| `MODELSCOPE_CACHE` | 否 | `/models/funasr` | Funasr 模型缓存目录 |
| `ULTRALYTICS_HOME` | 否 | `/models/yolo` | YOLO 模型缓存目录 |
| `HF_HOME` | 否 | `/models/huggingface` | HuggingFace 模型缓存 |
| `PADDLEOCR_HOME` | 否 | `/models/paddleocr` | PaddleOCR 模型缓存 |
| `PIP_INDEX_URL` | 否 | 清华镜像 | pip 镜像地址 |
| `LOG_LEVEL` | 否 | `info` | 日志级别 |

### .env 示例文件

```env
# .env.docker
MOONSHOT_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx
SPEECH_ENGINE=sensevoice
MOONSHOT_MODEL=moonshot-v1-8k

# 模型缓存（挂载的宿主机目录）
MODELSCOPE_CACHE=/models/funasr
ULTRALYTICS_HOME=/models/yolo
HF_HOME=/models/huggingface
PADDLEOCR_HOME=/models/paddleocr

# 国内 pip 镜像
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
```

---

## Dockerfile 说明

### 多阶段构建

```dockerfile
FROM python:3.11-slim AS builder    # 阶段1：编译依赖（可选）
FROM python:3.11-slim AS runtime    # 阶段2：最终运行镜像
```

### 关键设计点

#### 1. 国内 pip 镜像

```dockerfile
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
```

> 清华镜像覆盖主流深度学习包（PaddlePaddle、faiss 等有预编译 wheel）。  
> 如遇清华镜像缺失包，临时切换：`pip install xxx -i https://pypi.org/simple`

#### 2. 模型目录外部挂载

```dockerfile
ENV MODELSCOPE_CACHE=/models/funasr
ENV ULTRALYTICS_HOME=/models/yolo
```

所有大模型文件不进入镜像，通过 `-v /host/path:/container/path` 挂载。

#### 3. opencv-python-headless

```dockerfile
# ✅ 正确：Docker 专用，无 GUI 依赖
opencv-python-headless==4.8.1.78

# ❌ 避免：依赖 libgl1-mesa 等 GUI 库，镜像更大
opencv-python==4.8.1.78
```

#### 4. 非 root 用户

```dockerfile
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
USER appuser
```

安全最佳实践：容器进程不以 root 运行。

#### 5. Uvicorn 启动命令

```bash
uvicorn src.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  --access-log
```

- `src.main:app`：FastAPI 应用实例
- `--host 0.0.0.0`：接受任意来源连接
- `--access-log`：记录每个请求（便于排查）

---

## docker-compose 生产部署

### docker-compose.yml

```yaml
version: "3.9"

services:
  # ── 主服务 ──────────────────────────────────────────────
  langchain-ai-stack:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
    image: langchain-ai-stack:latest
    container_name: langchain-ai-stack
    restart: unless-stopped
    ports:
      - "8000:8000"    # API
      - "8080:8080"    # 前端静态文件
    volumes:
      # 模型缓存（宿主机 → 容器）
      - /data/models/funasr:/models/funasr
      - /data/models/yolo:/models/yolo
      - /data/models/huggingface:/models/huggingface
      - /data/models/paddleocr:/models/paddleocr
      # 源码热重载（开发模式可选）
      # - ./src:/app/src:ro
      # 环境变量
      - ./secrets/.env:/app/.env:ro
    environment:
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
      - SPEECH_ENGINE=sensevoice
      - LOG_LEVEL=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
    deploy:
      resources:
        limits:
          memory: 16G      # OCR + PaddlePaddle 较吃内存
        reservations:
          memory: 4G
    networks:
      - langchain-net

  # ── 前端（可选，单独容器）────────────────────────────────
  frontend:
    image: nginx:alpine
    container_name: langchain-frontend
    restart: unless-stopped
    ports:
      - "8080:80"
    volumes:
      - ./src/ui:/usr/share/nginx/html:ro
    depends_on:
      - langchain-ai-stack
    networks:
      - langchain-net

networks:
  langchain-net:
    driver: bridge
```

### 启动生产服务

```bash
# 创建密钥目录（API Key 不写入代码仓库）
mkdir -p /data/secrets
echo "MOONSHOT_API_KEY=sk-xxx" > /data/secrets/.env
chmod 600 /data/secrets/.env

# 拉取/构建并启动
docker-compose up -d --build

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f --tail=100

# 停止
docker-compose down
```

---

## GPU 部署

### NVIDIA GPU（CUDA）

```yaml
# docker-compose.gpu.yml 片段
services:
  langchain-ai-stack:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
    # 替换 CPU paddlepaddle 为 GPU 版本
    args:
      - PADDLEPADDLE=paddlepaddle-gpu
```

```bash
# 构建支持 GPU 的镜像
docker build \
  --build-arg PADDLEPADDLE=paddlepaddle-gpu \
  -t langchain-ai-stack:gpu \
  .

# 运行时需要 nvidia-container-toolkit
docker-compose -f docker-compose.gpu.yml up -d
```

### 修改 requirements-docker.txt（GPU 版本）

```txt
# CPU 版本
paddlepaddle==2.5.2

# GPU 版本（CUDA 11.8）
# paddlepaddle-gpu==2.5.2.post118
# GPU 版本（CUDA 12.x）
# paddlepaddle-gpu==2.5.2.post121
```

---

## 运维与监控

### 日志管理

```bash
# 查看实时日志
docker logs -f langchain-ai-stack

# 限制日志文件大小（Docker daemon 配置）
# /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  }
}
```

### 健康检查端点

```bash
# 容器内健康检查（自动）
curl http://localhost:8000/api/health

# 外部监控
curl -f --max-time 5 http://your-host:8000/api/health || echo "SERVICE DOWN"
```

### 资源限制建议

| 场景 | 内存 | CPU | 说明 |
|------|------|-----|------|
| 开发 / 轻量 | 4GB | 2核 | 单用户测试 |
| 生产（无 GPU） | 16GB | 4核 | PaddleOCR 较吃内存 |
| 生产（GPU 加速） | 8GB | 2核 + GPU | YOLO/Paddle OCR 加速 |

### 备份会话状态

AgentSingleton 的 MemorySaver 是内存存储，容器重启后会话丢失。  
如需持久化，在 `SessionManager` 中替换为 `SQLiteSaver`：

```python
# src/main.py 修改
from langgraph.checkpoint.sqlite import SqliteSaver

# 生产环境使用 SQLite持久化 checkpointer
checkpointer = SqliteSaver.from_conn_string("/data/sessions.db")
```

并添加 volume 挂载：
```yaml
volumes:
  - /data/sessions.db:/app/sessions.db
```

---

## 常见问题

### Q1: 容器启动后模型下载很慢

**原因**：首次启动从 ModelScope / HuggingFace CDN 下载，在国际带宽受限环境可能极慢。

**解决**：
```bash
# 方案A：手动在宿主机预下载，然后挂载目录
# ModelScope 镜像（国内）
export MODELSCOPE_CACHE=/data/models/funasr
python -c "from modelscope import snapshot_download; snapshot_download('Zhou_Wei/Paraformer')"

# 方案B：配置代理
docker run -e HTTP_PROXY=http://proxy:8080 ...

# 方案C：修改 Funasr 数据源（编辑 funasr 源码或环境变量）
```

### Q2: PaddleOCR / PaddlePaddle 安装失败

**原因**：清华镜像没有所有平台的预编译 wheel。

**解决**：
```dockerfile
# 改用官方源（最后回退）
RUN pip install paddlepaddle paddleocr \
    -i https://pypi.org/simple \
    --timeout 300
```

### Q3: YOLOv8n 报告内存不足

**原因**：容器内存限制太小（ultralytics 默认加载整个模型到内存）。

**解决**：增加容器内存至 4GB 以上，或使用 `yolov8n.pt` 轻量版本（已默认使用）。

### Q4: 如何更新代码而不重建镜像？

**方案**：源码挂载（开发模式）：

```bash
docker run -v $(pwd)/src:/app/src langchain-ai-stack:latest
```

**注意**：修改 `multi_agent.py` 后 LangGraph 单例不会自动重新编译，需重启容器。

### Q5: 是否需要 LangServe？

如前文所述，**不需要**。当前 FastAPI 架构已完整覆盖项目需求，LangServe 适合将单个 LangChain Chain 作为轻量微服务暴露的场景。项目已有：
- 自定义 REST API（文件上传、OCR、文档解析）
- WebSocket 实时对话
- Session 管理

这些都不是 LangServe 的设计目标。

### Q6: 镜像体积太大（>10GB）

**优化方向**：

```dockerfile
# 1. 使用 lighter 基础镜像
FROM python:3.11-slim

# 2. 分离构建（builder 阶段）
# 已在多阶段构建中实现

# 3. 分离 dev/prod 依赖
# → 开发用：完整 requirements.txt
# → 生产用：requirements-docker.txt（不含 pytest 等）

# 4. 清理 apt 缓存（已在 RUN 中执行）
```

---

## 参考资料

- [LangServe 官方文档](https://github.com/langchain-ai/langserve)
- [LangGraph Platform（生产级部署方案）](https://langchain-ai.github.io/langgraph/cloud/)
- [PaddleOCR Docker 部署](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/deploy/docker/)
- [ModelScope 模型下载](https://www.modelscope.cn/docs)
- [Ultralytics YOLO Docker](https://docs.ultralytics.com/zh/docker/)
- [清华 pip 镜像使用帮助](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
