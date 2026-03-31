# ── XRD AutoAnalyzer API 服务镜像 ──
# FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# 避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# 复制 py_env 目录内容到 /workspace
COPY py_env/ /workspace/

# 安装项目依赖（pip install -e . 安装 autoXRD 包）
# 同时安装基础及计算所需额外的依赖
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir pandas mp-api

# 默认提供交互式终端，以防止未指定前台服务导致容器启动后立即退出
CMD ["/bin/bash"]
