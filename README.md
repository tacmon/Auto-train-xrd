# Auto-train-xrd 🚀

本项目是一个专为 XRD（X-ray Diffraction，X射线衍射）数据设计的自动化训练框架。它利用 LLM Agent（大语言模型代理）来自动化管理微调、超参数优化和训练流程，旨在提升 XRD 相位分析模型的性能与开发效率。

## 🌟 核心特性

*   **智能代理 (LLM Agent)**：内置大模型逻辑，能够根据训练日志自动调整策略。
*   **子模块化设计**：集成 [XRD-1.1](https://github.com/tacmon/XRD-1.1) 作为核心算法库，保持算法与框架的解耦。
*   **容器化运行**：提供完整的 Docker 配置，确保环境一致性与快速部署。
*   **灵活配置**：通过 `configs/` 目录轻松定义超参数空间与训练任务。

## 📂 项目结构

```text
Auto-train-xrd/
├── agent/              # 存放 LLM Agent 的逻辑代码、Prompt 模板
├── libs/
│   └── XRD-1.1/       # 核心算法库（Git Submodule）
├── configs/            # 自动化训练的超参空间与任务配置
├── docker/             # Dockerfile 与 docker-compose.yaml 环境定义
├── data/               # 训练数据存放路径
├── main.py             # 项目入口文件
└── idea.md             # 项目设计思路与开发记录
```

## 🚀 快速开始

### 1. 克隆项目
由于项目包含子模块，请务必使用 `--recursive` 参数：
```bash
git clone --recursive https://github.com/tacmon/Auto-train-xrd.git
cd Auto-train-xrd
```

### 2. 环境部署
我们推荐使用 Docker 进行部署。请确保已安装 `docker-compose`：
```bash
cd docker
docker-compose up -d --build
```

### 3. 开始训练
修改 `configs/` 中的配置后，运行主程序：
```bash
python main.py
```

## 🛠️ 配置说明
*   **环境变量**：在根目录创建 `.env` 文件，配置 API Key 等敏感信息（参考 `.env.example`）。
*   **训练脚本**：框架会自动调用 `libs/XRD-1.1` 中的训练脚本，您也可以根据需要自定义接口。

## 📈 项目之间的跳转
本项目是系列工具的一部分，您可以点击下方链接跳转到其他版本：

*   [XRD-1.1](https://github.com/tacmon/XRD-1.1)：专注于 AlN216 的自动化分析工具箱与核心算法。
*   [xrd_server](https://github.com/tacmon/xrd_server)：为 XRD 相位识别提供生产级 API 支持的后端服务。

本项目致力于为您打造一个高效、稳健且智能的 XRD 模型训练体验。如果有任何建议，欢迎随时交流！

> [!NOTE]
> 本项目由 Google Gemini 2.0 Flash 大模型辅助开发与维护。
