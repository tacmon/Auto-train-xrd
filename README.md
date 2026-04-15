# Auto-train-xrd

本仓库现在是一个围绕 [XRD-1.1](https://github.com/tacmon/XRD-1.1) 的本地训练工作区，用于：

- 用 Docker 提供一致的训练和推理环境
- 管理真实 XRD 数据目录
- 通过本仓库内的 Codex skill 自动完成 CIF 选择、训练、推理和结果保存

它不再维护单独的“大模型 API 客户端 / Agent 框架 / 超参数自动搜索入口”。如果你在使用 Codex，就直接让 Codex 调用仓库里的 skill 和脚本即可。

## 当前结构

```text
Auto-train-xrd/
├── .codex/
│   └── skills/
│       └── xrd-formula-train-infer/   # 项目本地 Codex skill
├── libs/
│   └── XRD-1.1/                       # 核心算法库（git submodule）
├── docker/                            # 训练容器定义
├── data/                              # 真实 XRD 数据
├── idea.md                            # 历史设计记录
└── README.md
```

## 当前入口

### 1. 容器环境

```bash
docker compose -f docker/docker-compose.yaml up -d --build
```

### 2. Codex skill

项目级 skill 位于：

```text
.codex/skills/xrd-formula-train-infer/
```

这个 skill 会围绕 `libs/XRD-1.1/Novel-Space` 完成：

- 解析化学式 A 和可选的 B
- 从 Materials Project 查询候选结构并下载 CIF
- 训练 XRD 模型和 PDF 模型
- 读取 `./data` 下真实谱图并保存预测结果
- 当用户回复“不满意”时，保留 A 并重新尝试新的候选组合

### 3. 直接脚本入口

如果你不通过对话调用，也可以直接使用这两个脚本：

```bash
python3 .codex/skills/xrd-formula-train-infer/scripts/mp_formula_tool.py --help
bash .codex/skills/xrd-formula-train-infer/scripts/run_pipeline.sh --help
```

## 环境变量

根目录 `.env` 现在只需要保存与 Materials Project 相关的密钥：

```bash
MP_API_KEY=your_materials_project_api_key_here
```

不再需要额外配置大模型 API 地址、模型名或私有 SDK Key。

## 说明

- `docker/docker-compose.yaml` 已挂载整个仓库到容器内，因此容器可以直接访问当前项目文件。
- 训练和推理的真实实现仍在 `libs/XRD-1.1/Novel-Space`。
- `idea.md` 是历史设计草稿，不代表当前仓库的实际入口和目录结构。
