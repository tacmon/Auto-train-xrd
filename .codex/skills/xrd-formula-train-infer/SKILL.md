---
name: xrd-formula-train-infer
description: 当用户只给出化学式A，以及可选的化学式B，希望自动从 Materials Project 选 CIF、训练 XRD-1.1 的 XRD/PDF 模型、用 ./data 下真实谱图做推理并保存结果时触发。常见说法：用公式A/B训练、自动挑 CIF、对 data 真实数据跑预测、结果不满意就重训。
---

# XRD Formula Train Infer

这个 skill 面向当前仓库 `Auto-train-xrd`，只处理 `libs/XRD-1.1/Novel-Space` 这套工作流。

## 输入

用户至少提供：
- 化学式 A

用户可选提供：
- 化学式 B
- `./data` 下的某个子目录或文件路径

默认行为：
- 如果用户没指定真实数据路径，递归读取仓库 `./data` 下全部 `*.txt`、`*.xy`、`*.gk`
- 如果用户没指定 B，先生成 3 个对比度较大的候选 B 再请用户选

## 强制约束

1. 先使用 `docker/docker-compose.yaml` 启动容器，不要假设宿主机 Python 环境可直接运行。
2. 所有训练和推理都在容器里执行；所有数据和产物都保存在宿主机挂载目录。
3. 每次训练尝试都必须使用新的 `run_name`。不要覆盖旧结果，这样“备份”天然成立。
4. `construct_xrd_model.py` 训练前，`Novel-Space/References` 不能存在；`construct_pdf_model.py` 训练前，`Novel-Space/Models` 不能存在。
5. A 是正样本。用户说“不满意”后，下一轮仍要保留 A 这个化学式，只能换 A 的候选材料、换 B，或两者都换，但不能把 A 换成别的公式。
6. 如果缺少 `MP_API_KEY`，必须立即停下并告诉用户把它放到仓库根目录 `.env` 或 `libs/XRD-1.1/Novel-Space/.env`。

## 先用哪些脚本

优先使用本 skill 自带脚本，不要重写临时命令流：

- 候选查询和 CIF 下载：
  `python3 .codex/skills/xrd-formula-train-infer/scripts/mp_formula_tool.py`
- 一次完整训练+推理：
  `bash .codex/skills/xrd-formula-train-infer/scripts/run_pipeline.sh`

## 工作流

### Step 1: 检查前置条件

- 确认仓库路径是当前项目根目录
- 检查 `docker/docker-compose.yaml` 存在
- 检查 `libs/XRD-1.1/Novel-Space` 存在
- 检查 `MP_API_KEY` 是否可用

### Step 2: 解析 A/B 候选

运行：

```bash
python3 .codex/skills/xrd-formula-train-infer/scripts/mp_formula_tool.py candidates --formula-a "A" [--formula-b "B"]
```

处理规则：
- 如果 A 只返回 1 个明显更优的稳定候选，可以直接采用，但要把理由告诉用户
- 如果 A 有多个可行候选，向用户展示前 3 个，至少说明：
  - `material_id`
  - `formula_pretty`
  - `space group`
  - `energy_above_hull`
  - `top peaks (20-60 deg)`
- 如果 B 未提供，使用脚本给出的启发式 B 建议并让用户选一个
- 如果 B 已提供但有多个候选，也按同样方式向用户展示前 3 个

### Step 3: 运行一次完整流水线

在拿到最终的 `material_id_a` 和 `material_id_b` 后运行：

```bash
bash .codex/skills/xrd-formula-train-infer/scripts/run_pipeline.sh \
  --formula-a "A" \
  --formula-b "B" \
  --material-id-a "mp-xxxx" \
  --material-id-b "mp-yyyy" \
  [--spectra-source "./data/某个子目录或文件"]
```

这个脚本会：
- 启动或重建容器
- 创建新的命名运行目录
- 把真实谱图复制到本次运行的 `Spectra`
- 下载两份 CIF 到本次运行的 `All_CIFs`
- 训练 XRD 模型
- 训练 PDF 模型
- 对真实谱图推理
- 保存 `result.csv`、原始谱图预览图、模型和参考相

### Step 4: 结果汇报

完成后必须把以下路径告诉用户：

- 本轮运行目录名
- 预测结果：`libs/XRD-1.1/Novel-Space/soft_link/All_CIFs/<run_name>/results/result.csv`
- 模型目录：`libs/XRD-1.1/Novel-Space/soft_link/All_CIFs/<run_name>/Models`
- 参考相目录：`libs/XRD-1.1/Novel-Space/soft_link/All_CIFs/<run_name>/References`
- 本轮真实谱图副本：`libs/XRD-1.1/Novel-Space/soft_link/Spectra/<run_name>`
- 真实谱图预览：`libs/XRD-1.1/Novel-Space/soft_link/figure/<run_name>`

然后明确要求用户只回复：
- `满意`
- `不满意`

### Step 5: 用户回复“不满意”时

不要覆盖当前结果。当前 run 已经是备份。

按这个顺序处理：
- 保持化学式 A 不变
- 优先尝试更换 A 的另一个 MP 候选，或更换 B
- 生成新的 `run_name`
- 重新执行 Step 2 到 Step 4

## 交流规则

- 当需要用户选择候选 A 或候选 B 时，只问一个简短问题
- 不要把整个 MP 检索原始输出都贴给用户，只总结前三个候选
- 如果真实数据来自 `./data` 的多个不同实验体系，默认全量递归读取；只有当用户明确要限制范围时才缩小

## 禁止事项

- 不要调用 `train.sh` 或 `setup_links.sh` 的交互模式
- 不要在同一个 `run_name` 下反复清空重来
- 不要在训练前保留旧的 `References` 或 `Models`
- 不要在没有 `MP_API_KEY` 的情况下继续尝试 Materials Project 下载
