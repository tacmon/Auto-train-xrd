---
name: xrd-target-negative-loop
description: 当用户只给一个目标化学式 A，希望在当前 Auto-train-xrd 仓库中自动寻找一个或多个负样本、由主 Agent 负责选样与评估、由 subagent 负责容器内训练/推理，并基于真实数据的 processed_result.csv/score.json 迭代优化对 A 的区分效果时使用。任务完成与否必须按真实数据测试结果经过处理之后来判断，而不是按原始 result.csv、训练 loss 或理论谱图表现判断。常见说法：只给 A 自动找负样本、单目标迭代训练、让主 agent 自动重训直到真实数据 processed_result 达标、按 processed_result.csv 的效果决定继续或停止。
---

# XRD Target Negative Loop

## Overview

这个 skill 面向当前仓库 `Auto-train-xrd`，只处理 `libs/XRD-1.1/Novel-Space` 这套工作流。

最终目标不是“随便训练一个二分类器”，而是让主 Agent 在固定目标公式 A 的前提下，自动选择负样本组合、委托 subagent 训练、对真实数据做推理、把 `result.csv` 后处理成 `processed_result.csv`，并且优先在真实数据上做阈值搜索与评分，再根据 `processed_result.csv`/`score.json` 决定是否继续迭代。

## 输入

用户至少提供：
- 一个目标化学式 A

用户可选提供：
- 目标结构的精确要求，例如指定空间群、稳定性、MP material ID 或本地 CIF
- 真实数据范围，默认递归读取仓库 `./data` 下全部 `*.txt`、`*.xy`、`*.gk`
- 候选负样本范围或禁止使用的负样本
- 人工标签文件，或用于从文件名/目录名推断弱标签的规则
- 满意阈值，例如“真实数据 processed_result 准确率至少 95%”

## 这个 skill 依赖什么

先复用现有 skill 和脚本，不要重写整套训练流程：

- 结构查询与 MP 下载：
  `.codex/skills/xrd-formula-train-infer/scripts/mp_formula_tool.py`
- 双相训练推理：
  `.codex/skills/xrd-formula-train-infer/scripts/run_pipeline.sh`
- 多相或混合来源训练推理：
  `.codex/skills/xrd-formula-train-infer/scripts/run_multiphase_pipeline.sh`
- 本 skill 的非交互后处理：
  `.codex/skills/xrd-target-negative-loop/scripts/postprocess_target_results.py`
- 本 skill 的自动评分：
  `.codex/skills/xrd-target-negative-loop/scripts/score_processed_results.py`

## 强制约束

1. 主 Agent 负责选目标结构、选负样本、决定是否继续；subagent 只负责执行训练推理，不负责决定题目本身。
2. 只在用户明确允许 delegation / subagent 的前提下启动 subagent；这个 skill 默认就是这种场景。
3. 所有训练和推理仍然必须走 `docker/docker-compose.yaml`，并始终带 `LOCAL_UID=$(id -u)` 和 `LOCAL_GID=$(id -g)`。
4. 每次训练尝试都必须生成新的 `run_name`，不要覆盖旧结果。
5. 目标公式 A 一旦确认，不要因为结果不好而擅自把目标换成别的式子。
6. 如果用户指定精确空间群、稳定性或 MP material ID，先核实是否真实存在；MP 不存在时要明确告诉用户，不要伪造“已找到”。
7. 如果需要从 MP 下载，必须先确认 `MP_API_KEY` 可用。
8. `processed_result.csv` 不是天然“准确率”。只有当真实数据存在可解析标签或用户提供标签时，才能把评分当成近似准确率。
9. 任务是否完成，必须按真实数据 `processed_result.csv` 及其对应 `score.json` 判断；不要按原始 `result.csv`、训练 loss、模拟谱图区分度或单个样例的主观观感判断。
10. 在开始新一轮训练前，先对当前 run 的 `result.csv` 做 `--confidence-threshold` 阈值搜索；如果只靠后处理阈值就能让真实数据 processed_result 达标，优先停止，不要为形式而重训。
11. 如果真实数据里没有可解析的目标正样本标签，就不能声称“目标真实准确率达标”；此时最多只能声称负样本侧的 proxy 或弱标签准确率达标，并明确局限。
12. 迭代过程中必须保存“当前最佳 run + 最佳阈值”。如果新一轮训练使真实数据 processed_result 变差，保留旧 run，回退到旧 run 继续探索，不要把更差的新 run 当成结果。
13. 如果 `docker compose exec` 看不到 `MP_API_KEY`，先修复执行环境注入，再训练；不要在未确认 MP 下载真实可用时假设脚本能成功。

## 主 Agent 的职责

主 Agent 必须本地完成这些动作：

1. 检查前置条件。
2. 确定目标结构 A。
3. 选出第一轮负样本集合。
4. 生成训练计划并启动 subagent。
5. 等 subagent 返回 run 目录、`result.csv`、参考相和模型路径。
6. 本地运行后处理与评分脚本。
7. 对当前 run 做阈值搜索，记录最佳 `confidence_threshold`。
8. 根据真实数据 processed 评分结果判断：
   - 满意：结束并汇报路径。
   - 不满意：分析混淆来源，重新选负样本，再开新 subagent。

## subagent 的职责

subagent 只负责执行，不负责改题。给它的任务要具体、封闭、可验证。

subagent prompt 至少要包含：
- 固定的目标标签和负样本标签
- 每个相的来源：MP material ID 或本地 CIF
- 必须使用的脚本
- 必须返回的路径：
  - `result.csv`
  - `Models`
  - `References`
  - `run_manifest.txt`

不要让 subagent 自己发明新的负样本策略，也不要让它静默替换用户指定结构。

## 目标结构 A 的确定规则

沿用 `xrd-formula-train-infer` 的判断逻辑：

- 用户给本地 CIF：直接用本地 CIF，不要换成 MP。
- 用户给 MP material ID：直接采用，不再排序候选。
- 用户给精确空间群或稳定性要求：先核实 MP 是否存在该版本。
- 用户只给化学式 A：在 MP 中选优先候选，优先稳定、精确化学计量、`energy_above_hull` 更低的结构。

如果目标结构版本在 MP 不存在：
- 明确告诉用户 MP 实际有哪些候选。
- 不要拿“最接近”的版本冒充指定版本。
- 如果用户坚持该版本，要求本地 CIF。

## 负样本选择规则

负样本可以是 1 个，也可以是多个。默认先从 1 到 3 个开始，不要第一次就把相数堆得太多。

第一轮优先顺序：

1. 与 A 拓扑或衍射峰型差异较大、且在 MP 中稳定的候选。
2. `mp_formula_tool.py candidates` 或已有候选脚本中能给出的高对比候选。
3. 用户点名要求加入的负样本。
4. 仓库已有本地 CIF 中与 A 同体系、但容易混淆的相。

重训时优先顺序：

1. 先看 `processed_result.csv` 与评分报告中哪些文件被错分。
2. 再回看原始 `result.csv` 中最常作为高置信混淆项的相。
3. 优先替换最可疑的一个负样本，不要整组全部推倒。
4. 如果混淆来自目标结构版本选错，保持化学式 A 不变，只替换 A 的结构版本。

## 评分闭环

后处理和评分按这个顺序执行：

1. 先用默认阈值或候选阈值把 `result.csv` 非交互地处理成 `processed_result.csv`：

```bash
python3 .codex/skills/xrd-target-negative-loop/scripts/postprocess_target_results.py \
  --input ".../result.csv" \
  --output ".../processed_result.csv" \
  --target-formula "A" \
  --confidence-threshold 50
```

2. 再对 `processed_result.csv` 做评分。`known-formula` 不只包含训练时的负样本；如果真实数据文件名还能唯一解析出其他公式，也应一起传入，避免把大量真实负样本排除出评分范围：

```bash
python3 .codex/skills/xrd-target-negative-loop/scripts/score_processed_results.py \
  --input ".../processed_result.csv" \
  --target-formula "A" \
  --known-formula "A" \
  --known-formula "Negative1" \
  --known-formula "Negative2" \
  --known-formula "OtherRealFormula1" \
  --known-formula "OtherRealFormula2" \
  --output-json ".../score.json"
```

3. 在开始新一轮训练前，优先对同一个 `result.csv` 做阈值搜索。推荐搜索范围 `40` 到 `100`，步长 `1` 到 `5`，并把结果另存为 `processed_result_thrXX.csv` / `score_thrXX.json`。示例：

```bash
for thr in 40 50 60 70 80 90 95 98; do
  python3 .codex/skills/xrd-target-negative-loop/scripts/postprocess_target_results.py \
    --input ".../result.csv" \
    --output ".../processed_result_thr${thr}.csv" \
    --target-formula "A" \
    --confidence-threshold "$thr"

  python3 .codex/skills/xrd-target-negative-loop/scripts/score_processed_results.py \
    --input ".../processed_result_thr${thr}.csv" \
    --target-formula "A" \
    --known-formula "A" \
    --known-formula "Negative1" \
    --known-formula "Negative2" \
    --output-json ".../score_thr${thr}.json"
done
```

评分解释：

- 如果文件名或目录名里能唯一解析出 `known-formula` 之一，则记为弱标签样本。
- 如果弱标签样本足够多，就计算目标 A 的 precision / recall / F1，并用它决定是否满意。
- 如果只有负样本侧标签、没有目标 A 的真实正样本标签，则可以计算真实数据上的负样本侧 proxy accuracy / false-positive 情况，但不能把它说成目标真实准确率。
- 如果弱标签覆盖率过低，就只能把结果视为 proxy，不要把它说成严格 accuracy。

默认满意判据：

- `evaluation_mode == weak_labels`
- `labeled_rows >= 10`
- `coverage >= 0.10`
- 若用户未单独指定，默认完成阈值按真实数据 processed_result 准确率至少 `95%` 执行
- 若用户明确要求了其他阈值，优先满足用户明确要求的真实数据 processed_result 指标，例如“准确率至少 95%”
- 若存在目标正样本与负样本标签：
  - `precision >= 0.85`
  - `recall >= 0.70`
  - `f1 >= 0.75`
- 若不存在目标正样本标签、只能做负样本侧 proxy：
  - 明确声明是 proxy
  - 记录 `false_positive`、`target_prediction_ratio`、`unidentified_ratio`
  - 仅当用户接受这种 proxy 口径时，才能把阈值后的结果作为阶段性达标依据

如果用户提供更明确的阈值，按用户阈值覆盖。

## 执行脚本的选择

- 只有两相，且都来自 MP：优先用 `run_pipeline.sh`
- 只要出现以下任一情况，就改用 `run_multiphase_pipeline.sh`
  - 三相及以上
  - 任一相来自本地 CIF
  - 需要手工指定 `References` 名称
  - 为规避 `tabulate_cifs` 问题需要 `--skip_filter`

## 主 Agent 的推荐工作流

### Step 1: 检查前置条件

- 确认当前目录是仓库根目录
- 检查 `docker/docker-compose.yaml`
- 检查 `libs/XRD-1.1/Novel-Space`
- 如果需要 MP，检查 `MP_API_KEY`
- 如果通过 `docker compose exec` 调脚本，确认容器内执行时也能读到 `MP_API_KEY`
- 检查真实数据是否存在可用于弱标签的命名线索
- 检查真实数据里是否存在目标 A 的可解析正样本；若不存在，提前告知用户后续只能得到 proxy，而非完整真实准确率

### Step 2: 固定目标结构 A

- 如果用户指定精确结构，先核实
- 如果未指定，选最合理的第一候选
- 记录本轮的 target label，例如 `CrSiTe3_148` 或 `AlN_216`

### Step 3: 确定第一轮负样本

- 先选 1 到 3 个负样本
- 保持标签可读、可追踪
- 如果混合 MP 与本地 CIF，提前准备 `--manual-reference`

### Step 4: 启动 subagent 训练

把训练任务完整交给 subagent。它执行结束后，主 Agent 读取它返回的 run 路径，不要自己再补跑第二遍相同训练。

### Step 5: 后处理和评分

- 运行 `postprocess_target_results.py`
- 运行 `score_processed_results.py`
- 检查 `score.json`
- 对同一轮 `result.csv` 做阈值搜索，选出真实数据上最好的 `confidence_threshold`
- 把最佳阈值版本单独保存为 `processed_result_thrXX.csv` / `score_thrXX.json`

### Step 6: 决定满意或重训

- 若真实数据 processed 评分在某个阈值上已经达到用户要求，结束；不要为了“必须重训一次”继续折腾
- 若未达到，但存在明确高频混淆项，替换或增加负样本后重训
- 若新 run 比旧 run 更差，保留旧 run 为 best-so-far，回退并从旧 run 继续分析
- 若没有足够弱标签，不要宣称“准确率已经达标”；要明确说明现在只有 proxy

## 输出要求

完成后必须告诉用户：

- 哪个 `run_name` 是当前 best-so-far
- 哪个 `confidence_threshold` 是当前最佳阈值
- 本轮 `run_name`
- `result.csv`
- `processed_result.csv`
- `score.json`
- 若做了阈值搜索，再给出最佳阈值对应的 `processed_result_thrXX.csv` 与 `score_thrXX.json`
- `Models`
- `References`
- `Spectra`
- `figure`

如果用户问后处理列含义：

- `Predicted phases`：后处理后的单标签结果
- `Confidence`：该单标签结果在原融合结果中的最高置信度
- `未识别`：没有任一目标主物质标签超过阈值

## 禁止事项

- 不要把 subagent 变成“自己选题自己裁判”
- 不要在没有标签依据的情况下把 proxy 说成准确率
- 不要把“全部判成未识别”包装成对目标 A 的真实识别成功；这最多只说明阈值把误报压低了
- 不要在同一个 `run_name` 上反复覆盖
- 不要跳过 `processed_result.csv` 直接拿原始多标签结果当最终验收
- 不要因为某轮效果差就偷偷更换目标化学式 A
- 不要因为新 run 更新了文件，就忘记比较它是否真的优于旧 run
