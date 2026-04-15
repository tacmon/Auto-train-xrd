---
name: xrd-formula-train-infer
description: 当用户希望基于一个或多个化学式、标签或 CIF 文件，在当前仓库里自动选择/下载 Materials Project 结构、混合本地 CIF、训练 XRD-1.1 的 XRD/PDF 模型，并对 ./data 下真实谱图做推理时触发。常见说法：用公式A/B训练、三相训练、指定空间群版本、MP 加本地 CIF 混合训练、对 data 真实数据跑预测、结果不满意就重训。
---

# XRD Formula Train Infer

这个 skill 面向当前仓库 `Auto-train-xrd`，只处理 `libs/XRD-1.1/Novel-Space` 这套工作流。

## 输入

用户至少提供以下之一：
- 一个或多个化学式/标签
- 一个或多个明确的 MP material ID
- 一个或多个本地 CIF 路径

用户可选提供：
- 额外的对比相或更多相
- 精确空间群、稳定性或“就是这个结构”的约束
- `./data` 下的某个子目录或文件路径

默认行为：
- 如果用户没指定真实数据路径，递归读取仓库 `./data` 下全部 `*.txt`、`*.xy`、`*.gk`
- 如果用户只给一个正样本公式且没指定对比相，先生成 3 个对比度较大的候选再请用户选
- 如果用户给的是本地 CIF，就直接用本地 CIF，不要强行替换成 MP 结构

## 强制约束

1. 先使用 `docker/docker-compose.yaml` 启动容器，不要假设宿主机 Python 环境可直接运行。
2. 运行 `docker compose up/exec` 时始终带上 `LOCAL_UID=$(id -u)` 和 `LOCAL_GID=$(id -g)`，否则容易生成 root/nobody 拥有的输出，后续训练会失败。
3. 所有训练和推理都在容器里执行；所有数据和产物都保存在宿主机挂载目录。
4. 每次训练尝试都必须使用新的 `run_name`。不要覆盖旧结果，这样“备份”天然成立。
5. 训练前必须清理 `Novel-Space/References`、`Novel-Space/Models`、`Novel-Space/Model.pth`、`Novel-Space/PDF_Model.pth` 等旧状态。
6. `Novel-Space/Spectra` 和 `Novel-Space/All_CIFs` 要用相对软链接指向 `soft_link/...`，不要写宿主机绝对路径；容器里会按 `/workspace/project/...` 解析，绝对宿主机路径会失效。
7. 如果用户指定“某公式的某空间群/某不稳定版本”，必须先核实 MP 里是否真的存在；不存在时要明确说明，并让用户选择最近的 MP 候选或直接提供本地 CIF。
8. 只有在需要从 MP 下载时才强制要求 `MP_API_KEY`；纯本地 CIF 流程可以继续。
9. 如果用户说“不满意”，默认保持用户已确认的标签集合不变，优先换候选结构、补本地 CIF，或调整多相组合，不要擅自改题。

## 先用哪些脚本

优先使用本 skill 自带脚本，不要重写临时命令流：

- 候选查询和 CIF 下载：
  `python3 .codex/skills/xrd-formula-train-infer/scripts/mp_formula_tool.py`
- 双相、全 MP 的一次完整训练+推理：
  `bash .codex/skills/xrd-formula-train-infer/scripts/run_pipeline.sh`
- 三相及以上，或混合 MP + 本地 CIF 的训练+推理：
  `bash .codex/skills/xrd-formula-train-infer/scripts/run_multiphase_pipeline.sh`

## 工作流

### Step 1: 检查前置条件

- 确认仓库路径是当前项目根目录
- 检查 `docker/docker-compose.yaml` 存在
- 检查 `libs/XRD-1.1/Novel-Space` 存在
- 如果需要 MP 下载，检查 `MP_API_KEY` 是否可用

### Step 2: 先判定是“选结构”还是“用指定结构”

按这个顺序判断：

- 用户直接给了本地 CIF：直接采用，不再去 MP 替换
- 用户直接给了 MP material ID：直接采用，不再做候选排序
- 用户给的是化学式，但附带精确空间群/稳定性要求：先查 MP 是否存在该版本
- 用户只给了化学式：再进入候选排序

如果 MP 里不存在用户指定的精确版本：
- 明确告诉用户 MP 中实际找到的 `material_id`、`space group` 和稳定性
- 不要把“不存在的 MP 版本”当成已确认事实
- 如果用户坚持该结构，就要求本地 CIF

### Step 3: 解析 MP 候选

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

### Step 4: 选择执行脚本

按这个规则：

- 只有两个相，且都来自 MP：用 `run_pipeline.sh`
- 只要出现以下任一情况，就改用 `run_multiphase_pipeline.sh`
  - 三相及以上
  - 任一相来自本地 CIF
  - 需要手工指定 `References` 名称
  - 预计 `tabulate_cifs` 可能不稳定，需要 `--skip_filter`

### Step 5: 运行流水线

双相全 MP 示例：

```bash
bash .codex/skills/xrd-formula-train-infer/scripts/run_pipeline.sh \
  --formula-a "A" \
  --formula-b "B" \
  --material-id-a "mp-xxxx" \
  --material-id-b "mp-yyyy" \
  [--spectra-source "./data/某个子目录或文件"]
```

多相混合来源示例：

```bash
bash .codex/skills/xrd-formula-train-infer/scripts/run_multiphase_pipeline.sh \
  --phase-label "CrSiTe3_148" \
  --phase-label "AlN_216" \
  --phase-label "BiSiTe3_148" \
  --mp-material-id "mp-3779" \
  --mp-material-id "mp-1700" \
  --local-cif "./local_cifs/Bi2Si2Te6.cif" \
  --manual-reference "CrSiTe3_148=CrSiTe3__mp-3779.cif" \
  --manual-reference "AlN_216=AlN__mp-1700.cif" \
  --manual-reference "BiSiTe3_148=Bi2Si2Te6.cif"
```

这些脚本会：
- 启动或重建容器
- 创建新的命名运行目录
- 把真实谱图复制到本次运行的 `Spectra`
- 下载 MP CIF 并复制本地 CIF 到本次运行的 `All_CIFs`
- 训练 XRD 模型
- 训练 PDF 模型
- 对真实谱图推理
- 保存 `result.csv`、原始谱图预览图、模型、参考相和运行清单

### Step 6: `tabulate_cifs` 失败时的回退

已知混合 MP + 本地 CIF 时，可能出现：

```text
TypeError: check_array() got an unexpected keyword argument 'force_all_finite'
```

这是 `autoXRD/tabulate_cifs` 与当前 `pyts/sklearn` 兼容性问题。

回退方式：
- 直接在本轮目录准备好 `All_CIFs/*.cif`
- 使用 `--manual-reference RefName=SourceFilename` 为每个相手工命名参考相
- 让 `run_multiphase_pipeline.sh` 自动创建 `References/*.cif`
- 训练 XRD 时自动追加 `--skip_filter`

### Step 7: 结果汇报

完成后必须把以下路径告诉用户：

- 本轮运行目录名
- 预测结果：`libs/XRD-1.1/Novel-Space/soft_link/All_CIFs/<run_name>/results/result.csv`
- 模型目录：`libs/XRD-1.1/Novel-Space/soft_link/All_CIFs/<run_name>/Models`
- 参考相目录：`libs/XRD-1.1/Novel-Space/soft_link/All_CIFs/<run_name>/References`
- 本轮真实谱图副本：`libs/XRD-1.1/Novel-Space/soft_link/Spectra/<run_name>`
- 真实谱图预览：`libs/XRD-1.1/Novel-Space/soft_link/figure/<run_name>`

如果用户问 `result.csv` 的三列含义，直接说明：

- `Predicted phases`：`run_CNN.py --inc_pdf` 的融合结果
- `XRD predicted phases`：XRD-only 结果
- `PDF predicted phases`：PDF-only 结果

所以 `python run_CNN.py --inc_pdf` 产出的主预测列是 `Predicted phases`，不是 `XRD predicted phases`。

然后明确要求用户只回复：
- `满意`
- `不满意`

### Step 8: 用户回复“不满意”时

不要覆盖当前结果。当前 run 已经是备份。

按这个顺序处理：
- 保持用户已经确认的目标标签集合不变
- 优先尝试更换同一标签的另一个 MP 候选
- 如果用户要的是 MP 中不存在的结构版本，转为要求本地 CIF
- 多相任务里优先替换最可疑的那个相，而不是整组全部推倒重来
- 生成新的 `run_name`
- 重新执行 Step 2 到 Step 7

## 交流规则

- 当需要用户选择候选结构时，只问一个简短问题
- 不要把整个 MP 检索原始输出都贴给用户，只总结前三个候选
- 如果真实数据来自 `./data` 的多个不同实验体系，默认全量递归读取；只有当用户明确要限制范围时才缩小

## 禁止事项

- 不要调用 `train.sh` 或 `setup_links.sh` 的交互模式
- 不要在同一个 `run_name` 下反复清空重来
- 不要在训练前保留旧的 `References`、`Models`、`Model.pth` 或 `PDF_Model.pth`
- 不要在没有 `MP_API_KEY` 的情况下继续尝试 Materials Project 下载
- 不要把 MP 中不存在的空间群/不稳定版本说成“已经找到”
