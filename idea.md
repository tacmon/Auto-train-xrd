# 我与GPT的对话记录

> 注：这是早期设计草稿，包含已经废弃的 `agent/`、`configs/`、`main.py` 设想。
> 当前仓库请以 `README.md`、`docker/`、`.codex/skills/` 和 `libs/XRD-1.1/` 为准。
既然你决定采用 **Git Submodule** 方案，这不仅能保持 `XRD-1.1` 的独立性，还能让 `Auto-train-xrd` 优雅地管理依赖。

作为一个 CS PhD，你可能需要频繁在不同服务器或环境中同步代码，这个方案的优势在于它能通过 `.gitmodules` 文件自动记录依赖关系。

以下是具体的执行步骤：

---

## 第一步：准备阶段（清理物理路径）

为了避免嵌套冲突，我们先确保 `XRD-1.1` 处于一个安全的位置（比如它的父目录）。

1.  **确认现状：** 假设你现在的目录结构是：
    ```text
    /home/user/work/
    └── XRD-1.1/  (这是一个已经 git init 的仓库)
    ```
2.  **创建新仓库文件夹：** 在同级目录下创建 `Auto-train-xrd`。
    ```bash
    mkdir Auto-train-xrd
    cd Auto-train-xrd
    git init  # 初始化 Agent 框架的仓库
    ```

---

## 第二步：建立物理关联（Submodule）

现在我们将 `XRD-1.1` 引入作为子模块。为了工程规范，建议将其放在 `libs/` 或 `modules/` 目录下。

1.  **添加子模块：**
    如果你已经将 `XRD-1.1` 推送到远程仓库（GitHub/GitLab），直接执行：
    ```bash
    git submodule add <XRD-1.1的远程仓库URL> libs/XRD-1.1
    ```
    *如果你目前只有本地仓库*，也可以直接引用本地路径，但**强烈建议**先给 `XRD-1.1` 搞一个远程仓库，这样协作更规范。

2.  **检查生成的文件：** 执行后，你会发现：
    * 多了 `libs/XRD-1.1/` 文件夹。
    * 根目录多了一个 `.gitmodules` 文件，内容类似于：
        ```ini
        [submodule "libs/XRD-1.1"]
            path = libs/XRD-1.1
            url = https://github.com/yourname/XRD-1.1.git
        ```



---

## 第三步：Agent 框架的文件布局

现在你可以开始构建 Agent 框架了。推荐的目录规范如下：

```text
Auto-train-xrd/
├── .git/
├── .gitmodules         # 记录子模块信息
├── libs/
│   └── XRD-1.1/       # 这里就是你的核心算法库（子仓库）
├── agent/              # 存放 LLM Agent 的逻辑代码
│   ├── prompt_templates/
│   ├── tools.py        # 在这里定义调用 libs/XRD-1.1/scripts/train.sh 的接口
│   └── runner.py
├── configs/            # 自动化训练的超参空间定义
└── main.py             # 入口文件
```

---

## 第四步：如何在代码中优雅调用？

在 `Auto-train-xrd` 的 Python 代码中，你不再需要软链接，直接使用**相对路径**即可。

```python
import subprocess
import os

# 获取 XRD-1.1 的绝对路径
XRD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "libs/XRD-1.1"))

def run_xrd_train(learning_rate, batch_size):
    # 调用你之前写好的 shell 脚本
    script_path = os.path.join(XRD_PATH, "train.sh")
    cmd = f"bash {script_path} --lr {learning_rate} --bs {batch_size}"
    subprocess.run(cmd, shell=True)
```

---

## 关键操作注意事项

1.  **提交更改：** 在 `Auto-train-xrd` 中提交时，Git 只会记录 `libs/XRD-1.1` 的 **Commit ID**。如果你在 `XRD-1.1` 里改了代码并提交了，记得回到 `Auto-train-xrd` 目录再 `git add libs/XRD-1.1` 一次，更新这个“指针”。
2.  **克隆项目：** 未来你在其他机器（比如实验室服务器）克隆这个项目时，记得使用：
    ```bash
    git clone --recursive <Auto-train-xrd的URL>
    ```
    否则 `libs/XRD-1.1` 文件夹会是空的。

你打算如何设计这个 Agent？是打算让它通过解析 `XRD-1.1` 的训练日志（Loss/Accuracy）来自动调整下一轮的超参数吗？


# 人类（我）的补充
1. XRD-1.1的远程仓库地址为：https://github.com/tacmon/XRD-1.1
2. 代码的运行环境，可以每次通过docker-compose（具体的请参考./docker目录）来启动。
