import sys
import os

# 确保能导入 agent 包
sys.path.append(os.path.dirname(__file__))

from agent.runner import AgentRunner

def main():
    print("Welcome to Auto-train-xrd")
    runner = AgentRunner()
    # 模拟启动，可以根据需要进行参数修改
    runner.run_training_loop(initial_lr=0.001, initial_bs=32)

if __name__ == "__main__":
    main()
