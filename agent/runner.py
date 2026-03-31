from .tools import run_xrd_train

class AgentRunner:
    def __init__(self):
        print("AgentRunner initialized.")
        
    def run_training_loop(self, initial_lr=0.001, initial_bs=32):
        print(f"Starting training loop with lr={initial_lr}, bs={initial_bs}")
        # 这里可以加入解析训练日志并自动调整的逻辑
        # 比如：基于 Loss/Accuracy 自动调整下一轮的超参数
        run_xrd_train(initial_lr, initial_bs)
        print("Training step finished.")
