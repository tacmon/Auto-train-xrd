import subprocess
import os

# 获取 XRD-1.1 的绝对路径
XRD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../libs/XRD-1.1"))

def run_xrd_train(learning_rate, batch_size):
    # 调用你之前写好的 shell 脚本
    script_path = os.path.join(XRD_PATH, "train.sh")
    cmd = f"bash {script_path} --lr {learning_rate} --bs {batch_size}"
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True)
