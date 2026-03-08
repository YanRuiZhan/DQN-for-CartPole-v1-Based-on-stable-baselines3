# main.py
from train import train
from test import evaluate   
import torch    
import os 
from train import MODEL_SAVE_PATH
from stable_baselines3 import DQN
from test import render_episode

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print("现在使用的设备是："+str(device))

if __name__ == "__main__":
    model_path = MODEL_SAVE_PATH + ".zip"  # 添加.zip扩展名，因为stable_baselines3保存模型时会自动添加
    
    if os.path.exists(model_path):
        print(f"检测到已存在的模型文件: {model_path}")
        print("正在加载预训练模型...")
        model = DQN.load(model_path, env=None)  # 加载已保存的模型
        print("模型加载完成！")
    else:
        print(f"未检测到模型文件: {model_path}")
        print("开始训练新模型...")
        # 训练模型
        model = train()

    # 评估模型
    evaluate(model)

    #渲染模型动画
    render_episode(DQN.load(model_path, env=None), episodes=1)
