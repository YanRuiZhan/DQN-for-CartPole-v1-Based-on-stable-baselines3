#train.py
import gymnasium as gym
from config import LOG_DIR
from config import TOTAL_TIMESTEPS
from config import hyperparams
from config import MODEL_SAVE_PATH
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor 
from test import build_callbacks
#创建训练环境
def train_env_generate():
    train_env = gym.make("CartPole-v1")
    train_env = Monitor(train_env)
    return train_env

#创建评估环境
def eval_env_generate():
    eval_env = gym.make("CartPole-v1")
    eval_env = Monitor(eval_env)
    return eval_env

#训练函数
    #生成环境
def train():
    train_env = train_env_generate()
    eval_env = eval_env_generate()

    #初始化DQN模型
    model = DQN(
        policy = "MlpPolicy",
        env = train_env,
        max_grad_norm=10,
        tensorboard_log = LOG_DIR,
        **hyperparams
    )
    
    #构建回调
    callbacks = build_callbacks(eval_env)
    
    # 开始训练
    print(f"\n开始训练，最大步数：{TOTAL_TIMESTEPS}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name="DQN_CartPole",
        progress_bar=True,
    )

    # 保存最终模型
    model.save(MODEL_SAVE_PATH)
    print(f"\n模型已保存至：{MODEL_SAVE_PATH}.zip")

    train_env.close()
    eval_env.close()

    return model
