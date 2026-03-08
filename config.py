#config.py
#超参数设置

hyperparams = {
    "learning_rate": 1e-4,              #学习率
    "gamma": 0.99,                      #折扣因子
    "batch_size": 128,                  #批量大小
    "buffer_size": 100_000,              #经验池大小
    "learning_starts": 1000,            #学习开始时间
    "tau": 1,                         #软更新参数
    "train_freq": 1,                    #训练频率
    "gradient_steps": 1,                #梯度步数
    "target_update_interval": 500,     #目标网络更新间隔
    "exploration_fraction": 0.1,        #探索率衰减
    "exploration_initial_eps": 1.0,     #初始探索率
    "exploration_final_eps": 0.05,      #最终探索率
    "verbose": 1,
    "seed": None,
    "policy_kwargs": dict(net_arch=[128, 128]),     #MLP网络结构        
}

TOTAL_TIMESTEPS= 100_000              # 总步数
LOG_DIR = "./logs/"                     # TensorBoard 日志目录
MODEL_SAVE_PATH = "./dqn_cartpole"      # 模型保存路径