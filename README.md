# DQN for CartPole-v1 (Stable-Baselines3)

基于 Stable-Baselines3 框架的 DQN 实现，解决 CartPole-v1 环境。包含训练、评估、可视化全流程。

## 项目结构

```
├── config.py          # 超参数配置（学习率、网络结构、探索策略等）
├── train.py           # 训练脚本
├── test.py            # 评估与渲染脚本
├── plot.py            # 训练曲线可视化
├── main.py            # 主入口（自动检测已有模型或重新训练）
├── dqn_cartpole.zip   # 预训练模型
└── training_curves.png # 训练曲线图
```

## 环境

```bash
pip install stable-baselines3 gymnasium tensorboard matplotlib torch
```

## 使用

```bash
# 训练 + 评估
python main.py

# 单独训练
python train.py

# 查看训练曲线
python plot.py
```

## 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 1e-4 | Adam 优化器 |
| 折扣因子 | 0.99 | 未来奖励衰减 |
| 批量大小 | 128 | 经验回放采样 |
| 经验池 | 100,000 | Replay Buffer 容量 |
| 网络结构 | [128, 128] | 双层 MLP |
| 总步数 | 100,000 | 训练轮次 |
