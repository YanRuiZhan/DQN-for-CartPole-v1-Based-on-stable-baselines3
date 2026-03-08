#test.py
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from config import LOG_DIR
import os
import time
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor


class StableRewardCallback(BaseCallback):
    """连续 required_successes 次评估都达到阈值才停止训练"""
    def __init__(self, reward_threshold=495, required_successes=2, verbose=1):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.required_successes = required_successes
        self.success_count = 0

    def _on_step(self):
        mean_reward = self.parent.last_mean_reward
        if mean_reward >= self.reward_threshold:
            self.success_count += 1
            if self.verbose:
                print(f"达标 {self.success_count}/{self.required_successes} 次（mean={mean_reward:.1f}）")
            if self.success_count >= self.required_successes:
                print(f"连续 {self.required_successes} 次评估奖励 >= {self.reward_threshold}，停止训练！")
                return False
        else:
            if self.success_count > 0:
                print(f"未达标（mean={mean_reward:.1f}），计数重置")
            self.success_count = 0
        return True


def build_callbacks(eval_env):
    os.makedirs(LOG_DIR, exist_ok=True)
    return EvalCallback(
        eval_env,
        callback_after_eval=StableRewardCallback(reward_threshold=495, required_successes=3),
        eval_freq=10000,
        n_eval_episodes=100,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        verbose=1,
    )


def evaluate(model, n_eval_episodes=50):
    print("\n" + "=" * 50)
    print("  最终评估")
    print("=" * 50)

    eval_env = Monitor(gym.make("CartPole-v1"))
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
    )
    eval_env.close()

    print(f"评估轮数：{n_eval_episodes}")
    print(f"平均奖励：{mean_reward:.1f} ± {std_reward:.1f}")
    print(f"CartPole-v1 满分（最大步数）：500")
    print("训练成功！" if mean_reward >= 475 else "模型尚未完全收敛，尝试增加训练步数。")

    return mean_reward, std_reward


def render_episode(model, episodes=1):
    """使用训练好的模型渲染CartPole游戏过程"""
    render_env = gym.make("CartPole-v1", render_mode="human")

    for episode in range(episodes):
        obs, _ = render_env.reset()
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            steps += 1
            time.sleep(0.01)

        print(f"第 {episode+1} 局结束 - 总步数/奖励: {steps}")

    render_env.close()
    print("\n渲染完成！")