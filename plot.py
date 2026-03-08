import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os


def load_latest_run(log_dir):
    """只加载最新一次run的标量数据"""
    run_dirs = sorted(
        [root for root, _, files in os.walk(log_dir) if any('tfevents' in f for f in files)],
        key=os.path.getmtime
    )
    if not run_dirs:
        print(f"未找到日志: {log_dir}")
        return {}

    print(f"加载: {run_dirs[-1]}")
    ea = EventAccumulator(run_dirs[-1])
    ea.Reload()
    return {
        tag: {'steps': [e.step for e in ea.Scalars(tag)],
              'values': [e.value for e in ea.Scalars(tag)]}
        for tag in ea.Tags()['scalars']
    }


def smooth(values, weight=0.95):
    """指数移动平均平滑（仅用于Loss曲线）"""
    last = values[0]
    result = []
    for v in values:
        last = last * weight + v * (1 - weight)
        result.append(last)
    return result


def plot_training_curves(log_dir="./logs/", save_path="./training_curves.png"):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    scalars = load_latest_run(log_dir)
    if not scalars:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('DQN训练过程分析', fontsize=16, fontweight='bold')

    def plot_tags(ax, tags, title, ylabel, use_smooth=False, colors=None):
        if not tags:
            ax.text(0.5, 0.5, f'未找到{title}数据', ha='center', va='center', transform=ax.transAxes)
        for tag in tags:
            s, v = scalars[tag]['steps'], scalars[tag]['values']
            color = colors(tag) if colors else None
            if use_smooth:
                # Loss：原始数据半透明 + 平滑曲线
                ax.plot(s, v, color=color, alpha=0.2, linewidth=0.8)
                ax.plot(s, smooth(v), color=color, linewidth=1.8, label=tag)
            else:
                # Reward：直接显示原始数据
                ax.plot(s, v, color=color, linewidth=1.8, label=tag)
        ax.set(title=title, xlabel='训练步数 (Steps)', ylabel=ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    reward_tags = [t for t in scalars if 'reward' in t or 'ep_rew' in t]
    loss_tags   = [t for t in scalars if 'loss' in t]
    reward_color = lambda t: '#EBC183' if 'eval' in t else '#4C8EDA'

    plot_tags(ax1, reward_tags, '奖励变化曲线', '奖励值(Rewards)', use_smooth=False, colors=reward_color)
    plot_tags(ax2, loss_tags,   '损失变化曲线', '损失值(Loss)',    use_smooth=True,  colors=lambda _: '#E05C5C')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    import sys
    plot_training_curves(
        log_dir=sys.argv[1] if len(sys.argv) > 1 else "./logs/",
        save_path=sys.argv[2] if len(sys.argv) > 2 else "./training_curves.png"
    )