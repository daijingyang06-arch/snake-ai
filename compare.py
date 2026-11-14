"""
三种 AI 同台对比（第 7 周成果）
===============================
在完全相同的测试条件下，评测三种 AI：
    1. 随机 AI     —— 地板（不会玩）
    2. 贪心 AI     —— 参照线（会朝食物走，但不懂避障）
    3. DQN         —— 我们的成果

为什么这一步很重要？
  只报"我的 AI 平均 87 分"是没说服力的，老师无法判断 87 分算好还是差。
  但摆出「随机 -9 / 贪心 49 / DQN 88」，价值立刻清晰了。
  这在报告里叫"消融对比 / baseline comparison"，是标准做法。

输出：
    results/comparison.png        对比柱状图
    results/baseline_scores.json  基线数据（plot_results.py 会读取它画虚线）
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline import GreedyAgent, RandomAgent
from env import SnakeEnv
from play import DQNPlayer

# 复用 plot_results.py 里的中文字体设置和双语标签函数
# （import 这个模块时会自动执行 setup_chinese_font()，把字体配好）
from plot_results import L

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "dqn_snake_final.pth")
CHART_PATH = os.path.join(BASE_DIR, "results", "comparison.png")
JSON_PATH = os.path.join(BASE_DIR, "results", "baseline_scores.json")

N_GAMES = 100       # 每种 AI 玩多少局（局数越多越准，但越慢）
MAX_STEPS = 5000    # 单局步数上限（安全阀，防止 AI 原地绕圈时脚本卡死）


def evaluate(player, n_games=N_GAMES, seed=42):
    """
    让某个 AI 玩 n_games 局，统计奖励和吃到的食物数。

    参数 seed 固定随机种子：这样每次运行对比实验，食物出现的位置序列都一样，
    三种 AI 面对的是同一批"考试题"，对比才公平。
    """
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)

    env = SnakeEnv()
    returns, foods = [], []

    for _ in range(n_games):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = player.choose_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

        returns.append(total_reward)
        foods.append(env.game.score)     # 这局吃了几个食物

    return {
        "mean_reward": float(np.mean(returns)),
        "std_reward": float(np.std(returns)),
        "mean_food": float(np.mean(foods)),
        "max_food": int(np.max(foods)),
        "min_food": int(np.min(foods)),
    }


def main():
    print("=" * 58)
    print(f"三种 AI 对比实验（每种各玩 {N_GAMES} 局，相同随机种子）")
    print("=" * 58)

    results = {}

    print("评测随机 AI ...", flush=True)
    results["random"] = evaluate(RandomAgent())

    print("评测贪心 AI ...", flush=True)
    results["greedy"] = evaluate(GreedyAgent())

    print("评测 DQN ...", flush=True)
    results["dqn"] = evaluate(DQNPlayer(MODEL_PATH))

    # ---- 打印对比表格 ----
    names = {"random": "随机 AI", "greedy": "贪心 AI", "dqn": "DQN（本项）"}
    print()
    print(f"{'AI':<12}{'平均奖励':>10}{'标准差':>9}{'平均食物':>10}{'最好一局':>10}")
    print("-" * 58)
    for key in ("random", "greedy", "dqn"):
        r = results[key]
        print(f"{names[key]:<12}{r['mean_reward']:>9.1f}{r['std_reward']:>9.1f}"
              f"{r['mean_food']:>10.2f}{r['max_food']:>10d}")
    print("-" * 58)

    dqn = results["dqn"]["mean_reward"]
    greedy = results["greedy"]["mean_reward"]
    if greedy > 0:
        print(f"DQN 相比贪心 AI 提升：{(dqn - greedy) / abs(greedy) * 100:+.1f}%")

    # ---- 保存基线数据，供 plot_results.py 画虚线用 ----
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"n_games": N_GAMES, **results}, f, ensure_ascii=False, indent=2)

    # ---- 画对比柱状图 ----
    labels = [L("随机 AI", "Random"), L("贪心 AI", "Greedy"), L("DQN", "DQN")]
    colors = ["#9ca3af", "#f59e0b", "#2563eb"]
    means = [results[k]["mean_reward"] for k in ("random", "greedy", "dqn")]
    stds = [results[k]["std_reward"] for k in ("random", "greedy", "dqn")]
    foods = [results[k]["mean_food"] for k in ("random", "greedy", "dqn")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)

    # 左图：平均奖励（带误差棒，误差棒 = 标准差，表示"稳定程度"）
    ax = axes[0]
    bars = ax.bar(labels, means, color=colors, width=0.55, yerr=stds,
                  capsize=6, error_kw={"ecolor": "#4b5563", "lw": 1.2})
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (6 if v >= 0 else -14), f"{v:.1f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0, color="#374151", linewidth=0.8)
    ax.set_ylabel(L("平均累积奖励", "Mean episode return"), fontsize=11)
    ax.set_title(L("平均奖励对比（越高越好）", "Mean reward comparison"),
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    # 右图：平均吃到食物个数（更直观，非专业的人也能看懂）
    ax = axes[1]
    bars = ax.bar(labels, foods, color=colors, width=0.55)
    for bar, v in zip(bars, foods):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel(L("平均吃到食物数", "Mean foods eaten"), fontsize=11)
    ax.set_title(L(f"平均吃到食物数（{N_GAMES} 局）", f"Foods eaten ({N_GAMES} episodes)"),
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(CHART_PATH), exist_ok=True)
    fig.savefig(CHART_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"\n对比图已保存：{CHART_PATH}")
    print(f"基线数据已保存：{JSON_PATH}")
    print("（重新跑 plot_results.py，图里的虚线会自动变成这次的实测值）")


if __name__ == "__main__":
    main()
