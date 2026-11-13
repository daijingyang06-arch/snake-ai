"""
两个 baseline AI（第 4 周成果）
===============================
实现两个"笨 AI"，作为后面 DQN 的对照基线（baseline）。

为什么要写它们？
  期末报告里，光说"DQN 能拿 50 分"没有说服力。
  但如果有对比：
    随机 AI  ≈ 1 分（地板）
    贪心 AI  ≈ 8 分（参照线）
    DQN     ≈ 50+ 分（我们的成果）
  就能直观证明 DQN 的价值。

两个 AI 的策略：
  1. RandomAgent：每步随机选方向，完全不看局面。
  2. GreedyAgent：每步朝食物方向走，但不考虑危险（容易撞死）。
"""

import numpy as np
from env import SnakeEnv


class RandomAgent:
    """随机 AI：纯碰运气，作为"最差水平"的参照。"""

    def choose_action(self, state):
        # 完全不管 state，随机返回 0~3 中的一个动作
        return np.random.randint(0, 4)


class GreedyAgent:
    """贪心 AI：朝食物走，但只看目标、不看危险。

    它利用 env.py 里 state 的第 5~8 位（食物方向 one-hot）来判断该往哪走。
    """

    def choose_action(self, state):
        # state[4:8] 是食物方向：[上, 下, 左, 右]（见 env.py 的 _build_state）
        food_up, food_down, food_left, food_right = state[4], state[5], state[6], state[7]

        # 优先选择"能靠近食物"的方向。
        # 注意：动作编号 0=右, 1=左, 2=下, 3=上（见 env.py 的 ACTION_MAP）
        if food_right == 1:
            return 0   # 食物在右 → 向右
        elif food_left == 1:
            return 1   # 食物在左 → 向左
        elif food_down == 1:
            return 2   # 食物在下 → 向下
        elif food_up == 1:
            return 3   # 食物在上 → 向上
        else:
            # 理论上不会到这（食物总在某个方向），兜底随便走
            return np.random.randint(0, 4)


def evaluate(agent, env, n_games=100):
    """
    让某个 AI 玩 n_games 局，返回每局的得分列表。

    这就是"训练循环"的雏形：反复 reset → step → 直到 done。
    """
    scores = []
    for _ in range(n_games):
        state = env.reset()      # 开一局
        score = 0
        done = False

        while not done:
            action = agent.choose_action(state)   # AI 选动作
            state, reward, done = env.step(action) # 环境执行
            score += reward                        # 累积奖励（= 这一局的"回报"）

        scores.append(score)
    return scores


if __name__ == "__main__":
    env = SnakeEnv()

    # 评测随机 AI
    random_agent = RandomAgent()
    random_scores = evaluate(random_agent, env, n_games=100)
    print(f"随机 AI：平均得分 = {np.mean(random_scores):.2f}")

    # 评测贪心 AI
    greedy_agent = GreedyAgent()
    greedy_scores = evaluate(greedy_agent, env, n_games=100)
    print(f"贪心 AI：平均得分 = {np.mean(greedy_scores):.2f}")

    print("\n这两个数字，就是后面 DQN 要超越的'基线'。")
    print("DQN 的目标：显著超过贪心 AI，证明它真正学会了'权衡'（既吃食物又避危险）。")
