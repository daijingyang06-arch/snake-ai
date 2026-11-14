"""
训练脚本（第 6 周成果）
======================
把环境（env）和智能体（agent）串起来，让 AI 真正开始学习。

训练循环的结构（和第 4 周 baseline 的 evaluate 几乎一样）：
  for 每一局:
      reset 开局
      while 没死:
          选动作 → 执行 → 存经验 → 训练一步
      降低 ε

和 baseline 唯一的区别：choose_action 从"笨策略"换成了 DQN 网络。
"""

import time

import numpy as np

from env import SnakeEnv
from agent import DQNAgent

# ============ 训练参数（可以调整）============
EPISODES = 500              # 训练多少局（想训练更久可以改大，比如 2000）
MAX_STEPS_NO_FOOD = 200     # 连续这么多步没吃到食物，就强制结束本局
                            # （防止 AI 学会"绕圈圈不死"这种偷懒策略）
SAVE_EVERY = 100            # 每训练多少局保存一次模型
PRINT_EVERY = 10            # 每多少局打印一次进度


def train(episodes=EPISODES):
    env = SnakeEnv()
    agent = DQNAgent()

    scores = []          # 记录每局的总奖励
    start_time = time.time()

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps_since_food = 0
        done = False

        while not done:
            # ① 选动作（ε-greedy）
            action = agent.select_action(state)

            # ② 执行动作
            next_state, reward, done = env.step(action)

            # ③ 存经验
            agent.remember(state, action, reward, next_state, done)

            # ④ 训练一步
            agent.train_step()

            # ⑤ 前进
            state = next_state
            total_reward += reward

            # ⑥ 防止绕圈：太久没吃到食物就结束本局
            if reward > 0:
                steps_since_food = 0
            else:
                steps_since_food += 1
            if steps_since_food > MAX_STEPS_NO_FOOD:
                done = True

        # 一局结束：降低探索率
        agent.decay_epsilon()
        scores.append(total_reward)

        # 打印进度
        if ep % PRINT_EVERY == 0:
            recent = np.mean(scores[-PRINT_EVERY:])
            best = np.max(scores)
            elapsed = time.time() - start_time
            print(
                f"第 {ep:4d} 局 | 近{PRINT_EVERY}局平均奖励 {recent:7.2f} | "
                f"历史最佳 {best:6.2f} | ε={agent.epsilon:.3f} | "
                f"用时 {elapsed:.0f}s"
            )

        # 保存模型
        if ep % SAVE_EVERY == 0:
            agent.save(f"model/dqn_snake_{ep}.pth")
            print(f"    → 已保存模型: model/dqn_snake_{ep}.pth")

    # 训练结束：保存最终模型和得分记录
    agent.save("model/dqn_snake_final.pth")
    np.save("results/scores.npy", np.array(scores))

    print("\n训练完成！")
    print(f"最终模型: model/dqn_snake_final.pth")
    print(f"得分记录: results/scores.npy（第 7 周用它画曲线）")
    print(f"最后 20 局平均奖励: {np.mean(scores[-20:]):.2f}")


if __name__ == "__main__":
    train()
