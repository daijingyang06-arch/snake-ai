"""
DQN 智能体（第 6 周成果）
========================
把第 5 周的 Q 网络，加上 DQN 的三个关键组件，变成一个"会学习"的智能体。

包含四个部分：
  1. ReplayBuffer   —— 经验回放缓冲区（组件 1）
  2. DQNAgent       —— 智能体主体：选动作 + 存经验 + 训练
  3. 目标网络        —— 在 DQNAgent 内部（组件 2）
  4. ε-greedy       —— 在 select_action 里（组件 3）
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from model import QNetwork


class ReplayBuffer:
    """
    经验回放缓冲区（组件 1）。

    作用：把每一步的"经历"存起来，训练时随机抽一批出来学。
    为什么用 deque(maxlen=capacity)？
      满了之后自动挤掉最旧的，不用手动管理，也不占无限内存。
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存一条经历：(状态, 动作, 奖励, 下一个状态, 是否结束)"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        随机抽 batch_size 条经历，转成 PyTorch 张量。

        随机抽是关键——打乱时间顺序，消除样本之间的相关性。
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.array(next_states), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN 智能体：负责选动作和学习。"""

    def __init__(
        self,
        state_size=11,
        action_size=4,
        lr=1e-3,              # 学习率：每次调整权重的幅度
        gamma=0.9,            # 折扣因子：未来奖励打几折
        buffer_size=10000,    # 经验池容量
        batch_size=64,        # 每次训练抽多少条
        target_update=100,    # 每训练多少步，同步一次目标网络
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # ---- 两个网络（组件 2：目标网络）----
        # policy_net：主力网络，负责选动作，也是被训练的对象
        self.policy_net = QNetwork(state_size, 128, action_size)
        # target_net：目标网络，参数冻结，只用来算"学习目标"
        self.target_net = QNetwork(state_size, 128, action_size)
        # 一开始让两者完全一样
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # eval() 表示"只推理不训练"，关掉 dropout/batchnorm 之类
        self.target_net.eval()

        # 优化器：负责根据 loss 更新 policy_net 的权重
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # 经验池
        self.memory = ReplayBuffer(buffer_size)

        # ---- ε-greedy 的 ε（组件 3）----
        self.epsilon = 1.0    # 从 1.0 开始：百分百随机探索
        self.learn_step = 0   # 记录训练了多少步，用来决定何时同步目标网络

    def select_action(self, state):
        """
        选动作（组件 3：ε-greedy）。

        以 ε 的概率随机乱走（探索），否则选 Q 值最大的动作（利用）。
        """
        # 探索：随机
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # 利用：让网络算 Q 值，选最大的
        with torch.no_grad():    # 选动作不需要梯度，省内存
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        """把一步经历存进经验池。"""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        训练一步（DQN 的核心）。

        流程：
          1. 从经验池随机抽一批
          2. 用 policy_net 算出"我们当前认为的 Q 值"
          3. 用 target_net 算出"现实应该是什么"（贝尔曼方程右边）
          4. 计算误差 loss
          5. 反向传播，更新 policy_net
        """
        # 经验不够一批就先不学
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # ---- 2. 我们预测的 Q 值：只取"实际执行的那个动作"对应的 Q 值 ----
        # gather 的作用：从 4 个 Q 值里，按 action 挑出对应的那个
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ---- 3. 现实的目标值（贝尔曼方程右边）----
        with torch.no_grad():    # 目标值不需要梯度
            next_q_max = self.target_net(next_states).max(dim=1)[0]
            # 终止状态（撞死了）没有未来，所以乘 (1 - done)
            target = rewards + self.gamma * next_q_max * (1 - dones)

        # ---- 4. 误差 ----
        # smooth_l1_loss 比 MSE 更抗异常值，训练更稳（DQN 论文也用这个）
        loss = F.smooth_l1_loss(q_values, target)

        # ---- 5. 反向传播 + 更新权重 ----
        self.optimizer.zero_grad()      # 清空上一步的梯度
        loss.backward()                 # 反向传播，算出每个权重该怎么改
        # 梯度裁剪：防止梯度爆炸导致训练崩掉（实用技巧）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()           # 真正更新权重

        # ---- 6. 定期同步目标网络（组件 2 的关键）----
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self, decay=0.995, min_epsilon=0.05):
        """
        每局结束后降低一点 ε：探索逐步让位于利用。

        decay=0.995 意味着每局 ε 乘 0.995，大约 600 局后从 1.0 降到 0.05。
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay)

    def save(self, path):
        """保存训练好的模型。"""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """加载已有模型（用于接着训练或演示）。"""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == "__main__":
    # 快速自检：创建智能体，模拟几步，确认不报错
    agent = DQNAgent()
    fake_state = np.zeros(11, dtype=np.float32)

    print(f"初始 ε = {agent.epsilon}")
    print("随机选 5 次动作:", [agent.select_action(fake_state) for _ in range(5)])

    # 塞一些假经验，测试训练能不能跑
    for _ in range(100):
        agent.remember(np.random.rand(11), random.randrange(4), 0.0, np.random.rand(11), False)
    loss = agent.train_step()
    print(f"训练一步的 loss = {loss:.4f}")

    agent.decay_epsilon()
    print(f"衰减一次后 ε = {agent.epsilon:.4f}")
    print("\nDQN 智能体自检通过！下一步：用 train.py 真正训练。")
