"""
Q 网络（第 5 周成果）
====================
用 PyTorch 搭建的神经网络，负责"算 Q 值"。

它做什么？
  输入：11 个数字（游戏局面，来自 env.py）
  输出：4 个数字（4 个动作的 Q 值）

结构：
  输入层(11) → 隐藏层(128) → 隐藏层(128) → 输出层(4)
  每层之间用 ReLU 激活函数（非线性，让网络能学复杂规律）
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """全连接神经网络，用于估算 Q 值。"""

    def __init__(self, input_size=11, hidden_size=128, output_size=4):
        super().__init__()
        # 三层全连接网络
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),   # 11 → 128
            nn.ReLU(),                            # 激活函数
            nn.Linear(hidden_size, hidden_size),  # 128 → 128
            nn.ReLU(),                            # 激活函数
            nn.Linear(hidden_size, output_size),  # 128 → 4
        )

    def forward(self, x):
        """前向传播：输入状态，输出 4 个动作的 Q 值。"""
        return self.net(x)


if __name__ == "__main__":
    # 测试：创建一个 Q 网络，随机输入一个状态，看输出
    model = QNetwork()
    print("网络结构：")
    print(model)

    # 模拟一个状态：11 个数字
    dummy_state = torch.randn(1, 11)   # shape (1, 11)，1 表示"一批里有一个样本"
    q_values = model(dummy_state)
    print("\n输入：11 个随机数字")
    print(dummy_state)
    print("\n输出：4 个 Q 值（现在还是乱猜的，因为没训练）")
    print(q_values)
    print("\n输出形状：", q_values.shape, "（1 个样本 × 4 个动作）")
    print("\nQ 网络搭建成功！下一步（第6周）就是训练它，让 Q 值变得准确。")
