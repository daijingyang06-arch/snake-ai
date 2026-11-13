"""
贪吃蛇 RL 环境（第 3 周成果）
============================
把第 2 周的贪吃蛇游戏，包装成「标准强化学习环境」。

为什么需要这个文件？
  强化学习算法（比如 DQN）不关心"游戏长什么样"，它只认两个函数：
    env.reset()        -> 返回初始状态
    env.step(action)   -> 返回 (新状态, 奖励, 是否结束)

  只要满足这个接口，任何 RL 算法都能直接训练。这就是"游戏"和"AI"之间的
  标准桥梁。

关键设计（第 3 周的重点）：
  1. 状态（state）= 11 个数字，描述 AI 关心的关键信息（方案 B）
  2. 动作（action）= 0/1/2/3 四个编号，而不是方向向量
  3. 奖励做了优化，帮助 AI 更快学会
"""

import numpy as np
from game import SnakeGame, GRID_W, GRID_H

# 动作编号 -> 方向向量
# 0=右, 1=左, 2=下, 3=上
ACTION_MAP = {
    0: (1, 0),
    1: (-1, 0),
    2: (0, 1),
    3: (0, -1),
}


class SnakeEnv:
    """贪吃蛇的标准 RL 环境。内部复用了第 2 周的 SnakeGame 规则。"""

    def __init__(self, w=GRID_W, h=GRID_H):
        self.game = SnakeGame(w, h)
        # 状态是 11 个数字，动作有 4 种
        self.state_size = 11
        self.action_size = 4

    def reset(self):
        """开一局新游戏，返回初始状态（11 个数字的 numpy 数组）。"""
        self.game.reset()
        return self._build_state()

    def step(self, action):
        """
        执行一个动作，返回 (新状态, 奖励, 是否结束)。

        参数 action：0/1/2/3 的整数
        """
        # 把动作编号翻译成方向向量，交给底层游戏执行
        direction = ACTION_MAP[action]
        _, reward, done = self.game.step(direction)

        # 下面是对奖励的"优化"，帮助 AI 学得更快（见下方讲解）
        if done:
            reward = -10          # 死亡：重罚
        elif reward == 10:
            reward = 10           # 吃到食物：重奖
        else:
            reward = 0            # 其他情况：不给负奖励（避免 AI 因害怕扣分而不敢动）

        return self._build_state(), reward, done

    def _build_state(self):
        """
        把游戏局面转成 11 个数字（这就是 AI 的"眼睛"）。

        这 11 个数字的含义：
          [0:4]  当前方向（上/下/左/右 四选一，用 one-hot 表示）
          [4:8]  食物相对于蛇头的方向（四选一）
          [8:11] 前方 / 左侧 / 右侧是否有危险（撞墙或撞自己，1=危险 0=安全）

        用 one-hot（独热）编码，因为神经网络更容易理解"是/否"这种明确的信号。
        """
        head_x, head_y = self.game.snake[-1]
        food_x, food_y = self.game.food
        dx, dy = self.game.direction

        # --- 1. 当前方向（one-hot，4 位）---
        # 顺序：上、下、左、右
        dir_up = 1 if dy == -1 else 0
        dir_down = 1 if dy == 1 else 0
        dir_left = 1 if dx == -1 else 0
        dir_right = 1 if dx == 1 else 0

        # --- 2. 食物在哪个方向（one-hot，4 位）---
        # 食物在蛇头的上方/下方/左方/右方（可能同时多个为真，比如正上方）
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0
        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0

        # --- 3. 三个方向是否有危险（1=危险，3 位）---
        # 定义"前方、左侧、右侧"是相对当前朝向而言的
        # 先算出"前、左、右"三个方向向量
        forward = (dx, dy)
        if forward == (1, 0):     # 向右走时，左=上，右=下
            left, right = (0, -1), (0, 1)
        elif forward == (-1, 0):  # 向左走时，左=下，右=上
            left, right = (0, 1), (0, -1)
        elif forward == (0, 1):   # 向下走时，左=右，右=左
            left, right = (1, 0), (-1, 0)
        else:                     # 向上走时，左=左，右=右
            left, right = (-1, 0), (1, 0)

        danger_straight = self._is_danger(head_x, head_y, forward)
        danger_left = self._is_danger(head_x, head_y, left)
        danger_right = self._is_danger(head_x, head_y, right)

        # 把所有信号拼成一个 11 位的数组
        state = np.array([
            dir_up, dir_down, dir_left, dir_right,
            food_up, food_down, food_left, food_right,
            danger_straight, danger_left, danger_right,
        ], dtype=np.float32)

        return state

    def _is_danger(self, head_x, head_y, direction):
        """判断：如果朝某个方向走一步，会不会撞墙或撞到自己。"""
        nx, ny = head_x + direction[0], head_y + direction[1]
        # 撞墙
        if not (0 <= nx < self.game.w and 0 <= ny < self.game.h):
            return 1
        # 撞到自己身体
        if (nx, ny) in self.game.snake[:-1]:
            return 1
        return 0


# ============ 测试：验证环境能正常运转 ============
if __name__ == "__main__":
    env = SnakeEnv()

    # 开一局
    state = env.reset()
    print("初始状态（11 个数字）:", state)
    print("状态形状:", state.shape)

    # 随机走 5 步，看看每一步的状态、奖励、是否结束
    for i in range(5):
        action = np.random.randint(0, 4)   # 随机选一个动作 0~3
        state, reward, done = env.step(action)
        print(f"第{i+1}步: 动作={action}, 奖励={reward}, 结束={done}")

    print("\n环境测试通过！这个 env 就是 AI 将来要玩的'游戏'。")
