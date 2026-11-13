"""
贪吃蛇游戏（手动游玩版）
=======================
这是「贪吃蛇 AI」项目的第 2 周成果：一个可以手动玩的贪吃蛇。

为什么这一周要做这个？
  做 AI 之前，必须先有一个"游戏"让 AI 去玩。
  而要让 AI 能玩，游戏本身得写清楚：蛇在哪、食物在哪、怎么算赢、怎么算输。

代码结构（从下往上看更容易理解）：
  1. SnakeGame 类 —— 游戏的"大脑"，管状态和规则
  2. 各种画图函数 —— 负责把状态显示到屏幕上
  3. main() —— 游戏循环，把上面所有东西串起来跑

控制方式：
  方向键 ↑ ↓ ← → 控制蛇移动，按 ESC 或关闭窗口退出。
"""

import random
import pygame

# ============ 游戏的基本参数（改成这些数字就能调整游戏）============
BLOCK_SIZE = 20        # 每个格子的像素大小
GRID_W = 20            # 横向有多少个格子
GRID_H = 20            # 纵向有多少个格子
SPEED = 10             # 游戏速度：每秒蛇移动几步（数字越大越快）

# 颜色定义（RGB）
COLOR_BG = (30, 30, 30)          # 背景：深灰色
COLOR_SNAKE = (0, 200, 0)        # 蛇身：绿色
COLOR_SNAKE_HEAD = (0, 255, 0)   # 蛇头：亮绿色
COLOR_FOOD = (255, 0, 0)         # 食物：红色
COLOR_TEXT = (255, 255, 255)     # 文字：白色


class SnakeGame:
    """贪吃蛇游戏的"大脑"：只负责状态和规则，不负责画图。"""

    def __init__(self, w=GRID_W, h=GRID_H):
        self.w = w          # 网格宽（格数）
        self.h = h          # 网格高（格数）
        self.reset()

    def reset(self):
        """重置游戏，回到初始状态。返回初始状态（后面 AI 会用到）。"""
        # 蛇初始放在中间，长度为 3，头在右边
        cx, cy = self.w // 2, self.h // 2
        self.snake = [(cx - 2, cy), (cx - 1, cy), (cx, cy)]
        self.direction = (1, 0)   # 初始方向：向右
        self.score = 0
        self.game_over = False
        self.food = self._place_food()
        return self.get_state()

    def _place_food(self):
        """随机生成一个食物，保证不落在蛇身上。"""
        while True:
            pos = (random.randrange(self.w), random.randrange(self.h))
            if pos not in self.snake:
                return pos

    def step(self, action):
        """
        让蛇朝某个方向走一步。
        这是整个项目最关键的函数：后面做 AI 时，AI 就是反复调用这个函数。

        参数 action：方向，取值是 (1,0) 右 / (-1,0) 左 / (0,1) 下 / (0,-1) 上
        返回：新状态、奖励、是否结束
        """
        # 1. 更新方向（禁止 180 度掉头：不能直接反向）
        #    比如当前向右 (1,0)，就不能立刻改成向左 (-1,0)
        if (action[0] + self.direction[0], action[1] + self.direction[1]) != (0, 0):
            self.direction = action

        # 2. 计算新蛇头的位置 = 旧蛇头 + 方向
        head_x, head_y = self.snake[-1]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # 3. 判断是否撞墙（超出边界）
        if not (0 <= new_head[0] < self.w and 0 <= new_head[1] < self.h):
            self.game_over = True
            return self.get_state(), -10, True   # 撞墙：扣 10 分，游戏结束

        # 4. 判断是否撞到自己（新蛇头碰到自己的身体）
        #    注意：允许撞到"尾巴"，因为尾巴这一步会移走
        if new_head in self.snake[:-1]:
            self.game_over = True
            return self.get_state(), -10, True   # 撞自己：扣 10 分，游戏结束

        # 5. 移动：新蛇头加入，然后根据是否吃到食物决定是否删尾巴
        self.snake.append(new_head)
        if new_head == self.food:
            # 吃到食物：不删尾巴（变长），加分，重新生成食物
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            # 没吃到：删掉尾巴（长度不变），扣 1 分（惩罚绕圈、鼓励尽快吃）
            self.snake.pop(0)
            reward = -1

        return self.get_state(), reward, False

    def get_state(self):
        """返回当前游戏状态（后面 AI 要用这个来"看"游戏局面）。"""
        return {
            "snake": self.snake,
            "food": self.food,
            "direction": self.direction,
            "score": self.score,
            "game_over": self.game_over,
        }


# ============ 下面的代码都是"画图"，把状态显示到屏幕上 ============

def draw_game(screen, game):
    """把游戏状态画到屏幕上。"""
    screen.fill(COLOR_BG)   # 先清空成背景色

    # 画蛇身（先画身体，再画头，让头盖在身体上更明显）
    for i, (x, y) in enumerate(game.snake):
        rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        # 头部用亮绿色，身体用普通绿色
        color = COLOR_SNAKE_HEAD if i == len(game.snake) - 1 else COLOR_SNAKE
        pygame.draw.rect(screen, color, rect)

    # 画食物
    fx, fy = game.food
    food_rect = pygame.Rect(fx * BLOCK_SIZE, fy * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(screen, COLOR_FOOD, food_rect)

    # 显示分数
    font = pygame.font.Font(None, 28)   # None 表示用 pygame 默认字体，避免系统字体兼容问题
    text = font.render(f"Score: {game.score}", True, COLOR_TEXT)
    screen.blit(text, (10, 10))


def main():
    """游戏主入口：初始化 pygame，跑游戏循环。"""
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * BLOCK_SIZE, GRID_H * BLOCK_SIZE))
    pygame.display.set_caption("贪吃蛇 - 手动版")
    clock = pygame.time.Clock()   # 用来控制游戏速度

    game = SnakeGame()
    running = True

    while running:
        # ---- ① 处理输入 ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:          # 点关闭按钮
                running = False
            if event.type == pygame.KEYDOWN:        # 按下键盘
                if event.key == pygame.K_ESCAPE:    # 按 ESC 退出
                    running = False
                elif event.key == pygame.K_UP:
                    game.step((0, -1))              # 上
                elif event.key == pygame.K_DOWN:
                    game.step((0, 1))               # 下
                elif event.key == pygame.K_LEFT:
                    game.step((-1, 0))              # 左
                elif event.key == pygame.K_RIGHT:
                    game.step((1, 0))               # 右

        # 如果死了，按 R 重新开始（方便你反复玩）
        if game.game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                game = SnakeGame()

        # ---- ② 更新状态 ----
        # （这里蛇的移动是在"按键"时触发的，见上面的 KEYDOWN 处理）
        # 说明：真正的 AI 版会改成"每帧自动走一步"，这里为了手动玩，
        #       改成"按一下走一步"，方便你控制。

        # ---- ③ 画到屏幕 ----
        draw_game(screen, game)
        if game.game_over:
            font = pygame.font.Font(None, 40)
            text = font.render("Game Over! Press R to restart", True, COLOR_TEXT)
            screen.blit(text, (30, GRID_H * BLOCK_SIZE // 2))

        pygame.display.flip()          # 把画好的内容刷新到窗口
        clock.tick(SPEED)              # 控制帧率

    pygame.quit()


# 这一行保证：只有"直接运行 game.py"时才启动游戏；
# 如果别的文件 import 它（比如后面 train.py），不会自动弹窗。
if __name__ == "__main__":
    main()
