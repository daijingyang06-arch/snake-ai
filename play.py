"""
AI 现场演示（第 7 周成果）
==========================
加载训练好的模型，让你亲眼看着 AI 玩贪吃蛇。

为什么需要这个文件？
  期末答辩/演示时，光有曲线图和数字不够，老师更想看"东西真的动起来"。
  这个脚本就是你的现场演示工具。

用法（在 snake-ai 目录下）：
    python play.py              # 看训练好的 DQN 玩（默认）
    python play.py dqn 20       # 看 DQN 玩，速度 20 帧/秒
    python play.py greedy       # 看贪心 AI 玩（第 4 周的 baseline）
    python play.py random       # 看随机 AI 玩（最笨的，用来对比）

操作：
    空格  暂停/继续
    R     重新开一局
    + / - 加速 / 减速
    ESC   退出
"""

import os
import sys

import pygame

from baseline import GreedyAgent, RandomAgent
from env import SnakeEnv
from game import BLOCK_SIZE, COLOR_BG, COLOR_TEXT, GRID_H, GRID_W, draw_game

# ---- 路径统一基于本文件所在目录 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "dqn_snake_final.pth")

HUD_H = 48          # 底部信息栏高度（像素）
DEFAULT_FPS = 12    # 默认速度：每秒走 12 步
MIN_FPS, MAX_FPS = 2, 120


class DQNPlayer:
    """
    DQN 玩家的包装类。

    为什么需要包一层？
      agent.DQNAgent 的方法叫 select_action，而 baseline 的两个 AI 叫
      choose_action。名字不一样，play.py 就得多写 if 判断。
      包一层，把接口统一成 choose_action，主循环就干净了。
    """

    def __init__(self, model_path):
        # 延迟导入：只有真的要看 DQN 时才 import torch，
        # 看随机 AI 时不需要加载 torch，启动更快
        from agent import DQNAgent

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"找不到模型文件：{model_path}\n请先运行 python train.py 训练模型。"
            )

        self.agent = DQNAgent()
        self.agent.load(model_path)
        # ε 设为 0：完全不随机，永远选 Q 值最大的动作（这才是"考试"状态）
        self.agent.epsilon = 0.0

    def choose_action(self, state):
        return self.agent.select_action(state)


def build_player(mode):
    """根据命令行参数，创建对应的 AI。"""
    if mode == "dqn":
        print("正在加载训练好的 DQN 模型...")
        return DQNPlayer(MODEL_PATH), "DQN"
    elif mode == "greedy":
        return GreedyAgent(), "Greedy"
    elif mode == "random":
        return RandomAgent(), "Random"
    else:
        print(f"未知模式: {mode}")
        print("可选：dqn / greedy / random")
        sys.exit(1)


def draw_hud(screen, mode_name, foods, episode_return, fps, paused, episode):
    """
    画底部信息栏（用英文！）。

    注意：pygame 的默认字体 Font(None) 不支持中文，显示中文会变成方块。
    所以第 2 周我们就把界面文字全改成英文了。这里保持一致。
    """
    font = pygame.font.Font(None, 22)
    small = pygame.font.Font(None, 19)

    y0 = GRID_H * BLOCK_SIZE
    pygame.draw.rect(screen, (18, 18, 18), (0, y0, GRID_W * BLOCK_SIZE, HUD_H))

    line1 = f"Mode: {mode_name}   Ep: {episode}   Foods: {foods}   Return: {episode_return}"
    line2 = f"SPACE=pause  R=restart  +/-=speed  ESC=quit   speed={fps}fps"

    color1 = (255, 210, 80) if paused else COLOR_TEXT
    if paused:
        line1 += "   [PAUSED]"

    screen.blit(font.render(line1, True, color1), (8, y0 + 5))
    screen.blit(small.render(line2, True, (150, 150, 150)), (8, y0 + 26))


def main():
    # ---- 解析命令行参数 ----
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "dqn"
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FPS

    player, mode_name = build_player(mode)

    pygame.init()
    width, height = GRID_W * BLOCK_SIZE, GRID_H * BLOCK_SIZE
    screen = pygame.display.set_mode((width, height + HUD_H))
    pygame.display.set_caption(f"Snake AI Demo - {mode_name}")
    clock = pygame.time.Clock()

    # 游戏画面单独画在一个 surface 上，再贴到主窗口顶部，
    # 这样底部就能留出信息栏的位置，不会盖住棋盘
    game_surface = pygame.Surface((width, height))

    env = SnakeEnv()
    state = env.reset()

    episode = 1
    episode_return = 0      # 本局累积奖励
    paused = False
    running = True
    all_scores = []         # 记录每局的食物数，退出时打印

    while running:
        # ---- ① 处理输入 ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    state = env.reset()
                    episode_return = 0
                    paused = False
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps = min(MAX_FPS, fps * 2)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps = max(MIN_FPS, fps // 2)

        # ---- ② 更新状态：没暂停、没死，就让 AI 走一步 ----
        if not paused and not env.game.game_over:
            action = player.choose_action(state)
            state, reward, done = env.step(action)
            episode_return += reward

            if done:
                all_scores.append(env.game.score)

        # ---- ③ 画画面 ----
        draw_game(game_surface, env.game)
        screen.blit(game_surface, (0, 0))

        # 死了：显示提示（不自动重开，方便你截图/讲解）
        if env.game.game_over:
            font = pygame.font.Font(None, 40)
            text = font.render("Game Over - press R", True, (255, 80, 80))
            rect = text.get_rect(center=(width // 2, height // 2))
            pygame.draw.rect(screen, COLOR_BG,
                             rect.inflate(24, 16))
            screen.blit(text, rect)

        draw_hud(screen, mode_name, env.game.score, episode_return,
                 fps, paused, episode)

        pygame.display.flip()
        clock.tick(fps)

        # 死了且已经画完这一帧，就自动开下一局（方便长时间连续观察）
        if env.game.game_over and not paused:
            pygame.time.wait(600)          # 停 0.6 秒，让你看清是怎么死的
            episode += 1
            state = env.reset()
            episode_return = 0

    pygame.quit()

    # ---- 退出时打印本次观察的统计 ----
    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        print(f"\n本次演示：{mode_name} 玩了 {len(all_scores)} 局")
        print(f"平均吃到食物：{avg:.2f} 个")
        print(f"最高一局    ：{max(all_scores)} 个")
        print(f"最低一局    ：{min(all_scores)} 个")


if __name__ == "__main__":
    main()
