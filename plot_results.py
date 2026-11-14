"""
训练结果可视化（第 7 周成果）
============================
把 results/scores.npy 里那 500 个数字，画成一张"AI 成长曲线图"。

为什么需要这张图？
  报告里写"我的 AI 平均 87.8 分"没感觉，但一张从 -7 爬升到 88 的曲线，
  能让老师一眼看到"学习"这件事真实发生了。这是整个项目最有说服力的一张图。

图里画了什么？
  1. 灰细线  —— 每一局的原始得分（抖动很大，这是强化学习的正常现象）
  2. 蓝粗线  —— 每 20 局的平均分（平滑后就能看出趋势）
  3. 两条虚线 —— 第 4 周的两个 baseline（随机 AI / 贪心 AI），用来对比

生成的图片保存到 results/training_curve.png
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # 用不弹窗的后端，直接存图片（服务器/无显示器也能跑）
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 路径都基于本文件所在目录，这样在哪个目录下运行脚本都不会找不到文件 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCORES_PATH = os.path.join(BASE_DIR, "results", "scores.npy")
BASELINE_PATH = os.path.join(BASE_DIR, "results", "baseline_scores.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "training_curve.png")

# 平滑窗口：多少局取一次平均。20 局是经验值，太小看不出趋势，太大抹掉细节
SMOOTH_WINDOW = 20


def setup_chinese_font():
    """
    让 matplotlib 能显示中文。

    为什么需要这一步？
       matplotlib 默认字体不含中文字形，中文会显示成一个个方块（豆腐块）。
       所以要先在系统里找一个支持中文的字体，设置上去。
    返回：True 表示找到了中文字体，False 表示只能用英文标签。
    """
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False   # 修复负号显示成方块的问题
            return True
    return False


CHINESE_OK = setup_chinese_font()


def L(zh, en):
    """双语标签：有中文字体就用中文，没有就退回英文（保证图永远可读）。"""
    return zh if CHINESE_OK else en


def moving_average(x, w=SMOOTH_WINDOW):
    """
    滑动平均：把相邻 w 个数取平均。

    比如前 3 个数是 [1, 2, 3]，窗口 3 的平均就是 2。
    它的作用是"抹平毛刺"，让上升趋势变清楚。

    mode="valid" 表示不补零，只在数据够 w 个的地方计算，
    所以结果长度是 len(x) - w + 1。
    """
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def load_baselines():
    """
    读取第 4 周 baseline 的实测成绩（由 compare.py 生成）。

    如果文件不存在，就用第 4 周当时跑出来的默认值，
    这样即使没跑过 compare.py，图也能正常生成。
    """
    default = {"random": -9.3, "greedy": 49.1}
    if os.path.exists(BASELINE_PATH):
        try:
            with open(BASELINE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "random": data.get("random", {}).get("mean_reward", default["random"]),
                "greedy": data.get("greedy", {}).get("mean_reward", default["greedy"]),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return default


def main():
    if not os.path.exists(SCORES_PATH):
        print(f"找不到 {SCORES_PATH}")
        print("请先运行 python train.py 完成训练。")
        return

    scores = np.load(SCORES_PATH).astype(np.float32)
    n = len(scores)
    ma = moving_average(scores)
    # 滑动平均第 i 个点，对应的是原始数据第 (i + w/2) 局附近
    ma_x = np.arange(len(ma)) + SMOOTH_WINDOW // 2

    baselines = load_baselines()

    # ============ 画图 ============
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    # 1. 原始得分：半透明细线，铺在底层当"背景噪声"
    ax.plot(np.arange(1, n + 1), scores, color="#9ca3af", linewidth=0.8,
            alpha=0.55, label=L("每局得分（原始）", "Episode score (raw)"))

    # 2. 平滑曲线：粗蓝线，这是主角
    ax.plot(ma_x, ma, color="#2563eb", linewidth=2.4,
            label=L(f"每 {SMOOTH_WINDOW} 局平均", f"{SMOOTH_WINDOW}-episode average"))

    # 3. 两条 baseline 水平线，用来对比
    ax.axhline(baselines["greedy"], color="#f59e0b", linestyle="--", linewidth=1.6,
               label=L(f"贪心 AI 基线 ({baselines['greedy']:.1f})",
                       f"Greedy baseline ({baselines['greedy']:.1f})"))
    ax.axhline(baselines["random"], color="#6b7280", linestyle=":", linewidth=1.6,
               label=L(f"随机 AI 基线 ({baselines['random']:.1f})",
                       f"Random baseline ({baselines['random']:.1f})"))

    # 4. 在曲线末尾标注最终成绩，让人一眼看到结果
    #    加个白色半透明底框，避免数字压在曲线上看不清
    final_value = float(ma[-1])
    ax.annotate(f"{final_value:.1f}",
                xy=(ma_x[-1], final_value),
                xytext=(-10, 12), textcoords="offset points",
                fontsize=11, fontweight="bold", color="#1d4ed8", ha="right",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="none", alpha=0.85))

    # 5. 坐标轴、标题、网格、图例
    ax.set_xlabel(L("训练局数（episode）", "Episode"), fontsize=11)
    ax.set_ylabel(L("每局累积奖励", "Episode return"), fontsize=11)
    ax.set_title(L("DQN 训练过程：AI 从'乱撞'到'学会找食物'",
                   "DQN training: from random crashing to learned food-seeking"),
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.25, linestyle="-")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(1, n)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    # ============ 控制台打印关键数字（报告里可以直接引用）============
    print("=" * 52)
    print("训练结果统计")
    print("=" * 52)
    print(f"总训练局数      : {n}")
    print(f"起始阶段(前50局) : {scores[:50].mean():.1f}")
    print(f"最终阶段(后50局) : {scores[-50:].mean():.1f}")
    print(f"历史最高单局     : {scores.max():.1f}")
    print(f"历史最低单局     : {scores.min():.1f}")
    print(f"贪心 AI 基线     : {baselines['greedy']:.1f}")
    print(f"提升幅度         : {scores[-50:].mean() - scores[:50].mean():+.1f}")
    print("-" * 52)
    print(f"图片已保存：{OUTPUT_PATH}")
    print("（这张图可以直接放进 README 和期末报告）")


if __name__ == "__main__":
    main()
