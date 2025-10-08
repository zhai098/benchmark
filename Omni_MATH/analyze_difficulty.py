#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from collections import Counter
from typing import Dict, Iterable, List, Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# =============================
# 可调参数
# =============================
INPUT_PATH = "Omni_MATH_Long_Segmented.jsonl"

OUT_DIR = "Omni_MATH"
OUT_PNG = os.path.join(OUT_DIR, "segment_num_distribution.png")
OUT_TXT = os.path.join(OUT_DIR, "segment_num_distribution.txt")
OUT_CSV = os.path.join(OUT_DIR, "segment_num_distribution.csv")

# 排序模式: "numeric" 按难度从小到大；"count_desc" 按数量降序
SORT_MODE = "numeric"

# =============================
# 工具函数
# =============================

def analyze_difficulty_distribution(file_path: str) -> Dict[int, int]:
    """统计难度分布，返回 {difficulty(int): count}。"""
    counter: Counter = Counter()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            diff = obj.get("segment_num", None)
            counter[diff] += 1  # 修正：这里应使用解析后的整数 iv
    return dict(counter)

def order_items(items: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """根据 SORT_MODE 排序 (difficulty, count)。"""
    if SORT_MODE == "count_desc":
        return sorted(items, key=lambda kv: (-kv[1], kv[0]))
    return sorted(items, key=lambda kv: kv[0])  # numeric

def save_text_and_csv(d_counts: Dict[int, int], txt_path: str, csv_path: str) -> None:
    items = order_items(list(d_counts.items()))
    total = sum(d_counts.values())

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Difficulty Distribution\n")
        f.write("=======================\n")
        f.write(f"Total: {total}\n\n")
        for k, v in items:
            pct = (v / total * 100) if total else 0.0
            f.write(f"{k}\t{v}\t({pct:.2f}%)\n")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("difficulty,count,percentage\n")
        for k, v in items:
            pct = (v / total * 100) if total else 0.0
            f.write(f"{k},{v},{pct:.4f}\n")

# =============================
# 论文风格柱状图（seaborn）
# =============================
def setup_fonts_for_cn() -> None:
    try:
        matplotlib.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS"
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

def compute_figsize(n_bars: int) -> tuple:
    """根据柱子数量动态给出合适画布，避免标签重叠。"""
    # 基础宽度 6，每 10 个柱子增加 ~2.5 宽度
    width = 6 + max(0, (n_bars - 10)) * 0.25
    width = min(max(width, 6), 20)
    height = 4.5 if n_bars <= 15 else 5.5
    return (width, height)

def plot_difficulty_bar(difficulty_counts: Dict[int, int], save_path: str = None) -> None:
    # 排序与数据准备
    items = order_items(list(difficulty_counts.items()))
    xs = [k for k, _ in items]
    ys = [v for _, v in items]

    # seaborn 全局风格（论文级）
    setup_fonts_for_cn()
    sns.set_theme(style="whitegrid", context="paper", rc={
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
    })

    fig_size = compute_figsize(len(xs))
    fig, ax = plt.subplots(figsize=fig_size)

    # 使用 seaborn.barplot（不会指定调色，保持默认以利于出版中性）
    sns.barplot(x=[str(x) for x in xs], y=ys, ax=ax, edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Difficulty (int)")
    ax.set_ylabel("Count")
    ax.set_title("Analysis of Problem Difficulty Distribution")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    sns.despine(ax=ax, left=False, bottom=False)

    # 根据密度决定是否旋转标签与调整底部边距
    n = len(xs)
    if n >= 12:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
        plt.subplots_adjust(bottom=0.18)
    else:
        plt.subplots_adjust(bottom=0.12)

    # 在柱顶标注数值（避免与图顶端贴合）
    for p in ax.patches:
        h = p.get_height()
        if h is None:
            continue
        ax.annotate(
            f"{int(h)}",
            (p.get_x() + p.get_width() / 2., h),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points"
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    plt.show()

# =============================
# 主流程
# =============================
def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"输入文件不存在：{INPUT_PATH}")

    counts = analyze_difficulty_distribution(INPUT_PATH)

    items = order_items(list(counts.items()))
    total = sum(counts.values())
    print("\nDifficulty Distribution")
    print("=======================")
    print(f"Total: {total}")
    for k, v in items:
        pct = (v / total * 100) if total else 0.0
        print(f"{k}: {v} ({pct:.2f}%)")

    save_text_and_csv(counts, OUT_TXT, OUT_CSV)
    plot_difficulty_bar(counts, save_path=OUT_PNG)

if __name__ == "__main__":
    main()
