#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


"""
对 Progress_Prompt 的输出结果做统计分析并画论文级图表。

输入：JSONL，每行类似：
{
  "id": case_id,
  "gen_idx": idx,
  "problem": "...",
  "gen": "...",
  "progress_score": 0/1/2,
  "raw_output": ...
}

输出：
- 若干统计信息（打印到终端）
- 若干高质量图表（PDF + PNG）
- 一个 zero_scores.tsv，记录所有 score=0 的 (id, gen_idx)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_matplotlib_style():
    """设置比较接近论文风格的绘图样式。"""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (4.5, 3.2),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })
    sns.set_theme(style="whitegrid", font_scale=1.0)


def load_jsonl(path):
    """读取 JSONL 为 DataFrame。"""
    # 直接用 pandas 读更方便
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    return df


def ensure_outdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def print_basic_stats(df):
    """打印一些基础统计信息。"""
    df_valid = df.dropna(subset=["progress_score"])
    n = len(df_valid)
    mean_score = df_valid["progress_score"].mean()
    print(f"总记录数: {n}")
    print(f"平均 progress_score: {mean_score:.3f}")

    counts = df_valid["progress_score"].value_counts().sort_index()
    print("score 频数分布:")
    for s, c in counts.items():
        print(f"  score={s}: {c} 条")

    # 每个 case 的统计
    case_stats = (
        df_valid
        .groupby("id")["progress_score"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values("mean")
    )
    print("\n每个 case 的进展统计（前几行示例）:")
    print(case_stats.head(10).to_string(index=False))


def dump_zero_scores(df, outdir):
    """输出所有 score=0 的 (id, gen_idx) 以及全局顺序。"""
    df_valid = df.dropna(subset=["progress_score"])
    df_zero = df_valid[df_valid["progress_score"] == 0].copy()

    # 增加一个 global_idx（按原文件顺序）
    df_zero = df_zero.reset_index().rename(columns={"index": "global_idx"})
    df_zero = df_zero.sort_values(["id", "gen_idx"])

    out_path = os.path.join(outdir, "zero_scores.tsv")
    df_zero[["global_idx", "id", "gen_idx"]].to_csv(
        out_path, sep="\t", index=False
    )

    print(f"\nscore=0 的记录数: {len(df_zero)}")
    print(f"已写入: {out_path}")
    return df_zero


def plot_score_distribution(df, outdir):
    """图1：整体 score 分布柱状图（离散 0/1/2）。"""
    df_valid = df.dropna(subset=["progress_score"])
    counts = (
        df_valid["progress_score"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    counts.columns = ["progress_score", "count"]

    fig, ax = plt.subplots()
    sns.barplot(
        data=counts,
        x="progress_score",
        y="count",
        ax=ax,
        palette="viridis"
    )

    for i, row in counts.iterrows():
        ax.text(
            i,
            row["count"],
            f"{int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    ax.set_xlabel("progress_score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of progress scores")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "score_distribution_bar.pdf"))
    fig.savefig(os.path.join(outdir, "score_distribution_bar.png"))
    plt.close(fig)


def plot_score_vs_step(df, outdir):
    """
    图2：按 gen_idx 聚合的 score 曲线：
    - x: gen_idx
    - y: mean score
    - 阴影: 标准误差 / 标准差
    """
    df_valid = df.dropna(subset=["progress_score"])
    stats = (
        df_valid
        .groupby("gen_idx")["progress_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("gen_idx")
    )
    # 标准误差
    stats["sem"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(5, 3.2))

    ax.plot(
        stats["gen_idx"],
        stats["mean"],
        marker="o",
        linewidth=1.5,
        label="Mean score"
    )
    ax.fill_between(
        stats["gen_idx"],
        stats["mean"] - stats["sem"],
        stats["mean"] + stats["sem"],
        alpha=0.2,
        label="±1 SEM"
    )

    ax.set_xlabel("gen_idx (segment index)")
    ax.set_ylabel("Mean progress_score")
    ax.set_title("Score vs. step index")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "score_vs_step_mean_sem.pdf"))
    fig.savefig(os.path.join(outdir, "score_vs_step_mean_sem.png"))
    plt.close(fig)


def plot_case_level_stats(df, outdir, top_k=40):
    """
    图3：每个 case 的平均分柱状图（截取最多 top_k 个，可用于论文中的例子）。
    """
    df_valid = df.dropna(subset=["progress_score"])
    case_stats = (
        df_valid
        .groupby("id")["progress_score"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values("mean")
    )

    # 如果 case 很多，只画最前/最后的 top_k
    if len(case_stats) > top_k:
        case_stats_plot = case_stats.head(top_k)
    else:
        case_stats_plot = case_stats

    fig, ax = plt.subplots(figsize=(5, 0.2 * len(case_stats_plot) + 1.5))

    sns.barplot(
        data=case_stats_plot,
        x="mean",
        y="id",
        ax=ax,
        palette="mako"
    )
    ax.set_xlabel("Mean progress_score")
    ax.set_ylabel("Case id")
    ax.set_title("Per-case mean progress_score")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "case_mean_scores.pdf"))
    fig.savefig(os.path.join(outdir, "case_mean_scores.png"))
    plt.close(fig)


def plot_score_heatmap(df, outdir, max_cases=30):
    """
    图4：score 的热力图（id x gen_idx），适合展示结构模式。
    为了可读，默认最多画 max_cases 个 case。
    """
    df_valid = df.dropna(subset=["progress_score"])

    # 为了让 heatmap 更稳定，选取记录数最多的若干个 case
    case_counts = (
        df_valid
        .groupby("id")["progress_score"]
        .count()
        .sort_values(ascending=False)
    )
    selected_ids = case_counts.head(max_cases).index.tolist()
    df_sel = df_valid[df_valid["id"].isin(selected_ids)].copy()

    # 构造透视表：行是 id，列是 gen_idx
    pivot = df_sel.pivot_table(
        index="id",
        columns="gen_idx",
        values="progress_score",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(6, 0.25 * len(pivot) + 1.5))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=2,
        cbar_kws={"label": "progress_score"}
    )

    ax.set_xlabel("gen_idx")
    ax.set_ylabel("Case id")
    ax.set_title("Heatmap of progress scores")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "score_heatmap.pdf"))
    fig.savefig(os.path.join(outdir, "score_heatmap.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True,
        help="Progress_Prompt 输出的 JSONL 文件"
    )
    parser.add_argument(
        "--outdir", type=str, required=True,
        help="图表和统计结果的输出目录"
    )
    parser.add_argument(
        "--heatmap_max_cases", type=int, default=30,
        help="热力图中最多展示的 case 数量"
    )
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    setup_matplotlib_style()

    print(f"读取数据: {args.input}")
    df = load_jsonl(args.input)

    print_basic_stats(df)
    dump_zero_scores(df, args.outdir)

    print("\n开始绘图...")
    plot_score_distribution(df, args.outdir)
    plot_score_vs_step(df, args.outdir)
    plot_case_level_stats(df, args.outdir)
    plot_score_heatmap(df, args.outdir, max_cases=args.heatmap_max_cases)

    print(f"\n完成，所有图表已输出到目录: {args.outdir}")


if __name__ == "__main__":
    main()
