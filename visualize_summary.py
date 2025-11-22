#!/usr/bin/env python3
"""
visualize_summary.py

读取由 `summarize_cases.py` 生成的汇总文件（JSON 或 JSONL），并生成多张高质量、论文级别的图表：
- score 分布（直方图 + KDE）
- difficulty vs score（散点 + 回归）
- num_steps vs score（散点 + 回归）
- score 与 difficulty/num_steps 的箱线图/小提琴图（按难度分箱）
- 各步得分的按步分布（小提琴图或箱线图）
- 特征相关性热图
- 成对关系图（pairplot）

输出：PNG 和 PDF 文件，保存在汇总文件同级的 `figures/` 文件夹中。

用法示例：
  python visualize_summary.py /path/to/cases_summary.json

需要库：numpy, pandas, matplotlib, seaborn, scikit-learn
安装（如果需要）：
  pip install numpy pandas matplotlib seaborn scikit-learn
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


def load_summary(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    items: List[Dict[str, Any]] = []
    try:
        # 先尝试作为 JSON 列表/对象
        obj = json.loads(text)
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            # 单对象，转换为 single-element list
            items = [obj]
    except Exception:
        # 作为 jsonl，每行一个 json
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                # 忽略不能解析的行
                continue
    return items


def build_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    step_records = []
    for it in items:
        row = {
            "id": it.get("id"),
            "difficulty": try_float(it.get("difficulty")),
            "num_steps": try_int(it.get("num_steps")),
            "score": try_float(it.get("score")),
            "_source": it.get("_source"),
        }
        # 补充通过 step_scores 计算的统计量
        step_scores = it.get("step_scores") if isinstance(it.get("step_scores"), list) else None
        if not step_scores and isinstance(it.get("steps"), list):
            # 从 steps[*].score 中提取
            ss = []
            for s in it.get("steps"):
                if isinstance(s, dict):
                    if "score" in s:
                        ss.append(try_float(s.get("score")))
                    else:
                        # routes 尝试 pairwise/holistic
                        r = s.get("routes") if isinstance(s.get("routes"), dict) else None
                        if r and isinstance(r.get("pairwise"), (int, float)):
                            ss.append(try_float(r.get("pairwise")))
                        elif r and isinstance(r.get("holistic"), (int, float)):
                            ss.append(try_float(r.get("holistic")))
                        else:
                            ss.append(None)
            step_scores = ss

        if step_scores:
            cleaned = [try_float(x) for x in step_scores]
            cleaned = [x for x in cleaned if x is not None]
            row["step_mean"] = float(np.mean(cleaned)) if cleaned else None
            row["step_median"] = float(np.median(cleaned)) if cleaned else None
            row["step_std"] = float(np.std(cleaned)) if cleaned else None
            row["num_steps_detected"] = len(step_scores)
            # 记录逐步以用于画图
            for idx, sc in enumerate(step_scores, start=1):
                step_records.append({"id": it.get("id"), "step_index": idx, "step_score": try_float(sc)})
        else:
            row["step_mean"] = None
            row["step_median"] = None
            row["step_std"] = None
            row["num_steps_detected"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    steps_df = pd.DataFrame(step_records) if step_records else pd.DataFrame(columns=["id", "step_index", "step_score"])
    return df, steps_df


def try_float(x: Any):
    try:
        return float(x)
    except Exception:
        return None


def try_int(x: Any):
    try:
        return int(x)
    except Exception:
        return None


def setup_style():
    sns.set(style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def plot_score_distribution(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    scores = df["score"].dropna()
    sns.histplot(scores, kde=True, stat="density", color="#4C72B0", edgecolor="white", ax=ax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    fig.tight_layout()
    save_figs(fig, out_dir, "score_distribution")


def plot_difficulty_vs_score(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = df.dropna(subset=["difficulty", "score"])
    sns.scatterplot(x="difficulty", y="score", data=data, s=30, color="#2A9D8F", ax=ax)
    # 回归/平滑线：在主入口中控制 lowess 是否启用
    try:
        use_lowess = getattr(plot_difficulty_vs_score, "use_lowess", True)
    except Exception:
        use_lowess = True
    if use_lowess:
        sns.regplot(x="difficulty", y="score", data=data, scatter=False, lowess=True, ax=ax, color="#E76F51")
    else:
        sns.regplot(x="difficulty", y="score", data=data, scatter=False, ax=ax, color="#E76F51")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Score")
    ax.set_title("Difficulty vs Score")
    fig.tight_layout()
    save_figs(fig, out_dir, "difficulty_vs_score")


def plot_num_steps_vs_score(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = df.copy()
    # choose either provided num_steps or detected
    data["num_steps_final"] = data["num_steps"].fillna(data["num_steps_detected"]).astype(float)
    data = data.dropna(subset=["num_steps_final", "score"])
    sns.scatterplot(x="num_steps_final", y="score", data=data, s=30, color="#4C72B0", ax=ax)
    sns.regplot(x="num_steps_final", y="score", data=data, scatter=False, ax=ax, color="#F4A261")
    ax.set_xlabel("Num Steps")
    ax.set_ylabel("Score")
    ax.set_title("Num Steps vs Score")
    fig.tight_layout()
    save_figs(fig, out_dir, "num_steps_vs_score")


def plot_difficulty_bins_box(df: pd.DataFrame, out_dir: Path, n_bins: int = 5):
    data = df.dropna(subset=["difficulty", "score"]).copy()
    if data.empty:
        return
    data["diff_bin"] = pd.qcut(data["difficulty"], q=n_bins, duplicates="drop")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x="diff_bin", y="score", data=data, palette="Set3", ax=ax)
    ax.set_xlabel("Difficulty bins")
    ax.set_ylabel("Score")
    ax.set_title("Score by Difficulty Bins")
    plt.xticks(rotation=30)
    fig.tight_layout()
    save_figs(fig, out_dir, "score_by_difficulty_bins")


def plot_ecdf_score(df: pd.DataFrame, out_dir: Path):
    """绘制 Score 的 ECDF（累积分布函数），并标注常用分位点。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    scores = df["score"].dropna()
    if scores.empty:
        return
    try:
        sns.ecdfplot(scores, ax=ax, color="#2A9D8F")
    except Exception:
        # 备用实现
        s = np.sort(scores.values)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.step(s, y, where="post", color="#2A9D8F")

    # 标注 25/50/75 分位点
    q25, q50, q75 = np.percentile(scores.dropna(), [25, 50, 75])
    for q, lbl in [(q25, "25%"), (q50, "50%"), (q75, "75%")]:
        ax.axvline(q, color="#E76F51", linestyle="--", linewidth=0.8)
        ax.text(q, 0.02, lbl, rotation=90, verticalalignment="bottom", color="#E76F51", fontsize=8)

    ax.set_xlabel("Score")
    ax.set_ylabel("ECDF")
    ax.set_title("Score ECDF")
    fig.tight_layout()
    save_figs(fig, out_dir, "score_ecdf")


def plot_raincloud_by_difficulty(df: pd.DataFrame, out_dir: Path, n_bins: int = 4):
    """实现类似 raincloud 的复合图：violin + box + swarm，用于展示不同 difficulty bin 的分布。"""
    data = df.dropna(subset=["difficulty", "score"]).copy()
    if data.empty:
        return
    data["diff_bin"] = pd.qcut(data["difficulty"], q=n_bins, duplicates="drop")
    order = sorted(data["diff_bin"].unique(), key=lambda x: x.left if hasattr(x, 'left') else str(x))
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.violinplot(x="diff_bin", y="score", data=data, order=order, inner=None, palette="pastel", ax=ax)
    sns.boxplot(x="diff_bin", y="score", data=data, order=order, width=0.12, showcaps=True, boxprops={"facecolor":"none"}, showfliers=False, whiskerprops={"linewidth":0.8}, ax=ax)
    sns.stripplot(x="diff_bin", y="score", data=data, order=order, color="k", size=3, jitter=0.15, alpha=0.4, ax=ax)
    ax.set_xlabel("Difficulty bins")
    ax.set_ylabel("Score")
    ax.set_title("Raincloud-like: Score by Difficulty")
    plt.xticks(rotation=25)
    fig.tight_layout()
    save_figs(fig, out_dir, "raincloud_score_by_difficulty")


def plot_step_heatmap(df: pd.DataFrame, steps_df: pd.DataFrame, out_dir: Path, n_bins: int = 4, max_steps: int = 12):
    """按 difficulty bin 聚合每一步的平均分，绘制 heatmap（rows=difficulty bin, cols=step index）。"""
    if steps_df.empty or df.empty:
        return
    meta = df[["id", "difficulty"]].dropna()
    merged = steps_df.merge(meta, on="id", how="inner")
    if merged.empty:
        return
    merged["diff_bin"] = pd.qcut(merged["difficulty"], q=n_bins, duplicates="drop")
    merged = merged[merged["step_index"] <= max_steps]
    pivot = merged.groupby(["diff_bin", "step_index"]).agg(mean_score=("step_score", "mean"), cnt=("step_score", "count")).reset_index()
    if pivot.empty:
        return
    # 构造矩阵
    bins = list(pivot["diff_bin"].unique())
    steps = list(range(1, max_steps + 1))
    mat = np.full((len(bins), len(steps)), np.nan)
    for i, b in enumerate(bins):
        row = pivot[pivot["diff_bin"] == b]
        for j, s in enumerate(steps):
            v = row[row["step_index"] == s]["mean_score"]
            if not v.empty:
                mat[i, j] = float(v.values[0])

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(steps)), max(3, 0.8 * len(bins))))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="rocket", cbar_kws={"label": "mean step score"}, ax=ax, linewidths=0.5)
    ax.set_yticks(np.arange(len(bins)) + 0.5)
    ax.set_yticklabels([str(b) for b in bins])
    ax.set_xticks(np.arange(len(steps)) + 0.5)
    ax.set_xticklabels(steps)
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Difficulty bin")
    ax.set_title(f"Per-step Mean Score Heatmap (first {max_steps} steps)")
    fig.tight_layout()
    save_figs(fig, out_dir, "step_heatmap")


def plot_stepwise_distribution(steps_df: pd.DataFrame, out_dir: Path, max_steps: int = 12):
    if steps_df.empty:
        return
    # 仅保留前 max_steps 步以便图形整洁
    filtered = steps_df[steps_df["step_index"] <= max_steps].copy()
    if filtered.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.violinplot(x="step_index", y="step_score", data=filtered, inner="quartile", palette="Blues", ax=ax)
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Step Score")
    ax.set_title(f"Per-step Score Distribution (first {max_steps} steps)")
    fig.tight_layout()
    save_figs(fig, out_dir, "per_step_violin")


def plot_correlation(df: pd.DataFrame, out_dir: Path):
    cols = ["score", "difficulty", "num_steps", "step_mean", "step_median", "step_std"]
    sub = df[cols].copy()
    sub = sub.apply(pd.to_numeric, errors="coerce")
    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Feature Correlation")
    fig.tight_layout()
    save_figs(fig, out_dir, "feature_correlation")


def plot_pairwise(df: pd.DataFrame, out_dir: Path):
    cols = [c for c in ["score", "difficulty", "num_steps", "step_mean"] if c in df.columns]
    data = df[cols].dropna()
    if data.shape[0] < 2 or len(cols) < 2:
        return
    sns.set(style="ticks")
    g = sns.pairplot(data, diag_kind="kde", plot_kws={"s": 20, "alpha": 0.6})
    g.fig.suptitle("Pairwise Relationships", y=1.02)
    # 保存
    out_png = out_dir / "pairwise.png"
    out_pdf = out_dir / "pairwise.pdf"
    g.fig.savefig(out_png, bbox_inches="tight", dpi=300)
    g.fig.savefig(out_pdf, bbox_inches="tight", dpi=300)


def save_figs(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize summary JSON/JSONL from summarize_cases.py")
    parser.add_argument("summary", help="Path to summary JSON or JSONL file")
    parser.add_argument("--out-dir", help="Output directory for figures (defaults to summary parent/figures)")
    parser.add_argument("--lowess", action="store_true", help="Enable lowess smoothing for difficulty vs score (requires statsmodels)")
    parser.add_argument("--no-ecdf", action="store_true", help="Do not generate ECDF plot")
    parser.add_argument("--no-raincloud", action="store_true", help="Do not generate raincloud plot")
    parser.add_argument("--add-step-heatmap", action="store_true", help="Add per-step heatmap by difficulty bins")
    parser.add_argument("--max-step", type=int, default=30, help="Max step index to show in per-step plot")
    args = parser.parse_args()

    path = Path(args.summary)
    if not path.exists():
        raise SystemExit(f"Summary file not found: {path}")

    items = load_summary(path)
    df, steps_df = build_dataframe(items)

    # 风格设置
    setup_style()

    out_dir = Path(args.out_dir) if args.out_dir else path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 生成图表
    plot_score_distribution(df, out_dir)
    # 控制 lowess
    plot_difficulty_vs_score.use_lowess = bool(args.lowess)
    plot_difficulty_vs_score(df, out_dir)
    plot_num_steps_vs_score(df, out_dir)
    plot_difficulty_bins_box(df, out_dir)
    plot_stepwise_distribution(steps_df, out_dir, max_steps=args.max_step)
    # ECDF
    if not args.no_ecdf:
        plot_ecdf_score(df, out_dir)
    # Raincloud-like plot by difficulty
    if not args.no_raincloud:
        plot_raincloud_by_difficulty(df, out_dir)
    # Per-step heatmap by difficulty bins
    if args.add_step_heatmap:
        plot_step_heatmap(df, steps_df, out_dir, max_steps=args.max_step)
    plot_correlation(df, out_dir)
    plot_pairwise(df, out_dir)

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
