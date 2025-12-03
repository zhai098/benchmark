#!/usr/bin/env python3
"""
per_step_pairwise_holistic_stats.py

统计 `cases` 目录中每个 case 的每一步（step）中 pairwise 与 holistic 的分数，生成多维度统计与图表：
- 每步平均趋势（mean ± std）
- 每步箱线/小提琴图（分布）
- pairwise vs holistic 的散点图（可按 step 上色）
- 按 difficulty bin 的 per-step heatmap（均值）
- 整体分布与相关性

用法示例：
  python per_step_pairwise_holistic_stats.py --cases-dir /path/to/cases --out-dir /path/to/out --max-step 12

依赖：numpy pandas matplotlib seaborn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_cases(cases_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in sorted(cases_dir.rglob("*.json")):
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    for p in sorted(cases_dir.rglob("*.jsonl")):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def extract_step_scores(items: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从 cases 列表中抽取逐步 pairwise / holistic 的分数表格。
    返回 (df_steps, df_raw):
      df_steps: columns = [id, difficulty, step_index, pairwise, holistic, step_score]
      df_raw: columns = [id, difficulty, step_index, raw_pairwise]
    """
    records: List[Dict[str, Any]] = []
    raw_records: List[Dict[str, Any]] = []

    for it in items:
        cid = it.get("id")
        diff = it.get("difficulty")
        steps = it.get("steps") if isinstance(it.get("steps"), list) else None
        # 兼容部分格式：有时顶层有 gen_output 中的 steps 也可能为空
        if not steps:
            continue
        for s in steps:
            idx = s.get("index") if isinstance(s.get("index"), int) else None
            
            pairwise = None
            holistic = None
            pairwise_raw_vals = []

            # 优先 routes
            routes = s.get("routes") if isinstance(s.get("routes"), dict) else None
            if routes:
                if isinstance(routes.get("pairwise"), (int, float)):
                    pairwise = float(routes.get("pairwise"))
                if isinstance(routes.get("holistic"), (int, float)):
                    holistic = float(routes.get("holistic"))

            # 再尝试 judge_detail 中的聚合或者原始 detail
            jd = s.get("judge_detail") if isinstance(s.get("judge_detail"), dict) else None
            if jd:
                # 可能 judge_detail 包含 'pairwise' 和 'holistic' 子 dicts
                if isinstance(jd.get("pairwise"), dict):
                    p = jd.get("pairwise")
                    # Extract raw scores if available
                    if isinstance(p.get("scores"), list):
                        pairwise_raw_vals = [float(x) for x in p.get("scores") if isinstance(x, (int, float))]
                    
                    # Extract aggregated if not already set
                    if pairwise is None:
                        if isinstance(p.get("score"), (int, float)):
                            pairwise = float(p.get("score"))
                        elif pairwise_raw_vals:
                            pairwise = float(np.mean(pairwise_raw_vals))

                if holistic is None and isinstance(jd.get("holistic"), dict):
                    h = jd.get("holistic")
                    if isinstance(h.get("score"), (int, float)):
                        holistic = float(h.get("score"))

            # 兼容：某些 case 直接把路由分数放在 step['score'] 或 step['pairwise']
            if pairwise is None and isinstance(s.get("pairwise"), (int, float)):
                pairwise = float(s.get("pairwise"))
            if holistic is None and isinstance(s.get("holistic"), (int, float)):
                holistic = float(s.get("holistic"))

            # 一些格式会把 step 的总分放在 s['score']，但这不是 pairwise/holistic，仍记录
            step_score = None
            if isinstance(s.get("score"), (int, float)):
                step_score = float(s.get("score"))

            # Ensure idx exists: fallback to enumeration if missing
            if idx is None:
                continue

            records.append({
                "id": cid,
                "difficulty": try_float(diff),
                "step_index": int(idx),
                "pairwise": try_float(pairwise),
                "holistic": try_float(holistic),
                "step_score": try_float(step_score),
            })
            
            # Add raw records
            if pairwise_raw_vals:
                for v in pairwise_raw_vals:
                    raw_records.append({
                        "id": cid,
                        "difficulty": try_float(diff),
                        "step_index": int(idx),
                        "raw_pairwise": v
                    })
            elif pairwise is not None:
                 raw_records.append({
                    "id": cid,
                    "difficulty": try_float(diff),
                    "step_index": int(idx),
                    "raw_pairwise": pairwise
                })

    df = pd.DataFrame(records)
    df_raw = pd.DataFrame(raw_records)
    return df, df_raw


def try_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def plot_step_trends(df: pd.DataFrame, out_dir: Path, max_step: int = 12):
    if df.empty:
        return
    # only keep steps up to max_step
    data = df[df["step_index"] <= max_step]
    if data.empty:
        return

    stats = data.groupby("step_index").agg(
        pairwise_mean=("pairwise", lambda x: np.nanmean(x.values)),
        pairwise_std=("pairwise", lambda x: np.nanstd(x.values)),
        holistic_mean=("holistic", lambda x: np.nanmean(x.values)),
        holistic_std=("holistic", lambda x: np.nanstd(x.values)),
        cnt=("pairwise", lambda x: np.count_nonzero(~pd.isna(x)))
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(stats["step_index"], stats["pairwise_mean"], label="pairwise mean", marker="o", color="#4C72B0")
    ax.fill_between(stats["step_index"], stats["pairwise_mean"] - stats["pairwise_std"], stats["pairwise_mean"] + stats["pairwise_std"], color="#4C72B0", alpha=0.2)
    ax.plot(stats["step_index"], stats["holistic_mean"], label="holistic mean", marker="o", color="#E76F51")
    ax.fill_between(stats["step_index"], stats["holistic_mean"] - stats["holistic_std"], stats["holistic_mean"] + stats["holistic_std"], color="#E76F51", alpha=0.2)
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Score")
    ax.set_title("Per-step Mean ± STD (pairwise vs holistic)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir, "per_step_trends")


def plot_step_boxplots(df_steps: pd.DataFrame, df_raw: pd.DataFrame, out_dir: Path, max_step: int = 12):
    # Prepare data for pairwise from df_raw
    data_p = df_raw[df_raw["step_index"] <= max_step].copy()
    data_p["route"] = "pairwise"
    data_p = data_p.rename(columns={"raw_pairwise": "score"})
    
    # Prepare data for holistic from df_steps
    data_h = df_steps[df_steps["step_index"] <= max_step].copy()
    data_h["route"] = "holistic"
    data_h = data_h.rename(columns={"holistic": "score"})
    data_h = data_h.dropna(subset=["score"])
    
    # Combine
    combined = pd.concat([data_p[["step_index", "score", "route"]], data_h[["step_index", "score", "route"]]], ignore_index=True)
    
    if combined.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * max_step), 5))
    sns.boxplot(x="step_index", y="score", hue="route", data=combined, ax=ax, palette=["#4C72B0", "#E76F51"]) 
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Score")
    ax.set_title("Per-step Boxplots (Raw Pairwise Dist vs Holistic)")
    ax.legend(title="route")
    fig.tight_layout()
    save_fig(fig, out_dir, "per_step_boxplots")


def plot_pairwise_vs_holistic(df: pd.DataFrame, out_dir: Path, sample_limit: int = 5000):
    if df.empty:
        return
    data = df.dropna(subset=["pairwise", "holistic"]) 
    if data.empty:
        return
    # sample if too many points
    if len(data) > sample_limit:
        data = data.sample(sample_limit, random_state=42)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(data["pairwise"], data["holistic"], c=data["step_index"], cmap="viridis", alpha=0.6, s=12)
    ax.set_xlabel("Pairwise Score")
    ax.set_ylabel("Holistic Score")
    ax.set_title("Pairwise vs Holistic (colored by step index)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Step Index")
    # diag
    lims = [0, 5]
    ax.plot(lims, lims, linestyle="--", color="gray", linewidth=0.8)
    fig.tight_layout()
    save_fig(fig, out_dir, "pairwise_vs_holistic")


def plot_heatmap_by_difficulty(df: pd.DataFrame, out_dir: Path, n_bins: int = 4, max_steps: int = 12):
    if df.empty:
        return
    meta = df[["id", "difficulty"]].dropna()
    merged = df.merge(meta, on="id", how="left") if "difficulty" not in df.columns else df
    # Ensure difficulty exists
    if merged["difficulty"].isna().all():
        return
    merged = merged[merged["step_index"] <= max_steps]
    merged["diff_bin"] = pd.qcut(merged["difficulty"].fillna(0.0), q=n_bins, duplicates="drop")

    # pairwise heatmap
    pivot_p = merged.groupby(["diff_bin", "step_index"]).agg(mean_pairwise=("pairwise", "mean")).reset_index()
    bins = list(pivot_p["diff_bin"].unique())
    steps = list(range(1, max_steps + 1))
    mat = np.full((len(bins), len(steps)), np.nan)
    for i, b in enumerate(bins):
        row = pivot_p[pivot_p["diff_bin"] == b]
        for j, s in enumerate(steps):
            v = row[row["step_index"] == s]["mean_pairwise"]
            if not v.empty:
                mat[i, j] = float(v.values[0])
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(steps)), max(3, 0.8 * len(bins))))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="rocket", ax=ax)
    ax.set_title("Mean Pairwise per Step by Difficulty Bin")
    ax.set_xlabel("Step Index")
    fig.tight_layout()
    save_fig(fig, out_dir, "heatmap_pairwise_by_diff")

    # holistic heatmap
    pivot_h = merged.groupby(["diff_bin", "step_index"]).agg(mean_holistic=("holistic", "mean")).reset_index()
    mat_h = np.full((len(bins), len(steps)), np.nan)
    for i, b in enumerate(bins):
        row = pivot_h[pivot_h["diff_bin"] == b]
        for j, s in enumerate(steps):
            v = row[row["step_index"] == s]["mean_holistic"]
            if not v.empty:
                mat_h[i, j] = float(v.values[0])
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(steps)), max(3, 0.8 * len(bins))))
    sns.heatmap(mat_h, annot=True, fmt=".2f", cmap="rocket", ax=ax)
    ax.set_title("Mean Holistic per Step by Difficulty Bin")
    ax.set_xlabel("Step Index")
    fig.tight_layout()
    save_fig(fig, out_dir, "heatmap_holistic_by_diff")


def plot_overall_distributions(df_steps: pd.DataFrame, df_raw: pd.DataFrame, out_dir: Path):
    # 1. Original: Aggregated Pairwise vs Holistic
    if not df_steps.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df_steps["pairwise"].dropna(), kde=True, ax=axes[0], color="#4C72B0")
        axes[0].set_title("Aggregated Pairwise Distribution")
        sns.histplot(df_steps["holistic"].dropna(), kde=True, ax=axes[1], color="#E76F51")
        axes[1].set_title("Aggregated Holistic Distribution")
        fig.tight_layout()
        save_fig(fig, out_dir, "overall_distributions")

    # 2. New: Raw Pairwise vs Aggregated Pairwise
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    
    if not df_raw.empty and "raw_pairwise" in df_raw.columns:
        sns.histplot(df_raw["raw_pairwise"].dropna(), kde=True, ax=axes2[0], color="#8FBC8F") # DarkSeaGreen
    axes2[0].set_title("Raw Pairwise Scores (Individual Votes)")
    
    if not df_steps.empty and "pairwise" in df_steps.columns:
        sns.histplot(df_steps["pairwise"].dropna(), kde=True, ax=axes2[1], color="#4C72B0")
    axes2[1].set_title("Aggregated Pairwise Scores (Per Step)")
    
    fig2.tight_layout()
    save_fig(fig2, out_dir, "pairwise_raw_vs_agg_distribution")


def save_fig(fig: plt.Figure, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Per-step stats for pairwise and holistic scores from cases folder")
    parser.add_argument("--cases-dir", required=True, help="Path to cases folder (contains json/jsonl files)")
    parser.add_argument("--out-dir", required=True, help="Output directory for figures and CSVs")
    parser.add_argument("--max-step", type=int, default=30, help="Max step index to include in plots")
    parser.add_argument("--n-bins", type=int, default=6, help="Difficulty bins for heatmap")
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    out_dir = Path(args.out_dir)
    items = load_cases(cases_dir)
    df_steps, df_raw = extract_step_scores(items)

    # save raw table
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_step_scores.csv"
    df_steps.to_csv(csv_path, index=False)
    
    csv_raw_path = out_dir / "per_step_raw_pairwise_scores.csv"
    df_raw.to_csv(csv_raw_path, index=False)
    
    print(f"Wrote per-step table to {csv_path}, entries={len(df_steps)}")
    print(f"Wrote per-step raw pairwise table to {csv_raw_path}, entries={len(df_raw)}")

    # plots
    plot_step_trends(df_steps, out_dir, max_step=args.max_step)
    plot_step_boxplots(df_steps, df_raw, out_dir, max_step=args.max_step)
    plot_pairwise_vs_holistic(df_steps, out_dir)
    plot_heatmap_by_difficulty(df_steps, out_dir, n_bins=args.n_bins, max_steps=args.max_step)
    plot_overall_distributions(df_steps, df_raw, out_dir)


if __name__ == "__main__":
    main()
