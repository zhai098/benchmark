#!/usr/bin/env python3
"""
pairwise_aggregator.py

提供多个更合理的 pairwise 分数聚合策略，用于替换原先在 `judge.py` 中直接取最小值的做法。

策略包括：
- min / max / mean / median
- trimmed_mean（去除极端值后的均值）
- winsorized_mean（对极端值进行 Winsorize 后取均值）
- top_k_mean（只取最高 k 个或最低 k 个的均值）
- consensus（基于阈值的通过率，返回一个结合通过率与平均的分数）
- weighted_by_pos（支持根据 ref 步位置指定权重）

命令行用法示例：
  python pairwise_aggregator.py --scores 5,4,3,2 --method trimmed_mean

返回值：浮点数分数（保持原评分尺度不变）以及附带的 details 字典
"""
from __future__ import annotations

import argparse
import math
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple


def _validate_scores(scores: Iterable[float]) -> List[float]:
    s = [float(x) for x in scores if x is not None]
    return s


def agg_min(scores: List[float]) -> Tuple[float, Dict]:
    return (min(scores), {"method": "min"})


def agg_max(scores: List[float]) -> Tuple[float, Dict]:
    return (max(scores), {"method": "max"})


def agg_mean(scores: List[float]) -> Tuple[float, Dict]:
    return (mean(scores), {"method": "mean"})


def agg_median(scores: List[float]) -> Tuple[float, Dict]:
    return (median(scores), {"method": "median"})


def agg_trimmed_mean(scores: List[float], proportiontocut: float = 0.2) -> Tuple[float, Dict]:
    """Trimmed mean: 去掉两端各 proportiontocut 的数据后取均值。
    proportiontocut: 0..0.5
    """
    n = len(scores)
    if n == 0:
        return 0.0, {"method": "trimmed_mean", "proportiontocut": proportiontocut}
    k = int(math.floor(proportiontocut * n))
    if 2 * k >= n:
        # 如果去除后为空，回退为中位数
        return median(scores), {"method": "trimmed_mean", "proportiontocut": proportiontocut, "note": "trim_too_large, used_median"}
    s_sorted = sorted(scores)
    trimmed = s_sorted[k: n - k]
    return (mean(trimmed), {"method": "trimmed_mean", "proportiontocut": proportiontocut, "num_kept": len(trimmed)})


def agg_winsorized_mean(scores: List[float], proportiontocut: float = 0.1) -> Tuple[float, Dict]:
    """Winsorized mean: 将两端各 proportiontocut 的值替换为边界值后取均值。"""
    n = len(scores)
    if n == 0:
        return 0.0, {"method": "winsorized_mean", "proportiontocut": proportiontocut}
    k = int(math.floor(proportiontocut * n))
    s_sorted = sorted(scores)
    if k == 0:
        return mean(s_sorted), {"method": "winsorized_mean", "proportiontocut": proportiontocut}
    low = s_sorted[k]
    high = s_sorted[-k - 1]
    wins = [min(max(x, low), high) for x in s_sorted]
    return (mean(wins), {"method": "winsorized_mean", "proportiontocut": proportiontocut})


def agg_top_k_mean(scores: List[float], k: int = 1, take_highest: bool = True) -> Tuple[float, Dict]:
    s_sorted = sorted(scores, reverse=take_highest)
    k = max(1, min(k, len(s_sorted)))
    chosen = s_sorted[:k]
    return (mean(chosen), {"method": "top_k_mean", "k": k, "take_highest": take_highest})


def agg_consensus(scores: List[float], threshold: float = 3.0) -> Tuple[float, Dict]:
    """Consensus-based aggregator:
    - 计算达到阈值的比例 p
    - 计算平均得分 m
    - 返回 p * m + (1-p) * m * downweight，或更简单返回 m * (0.5 + 0.5 * p)
    这样既考虑通过率也保留平均信息。
    """
    if not scores:
        return 0.0, {"method": "consensus", "threshold": threshold}
    p = sum(1 for x in scores if x >= threshold) / len(scores)
    m = mean(scores)
    agg = m * (0.5 + 0.5 * p)
    return (agg, {"method": "consensus", "threshold": threshold, "pass_rate": p, "mean": m})


def agg_weighted_by_pos(scores: List[float], weights: Optional[List[float]] = None) -> Tuple[float, Dict]:
    """按位置加权：weights 长度应与 scores 一致；若未提供，则默认位置权重为指数衰减（更早的 ref 步权重大/小由 param 决定）。"""
    n = len(scores)
    if n == 0:
        return 0.0, {"method": "weighted_by_pos"}
    if weights:
        if len(weights) != n:
            raise ValueError("weights length must match scores length")
        w = [float(x) for x in weights]
    else:
        # 默认：对近步（较小 index）赋予较高权重 -> 指数衰减
        # 假设 scores 按 ref order 从近到远
        decay = 0.8
        w = [decay ** i for i in range(n)]
    s = sum(w)
    agg = sum(sv * wv for sv, wv in zip(scores, w)) / s
    return (agg, {"method": "weighted_by_pos", "weights": w})


def aggregate_pair_scores(scores: Iterable[float], *, method: str = "trimmed_mean", **kwargs) -> Tuple[float, Dict]:
    """统一入口，返回 (agg_score, details)。

    method 支持：min,max,mean,median,trimmed_mean,winsorized_mean,top_k_mean,consensus,weighted_by_pos
    """
    s = _validate_scores(scores)
    if not s:
        return 0.0, {"method": method, "note": "empty_scores"}

    method = method.lower()
    if method == "min":
        return agg_min(s)
    if method == "max":
        return agg_max(s)
    if method == "mean":
        return agg_mean(s)
    if method == "median":
        return agg_median(s)
    if method == "trimmed_mean":
        pct = float(kwargs.get("proportiontocut", 0.2))
        return agg_trimmed_mean(s, proportiontocut=pct)
    if method == "winsorized_mean":
        pct = float(kwargs.get("proportiontocut", 0.1))
        return agg_winsorized_mean(s, proportiontocut=pct)
    if method == "top_k_mean":
        k = int(kwargs.get("k", 1))
        take_high = bool(kwargs.get("take_highest", True))
        return agg_top_k_mean(s, k=k, take_highest=take_high)
    if method == "consensus":
        thr = float(kwargs.get("threshold", 3.0))
        return agg_consensus(s, threshold=thr)
    if method == "weighted_by_pos":
        weights = kwargs.get("weights")
        return agg_weighted_by_pos(s, weights=weights)

    # 回退
    return agg_mean(s)


def _parse_scores_arg(arg: str) -> List[float]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return [float(x) for x in parts]


def main():
    parser = argparse.ArgumentParser(description="Aggregate pairwise scores with robust strategies")
    parser.add_argument("--scores", type=str, help="Comma-separated list of scores, e.g. 5,4,3")
    parser.add_argument("--method", type=str, default="trimmed_mean", help="Aggregation method")
    parser.add_argument("--proportiontocut", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--take_highest", action="store_true")
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()

    if not args.scores:
        print("Please pass --scores")
        return
    scores = _parse_scores_arg(args.scores)
    agg, details = aggregate_pair_scores(scores, method=args.method, proportiontocut=args.proportiontocut, k=args.k, take_highest=args.take_highest, threshold=args.threshold)
    print(f"scores={scores}\nmethod={details.get('method')}\nagg={agg:.4f}\ndetails={details}")


if __name__ == "__main__":
    main()
