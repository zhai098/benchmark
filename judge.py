# stage2_judge.py
from __future__ import annotations

import os, json, time, argparse, logging
from typing import Dict, Any, List, Tuple
from config import Config
from runner import VLLMRunner
from data_process import Processor, _write_jsonl_line, _write_pretty_json, _normalize_generation_input
from prompt import Judge_Prompt, Pairwise_Prompt, Holistic_Prompt, SelfJudge_Prompt

logger = logging.getLogger(__name__)

# === 与 main.py 保持同逻辑的全局组件 ===
processor = Processor()

def build_judge_model():
    judge_model = VLLMRunner(
        Config["judge_model"],
        vllm_config=Config["judge_model_params"],
        sampling_config=Config["judge_sampling_params"],
        gpus=Config["judge_model_gpus"],
    )
    # 三路 evaluator
    pairwise = Pairwise_Prompt(judge_model)
    holistic = Holistic_Prompt(judge_model)
    selfjudge = SelfJudge_Prompt(judge_model)
    return pairwise, holistic, selfjudge


# ---- 聚合器实现：可用 Config 选择 ----
def _aggregate(scores: List[float], *,
               mode: str = "weighted",
               weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
               threshold: float = 3.0) -> Tuple[float, bool]:
    """
    scores = [pairwise, holistic, selfjudge], 各 ∈ [0,5]
    mode: 'weighted' | 'mean' | 'min' | 'geom' | 'harmonic' | 'vote'
    返回: (agg_score, is_hallucination)
    """
    import math

    # 清理 NaN
    clean = [0.0 if s is None else float(s) for s in scores]

    if mode == "weighted":
        w = list(weights)
        if len(w) != 3 or sum(w) <= 0:
            w = [0.4, 0.4, 0.2]
        agg = sum(s * w_i for s, w_i in zip(clean, w)) / sum(w)
    elif mode == "mean":
        agg = sum(clean) / len(clean)
    elif mode == "min":
        agg = min(clean)
    elif mode == "geom":
        eps = 1e-6
        prod = 1.0
        for s in clean:
            prod *= max(eps, s)
        agg = prod ** (1.0 / len(clean))
    elif mode == "harmonic":
        eps = 1e-6
        agg = len(clean) / sum(1.0 / max(eps, s) for s in clean)
    elif mode == "vote":
        # 将分数二值化：>=threshold 记为 1，否则 0，少数服从多数
        votes = sum(1 if s >= threshold else 0 for s in clean)
        agg = sum(clean) / len(clean)
        # 若投票不通过则强制更保守：把聚合分下拉到 min(agg, threshold - 0.5)
        if votes < 2:  # 三路至少两票通过才算通过
            agg = min(agg, threshold - 0.5)
    else:
        # 回退到 mean
        agg = sum(clean) / len(clean)

    is_hallu = bool(agg < threshold)
    return float(agg), is_hallu


def _evaluate_step_multiroute(
    *,
    gen_step: str,
    ref_steps: List[str],
    idx: int,
    builders: Dict[str, Any],
    agg_mode: str,
    agg_weights: Tuple[float, float, float],
    overall_threshold: float,
) -> Tuple[float, Dict[str, float], bool]:
    """
    对单步进行三路打分并聚合。
    返回: (agg_score, per_route_scores, is_hallu)
    """
    # --- Holistic
    prior_ref = "\n".join(ref_steps[: idx + 1]) if ref_steps else ""
    s_hol = builders["holistic"].run(gen_step, prior_ref)

    # --- Pairwise
    s_pairs = builders["pairwise"].run(gen_step, ref_steps[: idx + 1])
        
    s_pair = min(s_pairs) if s_pairs else 0.0

    # --- Self-judge
    s_self = builders["selfjudge"].run(gen_step)

    # --- aggregate
    agg, is_hallu = _aggregate(
        [s_pair, s_hol, s_self],
        mode=agg_mode,
        weights=agg_weights,
        threshold=overall_threshold,
    )

    per_route = {"pairwise": float(s_pair), "holistic": float(s_hol), "selfjudge": float(s_self)}
    return float(agg), per_route, bool(is_hallu)


def score_case(rec: Dict[str, Any], builders: Dict[str, Any]) -> Dict[str, Any]:
    """对 (gen_output[i], ref_steps[:i+1]) 逐步多路打分并聚合。"""
    ref_steps: List[str] = rec["ref_steps"]
    gen_output: List[str] = rec["gen_output"]
    N = len(ref_steps)
    
    # 读取聚合配置（若 Config 中无，则使用默认）
    agg_mode = Config.get("judge_aggregation", "weighted")
    agg_weights = tuple(Config.get("judge_aggregation_weights", (0.4, 0.4, 0.2)))
    overall_threshold = float(Config.get("overall threshold", 3.0))

    total_score = 0.0
    steps_log: List[Dict[str, Any]] = []

    i = 1
    # 与原逻辑一致的遍历
    for idx in range(len(gen_output) - 1):
        current_output = _normalize_generation_input(gen_output[idx])
        if not current_output:
            step_score = 0.0
            step_contrib = 0.0
            steps_log.append({
                "index": i,
                "score": step_score,
                "hallucination": 1,
                "routes": {"pairwise": 0.0, "holistic": 0.0, "selfjudge": 0.0},
            })
            i += 1
            continue
        
        agg_score, per_route, is_hallu = _evaluate_step_multiroute(
            gen_step=current_output,
            ref_steps=ref_steps,
            idx=idx,
            builders=builders,
            agg_mode=agg_mode,
            agg_weights=agg_weights,
            overall_threshold=overall_threshold,
        )

        step_score = float(agg_score)                 # 0..5
        step_contrib = step_score / max(1, N) * 20.0  
        total_score += step_contrib

        print(f"[DEBUG] Step {i}: pair={per_route['pairwise']:.2f}, hol={per_route['holistic']:.2f}, "
              f"self={per_route['selfjudge']:.2f} -> agg={step_score:.2f}, contrib={step_contrib:.4f}")

        steps_log.append({
            "index": i,
            "score": step_score,
            "hallucination": int(bool(is_hallu)),
            "routes": per_route,
        })
        i += 1

    return {
        "num_steps": N,
        "total_score": total_score,
        "steps": steps_log,
    }
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", type=str, required=True,
                        help="stage1 生成的 gen_only.jsonl 路径")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="输出目录（默认与 gen_file 同目录）")
    args = parser.parse_args()

    gen_file = os.path.abspath(args.gen_file)
    run_dir = os.path.abspath(args.run_dir) if args.run_dir else os.path.dirname(gen_file)
    os.makedirs(run_dir, exist_ok=True)

    out_cases = os.path.join(run_dir, "case_results.jsonl")
    out_cases_pretty = os.path.join(run_dir, "case_results_pretty.json")
    out_summary = os.path.join(run_dir, "summary.json")

    pairwise, holistic, selfjudge = build_judge_model()
    builders = {
        "pairwise": pairwise,
        "holistic": holistic,
        "selfjudge": selfjudge,
    }
    scores: List[float] = []
    num = 0
    with open(gen_file, "r", encoding="utf-8") as fin, \
         open(out_cases, "w", encoding="utf-8", buffering=1) as fout_cases, \
         open(out_cases_pretty, "w", encoding="utf-8", buffering=1) as fout_cases_pretty:

        for line in fin:
            if num > 50:
                break
            t0 = time.time()
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            num += 1

            eval_res = score_case(rec, builders)
            score = float(eval_res["total_score"])
            scores.append(score)

            case_record = {
                "id": rec["id"],
                "difficulty": float(rec.get("difficulty", 0.0)),
                "score": score,
                "num_steps": eval_res["num_steps"],
                "steps": eval_res["steps"],
                "problem": rec["problem"],
                "answer": rec.get("answer", ""),
            }
            _write_jsonl_line(fout_cases, case_record)
            _write_pretty_json(fout_cases_pretty, case_record)

            t1 = time.time()
            print(f"[INFO][JUDGE] scored case {rec['id']}, score={score:.4f}, time={t1 - t0:.2f}s")

    # 汇总（与 main.py 风格一致：近似百分制）
    model_score = (sum(scores) * 10 / max(1, num))
    with open(out_summary, "w", encoding="utf-8") as fsum:
        json.dump({"num": num, "avg_score": model_score}, fsum, ensure_ascii=False, indent=2)

    print(f"[RESULT][JUDGE] Processed {num} cases")
    print(f"[RESULT][JUDGE] Final model score ≈ {model_score:.2f}")

if __name__ == "__main__":
    main()
