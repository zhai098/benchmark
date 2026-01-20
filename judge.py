# stage2_judge.py
from __future__ import annotations

import os, json, time, argparse, logging, math
from typing import Dict, Any, List, Tuple
from config import Config
from runner import VLLMRunner, DEEPSEEK_API_runner, DOUBAO_deepseek_API_runner
from data_process import Processor, _write_jsonl_line, _write_pretty_json, _normalize_generation_input
from prompt import Judge_Prompt, Pairwise_Prompt, Holistic_Prompt, SelfJudge_Prompt
import numpy as np
from openai import OpenAI


logger = logging.getLogger(__name__)

# === 与 main.py 保持同逻辑的全局组件 ===
processor = Processor()

def build_judge_model():
    judge_model = DOUBAO_deepseek_API_runner()
    # 三路 evaluator
    pairwise = Pairwise_Prompt(judge_model)
    holistic = Holistic_Prompt(judge_model)
    return pairwise, holistic


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

def agg_top_k_mean(scores: List[float], k: int = 1, take_highest: bool = True) -> Tuple[float, Dict]:
    # 过滤掉负分（将负值视为无效/缺失），只使用 >= 0 的分数参与聚合
    valid = [float(x) for x in scores if x is not None and float(x) >= 0]
    if not valid:
        # 当没有有效分数时，返回 0 并标注原因
        return 0.0, {"method": "top_k_mean", "k": 0, "take_highest": take_highest, "note": "no_nonnegative_scores"}

    s_sorted = sorted(valid, reverse=take_highest)
    k = max(1, min(k, len(s_sorted)))
    chosen = s_sorted[:k]
    return float(np.mean(chosen)), {"method": "top_k_mean", "k": k, "take_highest": take_highest}

def _evaluate_step_multiroute(
    *,
    gen_step: str,
    ref_steps: List[str],
    idx: int,
    builders: Dict[str, Any],
) -> Tuple[float, Dict[str, float], bool]:
    """
    对单步进行三路打分并聚合。
    返回: (agg_score, per_route_scores, is_hallu)
    """
    prior_ref = "\n".join(ref_steps[: idx + 1]) if ref_steps else ""
    # --- Holistic
    time_hol = time.time()
    hol_res = builders["holistic"].run(gen_step, prior_ref)
    time_hol_fin = time.time() - time_hol
    #print(f"[DEBUG] Holistic result: {hol_res}")
    # --- Pairwise
    time_pair = time.time()
    pairs_res = builders["pairwise"].run(gen_step, ref_steps[: idx + 1], prior_ref)
    time_pair_fin = time.time() - time_pair
    #print(f"[DEBUG] Pairwise result: {pairs_res}")
    
    

    
        
    # --- Self-judge
    # self_res = builders["selfjudge"].run(gen_step)

    #pair_score = agg_top_k_mean(pairs_res["scores"], k=math.ceil(len(pairs_res["scores"])/2))[0]
    pair_score = sum(sorted(pairs_res["scores"])[:2]) / 2
    # --- aggregate
    agg = pair_score + hol_res["score"]  # + self_res["score"]
    agg /= 2  # 3 路平均

    per_route = {"pairwise": float(pair_score), "holistic": float(hol_res["score"])}  # , "selfjudge": float(self_res["score"])}
    detail = {
        "pairwise": pairs_res,
        "holistic": hol_res,
        "time_holistic": time_hol_fin,
        "time_pairwise": time_pair_fin,
        # "selfjudge": self_res,
    }
    return float(agg), per_route, detail


def score_case(rec: Dict[str, Any], builders: Dict[str, Any]) -> Dict[str, Any]:
    """对 (gen_output[i], ref_steps[:i+1]) 逐步多路打分并聚合。"""
    ref_steps: List[str] = rec["ref_steps"]
    gen_prefix: List[str] = rec["gen_prefix"]
    N = len(ref_steps)

    # 读取聚合配置（若 Config 中无，则使用默认）
    agg_mode = Config.get("judge_aggregation", "weighted")
    agg_weights = tuple(Config.get("judge_aggregation_weights", (0.4, 0.4, 0.2)))
    overall_threshold = float(Config.get("overall threshold", 3.0))

    total_score = 0.0
    steps_log: List[Dict[str, Any]] = []

    i = 1
    # 与原逻辑一致的遍历
    for idx in range(len(gen_prefix) - 2):        
        prefix = gen_prefix[idx]
        
        if not prefix.strip():
            step_score = 0.0
            step_contrib = 0.0
            steps_log.append({
                "index": i,
                "score": step_score,
                "hallucination": 1,
                "routes": {"pairwise": 0.0, "holistic": 0.0},
            })
            i += 1
            continue
        
        agg_score, per_route, detail = _evaluate_step_multiroute(
            gen_step=prefix,
            ref_steps=ref_steps,
            idx=idx,
            builders=builders,
        )

        step_score = float(agg_score)                 # 0..5
        step_contrib = step_score / max(1, N - 1) * 20.0  
        total_score += step_contrib

        print(f"[DEBUG] Step {i}: pair={per_route['pairwise']:.2f}, hol={per_route['holistic']:.2f} -> agg={step_score:.2f}, contrib={step_contrib:.4f}")
        print(f"[DEBUG] Step {i} detail: {detail['time_holistic']:.2f}s hol, {detail['time_pairwise']:.2f}s pair")

        steps_log.append({
            "index": i,
            "score": step_score,
            "routes": per_route,
            "judge_detail": detail,
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

    per_case_dir = os.path.join(run_dir, "cases")
    os.makedirs(per_case_dir, exist_ok=True)
    
    out_summary = os.path.join(run_dir, "summary.json")
    out_cases = os.path.join(run_dir, "case_results.jsonl")
    out_cases_pretty = os.path.join(run_dir, "case_results_pretty.json")
    
    pairwise, holistic = build_judge_model()
    builders = {
        "pairwise": pairwise,
        "holistic": holistic,
    }
    scores: List[float] = []
    num = 0
    with open(gen_file, "r", encoding="utf-8") as fin, \
         open(out_cases, "w", encoding="utf-8", buffering=1) as fout_cases, \
         open(out_cases_pretty, "w", encoding="utf-8", buffering=1) as fout_cases_pretty:
        total_time = 0.0
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
            simple_case_record = {
                "id": rec["id"],
                "score": score,
                "num_steps": eval_res["num_steps"],
                "problem": rec["problem"],
                "answer": rec.get("answer", ""),
            }
            _write_jsonl_line(fout_cases, simple_case_record)
            _write_pretty_json(fout_cases_pretty, simple_case_record)
            
            raw_id = str(rec["id"])
            safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw_id)
            case_path = os.path.join(per_case_dir, f"{safe_id}.json")
            with open(case_path, "w", encoding="utf-8") as fcase:
                json.dump(case_record, fcase, ensure_ascii=False, indent=2)
            

            t1 = time.time()
            total_time += (t1 - t0)
            print(f"[INFO][JUDGE] scored case {rec['id']}, score={score:.4f}, time={t1 - t0:.2f}s")
            
    print(f"[INFO][JUDGE] Total scoring time: {total_time:.2f}s")
    # 汇总（与 main.py 风格一致：近似百分制）
    model_score = (sum(scores) / num)
    with open(out_summary, "w", encoding="utf-8") as fsum:
        json.dump({"num": num, "avg_score": model_score}, fsum, ensure_ascii=False, indent=2)

    print(f"[RESULT][JUDGE] Processed {num} cases")
    print(f"[RESULT][JUDGE] Final model score ≈ {model_score:.2f}")

if __name__ == "__main__":
    main()
