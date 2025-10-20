# stage2_judge.py
from __future__ import annotations

import os, json, time, argparse, logging
from typing import Dict, Any, List, Tuple
from config import Config
from runner import VLLMRunner
from data_process import Processor, _write_jsonl_line, _write_pretty_json, _normalize_generation_input
from prompt import Judge_Prompt, PairwiseEntailmentPrompt

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
    judge_promptbuilder = Judge_Prompt(judge_model)
    entail_promptbuilder = PairwiseEntailmentPrompt(judge_model)
    return judge_promptbuilder, entail_promptbuilder

# === 复制 align_next_step_LLM_2（不改逻辑）===
def align_next_step_LLM_2(
    gen: str,
    ref: str,
    *,
    judge_promptbuilder: Judge_Prompt,
    ent: PairwiseEntailmentPrompt,
    threshold: float = Config["threshold"],
    overall_threshold: float = Config["overall threshold"],
    max_len: int = Config["max prefix_num"],
) -> Tuple[float, bool]:

    gen = _normalize_generation_input(gen)
    ref = _normalize_generation_input(ref)
    if not gen or not ref:
        logger.debug("Empty text in entailment alignment.")
        return 0.0, True

    gen_sents_all = processor.sentence_split_en(gen)
    if not gen_sents_all:
        logger.debug("Sentence splitter returned no sentences for generated text.")
        return 0.0, True

    K = max(1, min(max_len, len(gen_sents_all)))
    gen_prefix = " ".join(gen_sents_all[:K])                  # 

    score = judge_promptbuilder.run(gen_prefix, ref)           # 
    is_hallu = score < overall_threshold
    return float(score), bool(is_hallu)

def score_case(rec: Dict[str, Any], judge_promptbuilder: Judge_Prompt, ent: PairwiseEntailmentPrompt) -> Dict[str, Any]:
    """与 execute_evaluation() 的评测部分等价：对 (gen_output[i], ref_steps[i+1]) 逐步打分并累加。"""
    ref_steps: List[str] = rec["ref_steps"]
    gen_output: List[str] = rec["gen_output"]
    N = len(ref_steps)  # 与原逻辑一致，用参考步数计入分母

    total_score = 0.0
    steps_log: List[Dict[str, Any]] = []
    i = 1
    for idx in range(len(gen_output) - 1):                     # 
        current_output = gen_output[idx]
        next_ref_step = ref_steps[idx + 1]

        score, is_hallu = align_next_step_LLM_2(
            current_output, next_ref_step,
            judge_promptbuilder=judge_promptbuilder,
            ent=ent
        )
        # 原逻辑里步分=score；总分累加 = step_score / N * 20
        step_score = float(score)                               # 
        step_contrib = step_score / N * 20.0
        total_score += step_contrib

        print(f"[DEBUG] Step {i}: step_score={step_score:.4f}, contribution={step_contrib:.4f}")
        steps_log.append({
            "index": i,
            "score": step_score,
            "hallucination": int(bool(is_hallu)),
            "step_score": step_score,
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

    judge_promptbuilder, entail_promptbuilder = build_judge_model()

    scores: List[float] = []
    num = 0
    with open(gen_file, "r", encoding="utf-8") as fin, \
         open(out_cases, "w", encoding="utf-8", buffering=1) as fout_cases, \
         open(out_cases_pretty, "w", encoding="utf-8", buffering=1) as fout_cases_pretty:

        for line in fin:
            t0 = time.time()
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            num += 1

            eval_res = score_case(rec, judge_promptbuilder, entail_promptbuilder)
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
            print(f"[INFO][Stage2] scored case {rec['id']}, score={score:.4f}, time={t1 - t0:.2f}s")

    # 汇总（与 main.py 风格一致：近似百分制）
    model_score = (sum(scores) * 10 / max(1, num))
    with open(out_summary, "w", encoding="utf-8") as fsum:
        json.dump({"num": num, "avg_score": model_score}, fsum, ensure_ascii=False, indent=2)

    print(f"[RESULT][Stage2] Processed {num} cases")
    print(f"[RESULT][Stage2] Final model score ≈ {model_score:.2f}")

if __name__ == "__main__":
    main()
