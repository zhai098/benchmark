from __future__ import annotations

import json
import math
import time
from typing import Dict, Any, List, Iterable, Tuple, Optional
import logging
import os
import re
from functools import lru_cache
from config import Config
from runner import VLLMRunner
from data_process import (
    Processor,
    _write_case_text_log,
    _write_jsonl_line,
    _write_pretty_json,
    flatten_to_string,
    safe_json_loads,
    _normalize_generation_input
)
from prompt import (
    Claim_Segment_Prompt,
    Generate_Prompt,
    Judge_Prompt,
    On_Policy_Prompt,
    PairwiseEntailmentPrompt,
)
from data_process import safe_json_loads  # 文件顶部集中导入一次

logger = logging.getLogger(__name__)

reasoning_model = VLLMRunner(Config["reasoning_model"], vllm_config=Config["reasoning_model_params"], sampling_config=Config["reasoning_sampling_params"], gpus=Config["reasoning_model_gpus"])
judge_model = VLLMRunner(Config["judge_model"], vllm_config=Config["judge_model_params"], sampling_config=Config["judge_sampling_params"], gpus=Config["judge_model_gpus"])
processor = Processor()
judge_promptbuilder = Judge_Prompt(judge_model)
cs_promptbuilder = Claim_Segment_Prompt(judge_model)
on_policy_transformer = On_Policy_Prompt(reasoning_model)
entail_promptbuilder = PairwiseEntailmentPrompt(judge_model)


###不可靠的方法
"""def align_next_step_embedding(gen: str, ref: str, max_len: int = 10, threshold: float = 0.7):
    ###比较生成CoT的切分对应到goldenanswer的切分的那个位置
    ###切分+评估，语义相似度
    gen_sentences = split_sentences(gen)
    if not gen_sentences:
        return None, 0.0, True
    #从头开始取不同长短的前缀
    prefixes = [" ".join(gen_sentences[:i+1]) for i in range(max_len)]
    
    embeddings = embedding_model.encode([ref] + prefixes, convert_to_tensor=True)
    ref_emb = embeddings[0]
    prefix_embs = embeddings[1:]
    #计算相似度（与幻觉不一致）
    cos_scores = util.cos_sim(ref_emb, prefix_embs)[0]
    max_score, max_idx = torch.max(cos_scores, dim=0)
    is_hallucination = max_score < threshold
    return prefixes[max_idx], max_score.item(), is_hallucination"""

def claim_decompose(text: str) -> List[str]:
    claims = cs_promptbuilder.run(text)
    logger.debug("Claim decomposition output: %s", claims)
    return [segment["text"] for segment in claims.get("segments", [])]

@lru_cache(maxsize=5000)
def cached_claim_decompose(text: str) -> List[str]:
    return claim_decompose(text)

#按句子为单位评分，取最高分的前三个平均
def align_next_step_LLM_1(
    gen: str,
    ref: str,
    *,
    ent: PairwiseEntailmentPrompt,                           # 新增：传入上面的评测器实例
    threshold: float = Config["threshold"],                  # 单对 overall.score 阈值（决定该对是否“好”）
    overall_threshold: float = Config["overall threshold"],  # 所有配对平均后的总体阈值
    max_len: int = Config["max prefix_num"],                 # 仅在 gen 的前 K 句里找匹配
):

    # --- 规整输入（允许 list/tuple 进来） ---
    gen = _normalize_generation_input(gen)
    ref = _normalize_generation_input(ref)


    if not gen or not ref:
        logger.debug("Empty text encountered in entailment alignment.")
        return 0.0, True


    gen_sents_all = processor.sentence_split_en(gen)
    
    if not gen_sents_all or not ref:
        logger.debug("Sentence splitter returned no sentences for generated text: %s", gen)
        return 0.0, True

    # 仅考察 gen 的前 K 句
    K = max(1, min(max_len, len(gen_sents_all)))
    gen_sents = gen_sents_all[:K]
    print(f"gen_sents length: {len(gen_sents)}")
    print(f"gen_sents content: {gen_sents}")
    scored = []
    for idx, g in enumerate(gen_sents):
        scored.append((idx, g))

    matches = []
    picked = None
    for idx, g in scored:
        
        # 调用蕴含评测器（双向）得到严格 JSON
        res = ent.run(g, ref)   # {"forward":..., "backward":..., "overall":...}
        f_score = float(res[0])
        b_score = float(res[1])
        ov = float((f_score + b_score)/2.0)
        if ov < 0.4:
            continue
        picked = {
            "ref": ref,
            "gen_index": idx,
            "forward": f_score,
            "backward": b_score,
            "score": ov,
            "is_hallucination": ov < threshold
        }
        print("score:", ov, "gen_index:", idx)
        matches.append(picked)

    if picked is None:
        picked = ({
            "ref": ref, "best_gen": "", "gen_index": None, "sim": 0.0,
            "score": 0.0, "is_hallucination": True
        })

    # --- 汇总总体分 ---
    scores = [m["score"] for m in matches[:3]]
    overall_score = sum(scores) / 3 if scores else 0.0
    is_hallucination = overall_score < overall_threshold

    return overall_score, is_hallucination

def align_next_step_LLM_2(
    gen: str,
    ref: str,
    *,
    ent: PairwiseEntailmentPrompt,                           # 新增：传入上面的评测器实例
    threshold: float = Config["threshold"],                  # 单对 overall.score 阈值（决定该对是否“好”）
    overall_threshold: float = Config["overall threshold"],  # 所有配对平均后的总体阈值
    max_len: int = Config["max prefix_num"],                 # 仅在 gen 的前 K 句里找匹配
):

    # --- 规整输入（允许 list/tuple 进来） ---
    gen = _normalize_generation_input(gen)
    ref = _normalize_generation_input(ref)


    if not gen or not ref:
        logger.debug("Empty text encountered in entailment alignment.")
        return 0.0, True


    gen_sents_all = processor.sentence_split_en(gen)
    
    if not gen_sents_all or not ref:
        logger.debug("Sentence splitter returned no sentences for generated text: %s", gen)
        return 0.0, True

    # 仅考察 gen 的前 K 句
    K = max(1, min(max_len, len(gen_sents_all)))
    gen_sents = gen_sents_all[:K]
    gen_prefix = " ".join(gen_sents)

    score = judge_promptbuilder.run()
    
    is_hallucination = score < overall_threshold

    return score, is_hallucination

def align_next_step_LLM(
    gen: str,
    ref: str,
    *,
    threshold: float = Config["threshold"], 
    overall_threshold: float = Config["overall threshold"], 
    max_len: int = Config["max prefix_num"],  
):
    gen = flatten_to_string(gen, sep=" ")
    ref = flatten_to_string(ref, sep=" ")
    print("gen:", gen)
    print("ref:", ref)
    if not gen or not ref:
        return 0.0, True, {"reason": "empty input"}

    sents = processor.sentence_split_en(gen)
    
    if not sents:
        return 0.0, True, {"reason": "no sentences"}
    
    K = min(max_len, len(sents))
    head_sents = sents[:K]
    prefix = "".join(head_sents)
    
    gen_claims = claim_decompose(prefix)
    ref_claims = cached_claim_decompose(ref)
    if not gen_claims or not ref_claims:
        return 0.0, True, {"reason": "claim decomposition failed"}
    
    matches: List[Dict[str, Any]] = []
    for rc in ref_claims:
        best_score = -1.0
        best_gc = ""
        reason_ = ""
        for gc in gen_claims:
            score, label, reason = judge_hallucination(gc, rc)
            if score > best_score and label in ["entailed", "incomplete"]:
                best_score = score
                best_gc = gc
                reason_ = reason
        best = {
                "ref": rc,
                "best_gen": best_gc,
                "score": best_score,
                "justification": reason_,
                "is_hallucination": False if best_score > threshold else True
        }
        matches.append(best)
        
    scores = [m["score"] for m in matches]
    from collections import Counter

    best_gens = [m["best_gen"] for m in matches]
    counts = Counter(best_gens)

    # 找到重复的项（即出现次数 > 1）,匹配失败的情况不算
    duplicates = {gc: cnt for gc, cnt in counts.items() if cnt > 1 and gc != ""}
    overall_score = sum(scores) / len(scores)
    overall_score -= len(duplicates) * 0.5  # 每个重复项扣0.5分
    is_hallucination = overall_score < overall_threshold
    return overall_score, is_hallucination, matches
     
            
def judge_hallucination(
    gen_claim: str,
    ref_claim: str,
    runner=judge_model,
):
    
    data = judge_promptbuilder.run(gen_claim, ref_claim)
    
    # 提取信息
    score = float(max(0.0, min(1.0, float(data.get("score", 0.0)))))
    label = str(data.get("label", "irrelevant")).strip().lower()
    just = str(data.get("justification", ""))[:200]
    return score, label, just
        
def execute_evaluation(
        obj, 
        alpha: float = Config["alpha"], 
        lambda_h: float = Config["lambda_h"],
        max_len: int = Config["max prefix_num"],                 # 仅在 gen 的前 K 句里找匹配
    ) -> float:
    ### alpha控制前期权重衰减速度——0.5-2之间，lambda_h控制幻觉的惩罚力度——>=1
    ### 幻觉惩罚+按步权重控制
    problem = obj["problem"]
    thought_seg = obj["segments"]
    answer = obj["answer"]

    print(f"\n[DEBUG] Start evaluation for one sample\n")
    print(f"[DEBUG] Question: {problem}\n")
    print(f"[DEBUG] Number of thought steps: {len(thought_seg)}\n")
    
    processed_thought = []
    N = len(thought_seg)
    total_score = 0.0
    # 计算总权重的归一化因子
    # w(i) ∝ (N - i + 1)^α
    weight_denominator = sum((N - i + 1) ** alpha for i in range(1, N + 1))
    #初始化prompt
    generate_promptbuilder = Generate_Prompt(reasoning_model, problem)
    # on-policy转化
    current_prompt = generate_promptbuilder.return_prompt()
    #print(current_prompt)
    thought_policy = []
    for idx, seg in enumerate(thought_seg, 1):
        thought_policy.append(seg["content"])
    """for idx, seg in enumerate(thought_seg, 1):
        if seg["type"] == "text":
            op = on_policy_transformer.on_policy_trans(seg["content"])
            sen_policy = op.get("modified_text", "")
            thought_policy.append(sen_policy)
        else:
            thought_policy.append(seg["content"])
        #print(f"[DEBUG] On-policy transform step {idx}: {sen_policy}")"""

    unprocessed_thought = thought_policy
    processed_thought = []
    i = 1
    gen_output = []
    out_dir = os.path.abspath("./outputs")
    os.makedirs(out_dir, exist_ok=True)
    while unprocessed_thought:
        
        current_step = unprocessed_thought.pop(0)
        
        #更新prompt
        generate_promptbuilder.add_step(current_step)
        current_output = generate_promptbuilder.run()
        print(f"[DEBUG] Generation step {i}")
        gen_output.append(current_output)
        processed_thought.append(current_step)
        #可以在此处添加逻辑修改步长
        #这决定了评测效果
        if not unprocessed_thought:
            print("[DEBUG] Reached last step (generation), stop generation loop")
            break
        i += 1


    case_steps = []
    i = 1
    for idx in range(len(gen_output) - 1):
        current_output = gen_output[idx]
        current_step = processed_thought[idx]
        next_ref_step = processed_thought[idx+1]
        w_i = ((N - i + 1) ** alpha) / weight_denominator

        # 在为了优化评分策略，若无幻觉，则将next_step替换为模型生成的，不然评测的可靠性无法评估
        
        result = align_next_step_LLM_1(current_output, next_ref_step, ent=entail_promptbuilder)
        if not result or not isinstance(result, tuple):
            # fallback 安全值
            score, halluc_penalty = 0.0, True
        else:
            # 期望 (score: float, is_hallucination: bool, details: Any)
            score, halluc_penalty = result
            # 防护：确保类型正确
            try:
                score = float(score)
            except Exception:
                score = 0.0
            halluc_penalty = bool(halluc_penalty)
        # 出现幻觉要额外惩罚
        #step_score = score - lambda_h * int(halluc_penalty)      #halluc_penalty:0 / 1
        step_score = score
        step_contribution = w_i * step_score
        total_score += step_contribution
        
        print(f"[DEBUG] Step {i}: step_score={step_score:.4f}, contribution={step_contribution:.4f}")
        case_steps.append({
            "index": i,
            "score": score,
            "hallucination": int(halluc_penalty),
            "step_score": step_score,
        })
        i += 1
        
    print(f"[DEBUG] Final score for this sample = {total_score:.4f}")
    return {
        "problem": problem,
        "answer": answer,
        "num_steps": N,
        "total_score": total_score,
        "steps": case_steps,
        "gen_output": gen_output,
    }
        
    
def main(): 
    out_dir = os.path.abspath("./outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_cases = os.path.join(out_dir, "case_results.jsonl")
    out_genlog = os.path.join(out_dir, "gen_and_scores.jsonl")
    out_cases_pretty = os.path.join(out_dir, "case_results_pretty.json")
    out_genlog_pretty = os.path.join(out_dir, "gen_and_scores_pretty.json")
    out_case_text = os.path.join(out_dir, "case.log")
    scores = []
    # 难度控制因子
    beta = Config["beta"]
    num = 0
    input_path = Config["Input_path"]
    print(f"[INFO] Start evaluation, loading file: {input_path}")
    with open(out_cases, "w", encoding="utf-8", buffering=1) as fout_cases, \
         open(out_genlog, "w", encoding="utf-8", buffering=1) as fgen, \
         open(out_cases_pretty, "w", encoding="utf-8", buffering=1) as fout_cases_pretty, \
         open(out_genlog_pretty, "w", encoding="utf-8", buffering=1) as fgen_pretty, \
         open(out_case_text, "w", encoding="utf-8", buffering=1) as fcase_text, \
         open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            time_start = time.time()
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("[WARN] skip one bad json line")
                continue
            num += 1

            diff = float(obj["difficulty"])
            ###执行数据的评估
        
            case_eval = execute_evaluation(obj)
            score = float(case_eval["total_score"])
            #根据难度调整分数权重
            score *= math.exp(beta * (diff - 5) / 10) 
            ###需要并行化来提升效率
            scores.append(score)
            
            case_record = {
                "id": num,
                "difficulty": diff,
                "score": score,
                "num_steps": case_eval["num_steps"],
                "steps": case_eval["steps"],
                "problem": case_eval["problem"],
                "answer": case_eval["answer"],
            }
            _write_jsonl_line(fout_cases, case_record)
            _write_pretty_json(fout_cases_pretty, case_record)

            # 写入 gen_output + 评分（每个 case 一行 JSON）
            case_genlog = {
                "id": num,
                "difficulty": diff,
                "gen_output": case_eval.get("gen_output", []),
                "steps": case_eval["steps"],
                "final_total_score": score,
            }
            _write_jsonl_line(fgen, case_genlog)
            _write_pretty_json(fgen_pretty, case_genlog)
            _write_case_text_log(
                fcase_text,
                case_record=case_record,
                case_genlog=case_genlog,
            )            
            time_end = time.time()
            print(f"[INFO] Processed sample {num}, score={score:.4f}, time={time_end - time_start:.2f}s")
            if num % 50 == 0:
                print(f"[INFO] processed {num} samples, avg score: {sum(scores)/len(scores):.4f}")
    #近似百分制
    model_score = sum(scores) * 10 / num
    
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fsum:
        json.dump({"num": num, "avg_score": model_score}, fsum, ensure_ascii=False, indent=2)

    print(f"[RESULT] Processed {num} samples")
    print(f"[RESULT] Final model score ≈ {model_score:.2f}")

if __name__ == "__main__":
    main()