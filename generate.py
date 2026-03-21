# stage1_generate.py
from __future__ import annotations

import os, json, time, random, logging
from typing import Dict, Any, List, Tuple
from config import Config
from runner import VLLMRunner, Kimi_API_runner, DEEPSEEK_API_runner
from prompt import Generate_Prompt
from data_process import _write_jsonl_line, _write_pretty_json
from data_process import Processor, _write_jsonl_line, _write_pretty_json, _normalize_generation_input

logger = logging.getLogger(__name__)
processor = Processor()

def build_reasoning_model():
    """return VLLMRunner(
        model=Config["reasoning_model"],
        vllm_config=Config["reasoning_model_params"],
        sampling_config=Config["reasoning_sampling_params"],
        gpus=Config["reasoning_model_gpus"],
    )"""
    #return Kimi_API_runner()
    return DEEPSEEK_API_runner()

def generate_case(obj: Dict[str, Any], reasoning_model) -> Dict[str, Any]:
    """复用 main.py 的生成阶段：逐步 add_step -> run()，直到用尽参考步骤。
       逻辑等价于 execute_evaluation() 中生成部分。"""
    problem = obj["problem"]
    thought_seg = obj["segments"]
    answer = obj.get("answer", "")

    # 初始化 prompt（与原逻辑一致）
    generate_promptbuilder = Generate_Prompt(reasoning_model, query=problem)

    # 只取 content，等价于你现在的 thought_policy 构造
    thought_policy: List[str] = [seg["content"] for seg in thought_seg]  # 
    type_list: List[str] = [seg["type"] for seg in thought_seg]  
    unprocessed: List[Tuple[str, str]] = list(zip(thought_policy, type_list))
    processed: List[str] = []
    processed_types: List[str] = []
    gen_output: List[str] = []
    gen_prefixes: List[str] = []
    i = 1
    prompt_lists = []
    while unprocessed:
        current_step, current_type = unprocessed.pop(0)
        #print(f"[DEBUG] Generating step {i}, step={current_step}")
        generate_promptbuilder.add_step(current_step)
                
        # Use IDs for generation if available
        if hasattr(generate_promptbuilder, "return_prompt_ids"):
            prompt_lists.append(generate_promptbuilder.return_prompt_ids())
        else:
            prompt_lists.append(generate_promptbuilder.return_prompt())

        processed.append(current_step)
        processed_types.append(current_type)

        if not unprocessed:
            print("[DEBUG] Reached last step (generation), stop generation loop")
            break
        i += 1
    # 一次性生成所有步骤
    prompts = prompt_lists 
    reasonings, generations = reasoning_model.generate(prompts, None)
    gen_output.extend(generations)
    for gen in generations:
        current_output = _normalize_generation_input(gen)
        gen_sents_all = processor.sentence_split_en(current_output)
        K = max(1, min(Config["max prefix_num"], len(gen_sents_all)))
        gen_sents = gen_sents_all[:K]
        gen_prefix = " ".join(gen_sents)
        gen_prefixes.append(gen_prefix)
        #print(f"[DEBUG] Final generated prefix: {gen_prefix}\n")
        #print(f"[DEBUG] Full generated output: {current_output}\n")
    
    return {
        "problem": problem,
        "answer": answer,
        "prompts": prompts,       # 复现用
        "ref_steps": processed,     # 评测阶段需要参考步骤（与 processed_thought 等价）
        "gen_output": gen_output,   # 待评测的模型生成
        "gen_prefix": gen_prefixes,
        "reasoning": reasonings,
        "difficulty": float(obj.get("difficulty", 0.0)),
    }

def main():
    # 输出目录命名沿用 main.py 风格
    out_root = os.path.abspath("./gen_output")
    os.makedirs(out_root, exist_ok=True)

    tag = Config["tag"]
    run_dir_name = f"{Config['reasoning_model']}_{tag}"
    run_dir = os.path.join(out_root, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    # 只生成、不评测：生成专用文件名
    gen_only_jsonl = os.path.join(run_dir, "gen_only.jsonl")
    gen_only_pretty = os.path.join(run_dir, "gen_only_pretty.json")
    manifest_path = os.path.join(run_dir, "run_info.json")

    input_path = Config["Input_path"]
    print(f"[INFO][GENERATE] loading: {input_path}")
    reasoning_model = build_reasoning_model()
    num = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
        open(gen_only_jsonl, "w", encoding="utf-8", buffering=1) as fgen, \
        open(gen_only_pretty, "w", encoding="utf-8", buffering=1) as fgen_pretty:

        for line in fin:
            if num >= 100:
                break
            if num < Config.get("skip_generate_num", 0):
                num += 1
                continue
            t0 = time.time()
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("[WARN] skip one bad json line")
                continue
            num += 1

            res = generate_case(obj, reasoning_model)

            # 携带 id，方便第二阶段对齐
            out_record = {
                "id": num,
                "difficulty": res["difficulty"],
                "problem": res["problem"],
                "answer": res["answer"],
                "prompts": res["prompts"],        # 复现用
                "ref_steps": res["ref_steps"],      # 评测要用
                "gen_output": res["gen_output"],    # 评测要用
                "gen_prefix": res["gen_prefix"],    # 评测要用
                "reasoning_content": res["reasoning"],
            }
            _write_jsonl_line(fgen, out_record)
            _write_pretty_json(fgen_pretty, out_record)

            t1 = time.time()
            print(f"[INFO][GENERATE] generated case {num}, time={t1 - t0:.2f}s")

    # 写一个小 manifest，第二阶段方便指向文件
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_dir": run_dir,
            "gen_file": gen_only_jsonl,
            "model_reasoning": Config["reasoning_model"],
            "model_judge": Config["judge_model"],
            "num": num,
        }, f, ensure_ascii=False, indent=2)

    print(f"[RESULT][GENERATE] wrote {num} generations to: {gen_only_jsonl}")
    print(f"[RESULT][GENERATE] run_dir = {run_dir}")

if __name__ == "__main__":
    main()
