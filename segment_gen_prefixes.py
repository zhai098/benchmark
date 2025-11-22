
from __future__ import annotations

import argparse
import json, os, time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Config
from runner import VLLMRunner
from prompt import Prefix_segmenter


def prefix_segment(rec: dict, prefix_model: Prefix_segmenter):
    gens = rec["gen_output"]
    refs = rec["ref_steps"]
    len_gen = len(gens)
    prefix_list = []
    for i, gen in enumerate(gens):
        if i < len_gen - 1:
            res = prefix_model.run(gen, refs[i+1])
            prefix_list.append(res["prefix"])
        else:
            break
    
    
    return prefix_list

def build_prefix_model() -> VLLMRunner:
    """和 stage1_generate 一样，从 Config 里造一个模型实例。"""
    return VLLMRunner(
        Config["judge_model"],
        vllm_config=Config["judge_model_params"],
        sampling_config=Config["judge_sampling_params"],
        gpus=Config["judge_model_gpus"],
    )


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-file", required=True, help="Path to gen JSON/JSONL file")
    parser.add_argument("--out", type=str, default="segmented_prefixes.jsonl",
                        help="输出文件名，默认 segmented_prefixes.jsonl")
    args = parser.parse_args()

    out_path = os.path.abspath("./segmented_prefixes")
    os.makedirs(out_path, exist_ok=True) 
    out_file = os.path.join(out_path, args.out)
    gen_path = Path(args.gen_file)
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO][SEGMENT] loading: {gen_path}")

    prefix_model = build_prefix_model()
    segmenter = Prefix_segmenter(prefix_model)

    results: List[Dict[str, Any]] = []
    num = 0
    with open(gen_path, "r", encoding="utf-8") as fin, \
        open(out_file, "w", encoding="utf-8") as fout: 
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
            ref_steps = rec["ref_steps"]
            prefix_list = prefix_segment(rec, segmenter)
            
            case_prefix = {
                "id": rec["id"],
                "difficulty": float(rec.get("difficulty", 0.0)),
                "steps_prefix": prefix_list,
                "ref_steps": ref_steps,
                "problem": rec["problem"],
            }

            fout.write(json.dumps(case_prefix, ensure_ascii=False) + "\n")
            t1 = time.time()
            total_time += (t1 - t0)
            print(f"[INFO][SEGMENT] segment case {rec['id']}, time={t1 - t0:.2f}s")
            

        

if __name__ == "__main__":
    main()
