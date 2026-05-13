#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


"""
批量运行 Progress_Prompt 的小脚本，用来判断每个切分出来的前缀 GEN
是不是在“实质推进”，还是主要在复述 / 反思废话。

输入：JSONL，每行形如：
{
  "id": "case_1",
  "problem": "题目文本，可选",
  "segments": ["seg_0", "seg_1", "seg_2", ...]
}

输出：JSONL，每行一个 (case_id, seg_idx) 的打分结果：
{
  "id": "case_1",
  "seg_idx": 1,
  "problem": "...",
  "ref": "...",      # 拼好的 REF
  "gen": "...",      # 当前这个 segment
  "progress_score": 2,
  "raw_output": ...  # model 原始输出（可选）
}
"""

import os
import json
import time
import argparse
import logging

from benchmark_core.config import Config
from runner import VLLMRunner
from benchmark_core.prompt import Progress_Prompt
from benchmark_core.data_process import Processor, _write_jsonl_line, _write_pretty_json, _normalize_generation_input
# 如果你有统一的写文件工具，也可以用：
# from benchmark_core.data_process import _write_jsonl_line, _write_pretty_json

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def iter_jsonl(path):
    """逐行读取 JSONL。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl_line(f, obj):
    """简单的 jsonl 写入封装。"""
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, required=True,
                        help="输入 JSONL 文件（包含 problem + segments）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 JSONL 文件（每行一个前缀打分结果）")
    parser.add_argument("--pretty_dir", type=str, default="",
                        help="可选，把每个 case 的结果再单独写成 pretty json（调试用）")
    parser.add_argument("--max_cases", type=int, default=-1,
                        help="仅调试时用，限制最多处理多少个 case")

    args = parser.parse_args()

    # === 初始化模型 ===
    logger.info("Loading config and model...")
    processor = Processor()
    judge_model = VLLMRunner(
        Config["judge_model"],
        vllm_config=Config["judge_model_params"],
        sampling_config=Config["judge_sampling_params"],
        gpus=Config["judge_model_gpus"],
    )
    progress_prompt = Progress_Prompt(judge_model)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.pretty_dir:
        os.makedirs(args.pretty_dir, exist_ok=True)

    num_cases = 0
    num_segments = 0
    t0 = time.time()

    with open(args.output, "w", encoding="utf-8") as fout:
        for rec in iter_jsonl(args.input):
            case_id = rec.get("id", f"case_{num_cases}")
            problem = rec.get("problem", "")
            gens = rec.get("gen_output", None)
            refs = rec.get("ref_steps", None)

            logger.info(f"Processing case {case_id}, {len(gens)} segments")
            num_cases += 1

            # 逐个 segment 评估
            case_results = []

            for idx, gen in enumerate(gens):
                ref_text = "\n".join(refs[:idx + 1])  # 之前所有段拼成 REF
                
                current_output = _normalize_generation_input(gen)
                gen_sents_all = processor.sentence_split_en(current_output)
                K = max(1, min(Config["max prefix_num"], len(gen_sents_all)))
                gen_sents = gen_sents_all[:K]
                gen_prefix = " ".join(gen_sents)
                
                try:
                    res = progress_prompt.run(
                        gen=gen_prefix,
                        problem=problem,
                        ref=ref_text,
                    )
                    score = res.get("score")
                except Exception as e:
                    logger.exception(f"Progress_Prompt 失败, case={case_id}, seg_idx={idx}")
                    score = None
                    res = {"error": str(e)}

                out_obj = {
                    "id": case_id,
                    "gen_idx": idx,
                    "problem": problem,
                    "gen": gen_prefix,
                    "progress_score": score,
                    "raw_output": res.get("raw_output"),
                }

                write_jsonl_line(fout, out_obj)
                case_results.append(out_obj)

                # 更新 REF：把当前 seg 加入 prefix
                num_segments += 1

            # 可选：每个 case 单独存一个 pretty json，方便人工检查
            if args.pretty_dir:
                pretty_path = os.path.join(args.pretty_dir, f"{case_id}.json")
                with open(pretty_path, "w", encoding="utf-8") as pf:
                    json.dump(
                        {
                            "id": case_id,
                            "problem": problem,
                            #"segments": segments,
                            "progress_results": case_results,
                        },
                        pf,
                        ensure_ascii=False,
                        indent=2,
                    )

            if 0 < args.max_cases <= num_cases:
                logger.info("达到 max_cases 限制，提前终止。")
                break

    t1 = time.time()
    logger.info(f"Done. cases={num_cases}, segments={num_segments}, time={t1 - t0:.1f}s")


if __name__ == "__main__":
    main()
